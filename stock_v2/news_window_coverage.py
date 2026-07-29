from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from stock_v2.dataset_integrity import normalize_ticker


def _iter_jsonl(paths: Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSON row at {path}:{line_number}")
                yield row


def _provider(value: Any) -> str:
    text = str(value or "").lower()
    if "google" in text:
        return "google_rss"
    if "naver" in text:
        return "naver_search"
    return text or "unknown"


def _initial_windows(start: pd.Timestamp, end: pd.Timestamp, days: int) -> Iterable[tuple[pd.Timestamp, pd.Timestamp]]:
    left = start.normalize()
    final_exclusive = end.normalize() + pd.Timedelta(days=1)
    while left < final_exclusive:
        right = min(final_exclusive, left + pd.Timedelta(days=max(1, int(days))))
        yield left, right
        left = right


def _split_window(
    left: pd.Timestamp,
    right: pd.Timestamp,
    min_window_days: int,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]] | None:
    span_days = int((right - left).days)
    if span_days <= max(1, int(min_window_days)):
        return None
    midpoint = left + pd.Timedelta(days=max(1, span_days // 2))
    if midpoint <= left or midpoint >= right:
        return None
    return (left, midpoint), (midpoint, right)


def reconstruct_news_window_coverage(
    *,
    raw_paths: Sequence[str | Path],
    coverage_paths: Sequence[str | Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: dict[tuple[str, str, str, str], dict[str, int]] = {}
    observed_by_ticker: defaultdict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    raw_rows = 0
    for row in _iter_jsonl(raw_paths):
        window = row.get("query_window") if isinstance(row.get("query_window"), Mapping) else {}
        acquisition = row.get("acquisition") if isinstance(row.get("acquisition"), Mapping) else {}
        ticker = normalize_ticker(row.get("ticker"))
        start = str(window.get("start") or "")
        end = str(window.get("end") or "")
        provider = _provider(acquisition.get("provider") or row.get("source"))
        if not ticker or not start or not end:
            continue
        raw_rows += 1
        key = (provider, ticker, start, end)
        response_count = int(acquisition.get("response_count", 0) or 0)
        result_cap = int(acquisition.get("result_cap", 0) or 0)
        existing = observations.setdefault(
            key,
            {"response_count": response_count, "result_cap": result_cap, "raw_rows": 0},
        )
        if existing["response_count"] != response_count or existing["result_cap"] != result_cap:
            raise ValueError(f"inconsistent acquisition metadata for {key}")
        existing["raw_rows"] += 1
        observed_by_ticker[(provider, ticker)].add((start, end))

    coverage_rows = list(_iter_jsonl(coverage_paths))
    output: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    aggregate_statuses: Counter[str] = Counter()
    for coverage in coverage_rows:
        ticker = normalize_ticker(coverage.get("ticker"))
        provider = _provider(coverage.get("source"))
        start = pd.to_datetime(coverage.get("start"), errors="coerce")
        end = pd.to_datetime(coverage.get("end"), errors="coerce")
        if not ticker or pd.isna(start) or pd.isna(end):
            issues.append({"code": "invalid_aggregate_coverage", "ticker": ticker, "provider": provider})
            continue
        status = str(coverage.get("status") or "complete")
        aggregate_statuses[status] += 1
        if int(coverage.get("request_errors", 0) or 0) > 0:
            issues.append({"code": "aggregate_request_error", "ticker": ticker, "provider": provider})
            continue
        adaptive = bool(coverage.get("adaptive_split")) and provider == "google_rss"
        min_window_days = int(coverage.get("min_window_days", 1) or 1)
        result_cap = int(coverage.get("articles_per_window", 100) or 100)
        known = observed_by_ticker[(provider, ticker)]
        leaf_count = split_count = saturated_count = 0

        def descendants(left_text: str, right_text: str) -> bool:
            left_value = pd.Timestamp(left_text)
            right_value = pd.Timestamp(right_text)
            return any(
                (child_left != left_text or child_right != right_text)
                and pd.Timestamp(child_left) >= left_value
                and pd.Timestamp(child_right) <= right_value
                for child_left, child_right in known
            )

        def visit(left: pd.Timestamp, right: pd.Timestamp) -> None:
            nonlocal leaf_count, split_count, saturated_count
            left_text = str(left.date())
            right_text = str(right.date())
            key = (provider, ticker, left_text, right_text)
            observed = observations.get(key)
            split = _split_window(left, right, min_window_days) if adaptive else None
            if observed is None and split is not None and descendants(left_text, right_text):
                split_count += 1
                visit(*split[0])
                visit(*split[1])
                return
            leaf_count += 1
            response_count = int(observed["response_count"]) if observed is not None else 0
            cap = int(observed["result_cap"]) if observed is not None and observed["result_cap"] else result_cap
            saturated = bool(adaptive and response_count >= max(1, cap))
            saturated_count += int(saturated)
            window_id = hashlib.sha256(
                f"news-window-v1|{provider}|{ticker}|{left_text}|{right_text}".encode("utf-8")
            ).hexdigest()
            output.append(
                {
                    "schema_version": 1,
                    "window_id": window_id,
                    "provider": provider,
                    "ticker": ticker,
                    "start": left_text,
                    "end_exclusive": right_text,
                    "status": "incomplete_saturated" if saturated else "complete",
                    "response_count": response_count,
                    "result_cap": cap,
                    "raw_rows": int(observed["raw_rows"]) if observed is not None else 0,
                    "inferred_empty": observed is None,
                    "lineage": "deterministic_adaptive_window_reconstruction_v1",
                }
            )

        for initial_left, initial_right in _initial_windows(
            pd.Timestamp(start),
            pd.Timestamp(end),
            int(coverage.get("window_days", 1) or 1),
        ):
            visit(initial_left, initial_right)

        expected = {
            "leaf_windows": int(coverage.get("leaf_windows", leaf_count) or 0),
            "split_windows": int(coverage.get("split_windows", split_count) or 0),
            "saturated_leaf_windows": int(coverage.get("saturated_leaf_windows", saturated_count) or 0),
            "requests": int(coverage.get("requests", leaf_count + split_count) or 0),
        }
        actual = {
            "leaf_windows": leaf_count,
            "split_windows": split_count,
            "saturated_leaf_windows": saturated_count,
            "requests": leaf_count + split_count,
        }
        if actual != expected:
            issues.append(
                {
                    "code": "window_reconstruction_mismatch",
                    "ticker": ticker,
                    "provider": provider,
                    "expected": expected,
                    "actual": actual,
                }
            )

    output.sort(key=lambda row: (row["provider"], row["ticker"], row["start"], row["end_exclusive"]))
    report = {
        "schema_version": 1,
        "raw_rows_with_windows": raw_rows,
        "aggregate_coverage_rows": len(coverage_rows),
        "aggregate_statuses": dict(sorted(aggregate_statuses.items())),
        "window_rows": len(output),
        "complete_window_rows": sum(row["status"] == "complete" for row in output),
        "saturated_window_rows": sum(row["status"] == "incomplete_saturated" for row in output),
        "inferred_empty_window_rows": sum(bool(row["inferred_empty"]) for row in output),
        "issues": issues,
        "issue_count": len(issues),
    }
    return output, report
