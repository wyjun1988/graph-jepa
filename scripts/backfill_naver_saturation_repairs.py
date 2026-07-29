from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Iterable, Mapping, Sequence

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.dataset_integrity import load_json, normalize_ticker
from stock_v2.news_sources import fetch_naver_search


def stable_id(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\0")
    return digest.hexdigest()[:20]


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {path}:{line_number}")
            yield row


def find_saturated_google_windows(
    paths: Sequence[Path],
    universe: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, str]]:
    windows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            ticker = normalize_ticker(row.get("ticker"))
            if ticker not in universe:
                continue
            acquisition = row.get("acquisition") if isinstance(row.get("acquisition"), Mapping) else {}
            if str(acquisition.get("provider") or "") != "google_rss":
                continue
            try:
                response_count = int(acquisition.get("response_count", 0) or 0)
                result_cap = int(acquisition.get("result_cap", 0) or 0)
            except (TypeError, ValueError):
                continue
            query_window = row.get("query_window") if isinstance(row.get("query_window"), Mapping) else {}
            left = pd.to_datetime(query_window.get("start"), errors="coerce")
            right = pd.to_datetime(query_window.get("end"), errors="coerce")
            if (
                result_cap <= 0
                or response_count < result_cap
                or pd.isna(left)
                or pd.isna(right)
                or int((right - left).days) > 1
            ):
                continue
            start = str(pd.Timestamp(left).date())
            end = str(pd.Timestamp(right).date())
            key = (ticker, start, end)
            windows[key] = {
                "ticker": ticker,
                "name": str(universe[ticker].get("name") or row.get("name") or ""),
                "start": start,
                "end": end,
                "repair_for_provider": "google_rss",
                "detected_result_cap": str(result_cap),
            }
    return [windows[key] for key in sorted(windows)]


def fetch_naver_pages(
    query: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
    *,
    max_results: int,
    timeout_sec: float,
    sleep_sec: float,
    fetcher: Callable[..., list[dict[str, str]]] = fetch_naver_search,
) -> tuple[list[dict[str, str]], int, bool]:
    page_size = 10
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    requests_made = 0
    exhausted = False
    for start_index in range(1, max(1, int(max_results)) + 1, page_size):
        page = fetcher(
            query,
            left,
            right,
            page_size,
            timeout_sec,
            sort="1",
            start_index=start_index,
        )
        requests_made += 1
        new_rows = 0
        for row in page:
            key = (str(row.get("link") or ""), str(row.get("title") or ""), str(row.get("published") or ""))
            if key in seen:
                continue
            seen.add(key)
            rows.append(dict(row))
            new_rows += 1
        if len(page) < page_size or new_rows == 0:
            exhausted = True
            break
        if sleep_sec > 0:
            time.sleep(sleep_sec)
    return rows, requests_made, not exhausted


def load_completed_windows(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    completed: set[tuple[str, str, str]] = set()
    for row in iter_jsonl(path):
        if str(row.get("status")) == "complete":
            completed.add(
                (
                    normalize_ticker(row.get("ticker")),
                    str(row.get("start") or ""),
                    str(row.get("end") or ""),
                )
            )
    return completed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair Google daily result caps with paginated Naver snippets.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-output", required=True)
    parser.add_argument("--max-results-per-window", type=int, default=2000)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root or ROOT).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    universe_payload = load_json(repo_root / str(config["universe_manifest"]))
    universe = {
        normalize_ticker(row.get("ticker")): dict(row)
        for row in universe_payload.get("universe", [])
        if normalize_ticker(row.get("ticker"))
    }
    raw_paths = [repo_root / str(value) for value in config.get("raw_paths", [])]
    plan = find_saturated_google_windows(raw_paths, universe)
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    coverage = Path(args.coverage_output)
    if not coverage.is_absolute():
        coverage = repo_root / coverage
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_windows(coverage) if args.resume else set()
    seen_ids = {str(row.get("id")) for row in iter_jsonl(output)} if args.resume and output.exists() else set()
    mode = "a" if args.resume else "w"
    started = time.perf_counter()
    finished = written = requests_made = failures = saturated_windows = 0
    with output.open(mode, encoding="utf-8") as output_handle, coverage.open(mode, encoding="utf-8") as coverage_handle:
        for window in plan:
            key = (window["ticker"], window["start"], window["end"])
            if key in completed:
                continue
            left = pd.Timestamp(window["start"])
            right = pd.Timestamp(window["end"])
            articles: list[dict[str, str]] = []
            window_requests = 0
            saturated = False
            error = ""
            for attempt in range(max(0, args.request_retries) + 1):
                try:
                    articles, window_requests, saturated = fetch_naver_pages(
                        f'"{window["name"]}"',
                        left,
                        right,
                        max_results=args.max_results_per_window,
                        timeout_sec=args.timeout_sec,
                        sleep_sec=args.sleep_sec,
                    )
                    error = ""
                    break
                except requests.RequestException as exc:
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
                    response = getattr(exc, "response", None)
                    if int(getattr(response, "status_code", 0) or 0) in {403, 429}:
                        break
                    if attempt < max(0, args.request_retries):
                        time.sleep(max(0.0, args.retry_backoff_sec) * (2**attempt))
            requests_made += window_requests
            window_written = 0
            if not error:
                for response_rank, article in enumerate(articles, start=1):
                    article_id = stable_id(window["ticker"], article.get("title", ""), article.get("link", ""))
                    if article_id in seen_ids:
                        continue
                    seen_ids.add(article_id)
                    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    record = {
                        "schema_version": 2,
                        "id": article_id,
                        "collected_at_utc": collected_at,
                        "ts": collected_at,
                        "ticker": window["ticker"],
                        "name": window["name"],
                        "article": article,
                        "published": article.get("published", ""),
                        "source": "naver_public_search_saturation_repair_raw",
                        "query_window": {"start": window["start"], "end": window["end"]},
                        "acquisition": {
                            "provider": "naver_public_search",
                            "query": f'"{window["name"]}"',
                            "response_rank": response_rank,
                            "result_cap": args.max_results_per_window,
                            "repair_for_provider": "google_rss",
                            "detected_google_result_cap": int(window["detected_result_cap"]),
                        },
                    }
                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    window_written += 1
                output_handle.flush()
            status = "failed" if error else ("incomplete_saturated" if saturated else "complete")
            failures += int(bool(error))
            saturated_windows += int(saturated)
            finished += 1
            coverage_handle.write(
                json.dumps(
                    {
                        **window,
                        "status": status,
                        "requests": window_requests,
                        "written": window_written,
                        "request_errors": int(bool(error)),
                        "saturated_leaf_windows": int(saturated),
                        "error": error,
                        "source": "naver_public_search_saturation_repair_coverage",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            coverage_handle.flush()
            print(
                json.dumps(
                    {
                        "finished": finished,
                        "remaining": len(plan) - len(completed) - finished,
                        "ticker": window["ticker"],
                        "window": [window["start"], window["end"]],
                        "status": status,
                        "written": window_written,
                        "elapsed_sec": round(time.perf_counter() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            if error and ("403" in error or "429" in error):
                break
    summary = {
        "detected_windows": len(plan),
        "already_completed": len(completed),
        "finished": finished,
        "written": written,
        "requests": requests_made,
        "failures": failures,
        "saturated_windows": saturated_windows,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if failures == 0 and saturated_windows == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

