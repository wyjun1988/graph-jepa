from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np
import pandas as pd

from stock_v2.news_contract import (
    LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY,
    NEWS_QUEUE_INPUT_HASH_POLICY,
    SUPPORTED_NEWS_QUEUE_ID_POLICIES,
    SUPPORTED_NEWS_QUEUE_INPUT_HASH_POLICIES,
    expected_news_queue_id,
    news_queue_input_sha256,
)


REQUIRED_OHLCV_COLUMNS = ("Open", "High", "Low", "Close", "Volume")
REQUIRED_INVESTOR_COLUMNS = (
    "investor_individual_net_m",
    "investor_foreign_net_m",
    "investor_institution_net_m",
)
TRACKING_QUERY_PREFIXES = ("utm_", "fbclid", "gclid", "ref", "source")


@dataclass(frozen=True)
class IntegrityIssue:
    severity: str
    code: str
    message: str
    details: dict[str, Any]


class IssueLog:
    def __init__(self) -> None:
        self._issues: list[IntegrityIssue] = []

    def add(
        self,
        severity: str,
        code: str,
        message: str,
        **details: Any,
    ) -> None:
        if severity not in {"blocker", "warning", "info"}:
            raise ValueError(f"unsupported issue severity: {severity}")
        self._issues.append(
            IntegrityIssue(
                severity=severity,
                code=code,
                message=message,
                details=details,
            )
        )

    @property
    def issues(self) -> list[IntegrityIssue]:
        return list(self._issues)

    def counts(self) -> dict[str, int]:
        counts = Counter(issue.severity for issue in self._issues)
        return {severity: int(counts.get(severity, 0)) for severity in ("blocker", "warning", "info")}


def normalize_ticker(value: Any) -> str:
    raw = str(value or "").strip()
    if raw.startswith("A") and raw[1:].isdigit():
        raw = raw[1:]
    return raw.zfill(6) if raw.isdigit() and len(raw) <= 6 else ""


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, role: str) -> dict[str, Any]:
    stat = path.stat()
    return {
        "role": role,
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "sha256": sha256_file(path),
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    empty_lines = 0
    invalid_json = 0
    invalid_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(value, dict):
                invalid_rows += 1
                continue
            rows.append(value)
    return rows, {
        "rows": len(rows),
        "empty_lines": empty_lines,
        "invalid_json": invalid_json,
        "invalid_rows": invalid_rows,
    }


def scan_jsonl(
    path: Path,
    visit: Callable[[dict[str, Any]], None],
) -> dict[str, int]:
    """Parse JSONL and visit object rows without retaining the whole file."""

    rows = 0
    empty_lines = 0
    invalid_json = 0
    invalid_rows = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                empty_lines += 1
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if not isinstance(value, dict):
                invalid_rows += 1
                continue
            rows += 1
            visit(value)
    return {
        "rows": rows,
        "empty_lines": empty_lines,
        "invalid_json": invalid_json,
        "invalid_rows": invalid_rows,
    }


def load_report_has_errors(report: Mapping[str, Any]) -> bool:
    return any(int(report.get(key, 0) or 0) > 0 for key in ("invalid_json", "invalid_rows"))


def finite_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if not len(array):
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "mean": float(array.mean()),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
    }


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def parse_timestamp(value: Any) -> tuple[pd.Timestamp | None, str]:
    raw = str(value or "").strip()
    if not raw:
        return None, "missing"
    precision = "date" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw) else "datetime"
    try:
        parsed = pd.to_datetime(raw, errors="raise", utc=True)
    except (ValueError, TypeError, OverflowError):
        return None, "invalid"
    return pd.Timestamp(parsed), precision


def clean_text(value: Any) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def canonical_url(value: Any) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parts = urlsplit(raw)
    except ValueError:
        return raw
    query = [
        (key, val)
        for key, val in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(TRACKING_QUERY_PREFIXES)
    ]
    return urlunsplit(
        (
            parts.scheme.lower(),
            parts.netloc.lower(),
            parts.path.rstrip("/"),
            urlencode(sorted(query)),
            "",
        )
    )


def article_fingerprint(record: Mapping[str, Any]) -> str:
    article = record.get("article") if isinstance(record.get("article"), Mapping) else {}
    title = clean_text(article.get("title") or record.get("title")).lower()
    source = clean_text(article.get("source") or record.get("source")).lower()
    published = str(article.get("published") or record.get("published") or "")[:10]
    url = canonical_url(article.get("link") or article.get("url") or record.get("url"))
    identity = url if url else "|".join((title, source, published))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest() if identity else ""


def _path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def audit_universe(
    repo_root: Path,
    config: Mapping[str, Any],
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    path = _path(repo_root, str(config["manifest"]))
    if not path.exists():
        issues.add("blocker", "universe_manifest_missing", "Universe manifest is missing.", path=str(path))
        return {"path": str(path), "rows": 0}, {}
    files.append(file_record(path, role="universe_manifest"))
    payload = load_json(path)
    rows = payload.get("universe", []) if isinstance(payload, Mapping) else []
    as_of = pd.Timestamp(config["as_of"]).normalize()
    expected_count = int(config.get("expected_count", len(rows)))
    tickers: dict[str, dict[str, Any]] = {}
    invalid_rows = 0
    duplicate_tickers: list[str] = []
    lifecycle_errors: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            invalid_rows += 1
            continue
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker:
            invalid_rows += 1
            continue
        if ticker in tickers:
            duplicate_tickers.append(ticker)
            continue
        listing = pd.to_datetime(row.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(row.get("delisting_date"), errors="coerce")
        if pd.isna(listing) or listing > as_of or (not pd.isna(delisting) and delisting < as_of):
            lifecycle_errors.append(ticker)
        tickers[ticker] = dict(row)
    if len(tickers) != expected_count:
        issues.add(
            "blocker",
            "universe_count_mismatch",
            "Frozen universe does not contain the required number of unique securities.",
            expected=expected_count,
            actual=len(tickers),
        )
    if invalid_rows or duplicate_tickers or lifecycle_errors:
        issues.add(
            "blocker",
            "universe_schema_invalid",
            "Universe contains invalid, duplicated, or out-of-lifecycle records.",
            invalid_rows=invalid_rows,
            duplicate_tickers=duplicate_tickers[:20],
            lifecycle_errors=lifecycle_errors[:20],
        )
    return (
        {
            "path": str(path),
            "schema_version": payload.get("schema_version") if isinstance(payload, Mapping) else None,
            "as_of": str(as_of.date()),
            "expected_count": expected_count,
            "unique_tickers": len(tickers),
            "invalid_rows": invalid_rows,
            "duplicate_tickers": duplicate_tickers,
            "lifecycle_errors": lifecycle_errors,
            "markets": dict(Counter(str(row.get("market") or "unknown") for row in tickers.values())),
            "delisted_after_as_of": int(
                sum(bool(row.get("delisting_date")) for row in tickers.values())
            ),
        },
        tickers,
    )


def load_trading_calendar(
    repo_root: Path,
    paths: Sequence[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> pd.DatetimeIndex:
    calendars: list[pd.DatetimeIndex] = []
    for raw_path in paths:
        path = _path(repo_root, raw_path)
        if not path.exists():
            issues.add("blocker", "calendar_source_missing", "Trading-calendar source is missing.", path=str(path))
            continue
        files.append(file_record(path, role="trading_calendar_source"))
        frame = pd.read_csv(path)
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        dates = pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.normalize()
        calendars.append(pd.DatetimeIndex(dates))
    if not calendars:
        return pd.DatetimeIndex([])
    calendar = calendars[0]
    for values in calendars[1:]:
        calendar = calendar.union(values)
    return calendar[(calendar >= start) & (calendar <= end)].sort_values().unique()


def _cache_candidates(cache_dir: Path, ticker: str) -> list[tuple[pd.Timestamp, pd.Timestamp, Path]]:
    pattern = re.compile(rf"^{re.escape(ticker)}_(\d{{8}})_(\d{{8}})\.csv$")
    candidates: list[tuple[pd.Timestamp, pd.Timestamp, Path]] = []
    for path in cache_dir.glob(f"{ticker}_*.csv"):
        match = pattern.fullmatch(path.name)
        if match is None:
            continue
        start = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
        end = pd.to_datetime(match.group(2), format="%Y%m%d", errors="coerce")
        if not pd.isna(start) and not pd.isna(end):
            candidates.append((pd.Timestamp(start), pd.Timestamp(end), path))
    return candidates


def select_ohlcv_cache(
    cache_dir: Path,
    ticker: str,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> tuple[Path | None, bool]:
    candidates = _cache_candidates(cache_dir, ticker)
    covering = [item for item in candidates if item[0] <= required_start and item[1] >= required_end]
    if covering:
        covering.sort(key=lambda item: (-int(item[1].value), int((required_start - item[0]).days), item[2].name))
        return covering[0][2], True
    if not candidates:
        return None, False
    candidates.sort(
        key=lambda item: (
            -max(0, (min(item[1], required_end) - max(item[0], required_start)).days),
            -int(item[1].value),
            item[2].name,
        )
    )
    return candidates[0][2], False


def audit_ohlcv(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_dir = _path(repo_root, str(config["cache_dir"]))
    start = pd.Timestamp(config["start"]).normalize()
    end = pd.Timestamp(config["end"]).normalize()
    missing_files: list[str] = []
    noncovering_files: list[str] = []
    schema_failures: list[str] = []
    partial_nonpositive: list[str] = []
    impossible_bars: list[str] = []
    adjusted_rounding_tickers: dict[str, int] = {}
    lifecycle_violations: list[str] = []
    duplicate_dates: list[str] = []
    extreme_return_tickers: dict[str, int] = {}
    coverage_by_ticker: dict[str, float] = {}
    observed_rows: list[float] = []
    unavailable_rows: list[float] = []
    carried_close_suspension_rows: list[float] = []
    missing_sessions: list[float] = []
    selected_paths: dict[str, str] = {}

    source = {
        "provider": str(config.get("provider") or "unknown"),
        "official": bool(config.get("official", False)),
        "price_basis": str(config.get("price_basis") or "unknown"),
    }
    if bool(config.get("require_primary_source", False)) and not source["official"]:
        issues.add(
            "blocker",
            "ohlcv_primary_source_missing",
            "OHLCV release is not backed by the configured primary exchange source.",
            provider=source["provider"],
        )
    if source["price_basis"] not in {"raw", "adjusted", "raw_and_adjusted"}:
        issues.add(
            "blocker",
            "ohlcv_price_basis_undefined",
            "OHLCV source does not declare whether prices are raw or corporate-action adjusted.",
            price_basis=source["price_basis"],
        )

    for ticker, metadata in universe.items():
        listing = pd.to_datetime(metadata.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(metadata.get("delisting_date"), errors="coerce")
        required_start = max(start, pd.Timestamp(listing).normalize()) if not pd.isna(listing) else start
        required_end = min(end, pd.Timestamp(delisting).normalize()) if not pd.isna(delisting) else end
        path, covers = select_ohlcv_cache(cache_dir, ticker, required_start, required_end)
        if path is None:
            missing_files.append(ticker)
            continue
        selected_paths[ticker] = str(path)
        files.append(file_record(path, role=f"ohlcv:{ticker}"))
        if not covers:
            noncovering_files.append(ticker)
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            schema_failures.append(ticker)
            continue
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        if not set(REQUIRED_OHLCV_COLUMNS).issubset(frame.columns):
            schema_failures.append(ticker)
            continue
        dates = pd.to_datetime(frame[date_column], errors="coerce").dt.normalize()
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            duplicate_dates.append(ticker)
        numeric = frame[list(REQUIRED_OHLCV_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        prices = numeric[["Open", "High", "Low", "Close"]]
        all_positive = prices.gt(0.0).all(axis=1) & np.isfinite(prices).all(axis=1)
        all_zero = prices.eq(0.0).all(axis=1)
        carried_close_suspension = (
            prices[["Open", "High", "Low"]].eq(0.0).all(axis=1)
            & prices["Close"].gt(0.0)
            & numeric["Volume"].eq(0.0)
        )
        unavailable = all_zero | carried_close_suspension
        partial_bad = ~(all_positive | unavailable)
        if partial_bad.any():
            partial_nonpositive.append(ticker)
        observed = numeric.loc[all_positive]
        upper = observed[["Open", "Close", "Low"]].max(axis=1)
        lower = observed[["Open", "Close", "High"]].min(axis=1)
        high_gap = ((upper - observed["High"]) / upper.replace(0.0, np.nan)).clip(lower=0.0)
        low_gap = ((observed["Low"] - lower) / lower.replace(0.0, np.nan)).clip(lower=0.0)
        bar_gap = pd.concat((high_gap, low_gap), axis=1).max(axis=1)
        tolerance = float(config.get("bar_tolerance_ratio", 0.0))
        impossible = (
            (bar_gap > tolerance)
            | (~np.isfinite(observed["Volume"]))
            | (observed["Volume"] < 0.0)
        )
        if impossible.any():
            impossible_bars.append(ticker)
        rounding_count = int(((bar_gap > 0.0) & (bar_gap <= tolerance)).sum())
        if rounding_count:
            adjusted_rounding_tickers[ticker] = rounding_count
        observed_dates = dates[all_positive & dates.notna()]
        if not pd.isna(listing) and (observed_dates < pd.Timestamp(listing).normalize()).any():
            lifecycle_violations.append(ticker)
        if not pd.isna(delisting) and (observed_dates > pd.Timestamp(delisting).normalize()).any():
            lifecycle_violations.append(ticker)
        expected = calendar[(calendar >= required_start) & (calendar <= required_end)]
        present = pd.DatetimeIndex(dates.dropna())
        missing = expected.difference(present)
        observed_in_life = pd.DatetimeIndex(observed_dates).intersection(expected)
        coverage_by_ticker[ticker] = ratio(len(observed_in_life), len(expected))
        observed_rows.append(float(len(observed_in_life)))
        unavailable_rows.append(float(unavailable.sum()))
        carried_close_suspension_rows.append(float(carried_close_suspension.sum()))
        missing_sessions.append(float(len(missing)))
        close = observed["Close"].astype(float)
        raw_returns = close.pct_change()
        extreme_count = int((raw_returns.abs() > float(config.get("extreme_return_threshold", 0.35))).sum())
        if extreme_count:
            extreme_return_tickers[ticker] = extreme_count

    if missing_files:
        issues.add(
            "blocker",
            "ohlcv_universe_incomplete",
            "OHLCV does not cover every frozen-universe security.",
            missing_count=len(missing_files),
            tickers=missing_files[:50],
        )
    if noncovering_files:
        issues.add(
            "blocker",
            "ohlcv_range_incomplete",
            "Some OHLCV cache files do not cover the security lifecycle in the release window.",
            count=len(noncovering_files),
            tickers=noncovering_files[:50],
        )
    if schema_failures or partial_nonpositive or impossible_bars or duplicate_dates or lifecycle_violations:
        issues.add(
            "blocker",
            "ohlcv_integrity_failure",
            "OHLCV contains schema, bar, date, or lifecycle violations.",
            schema_failures=schema_failures[:30],
            partial_nonpositive=partial_nonpositive[:30],
            impossible_bars=impossible_bars[:30],
            duplicate_dates=duplicate_dates[:30],
            lifecycle_violations=lifecycle_violations[:30],
        )
    low_coverage = sorted(
        ((ticker, value) for ticker, value in coverage_by_ticker.items() if value < 0.50),
        key=lambda item: item[1],
    )
    if low_coverage:
        issues.add(
            "warning",
            "ohlcv_low_observed_coverage",
            "Some lifecycle windows contain mostly suspended or unavailable prices.",
            count=len(low_coverage),
            sample=low_coverage[:30],
        )
    if extreme_return_tickers:
        issues.add(
            "warning",
            "ohlcv_corporate_action_candidates",
            "Extreme close-to-close moves require corporate-action reconciliation.",
            ticker_count=len(extreme_return_tickers),
            sample=dict(list(sorted(extreme_return_tickers.items()))[:30]),
        )
    if adjusted_rounding_tickers:
        issues.add(
            "warning",
            "ohlcv_adjusted_rounding_detected",
            "Small adjusted-price rounding mismatches are present within the declared tolerance.",
            ticker_count=len(adjusted_rounding_tickers),
            row_count=sum(adjusted_rounding_tickers.values()),
            tolerance_ratio=float(config.get("bar_tolerance_ratio", 0.0)),
        )
    return {
        "cache_dir": str(cache_dir),
        "source": source,
        "expected_tickers": len(universe),
        "selected_files": len(selected_paths),
        "missing_tickers": missing_files,
        "noncovering_tickers": noncovering_files,
        "schema_failure_tickers": schema_failures,
        "partial_nonpositive_tickers": partial_nonpositive,
        "impossible_bar_tickers": impossible_bars,
        "adjusted_rounding_tickers": adjusted_rounding_tickers,
        "duplicate_date_tickers": duplicate_dates,
        "lifecycle_violation_tickers": lifecycle_violations,
        "observed_rows": finite_summary(observed_rows),
        "unavailable_zero_price_rows": finite_summary(unavailable_rows),
        "carried_close_suspension_rows": finite_summary(carried_close_suspension_rows),
        "missing_calendar_sessions": finite_summary(missing_sessions),
        "observed_lifecycle_coverage": finite_summary(list(coverage_by_ticker.values())),
        "lowest_coverage": low_coverage[:50],
        "corporate_action_candidates": extreme_return_tickers,
        "selected_paths": selected_paths,
    }


def audit_news(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    if str(config.get("format") or "legacy") == "canonical_v1":
        return audit_canonical_news(repo_root, config, universe, issues, files)
    raw_records: list[dict[str, Any]] = []
    raw_load_reports: dict[str, Any] = {}
    raw_load_errors = 0
    for value in config.get("raw_paths", []):
        path = _path(repo_root, value)
        if not path.exists():
            issues.add("blocker", "news_raw_missing", "Configured raw-news file is missing.", path=str(path))
            continue
        files.append(file_record(path, role="news_raw"))
        rows, load_report = load_jsonl(path)
        raw_records.extend(rows)
        raw_load_reports[str(path)] = load_report
        raw_load_errors += int(load_report["invalid_json"]) + int(load_report["invalid_rows"])

    structured_records: list[dict[str, Any]] = []
    structured_load_reports: dict[str, Any] = {}
    structured_load_errors = 0
    for value in config.get("structured_paths", []):
        path = _path(repo_root, value)
        if not path.exists():
            issues.add("blocker", "news_structured_missing", "Configured structured-news file is missing.", path=str(path))
            continue
        files.append(file_record(path, role="news_structured"))
        rows, load_report = load_jsonl(path)
        structured_records.extend(rows)
        structured_load_reports[str(path)] = load_report
        structured_load_errors += int(load_report["invalid_json"]) + int(load_report["invalid_rows"])

    def contract_date(keys: Sequence[str], *, field: str) -> pd.Timestamp:
        raw = next(
            (
                config.get(key)
                for key in keys
                if config.get(key) is not None and str(config.get(key)).strip()
            ),
            None,
        )
        parsed = pd.to_datetime(raw, errors="coerce", utc=True)
        if pd.isna(parsed):
            issues.add(
                "blocker",
                "news_contract_window_invalid",
                "News dataset contract is missing a valid audit window.",
                field=field,
                accepted_keys=list(keys),
                value=raw,
            )
            return pd.Timestamp("1970-01-01")
        return pd.Timestamp(parsed).tz_convert(None).normalize()

    coverage_tickers: set[str] = set()
    coverage_rows = 0
    failed_coverage_rows = 0
    incomplete_coverage_rows = 0
    coverage_load_errors = 0
    coverage_start = contract_date(
        ("coverage_start", "release_start", "start"),
        field="coverage_start",
    )
    coverage_end = contract_date(
        ("coverage_end", "release_end", "end"),
        field="coverage_end",
    )
    if coverage_end < coverage_start:
        issues.add(
            "blocker",
            "news_contract_window_invalid",
            "News dataset contract ends before it starts.",
            coverage_start=coverage_start.date().isoformat(),
            coverage_end=coverage_end.date().isoformat(),
        )
        coverage_end = coverage_start
    for value in config.get("coverage_paths", []):
        path = _path(repo_root, value)
        if not path.exists():
            issues.add("blocker", "news_coverage_ledger_missing", "Configured news-coverage ledger is missing.", path=str(path))
            continue
        files.append(file_record(path, role="news_coverage_ledger"))
        rows, load_report = load_jsonl(path)
        coverage_load_errors += int(load_report["invalid_json"]) + int(load_report["invalid_rows"])
        for row in rows:
            ticker = normalize_ticker(row.get("ticker"))
            if ticker:
                coverage_tickers.add(ticker)
            coverage_rows += 1
            if int(row.get("request_errors", 0) or 0) > 0:
                failed_coverage_rows += 1
            row_start = pd.to_datetime(row.get("start"), errors="coerce")
            row_end = pd.to_datetime(row.get("end"), errors="coerce")
            window_days = int(row.get("window_days", 0) or 0)
            requests = int(row.get("requests", 0) or 0)
            expected_requests = (
                int(math.ceil(((coverage_end - coverage_start).days + 1) / window_days))
                if window_days > 0
                else 0
            )
            if (
                pd.isna(row_start)
                or pd.isna(row_end)
                or row_start.normalize() > coverage_start
                or row_end.normalize() < coverage_end
                or expected_requests <= 0
                or requests < expected_requests
            ):
                incomplete_coverage_rows += 1

    universe_set = set(universe)
    article_ids = Counter()
    mapping_keys = Counter()
    unique_articles: set[str] = set()
    article_tickers: set[str] = set()
    invalid_ticker_rows = 0
    outside_universe_rows = 0
    invalid_published_rows = 0
    outside_release_window_rows = 0
    date_only_rows = 0
    precise_timestamp_rows = 0
    short_text_rows = 0
    future_timestamp_rows = 0
    per_ticker_year: Counter[tuple[str, int]] = Counter()
    now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
    min_text_chars = int(config.get("min_text_chars", 80))
    release_start = contract_date(
        ("release_start", "start", "coverage_start"),
        field="release_start",
    ).tz_localize("UTC")
    release_end = (
        contract_date(
            ("release_end", "end", "coverage_end"),
            field="release_end",
        )
        + pd.Timedelta(days=1)
    ).tz_localize("UTC")

    for record in raw_records:
        ticker = normalize_ticker(record.get("ticker"))
        if not ticker:
            invalid_ticker_rows += 1
        else:
            article_tickers.add(ticker)
            if ticker not in universe_set:
                outside_universe_rows += 1
        article = record.get("article") if isinstance(record.get("article"), Mapping) else {}
        published_raw = article.get("published") or record.get("published")
        published, precision = parse_timestamp(published_raw)
        if published is None:
            invalid_published_rows += 1
        else:
            if precision == "date":
                date_only_rows += 1
            else:
                precise_timestamp_rows += 1
            if published > now:
                future_timestamp_rows += 1
            if published < release_start or published >= release_end:
                outside_release_window_rows += 1
            if ticker:
                per_ticker_year[(ticker, int(published.year))] += 1
        title = clean_text(article.get("title") or record.get("title"))
        summary = clean_text(article.get("summary") or article.get("body") or record.get("summary"))
        if len(title) + len(summary) < min_text_chars:
            short_text_rows += 1
        record_id = str(record.get("id") or "").strip()
        if record_id:
            article_ids[record_id] += 1
        fingerprint = article_fingerprint(record)
        if fingerprint:
            unique_articles.add(fingerprint)
            mapping_keys[(ticker, fingerprint)] += 1

    duplicate_ids = sum(count - 1 for count in article_ids.values() if count > 1)
    duplicate_mappings = sum(count - 1 for count in mapping_keys.values() if count > 1)
    precise_ratio = ratio(precise_timestamp_rows, len(raw_records))
    coverage_ratio = ratio(len(coverage_tickers & universe_set), len(universe_set))
    article_ticker_ratio = ratio(len(article_tickers & universe_set), len(universe_set))

    llm_rows = 0
    heuristic_rows = 0
    llm_error_rows = 0
    schema_error_rows = 0
    event_value_error_rows = 0
    missing_model_metadata = 0
    structured_ids: set[str] = set()
    structured_id_counts: Counter[str] = Counter()
    orphan_structured_ids = 0
    for record in structured_records:
        record_id = str(record.get("id") or "").strip()
        if record_id:
            structured_ids.add(record_id)
            structured_id_counts[record_id] += 1
        llm_used = bool(record.get("llm_used"))
        event = record.get("event") if isinstance(record.get("event"), Mapping) else {}
        if llm_used:
            llm_rows += 1
            lineage = record.get("lineage") if isinstance(record.get("lineage"), Mapping) else {}
            rescore = record.get("rescore") if isinstance(record.get("rescore"), Mapping) else {}
            if not str(
                record.get("model")
                or event.get("model")
                or event.get("model_name")
                or lineage.get("model")
                or lineage.get("model_id")
                or rescore.get("model")
                or ""
            ).strip():
                raw = event.get("raw") if isinstance(event.get("raw"), Mapping) else {}
                if not str(raw.get("model") or raw.get("model_name") or "").strip():
                    missing_model_metadata += 1
        else:
            heuristic_rows += 1
        if str(record.get("llm_error") or "").strip():
            llm_error_rows += 1
        required = ("event_type", "polarity", "magnitude", "confidence", "horizon_days", "node_deltas")
        if any(key not in event for key in required):
            schema_error_rows += 1
            continue
        try:
            polarity = float(event["polarity"])
            magnitude = float(event["magnitude"])
            confidence = float(event["confidence"])
            horizon_days = int(event["horizon_days"])
            node_deltas = event["node_deltas"]
            valid_values = (
                all(math.isfinite(value) for value in (polarity, magnitude, confidence))
                and -1.0 <= polarity <= 1.0
                and 0.0 <= magnitude <= 1.0
                and 0.0 <= confidence <= 1.0
                and 1 <= horizon_days <= int(config.get("max_horizon_days", 365))
                and isinstance(node_deltas, Sequence)
                and not isinstance(node_deltas, (str, bytes))
            )
        except (TypeError, ValueError, OverflowError):
            valid_values = False
        if not valid_values:
            event_value_error_rows += 1

    raw_id_set = set(article_ids)
    orphan_structured_ids = len(structured_ids - raw_id_set)
    duplicate_structured_ids = sum(count - 1 for count in structured_id_counts.values() if count > 1)

    llm_ratio = ratio(llm_rows, len(structured_records))
    structured_raw_id_ratio = ratio(len(structured_ids & set(article_ids)), len(set(article_ids)))
    min_ledger_ratio = float(config.get("min_coverage_ledger_ratio", 0.95))
    min_precise_ratio = float(config.get("min_precise_timestamp_ratio", 0.90))
    min_llm_ratio = float(config.get("min_llm_ratio", 0.95))
    min_structured_ratio = float(config.get("min_structured_raw_id_ratio", 0.95))
    if raw_load_errors or structured_load_errors or coverage_load_errors:
        issues.add(
            "blocker",
            "news_jsonl_corrupt",
            "A news JSONL source contains malformed JSON or non-object rows.",
            raw_errors=raw_load_errors,
            structured_errors=structured_load_errors,
            coverage_errors=coverage_load_errors,
        )
    if coverage_ratio < min_ledger_ratio or failed_coverage_rows or incomplete_coverage_rows:
        issues.add(
            "blocker",
            "news_collection_incomplete",
            "News collection has no successful point-in-time coverage ledger for most universe securities.",
            ledger_ticker_ratio=coverage_ratio,
            required=min_ledger_ratio,
            failed_coverage_rows=failed_coverage_rows,
            incomplete_coverage_rows=incomplete_coverage_rows,
        )
    if precise_ratio < min_precise_ratio:
        issues.add(
            "blocker",
            "news_timestamp_precision_insufficient",
            "Too many news records have only a date or an invalid publication timestamp.",
            precise_ratio=precise_ratio,
            required=min_precise_ratio,
        )
    if (
        llm_ratio < min_llm_ratio
        or structured_raw_id_ratio < min_structured_ratio
        or schema_error_rows
        or event_value_error_rows
        or llm_error_rows
        or orphan_structured_ids
        or duplicate_structured_ids
    ):
        issues.add(
            "blocker",
            "news_structuring_incomplete",
            "Structured news has insufficient actual LLM coverage or schema failures.",
            llm_ratio=llm_ratio,
            required=min_llm_ratio,
            structured_raw_id_ratio=structured_raw_id_ratio,
            required_structured_raw_id_ratio=min_structured_ratio,
            schema_error_rows=schema_error_rows,
            event_value_error_rows=event_value_error_rows,
            llm_error_rows=llm_error_rows,
            orphan_structured_ids=orphan_structured_ids,
            duplicate_structured_ids=duplicate_structured_ids,
        )
    if duplicate_ids or duplicate_mappings:
        issues.add(
            "warning",
            "news_duplicates_present",
            "Raw news contains duplicate IDs or repeated ticker/article mappings.",
            duplicate_ids=duplicate_ids,
            duplicate_mappings=duplicate_mappings,
        )
    if (
        invalid_ticker_rows
        or outside_universe_rows
        or invalid_published_rows
        or outside_release_window_rows
        or future_timestamp_rows
    ):
        issues.add(
            "blocker",
            "news_record_integrity_failure",
            "Raw news contains invalid ticker or publication-time records.",
            invalid_ticker_rows=invalid_ticker_rows,
            outside_universe_rows=outside_universe_rows,
            invalid_published_rows=invalid_published_rows,
            outside_release_window_rows=outside_release_window_rows,
            future_timestamp_rows=future_timestamp_rows,
        )
    if short_text_rows:
        issues.add(
            "warning",
            "news_text_too_short",
            "Some records do not contain enough title/summary text for reliable event extraction.",
            count=short_text_rows,
            ratio=ratio(short_text_rows, len(raw_records)),
        )
    if missing_model_metadata:
        issues.add(
            "blocker",
            "news_model_lineage_missing",
            "LLM-derived records do not identify the model/version used.",
            rows=missing_model_metadata,
        )

    ticker_year_counts = list(per_ticker_year.values())
    return {
        "raw_load_reports": raw_load_reports,
        "structured_load_reports": structured_load_reports,
        "raw_rows": len(raw_records),
        "unique_articles": len(unique_articles),
        "unique_article_ids": len(article_ids),
        "duplicate_ids": duplicate_ids,
        "duplicate_ticker_article_mappings": duplicate_mappings,
        "article_tickers_in_universe": len(article_tickers & universe_set),
        "article_ticker_ratio": article_ticker_ratio,
        "coverage_ledger_rows": coverage_rows,
        "coverage_ledger_tickers": len(coverage_tickers & universe_set),
        "coverage_ledger_ticker_ratio": coverage_ratio,
        "failed_coverage_rows": failed_coverage_rows,
        "incomplete_coverage_rows": incomplete_coverage_rows,
        "raw_load_errors": raw_load_errors,
        "structured_load_errors": structured_load_errors,
        "coverage_load_errors": coverage_load_errors,
        "invalid_ticker_rows": invalid_ticker_rows,
        "outside_universe_rows": outside_universe_rows,
        "invalid_published_rows": invalid_published_rows,
        "date_only_rows": date_only_rows,
        "precise_timestamp_rows": precise_timestamp_rows,
        "precise_timestamp_ratio": precise_ratio,
        "future_timestamp_rows": future_timestamp_rows,
        "outside_release_window_rows": outside_release_window_rows,
        "short_text_rows": short_text_rows,
        "ticker_year_article_counts": finite_summary([float(value) for value in ticker_year_counts]),
        "structured_rows": len(structured_records),
        "structured_raw_id_ratio": structured_raw_id_ratio,
        "llm_rows": llm_rows,
        "heuristic_rows": heuristic_rows,
        "llm_ratio": llm_ratio,
        "llm_error_rows": llm_error_rows,
        "schema_error_rows": schema_error_rows,
        "event_value_error_rows": event_value_error_rows,
        "orphan_structured_ids": orphan_structured_ids,
        "duplicate_structured_ids": duplicate_structured_ids,
        "missing_model_metadata_rows": missing_model_metadata,
    }


def audit_canonical_news(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    def load_paths(key: str, role: str) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
        loaded: list[dict[str, Any]] = []
        reports: dict[str, Any] = {}
        errors = 0
        for value in config.get(key, []):
            path = _path(repo_root, value)
            if not path.exists():
                issues.add("blocker", f"{role}_missing", f"Configured {role} file is missing.", path=str(path))
                continue
            files.append(file_record(path, role=role))
            rows, report = load_jsonl(path)
            loaded.extend(rows)
            reports[str(path)] = report
            errors += int(report["invalid_json"]) + int(report["invalid_rows"])
        return loaded, reports, errors

    occurrence_paths: list[Path] = []
    occurrence_load_reports: dict[str, Any] = {}
    for value in config.get("occurrence_paths", []):
        path = _path(repo_root, value)
        if not path.exists():
            issues.add(
                "blocker",
                "news_occurrences_missing",
                "Configured news_occurrences file is missing.",
                path=str(path),
            )
            continue
        files.append(file_record(path, role="news_occurrences"))
        occurrence_paths.append(path)
    articles, article_load_reports, article_load_errors = load_paths("raw_paths", "news_articles")
    mappings, mapping_load_reports, mapping_load_errors = load_paths("mapping_paths", "news_mappings")
    queue, queue_load_reports, queue_load_errors = load_paths("queue_paths", "news_queue")
    structured, structured_load_reports, structured_load_errors = load_paths(
        "structured_paths", "news_structured"
    )
    neutral_events, neutral_load_reports, neutral_load_errors = load_paths(
        "neutral_event_paths", "news_neutral_events"
    )
    coverage, coverage_load_reports, coverage_load_errors = load_paths(
        "coverage_paths", "news_coverage"
    )
    coverage_windows, coverage_window_load_reports, coverage_window_load_errors = load_paths(
        "coverage_window_paths", "news_coverage_windows"
    )
    load_errors = (
        article_load_errors
        + mapping_load_errors
        + queue_load_errors
        + structured_load_errors
        + neutral_load_errors
        + coverage_load_errors
        + coverage_window_load_errors
    )
    universe_set = set(universe)
    release_start = pd.Timestamp(config["release_start"]).normalize()
    release_end = pd.Timestamp(config["release_end"]).normalize()
    article_by_id: dict[str, dict[str, Any]] = {}
    duplicate_article_ids = 0
    invalid_article_rows = 0
    outside_release_window_rows = 0
    date_only_rows = 0
    precise_timestamp_rows = 0
    title_only_rows = 0
    article_source_providers: Counter[str] = Counter()
    article_content_tiers: Counter[str] = Counter()
    point_in_time_selection_rows = 0
    retrospective_or_unknown_selection_rows = 0
    unsafe_availability_rows = 0
    not_yet_effective_rows = 0
    for row in articles:
        article_id = str(row.get("article_id") or "").strip()
        if not article_id or article_id in article_by_id:
            duplicate_article_ids += int(bool(article_id and article_id in article_by_id))
            invalid_article_rows += int(not article_id)
            continue
        published_date = pd.to_datetime(row.get("published_date_kst"), errors="coerce")
        precision = str(row.get("published_precision") or "")
        effective = pd.to_datetime(row.get("effective_session"), errors="coerce")
        if pd.isna(published_date) or precision not in {"date", "datetime"} or not str(row.get("title") or "").strip():
            invalid_article_rows += 1
            continue
        if published_date.normalize() < release_start or published_date.normalize() > release_end:
            outside_release_window_rows += 1
        if precision == "date":
            date_only_rows += 1
        else:
            precise_timestamp_rows += 1
        if str(row.get("content_tier")) == "title_only":
            title_only_rows += 1
        article_source_providers[str(row.get("source_provider") or "unknown")] += 1
        article_content_tiers[str(row.get("content_tier") or "unknown")] += 1
        if bool(row.get("selection_point_in_time")):
            point_in_time_selection_rows += 1
        else:
            retrospective_or_unknown_selection_rows += 1
        if pd.isna(effective):
            not_yet_effective_rows += 1
        elif (
            effective.normalize() <= published_date.normalize()
            or str(row.get("availability_policy")) != "next_krx_session"
        ):
            unsafe_availability_rows += 1
        article_by_id[article_id] = row

    mapping_keys: set[tuple[str, str]] = set()
    mapping_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_mappings = 0
    invalid_mapping_rows = 0
    lifecycle_mapping_rows = 0
    mapping_tickers: set[str] = set()
    mapping_methods: Counter[str] = Counter()
    relevance_required_mappings = 0
    for row in mappings:
        article_id = str(row.get("article_id") or "").strip()
        ticker = normalize_ticker(row.get("ticker"))
        key = (article_id, ticker)
        if not article_id or article_id not in article_by_id or ticker not in universe_set:
            invalid_mapping_rows += 1
            continue
        if key in mapping_keys:
            duplicate_mappings += 1
            continue
        mapping_keys.add(key)
        mapping_by_key[key] = row
        mapping_tickers.add(ticker)
        mapping_methods[str(row.get("mapping_method") or "unknown")] += 1
        relevance_required_mappings += int(bool(row.get("requires_relevance_classification")))
        published = pd.Timestamp(article_by_id[article_id]["published_date_kst"]).normalize()
        listing = pd.to_datetime(universe[ticker].get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(universe[ticker].get("delisting_date"), errors="coerce")
        if (
            (not pd.isna(listing) and published < pd.Timestamp(listing).normalize())
            or (not pd.isna(delisting) and published > pd.Timestamp(delisting).normalize())
        ):
            lifecycle_mapping_rows += 1

    occurrence_ids: set[str] = set()
    duplicate_occurrence_ids = 0
    invalid_occurrence_rows = 0
    invalid_occurrence_query_windows = 0
    outside_occurrence_query_window_rows = 0
    occurrence_rows = 0
    occurrence_acquisition_modes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    occurrence_pit_acquisition_modes: defaultdict[str, Counter[str]] = defaultdict(Counter)
    occurrence_pit_counts: Counter[str] = Counter()
    occurrence_non_pit_counts: Counter[str] = Counter()
    earliest_live_observation: dict[str, pd.Timestamp] = {}

    def audit_occurrence(row: dict[str, Any]) -> None:
        nonlocal duplicate_occurrence_ids
        nonlocal invalid_occurrence_rows
        nonlocal invalid_occurrence_query_windows
        nonlocal occurrence_rows
        nonlocal outside_occurrence_query_window_rows
        occurrence_rows += 1
        occurrence_id = str(row.get("occurrence_id") or "").strip()
        article_id = str(row.get("article_id") or "").strip()
        ticker = normalize_ticker(row.get("ticker"))
        if not occurrence_id:
            invalid_occurrence_rows += 1
            return
        if occurrence_id in occurrence_ids:
            duplicate_occurrence_ids += 1
            return
        occurrence_ids.add(occurrence_id)
        try:
            source_row_number = int(row.get("source_row_number", 0) or 0)
        except (TypeError, ValueError):
            source_row_number = 0
        if (
            (article_id, ticker) not in mapping_keys
            or not str(row.get("source_path") or "").strip()
            or source_row_number < 1
            or not str(row.get("acquisition_mode") or "").strip()
            or not isinstance(row.get("selection_point_in_time"), bool)
        ):
            invalid_occurrence_rows += 1
            return
        acquisition_mode = str(row.get("acquisition_mode") or "")
        selection_point_in_time = bool(row.get("selection_point_in_time"))
        occurrence_acquisition_modes[article_id][acquisition_mode] += 1
        if selection_point_in_time:
            occurrence_pit_counts[article_id] += 1
            occurrence_pit_acquisition_modes[article_id][acquisition_mode] += 1
        else:
            occurrence_non_pit_counts[article_id] += 1
        if acquisition_mode == "live_capture" and selection_point_in_time:
            collected = pd.to_datetime(
                row.get("collected_at_raw"), errors="coerce", utc=True
            )
            if pd.isna(collected):
                invalid_occurrence_rows += 1
                return
            collected_timestamp = pd.Timestamp(collected)
            previous = earliest_live_observation.get(article_id)
            if previous is None or collected_timestamp < previous:
                earliest_live_observation[article_id] = collected_timestamp
        query_window = row.get("query_window")
        if isinstance(query_window, Mapping):
            query_start = pd.to_datetime(query_window.get("start"), errors="coerce")
            query_end = pd.to_datetime(query_window.get("end"), errors="coerce")
            if pd.isna(query_start) or pd.isna(query_end) or query_start >= query_end:
                invalid_occurrence_query_windows += 1
                return
            published_date = pd.Timestamp(article_by_id[article_id]["published_date_kst"]).normalize()
            if (
                published_date < pd.Timestamp(query_start).normalize()
                or published_date >= pd.Timestamp(query_end).normalize()
            ):
                outside_occurrence_query_window_rows += 1

    occurrence_load_errors = 0
    for path in occurrence_paths:
        report = scan_jsonl(path, audit_occurrence)
        occurrence_load_reports[str(path)] = report
        occurrence_load_errors += int(report["invalid_json"]) + int(
            report["invalid_rows"]
        )
    load_errors += occurrence_load_errors
    if load_errors:
        issues.add(
            "blocker",
            "news_jsonl_corrupt",
            "A canonical news JSONL source contains malformed JSON or non-object rows.",
            errors=load_errors,
        )

    article_occurrence_lineage_mismatch_rows = 0
    unsafe_live_observation_availability_rows = 0
    if occurrence_paths:
        for article_id, article in article_by_id.items():
            pit_count = int(occurrence_pit_counts[article_id])
            non_pit_count = int(occurrence_non_pit_counts[article_id])
            expected_modes = (
                occurrence_pit_acquisition_modes[article_id]
                if pit_count
                else occurrence_acquisition_modes[article_id]
            )
            article_occurrence_lineage_mismatch_rows += int(
                not occurrence_acquisition_modes[article_id]
                or bool(article.get("selection_point_in_time")) != bool(pit_count)
                or int(article.get("pit_occurrence_count", 0) or 0) != pit_count
                or int(
                    article.get("retrospective_or_unknown_occurrence_count", 0)
                    or 0
                )
                != non_pit_count
                or dict(article.get("acquisition_modes") or {})
                != dict(sorted(expected_modes.items()))
            )
        for article_id, observed in earliest_live_observation.items():
            effective = pd.to_datetime(
                article_by_id[article_id].get("effective_session"), errors="coerce"
            )
            if pd.isna(effective):
                continue
            observed_date = (
                observed.tz_convert("Asia/Seoul").normalize().tz_localize(None)
            )
            unsafe_live_observation_availability_rows += int(
                pd.Timestamp(effective).normalize() <= observed_date
            )

    queue_by_id: dict[str, dict[str, Any]] = {}
    duplicate_queue_ids = 0
    invalid_queue_rows = 0
    queue_contract_mismatch_rows = 0
    queue_input_hash_mismatches = 0
    queue_input_hash_policies: Counter[str] = Counter()
    allowed_queue_hash_policies = {
        str(value)
        for value in config.get(
            "allowed_queue_input_hash_policies", [NEWS_QUEUE_INPUT_HASH_POLICY]
        )
    }
    unsupported_configured_queue_hash_policies = (
        allowed_queue_hash_policies - SUPPORTED_NEWS_QUEUE_INPUT_HASH_POLICIES
    )
    disallowed_queue_input_hash_rows = 0
    for row in queue:
        queue_id = str(row.get("queue_id") or "").strip()
        article_id = str(row.get("article_id") or "").strip()
        ticker = normalize_ticker(row.get("ticker"))
        effective = pd.to_datetime(row.get("effective_session"), errors="coerce")
        if queue_id in queue_by_id:
            duplicate_queue_ids += 1
            continue
        if (
            not queue_id
            or (article_id, ticker) not in mapping_keys
            or pd.isna(effective)
            or str(effective.date()) != str(article_by_id.get(article_id, {}).get("effective_session") or "")
            or not str(row.get("input_sha256") or "").strip()
        ):
            invalid_queue_rows += 1
            continue
        article = article_by_id[article_id]
        mapping = mapping_by_key[(article_id, ticker)]
        input_hash_policy = str(
            row.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
        )
        queue_input_hash_policies[input_hash_policy] += 1
        disallowed_queue_input_hash_rows += int(
            input_hash_policy not in allowed_queue_hash_policies
        )
        try:
            mapping_confidence_matches = abs(
                float(row.get("mapping_confidence"))
                - float(mapping.get("mapping_confidence"))
            ) <= 1e-12
        except (TypeError, ValueError, OverflowError):
            mapping_confidence_matches = False
        article_fields = (
            "title",
            "summary",
            "source",
            "published_date_kst",
            "published_precision",
            "content_tier",
        )
        if not row.get("semantic_cluster_id"):
            article_fields = ("event_cluster_id", "cluster_size", *article_fields)
        mapping_fields = (
            "company_name",
            "mapping_method",
            "matched_alias",
            "matched_alias_type",
            "matched_alias_source",
            "matched_alias_ambiguous",
        )
        try:
            queue_identity_matches = queue_id == expected_news_queue_id(row)
        except ValueError:
            queue_identity_matches = False
        hash_policy_fields_valid = (
            input_hash_policy == LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
            or (
                row.get("input_hash_policy") == NEWS_QUEUE_INPUT_HASH_POLICY
                and row.get("queue_identity_policy")
                in SUPPORTED_NEWS_QUEUE_ID_POLICIES
            )
        )
        queue_contract_matches = (
            queue_identity_matches
            and hash_policy_fields_valid
            and all(row.get(field) == article.get(field) for field in article_fields)
            and all(row.get(field) == mapping.get(field) for field in mapping_fields)
            and mapping_confidence_matches
            and isinstance(row.get("acquisition_modes"), Mapping)
            and isinstance(row.get("selection_point_in_time"), bool)
        )
        queue_contract_mismatch_rows += int(not queue_contract_matches)
        try:
            input_hash_matches = str(row.get("input_sha256") or "") == news_queue_input_sha256(
                row
            )
        except (TypeError, ValueError):
            input_hash_matches = False
        queue_input_hash_mismatches += int(not input_hash_matches)
        queue_by_id[queue_id] = row

    coverage_tickers: set[str] = set()
    coverage_tickers_by_source: defaultdict[str, set[str]] = defaultdict(set)
    incomplete_coverage_rows = 0
    failed_coverage_rows = 0
    for row in coverage:
        ticker = normalize_ticker(row.get("ticker"))
        start = pd.to_datetime(row.get("start"), errors="coerce")
        end = pd.to_datetime(row.get("end"), errors="coerce")
        if int(row.get("request_errors", 0) or 0) > 0:
            failed_coverage_rows += 1
            continue
        if (
            ticker not in universe_set
            or pd.isna(start)
            or pd.isna(end)
            or start.normalize() > release_start
            or end.normalize() < release_end
            or str(row.get("status") or "") != "complete"
        ):
            incomplete_coverage_rows += 1
            continue
        coverage_tickers.add(ticker)
        source_text = str(row.get("source") or "").lower()
        coverage_source = (
            "opendart"
            if "opendart" in source_text
            else "google_rss"
            if "google" in source_text
            else "naver_search"
            if "naver" in source_text
            else source_text or "unknown"
        )
        coverage_tickers_by_source[coverage_source].add(ticker)

    coverage_window_ids: set[str] = set()
    coverage_windows_by_source_ticker: defaultdict[
        tuple[str, str], list[tuple[pd.Timestamp, pd.Timestamp, str]]
    ] = defaultdict(list)
    invalid_coverage_window_rows = 0
    duplicate_coverage_window_ids = 0
    complete_coverage_window_rows = 0
    saturated_coverage_window_rows = 0
    for row in coverage_windows:
        window_id = str(row.get("window_id") or "").strip()
        ticker = normalize_ticker(row.get("ticker"))
        source = str(row.get("provider") or row.get("source") or "unknown").lower()
        source = (
            "google_rss"
            if "google" in source
            else "naver_search"
            if "naver" in source
            else source
        )
        start = pd.to_datetime(row.get("start"), errors="coerce")
        end_exclusive = pd.to_datetime(row.get("end_exclusive"), errors="coerce")
        status = str(row.get("status") or "")
        if window_id in coverage_window_ids:
            duplicate_coverage_window_ids += 1
            continue
        coverage_window_ids.add(window_id)
        valid = (
            bool(window_id)
            and ticker in universe_set
            and source != "unknown"
            and not pd.isna(start)
            and not pd.isna(end_exclusive)
            and pd.Timestamp(start).normalize() < pd.Timestamp(end_exclusive).normalize()
            and pd.Timestamp(start).normalize() >= release_start
            and pd.Timestamp(end_exclusive).normalize() <= release_end + pd.Timedelta(days=1)
            and status in {"complete", "incomplete_saturated"}
        )
        if not valid:
            invalid_coverage_window_rows += 1
            continue
        start_date = pd.Timestamp(start).normalize()
        end_date = pd.Timestamp(end_exclusive).normalize()
        coverage_windows_by_source_ticker[(source, ticker)].append((start_date, end_date, status))
        complete_coverage_window_rows += int(status == "complete")
        saturated_coverage_window_rows += int(status == "incomplete_saturated")

    window_partition_errors: list[dict[str, Any]] = []
    window_tickers_by_source: defaultdict[str, set[str]] = defaultdict(set)
    required_window_sources = {str(value) for value in config.get("required_window_coverage_sources", [])}
    for (source, ticker), windows in sorted(coverage_windows_by_source_ticker.items()):
        ordered = sorted(windows, key=lambda value: (value[0], value[1]))
        cursor = release_start
        valid_partition = True
        for start, end_exclusive, _status in ordered:
            if start != cursor:
                valid_partition = False
                break
            cursor = end_exclusive
        if cursor != release_end + pd.Timedelta(days=1):
            valid_partition = False
        if valid_partition:
            window_tickers_by_source[source].add(ticker)
        elif source in required_window_sources:
            window_partition_errors.append(
                {
                    "source": source,
                    "ticker": ticker,
                    "windows": len(ordered),
                    "covered_until": str(cursor.date()),
                }
            )

    neutral_ids: set[str] = set()
    invalid_neutral_rows = 0
    duplicate_neutral_ids = 0
    for row in neutral_events:
        neutral_id = str(row.get("queue_id") or "").strip()
        ticker = normalize_ticker(row.get("ticker"))
        published = pd.to_datetime(row.get("published"), errors="coerce")
        effective = pd.to_datetime(row.get("effective_session"), errors="coerce")
        source_article_ids = row.get("source_article_ids")
        event = row.get("event") if isinstance(row.get("event"), Mapping) else {}
        lineage = row.get("lineage") if isinstance(row.get("lineage"), Mapping) else {}
        if neutral_id in neutral_ids:
            duplicate_neutral_ids += 1
            continue
        neutral_ids.add(neutral_id)
        try:
            source_ids_valid = (
                isinstance(source_article_ids, Sequence)
                and not isinstance(source_article_ids, (str, bytes))
                and len(source_article_ids) == int(row.get("source_article_count", -1))
                and all(
                    str(article_id) in article_by_id
                    and (str(article_id), ticker) in mapping_keys
                    for article_id in source_article_ids
                )
            )
            event_valid = (
                float(event.get("relevance")) == 1.0
                and float(event.get("event_specificity")) == 1.0
                and event.get("sensor_accepted") is True
                and float(event.get("polarity")) == 0.0
                and float(event.get("magnitude")) == 0.0
                and float(event.get("evidence_quality")) == 1.0
                and bool(event.get("selection_point_in_time"))
                and isinstance(event.get("node_deltas"), Sequence)
                and not isinstance(event.get("node_deltas"), (str, bytes))
                and len(event.get("node_deltas")) == 0
            )
        except (TypeError, ValueError):
            source_ids_valid = False
            event_valid = False
        valid = (
            bool(neutral_id)
            and neutral_id not in queue_by_id
            and ticker in universe_set
            and not pd.isna(published)
            and not pd.isna(effective)
            and effective.normalize() > published.normalize()
            and source_ids_valid
            and event_valid
            and bool(row.get("selection_point_in_time"))
            and lineage.get("method") == "deterministic_standardized_filing_policy"
            and bool(lineage.get("policy_version"))
        )
        invalid_neutral_rows += int(not valid)

    structured_ids: set[str] = set()
    duplicate_structured_ids = 0
    orphan_structured_ids = 0
    input_hash_mismatches = 0
    llm_rows = 0
    llm_error_rows = 0
    schema_error_rows = 0
    event_value_error_rows = 0
    missing_model_metadata = 0
    accepted_sensor_rows = 0
    relevance_buckets: Counter[str] = Counter()
    event_specificity_buckets: Counter[str] = Counter()
    low_relevance_delta_rows = 0
    low_specificity_delta_rows = 0
    sensor_acceptance_mismatch_rows = 0
    materialization_error_rows = 0
    evidence_quality_error_rows = 0
    structured_input_hash_policy_mismatch_rows = 0
    for row in structured:
        queue_id = str(row.get("queue_id") or "").strip()
        if queue_id in structured_ids:
            duplicate_structured_ids += 1
            continue
        structured_ids.add(queue_id)
        expected = queue_by_id.get(queue_id)
        if expected is None:
            orphan_structured_ids += 1
            continue
        if str(row.get("input_sha256") or "") != str(expected.get("input_sha256") or ""):
            input_hash_mismatches += 1
        if bool(row.get("llm_used")):
            llm_rows += 1
        if str(row.get("llm_error") or "").strip():
            llm_error_rows += 1
        lineage = row.get("lineage") if isinstance(row.get("lineage"), Mapping) else {}
        required_lineage = ("model_id", "model_revision", "prompt_version", "output_schema_version")
        if any(not str(lineage.get(key) or "").strip() for key in required_lineage):
            missing_model_metadata += 1
        expected_input_hash_policy = str(
            expected.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
        )
        structured_input_hash_policy = str(
            lineage.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
        )
        structured_input_hash_policy_mismatch_rows += int(
            structured_input_hash_policy != expected_input_hash_policy
        )
        event = row.get("event") if isinstance(row.get("event"), Mapping) else {}
        required_event = (
            "event_type",
            "relevance",
            "event_specificity",
            "sensor_accepted",
            "polarity",
            "magnitude",
            "confidence",
            "evidence_quality",
            "content_tier",
            "mapping_method",
            "acquisition_modes",
            "selection_point_in_time",
            "horizon_days",
            "affected_nodes",
            "themes",
            "node_deltas",
            "edge_deltas",
        )
        if any(key not in event for key in required_event):
            schema_error_rows += 1
            continue
        try:
            relevance = float(event["relevance"])
            event_specificity = float(event["event_specificity"])
            polarity = float(event["polarity"])
            magnitude = float(event["magnitude"])
            confidence = float(event["confidence"])
            evidence_quality = float(event["evidence_quality"])
            horizon = int(event["horizon_days"])
            valid = (
                all(
                    math.isfinite(value)
                    for value in (
                        relevance,
                        event_specificity,
                        polarity,
                        magnitude,
                        confidence,
                        evidence_quality,
                    )
                )
                and 0.0 <= relevance <= 1.0
                and 0.0 <= event_specificity <= 1.0
                and -1.0 <= polarity <= 1.0
                and 0.0 <= magnitude <= 1.0
                and 0.0 <= confidence <= 1.0
                and 0.0 <= evidence_quality <= 1.0
                and 1 <= horizon <= int(config.get("max_horizon_days", 365))
                and isinstance(event["node_deltas"], Sequence)
                and not isinstance(event["node_deltas"], (str, bytes))
                and isinstance(event["edge_deltas"], Sequence)
                and not isinstance(event["edge_deltas"], (str, bytes))
                and isinstance(event["affected_nodes"], Sequence)
                and not isinstance(event["affected_nodes"], (str, bytes))
                and isinstance(event["themes"], Sequence)
                and not isinstance(event["themes"], (str, bytes))
                and isinstance(event["acquisition_modes"], Mapping)
                and isinstance(event["selection_point_in_time"], bool)
                and isinstance(event["sensor_accepted"], bool)
            )
        except (TypeError, ValueError, OverflowError):
            valid = False
        if not valid:
            event_value_error_rows += 1
            continue
        content_quality = {
            "title_only": 0.55,
            "title_summary": 0.85,
            "full_text": 1.0,
            "official_filing": 1.0,
        }.get(str(expected.get("content_tier") or ""))
        try:
            mapping_quality = max(
                0.0,
                min(1.0, float(expected.get("mapping_confidence"))),
            )
        except (TypeError, ValueError, OverflowError):
            mapping_quality = None
        expected_evidence_quality = (
            content_quality * mapping_quality
            if content_quality is not None and mapping_quality is not None
            else None
        )
        evidence_quality_error_rows += int(
            expected_evidence_quality is None
            or abs(evidence_quality - expected_evidence_quality) > 1e-9
            or str(event.get("content_tier") or "") != str(expected.get("content_tier") or "")
            or str(event.get("mapping_method") or "") != str(expected.get("mapping_method") or "")
            or bool(event.get("selection_point_in_time"))
            != bool(expected.get("selection_point_in_time"))
            or dict(event.get("acquisition_modes") or {})
            != dict(expected.get("acquisition_modes") or {})
        )
        expected_sensor_accepted = relevance >= 0.5 and event_specificity >= 0.5
        sensor_accepted = bool(event["sensor_accepted"])
        accepted_sensor_rows += int(sensor_accepted)
        sensor_acceptance_mismatch_rows += int(sensor_accepted != expected_sensor_accepted)
        labels = row.get("labels") if isinstance(row.get("labels"), Mapping) else None
        try:
            expected_confidence = (
                float(labels["confidence"])
                * relevance
                * event_specificity
                * evidence_quality
                if labels is not None and expected_sensor_accepted
                else 0.0
            )
            materialization_valid = (
                labels is not None
                and float(labels["relevance"]) == relevance
                and float(labels["event_specificity"]) == event_specificity
                and str(labels["event_type"]) == str(event["event_type"])
                and str(labels["summary"]) == str(event.get("summary") or "")
                and int(labels["horizon_days"]) == horizon
                and list(labels["themes"]) == list(event["themes"])
                and abs(confidence - expected_confidence) <= 1e-9
                and polarity
                == (float(labels["polarity"]) if expected_sensor_accepted else 0.0)
                and magnitude
                == (float(labels["magnitude"]) if expected_sensor_accepted else 0.0)
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            materialization_valid = False
        materialization_error_rows += int(not materialization_valid)
        relevance_buckets[
            "0.0-0.3"
            if relevance < 0.3
            else "0.3-0.5"
            if relevance < 0.5
            else "0.5-0.8"
            if relevance < 0.8
            else "0.8-1.0"
        ] += 1
        event_specificity_buckets[
            "0.0-0.3"
            if event_specificity < 0.3
            else "0.3-0.5"
            if event_specificity < 0.5
            else "0.5-0.8"
            if event_specificity < 0.8
            else "0.8-1.0"
        ] += 1
        forbidden_sensor_output = (
            bool(event["node_deltas"])
            or bool(event["edge_deltas"])
            or polarity != 0.0
            or magnitude != 0.0
            or confidence != 0.0
            or bool(event.get("affected_nodes"))
        )
        low_relevance_delta_rows += int(
            relevance < 0.5
            and forbidden_sensor_output
        )
        low_specificity_delta_rows += int(
            relevance >= 0.5
            and event_specificity < 0.5
            and forbidden_sensor_output
        )

    article_ticker_ratio = ratio(len(mapping_tickers), len(universe_set))
    coverage_ratio = ratio(len(coverage_tickers), len(universe_set))
    precise_ratio = ratio(precise_timestamp_rows, len(article_by_id))
    matched_structured_ids = structured_ids & set(queue_by_id)
    structured_queue_ratio = ratio(len(matched_structured_ids), len(queue_by_id))
    llm_ratio = ratio(llm_rows, len(queue_by_id))
    search_article_rows = len(article_by_id) - int(article_source_providers.get("opendart", 0))
    search_title_only_rows = int(article_content_tiers.get("title_only", 0))
    search_title_only_ratio = ratio(search_title_only_rows, search_article_rows)
    query_only_mapping_rows = int(mapping_methods.get("source_query_only", 0))
    query_only_mapping_ratio = ratio(query_only_mapping_rows, relevance_required_mappings)
    point_in_time_search_rows = sum(
        bool(row.get("selection_point_in_time"))
        for row in article_by_id.values()
        if str(row.get("source_provider") or "unknown") != "opendart"
    )
    non_point_in_time_search_rows = max(0, search_article_rows - point_in_time_search_rows)

    if (
        duplicate_article_ids
        or invalid_article_rows
        or outside_release_window_rows
        or unsafe_availability_rows
        or duplicate_mappings
        or invalid_mapping_rows
        or lifecycle_mapping_rows
        or duplicate_occurrence_ids
        or invalid_occurrence_rows
        or invalid_occurrence_query_windows
        or outside_occurrence_query_window_rows
        or article_occurrence_lineage_mismatch_rows
        or unsafe_live_observation_availability_rows
        or duplicate_queue_ids
        or invalid_queue_rows
        or queue_contract_mismatch_rows
        or queue_input_hash_mismatches
        or disallowed_queue_input_hash_rows
        or unsupported_configured_queue_hash_policies
    ):
        issues.add(
            "blocker",
            "news_record_integrity_failure",
            "Canonical articles, mappings, or queue rows violate identity, lifecycle, or availability contracts.",
            duplicate_article_ids=duplicate_article_ids,
            invalid_article_rows=invalid_article_rows,
            outside_release_window_rows=outside_release_window_rows,
            unsafe_availability_rows=unsafe_availability_rows,
            duplicate_mappings=duplicate_mappings,
            invalid_mapping_rows=invalid_mapping_rows,
            lifecycle_mapping_rows=lifecycle_mapping_rows,
            duplicate_occurrence_ids=duplicate_occurrence_ids,
            invalid_occurrence_rows=invalid_occurrence_rows,
            invalid_occurrence_query_windows=invalid_occurrence_query_windows,
            outside_occurrence_query_window_rows=outside_occurrence_query_window_rows,
            article_occurrence_lineage_mismatch_rows=(
                article_occurrence_lineage_mismatch_rows
            ),
            unsafe_live_observation_availability_rows=(
                unsafe_live_observation_availability_rows
            ),
            duplicate_queue_ids=duplicate_queue_ids,
            invalid_queue_rows=invalid_queue_rows,
            queue_contract_mismatch_rows=queue_contract_mismatch_rows,
            queue_input_hash_mismatches=queue_input_hash_mismatches,
            queue_input_hash_policies=dict(sorted(queue_input_hash_policies.items())),
            allowed_queue_input_hash_policies=sorted(allowed_queue_hash_policies),
            disallowed_queue_input_hash_rows=disallowed_queue_input_hash_rows,
            unsupported_configured_queue_hash_policies=sorted(
                unsupported_configured_queue_hash_policies
            ),
        )
    min_coverage = float(config.get("min_coverage_ledger_ratio", 0.95))
    if coverage_ratio < min_coverage or failed_coverage_rows or incomplete_coverage_rows:
        issues.add(
            "blocker",
            "news_collection_incomplete",
            "Canonical news coverage ledger is incomplete for the frozen universe.",
            ledger_ticker_ratio=coverage_ratio,
            required=min_coverage,
            failed_coverage_rows=failed_coverage_rows,
            incomplete_coverage_rows=incomplete_coverage_rows,
        )
    for required_source in config.get("required_coverage_sources", []):
        source_ratio = ratio(len(coverage_tickers_by_source[str(required_source)]), len(universe_set))
        if source_ratio < min_coverage:
            issues.add(
                "blocker",
                "news_required_source_incomplete",
                "A required canonical news source is incomplete for the frozen universe.",
                source=str(required_source),
                ticker_ratio=source_ratio,
                required=min_coverage,
            )
    if invalid_coverage_window_rows or duplicate_coverage_window_ids or window_partition_errors:
        issues.add(
            "blocker",
            "news_window_coverage_integrity_failure",
            "Per-window news coverage has invalid rows, duplicate identities, gaps, or overlaps.",
            invalid_rows=invalid_coverage_window_rows,
            duplicate_ids=duplicate_coverage_window_ids,
            partition_errors=window_partition_errors[:30],
        )
    for required_source in sorted(required_window_sources):
        source_ratio = ratio(len(window_tickers_by_source[required_source]), len(universe_set))
        if source_ratio < min_coverage:
            issues.add(
                "blocker",
                "news_required_window_source_incomplete",
                "A required per-window news source does not partition the full release range.",
                source=required_source,
                ticker_ratio=source_ratio,
                required=min_coverage,
            )
    max_saturated_window_ratio = config.get("max_saturated_window_ratio")
    window_rows = complete_coverage_window_rows + saturated_coverage_window_rows
    saturated_window_ratio = ratio(saturated_coverage_window_rows, window_rows)
    if (
        max_saturated_window_ratio is not None
        and saturated_window_ratio > float(max_saturated_window_ratio)
    ):
        issues.add(
            "blocker",
            "news_window_saturation_excessive",
            "Too many minimum-size news query windows reached the provider cap.",
            ratio=saturated_window_ratio,
            maximum=float(max_saturated_window_ratio),
        )
    if invalid_neutral_rows or duplicate_neutral_ids:
        issues.add(
            "blocker",
            "news_neutral_event_integrity_failure",
            "Deterministic filing-count events violate source, timing, or zero-direction contracts.",
            invalid_rows=invalid_neutral_rows,
            duplicate_ids=duplicate_neutral_ids,
        )
    max_title_only_ratio = config.get("max_search_title_only_ratio")
    max_query_only_ratio = config.get("max_query_only_mapping_ratio")
    content_quality_failures: dict[str, Any] = {}
    if max_title_only_ratio is not None and search_title_only_ratio > float(max_title_only_ratio):
        content_quality_failures["search_title_only_ratio"] = {
            "actual": search_title_only_ratio,
            "maximum": float(max_title_only_ratio),
        }
    if max_query_only_ratio is not None and query_only_mapping_ratio > float(max_query_only_ratio):
        content_quality_failures["query_only_mapping_ratio"] = {
            "actual": query_only_mapping_ratio,
            "maximum": float(max_query_only_ratio),
        }
    if content_quality_failures:
        issues.add(
            "blocker",
            "news_content_quality_insufficient",
            "Search-news evidence is too weak for the configured release role.",
            failures=content_quality_failures,
        )
    if bool(config.get("require_point_in_time_search_selection", False)) and non_point_in_time_search_rows:
        issues.add(
            "blocker",
            "news_retrospective_selection_not_pit",
            "Retrospective discovery-index results cannot serve as a strict point-in-time news sensor.",
            non_point_in_time_search_rows=non_point_in_time_search_rows,
            search_article_rows=search_article_rows,
        )
    min_llm_ratio = float(config.get("min_llm_ratio", 0.95))
    min_structured_ratio = float(config.get("min_structured_raw_id_ratio", 0.95))
    if (
        llm_ratio < min_llm_ratio
        or structured_queue_ratio < min_structured_ratio
        or duplicate_structured_ids
        or orphan_structured_ids
        or input_hash_mismatches
        or llm_error_rows
        or schema_error_rows
        or event_value_error_rows
        or low_relevance_delta_rows
        or low_specificity_delta_rows
        or sensor_acceptance_mismatch_rows
        or materialization_error_rows
        or evidence_quality_error_rows
        or structured_input_hash_policy_mismatch_rows
        or missing_model_metadata
    ):
        issues.add(
            "blocker",
            "news_structuring_incomplete",
            "Frozen news queue does not have one valid, lineage-complete LLM result per required row.",
            llm_ratio=llm_ratio,
            required=min_llm_ratio,
            structured_queue_ratio=structured_queue_ratio,
            required_structured_queue_ratio=min_structured_ratio,
            duplicate_structured_ids=duplicate_structured_ids,
            orphan_structured_ids=orphan_structured_ids,
            input_hash_mismatches=input_hash_mismatches,
            llm_error_rows=llm_error_rows,
            schema_error_rows=schema_error_rows,
            event_value_error_rows=event_value_error_rows,
            low_relevance_delta_rows=low_relevance_delta_rows,
            low_specificity_delta_rows=low_specificity_delta_rows,
            sensor_acceptance_mismatch_rows=sensor_acceptance_mismatch_rows,
            materialization_error_rows=materialization_error_rows,
            evidence_quality_error_rows=evidence_quality_error_rows,
            structured_input_hash_policy_mismatch_rows=structured_input_hash_policy_mismatch_rows,
            missing_model_metadata=missing_model_metadata,
        )
    if title_only_rows:
        issues.add(
            "warning",
            "news_title_only_content",
            "Some canonical articles contain a title but no substantive summary/body.",
            rows=title_only_rows,
            ratio=ratio(title_only_rows, len(article_by_id)),
        )
    legacy_queue_hash_rows = int(
        queue_input_hash_policies.get(LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY, 0)
    )
    if legacy_queue_hash_rows:
        issues.add(
            "warning",
            "news_legacy_queue_input_hash_policy",
            "Some queue rows use the legacy hash that omits prompt-visible fields.",
            rows=legacy_queue_hash_rows,
            policy=LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY,
        )
    if date_only_rows:
        issues.add(
            "info",
            "news_date_only_delayed",
            "Date-only articles are conservatively delayed to the next KRX session.",
            rows=date_only_rows,
        )
    return {
        "format": "canonical_v1",
        "occurrence_load_reports": occurrence_load_reports,
        "raw_load_reports": article_load_reports,
        "mapping_load_reports": mapping_load_reports,
        "queue_load_reports": queue_load_reports,
        "structured_load_reports": structured_load_reports,
        "neutral_event_load_reports": neutral_load_reports,
        "coverage_load_reports": coverage_load_reports,
        "coverage_window_load_reports": coverage_window_load_reports,
        "raw_rows": len(articles),
        "occurrence_rows": occurrence_rows,
        "unique_occurrence_ids": len(occurrence_ids),
        "duplicate_occurrence_ids": duplicate_occurrence_ids,
        "invalid_occurrence_rows": invalid_occurrence_rows,
        "invalid_occurrence_query_windows": invalid_occurrence_query_windows,
        "outside_occurrence_query_window_rows": outside_occurrence_query_window_rows,
        "article_occurrence_lineage_mismatch_rows": (
            article_occurrence_lineage_mismatch_rows
        ),
        "unsafe_live_observation_availability_rows": (
            unsafe_live_observation_availability_rows
        ),
        "unique_articles": len(article_by_id),
        "unique_article_ids": len(article_by_id),
        "duplicate_ids": duplicate_article_ids,
        "duplicate_ticker_article_mappings": duplicate_mappings,
        "article_tickers_in_universe": len(mapping_tickers),
        "article_ticker_ratio": article_ticker_ratio,
        "coverage_ledger_rows": len(coverage),
        "coverage_ledger_tickers": len(coverage_tickers),
        "coverage_ledger_ticker_ratio": coverage_ratio,
        "coverage_tickers_by_source": {
            source: len(tickers) for source, tickers in sorted(coverage_tickers_by_source.items())
        },
        "coverage_window_rows": len(coverage_windows),
        "complete_coverage_window_rows": complete_coverage_window_rows,
        "saturated_coverage_window_rows": saturated_coverage_window_rows,
        "saturated_coverage_window_ratio": saturated_window_ratio,
        "coverage_window_tickers_by_source": {
            source: len(tickers) for source, tickers in sorted(window_tickers_by_source.items())
        },
        "invalid_coverage_window_rows": invalid_coverage_window_rows,
        "duplicate_coverage_window_ids": duplicate_coverage_window_ids,
        "window_partition_errors": len(window_partition_errors),
        "failed_coverage_rows": failed_coverage_rows,
        "incomplete_coverage_rows": incomplete_coverage_rows,
        "date_only_rows": date_only_rows,
        "precise_timestamp_rows": precise_timestamp_rows,
        "precise_timestamp_ratio": precise_ratio,
        "not_yet_effective_rows": not_yet_effective_rows,
        "title_only_rows": title_only_rows,
        "article_source_providers": dict(sorted(article_source_providers.items())),
        "article_content_tiers": dict(sorted(article_content_tiers.items())),
        "search_article_rows": search_article_rows,
        "search_title_only_rows": search_title_only_rows,
        "search_title_only_ratio": search_title_only_ratio,
        "point_in_time_selection_rows": point_in_time_selection_rows,
        "retrospective_or_unknown_selection_rows": retrospective_or_unknown_selection_rows,
        "point_in_time_search_rows": point_in_time_search_rows,
        "non_point_in_time_search_rows": non_point_in_time_search_rows,
        "mapping_methods": dict(sorted(mapping_methods.items())),
        "relevance_required_mappings": relevance_required_mappings,
        "query_only_mapping_rows": query_only_mapping_rows,
        "query_only_mapping_ratio": query_only_mapping_ratio,
        "queue_rows": len(queue_by_id),
        "queue_contract_mismatch_rows": queue_contract_mismatch_rows,
        "queue_input_hash_mismatches": queue_input_hash_mismatches,
        "queue_input_hash_policies": dict(sorted(queue_input_hash_policies.items())),
        "allowed_queue_input_hash_policies": sorted(allowed_queue_hash_policies),
        "disallowed_queue_input_hash_rows": disallowed_queue_input_hash_rows,
        "neutral_event_rows": len(neutral_events),
        "invalid_neutral_event_rows": invalid_neutral_rows,
        "structured_rows": len(structured),
        "structured_raw_id_ratio": structured_queue_ratio,
        "llm_rows": llm_rows,
        "heuristic_rows": 0,
        "llm_ratio": llm_ratio,
        "llm_error_rows": llm_error_rows,
        "schema_error_rows": schema_error_rows,
        "event_value_error_rows": event_value_error_rows,
        "orphan_structured_ids": orphan_structured_ids,
        "duplicate_structured_ids": duplicate_structured_ids,
        "missing_model_metadata_rows": missing_model_metadata,
        "accepted_sensor_rows": accepted_sensor_rows,
        "sensor_acceptance_ratio": ratio(accepted_sensor_rows, len(structured)),
        "relevance_buckets": dict(sorted(relevance_buckets.items())),
        "event_specificity_buckets": dict(sorted(event_specificity_buckets.items())),
        "low_relevance_delta_rows": low_relevance_delta_rows,
        "low_specificity_delta_rows": low_specificity_delta_rows,
        "sensor_acceptance_mismatch_rows": sensor_acceptance_mismatch_rows,
        "materialization_error_rows": materialization_error_rows,
        "evidence_quality_error_rows": evidence_quality_error_rows,
        "structured_input_hash_policy_mismatch_rows": structured_input_hash_policy_mismatch_rows,
    }


def audit_fundamentals(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    path = _path(repo_root, str(config["observations_path"]))
    profile_path = _path(repo_root, str(config["profiles_path"]))
    observations: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    if path.exists():
        files.append(file_record(path, role="fundamental_observations"))
        observations, observation_load = load_jsonl(path)
    else:
        observation_load = {"rows": 0, "invalid_json": 0, "invalid_rows": 0, "empty_lines": 0}
        issues.add("blocker", "fundamental_file_missing", "Fundamental observation file is missing.", path=str(path))
    if profile_path.exists():
        files.append(file_record(profile_path, role="fundamental_profiles"))
        profiles, profile_load = load_jsonl(profile_path)
    else:
        profile_load = {"rows": 0, "invalid_json": 0, "invalid_rows": 0, "empty_lines": 0}
        issues.add("blocker", "fundamental_profile_missing", "Fundamental company-profile file is missing.", path=str(profile_path))

    if load_report_has_errors(observation_load) or load_report_has_errors(profile_load):
        issues.add(
            "blocker",
            "fundamental_jsonl_corrupt",
            "A fundamental JSONL source contains malformed JSON or non-object rows.",
            observations=observation_load,
            profiles=profile_load,
        )
    lag_sessions = int(config.get("availability_lag_sessions", 0))
    if lag_sessions < 1:
        issues.add(
            "blocker",
            "fundamental_availability_policy_unsafe",
            "Date-only filing timestamps require at least one full trading-session availability lag.",
            availability_lag_sessions=lag_sessions,
        )

    universe_set = set(universe)
    tickers: set[str] = set()
    profile_tickers = {normalize_ticker(row.get("ticker")) for row in profiles}
    invalid_tickers = 0
    invalid_dates = 0
    pre_period_availability = 0
    future_availability = 0
    empty_fields = 0
    nonfinite_fields = 0
    duplicate_keys = Counter()
    release_end = pd.Timestamp(config["release_end"]).normalize()
    for row in observations:
        ticker = normalize_ticker(row.get("ticker"))
        if not ticker or ticker not in universe_set:
            invalid_tickers += 1
        else:
            tickers.add(ticker)
        available = pd.to_datetime(row.get("available_at"), errors="coerce")
        period_end = pd.to_datetime(row.get("period_end"), errors="coerce")
        if pd.isna(available) or pd.isna(period_end):
            invalid_dates += 1
        else:
            if available.normalize() < period_end.normalize():
                pre_period_availability += 1
            if available.normalize() > release_end:
                future_availability += 1
            duplicate_keys[(ticker, str(period_end.date()), str(available.date()))] += 1
        fields = row.get("fields")
        if not isinstance(fields, Mapping) or not fields:
            empty_fields += 1
        else:
            for value in fields.values():
                try:
                    parsed = float(value)
                except (TypeError, ValueError):
                    nonfinite_fields += 1
                    continue
                if not math.isfinite(parsed):
                    nonfinite_fields += 1
    duplicate_rows = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    observation_coverage = ratio(len(tickers), len(universe_set))
    profile_coverage = ratio(len(profile_tickers & universe_set), len(universe_set))
    if (
        observation_coverage < float(config.get("min_observation_ticker_ratio", 0.85))
        or profile_coverage < float(config.get("min_profile_ticker_ratio", 0.95))
    ):
        issues.add(
            "blocker",
            "fundamental_coverage_insufficient",
            "OpenDART observations or company profiles do not cover enough of the universe.",
            observation_ticker_ratio=observation_coverage,
            profile_ticker_ratio=profile_coverage,
        )
    if invalid_tickers or invalid_dates or pre_period_availability or future_availability or empty_fields or nonfinite_fields:
        issues.add(
            "blocker",
            "fundamental_integrity_failure",
            "Fundamental records violate ticker, point-in-time, or numeric contracts.",
            invalid_tickers=invalid_tickers,
            invalid_dates=invalid_dates,
            pre_period_availability=pre_period_availability,
            future_availability=future_availability,
            empty_fields=empty_fields,
            nonfinite_fields=nonfinite_fields,
        )
    if duplicate_rows:
        issues.add(
            "warning",
            "fundamental_duplicates_present",
            "Duplicate fundamental observation keys are present.",
            rows=duplicate_rows,
        )
    return {
        "observation_load": observation_load,
        "profile_load": profile_load,
        "observation_tickers": len(tickers),
        "observation_ticker_ratio": observation_coverage,
        "profile_tickers": len(profile_tickers & universe_set),
        "profile_ticker_ratio": profile_coverage,
        "invalid_tickers": invalid_tickers,
        "invalid_dates": invalid_dates,
        "pre_period_availability": pre_period_availability,
        "future_availability": future_availability,
        "empty_fields": empty_fields,
        "nonfinite_fields": nonfinite_fields,
        "duplicate_rows": duplicate_rows,
        "availability_lag_sessions": lag_sessions,
    }


def _select_range_file(
    directory: Path,
    ticker: str,
    required_start: pd.Timestamp,
    required_end: pd.Timestamp,
) -> tuple[Path | None, bool]:
    return select_ohlcv_cache(directory, ticker, required_start, required_end)


def audit_investor(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    cache_dir = _path(repo_root, str(config["cache_dir"]))
    start = pd.Timestamp(config["start"]).normalize()
    end = pd.Timestamp(config["end"]).normalize()
    missing: list[str] = []
    noncovering: list[str] = []
    schema_failures: list[str] = []
    lifecycle_violations: list[str] = []
    duplicate_dates: list[str] = []
    negative_traded_value: list[str] = []
    legacy_volume_header: list[str] = []
    coverages: list[float] = []
    rows_per_ticker: list[float] = []
    for ticker, metadata in universe.items():
        listing = pd.to_datetime(metadata.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(metadata.get("delisting_date"), errors="coerce")
        required_start = max(start, pd.Timestamp(listing).normalize()) if not pd.isna(listing) else start
        required_end = min(end, pd.Timestamp(delisting).normalize()) if not pd.isna(delisting) else end
        path, covers = _select_range_file(cache_dir, ticker, required_start, required_end)
        if path is None:
            missing.append(ticker)
            continue
        if not covers:
            noncovering.append(ticker)
        files.append(file_record(path, role=f"investor:{ticker}"))
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            schema_failures.append(ticker)
            continue
        traded_volume_column = (
            "investor_traded_volume"
            if "investor_traded_volume" in frame.columns
            else "investor_traded_value_m" if "investor_traded_value_m" in frame.columns else None
        )
        if (
            "date" not in frame.columns
            or traded_volume_column is None
            or not set(REQUIRED_INVESTOR_COLUMNS).issubset(frame.columns)
        ):
            schema_failures.append(ticker)
            continue
        if traded_volume_column == "investor_traded_value_m":
            legacy_volume_header.append(ticker)
        dates = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            duplicate_dates.append(ticker)
        numeric = frame[[traded_volume_column, *REQUIRED_INVESTOR_COLUMNS]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            schema_failures.append(ticker)
        if (numeric[traded_volume_column] < 0.0).any():
            negative_traded_value.append(ticker)
        if not pd.isna(listing) and (dates.dropna() < pd.Timestamp(listing).normalize()).any():
            lifecycle_violations.append(ticker)
        if not pd.isna(delisting) and (dates.dropna() > pd.Timestamp(delisting).normalize()).any():
            lifecycle_violations.append(ticker)
        expected = calendar[(calendar >= required_start) & (calendar <= required_end)]
        coverages.append(ratio(len(pd.DatetimeIndex(dates.dropna()).intersection(expected)), len(expected)))
        rows_per_ticker.append(float(len(frame)))
    if missing or noncovering:
        issues.add(
            "blocker",
            "investor_universe_incomplete",
            "Investor-flow cache does not cover every frozen-universe security.",
            count=len(missing),
            tickers=missing[:50],
            noncovering_count=len(noncovering),
            noncovering_tickers=noncovering[:50],
        )
    if bool(config.get("require_official_source", False)) and not bool(config.get("official", False)):
        issues.add(
            "blocker",
            "investor_primary_source_missing",
            "Investor-flow release is not backed by the configured official brokerage/exchange source.",
            provider=config.get("provider"),
        )
    if schema_failures or lifecycle_violations or duplicate_dates or negative_traded_value:
        issues.add(
            "blocker",
            "investor_integrity_failure",
            "Investor-flow files violate schema, date, lifecycle, or numeric contracts.",
            schema_failures=sorted(set(schema_failures))[:30],
            lifecycle_violations=sorted(set(lifecycle_violations))[:30],
            duplicate_dates=sorted(set(duplicate_dates))[:30],
            negative_traded_value=sorted(set(negative_traded_value))[:30],
        )
    median_coverage = float(np.median(coverages)) if coverages else 0.0
    if median_coverage < float(config.get("min_median_coverage", 0.85)):
        issues.add(
            "blocker",
            "investor_coverage_insufficient",
            "Investor-flow median lifecycle coverage is below the release gate.",
            median=median_coverage,
        )
    if legacy_volume_header:
        issues.add(
            "warning",
            "investor_legacy_volume_header",
            "Kiwoom accumulated volume is stored under the historical traded-value column name.",
            count=len(legacy_volume_header),
            compatibility_policy="rename to investor_traded_volume in canonical release",
        )
    return {
        "cache_dir": str(cache_dir),
        "expected_tickers": len(universe),
        "loaded_tickers": len(universe) - len(missing),
        "missing_tickers": missing,
        "noncovering_tickers": noncovering,
        "provider": config.get("provider"),
        "official": bool(config.get("official", False)),
        "schema_failure_tickers": sorted(set(schema_failures)),
        "lifecycle_violation_tickers": sorted(set(lifecycle_violations)),
        "duplicate_date_tickers": sorted(set(duplicate_dates)),
        "negative_traded_value_tickers": sorted(set(negative_traded_value)),
        "legacy_volume_header_tickers": legacy_volume_header,
        "rows_per_ticker": finite_summary(rows_per_ticker),
        "lifecycle_coverage": finite_summary(coverages),
    }


def audit_external(
    repo_root: Path,
    config: Mapping[str, Any],
    issues: IssueLog,
    files: list[dict[str, Any]],
) -> dict[str, Any]:
    factor_reports: dict[str, Any] = {}
    missing: list[str] = []
    invalid: list[str] = []
    range_incomplete: list[str] = []
    start = pd.Timestamp(config["start"]).normalize()
    end = pd.Timestamp(config["end"]).normalize()
    max_start_lag = int(config.get("max_start_lag_days", 10))
    max_end_lag = int(config.get("max_end_lag_days", 10))
    for factor in config.get("factors", []):
        factor_id = str(factor["id"])
        path = _path(repo_root, str(factor["path"]))
        if not path.exists():
            missing.append(factor_id)
            continue
        files.append(file_record(path, role=f"external:{factor_id}"))
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            invalid.append(factor_id)
            continue
        date_column = "Date" if "Date" in frame.columns else frame.columns[0]
        configured_value_column = str(factor.get("value_column") or "")
        value_column = configured_value_column if configured_value_column in frame.columns else None
        if value_column is None and "Close" in frame.columns:
            value_column = "Close"
        if value_column is None:
            candidates = [column for column in frame.columns if column != date_column]
            if len(candidates) == 1:
                value_column = candidates[0]
        if value_column is None:
            invalid.append(factor_id)
            continue
        dates = pd.to_datetime(frame[date_column], errors="coerce")
        values = pd.to_numeric(frame[value_column], errors="coerce")
        if dates.isna().any() or dates.duplicated().any() or not dates.is_monotonic_increasing:
            invalid.append(factor_id)
        finite = np.isfinite(values.to_numpy(dtype=float))
        finite_ratio = float(finite.mean()) if len(finite) else 0.0
        usable_dates = dates[finite & dates.notna()]
        date_min = usable_dates.min().normalize() if len(usable_dates) else None
        date_max = usable_dates.max().normalize() if len(usable_dates) else None
        if (
            date_min is None
            or date_max is None
            or date_min > start + pd.Timedelta(days=max_start_lag)
            or date_max < end - pd.Timedelta(days=max_end_lag)
            or len(usable_dates) < int(config.get("min_observations", 80))
        ):
            range_incomplete.append(factor_id)
        factor_reports[factor_id] = {
            "path": str(path),
            "rows": int(len(frame)),
            "date_min": str(dates.min().date()) if dates.notna().any() else None,
            "date_max": str(dates.max().date()) if dates.notna().any() else None,
            "finite_ratio": finite_ratio,
            "usable_rows": int(len(usable_dates)),
            "empty_value_rows": int((~finite).sum()),
            "value_column": value_column,
            "provider": factor.get("provider"),
            "official": bool(factor.get("official", False)),
        }
    if missing or invalid or range_incomplete:
        issues.add(
            "blocker",
            "external_factor_integrity_failure",
            "Required external-factor files are missing or invalid.",
            missing=missing,
            invalid=invalid,
            range_incomplete=range_incomplete,
        )
    unofficial = [factor_id for factor_id, report in factor_reports.items() if not report["official"]]
    if unofficial:
        issues.add(
            "warning",
            "external_source_not_primary",
            "Some external factors are cached from non-primary market-data vendors.",
            factors=unofficial,
        )
    return {
        "required": len(config.get("factors", [])),
        "loaded": len(factor_reports),
        "missing": missing,
        "invalid": invalid,
        "range_incomplete": range_incomplete,
        "factors": factor_reports,
    }


def audit_splits(config: Mapping[str, Any], issues: IssueLog) -> dict[str, Any]:
    rows = list(config.get("folds", []))
    parsed: list[dict[str, Any]] = []
    overlap_pairs: list[list[str]] = []
    for row in rows:
        item = dict(row)
        item["start"] = str(pd.Timestamp(row["start"]).date())
        item["end"] = str(pd.Timestamp(row["end"]).date())
        parsed.append(item)
    for left_index, left in enumerate(parsed):
        left_start = pd.Timestamp(left["start"])
        left_end = pd.Timestamp(left["end"])
        if left_start > left_end:
            issues.add("blocker", "split_date_invalid", "A split starts after it ends.", split=left.get("name"))
        for right in parsed[left_index + 1 :]:
            right_start = pd.Timestamp(right["start"])
            right_end = pd.Timestamp(right["end"])
            if max(left_start, right_start) <= min(left_end, right_end):
                overlap_pairs.append([str(left.get("name")), str(right.get("name"))])
    if overlap_pairs:
        issues.add(
            "blocker",
            "split_overlap",
            "Dataset split windows overlap.",
            pairs=overlap_pairs,
        )
    final_rows = [row for row in parsed if row.get("role") == "final_evaluation"]
    if not final_rows or any(row.get("status") != "untouched" for row in final_rows):
        issues.add(
            "blocker",
            "untouched_final_holdout_missing",
            "No sufficiently sized untouched final-evaluation split is frozen.",
        )
    reused = [row.get("name") for row in parsed if row.get("status") == "reused_for_model_selection"]
    if reused:
        issues.add(
            "info",
            "development_folds_reused",
            "Historical folds are correctly marked as development-only after repeated model selection.",
            folds=reused,
        )
    return {"folds": parsed, "overlap_pairs": overlap_pairs, "final_evaluation_folds": final_rows}


def build_release_fingerprint(config: Mapping[str, Any], files: Sequence[Mapping[str, Any]]) -> str:
    payload = {
        "config": config,
        "files": sorted(
            (
                {"role": row["role"], "size_bytes": row["size_bytes"], "sha256": row["sha256"]}
                for row in files
            ),
            key=lambda row: (str(row["role"]), str(row["sha256"])),
        ),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def audit_dataset_release(repo_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    issues = IssueLog()
    files: list[dict[str, Any]] = []
    universe_report, universe = audit_universe(repo_root, config["universe"], issues, files)
    release_start = pd.Timestamp(config["release_window"]["start"]).normalize()
    release_end = pd.Timestamp(config["release_window"]["end"]).normalize()
    calendar = load_trading_calendar(
        repo_root,
        config["trading_calendar"]["paths"],
        release_start,
        release_end,
        issues,
        files,
    )
    if len(calendar) < int(config["trading_calendar"].get("min_sessions", 1000)):
        issues.add(
            "blocker",
            "trading_calendar_incomplete",
            "Trading calendar has too few sessions for the release window.",
            sessions=len(calendar),
        )
    reports = {
        "universe": universe_report,
        "trading_calendar": {
            "sessions": int(len(calendar)),
            "date_min": str(calendar.min().date()) if len(calendar) else None,
            "date_max": str(calendar.max().date()) if len(calendar) else None,
        },
        "ohlcv": audit_ohlcv(repo_root, config["ohlcv"], universe, calendar, issues, files),
        "news": audit_news(repo_root, config["news"], universe, issues, files),
        "fundamentals": audit_fundamentals(
            repo_root,
            config["fundamentals"],
            universe,
            issues,
            files,
        ),
        "investor": audit_investor(
            repo_root,
            config["investor"],
            universe,
            calendar,
            issues,
            files,
        ),
        "external": audit_external(repo_root, config["external"], issues, files),
        "splits": audit_splits(config["splits"], issues),
    }
    counts = issues.counts()
    status = "pass" if counts["blocker"] == 0 else "blocked"
    return {
        "schema_version": 1,
        "release_id": config["release_id"],
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "status": status,
        "release_window": config["release_window"],
        "fingerprint_sha256": build_release_fingerprint(config, files),
        "issue_counts": counts,
        "issues": [asdict(issue) for issue in issues.issues],
        "sources": files,
        "reports": reports,
    }


def render_dataset_card(report: Mapping[str, Any]) -> str:
    counts = report["issue_counts"]
    lines = [
        f"# Dataset Release {report['release_id']}",
        "",
        f"- Status: `{report['status']}`",
        f"- Fingerprint: `{report['fingerprint_sha256']}`",
        f"- Generated: `{report['generated_at_utc']}`",
        f"- Blockers: {counts['blocker']}",
        f"- Warnings: {counts['warning']}",
        "",
        "## Coverage",
        "",
    ]
    reports = report["reports"]
    lines.extend(
        [
            f"- Universe: {reports['universe']['unique_tickers']} / {reports['universe']['expected_count']}",
            f"- OHLCV files: {reports['ohlcv']['selected_files']} / {reports['ohlcv']['expected_tickers']}",
            f"- News collection-ledger ticker ratio: {reports['news']['coverage_ledger_ticker_ratio']:.3f}",
            f"- News actual LLM ratio: {reports['news']['llm_ratio']:.3f}",
            f"- Fundamental observation ticker ratio: {reports['fundamentals']['observation_ticker_ratio']:.3f}",
            f"- Investor files: {reports['investor']['loaded_tickers']} / {reports['investor']['expected_tickers']}",
            f"- External factors: {reports['external']['loaded']} / {reports['external']['required']}",
            "",
            "## Promotion Blockers",
            "",
        ]
    )
    blockers = [issue for issue in report["issues"] if issue["severity"] == "blocker"]
    if blockers:
        lines.extend(f"- `{issue['code']}`: {issue['message']}" for issue in blockers)
    else:
        lines.append("- None.")
    lines.extend(["", "## Warnings", ""])
    warnings = [issue for issue in report["issues"] if issue["severity"] == "warning"]
    if warnings:
        lines.extend(f"- `{issue['code']}`: {issue['message']}" for issue in warnings)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Contract",
            "",
            "This card is generated from immutable source hashes. A release is usable for",
            "model promotion only when `status` is `pass`; normalized tensors being finite",
            "is not sufficient. Reused temporal folds are development data, not final test data.",
            "",
        ]
    )
    return "\n".join(lines)
