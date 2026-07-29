from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_news_events import fetch_rss, is_source_blocked, is_source_overloaded
from stock_v2.dataset_integrity import load_json, normalize_ticker
from stock_v2.news_aliases import is_lexically_ambiguous_short_name, validate_alias_registry


GOOGLE_LIVE_QUERY_POLICY = "reviewed_identity_prefer_unambiguous_v2"


def stable_live_article_id(ticker: str, title: str, link: str) -> str:
    identity = f"google-live-v1|{ticker}|{title.strip()}|{link.strip()}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def build_google_live_query(
    ticker: str,
    company_name: str,
    aliases: Sequence[Mapping[str, Any]],
    *,
    include_ticker: bool = False,
    lookback: str = "2h",
) -> str:
    if not re.fullmatch(r"[1-9][0-9]*[hd]", lookback):
        raise ValueError(f"invalid Google News lookback: {lookback}")
    candidates = [
        (company_name, is_lexically_ambiguous_short_name(company_name)),
        *[
            (str(row.get("alias") or ""), bool(row.get("lexically_ambiguous", False)))
            for row in aliases
        ],
    ]
    safe_candidates = [candidate for candidate in candidates if not candidate[1]]
    selected_candidates = safe_candidates or candidates
    terms: list[str] = []
    for value, _ambiguous in selected_candidates:
        cleaned = " ".join(str(value or "").split()).strip()
        if cleaned and cleaned.casefold() not in {term.casefold() for term in terms}:
            terms.append(cleaned)
    if include_ticker:
        terms.append(ticker)
    quoted = [f'"{term.replace(chr(34), "")}"' for term in terms]
    if not quoted:
        raise ValueError(f"no live-news query terms for {ticker}")
    return f"({' OR '.join(quoted)}) when:{lookback}"


def _active_universe(payload: Mapping[str, Any], as_of: pd.Timestamp) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for value in payload.get("universe", []):
        if not isinstance(value, Mapping):
            continue
        ticker = normalize_ticker(value.get("ticker"))
        if not ticker:
            continue
        listing = pd.to_datetime(value.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(value.get("delisting_date"), errors="coerce")
        if (not pd.isna(listing) and pd.Timestamp(listing).normalize() > as_of) or (
            not pd.isna(delisting) and pd.Timestamp(delisting).normalize() < as_of
        ):
            continue
        result[ticker] = dict(value)
    return result


def _prepare_seen_cache(output: Path, database: Path) -> sqlite3.Connection:
    database.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(database))
    connection.execute("CREATE TABLE IF NOT EXISTS seen (article_id TEXT PRIMARY KEY)")
    if output.exists():
        pending: list[tuple[str]] = []
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article_id = str(value.get("id") or "") if isinstance(value, dict) else ""
                if article_id:
                    pending.append((article_id,))
                if len(pending) >= 10_000:
                    connection.executemany("INSERT OR IGNORE INTO seen(article_id) VALUES (?)", pending)
                    pending.clear()
        if pending:
            connection.executemany("INSERT OR IGNORE INTO seen(article_id) VALUES (?)", pending)
        connection.commit()
    return connection


def _write_request(handle: Any, payload: Mapping[str, Any]) -> None:
    handle.write(json.dumps(dict(payload), ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Forward-capture point-in-time Google RSS news.")
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--alias-registry", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--request-ledger", required=True)
    parser.add_argument("--seen-db", required=True)
    parser.add_argument("--poll-interval-sec", type=float, default=3600.0)
    parser.add_argument("--request-sleep-sec", type=float, default=0.75)
    parser.add_argument("--request-timeout-sec", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--articles-per-request", type=int, default=100)
    parser.add_argument("--include-ticker", action="store_true")
    parser.add_argument("--lookback", default="2h", help="Google News when: duration, e.g. 2h")
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 runs continuously")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("invalid shard selection")
    if args.poll_interval_sec <= 0 or args.articles_per_request <= 0:
        raise ValueError("poll interval and article limit must be positive")

    universe_path = Path(args.universe_manifest)
    universe_payload = load_json(universe_path)
    all_universe = {
        normalize_ticker(row.get("ticker")): dict(row)
        for row in universe_payload.get("universe", [])
        if isinstance(row, Mapping) and normalize_ticker(row.get("ticker"))
    }
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    active = _active_universe(universe_payload, today)
    combined_aliases: dict[str, Any] = {"schema_version": 1, "aliases": []}
    for raw_path in args.alias_registry:
        payload = load_json(Path(raw_path))
        rows = payload.get("aliases") if isinstance(payload, Mapping) else None
        if not isinstance(rows, list):
            raise ValueError(f"invalid alias registry: {raw_path}")
        combined_aliases["aliases"].extend(rows)
    aliases_by_ticker = validate_alias_registry(combined_aliases, all_universe)
    ticker_rows = [
        (ticker, active[ticker])
        for index, ticker in enumerate(sorted(active))
        if index % args.shard_count == args.shard_index
    ]

    output = Path(args.output)
    request_ledger = Path(args.request_ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    request_ledger.parent.mkdir(parents=True, exist_ok=True)
    seen = _prepare_seen_cache(output, Path(args.seen_db))
    cycles = 0
    source_pause_required = False
    try:
        with output.open("a", encoding="utf-8") as article_handle, request_ledger.open(
            "a", encoding="utf-8"
        ) as request_handle:
            while not args.max_cycles or cycles < args.max_cycles:
                cycles += 1
                cycle_started = datetime.now(timezone.utc)
                cycle_id = hashlib.sha256(
                    f"google-live-cycle-v1|{args.shard_index}|{cycle_started.isoformat()}".encode("utf-8")
                ).hexdigest()[:24]
                cycle_new = cycle_errors = cycle_saturated = 0
                cycle_requests = 0
                for ticker, row in ticker_rows:
                    eligible_aliases = [
                        alias
                        for alias in aliases_by_ticker.get(ticker, [])
                        if alias.get("relationship") == "identity"
                    ]
                    query = build_google_live_query(
                        ticker,
                        str(row.get("name") or ""),
                        eligible_aliases,
                        include_ticker=args.include_ticker,
                        lookback=args.lookback,
                    )
                    request_started = datetime.now(timezone.utc)
                    articles: list[dict[str, str]] = []
                    error = ""
                    request_error: requests.RequestException | None = None
                    for attempt in range(max(0, args.request_retries) + 1):
                        try:
                            articles = fetch_rss(query, args.articles_per_request, args.request_timeout_sec)
                            error = ""
                            request_error = None
                            break
                        except requests.RequestException as exc:
                            request_error = exc
                            error = f"{type(exc).__name__}: {str(exc)[:300]}"
                            if is_source_blocked(exc):
                                break
                            if attempt < args.request_retries:
                                time.sleep(args.retry_backoff_sec * (2**attempt))
                    source_blocked = bool(request_error and is_source_blocked(request_error))
                    source_overloaded = bool(request_error and is_source_overloaded(request_error))
                    source_pause_required = source_blocked or source_overloaded
                    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    new_rows: list[tuple[str]] = []
                    if not error:
                        for rank, article in enumerate(articles, start=1):
                            article_id = stable_live_article_id(
                                ticker,
                                str(article.get("title") or ""),
                                str(article.get("link") or ""),
                            )
                            if seen.execute(
                                "SELECT 1 FROM seen WHERE article_id = ?", (article_id,)
                            ).fetchone():
                                continue
                            payload = {
                                "schema_version": 1,
                                "id": article_id,
                                "ticker": ticker,
                                "name": row.get("name"),
                                "source": "google_news_rss_live_raw",
                                "collected_at_utc": collected_at,
                                "first_seen_at_utc": collected_at,
                                "published": article.get("published", ""),
                                "article": article,
                                "acquisition": {
                                    "provider": "google_rss",
                                    "mode": "live_capture",
                                    "query_policy": GOOGLE_LIVE_QUERY_POLICY,
                                    "query": query,
                                    "response_rank": rank,
                                    "response_count": len(articles),
                                    "cycle_id": cycle_id,
                                    "shard_count": args.shard_count,
                                    "shard_index": args.shard_index,
                                },
                            }
                            article_handle.write(
                                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
                            )
                            new_rows.append((article_id,))
                        article_handle.flush()
                        os.fsync(article_handle.fileno())
                        if new_rows:
                            seen.executemany(
                                "INSERT OR IGNORE INTO seen(article_id) VALUES (?)", new_rows
                            )
                            seen.commit()
                        cycle_new += len(new_rows)
                        saturated = len(articles) >= args.articles_per_request
                        cycle_saturated += int(saturated)
                    else:
                        saturated = False
                        cycle_errors += 1
                    _write_request(
                        request_handle,
                        {
                            "schema_version": 1,
                            "request_id": hashlib.sha256(
                                f"google-live-request-v1|{cycle_id}|{ticker}".encode("utf-8")
                            ).hexdigest(),
                            "cycle_id": cycle_id,
                            "ticker": ticker,
                            "company_name": row.get("name"),
                            "query": query,
                            "query_policy": GOOGLE_LIVE_QUERY_POLICY,
                            "request_started_at_utc": request_started.isoformat(timespec="seconds"),
                            "request_completed_at_utc": collected_at,
                            "status": (
                                "error"
                                if error
                                else "incomplete_saturated"
                                if saturated
                                else "complete"
                            ),
                            "error": error,
                            "source_blocked": source_blocked,
                            "source_overloaded": source_overloaded,
                            "source_pause_required": source_pause_required,
                            "response_count": len(articles),
                            "result_cap": args.articles_per_request,
                            "saturated": saturated,
                            "new_article_rows": len(new_rows),
                            "shard_count": args.shard_count,
                            "shard_index": args.shard_index,
                        },
                    )
                    cycle_requests += 1
                    if source_pause_required:
                        break
                    if args.request_sleep_sec > 0:
                        time.sleep(args.request_sleep_sec)
                elapsed = (datetime.now(timezone.utc) - cycle_started).total_seconds()
                print(
                    json.dumps(
                        {
                            "cycle": cycles,
                            "cycle_id": cycle_id,
                            "tickers": len(ticker_rows),
                            "requests_attempted": cycle_requests,
                            "new_articles": cycle_new,
                            "request_errors": cycle_errors,
                            "saturated_requests": cycle_saturated,
                            "elapsed_sec": round(elapsed, 3),
                            "source_pause_required": source_pause_required,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if source_pause_required:
                    return 75
                remaining = args.poll_interval_sec - elapsed
                if (not args.max_cycles or cycles < args.max_cycles) and remaining > 0:
                    time.sleep(remaining)
    finally:
        seen.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
