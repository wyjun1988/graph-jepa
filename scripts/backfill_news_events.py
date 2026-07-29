from __future__ import annotations

import argparse
from collections import deque
import hashlib
import json
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd
import requests as http_requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.news_sense import heuristic_event
from stock_v2.market_data import (
    load_universe_manifest,
    select_krx_universe_from_listing,
    select_universe,
)
from stock_v2.news_sources import fetch_naver_search, parse_naver_search_results


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="ignore"))
        h.update(b"\0")
    return h.hexdigest()[:20]


def load_seen(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict) and payload.get("id"):
                seen.add(str(payload["id"]))
    return seen


def load_completed_tickers(path: Path) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            ticker = str(payload.get("ticker", "")).replace("A", "").strip()
            status = str(payload.get("status") or "").strip()
            if status and status != "complete":
                continue
            if int(payload.get("request_errors", 0) or 0) > 0:
                continue
            if ticker.isdigit() and len(ticker) == 6:
                completed.add(ticker)
    return completed


def windows(start: str, end: str, days: int):
    left = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while left <= final:
        right = min(final + pd.Timedelta(days=1), left + pd.Timedelta(days=days))
        yield left, right
        left = right


def split_saturated_window(
    left: pd.Timestamp,
    right: pd.Timestamp,
    min_window_days: int,
) -> tuple[tuple[pd.Timestamp, pd.Timestamp], tuple[pd.Timestamp, pd.Timestamp]] | None:
    """Split an exclusive date window without overlap when a result cap is reached."""

    span_days = int((right - left).days)
    if span_days <= max(1, int(min_window_days)):
        return None
    midpoint = left + pd.Timedelta(days=max(1, span_days // 2))
    if midpoint <= left or midpoint >= right:
        return None
    return (left, midpoint), (midpoint, right)


def shard_universe(
    universe: list[tuple[str, str]],
    shard_count: int,
    shard_index: int,
) -> list[tuple[str, str]]:
    if shard_count < 1:
        raise ValueError("shard_count must be at least one")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [row for index, row in enumerate(universe) if index % shard_count == shard_index]


def fetch_rss(query: str, limit: int, timeout_sec: float) -> list[dict[str, str]]:
    url = "https://news.google.com/rss/search?q=" + urllib.parse.quote(query) + "&hl=ko&gl=KR&ceid=KR:ko"
    response = http_requests.get(
        url,
        timeout=max(float(timeout_sec), 0.1),
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    feed = feedparser.parse(response.content)
    rows = []
    for entry in feed.entries[:limit]:
        source = entry.get("source", {})
        source_name = str(source.get("title", "")) if isinstance(source, dict) else ""
        rows.append({
            "title": str(entry.get("title", "")),
            "summary": str(entry.get("summary", "")),
            "link": str(entry.get("link", "")),
            "published": str(entry.get("published", "")),
            "source": source_name,
        })
    return rows


def is_source_blocked(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    return int(getattr(response, "status_code", 0) or 0) in {403, 429}


def is_source_overloaded(error: BaseException) -> bool:
    response = getattr(error, "response", None)
    return int(getattr(response, "status_code", 0) or 0) in {502, 503, 504}


def build_historical_query(
    *,
    source: str,
    ticker: str,
    company_name: str,
    left: pd.Timestamp,
    right: pd.Timestamp,
    include_ticker: bool = False,
) -> tuple[str, str]:
    if source == "naver_search":
        return company_name, "exact_company_name_date_parameters_v1"
    terms = f'"{company_name}"'
    policy = "exact_company_name_date_bounds_v2"
    if include_ticker:
        terms = f"{terms} OR {ticker}"
        policy = "exact_company_name_or_ticker_date_bounds_v1"
    return (
        f"{terms} after:{left.date()} before:{right.date()}",
        policy,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill historical Korean stock-news events into JSONL.")
    parser.add_argument("--universe", choices=["manual", "krx"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--window-days", type=int, default=30)
    parser.add_argument("--articles-per-window", type=int, default=100)
    parser.add_argument(
        "--adaptive-split",
        action="store_true",
        help="Recursively split Google RSS windows that reach the result cap.",
    )
    parser.add_argument("--min-window-days", type=int, default=1)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    parser.add_argument("--request-timeout-sec", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--retry-backoff-sec", type=float, default=1.0)
    parser.add_argument(
        "--source",
        choices=["google_rss", "naver_search"],
        default="naver_search",
        help="Historical search source. Naver is the default because it supports date-bounded Korean results.",
    )
    parser.add_argument("--max-requests", type=int, default=0, help="0 means no explicit cap")
    parser.add_argument(
        "--include-ticker",
        action="store_true",
        help="Add the numeric ticker as an OR term. Disabled by default to reduce false matches.",
    )
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--with-heuristic",
        action="store_true",
        help="Store legacy keyword labels in addition to immutable acquisition data.",
    )
    parser.add_argument("--output", default="data/events/news_backfill_naver_krx100.jsonl")
    parser.add_argument("--coverage-output", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    elif args.universe == "krx":
        universe = select_krx_universe_from_listing(args.max_tickers)
    else:
        universe = select_universe(args.max_tickers)
    universe = list(universe)[: max(0, int(args.max_tickers))]
    full_universe_size = len(universe)
    universe = shard_universe(universe, args.shard_count, args.shard_index)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage_output = (
        Path(args.coverage_output)
        if args.coverage_output
        else Path(f"{output}.coverage.jsonl")
    )
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    seen = load_seen(output) if args.resume else set()
    completed_tickers = load_completed_tickers(coverage_output) if args.resume else set()
    mode = "a" if args.resume else "w"
    coverage_mode = "a" if args.resume else "w"
    requests = 0
    written = 0
    empty = 0
    request_errors = 0
    source_blocked = False
    source_overloaded = False
    source_paused = False
    started = time.perf_counter()
    with output.open(mode, encoding="utf-8") as handle, coverage_output.open(
        coverage_mode,
        encoding="utf-8",
    ) as coverage_handle:
        for ticker, name in universe:
            if ticker in completed_tickers:
                continue
            ticker_requests = 0
            ticker_written = 0
            ticker_failed = False
            stopped = False
            ticker_leaf_windows = 0
            ticker_split_windows = 0
            ticker_saturated_leaf_windows = 0
            ticker_out_of_window_articles = 0
            ticker_query_policy = ""
            pending_windows = deque(windows(args.start, args.end, args.window_days))
            while pending_windows:
                left, right = pending_windows.popleft()
                if args.max_requests and requests >= args.max_requests:
                    stopped = True
                    break
                query, ticker_query_policy = build_historical_query(
                    source=args.source,
                    ticker=ticker,
                    company_name=str(name),
                    left=left,
                    right=right,
                    include_ticker=bool(args.include_ticker),
                )
                error: Exception | None = None
                try:
                    articles: list[dict[str, str]] = []
                    for attempt in range(max(0, int(args.request_retries)) + 1):
                        try:
                            if args.source == "naver_search":
                                articles = fetch_naver_search(
                                    query,
                                    left,
                                    right,
                                    args.articles_per_window,
                                    args.request_timeout_sec,
                                )
                            else:
                                articles = fetch_rss(
                                    query,
                                    args.articles_per_window,
                                    args.request_timeout_sec,
                                )
                            error = None
                            break
                        except http_requests.RequestException as exc:
                            error = exc
                            if is_source_blocked(exc):
                                break
                            if attempt < max(0, int(args.request_retries)):
                                time.sleep(max(0.0, float(args.retry_backoff_sec)) * (2 ** attempt))
                    if error is not None:
                        raise error
                except http_requests.RequestException as exc:
                    request_errors += 1
                    ticker_failed = True
                    source_blocked = is_source_blocked(exc)
                    source_overloaded = is_source_overloaded(exc)
                    source_paused = source_blocked or source_overloaded
                    print(
                        json.dumps(
                            {
                                "ticker": ticker,
                                "window_start": str(left.date()),
                                "window_end": str(right.date()),
                                "error": type(exc).__name__,
                                "message": str(exc)[:180],
                                "source_blocked": source_blocked,
                                "source_overloaded": source_overloaded,
                                "source_pause_required": source_paused,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    break
                requests += 1
                ticker_requests += 1
                if not articles:
                    empty += 1
                saturated = len(articles) >= max(1, int(args.articles_per_window))
                if args.source == "google_rss" and args.adaptive_split and saturated:
                    split = split_saturated_window(left, right, args.min_window_days)
                    if split is not None:
                        ticker_split_windows += 1
                        first, second = split
                        pending_windows.appendleft(second)
                        pending_windows.appendleft(first)
                        if args.sleep_sec > 0:
                            time.sleep(args.sleep_sec)
                        continue
                    ticker_saturated_leaf_windows += 1
                ticker_leaf_windows += 1
                for response_rank, article in enumerate(articles, start=1):
                    article_id = stable_id(ticker, article.get("title", ""), article.get("link", ""))
                    if article_id in seen:
                        continue
                    seen.add(article_id)
                    published = pd.to_datetime(article.get("published"), errors="coerce", utc=True)
                    if not pd.isna(published):
                        published_date = published.tz_convert("Asia/Seoul").normalize().tz_localize(None)
                        if published_date < left.normalize() or published_date >= right.normalize():
                            ticker_out_of_window_articles += 1
                    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
                    record: dict[str, Any] = {
                        "schema_version": 2,
                        "id": article_id,
                        "collected_at_utc": collected_at,
                        "ts": collected_at,
                        "ticker": ticker,
                        "name": name,
                        "article": article,
                        "published": article.get("published", ""),
                        "source": f"{args.source}_backfill_raw",
                        "query_window": {"start": str(left.date()), "end": str(right.date())},
                        "acquisition": {
                            "provider": args.source,
                            "query": query,
                            "query_policy": ticker_query_policy,
                            "response_rank": response_rank,
                            "response_count": len(articles),
                            "result_cap": args.articles_per_window,
                            "shard_count": args.shard_count,
                            "shard_index": args.shard_index,
                        },
                    }
                    if args.with_heuristic:
                        event_payload = heuristic_event(
                            ticker=ticker,
                            name=name,
                            title=article.get("title", ""),
                            summary=article.get("summary", ""),
                            link=article.get("link", ""),
                        )
                        score = sum(
                            float(delta.get("delta", 0.0))
                            * float(delta.get("confidence", event_payload.get("confidence", 0.0)))
                            for delta in event_payload.get("node_deltas", [])
                            if str(delta.get("node", "")).replace("A", "") == ticker
                        )
                        record.update(
                            {
                                "llm_used": False,
                                "llm_error": "",
                                "event": event_payload,
                                "score_contribution": score,
                            }
                        )
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1
                    ticker_written += 1
                handle.flush()
                if requests % 100 == 0:
                    print(json.dumps({
                        "requests": requests,
                        "written": written,
                        "empty": empty,
                        "request_errors": request_errors,
                        "ticker": ticker,
                        "elapsed_sec": round(time.perf_counter() - started, 1),
                    }, ensure_ascii=False), flush=True)
                if args.sleep_sec > 0:
                    time.sleep(args.sleep_sec)
            if stopped:
                break
            if ticker_failed:
                if source_paused:
                    break
                continue
            coverage_handle.write(
                json.dumps(
                    {
                        "ticker": ticker,
                        "name": name,
                        "start": args.start,
                        "end": args.end,
                        "window_days": args.window_days,
                        "requests": ticker_requests,
                        "written": ticker_written,
                        "source": f"{args.source}_backfill_coverage",
                        "request_errors": 0,
                        "leaf_windows": ticker_leaf_windows,
                        "split_windows": ticker_split_windows,
                        "saturated_leaf_windows": ticker_saturated_leaf_windows,
                        "articles_per_window": args.articles_per_window,
                        "adaptive_split": bool(args.adaptive_split),
                        "min_window_days": args.min_window_days,
                        "out_of_window_articles": ticker_out_of_window_articles,
                        "status": (
                            "complete"
                            if ticker_saturated_leaf_windows == 0
                            else "incomplete_saturated"
                        ),
                        "query_policy": ticker_query_policy,
                        "shard_count": args.shard_count,
                        "shard_index": args.shard_index,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            coverage_handle.flush()
        if source_paused:
            print(
                json.dumps(
                    {
                        "source_blocked": source_blocked,
                        "source_overloaded": source_overloaded,
                        "requests": requests,
                        "written": written,
                        "request_errors": request_errors,
                        "message": "source returned a persistent block or overload response; wait before resuming",
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    print(json.dumps({
        "output": str(output),
        "coverage_output": str(coverage_output),
        "tickers": len(universe),
        "full_universe_tickers": full_universe_size,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "already_completed_tickers": len(completed_tickers),
        "requests": requests,
        "written": written,
        "empty": empty,
        "request_errors": request_errors,
        "source_blocked": source_blocked,
        "source_overloaded": source_overloaded,
        "source_pause_required": source_paused,
        "elapsed_sec": round(time.perf_counter() - started, 1),
    }, ensure_ascii=False, indent=2), flush=True)
    if source_paused:
        raise RuntimeError("news source requires a cooldown; resume only after waiting")


if __name__ == "__main__":
    main()
