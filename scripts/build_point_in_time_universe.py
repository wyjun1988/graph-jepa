from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.universe_lifecycle import (
    build_lifecycle_catalog,
    enrich_current_listing_dates,
    rank_lifecycle_universe,
    select_lifecycle_universe,
    summarize_trailing_turnover,
)
from stock_v2.market_data import fetch_naver_ohlcv


DEFAULT_EXCLUDE_PATTERN = r"\uc2a4\ud329|\ub9ac\uce20|\uc6b0$|\uc6b0B$|\uc6b0\uc120\uc8fc|ETF|ETN|\uc778\ubc84\uc2a4|\ub808\ubc84\ub9ac\uc9c0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a fixed KRX equity universe using only listings active at a "
            "historical date and trailing, pre-date turnover."
        )
    )
    parser.add_argument("--as-of", required=True, help="Last date that may influence membership.")
    parser.add_argument("--rank-start", required=True, help="Inclusive start of the turnover window.")
    parser.add_argument("--rank-end", required=True, help="Inclusive end of the turnover window.")
    parser.add_argument("--top-n", type=int, default=500)
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--turnover-stat", choices=["mean", "median"], default="median")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--retry-delay-seconds", type=float, default=0.5)
    parser.add_argument("--request-timeout-sec", type=float, default=20.0)
    parser.add_argument("--cache-dir", default="data/cache_pit_universe")
    parser.add_argument("--output", required=True)
    parser.add_argument("--catalog-output", default=None)
    parser.add_argument("--failures-output", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-non-common-stock", action="store_true")
    parser.add_argument("--exclude-name-pattern", default=DEFAULT_EXCLUDE_PATTERN)
    return parser.parse_args()


def cache_path(cache_dir: Path, ticker: str, rank_start: str, rank_end: str) -> Path:
    start = rank_start.replace("-", "")
    end = rank_end.replace("-", "")
    return cache_dir / f"{ticker}_{start}_{end}.csv"


def load_history(
    ticker: str,
    *,
    rank_start: str,
    rank_end: str,
    cache_dir: Path,
    refresh: bool,
    retries: int,
    retry_delay_seconds: float,
    request_timeout_sec: float,
) -> pd.DataFrame:
    path = cache_path(cache_dir, ticker, rank_start, rank_end)
    if path.exists() and not refresh:
        return pd.read_csv(path, parse_dates=["Date"], index_col="Date")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            frame = fetch_naver_ohlcv(
                ticker,
                rank_start,
                rank_end,
                timeout_sec=request_timeout_sec,
            )
            if frame is None or frame.empty:
                return pd.DataFrame()
            frame = frame.copy().sort_index()
            frame.index.name = "Date"
            temporary = path.with_suffix(".tmp")
            frame.to_csv(temporary)
            temporary.replace(path)
            return frame
        except Exception as exc:  # Network sources can be intermittently unavailable.
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay_seconds * (attempt + 1))
    assert last_error is not None
    raise last_error


def fetch_score(
    ticker: str,
    *,
    rank_start: str,
    rank_end: str,
    cache_dir: Path,
    refresh: bool,
    retries: int,
    retry_delay_seconds: float,
    request_timeout_sec: float,
) -> Tuple[str, Dict[str, float], str | None]:
    try:
        frame = load_history(
            ticker,
            rank_start=rank_start,
            rank_end=rank_end,
            cache_dir=cache_dir,
            refresh=refresh,
            retries=retries,
            retry_delay_seconds=retry_delay_seconds,
            request_timeout_sec=request_timeout_sec,
        )
        return ticker, summarize_trailing_turnover(frame, start=rank_start, end=rank_end), None
    except Exception as exc:
        return ticker, {}, f"{type(exc).__name__}: {exc}"


def main() -> None:
    args = parse_args()
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1")
    if args.min_observations < 1:
        raise ValueError("--min-observations must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    import FinanceDataReader as fdr

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    current = enrich_current_listing_dates(
        fdr.StockListing("KRX"),
        fdr.StockListing("KRX-DESC"),
    )
    delisted = fdr.StockListing("KRX-DELISTING")
    catalog = build_lifecycle_catalog(current, delisted)
    eligible = select_lifecycle_universe(
        catalog,
        as_of=args.as_of,
        require_common_stock=not args.include_non_common_stock,
        exclude_name_pattern=args.exclude_name_pattern or None,
    )
    print(
        f"lifecycle catalog={len(catalog)} eligible_as_of={len(eligible)} "
        f"as_of={args.as_of}",
        flush=True,
    )

    if args.catalog_output:
        catalog_output = Path(args.catalog_output)
        catalog_output.parent.mkdir(parents=True, exist_ok=True)
        catalog_output.write_text(
            eligible.to_json(
                orient="records",
                force_ascii=False,
                date_format="iso",
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.dry_run:
        return

    scores: Dict[str, Dict[str, float]] = {}
    failures: Dict[str, str] = {}
    tickers = eligible["ticker"].astype(str).tolist()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_score,
                ticker,
                rank_start=args.rank_start,
                rank_end=args.rank_end,
                cache_dir=cache_dir,
                refresh=args.refresh,
                retries=args.retries,
                retry_delay_seconds=args.retry_delay_seconds,
                request_timeout_sec=args.request_timeout_sec,
            ): ticker
            for ticker in tickers
        }
        completed = 0
        for future in as_completed(futures):
            ticker, score, error = future.result()
            if error:
                failures[ticker] = error
            else:
                scores[ticker] = score
            completed += 1
            if completed % 100 == 0 or completed == len(tickers):
                print(
                    f"rank histories completed={completed}/{len(tickers)} "
                    f"success={len(scores)} failures={len(failures)}",
                    flush=True,
                )

    ranked = rank_lifecycle_universe(
        eligible,
        scores,
        top_n=args.top_n,
        min_observations=args.min_observations,
        turnover_key=f"{args.turnover_stat}_turnover",
    )
    if len(ranked) < args.top_n:
        raise RuntimeError(
            f"only {len(ranked)} securities met the causal liquidity requirement; "
            f"requested {args.top_n}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2,
        "selection_policy": {
            "type": "point_in_time_trailing_turnover",
            "as_of": args.as_of,
            "rank_start": args.rank_start,
            "rank_end": args.rank_end,
            "top_n": args.top_n,
            "min_observations": args.min_observations,
            "turnover_definition": f"{args.turnover_stat}(close * volume)",
            "require_common_stock": not args.include_non_common_stock,
            "exclude_name_pattern": args.exclude_name_pattern,
            "catalog_sources": ["FinanceDataReader:KRX", "FinanceDataReader:KRX-DELISTING"],
        },
        "source_counts": {
            "lifecycle_rows": int(len(catalog)),
            "eligible_as_of": int(len(eligible)),
            "rank_history_success": int(len(scores)),
            "rank_history_failures": int(len(failures)),
        },
        "universe": [
            {
                "ticker": str(row.ticker),
                "name": str(row.name),
                "market": str(row.market),
                "listing_date": pd.Timestamp(row.listing_date).date().isoformat(),
                "delisting_date": (
                    pd.Timestamp(row.delisting_date).date().isoformat()
                    if pd.notna(row.delisting_date)
                    else None
                ),
                "trailing_turnover": float(row.trailing_turnover),
                "rank_observations": int(row.rank_observations),
                "liquidity_rank": int(row.liquidity_rank),
            }
            for row in ranked.itertuples()
        ],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    failures_output = (
        Path(args.failures_output)
        if args.failures_output
        else output.with_suffix(output.suffix + ".failures.json")
    )
    failures_output.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"wrote {output} tickers={len(ranked)} failures={len(failures)} "
        f"top={ranked.iloc[0].ticker}",
        flush=True,
    )


if __name__ == "__main__":
    main()
