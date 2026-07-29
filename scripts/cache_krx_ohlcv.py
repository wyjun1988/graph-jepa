from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.market_data import (
    fetch_krx_ohlcv,
    load_universe_manifest,
    select_krx_universe_from_listing,
    select_universe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache bounded KRX OHLCV histories for a fixed universe."
    )
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--universe", choices=["manual", "krx"], default="manual")
    parser.add_argument("--max-tickers", type=int, default=28)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--request-timeout-sec", type=float, default=20.0)
    parser.add_argument("--request-retries", type=int, default=2)
    parser.add_argument("--request-retry-delay-sec", type=float, default=0.5)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--summary-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    elif args.universe == "krx":
        universe = select_krx_universe_from_listing(args.max_tickers)
    else:
        universe = select_universe(args.max_tickers)

    data = fetch_krx_ohlcv(
        universe=universe,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
        min_rows=args.min_rows,
        request_timeout_sec=args.request_timeout_sec,
        request_retries=args.request_retries,
        request_retry_delay_sec=args.request_retry_delay_sec,
    )
    rows = {
        ticker: {
            "rows": int(len(frame)),
            "first_date": str(frame.index.min().date()),
            "last_date": str(frame.index.max().date()),
        }
        for ticker, frame in data.items()
    }
    payload = {
        "requested_tickers": len(universe),
        "loaded_tickers": len(data),
        "start": args.start,
        "end": args.end,
        "cache_dir": str(args.cache_dir),
        "tickers": rows,
    }
    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"cached requested={len(universe)} loaded={len(data)} cache_dir={args.cache_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
