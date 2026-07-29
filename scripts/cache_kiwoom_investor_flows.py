from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_investor import fetch_investor_history
from stock_v2.market_data import load_universe_manifest, select_krx_universe_from_listing, select_universe
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache Kiwoom daily investor-flow history for a KRX universe.")
    parser.add_argument("--universe", choices=["manual", "krx"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--cache-dir", default="data/kiwoom_investor_cache")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=["real", "mock"], default="real")
    parser.add_argument("--sleep-sec", type=float, default=0.28)
    parser.add_argument("--max-pages", type=int, default=40)
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
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    broker = KiwoomRestBroker(
        KiwoomConfig(server=args.server, env_file=args.env_file, timeout_sec=15.0),
        dry_run=True,
    )
    if not broker.authenticate():
        raise RuntimeError("Kiwoom authentication failed for investor-flow cache")
    succeeded = 0
    failed = 0
    for index, (ticker, _name) in enumerate(universe, start=1):
        output = cache_dir / f"{ticker}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
        if args.resume and output.exists():
            succeeded += 1
            continue
        try:
            frame = fetch_investor_history(
                broker,
                ticker=ticker,
                start=args.start,
                end=args.end,
                sleep_sec=args.sleep_sec,
                max_pages=args.max_pages,
            )
            frame.index.name = "date"
            temporary = output.with_suffix(".csv.tmp")
            frame.to_csv(temporary)
            temporary.replace(output)
            succeeded += 1
            print(f"ticker={ticker} rows={len(frame)} complete={index}/{len(universe)}", flush=True)
            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))
        except Exception as exc:
            failed += 1
            print(f"ticker={ticker} error={type(exc).__name__}:{str(exc)[:180]}", flush=True)
            time.sleep(max(0.0, float(args.sleep_sec)))
    print(f"cache_dir={cache_dir} succeeded={succeeded} failed={failed} total={len(universe)}", flush=True)


if __name__ == "__main__":
    main()
