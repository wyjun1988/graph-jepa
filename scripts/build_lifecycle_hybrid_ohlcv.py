from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.lifecycle_ohlcv import build_lifecycle_hybrid_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an immutable 500-node lifecycle OHLCV release."
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--causal-manifest", required=True)
    parser.add_argument("--proxy-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--expected-tickers", type=int, default=500)
    parser.add_argument("--expected-proxy-tickers", type=int, default=47)
    parser.add_argument("--skip-provider-overlap", action="store_true")
    args = parser.parse_args()
    manifest = build_lifecycle_hybrid_release(
        universe_manifest=args.universe_manifest,
        causal_manifest=args.causal_manifest,
        proxy_cache_dir=args.proxy_cache_dir,
        output_dir=args.output_dir,
        start=args.start,
        end=args.end,
        expected_tickers=args.expected_tickers,
        expected_proxy_tickers=args.expected_proxy_tickers,
        validate_provider_overlap=not args.skip_provider_overlap,
    )
    print(json.dumps({"manifest": str(manifest), "live_orders_allowed": False}))


if __name__ == "__main__":
    main()
