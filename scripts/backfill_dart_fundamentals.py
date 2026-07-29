from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.market_data import (
    load_universe_manifest,
    select_krx_universe_from_listing,
    select_universe,
)
from stock_v2.opendart_client import OpenDartClient, collect_fundamental_observations


def load_env_file(path: str) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill point-in-time OpenDART fundamental observations.")
    parser.add_argument("--universe", choices=["manual", "krx"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--ticker", action="append", default=[], help="Restrict collection to explicit ticker(s).")
    parser.add_argument("--start-year", type=int, default=2020)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--api-key-env", default="OPENDART_API_KEY")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--sleep-sec", type=float, default=0.08)
    parser.add_argument("--output", default="data/fundamentals/opendart_krx100_2020_2026.jsonl")
    parser.add_argument("--coverage-output", default=None)
    parser.add_argument("--raw-cache-dir", default="data/raw/opendart")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} before running the OpenDART backfill")
    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    else:
        universe = (
            select_krx_universe_from_listing(args.max_tickers)
            if args.universe == "krx"
            else select_universe(args.max_tickers)
        )
    if args.ticker:
        requested = {str(ticker).replace("A", "").zfill(6) for ticker in args.ticker}
        universe = [(ticker, name) for ticker, name in universe if ticker in requested]
        missing = requested - {ticker for ticker, _name in universe}
        if missing:
            raise RuntimeError(f"requested tickers are absent from the universe: {sorted(missing)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage_output = Path(args.coverage_output or f"{output}.coverage.jsonl")
    coverage_output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    completion_source = coverage_output if coverage_output.exists() else output
    if args.resume and completion_source.exists():
        for line in completion_source.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(str(json.loads(line).get("ticker", "")))
            except json.JSONDecodeError:
                continue
    pending = [ticker for ticker, _name in universe if ticker not in completed]
    mode = "a" if args.resume and output.exists() else "w"
    coverage_mode = "a" if args.resume and coverage_output.exists() else "w"
    written = 0
    with output.open(mode, encoding="utf-8") as handle, coverage_output.open(
        coverage_mode, encoding="utf-8"
    ) as coverage_handle:
        def persist(ticker: str, records: list[dict[str, object]]) -> None:
            nonlocal written
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            coverage_handle.write(
                json.dumps(
                    {
                        "ticker": ticker,
                        "status": "complete",
                        "observations": len(records),
                        "start_year": args.start_year,
                        "end_year": args.end_year,
                        "source": "opendart_fundamental_coverage",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            coverage_handle.flush()
            written += len(records)
            print(f"ticker={ticker} observations={len(records)} total={written}", flush=True)

        collect_fundamental_observations(
            OpenDartClient(
                api_key,
                sleep_sec=args.sleep_sec,
                raw_cache_dir=args.raw_cache_dir,
            ),
            pending,
            args.start_year,
            args.end_year,
            on_ticker=persist,
        )
    print(f"wrote {output} new_observations={written} completed_tickers={len(completed) + len(pending)}")


if __name__ == "__main__":
    main()
