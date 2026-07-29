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
from stock_v2.opendart_client import OpenDartClient, collect_company_profiles


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
    parser = argparse.ArgumentParser(description="Backfill OpenDART company metadata for graph relations.")
    parser.add_argument("--universe", choices=["manual", "krx"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--api-key-env", default="OPENDART_API_KEY")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--sleep-sec", type=float, default=0.08)
    parser.add_argument("--output", default="data/metadata/opendart_company_profiles_krx100.jsonl")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} before running the OpenDART profile backfill")
    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    else:
        universe = (
            select_krx_universe_from_listing(args.max_tickers)
            if args.universe == "krx"
            else select_universe(args.max_tickers)
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    if args.resume and output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            try:
                completed.add(str(json.loads(line).get("ticker", "")))
            except json.JSONDecodeError:
                continue
    pending = [ticker for ticker, _name in universe if ticker not in completed]
    mode = "a" if args.resume and output.exists() else "w"
    written = 0
    with output.open(mode, encoding="utf-8") as handle:
        def persist(ticker: str, profile: dict[str, object] | None) -> None:
            nonlocal written
            if profile is not None:
                handle.write(json.dumps(profile, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                written += 1
            print(f"ticker={ticker} profile={profile is not None} total={written}", flush=True)

        collect_company_profiles(
            OpenDartClient(api_key, sleep_sec=args.sleep_sec),
            pending,
            on_ticker=persist,
        )
    print(f"wrote {output} profiles={written} completed_tickers={len(completed) + len(pending)}")


if __name__ == "__main__":
    main()
