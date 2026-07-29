from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.krx_open_api import (
    MARKET_API_IDS,
    KrxOpenApiClient,
    build_ticker_frames,
    load_and_validate_raw,
    persist_raw_response,
    raw_response_path,
)
from stock_v2.news_dataset import load_calendar
from stock_v2.dataset_curation import load_universe


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze official KRX daily market responses and build ticker files.")
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--calendar-path", action="append", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--api-key-env", default="KRX_OPEN_API_KEY")
    parser.add_argument("--raw-dir", default="data/raw/krx_open_api")
    parser.add_argument("--output-dir", default="data/krx_official_cache")
    parser.add_argument("--sleep-sec", type=float, default=0.15)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-sessions", type=int, default=0)
    parser.add_argument("--collect-only", action="store_true")
    args = parser.parse_args()

    load_env_file(Path(args.env_file).expanduser())
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} in the secure M1 Pro environment")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    sessions = list(
        load_calendar([Path(path) for path in args.calendar_path], start, end)
    )
    if args.max_sessions > 0:
        sessions = sessions[: args.max_sessions]
    raw_dir = Path(args.raw_dir)
    client = KrxOpenApiClient(
        api_key,
        timeout_sec=args.timeout_sec,
        sleep_sec=args.sleep_sec,
        retries=args.retries,
    )
    fetched = reused = 0
    started = time.perf_counter()
    for session_index, session in enumerate(sessions, start=1):
        date = pd.Timestamp(session).strftime("%Y%m%d")
        for market, api_id in MARKET_API_IDS.items():
            expected_path = raw_dir / api_id / date[:4] / date[4:6] / f"{date}.json"
            if expected_path.exists():
                load_and_validate_raw(expected_path, market, date)
                reused += 1
                continue
            response = client.fetch_daily(market, date)
            persist_raw_response(raw_dir, response)
            fetched += 1
        if session_index % 25 == 0:
            print(
                json.dumps(
                    {
                        "sessions": session_index,
                        "total_sessions": len(sessions),
                        "fetched_responses": fetched,
                        "reused_responses": reused,
                        "elapsed_sec": round(time.perf_counter() - started, 1),
                    }
                ),
                flush=True,
            )
    if args.collect_only:
        return 0

    universe = load_universe(Path(args.universe_manifest))
    frames = build_ticker_frames(raw_dir, sessions, set(universe))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = []
    for ticker, frame in frames.items():
        metadata = universe[ticker]
        listing = pd.to_datetime(metadata.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(metadata.get("delisting_date"), errors="coerce")
        if not frame.empty:
            dates = pd.to_datetime(frame["Date"])
            keep = (dates >= start) & (dates <= end)
            if not pd.isna(listing):
                keep &= dates >= listing
            if not pd.isna(delisting):
                keep &= dates <= delisting
            frame = frame.loc[keep].copy()
        path = output_dir / f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        frame.to_csv(path, index=False)
        output_files.append({"ticker": ticker, "path": path.name, "rows": len(frame)})
    manifest = {
        "schema_version": 1,
        "provider": "Korea Exchange KRX Open API",
        "official": True,
        "price_basis": "raw_as_traded",
        "start": str(start.date()),
        "end": str(end.date()),
        "sessions": len(sessions),
        "fetched_responses": fetched,
        "reused_responses": reused,
        "output_files": output_files,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
