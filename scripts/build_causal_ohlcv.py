from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.corporate_actions import build_causal_ohlcv


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a causal Kiwoom adjusted-return index plus raw OHLCV.")
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--adjusted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--events-output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--minimum-jump-ratio", type=float, default=0.02)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    universe_path = Path(args.universe_manifest)
    universe_payload = json.loads(universe_path.read_text(encoding="utf-8"))
    universe = universe_payload.get("universe", [])
    raw_dir = Path(args.raw_dir)
    adjusted_dir = Path(args.adjusted_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events: list[dict[str, Any]] = []
    missing: list[str] = []
    outputs: list[dict[str, Any]] = []
    action_tickers: set[str] = set()
    suffix = f"{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
    for index, security in enumerate(universe, start=1):
        ticker = str(security.get("ticker", "")).replace("A", "").zfill(6)
        raw_path = raw_dir / f"{ticker}_{suffix}"
        adjusted_path = adjusted_dir / f"{ticker}_{suffix}"
        if not raw_path.exists() or not adjusted_path.exists():
            missing.append(ticker)
            continue
        raw = pd.read_csv(raw_path, parse_dates=["Date"], index_col="Date")
        adjusted = pd.read_csv(adjusted_path, parse_dates=["Date"], index_col="Date")
        canonical, ticker_events = build_causal_ohlcv(
            raw,
            adjusted,
            ticker=ticker,
            minimum_jump_ratio=args.minimum_jump_ratio,
        )
        output = output_dir / f"{ticker}_{suffix}"
        temporary = output.with_suffix(output.suffix + ".tmp")
        canonical.to_csv(temporary)
        temporary.replace(output)
        events.extend(ticker_events)
        action_tickers.update([ticker] if ticker_events else [])
        outputs.append(
            {
                "ticker": ticker,
                "rows": len(canonical),
                "first_date": str(canonical.index.min().date()) if len(canonical) else None,
                "last_date": str(canonical.index.max().date()) if len(canonical) else None,
                "corporate_actions": len(ticker_events),
                "path": str(output),
                "sha256": file_sha256(output),
            }
        )
        if index % 50 == 0:
            print(f"processed={index}/{len(universe)} outputs={len(outputs)} missing={len(missing)}", flush=True)

    events.sort(key=lambda row: (str(row["effective_date"]), str(row["ticker"])))
    write_jsonl(Path(args.events_output), events)
    manifest = {
        "schema_version": 2,
        "source": {
            "provider": "kiwoom_rest_ka10081",
            "raw_price_basis": "raw exchange price",
            "adjusted_price_basis": "vendor back-adjusted through release end",
            "canonical_price_basis": "forward index reconstructed from adjacent vendor-adjusted returns",
            "execution_price_basis": "RawOHLC columns only",
        },
        "method": {
            "minimum_factor_jump_ratio": args.minimum_jump_ratio,
            "factor": "median(adjusted/raw OHLC); volume is an independent corroboration",
            "causal_invariance": "future piecewise-constant back-adjustments cancel in adjacent returns",
            "notional_invariant": "canonical_close*canonical_volume == raw_close*raw_volume",
        },
        "universe_manifest": str(universe_path),
        "universe_sha256": file_sha256(universe_path),
        "start": args.start,
        "end": args.end,
        "expected_tickers": len(universe),
        "output_tickers": len(outputs),
        "missing_tickers": missing,
        "corporate_action_tickers": len(action_tickers),
        "corporate_action_events": len(events),
        "events_path": args.events_output,
        "events_sha256": file_sha256(Path(args.events_output)),
        "outputs": outputs,
    }
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_tickers": len(outputs),
                "missing_tickers": len(missing),
                "corporate_action_tickers": len(action_tickers),
                "corporate_action_events": len(events),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    return 0 if not missing else 2


if __name__ == "__main__":
    sys.exit(main())
