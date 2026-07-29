from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.event_features import clean_ticker
from stock_v2.market_data import load_universe_manifest


def filter_jsonl_to_universe(
    input_path: Path,
    output_path: Path,
    allowed_tickers: set[str],
    ticker_field: str = "ticker",
    source_value: str | None = None,
) -> dict[str, Any]:
    """Copy valid JSONL records whose ticker belongs to a frozen universe."""

    seen: set[str] = set()
    report = {
        "input": str(input_path),
        "output": str(output_path),
        "allowed_tickers": len(allowed_tickers),
        "input_rows": 0,
        "written_rows": 0,
        "invalid_rows": 0,
        "outside_universe_rows": 0,
        "duplicate_rows": 0,
        "observed_tickers": 0,
        "missing_tickers": 0,
    }
    observed: set[str] = set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open(encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            report["input_rows"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                report["invalid_rows"] += 1
                continue
            if not isinstance(record, dict):
                report["invalid_rows"] += 1
                continue
            ticker = clean_ticker(record.get(ticker_field))
            if ticker not in allowed_tickers:
                report["outside_universe_rows"] += 1
                continue
            record = dict(record)
            record[ticker_field] = ticker
            if source_value is not None:
                record["source"] = source_value
            canonical = json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if canonical in seen:
                report["duplicate_rows"] += 1
                continue
            seen.add(canonical)
            observed.add(ticker)
            target.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            report["written_rows"] += 1
    report["observed_tickers"] = len(observed)
    report["missing_tickers"] = len(allowed_tickers - observed)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter JSONL sensor records to a frozen stock universe.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--ticker-field", default="ticker")
    parser.add_argument("--set-source", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    allowed = {ticker for ticker, _name in load_universe_manifest(args.universe_manifest)}
    report = filter_jsonl_to_universe(
        input_path=Path(args.input),
        output_path=Path(args.output),
        allowed_tickers=allowed,
        ticker_field=args.ticker_field,
        source_value=args.set_source,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
