from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_ohlcv import (
    canonical_json_sha256,
    fetch_kiwoom_ohlcv_history,
    trim_to_security_lifecycle,
    write_immutable_raw_page,
)
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


SOURCE_URL = "https://openapi.kiwoom.com/m/guide/apiguide?jobTpCode=07&apiId=ka10081"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect immutable raw and adjusted Kiwoom ka10081 histories."
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--basis", choices=["raw", "adjusted", "both"], default="both")
    parser.add_argument("--cache-dir", default="data/kiwoom_ohlcv_cache")
    parser.add_argument("--raw-cache-dir", default="data/raw/kiwoom_ohlcv")
    parser.add_argument("--coverage-output", default="data/kiwoom_ohlcv_cache/coverage.jsonl")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=["real", "mock"], default="real")
    parser.add_argument("--sleep-sec", type=float, default=0.22)
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-tickers", type=int, default=0)
    return parser.parse_args()


def load_universe(path: Path, max_tickers: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    selected = rows[: max_tickers or None]
    tickers = [str(row.get("ticker", "")).replace("A", "").zfill(6) for row in selected]
    if any(not re.fullmatch(r"\d{6}", ticker) for ticker in tickers):
        raise ValueError("universe manifest contains an invalid ticker")
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe manifest contains duplicate tickers")
    return [dict(row, ticker=ticker) for row, ticker in zip(selected, tickers)]


def load_coverage(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[(str(record.get("ticker")), str(record.get("basis")))] = record
    return records


def write_coverage(path: Path, records: dict[tuple[str, str], dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and dash")
    universe_path = Path(args.universe_manifest)
    universe = load_universe(universe_path, max(0, args.max_tickers))
    cache_root = Path(args.cache_dir)
    raw_root = Path(args.raw_cache_dir) / args.run_id
    coverage_path = Path(args.coverage_output)
    coverage = load_coverage(coverage_path)
    bases = ["raw", "adjusted"] if args.basis == "both" else [args.basis]
    broker = KiwoomRestBroker(
        KiwoomConfig(server=args.server, env_file=args.env_file, timeout_sec=30.0),
        dry_run=True,
    )
    if not broker.authenticate():
        raise RuntimeError("Kiwoom authentication failed for read-only OHLCV collection")

    counts = {"ok": 0, "empty": 0, "errors": 0, "skipped": 0}
    for ticker_index, security in enumerate(universe, start=1):
        ticker = security["ticker"]
        for basis in bases:
            output = cache_root / basis / f"{ticker}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
            prior = coverage.get((ticker, basis), {})
            if (
                args.resume
                and prior.get("status") == "ok"
                and prior.get("run_id") == args.run_id
                and output.exists()
                and prior.get("output_sha256") == file_sha256(output)
            ):
                counts["skipped"] += 1
                continue

            page_digests: list[str] = []
            collected_at = datetime.now(tz=timezone.utc).isoformat()

            def save_page(page_index: int, response: dict[str, Any], has_more: bool) -> None:
                response_sha256 = canonical_json_sha256(response)
                envelope = {
                    "schema_version": 1,
                    "source": "kiwoom_rest",
                    "source_url": SOURCE_URL,
                    "endpoint": "/api/dostk/chart",
                    "api_id": "ka10081",
                    "run_id": args.run_id,
                    "ticker": ticker,
                    "basis": basis,
                    "request": {
                        "stk_cd": ticker,
                        "base_dt": args.end.replace("-", ""),
                        "upd_stkpc_tp": "1" if basis == "adjusted" else "0",
                    },
                    "page_index": page_index,
                    "has_more": bool(has_more),
                    "retrieved_at_utc": collected_at,
                    "response_sha256": response_sha256,
                    "response": response,
                }
                raw_path = raw_root / basis / ticker / f"page_{page_index:04d}.json"
                page_digests.append(write_immutable_raw_page(raw_path, envelope))

            try:
                frame = fetch_kiwoom_ohlcv_history(
                    broker,
                    ticker,
                    args.start,
                    args.end,
                    adjusted=basis == "adjusted",
                    sleep_sec=args.sleep_sec,
                    max_pages=args.max_pages,
                    raw_page_sink=save_page,
                )
                source_rows = len(frame)
                frame, removed = trim_to_security_lifecycle(
                    frame,
                    listing_date=security.get("listing_date"),
                    delisting_date=security.get("delisting_date"),
                    release_start=args.start,
                    release_end=args.end,
                )
                if frame.empty:
                    status = "empty"
                    output_sha256 = None
                    counts["empty"] += 1
                else:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    frame.index.name = "Date"
                    temporary = output.with_suffix(output.suffix + ".tmp")
                    frame.to_csv(temporary)
                    temporary.replace(output)
                    output_sha256 = file_sha256(output)
                    status = "ok"
                    counts["ok"] += 1
                record = {
                    "schema_version": 1,
                    "ticker": ticker,
                    "name": security.get("name"),
                    "market": security.get("market"),
                    "listing_date": security.get("listing_date"),
                    "delisting_date": security.get("delisting_date"),
                    "basis": basis,
                    "run_id": args.run_id,
                    "status": status,
                    "source_rows": source_rows,
                    "rows": len(frame),
                    "outside_lifecycle_rows": removed,
                    "first_date": str(frame.index.min().date()) if len(frame) else None,
                    "last_date": str(frame.index.max().date()) if len(frame) else None,
                    "raw_page_count": len(page_digests),
                    "raw_page_envelope_sha256": page_digests,
                    "output": str(output),
                    "output_sha256": output_sha256,
                    "price_basis": "back_adjusted" if basis == "adjusted" else "raw",
                    "collected_at_utc": collected_at,
                    "error": None,
                }
                print(
                    f"ticker={ticker} basis={basis} rows={len(frame)} pages={len(page_digests)} "
                    f"complete={ticker_index}/{len(universe)}",
                    flush=True,
                )
            except Exception as exc:
                counts["errors"] += 1
                record = {
                    "schema_version": 1,
                    "ticker": ticker,
                    "name": security.get("name"),
                    "market": security.get("market"),
                    "listing_date": security.get("listing_date"),
                    "delisting_date": security.get("delisting_date"),
                    "basis": basis,
                    "run_id": args.run_id,
                    "status": "error",
                    "raw_page_count": len(page_digests),
                    "raw_page_envelope_sha256": page_digests,
                    "collected_at_utc": collected_at,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
                print(f"ticker={ticker} basis={basis} error={record['error']}", flush=True)
            coverage[(ticker, basis)] = record
            write_coverage(coverage_path, coverage)
            if args.sleep_sec > 0:
                time.sleep(float(args.sleep_sec))

    summary = {
        "universe_manifest": str(universe_path),
        "universe_sha256": file_sha256(universe_path),
        "start": args.start,
        "end": args.end,
        "run_id": args.run_id,
        "bases": bases,
        "securities": len(universe),
        **counts,
    }
    summary_path = coverage_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if counts["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
