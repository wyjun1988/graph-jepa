from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any
import uuid

import requests


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.naver_ohlcv_proxy import (
    NAVER_DAILY_URL,
    NAVER_HEADERS,
    file_sha256,
    parse_naver_daily_xml,
    proxy_csv_bytes,
    trim_proxy_frame,
    validate_proxy_frame,
    write_bytes_atomic,
)


COLLECTION_CONTRACT = "immutable_naver_adjusted_ohlcv_proxy_collection_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze lifecycle-trimmed Naver daily histories as a non-executable proxy release."
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--sleep-sec", type=float, default=0.05)
    return parser.parse_args()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _universe(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    result: list[dict[str, Any]] = []
    for row in rows:
        ticker = str(row.get("ticker", "")).replace("A", "").zfill(6)
        if not re.fullmatch(r"\d{6}", ticker):
            raise ValueError(f"invalid universe ticker: {ticker}")
        result.append(dict(row, ticker=ticker))
    tickers = [row["ticker"] for row in result]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe manifest contains duplicate tickers")
    return result


def _load_coverage(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        ticker = str(record.get("ticker", ""))
        if ticker in records:
            raise ValueError(f"duplicate proxy coverage record: {ticker}")
        records[ticker] = record
    return records


def _write_coverage(path: Path, universe: list[dict[str, Any]], records: dict[str, dict[str, Any]]) -> None:
    payload = b"".join(
        (
            json.dumps(records[row["ticker"]], ensure_ascii=False, sort_keys=True)
            + "\n"
        ).encode("utf-8")
        for row in universe
        if row["ticker"] in records
    )
    write_bytes_atomic(path, payload)


def _record_is_complete(record: dict[str, Any], work_dir: Path, run_id: str) -> bool:
    if (
        record.get("schema_version") != 2
        or record.get("status") != "ok"
        or record.get("run_id") != run_id
    ):
        return False
    raw = work_dir / str(record.get("raw_path", ""))
    output = work_dir / str(record.get("path", ""))
    return (
        raw.is_file()
        and output.is_file()
        and file_sha256(raw) == record.get("raw_sha256")
        and file_sha256(output) == record.get("sha256")
    )


def _publish(
    *,
    work_dir: Path,
    output_dir: Path,
    universe_path: Path,
    universe: list[dict[str, Any]],
    coverage: dict[str, dict[str, Any]],
    run_id: str,
    start: str,
    end: str,
) -> Path:
    if output_dir.exists():
        raise FileExistsError(f"immutable proxy release already exists: {output_dir}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir(parents=True)
    published: list[dict[str, Any]] = []
    try:
        for security in universe:
            ticker = security["ticker"]
            record = dict(coverage[ticker])
            if not _record_is_complete(record, work_dir, run_id):
                raise ValueError(f"proxy work record is incomplete: {ticker}")
            for field in ("raw_path", "path"):
                source = work_dir / record[field]
                destination = temporary / record[field]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                expected = record["raw_sha256" if field == "raw_path" else "sha256"]
                if file_sha256(destination) != expected:
                    raise ValueError(f"proxy publish checksum mismatch: {ticker} {field}")
            published.append(record)
        coverage_path = temporary / "coverage.jsonl"
        coverage_path.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in published
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "collection_contract": COLLECTION_CONTRACT,
            "run_id": run_id,
            "created_at_utc": datetime.now(tz=timezone.utc).isoformat(),
            "requested_start": start,
            "requested_end": end,
            "source": {
                "provider": "naver_fchart",
                "endpoint": NAVER_DAILY_URL + "{ticker}",
                "price_basis": "vendor-adjusted display history",
                "raw_exchange_prices_available": False,
                "execution_prices_available": False,
                "volume_eligible_for_model_input": False,
            },
            "universe_tickers": len(universe),
            "universe_manifest": str(universe_path),
            "universe_manifest_sha256": file_sha256(universe_path),
            "coverage": "coverage.jsonl",
            "coverage_sha256": file_sha256(coverage_path),
            "records_sha256": canonical_sha256(published),
            "records": published,
            "contract": {
                "immutable": True,
                "transactional_publish": True,
                "lifecycle_trimmed": True,
                "raw_xml_preserved": True,
                "proxy_execution_rule": "not executable under any condition",
            },
            "promotion_eligible": False,
            "live_orders_allowed": False,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_dir)
        return output_dir / "manifest.json"
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise ValueError("run-id contains unsupported characters")
    if args.retries < 0 or args.timeout_sec <= 0.0 or args.sleep_sec < 0.0:
        raise ValueError("retry, timeout, and sleep arguments are invalid")
    universe_path = Path(args.universe_manifest).resolve()
    universe = _universe(universe_path)
    work_dir = Path(args.work_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"immutable proxy release already exists: {output_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)
    coverage_path = work_dir / "coverage.jsonl"
    coverage = _load_coverage(coverage_path)
    session = requests.Session()
    failures = 0
    for index, security in enumerate(universe, start=1):
        ticker = security["ticker"]
        prior = coverage.get(ticker, {})
        if _record_is_complete(prior, work_dir, args.run_id):
            continue
        retrieved_at = datetime.now(tz=timezone.utc).isoformat()
        error: str | None = None
        response: requests.Response | None = None
        for attempt in range(args.retries + 1):
            try:
                response = session.get(
                    NAVER_DAILY_URL + ticker,
                    headers=NAVER_HEADERS,
                    timeout=float(args.timeout_sec),
                )
                response.raise_for_status()
                metadata, full = parse_naver_daily_xml(response.content)
                if metadata["symbol"] != ticker:
                    raise ValueError("Naver response symbol differs from requested ticker")
                frame = trim_proxy_frame(
                    full,
                    start=args.start,
                    end=args.end,
                    listing_date=security.get("listing_date"),
                    delisting_date=security.get("delisting_date"),
                )
                validation = validate_proxy_frame(
                    frame,
                    start=args.start,
                    end=args.end,
                    listing_date=security.get("listing_date"),
                    delisting_date=security.get("delisting_date"),
                )
                break
            except Exception as exc:
                response = None
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.retries:
                    time.sleep(0.5 * (attempt + 1))
        if response is None:
            failures += 1
            coverage[ticker] = {
                "schema_version": 2,
                "run_id": args.run_id,
                "ticker": ticker,
                "name": security.get("name"),
                "status": "error",
                "error": error,
                "retrieved_at_utc": retrieved_at,
            }
        else:
            raw_relative = Path("raw") / f"{ticker}.xml"
            output_relative = (
                Path("outputs")
                / f"{ticker}_{args.start.replace('-', '')}_{args.end.replace('-', '')}.csv"
            )
            raw_path = work_dir / raw_relative
            output_path = work_dir / output_relative
            write_bytes_atomic(raw_path, response.content)
            write_bytes_atomic(output_path, proxy_csv_bytes(frame))
            coverage[ticker] = {
                "schema_version": 2,
                "run_id": args.run_id,
                "ticker": ticker,
                "name": security.get("name"),
                "listing_date": security.get("listing_date"),
                "delisting_date": security.get("delisting_date"),
                "status": "ok",
                "retrieved_at_utc": retrieved_at,
                "raw_path": raw_relative.as_posix(),
                "raw_sha256": file_sha256(raw_path),
                "raw_bytes": raw_path.stat().st_size,
                "path": output_relative.as_posix(),
                "sha256": file_sha256(output_path),
                "bytes": output_path.stat().st_size,
                "source_metadata": metadata,
                **validation,
                "execution_eligible": False,
                "volume_eligible_for_model_input": False,
            }
        _write_coverage(coverage_path, universe, coverage)
        if index % 25 == 0 or index == len(universe):
            print(
                f"collected={index}/{len(universe)} failures={failures}",
                flush=True,
            )
        time.sleep(float(args.sleep_sec))
    if failures:
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "failures": failures,
                    "coverage": str(coverage_path),
                    "promotion_eligible": False,
                    "live_orders_allowed": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 2
    manifest = _publish(
        work_dir=work_dir,
        output_dir=output_dir,
        universe_path=universe_path,
        universe=universe,
        coverage=coverage,
        run_id=args.run_id,
        start=args.start,
        end=args.end,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "tickers": len(universe),
                "manifest": str(manifest),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
