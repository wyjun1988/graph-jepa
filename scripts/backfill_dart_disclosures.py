from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.market_data import load_universe_manifest
from stock_v2.opendart_client import OpenDartClient


def load_env_file(path: str) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def disclosure_record(ticker: str, name: str, corp_code: str, row: Mapping[str, Any]) -> dict[str, Any]:
    receipt_no = str(row.get("rcept_no") or "").strip()
    receipt_date = str(row.get("rcept_dt") or "").strip()
    if len(receipt_date) != 8 or not receipt_date.isdigit():
        raise ValueError(f"invalid DART receipt date: {receipt_date!r}")
    published = f"{receipt_date[:4]}-{receipt_date[4:6]}-{receipt_date[6:]}"
    title = str(row.get("report_nm") or "").strip()
    if not receipt_no or not title:
        raise ValueError("DART disclosure requires receipt number and report name")
    identity = hashlib.sha256(f"opendart-disclosure-v1|{receipt_no}".encode("utf-8")).hexdigest()[:20]
    collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": 1,
        "id": identity,
        "collected_at_utc": collected_at,
        "ts": collected_at,
        "ticker": ticker,
        "name": name,
        "published": published,
        "source": "opendart_disclosure_raw",
        "article": {
            "title": title,
            "summary": "",
            "link": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt_no}",
            "published": published,
            "source": "금융감독원 DART",
        },
        "disclosure": {
            "corp_code": corp_code,
            "corp_name": str(row.get("corp_name") or name),
            "corp_class": str(row.get("corp_cls") or ""),
            "receipt_no": receipt_no,
            "filer_name": str(row.get("flr_nm") or ""),
            "remark": str(row.get("rm") or ""),
            "revision_preserved": True,
        },
        "acquisition": {
            "provider": "opendart",
            "endpoint": "list.json",
            "official": True,
        },
    }


def load_completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        row = json.loads(line)
        if row.get("status") == "complete":
            completed.add(str(row.get("ticker") or ""))
    return completed


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill all point-in-time OpenDART disclosure titles.")
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-07-10")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--api-key-env", default="OPENDART_API_KEY")
    parser.add_argument("--raw-cache-dir", default="data/raw/opendart_disclosures_v1")
    parser.add_argument("--output", required=True)
    parser.add_argument("--coverage-output", required=True)
    parser.add_argument("--sleep-sec", type=float, default=0.08)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_env_file(args.env_file)
    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"set {args.api_key_env} before collecting DART disclosures")
    universe = load_universe_manifest(args.universe_manifest)
    output = Path(args.output)
    coverage = Path(args.coverage_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    coverage.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(coverage) if args.resume else set()
    client = OpenDartClient(api_key, sleep_sec=args.sleep_sec, raw_cache_dir=args.raw_cache_dir)
    corp_codes = client.stock_to_corp_codes()
    mode = "a" if args.resume and output.exists() else "w"
    coverage_mode = "a" if args.resume and coverage.exists() else "w"
    written = failures = 0
    with output.open(mode, encoding="utf-8") as output_handle, coverage.open(
        coverage_mode, encoding="utf-8"
    ) as coverage_handle:
        for ticker, name in universe:
            if ticker in completed:
                continue
            corp_code = corp_codes.get(ticker, "")
            status = "complete"
            error = ""
            records: list[dict[str, Any]] = []
            if not corp_code:
                status = "missing_corp_code"
                error = "ticker is absent from OpenDART corporation-code registry"
            else:
                try:
                    seen_receipts: set[str] = set()
                    for row in client.disclosures(corp_code, args.start, args.end):
                        receipt = str(row.get("rcept_no") or "")
                        if receipt in seen_receipts:
                            continue
                        seen_receipts.add(receipt)
                        records.append(disclosure_record(ticker, name, corp_code, row))
                except Exception as exc:
                    status = "failed"
                    error = f"{type(exc).__name__}: {str(exc)[:500]}"
            for record in records:
                output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            output_handle.flush()
            coverage_handle.write(
                json.dumps(
                    {
                        "ticker": ticker,
                        "name": name,
                        "corp_code": corp_code,
                        "start": args.start,
                        "end": args.end,
                        "status": status,
                        "disclosures": len(records),
                        "request_errors": int(status == "failed"),
                        "error": error,
                        "source": "opendart_disclosure_coverage",
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            coverage_handle.flush()
            written += len(records)
            failures += int(status != "complete")
            print(
                json.dumps({"ticker": ticker, "status": status, "rows": len(records), "total": written}, ensure_ascii=False),
                flush=True,
            )
    print(
        json.dumps(
            {
                "tickers": len(universe),
                "already_completed": len(completed),
                "new_disclosures": written,
                "failures": failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
