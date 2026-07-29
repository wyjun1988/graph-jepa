from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_naver_proxy_ohlcv import COLLECTION_CONTRACT
from stock_v2.naver_ohlcv_proxy import (
    file_sha256,
    parse_naver_daily_xml,
    proxy_csv_bytes,
    trim_proxy_frame,
    validate_proxy_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild and audit an immutable Naver proxy OHLCV release."
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def audit_release(
    release_dir: Path,
    universe_manifest: Path,
) -> dict[str, Any]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if manifest.get("collection_contract") != COLLECTION_CONTRACT:
        failures.append("unsupported_collection_contract")
    if manifest.get("live_orders_allowed") is not False:
        failures.append("live_orders_not_explicitly_prohibited")
    contract = manifest.get("contract")
    if not isinstance(contract, dict) or any(
        contract.get(name) is not True
        for name in (
            "immutable",
            "transactional_publish",
            "lifecycle_trimmed",
            "raw_xml_preserved",
        )
    ):
        failures.append("release_contract_claims_are_incomplete")
    if file_sha256(universe_manifest) != manifest.get("universe_manifest_sha256"):
        failures.append("universe_manifest_sha256_mismatch")
    universe_payload = json.loads(universe_manifest.read_text(encoding="utf-8"))
    universe = universe_payload.get("universe", [])
    universe_by_ticker = {
        str(row.get("ticker", "")).replace("A", "").zfill(6): row
        for row in universe
    }
    records = list(manifest.get("records") or [])
    records_by_ticker = {str(row.get("ticker", "")): row for row in records}
    if (
        len(universe_by_ticker) != len(universe)
        or set(records_by_ticker) != set(universe_by_ticker)
        or len(records_by_ticker) != len(records)
    ):
        failures.append("universe_and_proxy_record_axes_differ")
    coverage_path = release_dir / str(manifest.get("coverage", ""))
    if not coverage_path.is_file() or file_sha256(coverage_path) != manifest.get(
        "coverage_sha256"
    ):
        failures.append("coverage_sha256_mismatch")
    if canonical_sha256(records) != manifest.get("records_sha256"):
        failures.append("records_sha256_mismatch")

    verified = 0
    observed_rows = 0
    suspended_rows = 0
    for ticker in sorted(set(records_by_ticker) & set(universe_by_ticker)):
        record = records_by_ticker[ticker]
        security = universe_by_ticker[ticker]
        try:
            if (
                record.get("schema_version") != 2
                or record.get("status") != "ok"
                or record.get("execution_eligible") is not False
            ):
                raise ValueError("proxy record status or execution contract is invalid")
            raw_path = release_dir / str(record["raw_path"])
            output_path = release_dir / str(record["path"])
            if file_sha256(raw_path) != record.get("raw_sha256"):
                raise ValueError("raw XML checksum mismatch")
            if file_sha256(output_path) != record.get("sha256"):
                raise ValueError("normalized CSV checksum mismatch")
            metadata, full = parse_naver_daily_xml(raw_path.read_bytes())
            if metadata != record.get("source_metadata") or metadata["symbol"] != ticker:
                raise ValueError("raw XML metadata differs from the release record")
            frame = trim_proxy_frame(
                full,
                start=manifest["requested_start"],
                end=manifest["requested_end"],
                listing_date=security.get("listing_date"),
                delisting_date=security.get("delisting_date"),
            )
            validation = validate_proxy_frame(
                frame,
                start=manifest["requested_start"],
                end=manifest["requested_end"],
                listing_date=security.get("listing_date"),
                delisting_date=security.get("delisting_date"),
            )
            regenerated_sha256 = hashlib.sha256(proxy_csv_bytes(frame)).hexdigest()
            if regenerated_sha256 != record.get("sha256"):
                raise ValueError("normalized CSV cannot be regenerated from raw XML")
            for name, value in validation.items():
                if record.get(name) != value:
                    raise ValueError(f"proxy validation metadata differs: {name}")
            observed_rows += int(validation["observed_price_rows"])
            suspended_rows += int(validation["suspended_rows"])
            verified += 1
        except Exception as exc:
            failures.append(f"{ticker}:{type(exc).__name__}:{exc}")
    report = {
        "schema_version": 1,
        "audit_contract": "immutable_naver_adjusted_ohlcv_proxy_audit_v1",
        "audited_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        "release_manifest_sha256": file_sha256(manifest_path),
        "universe_tickers": len(universe_by_ticker),
        "records": len(records),
        "records_verified": verified,
        "observed_price_rows": observed_rows,
        "suspended_rows": suspended_rows,
        "failures": failures,
        "integrity_gate_passed": not failures,
        "execution_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    return report


def main() -> int:
    args = parse_args()
    report = audit_release(
        Path(args.release_dir),
        Path(args.universe_manifest),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["integrity_gate_passed"] else "failed",
                "records_verified": report["records_verified"],
                "failures": len(report["failures"]),
                "output": str(output),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["integrity_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
