from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.rolling_validation import file_sha256, validate_rolling_contract


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a predeclared rolling validation contract."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    validated = validate_rolling_contract(contract)
    data_release = None
    if int(validated.get("schema_version", 0) or 0) >= 4:
        release = validated.get("data_release") or {}
        manifest_path = ROOT / str(release.get("manifest") or "")
        audit_path = ROOT / str(release.get("audit") or "")
        if not manifest_path.is_file() or not audit_path.is_file():
            raise ValueError("v4 lifecycle release artifacts are missing")
        manifest_sha256 = file_sha256(manifest_path)
        audit_sha256 = file_sha256(audit_path)
        if manifest_sha256 != str(release.get("manifest_sha256") or ""):
            raise ValueError("v4 lifecycle release manifest hash changed")
        if audit_sha256 != str(release.get("audit_sha256") or ""):
            raise ValueError("v4 lifecycle release audit hash changed")
        release_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if release_audit.get("status") != "pass":
            raise ValueError("v4 lifecycle release audit is blocked")
        data_release = {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": manifest_sha256,
            "audit": str(audit_path.relative_to(ROOT)),
            "audit_sha256": audit_sha256,
            "audit_status": release_audit.get("status"),
            "verified_files": int(release_audit.get("verified_files", 0)),
            "lifecycle_violations": int(
                release_audit.get("lifecycle_violations", -1)
            ),
        }
    payload = {
        "status": "pass",
        "contract_path": str(contract_path),
        "contract_file_sha256": file_sha256(contract_path),
        "canonical_contract_sha256": validated["contract_sha256"],
        "fold_count": len(validated["folds"]),
        "folds": validated["folds"],
        "data_release": data_release,
        "test_used_for_selection": bool(
            (validated.get("selection") or {}).get("test_used_for_selection")
        ),
        "live_orders_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "pass", "fold_count": payload["fold_count"]}))


if __name__ == "__main__":
    main()
