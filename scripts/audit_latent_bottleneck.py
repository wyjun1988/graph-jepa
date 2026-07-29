from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _major_metrics(root: Path) -> tuple[dict[str, Any], Path]:
    path = root / "major_trajectory" / "summary.json"
    payload = _load_json(path)
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe major-trajectory artifact: {path}")
    return payload, path


def evaluate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    project_root: Path,
    raw_root: Path,
) -> dict[str, Any]:
    if contract.get("role") != "retrospective_frozen_latent_bottleneck_diagnostic":
        raise ValueError("invalid latent bottleneck contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("latent bottleneck contract must prohibit live orders")

    variants = contract["variants"]
    projected_root = project_root / str(variants["projected_reference"])
    direct_root = project_root / str(variants["direct_reference"])
    projected, projected_path = _major_metrics(projected_root)
    raw, raw_path = _major_metrics(raw_root)
    direct, direct_path = _major_metrics(direct_root)

    projected_auc = float(projected["roc_auc"])
    raw_auc = float(raw["roc_auc"])
    direct_auc = float(direct["roc_auc"])
    projected_lift = float(projected["systemic_impact_mass_lift_at_major_rate"])
    raw_lift = float(raw["systemic_impact_mass_lift_at_major_rate"])
    direct_lift = float(direct["systemic_impact_mass_lift_at_major_rate"])
    thresholds = contract["primary_checks"]
    checks = {
        "raw_minus_projected_major_auc": (
            raw_auc - projected_auc
            >= float(thresholds["raw_minus_projected_major_auc_at_least"])
        ),
        "raw_minus_projected_major_mass_lift": (
            raw_lift - projected_lift
            >= float(thresholds["raw_minus_projected_major_mass_lift_at_least"])
        ),
        "raw_major_auc": raw_auc >= float(thresholds["raw_major_auc_at_least"]),
    }
    projector_primary = all(checks.values())
    contributes = raw_auc > projected_auc and raw_lift > projected_lift
    if projector_primary:
        decision = "projector_primary_bottleneck_supported"
    elif contributes:
        decision = "projector_contributes_but_not_primary"
    else:
        decision = "projector_bottleneck_not_supported"

    raw_summary_path = raw_root / "summary.json"
    raw_summary = _load_json(raw_summary_path)
    if raw_summary.get("live_orders_allowed") is not False:
        raise ValueError("unsafe raw latent summary")
    return {
        "status": "complete",
        "role": "frozen_latent_bottleneck_contract_audit",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "metrics": {
            "projected": {
                "major_auc": projected_auc,
                "major_mass_lift": projected_lift,
            },
            "raw": {
                "major_auc": raw_auc,
                "major_mass_lift": raw_lift,
                "best_validation_score": float(raw_summary["best_validation_score"]),
                "test_trajectory_auc": float(
                    raw_summary["metrics"]["test"]["trajectory"]["roc_auc"]
                ),
            },
            "direct": {
                "major_auc": direct_auc,
                "major_mass_lift": direct_lift,
            },
            "raw_minus_projected": {
                "major_auc": raw_auc - projected_auc,
                "major_mass_lift": raw_lift - projected_lift,
            },
            "remaining_gap_raw_to_direct": {
                "major_auc": direct_auc - raw_auc,
                "major_mass_lift": direct_lift - raw_lift,
            },
        },
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "projector_primary_bottleneck_supported": projector_primary,
        "decision": decision,
        "inputs": {
            "projected_major": {
                "path": str(projected_path),
                "sha256": _sha256(projected_path),
            },
            "raw_summary": {
                "path": str(raw_summary_path),
                "sha256": _sha256(raw_summary_path),
            },
            "raw_major": {"path": str(raw_path), "sha256": _sha256(raw_path)},
            "direct_major": {
                "path": str(direct_path),
                "sha256": _sha256(direct_path),
            },
        },
        "test_used_for_selection": True,
        "evidence_role": "diagnosis_only_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the frozen raw-latent projector bottleneck diagnostic."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    payload = evaluate_contract(
        _load_json(contract_path),
        contract_path=contract_path,
        project_root=Path(args.project_root),
        raw_root=Path(args.raw_root),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(json.dumps({"decision": payload["decision"], "checks": payload["checks"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
