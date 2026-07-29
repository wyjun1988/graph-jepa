from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


VARIANTS = ("baseline", "stock_quantiles", "external_identity", "combined")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_variant(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    major_path = root / "major_trajectory" / "summary.json"
    complete_path = root / "DIAGNOSTIC_COMPLETE"
    if not complete_path.is_file():
        raise ValueError(f"incomplete pooling variant: {root}")
    summary = _load_json(summary_path)
    major = _load_json(major_path)
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe variant summary: {summary_path}")
    if major.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe major-event summary: {major_path}")
    trajectory = summary["metrics"]["test"]["trajectory"]
    return {
        "best_validation_score": float(summary["best_validation_score"]),
        "trajectory_auc": float(trajectory["roc_auc"]),
        "trajectory_average_precision_lift": float(
            trajectory["average_precision_lift"]
        ),
        "trajectory_peak_horizon_accuracy": float(
            trajectory["peak_horizon_accuracy_on_events"]
        ),
        "trajectory_family_correlation": float(
            trajectory["mean_family_trajectory_correlation"]
        ),
        "trajectory_signature_cosine": float(
            trajectory["mean_transition_signature_cosine"]
        ),
        "major_auc": float(major["roc_auc"]),
        "major_average_precision_lift": float(major["average_precision_lift"]),
        "major_mass_lift": float(
            major["systemic_impact_mass_lift_at_major_rate"]
        ),
        "major_peak_horizon_accuracy": float(
            major["peak_horizon_accuracy_on_major_events"]
        ),
        "architecture": summary["architecture"],
        "parent_model_sha256": str(summary["parent_model_sha256"]),
        "model_dir": str(summary["model_dir"]),
        "split_dates": summary["split_dates"],
        "target_version": str(summary["target_version"]),
        "impact_metric_version": str(summary["impact_metric_version"]),
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "major": {"path": str(major_path), "sha256": _sha256(major_path)},
            "complete_marker": str(complete_path),
        },
    }


def _delta(candidate: Mapping[str, Any], baseline: Mapping[str, Any]) -> dict[str, float]:
    keys = (
        "best_validation_score",
        "trajectory_auc",
        "trajectory_average_precision_lift",
        "trajectory_peak_horizon_accuracy",
        "trajectory_family_correlation",
        "trajectory_signature_cosine",
        "major_auc",
        "major_average_precision_lift",
        "major_mass_lift",
        "major_peak_horizon_accuracy",
    )
    return {key: float(candidate[key]) - float(baseline[key]) for key in keys}


def evaluate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    roots: Mapping[str, Path],
) -> dict[str, Any]:
    if contract.get("role") != "retrospective_market_trajectory_pooling_factorial":
        raise ValueError("invalid pooling-factorial contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("pooling-factorial contract must prohibit live orders")
    missing = [variant for variant in VARIANTS if variant not in roots]
    if missing:
        raise ValueError(f"missing variant roots: {missing}")

    variants = {variant: _load_variant(roots[variant]) for variant in VARIANTS}
    baseline = variants["baseline"]
    deltas = {
        variant: _delta(variants[variant], baseline)
        for variant in VARIANTS
        if variant != "baseline"
    }
    ranking = sorted(
        VARIANTS,
        key=lambda variant: (
            variants[variant]["major_auc"],
            variants[variant]["major_mass_lift"],
            variants[variant]["trajectory_auc"],
        ),
        reverse=True,
    )

    expected_flags = {
        "baseline": (False, False),
        "stock_quantiles": (True, False),
        "external_identity": (False, True),
        "combined": (True, True),
    }
    architectures_match = all(
        bool(variants[name]["architecture"]["stock_quantile_pooling"])
        == expected_flags[name][0]
        and bool(variants[name]["architecture"]["preserve_external_identity"])
        == expected_flags[name][1]
        for name in VARIANTS
    )
    shared_fields = (
        "parent_model_sha256",
        "model_dir",
        "split_dates",
        "target_version",
        "impact_metric_version",
    )
    shared_inputs_match = all(
        variants[name][field] == baseline[field]
        for name in VARIANTS
        for field in shared_fields
    )

    combined_delta = deltas["combined"]
    combined_dominated = (
        combined_delta["major_auc"] <= 0.0
        and combined_delta["major_mass_lift"] <= 0.0
        and combined_delta["trajectory_auc"] <= 0.0
    )
    later_confirmation = contract["later_fold_confirmation"]
    checks = {
        "all_variants_completed": True,
        "variant_architectures_match_contract": architectures_match,
        "parent_split_and_targets_match": shared_inputs_match,
        "combined_not_dominated_by_baseline": not combined_dominated,
        "combined_major_auc_absolute": (
            variants["combined"]["major_auc"]
            >= float(later_confirmation["absolute_checks_each_fold"]["major_auc_at_least"])
        ),
        "combined_major_ap_lift_absolute": (
            variants["combined"]["major_average_precision_lift"]
            >= float(
                later_confirmation["absolute_checks_each_fold"]
                ["major_average_precision_lift_at_least"]
            )
        ),
        "combined_peak_accuracy_absolute": (
            variants["combined"]["major_peak_horizon_accuracy"]
            >= float(
                later_confirmation["absolute_checks_each_fold"]
                ["peak_horizon_accuracy_at_least"]
            )
        ),
    }
    combined_rejected = not all(checks.values())
    decision = (
        "combined_pooling_rejected_on_retrospective_diagnostic"
        if combined_rejected
        else "combined_pooling_requires_later_fold_confirmation"
    )
    return {
        "status": "complete",
        "role": "market_trajectory_pooling_factorial_contract_audit",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "variants": variants,
        "deltas_vs_baseline": deltas,
        "ranking_by_major_event_metrics": ranking,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "combined_rejected": combined_rejected,
        "decision": decision,
        "test_used_for_selection": True,
        "evidence_role": "retrospective_diagnosis_only_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the frozen-latent market trajectory pooling factorial."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--baseline-root", required=True)
    parser.add_argument("--stock-quantiles-root", required=True)
    parser.add_argument("--external-identity-root", required=True)
    parser.add_argument("--combined-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    payload = evaluate_contract(
        _load_json(contract_path),
        contract_path=contract_path,
        roots={
            "baseline": Path(args.baseline_root),
            "stock_quantiles": Path(args.stock_quantiles_root),
            "external_identity": Path(args.external_identity_root),
            "combined": Path(args.combined_root),
        },
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["checks"],
                "ranking": payload["ranking_by_major_event_metrics"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
