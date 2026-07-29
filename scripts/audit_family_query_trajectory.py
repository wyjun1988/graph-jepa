from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite diagnostic metric: {name}")
    return result


def _load_candidate(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    major_path = root / "major_trajectory" / "summary.json"
    head_path = root / "market_transition_head.pt"
    summary = _load_json(summary_path)
    major = _load_json(major_path)
    if summary.get("status") != "complete":
        raise ValueError(f"incomplete family-query summary: {summary_path}")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe family-query summary: {summary_path}")
    if major.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe family-query major summary: {major_path}")
    if not head_path.is_file():
        raise ValueError(f"missing family-query checkpoint: {head_path}")
    trajectory = summary["metrics"]["test"]["trajectory"]
    return {
        "parent_model_sha256": str(summary["parent_model_sha256"]),
        "split_dates": summary["split_dates"],
        "target_version": str(summary["target_version"]),
        "impact_metric_version": str(summary["impact_metric_version"]),
        "architecture": summary["architecture"],
        "best_validation_score": _finite_float(
            summary["best_validation_score"], "best_validation_score"
        ),
        "trajectory_auc": _finite_float(trajectory["roc_auc"], "trajectory_auc"),
        "trajectory_family_correlation": _finite_float(
            trajectory["mean_family_trajectory_correlation"],
            "trajectory_family_correlation",
        ),
        "major_auc": _finite_float(major["roc_auc"], "major_auc"),
        "major_mass_lift": _finite_float(
            major["systemic_impact_mass_lift_at_major_rate"], "major_mass_lift"
        ),
        "major_peak_horizon_accuracy": _finite_float(
            major["peak_horizon_accuracy_on_major_events"],
            "major_peak_horizon_accuracy",
        ),
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "major": {"path": str(major_path), "sha256": _sha256(major_path)},
            "head": {"path": str(head_path), "sha256": _sha256(head_path)},
        },
    }


def _load_stock_quantile_comparator(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    major_path = root / "major_trajectory" / "summary.json"
    summary = _load_json(summary_path)
    major = _load_json(major_path)
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe stock-quantile comparator: {summary_path}")
    if major.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe stock-quantile major comparator: {major_path}")
    architecture = summary["architecture"]
    if architecture.get("stock_quantile_pooling") is not True:
        raise ValueError("comparator is not the stock-quantile pooling variant")
    if architecture.get("family_query_pooling", False) is not False:
        raise ValueError("stock-quantile comparator unexpectedly uses family queries")
    return {
        "parent_model_sha256": str(summary["parent_model_sha256"]),
        "split_dates": summary["split_dates"],
        "target_version": str(summary["target_version"]),
        "impact_metric_version": str(summary["impact_metric_version"]),
        "major_auc": _finite_float(major["roc_auc"], "comparator_major_auc"),
        "inputs": {
            "summary": {"path": str(summary_path), "sha256": _sha256(summary_path)},
            "major": {"path": str(major_path), "sha256": _sha256(major_path)},
        },
    }


def evaluate_contract(
    contract: Mapping[str, Any],
    *,
    contract_path: Path,
    candidate_root: Path,
    stock_quantiles_root: Path,
    source_root: Path = Path("."),
) -> dict[str, Any]:
    if contract.get("role") != "retrospective_family_query_market_trajectory_diagnostic":
        raise ValueError("invalid family-query diagnostic contract role")
    if contract.get("test_used_for_hypothesis_generation") is not True:
        raise ValueError("family-query diagnostic must declare reused test data")
    if contract.get("promotion_eligible") is not False:
        raise ValueError("family-query diagnostic cannot be promotion eligible")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("family-query diagnostic must prohibit live orders")

    candidate = _load_candidate(candidate_root)
    comparator = _load_stock_quantile_comparator(stock_quantiles_root)
    architecture = candidate["architecture"]
    shared_fields = (
        "parent_model_sha256",
        "split_dates",
        "target_version",
        "impact_metric_version",
    )
    thresholds = contract["diagnostic_checks"]
    delta_major_auc = candidate["major_auc"] - comparator["major_auc"]
    source_pins = contract.get("source_pins", {})
    source_pin_results = {
        relative: {
            "expected_sha256": str(expected),
            "observed_sha256": (
                _sha256(source_root / relative)
                if (source_root / relative).is_file()
                else None
            ),
        }
        for relative, expected in source_pins.items()
    }
    source_pins_match = all(
        values["observed_sha256"] == values["expected_sha256"]
        for values in source_pin_results.values()
    )
    checks = {
        "source_pins_match_contract": source_pins_match,
        "parent_checkpoint_matches_contract": (
            candidate["parent_model_sha256"]
            == str(contract["parent"]["checkpoint_sha256"])
        ),
        "candidate_and_comparator_inputs_match": all(
            candidate[field] == comparator[field] for field in shared_fields
        ),
        "family_query_architecture_matches_contract": (
            architecture.get("family_query_pooling") is True
            and architecture.get("stock_quantile_pooling") is True
            and architecture.get("preserve_external_identity") is True
            and int(architecture.get("family_query_count", 0)) == 4
        ),
        "major_auc_at_least": (
            candidate["major_auc"] >= float(thresholds["major_auc_at_least"])
        ),
        "major_mass_lift_at_least": (
            candidate["major_mass_lift"]
            >= float(thresholds["major_mass_lift_at_least"])
        ),
        "major_peak_horizon_accuracy_at_least": (
            candidate["major_peak_horizon_accuracy"]
            >= float(thresholds["major_peak_horizon_accuracy_at_least"])
        ),
        "trajectory_auc_at_least": (
            candidate["trajectory_auc"]
            >= float(thresholds["trajectory_auc_at_least"])
        ),
        "mean_family_trajectory_correlation_at_least": (
            candidate["trajectory_family_correlation"]
            >= float(
                thresholds["mean_family_trajectory_correlation_at_least"]
            )
        ),
        "major_auc_delta_vs_stock_quantiles_at_least": (
            delta_major_auc
            >= float(thresholds["major_auc_delta_vs_stock_quantiles_at_least"])
        ),
    }
    passed = all(checks.values())
    decision = (
        "authorize_preregistered_fold4_fold5_confirmation_only"
        if passed
        else "reject_family_query_on_retrospective_fold3"
    )
    return {
        "status": "complete",
        "role": "family_query_market_trajectory_contract_audit",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "source_pins": source_pin_results,
        "candidate": candidate,
        "stock_quantiles_comparator": comparator,
        "major_auc_delta_vs_stock_quantiles": delta_major_auc,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "passed": passed,
        "failures": [name for name, value in checks.items() if not value],
        "decision": decision,
        "test_used_for_selection": True,
        "evidence_role": "retrospective_diagnosis_only_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a family-query market-trajectory diagnostic."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--stock-quantiles-root", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    contract_path = Path(args.contract)
    payload = evaluate_contract(
        _load_json(contract_path),
        contract_path=contract_path,
        candidate_root=Path(args.candidate_root),
        stock_quantiles_root=Path(args.stock_quantiles_root),
        source_root=Path(args.source_root),
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
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
