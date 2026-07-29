from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_contract(
    contract: Mapping[str, Any],
    summary: Mapping[str, Any],
    *,
    contract_path: Path,
    summary_path: Path,
    source_root: Path = Path("."),
) -> dict[str, Any]:
    if contract.get("role") != "retrospective_impact_direction_role_separation":
        raise ValueError("invalid role-separation contract")
    if contract.get("test_used_for_hypothesis_generation") is not True:
        raise ValueError("role-separation contract must declare reused test data")
    if contract.get("promotion_eligible") is not False:
        raise ValueError("role-separation contract cannot be promotion eligible")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("role-separation contract must prohibit live orders")
    if summary.get("status") != "complete":
        raise ValueError("incomplete role-separation summary")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError("unsafe role-separation summary")

    source_results = {
        relative: {
            "expected_sha256": str(expected),
            "observed_sha256": (
                _sha256(source_root / relative)
                if (source_root / relative).is_file()
                else None
            ),
        }
        for relative, expected in contract["source_pins"].items()
    }
    source_match = all(
        row["expected_sha256"] == row["observed_sha256"]
        for row in source_results.values()
    )
    inputs = summary["inputs"]
    expected_inputs = contract["input_pins"]
    input_match = (
        inputs["qlib_summary"]["sha256"] == expected_inputs["qlib_summary_sha256"]
        and inputs["family_query_summary"]["sha256"]
        == expected_inputs["family_query_summary_sha256"]
        and inputs["qlib_summary"]["checkpoint_sha256"]
        == expected_inputs["checkpoint_sha256"]
        and inputs["family_query_summary"]["parent_model_sha256"]
        == expected_inputs["checkpoint_sha256"]
    )
    calibration = summary["calibration"]
    calibration_match = (
        calibration["split"] == "validation_only"
        and "LogisticRegression(C=0.1)" in calibration["classifier"]
        and "Ridge(alpha=10.0)" in calibration["regressor"]
    )

    test = summary["test"]
    uncertainty = summary["uncertainty"]
    broad = test["broad_selloff"]
    family = test["family_query_broad_selloff_baseline"]
    direction = test["median_return_direction"]
    predicted_major = test["predicted_major_peak_direction"]
    thresholds = contract["diagnostic_checks"]

    def finite(value: Any) -> float:
        result = float(value)
        return result if math.isfinite(result) else float("nan")

    checks = {
        "source_pins_match_contract": source_match,
        "input_artifacts_match_contract": input_match,
        "calibration_contract_matches": calibration_match,
        "row_broad_selloff_auc_at_least": (
            finite(broad["roc_auc"])
            >= float(thresholds["row_broad_selloff_auc_at_least"])
        ),
        "row_broad_selloff_auc_not_below_family_query": (
            finite(broad["roc_auc"]) >= finite(family["roc_auc"])
        ),
        "row_broad_selloff_auc_bootstrap_lower_above": (
            finite(uncertainty["broad_selloff_auc"]["lower_95"])
            > float(thresholds["row_broad_selloff_auc_bootstrap_lower_above"])
        ),
        "row_direction_correlation_at_least": (
            finite(direction["pearson_correlation"])
            >= float(thresholds["row_direction_correlation_at_least"])
        ),
        "row_direction_correlation_bootstrap_lower_above": (
            finite(uncertainty["median_return_correlation"]["lower_95"])
            > float(thresholds["row_direction_correlation_bootstrap_lower_above"])
        ),
        "row_sign_accuracy_delta_vs_majority_at_least": (
            finite(direction["sign_accuracy_delta_vs_majority"])
            >= float(
                thresholds["row_sign_accuracy_delta_vs_majority_at_least"]
            )
        ),
        "predicted_major_rows_at_least": (
            int(predicted_major["rows"])
            >= int(thresholds["predicted_major_rows_at_least"])
        ),
        "predicted_major_direction_correlation_at_least": (
            finite(predicted_major["pearson_correlation"])
            >= float(
                thresholds["predicted_major_direction_correlation_at_least"]
            )
        ),
        "predicted_major_sign_delta_vs_majority_at_least": (
            finite(predicted_major["sign_accuracy_delta_vs_majority"])
            >= float(
                thresholds[
                    "predicted_major_sign_accuracy_delta_vs_majority_at_least"
                ]
            )
        ),
        "row_broad_selloff_auc_beats_95pct_placebo": (
            finite(broad["roc_auc"])
            > finite(uncertainty["placebo_broad_selloff_auc_95"])
        ),
        "predicted_major_sign_accuracy_beats_95pct_placebo": (
            finite(predicted_major["sign_accuracy"])
            > finite(uncertainty["placebo_predicted_major_sign_accuracy_95"])
        ),
    }
    passed = all(checks.values())
    return {
        "status": "complete",
        "role": "impact_direction_role_separation_contract_audit",
        "contract": str(contract_path),
        "contract_sha256": _sha256(contract_path),
        "summary": str(summary_path),
        "summary_sha256": _sha256(summary_path),
        "source_pins": source_results,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": len(checks),
        "passed": passed,
        "failures": [name for name, value in checks.items() if not value],
        "decision": (
            "authorize_role_separation_multifold_research_only"
            if passed
            else "reject_role_separation_on_retrospective_fold3"
        ),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Qlib direction behind a family-query impact gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--source-root", default=".")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    contract_path = Path(args.contract)
    summary_path = Path(args.summary)
    payload = evaluate_contract(
        _load(contract_path),
        _load(summary_path),
        contract_path=contract_path,
        summary_path=summary_path,
        source_root=Path(args.source_root),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(json.dumps({"decision": payload["decision"], "checks": payload["checks"]}))


if __name__ == "__main__":
    main()
