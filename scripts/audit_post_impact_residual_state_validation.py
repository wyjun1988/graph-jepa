from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.post_impact_residual_selection import select_residual_candidate


CONTRACT_ROLE = "post_impact_residual_state_validation_contract"
EVALUATION_ROLE = "post_impact_residual_state_evaluation"
AUDIT_ROLE = "post_impact_residual_state_validation_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit causal residual-state validation selection reports."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--report-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duplicate-overlap-observed", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _validate_timestamp_proof(
    report: Mapping[str, Any], contract: Mapping[str, Any], fold: str
) -> None:
    counts = report.get("causal_shock_timestamp_counts")
    digests = report.get("causal_shock_timestamp_sha256")
    if not isinstance(counts, Mapping) or not isinstance(digests, Mapping):
        raise ValueError(f"missing causal timestamp proof: {fold}")
    total = 0
    for bucket in map(str, contract["clock_buckets"]):
        if set(counts.get(bucket, {})) != set(contract["subsets"]):
            raise ValueError(f"causal timestamp count subsets differ: {fold}/{bucket}")
        if set(digests.get(bucket, {})) != set(contract["subsets"]):
            raise ValueError(f"causal timestamp hash subsets differ: {fold}/{bucket}")
        for subset in map(str, contract["subsets"]):
            count = int(counts[bucket][subset])
            digest = str(digests[bucket][subset])
            if count < 0 or len(digest) != 64:
                raise ValueError(f"invalid causal timestamp proof: {fold}/{bucket}/{subset}")
            total += count
    if total <= 0:
        raise ValueError(f"causal timestamp selection is empty: {fold}")


def _validate_causal_contract(report: Mapping[str, Any], fold: str) -> None:
    expected = {
        "mature_before_forecast": True,
        "residual_target_derived_from_observed_decision_price": True,
        "future_target_values_stored_in_pending_state": False,
        "future_target_availability_used_for_enqueue": False,
        "dynamic_residual_reset_at_session_boundary": True,
        "adapter_coefficients_persist_across_sessions": True,
    }
    if report.get("causal_contract") != expected:
        raise ValueError(f"causal residual-state contract mismatch: {fold}")


def _best_candidate(selection: Mapping[str, Any]) -> dict[str, Any]:
    records = selection.get("candidates")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("residual selection has no candidate records")
    name, record = max(
        records.items(),
        key=lambda item: (float(item[1]["selection_score"]), str(item[0])),
    )
    return {
        "name": str(name),
        "primary_pearson_delta": float(record["primary"]["pearson"]["mean_delta"]),
        "primary_positive_strata": int(
            record["primary"]["pearson"]["positive_strata"]
        ),
        "primary_skill_delta": float(
            record["primary"]["skill_vs_zero_mse"]["mean_delta"]
        ),
        "fast_exit_skill_delta": float(
            record["fast_exit"]["skill_vs_zero_mse"]["mean_delta"]
        ),
        "checks": dict(record["checks"]),
    }


def audit_decision(all_selected: bool, duplicate_overlap_observed: bool) -> str:
    if not all_selected:
        return "residual_state_validation_rejected"
    if duplicate_overlap_observed:
        return "clean_validation_rerun_required"
    return "eligible_for_separate_test_contract"


def audit(
    contract_path: Path,
    report_root: Path,
    *,
    duplicate_overlap_observed: bool,
) -> dict[str, Any]:
    contract = load_json(contract_path, "residual validation contract")
    if contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid residual validation contract role")
    if contract.get("promotion_eligible") is not False or contract.get(
        "live_orders_allowed"
    ) is not False:
        raise ValueError("unsafe residual validation contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"residual validation source pin mismatch: {relative}")

    contract_hash = file_sha256(contract_path)
    candidate_names = [str(record["name"]) for record in contract["candidates"]]
    if len(candidate_names) != len(set(candidate_names)):
        raise ValueError("residual validation candidate names are duplicated")
    folds: dict[str, Any] = {}
    input_hashes: dict[str, str] = {"contract": contract_hash}
    all_selected = True
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        report_path = report_root / fold / "latent.json"
        report = load_json(report_path, f"{fold} residual validation report")
        if (
            report.get("role") != EVALUATION_ROLE
            or report.get("phase") != "validation_selection"
            or report.get("fold") != fold
            or report.get("model_name") != contract["selection_model"]
        ):
            raise ValueError(f"residual validation report identity mismatch: {fold}")
        if report.get("promotion_eligible") is not False or report.get(
            "live_orders_allowed"
        ) is not False:
            raise ValueError(f"unsafe residual validation report: {fold}")
        if report.get("contract_sha256") != contract_hash:
            raise ValueError(f"residual validation contract hash mismatch: {fold}")
        if report.get("candidate_configs") != contract["candidates"]:
            raise ValueError(f"residual validation candidate configs differ: {fold}")
        parity = report.get("reference_inference_parity")
        if not isinstance(parity, Mapping) or parity.get("passed") is not True:
            raise ValueError(f"checkpoint inference parity failed: {fold}")
        _validate_causal_contract(report, fold)
        _validate_timestamp_proof(report, contract, fold)

        model_spec = fold_spec["models"][contract["selection_model"]]
        checkpoint = Path(model_spec["training_dir"]) / "post_impact_reforecast.pt"
        summary_path = Path(model_spec["training_dir"]) / "summary.json"
        checkpoint_hash = file_sha256(checkpoint)
        summary_hash = file_sha256(summary_path)
        summary = load_json(summary_path, f"{fold} training summary")
        if checkpoint_hash != model_spec["checkpoint_sha256"]:
            raise ValueError(f"checkpoint pin mismatch: {fold}")
        if summary_hash != model_spec["summary_sha256"]:
            raise ValueError(f"training summary pin mismatch: {fold}")
        if summary.get("promotion_eligible") is not False or summary.get(
            "live_orders_allowed"
        ) is not False:
            raise ValueError(f"unsafe residual source model: {fold}")
        inputs = report["inputs"]
        if inputs.get("checkpoint_sha256") != checkpoint_hash:
            raise ValueError(f"report checkpoint hash mismatch: {fold}")
        if inputs.get("reference_summary_sha256") != summary_hash:
            raise ValueError(f"report summary hash mismatch: {fold}")
        if inputs.get("day_release_manifest_sha256") != contract["data_pins"][
            "day_release_manifest_sha256"
        ]:
            raise ValueError(f"report day-release hash mismatch: {fold}")
        if inputs.get("stale_cache_manifest_sha256") != contract["data_pins"][
            "stale_graph_cache_manifest_sha256"
        ]:
            raise ValueError(f"report stale-cache hash mismatch: {fold}")
        splits = report["splits"]
        expected_split = fold_spec["split"]
        if (
            splits["train"]["end"] != expected_split["train_end"]
            or splits["validation"]["end"] != expected_split["validation_end"]
            or splits["test"]["end"] != expected_split["test_end"]
        ):
            raise ValueError(f"residual validation split mismatch: {fold}")

        replay = select_residual_candidate(
            report["daily_rows"],
            candidate_names,
            baseline="base",
            primary_cells=contract["primary_cells"],
            fast_exit_cells=contract["fast_exit_safety_cells"],
            gates=contract["selection_gates"],
        )
        if canonical_json(replay) != canonical_json(report.get("selection")):
            raise ValueError(f"residual validation selection replay mismatch: {fold}")
        selected = bool(replay["selection_passed"])
        all_selected = all_selected and selected
        folds[fold] = {
            "report_sha256": file_sha256(report_path),
            "selection_passed": selected,
            "selected_candidate": replay["selected_candidate"],
            "eligible_candidates": list(replay["eligible_candidates"]),
            "best_candidate": _best_candidate(replay),
            "reference_inference_parity_passed": True,
        }
        input_hashes[f"{fold}.report"] = file_sha256(report_path)
        input_hashes[f"{fold}.checkpoint"] = checkpoint_hash
        input_hashes[f"{fold}.summary"] = summary_hash

    clean_chain = not duplicate_overlap_observed
    decision = audit_decision(all_selected, duplicate_overlap_observed)
    return {
        "schema_version": 1,
        "role": AUDIT_ROLE,
        "decision": decision,
        "all_folds_selected": all_selected,
        "test_evaluation_allowed": bool(all_selected and clean_chain),
        "duplicate_overlap_observed": bool(duplicate_overlap_observed),
        "overlap_evidence_used_for_rejection_only": bool(
            duplicate_overlap_observed and not all_selected
        ),
        "folds": folds,
        "inputs": input_hashes,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> int:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite residual validation audit: {output_path}")
    result = audit(
        Path(args.contract),
        Path(args.report_root),
        duplicate_overlap_observed=bool(args.duplicate_overlap_observed),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": result["decision"],
                "all_folds_selected": result["all_folds_selected"],
                "test_evaluation_allowed": result["test_evaluation_allowed"],
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
