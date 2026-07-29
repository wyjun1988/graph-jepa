from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_causal_shock_buckets import (
    METRICS,
    SHOCK_CONTRACT,
    TIMESTAMP_FINGERPRINT,
    _aggregate_cells,
    _contract_cells,
)
from scripts.audit_post_impact_clock_bucket_increment import safe_json, sha256_file


CONTRACT_ROLE = "post_impact_node_surprise_message_screen_contract"
AUDIT_ROLE = "post_impact_node_surprise_message_screen_audit"
EXPECTED_MODELS = {
    "none",
    "surprise_disabled",
    "surprise_causal",
    "surprise_node_permuted",
}


def validation_shock_daily_frame(
    payload: Mapping[str, Any],
    horizon: str,
    bucket: str,
    subset: str,
) -> pd.DataFrame:
    if payload.get("test_evaluated") is not False or payload.get("test") is not None:
        raise ValueError("validation screen report evaluated the test split")
    rows = payload["validation"][
        "clock_bucket_causal_shock_daily_node_endpoint_rows"
    ][horizon][bucket][subset]
    frame = pd.DataFrame(rows)
    required = {"date", "count", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(
            f"validation causal-shock rows are missing fields: {sorted(missing)}"
        )
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("validation causal-shock rows are empty or duplicated")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("validation causal-shock metrics contain non-finite values")
    return frame[["date", "count", *METRICS]]


def _validate_shock_contract(
    payload: Mapping[str, Any],
    expected_subsets: list[str],
    expected_lookback: int,
    label: str,
) -> None:
    record = payload.get("validation", {}).get(
        "clock_bucket_causal_shock_contract"
    )
    if not isinstance(record, Mapping) or record.get("name") != SHOCK_CONTRACT:
        raise ValueError(f"validation causal-shock contract mismatch: {label}")
    if record.get("point_in_time_observed_only") is not True:
        raise ValueError(f"validation shock selection is not point-in-time: {label}")
    if record.get("future_realized_labels_used_for_selection") is not False:
        raise ValueError(f"future labels selected validation shock rows: {label}")
    if int(record.get("recent_lookback_minutes", -1)) != expected_lookback:
        raise ValueError(f"validation shock lookback mismatch: {label}")
    if list(record.get("subsets", [])) != expected_subsets:
        raise ValueError(f"validation shock subset mismatch: {label}")
    if record.get("timestamp_fingerprint") != TIMESTAMP_FINGERPRINT:
        raise ValueError(f"validation shock fingerprint mismatch: {label}")


def _validate_model_artifacts(
    contract: Mapping[str, Any],
    name: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    report_path = Path(str(spec["report"]))
    training_dir = Path(str(spec["training_dir"]))
    summary_path = training_dir / "summary.json"
    checkpoint_path = training_dir / "post_impact_reforecast.pt"
    report = safe_json(report_path, f"{name} validation shock report")
    summary = safe_json(summary_path, f"{name} training summary")

    for label, payload in (("report", report), ("summary", summary)):
        if payload.get("live_orders_allowed") is not False:
            raise ValueError(f"unsafe {name} {label}")
        if payload.get("promotion_eligible") is not False:
            raise ValueError(f"promotion-enabled {name} {label}")
        if payload.get("evaluation_scope") != "validation_only":
            raise ValueError(f"{name} {label} is not validation-only")
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"{name} {label} evaluated test labels")
        if payload.get("test") is not None:
            raise ValueError(f"{name} {label} contains test metrics")

    if summary.get("test_loss") is not None:
        raise ValueError(f"{name} summary contains a test loss")
    if report.get("variant") != "latent" or summary.get("variant") != "latent":
        raise ValueError(f"{name} model variant mismatch")
    expected_mode = str(spec["graph_message_mode"])
    if report.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{name} report graph-message mode mismatch")
    if summary.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{name} summary graph-message mode mismatch")
    if summary.get("graph_message_edges_used") is not bool(
        spec["graph_message_edges_used"]
    ):
        raise ValueError(f"{name} graph edge-use contract mismatch")
    if summary.get("stale_stock_graph_used") is not False:
        raise ValueError(f"{name} unexpectedly used the stale graph encoder")
    if report.get("splits") != summary.get("splits"):
        raise ValueError(f"{name} report and summary splits differ")
    split = contract["split"]
    realized = report["splits"]
    if (
        realized["train"]["end"] != split["train_end"]
        or realized["validation"]["end"] != split["validation_end"]
        or realized["test"]["end"] != split["test_end"]
    ):
        raise ValueError(f"{name} split does not match the frozen contract")
    expected_audit_days = int(realized["train"]["days"]) + int(
        realized["validation"]["days"]
    )
    for audit_name in ("context_map_audit", "latent_context_map_audit"):
        report_audit = report.get(audit_name)
        summary_audit = summary.get(audit_name)
        if report_audit != summary_audit:
            raise ValueError(f"{name} {audit_name} differs after checkpoint reload")
        if not isinstance(report_audit, Mapping):
            raise ValueError(f"{name} is missing {audit_name}")
        if int(report_audit.get("dates", -1)) != expected_audit_days:
            raise ValueError(f"{name} {audit_name} includes a held-out test date")
        if int(report_audit.get("future_context_violations", -1)) != 0:
            raise ValueError(f"{name} {audit_name} is non-causal")

    parity = report.get("reference_inference_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError(f"{name} checkpoint inference parity failed")
    _validate_shock_contract(
        report,
        [str(value) for value in contract["subsets"]],
        int(contract["recent_lookback_minutes"]),
        name,
    )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    summary_sha256 = sha256_file(summary_path)
    data_pins = contract["data_pins"]
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{name} summary checkpoint hash mismatch")
    if report["inputs"].get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{name} report checkpoint hash mismatch")
    if report["inputs"].get("reference_summary_sha256") != summary_sha256:
        raise ValueError(f"{name} report summary hash mismatch")
    if report["inputs"].get("day_release_manifest_sha256") != data_pins[
        "day_release_manifest_sha256"
    ]:
        raise ValueError(f"{name} day-release hash mismatch")
    if report["inputs"].get("stale_cache_manifest_sha256") != data_pins[
        "stale_graph_cache_manifest_sha256"
    ]:
        raise ValueError(f"{name} stale-cache hash mismatch")

    return report, summary, {
        "report": sha256_file(report_path),
        "summary": summary_sha256,
        "checkpoint": checkpoint_sha256,
    }


def _comparison_result(
    contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
    comparison: Mapping[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_name = str(comparison["candidate"])
    comparator_name = str(comparison["comparator"])
    primary_cells = _contract_cells(contract, "primary_cells")
    fast_cells = _contract_cells(contract, "fast_exit_safety_cells")
    cells = list(dict.fromkeys(primary_cells + fast_cells))
    cell_frames: dict[
        tuple[str, str, str],
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ] = {}
    rows: list[dict[str, Any]] = []
    for horizon, bucket, subset in cells:
        candidate = validation_shock_daily_frame(
            reports[candidate_name], horizon, bucket, subset
        )
        comparator = validation_shock_daily_frame(
            reports[comparator_name], horizon, bucket, subset
        )
        cell_frames[(horizon, bucket, subset)] = [
            (str(contract["fold_name"]), candidate, comparator)
        ]
        paired = candidate.merge(
            comparator,
            on="date",
            suffixes=("_candidate", "_comparator"),
            validate="one_to_one",
        )
        if len(paired) != len(candidate) or len(paired) != len(comparator):
            raise ValueError(
                f"comparison dates differ for {candidate_name}/{comparator_name}"
            )
        for metric in METRICS:
            for row in paired.itertuples(index=False):
                candidate_value = float(getattr(row, f"{metric}_candidate"))
                comparator_value = float(getattr(row, f"{metric}_comparator"))
                rows.append(
                    {
                        "comparison": str(comparison["name"]),
                        "horizon": horizon,
                        "bucket": bucket,
                        "subset": subset,
                        "metric": metric,
                        "date": row.date,
                        "candidate": candidate_value,
                        "comparator": comparator_value,
                        "delta": candidate_value - comparator_value,
                    }
                )

    bootstrap = contract["bootstrap"]
    primary = {
        metric: _aggregate_cells(
            cell_frames,
            primary_cells,
            metric,
            bootstrap=bootstrap,
            seed_offset=index * 10_000 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    fast = {
        metric: _aggregate_cells(
            cell_frames,
            fast_cells,
            metric,
            bootstrap=bootstrap,
            seed_offset=index * 10_000 + 5_000 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    candidate_loss = float(summaries[candidate_name]["validation_loss"])
    comparator_loss = float(summaries[comparator_name]["validation_loss"])
    relative_loss = candidate_loss / comparator_loss - 1.0
    gates = contract["gates"]
    primary_pearson = primary["pearson"]
    primary_skill = primary["skill_vs_zero_mse"]
    fast_skill = fast["skill_vs_zero_mse"]
    checks = {
        "minimum_primary_days": all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in primary_pearson["per_fold"].values()
        ),
        "validation_loss": relative_loss
        <= float(gates["maximum_relative_validation_loss_degradation"]),
        "primary_pearson_mean": float(primary_pearson["mean_delta"])
        > float(gates["minimum_primary_pearson_delta"]),
        "primary_pearson_positive_strata": int(
            primary_pearson["positive_fold_count"]
        )
        >= int(gates["minimum_positive_primary_strata"]),
        "primary_pearson_lower95_safety": float(
            primary_pearson["stratified_block_bootstrap"]["lower_95"]
        )
        >= -float(gates["maximum_primary_pearson_bootstrap_degradation"]),
        "primary_skill_mean": float(primary_skill["mean_delta"])
        >= float(gates["minimum_primary_skill_delta"]),
        "primary_skill_lower95_safety": float(
            primary_skill["stratified_block_bootstrap"]["lower_95"]
        )
        >= -float(gates["maximum_primary_skill_bootstrap_degradation"]),
        "minimum_fast_exit_days": all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in fast_skill["per_fold"].values()
        ),
        "fast_exit_skill_mean_safety": float(fast_skill["mean_delta"])
        >= -float(gates["maximum_fast_exit_mean_skill_degradation"]),
        "fast_exit_skill_lower95_safety": float(
            fast_skill["stratified_block_bootstrap"]["lower_95"]
        )
        >= -float(gates["maximum_fast_exit_bootstrap_degradation"]),
    }
    return {
        "candidate": candidate_name,
        "comparator": comparator_name,
        "role": str(comparison["role"]),
        "candidate_validation_loss": candidate_loss,
        "comparator_validation_loss": comparator_loss,
        "relative_validation_loss_delta": relative_loss,
        "primary": primary,
        "fast_exit_safety": fast,
        "checks": checks,
        "passed": all(checks.values()),
    }, rows


def select_candidate(
    comparison_passes: Mapping[str, bool],
    selection: Mapping[str, Any],
) -> tuple[str | None, str]:
    causal_required = [str(value) for value in selection["causal_required"]]
    own_required = [str(value) for value in selection["own_surprise_required"]]
    if all(comparison_passes.get(name) is True for name in causal_required):
        return "surprise_causal", "expand_causal_node_surprise_to_multifold_validation"
    if all(comparison_passes.get(name) is True for name in own_required):
        return "surprise_disabled", "expand_own_node_surprise_to_multifold_validation"
    return None, "reject_node_surprise_message_screen"


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = safe_json(contract_path, "node-surprise screen contract")
    if contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid node-surprise screen contract role")
    if contract.get("live_orders_allowed") is not False or contract.get(
        "promotion_eligible"
    ) is not False:
        raise ValueError("unsafe node-surprise screen contract")
    if contract.get("test_split_evaluation_allowed") is not False:
        raise ValueError("node-surprise screen permits test evaluation")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    models = contract["models"]
    if set(models) != EXPECTED_MODELS:
        raise ValueError("node-surprise screen model set mismatch")
    reports: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {"contract": sha256_file(contract_path)}
    timestamp_counts: set[str] = set()
    timestamp_fingerprints: set[str] = set()
    for name, spec in models.items():
        report, summary, hashes = _validate_model_artifacts(
            contract, str(name), spec
        )
        reports[str(name)] = report
        summaries[str(name)] = summary
        inputs.update({f"{name}.{key}": value for key, value in hashes.items()})
        validation = report["validation"]
        timestamp_counts.add(
            json.dumps(
                validation["clock_bucket_causal_shock_timestamp_counts"],
                sort_keys=True,
            )
        )
        timestamp_fingerprints.add(
            json.dumps(
                validation["clock_bucket_causal_shock_timestamp_sha256"],
                sort_keys=True,
            )
        )
    if len(timestamp_counts) != 1 or len(timestamp_fingerprints) != 1:
        raise ValueError("model arms do not share validation shock timestamps")

    comparison_results: dict[str, Any] = {}
    daily_rows: list[dict[str, Any]] = []
    for index, comparison in enumerate(contract["comparisons"], start=1):
        name = str(comparison["name"])
        if name in comparison_results:
            raise ValueError(f"duplicate comparison name: {name}")
        result, rows = _comparison_result(
            contract, reports, summaries, comparison, index
        )
        comparison_results[name] = result
        daily_rows.extend(rows)
    passes = {
        name: bool(result["passed"])
        for name, result in comparison_results.items()
    }
    selected, decision = select_candidate(passes, contract["selection"])
    summary = {
        "schema_version": 1,
        "role": AUDIT_ROLE,
        "status": "complete",
        "evaluation_scope": "validation_only",
        "test_evaluated": False,
        "test": None,
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "retrospective_validation_period_previously_inspected": True,
        "fold": str(contract["fold_name"]),
        "comparison_results": comparison_results,
        "comparison_passes": passes,
        "selected_candidate": selected,
        "decision": decision,
        "next_gate": (
            "frozen_three_fold_validation_only_comparison"
            if selected is not None
            else "do_not_expand_node_surprise_message_path"
        ),
        "inputs": inputs,
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a validation-only node-surprise graph-message screen."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily_path = output_dir / "daily_paired_deltas.csv"
    daily.to_csv(daily_path, index=False)
    summary["daily_paired_deltas_sha256"] = sha256_file(daily_path)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "selected_candidate": summary["selected_candidate"],
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
