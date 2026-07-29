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
    _aggregate_cells,
    _contract_cells,
)
from scripts.audit_post_impact_clock_bucket_increment import safe_json, sha256_file
from scripts.audit_post_impact_node_surprise_screen import (
    _validate_shock_contract,
    validation_shock_daily_frame,
)


CONTRACT_ROLE = "post_impact_rank_adapter_multifold_contract"
AUDIT_ROLE = "post_impact_rank_adapter_multifold_audit"
EXPECTED_MODELS = {"baseline", "aligned", "own_permuted"}


def _objective_components(summary: Mapping[str, Any]) -> Mapping[str, Any]:
    components = summary.get("validation", {}).get("objective_components")
    if not isinstance(components, Mapping):
        raise ValueError("training summary is missing validation objective components")
    return components


def _validate_model(
    contract: Mapping[str, Any],
    artifact_root: Path,
    fold_name: str,
    split: Mapping[str, str],
    name: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    training_dir = artifact_root / str(spec["training_dir"])
    report_path = artifact_root / str(spec["report"])
    summary_path = training_dir / "summary.json"
    checkpoint_path = training_dir / "post_impact_reforecast.pt"
    report = safe_json(report_path, f"{fold_name}/{name} validation report")
    summary = safe_json(summary_path, f"{fold_name}/{name} training summary")

    for label, payload in (("report", report), ("summary", summary)):
        if payload.get("live_orders_allowed") is not False:
            raise ValueError(f"unsafe {fold_name}/{name} {label}")
        if payload.get("promotion_eligible") is not False:
            raise ValueError(f"promotion-enabled {fold_name}/{name} {label}")
        if payload.get("evaluation_scope") != "validation_only":
            raise ValueError(f"{fold_name}/{name} {label} is not validation-only")
        if payload.get("test_evaluated") is not False:
            raise ValueError(f"{fold_name}/{name} {label} evaluated test labels")
        if payload.get("test") is not None:
            raise ValueError(f"{fold_name}/{name} {label} contains test metrics")
    if summary.get("test_loss") is not None:
        raise ValueError(f"{fold_name}/{name} summary contains test loss")
    if report.get("variant") != "latent" or summary.get("variant") != "latent":
        raise ValueError(f"{fold_name}/{name} variant mismatch")

    expected_mode = str(spec["graph_message_mode"])
    expected_fusion = str(spec["graph_message_fusion"])
    expected_frozen = bool(spec["frozen_message_adapter"])
    expected_weight = float(spec["post_shock_correlation_weight"])
    if report.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{fold_name}/{name} report message mode mismatch")
    if summary.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{fold_name}/{name} summary message mode mismatch")
    if str(report.get("graph_message_fusion", "shared")) != expected_fusion:
        raise ValueError(f"{fold_name}/{name} report fusion mismatch")
    if str(summary.get("graph_message_fusion", "shared")) != expected_fusion:
        raise ValueError(f"{fold_name}/{name} summary fusion mismatch")
    if bool(summary.get("freeze_base_for_message_adapter", False)) is not expected_frozen:
        raise ValueError(f"{fold_name}/{name} frozen flag mismatch")
    if bool(report.get("freeze_base_for_message_adapter", False)) is not expected_frozen:
        raise ValueError(f"{fold_name}/{name} report frozen flag mismatch")
    if summary.get("graph_message_edges_used") is not False:
        raise ValueError(f"{fold_name}/{name} unexpectedly used message edges")
    if summary.get("stale_stock_graph_used") is not False:
        raise ValueError(f"{fold_name}/{name} unexpectedly used stale graph inputs")

    summary_rank = summary.get("post_shock_correlation_contract")
    report_rank = report.get("post_shock_correlation_contract")
    if not isinstance(summary_rank, Mapping) or not isinstance(report_rank, Mapping):
        raise ValueError(f"{fold_name}/{name} is missing correlation metadata")
    if float(summary["objective_weights"]["post_shock_correlation"]) != expected_weight:
        raise ValueError(f"{fold_name}/{name} objective weight mismatch")
    if bool(summary_rank.get("enabled", False)) is not bool(expected_weight > 0.0):
        raise ValueError(f"{fold_name}/{name} objective enabled flag mismatch")
    if float(report_rank.get("weight", -1.0)) != expected_weight:
        raise ValueError(f"{fold_name}/{name} report objective weight mismatch")
    expected_horizons = list(contract["objective"]["horizons"])
    for record in (summary_rank, report_rank):
        if list(record.get("horizons", [])) != expected_horizons:
            raise ValueError(f"{fold_name}/{name} correlation horizons mismatch")
        if int(record.get("lookback_minutes", -1)) != int(
            contract["objective"]["lookback_minutes"]
        ):
            raise ValueError(f"{fold_name}/{name} shock lookback mismatch")
        if int(record.get("minimum_nodes", -1)) != int(
            contract["objective"]["minimum_nodes"]
        ):
            raise ValueError(f"{fold_name}/{name} minimum nodes mismatch")
        if record.get("point_in_time_observed_shock_only") is not True:
            raise ValueError(f"{fold_name}/{name} event selection is not causal")
        if record.get("future_labels_used_for_event_selection") is not False:
            raise ValueError(f"{fold_name}/{name} event selection used future labels")

    components = _objective_components(summary)
    if float(components["post_shock_correlation_weight"]) != expected_weight:
        raise ValueError(f"{fold_name}/{name} validation objective mismatch")
    reconstructed = float(components["multitask"]) + expected_weight * float(
        components["post_shock_correlation"]
    )
    if abs(float(components["total"]) - reconstructed) > 1e-7:
        raise ValueError(f"{fold_name}/{name} objective components do not sum")

    if report.get("splits") != summary.get("splits"):
        raise ValueError(f"{fold_name}/{name} report and summary splits differ")
    realized = report["splits"]
    if (
        realized["train"]["end"] != split["train_end"]
        or realized["validation"]["end"] != split["validation_end"]
        or realized["test"]["end"] != split["test_end"]
    ):
        raise ValueError(f"{fold_name}/{name} split mismatch")
    expected_context_days = int(realized["train"]["days"]) + int(
        realized["validation"]["days"]
    )
    for audit_name in ("context_map_audit", "latent_context_map_audit"):
        if report.get(audit_name) != summary.get(audit_name):
            raise ValueError(f"{fold_name}/{name} {audit_name} reload mismatch")
        audit = report.get(audit_name)
        if not isinstance(audit, Mapping):
            raise ValueError(f"{fold_name}/{name} is missing {audit_name}")
        if int(audit.get("dates", -1)) != expected_context_days:
            raise ValueError(f"{fold_name}/{name} {audit_name} includes test dates")
        if int(audit.get("future_context_violations", -1)) != 0:
            raise ValueError(f"{fold_name}/{name} {audit_name} is non-causal")

    parity = report.get("reference_inference_parity")
    if not isinstance(parity, Mapping) or parity.get("passed") is not True:
        raise ValueError(f"{fold_name}/{name} inference parity failed")
    _validate_shock_contract(
        report,
        [str(value) for value in contract["subsets"]],
        int(contract["recent_lookback_minutes"]),
        f"{fold_name}/{name}",
    )
    checkpoint_sha256 = sha256_file(checkpoint_path)
    summary_sha256 = sha256_file(summary_path)
    if summary.get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{fold_name}/{name} summary checkpoint hash mismatch")
    if report["inputs"].get("checkpoint_sha256") != checkpoint_sha256:
        raise ValueError(f"{fold_name}/{name} report checkpoint hash mismatch")
    if report["inputs"].get("reference_summary_sha256") != summary_sha256:
        raise ValueError(f"{fold_name}/{name} report summary hash mismatch")
    data_pins = contract["data_pins"]
    if report["inputs"].get("day_release_manifest_sha256") != data_pins[
        "day_release_manifest_sha256"
    ]:
        raise ValueError(f"{fold_name}/{name} day-release hash mismatch")
    if report["inputs"].get("stale_cache_manifest_sha256") != data_pins[
        "stale_graph_cache_manifest_sha256"
    ]:
        raise ValueError(f"{fold_name}/{name} stale-cache hash mismatch")
    return report, summary, {
        "checkpoint": checkpoint_sha256,
        "summary": summary_sha256,
        "report": sha256_file(report_path),
    }


def _validate_adapter_lineage(
    fold_name: str,
    summaries: Mapping[str, Mapping[str, Any]],
    hashes: Mapping[str, Mapping[str, str]],
) -> None:
    baseline_components = _objective_components(summaries["baseline"])
    baseline_multitask = float(baseline_components["multitask"])
    if float(summaries["baseline"]["validation_loss"]) != baseline_multitask:
        raise ValueError(f"{fold_name} baseline objective is not pure multitask loss")
    for name in ("aligned", "own_permuted"):
        summary = summaries[name]
        lineage = summary.get("frozen_message_base")
        if not isinstance(lineage, Mapping):
            raise ValueError(f"{fold_name}/{name} is missing base lineage")
        if lineage.get("base_checkpoint_sha256") != hashes["baseline"]["checkpoint"]:
            raise ValueError(f"{fold_name}/{name} checkpoint lineage mismatch")
        if lineage.get("base_summary_sha256") != hashes["baseline"]["summary"]:
            raise ValueError(f"{fold_name}/{name} summary lineage mismatch")
        if lineage.get("protected_horizons") != ["5m"]:
            raise ValueError(f"{fold_name}/{name} does not protect 5m")
        initial = summary.get("initial_validation_objective_components")
        if not isinstance(initial, Mapping):
            raise ValueError(f"{fold_name}/{name} is missing initial objective")
        if float(initial["multitask"]) != baseline_multitask:
            raise ValueError(f"{fold_name}/{name} did not start from exact baseline")
        runtime = summary.get("runtime", {})
        if int(runtime.get("trainable_parameters", 0)) <= 0:
            raise ValueError(f"{fold_name}/{name} has no trainable adapter")
        if int(runtime.get("frozen_parameters", 0)) <= 0:
            raise ValueError(f"{fold_name}/{name} has no frozen base")


def _standard_multitask_loss(summary: Mapping[str, Any]) -> float:
    return float(_objective_components(summary)["multitask"])


def _comparison(
    contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Mapping[str, Any]]],
    summaries: Mapping[str, Mapping[str, Mapping[str, Any]]],
    spec: Mapping[str, Any],
    index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidate_name = str(spec["candidate"])
    comparator_name = str(spec["comparator"])
    primary_cells = _contract_cells(contract, "primary_cells")
    protected_cells = _contract_cells(contract, "protected_cells")
    cells = list(dict.fromkeys(primary_cells + protected_cells))
    frames: dict[
        tuple[str, str, str],
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ] = {cell: [] for cell in cells}
    rows: list[dict[str, Any]] = []
    protected_max_absolute_delta = 0.0
    protected_counts_equal = True
    relative_losses: dict[str, float] = {}
    for fold_name in reports:
        candidate_summary = summaries[fold_name][candidate_name]
        comparator_summary = summaries[fold_name][comparator_name]
        candidate_loss = _standard_multitask_loss(candidate_summary)
        comparator_loss = _standard_multitask_loss(comparator_summary)
        relative_losses[fold_name] = candidate_loss / comparator_loss - 1.0
        for horizon, bucket, subset in cells:
            candidate = validation_shock_daily_frame(
                reports[fold_name][candidate_name], horizon, bucket, subset
            )
            comparator = validation_shock_daily_frame(
                reports[fold_name][comparator_name], horizon, bucket, subset
            )
            frames[(horizon, bucket, subset)].append(
                (fold_name, candidate, comparator)
            )
            paired = candidate.merge(
                comparator,
                on="date",
                suffixes=("_candidate", "_comparator"),
                validate="one_to_one",
            )
            if len(paired) != len(candidate) or len(paired) != len(comparator):
                raise ValueError(
                    f"comparison dates differ for {fold_name}: "
                    f"{candidate_name}/{comparator_name}"
                )
            protected = (horizon, bucket, subset) in protected_cells
            if protected:
                protected_counts_equal &= bool(
                    np.array_equal(
                        paired["count_candidate"].to_numpy(),
                        paired["count_comparator"].to_numpy(),
                    )
                )
            for metric in METRICS:
                delta = (
                    paired[f"{metric}_candidate"].to_numpy(dtype=np.float64)
                    - paired[f"{metric}_comparator"].to_numpy(dtype=np.float64)
                )
                if protected:
                    protected_max_absolute_delta = max(
                        protected_max_absolute_delta,
                        float(np.max(np.abs(delta), initial=0.0)),
                    )
                for date, candidate_value, comparator_value, difference in zip(
                    paired["date"],
                    paired[f"{metric}_candidate"],
                    paired[f"{metric}_comparator"],
                    delta,
                ):
                    rows.append(
                        {
                            "comparison": str(spec["name"]),
                            "fold": fold_name,
                            "horizon": horizon,
                            "bucket": bucket,
                            "subset": subset,
                            "metric": metric,
                            "date": date,
                            "candidate": float(candidate_value),
                            "comparator": float(comparator_value),
                            "delta": float(difference),
                        }
                    )

    bootstrap = contract["bootstrap"]
    primary = {
        metric: _aggregate_cells(
            frames,
            primary_cells,
            metric,
            bootstrap=bootstrap,
            seed_offset=index * 10_000 + metric_index,
        )
        for metric_index, metric in enumerate(METRICS)
    }
    gates = contract["gates"]
    pearson = primary["pearson"]
    skill = primary["skill_vs_zero_mse"]
    checks = {
        "minimum_primary_days": all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in pearson["per_fold"].values()
        ),
        "multitask_loss_safety": max(relative_losses.values())
        <= float(gates["maximum_relative_multitask_loss_degradation"]),
        "primary_pearson_mean": float(pearson["mean_delta"])
        > float(gates["minimum_primary_pearson_delta"]),
        "primary_pearson_positive_strata": int(pearson["positive_fold_count"])
        >= int(gates["minimum_positive_primary_strata"]),
        "primary_pearson_positive_folds": int(
            pearson["positive_original_fold_count"]
        )
        >= int(gates["minimum_positive_original_folds"]),
        "primary_pearson_lower95_safety": float(
            pearson["stratified_block_bootstrap"]["lower_95"]
        )
        >= -float(gates["maximum_primary_pearson_bootstrap_degradation"]),
        "primary_skill_mean": float(skill["mean_delta"])
        >= float(gates["minimum_primary_skill_delta"]),
        "primary_skill_lower95_safety": float(
            skill["stratified_block_bootstrap"]["lower_95"]
        )
        >= -float(gates["maximum_primary_skill_bootstrap_degradation"]),
        "protected_5m_counts_exact": protected_counts_equal,
        "protected_5m_metrics_exact": protected_max_absolute_delta == 0.0,
    }
    return {
        "candidate": candidate_name,
        "comparator": comparator_name,
        "role": str(spec["role"]),
        "relative_multitask_loss_delta_by_fold": relative_losses,
        "primary": primary,
        "protected_5m": {
            "counts_exact": protected_counts_equal,
            "maximum_absolute_metric_delta": protected_max_absolute_delta,
        },
        "checks": checks,
        "passed": all(checks.values()),
    }, rows


def evaluate(
    contract_path: Path, artifact_root: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = safe_json(contract_path, "rank-adapter multifold contract")
    if contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid rank-adapter multifold contract role")
    if contract.get("live_orders_allowed") is not False or contract.get(
        "promotion_eligible"
    ) is not False:
        raise ValueError("unsafe rank-adapter multifold contract")
    if contract.get("test_split_evaluation_allowed") is not False:
        raise ValueError("rank-adapter multifold contract permits test evaluation")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    reports: dict[str, dict[str, dict[str, Any]]] = {}
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, str] = {"contract": sha256_file(contract_path)}
    validation_ranges: list[tuple[str, str, str]] = []
    for fold in contract["folds"]:
        fold_name = str(fold["name"])
        if fold_name in reports:
            raise ValueError(f"duplicate fold name: {fold_name}")
        split = fold["split"]
        validation_ranges.append(
            (
                fold_name,
                str(split["train_end"]),
                str(split["validation_end"]),
            )
        )
        models = fold["models"]
        if set(models) != EXPECTED_MODELS:
            raise ValueError(f"{fold_name} model set mismatch")
        fold_reports: dict[str, dict[str, Any]] = {}
        fold_summaries: dict[str, dict[str, Any]] = {}
        fold_hashes: dict[str, dict[str, str]] = {}
        timestamp_counts: set[str] = set()
        timestamp_fingerprints: set[str] = set()
        for name, spec in models.items():
            report, summary, hashes = _validate_model(
                contract,
                artifact_root,
                fold_name,
                split,
                str(name),
                spec,
            )
            fold_reports[str(name)] = report
            fold_summaries[str(name)] = summary
            fold_hashes[str(name)] = hashes
            inputs.update(
                {
                    f"{fold_name}.{name}.{key}": value
                    for key, value in hashes.items()
                }
            )
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
            raise ValueError(f"{fold_name} model arms use different shock timestamps")
        _validate_adapter_lineage(fold_name, fold_summaries, fold_hashes)
        reports[fold_name] = fold_reports
        summaries[fold_name] = fold_summaries
    ordered_ranges = sorted(validation_ranges, key=lambda value: value[1])
    for previous, current in zip(ordered_ranges, ordered_ranges[1:]):
        if previous[2] >= current[1]:
            raise ValueError("multifold validation periods overlap")

    comparison_results: dict[str, Any] = {}
    daily_rows: list[dict[str, Any]] = []
    for index, comparison in enumerate(contract["comparisons"], start=1):
        name = str(comparison["name"])
        result, rows = _comparison(
            contract, reports, summaries, comparison, index
        )
        comparison_results[name] = result
        daily_rows.extend(rows)
    passes = {
        name: bool(result["passed"])
        for name, result in comparison_results.items()
    }
    required = [str(value) for value in contract["selection"]["required"]]
    selected = "aligned" if all(passes.get(name) is True for name in required) else None
    decision = (
        "admit_rank_adapter_to_prospective_read_only_shadow_gate"
        if selected is not None
        else "reject_rank_adapter_multifold"
    )
    summary = {
        "schema_version": 1,
        "role": AUDIT_ROLE,
        "status": "complete",
        "evaluation_scope": "validation_only",
        "test_evaluated": False,
        "test": None,
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "retrospective_periods_previously_inspected": True,
        "folds": list(reports),
        "comparison_results": comparison_results,
        "comparison_passes": passes,
        "selected_candidate": selected,
        "decision": decision,
        "next_gate": (
            "prospective_read_only_shadow_with_no_order_route"
            if selected is not None
            else "do_not_deploy_rank_adapter"
        ),
        "inputs": inputs,
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit validation-only post-shock rank adapters across folds."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract), Path(args.artifact_root))
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
