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


CONTRACT_ROLE = "post_impact_long_horizon_adapter_screen_contract"
AUDIT_ROLE = "post_impact_long_horizon_adapter_screen_audit"
EXPECTED_MODELS = {"baseline", "aligned", "own_permuted"}


def _validate_model(
    contract: Mapping[str, Any],
    artifact_root: Path,
    name: str,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    training_dir = artifact_root / str(spec["training_dir"])
    report_path = artifact_root / str(spec["report"])
    summary_path = training_dir / "summary.json"
    checkpoint_path = training_dir / "post_impact_reforecast.pt"
    report = safe_json(report_path, f"{name} validation report")
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
        raise ValueError(f"{name} summary contains test loss")
    if report.get("variant") != "latent" or summary.get("variant") != "latent":
        raise ValueError(f"{name} variant mismatch")

    expected_mode = str(spec["graph_message_mode"])
    expected_fusion = str(spec["graph_message_fusion"])
    if report.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{name} report graph-message mode mismatch")
    if summary.get("graph_message_mode") != expected_mode:
        raise ValueError(f"{name} summary graph-message mode mismatch")
    if str(report.get("graph_message_fusion", "shared")) != expected_fusion:
        raise ValueError(f"{name} report graph-message fusion mismatch")
    if str(summary.get("graph_message_fusion", "shared")) != expected_fusion:
        raise ValueError(f"{name} summary graph-message fusion mismatch")
    if summary.get("graph_message_edges_used") is not False:
        raise ValueError(f"{name} unexpectedly used graph edges")
    if summary.get("stale_stock_graph_used") is not False:
        raise ValueError(f"{name} unexpectedly used the stale graph encoder")

    frozen = bool(spec["frozen_message_adapter"])
    if bool(summary.get("freeze_base_for_message_adapter", False)) is not frozen:
        raise ValueError(f"{name} frozen-adapter flag mismatch")
    if bool(report.get("freeze_base_for_message_adapter", False)) is not frozen:
        raise ValueError(f"{name} report frozen-adapter flag mismatch")
    lineage = summary.get("frozen_message_base")
    if frozen:
        if not isinstance(lineage, Mapping):
            raise ValueError(f"{name} is missing frozen-base lineage")
        base_pins = contract["base_pins"]
        if lineage.get("base_checkpoint_sha256") != base_pins[
            "checkpoint_sha256"
        ]:
            raise ValueError(f"{name} frozen checkpoint lineage mismatch")
        if lineage.get("base_summary_sha256") != base_pins["summary_sha256"]:
            raise ValueError(f"{name} frozen summary lineage mismatch")
        if lineage.get("protected_horizons") != ["5m"]:
            raise ValueError(f"{name} does not protect the 5m horizon")
        runtime = summary.get("runtime", {})
        if int(runtime.get("trainable_parameters", 0)) <= 0:
            raise ValueError(f"{name} has no trainable adapter parameters")
        if int(runtime.get("frozen_parameters", 0)) <= 0:
            raise ValueError(f"{name} has no frozen base parameters")
    elif lineage is not None:
        raise ValueError(f"{name} baseline unexpectedly has adapter lineage")

    if report.get("splits") != summary.get("splits"):
        raise ValueError(f"{name} report and summary splits differ")
    split = contract["split"]
    realized = report["splits"]
    if (
        realized["train"]["end"] != split["train_end"]
        or realized["validation"]["end"] != split["validation_end"]
        or realized["test"]["end"] != split["test_end"]
    ):
        raise ValueError(f"{name} split mismatch")
    expected_context_days = int(realized["train"]["days"]) + int(
        realized["validation"]["days"]
    )
    for audit_name in ("context_map_audit", "latent_context_map_audit"):
        if report.get(audit_name) != summary.get(audit_name):
            raise ValueError(f"{name} {audit_name} reload mismatch")
        audit = report.get(audit_name)
        if not isinstance(audit, Mapping):
            raise ValueError(f"{name} is missing {audit_name}")
        if int(audit.get("dates", -1)) != expected_context_days:
            raise ValueError(f"{name} {audit_name} includes test dates")
        if int(audit.get("future_context_violations", -1)) != 0:
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


def _comparison(
    contract: Mapping[str, Any],
    reports: Mapping[str, Mapping[str, Any]],
    summaries: Mapping[str, Mapping[str, Any]],
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
    ] = {}
    rows: list[dict[str, Any]] = []
    protected_max_absolute_delta = 0.0
    protected_counts_equal = True
    for horizon, bucket, subset in cells:
        candidate = validation_shock_daily_frame(
            reports[candidate_name], horizon, bucket, subset
        )
        comparator = validation_shock_daily_frame(
            reports[comparator_name], horizon, bucket, subset
        )
        frames[(horizon, bucket, subset)] = [
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
        if (horizon, bucket, subset) in protected_cells:
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
            if (horizon, bucket, subset) in protected_cells:
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
    candidate_loss = float(summaries[candidate_name]["validation_loss"])
    comparator_loss = float(summaries[comparator_name]["validation_loss"])
    relative_loss = candidate_loss / comparator_loss - 1.0
    gates = contract["gates"]
    pearson = primary["pearson"]
    skill = primary["skill_vs_zero_mse"]
    checks = {
        "minimum_primary_days": all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in pearson["per_fold"].values()
        ),
        "validation_loss": relative_loss
        <= float(gates["maximum_relative_validation_loss_degradation"]),
        "primary_pearson_mean": float(pearson["mean_delta"])
        > float(gates["minimum_primary_pearson_delta"]),
        "primary_pearson_positive_strata": int(pearson["positive_fold_count"])
        >= int(gates["minimum_positive_primary_strata"]),
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
        "candidate_validation_loss": candidate_loss,
        "comparator_validation_loss": comparator_loss,
        "relative_validation_loss_delta": relative_loss,
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
    contract = safe_json(contract_path, "long-horizon adapter contract")
    if contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid long-horizon adapter contract role")
    if contract.get("live_orders_allowed") is not False or contract.get(
        "promotion_eligible"
    ) is not False:
        raise ValueError("unsafe long-horizon adapter contract")
    if contract.get("test_split_evaluation_allowed") is not False:
        raise ValueError("long-horizon adapter contract permits test evaluation")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    models = contract["models"]
    if set(models) != EXPECTED_MODELS:
        raise ValueError("long-horizon adapter model set mismatch")
    reports: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    inputs: dict[str, str] = {"contract": sha256_file(contract_path)}
    timestamp_counts: set[str] = set()
    timestamp_fingerprints: set[str] = set()
    for name, spec in models.items():
        report, summary, hashes = _validate_model(
            contract, artifact_root, str(name), spec
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

    baseline_loss = float(summaries["baseline"]["validation_loss"])
    for name in ("aligned", "own_permuted"):
        initial = summaries[name].get("initial_validation_loss")
        if initial is None or float(initial) != baseline_loss:
            raise ValueError(f"{name} did not start from the exact baseline loss")

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
        "expand_long_horizon_adapter_to_frozen_multifold_validation"
        if selected is not None
        else "reject_long_horizon_adapter_screen"
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
        "retrospective_validation_period_previously_inspected": True,
        "fold": str(contract["fold_name"]),
        "comparison_results": comparison_results,
        "comparison_passes": passes,
        "selected_candidate": selected,
        "decision": decision,
        "next_gate": (
            "separately_frozen_multifold_validation_only_comparison"
            if selected is not None
            else "do_not_expand_long_horizon_adapter"
        ),
        "inputs": inputs,
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a validation-only frozen-base long-horizon adapter screen."
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
