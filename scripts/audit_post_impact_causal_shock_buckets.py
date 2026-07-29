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

from scripts.audit_post_impact_clock_bucket_increment import (
    safe_json,
    sha256_file,
)
from scripts.audit_post_impact_multifold_increment import (
    _assert_context_map_audits,
    _validate_nonoverlapping_tests,
    paired_multifold_result,
)


CONTRACT_ROLE = "post_impact_causal_shock_bucket_screen_contract"
AUDIT_ROLE = "post_impact_causal_shock_bucket_screen_audit"
SHOCK_CONTRACT = "causal_observed_surprise_clock_subsets_v1"
TIMESTAMP_FINGERPRINT = (
    "per_test_day_uint64_count_then_little_endian_int64_utc_ns_sha256_v1"
)
METRICS = ("pearson", "skill_vs_zero_mse")


def causal_shock_daily_frame(
    payload: Mapping[str, Any],
    horizon: str,
    bucket: str,
    subset: str,
) -> pd.DataFrame:
    rows = payload["test"][
        "clock_bucket_causal_shock_daily_node_endpoint_rows"
    ][horizon][bucket][subset]
    frame = pd.DataFrame(rows)
    required = {"date", "count", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"causal-shock daily rows are missing fields: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("causal-shock daily rows are empty or duplicated")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("causal-shock daily metrics contain non-finite values")
    return frame[["date", "count", *METRICS]]


def _validate_report_contract(
    payload: Mapping[str, Any],
    expected_subsets: list[str],
    expected_lookback: int,
    label: str,
) -> None:
    record = payload.get("test", {}).get("clock_bucket_causal_shock_contract")
    if not isinstance(record, Mapping) or record.get("name") != SHOCK_CONTRACT:
        raise ValueError(f"causal-shock report contract mismatch: {label}")
    if record.get("point_in_time_observed_only") is not True:
        raise ValueError(f"causal-shock subset is not point-in-time: {label}")
    if record.get("future_realized_labels_used_for_selection") is not False:
        raise ValueError(f"future labels selected a causal-shock subset: {label}")
    if int(record.get("recent_lookback_minutes", -1)) != int(expected_lookback):
        raise ValueError(f"causal-shock lookback mismatch: {label}")
    if list(record.get("subsets", [])) != expected_subsets:
        raise ValueError(f"causal-shock subset list mismatch: {label}")
    if record.get("timestamp_fingerprint") != TIMESTAMP_FINGERPRINT:
        raise ValueError(f"causal-shock timestamp fingerprint mismatch: {label}")


def _load_inputs(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    reports: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, Any] = {}
    split_records: list[dict[str, Any]] = []
    expected_subsets = [str(value) for value in contract["subsets"]]
    expected_lookback = int(contract["recent_lookback_minutes"])
    data_pins = contract["data_pins"]
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        if fold in reports:
            raise ValueError(f"duplicate fold name: {fold}")
        split = fold_spec["split"]
        fold_reports: dict[str, dict[str, Any]] = {}
        shared_splits = set()
        shared_timestamp_counts = set()
        shared_timestamp_fingerprints = set()
        for name, spec in fold_spec["models"].items():
            report_path = Path(spec["report"])
            summary_path = Path(spec["training_dir"]) / "summary.json"
            checkpoint_path = Path(spec["training_dir"]) / "post_impact_reforecast.pt"
            report = safe_json(report_path, f"{fold}/{name} causal-shock report")
            summary = safe_json(summary_path, f"{fold}/{name} training summary")
            if report.get("live_orders_allowed") is not False:
                raise ValueError(f"unsafe causal-shock report: {fold}/{name}")
            if summary.get("promotion_eligible") is not False:
                raise ValueError(f"unsafe training summary: {fold}/{name}")
            if report.get("daily_context_placebo_mode") != spec["mode"]:
                raise ValueError(f"report context mode mismatch: {fold}/{name}")
            if summary.get("daily_context_placebo_mode") != spec["mode"]:
                raise ValueError(f"summary context mode mismatch: {fold}/{name}")
            if report.get("variant") != spec["variant"] or summary.get(
                "variant"
            ) != spec["variant"]:
                raise ValueError(f"model variant mismatch: {fold}/{name}")
            _assert_context_map_audits(report, str(spec["mode"]), f"{fold}/{name}")
            if report.get("context_map_audit") != summary.get("context_map_audit"):
                raise ValueError(f"observable context audit mismatch: {fold}/{name}")
            if report.get("latent_context_map_audit") != summary.get(
                "latent_context_map_audit"
            ):
                raise ValueError(f"latent context audit mismatch: {fold}/{name}")
            _validate_report_contract(
                report,
                expected_subsets,
                expected_lookback,
                f"{fold}/{name}",
            )
            parity = report.get("reference_inference_parity")
            if not isinstance(parity, Mapping) or parity.get("passed") is not True:
                raise ValueError(f"reference inference parity failed: {fold}/{name}")
            checkpoint_sha256 = sha256_file(checkpoint_path)
            summary_sha256 = sha256_file(summary_path)
            if checkpoint_sha256 != spec["checkpoint_sha256"]:
                raise ValueError(f"contract checkpoint pin mismatch: {fold}/{name}")
            if summary_sha256 != spec["summary_sha256"]:
                raise ValueError(f"contract summary pin mismatch: {fold}/{name}")
            if report["inputs"].get("checkpoint_sha256") != checkpoint_sha256:
                raise ValueError(f"checkpoint hash mismatch: {fold}/{name}")
            if report["inputs"].get("reference_summary_sha256") != summary_sha256:
                raise ValueError(f"summary hash mismatch: {fold}/{name}")
            if report["inputs"].get("day_release_manifest_sha256") != data_pins[
                "day_release_manifest_sha256"
            ]:
                raise ValueError(f"day release mismatch: {fold}/{name}")
            if report["inputs"].get("stale_cache_manifest_sha256") != data_pins[
                "stale_graph_cache_manifest_sha256"
            ]:
                raise ValueError(f"stale cache mismatch: {fold}/{name}")
            if report.get("splits") != summary.get("splits"):
                raise ValueError(f"split mismatch: {fold}/{name}")
            shared_splits.add(json.dumps(report["splits"], sort_keys=True))
            counts = report["test"].get(
                "clock_bucket_causal_shock_timestamp_counts"
            )
            fingerprints = report["test"].get(
                "clock_bucket_causal_shock_timestamp_sha256"
            )
            if not isinstance(counts, Mapping) or not isinstance(
                fingerprints, Mapping
            ):
                raise ValueError(f"missing causal-shock timestamp proof: {fold}/{name}")
            shared_timestamp_counts.add(json.dumps(counts, sort_keys=True))
            shared_timestamp_fingerprints.add(
                json.dumps(fingerprints, sort_keys=True)
            )
            fold_reports[str(name)] = report
            inputs[f"{fold}.{name}.report"] = sha256_file(report_path)
            inputs[f"{fold}.{name}.summary"] = sha256_file(summary_path)
            inputs[f"{fold}.{name}.checkpoint"] = sha256_file(checkpoint_path)
        if (
            len(shared_splits) != 1
            or len(shared_timestamp_counts) != 1
            or len(shared_timestamp_fingerprints) != 1
        ):
            raise ValueError(f"models do not share split and shock timestamps: {fold}")
        realized_splits = next(iter(fold_reports.values()))["splits"]
        if (
            realized_splits["train"]["end"] != split["train_end"]
            or realized_splits["validation"]["end"] != split["validation_end"]
            or realized_splits["test"]["end"] != split["test_end"]
        ):
            raise ValueError(f"realized split does not match contract: {fold}")
        split_records.append(realized_splits)
        reports[fold] = fold_reports
    _validate_nonoverlapping_tests(split_records)
    return reports, inputs, split_records


def _contract_cells(
    contract: Mapping[str, Any],
    name: str,
) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    allowed_horizons = set(map(str, contract["horizons"]))
    allowed_buckets = set(map(str, contract["buckets"]))
    allowed_subsets = set(map(str, contract["subsets"]))
    for record in contract[name]:
        cell = (
            str(record["horizon"]),
            str(record["bucket"]),
            str(record["subset"]),
        )
        if (
            cell[0] not in allowed_horizons
            or cell[1] not in allowed_buckets
            or cell[2] not in allowed_subsets
        ):
            raise ValueError(f"{name} contains an undeclared diagnostic cell: {cell}")
        if cell in cells:
            raise ValueError(f"{name} contains a duplicate cell: {cell}")
        cells.append(cell)
    if not cells:
        raise ValueError(f"{name} must contain at least one cell")
    return cells


def _aggregate_cells(
    cell_frames: Mapping[
        tuple[str, str, str],
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ],
    cells: list[tuple[str, str, str]],
    metric: str,
    *,
    bootstrap: Mapping[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    strata = [
        (
            f"{fold}|{horizon}|{bucket}|{subset}",
            actual,
            comparator,
        )
        for horizon, bucket, subset in cells
        for fold, actual, comparator in cell_frames[(horizon, bucket, subset)]
    ]
    result, _paired = paired_multifold_result(
        strata,
        metric,
        bootstrap=bootstrap,
        seed_offset=seed_offset,
    )
    by_fold: dict[str, list[float]] = {}
    for stratum, record in result["per_fold"].items():
        fold = str(stratum).split("|", maxsplit=1)[0]
        by_fold.setdefault(fold, []).append(float(record["mean_delta"]))
    fold_mean_delta = {
        fold: float(np.mean(values)) for fold, values in by_fold.items()
    }
    result["positive_original_fold_count"] = int(
        sum(value > 0.0 for value in fold_mean_delta.values())
    )
    result["original_fold_mean_delta"] = fold_mean_delta
    return result


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid causal-shock screen contract role")
    if contract.get("live_orders_allowed") is not False or contract.get(
        "promotion_eligible"
    ) is not False:
        raise ValueError("unsafe causal-shock screen contract")
    if contract.get("retrospective_period_previously_inspected") is not True:
        raise ValueError("causal-shock screen must disclose test reuse")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    reports, inputs, _split_records = _load_inputs(contract)
    inputs["contract"] = sha256_file(contract_path)
    fold_names = list(reports)
    actual_name = str(contract["actual_model"])
    comparator = str(contract["comparator"])
    if any(set(models) != {actual_name, comparator} for models in reports.values()):
        raise ValueError("causal-shock screen model set mismatch")

    bootstrap = contract["bootstrap"]
    gates = contract["gates"]
    primary_cells = _contract_cells(contract, "primary_cells")
    fast_exit_cells = _contract_cells(contract, "fast_exit_safety_cells")
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    cell_frames: dict[
        tuple[str, str, str],
        list[tuple[str, pd.DataFrame, pd.DataFrame]],
    ] = {}
    for horizon_index, horizon in enumerate(contract["horizons"]):
        results[horizon] = {}
        for bucket_index, bucket in enumerate(contract["buckets"]):
            results[horizon][bucket] = {}
            for subset_index, subset in enumerate(contract["subsets"]):
                fold_frames = [
                    (
                        fold,
                        causal_shock_daily_frame(
                            reports[fold][actual_name], horizon, bucket, subset
                        ),
                        causal_shock_daily_frame(
                            reports[fold][comparator], horizon, bucket, subset
                        ),
                    )
                    for fold in fold_names
                ]
                cell_frames[(str(horizon), str(bucket), str(subset))] = fold_frames
                cell: dict[str, Any] = {}
                for metric_index, metric in enumerate(METRICS):
                    result, paired_folds = paired_multifold_result(
                        fold_frames,
                        metric,
                        bootstrap=bootstrap,
                        seed_offset=(
                            horizon_index * 100
                            + bucket_index * 20
                            + subset_index * 4
                            + metric_index
                        ),
                    )
                    cell[metric] = result
                    for fold, paired in paired_folds:
                        for row in paired.itertuples(index=False):
                            rows.append(
                                {
                                    "fold": fold,
                                    "horizon": horizon,
                                    "bucket": bucket,
                                    "subset": subset,
                                    "metric": metric,
                                    "date": row.date,
                                    "actual": float(row.actual),
                                    "comparator": float(row.comparator),
                                    "delta": float(row.delta),
                                }
                            )
                results[horizon][bucket][subset] = cell

    primary = {
        metric: _aggregate_cells(
            cell_frames,
            primary_cells,
            metric,
            bootstrap=bootstrap,
            seed_offset=10_000 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    fast_exit_safety = {
        metric: _aggregate_cells(
            cell_frames,
            fast_exit_cells,
            metric,
            bootstrap=bootstrap,
            seed_offset=20_000 + index,
        )
        for index, metric in enumerate(METRICS)
    }
    primary_pearson = primary["pearson"]
    primary_skill = primary["skill_vs_zero_mse"]
    fast_exit_skill = fast_exit_safety["skill_vs_zero_mse"]
    checks["primary.minimum_days_per_stratum"] = all(
        int(record["rows"]) >= int(gates["minimum_days_per_fold"])
        for record in primary_pearson["per_fold"].values()
    )
    checks["primary.pearson_mean"] = float(
        primary_pearson["mean_delta"]
    ) > float(gates["minimum_pearson_delta"])
    checks["primary.pearson_lower95"] = float(
        primary_pearson["stratified_block_bootstrap"]["lower_95"]
    ) > float(gates["minimum_pearson_bootstrap_lower_95"])
    checks["primary.pearson_positive_strata"] = int(
        primary_pearson["positive_fold_count"]
    ) >= int(gates["minimum_positive_strata"])
    checks["primary.pearson_positive_original_folds"] = int(
        primary_pearson["positive_original_fold_count"]
    ) >= int(gates["minimum_positive_folds"])
    checks["primary.skill_mean"] = float(primary_skill["mean_delta"]) >= float(
        gates["minimum_skill_delta"]
    )
    checks["primary.skill_lower95"] = float(
        primary_skill["stratified_block_bootstrap"]["lower_95"]
    ) >= -float(gates["maximum_skill_bootstrap_degradation"])
    checks["fast_exit.minimum_days_per_stratum"] = all(
        int(record["rows"]) >= int(gates["minimum_days_per_fold"])
        for record in fast_exit_skill["per_fold"].values()
    )
    checks["fast_exit.skill_mean_safety"] = float(
        fast_exit_skill["mean_delta"]
    ) >= -float(gates["maximum_fast_exit_mean_skill_degradation"])
    checks["fast_exit.skill_lower95_safety"] = float(
        fast_exit_skill["stratified_block_bootstrap"]["lower_95"]
    ) >= -float(gates["maximum_fast_exit_bootstrap_degradation"])

    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "role": AUDIT_ROLE,
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "retrospective_period_previously_inspected": True,
        "actual_model": actual_name,
        "comparator": comparator,
        "folds": fold_names,
        "contract_sha256": sha256_file(contract_path),
        "results": results,
        "primary_cells": [
            {"horizon": value[0], "bucket": value[1], "subset": value[2]}
            for value in primary_cells
        ],
        "primary_aggregate": primary,
        "fast_exit_safety_cells": [
            {"horizon": value[0], "bucket": value[1], "subset": value[2]}
            for value in fast_exit_cells
        ],
        "fast_exit_safety_aggregate": fast_exit_safety,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "decision": (
            "causal_shock_scope_confirmed_for_new_prospective_design"
            if passed
            else "causal_shock_scope_not_confirmed"
        ),
        "next_gate": (
            "pre_register_new_forward_only_conditional_shadow_gate"
            if passed
            else "do_not_create_conditional_shadow_scope"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    return summary, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit post-impact forecasts after causally observed shocks."
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
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "checks_passed": summary["checks_passed"],
                "checks_total": summary["checks_total"],
                "promotion_eligible": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
