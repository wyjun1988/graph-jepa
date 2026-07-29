from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.audit_post_impact_clock_bucket_increment import safe_json, sha256_file
from scripts.audit_post_impact_multifold_increment import (
    HORIZONS,
    _assert_checkpoint_args,
    _validate_nonoverlapping_tests,
)


METRICS = ("pearson", "skill_vs_zero_mse")


def _target_metric(
    summary: Mapping[str, Any], target: str, horizon: str, metric: str
) -> float:
    value = float(summary["test"]["node_targets"][target][horizon]["all"][metric])
    if not np.isfinite(value):
        raise ValueError(f"non-finite target metric: {target}/{horizon}/{metric}")
    return value


def _load_models(
    contract: Mapping[str, Any],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, str]]:
    training: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, str] = {}
    split_records: list[dict[str, Any]] = []
    data_pins = contract["data_pins"]
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        if fold in training:
            raise ValueError(f"duplicate confirmation fold: {fold}")
        split = fold_spec["split"]
        models: dict[str, dict[str, Any]] = {}
        realized_splits = set()
        for name, spec in fold_spec["models"].items():
            root = Path(spec["training_dir"])
            summary_path = root / "summary.json"
            checkpoint_path = root / "post_impact_reforecast.pt"
            summary = safe_json(summary_path, f"{fold}/{name} training summary")
            if summary.get("promotion_eligible") is not False:
                raise ValueError(f"unsafe promotion field: {fold}/{name}")
            if summary.get("strict_out_of_sample_stale_jepa") is not True:
                raise ValueError(f"non-strict stale JEPA input: {fold}/{name}")
            if summary.get("stale_stock_graph_mode") != "disabled":
                raise ValueError(f"graph sensor was not disabled: {fold}/{name}")
            if summary.get("variant") != spec["variant"] or bool(
                summary.get("shuffle_daily_context")
            ) != bool(spec["shuffle_daily_context"]):
                raise ValueError(f"model mode mismatch: {fold}/{name}")
            if summary["inputs"].get("day_release_manifest_sha256") != data_pins[
                "day_release_manifest_sha256"
            ]:
                raise ValueError(f"day release mismatch: {fold}/{name}")
            if summary["inputs"].get("stale_cache_manifest_sha256") != data_pins[
                "stale_graph_cache_manifest_sha256"
            ]:
                raise ValueError(f"stale cache mismatch: {fold}/{name}")
            if sha256_file(checkpoint_path) != str(summary["checkpoint_sha256"]):
                raise ValueError(f"checkpoint checksum mismatch: {fold}/{name}")
            _assert_checkpoint_args(
                checkpoint_path,
                contract["training_args"],
                split,
                spec,
            )
            realized_splits.add(json.dumps(summary["splits"], sort_keys=True))
            models[str(name)] = summary
            inputs[f"{fold}.{name}.summary"] = sha256_file(summary_path)
            inputs[f"{fold}.{name}.checkpoint"] = sha256_file(checkpoint_path)
        if len(realized_splits) != 1:
            raise ValueError(f"models do not share one split in {fold}")
        splits = next(iter(models.values()))["splits"]
        if (
            splits["train"]["end"] != split["train_end"]
            or splits["validation"]["end"] != split["validation_end"]
            or splits["test"]["end"] != split["test_end"]
        ):
            raise ValueError(f"realized split does not match contract: {fold}")
        split_records.append(splits)
        training[fold] = models
    _validate_nonoverlapping_tests(split_records)
    return training, inputs


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "post_impact_magnitude_confirmation_contract":
        raise ValueError("invalid post-impact magnitude contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe post-impact magnitude contract")
    if contract.get("hypothesis_generation_fold") != "fold1":
        raise ValueError("magnitude hypothesis-generation lineage is missing")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    training, inputs = _load_models(contract)
    inputs["contract"] = sha256_file(contract_path)
    fold_names = list(training)
    if "fold1" in fold_names:
        raise ValueError("hypothesis-generation fold leaked into confirmation")
    actual_name = str(contract["actual_model"])
    comparator_names = [str(value) for value in contract["comparators"]]
    required_models = {actual_name, *comparator_names}
    for fold, models in training.items():
        if set(models) != required_models:
            raise ValueError(f"unexpected model set in {fold}")

    targets = [str(value) for value in contract["targets"]]
    if len(targets) != len(set(targets)) or not targets:
        raise ValueError("magnitude targets must be unique and nonempty")
    cells: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    for comparator in comparator_names:
        comparisons[comparator] = {}
        for target in targets:
            comparisons[comparator][target] = {}
            for metric in METRICS:
                values: list[float] = []
                per_fold: dict[str, Any] = {}
                for fold in fold_names:
                    horizon_deltas = {}
                    for horizon in HORIZONS:
                        actual = _target_metric(
                            training[fold][actual_name], target, horizon, metric
                        )
                        baseline = _target_metric(
                            training[fold][comparator], target, horizon, metric
                        )
                        delta = actual - baseline
                        horizon_deltas[horizon] = float(delta)
                        values.append(float(delta))
                        cells.append(
                            {
                                "fold": fold,
                                "target": target,
                                "horizon": horizon,
                                "comparator": comparator,
                                "metric": metric,
                                "actual": actual,
                                "comparator_value": baseline,
                                "delta": float(delta),
                            }
                        )
                    fold_values = np.asarray(
                        list(horizon_deltas.values()), dtype=np.float64
                    )
                    per_fold[fold] = {
                        "horizons": horizon_deltas,
                        "mean_delta": float(fold_values.mean()),
                        "positive_cells": int((fold_values > 0.0).sum()),
                    }
                array = np.asarray(values, dtype=np.float64)
                comparisons[comparator][target][metric] = {
                    "cells": int(len(array)),
                    "mean_delta": float(array.mean()),
                    "worst_delta": float(array.min()),
                    "positive_cells": int((array > 0.0).sum()),
                    "positive_fold_count": int(
                        sum(
                            record["mean_delta"] > 0.0
                            for record in per_fold.values()
                        )
                    ),
                    "per_fold": per_fold,
                }

    test_loss = {
        fold: {
            name: float(summary["test_loss"])
            for name, summary in models.items()
        }
        for fold, models in training.items()
    }
    gates = contract["gates"]
    checks: dict[str, bool] = {}
    for comparator in comparator_names:
        checks[f"test_loss_vs_{comparator}"] = all(
            losses[actual_name]
            <= losses[comparator]
            * (1.0 + float(gates["maximum_relative_test_loss_degradation"]))
            for losses in test_loss.values()
        )
        for target in targets:
            pearson = comparisons[comparator][target]["pearson"]
            skill = comparisons[comparator][target]["skill_vs_zero_mse"]
            prefix = f"{target}_vs_{comparator}"
            checks[f"{prefix}_pearson_mean"] = float(
                pearson["mean_delta"]
            ) > float(gates["minimum_mean_pearson_delta"])
            checks[f"{prefix}_pearson_cells"] = int(
                pearson["positive_cells"]
            ) >= int(gates["minimum_positive_pearson_cells"])
            checks[f"{prefix}_pearson_folds"] = int(
                pearson["positive_fold_count"]
            ) >= int(gates["minimum_positive_folds"])
            checks[f"{prefix}_skill_mean"] = float(
                skill["mean_delta"]
            ) >= float(gates["minimum_mean_skill_delta"])
            checks[f"{prefix}_skill_cells"] = int(
                skill["positive_cells"]
            ) >= int(gates["minimum_positive_skill_cells"])
            checks[f"{prefix}_skill_folds"] = int(
                skill["positive_fold_count"]
            ) >= int(gates["minimum_positive_folds"])
            checks[f"{prefix}_skill_worst"] = float(
                skill["worst_delta"]
            ) >= -float(gates["maximum_single_cell_skill_degradation"])

    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "role": "post_impact_magnitude_confirmation_audit",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "hypothesis_generation_fold": "fold1",
        "confirmation_folds": fold_names,
        "actual_model": actual_name,
        "comparators": comparator_names,
        "targets": targets,
        "contract_sha256": sha256_file(contract_path),
        "test_loss": test_loss,
        "comparisons": comparisons,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "decision": (
            "magnitude_path_increment_confirmed_for_daily_metric_audit"
            if passed
            else "magnitude_path_increment_not_confirmed"
        ),
        "next_gate": (
            "add_daily_target_rows_and_run_block_bootstrap_confirmation"
            if passed
            else "do_not_promote_current_latent_magnitude_heads"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    return summary, pd.DataFrame(cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Confirm latent post-impact magnitude heads on untouched folds."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, cells = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    cells_path = output_dir / "magnitude_cells.csv"
    cells.to_csv(cells_path, index=False)
    summary["magnitude_cells_sha256"] = sha256_file(cells_path)
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
