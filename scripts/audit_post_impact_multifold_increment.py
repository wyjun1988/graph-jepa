from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_clock_bucket_increment import (
    circular_indices,
    daily_frame,
    paired_delta,
    safe_json,
    sha256_file,
)


HORIZONS = ("5m", "15m", "30m", "60m", "close")
METRICS = ("pearson", "skill_vs_zero_mse")
MULTIFOLD_CONTRACT_ROLE = "post_impact_multifold_increment_contract"
LATENT_ONLY_CONTRACT_ROLE = "post_impact_latent_only_placebo_contract"
IDENTITY_CONTEXT_CONTRACT = "identity_strict_oos_stale_h1_v1"
LATENT_PLACEBO_CONTEXT_PREFIX = "causal_historical_latent_placebo_last_"


def _expected_context_mode(spec: Mapping[str, Any]) -> str:
    explicit = spec.get("daily_context_placebo_mode")
    if explicit is not None:
        mode = str(explicit)
        if mode not in {"none", "all", "latent_only"}:
            raise ValueError(f"invalid daily context placebo mode: {mode}")
        return mode
    return "all" if bool(spec.get("shuffle_daily_context", False)) else "none"


def _payload_context_mode(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("daily_context_placebo_mode")
    if explicit is not None:
        return str(explicit)
    return "all" if bool(payload.get("shuffle_daily_context", False)) else "none"


def _assert_context_map_audits(
    payload: Mapping[str, Any],
    expected_mode: str,
    label: str,
) -> None:
    state_audit = payload.get("context_map_audit")
    latent_audit = payload.get("latent_context_map_audit")
    if not isinstance(state_audit, dict) or not isinstance(latent_audit, dict):
        raise ValueError(f"context-map audits are missing: {label}")
    if state_audit.get("contract") != IDENTITY_CONTEXT_CONTRACT:
        raise ValueError(f"observable state context is not identity-mapped: {label}")
    state_dates = int(state_audit.get("dates", 0))
    if state_dates <= 0 or int(state_audit.get("same_target_date_count", -1)) != state_dates:
        raise ValueError(f"observable state context changed target dates: {label}")
    if int(state_audit.get("future_context_violations", -1)) != 0:
        raise ValueError(f"observable state context is non-causal: {label}")
    if int(latent_audit.get("future_context_violations", -1)) != 0:
        raise ValueError(f"latent context is non-causal: {label}")
    if int(latent_audit.get("dates", 0)) != state_dates:
        raise ValueError(f"state/latent context dates differ: {label}")
    same_target = int(latent_audit.get("same_target_date_count", -1))
    if expected_mode == "none":
        if latent_audit.get("contract") != IDENTITY_CONTEXT_CONTRACT:
            raise ValueError(f"identity latent context contract mismatch: {label}")
        if same_target != state_dates:
            raise ValueError(f"identity latent context changed target dates: {label}")
    elif expected_mode == "latent_only":
        if not str(latent_audit.get("contract", "")).startswith(
            LATENT_PLACEBO_CONTEXT_PREFIX
        ):
            raise ValueError(f"latent-only placebo contract mismatch: {label}")
        if not 0 <= same_target < state_dates:
            raise ValueError(f"latent-only placebo did not replace context dates: {label}")
    else:
        raise ValueError(f"unsupported strict context audit mode: {expected_mode}")


def summary_daily(payload: Mapping[str, Any], horizon: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload["test"]["daily_node_endpoint_rows"][horizon])
    required = {"date", "count", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training daily rows are missing fields: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("training daily rows are empty or duplicated")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("training daily metrics contain non-finite values")
    return frame[["date", "count", *METRICS]]


def stratified_block_bootstrap_mean(
    fold_values: Sequence[np.ndarray],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    arrays = [np.asarray(values, dtype=np.float64) for values in fold_values]
    if not arrays or any(len(values) < 2 for values in arrays):
        raise ValueError("each bootstrap fold must contain at least two rows")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("bootstrap fold values must be finite")
    if int(samples) < 1 or int(block_length) < 1:
        raise ValueError("bootstrap samples and block length must be positive")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        resampled = [
            values[circular_indices(len(values), int(block_length), rng)]
            for values in arrays
        ]
        draws[draw] = float(np.concatenate(resampled).mean())
    return {
        "samples": int(samples),
        "block_length": int(block_length),
        "folds": int(len(arrays)),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def paired_multifold_result(
    fold_frames: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    metric: str,
    *,
    bootstrap: Mapping[str, Any],
    seed_offset: int,
) -> tuple[dict[str, Any], list[tuple[str, pd.DataFrame]]]:
    paired_folds: list[tuple[str, pd.DataFrame]] = []
    values: list[np.ndarray] = []
    fold_records: dict[str, Any] = {}
    for fold, actual, comparator in fold_frames:
        paired = paired_delta(actual, comparator, metric)
        fold_values = paired["delta"].to_numpy(dtype=np.float64)
        paired_folds.append((fold, paired))
        values.append(fold_values)
        fold_records[fold] = {
            "rows": int(len(fold_values)),
            "mean_delta": float(fold_values.mean()),
            "positive_day_fraction": float(np.mean(fold_values > 0.0)),
        }
    pooled = np.concatenate(values)
    result = {
        "rows": int(len(pooled)),
        "folds": int(len(values)),
        "mean_delta": float(pooled.mean()),
        "positive_day_fraction": float(np.mean(pooled > 0.0)),
        "positive_fold_count": int(
            sum(record["mean_delta"] > 0.0 for record in fold_records.values())
        ),
        "per_fold": fold_records,
        "stratified_block_bootstrap": stratified_block_bootstrap_mean(
            values,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["block_length"]),
            seed=int(bootstrap["seed"]) + int(seed_offset),
        ),
    }
    return result, paired_folds


def _assert_checkpoint_args(
    checkpoint_path: Path,
    expected_common: Mapping[str, Any],
    split: Mapping[str, str],
    spec: Mapping[str, Any],
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"checkpoint args are missing: {checkpoint_path}")
    expected = dict(expected_common)
    context_mode = _expected_context_mode(spec)
    expected.update(
        {
            "train_end": str(split["train_end"]),
            "validation_end": str(split["validation_end"]),
            "test_end": str(split["test_end"]),
            "variant": str(spec["variant"]),
            "shuffle_daily_context": context_mode == "all",
            "disable_stale_graph": True,
            "permute_stale_graph_nodes": False,
        }
    )
    if "daily_context_placebo_mode" in spec:
        expected["daily_context_placebo_mode"] = context_mode
    for name, expected_value in expected.items():
        actual = args.get(name)
        if isinstance(expected_value, float):
            if not np.isclose(float(actual), expected_value, rtol=0.0, atol=1e-12):
                raise ValueError(f"checkpoint arg mismatch: {name}")
        elif actual != expected_value:
            raise ValueError(f"checkpoint arg mismatch: {name}")


def _validate_nonoverlapping_tests(split_records: Sequence[Mapping[str, Any]]) -> None:
    intervals = sorted(
        (
            pd.Timestamp(record["test"]["start"]),
            pd.Timestamp(record["test"]["end"]),
        )
        for record in split_records
    )
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] <= previous[1]:
            raise ValueError("multifold test intervals overlap")


def _load_contract_models(
    contract: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    dict[str, dict[str, dict[str, Any]]],
    dict[str, Any],
]:
    training: dict[str, dict[str, dict[str, Any]]] = {}
    clock: dict[str, dict[str, dict[str, Any]]] = {}
    inputs: dict[str, Any] = {}
    split_records: list[dict[str, Any]] = []
    data_pins = contract["data_pins"]
    for fold_spec in contract["folds"]:
        fold = str(fold_spec["name"])
        if fold in training:
            raise ValueError(f"duplicate fold name: {fold}")
        split = fold_spec["split"]
        fold_training: dict[str, dict[str, Any]] = {}
        fold_clock: dict[str, dict[str, Any]] = {}
        shared_splits = set()
        for name, spec in fold_spec["models"].items():
            root = Path(spec["training_dir"])
            summary_path = root / "summary.json"
            checkpoint_path = root / "post_impact_reforecast.pt"
            clock_path = Path(spec["clock_report"])
            summary = safe_json(summary_path, f"{fold}/{name} training summary")
            clock_payload = safe_json(clock_path, f"{fold}/{name} clock report")
            if summary.get("promotion_eligible") is not False:
                raise ValueError(f"unsafe promotion field: {fold}/{name}")
            if summary.get("strict_out_of_sample_stale_jepa") is not True:
                raise ValueError(f"non-strict stale JEPA input: {fold}/{name}")
            if summary.get("stale_stock_graph_mode") != "disabled":
                raise ValueError(f"graph sensor was not disabled: {fold}/{name}")
            if summary.get("variant") != spec["variant"]:
                raise ValueError(f"training variant mismatch: {fold}/{name}")
            context_mode = _expected_context_mode(spec)
            if _payload_context_mode(summary) != context_mode:
                raise ValueError(f"training context mode mismatch: {fold}/{name}")
            if "daily_context_placebo_mode" in spec:
                _assert_context_map_audits(summary, context_mode, f"{fold}/{name}/training")
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
            parity = clock_payload.get("reference_inference_parity")
            if not isinstance(parity, dict) or parity.get("passed") is not True:
                raise ValueError(f"clock inference parity failed: {fold}/{name}")
            if (
                clock_payload.get("variant") != spec["variant"]
                or _payload_context_mode(clock_payload) != context_mode
            ):
                raise ValueError(f"clock model mode mismatch: {fold}/{name}")
            if "daily_context_placebo_mode" in spec:
                _assert_context_map_audits(
                    clock_payload, context_mode, f"{fold}/{name}/clock"
                )
                if clock_payload.get("context_map_audit") != summary.get(
                    "context_map_audit"
                ) or clock_payload.get("latent_context_map_audit") != summary.get(
                    "latent_context_map_audit"
                ):
                    raise ValueError(f"clock context audit mismatch: {fold}/{name}")
            if clock_payload["inputs"].get("checkpoint_sha256") != summary[
                "checkpoint_sha256"
            ]:
                raise ValueError(f"clock checkpoint mismatch: {fold}/{name}")
            if clock_payload["inputs"].get(
                "reference_summary_sha256"
            ) != sha256_file(summary_path):
                raise ValueError(f"clock reference summary mismatch: {fold}/{name}")
            if clock_payload.get("splits") != summary.get("splits"):
                raise ValueError(f"clock split mismatch: {fold}/{name}")
            shared_splits.add(json.dumps(summary["splits"], sort_keys=True))
            fold_training[str(name)] = summary
            fold_clock[str(name)] = clock_payload
            inputs[f"{fold}.{name}.summary"] = sha256_file(summary_path)
            inputs[f"{fold}.{name}.checkpoint"] = sha256_file(checkpoint_path)
            inputs[f"{fold}.{name}.clock"] = sha256_file(clock_path)
        if len(shared_splits) != 1:
            raise ValueError(f"models do not share one split in {fold}")
        splits = next(iter(fold_training.values()))["splits"]
        if (
            splits["train"]["end"] != split["train_end"]
            or splits["validation"]["end"] != split["validation_end"]
            or splits["test"]["end"] != split["test_end"]
        ):
            raise ValueError(f"realized split does not match contract: {fold}")
        split_records.append(splits)
        training[fold] = fold_training
        clock[fold] = fold_clock
    _validate_nonoverlapping_tests(split_records)
    return training, clock, inputs


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract_role = str(contract.get("role", ""))
    if contract_role not in {MULTIFOLD_CONTRACT_ROLE, LATENT_ONLY_CONTRACT_ROLE}:
        raise ValueError("invalid post-impact multifold contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe post-impact multifold contract")
    latent_only_contract = contract_role == LATENT_ONLY_CONTRACT_ROLE
    if latent_only_contract:
        if contract.get("promotion_eligible") is not False:
            raise ValueError("latent-only placebo contract must be research-only")
        if contract.get("retrospective_period_previously_inspected") is not True:
            raise ValueError("latent-only placebo contract must disclose test reuse")
        if contract.get("actual_model") != "latent" or contract.get("comparators") != [
            "latent_only_placebo"
        ]:
            raise ValueError("latent-only placebo model set mismatch")
        for fold in contract.get("folds", []):
            models = fold.get("models", {})
            if set(models) != {"latent", "latent_only_placebo"}:
                raise ValueError("latent-only placebo fold model set mismatch")
            if any(spec.get("variant") != "latent" for spec in models.values()):
                raise ValueError("latent-only placebo requires latent variants")
            if _expected_context_mode(models["latent"]) != "none" or _expected_context_mode(
                models["latent_only_placebo"]
            ) != "latent_only":
                raise ValueError("latent-only placebo fold modes mismatch")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    training, clock, inputs = _load_contract_models(contract)
    inputs["contract"] = sha256_file(contract_path)
    fold_names = list(training)
    actual_name = str(contract["actual_model"])
    comparator_names = [str(value) for value in contract["comparators"]]
    required_models = {actual_name, *comparator_names}
    for fold, models in training.items():
        if set(models) != required_models:
            raise ValueError(f"unexpected model set in {fold}")
        if latent_only_contract and models[actual_name]["context_map_audit"] != models[
            comparator_names[0]
        ]["context_map_audit"]:
            raise ValueError(f"observable state context differs within {fold}")

    bootstrap = contract["bootstrap"]
    daily_rows: list[dict[str, Any]] = []
    full_session: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        full_session[horizon] = {}
        for comparator_index, comparator in enumerate(comparator_names):
            fold_frames = [
                (
                    fold,
                    summary_daily(training[fold][actual_name], horizon),
                    summary_daily(training[fold][comparator], horizon),
                )
                for fold in fold_names
            ]
            full_session[horizon][comparator] = {}
            for metric_index, metric in enumerate(METRICS):
                result, paired_folds = paired_multifold_result(
                    fold_frames,
                    metric,
                    bootstrap=bootstrap,
                    seed_offset=horizon_index * 30 + comparator_index * 2 + metric_index,
                )
                full_session[horizon][comparator][metric] = result
                for fold, paired in paired_folds:
                    for row in paired.itertuples(index=False):
                        daily_rows.append(
                            {
                                "scope": "full_session",
                                "fold": fold,
                                "horizon": horizon,
                                "bucket": "all",
                                "comparator": comparator,
                                "metric": metric,
                                "date": row.date,
                                "actual": float(row.actual),
                                "comparator_value": float(row.comparator),
                                "delta": float(row.delta),
                            }
                        )

    early_close: dict[str, Any] = {}
    for bucket_index, bucket in enumerate(contract["gates"]["actionable_buckets"]):
        early_close[bucket] = {}
        for comparator_index, comparator in enumerate(comparator_names):
            fold_frames = [
                (
                    fold,
                    daily_frame(clock[fold][actual_name], "close", bucket),
                    daily_frame(clock[fold][comparator], "close", bucket),
                )
                for fold in fold_names
            ]
            early_close[bucket][comparator] = {}
            for metric_index, metric in enumerate(METRICS):
                result, paired_folds = paired_multifold_result(
                    fold_frames,
                    metric,
                    bootstrap=bootstrap,
                    seed_offset=500 + bucket_index * 30 + comparator_index * 2 + metric_index,
                )
                early_close[bucket][comparator][metric] = result
                for fold, paired in paired_folds:
                    for row in paired.itertuples(index=False):
                        daily_rows.append(
                            {
                                "scope": "early_close_endpoint",
                                "fold": fold,
                                "horizon": "close",
                                "bucket": bucket,
                                "comparator": comparator,
                                "metric": metric,
                                "date": row.date,
                                "actual": float(row.actual),
                                "comparator_value": float(row.comparator),
                                "delta": float(row.delta),
                            }
                        )

    systemic: dict[str, Any] = {}
    for comparator in comparator_names:
        per_fold = {}
        cells = []
        for fold in fold_names:
            per_horizon = {
                horizon: float(
                    training[fold][actual_name]["test"][
                        "systemic_state_change_energy"
                    ][horizon]["pearson"]
                )
                - float(
                    training[fold][comparator]["test"][
                        "systemic_state_change_energy"
                    ][horizon]["pearson"]
                )
                for horizon in HORIZONS
            }
            values = np.asarray(list(per_horizon.values()), dtype=np.float64)
            per_fold[fold] = {
                "horizons": per_horizon,
                "mean_delta": float(values.mean()),
                "positive_horizons": int((values > 0.0).sum()),
            }
            cells.extend(values.tolist())
        cell_values = np.asarray(cells, dtype=np.float64)
        systemic[comparator] = {
            "mean_delta": float(cell_values.mean()),
            "positive_cells": int((cell_values > 0.0).sum()),
            "positive_fold_count": int(
                sum(value["mean_delta"] > 0.0 for value in per_fold.values())
            ),
            "per_fold": per_fold,
        }

    test_loss = {
        fold: {
            name: float(payload["test_loss"])
            for name, payload in models.items()
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
        systemic_result = systemic[comparator]
        checks[f"systemic_mean_vs_{comparator}"] = float(
            systemic_result["mean_delta"]
        ) > float(gates["minimum_systemic_mean_pearson_delta"])
        checks[f"systemic_cells_vs_{comparator}"] = int(
            systemic_result["positive_cells"]
        ) >= int(gates["minimum_systemic_positive_cells"])
        checks[f"systemic_folds_vs_{comparator}"] = int(
            systemic_result["positive_fold_count"]
        ) >= int(gates["minimum_positive_folds"])

        close_metrics = full_session["close"][comparator]
        checks[f"full_close_pearson_mean_vs_{comparator}"] = float(
            close_metrics["pearson"]["mean_delta"]
        ) > float(gates["minimum_full_close_pearson_delta"])
        checks[f"full_close_pearson_lower95_vs_{comparator}"] = float(
            close_metrics["pearson"]["stratified_block_bootstrap"]["lower_95"]
        ) > float(gates["minimum_pearson_bootstrap_lower_95"])
        checks[f"full_close_pearson_folds_vs_{comparator}"] = int(
            close_metrics["pearson"]["positive_fold_count"]
        ) >= int(gates["minimum_positive_folds"])
        checks[f"full_close_skill_mean_vs_{comparator}"] = float(
            close_metrics["skill_vs_zero_mse"]["mean_delta"]
        ) >= float(gates["minimum_mean_skill_delta"])
        checks[f"full_close_skill_lower95_vs_{comparator}"] = float(
            close_metrics["skill_vs_zero_mse"]["stratified_block_bootstrap"][
                "lower_95"
            ]
        ) >= -float(gates["maximum_skill_bootstrap_degradation"])
        for horizon in gates["short_horizons"]:
            metrics = full_session[horizon][comparator]["skill_vs_zero_mse"]
            checks[f"{horizon}_skill_vs_{comparator}"] = float(
                metrics["mean_delta"]
            ) >= -float(gates["maximum_short_horizon_skill_degradation"])

        for bucket in gates["actionable_buckets"]:
            metrics = early_close[bucket][comparator]
            prefix = f"{bucket}_vs_{comparator}"
            checks[f"{prefix}_pearson_mean"] = float(
                metrics["pearson"]["mean_delta"]
            ) > float(gates["minimum_early_pearson_delta"])
            checks[f"{prefix}_pearson_lower95"] = float(
                metrics["pearson"]["stratified_block_bootstrap"]["lower_95"]
            ) > float(gates["minimum_pearson_bootstrap_lower_95"])
            checks[f"{prefix}_pearson_folds"] = int(
                metrics["pearson"]["positive_fold_count"]
            ) >= int(gates["minimum_positive_folds"])
            checks[f"{prefix}_skill_mean"] = float(
                metrics["skill_vs_zero_mse"]["mean_delta"]
            ) >= float(gates["minimum_mean_skill_delta"])
            checks[f"{prefix}_skill_lower95"] = float(
                metrics["skill_vs_zero_mse"]["stratified_block_bootstrap"][
                    "lower_95"
                ]
            ) >= -float(gates["maximum_skill_bootstrap_degradation"])

    passed = all(checks.values())
    retrospective_reuse = bool(
        contract.get("retrospective_period_previously_inspected", False)
    )
    summary = {
        "schema_version": 1,
        "role": (
            "post_impact_latent_only_placebo_audit"
            if latent_only_contract
            else "post_impact_multifold_increment_audit"
        ),
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "actual_model": actual_name,
        "comparators": comparator_names,
        "folds": fold_names,
        "contract_sha256": sha256_file(contract_path),
        "test_loss": test_loss,
        "systemic_state_change_energy": systemic,
        "full_session": full_session,
        "early_close_endpoint": early_close,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "retrospective_period_previously_inspected": retrospective_reuse,
        "decision": (
            "latent_only_placebo_increment_confirmed_research_only"
            if latent_only_contract and passed
            else "latent_only_placebo_increment_not_confirmed"
            if latent_only_contract
            else "multifold_jepa_increment_confirmed_research_only"
            if passed and retrospective_reuse
            else "multifold_jepa_increment_confirmed_for_shadow_candidate_development"
            if passed
            else "multifold_jepa_increment_not_confirmed"
        ),
        "next_gate": (
            "require_new_prospective_readonly_period_before_shadow_candidate_status"
            if latent_only_contract and passed
            else "do_not_promote_latent_only_placebo_candidate"
            if latent_only_contract
            else "require_new_prospective_readonly_period_before_shadow_candidate_status"
            if passed and retrospective_reuse
            else "build_max_only_readonly_latency_and_safety_shadow"
            if passed
            else "do_not_promote_current_jepa_latent_post_impact_head"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit post-impact JEPA increment across non-overlapping test folds."
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
