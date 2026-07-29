from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_post_impact_adaptive_events import (
    _annotate_context_map_contracts,
    _calibration,
    _scaler,
    build_adaptive_event_calendar,
    causal_recent_event_mask,
    clock_bucket_masks,
    update_causal_shock_timestamp_fingerprint,
)
from scripts.train_post_impact_reforecast import (
    DayRelease,
    StaleCache,
    _amp_dtype,
    _batches,
    _daily_context_maps,
    _device,
    _graph_message_feature_dim,
    _graph_message_feature_names,
    _pad_batch,
    _resolved_daily_context_placebo_mode,
    _split_dates,
    _strict_json_value,
    _tensor_batch,
    file_sha256,
)
from stock_v2.post_impact_reforecast import (
    CausalPostImpactReforecast,
    RegressionMetricAccumulator,
)
from stock_v2.post_impact_residual_selection import select_residual_candidate
from stock_v2.post_impact_residual_state import (
    CausalPostImpactResidualState,
    PostImpactResidualConfig,
)
from stock_v2.surprise_reforecast import SURPRISE_STATISTIC_NAMES


VALIDATION_ROLE = "post_impact_residual_state_validation_contract"
TEST_ROLE = "post_impact_residual_state_test_contract"
EVALUATION_ROLE = "post_impact_residual_state_evaluation"
SUBSET_NAMES = (
    "adaptive_observed_surprise_current",
    "adaptive_observed_surprise_recent_30m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate causal intraday residual-state endpoint corrections."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--phase", choices=["validation_selection", "test"], required=True
    )
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--stale-cache-dir", required=True)
    parser.add_argument("--batch-days", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="bfloat16",
    )
    parser.add_argument("--cache-day-shards", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _fold_spec(contract: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    matches = [record for record in contract["folds"] if record["name"] == name]
    if len(matches) != 1:
        raise ValueError(f"residual-state fold is missing or duplicated: {name}")
    return matches[0]


def _validate_contract(
    contract: Mapping[str, Any],
    phase: str,
    fold: str,
    model_name: str,
    output: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    expected_role = VALIDATION_ROLE if phase == "validation_selection" else TEST_ROLE
    if contract.get("role") != expected_role:
        raise ValueError("residual-state contract role does not match the phase")
    if contract.get("promotion_eligible") is not False or contract.get(
        "live_orders_allowed"
    ) is not False:
        raise ValueError("unsafe residual-state evaluation contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or file_sha256(path) != str(expected):
            raise ValueError(f"residual-state source pin mismatch: {relative}")
    spec = _fold_spec(contract, fold)
    models = spec["models"]
    if model_name not in models:
        raise ValueError(f"residual-state model is not declared: {model_name}")
    if phase == "validation_selection" and model_name != contract["selection_model"]:
        raise ValueError("validation selection must use the declared actual model")
    model = models[model_name]
    expected_output = (
        spec["validation_report"]
        if phase == "validation_selection"
        else spec["test_reports"][model_name]
    )
    if output.resolve() != Path(expected_output).resolve():
        raise ValueError("residual-state output path does not match the contract")
    training_dir = Path(model["training_dir"])
    checkpoint = training_dir / "post_impact_reforecast.pt"
    summary = training_dir / "summary.json"
    if file_sha256(checkpoint) != model["checkpoint_sha256"]:
        raise ValueError("residual-state checkpoint pin mismatch")
    if file_sha256(summary) != model["summary_sha256"]:
        raise ValueError("residual-state training summary pin mismatch")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    if summary_payload.get("promotion_eligible") is not False or summary_payload.get(
        "live_orders_allowed"
    ) is not False:
        raise ValueError("unsafe residual-state source training summary")
    if phase == "validation_selection":
        candidates = [str(record["name"]) for record in contract["candidates"]]
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError("residual-state candidate names must be unique")
    if phase == "test":
        selected = contract.get("selected_config")
        if not isinstance(selected, Mapping):
            raise ValueError("test contract is missing a selected residual config")
        validation_path = Path(spec["validation_report"])
        if file_sha256(validation_path) != spec["validation_report_sha256"]:
            raise ValueError("test contract validation-report pin mismatch")
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        if (
            validation.get("role") != EVALUATION_ROLE
            or validation.get("phase") != "validation_selection"
            or validation.get("promotion_eligible") is not False
            or validation.get("live_orders_allowed") is not False
        ):
            raise ValueError("unsafe or invalid residual-state validation report")
        if validation.get("contract_sha256") != contract[
            "validation_contract_sha256"
        ]:
            raise ValueError("test contract validation-contract pin mismatch")
        realized = validation.get("selection", {}).get("selected_candidate")
        if realized != selected.get("name"):
            raise ValueError("test config differs from validation selection")
        candidate_configs = {
            str(record["name"]): record
            for record in validation.get("candidate_configs", [])
        }
        if candidate_configs.get(str(realized)) != selected:
            raise ValueError("test hyperparameters differ from validation candidate")
    return spec, model


def _residual_config(record: Mapping[str, Any], mode: str) -> PostImpactResidualConfig:
    return PostImpactResidualConfig(
        alpha=float(record["alpha"]),
        ridge=float(record["ridge"]),
        min_updates=int(record["min_updates"]),
        gain_clip=float(record["gain_clip"]),
        bias_clip=float(record["bias_clip"]),
        correction_clip=float(record["correction_clip"]),
        mode=mode,
        permutation_seed=int(record["permutation_seed"]),
    )


def _metric_containers(
    variants: Sequence[str],
    horizons: Sequence[str],
    buckets: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    aggregate = {
        variant: {
            horizon: {
                bucket: {
                    subset: RegressionMetricAccumulator()
                    for subset in SUBSET_NAMES
                }
                for bucket in buckets
            }
            for horizon in horizons
        }
        for variant in variants
    }
    daily = {
        variant: {
            horizon: {
                bucket: {subset: [] for subset in SUBSET_NAMES}
                for bucket in buckets
            }
            for horizon in horizons
        }
        for variant in variants
    }
    return aggregate, daily


def _metrics_payload(aggregate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        variant: {
            horizon: {
                bucket: {
                    subset: accumulator.metrics()
                    for subset, accumulator in subsets.items()
                }
                for bucket, subsets in buckets.items()
            }
            for horizon, buckets in horizons.items()
        }
        for variant, horizons in aggregate.items()
    }


def _reference_parity(
    realized: Mapping[str, RegressionMetricAccumulator],
    reference: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    tolerances = {
        "mae": 1e-5,
        "mse": 1e-7,
        "pearson": 2e-3,
        "skill_vs_zero_mse": 2e-3,
        "direction_accuracy": 5e-3,
    }
    maximum = {name: 0.0 for name in tolerances}
    count_mismatches = 0
    for horizon, accumulator in realized.items():
        current = accumulator.metrics()
        expected = reference[split]["node_targets"]["endpoint_return"][horizon][
            "all"
        ]
        count_mismatches += int(int(current["count"]) != int(expected["count"]))
        for metric in tolerances:
            difference = abs(float(current[metric]) - float(expected[metric]))
            maximum[metric] = max(maximum[metric], difference)
    passed = count_mismatches == 0 and all(
        maximum[name] <= tolerance for name, tolerance in tolerances.items()
    )
    return {
        "passed": passed,
        "count_mismatches": int(count_mismatches),
        "maximum_absolute_difference": maximum,
        "absolute_tolerances": tolerances,
    }


def _infer_batch(
    model: CausalPostImpactReforecast,
    numpy_batch: Mapping[str, np.ndarray],
    checkpoint_args: argparse.Namespace,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> np.ndarray:
    batch = _tensor_batch(dict(numpy_batch), device)
    with torch.no_grad(), torch.autocast(
        device_type=device.type,
        dtype=amp_dtype or torch.float32,
        enabled=amp_dtype is not None and device.type in {"cuda", "mps"},
    ):
        prediction = model(
            batch["node_values"],
            batch["node_available"],
            stale_state=(
                batch["stale_state"]
                if checkpoint_args.variant != "direct"
                else None
            ),
            context_latent=(
                batch["context_latent"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            predicted_delta=(
                batch["predicted_delta"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            surprise_values=batch["surprise"],
            graph_neighbor_values=(
                batch["graph_neighbor_values"].float()
                if "graph_neighbor_values" in batch
                else None
            ),
            graph_neighbor_available=batch.get("graph_neighbor_available"),
        )
    return prediction.node.float().cpu().numpy()


def _load_model_state_for_inference(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
) -> None:
    model.load_state_dict(state_dict)
    model.eval()


def _process_dates(
    model: CausalPostImpactReforecast,
    release: DayRelease,
    stale: StaleCache,
    dates: Sequence[str],
    state_context_map: Mapping[str, str],
    latent_context_map: Mapping[str, str],
    calendar: Mapping[str, Any],
    observed_calibration: Any,
    model_calibration: Any,
    node_scaler: Any,
    stale_scaler: Any,
    impact_thresholds: Mapping[str, np.ndarray],
    checkpoint_args: argparse.Namespace,
    states: Mapping[str, CausalPostImpactResidualState],
    fixed_labels: Sequence[str],
    fixed_minutes: Sequence[int],
    buckets: Sequence[str],
    *,
    record_after_sessions: int,
    record_metrics: bool,
    batch_days: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    variants = ("base", *states.keys())
    aggregate, daily = _metric_containers(variants, fixed_labels, buckets)
    full_baseline = {
        label: RegressionMetricAccumulator() for label in fixed_labels
    }
    timestamp_counts = {
        bucket: {subset: 0 for subset in SUBSET_NAMES} for bucket in buckets
    }
    timestamp_hashers = {
        bucket: {subset: hashlib.sha256() for subset in SUBSET_NAMES}
        for bucket in buckets
    }
    endpoint_index = release.target_names.index("endpoint_return")
    horizon_indices = [release.horizon_labels.index(label) for label in fixed_labels]
    recorded_dates = 0
    for date_batch in _batches(list(dates), int(batch_days), None):
        numpy_batch = _pad_batch(
            release,
            stale,
            date_batch,
            state_context_map,
            latent_context_map,
            observed_calibration,
            model_calibration,
            node_scaler,
            stale_scaler,
            impact_thresholds,
            checkpoint_args,
        )
        node_prediction = _infer_batch(
            model, numpy_batch, checkpoint_args, device, amp_dtype
        )
        for batch_index, date in enumerate(date_batch):
            day = release.load(date)
            timestamps = np.asarray(day["timestamps_utc_ns"], dtype=np.int64)
            count = len(timestamps)
            base = node_prediction[
                batch_index, :count, :, :, endpoint_index
            ][:, :, horizon_indices]
            targets = np.asarray(day["targets"], dtype=np.float32)[
                :, :, horizon_indices, endpoint_index
            ]
            available = np.asarray(day["target_available"], dtype=bool)[
                :, :, horizon_indices, endpoint_index
            ]
            prices = np.asarray(day["decision_price"], dtype=np.float32)
            variant_values: dict[str, np.ndarray] = {"base": base}
            for name, state in states.items():
                state.start_session(date)
                corrected = np.empty_like(base)
                for time_index, timestamp in enumerate(timestamps):
                    corrected[time_index] = state.step(
                        int(timestamp), prices[time_index], base[time_index]
                    )
                state.finish_session()
                variant_values[name] = corrected

            should_record = record_metrics and (
                recorded_dates >= int(record_after_sessions)
            )
            if record_metrics:
                recorded_dates += 1
            if record_metrics:
                for horizon_position, label in enumerate(fixed_labels):
                    full_baseline[label].update(
                        base[:, :, horizon_position],
                        targets[:, :, horizon_position],
                        available[:, :, horizon_position],
                    )
            if not should_record:
                continue
            clock_masks = clock_bucket_masks(timestamps)
            events = calendar[date]
            shock_masks = {
                "adaptive_observed_surprise_current": events.observed,
                "adaptive_observed_surprise_recent_30m": causal_recent_event_mask(
                    timestamps, events.observed
                ),
            }
            for bucket in buckets:
                time_mask = clock_masks[bucket]
                for subset, shock_mask in shock_masks.items():
                    selected_timestamps = timestamps[time_mask & shock_mask]
                    timestamp_counts[bucket][subset] += int(len(selected_timestamps))
                    update_causal_shock_timestamp_fingerprint(
                        timestamp_hashers[bucket][subset], selected_timestamps
                    )
            for horizon_position, label in enumerate(fixed_labels):
                for bucket in buckets:
                    clock_mask = clock_masks[bucket]
                    for subset, shock_mask in shock_masks.items():
                        selected = available[:, :, horizon_position] & (
                            clock_mask & shock_mask
                        )[:, None]
                        for variant, values in variant_values.items():
                            prediction = values[:, :, horizon_position]
                            aggregate[variant][label][bucket][subset].update(
                                prediction,
                                targets[:, :, horizon_position],
                                selected,
                            )
                            day_metric = RegressionMetricAccumulator()
                            day_metric.update(
                                prediction,
                                targets[:, :, horizon_position],
                                selected,
                            )
                            metrics = day_metric.metrics()
                            if int(metrics["count"]) > 0:
                                daily[variant][label][bucket][subset].append(
                                    {"date": date, **metrics}
                                )
    return {
        "metrics": _metrics_payload(aggregate),
        "daily_rows": daily,
        "full_baseline": full_baseline,
        "timestamp_counts": timestamp_counts,
        "timestamp_sha256": {
            bucket: {
                subset: digest.hexdigest()
                for subset, digest in subsets.items()
            }
            for bucket, subsets in timestamp_hashers.items()
        },
        "state_diagnostics": {
            name: state.diagnostics() for name, state in states.items()
        },
    }


def main() -> int:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    output_path = Path(args.output)
    if output_path.exists():
        raise ValueError(f"refusing to overwrite residual-state output: {output_path}")
    fold_spec, model_spec = _validate_contract(
        contract, args.phase, args.fold, args.model_name, output_path
    )
    device = _device(args.device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    training_dir = Path(model_spec["training_dir"])
    checkpoint_path = training_dir / "post_impact_reforecast.pt"
    summary_path = training_dir / "summary.json"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    context_mode = _resolved_daily_context_placebo_mode(checkpoint_args)
    checkpoint_args.daily_context_placebo_mode = context_mode
    checkpoint_args.shuffle_daily_context = context_mode == "all"
    if context_mode != model_spec["mode"]:
        raise ValueError("residual-state checkpoint context mode mismatch")
    if checkpoint_args.variant != model_spec["variant"]:
        raise ValueError("residual-state checkpoint variant mismatch")

    release = DayRelease(Path(args.day_release_dir), cache=bool(args.cache_day_shards))
    stale = StaleCache(Path(args.stale_cache_dir))
    stale.align_tickers(release.tickers)
    if tuple(checkpoint["feature_names"]) != release.feature_names:
        raise ValueError("residual-state checkpoint feature contract mismatch")
    expected_graph_features = _graph_message_feature_names(
        release.feature_names,
        str(getattr(checkpoint_args, "graph_message_mode", "none")),
    )
    if tuple(
        checkpoint.get("graph_message_feature_names", expected_graph_features)
    ) != expected_graph_features:
        raise ValueError("residual-state graph-message feature contract mismatch")
    if tuple(checkpoint["state_feature_names"]) != stale.state_feature_names:
        raise ValueError("residual-state checkpoint stale-state contract mismatch")
    if tuple(checkpoint["horizon_labels"]) != release.horizon_labels:
        raise ValueError("residual-state checkpoint horizon contract mismatch")
    if tuple(checkpoint["target_names"]) != release.target_names:
        raise ValueError("residual-state checkpoint target contract mismatch")
    if file_sha256(release.manifest_path) != contract["data_pins"][
        "day_release_manifest_sha256"
    ]:
        raise ValueError("residual-state day-release pin mismatch")
    if file_sha256(stale.manifest_path) != contract["data_pins"][
        "stale_graph_cache_manifest_sha256"
    ]:
        raise ValueError("residual-state stale-cache pin mismatch")
    split = fold_spec["split"]
    common_dates = sorted(set(release.dates) & set(stale.dates))
    train_dates, validation_dates, test_dates = _split_dates(
        common_dates,
        split["train_end"],
        split["validation_end"],
        split["test_end"],
    )
    state_context_map, latent_context_map = _daily_context_maps(
        (train_dates, validation_dates, test_dates),
        mode=context_mode,
        seed=int(checkpoint_args.seed),
    )
    selected_dates = train_dates + validation_dates + test_dates
    context_map_audit = stale.audit_context_map(selected_dates, state_context_map)
    latent_context_map_audit = stale.audit_context_map(
        selected_dates, latent_context_map
    )
    _annotate_context_map_contracts(
        context_map_audit, latent_context_map_audit, context_mode
    )
    node_scaler = _scaler(checkpoint["node_scaler"])
    target_scaler = _scaler(checkpoint["target_scaler"])
    stale_scaler = _scaler(checkpoint["stale_scaler"])
    observed_calibration = _calibration(checkpoint["observed_surprise_calibration"])
    model_calibration = _calibration(checkpoint["model_surprise_calibration"])
    impact_thresholds = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in checkpoint["impact_thresholds"].items()
    }
    calendar = build_adaptive_event_calendar(
        release,
        stale,
        selected_dates,
        observed_calibration,
        impact_thresholds,
        quantile=float(contract["event_selection"]["quantile"]),
        window_sessions=int(contract["event_selection"]["window_sessions"]),
        minimum_history=int(contract["event_selection"]["minimum_history"]),
    )
    model = CausalPostImpactReforecast(
        node_feature_dim=len(release.feature_names),
        stale_state_dim=len(stale.state_feature_names),
        latent_dim=int(stale.context.shape[-1]),
        horizons=release.horizon_labels,
        systemic_target_dim=len(release.systemic_target_names),
        variant=checkpoint_args.variant,
        hidden_dim=int(checkpoint_args.hidden_dim),
        latent_projection_dim=int(checkpoint_args.latent_projection_dim),
        temporal_layers=int(checkpoint_args.temporal_layers),
        dropout=float(checkpoint_args.dropout),
        surprise_dim=len(SURPRISE_STATISTIC_NAMES),
        graph_message_dim=_graph_message_feature_dim(
            release.feature_names,
            str(getattr(checkpoint_args, "graph_message_mode", "none")),
        ),
        graph_message_fusion=str(
            getattr(checkpoint_args, "graph_message_fusion", "shared")
        ),
        target_names=release.target_names,
        output_scales=target_scaler.scale,
    ).to(device)
    _load_model_state_for_inference(model, checkpoint["model_state"])

    fixed = contract["fixed_horizons"]
    fixed_labels = tuple(str(record["label"]) for record in fixed)
    fixed_minutes = tuple(int(record["minutes"]) for record in fixed)
    buckets = tuple(str(value) for value in contract["clock_buckets"])
    if set(contract["subsets"]) != set(SUBSET_NAMES):
        raise ValueError("residual-state subset contract mismatch")
    if args.phase == "validation_selection":
        states = {
            str(record["name"]): CausalPostImpactResidualState(
                len(release.tickers),
                fixed_minutes,
                _residual_config(record, "dynamic"),
            )
            for record in contract["candidates"]
        }
        result = _process_dates(
            model,
            release,
            stale,
            validation_dates,
            state_context_map,
            latent_context_map,
            calendar,
            observed_calibration,
            model_calibration,
            node_scaler,
            stale_scaler,
            impact_thresholds,
            checkpoint_args,
            states,
            fixed_labels,
            fixed_minutes,
            buckets,
            record_after_sessions=int(contract["warmup_sessions"]),
            record_metrics=True,
            batch_days=int(args.batch_days),
            device=device,
            amp_dtype=amp_dtype,
        )
        selection = select_residual_candidate(
            result["daily_rows"],
            list(states),
            baseline="base",
            primary_cells=contract["primary_cells"],
            fast_exit_cells=contract["fast_exit_safety_cells"],
            gates=contract["selection_gates"],
        )
        evaluated_split = "validation"
    else:
        selected = contract["selected_config"]
        states = {
            mode: CausalPostImpactResidualState(
                len(release.tickers),
                fixed_minutes,
                _residual_config(selected, mode),
            )
            for mode in ("dynamic", "bias_only", "node_permuted")
        }
        _process_dates(
            model,
            release,
            stale,
            validation_dates,
            state_context_map,
            latent_context_map,
            calendar,
            observed_calibration,
            model_calibration,
            node_scaler,
            stale_scaler,
            impact_thresholds,
            checkpoint_args,
            states,
            fixed_labels,
            fixed_minutes,
            buckets,
            record_after_sessions=0,
            record_metrics=False,
            batch_days=int(args.batch_days),
            device=device,
            amp_dtype=amp_dtype,
        )
        result = _process_dates(
            model,
            release,
            stale,
            test_dates,
            state_context_map,
            latent_context_map,
            calendar,
            observed_calibration,
            model_calibration,
            node_scaler,
            stale_scaler,
            impact_thresholds,
            checkpoint_args,
            states,
            fixed_labels,
            fixed_minutes,
            buckets,
            record_after_sessions=0,
            record_metrics=True,
            batch_days=int(args.batch_days),
            device=device,
            amp_dtype=amp_dtype,
        )
        selection = None
        evaluated_split = "test"

    reference = json.loads(summary_path.read_text(encoding="utf-8"))
    parity = _reference_parity(result["full_baseline"], reference, evaluated_split)
    if parity["passed"] is not True:
        raise ValueError("residual-state baseline failed checkpoint inference parity")
    output = {
        "schema_version": 1,
        "role": EVALUATION_ROLE,
        "phase": args.phase,
        "fold": args.fold,
        "model_name": args.model_name,
        "variant": checkpoint_args.variant,
        "daily_context_placebo_mode": context_mode,
        "contract_sha256": file_sha256(contract_path),
        "selection": selection,
        "candidate_configs": (
            list(contract["candidates"])
            if args.phase == "validation_selection"
            else [dict(contract["selected_config"])]
        ),
        "fixed_horizons": list(fixed),
        "clock_buckets": list(buckets),
        "subsets": list(SUBSET_NAMES),
        "context_map_audit": context_map_audit,
        "latent_context_map_audit": latent_context_map_audit,
        "reference_inference_parity": parity,
        "metrics": result["metrics"],
        "daily_rows": result["daily_rows"],
        "causal_shock_timestamp_counts": result["timestamp_counts"],
        "causal_shock_timestamp_sha256": result["timestamp_sha256"],
        "state_diagnostics": result["state_diagnostics"],
        "causal_contract": {
            "mature_before_forecast": True,
            "residual_target_derived_from_observed_decision_price": True,
            "future_target_values_stored_in_pending_state": False,
            "future_target_availability_used_for_enqueue": False,
            "dynamic_residual_reset_at_session_boundary": True,
            "adapter_coefficients_persist_across_sessions": True,
        },
        "splits": {
            "train": {"start": train_dates[0], "end": train_dates[-1], "days": len(train_dates)},
            "validation": {"start": validation_dates[0], "end": validation_dates[-1], "days": len(validation_dates)},
            "test": {"start": test_dates[0], "end": test_dates[-1], "days": len(test_dates)},
        },
        "inputs": {
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "reference_summary_sha256": file_sha256(summary_path),
            "day_release_manifest_sha256": file_sha256(release.manifest_path),
            "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            _strict_json_value(output),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "phase": args.phase,
                "fold": args.fold,
                "model_name": args.model_name,
                "selection_passed": (
                    selection["selection_passed"] if selection is not None else None
                ),
                "selected_candidate": (
                    selection["selected_candidate"] if selection is not None else None
                ),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
