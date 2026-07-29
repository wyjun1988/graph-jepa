from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_post_impact_reforecast import (
    CONTEXT_PLACEBO_LOOKBACK_SESSIONS,
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
    _surprise_residuals,
    _tensor_batch,
    file_sha256,
)
from stock_v2.causal_quantile import causal_grouped_upper_tail
from stock_v2.post_impact_reforecast import (
    CausalPostImpactReforecast,
    RegressionMetricAccumulator,
    RobustArrayScaler,
)
from stock_v2.surprise_reforecast import (
    SURPRISE_STATISTIC_NAMES,
    ResidualSurpriseCalibration,
    summarize_residual_surprise,
)


ADAPTIVE_EVENT_CONTRACT = "causal_same_clock_rolling_upper_tail_v1"
CLOCK_BUCKET_CONTRACT = "krx_regular_session_fixed_clock_buckets_v1"
CLOCK_CAUSAL_SHOCK_CONTRACT = "causal_observed_surprise_clock_subsets_v1"
CAUSAL_SHOCK_TIMESTAMP_FINGERPRINT = (
    "per_test_day_uint64_count_then_little_endian_int64_utc_ns_sha256_v1"
)
RECENT_SHOCK_LOOKBACK_MINUTES = 30
CLOCK_CAUSAL_SHOCK_SUBSETS = (
    "adaptive_observed_surprise_current",
    "adaptive_observed_surprise_recent_30m",
)
CLOCK_BUCKETS = (
    ("open_0900_0929", 9 * 60, 9 * 60 + 30),
    ("morning_0930_1059", 9 * 60 + 30, 11 * 60),
    ("midday_1100_1329", 11 * 60, 13 * 60 + 30),
    ("afternoon_1330_1459", 13 * 60 + 30, 15 * 60),
    ("close_1500_1530", 15 * 60, 15 * 60 + 31),
)
SUBSETS = (
    "all",
    "adaptive_observed_surprise",
    "adaptive_realized_impact",
    "adaptive_surprise_and_impact",
)


def _annotate_context_map_contracts(
    state_audit: dict[str, Any],
    latent_audit: dict[str, Any],
    mode: str,
) -> None:
    identity = "identity_strict_oos_stale_h1_v1"
    state_audit["contract"] = (
        f"causal_historical_placebo_last_"
        f"{CONTEXT_PLACEBO_LOOKBACK_SESSIONS}_sessions_v1"
        if mode == "all"
        else identity
    )
    latent_audit["contract"] = (
        f"causal_historical_latent_placebo_last_"
        f"{CONTEXT_PLACEBO_LOOKBACK_SESSIONS}_sessions_v1"
        if mode in {"all", "latent_only"}
        else identity
    )


@dataclass(frozen=True)
class AdaptiveDayEvents:
    observed: np.ndarray
    observed_available: np.ndarray
    realized: np.ndarray
    realized_available: np.ndarray


def daily_event_count_row(
    date: str,
    events: AdaptiveDayEvents,
    horizon_labels: tuple[str, ...],
) -> dict[str, Any]:
    if (
        events.realized.ndim != 2
        or events.realized_available.shape != events.realized.shape
    ):
        raise ValueError("daily realized-event masks must be aligned matrices")
    if events.realized.shape[1] != len(horizon_labels):
        raise ValueError("daily event horizons do not match the release contract")
    if events.observed.shape != events.observed_available.shape:
        raise ValueError("daily observed-event masks must be aligned")
    if events.observed.shape != (events.realized.shape[0],):
        raise ValueError("daily observed and realized event timestamps must align")
    realized: dict[str, Any] = {}
    joint: dict[str, Any] = {}
    for horizon, label in enumerate(horizon_labels):
        realized_event = events.realized[:, horizon]
        realized_available = events.realized_available[:, horizon]
        joint_event = events.observed & realized_event
        joint_available = events.observed_available & realized_available
        realized[label] = {
            "positive_timestamps": int(realized_event.sum()),
            "valid_timestamps": int(realized_available.sum()),
        }
        joint[label] = {
            "positive_timestamps": int(joint_event.sum()),
            "valid_timestamps": int(joint_available.sum()),
        }
    return {
        "date": str(date),
        "adaptive_observed_surprise": {
            "positive_timestamps": int(events.observed.sum()),
            "valid_timestamps": int(events.observed_available.sum()),
        },
        "adaptive_realized_impact": realized,
        "adaptive_surprise_and_impact": joint,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a post-impact checkpoint on causal adaptive event tails."
    )
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--stale-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference-summary")
    parser.add_argument("--require-reference-parity", action="store_true")
    parser.add_argument(
        "--evaluation-scope",
        choices=["full", "validation_only"],
        default="full",
    )
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--validation-end", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--quantile", type=float, default=0.80)
    parser.add_argument("--window-sessions", type=int, default=60)
    parser.add_argument("--minimum-history", type=int, default=20)
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


def _evaluation_splits(scope: str) -> tuple[str, ...]:
    if scope == "full":
        return ("validation", "test")
    if scope == "validation_only":
        return ("validation",)
    raise ValueError(f"unsupported evaluation scope: {scope}")


def _scaler(record: dict[str, Any]) -> RobustArrayScaler:
    return RobustArrayScaler(
        center=np.asarray(record["center"], dtype=np.float64),
        scale=np.asarray(record["scale"], dtype=np.float64),
    )


def _calibration(record: dict[str, Any]) -> ResidualSurpriseCalibration:
    return ResidualSurpriseCalibration(
        feature_center=np.asarray(record["feature_center"], dtype=np.float64),
        feature_scale=np.asarray(record["feature_scale"], dtype=np.float64),
        energy_threshold=float(record["energy_threshold"]),
        threshold_quantile=float(record["threshold_quantile"]),
        min_nodes=int(record["min_nodes"]),
        node_z_threshold=float(record["node_z_threshold"]),
        clip=float(record["clip"]),
    )


def _clock_groups(timestamps_utc_ns: np.ndarray) -> np.ndarray:
    timestamps = pd.to_datetime(
        np.asarray(timestamps_utc_ns, dtype=np.int64), unit="ns", utc=True
    ).tz_convert("Asia/Seoul")
    return np.asarray(timestamps.hour * 60 + timestamps.minute, dtype=np.int16)


def clock_bucket_masks(timestamps_utc_ns: np.ndarray) -> dict[str, np.ndarray]:
    minute = _clock_groups(timestamps_utc_ns)
    result = {
        name: (minute >= start) & (minute < end)
        for name, start, end in CLOCK_BUCKETS
    }
    coverage = np.stack(list(result.values()), axis=0).sum(axis=0)
    if (coverage > 1).any():
        raise ValueError("KRX clock buckets overlap")
    return result


def causal_recent_event_mask(
    timestamps_utc_ns: np.ndarray,
    events: np.ndarray,
    *,
    lookback_minutes: int = RECENT_SHOCK_LOOKBACK_MINUTES,
) -> np.ndarray:
    timestamps = np.asarray(timestamps_utc_ns, dtype=np.int64)
    event_values = np.asarray(events, dtype=bool)
    if timestamps.ndim != 1 or event_values.shape != timestamps.shape:
        raise ValueError("recent-event timestamps and events must be aligned vectors")
    if int(lookback_minutes) <= 0:
        raise ValueError("recent-event lookback must be positive")
    if len(timestamps) > 1 and bool((np.diff(timestamps) < 0).any()):
        raise ValueError("recent-event timestamps must be sorted")
    result = np.zeros(len(timestamps), dtype=bool)
    lookback_ns = int(lookback_minutes) * 60 * 1_000_000_000
    last_event_ns: int | None = None
    for index, timestamp in enumerate(timestamps):
        if event_values[index]:
            last_event_ns = int(timestamp)
        if last_event_ns is not None and int(timestamp) - last_event_ns <= lookback_ns:
            result[index] = True
    return result


def update_causal_shock_timestamp_fingerprint(
    digest: Any,
    timestamps_utc_ns: np.ndarray,
) -> None:
    """Append one test day's selected UTC timestamps to a stable digest."""
    timestamps = np.asarray(timestamps_utc_ns, dtype="<i8")
    if timestamps.ndim != 1:
        raise ValueError("causal-shock fingerprint timestamps must be a vector")
    digest.update(int(len(timestamps)).to_bytes(8, "little", signed=False))
    digest.update(timestamps.tobytes(order="C"))


def _observed_surprise_energy(
    release: DayRelease,
    stale: StaleCache,
    date: str,
    calibration: ResidualSurpriseCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    day = release.load(date)
    stale_state = stale.state_row(stale.date_to_row[date])
    residuals, valid = _surprise_residuals(
        release,
        day,
        stale_state,
        stale.state_feature_names,
        residual_conditioned=False,
    )
    values = summarize_residual_surprise(
        residuals,
        valid,
        feature_center=calibration.feature_center,
        feature_scale=calibration.feature_scale,
        stock_count=len(release.tickers),
        min_nodes=calibration.min_nodes,
        node_z_threshold=calibration.node_z_threshold,
        clip=calibration.clip,
    )
    energy = values[
        :, SURPRISE_STATISTIC_NAMES.index("systemic_surprise_energy")
    ]
    return energy, np.isfinite(energy)


def build_adaptive_event_calendar(
    release: DayRelease,
    stale: StaleCache,
    dates: list[str],
    observed_calibration: ResidualSurpriseCalibration,
    impact_thresholds: dict[str, np.ndarray],
    *,
    quantile: float,
    window_sessions: int,
    minimum_history: int,
) -> dict[str, AdaptiveDayEvents]:
    state_index = release.systemic_target_names.index("state_change_energy")
    volume_index = release.systemic_target_names.index("volume_expansion_breadth")
    horizons = len(release.horizon_labels)
    lengths: list[int] = []
    observed_values: list[np.ndarray] = []
    observed_valid: list[np.ndarray] = []
    realized_values: list[np.ndarray] = []
    realized_valid: list[np.ndarray] = []
    clock_groups: list[np.ndarray] = []
    for date in dates:
        day = release.load(date)
        lengths.append(len(day["timestamps_utc_ns"]))
        energy, energy_valid = _observed_surprise_energy(
            release, stale, date, observed_calibration
        )
        observed_values.append(energy)
        observed_valid.append(energy_valid)
        systemic = np.asarray(day["systemic_targets"], dtype=np.float64)
        systemic_valid = np.asarray(day["systemic_available"], dtype=bool)
        state_valid = systemic_valid[..., state_index]
        volume_valid = systemic_valid[..., volume_index]
        state_ratio = np.where(
            state_valid,
            systemic[..., state_index] / impact_thresholds["state"][None],
            -np.inf,
        )
        volume_ratio = np.where(
            volume_valid,
            systemic[..., volume_index] / impact_thresholds["volume"][None],
            -np.inf,
        )
        score = np.maximum(state_ratio, volume_ratio)
        score_valid = (state_valid | volume_valid) & np.isfinite(score)
        realized_values.append(score)
        realized_valid.append(score_valid)
        clock_groups.append(_clock_groups(day["timestamps_utc_ns"]))

    flat_clock = np.concatenate(clock_groups)
    observed_result = causal_grouped_upper_tail(
        np.concatenate(observed_values),
        np.concatenate(observed_valid),
        flat_clock,
        quantile=quantile,
        window=window_sessions,
        minimum_history=minimum_history,
    )
    flat_realized = np.concatenate(realized_values)
    flat_realized_valid = np.concatenate(realized_valid)
    realized_events = np.zeros(flat_realized.shape, dtype=bool)
    realized_available = np.zeros(flat_realized.shape, dtype=bool)
    for horizon in range(horizons):
        result = causal_grouped_upper_tail(
            flat_realized[:, horizon],
            flat_realized_valid[:, horizon],
            flat_clock,
            quantile=quantile,
            window=window_sessions,
            minimum_history=minimum_history,
        )
        realized_events[:, horizon] = result.events
        realized_available[:, horizon] = result.available

    calendar: dict[str, AdaptiveDayEvents] = {}
    start = 0
    for date, length in zip(dates, lengths):
        stop = start + length
        calendar[date] = AdaptiveDayEvents(
            observed=observed_result.events[start:stop],
            observed_available=observed_result.available[start:stop],
            realized=realized_events[start:stop],
            realized_available=realized_available[start:stop],
        )
        start = stop
    return calendar


def _prevalence(positive: int, valid: int) -> dict[str, int | float]:
    return {
        "positive_timestamps": int(positive),
        "valid_timestamps": int(valid),
        "fraction": float(positive / valid) if valid else float("nan"),
    }


def compare_all_subset_metrics(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    splits: tuple[str, ...] = ("validation", "test"),
) -> dict[str, Any]:
    absolute_tolerances = {
        "mae": 1e-5,
        "mse": 1e-7,
        "pearson": 2e-3,
        "skill_vs_zero_mse": 2e-3,
        "direction_accuracy": 5e-3,
    }
    relative_tolerances = {"mae": 5e-3, "mse": 5e-3}
    maximum_difference = {name: 0.0 for name in absolute_tolerances}
    maximum_relative_difference = {
        name: 0.0 for name in relative_tolerances
    }
    compared = 0
    count_mismatches = 0
    finite_mismatches = 0
    if not splits or any(split not in {"validation", "test"} for split in splits):
        raise ValueError("reference parity splits must be validation and/or test")
    for split in splits:
        for family in ("node_targets", "systemic_targets"):
            for target, horizons in candidate[split][family].items():
                for horizon, subsets in horizons.items():
                    current = subsets["all"]
                    expected = reference[split][family][target][horizon]["all"]
                    count_mismatches += int(
                        int(current["count"]) != int(expected["count"])
                    )
                    for metric, tolerance in absolute_tolerances.items():
                        current_value = current[metric]
                        expected_value = expected[metric]
                        current_finite = current_value is not None and np.isfinite(
                            float(current_value)
                        )
                        expected_finite = expected_value is not None and np.isfinite(
                            float(expected_value)
                        )
                        if not current_finite or not expected_finite:
                            finite_mismatches += int(
                                current_finite != expected_finite
                            )
                            continue
                        difference = abs(float(current_value) - float(expected_value))
                        maximum_difference[metric] = max(
                            maximum_difference[metric], difference
                        )
                        relative_difference = difference / max(
                            abs(float(expected_value)), 1e-12
                        )
                        if metric in relative_tolerances:
                            maximum_relative_difference[metric] = max(
                                maximum_relative_difference[metric],
                                relative_difference,
                            )
                        compared += 1
                        exceeds = difference > tolerance
                        if metric in relative_tolerances:
                            exceeds = exceeds and (
                                relative_difference > relative_tolerances[metric]
                            )
                        finite_mismatches += int(exceeds)
    return {
        "passed": count_mismatches == 0 and finite_mismatches == 0,
        "compared_finite_metrics": compared,
        "count_mismatches": count_mismatches,
        "tolerance_exceedances_or_finite_mismatches": finite_mismatches,
        "maximum_absolute_difference": maximum_difference,
        "maximum_relative_difference": maximum_relative_difference,
        "absolute_tolerances": absolute_tolerances,
        "relative_tolerances": relative_tolerances,
    }


def evaluate_split(
    model: CausalPostImpactReforecast,
    release: DayRelease,
    stale: StaleCache,
    dates: list[str],
    state_context_map: dict[str, str],
    latent_context_map: dict[str, str],
    calendar: dict[str, AdaptiveDayEvents],
    observed_calibration: ResidualSurpriseCalibration,
    model_calibration: ResidualSurpriseCalibration,
    node_scaler: RobustArrayScaler,
    stale_scaler: RobustArrayScaler,
    systemic_scaler: RobustArrayScaler,
    impact_thresholds: dict[str, np.ndarray],
    checkpoint_args: argparse.Namespace,
    *,
    batch_days: int,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> dict[str, Any]:
    model.eval()
    node_statistics = {
        target: {
            label: {
                subset: RegressionMetricAccumulator() for subset in SUBSETS
            }
            for label in release.horizon_labels
        }
        for target in release.target_names
    }
    systemic_statistics = {
        target: {
            label: {
                subset: RegressionMetricAccumulator() for subset in SUBSETS
            }
            for label in release.horizon_labels
        }
        for target in release.systemic_target_names
    }
    clock_statistics = {
        label: {
            name: RegressionMetricAccumulator()
            for name, _start, _end in CLOCK_BUCKETS
        }
        for label in release.horizon_labels
    }
    clock_daily_rows = {
        label: {name: [] for name, _start, _end in CLOCK_BUCKETS}
        for label in release.horizon_labels
    }
    clock_causal_shock_statistics = {
        label: {
            name: {
                subset: RegressionMetricAccumulator()
                for subset in CLOCK_CAUSAL_SHOCK_SUBSETS
            }
            for name, _start, _end in CLOCK_BUCKETS
        }
        for label in release.horizon_labels
    }
    clock_causal_shock_daily_rows = {
        label: {
            name: {subset: [] for subset in CLOCK_CAUSAL_SHOCK_SUBSETS}
            for name, _start, _end in CLOCK_BUCKETS
        }
        for label in release.horizon_labels
    }
    clock_causal_shock_timestamp_counts = {
        name: {subset: 0 for subset in CLOCK_CAUSAL_SHOCK_SUBSETS}
        for name, _start, _end in CLOCK_BUCKETS
    }
    clock_causal_shock_timestamp_hashers = {
        name: {subset: hashlib.sha256() for subset in CLOCK_CAUSAL_SHOCK_SUBSETS}
        for name, _start, _end in CLOCK_BUCKETS
    }
    daily_event_rows: list[dict[str, Any]] = []
    endpoint_index = release.target_names.index("endpoint_return")
    observed_positive = 0
    observed_valid = 0
    realized_positive = {label: 0 for label in release.horizon_labels}
    realized_valid = {label: 0 for label in release.horizon_labels}
    joint_positive = {label: 0 for label in release.horizon_labels}
    joint_valid = {label: 0 for label in release.horizon_labels}
    for date_batch in _batches(dates, batch_days, None):
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
        batch = _tensor_batch(numpy_batch, device)
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
        node_prediction = prediction.node.float().cpu().numpy()
        systemic_prediction = (
            prediction.systemic.float().cpu().numpy()
            * systemic_scaler.scale[None, None, None, :]
            + systemic_scaler.center[None, None, None, :]
        )
        for batch_index, date in enumerate(date_batch):
            day = release.load(date)
            count = len(day["timestamps_utc_ns"])
            clock_masks = clock_bucket_masks(day["timestamps_utc_ns"])
            events = calendar[date]
            daily_event_row = daily_event_count_row(
                date, events, release.horizon_labels
            )
            causal_shock_masks = {
                "adaptive_observed_surprise_current": events.observed,
                "adaptive_observed_surprise_recent_30m": causal_recent_event_mask(
                    day["timestamps_utc_ns"], events.observed
                ),
            }
            for bucket, time_mask in clock_masks.items():
                for subset, shock_mask in causal_shock_masks.items():
                    selected_timestamps = np.asarray(
                        day["timestamps_utc_ns"], dtype=np.int64
                    )[time_mask & shock_mask]
                    clock_causal_shock_timestamp_counts[bucket][subset] += int(
                        len(selected_timestamps)
                    )
                    update_causal_shock_timestamp_fingerprint(
                        clock_causal_shock_timestamp_hashers[bucket][subset],
                        selected_timestamps,
                    )
            observed_positive += int(events.observed.sum())
            observed_valid += int(events.observed_available.sum())
            node_target = numpy_batch["targets"][batch_index, :count]
            node_valid = numpy_batch["target_available"][batch_index, :count]
            systemic_target = numpy_batch["systemic_targets"][batch_index, :count]
            systemic_valid_mask = numpy_batch["systemic_available"][
                batch_index, :count
            ]
            for horizon, label in enumerate(release.horizon_labels):
                adaptive_realized = events.realized[:, horizon]
                adaptive_realized_valid = events.realized_available[:, horizon]
                adaptive_joint = events.observed & adaptive_realized
                adaptive_joint_valid = (
                    events.observed_available & adaptive_realized_valid
                )
                realized_positive[label] += int(adaptive_realized.sum())
                realized_valid[label] += int(adaptive_realized_valid.sum())
                joint_positive[label] += int(adaptive_joint.sum())
                joint_valid[label] += int(adaptive_joint_valid.sum())
                subset_masks = {
                    "all": np.ones(count, dtype=bool),
                    "adaptive_observed_surprise": events.observed,
                    "adaptive_realized_impact": adaptive_realized,
                    "adaptive_surprise_and_impact": adaptive_joint,
                }
                for target_index, target_name in enumerate(release.target_names):
                    prediction_values = node_prediction[
                        batch_index, :count, :, horizon, target_index
                    ]
                    target_values = node_target[:, :, horizon, target_index]
                    valid_values = node_valid[:, :, horizon, target_index]
                    for subset, time_mask in subset_masks.items():
                        node_statistics[target_name][label][subset].update(
                            prediction_values,
                            target_values,
                            valid_values & time_mask[:, None],
                        )
                endpoint_prediction = node_prediction[
                    batch_index, :count, :, horizon, endpoint_index
                ]
                endpoint_target = node_target[:, :, horizon, endpoint_index]
                endpoint_valid = node_valid[:, :, horizon, endpoint_index]
                for bucket, time_mask in clock_masks.items():
                    selected = endpoint_valid & time_mask[:, None]
                    clock_statistics[label][bucket].update(
                        endpoint_prediction, endpoint_target, selected
                    )
                    daily = RegressionMetricAccumulator()
                    daily.update(endpoint_prediction, endpoint_target, selected)
                    metrics = daily.metrics()
                    if int(metrics["count"]) > 0:
                        clock_daily_rows[label][bucket].append(
                            {"date": date, **metrics}
                        )
                    for subset, shock_mask in causal_shock_masks.items():
                        shock_selected = endpoint_valid & (
                            time_mask & shock_mask
                        )[:, None]
                        clock_causal_shock_statistics[label][bucket][subset].update(
                            endpoint_prediction,
                            endpoint_target,
                            shock_selected,
                        )
                        shock_daily = RegressionMetricAccumulator()
                        shock_daily.update(
                            endpoint_prediction,
                            endpoint_target,
                            shock_selected,
                        )
                        shock_metrics = shock_daily.metrics()
                        if int(shock_metrics["count"]) > 0:
                            clock_causal_shock_daily_rows[label][bucket][subset].append(
                                {"date": date, **shock_metrics}
                            )
                for target_index, target_name in enumerate(
                    release.systemic_target_names
                ):
                    prediction_values = systemic_prediction[
                        batch_index, :count, horizon, target_index
                    ]
                    target_values = systemic_target[:, horizon, target_index]
                    valid_values = systemic_valid_mask[:, horizon, target_index]
                    for subset, time_mask in subset_masks.items():
                        systemic_statistics[target_name][label][subset].update(
                            prediction_values,
                            target_values,
                            valid_values & time_mask,
                        )
            daily_event_rows.append(daily_event_row)
    node_metrics = {
        target: {
            label: {
                subset: accumulator.metrics()
                for subset, accumulator in subsets.items()
            }
            for label, subsets in horizons.items()
        }
        for target, horizons in node_statistics.items()
    }
    systemic_metrics = {
        target: {
            label: {
                subset: accumulator.metrics()
                for subset, accumulator in subsets.items()
            }
            for label, subsets in horizons.items()
        }
        for target, horizons in systemic_statistics.items()
    }
    return {
        "node_endpoint": node_metrics["endpoint_return"],
        "node_targets": node_metrics,
        "systemic_targets": systemic_metrics,
        "clock_bucket_contract": {
            "name": CLOCK_BUCKET_CONTRACT,
            "timezone": "Asia/Seoul",
            "buckets": {
                name: {
                    "start_minute_inclusive": int(start),
                    "end_minute_exclusive": int(end),
                }
                for name, start, end in CLOCK_BUCKETS
            },
        },
        "clock_bucket_node_endpoint": {
            label: {
                bucket: accumulator.metrics()
                for bucket, accumulator in buckets.items()
            }
            for label, buckets in clock_statistics.items()
        },
        "clock_bucket_daily_node_endpoint_rows": clock_daily_rows,
        "clock_bucket_causal_shock_contract": {
            "name": CLOCK_CAUSAL_SHOCK_CONTRACT,
            "point_in_time_observed_only": True,
            "future_realized_labels_used_for_selection": False,
            "recent_lookback_minutes": RECENT_SHOCK_LOOKBACK_MINUTES,
            "subsets": list(CLOCK_CAUSAL_SHOCK_SUBSETS),
            "timestamp_fingerprint": CAUSAL_SHOCK_TIMESTAMP_FINGERPRINT,
        },
        "clock_bucket_causal_shock_node_endpoint": {
            label: {
                bucket: {
                    subset: accumulator.metrics()
                    for subset, accumulator in subsets.items()
                }
                for bucket, subsets in buckets.items()
            }
            for label, buckets in clock_causal_shock_statistics.items()
        },
        "clock_bucket_causal_shock_daily_node_endpoint_rows": (
            clock_causal_shock_daily_rows
        ),
        "clock_bucket_causal_shock_timestamp_counts": (
            clock_causal_shock_timestamp_counts
        ),
        "clock_bucket_causal_shock_timestamp_sha256": {
            bucket: {
                subset: digest.hexdigest()
                for subset, digest in subsets.items()
            }
            for bucket, subsets in clock_causal_shock_timestamp_hashers.items()
        },
        "event_prevalence": {
            "adaptive_observed_surprise": _prevalence(
                observed_positive, observed_valid
            ),
            "adaptive_realized_impact": {
                label: _prevalence(
                    realized_positive[label], realized_valid[label]
                )
                for label in release.horizon_labels
            },
            "adaptive_surprise_and_impact": {
                label: _prevalence(joint_positive[label], joint_valid[label])
                for label in release.horizon_labels
            },
        },
        "daily_event_rows": daily_event_rows,
    }


def main() -> int:
    args = parse_args()
    evaluation_splits = _evaluation_splits(args.evaluation_scope)
    if args.require_reference_parity and not args.reference_summary:
        raise ValueError("reference parity requires --reference-summary")
    device = _device(args.device)
    amp_dtype = _amp_dtype(args.amp_dtype)
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    daily_context_placebo_mode = _resolved_daily_context_placebo_mode(
        checkpoint_args
    )
    checkpoint_args.daily_context_placebo_mode = daily_context_placebo_mode
    checkpoint_args.shuffle_daily_context = daily_context_placebo_mode == "all"
    release = DayRelease(
        Path(args.day_release_dir), cache=bool(args.cache_day_shards)
    )
    stale = StaleCache(Path(args.stale_cache_dir))
    stale.align_tickers(release.tickers)
    if tuple(checkpoint["feature_names"]) != release.feature_names:
        raise ValueError("checkpoint and day-release feature contracts differ")
    expected_graph_features = _graph_message_feature_names(
        release.feature_names,
        str(getattr(checkpoint_args, "graph_message_mode", "none")),
    )
    if tuple(
        checkpoint.get("graph_message_feature_names", expected_graph_features)
    ) != expected_graph_features:
        raise ValueError("checkpoint graph-message feature contract differs")
    if tuple(checkpoint["state_feature_names"]) != stale.state_feature_names:
        raise ValueError("checkpoint and stale-cache state contracts differ")
    if tuple(checkpoint["horizon_labels"]) != release.horizon_labels:
        raise ValueError("checkpoint and day-release horizons differ")
    if tuple(checkpoint["target_names"]) != release.target_names:
        raise ValueError("checkpoint and day-release targets differ")
    if tuple(checkpoint["systemic_target_names"]) != release.systemic_target_names:
        raise ValueError("checkpoint and day-release systemic targets differ")

    common_dates = sorted(set(release.dates) & set(stale.dates))
    train_dates, validation_dates, test_dates = _split_dates(
        common_dates, args.train_end, args.validation_end, args.test_end
    )
    evaluation_date_splits = [train_dates, validation_dates]
    if "test" in evaluation_splits:
        evaluation_date_splits.append(test_dates)
    state_context_map, latent_context_map = _daily_context_maps(
        tuple(evaluation_date_splits),
        mode=daily_context_placebo_mode,
        seed=int(checkpoint_args.seed),
    )
    selected_dates = [
        date for split_dates in evaluation_date_splits for date in split_dates
    ]
    context_map_audit = stale.audit_context_map(
        selected_dates, state_context_map
    )
    latent_context_map_audit = stale.audit_context_map(
        selected_dates, latent_context_map
    )
    _annotate_context_map_contracts(
        context_map_audit,
        latent_context_map_audit,
        daily_context_placebo_mode,
    )
    node_scaler = _scaler(checkpoint["node_scaler"])
    target_scaler = _scaler(checkpoint["target_scaler"])
    systemic_scaler = _scaler(checkpoint["systemic_scaler"])
    stale_scaler = _scaler(checkpoint["stale_scaler"])
    observed_calibration = _calibration(
        checkpoint["observed_surprise_calibration"]
    )
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
        quantile=float(args.quantile),
        window_sessions=int(args.window_sessions),
        minimum_history=int(args.minimum_history),
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
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate_split(
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
        systemic_scaler,
        impact_thresholds,
        checkpoint_args,
        batch_days=int(args.batch_days),
        device=device,
        amp_dtype=amp_dtype,
    )
    test_evaluated = "test" in evaluation_splits
    if test_evaluated:
        test = evaluate_split(
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
            systemic_scaler,
            impact_thresholds,
            checkpoint_args,
            batch_days=int(args.batch_days),
            device=device,
            amp_dtype=amp_dtype,
        )
    else:
        test = None
    candidate_metrics = {"validation": validation}
    if test_evaluated:
        candidate_metrics["test"] = test
    reference_parity = None
    reference_path = None
    if args.reference_summary:
        reference_path = Path(args.reference_summary)
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        reference_parity = compare_all_subset_metrics(
            reference,
            candidate_metrics,
            splits=evaluation_splits,
        )
    output = {
        "schema_version": 1,
        "adaptive_event_contract": ADAPTIVE_EVENT_CONTRACT,
        "evaluation_scope": args.evaluation_scope,
        "test_evaluated": test_evaluated,
        "variant": checkpoint_args.variant,
        "shuffle_daily_context": bool(checkpoint_args.shuffle_daily_context),
        "daily_context_placebo_mode": daily_context_placebo_mode,
        "graph_message_mode": str(
            getattr(checkpoint_args, "graph_message_mode", "none")
        ),
        "graph_message_fusion": str(
            getattr(checkpoint_args, "graph_message_fusion", "shared")
        ),
        "freeze_base_for_message_adapter": bool(
            getattr(checkpoint_args, "freeze_base_for_message_adapter", False)
        ),
        "post_shock_correlation_contract": {
            "weight": float(
                getattr(
                    checkpoint_args,
                    "post_shock_correlation_loss_weight",
                    0.0,
                )
            ),
            "horizons": [
                value.strip()
                for value in str(
                    getattr(
                        checkpoint_args,
                        "post_shock_correlation_horizons",
                        "15m,30m,60m",
                    )
                ).split(",")
                if value.strip()
            ],
            "lookback_minutes": int(
                getattr(checkpoint_args, "post_shock_lookback_minutes", 30)
            ),
            "minimum_nodes": int(
                getattr(checkpoint_args, "post_shock_minimum_nodes", 100)
            ),
            "point_in_time_observed_shock_only": True,
            "future_labels_used_for_event_selection": False,
        },
        "parameters": {
            "quantile": float(args.quantile),
            "window_sessions": int(args.window_sessions),
            "minimum_history": int(args.minimum_history),
        },
        "splits": {
            "train": {
                "start": train_dates[0],
                "end": train_dates[-1],
                "days": len(train_dates),
            },
            "validation": {
                "start": validation_dates[0],
                "end": validation_dates[-1],
                "days": len(validation_dates),
            },
            "test": {
                "start": test_dates[0],
                "end": test_dates[-1],
                "days": len(test_dates),
            },
        },
        "context_map_audit": context_map_audit,
        "latent_context_map_audit": latent_context_map_audit,
        "reference_inference_parity": reference_parity,
        "validation": validation,
        "test": test,
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "reference_summary": str(reference_path) if reference_path else None,
            "reference_summary_sha256": (
                file_sha256(reference_path) if reference_path else None
            ),
            "day_release_manifest_sha256": file_sha256(release.manifest_path),
            "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    output_path = Path(args.output)
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
                "status": "complete",
                "variant": checkpoint_args.variant,
                "output": str(output_path),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if (
        args.require_reference_parity
        and reference_parity is not None
        and reference_parity["passed"] is not True
    ):
        raise ValueError("adaptive evaluator failed checkpoint inference parity")
    return 0


if __name__ == "__main__":
    sys.exit(main())
