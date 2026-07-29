from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_post_impact_adaptive_events import _calibration, _scaler
from scripts.train_post_impact_reforecast import (
    DayRelease,
    StaleCache,
    _graph_message_feature_dim,
    _graph_message_feature_names,
    _pad_batch,
    _resolved_daily_context_placebo_mode,
)
from stock_v2.post_impact_reforecast import CausalPostImpactReforecast
from stock_v2.surprise_reforecast import SURPRISE_STATISTIC_NAMES


DEFAULT_CLOCKS = (9 * 60 + 15, 11 * 60, 13 * 60 + 30, 15 * 60, 15 * 60 + 15)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_clocks(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        clocks = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        clocks = tuple(int(item) for item in value)
    if not clocks or len(clocks) != len(set(clocks)):
        raise ValueError("benchmark clocks must be non-empty and unique")
    if any(clock < 9 * 60 or clock > 15 * 60 + 30 for clock in clocks):
        raise ValueError("benchmark clocks must belong to the KRX regular session")
    return clocks


def prefix_indices(
    timestamps_utc_ns: np.ndarray,
    clocks: Sequence[int],
) -> dict[int, int]:
    timestamps = pd.to_datetime(
        np.asarray(timestamps_utc_ns, dtype=np.int64), unit="ns", utc=True
    ).tz_convert("Asia/Seoul")
    minutes = np.asarray(timestamps.hour * 60 + timestamps.minute, dtype=np.int16)
    result: dict[int, int] = {}
    for clock in clocks:
        matches = np.flatnonzero(minutes == int(clock))
        if len(matches) != 1:
            raise ValueError(f"benchmark clock is absent or duplicated: {clock}")
        result[int(clock)] = int(matches[0]) + 1
    return result


def summarize_ms(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError("latency timings must be finite non-negative values")
    return {
        "count": int(len(array)),
        "mean": float(statistics.mean(array)),
        "median": float(statistics.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def accelerator_memory(device: torch.device) -> dict[str, int]:
    if device.type == "mps":
        return {
            "current_allocated_bytes": int(torch.mps.current_allocated_memory()),
            "driver_allocated_bytes": int(torch.mps.driver_allocated_memory()),
            "recommended_max_bytes": int(torch.mps.recommended_max_memory()),
        }
    if device.type == "cuda":
        return {
            "current_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    return {}


def maximum_latest_prefix_difference(
    full: torch.Tensor,
    prefix: torch.Tensor,
    prefix_length: int,
) -> float:
    """Compare one prefix's latest output with the same row from a full pass."""

    length = int(prefix_length)
    if full.ndim < 2 or prefix.ndim != full.ndim:
        raise ValueError("prefix-invariance tensors must include batch and time axes")
    if length <= 0 or length > full.shape[1] or prefix.shape[1] != length:
        raise ValueError("prefix-invariance length does not match tensor time axes")
    if full.shape[:1] + full.shape[2:] != prefix.shape[:1] + prefix.shape[2:]:
        raise ValueError("prefix-invariance tensor shapes are incompatible")
    expected = full[:, length - 1]
    observed = prefix[:, -1]
    if not torch.isfinite(expected).all() or not torch.isfinite(observed).all():
        raise ValueError("prefix-invariance tensors contain non-finite outputs")
    return float(torch.max(torch.abs(expected - observed)).detach().cpu())


def float32_roundoff_tolerance(
    reference: torch.Tensor,
    observed: torch.Tensor,
    *,
    minimum_absolute_tolerance: float,
    ulps: int,
) -> float:
    """Return a scale-aware float32 budget for shape-dependent kernels."""

    if reference.shape != observed.shape or not reference.numel():
        raise ValueError("roundoff reference tensors must be non-empty and aligned")
    if int(ulps) <= 0:
        raise ValueError("roundoff ULP budget must be positive")
    minimum = float(minimum_absolute_tolerance)
    if not np.isfinite(minimum) or minimum < 0.0:
        raise ValueError("minimum roundoff tolerance must be finite and non-negative")
    if not torch.isfinite(reference).all() or not torch.isfinite(observed).all():
        raise ValueError("roundoff reference tensors contain non-finite values")
    magnitude = max(
        1.0,
        float(reference.detach().abs().max().cpu()),
        float(observed.detach().abs().max().cpu()),
    )
    return max(
        minimum,
        float(int(ulps) * np.finfo(np.float32).eps * magnitude),
    )


def perturb_future_inputs(
    tensors: Mapping[str, torch.Tensor],
    prefix_length: int,
) -> dict[str, torch.Tensor]:
    """Change every dynamic input after a prefix while preserving its shape."""

    if "node_values" not in tensors:
        raise ValueError("future perturbation requires node values")
    length = int(prefix_length)
    timestamps = int(tensors["node_values"].shape[1])
    if length <= 0 or length >= timestamps:
        raise ValueError("future perturbation requires a non-empty future suffix")
    values = {
        "node_values",
        "surprise",
        "graph_neighbor_values",
    }
    availability = {
        "node_available",
        "graph_neighbor_available",
    }
    result = dict(tensors)
    for name in values:
        if name not in tensors:
            continue
        changed = tensors[name].clone()
        changed[:, length:] = -changed[:, length:] + 0.125
        result[name] = changed
    for name in availability:
        if name not in tensors:
            continue
        changed = tensors[name].clone()
        changed[:, length:] = ~changed[:, length:].bool()
        result[name] = changed
    return result


def _input_tensors(
    batch: Mapping[str, np.ndarray],
    *,
    prefix: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    timed = {"node_values", "node_available", "surprise"}
    optional = {"graph_neighbor_values", "graph_neighbor_available"}
    result = {
        name: torch.as_tensor(value[:, :prefix], device=device)
        for name, value in batch.items()
        if name in timed or name in optional
    }
    for name in ("stale_state", "context_latent", "predicted_delta"):
        result[name] = torch.as_tensor(batch[name], device=device)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a causal post-impact prefix inference cycle."
    )
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--stale-cache-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--clocks", default=",".join(str(value) for value in DEFAULT_CLOCKS)
    )
    parser.add_argument("--cycles", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--latency-p95-ms-max", type=float, default=250.0)
    parser.add_argument("--prefix-invariance-atol", type=float, default=2e-6)
    parser.add_argument("--prefix-invariance-float32-ulps", type=int, default=32)
    parser.add_argument("--future-perturbation-atol", type=float, default=0.0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(args.cycles) <= 0 or int(args.warmup) < 0:
        raise ValueError("cycles must be positive and warmup must be non-negative")
    if not np.isfinite(args.prefix_invariance_atol) or args.prefix_invariance_atol < 0:
        raise ValueError("prefix-invariance tolerance must be finite and non-negative")
    if int(args.prefix_invariance_float32_ulps) <= 0:
        raise ValueError("prefix-invariance float32 ULP budget must be positive")
    if not np.isfinite(args.future_perturbation_atol) or args.future_perturbation_atol < 0:
        raise ValueError("future-perturbation tolerance must be finite and non-negative")
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    release = DayRelease(Path(args.day_release_dir), cache=True)
    stale = StaleCache(Path(args.stale_cache_dir))
    stale.align_tickers(release.tickers)
    date = str(args.date)
    if date not in release.records or date not in stale.date_to_row:
        raise ValueError("benchmark date is absent from the day or stale release")
    day = release.load(date)
    clocks = parse_clocks(args.clocks)
    prefixes = prefix_indices(day["timestamps_utc_ns"], clocks)

    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    checkpoint_args.daily_context_placebo_mode = _resolved_daily_context_placebo_mode(
        checkpoint_args
    )
    checkpoint_args.shuffle_daily_context = (
        checkpoint_args.daily_context_placebo_mode == "all"
    )
    if checkpoint_args.daily_context_placebo_mode != "none":
        raise ValueError("latency benchmark does not permit placebo context models")
    if tuple(checkpoint["feature_names"]) != release.feature_names:
        raise ValueError("checkpoint and runtime feature contracts differ")
    expected_graph_features = _graph_message_feature_names(
        release.feature_names,
        str(getattr(checkpoint_args, "graph_message_mode", "none")),
    )
    if tuple(
        checkpoint.get("graph_message_feature_names", expected_graph_features)
    ) != expected_graph_features:
        raise ValueError("checkpoint and graph-message feature contracts differ")
    if tuple(checkpoint["state_feature_names"]) != stale.state_feature_names:
        raise ValueError("checkpoint and stale state contracts differ")
    if tuple(checkpoint["horizon_labels"]) != release.horizon_labels:
        raise ValueError("checkpoint and runtime horizon contracts differ")
    if tuple(checkpoint["target_names"]) != release.target_names:
        raise ValueError("checkpoint and runtime target contracts differ")

    node_scaler = _scaler(checkpoint["node_scaler"])
    target_scaler = _scaler(checkpoint["target_scaler"])
    stale_scaler = _scaler(checkpoint["stale_scaler"])
    observed_calibration = _calibration(
        checkpoint["observed_surprise_calibration"]
    )
    model_calibration = _calibration(checkpoint["model_surprise_calibration"])
    impact_thresholds = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in checkpoint["impact_thresholds"].items()
    }
    numpy_batch = _pad_batch(
        release,
        stale,
        [date],
        {date: date},
        {date: date},
        observed_calibration,
        model_calibration,
        node_scaler,
        stale_scaler,
        impact_thresholds,
        checkpoint_args,
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
    model.eval()

    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        full_tensors = _input_tensors(
            numpy_batch,
            prefix=int(numpy_batch["node_values"].shape[1]),
            device=device,
        )
        full_prediction = model(
            full_tensors["node_values"],
            full_tensors["node_available"],
            stale_state=(
                full_tensors["stale_state"]
                if checkpoint_args.variant != "direct"
                else None
            ),
            context_latent=(
                full_tensors["context_latent"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            predicted_delta=(
                full_tensors["predicted_delta"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            surprise_values=full_tensors["surprise"],
            graph_neighbor_values=full_tensors.get("graph_neighbor_values"),
            graph_neighbor_available=full_tensors.get(
                "graph_neighbor_available"
            ),
        )
        synchronize(device)
        for clock, prefix in prefixes.items():
            node_future_difference: float | None = None
            systemic_future_difference: float | None = None
            if prefix < int(full_tensors["node_values"].shape[1]):
                perturbed_tensors = perturb_future_inputs(full_tensors, prefix)
                perturbed_prediction = model(
                    perturbed_tensors["node_values"],
                    perturbed_tensors["node_available"],
                    stale_state=(
                        perturbed_tensors["stale_state"]
                        if checkpoint_args.variant != "direct"
                        else None
                    ),
                    context_latent=(
                        perturbed_tensors["context_latent"].float()
                        if checkpoint_args.variant == "latent"
                        else None
                    ),
                    predicted_delta=(
                        perturbed_tensors["predicted_delta"].float()
                        if checkpoint_args.variant == "latent"
                        else None
                    ),
                    surprise_values=perturbed_tensors["surprise"],
                    graph_neighbor_values=perturbed_tensors.get(
                        "graph_neighbor_values"
                    ),
                    graph_neighbor_available=perturbed_tensors.get(
                        "graph_neighbor_available"
                    ),
                )
                synchronize(device)
                node_future_difference = maximum_latest_prefix_difference(
                    full_prediction.node,
                    perturbed_prediction.node[:, :prefix],
                    prefix,
                )
                systemic_future_difference = maximum_latest_prefix_difference(
                    full_prediction.systemic,
                    perturbed_prediction.systemic[:, :prefix],
                    prefix,
                )
            for cycle in range(int(args.warmup) + int(args.cycles)):
                synchronize(device)
                started = time.perf_counter()
                tensors = _input_tensors(
                    numpy_batch, prefix=prefix, device=device
                )
                synchronize(device)
                transferred = time.perf_counter()
                prediction = model(
                    tensors["node_values"],
                    tensors["node_available"],
                    stale_state=(
                        tensors["stale_state"]
                        if checkpoint_args.variant != "direct"
                        else None
                    ),
                    context_latent=(
                        tensors["context_latent"].float()
                        if checkpoint_args.variant == "latent"
                        else None
                    ),
                    predicted_delta=(
                        tensors["predicted_delta"].float()
                        if checkpoint_args.variant == "latent"
                        else None
                    ),
                    surprise_values=tensors["surprise"],
                    graph_neighbor_values=tensors.get("graph_neighbor_values"),
                    graph_neighbor_available=tensors.get(
                        "graph_neighbor_available"
                    ),
                )
                synchronize(device)
                finished = time.perf_counter()
                latest = prediction.node[:, -1]
                if not torch.isfinite(latest).all():
                    raise RuntimeError(
                        "post-impact inference produced non-finite values"
                    )
                node_prefix_difference = maximum_latest_prefix_difference(
                    full_prediction.node,
                    prediction.node,
                    prefix,
                )
                systemic_prefix_difference = maximum_latest_prefix_difference(
                    full_prediction.systemic,
                    prediction.systemic,
                    prefix,
                )
                node_prefix_tolerance = float32_roundoff_tolerance(
                    full_prediction.node[:, prefix - 1],
                    prediction.node[:, -1],
                    minimum_absolute_tolerance=args.prefix_invariance_atol,
                    ulps=args.prefix_invariance_float32_ulps,
                )
                systemic_prefix_tolerance = float32_roundoff_tolerance(
                    full_prediction.systemic[:, prefix - 1],
                    prediction.systemic[:, -1],
                    minimum_absolute_tolerance=args.prefix_invariance_atol,
                    ulps=args.prefix_invariance_float32_ulps,
                )
                if cycle >= int(args.warmup):
                    rows.append(
                        {
                            "clock_minute": int(clock),
                            "prefix_timestamps": int(prefix),
                            "input_transfer_ms": (transferred - started) * 1000.0,
                            "inference_ms": (finished - transferred) * 1000.0,
                            "total_ms": (finished - started) * 1000.0,
                            "node_prefix_max_abs_difference": node_prefix_difference,
                            "node_prefix_allowed_difference": node_prefix_tolerance,
                            "systemic_prefix_max_abs_difference": (
                                systemic_prefix_difference
                            ),
                            "systemic_prefix_allowed_difference": (
                                systemic_prefix_tolerance
                            ),
                            "node_future_perturbation_max_abs_difference": (
                                node_future_difference
                            ),
                            "systemic_future_perturbation_max_abs_difference": (
                                systemic_future_difference
                            ),
                        }
                    )
    overall = {
        metric: summarize_ms([row[metric] for row in rows])
        for metric in ("input_transfer_ms", "inference_ms", "total_ms")
    }
    by_clock = {
        str(clock): {
            metric: summarize_ms(
                [row[metric] for row in rows if row["clock_minute"] == clock]
            )
            for metric in ("input_transfer_ms", "inference_ms", "total_ms")
        }
        for clock in clocks
    }
    threshold = float(args.latency_p95_ms_max)
    prefix_tolerance_floor = float(args.prefix_invariance_atol)
    node_prefix_maximum = max(
        row["node_prefix_max_abs_difference"] for row in rows
    )
    systemic_prefix_maximum = max(
        row["systemic_prefix_max_abs_difference"] for row in rows
    )
    node_prefix_allowed_maximum = max(
        row["node_prefix_allowed_difference"] for row in rows
    )
    systemic_prefix_allowed_maximum = max(
        row["systemic_prefix_allowed_difference"] for row in rows
    )
    prefix_invariance_passed = all(
        row["node_prefix_max_abs_difference"]
        <= row["node_prefix_allowed_difference"]
        and row["systemic_prefix_max_abs_difference"]
        <= row["systemic_prefix_allowed_difference"]
        for row in rows
    )
    future_rows = [
        row
        for row in rows
        if row["node_future_perturbation_max_abs_difference"] is not None
        and row["systemic_future_perturbation_max_abs_difference"] is not None
    ]
    if not future_rows:
        raise ValueError("future-perturbation gate has no clock with a future suffix")
    node_future_maximum = max(
        row["node_future_perturbation_max_abs_difference"]
        for row in future_rows
    )
    systemic_future_maximum = max(
        row["systemic_future_perturbation_max_abs_difference"]
        for row in future_rows
    )
    future_tolerance = float(args.future_perturbation_atol)
    future_perturbation_passed = (
        node_future_maximum <= future_tolerance
        and systemic_future_maximum <= future_tolerance
    )
    latency_passed = float(overall["total_ms"]["p95"]) <= threshold
    passed = (
        latency_passed
        and prefix_invariance_passed
        and future_perturbation_passed
    )
    day_record = release.records[date]
    output = {
        "schema_version": 1,
        "role": "post_impact_causal_prefix_latency_benchmark",
        "status": "pass" if passed else "blocked",
        "variant": checkpoint_args.variant,
        "device": str(device),
        "date": date,
        "stocks": len(release.tickers),
        "clocks": list(clocks),
        "cycles_per_clock": int(args.cycles),
        "warmup_per_clock": int(args.warmup),
        "future_intraday_rows_visible": False,
        "future_intraday_rows_visible_scope": "timed_prefix_inference_only",
        "prefix_invariance_reference_reads_full_day": True,
        "timing_includes_input_transfer": True,
        "latency_p95_ms_max": threshold,
        "latency_passed": latency_passed,
        "timings": overall,
        "timings_by_clock": by_clock,
        "prefix_invariance": {
            "status": "pass" if prefix_invariance_passed else "blocked",
            "absolute_tolerance_floor": prefix_tolerance_floor,
            "float32_ulps": int(args.prefix_invariance_float32_ulps),
            "node_maximum_absolute_difference": node_prefix_maximum,
            "node_maximum_allowed_difference": node_prefix_allowed_maximum,
            "systemic_maximum_absolute_difference": systemic_prefix_maximum,
            "systemic_maximum_allowed_difference": (
                systemic_prefix_allowed_maximum
            ),
            "interpretation": (
                "A full-day pass and every causally truncated prefix agree at "
                "the prefix endpoint within a scale-aware float32 budget. This "
                "checks shape-dependent numerical consistency, not future use."
            ),
        },
        "future_perturbation_invariance": {
            "status": "pass" if future_perturbation_passed else "blocked",
            "absolute_tolerance": future_tolerance,
            "node_maximum_absolute_difference": node_future_maximum,
            "systemic_maximum_absolute_difference": systemic_future_maximum,
            "same_sequence_length": True,
            "all_dynamic_future_inputs_perturbed": True,
            "clocks_tested": sorted(
                {int(row["clock_minute"]) for row in future_rows}
            ),
            "clocks_without_future_suffix": sorted(
                {
                    int(row["clock_minute"])
                    for row in rows
                    if row["node_future_perturbation_max_abs_difference"] is None
                }
            ),
            "interpretation": (
                "Changing every dynamic input after each prefix must not change "
                "the output at that prefix. This is the strict future-leakage gate."
            ),
        },
        "accelerator_memory": accelerator_memory(device),
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "day_release_manifest_sha256": file_sha256(release.manifest_path),
            "day_shard_sha256": day_record["sha256"],
            "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
            "benchmark_source_sha256": file_sha256(Path(__file__)),
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": output["status"],
                "variant": output["variant"],
                "total_p95_ms": overall["total_ms"]["p95"],
                "prefix_invariance": output["prefix_invariance"]["status"],
                "future_perturbation_invariance": output[
                    "future_perturbation_invariance"
                ]["status"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
