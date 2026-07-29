from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    context_rows_for_step,
    evaluator_namespace,
    graph_neighbor_state,
)
from scripts.benchmark_qlib_lgb import validate_contract
from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    graph_edge_kwargs,
    load_model,
)
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.real_features import make_real_snapshot


DEFAULT_HORIZONS = (1, 2, 3, 5, 10)


def parse_horizons(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    else:
        parsed = tuple(int(item) for item in value)
    if not parsed or len(parsed) != len(set(parsed)) or any(item <= 0 for item in parsed):
        raise ValueError("horizons must be unique positive integers")
    return parsed


def build_rollout_plan(
    checkpoint_args: Mapping[str, Any], horizons: Sequence[int]
) -> dict[int, int]:
    namespace = argparse.Namespace(**dict(checkpoint_args))
    return {
        int(horizon): int(rollout_steps_for_offset(namespace, int(horizon)))
        for horizon in horizons
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def summarize_timings(rows: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("at least one timing row is required")
    keys = tuple(rows[0])
    if any(tuple(row) != keys for row in rows):
        raise ValueError("timing rows must have identical ordered keys")
    result: dict[str, Any] = {}
    for key in keys:
        values = np.asarray([float(row[key]) * 1000.0 for row in rows], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"invalid timing values for {key}")
        result[key.removesuffix("_sec") + "_ms"] = {
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(values.max()),
        }
    return result


def validate_feature_contract(
    actual_names: Sequence[str], contract: Mapping[str, Any]
) -> None:
    expected_names = [str(value) for value in contract.get("feature_names", ())]
    if list(actual_names) != expected_names:
        mismatch = next(
            (
                index
                for index, (actual, expected) in enumerate(
                    zip(actual_names, expected_names)
                )
                if actual != expected
            ),
            min(len(actual_names), len(expected_names)),
        )
        raise ValueError(
            "live Qlib feature layout differs from the PIT bundle at "
            f"column {mismatch}: actual={len(actual_names)} expected={len(expected_names)}"
        )
    if int(contract.get("feature_count", -1)) != len(actual_names):
        raise ValueError("Qlib feature count differs from the PIT bundle")


def _date_steps(features, start: str, end: str) -> np.ndarray:
    dates = np.asarray(features.dates, dtype="datetime64[D]")
    mask = (dates >= np.datetime64(start)) & (dates <= np.datetime64(end))
    return np.flatnonzero(mask).astype(np.int64)


def resolve_benchmark_step(features, contract: Mapping[str, Any], date: str | None) -> int:
    requested = str(date or contract["splits"]["test"]["end"])
    dates = np.asarray(features.dates, dtype="datetime64[D]")
    matches = np.flatnonzero(dates == np.datetime64(requested))
    if len(matches) != 1:
        raise ValueError(f"benchmark date is absent or duplicated: {requested}")
    test_start = np.datetime64(str(contract["splits"]["test"]["start"]))
    test_end = np.datetime64(str(contract["splits"]["test"]["end"]))
    if not (test_start <= dates[matches[0]] <= test_end):
        raise ValueError("benchmark date must belong to the frozen Qlib test split")
    return int(matches[0])


def load_qlib_boosters(
    model_dir: Path,
    horizons: Sequence[int],
    *,
    expected_checkpoint_sha256: str,
):
    import lightgbm

    summary_path = model_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("role") != "research_only_qlib_lightgbm_baseline":
        raise ValueError("unexpected Qlib model role")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError("Qlib model artifact does not prohibit live orders")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError("Qlib model artifact used test data for selection")
    if summary.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("Qlib model checkpoint differs from the JEPA checkpoint")

    boosters = {}
    hashes = {}
    for horizon in horizons:
        path = model_dir / f"lightgbm_h{int(horizon)}.txt"
        actual_hash = sha256_file(path)
        expected_hash = (
            summary.get("horizons", {}).get(str(horizon), {}).get("model_sha256")
        )
        if actual_hash != expected_hash:
            raise ValueError(f"Qlib model hash mismatch for horizon {horizon}")
        boosters[int(horizon)] = lightgbm.Booster(model_file=str(path))
        hashes[str(horizon)] = actual_hash
    return boosters, hashes, summary


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
            "current_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    return {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one complete JEPA state plus Qlib return sensing cycle."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--qlib-bundle-dir", required=True)
    parser.add_argument("--qlib-model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--date")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--latency-p95-ms-max", type=float, default=250.0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cycles <= 0 or args.warmup < 0:
        raise ValueError("cycles must be positive and warmup must be non-negative")
    horizons = parse_horizons(args.horizons)
    device = torch.device(args.device)
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    checkpoint_hash = sha256_file(checkpoint_path)

    bundle_dir = Path(args.qlib_bundle_dir).resolve()
    contract = validate_contract(bundle_dir)
    if contract.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("Qlib PIT bundle differs from the JEPA checkpoint")
    if tuple(int(value) for value in contract.get("horizons", ())) != horizons:
        raise ValueError("requested horizons differ from the frozen Qlib PIT bundle")

    model, ckpt = load_model(model_dir, device)
    feature_args = deepcopy(args)
    feature_args.horizons = ",".join(str(value) for value in horizons)
    features, ckpt_args = build_features_from_ckpt(
        ckpt, evaluator_namespace(feature_args)
    )
    fit_steps = _date_steps(
        features,
        str(contract["splits"]["fit"]["start"]),
        str(contract["splits"]["fit"]["end"]),
    )
    if len(fit_steps) != int(contract["splits"]["fit"]["dates"]):
        raise ValueError("rebuilt fit dates differ from the Qlib PIT bundle")
    layout = build_context_layout(features, fit_steps, include_calendar=False)
    actual_names = list(layout.base_feature_names)
    if bool(contract.get("uses_graph_neighbor_state")):
        actual_names.extend(layout.graph_feature_names)
    validate_feature_contract(actual_names, contract)

    step = resolve_benchmark_step(features, contract, args.date)
    edge_window = int(ckpt_args.get("edge_window", 60))
    if step < edge_window:
        raise ValueError("benchmark step does not have enough edge history")
    boosters, model_hashes, qlib_summary = load_qlib_boosters(
        Path(args.qlib_model_dir).resolve(),
        horizons,
        expected_checkpoint_sha256=checkpoint_hash,
    )

    timings: list[dict[str, float]] = []
    prediction_means: dict[str, float] = {}
    state_means: dict[str, float] = {}
    edge_top_k = int(ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(ckpt_args.get("min_abs_corr", 0.2))
    rollout_plan = build_rollout_plan(ckpt_args, horizons)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for cycle in range(int(args.warmup) + int(args.cycles)):
            t0 = time.perf_counter()
            batch = make_real_snapshot(
                features,
                step=step,
                full_observation=True,
                edge_window=edge_window,
                top_k=edge_top_k,
                min_abs_corr=min_abs_corr,
                **graph_edge_kwargs(ckpt_args, evaluator_namespace(feature_args)),
            ).to(device)
            synchronize(device)
            t1 = time.perf_counter()

            latent_context = model.encode_temporal_context(batch)
            predicted_states = {}
            for horizon in horizons:
                steps_forward = rollout_plan[int(horizon)]
                rollout = model.rollout_latent(
                    latent_context, steps=max(1, steps_forward)
                )
                predicted_states[int(horizon)] = model.predict_temporal_state(
                    batch,
                    rollout,
                    rollout_steps=max(1, steps_forward),
                    z_context=latent_context,
                )
            synchronize(device)
            state_arrays = {
                horizon: state.detach().cpu().numpy()
                for horizon, state in predicted_states.items()
            }
            t2 = time.perf_counter()

            neighbor = graph_neighbor_state(features, step, ckpt_args)
            qlib_context = context_rows_for_step(features, step, layout, neighbor)
            qlib_context = qlib_context[:, : int(contract["feature_count"])]
            t3 = time.perf_counter()

            predictions = {
                horizon: np.asarray(
                    booster.predict(
                        qlib_context,
                        num_threads=max(1, int(args.num_threads)),
                    ),
                    dtype=np.float64,
                )
                for horizon, booster in boosters.items()
            }
            t4 = time.perf_counter()
            expected_state_shape = (features.node_count, len(features.feature_names))
            if any(
                values.shape != expected_state_shape or not np.isfinite(values).all()
                for values in state_arrays.values()
            ) or any(
                values.shape != (features.tradable_count,)
                or not np.isfinite(values).all()
                for values in predictions.values()
            ):
                raise RuntimeError("JEPA or Qlib produced invalid outputs")
            state_means = {
                str(horizon): float(values.mean())
                for horizon, values in state_arrays.items()
            }
            prediction_means = {
                str(horizon): float(values.mean())
                for horizon, values in predictions.items()
            }
            if cycle >= int(args.warmup):
                timings.append(
                    {
                        "snapshot_sec": t1 - t0,
                        "jepa_sec": t2 - t1,
                        "qlib_context_sec": t3 - t2,
                        "qlib_predict_sec": t4 - t3,
                        "total_sec": t4 - t0,
                    }
                )

    timing_summary = summarize_timings(timings)
    total_p95_ms = float(timing_summary["total_ms"]["p95"])
    threshold = float(args.latency_p95_ms_max)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "pass" if total_p95_ms <= threshold else "blocked",
        "role": "research_only_modular_jepa_qlib_latency",
        "jepa_and_qlib_executed": True,
        "live_orders_allowed": False,
        "test_used_for_selection": False,
        "device": str(device),
        "date": str(pd.Timestamp(features.dates[step]).date()),
        "step": step,
        "cycles": int(args.cycles),
        "warmup": int(args.warmup),
        "stocks": int(features.tradable_count),
        "nodes": int(features.node_count),
        "state_features": int(len(features.feature_names)),
        "qlib_features": int(contract["feature_count"]),
        "horizons": list(horizons),
        "rollout_steps": max(rollout_plan.values()),
        "rollout_steps_by_horizon": {
            str(horizon): steps for horizon, steps in rollout_plan.items()
        },
        "total_p95_ms": total_p95_ms,
        "latency_p95_ms_max": threshold,
        "timings": timing_summary,
        "output_checksums": {
            "jepa_state_mean": state_means[str(max(horizons))],
            "jepa_state_means": state_means,
            "qlib_prediction_means": prediction_means,
        },
        "checkpoint_sha256": checkpoint_hash,
        "qlib_bundle_contract_sha256": contract["bundle_contract_sha256"],
        "qlib_model_sha256": model_hashes,
        "qlib_framework": qlib_summary.get("framework"),
        "accelerator_memory": accelerator_memory(device),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
