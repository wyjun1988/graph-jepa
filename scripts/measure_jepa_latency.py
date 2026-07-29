from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.evaluate_node_prediction import build_features_from_ckpt, graph_edge_kwargs, load_model
from stock_v2.real_features import make_real_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Graph-JEPA snapshot/rollout latency.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--output", default="reports/latency/jepa_latency.json")
    parser.add_argument("--event-path", action="append", default=[])
    parser.add_argument("--event-half-life-days", type=float, default=None)
    parser.add_argument("--event-lag-days", type=int, default=None)
    parser.add_argument("--event-max-decay-days", type=int, default=None)
    parser.add_argument("--event-edge-top-k", type=int, default=None)
    parser.add_argument("--event-edge-min-weight", type=float, default=None)
    parser.add_argument("--event-edge-scale", type=float, default=None)
    parser.add_argument("--event-edge-max-themes", type=int, default=None)
    parser.add_argument("--event-edge-min-theme-count", type=int, default=None)
    parser.add_argument("--fundamental-path", action="append", default=[])
    parser.add_argument("--fundamental-lag-days", type=int, default=None)
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", type=int, default=None)
    parser.add_argument(
        "--external-preset",
        choices=["none", "kr_global", "kr_global_rates"],
        default=None,
    )
    parser.add_argument("--external-symbol", action="append", default=[])
    parser.add_argument("--external-lag-days", type=int, default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--industry-profile-path", action="append", default=[])
    parser.add_argument("--industry-prefix-length", type=int, default=None)
    parser.add_argument("--industry-edge-scale", type=float, default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--override-universe", action="store_true")
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    parser.add_argument("--edge-window", type=int, default=None)
    parser.add_argument("--edge-top-k", type=int, default=None)
    parser.add_argument("--min-abs-corr", type=float, default=None)
    parser.add_argument("--edge-correlation-mode", choices=["signed", "abs", "positive", "negative", "none"], default=None)
    parser.add_argument("--partial-corr-top-k", type=int, default=None)
    parser.add_argument("--partial-corr-min-abs", type=float, default=None)
    parser.add_argument("--partial-corr-mode", choices=["signed", "abs", "positive", "negative"], default=None)
    parser.add_argument("--partial-corr-scale", type=float, default=None)
    parser.add_argument("--lead-lag-top-k", type=int, default=None)
    parser.add_argument("--lead-lag-days", type=int, default=None)
    parser.add_argument("--lead-lag-min-abs-corr", type=float, default=None)
    parser.add_argument("--lead-lag-mode", choices=["signed", "abs", "positive", "negative"], default=None)
    parser.add_argument("--lead-lag-scale", type=float, default=None)
    parser.add_argument("--policy-rate-edge-scale", type=float, default=None)
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--horizons", default="1,2,3,5,10")
    return parser.parse_args()


def pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_contract(model_dir: Path, ckpt: dict[str, Any]) -> dict[str, Any]:
    checkpoint = model_dir / "graph_jepa_real.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    return {
        "role": "research_only_latency_benchmark",
        "live_orders_allowed": False,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get(
            "sha256"
        ),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get(
            "sha256"
        ),
    }


def model_memory_contract(model: torch.nn.Module) -> dict[str, int]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    return {
        "parameter_count": sum(tensor.numel() for tensor in parameters),
        "parameter_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in parameters
        ),
        "buffer_bytes": sum(
            tensor.numel() * tensor.element_size() for tensor in buffers
        ),
    }


def accelerator_memory_snapshot(device: torch.device) -> dict[str, int]:
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


def summarize_accelerator_memory(
    samples: list[dict[str, int]],
) -> dict[str, Any]:
    result: dict[str, Any] = {"unit": "bytes", "samples": len(samples)}
    for key in sorted({key for sample in samples for key in sample}):
        values = [sample[key] for sample in samples if key in sample]
        result[f"max_{key}"] = max(values)
        result[f"last_{key}"] = values[-1]
    return result


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def fill_evaluator_contract(args: argparse.Namespace) -> argparse.Namespace:
    """Supply evaluator fields this script has no flag of its own for.

    build_features_from_ckpt reads its cli_args attributes unconditionally, so a
    field the evaluator gains after this script was last touched becomes an
    AttributeError here rather than a default. external_etf_panel arrived with
    the US-ETF node work and was never mirrored outward -- the same omission
    broke benchmark_direct_baselines and, through it, the daily chain's cache
    build. Here it blocked the rolling gate, whose latency check reads this
    script's output.

    Filling the whole contract rather than the one field that happens to be
    missing today means the next evaluator field lands as a checkpoint fallback
    instead of a crash. None/False/[] mean "use what the checkpoint recorded",
    matching every other caller. Flags the parser already defines are untouched.
    """

    contract: dict[str, Any] = {
        "override_universe": False,
        "universe_manifest": None,
        "allow_unverified_legacy": False,
        "fundamental_path": [],
        "fundamental_lag_days": None,
        "investor_cache_dir": None,
        "investor_flow_lag_days": None,
        "external_symbol": [],
        "external_preset": None,
        "external_lag_days": None,
        "external_cache_dir": None,
        "external_etf_panel": None,
        "external_etf_symbols": None,
        "industry_profile_path": [],
        "industry_prefix_length": None,
        "industry_edge_scale": None,
    }
    for field, default in contract.items():
        if not hasattr(args, field):
            setattr(args, field, default)
    return args


def main() -> None:
    args = fill_evaluator_contract(parse_args())
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    model_memory = model_memory_contract(model)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    edge_window = int(args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(args.min_abs_corr if args.min_abs_corr is not None else ckpt_args.get("min_abs_corr", 0.2))
    step = args.step if args.step >= 0 else len(features.dates) - 1
    step = max(edge_window, min(step, len(features.dates) - 1))
    timings: list[dict[str, float]] = []
    memory_samples: list[dict[str, int]] = []
    with torch.no_grad():
        for idx in range(args.warmup + args.cycles):
            t0 = time.perf_counter()
            batch = make_real_snapshot(
                features,
                step=step,
                full_observation=True,
                edge_window=edge_window,
                top_k=edge_top_k,
                min_abs_corr=min_abs_corr,
                **graph_edge_kwargs(ckpt_args, args),
            ).to(device)
            synchronize(device)
            t1 = time.perf_counter()
            context = model.encode_temporal_context(batch)
            synchronize(device)
            t2 = time.perf_counter()
            z = model.rollout_latent(context, steps=max(1, args.rollout_steps))
            state = model.predict_temporal_state(
                batch,
                z,
                rollout_steps=max(1, args.rollout_steps),
                z_context=context,
            )
            synchronize(device)
            _ = state.detach().cpu().numpy()
            t3 = time.perf_counter()
            if idx >= args.warmup:
                timings.append({
                    "snapshot_sec": t1 - t0,
                    "forward_sec": t2 - t1,
                    "rollout_sec": t3 - t2,
                    "total_sec": t3 - t0,
                })
                memory_samples.append(accelerator_memory_snapshot(device))
    summary: dict[str, Any] = {
        **checkpoint_contract(model_dir, ckpt),
        "model_dir": args.model_dir,
        "device": args.device,
        "tickers": len(features.tickers),
        "nodes": features.node_count,
        "stock_node_count": features.tradable_count,
        "features": len(features.feature_names),
        "date": str(features.dates[step].date()),
        "step": int(step),
        "rollout_steps": args.rollout_steps,
        "cycles": args.cycles,
        "model_memory": model_memory,
        "accelerator_memory": summarize_accelerator_memory(memory_samples),
    }
    for key in ["snapshot_sec", "forward_sec", "rollout_sec", "total_sec"]:
        values = [row[key] for row in timings]
        summary[key] = {
            "mean": float(statistics.mean(values)),
            "median": float(statistics.median(values)),
            "p95": pct(values, 95),
            "max": max(values),
        }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
