from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import (
    _edge_settings,
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
    rows_for_steps,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.downstream_probes import (
    CONTINUOUS_TASKS,
    DownstreamTargets,
    FrozenEncoderProbe,
    build_downstream_targets,
    causal_probe_splits,
    evaluate_probe_predictions,
    masked_probe_loss,
    newey_west_mean,
)
from stock_v2.latent_path_head import sha256_file
from stock_v2.real_features import build_edge_tensor, make_real_snapshot


INPUT_VARIANTS = ("raw", "latent", "raw_latent", "raw_shuffled_latent")
TRAINING_MODES = ("single", "multi")


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_names(value: str, allowed: Sequence[str]) -> list[str]:
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(allowed))
    if not parsed or unknown:
        raise ValueError(f"invalid values {unknown or parsed}; allowed={list(allowed)}")
    return list(dict.fromkeys(parsed))


def as_rollout_namespace(ckpt_args: dict[str, Any]) -> argparse.Namespace:
    values = dict(ckpt_args)
    values.setdefault("temporal_offset", ckpt_args.get("horizon", 1))
    values.setdefault("latent_rollout_steps", 1)
    return argparse.Namespace(**values)


def latent_cache_contract(
    checkpoint_path: Path,
    ckpt: dict[str, Any],
    steps: np.ndarray,
    horizons: Sequence[int],
    stock_count: int,
    hidden_dim: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "steps": [int(value) for value in steps],
        "horizons": [int(value) for value in horizons],
        "stock_count": int(stock_count),
        "hidden_dim": int(hidden_dim),
        "dtype": "float16",
    }


def load_or_build_latent_cache(
    model,
    features,
    ckpt: dict[str, Any],
    ckpt_args: dict[str, Any],
    steps: np.ndarray,
    horizons: Sequence[int],
    cache_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(ckpt_args["models_dir"]) / "graph_jepa_real.pt"
    if not checkpoint_path.exists():
        checkpoint_path = Path(model._checkpoint_path)  # type: ignore[attr-defined]
    stock_count = int(features.tradable_count)
    hidden_dim = int(ckpt_args["hidden_dim"])
    row_count = len(steps) * stock_count
    contract = latent_cache_contract(
        checkpoint_path,
        ckpt,
        steps,
        horizons,
        stock_count,
        hidden_dim,
    )
    metadata_path = cache_dir / "metadata.json"
    progress_path = cache_dir / "progress.json"
    context_path = cache_dir / "context.npy"
    delta_paths = {int(h): cache_dir / f"delta_h{int(h)}.npy" for h in horizons}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata == contract and context_path.exists() and all(path.exists() for path in delta_paths.values()):
            context = np.load(context_path, mmap_mode="r")
            deltas = {h: np.load(path, mmap_mode="r") for h, path in delta_paths.items()}
            expected = (row_count, hidden_dim)
            if context.shape == expected and all(values.shape == expected for values in deltas.values()):
                print(f"loaded latent cache: {cache_dir}", flush=True)
                return context, deltas, contract

    start_position = 0
    progress = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
    can_resume = (
        isinstance(progress, dict)
        and progress.get("contract") == contract
        and context_path.exists()
        and all(path.exists() for path in delta_paths.values())
    )
    if can_resume:
        context = np.lib.format.open_memmap(context_path, mode="r+")
        deltas = {
            h: np.lib.format.open_memmap(path, mode="r+")
            for h, path in delta_paths.items()
        }
        expected = (row_count, hidden_dim)
        if context.shape != expected or any(values.shape != expected for values in deltas.values()):
            raise ValueError("partial latent cache shape does not match its progress contract")
        start_position = int(progress.get("completed_dates", 0))
        if not 0 <= start_position <= len(steps):
            raise ValueError("partial latent cache has an invalid completed date count")
        print(
            f"resuming latent cache at date {start_position}/{len(steps)}",
            flush=True,
        )
    else:
        context = np.lib.format.open_memmap(
            context_path,
            mode="w+",
            dtype=np.float16,
            shape=(row_count, hidden_dim),
        )
        deltas = {
            h: np.lib.format.open_memmap(
                path,
                mode="w+",
                dtype=np.float16,
                shape=(row_count, hidden_dim),
            )
            for h, path in delta_paths.items()
        }
    edge_settings = _edge_settings(ckpt_args)
    rollout_args = as_rollout_namespace(ckpt_args)
    model.eval()
    for position in range(start_position, len(steps)):
        step = int(steps[position])
        edge_index, edge_weight = build_edge_tensor(
            features,
            step=step,
            **edge_settings,
        )
        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=int(edge_settings["edge_window"]),
            top_k=int(edge_settings["top_k"]),
            min_abs_corr=float(edge_settings["min_abs_corr"]),
            edge_cache={step: (edge_index, edge_weight)},
        ).to(device)
        with torch.inference_mode():
            latent_context = model.encode_temporal_context(batch)
        start = position * stock_count
        end = start + stock_count
        stock_context = latent_context[:stock_count]
        context[start:end] = stock_context.detach().float().cpu().numpy().astype(np.float16)
        for horizon in horizons:
            rollout_steps = rollout_steps_for_offset(rollout_args, int(horizon))
            with torch.inference_mode():
                predicted = model.rollout_latent(
                    latent_context,
                    steps=max(1, int(rollout_steps)),
                )[:stock_count]
            delta = predicted - stock_context
            deltas[int(horizon)][start:end] = (
                delta.detach().float().cpu().numpy().astype(np.float16)
            )
        if (position + 1) % 25 == 0 or position + 1 == len(steps):
            context.flush()
            for values in deltas.values():
                values.flush()
            progress_payload = {
                "contract": contract,
                "completed_dates": int(position + 1),
            }
            temporary_progress = progress_path.with_suffix(".json.tmp")
            temporary_progress.write_text(
                json.dumps(progress_payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_progress.replace(progress_path)
            print(f"latent cache: {position + 1}/{len(steps)} dates", flush=True)
    context.flush()
    for values in deltas.values():
        values.flush()
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    progress_path.unlink(missing_ok=True)
    return (
        np.load(context_path, mmap_mode="r"),
        {h: np.load(path, mmap_mode="r") for h, path in delta_paths.items()},
        contract,
    )


class ProbeInputs:
    def __init__(
        self,
        raw: np.ndarray,
        context: np.ndarray,
        deltas: dict[int, np.ndarray],
        stock_count: int,
        seed: int,
    ) -> None:
        if len(raw) != len(context) or any(len(values) != len(raw) for values in deltas.values()):
            raise ValueError("raw and latent rows must align")
        self.raw = raw
        self.context = context
        self.deltas = deltas
        if len(raw) % int(stock_count):
            raise ValueError("probe rows must contain complete stock-date blocks")
        generator = np.random.default_rng(int(seed))
        shuffled = np.empty(len(raw), dtype=np.int64)
        for start in range(0, len(raw), int(stock_count)):
            shuffled[start : start + int(stock_count)] = (
                start + generator.permutation(int(stock_count))
            )
        self.shuffled_rows = shuffled

    def dimension(self, variant: str) -> int:
        raw_dim = int(self.raw.shape[1])
        latent_dim = 2 * int(self.context.shape[1])
        if variant == "raw":
            return raw_dim
        if variant == "latent":
            return latent_dim
        if variant in {"raw_latent", "raw_shuffled_latent"}:
            return raw_dim + latent_dim
        raise ValueError(f"unknown input variant: {variant}")

    def batch(
        self,
        variant: str,
        horizon: int,
        rows: np.ndarray,
        device: torch.device,
    ) -> torch.Tensor:
        parts = []
        if variant in {"raw", "raw_latent", "raw_shuffled_latent"}:
            parts.append(np.asarray(self.raw[rows], dtype=np.float32))
        if variant in {"latent", "raw_latent", "raw_shuffled_latent"}:
            latent_rows = self.shuffled_rows[rows] if variant == "raw_shuffled_latent" else rows
            parts.append(np.asarray(self.context[latent_rows], dtype=np.float32))
            parts.append(np.asarray(self.deltas[int(horizon)][latent_rows], dtype=np.float32))
        values = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        return torch.from_numpy(np.ascontiguousarray(values)).to(device=device)


def amp_context(device: torch.device, enabled: bool):
    return (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if device.type == "cuda" and enabled
        else nullcontext()
    )


def make_scaler(device: torch.device, enabled: bool):
    active = device.type == "cuda" and enabled
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=active)
        except TypeError:
            return torch.amp.GradScaler(enabled=active)
    return torch.cuda.amp.GradScaler(enabled=active)


def target_tensors(
    targets: DownstreamTargets,
    rows: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous = torch.from_numpy(
        np.ascontiguousarray(targets.continuous[rows], dtype=np.float32)
    ).to(device)
    continuous_valid = torch.from_numpy(
        np.ascontiguousarray(targets.continuous_valid[rows])
    ).to(device)
    direction = torch.from_numpy(
        np.ascontiguousarray(targets.direction[rows], dtype=np.float32)
    ).to(device)
    direction_valid = torch.from_numpy(
        np.ascontiguousarray(targets.direction_valid[rows])
    ).to(device)
    return continuous, continuous_valid, direction, direction_valid


def evaluation_loss(
    model: FrozenEncoderProbe,
    inputs: ProbeInputs,
    variant: str,
    horizon: int,
    targets: DownstreamTargets,
    rows: np.ndarray,
    task_indices: Sequence[int],
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            batch_rows = rows[start : start + int(batch_size)]
            values = inputs.batch(variant, horizon, batch_rows, device)
            target = target_tensors(targets, batch_rows, device)
            with amp_context(device, amp):
                continuous, direction = model(values)
                loss = masked_probe_loss(
                    continuous.float(),
                    direction.float(),
                    *target,
                    task_indices=task_indices,
                )
            total += float(loss.item()) * len(batch_rows)
            count += len(batch_rows)
    return total / max(count, 1)


def fit_probe(
    inputs: ProbeInputs,
    variant: str,
    mode: str,
    horizon: int,
    targets: DownstreamTargets,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[FrozenEncoderProbe, dict[str, Any]]:
    task_indices = [0] if mode == "single" else list(range(len(CONTINUOUS_TASKS)))
    set_seed(int(args.seed))
    model = FrozenEncoderProbe(
        input_dim=inputs.dimension(variant),
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scaler = make_scaler(device, bool(args.amp))
    generator = np.random.default_rng(int(args.seed))
    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = generator.permutation(fit_rows)
        loss_sum = 0.0
        observed = 0
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start : start + int(args.batch_size)]
            values = inputs.batch(variant, horizon, rows, device)
            target = target_tensors(targets, rows, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, bool(args.amp)):
                continuous, direction = model(values)
                loss = masked_probe_loss(
                    continuous.float(),
                    direction.float(),
                    *target,
                    task_indices=task_indices,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.item()) * len(rows)
            observed += len(rows)
        validation_loss = evaluation_loss(
            model,
            inputs,
            variant,
            horizon,
            targets,
            validation_rows,
            task_indices,
            int(args.batch_size),
            device,
            bool(args.amp),
        )
        row = {
            "epoch": int(epoch),
            "train_loss": float(loss_sum / max(observed, 1)),
            "validation_loss": float(validation_loss),
        }
        history.append(row)
        print(
            f"probe variant={variant} mode={mode} h={horizon} epoch={epoch:02d} "
            f"train={row['train_loss']:.6f} validation={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss - 1e-5:
            best_loss = float(validation_loss)
            best_epoch = int(epoch)
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("probe training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    model.eval()
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "history": history,
        "input_dim": inputs.dimension(variant),
        "task_indices": task_indices,
    }


def predict_probe(
    model: FrozenEncoderProbe,
    inputs: ProbeInputs,
    variant: str,
    horizon: int,
    rows: np.ndarray,
    batch_size: int,
    device: torch.device,
    amp: bool,
) -> tuple[np.ndarray, np.ndarray]:
    continuous = np.empty((len(rows), len(CONTINUOUS_TASKS)), dtype=np.float32)
    direction = np.empty(len(rows), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), int(batch_size)):
            end = min(start + int(batch_size), len(rows))
            values = inputs.batch(variant, horizon, rows[start:end], device)
            with amp_context(device, amp):
                batch_continuous, batch_direction = model(values)
            continuous[start:end] = batch_continuous.float().cpu().numpy()
            direction[start:end] = batch_direction.float().cpu().numpy()
    return continuous, direction


def paired_premium(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    horizon: int,
) -> dict[str, Any]:
    result = {"tasks": {}}
    task_names = sorted(set(candidate["tasks"]) & set(baseline["tasks"]))
    for name in task_names:
        left = candidate["tasks"][name]["daily_ic_values"]
        right = baseline["tasks"][name]["daily_ic_values"]
        differences = [
            float(a) - float(b)
            for a, b in zip(left, right)
            if a is not None and b is not None
        ]
        result["tasks"][name] = newey_west_mean(differences, lag=int(horizon))
    result["direction_brier_improvement"] = float(
        baseline["direction"]["brier"] - candidate["direction"]["brier"]
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test whether a frozen Graph-JEPA encoder adds value across downstream tasks."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--raw-context-cache", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--variants", default=",".join(INPUT_VARIANTS))
    parser.add_argument("--modes", default=",".join(TRAINING_MODES))
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--feature-workers", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    horizons = sorted({int(value) for value in args.horizons.split(",") if value.strip()})
    variants = parse_names(args.variants, INPUT_VARIANTS)
    modes = parse_names(args.modes, TRAINING_MODES)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    model._checkpoint_path = str(model_dir / "graph_jepa_real.pt")  # type: ignore[attr-defined]
    ckpt_args = dict(ckpt.get("args", {}))
    feature_args = deepcopy(args)
    configured_horizons = ckpt_args.get("rollout_offsets", horizons)
    if isinstance(configured_horizons, str):
        feature_args.horizons = configured_horizons
    else:
        feature_args.horizons = ",".join(str(int(value)) for value in configured_horizons)
    features, ckpt_args = build_features_from_ckpt(
        ckpt,
        evaluator_namespace(feature_args),
    )
    train_end = str(ckpt_args["train_end"])
    splits = causal_probe_splits(
        features.dates,
        train_end=train_end,
        edge_window=int(ckpt_args.get("edge_window", 60)),
        max_horizon=max(horizons),
        validation_days=int(args.validation_days),
        max_test_steps=int(args.max_test_steps),
        test_end=args.test_end,
    )
    all_steps = np.unique(
        np.concatenate([splits.fit_steps, splits.validation_steps, splits.test_steps])
    ).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    stock_count = int(features.tradable_count)
    layout = build_context_layout(features, splits.fit_steps)
    raw = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.raw_context_cache),
    )
    context, deltas, latent_contract = load_or_build_latent_cache(
        model,
        features,
        ckpt,
        ckpt_args,
        all_steps,
        horizons,
        Path(args.latent_cache_dir),
        device,
    )
    inputs = ProbeInputs(raw, context, deltas, stock_count=stock_count, seed=int(args.seed))
    fit_rows = rows_for_steps(splits.fit_steps, step_positions, stock_count)
    validation_rows = rows_for_steps(splits.validation_steps, step_positions, stock_count)
    test_rows = rows_for_steps(splits.test_steps, step_positions, stock_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    fresh_summary: dict[str, Any] = {
        "status": "running",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "checkpoint": str(model_dir),
        "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "train_end": train_end,
        "fit_start": str(features.dates[int(splits.fit_steps[0])].date()),
        "fit_end": str(features.dates[int(splits.fit_steps[-1])].date()),
        "validation_start": str(features.dates[int(splits.validation_steps[0])].date()),
        "validation_end": str(features.dates[int(splits.validation_steps[-1])].date()),
        "test_start": str(features.dates[int(splits.test_steps[0])].date()),
        "test_end": str(features.dates[int(splits.test_steps[-1])].date()),
        "fit_dates": int(len(splits.fit_steps)),
        "validation_dates": int(len(splits.validation_steps)),
        "test_dates": int(len(splits.test_steps)),
        "stocks": stock_count,
        "continuous_tasks": list(CONTINUOUS_TASKS),
        "latent_cache_contract": latent_contract,
        "results": {},
        "premiums": {},
    }
    if summary_path.exists():
        existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            existing_summary.get("checkpoint_sha256")
            == fresh_summary["checkpoint_sha256"]
            and existing_summary.get("train_data_manifest_sha256")
            == fresh_summary["train_data_manifest_sha256"]
            and existing_summary.get("latent_cache_contract") == latent_contract
        ):
            summary = existing_summary
            summary["status"] = "running"
            print(f"resuming probe results from {summary_path}", flush=True)
        else:
            raise ValueError("existing probe summary does not match the current run contract")
    else:
        summary = fresh_summary

    for horizon in horizons:
        all_targets = build_downstream_targets(features, all_steps, horizon)
        test_targets = build_downstream_targets(features, splits.test_steps, horizon)
        horizon_result: dict[str, Any] = dict(
            summary["results"].get(str(horizon), {})
        )
        for mode in modes:
            task_indices = [0] if mode == "single" else list(range(len(CONTINUOUS_TASKS)))
            selected_valid = all_targets.continuous_valid[:, task_indices].any(axis=1)
            fit_selected = fit_rows[selected_valid[fit_rows]]
            validation_selected = validation_rows[selected_valid[validation_rows]]
            mode_result: dict[str, Any] = dict(horizon_result.get(mode, {}))
            for variant in variants:
                completed = mode_result.get(variant)
                if completed:
                    artifact_path = Path(str(completed.get("artifact", "")))
                    artifact_sha = str(completed.get("artifact_sha256", ""))
                    if artifact_path.exists() and artifact_sha and sha256_file(artifact_path) == artifact_sha:
                        print(
                            f"skipping completed probe h={horizon} mode={mode} variant={variant}",
                            flush=True,
                        )
                        continue
                    raise ValueError("completed probe metadata has a missing or corrupt artifact")
                print(
                    f"training downstream probe h={horizon} mode={mode} variant={variant} "
                    f"fit_rows={len(fit_selected)} validation_rows={len(validation_selected)}",
                    flush=True,
                )
                probe, fit_metadata = fit_probe(
                    inputs,
                    variant,
                    mode,
                    horizon,
                    all_targets,
                    fit_selected,
                    validation_selected,
                    args,
                    device,
                )
                continuous, direction = predict_probe(
                    probe,
                    inputs,
                    variant,
                    horizon,
                    test_rows,
                    int(args.batch_size),
                    device,
                    bool(args.amp),
                )
                metrics = evaluate_probe_predictions(
                    continuous,
                    direction,
                    test_targets,
                    len(splits.test_steps),
                    stock_count,
                    horizon,
                )
                if mode == "single":
                    metrics["tasks"] = {
                        "path_return": metrics["tasks"]["path_return"]
                    }
                artifact = {
                    "state_dict": probe.state_dict(),
                    "input_variant": variant,
                    "mode": mode,
                    "horizon": int(horizon),
                    "input_dim": inputs.dimension(variant),
                    "continuous_tasks": list(CONTINUOUS_TASKS),
                    "parent_model_sha256": summary["checkpoint_sha256"],
                    "train_data_manifest_sha256": summary["train_data_manifest_sha256"],
                    "train_edge_manifest_sha256": summary["train_edge_manifest_sha256"],
                    "live_orders_allowed": False,
                }
                artifact_path = output_dir / "models" / f"h{horizon}_{mode}_{variant}.pt"
                artifact_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(artifact, artifact_path)
                mode_result[variant] = {
                    "fit": fit_metadata,
                    "metrics": metrics,
                    "artifact": str(artifact_path),
                    "artifact_sha256": sha256_file(artifact_path),
                }
                horizon_result[mode] = mode_result
                summary["results"][str(horizon)] = horizon_result
                summary_path.write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                path_ic = metrics["tasks"]["path_return"]["daily_ic"]
                print(
                    f"result h={horizon} mode={mode} variant={variant} "
                    f"path_ic={path_ic['mean']:+.6f} t={path_ic['newey_west_t']:+.3f} "
                    f"direction_brier={metrics['direction']['brier']:.6f}",
                    flush=True,
                )
        summary["results"][str(horizon)] = horizon_result

    for horizon in horizons:
        horizon_key = str(horizon)
        summary["premiums"][horizon_key] = {}
        for mode in modes:
            rows = summary["results"][horizon_key][mode]
            if "raw" not in rows:
                continue
            summary["premiums"][horizon_key][mode] = {
                variant: paired_premium(
                    rows[variant]["metrics"], rows["raw"]["metrics"], horizon
                )
                for variant in ("latent", "raw_latent", "raw_shuffled_latent")
                if variant in rows
            }
            if "raw_latent" in rows and "raw_shuffled_latent" in rows:
                summary["premiums"][horizon_key][mode]["raw_latent_vs_shuffled"] = paired_premium(
                    rows["raw_latent"]["metrics"],
                    rows["raw_shuffled_latent"]["metrics"],
                    horizon,
                )
    summary["status"] = "complete"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
