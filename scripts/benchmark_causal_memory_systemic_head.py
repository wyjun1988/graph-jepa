from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.audit_systemic_transition_targets import _actual_rows, _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_direct_systemic_transition_head import (
    build_design,
    normalize_design,
)
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.benchmark_systemic_transition_head import (
    _batch_target,
    _masked_component_loss,
    _subsample,
    _target_arrays,
    _validation_score,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    summarize_predictions,
    trajectory_metrics,
)
from scripts.compare_systemic_transition_heads import absolute_gate
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list, rollout_steps_for_offset
from stock_v2.causal_residual_memory import (
    align_matured_residuals,
    build_causal_residual_memory,
    shuffled_within_splits,
)
from stock_v2.systemic_head import (
    CausalMemorySystemicTransitionHead,
    SYSTEMIC_COMPONENT_TARGETS,
    SYSTEMIC_EVENT_TARGETS,
    correlation_rank_loss,
    focal_binary_loss,
    weighted_smooth_l1_loss,
)
from stock_v2.systemic_transition import (
    event_labels,
    score_systemic_components,
)


VARIANTS = (
    "latent_only",
    "latent_raw",
    "latent_raw_memory",
    "latent_raw_shuffled_memory",
)
LOSS_WEIGHTS = {
    "components": 0.18,
    "energy": 0.16,
    "energy_rank": 0.10,
    "event": 0.12,
    "subtypes": 0.16,
    "broad_selloff": 0.12,
    "direction": 0.16,
}


def _sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    view = np.ascontiguousarray(values).view(np.uint8).ravel()
    for start in range(0, len(view), 1024 * 1024):
        digest.update(view[start : start + 1024 * 1024])
    return digest.hexdigest()


def _cache_contract(
    model_dir: Path,
    steps: np.ndarray,
    horizons: Sequence[int],
    node_count: int,
    latent_dim: int,
    eligible_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": checkpoint_sha256(model_dir),
        "steps_sha256": _sha256_array(np.asarray(steps, dtype=np.int64)),
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "rows": int(len(steps)),
        "horizons": [int(value) for value in horizons],
        "node_count": int(node_count),
        "latent_dim": int(latent_dim),
        "eligible_indices": np.asarray(eligible_indices, dtype=np.int64).tolist(),
        "dtype": "float16",
    }


def _load_cache_arrays(
    cache_dir: Path,
    horizons: Sequence[int],
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    context = np.load(cache_dir / "context.npy", mmap_mode="r")
    predicted = {
        int(horizon): np.load(
            cache_dir / f"predicted_h{int(horizon)}.npy", mmap_mode="r"
        )
        for horizon in horizons
    }
    state = {
        int(horizon): np.load(
            cache_dir / f"state_h{int(horizon)}.npy", mmap_mode="r"
        )
        for horizon in horizons
    }
    return context, predicted, state


def build_or_load_forecast_cache(
    model,
    model_dir: Path,
    features,
    steps: np.ndarray,
    horizons: Sequence[int],
    checkpoint_args: Mapping[str, Any],
    feature_args,
    edge_cache,
    eligible_indices: np.ndarray,
    cache_dir: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    contract = _cache_contract(
        model_dir,
        steps,
        horizons,
        features.node_count,
        int(checkpoint_args["hidden_dim"]),
        eligible_indices,
    )
    contract_path = cache_dir / "contract.json"
    complete_path = cache_dir / "CACHE_COMPLETE"
    if complete_path.exists() and contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("forecast cache contract differs from the requested run")
        return (*_load_cache_arrays(cache_dir, horizons), contract)

    progress_path = cache_dir / "progress.json"
    if contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("partial forecast cache contract differs from the run")
        start_row = int(
            json.loads(progress_path.read_text(encoding="utf-8")).get("rows", 0)
        ) if progress_path.exists() else 0
        context = np.lib.format.open_memmap(cache_dir / "context.npy", mode="r+")
        predicted = {
            int(horizon): np.lib.format.open_memmap(
                cache_dir / f"predicted_h{int(horizon)}.npy", mode="r+"
            )
            for horizon in horizons
        }
        state = {
            int(horizon): np.lib.format.open_memmap(
                cache_dir / f"state_h{int(horizon)}.npy", mode="r+"
            )
            for horizon in horizons
        }
    else:
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
        start_row = 0
        context = np.lib.format.open_memmap(
            cache_dir / "context.npy",
            mode="w+",
            dtype=np.float16,
            shape=(len(steps), features.node_count, int(checkpoint_args["hidden_dim"])),
        )
        predicted = {
            int(horizon): np.lib.format.open_memmap(
                cache_dir / f"predicted_h{int(horizon)}.npy",
                mode="w+",
                dtype=np.float16,
                shape=context.shape,
            )
            for horizon in horizons
        }
        state = {
            int(horizon): np.lib.format.open_memmap(
                cache_dir / f"state_h{int(horizon)}.npy",
                mode="w+",
                dtype=np.float16,
                shape=(len(steps), features.node_count, len(eligible_indices)),
            )
            for horizon in horizons
        }

    rollout_args = dict(checkpoint_args)
    rollout_args.setdefault("temporal_offset", checkpoint_args.get("horizon", max(horizons)))
    rollout_args.setdefault("latent_rollout_steps", 1)
    rollout_namespace = argparse.Namespace(**rollout_args)
    model.eval()
    for start in range(start_row, len(steps), int(batch_size)):
        end = min(start + int(batch_size), len(steps))
        selected_steps = steps[start:end]
        batch = snapshot_batch(
            features,
            selected_steps,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
        )
        latent_context, latent_predictions = latent_trajectories(
            model, batch, horizons, checkpoint_args
        )
        batch_rows = len(selected_steps)
        context[start:end] = (
            latent_context.reshape(batch_rows, features.node_count, -1)
            .to(dtype=torch.float16)
            .cpu()
            .numpy()
        )
        for horizon in horizons:
            horizon = int(horizon)
            latent = latent_predictions[horizon]
            predicted[horizon][start:end] = (
                latent.reshape(batch_rows, features.node_count, -1)
                .to(dtype=torch.float16)
                .cpu()
                .numpy()
            )
            rollout_steps = rollout_steps_for_offset(rollout_namespace, horizon)
            with torch.no_grad():
                state_prediction = model.predict_temporal_state(
                    batch,
                    latent,
                    rollout_steps=rollout_steps,
                    z_context=latent_context,
                )
            state[horizon][start:end] = (
                state_prediction[:, eligible_indices]
                .reshape(batch_rows, features.node_count, -1)
                .to(dtype=torch.float16)
                .cpu()
                .numpy()
            )
        for values in (context, *predicted.values(), *state.values()):
            values.flush()
        progress_path.write_text(
            json.dumps({"rows": end}) + "\n", encoding="utf-8"
        )
        print(f"forecast_cache={end}/{len(steps)}", flush=True)
    complete_path.touch()
    return (*_load_cache_arrays(cache_dir, horizons), contract)


def _row_positions(chronology_steps: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    lookup = {int(step): index for index, step in enumerate(chronology_steps)}
    positions = np.asarray([lookup[int(step)] for step in selected], dtype=np.int64)
    if len(np.unique(positions)) != len(positions):
        raise ValueError("split rows do not map uniquely into chronology")
    return positions


def prepare_auxiliary_design(
    features,
    chronology_steps: np.ndarray,
    splits: Mapping[str, np.ndarray],
    forecasts: Mapping[int, np.ndarray],
    eligible_indices: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    actual = features.features[chronology_steps][:, :, eligible_indices]
    available = features.available_mask[chronology_steps][:, :, eligible_indices] > 0.5
    matured, aligned_horizons = align_matured_residuals(
        forecasts,
        actual,
        available,
        available,
        chronology_steps,
    )
    split_rows = {
        name: _row_positions(chronology_steps, steps)
        for name, steps in splits.items()
    }
    memory = build_causal_residual_memory(
        matured,
        split_rows["fit"],
        [features.feature_names[int(index)] for index in eligible_indices],
        stock_count=features.tradable_count,
    )
    raw, raw_names = build_design(features, chronology_steps)
    _, raw_mean, raw_std, raw_transformed = normalize_design(
        raw[split_rows["fit"]],
        raw,
    )
    raw_normalized = raw_transformed[0]
    _, memory_mean, memory_std, memory_transformed = normalize_design(
        memory.values[split_rows["fit"]],
        memory.values,
    )
    memory_normalized = memory_transformed[0]
    shuffled_memory = shuffled_within_splits(
        memory_normalized,
        split_rows,
        seed=int(seed) + 991,
    )
    zero_raw = np.zeros_like(raw_normalized)
    zero_memory = np.zeros_like(memory_normalized)
    variants = {
        "latent_only": np.concatenate((zero_raw, zero_memory), axis=1),
        "latent_raw": np.concatenate((raw_normalized, zero_memory), axis=1),
        "latent_raw_memory": np.concatenate((raw_normalized, memory_normalized), axis=1),
        "latent_raw_shuffled_memory": np.concatenate(
            (raw_normalized, shuffled_memory), axis=1
        ),
    }
    diagnostics = {
        "aligned_horizons": list(aligned_horizons),
        "raw_features": int(raw.shape[1]),
        "memory_features": int(memory.values.shape[1]),
        "auxiliary_features": int(next(iter(variants.values())).shape[1]),
        "raw_feature_names": raw_names.tolist(),
        "memory_feature_names": list(memory.feature_names),
        "memory_group_names": list(memory.group_names),
        "memory_diagnostics": memory.diagnostics,
        "raw_mean_sha256": _sha256_array(raw_mean),
        "raw_std_sha256": _sha256_array(raw_std),
        "memory_mean_sha256": _sha256_array(memory_mean),
        "memory_std_sha256": _sha256_array(memory_std),
        "memory_scale_sha256": _sha256_array(memory.feature_scale),
        "split_rows": {name: values.tolist() for name, values in split_rows.items()},
    }
    return variants, diagnostics


def _direction_targets(targets, horizons: Sequence[int]) -> dict[int, np.ndarray]:
    return {
        int(horizon): np.asarray(
            [float(row["market_return"]) >= 0.0 for row in targets[int(horizon)]["rows"]],
            dtype=np.float32,
        )
        for horizon in horizons
    }


def _weighted_binary_loss(logits, labels, sample_weight):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    weight = sample_weight.to(dtype=loss.dtype).clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def train_epoch(
    head,
    cached_context,
    cached_predicted,
    chronology_rows,
    auxiliary,
    targets,
    direction_targets,
    contracts,
    horizons,
    optimizer,
    device,
    batch_size,
    node_count,
    stock_count,
    seed,
):
    head.train()
    order = np.random.default_rng(seed).permutation(len(chronology_rows))
    losses = []
    components_history = {name: [] for name in LOSS_WEIGHTS}
    for start in range(0, len(order), int(batch_size)):
        positions = order[start : start + int(batch_size)]
        rows = chronology_rows[positions]
        context = torch.as_tensor(
            np.asarray(cached_context[rows]), dtype=torch.float32, device=device
        ).reshape(-1, cached_context.shape[-1])
        aux = torch.as_tensor(auxiliary[rows], dtype=torch.float32, device=device)
        horizon_losses = []
        horizon_weight_sum = 0.0
        batch_components = {name: [] for name in LOSS_WEIGHTS}
        for horizon in horizons:
            predicted = torch.as_tensor(
                np.asarray(cached_predicted[int(horizon)][rows]),
                dtype=torch.float32,
                device=device,
            ).reshape(-1, cached_context.shape[-1])
            component_prediction, energy_prediction, event_logits, direction_logits = head(
                context,
                predicted,
                aux,
                batch_size=len(positions),
                node_count=node_count,
                stock_count=stock_count,
                horizon=int(horizon),
            )
            target = _batch_target(targets, horizon, positions, device)
            component_loss = _masked_component_loss(
                component_prediction,
                target["components"],
                target["component_valid"],
                target["sample_weight"],
            )
            energy_loss = weighted_smooth_l1_loss(
                energy_prediction, target["log_energy"], target["sample_weight"]
            )
            rank_loss = correlation_rank_loss(energy_prediction, target["log_energy"])
            event_loss = focal_binary_loss(event_logits[:, 0], target["labels"][:, 0])
            subtype_pos_weight = torch.as_tensor(
                contracts[int(horizon)].subtype_pos_weight,
                dtype=event_logits.dtype,
                device=device,
            )
            subtype_loss = F.binary_cross_entropy_with_logits(
                event_logits[:, 1:],
                target["labels"][:, 1:],
                pos_weight=subtype_pos_weight,
            )
            broad_selloff_loss = focal_binary_loss(
                event_logits[:, 1], target["labels"][:, 1], alpha=0.80, gamma=2.0
            )
            direction_label = torch.as_tensor(
                direction_targets[int(horizon)][positions], device=device
            )
            direction_loss = _weighted_binary_loss(
                direction_logits, direction_label, target["sample_weight"]
            )
            parts = {
                "components": component_loss,
                "energy": energy_loss,
                "energy_rank": rank_loss,
                "event": event_loss,
                "subtypes": subtype_loss,
                "broad_selloff": broad_selloff_loss,
                "direction": direction_loss,
            }
            loss = sum(LOSS_WEIGHTS[name] * parts[name] for name in LOSS_WEIGHTS)
            weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            horizon_losses.append(weight * loss)
            horizon_weight_sum += weight
            for name, value in parts.items():
                batch_components[name].append(value)
        loss = torch.stack(horizon_losses).sum() / horizon_weight_sum
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, values in batch_components.items():
            components_history[name].append(
                float(torch.stack(values).mean().detach().cpu())
            )
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in components_history.items()
    }


def predict_steps(
    head,
    cached_context,
    cached_predicted,
    chronology_rows,
    auxiliary,
    targets,
    contracts,
    horizons,
    device,
    batch_size,
    node_count,
    stock_count,
):
    head.eval()
    output = {int(horizon): [] for horizon in horizons}
    for start in range(0, len(chronology_rows), int(batch_size)):
        end = min(start + int(batch_size), len(chronology_rows))
        positions = np.arange(start, end, dtype=np.int64)
        rows = chronology_rows[positions]
        context = torch.as_tensor(
            np.asarray(cached_context[rows]), dtype=torch.float32, device=device
        ).reshape(-1, cached_context.shape[-1])
        aux = torch.as_tensor(auxiliary[rows], dtype=torch.float32, device=device)
        with torch.no_grad():
            for horizon in horizons:
                predicted = torch.as_tensor(
                    np.asarray(cached_predicted[int(horizon)][rows]),
                    dtype=torch.float32,
                    device=device,
                ).reshape(-1, cached_context.shape[-1])
                normalized_components, log_energy, logits, direction_logits = head(
                    context,
                    predicted,
                    aux,
                    batch_size=len(positions),
                    node_count=node_count,
                    stock_count=stock_count,
                    horizon=int(horizon),
                )
                contract = contracts[int(horizon)]
                raw_components = (
                    normalized_components.float().cpu().numpy()
                    * contract.component_std[None, :]
                    + contract.component_mean[None, :]
                )
                energy = np.maximum(
                    np.expm1(np.clip(log_energy.float().cpu().numpy(), -5.0, 5.0)),
                    0.0,
                )
                event_scores = logits.float().cpu().numpy()
                direction_scores = direction_logits.float().cpu().numpy()
                for local_position, target_position in enumerate(positions):
                    actual = targets[int(horizon)]["rows"][int(target_position)]
                    predicted_row = {
                        name: float(raw_components[local_position, index])
                        for index, name in enumerate(SYSTEMIC_COMPONENT_TARGETS)
                    }
                    output[int(horizon)].append(
                        {
                            "step": int(actual["step"]),
                            "date": str(actual["date"]),
                            "horizon": int(horizon),
                            "actual": actual,
                            "predicted": predicted_row,
                            "predicted_energy": float(energy[local_position]),
                            "event_logits": event_scores[local_position].tolist(),
                            "direction_logit": float(direction_scores[local_position]),
                        }
                    )
    return output


def _impact_direction_accuracy(records, contract) -> float:
    energies = np.asarray(
        [
            score_systemic_components(row["actual"], contract.calibration)[
                "systemic_energy"
            ]
            for row in records
        ],
        dtype=np.float64,
    )
    events = energies >= float(contract.calibration.event_threshold)
    actual = np.asarray(
        [float(row["actual"]["market_return"]) for row in records], dtype=np.float64
    )
    predicted_up = np.asarray(
        [float(row["direction_logit"]) >= 0.0 for row in records], dtype=bool
    )
    valid = events & np.isfinite(actual)
    if not valid.any():
        return float("nan")
    correct = predicted_up[valid] == (actual[valid] >= 0.0)
    return float(np.average(correct.astype(np.float64), weights=energies[valid]))


def summarize_with_direction(predictions, contracts, horizons):
    metrics, base_score = summarize_predictions(predictions, contracts, horizons)
    direction_values = []
    broad_auc_values = []
    minimum_subtype_values = []
    for horizon in horizons:
        row = metrics[str(int(horizon))]
        direction = _impact_direction_accuracy(
            predictions[int(horizon)], contracts[int(horizon)]
        )
        row["energy_head"][
            "event_impact_weighted_market_direction_accuracy"
        ] = direction
        direction_values.append(direction)
        broad_auc_values.append(float(row["subtypes"]["broad_selloff"]["roc_auc"]))
        minimum_subtype_values.extend(
            float(value["roc_auc"]) for value in row["subtypes"].values()
        )
    direction_score = float(np.nanmean(direction_values))
    broad_score = float(np.nanmean(broad_auc_values))
    minimum_subtype = float(np.nanmin(minimum_subtype_values))
    score = (
        0.70 * float(base_score)
        + 0.15 * np.clip(2.0 * (direction_score - 0.5), -1.0, 1.0)
        + 0.10 * np.clip(2.0 * (broad_score - 0.5), -1.0, 1.0)
        + 0.05 * np.clip(2.0 * (minimum_subtype - 0.5), -1.0, 1.0)
    )
    return metrics, float(score)


def _daily_rows(predictions, contracts, horizons, split):
    rows = []
    for horizon in horizons:
        calibration = contracts[int(horizon)].calibration
        for row in predictions[int(horizon)]:
            actual_energy = score_systemic_components(
                row["actual"], calibration
            )["systemic_energy"]
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(np.asarray(row["event_logits"]), -30.0, 30.0))
            )
            rows.append(
                {
                    "split": split,
                    "date": row["date"],
                    "step": row["step"],
                    "horizon": int(horizon),
                    "actual_systemic_energy": actual_energy,
                    "predicted_systemic_energy": row["predicted_energy"],
                    "actual_market_return": row["actual"]["market_return"],
                    "predicted_market_return": row["predicted"]["market_return"],
                    "predicted_up_probability": float(
                        1.0 / (1.0 + math.exp(-np.clip(row["direction_logit"], -30.0, 30.0)))
                    ),
                    **{
                        f"actual_{name}": bool(
                            event_labels(row["actual"], calibration)[name]
                        )
                        for name in SYSTEMIC_EVENT_TARGETS
                    },
                    **{
                        f"probability_{name}": float(probabilities[index])
                        for index, name in enumerate(SYSTEMIC_EVENT_TARGETS)
                    },
                }
            )
    return sorted(rows, key=lambda item: (item["date"], item["horizon"]))


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def train_variant(
    variant,
    cached_context,
    cached_predicted,
    split_rows,
    auxiliary,
    targets,
    contracts,
    horizons,
    args,
    features,
    device,
):
    torch.manual_seed(int(args.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(args.seed))
    head = CausalMemorySystemicTransitionHead(
        cached_context.shape[-1],
        auxiliary.shape[1],
        horizons,
        projection_dim=int(args.projection_dim),
        auxiliary_projection_dim=int(args.auxiliary_projection_dim),
        hidden_dim=int(args.hidden_dim),
        horizon_dim=int(args.horizon_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    direction_fit = _direction_targets(targets["fit"], horizons)
    history = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, train_components = train_epoch(
            head,
            cached_context,
            cached_predicted,
            split_rows["fit"],
            auxiliary,
            targets["fit"],
            direction_fit,
            contracts,
            horizons,
            optimizer,
            device,
            int(args.batch_size),
            features.node_count,
            features.tradable_count,
            int(args.seed) + epoch,
        )
        validation_predictions = predict_steps(
            head,
            cached_context,
            cached_predicted,
            split_rows["validation"],
            auxiliary,
            targets["validation"],
            contracts,
            horizons,
            device,
            int(args.eval_batch_size),
            features.node_count,
            features.tradable_count,
        )
        _, validation_score = summarize_with_direction(
            validation_predictions, contracts, horizons
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_components": train_components,
                "validation_score": validation_score,
            }
        )
        print(
            f"variant={variant} epoch={epoch:02d} loss={train_loss:.6f} "
            f"validation_score={validation_score:+.6f}",
            flush=True,
        )
        if math.isfinite(validation_score) and validation_score > best_score + 1e-4:
            best_score = validation_score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError(f"{variant} did not produce a validation checkpoint")
    head.load_state_dict(best_state)
    predictions = {
        split: predict_steps(
            head,
            cached_context,
            cached_predicted,
            split_rows[split],
            auxiliary,
            targets[split],
            contracts,
            horizons,
            device,
            int(args.eval_batch_size),
            features.node_count,
            features.tradable_count,
        )
        for split in ("validation", "test")
    }
    summaries = {}
    for split, values in predictions.items():
        horizon_metrics, score = summarize_with_direction(values, contracts, horizons)
        summaries[split] = {
            "horizons": horizon_metrics,
            "weighted_validation_formula_score": score,
            "trajectory": trajectory_metrics(
                values,
                contracts,
                horizons,
                fit_trajectory_event_rate(targets["fit"], contracts, horizons),
            ),
        }
    return head, history, best_score, predictions, summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test causal matured JEPA residual memory in a robust systemic head."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--forecast-cache-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--auxiliary-projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--horizon-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not variants or any(value not in VARIANTS for value in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    horizons = tuple(parse_int_list(args.horizons))
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    splits["fit"] = _subsample(splits["fit"], int(args.max_fit_steps))
    splits["validation"] = _subsample(
        splits["validation"], int(args.max_validation_steps)
    )
    splits["test"] = _subsample(splits["test"], int(args.max_test_steps))
    if any(len(values) == 0 for values in splits.values()):
        raise ValueError("every split must contain at least one step")
    minimum_step = max(0, int(splits["fit"].min()) - max(horizons))
    maximum_step = int(splits["test"].max())
    chronology_steps = np.arange(minimum_step, maximum_step + 1, dtype=np.int64)
    temporal_weights = checkpoint.get("temporal_state_feature_weights")
    if torch.is_tensor(temporal_weights):
        temporal_weights = temporal_weights.detach().cpu().numpy()
    eligible_indices = np.flatnonzero(
        np.asarray(temporal_weights, dtype=np.float32) > 0.0
    )
    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    targets = {
        name: _target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    edge_cache = build_evaluation_edge_cache(
        features, chronology_steps, checkpoint_args, feature_args
    )
    cached_context, cached_predicted, state_forecasts, cache_contract = (
        build_or_load_forecast_cache(
            model,
            model_dir,
            features,
            chronology_steps,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            eligible_indices,
            Path(args.forecast_cache_dir),
            device,
            int(args.cache_batch_size),
        )
    )
    model.to("cpu")
    del model, checkpoint, edge_cache
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    auxiliary_variants, auxiliary_diagnostics = prepare_auxiliary_design(
        features,
        chronology_steps,
        splits,
        state_forecasts,
        eligible_indices,
        seed=int(args.seed),
    )
    split_rows = {
        name: _row_positions(chronology_steps, steps)
        for name, steps in splits.items()
    }
    variant_results = {}
    for variant in variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        head, history, best_score, predictions, metrics = train_variant(
            variant,
            cached_context,
            cached_predicted,
            split_rows,
            auxiliary_variants[variant],
            targets,
            contracts,
            horizons,
            args,
            features,
            device,
        )
        variant_summary = {
            "status": "complete",
            "role": "research_only_causal_memory_systemic_head",
            "variant": variant,
            "best_validation_score": best_score,
            "history": history,
            "metrics": metrics,
            "test_absolute_gate": None,
            "test_used_for_selection": False,
            "live_orders_allowed": False,
        }
        variant_summary["test_absolute_gate"] = absolute_gate(variant_summary)
        (variant_dir / "summary.json").write_text(
            json.dumps(variant_summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        for split in ("validation", "test"):
            _write_csv(
                variant_dir / f"daily_{split}.csv",
                _daily_rows(predictions[split], contracts, horizons, split),
            )
        torch.save(
            {
                "state_dict": head.state_dict(),
                "variant": variant,
                "parent_model_sha256": cache_contract["checkpoint_sha256"],
                "horizons": list(horizons),
                "auxiliary_dim": int(auxiliary_variants[variant].shape[1]),
                "loss_weights": LOSS_WEIGHTS,
                "live_orders_allowed": False,
            },
            variant_dir / "causal_memory_systemic_head.pt",
        )
        variant_results[variant] = {
            "best_validation_score": best_score,
            "test_absolute_gate": variant_summary["test_absolute_gate"],
        }
        del head, predictions
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    selected_variant = max(
        variant_results,
        key=lambda name: float(variant_results[name]["best_validation_score"]),
    )
    memory_gate = None
    if all(name in variant_results for name in (
        "latent_raw",
        "latent_raw_memory",
        "latent_raw_shuffled_memory",
    )):
        real = float(variant_results["latent_raw_memory"]["best_validation_score"])
        raw = float(variant_results["latent_raw"]["best_validation_score"])
        shuffled = float(
            variant_results["latent_raw_shuffled_memory"]["best_validation_score"]
        )
        memory_gate = {
            "passed": real > raw and real > shuffled,
            "real_minus_raw": real - raw,
            "real_minus_shuffled": real - shuffled,
            "selection_split": "validation",
            "test_used_for_selection": False,
        }
    summary = {
        "schema_version": 1,
        "status": "complete",
        "role": "research_only_causal_matured_residual_memory_ablation",
        "model_dir": str(model_dir),
        "parent_model_sha256": cache_contract["checkpoint_sha256"],
        "horizons": list(horizons),
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "chronology": {
            "start": str(features.dates[int(chronology_steps[0])].date()),
            "end": str(features.dates[int(chronology_steps[-1])].date()),
            "rows": len(chronology_steps),
        },
        "variants": variant_results,
        "validation_selected_variant": selected_variant,
        "memory_validation_gate": memory_gate,
        "architecture": {
            "projection_dim": int(args.projection_dim),
            "auxiliary_projection_dim": int(args.auxiliary_projection_dim),
            "hidden_dim": int(args.hidden_dim),
            "horizon_dim": int(args.horizon_dim),
            "dropout": float(args.dropout),
            "pooling": "stock_mean_std_median_q10_q90_attention_external_robust",
            "horizon_specific_output_heads": True,
            "dedicated_direction_head": True,
        },
        "loss_weights": LOSS_WEIGHTS,
        "causal_contract": {
            "memory_updates": "only forecasts whose targets are observed at the current step",
            "feature_scale": "fit split only",
            "stop_gradient": True,
            "future_targets_in_memory": False,
            "test_used_for_selection": False,
        },
        "auxiliary_diagnostics": auxiliary_diagnostics,
        "cache_contract": cache_contract,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_variant": selected_variant,
                "memory_validation_gate": memory_gate,
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
