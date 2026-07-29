from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.benchmark_direct_baselines import _edge_settings, evaluator_namespace
from scripts.benchmark_frozen_downstream import as_rollout_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.auxiliary_policy import (
    evaluate_ranked_strategy,
    liquid_universe_mask,
    paired_strategy_premium,
)
from stock_v2.downstream_probes import (
    build_downstream_targets,
    causal_probe_splits,
    newey_west_mean,
    pearson,
)
from stock_v2.external_factors import (
    POLICY_RATE_FACTORS,
    build_risk_free_period_returns,
    fetch_external_factor_closes,
)
from stock_v2.latent_path_head import (
    blend_latent_path_scores,
    load_latent_path_head,
    sha256_file,
)
from stock_v2.ops.signals import world_model_state_scores
from stock_v2.real_features import build_edge_tensor, make_real_snapshot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a cost-aware economic audit of the frozen latent path head."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latent-path-head", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--policy-horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--liquidity-top-n", type=int, default=300)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--min-price", type=float, default=1000.0)
    parser.add_argument("--max-price", type=float, default=2_000_000.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def prediction_contract(
    checkpoint_path: Path,
    head_path: Path,
    checkpoint: Mapping[str, Any],
    steps: np.ndarray,
    horizons: list[int],
    stock_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "head_sha256": sha256_file(head_path),
        "train_data_manifest_sha256": checkpoint.get("train_data_manifest", {}).get(
            "sha256"
        ),
        "train_edge_manifest_sha256": checkpoint.get("train_edge_manifest", {}).get(
            "sha256"
        ),
        "steps": [int(value) for value in steps],
        "horizons": horizons,
        "stock_count": int(stock_count),
        "dtype": "float32",
    }


def load_or_build_predictions(
    model,
    head,
    features,
    checkpoint: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    steps: np.ndarray,
    horizons: list[int],
    cache_dir: Path,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(str(checkpoint_args["models_dir"])) / "graph_jepa_real.pt"
    contract = prediction_contract(
        checkpoint_path,
        head.checkpoint_path,
        checkpoint,
        steps,
        horizons,
        int(features.tradable_count),
    )
    metadata_path = cache_dir / "metadata.json"
    progress_path = cache_dir / "progress.json"
    paths = {
        "base": cache_dir / "base_paths.npy",
        "latent": cache_dir / "latent_paths.npy",
        "blended": cache_dir / "blended_paths.npy",
        "aggregate": cache_dir / "aggregate.npy",
    }
    shape = (len(steps), len(horizons), int(features.tradable_count))
    aggregate_shape = (len(steps), int(features.tradable_count))
    if metadata_path.exists() and all(path.exists() for path in paths.values()):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        loaded = {name: np.load(path, mmap_mode="r") for name, path in paths.items()}
        if (
            metadata == contract
            and loaded["base"].shape == shape
            and loaded["latent"].shape == shape
            and loaded["blended"].shape == shape
            and loaded["aggregate"].shape == aggregate_shape
        ):
            print(f"loaded latent path prediction cache: {cache_dir}", flush=True)
            return loaded, contract

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else None
    )
    can_resume = (
        isinstance(progress, dict)
        and progress.get("contract") == contract
        and all(path.exists() for path in paths.values())
    )
    if can_resume:
        arrays = {
            name: np.lib.format.open_memmap(path, mode="r+")
            for name, path in paths.items()
        }
        start_position = int(progress.get("completed_dates", 0))
    else:
        arrays = {
            "base": np.lib.format.open_memmap(paths["base"], mode="w+", dtype=np.float32, shape=shape),
            "latent": np.lib.format.open_memmap(paths["latent"], mode="w+", dtype=np.float32, shape=shape),
            "blended": np.lib.format.open_memmap(paths["blended"], mode="w+", dtype=np.float32, shape=shape),
            "aggregate": np.lib.format.open_memmap(
                paths["aggregate"], mode="w+", dtype=np.float32, shape=aggregate_shape
            ),
        }
        for values in arrays.values():
            values[:] = np.nan
        start_position = 0

    edge_settings = _edge_settings(dict(checkpoint_args))
    rollout_args = as_rollout_namespace(dict(checkpoint_args))
    stock_count = int(features.tradable_count)
    model.eval()
    head.model.eval()
    for position in range(start_position, len(steps)):
        step = int(steps[position])
        edge_index, edge_weight = build_edge_tensor(features, step=step, **edge_settings)
        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=int(edge_settings["edge_window"]),
            top_k=int(edge_settings["top_k"]),
            min_abs_corr=float(edge_settings["min_abs_corr"]),
            edge_cache={step: (edge_index, edge_weight)},
        ).to(device)
        forecasts: dict[int, np.ndarray] = {}
        latent_matrix = np.empty((stock_count, len(horizons)), dtype=np.float32)
        with torch.inference_mode():
            context = model.encode_temporal_context(batch)
            for horizon_position, horizon in enumerate(horizons):
                rollout_steps = rollout_steps_for_offset(rollout_args, horizon)
                predicted = model.rollout_latent(context, steps=rollout_steps)
                forecast = model.predict_temporal_state(
                    batch,
                    predicted,
                    rollout_steps=rollout_steps,
                    z_context=context,
                )
                forecasts[horizon] = forecast[:stock_count].float().cpu().numpy()
                latent_matrix[:, horizon_position] = (
                    head.model(
                        context[:stock_count],
                        predicted[:stock_count],
                        horizon,
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
        _state_score, state_diagnostics = world_model_state_scores(
            forecasts,
            list(features.feature_names),
            features.train_mean,
            features.train_std,
        )
        base_matrix = state_diagnostics["predicted_entry_path_returns"]
        aggregate, blend_diagnostics = blend_latent_path_scores(
            base_matrix,
            latent_matrix,
            horizons,
            head.latent_blend_weight,
        )
        arrays["base"][position] = base_matrix.T
        arrays["latent"][position] = latent_matrix.T
        arrays["blended"][position] = blend_diagnostics[
            "blended_entry_path_scores"
        ].T
        arrays["aggregate"][position] = aggregate
        if (position + 1) % 10 == 0 or position + 1 == len(steps):
            for values in arrays.values():
                values.flush()
            temporary = progress_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(
                    {"contract": contract, "completed_dates": int(position + 1)},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(progress_path)
            print(f"latent path predictions: {position + 1}/{len(steps)} dates", flush=True)
    for values in arrays.values():
        values.flush()
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    temporary.replace(metadata_path)
    progress_path.unlink(missing_ok=True)
    return {name: np.load(path, mmap_mode="r") for name, path in paths.items()}, contract


def daily_ic(
    scores: np.ndarray,
    target: np.ndarray,
    eligible: np.ndarray,
    horizon: int,
) -> dict[str, float | int]:
    values = []
    for position in range(len(scores)):
        valid = eligible[position] & np.isfinite(scores[position]) & np.isfinite(target[position])
        values.append(pearson(scores[position, valid], target[position, valid]))
    return newey_west_mean(values, lag=horizon)


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    policy_horizon = int(args.policy_horizon)
    if policy_horizon not in horizons:
        raise ValueError("policy horizon must be configured in the parent model")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    model, checkpoint = load_model(model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    head = load_latent_path_head(
        args.latent_path_head,
        checkpoint_path,
        checkpoint,
        device,
    )
    feature_args = deepcopy(args)
    configured_horizons = checkpoint_args.get("rollout_offsets", horizons)
    feature_args.horizons = (
        configured_horizons
        if isinstance(configured_horizons, str)
        else ",".join(str(int(value)) for value in configured_horizons)
    )
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint,
        evaluator_namespace(feature_args),
    )
    splits = causal_probe_splits(
        features.dates,
        train_end=str(checkpoint_args["train_end"]),
        edge_window=int(checkpoint_args.get("edge_window", 60)),
        max_horizon=max(horizons),
        validation_days=int(args.validation_days),
        test_end=args.test_end,
    )
    test_steps = splits.test_steps.astype(np.int64)
    predictions, prediction_metadata = load_or_build_predictions(
        model,
        head,
        features,
        checkpoint,
        checkpoint_args,
        test_steps,
        horizons,
        Path(args.prediction_cache_dir),
        device,
    )
    stock_count = int(features.tradable_count)
    targets = build_downstream_targets(features, test_steps, policy_horizon)
    target_path = targets.continuous_raw.reshape(
        len(test_steps), stock_count, -1
    )[:, :, 0]
    prices = (
        features.execution_close[test_steps, :stock_count]
        if features.execution_close is not None
        else features.close[test_steps, :stock_count]
    ).astype(np.float64)
    return_index = features.feature_names.index("return_1d")
    liquidity_index = features.feature_names.index("value_ma20_log")
    momentum_index = features.feature_names.index("return_20d")
    observed = features.available_mask[test_steps, :stock_count, return_index] > 0.5
    liquidity = features.raw_features[test_steps, :stock_count, liquidity_index]
    momentum = features.raw_features[test_steps, :stock_count, momentum_index]
    eligible = np.zeros((len(test_steps), stock_count), dtype=bool)
    for position in range(len(test_steps)):
        base = (
            observed[position]
            & np.isfinite(prices[position])
            & (prices[position] >= float(args.min_price))
            & (prices[position] <= float(args.max_price))
        )
        eligible[position] = liquid_universe_mask(
            liquidity[position], base, int(args.liquidity_top_n)
        )

    horizon_position = horizons.index(policy_horizon)
    strategy_scores = {
        "latent_blend_h10": predictions["blended"][:, horizon_position],
        "latent_blend_aggregate": predictions["aggregate"],
        "state_path_h10": predictions["base"][:, horizon_position],
        "latent_head_h10": predictions["latent"][:, horizon_position],
        "momentum_20d": momentum,
    }
    common = eligible.copy()
    for scores in strategy_scores.values():
        common &= np.isfinite(scores)

    factors = fetch_external_factor_closes(
        [POLICY_RATE_FACTORS[0]],
        start=str(features.dates[0].date()),
        end=str(features.dates[-1].date()),
        cache_dir=str(args.external_cache_dir),
        refresh=False,
    )
    risk_free = build_risk_free_period_returns(
        features.dates,
        factors["bok_base_rate"],
        [policy_horizon],
    )[policy_horizon][test_steps]
    dates = [str(features.dates[int(step)].date()) for step in test_steps]
    evaluations: dict[str, dict[str, object]] = {}
    for cost in sorted({float(args.cost_bps), float(args.stress_cost_bps)}):
        key = f"{cost:g}bps"
        evaluations[key] = {
            name: evaluate_ranked_strategy(
                scores,
                target_path,
                common,
                dates,
                features.tickers,
                top_k=int(args.top_k),
                stride=policy_horizon,
                cost_bps=cost,
                risk_free_returns=risk_free,
            )
            for name, scores in strategy_scores.items()
        }
        candidate = evaluations[key]["latent_blend_h10"]
        evaluations[key]["paired_premiums"] = {
            name: paired_strategy_premium(candidate, result)
            for name, result in evaluations[key].items()
            if name not in {"latent_blend_h10", "paired_premiums"}
        }

    output = {
        "status": "complete",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "head": str(head.checkpoint_path),
        "head_sha256": head.checkpoint_sha256,
        "prediction_contract": prediction_metadata,
        "test_start": dates[0],
        "test_end": dates[-1],
        "test_dates": len(dates),
        "stocks": stock_count,
        "policy": {
            "horizon": policy_horizon,
            "top_k": int(args.top_k),
            "liquidity_top_n": int(args.liquidity_top_n),
            "entry": "next_open",
            "exit": f"close_t_plus_{policy_horizon}",
            "rebalance_stride": policy_horizon,
            "risk_free": "BOK_base_rate_ACT_365_effective",
        },
        "daily_ic": {
            name: daily_ic(scores, target_path, common, policy_horizon)
            for name, scores in strategy_scores.items()
        },
        "evaluations": evaluations,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.fold}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
