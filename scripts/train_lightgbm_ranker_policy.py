from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.auxiliary_policy import (
    cross_sectional_zscore,
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
from stock_v2.latent_path_head import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and audit a causal LightGBM LambdaRank trading baseline."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--context-cache", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--liquidity-top-n", type=int, default=300)
    parser.add_argument("--top-k-values", default="5,10,20,30")
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--relevance-bins", type=int, default=10)
    parser.add_argument("--feature-workers", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--auxiliary-prediction-cache")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def relevance_labels(values: np.ndarray, bins: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < bins:
        raise ValueError("not enough values for cross-sectional relevance labels")
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.int64)
    ranks[order] = np.arange(len(values), dtype=np.int64)
    return np.minimum((ranks * int(bins)) // len(values), int(bins) - 1).astype(
        np.int32
    )


def current_liquid_masks(
    features,
    steps: Sequence[int],
    liquidity_top_n: int,
) -> np.ndarray:
    stock_count = int(features.tradable_count)
    return_index = features.feature_names.index("return_1d")
    liquidity_index = features.feature_names.index("value_ma20_log")
    result = np.zeros((len(steps), stock_count), dtype=bool)
    for position, raw_step in enumerate(steps):
        step = int(raw_step)
        observed = features.available_mask[step, :stock_count, return_index] > 0.5
        prices = (
            features.execution_close[step, :stock_count]
            if features.execution_close is not None
            else features.close[step, :stock_count]
        )
        eligible = observed & np.isfinite(prices) & (prices >= 1000.0) & (prices <= 2_000_000.0)
        result[position] = liquid_universe_mask(
            features.raw_features[step, :stock_count, liquidity_index],
            eligible,
            int(liquidity_top_n),
        )
    return result


def ranking_rows(
    features,
    steps: np.ndarray,
    step_positions: dict[int, int],
    liquid: np.ndarray,
    horizon: int,
    bins: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    stock_count = int(features.tradable_count)
    targets = build_downstream_targets(features, steps, int(horizon))
    raw_path = targets.continuous_raw.reshape(len(steps), stock_count, -1)[:, :, 0]
    rows = []
    labels = []
    groups = []
    for position, step in enumerate(steps):
        valid = liquid[position] & np.isfinite(raw_path[position])
        indices = np.flatnonzero(valid)
        if len(indices) < max(20, int(bins)):
            continue
        rows.append(step_positions[int(step)] * stock_count + indices)
        labels.append(relevance_labels(raw_path[position, indices], int(bins)))
        groups.append(int(len(indices)))
    if not rows:
        raise ValueError("ranking split has no usable dates")
    return (
        np.concatenate(rows).astype(np.int64),
        np.concatenate(labels).astype(np.int32),
        groups,
    )


def load_auxiliary_risk_scores(
    cache_dir: Path,
    test_steps: np.ndarray,
    stock_count: int,
    eligible: np.ndarray,
    horizon: int,
) -> np.ndarray:
    metadata = json.loads((cache_dir / "metadata.json").read_text(encoding="utf-8"))
    if metadata.get("steps") != [int(value) for value in test_steps]:
        raise ValueError("auxiliary cache steps do not match ranker test steps")
    if int(metadata.get("stock_count", -1)) != int(stock_count):
        raise ValueError("auxiliary cache stock count does not match")
    horizons = [int(value) for value in metadata["horizons"]]
    horizon_position = horizons.index(int(horizon))
    auxiliary = np.load(cache_dir / "auxiliary.npy", mmap_mode="r")
    result = np.full((len(test_steps), stock_count), np.nan, dtype=np.float32)
    for date_index in range(len(test_steps)):
        selected = eligible[date_index]
        mfe = cross_sectional_zscore(auxiliary[date_index, horizon_position, :, 1], selected)
        mae = cross_sectional_zscore(auxiliary[date_index, horizon_position, :, 2], selected)
        volatility = cross_sectional_zscore(
            auxiliary[date_index, horizon_position, :, 3], selected
        )
        valid = np.isfinite(mfe) & np.isfinite(mae) & np.isfinite(volatility)
        result[date_index, valid] = (
            0.25 * mfe[valid] + 0.50 * mae[valid] - 0.50 * volatility[valid]
        )
    return result


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
    return newey_west_mean(values, lag=int(horizon))


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)
    top_k_values = sorted({int(value) for value in args.top_k_values.split(",") if value})
    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = deepcopy(args)
    configured_horizons = checkpoint_args.get("rollout_offsets", [horizon])
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
        max_horizon=horizon,
        validation_days=int(args.validation_days),
        test_end=args.test_end,
    )
    all_steps = np.unique(
        np.concatenate([splits.fit_steps, splits.validation_steps, splits.test_steps])
    ).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    layout = build_context_layout(features, splits.fit_steps)
    context = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        checkpoint,
        checkpoint_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache),
    )
    fit_liquid = current_liquid_masks(features, splits.fit_steps, int(args.liquidity_top_n))
    validation_liquid = current_liquid_masks(
        features, splits.validation_steps, int(args.liquidity_top_n)
    )
    test_liquid = current_liquid_masks(features, splits.test_steps, int(args.liquidity_top_n))
    fit_rows, fit_labels, fit_groups = ranking_rows(
        features,
        splits.fit_steps,
        step_positions,
        fit_liquid,
        horizon,
        int(args.relevance_bins),
    )
    validation_rows, validation_labels, validation_groups = ranking_rows(
        features,
        splits.validation_steps,
        step_positions,
        validation_liquid,
        horizon,
        int(args.relevance_bins),
    )
    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.70,
        reg_alpha=0.10,
        reg_lambda=1.0,
        random_state=17,
        n_jobs=int(args.num_threads),
        verbosity=-1,
    )
    ranker.fit(
        np.asarray(context[fit_rows], dtype=np.float32),
        fit_labels,
        group=fit_groups,
        eval_set=[
            (
                np.asarray(context[validation_rows], dtype=np.float32),
                validation_labels,
            )
        ],
        eval_group=[validation_groups],
        eval_at=[10, 20, 30],
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)],
    )

    stock_count = int(features.tradable_count)
    test_scores = np.full((len(splits.test_steps), stock_count), np.nan, dtype=np.float32)
    for position, step in enumerate(splits.test_steps):
        indices = np.flatnonzero(test_liquid[position])
        rows = step_positions[int(step)] * stock_count + indices
        test_scores[position, indices] = ranker.predict(
            np.asarray(context[rows], dtype=np.float32),
            num_iteration=ranker.best_iteration_,
        ).astype(np.float32)
    strategy_scores = {"lightgbm_ranker": test_scores}
    if args.auxiliary_prediction_cache:
        risk = load_auxiliary_risk_scores(
            Path(args.auxiliary_prediction_cache),
            splits.test_steps,
            stock_count,
            test_liquid,
            horizon,
        )
        adjusted = np.full_like(test_scores, np.nan)
        for position in range(len(test_scores)):
            ranker_z = cross_sectional_zscore(test_scores[position], test_liquid[position])
            valid = np.isfinite(ranker_z) & np.isfinite(risk[position])
            adjusted[position, valid] = ranker_z[valid] + risk[position, valid]
        strategy_scores["lightgbm_plus_jepa_risk"] = adjusted

    targets = build_downstream_targets(features, splits.test_steps, horizon)
    target_path = targets.continuous_raw.reshape(len(splits.test_steps), stock_count, -1)[:, :, 0]
    momentum_index = features.feature_names.index("return_20d")
    strategy_scores["momentum_20d"] = features.raw_features[
        splits.test_steps, :stock_count, momentum_index
    ]
    common = test_liquid.copy()
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
        features.dates, factors["bok_base_rate"], [horizon]
    )[horizon][splits.test_steps]
    dates = [str(features.dates[int(step)].date()) for step in splits.test_steps]
    evaluations: dict[str, Any] = {}
    for top_k in top_k_values:
        top_key = f"top{top_k}"
        evaluations[top_key] = {}
        for cost in sorted({float(args.cost_bps), float(args.stress_cost_bps)}):
            cost_key = f"{cost:g}bps"
            rows = {
                name: evaluate_ranked_strategy(
                    scores,
                    target_path,
                    common,
                    dates,
                    features.tickers,
                    top_k=top_k,
                    stride=horizon,
                    cost_bps=cost,
                    risk_free_returns=risk_free,
                )
                for name, scores in strategy_scores.items()
            }
            candidate_name = (
                "lightgbm_plus_jepa_risk"
                if "lightgbm_plus_jepa_risk" in rows
                else "lightgbm_ranker"
            )
            rows["paired_premiums"] = {
                name: paired_strategy_premium(rows[candidate_name], result)
                for name, result in rows.items()
                if name not in {candidate_name, "paired_premiums"}
            }
            evaluations[top_key][cost_key] = rows

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{args.fold}_lightgbm_ranker.txt"
    ranker.booster_.save_model(artifact_path)
    output = {
        "status": "complete",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "fold": args.fold,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_data_manifest_sha256": checkpoint.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": checkpoint.get("train_edge_manifest", {}).get("sha256"),
        "horizon": horizon,
        "liquidity_top_n": int(args.liquidity_top_n),
        "fit_dates": int(len(splits.fit_steps)),
        "validation_dates": int(len(splits.validation_steps)),
        "test_dates": int(len(splits.test_steps)),
        "best_iteration": int(ranker.best_iteration_),
        "feature_count": int(context.shape[1]),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "daily_ic": {
            name: daily_ic(scores, target_path, common, horizon)
            for name, scores in strategy_scores.items()
        },
        "evaluations": evaluations,
    }
    (output_dir / f"{args.fold}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
