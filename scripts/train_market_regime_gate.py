from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lightgbm as lgb
import numpy as np
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_auxiliary_trading_policy import (
    _external_state_features,
    _masked_stock_moments,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.auxiliary_policy import liquid_universe_mask
from stock_v2.backtest import performance_metrics
from stock_v2.downstream_probes import (
    build_downstream_targets,
    causal_probe_splits,
    newey_west_mean,
    pearson,
)
from stock_v2.latent_path_head import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal absolute-market-return gate for a ranked strategy."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--strategy-report", required=True)
    parser.add_argument("--strategy-name", default="jepa_auxiliary_risk_adjusted")
    parser.add_argument("--source-cost-bps", type=float, default=30.0)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--liquidity-top-n", type=int, default=300)
    parser.add_argument("--min-price", type=float, default=1_000.0)
    parser.add_argument("--max-price", type=float, default=2_000_000.0)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def market_target(
    features,
    steps: np.ndarray,
    horizon: int,
    liquidity_top_n: int,
    min_price: float,
    max_price: float,
) -> np.ndarray:
    stock_count = int(features.tradable_count)
    targets = build_downstream_targets(features, steps, horizon)
    path = targets.continuous_raw.reshape(len(steps), stock_count, -1)[:, :, 0]
    return_index = features.feature_names.index("return_1d")
    liquidity_index = features.feature_names.index("value_ma20_log")
    prices = (
        features.execution_close[steps, :stock_count]
        if features.execution_close is not None
        else features.close[steps, :stock_count]
    ).astype(np.float64)
    result = np.full(len(steps), np.nan, dtype=np.float64)
    for position, raw_step in enumerate(steps):
        step = int(raw_step)
        observed = features.available_mask[step, :stock_count, return_index] > 0.5
        eligible = (
            observed
            & np.isfinite(prices[position])
            & (prices[position] >= float(min_price))
            & (prices[position] <= float(max_price))
        )
        liquid = liquid_universe_mask(
            features.raw_features[step, :stock_count, liquidity_index],
            eligible,
            liquidity_top_n,
        )
        valid = liquid & np.isfinite(path[position])
        if valid.sum() >= 20:
            result[position] = float(path[position, valid].mean())
    return result


def build_design(features, steps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    stock_values, stock_names = _masked_stock_moments(features, steps)
    external_values, external_names = _external_state_features(features, steps)
    return (
        np.concatenate([stock_values, external_values], axis=1).astype(np.float32),
        np.asarray(stock_names + external_names),
    )


def regression_metrics(prediction: np.ndarray, target: np.ndarray, baseline: float) -> dict[str, float | int]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    model_sse = float(np.square(prediction - target).sum())
    baseline_sse = float(np.square(target - baseline).sum())
    return {
        "rows": int(len(target)),
        "correlation": pearson(prediction, target),
        "mse": model_sse / max(len(target), 1),
        "mse_skill_vs_fit_mean": 1.0 - model_sse / baseline_sse
        if baseline_sse > 1e-12
        else float("nan"),
        "direction_accuracy": float(((prediction > 0.0) == (target > 0.0)).mean()),
    }


def affine_calibration(
    prediction: np.ndarray, target: np.ndarray
) -> tuple[float, float]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    design = np.column_stack(
        [np.ones(int(valid.sum()), dtype=np.float64), prediction[valid]]
    )
    intercept, slope = np.linalg.lstsq(design, target[valid], rcond=None)[0]
    return float(intercept), float(slope)


def promotion_gate(
    validation: dict[str, float | int], evaluations: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "validation_positive_correlation": float(validation["correlation"]) > 0.0,
        "validation_positive_mse_skill": float(validation["mse_skill_vs_fit_mean"]) > 0.0,
        "validation_direction_above_chance": float(validation["direction_accuracy"]) > 0.5,
    }
    for cost_key, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        baselines = evaluation["baselines"]
        active_fraction = float(evaluation["active_fraction"])
        checks[f"{cost_key}_nondegenerate_exposure"] = 0.15 <= active_fraction <= 0.85
        checks[f"{cost_key}_positive_excess_sharpe"] = float(metrics["excess_sharpe"]) > 0.0
        checks[f"{cost_key}_drawdown_within_25pct"] = float(metrics["max_drawdown"]) >= -0.25
        for baseline in ("candidate_always", "equal_weight", "cash"):
            checks[f"{cost_key}_beats_{baseline}"] = float(metrics["total_return"]) > float(
                baselines[baseline]["total_return"]
            )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "decision": "paper_shadow_candidate" if not failures else "research_only",
    }


def evaluate_strategy_gate(
    report: dict[str, Any],
    strategy_name: str,
    predictions_by_date: dict[str, float],
    *,
    source_cost_bps: float,
    cost_bps: float,
    periods_per_year: float,
) -> dict[str, Any]:
    source_key = f"{float(source_cost_bps):g}bps"
    rows = report["evaluations"][source_key][strategy_name]["rows"]
    source_cost = float(source_cost_bps) / 10_000.0
    target_cost = float(cost_bps) / 10_000.0
    gated = []
    always = []
    equal = []
    cash = []
    output_rows = []
    for row in rows:
        date = str(row["date"])
        prediction = float(predictions_by_date[date])
        risk_free = float(row["risk_free_return"])
        gross = float(row["period_return"]) + source_cost
        equal_gross = float(row["benchmark_return"]) + source_cost
        active = prediction > target_cost + risk_free
        period_return = gross - target_cost if active else risk_free
        gated.append(period_return)
        always.append(gross - target_cost)
        equal.append(equal_gross - target_cost)
        cash.append(risk_free)
        output_rows.append(
            {
                "date": date,
                "predicted_market_return": prediction,
                "active": bool(active),
                "period_return": period_return,
                "always_return": always[-1],
                "equal_weight_return": equal[-1],
                "risk_free_return": risk_free,
            }
        )
    gated_array = np.asarray(gated, dtype=np.float64)
    risk_free_array = np.asarray(cash, dtype=np.float64)
    baselines = {
        "candidate_always": np.asarray(always, dtype=np.float64),
        "equal_weight": np.asarray(equal, dtype=np.float64),
        "cash": risk_free_array,
    }
    return {
        "cost_bps": float(cost_bps),
        "periods": len(output_rows),
        "active_periods": int(sum(row["active"] for row in output_rows)),
        "active_fraction": float(np.mean([row["active"] for row in output_rows])),
        "metrics": performance_metrics(gated_array, periods_per_year, risk_free_array),
        "baselines": {
            name: performance_metrics(values, periods_per_year, risk_free_array)
            for name, values in baselines.items()
        },
        "premiums": {
            name: newey_west_mean(gated_array - values, lag=1)
            for name, values in baselines.items()
        },
        "rows": output_rows,
    }


def main() -> None:
    args = parse_args()
    horizon = int(args.horizon)
    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
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
    fit_x, feature_names = build_design(features, splits.fit_steps)
    validation_x, validation_names = build_design(features, splits.validation_steps)
    test_x, test_names = build_design(features, splits.test_steps)
    if not np.array_equal(feature_names, validation_names) or not np.array_equal(feature_names, test_names):
        raise ValueError("market regime feature contracts do not align")
    target_args = (
        horizon,
        int(args.liquidity_top_n),
        float(args.min_price),
        float(args.max_price),
    )
    fit_y = market_target(features, splits.fit_steps, *target_args)
    validation_y = market_target(features, splits.validation_steps, *target_args)
    test_y = market_target(features, splits.test_steps, *target_args)
    fit_valid = np.isfinite(fit_y)
    validation_valid = np.isfinite(validation_y)
    if fit_valid.sum() < 260 or validation_valid.sum() < 60:
        raise ValueError("market regime training split is too short")

    model = lgb.LGBMRegressor(
        objective="regression_l2",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,
        max_depth=-1,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        random_state=17,
        n_jobs=int(args.num_threads),
        verbosity=-1,
    )
    model.fit(
        fit_x[fit_valid],
        100.0 * fit_y[fit_valid],
        eval_set=[(validation_x[validation_valid], 100.0 * validation_y[validation_valid])],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)],
    )
    validation_prediction = model.predict(
        validation_x, num_iteration=model.best_iteration_
    ) / 100.0
    test_prediction_raw = model.predict(test_x, num_iteration=model.best_iteration_) / 100.0
    calibration_intercept, calibration_slope = affine_calibration(
        validation_prediction, validation_y
    )
    validation_prediction_calibrated = (
        calibration_intercept + calibration_slope * validation_prediction
    )
    test_prediction = calibration_intercept + calibration_slope * test_prediction_raw
    fit_mean = float(fit_y[fit_valid].mean())
    report = json.loads(Path(args.strategy_report).read_text(encoding="utf-8"))
    predictions_by_date = {
        str(features.dates[int(step)].date()): float(value)
        for step, value in zip(splits.test_steps, test_prediction)
    }
    evaluations = {
        f"{cost:g}bps": evaluate_strategy_gate(
            report,
            args.strategy_name,
            predictions_by_date,
            source_cost_bps=float(args.source_cost_bps),
            cost_bps=cost,
            periods_per_year=252.0 / float(horizon),
        )
        for cost in sorted({float(args.cost_bps), float(args.stress_cost_bps)})
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{args.fold}_market_regime.txt"
    model.booster_.save_model(artifact_path)
    feature_path = output_dir / f"{args.fold}_market_regime_features.npz"
    np.savez_compressed(
        feature_path,
        schema_version=np.asarray([1], dtype=np.int64),
        feature_names=feature_names,
        horizon=np.asarray([horizon], dtype=np.int64),
        checkpoint_sha256=np.asarray([sha256_file(checkpoint_path)]),
        live_orders_allowed=np.asarray([False]),
    )
    validation_metrics = regression_metrics(
        validation_prediction_calibrated, validation_y, fit_mean
    )
    test_metrics = regression_metrics(test_prediction, test_y, fit_mean)
    gate = promotion_gate(validation_metrics, evaluations)
    output = {
        "status": "complete",
        "approval_scope": gate["decision"],
        "live_orders_allowed": False,
        "fold": args.fold,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "horizon": horizon,
        "feature_count": int(fit_x.shape[1]),
        "best_iteration": int(model.best_iteration_),
        "fit_dates": int(len(splits.fit_steps)),
        "validation_dates": int(len(splits.validation_steps)),
        "test_dates": int(len(splits.test_steps)),
        "calibration": {
            "source": "validation_only",
            "intercept": calibration_intercept,
            "slope": calibration_slope,
        },
        "validation_regression_raw": regression_metrics(
            validation_prediction, validation_y, fit_mean
        ),
        "validation_regression": validation_metrics,
        "test_regression_raw": regression_metrics(test_prediction_raw, test_y, fit_mean),
        "test_regression": test_metrics,
        "evaluations": evaluations,
        "promotion_gate": gate,
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "feature_contract": str(feature_path),
        "feature_contract_sha256": sha256_file(feature_path),
    }
    (output_dir / f"{args.fold}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
