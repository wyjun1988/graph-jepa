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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.train_market_regime_gate import build_design, market_target
from stock_v2.backtest import performance_metrics
from stock_v2.downstream_probes import causal_probe_splits, newey_west_mean
from stock_v2.external_factors import (
    POLICY_RATE_FACTORS,
    build_risk_free_period_returns,
    fetch_external_factor_closes,
)
from stock_v2.latent_path_head import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a causal probability-of-cost-exceedance exposure head."
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
    parser.add_argument("--label-cost-bps", type=float, default=50.0)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--num-threads", type=int, default=16)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def classification_metrics(
    probability: np.ndarray, label: np.ndarray, fit_prior: float
) -> dict[str, float | int]:
    valid = np.isfinite(probability) & np.isfinite(label)
    probability = np.clip(probability[valid], 1e-6, 1.0 - 1e-6)
    label = label[valid].astype(np.int64)
    brier = float(brier_score_loss(label, probability))
    prior_brier = float(np.mean(np.square(label - float(fit_prior))))
    return {
        "rows": int(len(label)),
        "positive_fraction": float(label.mean()),
        "roc_auc": float(roc_auc_score(label, probability)),
        "brier": brier,
        "brier_skill_vs_fit_prior": 1.0 - brier / prior_brier
        if prior_brier > 1e-12
        else float("nan"),
        "accuracy": float(np.mean((probability >= 0.5) == label)),
        "balanced_accuracy": float(
            balanced_accuracy_score(label, probability >= 0.5)
        ),
    }


def platt_calibration(
    probability: np.ndarray, label: np.ndarray
) -> tuple[LogisticRegression, np.ndarray]:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", random_state=17)
    calibrator.fit(logits, label.astype(np.int64))
    return calibrator, calibrator.predict_proba(logits)[:, 1]


def calibrated_probability(
    calibrator: LogisticRegression, probability: np.ndarray
) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def evaluate_probability_gate(
    report: dict[str, Any],
    strategy_name: str,
    probability_by_date: dict[str, float],
    *,
    source_cost_bps: float,
    cost_bps: float,
    periods_per_year: float,
) -> dict[str, Any]:
    source_key = f"{float(source_cost_bps):g}bps"
    rows = report["evaluations"][source_key][strategy_name]["rows"]
    source_cost = float(source_cost_bps) / 10_000.0
    target_cost = float(cost_bps) / 10_000.0
    gated: list[float] = []
    always: list[float] = []
    equal: list[float] = []
    cash: list[float] = []
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        date = str(row["date"])
        probability = float(probability_by_date[date])
        active = probability >= 0.5
        risk_free = float(row["risk_free_return"])
        gross = float(row["period_return"]) + source_cost
        equal_gross = float(row["benchmark_return"]) + source_cost
        period_return = gross - target_cost if active else risk_free
        gated.append(period_return)
        always.append(gross - target_cost)
        equal.append(equal_gross - target_cost)
        cash.append(risk_free)
        output_rows.append(
            {
                "date": date,
                "probability_cost_exceedance": probability,
                "active": bool(active),
                "period_return": period_return,
                "always_return": always[-1],
                "equal_weight_return": equal[-1],
                "risk_free_return": risk_free,
            }
        )
    gated_values = np.asarray(gated, dtype=np.float64)
    risk_free_values = np.asarray(cash, dtype=np.float64)
    baselines = {
        "candidate_always": np.asarray(always, dtype=np.float64),
        "equal_weight": np.asarray(equal, dtype=np.float64),
        "cash": risk_free_values,
    }
    return {
        "cost_bps": float(cost_bps),
        "periods": len(output_rows),
        "active_periods": int(sum(row["active"] for row in output_rows)),
        "active_fraction": float(np.mean([row["active"] for row in output_rows])),
        "metrics": performance_metrics(
            gated_values, periods_per_year, risk_free_values
        ),
        "baselines": {
            name: performance_metrics(values, periods_per_year, risk_free_values)
            for name, values in baselines.items()
        },
        "premiums": {
            name: newey_west_mean(gated_values - values, lag=1)
            for name, values in baselines.items()
        },
        "rows": output_rows,
    }


def promotion_gate(
    validation: dict[str, float | int], evaluations: dict[str, Any]
) -> dict[str, Any]:
    checks = {
        "validation_auc_above_0_52": float(validation["roc_auc"]) > 0.52,
        "validation_positive_brier_skill": float(
            validation["brier_skill_vs_fit_prior"]
        )
        > 0.0,
        "validation_balanced_accuracy_above_chance": float(
            validation["balanced_accuracy"]
        )
        > 0.5,
    }
    for cost_key, evaluation in evaluations.items():
        metrics = evaluation["metrics"]
        baselines = evaluation["baselines"]
        active_fraction = float(evaluation["active_fraction"])
        checks[f"{cost_key}_nondegenerate_exposure"] = 0.15 <= active_fraction <= 0.85
        checks[f"{cost_key}_positive_excess_sharpe"] = float(
            metrics["excess_sharpe"]
        ) > 0.0
        checks[f"{cost_key}_drawdown_within_25pct"] = float(
            metrics["max_drawdown"]
        ) >= -0.25
        for baseline in ("candidate_always", "equal_weight", "cash"):
            checks[f"{cost_key}_beats_{baseline}"] = float(
                metrics["total_return"]
            ) > float(baselines[baseline]["total_return"])
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "decision": "paper_shadow_candidate" if not failures else "research_only",
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
        checkpoint, evaluator_namespace(feature_args)
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
    if not np.array_equal(feature_names, validation_names) or not np.array_equal(
        feature_names, test_names
    ):
        raise ValueError("market regime feature contracts do not align")

    target_args = (
        horizon,
        int(args.liquidity_top_n),
        float(args.min_price),
        float(args.max_price),
    )
    fit_return = market_target(features, splits.fit_steps, *target_args)
    validation_return = market_target(features, splits.validation_steps, *target_args)
    test_return = market_target(features, splits.test_steps, *target_args)
    factors = fetch_external_factor_closes(
        [POLICY_RATE_FACTORS[0]],
        start=str(features.dates[0].date()),
        end=str(features.dates[-1].date()),
        cache_dir=str(args.external_cache_dir),
        refresh=False,
    )
    bok_rate = factors.get("bok_base_rate")
    if bok_rate is None:
        raise RuntimeError("BOK base-rate history is required")
    risk_free = build_risk_free_period_returns(
        features.dates, bok_rate, [horizon]
    )[horizon]
    threshold_return = float(args.label_cost_bps) / 10_000.0
    fit_valid = np.isfinite(fit_return) & np.isfinite(risk_free[splits.fit_steps])
    validation_valid = np.isfinite(validation_return) & np.isfinite(
        risk_free[splits.validation_steps]
    )
    test_valid = np.isfinite(test_return) & np.isfinite(risk_free[splits.test_steps])
    fit_label = (
        fit_return > threshold_return + risk_free[splits.fit_steps]
    ).astype(np.int64)
    validation_label = (
        validation_return
        > threshold_return + risk_free[splits.validation_steps]
    ).astype(np.int64)
    test_label = (
        test_return > threshold_return + risk_free[splits.test_steps]
    ).astype(np.int64)
    if fit_valid.sum() < 260 or validation_valid.sum() < 60:
        raise ValueError("market regime classification split is too short")
    fit_prior = float(fit_label[fit_valid].mean())
    positives = int(fit_label[fit_valid].sum())
    negatives = int(fit_valid.sum()) - positives
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=2000,
        learning_rate=0.02,
        num_leaves=15,
        min_child_samples=30,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5.0,
        scale_pos_weight=float(negatives) / max(float(positives), 1.0),
        random_state=17,
        n_jobs=int(args.num_threads),
        verbosity=-1,
    )
    model.fit(
        fit_x[fit_valid],
        fit_label[fit_valid],
        eval_set=[(validation_x[validation_valid], validation_label[validation_valid])],
        eval_metric="binary_logloss",
        callbacks=[lgb.early_stopping(100, verbose=True), lgb.log_evaluation(50)],
    )
    validation_probability_raw = model.predict_proba(
        validation_x, num_iteration=model.best_iteration_
    )[:, 1]
    test_probability_raw = model.predict_proba(
        test_x, num_iteration=model.best_iteration_
    )[:, 1]
    calibrator, validation_probability = platt_calibration(
        validation_probability_raw[validation_valid], validation_label[validation_valid]
    )
    validation_probability_full = np.full(len(validation_x), np.nan, dtype=np.float64)
    validation_probability_full[validation_valid] = validation_probability
    test_probability = calibrated_probability(calibrator, test_probability_raw)
    report = json.loads(Path(args.strategy_report).read_text(encoding="utf-8"))
    probability_by_date = {
        str(features.dates[int(step)].date()): float(value)
        for step, value in zip(splits.test_steps, test_probability)
    }
    evaluations = {
        f"{cost:g}bps": evaluate_probability_gate(
            report,
            args.strategy_name,
            probability_by_date,
            source_cost_bps=float(args.source_cost_bps),
            cost_bps=cost,
            periods_per_year=252.0 / float(horizon),
        )
        for cost in sorted({float(args.cost_bps), float(args.stress_cost_bps)})
    }
    validation_metrics = classification_metrics(
        validation_probability_full, validation_label, fit_prior
    )
    test_metrics = classification_metrics(test_probability[test_valid], test_label[test_valid], fit_prior)
    gate = promotion_gate(validation_metrics, evaluations)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{args.fold}_market_classifier.txt"
    model.booster_.save_model(model_path)
    contract_path = output_dir / f"{args.fold}_market_classifier_contract.npz"
    np.savez_compressed(
        contract_path,
        schema_version=np.asarray([1], dtype=np.int64),
        feature_names=feature_names,
        horizon=np.asarray([horizon], dtype=np.int64),
        label_cost_bps=np.asarray([float(args.label_cost_bps)], dtype=np.float64),
        calibration_intercept=np.asarray([float(calibrator.intercept_[0])]),
        calibration_coefficient=np.asarray([float(calibrator.coef_[0, 0])]),
        decision_probability=np.asarray([0.5], dtype=np.float64),
        checkpoint_sha256=np.asarray([sha256_file(checkpoint_path)]),
        live_orders_allowed=np.asarray([False]),
    )
    output = {
        "status": "complete",
        "approval_scope": gate["decision"],
        "live_orders_allowed": False,
        "fold": args.fold,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "horizon": horizon,
        "label_cost_bps": float(args.label_cost_bps),
        "feature_count": int(fit_x.shape[1]),
        "best_iteration": int(model.best_iteration_),
        "fit_dates": int(len(splits.fit_steps)),
        "validation_dates": int(len(splits.validation_steps)),
        "test_dates": int(len(splits.test_steps)),
        "fit_positive_fraction": fit_prior,
        "calibration": {
            "source": "validation_only",
            "intercept": float(calibrator.intercept_[0]),
            "coefficient": float(calibrator.coef_[0, 0]),
            "decision_probability": 0.5,
        },
        "validation_classification_raw": classification_metrics(
            validation_probability_raw[validation_valid],
            validation_label[validation_valid],
            fit_prior,
        ),
        "validation_classification": validation_metrics,
        "test_classification_raw": classification_metrics(
            test_probability_raw[test_valid], test_label[test_valid], fit_prior
        ),
        "test_classification": test_metrics,
        "evaluations": evaluations,
        "promotion_gate": gate,
        "artifact": str(model_path),
        "artifact_sha256": sha256_file(model_path),
        "feature_contract": str(contract_path),
        "feature_contract_sha256": sha256_file(contract_path),
    }
    (output_dir / f"{args.fold}.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
