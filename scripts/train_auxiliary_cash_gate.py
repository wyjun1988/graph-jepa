from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import Ridge

from stock_v2.backtest import performance_metrics
from stock_v2.downstream_probes import newey_west_mean, pearson


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a portable absolute-return cash gate on Fold-1 OOS outputs."
    )
    parser.add_argument("--development", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--development-fraction", type=float, default=0.60)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000")
    return parser.parse_args()


def load_dataset(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as source:
        return {name: source[name].copy() for name in source.files}


def validate_contract(development: dict[str, Any], confirmation: dict[str, Any]) -> None:
    for key in ("horizon", "top_k"):
        if int(development[key][0]) != int(confirmation[key][0]):
            raise ValueError(f"cash-gate datasets disagree on {key}")
    for key in ("model_feature_names", "market_feature_names"):
        if not np.array_equal(development[key], confirmation[key]):
            raise ValueError(f"cash-gate datasets disagree on {key}")


def design_matrix(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    names = np.concatenate(
        [dataset["model_feature_names"], dataset["market_feature_names"]]
    )
    values = np.concatenate(
        [dataset["model_features"], dataset["market_features"]], axis=1
    ).astype(np.float64)
    return values, names


def fit_transform_contract(
    values: np.ndarray,
    fit_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    fit = values[fit_indices]
    finite_count = np.isfinite(fit).sum(axis=0)
    usable = finite_count >= max(3, int(np.ceil(len(fit_indices) * 0.50)))
    if not usable.any():
        raise ValueError("cash-gate design has no sufficiently observed features")
    selected = fit[:, usable]
    median = np.nanmedian(selected, axis=0)
    imputed = np.where(np.isfinite(selected), selected, median[None, :])
    mean = imputed.mean(axis=0)
    scale = imputed.std(axis=0)
    nonconstant = np.isfinite(scale) & (scale > 1e-8)
    if not nonconstant.any():
        raise ValueError("cash-gate design has no varying features")
    selected_indices = np.flatnonzero(usable)[nonconstant]
    return {
        "selected_indices": selected_indices.astype(np.int64),
        "median": median[nonconstant].astype(np.float64),
        "mean": mean[nonconstant].astype(np.float64),
        "scale": scale[nonconstant].astype(np.float64),
    }


def transform(values: np.ndarray, contract: dict[str, np.ndarray]) -> np.ndarray:
    selected = values[:, contract["selected_indices"]]
    imputed = np.where(
        np.isfinite(selected),
        selected,
        contract["median"][None, :],
    )
    return ((imputed - contract["mean"][None, :]) / contract["scale"][None, :]).astype(
        np.float64
    )


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    baseline_mean: float,
) -> dict[str, float | int]:
    valid = np.isfinite(prediction) & np.isfinite(target)
    prediction = prediction[valid]
    target = target[valid]
    model_sse = float(np.square(prediction - target).sum())
    baseline_sse = float(np.square(target - float(baseline_mean)).sum())
    return {
        "rows": int(len(target)),
        "correlation": pearson(prediction, target),
        "mse": model_sse / max(len(target), 1),
        "mse_skill_vs_development_mean": (
            1.0 - model_sse / baseline_sse if baseline_sse > 1e-12 else float("nan")
        ),
        "direction_accuracy": float(((prediction > 0.0) == (target > 0.0)).mean())
        if len(target)
        else float("nan"),
    }


def evaluate_gate(
    dataset: dict[str, Any],
    prediction: np.ndarray,
    *,
    indices: np.ndarray,
    horizon: int,
    cost_bps: float,
) -> dict[str, Any]:
    gross = dataset["candidate_gross_return"].astype(np.float64)
    direct = dataset["direct_gross_return"].astype(np.float64)
    equal_weight = dataset["equal_weight_gross_return"].astype(np.float64)
    risk_free = dataset["risk_free_return"].astype(np.float64)
    dates = dataset["dates"].astype(str)
    selected = indices[:: int(horizon)]
    valid = (
        np.isfinite(prediction[selected])
        & np.isfinite(gross[selected])
        & np.isfinite(equal_weight[selected])
        & np.isfinite(risk_free[selected])
    )
    selected = selected[valid]
    cost = float(cost_bps) / 10_000.0
    active = prediction[selected] > risk_free[selected] + cost
    candidate_always = gross[selected] - cost
    gated = np.where(active, candidate_always, risk_free[selected])
    baselines = {
        "candidate_always": candidate_always,
        "equal_weight": equal_weight[selected] - cost,
        "cash": risk_free[selected],
    }
    if np.isfinite(direct[selected]).all():
        baselines["direct_raw_mlp"] = direct[selected] - cost
    metrics = performance_metrics(
        gated,
        periods_per_year=252.0 / float(horizon),
        risk_free_returns=risk_free[selected],
    )
    premiums = {
        name: newey_west_mean(gated - values, lag=1)
        for name, values in baselines.items()
    }
    rows = [
        {
            "date": str(dates[index]),
            "prediction": float(prediction[index]),
            "active": bool(is_active),
            "period_return": float(period_return),
            "candidate_always_return": float(always_return),
            "direct_raw_mlp_return": (
                float(direct_return) if np.isfinite(direct_return) else None
            ),
            "equal_weight_return": float(equal_return),
            "risk_free_return": float(risk_free[index]),
        }
        for index, is_active, period_return, always_return, direct_return, equal_return in zip(
            selected,
            active,
            gated,
            candidate_always,
            direct[selected] - cost,
            baselines["equal_weight"],
        )
    ]
    return {
        "cost_bps": float(cost_bps),
        "periods": int(len(selected)),
        "active_periods": int(active.sum()),
        "active_fraction": float(active.mean()) if len(active) else float("nan"),
        "metrics": metrics,
        "premiums": premiums,
        "baselines": {
            name: performance_metrics(
                values,
                periods_per_year=252.0 / float(horizon),
                risk_free_returns=risk_free[selected],
            )
            for name, values in baselines.items()
        },
        "rows": rows,
    }


def main() -> None:
    args = parse_args()
    if not 0.5 <= float(args.development_fraction) <= 0.8:
        raise ValueError("development fraction must be between 0.5 and 0.8")
    development = load_dataset(args.development)
    confirmation = load_dataset(args.confirmation)
    validate_contract(development, confirmation)
    horizon = int(development["horizon"][0])
    top_k = int(development["top_k"][0])
    dev_values, feature_names = design_matrix(development)
    confirmation_values, _ = design_matrix(confirmation)
    dev_target = development["candidate_gross_return"].astype(np.float64)
    finite_dev = np.flatnonzero(np.isfinite(dev_target))
    split_position = int(np.floor(len(development["dates"]) * float(args.development_fraction)))
    fit_indices = finite_dev[finite_dev + horizon < split_position]
    validation_indices = finite_dev[finite_dev >= split_position]
    if len(fit_indices) < 80 or len(validation_indices) < 40:
        raise ValueError("cash-gate development split is too short")

    contract = fit_transform_contract(dev_values, fit_indices)
    transformed_dev = transform(dev_values, contract)
    baseline_mean = float(dev_target[fit_indices].mean())
    alphas = sorted({float(value) for value in args.alphas.split(",") if value})
    candidates = []
    for alpha in alphas:
        model = Ridge(alpha=alpha, solver="lsqr")
        model.fit(transformed_dev[fit_indices], dev_target[fit_indices])
        validation_prediction = model.predict(transformed_dev[validation_indices])
        metrics = regression_metrics(
            validation_prediction,
            dev_target[validation_indices],
            baseline_mean,
        )
        candidates.append({"alpha": alpha, "metrics": metrics, "model": model})
    selected = min(candidates, key=lambda row: float(row["metrics"]["mse"]))

    all_dev_indices = finite_dev
    final_contract = fit_transform_contract(dev_values, all_dev_indices)
    final_dev = transform(dev_values, final_contract)
    final_confirmation = transform(confirmation_values, final_contract)
    final_model = Ridge(alpha=float(selected["alpha"]), solver="lsqr")
    final_model.fit(final_dev[all_dev_indices], dev_target[all_dev_indices])
    dev_prediction = final_model.predict(final_dev)
    confirmation_prediction = final_model.predict(final_confirmation)
    full_dev_mean = float(dev_target[all_dev_indices].mean())

    costs = sorted({float(args.cost_bps), float(args.stress_cost_bps)})
    validation_evaluations = {
        f"{cost:g}bps": evaluate_gate(
            development,
            dev_prediction,
            indices=validation_indices,
            horizon=horizon,
            cost_bps=cost,
        )
        for cost in costs
    }
    confirmation_indices = np.arange(len(confirmation["dates"]), dtype=np.int64)
    confirmation_evaluations = {
        f"{cost:g}bps": evaluate_gate(
            confirmation,
            confirmation_prediction,
            indices=confirmation_indices,
            horizon=horizon,
            cost_bps=cost,
        )
        for cost in costs
    }
    primary = confirmation_evaluations[f"{float(args.cost_bps):g}bps"]
    stress = confirmation_evaluations[f"{float(args.stress_cost_bps):g}bps"]
    confirmation_regression = regression_metrics(
        confirmation_prediction,
        confirmation["candidate_gross_return"].astype(np.float64),
        full_dev_mean,
    )
    direct_premium = primary["premiums"].get("direct_raw_mlp")
    passed = bool(
        confirmation_regression["correlation"] > 0.0
        and primary["metrics"]["total_return"] > 0.0
        and primary["metrics"]["sharpe"] > 0.0
        and primary["premiums"]["candidate_always"]["mean"] >= 0.0
        and (direct_premium is None or direct_premium["mean"] >= 0.0)
        and stress["metrics"]["total_return"] > 0.0
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "cash_gate_head.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray([1], dtype=np.int64),
        horizon=np.asarray([horizon], dtype=np.int64),
        top_k=np.asarray([top_k], dtype=np.int64),
        alpha=np.asarray([float(selected["alpha"])], dtype=np.float64),
        feature_names=feature_names[final_contract["selected_indices"]],
        selected_indices=final_contract["selected_indices"],
        median=final_contract["median"],
        mean=final_contract["mean"],
        scale=final_contract["scale"],
        coefficient=np.asarray(final_model.coef_, dtype=np.float64),
        intercept=np.asarray([float(final_model.intercept_)], dtype=np.float64),
        development_checkpoint_sha256=development["checkpoint_sha256"],
        confirmation_checkpoint_sha256=confirmation["checkpoint_sha256"],
        live_orders_allowed=np.asarray([False]),
    )
    output = {
        "status": "pass" if passed else "blocked",
        "approval_scope": "read_only_shadow" if passed else "none",
        "live_orders_allowed": False,
        "horizon": horizon,
        "top_k": top_k,
        "feature_count": int(len(final_contract["selected_indices"])),
        "selected_alpha": float(selected["alpha"]),
        "development": {
            "fit_start": str(development["dates"][fit_indices[0]]),
            "fit_end": str(development["dates"][fit_indices[-1]]),
            "validation_start": str(development["dates"][validation_indices[0]]),
            "validation_end": str(development["dates"][validation_indices[-1]]),
            "alpha_candidates": [
                {"alpha": row["alpha"], "metrics": row["metrics"]}
                for row in candidates
            ],
            "validation_evaluations": validation_evaluations,
        },
        "confirmation": {
            "start": str(confirmation["dates"][0]),
            "end": str(confirmation["dates"][-1]),
            "regression": confirmation_regression,
            "evaluations": confirmation_evaluations,
        },
        "artifact": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
