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
from sklearn.metrics import roc_auc_score

from scripts.benchmark_direct_baselines import (
    _edge_settings,
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
)
from scripts.benchmark_frozen_downstream import as_rollout_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.auxiliary_policy import (
    AuxiliaryPolicy,
    combine_auxiliary_predictions,
    evaluate_ranked_strategy,
    liquid_universe_mask,
    paired_strategy_premium,
)
from stock_v2.backtest import performance_metrics
from stock_v2.downstream_probes import (
    CONTINUOUS_TASKS,
    FrozenEncoderProbe,
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
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.latent_path_head import sha256_file
from stock_v2.real_features import build_edge_tensor, make_real_snapshot


def parse_task_weights(values: list[str]) -> dict[str, float]:
    weights = {
        "path_return": 1.0,
        "max_favorable_excursion": 0.25,
        "max_adverse_excursion": 0.50,
        "realized_volatility": -0.50,
    }
    for value in values:
        name, separator, raw_weight = value.partition("=")
        if not separator or name not in DOWNSTREAM_AUXILIARY_TASKS:
            raise ValueError(f"invalid auxiliary task weight: {value}")
        weights[name] = float(raw_weight)
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed jointly trained JEPA specialist policy."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--direct-probe-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prediction-cache-dir", required=True)
    parser.add_argument("--raw-context-cache")
    parser.add_argument("--fold", required=True)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--policy-horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--liquidity-top-n", type=int, default=300)
    parser.add_argument("--task-weight", action="append", default=[])
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--min-price", type=float, default=1000.0)
    parser.add_argument("--max-price", type=float, default=2_000_000.0)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--feature-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-direct-probes", action="store_true")
    parser.add_argument(
        "--save-cash-gate-dataset",
        action="store_true",
        help="Save the daily absolute-return calibration dataset without requiring direct probes.",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def _load_probe(path: Path, parent_sha256: str, device: torch.device) -> FrozenEncoderProbe:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact.get("parent_model_sha256") != parent_sha256:
        raise ValueError(f"direct probe parent mismatch: {path}")
    state = artifact["state_dict"]
    linear_weights = [
        value
        for key, value in state.items()
        if key.startswith("trunk.") and key.endswith(".weight") and value.ndim == 2
    ]
    if not linear_weights:
        raise ValueError(f"cannot infer direct probe architecture: {path}")
    model = FrozenEncoderProbe(
        input_dim=int(artifact["input_dim"]),
        task_count=len(artifact.get("continuous_tasks", CONTINUOUS_TASKS)),
        hidden_dim=int(linear_weights[0].shape[0]),
        layers=len(linear_weights),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_direct_path(
    model: FrozenEncoderProbe,
    raw: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    result = np.empty(len(raw), dtype=np.float32)
    for start in range(0, len(raw), int(batch_size)):
        end = min(start + int(batch_size), len(raw))
        values = torch.from_numpy(
            np.ascontiguousarray(np.asarray(raw[start:end], dtype=np.float32))
        ).to(device)
        with torch.inference_mode():
            continuous, _direction = model(values)
        result[start:end] = continuous[:, 0].float().cpu().numpy()
    return result


def _prediction_contract(
    checkpoint_path: Path,
    checkpoint: Mapping[str, Any],
    steps: np.ndarray,
    horizons: list[int],
    stock_count: int,
    has_market_head: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_data_manifest_sha256": checkpoint.get("train_data_manifest", {}).get(
            "sha256"
        ),
        "train_edge_manifest_sha256": checkpoint.get("train_edge_manifest", {}).get(
            "sha256"
        ),
        "steps": [int(value) for value in steps],
        "horizons": horizons,
        "stock_count": int(stock_count),
        "tasks": list(DOWNSTREAM_AUXILIARY_TASKS),
        "market_outputs": ["return_percent", "cost_exceedance_logit"]
        if has_market_head
        else [],
        "dtype": "float32",
    }


def load_or_build_auxiliary_predictions(
    model,
    features,
    checkpoint: Mapping[str, Any],
    checkpoint_args: Mapping[str, Any],
    steps: np.ndarray,
    horizons: list[int],
    cache_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(str(checkpoint_args["models_dir"])) / "graph_jepa_real.pt"
    has_market_head = bool(model.downstream_market_heads)
    contract = _prediction_contract(
        checkpoint_path,
        checkpoint,
        steps,
        horizons,
        int(features.tradable_count),
        has_market_head,
    )
    metadata_path = cache_dir / "metadata.json"
    progress_path = cache_dir / "progress.json"
    values_path = cache_dir / "auxiliary.npy"
    market_path = cache_dir / "market.npy"
    shape = (
        len(steps),
        len(horizons),
        int(features.tradable_count),
        len(DOWNSTREAM_AUXILIARY_TASKS),
    )
    cache_files_exist = values_path.exists() and (
        market_path.exists() if has_market_head else True
    )
    if metadata_path.exists() and cache_files_exist:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        values = np.load(values_path, mmap_mode="r")
        market_values = (
            np.load(market_path, mmap_mode="r") if has_market_head else None
        )
        market_valid = not has_market_head or (
            market_values is not None
            and market_values.shape == (len(steps), len(horizons), 2)
            and market_values.dtype == np.float32
        )
        if (
            metadata == contract
            and values.shape == shape
            and values.dtype == np.float32
            and market_valid
        ):
            print(f"loaded auxiliary prediction cache: {cache_dir}", flush=True)
            return values, market_values, contract

    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.exists()
        else None
    )
    can_resume = (
        isinstance(progress, dict)
        and progress.get("contract") == contract
        and values_path.exists()
    )
    if can_resume:
        values = np.lib.format.open_memmap(values_path, mode="r+")
        if values.shape != shape or values.dtype != np.float32:
            raise ValueError("partial auxiliary cache has the wrong shape")
        start_position = int(progress.get("completed_dates", 0))
        market_values = (
            np.lib.format.open_memmap(market_path, mode="r+")
            if has_market_head
            else None
        )
    else:
        values = np.lib.format.open_memmap(
            values_path,
            mode="w+",
            dtype=np.float32,
            shape=shape,
        )
        values[:] = np.nan
        market_values = (
            np.lib.format.open_memmap(
                market_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(steps), len(horizons), 2),
            )
            if has_market_head
            else None
        )
        if market_values is not None:
            market_values[:] = np.nan
        start_position = 0

    edge_settings = _edge_settings(dict(checkpoint_args))
    rollout_args = as_rollout_namespace(dict(checkpoint_args))
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
            context = model.encode_temporal_context(batch)
            for horizon_position, horizon in enumerate(horizons):
                rollout_steps = rollout_steps_for_offset(rollout_args, int(horizon))
                predicted = model.rollout_latent(context, steps=rollout_steps)
                auxiliary = model.predict_downstream_auxiliary(
                    context,
                    predicted,
                    rollout_steps,
                )
                values[position, horizon_position] = (
                    auxiliary[: features.tradable_count].float().cpu().numpy()
                )
                if market_values is not None:
                    market = model.predict_downstream_market(
                        context,
                        predicted,
                        batch.supervision_node_mask,
                        batch.graph_index,
                        rollout_steps,
                    )
                    if market.shape != (1, 2):
                        raise ValueError("one-date inference requires one market output")
                    market_values[position, horizon_position] = (
                        market[0].float().cpu().numpy()
                    )
        if (position + 1) % 10 == 0 or position + 1 == len(steps):
            values.flush()
            if market_values is not None:
                market_values.flush()
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
            print(f"auxiliary predictions: {position + 1}/{len(steps)} dates", flush=True)
    values.flush()
    if market_values is not None:
        market_values.flush()
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(metadata_path)
    progress_path.unlink(missing_ok=True)
    return (
        np.load(values_path, mmap_mode="r"),
        np.load(market_path, mmap_mode="r") if has_market_head else None,
        contract,
    )


def daily_ic(
    scores: np.ndarray,
    targets: np.ndarray,
    eligible: np.ndarray,
    horizon: int,
) -> dict[str, float | int]:
    values = []
    for date_index in range(len(scores)):
        valid = (
            eligible[date_index]
            & np.isfinite(scores[date_index])
            & np.isfinite(targets[date_index])
        )
        values.append(pearson(scores[date_index, valid], targets[date_index, valid]))
    return newey_west_mean(values, lag=int(horizon))


def evaluate_market_gated_strategy(
    candidate: Mapping[str, Any],
    score_by_date: Mapping[str, float],
    *,
    threshold: float = 0.5,
    score_field: str = "market_cost_exceedance_probability",
) -> dict[str, Any]:
    rows = []
    period_returns = []
    benchmark_returns = []
    risk_free_returns = []
    for source in candidate["rows"]:
        date = str(source["date"])
        score = float(score_by_date[date])
        active = score >= float(threshold)
        period_return = (
            float(source["period_return"])
            if active
            else float(source["risk_free_return"])
        )
        rows.append(
            {
                **source,
                "selected": list(source["selected"]) if active else [],
                "period_return": period_return,
                score_field: score,
                "active": bool(active),
            }
        )
        period_returns.append(period_return)
        benchmark_returns.append(float(source["benchmark_return"]))
        risk_free_returns.append(float(source["risk_free_return"]))
    values = np.asarray(period_returns, dtype=np.float64)
    benchmark = np.asarray(benchmark_returns, dtype=np.float64)
    risk_free = np.asarray(risk_free_returns, dtype=np.float64)
    active = np.asarray([bool(row["active"]) for row in rows], dtype=bool)
    active_excess_cash = values[active] - risk_free[active]
    active_alpha = values[active] - benchmark[active]
    active_months = sorted({str(row["date"])[:7] for row in rows if row["active"]})
    stride = int(candidate["stride"])
    metrics = performance_metrics(
        values,
        periods_per_year=252.0 / float(stride),
        risk_free_returns=risk_free,
    )
    metrics.update(
        {
            "mean_turnover": float(
                np.mean([row["turnover"] if row["active"] else 0.0 for row in rows])
            )
            if rows
            else float("nan"),
            "worst_period_return": float(values.min()) if len(values) else float("nan"),
            "alpha_vs_equal_weight": newey_west_mean(values - benchmark, lag=1),
        }
    )
    return {
        "cost_bps": float(candidate["cost_bps"]),
        "top_k": int(candidate["top_k"]),
        "stride": stride,
        "active_periods": int(sum(bool(row["active"]) for row in rows)),
        "active_fraction": float(active.mean())
        if rows
        else float("nan"),
        "active_diagnostics": {
            "calendar_months": active_months,
            "calendar_month_count": len(active_months),
            "mean_return": float(values[active].mean())
            if active.any()
            else float("nan"),
            "cash_excess_hit_rate": float((active_excess_cash > 0.0).mean())
            if active.any()
            else float("nan"),
            "cash_excess": newey_west_mean(active_excess_cash, lag=1),
            "alpha_vs_equal_weight": newey_west_mean(active_alpha, lag=1),
        },
        "decision_rule": {
            "score_field": score_field,
            "operator": ">=",
            "threshold": float(threshold),
        },
        "metrics": metrics,
        "period_returns": values.tolist(),
        "rows": rows,
    }


def market_head_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    cost_bps: float,
) -> dict[str, float | int]:
    valid = (
        np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(probability)
    )
    prediction = prediction[valid]
    target = target[valid]
    probability = probability[valid]
    label = target > float(cost_bps) / 10_000.0
    return {
        "rows": int(len(target)),
        "return_correlation": pearson(prediction, target),
        "return_mse": float(np.mean(np.square(prediction - target))),
        "direction_accuracy": float(
            np.mean((prediction > 0.0) == (target > 0.0))
        ),
        "cost_exceedance_fraction": float(label.mean()),
        "cost_exceedance_auc": float(roc_auc_score(label, probability)),
        "cost_exceedance_accuracy": float(
            np.mean((probability >= 0.5) == label)
        ),
    }


def market_head_promotion_gate(
    metrics: Mapping[str, float | int],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "market_return_positive_correlation": float(metrics["return_correlation"]) > 0.0,
        "market_cost_auc_above_0_52": float(metrics["cost_exceedance_auc"]) > 0.52,
    }
    for cost_key, strategies in evaluations.items():
        gated = strategies["jepa_auxiliary_market_gated"]
        candidate = strategies["jepa_auxiliary_risk_adjusted"]
        gated_return = float(gated["metrics"]["total_return"])
        candidate_return = float(candidate["metrics"]["total_return"])
        cash_return = float(candidate["metrics"]["risk_free_total_return"])
        checks[f"{cost_key}_nondegenerate_exposure"] = (
            0.15 <= float(gated["active_fraction"]) <= 0.85
        )
        checks[f"{cost_key}_positive_excess_sharpe"] = float(
            gated["metrics"]["excess_sharpe"]
        ) > 0.0
        checks[f"{cost_key}_beats_ungated_candidate"] = gated_return > candidate_return
        checks[f"{cost_key}_beats_cash"] = gated_return > cash_return
        checks[f"{cost_key}_positive_alpha_vs_equal_weight"] = float(
            gated["metrics"]["alpha_vs_equal_weight"]["mean"]
        ) > 0.0
        checks[f"{cost_key}_drawdown_within_25pct"] = float(
            gated["metrics"]["max_drawdown"]
        ) >= -0.25
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "decision": "fold1_development_pass" if not failures else "research_only",
    }


def market_return_head_promotion_gate(
    metrics: Mapping[str, float | int],
    evaluations: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checks = {
        "market_return_correlation_above_0_05": float(
            metrics["return_correlation"]
        ) > 0.05,
    }
    for cost_key, strategies in evaluations.items():
        gated = strategies["jepa_auxiliary_market_return_gated"]
        candidate = strategies["jepa_auxiliary_risk_adjusted"]
        gated_return = float(gated["metrics"]["total_return"])
        candidate_return = float(candidate["metrics"]["total_return"])
        cash_return = float(candidate["metrics"]["risk_free_total_return"])
        checks[f"{cost_key}_at_least_12_active_periods"] = int(
            gated["active_periods"]
        ) >= 12
        checks[f"{cost_key}_active_fraction_at_most_0_85"] = float(
            gated["active_fraction"]
        ) <= 0.85
        checks[f"{cost_key}_active_in_at_least_4_months"] = int(
            gated["active_diagnostics"]["calendar_month_count"]
        ) >= 4
        checks[f"{cost_key}_positive_active_cash_premium"] = float(
            gated["active_diagnostics"]["cash_excess"]["mean"]
        ) > 0.0
        checks[f"{cost_key}_positive_excess_sharpe"] = float(
            gated["metrics"]["excess_sharpe"]
        ) > 0.0
        checks[f"{cost_key}_beats_ungated_candidate"] = gated_return > candidate_return
        checks[f"{cost_key}_beats_cash"] = gated_return > cash_return
        checks[f"{cost_key}_positive_alpha_vs_equal_weight"] = float(
            gated["metrics"]["alpha_vs_equal_weight"]["mean"]
        ) > 0.0
        checks[f"{cost_key}_drawdown_within_25pct"] = float(
            gated["metrics"]["max_drawdown"]
        ) >= -0.25
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not failures,
        "checks": checks,
        "failures": failures,
        "decision": "fold1_development_pass" if not failures else "research_only",
        "policy": "predicted_market_return_at_least_roundtrip_cost",
    }


def _masked_stock_moments(features, steps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stock_count = int(features.tradable_count)
    raw = features.raw_features[steps, :stock_count].astype(np.float64)
    available = features.available_mask[steps, :stock_count] > 0.5
    count = available.sum(axis=1).astype(np.float64)
    total = np.where(available, raw, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    centered = np.where(available, raw - mean[:, None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        count,
        out=np.zeros_like(total),
        where=count > 0.0,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    availability = count / float(stock_count)
    values = np.concatenate([mean, standard_deviation, availability], axis=1)
    names = []
    for prefix in ("stock_mean", "stock_std", "stock_available"):
        names.extend(f"{prefix}:{name}" for name in features.feature_names)
    return values.astype(np.float32), names


def _external_state_features(features, steps: np.ndarray) -> tuple[np.ndarray, list[str]]:
    stock_count = int(features.tradable_count)
    node_tickers = list(features.node_tickers or [])
    values = []
    names = []
    for node_index in range(stock_count, int(features.node_count)):
        node_id = (
            node_tickers[node_index]
            if node_index < len(node_tickers)
            else f"EXT:{node_index - stock_count}"
        )
        factor = node_id.split(":", 1)[-1]
        prefix = f"ext_{factor}_"
        for feature_index, feature_name in enumerate(features.feature_names):
            if not feature_name.startswith(prefix):
                continue
            observed = (
                features.available_mask[steps, node_index, feature_index] > 0.5
            )
            raw = features.raw_features[steps, node_index, feature_index].astype(
                np.float64
            )
            values.append(np.where(observed & np.isfinite(raw), raw, 0.0))
            values.append(observed.astype(np.float64))
            names.extend(
                [
                    f"external_value:{node_id}:{feature_name}",
                    f"external_available:{node_id}:{feature_name}",
                ]
            )
    if not values:
        return np.empty((len(steps), 0), dtype=np.float32), []
    return np.stack(values, axis=1).astype(np.float32), names


def _auxiliary_distribution_features(
    auxiliary: np.ndarray,
    horizons: list[int],
    eligible: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    statistics = ("q10", "q25", "mean", "std", "q75", "q90")
    result = np.full(
        (
            auxiliary.shape[0],
            len(horizons) * len(DOWNSTREAM_AUXILIARY_TASKS) * len(statistics),
        ),
        np.nan,
        dtype=np.float32,
    )
    names = [
        f"aux_h{horizon}:{task}:{statistic}"
        for horizon in horizons
        for task in DOWNSTREAM_AUXILIARY_TASKS
        for statistic in statistics
    ]
    for date_index in range(auxiliary.shape[0]):
        cursor = 0
        selected = eligible[date_index]
        for horizon_position in range(len(horizons)):
            for task_index in range(len(DOWNSTREAM_AUXILIARY_TASKS)):
                sample = auxiliary[
                    date_index, horizon_position, selected, task_index
                ].astype(np.float64)
                sample = sample[np.isfinite(sample)]
                if len(sample) >= 3:
                    result[date_index, cursor : cursor + len(statistics)] = np.asarray(
                        [
                            np.quantile(sample, 0.10),
                            np.quantile(sample, 0.25),
                            sample.mean(),
                            sample.std(),
                            np.quantile(sample, 0.75),
                            np.quantile(sample, 0.90),
                        ],
                        dtype=np.float32,
                    )
                cursor += len(statistics)
    return result, names


def _realized_portfolio_mean(
    targets: np.ndarray,
    indices: np.ndarray,
    fallback_return: float,
) -> tuple[float, int]:
    selected = np.asarray(targets, dtype=np.float64)[np.asarray(indices, dtype=np.int64)]
    if not len(selected) or not np.isfinite(fallback_return):
        return float("nan"), 0
    missing = ~np.isfinite(selected)
    realized = np.where(missing, float(fallback_return), selected)
    return float(realized.mean()), int(missing.sum())


def _ranked_portfolio_mean(
    scores: np.ndarray,
    targets: np.ndarray,
    eligible: np.ndarray,
    *,
    top_k: int,
    fallback_return: float,
) -> tuple[float, int]:
    candidates = np.flatnonzero(np.asarray(eligible, dtype=bool) & np.isfinite(scores))
    if len(candidates) < int(top_k):
        return float("nan"), 0
    selected = candidates[
        np.argsort(np.asarray(scores)[candidates], kind="stable")[::-1]
    ][: int(top_k)]
    return _realized_portfolio_mean(targets, selected, fallback_return)


def save_cash_gate_dataset(
    output_path: Path,
    *,
    features,
    test_steps: np.ndarray,
    horizons: list[int],
    auxiliary: np.ndarray,
    eligible: np.ndarray,
    candidate_scores: np.ndarray,
    direct_scores: np.ndarray,
    target_path: np.ndarray,
    risk_free: np.ndarray,
    market_predictions: np.ndarray | None,
    policy: AuxiliaryPolicy,
    checkpoint_sha256: str,
) -> None:
    stock_values, stock_names = _masked_stock_moments(features, test_steps)
    external_values, external_names = _external_state_features(features, test_steps)
    auxiliary_values, auxiliary_names = _auxiliary_distribution_features(
        auxiliary,
        horizons,
        eligible,
    )
    market_values = np.concatenate([stock_values, external_values], axis=1)
    market_names = stock_names + external_names
    model_values = auxiliary_values
    model_names = auxiliary_names
    if market_predictions is not None:
        market_head = np.asarray(market_predictions, dtype=np.float32)
        if market_head.shape != (len(test_steps), len(horizons), 2):
            raise ValueError("market-head predictions do not match the cash-gate contract")
        model_values = np.concatenate(
            [model_values, market_head.reshape(len(test_steps), -1)],
            axis=1,
        )
        model_names = model_names + [
            f"market_head_h{horizon}:{name}"
            for horizon in horizons
            for name in ("return_percent", "cost_exceedance_logit")
        ]

    candidate_gross = np.full(len(test_steps), np.nan, dtype=np.float32)
    direct_gross = np.full(len(test_steps), np.nan, dtype=np.float32)
    equal_weight_gross = np.full(len(test_steps), np.nan, dtype=np.float32)
    candidate_missing_targets = np.zeros(len(test_steps), dtype=np.int32)
    direct_missing_targets = np.zeros(len(test_steps), dtype=np.int32)
    equal_weight_missing_targets = np.zeros(len(test_steps), dtype=np.int32)
    for date_index in range(len(test_steps)):
        indices = np.flatnonzero(eligible[date_index])
        if len(indices) < policy.top_k or not np.isfinite(risk_free[date_index]):
            continue
        equal_weight_gross[date_index], equal_weight_missing_targets[date_index] = (
            _realized_portfolio_mean(
                target_path[date_index],
                indices,
                float(risk_free[date_index]),
            )
        )
        for score, destination, missing_destination in (
            (candidate_scores, candidate_gross, candidate_missing_targets),
            (direct_scores, direct_gross, direct_missing_targets),
        ):
            destination[date_index], missing_destination[date_index] = (
                _ranked_portfolio_mean(
                    score[date_index],
                    target_path[date_index],
                    eligible[date_index],
                    top_k=policy.top_k,
                    fallback_return=float(risk_free[date_index]),
                )
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        schema_version=np.asarray([2], dtype=np.int64),
        checkpoint_sha256=np.asarray([checkpoint_sha256]),
        dates=np.asarray(
            [str(features.dates[int(step)].date()) for step in test_steps]
        ),
        horizon=np.asarray([policy.horizon], dtype=np.int64),
        top_k=np.asarray([policy.top_k], dtype=np.int64),
        model_feature_names=np.asarray(model_names),
        model_features=model_values,
        market_feature_names=np.asarray(market_names),
        market_features=market_values,
        candidate_gross_return=candidate_gross,
        direct_gross_return=direct_gross,
        equal_weight_gross_return=equal_weight_gross,
        candidate_missing_target_count=candidate_missing_targets,
        direct_missing_target_count=direct_missing_targets,
        equal_weight_missing_target_count=equal_weight_missing_targets,
        risk_free_return=np.asarray(risk_free, dtype=np.float32),
    )


def main() -> None:
    args = parse_args()
    if not args.skip_direct_probes and (
        not args.direct_probe_dir or not args.raw_context_cache
    ):
        raise ValueError(
            "direct probe evaluation requires --direct-probe-dir and "
            "--raw-context-cache"
        )
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    weights = parse_task_weights(args.task_weight)
    policy = AuxiliaryPolicy(
        horizon=int(args.policy_horizon),
        top_k=int(args.top_k),
        liquidity_top_n=int(args.liquidity_top_n),
        task_weights=weights,
    )
    if policy.horizon not in horizons:
        raise ValueError("policy horizon must be one of the evaluated horizons")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    model, checkpoint = load_model(model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
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
    stock_count = int(features.tradable_count)
    auxiliary, market_predictions, auxiliary_contract = load_or_build_auxiliary_predictions(
        model,
        features,
        checkpoint,
        checkpoint_args,
        test_steps,
        horizons,
        Path(args.prediction_cache_dir),
        device,
    )

    checkpoint_sha = sha256_file(checkpoint_path)
    direct_scores: dict[int, np.ndarray] = {}
    direct_artifacts: dict[str, dict[str, str]] = {}
    if not args.skip_direct_probes:
        layout = build_context_layout(features, splits.fit_steps)
        raw = load_or_build_context_matrix(
            features,
            test_steps,
            layout,
            checkpoint,
            checkpoint_args,
            workers=max(1, int(args.feature_workers)),
            cache_path=Path(args.raw_context_cache),
        )
        for horizon in horizons:
            path = Path(args.direct_probe_dir) / "models" / f"h{horizon}_single_raw.pt"
            probe = _load_probe(path, checkpoint_sha, device)
            direct_scores[horizon] = _predict_direct_path(
                probe,
                raw,
                int(args.batch_size),
                device,
            ).reshape(len(test_steps), stock_count)
            direct_artifacts[str(horizon)] = {
                "path": str(path),
                "sha256": sha256_file(path),
            }

    horizon_position = horizons.index(policy.horizon)
    targets = build_downstream_targets(features, test_steps, policy.horizon)
    target_raw = targets.continuous_raw.reshape(
        len(test_steps), stock_count, len(CONTINUOUS_TASKS)
    )
    target_path = target_raw[:, :, 0]
    current_prices = (
        features.execution_close[test_steps, :stock_count]
        if features.execution_close is not None
        else features.close[test_steps, :stock_count]
    ).astype(np.float64)
    return_index = features.feature_names.index("return_1d")
    liquidity_index = features.feature_names.index("value_ma20_log")
    momentum_index = features.feature_names.index("return_20d")
    observed = features.available_mask[test_steps, :stock_count, return_index] > 0.5
    liquidity = features.raw_features[
        test_steps, :stock_count, liquidity_index
    ].astype(np.float64)
    momentum = features.raw_features[
        test_steps, :stock_count, momentum_index
    ].astype(np.float64)

    candidate = np.full((len(test_steps), stock_count), np.nan, dtype=np.float32)
    path_only = np.full_like(candidate, np.nan)
    liquid = np.zeros((len(test_steps), stock_count), dtype=bool)
    for date_index in range(len(test_steps)):
        base_eligible = (
            observed[date_index]
            & np.isfinite(current_prices[date_index])
            & (current_prices[date_index] >= float(args.min_price))
            & (current_prices[date_index] <= float(args.max_price))
        )
        liquid[date_index] = liquid_universe_mask(
            liquidity[date_index],
            base_eligible,
            policy.liquidity_top_n,
        )
        candidate[date_index], _components = combine_auxiliary_predictions(
            auxiliary[date_index, horizon_position],
            liquid[date_index],
            policy.task_weights,
        )
        path_only[date_index], _path_components = combine_auxiliary_predictions(
            auxiliary[date_index, horizon_position],
            liquid[date_index],
            {"path_return": 1.0},
        )

    direct = direct_scores.get(
        policy.horizon,
        np.full_like(candidate, np.nan, dtype=np.float32),
    )
    common = (
        liquid
        & np.isfinite(candidate)
        & np.isfinite(path_only)
        & np.isfinite(momentum)
    )
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
        features.dates,
        bok_rate,
        [policy.horizon],
    )[policy.horizon][test_steps]
    dates = [str(features.dates[int(step)].date()) for step in test_steps]
    strategy_scores = {
        "jepa_auxiliary_risk_adjusted": candidate,
        "jepa_auxiliary_path_only": path_only,
        "momentum_20d": momentum,
    }
    if not args.skip_direct_probes:
        strategy_scores["direct_raw_mlp"] = direct
    market_summary = None
    market_probability_by_date: dict[str, float] | None = None
    market_return_by_date: dict[str, float] | None = None
    if market_predictions is not None:
        market_output = np.asarray(
            market_predictions[:, horizon_position], dtype=np.float64
        )
        market_return_prediction = market_output[:, 0] / 100.0
        market_probability = 1.0 / (
            1.0 + np.exp(-np.clip(market_output[:, 1], -30.0, 30.0))
        )
        market_target = np.full(len(test_steps), np.nan, dtype=np.float64)
        for date_index in range(len(test_steps)):
            valid = observed[date_index] & np.isfinite(target_path[date_index])
            if valid.sum() >= 20:
                market_target[date_index] = float(
                    target_path[date_index, valid].mean()
                )
        market_cost_bps = float(
            checkpoint_args.get("downstream_market_cost_bps", 50.0)
        )
        market_summary = market_head_metrics(
            market_return_prediction,
            market_target,
            market_probability,
            market_cost_bps,
        )
        market_summary.update(
            {
                "training_cost_bps": market_cost_bps,
                "predicted_return_unit": "decimal",
                "decision_probability": 0.5,
            }
        )
        market_probability_by_date = dict(zip(dates, market_probability.tolist()))
        market_return_by_date = dict(
            zip(dates, market_return_prediction.tolist())
        )
    costs = sorted({float(args.cost_bps), float(args.stress_cost_bps)})
    evaluations: dict[str, dict[str, object]] = {}
    for cost_bps in costs:
        cost_key = f"{cost_bps:g}bps"
        evaluations[cost_key] = {
            name: evaluate_ranked_strategy(
                scores,
                target_path,
                common,
                dates,
                features.tickers,
                top_k=policy.top_k,
                stride=policy.horizon,
                cost_bps=cost_bps,
                risk_free_returns=risk_free,
            )
            for name, scores in strategy_scores.items()
        }
        candidate_result = evaluations[cost_key]["jepa_auxiliary_risk_adjusted"]
        if market_probability_by_date is not None:
            evaluations[cost_key]["jepa_auxiliary_market_gated"] = (
                evaluate_market_gated_strategy(
                    candidate_result,
                    market_probability_by_date,
                )
            )
        if market_return_by_date is not None:
            evaluations[cost_key]["jepa_auxiliary_market_return_gated"] = (
                evaluate_market_gated_strategy(
                    candidate_result,
                    market_return_by_date,
                    threshold=cost_bps / 10_000.0,
                    score_field="market_predicted_return",
                )
            )
        evaluations[cost_key]["paired_premiums"] = {
            baseline: paired_strategy_premium(candidate_result, result)
            for baseline, result in evaluations[cost_key].items()
            if baseline not in {"jepa_auxiliary_risk_adjusted", "paired_premiums"}
        }

    classification_promotion_gate = (
        market_head_promotion_gate(market_summary, evaluations)
        if market_summary is not None
        else None
    )
    return_promotion_gate = (
        market_return_head_promotion_gate(market_summary, evaluations)
        if market_summary is not None
        else None
    )
    promotion_gate = return_promotion_gate
    output = {
        "status": "complete",
        "approval_scope": (
            promotion_gate["decision"]
            if promotion_gate is not None
            else "research_only"
        ),
        "live_orders_allowed": False,
        "fold": args.fold,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "auxiliary_prediction_contract": auxiliary_contract,
        "direct_probe_artifacts": direct_artifacts,
        "test_start": dates[0],
        "test_end": dates[-1],
        "test_dates": len(dates),
        "stocks": stock_count,
        "policy": {
            "horizon": policy.horizon,
            "top_k": policy.top_k,
            "liquidity_top_n": policy.liquidity_top_n,
            "task_weights": dict(policy.task_weights),
            "entry": "next_open",
            "exit": f"close_t_plus_{policy.horizon}",
            "rebalance_stride": policy.horizon,
            "cost_convention": "conservative_full_roundtrip_each_rebalance",
            "risk_free": "BOK_base_rate_ACT_365_effective",
        },
        "daily_ic": {
            name: daily_ic(scores, target_path, common, policy.horizon)
            for name, scores in strategy_scores.items()
        },
        "market_head": market_summary,
        "market_head_promotion_gate": promotion_gate,
        "market_classification_promotion_gate": classification_promotion_gate,
        "market_return_promotion_gate": return_promotion_gate,
        "evaluations": evaluations,
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_cash_gate_dataset or not args.skip_direct_probes:
        save_cash_gate_dataset(
            output_dir / f"{args.fold}_cash_gate_dataset.npz",
            features=features,
            test_steps=test_steps,
            horizons=horizons,
            auxiliary=auxiliary,
            eligible=common,
            candidate_scores=candidate,
            direct_scores=direct,
            target_path=target_path,
            risk_free=risk_free,
            market_predictions=market_predictions,
            policy=policy,
            checkpoint_sha256=checkpoint_sha,
        )
    output_path = output_dir / f"{args.fold}.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
