from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {name}")
    return result


def validation_threshold(values: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 20:
        raise ValueError("fewer than twenty finite validation risk scores")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("risk quantile must be between 0.5 and 1.0")
    return float(np.quantile(array, float(quantile), method="higher"))


def _load_json(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe {role} artifact: {path}")
    return payload


def _risk_scores(root: Path, split: str) -> pd.DataFrame:
    path = root / f"daily_{split}.csv"
    frame = pd.read_csv(path)
    required = {"date", "horizon", "predicted_normalized_salience"}
    if not required.issubset(frame.columns):
        raise ValueError(f"risk score schema mismatch: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["date", "horizon"]).any():
        raise ValueError(f"duplicate risk score keys: {path}")
    score = (
        frame.groupby("date", sort=True)["predicted_normalized_salience"]
        .max()
        .rename("risk_score")
        .reset_index()
    )
    if not np.isfinite(score["risk_score"].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"non-finite risk scores: {path}")
    return score


def _major_labels(family_root: Path) -> pd.DataFrame:
    path = family_root / "major_trajectory" / "daily_major_trajectory.csv"
    frame = pd.read_csv(path)
    required = {"date", "major_trajectory_event"}
    if not required.issubset(frame.columns):
        raise ValueError(f"major-event schema mismatch: {path}")
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError(f"duplicate major-event dates: {path}")
    labels = frame["major_trajectory_event"]
    if labels.dtype != bool:
        labels = labels.astype(str).str.lower().map({"true": True, "false": False})
    if labels.isna().any():
        raise ValueError(f"invalid major-event labels: {path}")
    return pd.DataFrame({"date": frame["date"], "major_event": labels.astype(bool)})


def _load_qlib(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index()
    required = {
        "datetime",
        "instrument",
        "prediction",
        "label",
        "liquidity",
        "current_available",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Qlib prediction schema mismatch: {path}")
    frame = frame.rename(columns={"datetime": "date", "instrument": "ticker"})
    frame["date"] = pd.to_datetime(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str).str.zfill(6)
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError(f"duplicate Qlib keys: {path}")
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def historical_stress_scores(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(window) < 5:
        raise ValueError("historical stress window must be at least five")

    def aggregate(frame: pd.DataFrame, split: str) -> pd.DataFrame:
        rows = []
        for date, group in frame.groupby("date", sort=True):
            valid = (
                group["current_available"].astype(bool).to_numpy()
                & np.isfinite(group["label"].to_numpy(dtype=np.float64))
            )
            values = group.loc[valid, "label"].to_numpy(dtype=np.float64)
            if values.size < 20:
                raise ValueError(f"fewer than twenty historical returns on {date}")
            median = float(np.median(values))
            dispersion = float(np.median(np.abs(values - median)))
            negative_fraction = float(np.mean(values < 0.0))
            rows.append(
                {
                    "date": pd.Timestamp(date),
                    "split": split,
                    "realized_stress": abs(median)
                    + dispersion
                    + 0.02 * abs(negative_fraction - 0.5),
                }
            )
        return pd.DataFrame(rows)

    combined = pd.concat(
        [aggregate(validation, "validation"), aggregate(test, "test")],
        ignore_index=True,
    ).sort_values("date")
    if combined["date"].duplicated().any():
        raise ValueError("validation and test Qlib dates overlap")
    lagged = combined["realized_stress"].shift(1)
    combined["risk_score"] = (
        lagged.rolling(int(window), min_periods=5).mean()
        + lagged.rolling(5, min_periods=5).max()
    )
    validation_scores = combined.loc[
        combined["split"] == "validation", ["date", "risk_score"]
    ].reset_index(drop=True)
    test_scores = combined.loc[
        combined["split"] == "test", ["date", "risk_score"]
    ].reset_index(drop=True)
    if test_scores["risk_score"].isna().any():
        raise ValueError("historical test risk score lacks causal warmup")
    return validation_scores, test_scores


def _target_dates(target_root: Path, split: str) -> pd.DataFrame:
    path = target_root / "daily_market_transition_targets.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["split"].astype(str) == split)
        & (pd.to_numeric(frame["horizon"], errors="raise") == 1),
        ["date", "target_date"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected["target_date"] = pd.to_datetime(selected["target_date"])
    if selected["date"].duplicated().any() or selected.empty:
        raise ValueError(f"invalid target-date mapping: {path} ({split})")
    if not (selected["target_date"] > selected["date"]).all():
        raise ValueError("target dates must be after decision dates")
    return selected.sort_values("date").reset_index(drop=True)


def _load_policy_rate(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["Date"])
    if "Close" not in frame:
        raise ValueError("BOK cache has no Close column")
    rate = pd.to_numeric(frame["Close"], errors="coerce")
    result = pd.Series(rate.to_numpy(dtype=np.float64), index=frame["Date"])
    result.index = pd.DatetimeIndex(result.index).normalize()
    result = result[~result.index.duplicated(keep="last")].sort_index().dropna()
    if result.empty or (result <= -100.0).any():
        raise ValueError("invalid BOK policy-rate cache")
    return result


def effective_cash_returns(mapping: pd.DataFrame, annual_rate: pd.Series) -> pd.Series:
    start = mapping["date"].min().normalize()
    end = mapping["target_date"].max().normalize()
    calendar = pd.date_range(start, end, freq="D")
    effective = (
        annual_rate.reindex(annual_rate.index.union(calendar))
        .sort_index()
        .ffill()
        .reindex(calendar)
    )
    if effective.isna().any():
        raise ValueError("policy-rate history does not cover the backtest")
    daily_log = np.log1p(effective / 100.0) / 365.0
    values = []
    for row in mapping.itertuples(index=False):
        accrued = daily_log.loc[
            (daily_log.index >= pd.Timestamp(row.date).normalize())
            & (daily_log.index < pd.Timestamp(row.target_date).normalize())
        ]
        values.append(float(np.expm1(accrued.sum())))
    return pd.Series(values, index=mapping["date"].to_numpy(), name="cash_return")


def _score_map(scores: pd.DataFrame, expected_dates: pd.Index, name: str) -> dict[pd.Timestamp, float]:
    if scores["date"].duplicated().any():
        raise ValueError(f"duplicate {name} score dates")
    indexed = scores.set_index("date")["risk_score"].reindex(expected_dates)
    if indexed.isna().any() or not np.isfinite(indexed.to_numpy(dtype=np.float64)).all():
        raise ValueError(f"{name} scores do not align with Qlib test dates")
    return {pd.Timestamp(date): float(value) for date, value in indexed.items()}


def ranked_exposure_backtest(
    frame: pd.DataFrame,
    target_dates: pd.DataFrame,
    cash_returns: pd.Series,
    *,
    risk_scores: Mapping[pd.Timestamp, float] | None,
    threshold: float | None,
    high_risk_exposure: float,
    top_k: int,
    liquidity_top_n: int,
    cost_bps: float,
) -> pd.DataFrame:
    if not 0.0 <= float(high_risk_exposure) <= 1.0:
        raise ValueError("high-risk exposure must be in [0, 1]")
    if int(top_k) < 1 or int(liquidity_top_n) < int(top_k):
        raise ValueError("invalid ranking dimensions")
    target_map = target_dates.set_index("date")["target_date"]
    if set(frame["date"].unique()) != set(target_map.index):
        raise ValueError("Qlib and target-audit dates do not align")
    cost_rate = float(cost_bps) / 10_000.0
    previous_weights: dict[str, float] = {}
    previous_cash = 1.0
    rows = []
    for date, group in frame.groupby("date", sort=True):
        valid = (
            group["current_available"].astype(bool).to_numpy()
            & np.isfinite(group["prediction"].to_numpy(dtype=np.float64))
            & np.isfinite(group["label"].to_numpy(dtype=np.float64))
            & np.isfinite(group["liquidity"].to_numpy(dtype=np.float64))
        )
        eligible = group.loc[valid].nlargest(int(liquidity_top_n), "liquidity")
        selected = eligible.nlargest(int(top_k), "prediction")
        if len(selected) != int(top_k):
            raise ValueError(f"insufficient liquid Qlib candidates on {date}")
        risk_score = float("nan")
        high_risk = False
        exposure = 1.0
        if risk_scores is not None:
            risk_score = finite_float(risk_scores[pd.Timestamp(date)], "risk score")
            high_risk = risk_score >= finite_float(threshold, "risk threshold")
            exposure = float(high_risk_exposure) if high_risk else 1.0
        weight = exposure / float(top_k)
        weights = {str(ticker): weight for ticker in selected["ticker"]}
        cash_weight = 1.0 - exposure
        assets = set(previous_weights) | set(weights)
        turnover = 0.5 * (
            sum(abs(weights.get(item, 0.0) - previous_weights.get(item, 0.0)) for item in assets)
            + abs(cash_weight - previous_cash)
        )
        selected_returns = {
            str(row.ticker): float(row.label) for row in selected.itertuples()
        }
        selected_return = float(selected["label"].mean())
        cash_return = finite_float(cash_returns.loc[pd.Timestamp(date)], "cash return")
        gross_return = exposure * selected_return + cash_weight * cash_return
        trading_cost = turnover * cost_rate
        rows.append(
            {
                "date": pd.Timestamp(date),
                "target_date": pd.Timestamp(target_map.loc[pd.Timestamp(date)]),
                "risk_score": risk_score,
                "high_risk": bool(high_risk),
                "exposure": exposure,
                "selected_return": selected_return,
                "cash_return": cash_return,
                "turnover": float(turnover),
                "trading_cost": float(trading_cost),
                "net_return": float(gross_return - trading_cost),
                "selected": ",".join(weights),
            }
        )
        end_asset_values = {
            ticker: value * (1.0 + selected_returns[ticker])
            for ticker, value in weights.items()
        }
        end_cash_value = cash_weight * (1.0 + cash_return)
        end_value = sum(end_asset_values.values()) + end_cash_value
        if not math.isfinite(end_value) or end_value <= 0.0:
            raise ValueError(f"non-positive portfolio value on {date}")
        previous_weights = {
            ticker: value / end_value for ticker, value in end_asset_values.items()
        }
        previous_cash = end_cash_value / end_value
    return pd.DataFrame(rows)


def strategy_metrics(returns: Sequence[float]) -> dict[str, float | int]:
    values = np.asarray(returns, dtype=np.float64)
    if values.size < 20 or not np.isfinite(values).all():
        raise ValueError("strategy metrics require at least twenty finite returns")
    equity = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0
    tail_count = max(1, int(math.ceil(0.05 * len(values))))
    volatility = float(np.std(values, ddof=1))
    return {
        "periods": int(len(values)),
        "total_return": float(equity[-1] - 1.0),
        "mean_return": float(np.mean(values)),
        "annualized_volatility": volatility * math.sqrt(252.0),
        "sharpe_zero_cash": float(np.mean(values) / volatility * math.sqrt(252.0))
        if volatility > 0.0
        else 0.0,
        "max_drawdown": float(np.min(drawdown)),
        "cvar_05": float(np.mean(np.sort(values)[:tail_count])),
        "worst_return": float(np.min(values)),
        "hit_rate": float(np.mean(values > 0.0)),
    }


def _paired_metrics(candidate: pd.DataFrame, baseline: pd.DataFrame) -> dict[str, Any]:
    merged = candidate[["date", "net_return"]].merge(
        baseline[["date", "net_return"]],
        on="date",
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if len(merged) != len(candidate) or len(merged) != len(baseline):
        raise ValueError("paired strategy dates do not align")
    candidate_metrics = strategy_metrics(merged["net_return_candidate"])
    baseline_metrics = strategy_metrics(merged["net_return_baseline"])
    return {
        "candidate": candidate_metrics,
        "baseline": baseline_metrics,
        "mean_return_delta": float(
            merged["net_return_candidate"].mean()
            - merged["net_return_baseline"].mean()
        ),
        "cvar_05_improvement": float(
            candidate_metrics["cvar_05"] - baseline_metrics["cvar_05"]
        ),
        "max_drawdown_improvement": float(
            candidate_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
        ),
        "volatility_reduction": float(
            baseline_metrics["annualized_volatility"]
            - candidate_metrics["annualized_volatility"]
        ),
    }


def _circular_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(math.ceil(length / float(block_length)))
    starts = rng.integers(0, length, size=blocks)
    return np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )[:length]


def paired_block_bootstrap(
    fold_frames: Mapping[str, pd.DataFrame],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    cvar = []
    drawdown = []
    mean_delta = []
    for _ in range(int(samples)):
        baseline_parts = []
        candidate_parts = []
        for frame in fold_frames.values():
            indices = _circular_indices(len(frame), int(block_length), rng)
            baseline_parts.append(frame["baseline"].to_numpy(dtype=np.float64)[indices])
            candidate_parts.append(frame["candidate"].to_numpy(dtype=np.float64)[indices])
        baseline = np.concatenate(baseline_parts)
        candidate = np.concatenate(candidate_parts)
        baseline_metrics = strategy_metrics(baseline)
        candidate_metrics = strategy_metrics(candidate)
        cvar.append(candidate_metrics["cvar_05"] - baseline_metrics["cvar_05"])
        drawdown.append(
            candidate_metrics["max_drawdown"] - baseline_metrics["max_drawdown"]
        )
        mean_delta.append(float(np.mean(candidate - baseline)))

    def interval(values: Sequence[float]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        return {
            "lower_95": float(np.quantile(array, 0.025)),
            "median": float(np.quantile(array, 0.5)),
            "upper_95": float(np.quantile(array, 0.975)),
        }

    return {
        "method": "within_fold_circular_moving_block_bootstrap",
        "samples": int(samples),
        "block_length_days": int(block_length),
        "seed": int(seed),
        "cvar_05_improvement": interval(cvar),
        "max_drawdown_improvement": interval(drawdown),
        "mean_return_delta": interval(mean_delta),
    }


def _percentile_auc(frame: pd.DataFrame, score_name: str) -> float:
    ranked = frame.groupby("fold", sort=False)[score_name].rank(pct=True)
    labels = frame["major_event"].to_numpy(dtype=bool)
    if np.unique(labels).size != 2:
        raise ValueError("pooled major-event labels contain one class")
    return float(roc_auc_score(labels, ranked.to_numpy(dtype=np.float64)))


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "magnitude_risk_exposure_reduction_contract":
        raise ValueError("invalid exposure contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe exposure contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source pin mismatch: {relative}")
    policy = contract["policy"]
    quantile = float(policy["validation_risk_quantile"])
    high_exposure = float(policy["high_risk_exposure"])
    top_k = int(policy["top_k"])
    liquidity_top_n = int(policy["liquidity_top_n"])
    costs = [float(value) for value in policy["turnover_cost_bps"]]
    bok_path = Path(contract["bok_policy_rate_cache"]["path"])
    if sha256(bok_path) != contract["bok_policy_rate_cache"]["sha256"]:
        raise ValueError("BOK policy-rate cache hash mismatch")
    annual_rate = _load_policy_rate(bok_path)

    results: dict[str, Any] = {}
    pooled_risk = []
    pooled_strategy: dict[float, dict[str, list[pd.DataFrame]]] = {
        cost: {"baseline": [], "family": [], "direct": [], "historical": []}
        for cost in costs
    }
    bootstrap_frames: dict[float, dict[str, pd.DataFrame]] = {cost: {} for cost in costs}
    input_hashes: dict[str, str] = {"contract": sha256(contract_path), "bok": sha256(bok_path)}
    seen_test_dates: set[pd.Timestamp] = set()

    for fold, spec in contract["folds"].items():
        family_root = Path(spec["family_root"])
        direct_root = Path(spec["direct_root"])
        qlib_root = Path(spec["qlib_root"])
        target_root = Path(spec["target_root"])
        family_summary_path = family_root / "summary.json"
        direct_summary_path = direct_root / "summary.json"
        qlib_summary_path = qlib_root / "summary.json"
        family_summary = _load_json(family_summary_path, "family")
        direct_summary = _load_json(direct_summary_path, "direct")
        qlib_summary = _load_json(qlib_summary_path, "Qlib")
        checkpoint = str(spec["checkpoint_sha256"])
        observed = {
            str(family_summary.get("parent_model_sha256")),
            str(direct_summary.get("parent_model_sha256")),
            str(qlib_summary.get("checkpoint_sha256")),
        }
        if observed != {checkpoint}:
            raise ValueError(f"checkpoint mismatch for {fold}: {sorted(observed)}")
        if direct_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"direct comparator used test selection: {fold}")
        if qlib_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"Qlib used test selection: {fold}")
        direct_manifest = _load_json(direct_root / "run_manifest.json", "direct manifest")
        if direct_manifest.get("role") != "direct_current_state_magnitude_risk_comparator_fold_manifest":
            raise ValueError(f"invalid direct manifest role: {fold}")

        family_validation = _risk_scores(family_root, "validation")
        family_test = _risk_scores(family_root, "test")
        direct_validation = _risk_scores(direct_root, "validation")
        direct_test = _risk_scores(direct_root, "test")
        qlib_validation_path = qlib_root / "predictions_h1_valid.parquet"
        qlib_test_path = qlib_root / "predictions_h1_test.parquet"
        qlib_validation = _load_qlib(qlib_validation_path)
        qlib_test = _load_qlib(qlib_test_path)
        validation_dates = pd.Index(
            sorted(pd.to_datetime(qlib_validation["date"].unique()))
        )
        test_dates = pd.Index(sorted(pd.to_datetime(qlib_test["date"].unique())))
        for name, frame, expected_dates in (
            ("family validation", family_validation, validation_dates),
            ("direct validation", direct_validation, validation_dates),
            ("family test", family_test, test_dates),
            ("direct test", direct_test, test_dates),
        ):
            observed_dates = pd.Index(sorted(pd.to_datetime(frame["date"].unique())))
            if not observed_dates.equals(expected_dates):
                raise ValueError(f"{name} dates do not align with Qlib")
        historical_validation, historical_test = historical_stress_scores(
            qlib_validation,
            qlib_test,
            window=int(policy["historical_stress_window"]),
        )
        overlap = seen_test_dates.intersection(set(test_dates))
        if overlap:
            raise ValueError(f"test dates overlap across folds: {fold}")
        seen_test_dates.update(test_dates)
        targets = _target_dates(target_root, "test")
        cash = effective_cash_returns(targets, annual_rate)

        scores = {
            "family": (
                family_validation,
                family_test,
                validation_threshold(family_validation["risk_score"], quantile),
            ),
            "direct": (
                direct_validation,
                direct_test,
                validation_threshold(direct_validation["risk_score"], quantile),
            ),
            "historical": (
                historical_validation,
                historical_test,
                validation_threshold(historical_validation["risk_score"], quantile),
            ),
        }
        labels = _major_labels(family_root)
        risk = labels.copy()
        for name, (_validation, test_score, threshold) in scores.items():
            risk = risk.merge(
                test_score.rename(columns={"risk_score": name}),
                on="date",
                validate="one_to_one",
            )
            risk[f"{name}_high_risk"] = risk[name] >= threshold
        if len(risk) != len(test_dates):
            raise ValueError(f"risk labels do not align with Qlib dates: {fold}")
        risk.insert(0, "fold", fold)
        pooled_risk.append(risk)

        fold_payload: dict[str, Any] = {
            "dates": int(len(test_dates)),
            "threshold_source": "validation_only",
            "thresholds": {name: values[2] for name, values in scores.items()},
            "test_high_risk_fraction": {
                name: float(risk[f"{name}_high_risk"].mean()) for name in scores
            },
            "costs": {},
        }
        for cost in costs:
            baseline = ranked_exposure_backtest(
                qlib_test,
                targets,
                cash,
                risk_scores=None,
                threshold=None,
                high_risk_exposure=high_exposure,
                top_k=top_k,
                liquidity_top_n=liquidity_top_n,
                cost_bps=cost,
            )
            strategies = {"baseline": baseline}
            for name, (_validation, test_score, threshold) in scores.items():
                strategies[name] = ranked_exposure_backtest(
                    qlib_test,
                    targets,
                    cash,
                    risk_scores=_score_map(test_score, test_dates, name),
                    threshold=threshold,
                    high_risk_exposure=high_exposure,
                    top_k=top_k,
                    liquidity_top_n=liquidity_top_n,
                    cost_bps=cost,
                )
            fold_payload["costs"][str(cost)] = {
                name: _paired_metrics(frame, baseline)
                for name, frame in strategies.items()
                if name != "baseline"
            }
            for name, frame in strategies.items():
                tagged = frame.copy()
                tagged.insert(0, "fold", fold)
                tagged.insert(1, "gate", name)
                tagged.insert(2, "cost_bps", cost)
                pooled_strategy[cost][name].append(tagged)
            bootstrap_frames[cost][fold] = pd.DataFrame(
                {
                    "baseline": baseline["net_return"],
                    "candidate": strategies["family"]["net_return"],
                }
            )
        results[fold] = fold_payload

        for name, path in {
            f"{fold}.family_summary": family_summary_path,
            f"{fold}.direct_summary": direct_summary_path,
            f"{fold}.direct_manifest": direct_root / "run_manifest.json",
            f"{fold}.qlib_summary": qlib_summary_path,
            f"{fold}.qlib_validation": qlib_validation_path,
            f"{fold}.qlib_test": qlib_test_path,
            f"{fold}.target": target_root / "daily_market_transition_targets.csv",
        }.items():
            input_hashes[name] = sha256(path)

    risk_frame = pd.concat(pooled_risk, ignore_index=True).sort_values(["date", "fold"])
    risk_metrics = {
        name: {
            "pooled_fold_rank_auc": _percentile_auc(risk_frame, name),
            "high_risk_fraction": float(risk_frame[f"{name}_high_risk"].mean()),
        }
        for name in ("family", "direct", "historical")
    }
    pooled_payload = {"risk": risk_metrics, "costs": {}}
    daily_parts = []
    for cost in costs:
        baseline = pd.concat(pooled_strategy[cost]["baseline"], ignore_index=True)
        cost_payload = {}
        for name in ("family", "direct", "historical"):
            candidate = pd.concat(pooled_strategy[cost][name], ignore_index=True)
            cost_payload[name] = _paired_metrics(candidate, baseline)
        cost_payload["family_uncertainty"] = paired_block_bootstrap(
            bootstrap_frames[cost],
            samples=int(contract["uncertainty"]["bootstrap_samples"]),
            block_length=int(contract["uncertainty"]["block_length_days"]),
            seed=int(contract["uncertainty"]["seed"]) + int(round(cost)),
        )
        pooled_payload["costs"][str(cost)] = cost_payload
        for name, frames in pooled_strategy[cost].items():
            daily_parts.append(pd.concat(frames, ignore_index=True))

    primary_cost = str(float(policy["primary_turnover_cost_bps"]))
    primary = pooled_payload["costs"][primary_cost]
    checks = {
        "all_test_dates_non_overlapping": len(seen_test_dates) == len(risk_frame),
        "family_pooled_rank_auc_at_least": risk_metrics["family"]["pooled_fold_rank_auc"]
        >= float(contract["checks"]["family_pooled_rank_auc_at_least"]),
        "family_auc_beats_historical_by": risk_metrics["family"]["pooled_fold_rank_auc"]
        - risk_metrics["historical"]["pooled_fold_rank_auc"]
        >= float(contract["checks"]["family_auc_beats_historical_by"]),
        "family_auc_not_below_direct_by_more_than": risk_metrics["family"]["pooled_fold_rank_auc"]
        - risk_metrics["direct"]["pooled_fold_rank_auc"]
        >= -float(contract["checks"]["family_auc_noninferiority_margin_vs_direct"]),
        "family_primary_cvar_improves": primary["family"]["cvar_05_improvement"] > 0.0,
        "family_primary_drawdown_improves": primary["family"]["max_drawdown_improvement"] > 0.0,
        "family_primary_volatility_reduces": primary["family"]["volatility_reduction"] > 0.0,
        "family_primary_mean_degradation_within": primary["family"]["mean_return_delta"]
        >= -float(contract["checks"]["maximum_daily_mean_return_degradation"]),
        "family_cvar_improves_in_required_folds": sum(
            results[fold]["costs"][primary_cost]["family"]["cvar_05_improvement"] > 0.0
            for fold in results
        )
        >= int(contract["checks"]["cvar_improvement_folds_at_least"]),
        "family_cvar_improvement_not_below_direct": primary["family"]["cvar_05_improvement"]
        >= primary["direct"]["cvar_05_improvement"],
    }
    absolute_names = {
        "family_pooled_rank_auc_at_least",
        "family_auc_beats_historical_by",
        "family_primary_cvar_improves",
        "family_primary_drawdown_improves",
        "family_primary_volatility_reduces",
        "family_primary_mean_degradation_within",
        "family_cvar_improves_in_required_folds",
    }
    specific_names = {
        "family_auc_not_below_direct_by_more_than",
        "family_cvar_improvement_not_below_direct",
    }
    absolute_passed = all(checks[name] for name in absolute_names)
    specific_passed = all(checks[name] for name in specific_names)
    if absolute_passed and specific_passed:
        decision = "risk_gate_candidate_requires_m1max_read_only_safety_validation"
    elif absolute_passed:
        decision = "risk_reduction_supported_but_graph_jepa_not_incremental"
    else:
        decision = "reject_family_query_exposure_gate"
    payload = {
        "status": "complete",
        "role": "magnitude_risk_exposure_reduction_audit",
        "contract_sha256": sha256(contract_path),
        "folds": results,
        "pooled": pooled_payload,
        "checks": checks,
        "absolute_risk_reduction_passed": absolute_passed,
        "graph_jepa_specific_utility_passed": specific_passed,
        "decision": decision,
        "inputs": input_hashes,
        "test_used_for_selection": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["cost_bps", "date", "gate"]
    )
    return payload, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate validation-calibrated magnitude scores as Qlib exposure reducers."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite exposure audit: {output_dir}")
    payload, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_exposure_strategies.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(
        json.dumps(
            {
                "decision": payload["decision"],
                "checks": payload["checks"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
