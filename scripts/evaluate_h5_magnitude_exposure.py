from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_magnitude_risk_exposure import (
    _load_json,
    _load_policy_rate,
    _load_qlib,
    _major_labels,
    _paired_metrics,
    _percentile_auc,
    _risk_scores,
    _score_map,
    effective_cash_returns,
    historical_stress_scores,
    paired_block_bootstrap,
    ranked_exposure_backtest,
    sha256,
    validation_threshold,
)


def target_dates_for_horizon(
    target_root: Path, split: str, horizon: int
) -> pd.DataFrame:
    path = target_root / "daily_market_transition_targets.csv"
    frame = pd.read_csv(path)
    selected = frame.loc[
        (frame["split"].astype(str) == split)
        & (pd.to_numeric(frame["horizon"], errors="raise") == int(horizon)),
        ["date", "target_date"],
    ].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected["target_date"] = pd.to_datetime(selected["target_date"])
    if selected.empty or selected["date"].duplicated().any():
        raise ValueError(f"invalid h{horizon} target-date mapping: {path}")
    if not (selected["target_date"] > selected["date"]).all():
        raise ValueError("target dates must be after decision dates")
    return selected.sort_values("date").reset_index(drop=True)


def select_non_overlapping_dates(
    target_dates: pd.DataFrame, stride: int
) -> pd.DataFrame:
    if int(stride) < 1:
        raise ValueError("stride must be positive")
    selected = target_dates.iloc[:: int(stride)].reset_index(drop=True)
    if len(selected) < 20:
        raise ValueError("fewer than twenty non-overlapping periods")
    next_dates = selected["date"].shift(-1)
    overlap = selected["target_date"].iloc[:-1].to_numpy() > next_dates.iloc[:-1].to_numpy()
    if bool(np.any(overlap)):
        raise ValueError("selected holding periods overlap")
    return selected


def _filter_dates(frame: pd.DataFrame, dates: pd.Index, role: str) -> pd.DataFrame:
    result = frame.loc[frame["date"].isin(dates)].copy()
    if set(pd.to_datetime(result["date"].unique())) != set(pd.to_datetime(dates)):
        raise ValueError(f"{role} does not cover selected dates")
    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "h5_nonoverlapping_magnitude_exposure_contract":
        raise ValueError("invalid h5 exposure contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe h5 exposure contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source pin mismatch: {relative}")
    policy = contract["policy"]
    horizon = int(policy["horizon"])
    stride = int(policy["rebalance_stride"])
    if horizon != 5 or stride != 5:
        raise ValueError("this contract requires h5 with stride 5")
    costs = [float(value) for value in policy["turnover_cost_bps"]]
    primary_cost = str(float(policy["primary_turnover_cost_bps"]))
    bok_path = Path(contract["bok_policy_rate_cache"]["path"])
    if sha256(bok_path) != contract["bok_policy_rate_cache"]["sha256"]:
        raise ValueError("BOK policy-rate cache hash mismatch")
    annual_rate = _load_policy_rate(bok_path)

    fold_results = {}
    risk_parts = []
    daily_parts = []
    pooled_strategy: dict[float, dict[str, list[pd.DataFrame]]] = {
        cost: {"baseline": [], "family": [], "direct": [], "historical": []}
        for cost in costs
    }
    bootstrap_frames: dict[float, dict[str, pd.DataFrame]] = {cost: {} for cost in costs}
    input_hashes = {"contract": sha256(contract_path), "bok": sha256(bok_path)}
    seen_dates: set[pd.Timestamp] = set()

    for fold, spec in contract["folds"].items():
        family_root = Path(spec["family_root"])
        direct_root = Path(spec["direct_root"])
        qlib_root = Path(spec["qlib_root"])
        target_root = Path(spec["target_root"])
        family_summary = _load_json(family_root / "summary.json", "family")
        direct_summary = _load_json(direct_root / "summary.json", "direct")
        qlib_summary = _load_json(qlib_root / "summary.json", "Qlib")
        expected_checkpoint = str(spec["checkpoint_sha256"])
        if {
            str(family_summary.get("parent_model_sha256")),
            str(direct_summary.get("parent_model_sha256")),
            str(qlib_summary.get("checkpoint_sha256")),
        } != {expected_checkpoint}:
            raise ValueError(f"checkpoint mismatch: {fold}")
        if direct_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"direct comparator used test selection: {fold}")
        if qlib_summary.get("test_used_for_selection") is not False:
            raise ValueError(f"Qlib used test selection: {fold}")

        family_validation = _risk_scores(family_root, "validation")
        family_test = _risk_scores(family_root, "test")
        direct_validation = _risk_scores(direct_root, "validation")
        direct_test = _risk_scores(direct_root, "test")
        qlib_h5_validation_path = qlib_root / "predictions_h5_valid.parquet"
        qlib_h5_test_path = qlib_root / "predictions_h5_test.parquet"
        qlib_h1_validation_path = qlib_root / "predictions_h1_valid.parquet"
        qlib_h1_test_path = qlib_root / "predictions_h1_test.parquet"
        qlib_h5_validation = _load_qlib(qlib_h5_validation_path)
        qlib_h5_test = _load_qlib(qlib_h5_test_path)
        qlib_h1_validation = _load_qlib(qlib_h1_validation_path)
        qlib_h1_test = _load_qlib(qlib_h1_test_path)
        historical_validation, historical_test = historical_stress_scores(
            qlib_h1_validation,
            qlib_h1_test,
            window=int(policy["historical_stress_window"]),
        )
        full_dates = pd.Index(sorted(pd.to_datetime(qlib_h5_test["date"].unique())))
        for role, frame in {
            "family test": family_test,
            "direct test": direct_test,
            "historical test": historical_test,
        }.items():
            observed = pd.Index(sorted(pd.to_datetime(frame["date"].unique())))
            if not observed.equals(full_dates):
                raise ValueError(f"{role} dates do not align with Qlib h5: {fold}")
        all_targets = target_dates_for_horizon(target_root, "test", horizon)
        if not pd.Index(all_targets["date"]).equals(full_dates):
            raise ValueError(f"h5 target dates do not align: {fold}")
        selected_targets = select_non_overlapping_dates(all_targets, stride)
        selected_dates = pd.Index(selected_targets["date"])
        overlap = seen_dates.intersection(set(selected_dates))
        if overlap:
            raise ValueError(f"selected dates overlap across folds: {fold}")
        seen_dates.update(selected_dates)
        selected_qlib = _filter_dates(qlib_h5_test, selected_dates, "Qlib h5")
        cash = effective_cash_returns(selected_targets, annual_rate)

        quantile = float(policy["validation_risk_quantile"])
        scores = {
            "family": (
                family_test,
                validation_threshold(family_validation["risk_score"], quantile),
            ),
            "direct": (
                direct_test,
                validation_threshold(direct_validation["risk_score"], quantile),
            ),
            "historical": (
                historical_test,
                validation_threshold(historical_validation["risk_score"], quantile),
            ),
        }
        labels = _major_labels(family_root)
        risk = labels.merge(
            family_test.rename(columns={"risk_score": "family"}),
            on="date",
            validate="one_to_one",
        ).merge(
            direct_test.rename(columns={"risk_score": "direct"}),
            on="date",
            validate="one_to_one",
        ).merge(
            historical_test.rename(columns={"risk_score": "historical"}),
            on="date",
            validate="one_to_one",
        )
        risk.insert(0, "fold", fold)
        risk_parts.append(risk)

        fold_payload: dict[str, Any] = {
            "full_test_dates": int(len(full_dates)),
            "non_overlapping_periods": int(len(selected_dates)),
            "first_date": str(selected_dates[0].date()),
            "last_date": str(selected_dates[-1].date()),
            "threshold_source": "validation_only",
            "thresholds": {name: threshold for name, (_frame, threshold) in scores.items()},
            "costs": {},
        }
        for cost in costs:
            baseline = ranked_exposure_backtest(
                selected_qlib,
                selected_targets,
                cash,
                risk_scores=None,
                threshold=None,
                high_risk_exposure=float(policy["high_risk_exposure"]),
                top_k=int(policy["top_k"]),
                liquidity_top_n=int(policy["liquidity_top_n"]),
                cost_bps=cost,
            )
            strategies = {"baseline": baseline}
            for name, (score_frame, threshold) in scores.items():
                selected_score = score_frame.loc[
                    score_frame["date"].isin(selected_dates)
                ].reset_index(drop=True)
                strategies[name] = ranked_exposure_backtest(
                    selected_qlib,
                    selected_targets,
                    cash,
                    risk_scores=_score_map(selected_score, selected_dates, name),
                    threshold=threshold,
                    high_risk_exposure=float(policy["high_risk_exposure"]),
                    top_k=int(policy["top_k"]),
                    liquidity_top_n=int(policy["liquidity_top_n"]),
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
                daily_parts.append(tagged)
            bootstrap_frames[cost][fold] = pd.DataFrame(
                {
                    "baseline": baseline["net_return"],
                    "candidate": strategies["family"]["net_return"],
                }
            )
        fold_results[fold] = fold_payload
        for name, path in {
            f"{fold}.family_summary": family_root / "summary.json",
            f"{fold}.direct_summary": direct_root / "summary.json",
            f"{fold}.qlib_summary": qlib_root / "summary.json",
            f"{fold}.qlib_h5_validation": qlib_h5_validation_path,
            f"{fold}.qlib_h5_test": qlib_h5_test_path,
            f"{fold}.qlib_h1_validation": qlib_h1_validation_path,
            f"{fold}.qlib_h1_test": qlib_h1_test_path,
            f"{fold}.target": target_root / "daily_market_transition_targets.csv",
        }.items():
            input_hashes[name] = sha256(path)

    risk_frame = pd.concat(risk_parts, ignore_index=True)
    pooled = {"risk": {}, "costs": {}}
    for name in ("family", "direct", "historical"):
        pooled["risk"][name] = {
            "fold_rank_auc": _percentile_auc(risk_frame, name)
        }
    for cost in costs:
        baseline = pd.concat(pooled_strategy[cost]["baseline"], ignore_index=True)
        cost_payload = {}
        for name in ("family", "direct", "historical"):
            candidate = pd.concat(pooled_strategy[cost][name], ignore_index=True)
            cost_payload[name] = _paired_metrics(candidate, baseline)
        cost_payload["family_uncertainty"] = paired_block_bootstrap(
            bootstrap_frames[cost],
            samples=int(contract["uncertainty"]["bootstrap_samples"]),
            block_length=int(contract["uncertainty"]["block_length_periods"]),
            seed=int(contract["uncertainty"]["seed"]) + int(round(cost)),
        )
        pooled["costs"][str(cost)] = cost_payload

    primary = pooled["costs"][primary_cost]
    baseline_metrics = primary["family"]["baseline"]
    family_metrics = primary["family"]["candidate"]
    checks = {
        "holding_periods_non_overlapping": sum(
            item["non_overlapping_periods"] for item in fold_results.values()
        )
        == len(seen_dates),
        "baseline_total_return_positive": baseline_metrics["total_return"] > 0.0,
        "baseline_sharpe_at_least": baseline_metrics["sharpe_zero_cash"]
        >= float(contract["checks"]["baseline_sharpe_at_least"]),
        "family_total_return_positive": family_metrics["total_return"] > 0.0,
        "family_sharpe_at_least": family_metrics["sharpe_zero_cash"]
        >= float(contract["checks"]["family_sharpe_at_least"]),
        "family_cvar_improves": primary["family"]["cvar_05_improvement"] > 0.0,
        "family_drawdown_improves": primary["family"]["max_drawdown_improvement"] > 0.0,
        "family_mean_degradation_within": primary["family"]["mean_return_delta"]
        >= -float(contract["checks"]["maximum_period_mean_return_degradation"]),
        "family_cvar_improves_in_required_folds": sum(
            fold_results[fold]["costs"][primary_cost]["family"][
                "cvar_05_improvement"
            ]
            > 0.0
            for fold in fold_results
        )
        >= int(contract["checks"]["cvar_improvement_folds_at_least"]),
        "family_bootstrap_cvar_lower_above_zero": primary[
            "family_uncertainty"
        ]["cvar_05_improvement"]["lower_95"]
        > 0.0,
        "family_cvar_not_below_historical": primary["family"][
            "cvar_05_improvement"
        ]
        >= primary["historical"]["cvar_05_improvement"],
    }
    passed = bool(all(checks.values()))
    payload = {
        "status": "complete",
        "role": "h5_nonoverlapping_magnitude_exposure_audit",
        "folds": fold_results,
        "pooled": pooled,
        "checks": checks,
        "passed": passed,
        "decision": (
            "candidate_requires_forward_only_shadow_confirmation"
            if passed
            else "reject_h5_magnitude_exposure_policy"
        ),
        "inputs": input_hashes,
        "test_used_for_selection": False,
        "evidence_role": "retrospective_post_h1_diagnostic_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    daily = pd.concat(daily_parts, ignore_index=True).sort_values(
        ["cost_bps", "date", "gate"]
    )
    return payload, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate h5 non-overlapping Qlib returns with magnitude exposure gates."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite h5 exposure audit: {output_dir}")
    payload, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_h5_exposure.csv", index=False)
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
            }
        )
    )


if __name__ == "__main__":
    main()
