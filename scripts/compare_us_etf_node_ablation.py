from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean


REQUIRED_COLUMNS = {
    "date",
    "horizon",
    "observed_cells",
    "model_sse",
    "persistence_sse",
    "zero_baseline_sse",
    "mse_skill_vs_persistence",
    "delta_corr",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite values in {column}")
    return values


def pair_daily_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    outcome_rtol: float = 1e-7,
    outcome_atol: float = 1e-6,
) -> pd.DataFrame:
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"{label} daily rollout columns missing: {missing}")
        if frame.duplicated(["date", "horizon"]).any():
            raise ValueError(f"{label} has duplicate date/horizon rows")

    columns = sorted(REQUIRED_COLUMNS.difference({"date", "horizon"}))
    paired = baseline[["date", "horizon", *columns]].merge(
        candidate[["date", "horizon", *columns]],
        on=["date", "horizon"],
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(candidate):
        raise ValueError("baseline and candidate evaluation dates do not match exactly")
    paired = paired.sort_values(["date", "horizon"], kind="stable").reset_index(drop=True)

    baseline_cells = _finite_float(paired, "observed_cells_baseline")
    candidate_cells = _finite_float(paired, "observed_cells_candidate")
    if not np.array_equal(baseline_cells, candidate_cells):
        raise ValueError("stock outcome observed-cell geometry differs")
    if (baseline_cells <= 0).any():
        raise ValueError("observed stock outcome cells must be positive")
    for column in ("persistence_sse", "zero_baseline_sse"):
        left = _finite_float(paired, f"{column}_baseline")
        right = _finite_float(paired, f"{column}_candidate")
        if not np.allclose(left, right, rtol=outcome_rtol, atol=outcome_atol):
            maximum = float(np.max(np.abs(left - right)))
            raise ValueError(f"stock outcome geometry differs for {column}: {maximum}")
    return paired


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights >= 0.0)
    if not valid.any() or float(weights[valid].sum()) <= 0.0:
        return float("nan")
    return float(np.average(values[valid], weights=weights[valid]))


def _relative_improvement(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) <= 1e-12:
        return float("nan")
    return float((baseline - candidate) / baseline)


def compare_horizon(paired: pd.DataFrame, horizon: int) -> dict[str, Any]:
    rows = paired.loc[paired["horizon"].astype(int).eq(int(horizon))].copy()
    if len(rows) < 20:
        raise ValueError(f"too few paired dates for horizon {horizon}")
    cells = _finite_float(rows, "observed_cells_baseline")
    persistence_mse = _finite_float(rows, "persistence_sse_baseline") / cells
    baseline_mse = _finite_float(rows, "model_sse_baseline") / cells
    candidate_mse = _finite_float(rows, "model_sse_candidate") / cells
    delta = candidate_mse - baseline_mse

    ordinary_baseline = float(baseline_mse.mean())
    ordinary_candidate = float(candidate_mse.mean())
    impact_baseline = _weighted_mean(baseline_mse, persistence_mse)
    impact_candidate = _weighted_mean(candidate_mse, persistence_mse)
    normalized_weights = persistence_mse / float(persistence_mse.mean())
    result: dict[str, Any] = {
        "rows": len(rows),
        "transition_energy": {
            "mean": float(persistence_mse.mean()),
            "median": float(np.median(persistence_mse)),
            "p80": float(np.quantile(persistence_mse, 0.8)),
            "p90": float(np.quantile(persistence_mse, 0.9)),
        },
        "ordinary": {
            "baseline_model_mse": ordinary_baseline,
            "candidate_model_mse": ordinary_candidate,
            "relative_mse_improvement": _relative_improvement(
                ordinary_baseline, ordinary_candidate
            ),
            "candidate_minus_baseline_daily_mse": newey_west_mean(
                delta, lag=int(horizon)
            ),
        },
        "impact_weighted": {
            "baseline_model_mse": impact_baseline,
            "candidate_model_mse": impact_candidate,
            "relative_mse_improvement": _relative_improvement(
                impact_baseline, impact_candidate
            ),
            "candidate_minus_baseline_weighted_daily_mse": newey_west_mean(
                delta * normalized_weights, lag=int(horizon)
            ),
        },
    }
    for quantile, label in ((0.8, "top_20_percent"), (0.9, "top_10_percent")):
        threshold = float(np.quantile(persistence_mse, quantile))
        selected = persistence_mse >= threshold
        base = float(baseline_mse[selected].mean())
        candidate_value = float(candidate_mse[selected].mean())
        result[label] = {
            "rows": int(selected.sum()),
            "transition_energy_threshold": threshold,
            "baseline_model_mse": base,
            "candidate_model_mse": candidate_value,
            "relative_mse_improvement": _relative_improvement(
                base, candidate_value
            ),
            "candidate_minus_baseline_daily_mse": newey_west_mean(
                delta[selected], lag=int(horizon)
            ),
        }
    return result


def compare_folds(
    folds: Sequence[tuple[str, pd.DataFrame, pd.DataFrame]],
    horizons: Sequence[int],
) -> dict[str, Any]:
    per_fold: dict[str, Any] = {}
    aggregate_rows: list[pd.DataFrame] = []
    for label, baseline, candidate in folds:
        paired = pair_daily_frames(baseline, candidate)
        paired.insert(0, "fold", str(label))
        aggregate_rows.append(paired)
        per_fold[str(label)] = {
            str(horizon): compare_horizon(paired, int(horizon))
            for horizon in horizons
        }
    aggregate = pd.concat(aggregate_rows, ignore_index=True)
    return {
        "per_fold": per_fold,
        "aggregate": {
            str(horizon): compare_horizon(aggregate, int(horizon))
            for horizon in horizons
        },
    }


def apply_screening_gate(
    result: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    gate = contract["screening_gate_for_five_fold_followup"]
    horizons = [str(value) for value in contract["evaluation"]["horizons"]]
    if "5" not in horizons or "10" not in horizons:
        raise ValueError("screening gate requires horizons 5 and 10")

    def block_metrics(block: dict[str, Any]) -> dict[str, Any]:
        ordinary = np.asarray(
            [
                float(block[h]["ordinary"]["relative_mse_improvement"])
                for h in horizons
            ],
            dtype=np.float64,
        )
        impact = np.asarray(
            [
                float(block[h]["impact_weighted"]["relative_mse_improvement"])
                for h in horizons
            ],
            dtype=np.float64,
        )
        top20 = np.asarray(
            [
                float(block[h]["top_20_percent"]["relative_mse_improvement"])
                for h in horizons
            ],
            dtype=np.float64,
        )
        finite = bool(
            np.isfinite(ordinary).all()
            and np.isfinite(impact).all()
            and np.isfinite(top20).all()
        )
        ordinary_mean = float(np.mean(ordinary))
        impact_mean = float(np.mean(impact))
        h5_h10_mean = float(
            np.mean([impact[horizons.index("5")], impact[horizons.index("10")]])
        )
        checks = {
            "finite_metrics": finite,
            "ordinary_mean_noninferiority": ordinary_mean
            >= -float(gate["ordinary_mean_model_mse_relative_regression_max"]),
            "impact_weighted_mean_positive": impact_mean
            > float(
                gate[
                    "impact_weighted_relative_mse_improvement_mean_min_exclusive"
                ]
            ),
            "impact_weighted_horizon_count": int(np.sum(impact > 0.0))
            >= int(gate["impact_weighted_horizons_improved_min"]),
            "top20_horizon_count": int(np.sum(top20 > 0.0))
            >= int(gate["top_20_percent_horizons_improved_min"]),
            "h5_h10_impact_noninferiority": h5_h10_mean
            >= float(
                gate["h5_h10_impact_weighted_mean_relative_improvement_min"]
            ),
        }
        return {
            "passes": all(checks.values()),
            "checks": checks,
            "ordinary_mean_relative_mse_improvement": ordinary_mean,
            "impact_weighted_mean_relative_mse_improvement": impact_mean,
            "impact_weighted_horizons_improved": int(np.sum(impact > 0.0)),
            "top20_horizons_improved": int(np.sum(top20 > 0.0)),
            "h5_h10_impact_weighted_mean_relative_improvement": h5_h10_mean,
        }

    aggregate_metrics = block_metrics(result["aggregate"])
    per_fold_metrics = {
        str(label): block_metrics(block)
        for label, block in result["per_fold"].items()
    }
    both_folds_present = len(per_fold_metrics) == 2
    require_individual = bool(
        gate.get(
            "both_folds_must_individually_pass_all_numeric_checks",
            gate.get("both_folds_required", False),
        )
    )
    both_folds_individually_pass = both_folds_present and all(
        metrics["passes"] for metrics in per_fold_metrics.values()
    )
    checks = {
        **aggregate_metrics["checks"],
        "both_folds_present": both_folds_present,
        "both_folds_individually_pass": (
            both_folds_individually_pass if require_individual else True
        ),
    }
    return {
        "decision": "advance_to_five_fold" if all(checks.values()) else "do_not_advance",
        "checks": checks,
        "ordinary_mean_relative_mse_improvement": aggregate_metrics[
            "ordinary_mean_relative_mse_improvement"
        ],
        "impact_weighted_mean_relative_mse_improvement": aggregate_metrics[
            "impact_weighted_mean_relative_mse_improvement"
        ],
        "impact_weighted_horizons_improved": aggregate_metrics[
            "impact_weighted_horizons_improved"
        ],
        "top20_horizons_improved": aggregate_metrics[
            "top20_horizons_improved"
        ],
        "h5_h10_impact_weighted_mean_relative_improvement": aggregate_metrics[
            "h5_h10_impact_weighted_mean_relative_improvement"
        ],
        "per_fold": per_fold_metrics,
        "promotion_eligible": False,
        "deployment_eligible": False,
        "live_orders_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and US-ETF node rollouts with systemic-impact weighting."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--fold-label", action="append", required=True)
    parser.add_argument("--baseline-daily", action="append", required=True)
    parser.add_argument("--candidate-daily", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not (
        len(args.fold_label)
        == len(args.baseline_daily)
        == len(args.candidate_daily)
        == 2
    ):
        raise ValueError("exactly two fold labels, baseline files, and candidate files required")

    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "predeclared_us_etf_external_node_ablation":
        raise ValueError("invalid US ETF ablation contract")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("ablation contract does not prohibit live orders")

    inputs = []
    file_audit = []
    fold_contracts = {str(row["label"]): row for row in contract["folds"]}
    for label, baseline_path, candidate_path in zip(
        args.fold_label, args.baseline_daily, args.candidate_daily
    ):
        if label not in fold_contracts:
            raise ValueError(f"unknown fold label: {label}")
        baseline_file = Path(baseline_path)
        candidate_file = Path(candidate_path)
        baseline_sha = file_sha256(baseline_file)
        if baseline_sha != fold_contracts[label]["baseline_future_rollout_csv_sha256"]:
            raise ValueError(f"{label} frozen baseline rollout file changed")
        inputs.append(
            (label, pd.read_csv(baseline_file), pd.read_csv(candidate_file))
        )
        file_audit.append(
            {
                "label": label,
                "baseline_daily": str(baseline_file),
                "baseline_sha256": baseline_sha,
                "candidate_daily": str(candidate_file),
                "candidate_sha256": file_sha256(candidate_file),
            }
        )

    result = compare_folds(inputs, contract["evaluation"]["horizons"])
    result.update(
        {
            "schema_version": 1,
            "role": "us_etf_node_ablation_impact_comparison",
            "contract": str(contract_path),
            "contract_sha256": file_sha256(contract_path),
            "input_files": file_audit,
            "impact_definition": contract["evaluation"]["impact_definition"],
            "screening_gate": apply_screening_gate(result, contract),
            "promotion_eligible": False,
            "deployment_eligible": False,
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["screening_gate"], sort_keys=True))


if __name__ == "__main__":
    main()
