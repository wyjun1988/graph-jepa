from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HORIZONS = ("15m", "30m", "60m")
BUCKETS = ("open_0900_0929", "morning_0930_1059")
SUBSET = "adaptive_observed_surprise_recent_30m"
METRICS = ("pearson", "skill_vs_zero_mse")
REPORT_ROLE = "post_impact_adaptive_event_sufficient_statistics_diagnostic"
REPORT_CONTRACT = "regression_daily_sufficient_statistics_v1"
COMPARISONS = (
    "raw_aligned_vs_raw_baseline",
    "affine_aligned_vs_raw_baseline",
    "affine_aligned_vs_affine_baseline",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def resolve(value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("role") != (
        "post_impact_rank_adapter_calibration_diagnostic_contract"
    ):
        raise ValueError("invalid calibration diagnostic contract")
    if contract.get("test_split_evaluation_allowed") is not False:
        raise ValueError("calibration diagnostic permits test evaluation")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("calibration diagnostic permits live orders")
    if contract.get("promotion_eligible") is not False:
        raise ValueError("calibration diagnostic permits promotion")
    if tuple(contract.get("primary_horizons") or ()) != HORIZONS:
        raise ValueError("calibration diagnostic horizons changed")
    if tuple(contract.get("clock_buckets") or ()) != BUCKETS:
        raise ValueError("calibration diagnostic buckets changed")
    if contract.get("event_subset") != SUBSET:
        raise ValueError("calibration diagnostic subset changed")
    bounds = contract.get("affine_slope_bounds")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("calibration diagnostic slope bounds are invalid")
    if not 0.0 < float(bounds[0]) < float(bounds[1]):
        raise ValueError("calibration diagnostic slope bounds are not positive")
    for relative, expected in contract.get("source_pins", {}).items():
        path_value = resolve(relative)
        if not path_value.is_file() or sha256_file(path_value) != str(expected):
            raise ValueError(f"calibration source pin changed: {relative}")
    if len(contract.get("draws") or ()) != 4:
        raise ValueError("calibration diagnostic needs four seed-fold draws")
    return contract


def load_report(path: Path, expected_checkpoint: str, *, parity: bool) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != REPORT_ROLE:
        raise ValueError(f"unexpected sufficient-statistics report role: {path}")
    if payload.get("sufficient_statistics_contract") != REPORT_CONTRACT:
        raise ValueError(f"unexpected sufficient-statistics contract: {path}")
    if payload.get("test_evaluated") is not False or payload.get("test") is not None:
        raise ValueError(f"sufficient-statistics report evaluated test: {path}")
    if payload.get("promotion_eligible") is not False:
        raise ValueError(f"sufficient-statistics report permits promotion: {path}")
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"sufficient-statistics report permits live orders: {path}")
    if payload.get("inputs", {}).get("checkpoint_sha256") != expected_checkpoint:
        raise ValueError(f"sufficient-statistics checkpoint changed: {path}")
    reference = payload.get("reference_inference_parity")
    if parity and (not isinstance(reference, Mapping) or reference.get("passed") is not True):
        raise ValueError(f"validation inference parity failed: {path}")
    return payload


def primary_daily_rows(payload: Mapping[str, Any], horizon: str, bucket: str) -> list[dict[str, Any]]:
    rows = payload["validation"][
        "clock_bucket_causal_shock_daily_node_endpoint_rows"
    ][horizon][bucket][SUBSET]
    if not rows:
        raise ValueError(f"sufficient-statistics cell is empty: {horizon}/{bucket}")
    return list(rows)


def sufficient_statistics(row: Mapping[str, Any]) -> dict[str, float]:
    values = row.get("sufficient_statistics")
    required = {
        "count",
        "prediction_sum",
        "target_sum",
        "prediction_squared_sum",
        "target_squared_sum",
        "prediction_target_cross_sum",
        "squared_error_sum",
    }
    if not isinstance(values, Mapping) or set(values) != required:
        raise ValueError("daily sufficient statistics are incomplete")
    result = {name: float(values[name]) for name in required}
    if int(result["count"]) < 3 or not np.isfinite(list(result.values())).all():
        raise ValueError("daily sufficient statistics are invalid")
    return result


def combine_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    combined = {
        "count": 0.0,
        "prediction_sum": 0.0,
        "target_sum": 0.0,
        "prediction_squared_sum": 0.0,
        "target_squared_sum": 0.0,
        "prediction_target_cross_sum": 0.0,
        "squared_error_sum": 0.0,
    }
    for row in rows:
        values = sufficient_statistics(row)
        for name, value in values.items():
            combined[name] += value
    return combined


def fit_affine(
    rows: Sequence[Mapping[str, Any]], bounds: Sequence[float]
) -> dict[str, float | bool | int]:
    stats = combine_statistics(rows)
    count = stats["count"]
    prediction_variation = (
        stats["prediction_squared_sum"] - stats["prediction_sum"] ** 2 / count
    )
    covariance = (
        stats["prediction_target_cross_sum"]
        - stats["prediction_sum"] * stats["target_sum"] / count
    )
    if prediction_variation <= count * 1e-24:
        raise ValueError("calibration predictions have no variation")
    raw_slope = covariance / prediction_variation
    slope = float(np.clip(raw_slope, float(bounds[0]), float(bounds[1])))
    intercept = stats["target_sum"] / count - slope * stats["prediction_sum"] / count
    return {
        "count": int(count),
        "raw_slope": float(raw_slope),
        "slope": slope,
        "intercept": float(intercept),
        "slope_clipped": bool(not math.isclose(raw_slope, slope, rel_tol=0.0, abs_tol=0.0)),
    }


def transformed_metrics(
    row: Mapping[str, Any], coefficient: Mapping[str, Any] | None
) -> dict[str, float]:
    if coefficient is None:
        return {metric: float(row[metric]) for metric in METRICS}
    stats = sufficient_statistics(row)
    count = stats["count"]
    intercept = float(coefficient["intercept"])
    slope = float(coefficient["slope"])
    prediction_sum = intercept * count + slope * stats["prediction_sum"]
    prediction_squared_sum = (
        intercept**2 * count
        + 2.0 * intercept * slope * stats["prediction_sum"]
        + slope**2 * stats["prediction_squared_sum"]
    )
    cross_sum = (
        intercept * stats["target_sum"]
        + slope * stats["prediction_target_cross_sum"]
    )
    squared_error_sum = (
        prediction_squared_sum
        - 2.0 * cross_sum
        + stats["target_squared_sum"]
    )
    squared_error_sum = max(float(squared_error_sum), 0.0)
    prediction_variation = prediction_squared_sum - prediction_sum**2 / count
    target_variation = (
        stats["target_squared_sum"] - stats["target_sum"] ** 2 / count
    )
    covariance = cross_sum - prediction_sum * stats["target_sum"] / count
    if prediction_variation <= count * 1e-24 or target_variation <= count * 1e-24:
        raise ValueError("transformed daily metric has no variation")
    pearson = covariance / math.sqrt(prediction_variation * target_variation)
    if stats["target_squared_sum"] <= 1e-12:
        raise ValueError("transformed daily metric has zero target energy")
    return {
        "pearson": float(np.clip(pearson, -1.0, 1.0)),
        "skill_vs_zero_mse": float(
            1.0 - squared_error_sum / stats["target_squared_sum"]
        ),
    }


def assert_target_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> None:
    left_stats = sufficient_statistics(left)
    right_stats = sufficient_statistics(right)
    for name in ("count", "target_sum", "target_squared_sum"):
        if not np.isclose(
            left_stats[name], right_stats[name], rtol=0.0, atol=1e-10
        ):
            raise ValueError(f"baseline/aligned target statistics differ: {name}")


def circular_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def block_bootstrap(
    arrays: Sequence[np.ndarray], *, samples: int, block_length: int, seed: int
) -> dict[str, float | int]:
    values = [np.asarray(array, dtype=np.float64) for array in arrays]
    if len(values) != 2 or any(len(array) < 2 for array in values):
        raise ValueError("session bootstrap requires two populated folds")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        sampled = [
            array[circular_indices(len(array), block_length, rng)]
            for array in values
        ]
        draws[index] = float(np.concatenate(sampled).mean())
    return {
        "samples": samples,
        "block_length": block_length,
        "folds": len(values),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def summarize(
    frame: pd.DataFrame,
    *,
    comparison: str,
    bucket: str | None,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    selected = frame.loc[frame["comparison"] == comparison]
    if bucket is not None:
        selected = selected.loc[selected["bucket"] == bucket]
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        values = selected.loc[selected["metric"] == metric]
        session = (
            values.groupby(["fold", "date"], as_index=False, observed=True)["delta"]
            .mean()
            .sort_values(["fold", "date"])
        )
        arrays: list[np.ndarray] = []
        per_fold: dict[str, Any] = {}
        for fold in ("fold1", "fold3"):
            array = session.loc[session["fold"] == fold, "delta"].to_numpy(
                dtype=np.float64
            )
            arrays.append(array)
            per_fold[fold] = {
                "sessions": len(array),
                "mean_delta": float(array.mean()),
                "positive_session_fraction": float(np.mean(array > 0.0)),
            }
        bootstrap = block_bootstrap(
            arrays,
            samples=samples,
            block_length=block_length,
            seed=seed + metric_index,
        )
        pooled = np.concatenate(arrays)
        positive_folds = sum(row["mean_delta"] > 0.0 for row in per_fold.values())
        result[metric] = {
            "sessions": len(pooled),
            "mean_delta": float(pooled.mean()),
            "positive_session_fraction": float(np.mean(pooled > 0.0)),
            "positive_fold_count": positive_folds,
            "per_fold": per_fold,
            "session_clustered_block_bootstrap": bootstrap,
            "confirmed_positive": bool(
                positive_folds == 2 and float(bootstrap["lower_95"]) > 0.0
            ),
        }
    return result


def report_paths(draw: Mapping[str, Any], phase: str) -> dict[str, Path]:
    record = draw[phase]
    return {model: resolve(record[model]) for model in ("baseline", "aligned")}


def main() -> None:
    args = parse_args()
    contract = load_contract(args.contract)
    bounds = [float(value) for value in contract["affine_slope_bounds"]]
    cell_records: list[dict[str, Any]] = []
    coefficient_records: list[dict[str, Any]] = []
    report_hashes: dict[str, str] = {}

    for draw in contract["draws"]:
        seed = int(draw["seed"])
        fold = str(draw["fold"])
        checkpoints = draw["checkpoint_sha256"]
        calibration_paths = report_paths(draw, "calibration_reports")
        validation_paths = report_paths(draw, "validation_reports")
        calibration = {
            model: load_report(
                path, str(checkpoints[model]), parity=False
            )
            for model, path in calibration_paths.items()
        }
        validation = {
            model: load_report(
                path, str(checkpoints[model]), parity=True
            )
            for model, path in validation_paths.items()
        }
        for phase, paths in (
            ("calibration", calibration_paths),
            ("validation", validation_paths),
        ):
            for model, path in paths.items():
                report_hashes[f"seed{seed}.{fold}.{phase}.{model}"] = sha256_file(path)

        for horizon in HORIZONS:
            for bucket in BUCKETS:
                coefficients: dict[str, dict[str, Any]] = {}
                for model in ("baseline", "aligned"):
                    rows = primary_daily_rows(calibration[model], horizon, bucket)
                    coefficients[model] = fit_affine(rows, bounds)
                    coefficient_records.append(
                        {
                            "seed": seed,
                            "fold": fold,
                            "model": model,
                            "horizon": horizon,
                            "bucket": bucket,
                            **coefficients[model],
                        }
                    )

                validation_rows = {
                    model: {
                        str(row["date"]): row
                        for row in primary_daily_rows(
                            validation[model], horizon, bucket
                        )
                    }
                    for model in ("baseline", "aligned")
                }
                if set(validation_rows["baseline"]) != set(validation_rows["aligned"]):
                    raise ValueError("baseline/aligned validation dates differ")
                for date in sorted(validation_rows["baseline"]):
                    baseline = validation_rows["baseline"][date]
                    aligned = validation_rows["aligned"][date]
                    assert_target_identity(baseline, aligned)
                    raw_baseline = transformed_metrics(baseline, None)
                    raw_aligned = transformed_metrics(aligned, None)
                    affine_baseline = transformed_metrics(
                        baseline, coefficients["baseline"]
                    )
                    affine_aligned = transformed_metrics(
                        aligned, coefficients["aligned"]
                    )
                    pairs = {
                        "raw_aligned_vs_raw_baseline": (
                            raw_aligned,
                            raw_baseline,
                        ),
                        "affine_aligned_vs_raw_baseline": (
                            affine_aligned,
                            raw_baseline,
                        ),
                        "affine_aligned_vs_affine_baseline": (
                            affine_aligned,
                            affine_baseline,
                        ),
                    }
                    for comparison, (candidate, comparator) in pairs.items():
                        for metric in METRICS:
                            cell_records.append(
                                {
                                    "comparison": comparison,
                                    "seed": seed,
                                    "fold": fold,
                                    "horizon": horizon,
                                    "bucket": bucket,
                                    "subset": SUBSET,
                                    "metric": metric,
                                    "date": date,
                                    "candidate": float(candidate[metric]),
                                    "comparator": float(comparator[metric]),
                                    "delta": float(
                                        candidate[metric] - comparator[metric]
                                    ),
                                }
                            )

    cells = pd.DataFrame(cell_records).sort_values(
        ["comparison", "metric", "fold", "date", "bucket", "horizon", "seed"]
    ).reset_index(drop=True)
    coefficients = pd.DataFrame(coefficient_records).sort_values(
        ["model", "fold", "bucket", "horizon", "seed"]
    ).reset_index(drop=True)

    raw_expected = []
    for seed_record in contract["original_daily_paired_deltas"]:
        seed = int(seed_record["seed"])
        path = resolve(seed_record["path"])
        if sha256_file(path) != seed_record["sha256"]:
            raise ValueError("original paired-delta source changed")
        frame = pd.read_csv(path)
        frame = frame.loc[
            (frame["comparison"] == "aligned_vs_baseline")
            & frame["horizon"].isin(HORIZONS)
            & frame["bucket"].isin(BUCKETS)
            & (frame["subset"] == SUBSET)
            & frame["metric"].isin(METRICS)
        ].copy()
        frame["seed"] = seed
        raw_expected.append(frame)
    expected = pd.concat(raw_expected, ignore_index=True)
    observed = cells.loc[
        cells["comparison"] == "raw_aligned_vs_raw_baseline"
    ]
    keys = ["seed", "fold", "horizon", "bucket", "subset", "metric", "date"]
    parity = expected.merge(
        observed[keys + ["delta"]],
        on=keys,
        how="outer",
        suffixes=("_expected", "_observed"),
        indicator=True,
    )
    if not (parity["_merge"] == "both").all():
        raise ValueError("raw sufficient-statistics cell set differs from source audit")
    maximum_raw_parity_difference = float(
        np.max(np.abs(parity["delta_expected"] - parity["delta_observed"]))
    )
    if maximum_raw_parity_difference > float(contract["raw_metric_parity_tolerance"]):
        raise ValueError("raw sufficient-statistics inference parity changed")

    bootstrap = contract["bootstrap"]
    results: dict[str, Any] = {}
    for comparison_index, comparison in enumerate(COMPARISONS):
        results[comparison] = {
            "combined_available_buckets": summarize(
                cells,
                comparison=comparison,
                bucket=None,
                samples=int(bootstrap["samples"]),
                block_length=int(bootstrap["block_length_sessions"]),
                seed=int(bootstrap["seed"]) + comparison_index * 100,
            ),
            "clock_buckets": {
                bucket: summarize(
                    cells,
                    comparison=comparison,
                    bucket=bucket,
                    samples=int(bootstrap["samples"]),
                    block_length=int(bootstrap["block_length_sessions"]),
                    seed=int(bootstrap["seed"])
                    + comparison_index * 100
                    + (bucket_index + 1) * 10,
                )
                for bucket_index, bucket in enumerate(BUCKETS)
            },
        }

    fair = results["affine_aligned_vs_affine_baseline"]
    raw = results["raw_aligned_vs_raw_baseline"]
    conclusions = {
        "raw_overall_skill_confirmed": raw["combined_available_buckets"][
            "skill_vs_zero_mse"
        ]["confirmed_positive"],
        "fair_affine_overall_skill_confirmed": fair[
            "combined_available_buckets"
        ]["skill_vs_zero_mse"]["confirmed_positive"],
        "fair_affine_morning_skill_confirmed": fair["clock_buckets"][
            "morning_0930_1059"
        ]["skill_vs_zero_mse"]["confirmed_positive"],
        "fair_affine_open_skill_confirmed": fair["clock_buckets"][
            "open_0900_0929"
        ]["skill_vs_zero_mse"]["confirmed_positive"],
    }
    conclusions["calibration_explains_overall_skill_gap"] = bool(
        not conclusions["raw_overall_skill_confirmed"]
        and conclusions["fair_affine_overall_skill_confirmed"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells_path = args.output_dir / "daily_cell_deltas.csv"
    coefficients_path = args.output_dir / "affine_coefficients.csv"
    summary_path = args.output_dir / "summary.json"
    cells.to_csv(cells_path, index=False)
    coefficients.to_csv(coefficients_path, index=False)
    coefficient_summary = {
        model: {
            "slope_minimum": float(group["slope"].min()),
            "slope_median": float(group["slope"].median()),
            "slope_maximum": float(group["slope"].max()),
            "clipped_cells": int(group["slope_clipped"].sum()),
            "cells": int(len(group)),
        }
        for model, group in coefficients.groupby("model", observed=True)
    }
    summary = {
        "schema_version": 1,
        "role": "post_impact_rank_adapter_train_only_affine_calibration_audit",
        "status": "complete",
        "evidence_class": "post_selection_retrospective_calibration_diagnostic",
        "test_evaluated": False,
        "counts_as_primary_forward_evidence": False,
        "changes_frozen_prospective_candidate": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "contract": portable(args.contract),
        "contract_sha256": sha256_file(args.contract),
        "auditor": portable(Path(__file__)),
        "auditor_sha256": sha256_file(Path(__file__).resolve()),
        "method": (
            "Fit one positive affine map per optimizer seed, validation fold, model, "
            "clock bucket, and primary horizon using only that model's original "
            "training-period sufficient statistics; apply the frozen map to the "
            "non-overlapping validation period and cluster both seeds and all cells "
            "at the trading-session level. Apply the same procedure to the baseline."
        ),
        "affine_slope_bounds": bounds,
        "coefficient_summary": coefficient_summary,
        "raw_metric_parity": {
            "maximum_absolute_delta_difference": maximum_raw_parity_difference,
            "tolerance": float(contract["raw_metric_parity_tolerance"]),
            "passed": True,
        },
        "report_sha256": report_hashes,
        "daily_cell_deltas_sha256": sha256_file(cells_path),
        "affine_coefficients_sha256": sha256_file(coefficients_path),
        "results": results,
        "conclusions": conclusions,
        "decision": (
            "train_only_affine_calibration_explains_skill_gap"
            if conclusions["calibration_explains_overall_skill_gap"]
            else "train_only_affine_calibration_does_not_explain_skill_gap"
        ),
        "prospective_policy": (
            "Do not alter the frozen candidate or forward gate after this diagnostic. "
            "Any calibrated successor requires a new predeclared contract and untouched "
            "forward sessions."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "decision": summary["decision"],
                "summary": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "test_evaluated": False,
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
