from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


COMPARISON = "aligned_vs_baseline"
FOLDS = ("fold1", "fold3")
HORIZONS = ("15m", "30m", "60m")
BUCKETS = ("open_0900_0929", "morning_0930_1059")
METRICS = ("pearson", "skill_vs_zero_mse")
SUBSET = "adaptive_observed_surprise_recent_30m"
REQUIRED_COLUMNS = {
    "comparison",
    "fold",
    "horizon",
    "bucket",
    "subset",
    "metric",
    "date",
    "candidate",
    "comparator",
    "delta",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_indices(
    length: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    blocks = int(np.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).reshape(-1)[:length]


def stratified_block_bootstrap_mean(
    fold_values: Sequence[np.ndarray],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    arrays = [np.asarray(values, dtype=np.float64) for values in fold_values]
    if not arrays or any(len(values) < 2 for values in arrays):
        raise ValueError("each fold needs at least two trading sessions")
    if any(not np.isfinite(values).all() for values in arrays):
        raise ValueError("bootstrap values must be finite")
    if samples < 1 or block_length < 1:
        raise ValueError("bootstrap samples and block length must be positive")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        resampled = [
            values[circular_indices(len(values), block_length, rng)]
            for values in arrays
        ]
        draws[draw] = float(np.concatenate(resampled).mean())
    return {
        "samples": samples,
        "block_length": block_length,
        "folds": len(arrays),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def load_seed(path: Path, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.loc[
        (frame["comparison"] == COMPARISON)
        & frame["fold"].isin(FOLDS)
        & frame["horizon"].isin(HORIZONS)
        & frame["bucket"].isin(BUCKETS)
        & (frame["subset"] == SUBSET)
        & frame["metric"].isin(METRICS)
    ].copy()
    if frame.empty:
        raise ValueError(f"{path} has no primary aligned-vs-baseline rows")
    for column, expected in (
        ("fold", FOLDS),
        ("horizon", HORIZONS),
        ("bucket", BUCKETS),
        ("metric", METRICS),
    ):
        if set(frame[column]) != set(expected):
            raise ValueError(f"{path} has unexpected {column} coverage")
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime(
        "%Y-%m-%d"
    )
    numeric = frame[["candidate", "comparator", "delta"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise ValueError(f"{path} contains non-finite values")
    recomputed = frame["candidate"].to_numpy() - frame["comparator"].to_numpy()
    if not np.allclose(frame["delta"].to_numpy(), recomputed, rtol=0.0, atol=1e-12):
        raise ValueError(f"{path} contains inconsistent paired deltas")
    key = ["fold", "horizon", "bucket", "subset", "metric", "date"]
    if frame.duplicated(key).any():
        raise ValueError(f"{path} contains duplicate paired cells")
    frame["seed"] = seed
    return frame.sort_values([*key, "seed"]).reset_index(drop=True)


def validate_pairing(seed_frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    key = ["fold", "horizon", "bucket", "subset", "metric", "date"]
    reference = set(map(tuple, seed_frames[0][key].itertuples(index=False, name=None)))
    for frame in seed_frames[1:]:
        current = set(map(tuple, frame[key].itertuples(index=False, name=None)))
        if current != reference:
            raise ValueError("optimizer seeds do not have exact cell/date pairing")
    combined = pd.concat(seed_frames, ignore_index=True)
    counts = combined.groupby(key, observed=True)["seed"].agg(["size", "nunique"])
    expected_seeds = len(seed_frames)
    if not ((counts["size"] == expected_seeds) & (counts["nunique"] == expected_seeds)).all():
        raise ValueError("each source cell/date must contain every seed exactly once")
    fold_dates = {
        fold: set(combined.loc[combined["fold"] == fold, "date"])
        for fold in FOLDS
    }
    if fold_dates[FOLDS[0]].intersection(fold_dates[FOLDS[1]]):
        raise ValueError("validation fold trading dates overlap")
    return combined


def aggregate_sessions(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    combined = (
        frame.groupby(["fold", "date", "metric"], as_index=False, observed=True)
        .agg(
            delta=("delta", "mean"),
            source_cells=("delta", "size"),
            seed_count=("seed", "nunique"),
            horizon_count=("horizon", "nunique"),
            bucket_count=("bucket", "nunique"),
        )
        .assign(scope="combined_available_buckets", bucket="all")
    )
    rows.append(combined)
    by_bucket = (
        frame.groupby(
            ["fold", "date", "bucket", "metric"],
            as_index=False,
            observed=True,
        )
        .agg(
            delta=("delta", "mean"),
            source_cells=("delta", "size"),
            seed_count=("seed", "nunique"),
            horizon_count=("horizon", "nunique"),
            bucket_count=("bucket", "nunique"),
        )
        .assign(scope="clock_bucket")
    )
    rows.append(by_bucket)
    output = pd.concat(rows, ignore_index=True)
    expected_cell_multiple = len(HORIZONS) * len(seed_frames_from_count(frame))
    if not (output["source_cells"] % expected_cell_multiple == 0).all():
        raise ValueError("session clusters have incomplete seed/horizon cells")
    if not (output["seed_count"] == len(seed_frames_from_count(frame))).all():
        raise ValueError("session clusters have incomplete seed coverage")
    if not (output["horizon_count"] == len(HORIZONS)).all():
        raise ValueError("session clusters have incomplete horizon coverage")
    columns = [
        "scope",
        "fold",
        "date",
        "bucket",
        "metric",
        "delta",
        "source_cells",
        "seed_count",
        "horizon_count",
        "bucket_count",
    ]
    return output[columns].sort_values(
        ["scope", "bucket", "metric", "fold", "date"]
    ).reset_index(drop=True)


def seed_frames_from_count(frame: pd.DataFrame) -> tuple[int, ...]:
    seeds = tuple(sorted(int(value) for value in frame["seed"].unique()))
    if len(seeds) < 2:
        raise ValueError("at least two optimizer seeds are required")
    return seeds


def summarize_scope(
    session_rows: pd.DataFrame,
    raw_rows: pd.DataFrame,
    *,
    scope: str,
    bucket: str,
    samples: int,
    block_length: int,
    seed_base: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        selected = session_rows.loc[
            (session_rows["scope"] == scope)
            & (session_rows["bucket"] == bucket)
            & (session_rows["metric"] == metric)
        ]
        fold_arrays: list[np.ndarray] = []
        per_fold: dict[str, Any] = {}
        for fold in FOLDS:
            values = selected.loc[selected["fold"] == fold].sort_values("date")
            array = values["delta"].to_numpy(dtype=np.float64)
            fold_arrays.append(array)
            per_fold[fold] = {
                "sessions": len(array),
                "mean_delta": float(array.mean()),
                "positive_session_fraction": float(np.mean(array > 0.0)),
            }
        if scope == "combined_available_buckets":
            raw_selected = raw_rows.loc[raw_rows["metric"] == metric]
        else:
            raw_selected = raw_rows.loc[
                (raw_rows["bucket"] == bucket) & (raw_rows["metric"] == metric)
            ]
        seed_fold = (
            raw_selected.groupby(
                ["seed", "fold", "date"], as_index=False, observed=True
            )["delta"]
            .mean()
            .groupby(["seed", "fold"], observed=True)["delta"]
            .mean()
            .sort_index()
        )
        seed_fold_diagnostics = {
            f"seed{int(seed)}_{fold}": float(value)
            for (seed, fold), value in seed_fold.items()
        }
        bootstrap = stratified_block_bootstrap_mean(
            fold_arrays,
            samples=samples,
            block_length=block_length,
            seed=seed_base + metric_index,
        )
        pooled = np.concatenate(fold_arrays)
        positive_folds = sum(value["mean_delta"] > 0.0 for value in per_fold.values())
        result[metric] = {
            "sessions": len(pooled),
            "mean_delta": float(pooled.mean()),
            "positive_session_fraction": float(np.mean(pooled > 0.0)),
            "positive_fold_count": positive_folds,
            "per_fold": per_fold,
            "per_seed_fold_mean_delta": seed_fold_diagnostics,
            "session_clustered_block_bootstrap": bootstrap,
            "confirmed_positive": bool(
                positive_folds == len(FOLDS) and float(bootstrap["lower_95"]) > 0.0
            ),
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed17-daily", type=Path, required=True)
    parser.add_argument("--seed43-daily", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_paths = {17: args.seed17_daily, 43: args.seed43_daily}
    seed_frames = [load_seed(path, seed) for seed, path in source_paths.items()]
    raw = validate_pairing(seed_frames)
    sessions = aggregate_sessions(raw)

    overall = summarize_scope(
        sessions,
        raw,
        scope="combined_available_buckets",
        bucket="all",
        samples=args.samples,
        block_length=args.block_length,
        seed_base=args.bootstrap_seed,
    )
    buckets = {
        bucket: summarize_scope(
            sessions,
            raw,
            scope="clock_bucket",
            bucket=bucket,
            samples=args.samples,
            block_length=args.block_length,
            seed_base=args.bootstrap_seed + 100 * (index + 1),
        )
        for index, bucket in enumerate(BUCKETS)
    }
    conclusions = {
        "overall_increment_confirmed": all(
            overall[metric]["confirmed_positive"] for metric in METRICS
        ),
        "open_increment_confirmed": all(
            buckets[BUCKETS[0]][metric]["confirmed_positive"] for metric in METRICS
        ),
        "morning_increment_confirmed": all(
            buckets[BUCKETS[1]][metric]["confirmed_positive"] for metric in METRICS
        ),
    }
    summary = {
        "schema_version": 1,
        "role": "post_impact_rank_adapter_baseline_session_sensitivity_audit",
        "status": "complete",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "counts_as_primary_forward_evidence": False,
        "evidence_class": "post_selection_retrospective_sensitivity_diagnostic",
        "retrospective_test_period_previously_inspected": True,
        "changes_frozen_prospective_candidate": False,
        "comparison": COMPARISON,
        "primary_horizons": list(HORIZONS),
        "primary_subset": SUBSET,
        "optimizer_seeds_jointly_clustered": [17, 43],
        "statistical_unit": "trading_session",
        "method": (
            "Within each original validation fold and trading date, average paired "
            "aligned-minus-baseline deltas jointly across optimizer seeds and primary "
            "horizons before applying a two-fold circular moving-block bootstrap. "
            "Report all available actionable buckets jointly and each clock bucket "
            "separately so repeated market dates never act as independent samples."
        ),
        "bootstrap": {
            "samples": args.samples,
            "block_length_sessions": args.block_length,
            "base_seed": args.bootstrap_seed,
        },
        "auditor": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "source_daily": {
            f"seed{seed}": {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for seed, path in source_paths.items()
        },
        "checks": {
            "exact_two_seed_pairing_per_cell_date": True,
            "two_nonoverlapping_folds": True,
            "primary_horizon_coverage_complete": True,
            "paired_delta_arithmetic_exact": True,
        },
        "results": {
            "combined_available_buckets": overall,
            "clock_buckets": buckets,
        },
        "conclusions": conclusions,
        "decision": (
            "baseline_increment_confirmed_in_all_scopes"
            if all(conclusions.values())
            else "baseline_increment_not_confirmed_in_every_scope"
        ),
        "prospective_policy": (
            "Do not alter candidate, clocks, thresholds, or minimum forward-session "
            "count after inspecting this retrospective sensitivity result."
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session_path = args.output_dir / "session_paired_deltas.csv"
    summary_path = args.output_dir / "summary.json"
    sessions.to_csv(session_path, index=False)
    summary["session_paired_deltas_sha256"] = sha256_file(session_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(summary_path),
                "summary_sha256": sha256_file(summary_path),
                "conclusions": conclusions,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
