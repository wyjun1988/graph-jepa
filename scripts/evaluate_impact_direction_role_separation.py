from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


QLIB_FEATURES = (
    "qlib_mean",
    "qlib_median",
    "qlib_std",
    "qlib_q10",
    "qlib_q25",
    "qlib_q75",
    "qlib_q90",
    "qlib_negative_fraction",
    "qlib_liquidity_weighted_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value for {name}")
    return result


def _aggregate_predictions(path: Path, horizon: int) -> pd.DataFrame:
    frame = pd.read_parquet(path).reset_index()
    required = {
        "datetime",
        "prediction",
        "label",
        "liquidity",
        "current_available",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Qlib prediction schema mismatch: {path}")
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    rows = []
    for date, group in frame.groupby("datetime", sort=True):
        valid = (
            group["current_available"].astype(bool).to_numpy()
            & np.isfinite(group["prediction"].to_numpy(dtype=np.float64))
            & np.isfinite(group["label"].to_numpy(dtype=np.float64))
        )
        prediction = group.loc[valid, "prediction"].to_numpy(dtype=np.float64)
        label = group.loc[valid, "label"].to_numpy(dtype=np.float64)
        liquidity = group.loc[valid, "liquidity"].to_numpy(dtype=np.float64)
        if prediction.size < 20:
            raise ValueError(f"fewer than twenty Qlib rows for {date}: {path}")
        finite_liquidity = np.isfinite(liquidity)
        if finite_liquidity.any():
            centered = liquidity[finite_liquidity] - np.max(liquidity[finite_liquidity])
            weights = np.exp(np.clip(centered, -20.0, 0.0))
            liquidity_weighted = float(
                np.average(prediction[finite_liquidity], weights=weights)
            )
        else:
            liquidity_weighted = float(np.mean(prediction))
        rows.append(
            {
                "date": pd.Timestamp(date),
                "horizon": int(horizon),
                "stocks": int(prediction.size),
                "qlib_mean": float(np.mean(prediction)),
                "qlib_median": float(np.median(prediction)),
                "qlib_std": float(np.std(prediction)),
                "qlib_q10": float(np.quantile(prediction, 0.10)),
                "qlib_q25": float(np.quantile(prediction, 0.25)),
                "qlib_q75": float(np.quantile(prediction, 0.75)),
                "qlib_q90": float(np.quantile(prediction, 0.90)),
                "qlib_negative_fraction": float(np.mean(prediction < 0.0)),
                "qlib_liquidity_weighted_mean": liquidity_weighted,
                "actual_median_return": float(np.median(label)),
            }
        )
    return pd.DataFrame(rows)


def _load_split(
    qlib_root: Path,
    family_root: Path,
    split: str,
    horizons: Sequence[int],
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    family_path = family_root / f"daily_{split}.csv"
    family = pd.read_csv(family_path)
    family["date"] = pd.to_datetime(family["date"])
    required_family = {
        "date",
        "horizon",
        "actual_broad_selloff",
        "probability_broad_selloff",
        "actual_normalized_salience",
        "predicted_normalized_salience",
    }
    if not required_family.issubset(family.columns):
        raise ValueError(f"family-query daily schema mismatch: {family_path}")
    parts = []
    inputs = [{"path": str(family_path), "sha256": _sha256(family_path)}]
    qlib_split = "valid" if split == "validation" else split
    for horizon in horizons:
        qlib_path = qlib_root / f"predictions_h{int(horizon)}_{qlib_split}.parquet"
        aggregate = _aggregate_predictions(qlib_path, int(horizon))
        selected = family[family["horizon"] == int(horizon)].copy()
        merged = aggregate.merge(
            selected,
            on=("date", "horizon"),
            how="inner",
            validate="one_to_one",
        )
        if len(merged) != len(aggregate) or len(merged) != len(selected):
            raise ValueError(f"Qlib and family-query dates do not align at h{horizon}")
        parts.append(merged)
        inputs.append({"path": str(qlib_path), "sha256": _sha256(qlib_path)})
    result = pd.concat(parts, ignore_index=True).sort_values(
        ["date", "horizon"]
    )
    result = result.reset_index(drop=True)
    return result, inputs


def _design(frame: pd.DataFrame, horizons: Sequence[int]) -> np.ndarray:
    columns = [frame[name].to_numpy(dtype=np.float64) for name in QLIB_FEATURES]
    observed_horizon = frame["horizon"].to_numpy(dtype=np.int64)
    columns.extend((observed_horizon == int(horizon)).astype(np.float64) for horizon in horizons)
    result = np.column_stack(columns)
    if not np.isfinite(result).all():
        raise ValueError("role-separation design contains non-finite values")
    return result


def _select_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    candidates = np.unique(
        np.concatenate(
            (
                np.asarray([0.5]),
                np.quantile(probability, np.linspace(0.02, 0.98, 97)),
            )
        )
    )
    ranked = []
    for threshold in candidates:
        prediction = probability >= float(threshold)
        score = balanced_accuracy_score(labels, prediction)
        ranked.append((float(score), -abs(float(threshold) - 0.5), float(threshold)))
    return max(ranked)[2]


def _binary_metrics(
    labels: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    prediction = probability >= float(threshold)
    two_classes = np.unique(labels).size == 2
    return {
        "rows": int(labels.size),
        "events": int(labels.sum()),
        "roc_auc": float(roc_auc_score(labels, probability)) if two_classes else float("nan"),
        "average_precision": (
            float(average_precision_score(labels, probability))
            if labels.any()
            else float("nan")
        ),
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "precision": float(precision_score(labels, prediction, zero_division=0)),
        "recall": float(recall_score(labels, prediction, zero_division=0)),
        "threshold": float(threshold),
    }


def _direction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    correlation = (
        float(np.corrcoef(actual, predicted)[0, 1])
        if actual.size >= 3 and np.std(actual) > 0 and np.std(predicted) > 0
        else float("nan")
    )
    negative_rate = float(np.mean(actual < 0.0))
    majority_accuracy = max(negative_rate, 1.0 - negative_rate)
    return {
        "rows": int(actual.size),
        "pearson_correlation": correlation,
        "sign_accuracy": float(np.mean((actual < 0.0) == (predicted < 0.0))),
        "majority_sign_accuracy": majority_accuracy,
        "sign_accuracy_delta_vs_majority": float(
            np.mean((actual < 0.0) == (predicted < 0.0)) - majority_accuracy
        ),
    }


def _impact_peak_rows(
    frame: pd.DataFrame,
    *,
    score_column: str,
    selection_rate: float | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    peak_indices = frame.groupby("date", sort=True)[score_column].idxmax()
    peaks = frame.loc[peak_indices].copy()
    if selection_rate is not None:
        count = max(1, int(math.ceil(float(selection_rate) * len(peaks))))
        peaks = peaks.nlargest(count, score_column)
    if threshold is not None:
        peaks = peaks[peaks[score_column] >= float(threshold)]
    return peaks.sort_values("date")


def _bootstrap_by_date(
    frame: pd.DataFrame,
    probability: np.ndarray,
    direction: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> dict[str, list[float]]:
    probability = np.asarray(probability, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    labels = frame["actual_broad_selloff"].astype(bool).to_numpy()
    actual = frame["actual_median_return"].to_numpy(dtype=np.float64)
    groups = [indices.to_numpy() for _, indices in frame.groupby("date").groups.items()]
    rng = np.random.default_rng(int(seed))
    auc_values = []
    correlation_values = []
    for _ in range(int(samples)):
        chosen = rng.integers(0, len(groups), size=len(groups))
        indices = np.concatenate([groups[index] for index in chosen])
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size == 2:
            auc_values.append(float(roc_auc_score(sampled_labels, probability[indices])))
        sampled_actual = actual[indices]
        sampled_direction = direction[indices]
        if np.std(sampled_actual) > 0 and np.std(sampled_direction) > 0:
            correlation_values.append(
                float(np.corrcoef(sampled_actual, sampled_direction)[0, 1])
            )
    return {"roc_auc": auc_values, "direction_correlation": correlation_values}


def _interval(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"lower_95": float("nan"), "upper_95": float("nan")}
    return {
        "lower_95": float(np.quantile(array, 0.025)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def evaluate(
    *,
    qlib_root: Path,
    family_root: Path,
    horizons: Sequence[int],
    seed: int,
    bootstrap_samples: int,
    placebo_samples: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    qlib_summary_path = qlib_root / "summary.json"
    family_summary_path = family_root / "summary.json"
    qlib_summary = json.loads(qlib_summary_path.read_text(encoding="utf-8"))
    family_summary = json.loads(family_summary_path.read_text(encoding="utf-8"))
    if qlib_summary.get("live_orders_allowed") is not False:
        raise ValueError("unsafe Qlib summary")
    if family_summary.get("live_orders_allowed") is not False:
        raise ValueError("unsafe family-query summary")
    if qlib_summary.get("checkpoint_sha256") != family_summary.get(
        "parent_model_sha256"
    ):
        raise ValueError("Qlib and family-query artifacts use different checkpoints")
    validation, validation_inputs = _load_split(
        qlib_root, family_root, "validation", horizons
    )
    test, test_inputs = _load_split(qlib_root, family_root, "test", horizons)
    validation_design = _design(validation, horizons)
    test_design = _design(test, horizons)
    validation_labels = validation["actual_broad_selloff"].astype(bool).to_numpy()
    test_labels = test["actual_broad_selloff"].astype(bool).to_numpy()

    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=2000,
            random_state=int(seed),
        ),
    )
    classifier.fit(validation_design, validation_labels)
    validation_probability = classifier.predict_proba(validation_design)[:, 1]
    threshold = _select_threshold(validation_labels, validation_probability)
    test_probability = classifier.predict_proba(test_design)[:, 1]

    regressor = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    regressor.fit(
        validation_design,
        validation["actual_median_return"].to_numpy(dtype=np.float64),
    )
    test_direction = regressor.predict(test_design)
    test = test.copy()
    test["role_direction_probability"] = test_probability
    test["role_median_return_prediction"] = test_direction

    major_summary_path = family_root / "major_trajectory" / "summary.json"
    major_summary = json.loads(major_summary_path.read_text(encoding="utf-8"))
    selection_rate = _finite_float(
        major_summary["fit_major_event_rate"], "fit_major_event_rate"
    )
    major_threshold = _finite_float(
        major_summary["fit_major_event_threshold"], "fit_major_event_threshold"
    )
    predicted_major = _impact_peak_rows(
        test,
        score_column="predicted_normalized_salience",
        selection_rate=selection_rate,
    )
    actual_major = _impact_peak_rows(
        test,
        score_column="actual_normalized_salience",
        threshold=major_threshold,
    )

    row_binary = _binary_metrics(test_labels, test_probability, threshold)
    family_binary = _binary_metrics(
        test_labels,
        test["probability_broad_selloff"].to_numpy(dtype=np.float64),
        0.5,
    )
    row_direction = _direction_metrics(
        test["actual_median_return"], test_direction
    )
    predicted_major_direction = _direction_metrics(
        predicted_major["actual_median_return"],
        predicted_major["role_median_return_prediction"],
    )
    actual_major_direction = _direction_metrics(
        actual_major["actual_median_return"],
        actual_major["role_median_return_prediction"],
    )

    bootstrap = _bootstrap_by_date(
        test,
        test_probability,
        test_direction,
        samples=int(bootstrap_samples),
        seed=int(seed),
    )
    rng = np.random.default_rng(int(seed) + 1000)
    placebo_auc = []
    placebo_major_sign = []
    major_indices = predicted_major.index.to_numpy(dtype=np.int64)
    for _ in range(int(placebo_samples)):
        shuffled = test.copy()
        for horizon in horizons:
            positions = np.flatnonzero(
                shuffled["horizon"].to_numpy(dtype=np.int64) == int(horizon)
            )
            source = rng.permutation(positions)
            shuffled.loc[positions, list(QLIB_FEATURES)] = test.loc[
                source, list(QLIB_FEATURES)
            ].to_numpy()
        design = _design(shuffled, horizons)
        probability = classifier.predict_proba(design)[:, 1]
        direction = regressor.predict(design)
        if np.unique(test_labels).size == 2:
            placebo_auc.append(float(roc_auc_score(test_labels, probability)))
        placebo_major_sign.append(
            float(
                np.mean(
                    (test.loc[major_indices, "actual_median_return"].to_numpy() < 0.0)
                    == (direction[major_indices] < 0.0)
                )
            )
        )

    payload = {
        "status": "complete",
        "role": "qlib_direction_family_query_impact_role_separation_diagnostic",
        "horizons": [int(value) for value in horizons],
        "calibration": {
            "split": "validation_only",
            "rows": int(len(validation)),
            "dates": int(validation["date"].nunique()),
            "classifier": "StandardScaler + class-weighted LogisticRegression(C=0.1)",
            "regressor": "StandardScaler + Ridge(alpha=10.0)",
            "probability_threshold": float(threshold),
        },
        "test": {
            "rows": int(len(test)),
            "dates": int(test["date"].nunique()),
            "broad_selloff": row_binary,
            "family_query_broad_selloff_baseline": family_binary,
            "median_return_direction": row_direction,
            "predicted_major_peak_direction": predicted_major_direction,
            "actual_major_peak_direction": actual_major_direction,
            "predicted_major_dates": predicted_major["date"].dt.strftime("%Y-%m-%d").tolist(),
            "actual_major_dates": actual_major["date"].dt.strftime("%Y-%m-%d").tolist(),
        },
        "uncertainty": {
            "date_block_bootstrap_samples": int(bootstrap_samples),
            "broad_selloff_auc": _interval(bootstrap["roc_auc"]),
            "median_return_correlation": _interval(
                bootstrap["direction_correlation"]
            ),
            "placebo_samples": int(placebo_samples),
            "placebo_broad_selloff_auc_95": float(np.quantile(placebo_auc, 0.95)),
            "placebo_predicted_major_sign_accuracy_95": float(
                np.quantile(placebo_major_sign, 0.95)
            ),
        },
        "inputs": {
            "qlib_root": str(qlib_root),
            "family_query_root": str(family_root),
            "qlib_summary": {
                "path": str(qlib_summary_path),
                "sha256": _sha256(qlib_summary_path),
                "checkpoint_sha256": str(qlib_summary["checkpoint_sha256"]),
            },
            "family_query_summary": {
                "path": str(family_summary_path),
                "sha256": _sha256(family_summary_path),
                "parent_model_sha256": str(family_summary["parent_model_sha256"]),
            },
            "validation": validation_inputs,
            "test": test_inputs,
            "major_summary": {
                "path": str(major_summary_path),
                "sha256": _sha256(major_summary_path),
            },
        },
        "test_used_for_selection": True,
        "evidence_role": "retrospective_diagnosis_only_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    return payload, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Qlib direction behind a family-query impact gate."
    )
    parser.add_argument("--qlib-root", required=True)
    parser.add_argument("--family-query-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--placebo-samples", type=int, default=200)
    args = parser.parse_args()

    horizons = tuple(int(value) for value in args.horizons.split(","))
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique and non-empty")
    payload, daily = evaluate(
        qlib_root=Path(args.qlib_root),
        family_root=Path(args.family_query_root),
        horizons=horizons,
        seed=int(args.seed),
        bootstrap_samples=int(args.bootstrap_samples),
        placebo_samples=int(args.placebo_samples),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    daily.to_csv(output_dir / "test_role_predictions.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "DIAGNOSTIC_ONLY").touch()
    print(json.dumps(payload["test"], ensure_ascii=False))


if __name__ == "__main__":
    main()
