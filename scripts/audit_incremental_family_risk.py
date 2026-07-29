from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.evaluate_magnitude_risk_exposure import (
    _load_qlib,
    historical_stress_scores,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe {role}: {path}")
    return payload


def labeled_scores(root: Path, split: str) -> pd.DataFrame:
    daily_path = root / f"daily_{split}.csv"
    major_path = root / "major_trajectory" / "summary.json"
    daily = pd.read_csv(daily_path)
    required = {
        "date",
        "horizon",
        "actual_normalized_salience",
        "predicted_normalized_salience",
    }
    if not required.issubset(daily.columns):
        raise ValueError(f"risk daily schema mismatch: {daily_path}")
    if daily.duplicated(["date", "horizon"]).any():
        raise ValueError(f"duplicate risk daily keys: {daily_path}")
    major = _safe_json(major_path, "major summary")
    threshold = float(major["fit_major_event_threshold"])
    daily["date"] = pd.to_datetime(daily["date"])
    grouped = daily.groupby("date", sort=True).agg(
        risk_score=("predicted_normalized_salience", "max"),
        actual_salience=("actual_normalized_salience", "max"),
    )
    grouped["major_event"] = grouped["actual_salience"] >= threshold
    result = grouped.reset_index()
    numeric = result[["risk_score", "actual_salience"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"non-finite labeled risk scores: {daily_path}")
    return result


def fit_models(validation: pd.DataFrame, seed: int) -> dict[str, Any]:
    labels = validation["major_event"].to_numpy(dtype=bool)
    if len(validation) < 50 or np.unique(labels).size != 2:
        raise ValueError("validation data cannot fit an incremental classifier")

    def fit(columns: Sequence[str]):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=2000,
                random_state=int(seed),
            ),
        )
        model.fit(validation[list(columns)].to_numpy(dtype=np.float64), labels)
        return model

    return {
        "historical": fit(["historical"]),
        "historical_plus_family": fit(["historical", "family"]),
        "historical_plus_direct": fit(["historical", "direct"]),
    }


def predict_models(models: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[["date", "major_event"]].copy()
    columns = {
        "historical": ["historical"],
        "historical_plus_family": ["historical", "family"],
        "historical_plus_direct": ["historical", "direct"],
    }
    for name, selected in columns.items():
        result[name] = models[name].predict_proba(
            frame[selected].to_numpy(dtype=np.float64)
        )[:, 1]
    return result


def _metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    probability = np.asarray(probability, dtype=np.float64)
    return {
        "rows": int(len(labels)),
        "events": int(labels.sum()),
        "event_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, probability)),
        "average_precision": float(average_precision_score(labels, probability)),
        "average_precision_lift": float(
            average_precision_score(labels, probability) / labels.mean()
        ),
    }


def _pooled_auc(frame: pd.DataFrame, column: str) -> float:
    rank = frame.groupby("fold", sort=False)[column].rank(pct=True)
    return float(roc_auc_score(frame["major_event"], rank))


def _circular_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(math.ceil(length / float(block)))
    starts = rng.integers(0, length, size=blocks)
    return np.concatenate([(start + np.arange(block)) % length for start in starts])[
        :length
    ]


def bootstrap_auc_delta(
    folds: Mapping[str, pd.DataFrame],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(samples)):
        parts = []
        for fold, frame in folds.items():
            sampled = frame.iloc[_circular_indices(len(frame), int(block_length), rng)].copy()
            sampled["fold"] = fold
            parts.append(sampled)
        pooled = pd.concat(parts, ignore_index=True)
        if np.unique(pooled["major_event"]).size != 2:
            continue
        values.append(
            _pooled_auc(pooled, "historical_plus_family")
            - _pooled_auc(pooled, "historical")
        )
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 0.95 * int(samples):
        raise ValueError("too few valid incremental bootstrap samples")
    return {
        "samples": int(len(array)),
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _predict_augmented(model: Any, frame: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(
        frame[["historical", "family"]].to_numpy(dtype=np.float64)
    )[:, 1]


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "incremental_family_risk_diagnostic_contract":
        raise ValueError("invalid incremental risk contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe incremental risk contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"source pin mismatch: {relative}")

    fold_results = {}
    fold_predictions = {}
    placebo_inputs = {}
    inputs = {"contract": sha256(contract_path)}
    for fold, spec in contract["folds"].items():
        family_root = Path(spec["family_root"])
        direct_root = Path(spec["direct_root"])
        qlib_root = Path(spec["qlib_root"])
        family_summary = _safe_json(family_root / "summary.json", "family summary")
        direct_summary = _safe_json(direct_root / "summary.json", "direct summary")
        qlib_summary = _safe_json(qlib_root / "summary.json", "Qlib summary")
        expected_checkpoint = str(spec["checkpoint_sha256"])
        if {
            str(family_summary.get("parent_model_sha256")),
            str(direct_summary.get("parent_model_sha256")),
            str(qlib_summary.get("checkpoint_sha256")),
        } != {expected_checkpoint}:
            raise ValueError(f"checkpoint mismatch: {fold}")
        family_validation = labeled_scores(family_root, "validation").rename(
            columns={"risk_score": "family"}
        )
        family_test = labeled_scores(family_root, "test").rename(
            columns={"risk_score": "family"}
        )
        direct_validation = labeled_scores(direct_root, "validation").rename(
            columns={"risk_score": "direct"}
        )
        direct_test = labeled_scores(direct_root, "test").rename(
            columns={"risk_score": "direct"}
        )
        qlib_validation_path = qlib_root / "predictions_h1_valid.parquet"
        qlib_test_path = qlib_root / "predictions_h1_test.parquet"
        qlib_validation = _load_qlib(qlib_validation_path)
        qlib_test = _load_qlib(qlib_test_path)
        historical_validation, historical_test = historical_stress_scores(
            qlib_validation,
            qlib_test,
            window=int(contract["historical_stress_window"]),
        )
        historical_validation = historical_validation.rename(
            columns={"risk_score": "historical"}
        ).dropna()
        historical_test = historical_test.rename(columns={"risk_score": "historical"})

        def merge(
            family: pd.DataFrame,
            direct: pd.DataFrame,
            historical: pd.DataFrame,
        ) -> pd.DataFrame:
            selected = family[["date", "major_event", "family"]].merge(
                direct[["date", "major_event", "direct"]],
                on="date",
                suffixes=("", "_direct"),
                validate="one_to_one",
            )
            if not np.array_equal(selected["major_event"], selected["major_event_direct"]):
                raise ValueError(f"family/direct labels differ: {fold}")
            selected = selected.drop(columns="major_event_direct").merge(
                historical[["date", "historical"]], on="date", validate="one_to_one"
            )
            if not np.isfinite(
                selected[["family", "direct", "historical"]].to_numpy(dtype=np.float64)
            ).all():
                raise ValueError(f"non-finite incremental design: {fold}")
            return selected.sort_values("date").reset_index(drop=True)

        validation = merge(family_validation, direct_validation, historical_validation)
        test = merge(family_test, direct_test, historical_test)
        models = fit_models(validation, int(contract["seed"]))
        prediction = predict_models(models, test)
        prediction["fold"] = fold
        test_with_features = test.copy()
        test_with_features["historical"] = prediction["historical"]
        test_with_features["historical_plus_family"] = prediction[
            "historical_plus_family"
        ]
        test_with_features["historical_plus_direct"] = prediction[
            "historical_plus_direct"
        ]
        test_with_features["fold"] = fold
        fold_predictions[fold] = test_with_features
        placebo_inputs[fold] = (models["historical_plus_family"], test.copy())
        labels = test["major_event"].to_numpy(dtype=bool)
        metrics = {
            name: _metrics(labels, prediction[name].to_numpy(dtype=np.float64))
            for name in (
                "historical",
                "historical_plus_family",
                "historical_plus_direct",
            )
        }
        coefficient = models["historical_plus_family"].named_steps[
            "logisticregression"
        ].coef_[0]
        fold_results[fold] = {
            "validation_rows": int(len(validation)),
            "validation_events": int(validation["major_event"].sum()),
            "test_rows": int(len(test)),
            "test_events": int(test["major_event"].sum()),
            "metrics": metrics,
            "family_auc_delta": float(
                metrics["historical_plus_family"]["roc_auc"]
                - metrics["historical"]["roc_auc"]
            ),
            "direct_auc_delta": float(
                metrics["historical_plus_direct"]["roc_auc"]
                - metrics["historical"]["roc_auc"]
            ),
            "standardized_coefficients": {
                "historical": float(coefficient[0]),
                "family": float(coefficient[1]),
            },
        }
        for name, path in {
            f"{fold}.family_summary": family_root / "summary.json",
            f"{fold}.direct_summary": direct_root / "summary.json",
            f"{fold}.qlib_summary": qlib_root / "summary.json",
            f"{fold}.qlib_validation": qlib_validation_path,
            f"{fold}.qlib_test": qlib_test_path,
        }.items():
            inputs[name] = sha256(path)

    pooled = pd.concat(fold_predictions.values(), ignore_index=True)
    pooled_metrics = {
        name: {
            "fold_rank_auc": _pooled_auc(pooled, name),
            "average_precision": float(
                average_precision_score(pooled["major_event"], pooled[name])
            ),
        }
        for name in (
            "historical",
            "historical_plus_family",
            "historical_plus_direct",
        )
    }
    observed_delta = float(
        pooled_metrics["historical_plus_family"]["fold_rank_auc"]
        - pooled_metrics["historical"]["fold_rank_auc"]
    )
    uncertainty = bootstrap_auc_delta(
        fold_predictions,
        samples=int(contract["uncertainty"]["bootstrap_samples"]),
        block_length=int(contract["uncertainty"]["block_length_days"]),
        seed=int(contract["seed"]),
    )

    # Shift only the family input while retaining the validation-fitted augmented model.
    rng = np.random.default_rng(int(contract["seed"]) + 1000)
    placebo_auc = []
    for _ in range(int(contract["uncertainty"]["placebo_samples"])):
        parts = []
        for fold, (model, frame) in placebo_inputs.items():
            length = len(frame)
            minimum = int(contract["uncertainty"]["minimum_shift_days"])
            choices = np.arange(minimum, length - minimum + 1)
            shifted = frame.copy()
            shifted["family"] = np.roll(
                shifted["family"].to_numpy(dtype=np.float64), int(rng.choice(choices))
            )
            shifted["historical_plus_family"] = _predict_augmented(model, shifted)
            shifted["fold"] = fold
            parts.append(shifted)
        placebo_auc.append(
            _pooled_auc(pd.concat(parts, ignore_index=True), "historical_plus_family")
        )
    placebo_99 = float(np.quantile(placebo_auc, 0.99))
    checks = {
        "pooled_augmented_auc_improves_by_at_least": observed_delta
        >= float(contract["checks"]["pooled_augmented_auc_improvement_at_least"]),
        "bootstrap_auc_delta_lower_above_zero": float(uncertainty["lower_95"]) > 0.0,
        "augmented_auc_beats_shift_placebo_99": pooled_metrics[
            "historical_plus_family"
        ]["fold_rank_auc"]
        > placebo_99,
        "positive_family_coefficient_in_required_folds": sum(
            row["standardized_coefficients"]["family"] > 0.0
            for row in fold_results.values()
        )
        >= int(contract["checks"]["positive_family_coefficient_folds_at_least"]),
    }
    payload = {
        "status": "complete",
        "role": "incremental_family_risk_diagnostic",
        "folds": fold_results,
        "pooled": pooled_metrics,
        "observed_family_auc_delta": observed_delta,
        "uncertainty": uncertainty,
        "shift_placebo_auc_99": placebo_99,
        "checks": checks,
        "passed": bool(all(checks.values())),
        "decision": (
            "design_residual_family_innovation_readout"
            if all(checks.values())
            else "no_confirmed_incremental_family_information"
        ),
        "inputs": inputs,
        "test_used_for_selection": False,
        "evidence_role": "post_h1_diagnostic_only_no_promotion",
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    return payload, pooled.sort_values(["date", "fold"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit family-query information conditional on causal historical stress."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise SystemExit(f"refusing to overwrite incremental audit: {output_dir}")
    payload, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_incremental_risk.csv", index=False)
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
