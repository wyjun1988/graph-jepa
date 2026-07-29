from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.open_innovation import (
    OPEN_GAP_STATISTICS,
    build_jepa_open_innovation_design,
    build_open_sensor_design,
    fit_residual_jepa_pca_design,
    shuffled_feature_block,
)
from stock_v2.systemic_head import SYSTEMIC_COMPONENT_TARGETS
from stock_v2.systemic_transition import (
    DEFAULT_SYSTEMIC_STATE_FEATURES,
    binary_ranking_metrics,
    derived_subtype_scores,
    event_labels,
    fit_systemic_calibration,
    score_systemic_components,
    transition_components,
)


MODEL_CONFIGS = {
    "shallow": {
        "num_leaves": 7,
        "max_depth": 3,
        "min_child_samples": 40,
        "learning_rate": 0.025,
        "n_estimators": 700,
        "reg_alpha": 1.0,
        "reg_lambda": 12.0,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
    },
    "medium": {
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 30,
        "learning_rate": 0.02,
        "n_estimators": 850,
        "reg_alpha": 1.0,
        "reg_lambda": 16.0,
        "feature_fraction": 0.55,
        "bagging_fraction": 0.80,
        "bagging_freq": 1,
    },
}
SPLIT_NAMES = ("fit", "validation", "test")
SUBTYPE_NAMES = ("broad_selloff", "turnover_explosion", "graph_state_shift")
OPEN_NOWCAST_TARGET_VERSION = "krx_open_unknown_impact_v1_20260715"
OPEN_NOWCAST_STATE_FEATURES = tuple(
    name
    for name in DEFAULT_SYSTEMIC_STATE_FEATURES
    if name != "gap_open" and not name.startswith("investor_")
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    data = np.ascontiguousarray(values).view(np.uint8).ravel()
    for start in range(0, len(data), 1024 * 1024):
        digest.update(data[start : start + 1024 * 1024])
    return digest.hexdigest()


def parse_int_list(text: str) -> list[int]:
    result = [int(value.strip()) for value in str(text).split(",") if value.strip()]
    if not result:
        raise ValueError("at least one integer is required")
    return result


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3 or left[valid].std() <= 1e-12 or right[valid].std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left[valid], right[valid])[0, 1])


def _split_rows(splits: Mapping[str, np.ndarray]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ordered = []
    rows: dict[str, np.ndarray] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        values = np.asarray(splits[name], dtype=np.int64)
        ordered.append(values)
        rows[name] = np.arange(cursor, cursor + len(values), dtype=np.int64)
        cursor += len(values)
    steps = np.concatenate(ordered)
    if len(np.unique(steps)) != len(steps):
        raise ValueError("open nowcast splits overlap")
    return steps, rows


def load_forecast_state(
    cache_dir: Path,
    model_dir: Path,
    requested_steps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    contract = json.loads((cache_dir / "contract.json").read_text(encoding="utf-8"))
    if not (cache_dir / "CACHE_COMPLETE").exists():
        raise ValueError("JEPA forecast cache is incomplete")
    checkpoint_hash = sha256_file(model_dir / "graph_jepa_real.pt")
    if contract.get("checkpoint_sha256") != checkpoint_hash:
        raise ValueError("JEPA forecast cache checkpoint differs from the model")
    start = int(contract["step_start"])
    end = int(contract["step_end"])
    cache_steps = np.arange(start, end + 1, dtype=np.int64)
    if len(cache_steps) != int(contract["rows"]):
        raise ValueError("JEPA cache requires contiguous chronology steps")
    lookup = {int(step): index for index, step in enumerate(cache_steps)}
    try:
        rows = np.asarray([lookup[int(step)] for step in requested_steps], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"requested step is absent from the JEPA cache: {error}") from error
    state = np.load(cache_dir / "state_h1.npy", mmap_mode="r")
    expected = (
        int(contract["rows"]),
        int(contract["node_count"]),
        len(contract["eligible_indices"]),
    )
    if state.shape != expected or state.dtype != np.float16:
        raise ValueError(f"JEPA h1 state cache shape/dtype differs from {expected}/float16")
    return np.asarray(state[rows], dtype=np.float32), rows, contract


def _open_actual_rows(features, steps: Sequence[int], split: str):
    stock_count = int(features.tradable_count)
    return_index = features.feature_names.index("return_1d")
    rows = []
    for step in np.asarray(steps, dtype=np.int64):
        target_step = int(step) + 1
        current_state = features.features[int(step), :stock_count]
        future_state = features.features[target_step, :stock_count]
        current_raw = features.raw_features[int(step), :stock_count]
        future_raw = features.raw_features[target_step, :stock_count]
        current_available = features.available_mask[int(step), :stock_count] > 0.5
        future_available = features.available_mask[target_step, :stock_count] > 0.5
        path = np.asarray(
            features.target_return_paths[1][int(step), :stock_count],
            dtype=np.float64,
        )
        node_mask = (
            current_available[:, return_index]
            & future_available[:, return_index]
            & np.isfinite(path)
        )
        components = transition_components(
            current_state=current_state,
            future_state=future_state,
            current_raw=current_raw,
            future_raw=future_raw,
            current_available=current_available,
            future_available=future_available,
            feature_names=features.feature_names,
            entry_path_returns=path,
            node_mask=node_mask,
            state_feature_names=OPEN_NOWCAST_STATE_FEATURES,
        )
        rows.append(
            {
                "split": str(split),
                "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                "target_date": str(pd.Timestamp(features.dates[target_step]).date()),
                "step": int(step),
                "horizon": 1,
                **components,
            }
        )
    return rows


def build_targets(features, splits: Mapping[str, np.ndarray], split_rows):
    rows = []
    for split in SPLIT_NAMES:
        selected = _open_actual_rows(features, splits[split], split)
        expected_steps = np.asarray(splits[split], dtype=np.int64)
        actual_steps = np.asarray([int(row["step"]) for row in selected], dtype=np.int64)
        if not np.array_equal(actual_steps, expected_steps):
            raise ValueError(f"{split} target rows do not align with origin steps")
        rows.extend(selected)
    fit = [rows[int(index)] for index in split_rows["fit"]]
    calibration = fit_systemic_calibration(fit)
    component = np.asarray(
        [[float(row[name]) for name in SYSTEMIC_COMPONENT_TARGETS] for row in rows],
        dtype=np.float64,
    )
    energy = np.asarray(
        [score_systemic_components(row, calibration)["systemic_energy"] for row in rows],
        dtype=np.float64,
    )
    labels = {
        name: np.asarray(
            [bool(event_labels(row, calibration)[name]) for row in rows], dtype=bool
        )
        for name in ("systemic_event", *SUBTYPE_NAMES)
    }
    fit_label_rates = {
        name: float(labels[name][split_rows["fit"]].mean())
        for name in ("systemic_event", *SUBTYPE_NAMES)
    }
    threshold = max(float(calibration.event_threshold), 1e-8)
    sample_weight = 1.0 + 3.0 * np.minimum(np.maximum(energy / threshold, 0.0), 3.0)
    return (
        rows,
        calibration,
        component,
        energy,
        labels,
        fit_label_rates,
        sample_weight,
    )


def prepare_design(values: np.ndarray, fit_rows: np.ndarray):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("open nowcast design must be a finite row matrix")
    fit = values[np.asarray(fit_rows, dtype=np.int64)].astype(np.float64)
    variance = fit.var(axis=0)
    keep = np.isfinite(variance) & (variance > 1e-10)
    if not keep.any():
        raise ValueError("open nowcast design has no varying fit features")
    return values[:, keep], keep


def train_predictions(
    design: np.ndarray,
    components: np.ndarray,
    energy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    sample_weight: np.ndarray,
    split_rows: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    seed: int,
    num_threads: int,
):
    import lightgbm as lgb

    fit_rows = np.asarray(split_rows["fit"], dtype=np.int64)
    validation_rows = np.asarray(split_rows["validation"], dtype=np.int64)
    common = {
        "objective": "huber",
        "max_bin": 63,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": int(num_threads),
        **dict(config),
    }
    component_prediction = np.full_like(components, np.nan, dtype=np.float64)
    best_iterations: dict[str, int] = {}
    for target_index, target_name in enumerate(SYSTEMIC_COMPONENT_TARGETS):
        target = components[:, target_index]
        fit_valid = fit_rows[np.isfinite(target[fit_rows])]
        validation_valid = validation_rows[np.isfinite(target[validation_rows])]
        if len(fit_valid) < 40 or len(validation_valid) < 20:
            raise ValueError(f"too few finite rows for component {target_name}")
        model = lgb.LGBMRegressor(
            random_state=int(seed) + target_index,
            **common,
        )
        model.fit(
            design[fit_valid],
            target[fit_valid],
            sample_weight=sample_weight[fit_valid],
            eval_set=[(design[validation_valid], target[validation_valid])],
            eval_sample_weight=[sample_weight[validation_valid]],
            eval_metric="l1",
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        component_prediction[:, target_index] = model.booster_.predict(
            design, num_iteration=model.best_iteration_
        )
        best_iterations[target_name] = int(model.best_iteration_ or config["n_estimators"])

    log_energy = np.log1p(np.maximum(energy, 0.0))
    valid_fit = fit_rows[np.isfinite(log_energy[fit_rows])]
    valid_validation = validation_rows[np.isfinite(log_energy[validation_rows])]
    energy_model = lgb.LGBMRegressor(random_state=int(seed) + 1009, **common)
    energy_model.fit(
        design[valid_fit],
        log_energy[valid_fit],
        sample_weight=sample_weight[valid_fit],
        eval_set=[(design[valid_validation], log_energy[valid_validation])],
        eval_sample_weight=[sample_weight[valid_validation]],
        eval_metric="l1",
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    predicted_energy = np.maximum(
        np.expm1(
            energy_model.booster_.predict(
                design, num_iteration=energy_model.best_iteration_
            )
        ),
        0.0,
    )
    best_iterations["systemic_energy"] = int(
        energy_model.best_iteration_ or config["n_estimators"]
    )
    subtype_scores: dict[str, np.ndarray] = {}
    classifier_common = {
        **common,
        "objective": "binary",
        "metric": "average_precision",
    }
    for subtype_index, subtype_name in enumerate(SUBTYPE_NAMES):
        target = np.asarray(labels[subtype_name], dtype=np.int8)
        fit_rate = float(target[fit_rows].mean())
        if not 0.0 < fit_rate < 1.0 or int(target[fit_rows].sum()) < 10:
            raise ValueError(f"too few fit events for subtype {subtype_name}")
        balanced_weight = sample_weight * np.where(
            target > 0,
            0.5 / fit_rate,
            0.5 / (1.0 - fit_rate),
        )
        model = lgb.LGBMClassifier(
            random_state=int(seed) + 2001 + subtype_index,
            **classifier_common,
        )
        model.fit(
            design[fit_rows],
            target[fit_rows],
            sample_weight=balanced_weight[fit_rows],
            eval_set=[(design[validation_rows], target[validation_rows])],
            eval_sample_weight=[balanced_weight[validation_rows]],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(60, first_metric_only=True, verbose=False)],
        )
        subtype_scores[subtype_name] = np.asarray(
            model.booster_.predict(
                design,
                num_iteration=model.best_iteration_,
                raw_score=True,
            ),
            dtype=np.float64,
        )
        best_iterations[f"classifier:{subtype_name}"] = int(
            model.best_iteration_ or config["n_estimators"]
        )
    return component_prediction, predicted_energy, subtype_scores, best_iterations


def refit_predictions(
    design: np.ndarray,
    components: np.ndarray,
    energy: np.ndarray,
    labels: Mapping[str, np.ndarray],
    sample_weight: np.ndarray,
    development_rows: Sequence[int],
    config: Mapping[str, Any],
    best_iterations: Mapping[str, int],
    *,
    seed: int,
    num_threads: int,
):
    """Refit fixed validation-selected tree counts on all pre-test rows."""

    import lightgbm as lgb

    rows = np.asarray(development_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) < 100:
        raise ValueError("final refit requires at least one hundred development rows")
    base = {
        "objective": "huber",
        "max_bin": 63,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
        "n_jobs": int(num_threads),
        **dict(config),
    }
    base.pop("n_estimators", None)
    component_prediction = np.full_like(components, np.nan, dtype=np.float64)
    for target_index, target_name in enumerate(SYSTEMIC_COMPONENT_TARGETS):
        target = components[:, target_index]
        valid = rows[np.isfinite(target[rows])]
        model = lgb.LGBMRegressor(
            random_state=int(seed) + target_index,
            n_estimators=max(1, int(best_iterations[target_name])),
            **base,
        )
        model.fit(
            design[valid],
            target[valid],
            sample_weight=sample_weight[valid],
        )
        component_prediction[:, target_index] = model.booster_.predict(design)

    log_energy = np.log1p(np.maximum(energy, 0.0))
    valid = rows[np.isfinite(log_energy[rows])]
    energy_model = lgb.LGBMRegressor(
        random_state=int(seed) + 1009,
        n_estimators=max(1, int(best_iterations["systemic_energy"])),
        **base,
    )
    energy_model.fit(
        design[valid],
        log_energy[valid],
        sample_weight=sample_weight[valid],
    )
    predicted_energy = np.maximum(
        np.expm1(energy_model.booster_.predict(design)), 0.0
    )

    subtype_scores: dict[str, np.ndarray] = {}
    classifier_base = {**base, "objective": "binary", "metric": "average_precision"}
    for subtype_index, subtype_name in enumerate(SUBTYPE_NAMES):
        target = np.asarray(labels[subtype_name], dtype=np.int8)
        event_rate = float(target[rows].mean())
        if not 0.0 < event_rate < 1.0:
            raise ValueError(f"invalid refit event rate for subtype {subtype_name}")
        balanced_weight = sample_weight * np.where(
            target > 0,
            0.5 / event_rate,
            0.5 / (1.0 - event_rate),
        )
        model = lgb.LGBMClassifier(
            random_state=int(seed) + 2001 + subtype_index,
            n_estimators=max(
                1, int(best_iterations[f"classifier:{subtype_name}"])
            ),
            **classifier_base,
        )
        model.fit(
            design[rows],
            target[rows],
            sample_weight=balanced_weight[rows],
        )
        subtype_scores[subtype_name] = np.asarray(
            model.booster_.predict(design, raw_score=True), dtype=np.float64
        )
    return component_prediction, predicted_energy, subtype_scores


def evaluate_split(
    actual_rows,
    calibration,
    actual_components,
    actual_energy,
    labels,
    fit_label_rates,
    predicted_components,
    predicted_energy,
    subtype_scores,
    rows: Sequence[int],
) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    event = labels["systemic_event"][rows]
    ranking = binary_ranking_metrics(
        event,
        predicted_energy[rows],
        selection_rate=float(calibration.fit_event_rate),
    )
    selected_count = int(ranking["selected_count"])
    selected = rows[
        np.argsort(predicted_energy[rows], kind="mergesort")[-selected_count:]
    ]
    total_mass = float(np.nansum(actual_energy[rows]))
    actual_return = actual_components[rows, 0]
    predicted_return = predicted_components[rows, 0]
    direction_valid = event & np.isfinite(actual_return) & np.isfinite(predicted_return)
    direction = float("nan")
    if direction_valid.any():
        correct = np.sign(actual_return[direction_valid]) == np.sign(
            predicted_return[direction_valid]
        )
        direction = float(
            np.average(
                correct.astype(np.float64),
                weights=actual_energy[rows][direction_valid],
            )
        )
    ranking.update(
        {
            "fit_selection_rate": float(calibration.fit_event_rate),
            "average_precision_lift": (
                float(ranking["average_precision"]) / float(ranking["event_rate"])
                if float(ranking["event_rate"]) > 0.0
                else float("nan")
            ),
            "systemic_energy_correlation": _pearson(
                predicted_energy[rows], actual_energy[rows]
            ),
            "tail_mass_recall_at_fit_event_rate": (
                float(np.nansum(actual_energy[selected]) / total_mass)
                if total_mass > 1e-12
                else float("nan")
            ),
            "event_impact_weighted_market_direction_accuracy": direction,
        }
    )
    subtype_metrics = {}
    derived_subtype_metrics = {}
    for name in SUBTYPE_NAMES:
        derived_scores = np.asarray(
            [
                derived_subtype_scores(
                    {
                        component: float(predicted_components[index, position])
                        for position, component in enumerate(SYSTEMIC_COMPONENT_TARGETS)
                    },
                    calibration,
                )[name]
                for index in rows
            ],
            dtype=np.float64,
        )
        selection_rate = max(float(fit_label_rates[name]), 1e-6)
        subtype_metrics[name] = binary_ranking_metrics(
            labels[name][rows],
            np.asarray(subtype_scores[name], dtype=np.float64)[rows],
            selection_rate=selection_rate,
        )
        derived_subtype_metrics[name] = binary_ranking_metrics(
            labels[name][rows],
            derived_scores,
            selection_rate=selection_rate,
        )
    component_metrics = {}
    for index, name in enumerate(SYSTEMIC_COMPONENT_TARGETS):
        actual = actual_components[rows, index]
        predicted = predicted_components[rows, index]
        valid = np.isfinite(actual) & np.isfinite(predicted)
        component_metrics[name] = {
            "count": int(valid.sum()),
            "mae": float(np.abs(actual[valid] - predicted[valid]).mean()),
            "correlation": _pearson(actual[valid], predicted[valid]),
        }
    return_metrics = {
        "correlation": _pearson(actual_return, predicted_return),
        "mae": float(np.nanmean(np.abs(actual_return - predicted_return))),
        "direction_accuracy": float(
            np.nanmean((np.sign(actual_return) == np.sign(predicted_return)).astype(float))
        ),
    }
    return {
        "energy": ranking,
        "subtypes": subtype_metrics,
        "derived_subtypes": derived_subtype_metrics,
        "components": component_metrics,
        "open_to_close_market_return": return_metrics,
    }


def composite_score(metrics: Mapping[str, Any]) -> float:
    primary = metrics["energy"]
    subtype_auc = np.nanmean(
        [float(value["roc_auc"]) for value in metrics["subtypes"].values()]
    )
    selection_rate = float(primary["fit_selection_rate"])
    mass = float(primary["tail_mass_recall_at_fit_event_rate"])
    terms = {
        "auc": np.clip(2.0 * (float(primary["roc_auc"]) - 0.5), -1.0, 1.0),
        "ap": np.clip(float(primary["average_precision_lift"]) - 1.0, -1.0, 1.0),
        "mass": np.clip(
            (mass - selection_rate) / max(1.0 - selection_rate, 1e-6), -1.0, 1.0
        ),
        "energy_corr": np.clip(
            float(primary["systemic_energy_correlation"]), -1.0, 1.0
        ),
        "subtypes": np.clip(2.0 * (float(subtype_auc) - 0.5), -1.0, 1.0),
        "direction": np.clip(
            2.0
            * (
                float(primary["event_impact_weighted_market_direction_accuracy"])
                - 0.5
            ),
            -1.0,
            1.0,
        ),
        "return_corr": np.clip(
            float(metrics["open_to_close_market_return"]["correlation"]), -1.0, 1.0
        ),
    }
    return float(
        0.25 * terms["auc"]
        + 0.15 * terms["ap"]
        + 0.15 * terms["mass"]
        + 0.15 * terms["energy_corr"]
        + 0.10 * terms["subtypes"]
        + 0.10 * terms["direction"]
        + 0.10 * terms["return_corr"]
    )


def validation_head_scores(metrics: Mapping[str, Any]) -> dict[str, float]:
    energy = metrics["energy"]
    energy_score = (
        0.40 * np.clip(2.0 * (float(energy["roc_auc"]) - 0.5), -1.0, 1.0)
        + 0.30
        * np.clip(float(energy["systemic_energy_correlation"]), -1.0, 1.0)
        + 0.15
        * np.clip(
            2.0
            * (
                float(energy["event_impact_weighted_market_direction_accuracy"])
                - 0.5
            ),
            -1.0,
            1.0,
        )
        + 0.15
        * np.clip(float(energy["tail_mass_recall_at_fit_event_rate"]), 0.0, 1.0)
    )
    scores = {"energy": float(energy_score)}
    for name, row in metrics["components"].items():
        score = float(row["correlation"])
        scores[f"component:{name}"] = score if np.isfinite(score) else -1.0
    for name, row in metrics["subtypes"].items():
        score = 2.0 * (float(row["roc_auc"]) - 0.5)
        scores[f"subtype:{name}"] = float(np.clip(score, -1.0, 1.0))
    return scores


def select_placebo_guarded_heads(
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Any],
    placebo_metrics: Sequence[Mapping[str, Any]],
    *,
    minimum_margin: float = 0.01,
) -> dict[str, dict[str, Any]]:
    if len(placebo_metrics) < 5:
        raise ValueError("modular JEPA selection requires at least five placebos")
    baseline = validation_head_scores(baseline_metrics)
    candidate = validation_head_scores(candidate_metrics)
    placebo = [validation_head_scores(value) for value in placebo_metrics]
    if set(baseline) != set(candidate) or any(set(value) != set(baseline) for value in placebo):
        raise ValueError("modular head score contracts differ across variants")
    selected = {}
    for name in baseline:
        placebo_max = max(float(value[name]) for value in placebo)
        candidate_score = float(candidate[name])
        reference = max(float(baseline[name]), placebo_max)
        use_jepa = candidate_score >= reference + float(minimum_margin)
        selected[name] = {
            "source": "open_sensors_plus_jepa" if use_jepa else "open_sensors",
            "candidate_validation_score": candidate_score,
            "baseline_validation_score": float(baseline[name]),
            "maximum_placebo_validation_score": placebo_max,
            "candidate_margin_over_reference": candidate_score - reference,
        }
    return selected


def compose_modular_predictions(
    predictions: Mapping[str, tuple[np.ndarray, np.ndarray, Mapping[str, np.ndarray]]],
    selection: Mapping[str, Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    required = {"open_sensors", "open_sensors_plus_jepa"}
    if not required.issubset(predictions):
        raise ValueError("modular composition requires baseline and JEPA predictions")
    baseline = predictions["open_sensors"]
    candidate = predictions["open_sensors_plus_jepa"]
    components = np.asarray(baseline[0], dtype=np.float64).copy()
    for index, name in enumerate(SYSTEMIC_COMPONENT_TARGETS):
        key = f"component:{name}"
        source = str(selection[key]["source"])
        components[:, index] = np.asarray(predictions[source][0])[:, index]
    energy_source = str(selection["energy"]["source"])
    energy = np.asarray(predictions[energy_source][1], dtype=np.float64).copy()
    subtypes = {}
    for name in SUBTYPE_NAMES:
        source = str(selection[f"subtype:{name}"]["source"])
        subtypes[name] = np.asarray(predictions[source][2][name], dtype=np.float64).copy()
    if components.shape != np.asarray(candidate[0]).shape:
        raise ValueError("baseline and JEPA component predictions do not align")
    return components, energy, subtypes


def absolute_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    energy = metrics["energy"]
    subtype_auc = [float(value["roc_auc"]) for value in metrics["subtypes"].values()]
    checks = {
        "event_auc_at_least_0_60": float(energy["roc_auc"]) >= 0.60,
        "event_ap_lift_at_least_1_50": float(energy["average_precision_lift"]) >= 1.50,
        "energy_correlation_at_least_0_15": float(
            energy["systemic_energy_correlation"]
        )
        >= 0.15,
        "tail_mass_recall_at_least_0_20": float(
            energy["tail_mass_recall_at_fit_event_rate"]
        )
        >= 0.20,
        "impact_direction_at_least_0_55": float(
            energy["event_impact_weighted_market_direction_accuracy"]
        )
        >= 0.55,
        "broad_selloff_recall_at_least_0_25": float(
            metrics["subtypes"]["broad_selloff"]["recall_at_selection_rate"]
        )
        >= 0.25,
        "all_subtype_auc_at_least_0_52": min(subtype_auc) >= 0.52,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
    }


def _write_daily(
    path: Path,
    split_rows: Mapping[str, np.ndarray],
    actual_rows,
    actual_energy,
    labels,
    predicted_components,
    predicted_energy,
    subtype_scores,
    *,
    splits: Sequence[str] = SPLIT_NAMES,
) -> None:
    output = []
    for split in splits:
        if split not in SPLIT_NAMES:
            raise ValueError(f"unknown daily prediction split: {split}")
        for index in split_rows[split]:
            row = actual_rows[int(index)]
            output.append(
                {
                    "split": split,
                    "date": row["date"],
                    "target_date": row["target_date"],
                    "step": int(row["step"]),
                    "actual_systemic_energy": float(actual_energy[int(index)]),
                    "predicted_systemic_energy": float(predicted_energy[int(index)]),
                    **{
                        f"actual_{name}": bool(labels[name][int(index)])
                        for name in ("systemic_event", *SUBTYPE_NAMES)
                    },
                    **{
                        f"score_{name}": float(subtype_scores[name][int(index)])
                        for name in SUBTYPE_NAMES
                    },
                    **{
                        f"actual_{name}": float(row[name])
                        for name in SYSTEMIC_COMPONENT_TARGETS
                    },
                    **{
                        f"predicted_{name}": float(predicted_components[int(index), position])
                        for position, name in enumerate(SYSTEMIC_COMPONENT_TARGETS)
                    },
                }
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a causal KRX-open learned JEPA innovation nowcast."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forecast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--split-horizons", default="1,2,3,5,10")
    parser.add_argument("--configs", default="shallow,medium")
    parser.add_argument("--placebo-seeds", default="101,211,307,401,503")
    parser.add_argument(
        "--jepa-feature-mode",
        choices=("raw", "sensor_residual_pca"),
        default="raw",
    )
    parser.add_argument("--jepa-pca-rank", type=int, default=32)
    parser.add_argument("--sensor-pca-rank", type=int, default=64)
    parser.add_argument("--projection-ridge-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    cache_dir = Path(args.forecast_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_names = [value.strip() for value in args.configs.split(",") if value.strip()]
    unknown = [name for name in config_names if name not in MODEL_CONFIGS]
    if unknown:
        raise ValueError(f"unknown model configs: {unknown}")
    placebo_seeds = parse_int_list(args.placebo_seeds)

    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    args.horizons = args.split_horizons
    feature_args = evaluator_namespace(args)
    feature_args.horizons = args.split_horizons
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    split_horizons = parse_int_list(args.split_horizons)
    splits = _split_steps(
        features, checkpoint_args, split_horizons, int(args.validation_days)
    )
    steps, split_rows = _split_rows(splits)
    predicted_state, cache_rows, cache_contract = load_forecast_state(
        cache_dir, model_dir, steps
    )

    sensor = build_open_sensor_design(features, steps)
    jepa = build_jepa_open_innovation_design(
        features,
        steps,
        predicted_state,
        cache_contract["eligible_indices"],
    )
    if args.jepa_feature_mode == "sensor_residual_pca":
        projected_jepa = fit_residual_jepa_pca_design(
            sensor.values,
            jepa.values,
            split_rows["fit"],
            rank=int(args.jepa_pca_rank),
            sensor_rank=int(args.sensor_pca_rank),
            ridge_alpha=float(args.projection_ridge_alpha),
        )
        jepa_values = projected_jepa.values
        jepa_feature_names = projected_jepa.feature_names
        jepa_feature_contract = dict(projected_jepa.contract)
    else:
        jepa_values = jepa.values
        jepa_feature_names = jepa.feature_names
        jepa_feature_contract = {
            "fit_only": False,
            "target_used_for_projection": False,
            "mode": "raw_open_innovation_design",
        }
    variants = {"open_sensors": sensor.values}
    variants["open_sensors_plus_jepa"] = np.concatenate(
        (sensor.values, jepa_values), axis=1
    )
    for seed in placebo_seeds:
        variants[f"open_sensors_plus_shuffled_jepa_seed{seed}"] = np.concatenate(
            (
                sensor.values,
                shuffled_feature_block(jepa_values, split_rows, seed=int(seed)),
            ),
            axis=1,
        )

    (
        actual_rows,
        calibration,
        components,
        energy,
        labels,
        fit_label_rates,
        sample_weight,
    ) = build_targets(features, splits, split_rows)
    results: dict[str, Any] = {}
    validation_predictions_by_variant = {}
    refit_predictions_by_variant = {}
    for variant, raw_design in variants.items():
        design, keep = prepare_design(raw_design, split_rows["fit"])
        config_results = {}
        config_predictions = {}
        for config_index, config_name in enumerate(config_names):
            (
                predicted_components,
                predicted_energy,
                subtype_scores,
                iterations,
            ) = train_predictions(
                design,
                components,
                energy,
                labels,
                sample_weight,
                split_rows,
                MODEL_CONFIGS[config_name],
                seed=int(args.seed) + 100 * config_index,
                num_threads=int(args.num_threads),
            )
            metrics = {
                "validation": evaluate_split(
                    actual_rows,
                    calibration,
                    components,
                    energy,
                    labels,
                    fit_label_rates,
                    predicted_components,
                    predicted_energy,
                    subtype_scores,
                    split_rows["validation"],
                )
            }
            config_results[config_name] = {
                "validation_score": composite_score(metrics["validation"]),
                "metrics": metrics,
                "best_iterations": iterations,
            }
            config_predictions[config_name] = (
                predicted_components,
                predicted_energy,
                subtype_scores,
            )
            print(
                f"variant={variant} config={config_name} "
                f"validation={config_results[config_name]['validation_score']:.6f}",
                flush=True,
            )
        selected_config = max(
            config_names, key=lambda name: float(config_results[name]["validation_score"])
        )
        selected_validation = config_results[selected_config]
        validation_predictions_by_variant[variant] = config_predictions[selected_config]
        development_rows = np.concatenate(
            (split_rows["fit"], split_rows["validation"])
        )
        refit = refit_predictions(
            design,
            components,
            energy,
            labels,
            sample_weight,
            development_rows,
            MODEL_CONFIGS[selected_config],
            selected_validation["best_iterations"],
            seed=int(args.seed) + 100 * config_names.index(selected_config),
            num_threads=int(args.num_threads),
        )
        test_metrics = evaluate_split(
            actual_rows,
            calibration,
            components,
            energy,
            labels,
            fit_label_rates,
            refit[0],
            refit[1],
            refit[2],
            split_rows["test"],
        )
        selected_result = {
            "config": selected_config,
            "validation_score": float(selected_validation["validation_score"]),
            "test_score": composite_score(test_metrics),
            "metrics": {
                "validation": selected_validation["metrics"]["validation"],
                "test": test_metrics,
            },
            "best_iterations": selected_validation["best_iterations"],
            "refit_dates": int(len(development_rows)),
            "refit_uses_test_rows": False,
        }
        results[variant] = {
            "raw_features": int(raw_design.shape[1]),
            "retained_features": int(keep.sum()),
            "retained_feature_mask_sha256": sha256_array(keep),
            "validation_selected_config": selected_config,
            "selected": selected_result,
            "all_configs": config_results,
        }
        print(
            f"variant={variant} selected={selected_config} "
            f"refit_test={selected_result['test_score']:.6f}",
            flush=True,
        )
        refit_predictions_by_variant[variant] = refit

    candidate = results["open_sensors_plus_jepa"]["selected"]
    baseline = results["open_sensors"]["selected"]
    placebo = [
        results[f"open_sensors_plus_shuffled_jepa_seed{seed}"]["selected"]
        for seed in placebo_seeds
    ]
    placebo_validation = np.asarray(
        [float(item["validation_score"]) for item in placebo], dtype=np.float64
    )
    placebo_test = np.asarray(
        [float(item["test_score"]) for item in placebo], dtype=np.float64
    )
    absolute = absolute_gate(candidate["metrics"]["test"])
    advantage_checks = {
        "validation_beats_open_sensor_baseline": float(candidate["validation_score"])
        > float(baseline["validation_score"]),
        "validation_beats_every_shuffled_placebo": float(candidate["validation_score"])
        > float(placebo_validation.max()),
        "test_beats_open_sensor_baseline": float(candidate["test_score"])
        > float(baseline["test_score"]),
        "test_beats_95pct_shuffled_placebo": float(candidate["test_score"])
        > float(np.quantile(placebo_test, 0.95)),
    }
    advantage = {
        "passed": all(advantage_checks.values()),
        "checks": advantage_checks,
        "failures": [name for name, passed in advantage_checks.items() if not passed],
        "candidate_validation_score": float(candidate["validation_score"]),
        "baseline_validation_score": float(baseline["validation_score"]),
        "placebo_validation_scores": placebo_validation.tolist(),
        "candidate_test_score": float(candidate["test_score"]),
        "baseline_test_score": float(baseline["test_score"]),
        "placebo_test_scores": placebo_test.tolist(),
    }
    gate = {
        "passed": bool(absolute["passed"] and advantage["passed"]),
        "absolute": absolute,
        "jepa_specific_advantage": advantage,
        "requirements": (
            "the validation-selected learned open nowcast must pass every absolute h1 "
            "impact check and beat both the direct open-sensor model and split-local "
            "shuffled JEPA placebos"
        ),
    }
    placebo_names = [
        f"open_sensors_plus_shuffled_jepa_seed{seed}" for seed in placebo_seeds
    ]
    head_selection = select_placebo_guarded_heads(
        baseline["metrics"]["validation"],
        candidate["metrics"]["validation"],
        [results[name]["selected"]["metrics"]["validation"] for name in placebo_names],
        minimum_margin=0.01,
    )
    modular_validation_predictions = compose_modular_predictions(
        validation_predictions_by_variant,
        head_selection,
    )
    modular_test_predictions = compose_modular_predictions(
        refit_predictions_by_variant,
        head_selection,
    )
    modular_validation_metrics = evaluate_split(
        actual_rows,
        calibration,
        components,
        energy,
        labels,
        fit_label_rates,
        modular_validation_predictions[0],
        modular_validation_predictions[1],
        modular_validation_predictions[2],
        split_rows["validation"],
    )
    modular_test_metrics = evaluate_split(
        actual_rows,
        calibration,
        components,
        energy,
        labels,
        fit_label_rates,
        modular_test_predictions[0],
        modular_test_predictions[1],
        modular_test_predictions[2],
        split_rows["test"],
    )
    modular_validation_score = composite_score(modular_validation_metrics)
    modular_test_score = composite_score(modular_test_metrics)
    enabled_jepa_heads = [
        name
        for name, row in head_selection.items()
        if row["source"] == "open_sensors_plus_jepa"
    ]
    modular_absolute = absolute_gate(modular_test_metrics)
    modular_checks = {
        "at_least_one_placebo_guarded_jepa_head": bool(enabled_jepa_heads),
        "validation_score_not_below_direct": modular_validation_score
        >= float(baseline["validation_score"]),
        "test_score_not_below_direct": modular_test_score
        >= float(baseline["test_score"]),
        "absolute_test_gate_passes": bool(modular_absolute["passed"]),
    }
    modular = {
        "selection_rule": (
            "per-head candidate validation score must exceed direct and every "
            "split-local shuffled placebo by at least 0.01"
        ),
        "minimum_validation_margin": 0.01,
        "head_selection": head_selection,
        "enabled_jepa_heads": enabled_jepa_heads,
        "validation_score": modular_validation_score,
        "test_score": modular_test_score,
        "metrics": {
            "validation": modular_validation_metrics,
            "test": modular_test_metrics,
        },
        "absolute_gate": modular_absolute,
        "gate": {
            "passed": all(modular_checks.values()),
            "checks": modular_checks,
            "failures": [
                name for name, passed in modular_checks.items() if not passed
            ],
        },
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }
    summary = {
        "schema_version": 5 if args.jepa_feature_mode != "raw" else 4,
        "status": "complete",
        "role": "research_only_krx_open_learned_jepa_innovation_nowcast",
        "target_version": OPEN_NOWCAST_TARGET_VERSION,
        "target_state_features": list(OPEN_NOWCAST_STATE_FEATURES),
        "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "forecast_cache_contract": cache_contract,
        "forecast_cache_rows_sha256": sha256_array(cache_rows),
        "stocks": int(features.tradable_count),
        "nodes": int(features.node_count),
        "state_features": len(features.feature_names),
        "split_dates": {name: int(len(splits[name])) for name in SPLIT_NAMES},
        "open_sensor_features": int(sensor.values.shape[1]),
        "open_sensor_contract": "raw_open_gap_breadth_v2",
        "raw_open_gap_statistics": list(OPEN_GAP_STATISTICS),
        "jepa_feature_mode": str(args.jepa_feature_mode),
        "jepa_feature_contract": jepa_feature_contract,
        "jepa_raw_innovation_features": int(jepa.values.shape[1]),
        "jepa_innovation_features": int(jepa_values.shape[1]),
        "jepa_feature_names": list(jepa_feature_names),
        "current_open_sensor_names": list(sensor.current_feature_names),
        "current_jepa_innovation_names": list(jepa.current_feature_names),
        "model_configs": {name: MODEL_CONFIGS[name] for name in config_names},
        "placebo_seeds": placebo_seeds,
        "calibration": calibration.to_dict(),
        "variants": results,
        "gate": gate,
        "placebo_guarded_modular": modular,
        "daily_prediction_contract": {
            "candidate_selection_validation_daily.csv": (
                "fit-only model using validation-selected config; validation rows only"
            ),
            "candidate_refit_test_daily.csv": (
                "fit-plus-validation refit with fixed tree counts; test rows only"
            ),
            "modular_selection_validation_daily.csv": (
                "validation-only placebo-guarded head selection inputs; validation rows only"
            ),
            "modular_refit_test_daily.csv": (
                "validation-selected modular heads after pre-test refit; test rows only"
            ),
        },
        "causal_contract": {
            "forecast_origin": "previous_krx_close",
            "target": "current_krx_open_to_close_and_close_state_transition",
            "current_close_high_low_volume_in_input": False,
            "current_gap_open_in_input": True,
            "lagged_news_fundamental_investor_external_in_input": True,
            "known_gap_and_lagged_investor_excluded_from_state_energy_target": True,
            "test_used_for_model_or_config_selection": False,
            "fit_and_validation_refit_before_test": True,
            "refit_tree_counts_fixed_from_validation": True,
            "placebo_shuffled_only_within_fit_validation_test": True,
            "jepa_projection_fit_rows_only": bool(
                args.jepa_feature_mode == "sensor_residual_pca"
            ),
            "jepa_projection_uses_targets": False,
        },
        "live_orders_allowed": False,
    }
    _write_daily(
        output_dir / "candidate_selection_validation_daily.csv",
        split_rows,
        actual_rows,
        energy,
        labels,
        validation_predictions_by_variant["open_sensors_plus_jepa"][0],
        validation_predictions_by_variant["open_sensors_plus_jepa"][1],
        validation_predictions_by_variant["open_sensors_plus_jepa"][2],
        splits=("validation",),
    )
    _write_daily(
        output_dir / "candidate_refit_test_daily.csv",
        split_rows,
        actual_rows,
        energy,
        labels,
        refit_predictions_by_variant["open_sensors_plus_jepa"][0],
        refit_predictions_by_variant["open_sensors_plus_jepa"][1],
        refit_predictions_by_variant["open_sensors_plus_jepa"][2],
        splits=("test",),
    )
    _write_daily(
        output_dir / "modular_selection_validation_daily.csv",
        split_rows,
        actual_rows,
        energy,
        labels,
        modular_validation_predictions[0],
        modular_validation_predictions[1],
        modular_validation_predictions[2],
        splits=("validation",),
    )
    _write_daily(
        output_dir / "modular_refit_test_daily.csv",
        split_rows,
        actual_rows,
        energy,
        labels,
        modular_test_predictions[0],
        modular_test_predictions[1],
        modular_test_predictions[2],
        splits=("test",),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "gate": gate,
                "candidate_test_score": candidate["test_score"],
                "baseline_test_score": baseline["test_score"],
                "modular_test_score": modular_test_score,
                "modular_gate": modular["gate"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
