from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
import torch

from scripts.audit_market_transition_targets import _actual_rows
from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_direct_market_transition_head import (
    build_design_with_transition_history,
    parse_transition_history_lags,
)
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    checkpoint_sha256,
)
from scripts.benchmark_market_transition_head import (
    _daily_rows,
    _subsample,
    _write_csv,
    build_target_arrays,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    summarize,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)
from stock_v2.market_transition_head import (
    MARKET_COMPONENT_TARGETS,
    MARKET_EVENT_TARGETS,
    MARKET_FAMILY_TARGETS,
)


PARAMETER_GRID = (
    {"max_depth": 6, "min_samples_leaf": 8, "max_features": 0.5},
    {"max_depth": 10, "min_samples_leaf": 4, "max_features": 0.7},
    {"max_depth": None, "min_samples_leaf": 8, "max_features": 1.0},
)


ACTIVITY_STOCK_FEATURES = frozenset(
    {
        "return_1d",
        "return_2d",
        "return_3d",
        "return_5d",
        "volatility_5d",
        "volatility_10d",
        "volatility_20d",
        "downside_volatility_20d",
        "volume_z20",
        "volume_z60",
        "value_z20",
        "value_z60",
        "value_ma20_log",
        "amihud_20d",
        "range_z20",
        "gap_open",
        "intraday_return",
        "market_return_1d",
        "market_return_5d",
        "market_corr_60d",
    }
)


EVENT_FAMILY = {
    "price_transition": "price_co_movement",
    "activity_transition": "market_activity",
    "node_state_transition": "node_state",
    "topology_transition": "topology",
}


def activity_event_feature_indices(feature_names) -> np.ndarray:
    selected = []
    for index, raw_name in enumerate(feature_names):
        name = str(raw_name)
        parts = name.split(":")
        pool = parts[0]
        feature = parts[-1]
        if pool.startswith("transition_lag"):
            if any(
                token in name
                for token in (
                    "volume_",
                    "value_",
                    "market_activity",
                    "activity_transition",
                    "systemic_event",
                )
            ):
                selected.append(index)
            continue
        if pool.startswith("stock_") and "available" not in pool:
            if (
                feature in ACTIVITY_STOCK_FEATURES
                or feature.startswith("investor_")
                or feature
                in {
                    "news_abs_decay",
                    "news_count_decay",
                    "news_count_3d",
                    "news_count_10d",
                    "news_confidence_1d",
                }
            ):
                selected.append(index)
            continue
        if pool == "external_value" and feature.startswith(
            (
                "ext_kospi_",
                "ext_kosdaq_",
                "ext_sp500_",
                "ext_nasdaq_",
                "ext_dow_",
                "ext_vix_",
            )
        ):
            selected.append(index)
    indices = np.asarray(selected, dtype=np.int64)
    if indices.size < 20:
        raise ValueError("activity sensor contract selected too few features")
    return indices


def event_feature_indices(event_name, feature_names, activity_feature_mode):
    if (
        str(event_name) == "activity_transition"
        and str(activity_feature_mode) == "domain_pruned_v1"
    ):
        return activity_event_feature_indices(feature_names)
    if str(activity_feature_mode) not in {"all", "domain_pruned_v1"}:
        raise ValueError("unknown activity feature mode")
    return np.arange(len(feature_names), dtype=np.int64)


def validation_event_auc(validation, event_name, horizons) -> float:
    values = []
    weights = []
    for horizon in horizons:
        item = validation["horizons"][str(int(horizon))]
        if event_name == "systemic_event":
            metric = item["systemic_event"]
        elif event_name == "broad_selloff":
            metric = item["broad_selloff"]
        else:
            metric = item["family_events"][EVENT_FAMILY[str(event_name)]]
        value = float(metric["roc_auc"])
        if not math.isfinite(value):
            continue
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        values.append(value * weight)
        weights.append(weight)
    return float(sum(values) / sum(weights)) if weights else float("nan")


def validation_event_scores(validation, horizons) -> dict[str, float]:
    return {
        name: validation_event_auc(validation, name, horizons)
        for name in MARKET_EVENT_TARGETS
    }


def regression_targets(targets: dict[str, object]) -> np.ndarray:
    if not np.asarray(targets["component_valid"], dtype=bool).all():
        raise ValueError("classical comparator requires complete component targets")
    if not np.asarray(targets["family_valid"], dtype=bool).all():
        raise ValueError("classical comparator requires complete family targets")
    component = np.asarray(targets["components"], dtype=np.float32)
    family = np.asarray(targets["family_log"], dtype=np.float32)
    return np.concatenate((component, family), axis=2).reshape(len(component), -1)


def event_class_weights(labels: np.ndarray) -> list[dict[int, float]]:
    flattened = np.asarray(labels, dtype=np.int64).reshape(len(labels), -1)
    output = []
    for column in flattened.T:
        positives = max(int(column.sum()), 1)
        negatives = max(int(len(column) - column.sum()), 1)
        output.append({0: 1.0, 1: float(min(negatives / positives, 20.0))})
    return output


def _fit_regressor(
    design,
    targets,
    parameters,
    *,
    estimators: int,
    seed: int,
    jobs: int,
):
    sample_weight = np.asarray(targets["sample_weight"], dtype=np.float64).max(
        axis=1
    )
    regressor = ExtraTreesRegressor(
        n_estimators=int(estimators),
        criterion="squared_error",
        random_state=int(seed) + 1,
        n_jobs=int(jobs),
        bootstrap=False,
        **parameters,
    )
    regressor.fit(
        design, regression_targets(targets), sample_weight=sample_weight
    )
    return regressor


def _fit_event_classifier(
    design,
    labels,
    sample_weight,
    parameters,
    *,
    estimators: int,
    seed: int,
    jobs: int,
    mode: str,
    feature_names,
    activity_feature_mode: str,
):
    labels = np.asarray(labels, dtype=np.int64)
    common = {
        "n_estimators": int(estimators),
        "criterion": "log_loss",
        "n_jobs": int(jobs),
        "bootstrap": False,
        **parameters,
    }
    if mode == "joint":
        classifier = ExtraTreesClassifier(
            class_weight=event_class_weights(labels),
            random_state=int(seed),
            **common,
        )
        classifier.fit(
            design,
            labels.reshape(len(labels), -1),
            sample_weight=sample_weight,
        )
        return classifier
    if mode != "by_event":
        raise ValueError("event classifier mode must be 'joint' or 'by_event'")
    if labels.shape[2] != len(MARKET_EVENT_TARGETS):
        raise ValueError("event labels do not match the event contract")
    models = []
    feature_indices = []
    for event_index in range(labels.shape[2]):
        event_labels = labels[:, :, event_index]
        indices = event_feature_indices(
            MARKET_EVENT_TARGETS[event_index],
            feature_names,
            activity_feature_mode,
        )
        classifier = ExtraTreesClassifier(
            class_weight=event_class_weights(event_labels),
            random_state=int(seed) + 1000 + event_index,
            **common,
        )
        classifier.fit(
            design[:, indices], event_labels, sample_weight=sample_weight
        )
        models.append(classifier)
        feature_indices.append(indices.tolist())
    return {
        "mode": "by_event",
        "models": models,
        "feature_indices": feature_indices,
        "activity_feature_mode": str(activity_feature_mode),
        "horizons": int(labels.shape[1]),
        "events": int(labels.shape[2]),
    }


def fit_models(
    design,
    targets,
    parameters,
    *,
    estimators: int,
    seed: int,
    jobs: int,
    event_classifier_mode: str = "joint",
    feature_names=None,
    activity_feature_mode: str = "all",
):
    sample_weight = np.asarray(targets["sample_weight"], dtype=np.float64).max(axis=1)
    labels = np.asarray(targets["labels"], dtype=np.int64)
    regressor = _fit_regressor(
        design,
        targets,
        parameters,
        estimators=int(estimators),
        seed=int(seed),
        jobs=int(jobs),
    )
    classifier = _fit_event_classifier(
        design,
        labels,
        sample_weight,
        parameters,
        estimators=int(estimators),
        seed=int(seed),
        jobs=int(jobs),
        mode=str(event_classifier_mode),
        feature_names=feature_names,
        activity_feature_mode=str(activity_feature_mode),
    )
    return regressor, classifier


def fit_models_with_event_parameters(
    design,
    targets,
    regressor_parameters,
    regressor_candidate_index,
    event_selection,
    *,
    estimators: int,
    base_seed: int,
    jobs: int,
    feature_names,
    activity_feature_mode: str,
):
    sample_weight = np.asarray(targets["sample_weight"], dtype=np.float64).max(
        axis=1
    )
    labels = np.asarray(targets["labels"], dtype=np.int64)
    regressor = _fit_regressor(
        design,
        targets,
        regressor_parameters,
        estimators=int(estimators),
        seed=int(base_seed) + 100 * int(regressor_candidate_index),
        jobs=int(jobs),
    )
    models = []
    feature_indices = []
    for event_index, event_name in enumerate(MARKET_EVENT_TARGETS):
        selection = event_selection[str(event_name)]
        indices = event_feature_indices(
            event_name, feature_names, activity_feature_mode
        )
        classifier = ExtraTreesClassifier(
            n_estimators=int(estimators),
            criterion="log_loss",
            class_weight=event_class_weights(labels[:, :, event_index]),
            random_state=(
                int(base_seed)
                + 100 * int(selection["candidate_index"])
                + 1000
                + event_index
            ),
            n_jobs=int(jobs),
            bootstrap=False,
            **selection["parameters"],
        )
        classifier.fit(
            design[:, indices],
            labels[:, :, event_index],
            sample_weight=sample_weight,
        )
        models.append(classifier)
        feature_indices.append(indices.tolist())
    return regressor, {
        "mode": "by_event",
        "models": models,
        "feature_indices": feature_indices,
        "activity_feature_mode": str(activity_feature_mode),
        "event_model_selection": "per_event_validation_auc",
        "horizons": int(labels.shape[1]),
        "events": int(labels.shape[2]),
    }


def positive_probabilities(classifier, design) -> np.ndarray:
    if isinstance(classifier, dict):
        if classifier.get("mode") != "by_event":
            raise ValueError("unknown event classifier bundle")
        probabilities = [
            positive_probabilities(
                model,
                design[:, np.asarray(indices, dtype=np.int64)],
            )
            for model, indices in zip(
                classifier["models"],
                classifier.get(
                    "feature_indices",
                    [range(design.shape[1])] * len(classifier["models"]),
                ),
            )
        ]
        expected = (
            len(design),
            int(classifier["horizons"]),
            int(classifier["events"]),
        )
        stacked = np.stack(probabilities, axis=2)
        if stacked.shape != expected:
            raise ValueError("by-event classifier probabilities are misaligned")
        return stacked.reshape(len(design), -1)
    raw = classifier.predict_proba(design)
    raw = raw if isinstance(raw, list) else [raw]
    classes = classifier.classes_
    classes = classes if isinstance(classes, list) else [classes]
    columns = []
    for probability, values in zip(raw, classes):
        values = np.asarray(values)
        match = np.flatnonzero(values == 1)
        columns.append(
            probability[:, int(match[0])]
            if match.size
            else np.zeros(len(design), dtype=np.float64)
        )
    return np.stack(columns, axis=1)


def prediction_rows(regressor, classifier, design, targets, contracts, horizons):
    batch = len(design)
    component_count = len(MARKET_COMPONENT_TARGETS)
    family_count = len(MARKET_FAMILY_TARGETS)
    event_count = len(MARKET_EVENT_TARGETS)
    regression = regressor.predict(design).reshape(
        batch, len(horizons), component_count + family_count
    )
    component = regression[:, :, :component_count]
    family = np.maximum(
        np.expm1(np.clip(regression[:, :, component_count:], -5.0, 5.0)), 0.0
    )
    probability = positive_probabilities(classifier, design).reshape(
        batch, len(horizons), event_count
    )
    logits = np.log(
        np.clip(probability, 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - probability, 1e-6, 1.0)
    )
    output = {int(horizon): [] for horizon in horizons}
    for horizon_index, horizon in enumerate(horizons):
        contract = contracts[int(horizon)]
        raw_component = (
            component[:, horizon_index] * contract.component_std[None, :]
            + contract.component_mean[None, :]
        )
        for position in range(batch):
            actual = targets["rows"][position][horizon_index]
            output[int(horizon)].append(
                {
                    "step": int(actual["step"]),
                    "date": str(actual["date"]),
                    "horizon": int(horizon),
                    "actual": actual,
                    "predicted": {
                        name: float(raw_component[position, index])
                        for index, name in enumerate(MARKET_COMPONENT_TARGETS)
                    },
                    "predicted_families": family[position, horizon_index].tolist(),
                    "event_logits": logits[position, horizon_index].tolist(),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classical same-objective market-transition comparator."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--estimators", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument(
        "--event-classifier-mode",
        choices=["joint", "by_event"],
        default="joint",
    )
    parser.add_argument(
        "--activity-feature-mode",
        choices=["all", "domain_pruned_v1"],
        default="all",
    )
    parser.add_argument(
        "--event-model-selection",
        choices=["joint_bundle", "per_event_validation_auc"],
        default="joint_bundle",
    )
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument(
        "--transition-history-lags",
        default="",
        help="Completed one-step market-transition lags, for example 1,2,5.",
    )
    args = parser.parse_args()
    if (
        args.event_model_selection == "per_event_validation_auc"
        and args.event_classifier_mode != "by_event"
    ):
        parser.error("per-event model selection requires by-event classifiers")
    if (
        args.activity_feature_mode != "all"
        and args.event_classifier_mode != "by_event"
    ):
        parser.error("activity feature pruning requires by-event classifiers")

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_int_list(args.horizons)
    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(features, checkpoint_args, horizons, int(args.validation_days))
    for name, option in (
        ("fit", args.max_fit_steps),
        ("validation", args.max_validation_steps),
        ("test", args.max_test_steps),
    ):
        splits[name] = _subsample(splits[name], int(option))
    rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(rows["fit"], horizons)
    targets = {
        name: build_target_arrays(rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    fit_event_rate = fit_trajectory_event_rate(targets["fit"])
    transition_history_lags = parse_transition_history_lags(
        args.transition_history_lags
    )
    designs = {}
    feature_names = None
    for name, steps in splits.items():
        design, names = build_design_with_transition_history(
            features, steps, contracts, transition_history_lags
        )
        if feature_names is None:
            feature_names = names
        elif not np.array_equal(feature_names, names):
            raise ValueError("classical comparator feature contracts differ")
        designs[name] = design

    candidates = []
    best = None
    best_score = -math.inf
    best_index = None
    event_selection = {
        name: {
            "validation_auc": -math.inf,
            "candidate_index": None,
            "parameters": None,
        }
        for name in MARKET_EVENT_TARGETS
    }
    for index, parameters in enumerate(PARAMETER_GRID):
        regressor, classifier = fit_models(
            designs["fit"],
            targets["fit"],
            parameters,
            estimators=int(args.estimators),
            seed=int(args.seed) + 100 * index,
            jobs=int(args.jobs),
            event_classifier_mode=str(args.event_classifier_mode),
            feature_names=feature_names,
            activity_feature_mode=str(args.activity_feature_mode),
        )
        validation_predictions = prediction_rows(
            regressor,
            classifier,
            designs["validation"],
            targets["validation"],
            contracts,
            horizons,
        )
        validation = summarize(
            validation_predictions, contracts, horizons, fit_event_rate
        )
        score = float(validation["validation_formula_score"])
        event_scores = validation_event_scores(validation, horizons)
        candidates.append(
            {
                "candidate_index": index,
                "parameters": parameters,
                "validation_score": score,
                "validation_event_auc": event_scores,
                "validation_trajectory": validation["trajectory"],
            }
        )
        print(
            f"candidate={index} validation_market_score={score:+.6f} "
            f"parameters={parameters}",
            flush=True,
        )
        for event_name, event_score in event_scores.items():
            if (
                math.isfinite(event_score)
                and event_score
                > float(event_selection[event_name]["validation_auc"])
            ):
                event_selection[event_name] = {
                    "validation_auc": float(event_score),
                    "candidate_index": int(index),
                    "parameters": dict(parameters),
                }
        if math.isfinite(score) and score > best_score:
            best_score = score
            best_index = int(index)
            best = (
                regressor,
                classifier,
                parameters,
                validation_predictions,
            )
    if best is None:
        raise RuntimeError("classical comparator produced no valid candidate")
    regressor, classifier, best_parameters, validation_predictions = best
    if args.event_model_selection == "per_event_validation_auc":
        if any(
            value["candidate_index"] is None
            for value in event_selection.values()
        ):
            raise RuntimeError("an event classifier produced no validation score")
        best = None
        regressor = None
        classifier = None
        validation_predictions = None
        regressor, classifier = fit_models_with_event_parameters(
            designs["fit"],
            targets["fit"],
            best_parameters,
            best_index,
            event_selection,
            estimators=int(args.estimators),
            base_seed=int(args.seed),
            jobs=int(args.jobs),
            feature_names=feature_names,
            activity_feature_mode=str(args.activity_feature_mode),
        )
        validation_predictions = prediction_rows(
            regressor,
            classifier,
            designs["validation"],
            targets["validation"],
            contracts,
            horizons,
        )
        best_score = float(
            summarize(
                validation_predictions, contracts, horizons, fit_event_rate
            )["validation_formula_score"]
        )
    predictions = {
        "validation": validation_predictions,
        "test": prediction_rows(
            regressor,
            classifier,
            designs["test"],
            targets["test"],
            contracts,
            horizons,
        ),
    }
    metrics = {
        name: summarize(values, contracts, horizons, fit_event_rate)
        for name, values in predictions.items()
    }
    for name, values in predictions.items():
        _write_csv(
            output_dir / f"daily_{name}.csv",
            _daily_rows(values, contracts, horizons, name),
        )

    summary = {
        "status": "complete",
        "role": "same_objective_classical_extra_trees_market_transition_head",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": checkpoint_sha256(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "input_features": int(designs["fit"].shape[1]),
        "input_feature_names": feature_names.tolist(),
        "transition_history_lags": list(transition_history_lags),
        "transition_history_semantics": (
            "lag L is the completed one-step transition from t-L to t-L+1"
        ),
        "estimators": int(args.estimators),
        "event_classifier_mode": str(args.event_classifier_mode),
        "event_model_selection": str(args.event_model_selection),
        "event_best_validation_selection": (
            event_selection
            if args.event_model_selection == "per_event_validation_auc"
            else None
        ),
        "activity_feature_mode": str(args.activity_feature_mode),
        "event_input_features": (
            {
                name: len(indices)
                for name, indices in zip(
                    MARKET_EVENT_TARGETS, classifier["feature_indices"]
                )
            }
            if isinstance(classifier, dict)
            else {"joint": len(feature_names)}
        ),
        "best_parameters": best_parameters,
        "candidate_validation_results": candidates,
        "best_validation_score": best_score,
        "sample_weight": "max horizon systemic-impact weight per context",
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_event_rate,
        "metrics": metrics,
        "test_used_for_selection": False,
        "selection_status": "research_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    joblib.dump(
        {
            "regressor": regressor,
            "classifier": classifier,
            "target_version": MARKET_TRANSITION_TARGET_VERSION,
            "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
            "feature_names": feature_names,
            "horizons": horizons,
            "live_orders_allowed": False,
        },
        output_dir / "classical_market_transition_head.joblib",
        compress=3,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_trajectory": metrics["test"]["trajectory"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
