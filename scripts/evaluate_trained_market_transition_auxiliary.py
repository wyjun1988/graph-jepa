from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import (
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.benchmark_systemic_transition_head import configured_horizon_text
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import (
    date_indices,
    parse_int_list,
    rollout_steps_for_offset,
    temporal_training_indices,
)
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
    MarketTransitionCalibration,
    binary_ranking_metrics,
)
from stock_v2.market_transition_auxiliary import (
    MARKET_TRANSITION_AUXILIARY_FAMILIES,
    apply_market_transition_auxiliary_calibration,
)


EVALUATION_VERSION = "trained_market_transition_auxiliary_v3_20260714"


def _calibrations(checkpoint: dict, horizons: list[int]):
    contract = checkpoint.get("market_transition_auxiliary_contract")
    if not isinstance(contract, dict):
        raise ValueError("checkpoint has no market transition auxiliary contract")
    if bool(contract.get("live_orders_allowed", True)):
        raise ValueError("market transition auxiliary contract is not research-only")
    if contract.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError(
            "market transition auxiliary target version differs: "
            f"checkpoint={contract.get('target_version')!r} "
            f"evaluator={MARKET_TRANSITION_TARGET_VERSION!r}"
        )
    if tuple(contract.get("family_names", ())) != tuple(
        MARKET_TRANSITION_AUXILIARY_FAMILIES
    ):
        raise ValueError("market transition auxiliary family contract differs")
    output = {}
    for horizon in horizons:
        item = contract.get("horizons", {}).get(str(int(horizon)))
        if not isinstance(item, dict):
            raise ValueError(f"missing auxiliary calibration for horizon {horizon}")
        output[int(horizon)] = MarketTransitionCalibration(
            **item["calibration"]
        )
    return output, contract


def _subsample(steps: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(steps) <= int(maximum):
        return np.asarray(steps, dtype=np.int64)
    positions = np.linspace(0, len(steps) - 1, int(maximum)).round().astype(int)
    return np.asarray(steps, dtype=np.int64)[positions]


def _predict(
    model,
    features,
    steps,
    targets,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
):
    rows = []
    rollout_args = dict(checkpoint_args)
    rollout_args.setdefault(
        "temporal_offset", checkpoint_args.get("horizon", max(horizons))
    )
    rollout_args.setdefault("latent_rollout_steps", 1)
    rollout_namespace = argparse.Namespace(**rollout_args)
    family_count = len(MARKET_TRANSITION_AUXILIARY_FAMILIES)
    model.eval()
    for start in range(0, len(steps), int(batch_size)):
        selected = np.asarray(steps[start : start + int(batch_size)], dtype=np.int64)
        batch = snapshot_batch(
            features, selected, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
            stock_mask = model._supervision_node_mask(batch)
            available = (
                torch.ones_like(batch.node_features, dtype=torch.bool)
                if batch.available_mask is None
                else batch.available_mask > 0.5
            )
            stock_pool_mask = stock_mask & available.any(dim=-1)
            outputs = {
                int(horizon): model.predict_downstream_transition(
                    context,
                    predicted[int(horizon)],
                    stock_mask,
                    batch.graph_index,
                    rollout_steps_for_offset(rollout_namespace, int(horizon)),
                    stock_pool_mask=stock_pool_mask,
                ).float()
                for horizon in horizons
            }
        for position, step in enumerate(selected):
            for horizon in horizons:
                output = outputs[int(horizon)][position]
                actual = targets[int(horizon)][int(step)]
                predicted_family = F.softplus(output[:family_count]).cpu().numpy()
                rows.append(
                    {
                        "step": int(step),
                        "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                        "horizon": int(horizon),
                        **{
                            f"actual_family_log:{name}": float(actual[index])
                            for index, name in enumerate(
                                MARKET_TRANSITION_AUXILIARY_FAMILIES
                            )
                        },
                        **{
                            f"predicted_family_log:{name}": float(
                                predicted_family[index]
                            )
                            for index, name in enumerate(
                                MARKET_TRANSITION_AUXILIARY_FAMILIES
                            )
                        },
                        **{
                            f"actual_family_event:{name}": bool(
                                actual[family_count + index]
                            )
                            for index, name in enumerate(
                                MARKET_TRANSITION_AUXILIARY_FAMILIES
                            )
                        },
                        **{
                            f"family_event_logit:{name}": float(
                                output[family_count + index].cpu()
                            )
                            for index, name in enumerate(
                                MARKET_TRANSITION_AUXILIARY_FAMILIES
                            )
                        },
                        "actual_broad_selloff": bool(actual[-2]),
                        "broad_selloff_logit": float(output[-2].cpu()),
                        "actual_systemic_event": bool(actual[-1]),
                        "systemic_event_logit": float(output[-1].cpu()),
                    }
                )
    return rows


def _ranking(labels, scores, selection_rate):
    metrics = binary_ranking_metrics(
        np.asarray(labels, dtype=bool),
        np.asarray(scores, dtype=np.float64),
        selection_rate=float(selection_rate),
    )
    rate = float(metrics["event_rate"])
    metrics["average_precision_lift"] = (
        float(metrics["average_precision"]) / rate if rate > 0.0 else float("nan")
    )
    return metrics


def _ranking_with_systemic_impact(labels, scores, selection_rate, actual_impact):
    metrics = _ranking(labels, scores, selection_rate)
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    actual_impact = np.asarray(actual_impact, dtype=np.float64)
    valid = np.isfinite(scores) & np.isfinite(actual_impact)
    labels = labels[valid]
    scores = scores[valid]
    actual_impact = actual_impact[valid]
    selected_count = int(metrics["selected_count"])
    selected = np.argsort(scores, kind="mergesort")[-selected_count:]
    event_mass = float(actual_impact[labels].sum())
    captured = float(actual_impact[selected][labels[selected]].sum())
    metrics["systemic_impact_mass_recall_at_selection_rate"] = (
        captured / event_mass if event_mass > 1e-12 else float("nan")
    )
    return metrics


def _horizon_metrics(rows, contract):
    output = {}
    for horizon, selected in pd.DataFrame(rows).groupby("horizon", sort=True):
        horizon_contract = contract["horizons"][str(int(horizon))]
        family = {}
        for index, name in enumerate(MARKET_TRANSITION_AUXILIARY_FAMILIES):
            actual = selected[f"actual_family_log:{name}"].to_numpy(float)
            predicted = selected[f"predicted_family_log:{name}"].to_numpy(float)
            labels = selected[f"actual_family_event:{name}"].to_numpy(bool)
            logits = selected[f"family_event_logit:{name}"].to_numpy(float)
            family[name] = {
                "log_intensity_mae": float(np.mean(np.abs(predicted - actual))),
                "log_intensity_correlation": pearson(predicted, actual),
                "event": _ranking(
                    labels,
                    logits,
                    horizon_contract["fit_family_event_rate"][index],
                ),
            }
        selloff_labels = selected["actual_broad_selloff"].to_numpy(bool)
        actual_impact = np.maximum(
            np.max(
                np.stack(
                    [
                        np.expm1(
                            selected[f"actual_family_log:{name}"].to_numpy(float)
                        )
                        for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
                    ],
                    axis=1,
                ),
                axis=1,
            ),
            selloff_labels.astype(np.float64),
        )
        output[str(int(horizon))] = {
            "rows": len(selected),
            "family": family,
            "broad_selloff": _ranking(
                selloff_labels,
                selected["broad_selloff_logit"].to_numpy(float),
                horizon_contract["fit_broad_selloff_rate"],
            ),
            "systemic_event": _ranking_with_systemic_impact(
                selected["actual_systemic_event"].to_numpy(bool),
                selected["systemic_event_logit"].to_numpy(float),
                horizon_contract["fit_systemic_event_rate"],
                actual_impact,
            ),
        }
    return output


def _path_metrics(rows, fit_targets, steps, horizons):
    family_count = len(MARKET_TRANSITION_AUXILIARY_FAMILIES)
    fit_path = []
    for step in steps:
        fit_path.append(
            max(
                max(
                    float(
                        np.expm1(
                            fit_targets[int(horizon)][int(step)][:family_count]
                        ).max()
                    ),
                    float(fit_targets[int(horizon)][int(step)][-2]),
                )
                for horizon in horizons
            )
        )
    threshold = float(np.quantile(fit_path, 0.90))
    date_rows = []
    frame = pd.DataFrame(rows)
    for date, selected in frame.groupby("date", sort=True):
        selected = selected.sort_values("horizon")
        actual_horizon = np.asarray(
            [
                max(
                    max(
                        np.expm1(row[f"actual_family_log:{name}"])
                        for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
                    ),
                    float(row["actual_broad_selloff"]),
                )
                for _, row in selected.iterrows()
            ],
            dtype=np.float64,
        )
        predicted_horizon = np.asarray(
            [
                max(
                    np.expm1(row[f"predicted_family_log:{name}"])
                    for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
                )
                for _, row in selected.iterrows()
            ],
            dtype=np.float64,
        )
        horizons_for_date = selected["horizon"].to_numpy(int)
        actual_path = float(actual_horizon.max())
        date_rows.append(
            {
                "date": date,
                "actual_path_salience": actual_path,
                "predicted_path_salience": float(predicted_horizon.max()),
                "actual_major_event": actual_path >= threshold,
                "actual_peak_horizon": int(
                    horizons_for_date[int(np.argmax(actual_horizon))]
                ),
                "predicted_peak_horizon": int(
                    horizons_for_date[int(np.argmax(predicted_horizon))]
                ),
            }
        )
    daily = pd.DataFrame(date_rows)
    labels = daily["actual_major_event"].to_numpy(bool)
    ranking = _ranking_with_systemic_impact(
        labels,
        daily["predicted_path_salience"],
        0.10,
        daily["actual_path_salience"],
    )
    ranking.update(
        {
            "fit_major_event_threshold": threshold,
            "path_log_mae": float(
                np.mean(
                    np.abs(
                        np.log1p(daily["predicted_path_salience"])
                        - np.log1p(daily["actual_path_salience"])
                    )
                )
            ),
            "path_log_correlation": pearson(
                np.log1p(daily["predicted_path_salience"].to_numpy(float)),
                np.log1p(daily["actual_path_salience"].to_numpy(float)),
            ),
            "peak_horizon_accuracy_on_major_events": (
                float(
                    np.mean(
                        daily.loc[labels, "actual_peak_horizon"].to_numpy(int)
                        == daily.loc[labels, "predicted_peak_horizon"].to_numpy(int)
                    )
                )
                if labels.any()
                else float("nan")
            ),
        }
    )
    return ranking, daily


def _write_csv(path: Path, rows) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the JEPA checkpoint's trained broad-transition heads."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    if float(checkpoint_args.get("downstream_transition_loss_weight", 0.0)) <= 0.0:
        raise ValueError("checkpoint was not trained with broad-transition auxiliary loss")
    horizons = parse_int_list(args.horizons)
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    calibrations, contract = _calibrations(checkpoint, horizons)
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    feature_args.edge_cache_workers = min(16, int(args.batch_size))
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    test_steps = _subsample(splits["test"], int(args.max_test_steps))
    train_steps = temporal_training_indices(
        date_indices(features.dates, end=str(checkpoint_args["train_end"])),
        edge_window=int(checkpoint_args.get("edge_window", 60)),
        max_rollout_offset=max(horizons),
        total_steps=len(features.dates),
    )
    test_targets = apply_market_transition_auxiliary_calibration(
        features, test_steps, calibrations
    )
    fit_targets = apply_market_transition_auxiliary_calibration(
        features, train_steps, calibrations
    )
    edge_cache = build_evaluation_edge_cache(
        features, test_steps, checkpoint_args, feature_args
    )
    rows = _predict(
        model,
        features,
        test_steps,
        test_targets,
        horizons,
        checkpoint_args,
        feature_args,
        edge_cache,
        device,
        int(args.batch_size),
    )
    path, path_daily = _path_metrics(
        rows, fit_targets, train_steps, horizons
    )
    summary = {
        "status": "complete",
        "role": "trained_graph_jepa_broad_transition_auxiliary",
        "evaluation_version": EVALUATION_VERSION,
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": checkpoint_sha256(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "test_steps": len(test_steps),
        "horizon_metrics": _horizon_metrics(rows, contract),
        "major_path": path,
        "test_used_for_selection": False,
        "selection_status": "research_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    _write_csv(output_dir / "daily_test.csv", rows)
    path_daily.to_csv(output_dir / "daily_major_path_test.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "test_steps": len(test_steps),
                "major_path": path,
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
