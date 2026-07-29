from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_systemic_transition_targets import _actual_rows, _split_steps
from scripts.benchmark_causal_memory_systemic_head import (
    VARIANTS,
    _cache_contract,
    _load_cache_arrays,
    _row_positions,
    prepare_auxiliary_design,
    predict_steps,
    summarize_with_direction,
)
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import HORIZON_WEIGHTS
from scripts.benchmark_systemic_transition_head import (
    _subsample,
    _target_arrays,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    trajectory_metrics,
)
from scripts.compare_systemic_transition_heads import absolute_gate
from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.systemic_head import CausalMemorySystemicTransitionHead


MIN_AUC_ADVANTAGE = 0.01
MIN_BROAD_RECALL_ADVANTAGE = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _weighted_metric(
    horizons: Mapping[str, Mapping[str, Any]],
    subtype_key: str,
    subtype: str,
    metric: str,
) -> float:
    total = 0.0
    weight_sum = 0.0
    for raw_horizon, row in horizons.items():
        value = float(row[subtype_key][subtype][metric])
        weight = float(HORIZON_WEIGHTS.get(int(raw_horizon), 1.0))
        total += weight * value
        weight_sum += weight
    return float(total / weight_sum)


def _weighted_mean_subtype_auc(
    horizons: Mapping[str, Mapping[str, Any]], subtype_key: str
) -> float:
    total = 0.0
    weight_sum = 0.0
    for raw_horizon, row in horizons.items():
        horizon_values = [
            float(metrics["roc_auc"])
            for metrics in row[subtype_key].values()
        ]
        weight = float(HORIZON_WEIGHTS.get(int(raw_horizon), 1.0))
        total += weight * float(np.nanmean(horizon_values))
        weight_sum += weight
    return float(total / weight_sum)


def readout_validation_gate(
    horizons: Mapping[str, Mapping[str, Any]],
    *,
    min_auc_advantage: float = MIN_AUC_ADVANTAGE,
    min_broad_recall_advantage: float = MIN_BROAD_RECALL_ADVANTAGE,
) -> dict[str, Any]:
    dedicated_auc = _weighted_mean_subtype_auc(horizons, "subtypes")
    derived_auc = _weighted_mean_subtype_auc(horizons, "derived_subtypes")
    dedicated_recall = _weighted_metric(
        horizons, "subtypes", "broad_selloff", "recall_at_selection_rate"
    )
    derived_recall = _weighted_metric(
        horizons,
        "derived_subtypes",
        "broad_selloff",
        "recall_at_selection_rate",
    )
    auc_advantage = derived_auc - dedicated_auc
    recall_advantage = derived_recall - dedicated_recall
    checks = {
        "derived_mean_subtype_auc_advantage": auc_advantage
        >= float(min_auc_advantage),
        "derived_broad_selloff_recall_advantage": recall_advantage
        >= float(min_broad_recall_advantage),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "dedicated_weighted_mean_subtype_auc": dedicated_auc,
        "derived_weighted_mean_subtype_auc": derived_auc,
        "auc_advantage": auc_advantage,
        "dedicated_weighted_broad_selloff_recall": dedicated_recall,
        "derived_weighted_broad_selloff_recall": derived_recall,
        "broad_selloff_recall_advantage": recall_advantage,
        "minimum_auc_advantage": float(min_auc_advantage),
        "minimum_broad_recall_advantage": float(min_broad_recall_advantage),
        "selection_split": "validation",
        "test_used_for_selection": False,
    }


def use_derived_subtypes(
    horizons: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = copy.deepcopy(dict(horizons))
    for row in result.values():
        row["dedicated_subtypes"] = row["subtypes"]
        row["subtypes"] = row["derived_subtypes"]
    return result


def _load_head(
    path: Path,
    architecture: Mapping[str, Any],
    latent_dim: int,
    auxiliary_dim: int,
    horizons: Sequence[int],
    variant: str,
    parent_sha256: str,
    device: torch.device,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("variant") != variant:
        raise ValueError(f"head variant mismatch: {variant}")
    if payload.get("parent_model_sha256") != parent_sha256:
        raise ValueError(f"head parent checkpoint mismatch: {variant}")
    if tuple(int(value) for value in payload.get("horizons", ())) != tuple(horizons):
        raise ValueError(f"head horizons mismatch: {variant}")
    if int(payload.get("auxiliary_dim", -1)) != int(auxiliary_dim):
        raise ValueError(f"head auxiliary width mismatch: {variant}")
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"head does not prohibit live orders: {variant}")
    head = CausalMemorySystemicTransitionHead(
        int(latent_dim),
        int(auxiliary_dim),
        horizons,
        projection_dim=int(architecture["projection_dim"]),
        auxiliary_projection_dim=int(architecture["auxiliary_projection_dim"]),
        hidden_dim=int(architecture["hidden_dim"]),
        horizon_dim=int(architecture["horizon_dim"]),
        dropout=float(architecture["dropout"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    return head


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate dense component-derived systemic subtype readouts."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--head-report-dir", required=True)
    parser.add_argument("--forecast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not variants or any(value not in VARIANTS for value in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device(args.device)
    horizons = tuple(parse_int_list(args.horizons))
    model_dir = Path(args.model_dir).resolve()
    report_dir = Path(args.head_report_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_summary = json.loads(
        (report_dir / "summary.json").read_text(encoding="utf-8")
    )
    if parent_summary.get("live_orders_allowed") is not False:
        raise ValueError("parent report does not prohibit live orders")
    parent_checkpoint = model_dir / "graph_jepa_real.pt"
    parent_sha256 = sha256_file(parent_checkpoint)
    if parent_summary.get("parent_model_sha256") != parent_sha256:
        raise ValueError("parent report differs from the requested JEPA checkpoint")

    checkpoint = torch.load(parent_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    splits = {name: _subsample(values, 0) for name, values in splits.items()}
    minimum_step = max(0, int(splits["fit"].min()) - max(horizons))
    maximum_step = int(splits["test"].max())
    chronology_steps = np.arange(minimum_step, maximum_step + 1, dtype=np.int64)

    temporal_weights = checkpoint.get("temporal_state_feature_weights")
    if torch.is_tensor(temporal_weights):
        temporal_weights = temporal_weights.detach().cpu().numpy()
    eligible_indices = np.flatnonzero(
        np.asarray(temporal_weights, dtype=np.float32) > 0.0
    )
    expected_cache_contract = _cache_contract(
        model_dir,
        chronology_steps,
        horizons,
        features.node_count,
        int(checkpoint_args["hidden_dim"]),
        eligible_indices,
    )
    cache_dir = Path(args.forecast_cache_dir).resolve()
    if not (cache_dir / "CACHE_COMPLETE").exists():
        raise ValueError("forecast cache is incomplete")
    actual_cache_contract = json.loads(
        (cache_dir / "contract.json").read_text(encoding="utf-8")
    )
    if actual_cache_contract != expected_cache_contract:
        raise ValueError("forecast cache contract differs from the evaluation")
    cached_context, cached_predicted, state_forecasts = _load_cache_arrays(
        cache_dir, horizons
    )

    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    targets = {
        name: _target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    auxiliary_variants, auxiliary_diagnostics = prepare_auxiliary_design(
        features,
        chronology_steps,
        splits,
        state_forecasts,
        eligible_indices,
        seed=int(args.seed),
    )
    split_rows = {
        name: _row_positions(chronology_steps, steps)
        for name, steps in splits.items()
    }
    architecture = parent_summary["architecture"]
    variant_results = {}
    for variant in variants:
        auxiliary = auxiliary_variants[variant]
        head_path = report_dir / variant / "causal_memory_systemic_head.pt"
        head = _load_head(
            head_path,
            architecture,
            cached_context.shape[-1],
            auxiliary.shape[1],
            horizons,
            variant,
            parent_sha256,
            device,
        )
        split_result = {}
        for split in ("validation", "test"):
            predictions = predict_steps(
                head,
                cached_context,
                cached_predicted,
                split_rows[split],
                auxiliary,
                targets[split],
                contracts,
                horizons,
                device,
                int(args.eval_batch_size),
                features.node_count,
                features.tradable_count,
            )
            horizon_metrics, dedicated_score = summarize_with_direction(
                predictions, contracts, horizons
            )
            trajectory = trajectory_metrics(
                predictions,
                contracts,
                horizons,
                fit_trajectory_event_rate(targets["fit"], contracts, horizons),
            )
            split_result[split] = {
                "horizons": horizon_metrics,
                "trajectory": trajectory,
                "dedicated_validation_formula_score": dedicated_score,
            }
        validation_gate = readout_validation_gate(
            split_result["validation"]["horizons"]
        )
        dedicated_gate = absolute_gate(
            {"metrics": {"test": split_result["test"]}}
        )
        derived_test = copy.deepcopy(split_result["test"])
        derived_test["horizons"] = use_derived_subtypes(
            split_result["test"]["horizons"]
        )
        derived_gate = absolute_gate({"metrics": {"test": derived_test}})
        variant_results[variant] = {
            "head_sha256": sha256_file(head_path),
            "validation_readout_gate": validation_gate,
            "dedicated_test_absolute_gate": dedicated_gate,
            "derived_test_absolute_gate": derived_gate,
            "derived_readout_qualified": bool(validation_gate["passed"])
            and bool(derived_gate["passed"]),
            "metrics": split_result,
        }
        del head
        if device.type == "cuda":
            torch.cuda.empty_cache()

    parent_selected = str(parent_summary["validation_selected_variant"])
    selected_result = variant_results.get(parent_selected)
    summary = {
        "schema_version": 1,
        "status": "complete",
        "role": "research_only_dense_component_derived_subtype_ablation",
        "parent_report": str(report_dir),
        "parent_report_sha256": sha256_file(report_dir / "summary.json"),
        "parent_model_sha256": parent_sha256,
        "parent_validation_selected_variant": parent_selected,
        "parent_selected_variant_evaluated": selected_result is not None,
        "parent_selected_variant_derived_readout_qualified": bool(
            selected_result and selected_result["derived_readout_qualified"]
        ),
        "selection_rule": {
            "split": "validation",
            "minimum_weighted_mean_subtype_auc_advantage": MIN_AUC_ADVANTAGE,
            "minimum_weighted_broad_selloff_recall_advantage": MIN_BROAD_RECALL_ADVANTAGE,
            "test_used_for_selection": False,
        },
        "variants": variant_results,
        "auxiliary_diagnostics": {
            "raw_features": auxiliary_diagnostics["raw_features"],
            "memory_features": auxiliary_diagnostics["memory_features"],
            "auxiliary_features": auxiliary_diagnostics["auxiliary_features"],
            "memory_group_names": auxiliary_diagnostics["memory_group_names"],
        },
        "live_orders_allowed": False,
        "test_used_for_selection": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "status": "complete",
                "parent_selected_variant": parent_selected,
                "parent_selected_variant_derived_readout_qualified": summary[
                    "parent_selected_variant_derived_readout_qualified"
                ],
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
