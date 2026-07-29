from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_residual_predictability import (
    collect_baseline_errors,
    corrected_model_sse,
)
from scripts.evaluate_causal_innovation_filter import (
    REQUIRED_HORIZONS,
    build_parser as build_data_parser,
    sha256_file,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    select_steps,
    validate_future_rollout_contract,
    write_csv,
)
from scripts.evaluate_residual_adapter_transfer import summarize_transfer
from scripts.run_real_backtest import parse_int_list
from stock_v2.online_residual_adapter import (
    OnlineResidualAdapter,
    OnlineResidualConfig,
)


ONLINE_CANDIDATE = "online_node_ar"


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(value.strip()) for value in str(text).split(",") if value.strip())
    if not values or any(not 0.0 < value <= 1.0 for value in values):
        raise ValueError("online residual alphas must be in (0, 1]")
    if len(values) != len(set(values)):
        raise ValueError("online residual alphas must be unique")
    return values


def build_adapters(
    horizons: Sequence[int],
    feature_count: int,
    config: OnlineResidualConfig,
    states: Mapping[int, Mapping[str, np.ndarray]] | None = None,
) -> dict[int, OnlineResidualAdapter]:
    return {
        int(horizon): OnlineResidualAdapter(
            int(horizon),
            int(feature_count),
            config,
            None if states is None else states[int(horizon)],
        )
        for horizon in horizons
    }


def _adjusted_row(
    base: Mapping[str, Any],
    target_error: np.ndarray,
    correction: np.ndarray,
    *,
    candidate: str,
    dynamic_available: bool,
) -> dict[str, Any]:
    model_sse, corrected_cells = corrected_model_sse(
        float(base["model_sse"]), target_error, correction
    )
    persistence_sse = float(base["persistence_sse"])
    skill = (
        float(1.0 - model_sse / persistence_sse)
        if persistence_sse > 1e-12
        else float("nan")
    )
    valid = np.isfinite(target_error) & np.isfinite(correction)
    return {
        "candidate": candidate,
        **dict(base),
        "model_sse": model_sse,
        "mse_skill_vs_persistence": skill,
        "dynamic_available": bool(dynamic_available),
        "eligible_corrected_cells": int(corrected_cells),
        "correction_rms": (
            float(np.sqrt(np.mean(np.square(correction[valid]))))
            if valid.any()
            else 0.0
        ),
    }


def evaluate_online(
    errors: Mapping[int, np.ndarray],
    baseline_rows: Mapping[int, Sequence[Mapping[str, Any]]],
    adapters: Mapping[int, OnlineResidualAdapter],
    *,
    score_start: int = 0,
) -> list[dict[str, Any]]:
    output = []
    for horizon in sorted(errors):
        values = np.asarray(errors[int(horizon)], dtype=np.float32)
        rows = list(baseline_rows[int(horizon)])
        if len(values) != len(rows):
            raise ValueError("online residual errors and baseline rows do not align")
        adapter = adapters[int(horizon)]
        for ordinal, (target_error, base) in enumerate(zip(values, rows)):
            bias, correction, dynamic = adapter.advance(values, ordinal)
            if ordinal < int(score_start):
                continue
            output.append(
                {
                    "candidate": "baseline",
                    **dict(base),
                    "dynamic_available": bool(dynamic),
                    "eligible_corrected_cells": 0,
                    "correction_rms": 0.0,
                }
            )
            output.append(
                _adjusted_row(
                    base,
                    target_error,
                    bias,
                    candidate="bias_only",
                    dynamic_available=dynamic,
                )
            )
            output.append(
                _adjusted_row(
                    base,
                    target_error,
                    correction,
                    candidate=ONLINE_CANDIDATE,
                    dynamic_available=dynamic,
                )
            )
    return output


def _selection_margin(summary: Mapping[str, Any]) -> tuple[float, float]:
    targeted = []
    for horizon in (2, 3):
        row = summary["horizons"][str(horizon)]
        for reference in ("baseline", "bias_only"):
            for scope in ("daily", "top_impact", "post_cold_start"):
                targeted.append(
                    float(row[f"{scope}_delta_vs_{reference}"]["mean"])
                )
    pooled = [
        float(row["pooled_delta_vs_baseline"])
        for row in summary["horizons"].values()
    ]
    return min(targeted), float(np.mean(pooled))


def choose_alpha(results: Mapping[str, Mapping[str, Any]]) -> str:
    if not results:
        raise ValueError("online residual calibration has no alpha results")
    return max(
        results,
        key=lambda key: (
            bool(results[key]["transfer_gate_passed"]),
            *_selection_margin(results[key]),
            -float(key),
        ),
    )


def save_state(
    output_dir: Path,
    adapters: Mapping[int, OnlineResidualAdapter],
    *,
    eligible_indices: np.ndarray,
    feature_names: Sequence[str],
    model_dir: Path,
    checkpoint: Mapping[str, Any],
    steps: np.ndarray,
    features,
    source_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "eligible_indices": np.asarray(eligible_indices, dtype=np.int64)
    }
    for horizon, adapter in adapters.items():
        for key, value in adapter.state_dict().items():
            arrays[f"h{int(horizon)}_{key}"] = value
    artifact = output_dir / "online_residual_state.npz"
    np.savez_compressed(artifact, **arrays)
    first = next(iter(adapters.values()))
    config = {
        "alpha": float(first.config.alpha),
        "ridge": float(first.config.ridge),
        "min_updates": int(first.config.min_updates),
        "gain_clip": float(first.config.gain_clip),
        "bias_clip": float(first.config.bias_clip),
        "correction_clip": float(first.config.correction_clip),
    }
    horizons = [int(value) for value in sorted(adapters)]
    chain = [] if source_metadata is None else list(source_metadata.get("source_chain", []))
    chain.append(
        {
            "model_dir": str(model_dir),
            "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
            "eval_start": str(features.dates[int(steps[0])].date()),
            "eval_end": str(features.dates[int(steps[-1])].date()),
        }
    )
    metadata = {
        "schema_version": 1,
        "candidate": ONLINE_CANDIDATE,
        "causal_update": "at origin t update only with residuals matured by t, then predict",
        "configuration_selected_on_source_chronological_holdout": (
            source_metadata is None
        ),
        "action_outputs_fed_back": False,
        "horizons": horizons,
        "feature_names": [str(value) for value in feature_names],
        "eligible_indices": [int(value) for value in eligible_indices],
        "config": config,
        "source_model_dir": str(model_dir),
        "source_checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "source_train_data_manifest_sha256": checkpoint.get(
            "train_data_manifest", {}
        ).get("sha256"),
        "source_eval_start": str(features.dates[int(steps[0])].date()),
        "source_eval_end": str(features.dates[int(steps[-1])].date()),
        "source_latest_target_date_by_horizon": {
            str(horizon): str(features.dates[int(steps[-1]) + horizon].date())
            for horizon in horizons
        },
        "source_chain": chain,
        "artifact_file": artifact.name,
        "artifact_sha256": sha256_file(artifact),
        "implementation_sha256": sha256_file(Path(__file__)),
        "adapter_implementation_sha256": sha256_file(
            ROOT / "stock_v2" / "online_residual_adapter.py"
        ),
        "live_orders_allowed": False,
    }
    metadata_path = output_dir / "online_residual_state.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def load_state(
    metadata_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1 or metadata.get("candidate") != ONLINE_CANDIDATE:
        raise ValueError("unsupported online residual state metadata")
    if metadata.get("live_orders_allowed") is not False:
        raise ValueError("online residual state does not prohibit live orders")
    artifact = metadata_path.parent / str(metadata["artifact_file"])
    if sha256_file(artifact) != metadata.get("artifact_sha256"):
        raise ValueError("online residual state artifact hash differs")
    states: dict[int, dict[str, np.ndarray]] = {}
    with np.load(artifact, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["eligible_indices"],
            np.asarray(metadata["eligible_indices"], dtype=np.int64),
        ):
            raise ValueError("online residual state eligible indices differ")
        for horizon in metadata["horizons"]:
            states[int(horizon)] = {
                key: archive[f"h{int(horizon)}_{key}"]
                for key in OnlineResidualAdapter.STATE_KEYS
            }
    return metadata, states


def build_parser() -> argparse.ArgumentParser:
    parser = build_data_parser()
    parser.description = "Calibrate or transfer a causal online matured-residual adapter."
    parser.add_argument("--mode", choices=("calibration", "transfer"), required=True)
    parser.add_argument("--state-metadata", type=Path)
    parser.add_argument("--alphas", default="0.02,0.05,0.10")
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--min-updates", type=int, default=10)
    parser.add_argument("--gain-clip", type=float, default=1.0)
    parser.add_argument("--bias-clip", type=float, default=0.5)
    parser.add_argument("--correction-clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected_role = "calibration" if args.mode == "calibration" else "test"
    if args.evaluation_role != expected_role:
        raise ValueError(f"online residual {args.mode} requires {expected_role} role")
    if args.mode == "calibration" and args.state_metadata is not None:
        raise ValueError("calibration cannot load a prior online residual state")
    if args.mode == "transfer" and args.state_metadata is None:
        raise ValueError("transfer requires prior online residual state metadata")
    if not 0.35 <= float(args.fit_fraction) <= 0.65:
        raise ValueError("online residual fit fraction must be between 0.35 and 0.65")
    args.max_steps = 0
    horizons = tuple(parse_int_list(args.horizons))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(f"online residual evaluation requires {list(REQUIRED_HORIZONS)}")
    model, checkpoint = load_model(args.model_dir, torch.device(args.device))
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(
        checkpoint_args, list(horizons), args.allow_extrapolated_horizons
    )
    features, checkpoint_args = build_features_from_ckpt(checkpoint, args)
    steps = select_steps(features, checkpoint_args, args)
    if args.limit_steps:
        steps = steps[: args.limit_steps]
    if np.any(np.diff(steps) != 1):
        raise ValueError("online residual evaluation requires contiguous steps")
    edge_cache = build_evaluation_edge_cache(features, steps, checkpoint_args, args)
    errors, baseline_rows, eligible_indices = collect_baseline_errors(
        model,
        features,
        steps,
        checkpoint,
        checkpoint_args,
        args,
        torch.device(args.device),
        edge_cache,
    )
    feature_count = int(len(eligible_indices))
    output_dir = args.output_dir / args.model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = None

    if args.mode == "calibration":
        split_index = int(round(len(steps) * float(args.fit_fraction)))
        if split_index <= 2 * max(horizons) or len(steps) - split_index < 20:
            raise ValueError("online residual calibration split is too small")
        alpha_results = {}
        rows_by_alpha = {}
        for alpha in parse_float_list(args.alphas):
            config = OnlineResidualConfig(
                alpha=alpha,
                ridge=args.ridge,
                min_updates=args.min_updates,
                gain_clip=args.gain_clip,
                bias_clip=args.bias_clip,
                correction_clip=args.correction_clip,
            )
            adapters = build_adapters(horizons, feature_count, config)
            rows = evaluate_online(
                errors, baseline_rows, adapters, score_start=split_index
            )
            rows_by_alpha[f"{alpha:g}"] = rows
            alpha_results[f"{alpha:g}"] = summarize_transfer(
                rows,
                ONLINE_CANDIDATE,
                horizons,
                impact_quantile=args.impact_quantile,
            )
        selected_key = choose_alpha(alpha_results)
        selected_result = alpha_results[selected_key]
        selected_rows = rows_by_alpha[selected_key]
        selected_config = OnlineResidualConfig(
            alpha=float(selected_key),
            ridge=args.ridge,
            min_updates=args.min_updates,
            gain_clip=args.gain_clip,
            bias_clip=args.bias_clip,
            correction_clip=args.correction_clip,
        )
        adapters = build_adapters(horizons, feature_count, selected_config)
        evaluate_online(errors, baseline_rows, adapters)
        for horizon, adapter in adapters.items():
            adapter.flush(errors[horizon])
        report = {
            "schema_version": 1,
            "mode": "calibration",
            "selection_split_start": int(split_index),
            "selection_rule": (
                "pass h2/h3 transfer checks, then maximize worst baseline/bias "
                "margin and mean pooled improvement"
            ),
            "selected_alpha": float(selected_key),
            "selected": selected_result,
            "all_alphas": alpha_results,
            "test_used_for_selection": False,
            "live_orders_allowed": False,
        }
    else:
        source_metadata, states = load_state(args.state_metadata)
        if tuple(int(value) for value in source_metadata["horizons"]) != horizons:
            raise ValueError("online residual state horizons differ from the target")
        if list(features.feature_names) != list(source_metadata["feature_names"]):
            raise ValueError("online residual state feature schema differs from the target")
        if not np.array_equal(
            eligible_indices,
            np.asarray(source_metadata["eligible_indices"], dtype=np.int64),
        ):
            raise ValueError("online residual eligible features differ from the target")
        target_start = str(features.dates[int(steps[0])].date())
        latest_source_target = max(
            str(value)
            for value in source_metadata["source_latest_target_date_by_horizon"].values()
        )
        if latest_source_target >= target_start:
            raise ValueError("online residual source targets overlap the target fold")
        config = OnlineResidualConfig(**source_metadata["config"])
        adapters = build_adapters(horizons, feature_count, config, states)
        selected_rows = evaluate_online(errors, baseline_rows, adapters)
        selected_result = summarize_transfer(
            selected_rows,
            ONLINE_CANDIDATE,
            horizons,
            impact_quantile=args.impact_quantile,
        )
        for horizon, adapter in adapters.items():
            adapter.flush(errors[horizon])
        report = {
            "schema_version": 1,
            "mode": "transfer",
            "source_state_metadata": str(args.state_metadata),
            "source_state_metadata_sha256": sha256_file(args.state_metadata),
            "selection": selected_result,
            "target_parameters_selected": False,
            "state_updates_use_matured_errors_only": True,
            "action_outputs_fed_back": False,
            "live_orders_allowed": False,
        }

    for candidate in ("baseline", "bias_only", ONLINE_CANDIDATE):
        write_csv(
            output_dir / f"{candidate}.csv",
            [row for row in selected_rows if row["candidate"] == candidate],
        )
    state_metadata = save_state(
        output_dir,
        adapters,
        eligible_indices=eligible_indices,
        feature_names=features.feature_names,
        model_dir=args.model_dir,
        checkpoint=checkpoint,
        steps=steps,
        features=features,
        source_metadata=source_metadata,
    )
    report["state_metadata"] = str(output_dir / "online_residual_state.json")
    report["state_artifact_sha256"] = state_metadata["artifact_sha256"]
    report["target_model_dir"] = str(args.model_dir)
    report["target_checkpoint_sha256"] = sha256_file(
        args.model_dir / "graph_jepa_real.pt"
    )
    report["live_orders_allowed"] = False
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    passed = bool(selected_result["transfer_gate_passed"])
    (output_dir / ("ONLINE_GATE_PASSED" if passed else "ONLINE_GATE_FAILED")).touch()
    print(
        json.dumps(
            {
                "mode": args.mode,
                "selected_alpha": float(adapters[horizons[0]].config.alpha),
                "gate_passed": passed,
                "output": str(report_path),
                "live_orders_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
