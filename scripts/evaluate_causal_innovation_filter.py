from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.evaluate_node_prediction import (
    as_namespace,
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    future_state_metrics,
    graph_edge_kwargs,
    load_model,
    select_steps,
    validate_future_rollout_contract,
    write_csv,
)
from scripts.run_real_backtest import parse_int_list, rollout_steps_for_offset
from stock_v2.innovation_filter import (
    CausalHorizonInnovationState,
    InnovationFilterConfig,
)
from stock_v2.real_features import make_real_snapshot


DEFAULT_CANDIDATES = (
    "common_a005,0.05,0.0,1.0",
    "common_a010,0.10,0.0,1.0",
    "common_a020,0.20,0.0,1.0",
    "node_a005,0.05,1.0,1.0",
    "node_a010,0.10,1.0,1.0",
    "node_a020,0.20,1.0,1.0",
    "hybrid_a005,0.05,0.5,1.0",
    "hybrid_a010,0.10,0.5,1.0",
    "hybrid_a020,0.20,0.5,1.0",
)
REQUIRED_HORIZONS = (1, 2, 3, 5, 10)
SELECTION_HORIZONS = (2, 3)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_candidate_spec(spec: str) -> InnovationFilterConfig:
    values = [value.strip() for value in str(spec).split(",")]
    if len(values) != 4:
        raise ValueError(
            "innovation candidate must be name,alpha,node_mix,clip"
        )
    name = values[0]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("innovation candidate name contains unsafe characters")
    return InnovationFilterConfig(
        name=name,
        alpha=float(values[1]),
        node_mix=float(values[2]),
        clip=float(values[3]),
    )


def _pooled_skill(rows: Sequence[dict[str, Any]]) -> float:
    model_sse = sum(float(row["model_sse"]) for row in rows)
    persistence_sse = sum(float(row["persistence_sse"]) for row in rows)
    return (
        float(1.0 - model_sse / persistence_sse)
        if persistence_sse > 1e-12
        else float("nan")
    )


def summarize_candidates(
    rows: list[dict[str, Any]],
    configs: Sequence[InnovationFilterConfig],
    horizons: Sequence[int],
    *,
    impact_quantile: float,
    warmup_sessions: int,
    pooled_floor: float = -0.002,
) -> dict[str, object]:
    if not 0.0 < impact_quantile < 1.0:
        raise ValueError("impact_quantile must be between 0 and 1")
    baseline = {
        (int(row["session_ordinal"]), int(row["horizon"])): row
        for row in rows
        if row["candidate"] == "baseline"
    }
    if not baseline:
        raise ValueError("innovation summary requires baseline rows")
    output: dict[str, object] = {}
    eligible_candidates = []
    for config in configs:
        candidate_rows = {
            (int(row["session_ordinal"]), int(row["horizon"])): row
            for row in rows
            if row["candidate"] == config.name
        }
        if set(candidate_rows) != set(baseline):
            raise ValueError(
                f"innovation rows do not align for candidate {config.name}"
            )
        horizon_results = {}
        for horizon in horizons:
            keys = sorted(
                key for key in baseline if int(key[1]) == int(horizon)
            )
            base_rows = [baseline[key] for key in keys]
            filtered_rows = [candidate_rows[key] for key in keys]
            base_daily = np.asarray(
                [float(row["mse_skill_vs_persistence"]) for row in base_rows],
                dtype=np.float64,
            )
            filtered_daily = np.asarray(
                [
                    float(row["mse_skill_vs_persistence"])
                    for row in filtered_rows
                ],
                dtype=np.float64,
            )
            delta = filtered_daily - base_daily
            energy = np.asarray(
                [
                    float(row["persistence_sse"])
                    / float(row["observed_cells"])
                    for row in base_rows
                ],
                dtype=np.float64,
            )
            threshold = float(np.quantile(energy, impact_quantile))
            top_mask = energy >= threshold
            post_warmup = np.asarray(
                [int(row["session_ordinal"]) >= warmup_sessions for row in base_rows],
                dtype=bool,
            )
            if not top_mask.any() or not post_warmup.any():
                raise ValueError("innovation diagnostic stratum is empty")
            base_pooled = _pooled_skill(base_rows)
            filtered_pooled = _pooled_skill(filtered_rows)
            horizon_results[str(int(horizon))] = {
                "rows": len(keys),
                "baseline_pooled_skill": base_pooled,
                "filtered_pooled_skill": filtered_pooled,
                "pooled_delta_vs_baseline": filtered_pooled - base_pooled,
                "daily_delta_vs_baseline": newey_west_mean(
                    delta,
                    lag=int(horizon),
                ),
                "top_impact_delta_vs_baseline": newey_west_mean(
                    delta[top_mask],
                    lag=int(horizon),
                ),
                "post_warmup_delta_vs_baseline": newey_west_mean(
                    delta[post_warmup],
                    lag=int(horizon),
                ),
                "top_impact_post_warmup_delta_vs_baseline": newey_west_mean(
                    delta[top_mask & post_warmup],
                    lag=int(horizon),
                ),
                "impact_threshold": threshold,
                "top_impact_rows": int(top_mask.sum()),
                "mean_correction_rms": float(
                    np.mean(
                        [float(row["correction_rms"]) for row in filtered_rows]
                    )
                ),
            }
        floor_pass = all(
            horizon_results[str(int(horizon))]["pooled_delta_vs_baseline"]
            >= pooled_floor
            for horizon in horizons
        )
        targeted_pass = all(
            horizon_results[str(horizon)]["daily_delta_vs_baseline"]["mean"]
            >= 0.0
            and horizon_results[str(horizon)][
                "top_impact_delta_vs_baseline"
            ]["mean"]
            >= 0.0
            and horizon_results[str(horizon)][
                "post_warmup_delta_vs_baseline"
            ]["mean"]
            >= 0.0
            for horizon in SELECTION_HORIZONS
        )
        targeted_values = [
            horizon_results[str(horizon)][
                "top_impact_delta_vs_baseline"
            ]["mean"]
            for horizon in SELECTION_HORIZONS
        ]
        all_values = [
            horizon_results[str(horizon)]["daily_delta_vs_baseline"]["mean"]
            for horizon in SELECTION_HORIZONS
        ]
        score = float(min(targeted_values) + 0.5 * min(all_values))
        candidate_result = {
            "config": {
                "alpha": config.alpha,
                "node_mix": config.node_mix,
                "clip": config.clip,
            },
            "horizons": horizon_results,
            "selection_floor_passed": bool(floor_pass),
            "selection_targeted_passed": bool(targeted_pass),
            "selection_eligible": bool(floor_pass and targeted_pass),
            "selection_score": score,
        }
        output[config.name] = candidate_result
        if candidate_result["selection_eligible"]:
            eligible_candidates.append((score, config.name))

    selected = (
        max(eligible_candidates, key=lambda item: (item[0], item[1]))[1]
        if eligible_candidates
        else None
    )
    return {
        "impact_quantile": float(impact_quantile),
        "warmup_sessions": int(warmup_sessions),
        "pooled_delta_floor": float(pooled_floor),
        "selection_horizons": list(SELECTION_HORIZONS),
        "candidates": output,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
    }


def evaluate_filters(
    model,
    features,
    steps: np.ndarray,
    ckpt: dict[str, Any],
    ckpt_args: dict[str, Any],
    cli_args: argparse.Namespace,
    device: torch.device,
    configs: Sequence[InnovationFilterConfig],
    edge_cache,
) -> tuple[list[dict[str, Any]], dict[str, object]]:
    horizons = tuple(parse_int_list(cli_args.horizons))
    if tuple(sorted(horizons)) != tuple(horizons):
        raise ValueError("innovation horizons must be sorted")
    stock_count = int(features.tradable_count)
    feature_count = len(features.feature_names)
    temporal_weights = ckpt.get("temporal_state_feature_weights")
    if temporal_weights is None:
        eligible_features = np.ones(feature_count, dtype=bool)
    else:
        if torch.is_tensor(temporal_weights):
            temporal_weights = temporal_weights.detach().cpu().numpy()
        eligible_features = (
            np.asarray(temporal_weights, dtype=np.float32) > 0.0
        )
    if eligible_features.shape != (feature_count,):
        raise ValueError(
            "temporal state feature weights do not match checkpoint features"
        )
    states = {
        config.name: {
            horizon: CausalHorizonInnovationState(
                stock_count,
                feature_count,
                eligible_features,
                config,
            )
            for horizon in horizons
        }
        for config in configs
    }
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(
        cli_args.min_abs_corr
        if cli_args.min_abs_corr is not None
        else ckpt_args.get("min_abs_corr", 0.2)
    )
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault(
        "temporal_offset",
        ckpt_args.get("horizon", max(horizons)),
    )
    rollout_args.setdefault("latent_rollout_steps", 1)
    rollout_namespace = as_namespace(rollout_args)
    rows: list[dict[str, Any]] = []

    for ordinal, raw_step in enumerate(steps):
        step = int(raw_step)
        current_target = features.features[step, :stock_count]
        current_available = features.available_mask[step, :stock_count] > 0.5
        for config in configs:
            for horizon in horizons:
                states[config.name][horizon].mature(
                    step,
                    current_target,
                    current_available,
                )

        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, cli_args),
            edge_cache=edge_cache,
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)
        x0 = features.features[step, :stock_count]
        source_available = features.available_mask[step, :stock_count] > 0.5
        for horizon in horizons:
            target_step = step + int(horizon)
            target = features.features[target_step, :stock_count]
            target_available = (
                features.available_mask[target_step, :stock_count] > 0.5
            )
            rollout_steps = rollout_steps_for_offset(
                rollout_namespace,
                int(horizon),
            )
            with torch.no_grad():
                latent = model.rollout_latent(context, steps=rollout_steps)
                baseline_prediction = model.predict_temporal_state(
                    batch,
                    latent,
                    rollout_steps=rollout_steps,
                    z_context=context,
                ).detach().cpu().numpy()[:stock_count]
            baseline_metrics = future_state_metrics(
                baseline_prediction,
                target,
                x0,
                target_available,
                source_available,
            )
            if baseline_metrics is None:
                raise ValueError("baseline innovation evaluation produced no cells")
            base_row = {
                "candidate": "baseline",
                "session_ordinal": int(ordinal),
                "date": str(features.dates[step].date()),
                "target_date": str(features.dates[target_step].date()),
                "horizon": int(horizon),
                "rollout_steps": int(rollout_steps),
                "correction_rms": 0.0,
                "update_calls": 0,
                **baseline_metrics,
            }
            rows.append(base_row)
            for config in configs:
                state = states[config.name][horizon]
                corrected = state.correct_and_enqueue(
                    step,
                    int(horizon),
                    baseline_prediction,
                    source_available,
                )
                filtered_metrics = future_state_metrics(
                    corrected,
                    target,
                    x0,
                    target_available,
                    source_available,
                )
                if filtered_metrics is None:
                    raise ValueError(
                        f"innovation candidate {config.name} produced no cells"
                    )
                diagnostics = state.filter.diagnostics()
                rows.append(
                    {
                        "candidate": config.name,
                        "session_ordinal": int(ordinal),
                        "date": str(features.dates[step].date()),
                        "target_date": str(features.dates[target_step].date()),
                        "horizon": int(horizon),
                        "rollout_steps": int(rollout_steps),
                        "correction_rms": diagnostics["correction_rms"],
                        "update_calls": diagnostics["update_calls"],
                        **filtered_metrics,
                    }
                )
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(steps):
            print(
                f"innovation evaluation rows={ordinal + 1}/{len(steps)}",
                flush=True,
            )

    final_memory = {
        config.name: {
            str(horizon): states[config.name][horizon].filter.diagnostics()
            for horizon in horizons
        }
        for config in configs
    }
    return rows, final_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a causal matured-forecast innovation filter.",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-role", choices=["calibration", "test"], required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--impact-quantile", type=float, default=0.75)
    parser.add_argument("--warmup-sessions", type=int, default=20)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--edge-cache-workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--override-universe", action="store_true")
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    parser.add_argument("--allow-extrapolated-horizons", action="store_true")
    parser.add_argument("--edge-window", type=int, default=None)
    parser.add_argument("--edge-top-k", type=int, default=None)
    parser.add_argument("--min-abs-corr", type=float, default=None)
    parser.add_argument("--edge-correlation-mode", default=None)
    parser.add_argument("--partial-corr-top-k", type=int, default=None)
    parser.add_argument("--partial-corr-min-abs", type=float, default=None)
    parser.add_argument("--partial-corr-mode", default=None)
    parser.add_argument("--partial-corr-scale", type=float, default=None)
    parser.add_argument("--lead-lag-top-k", type=int, default=None)
    parser.add_argument("--lead-lag-days", type=int, default=None)
    parser.add_argument("--lead-lag-min-abs-corr", type=float, default=None)
    parser.add_argument("--lead-lag-mode", default=None)
    parser.add_argument("--lead-lag-scale", type=float, default=None)
    parser.add_argument("--policy-rate-edge-scale", type=float, default=None)
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--event-path", action="append", default=[])
    parser.add_argument("--event-half-life-days", type=float, default=None)
    parser.add_argument("--event-lag-days", type=int, default=None)
    parser.add_argument("--event-max-decay-days", type=int, default=None)
    parser.add_argument("--event-edge-top-k", type=int, default=None)
    parser.add_argument("--event-edge-min-weight", type=float, default=None)
    parser.add_argument("--event-edge-scale", type=float, default=None)
    parser.add_argument("--event-edge-max-themes", type=int, default=None)
    parser.add_argument("--event-edge-min-theme-count", type=int, default=None)
    parser.add_argument("--industry-profile-path", action="append", default=[])
    parser.add_argument("--industry-prefix-length", type=int, default=None)
    parser.add_argument("--industry-edge-scale", type=float, default=None)
    parser.add_argument("--fundamental-path", action="append", default=[])
    parser.add_argument("--fundamental-lag-days", type=int, default=None)
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", type=int, default=None)
    parser.add_argument("--external-preset", default=None)
    parser.add_argument("--external-symbol", action="append", default=[])
    parser.add_argument("--external-lag-days", type=int, default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.warmup_sessions < 0 or args.limit_steps < 0:
        raise ValueError("warmup and limit steps must be non-negative")
    args.max_steps = 0
    specs = tuple(args.candidate or DEFAULT_CANDIDATES)
    configs = tuple(parse_candidate_spec(spec) for spec in specs)
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("innovation candidate names must be unique")
    horizons = tuple(parse_int_list(args.horizons))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(
            f"innovation evaluation requires horizons {list(REQUIRED_HORIZONS)}"
        )

    device = torch.device(args.device)
    model, checkpoint = load_model(args.model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(
        checkpoint_args,
        list(horizons),
        args.allow_extrapolated_horizons,
    )
    features, checkpoint_args = build_features_from_ckpt(checkpoint, args)
    steps = select_steps(features, checkpoint_args, args)
    if args.limit_steps:
        steps = steps[: args.limit_steps]
    if len(steps) < max(horizons) + 3:
        raise ValueError("innovation evaluation needs a contiguous maturity window")
    if np.any(np.diff(steps) != 1):
        raise ValueError("innovation evaluation steps must be contiguous")
    edge_cache = build_evaluation_edge_cache(
        features,
        steps,
        checkpoint_args,
        args,
    )
    rows, final_memory = evaluate_filters(
        model,
        features,
        steps,
        checkpoint,
        checkpoint_args,
        args,
        device,
        configs,
        edge_cache,
    )
    summary = summarize_candidates(
        rows,
        configs,
        horizons,
        impact_quantile=args.impact_quantile,
        warmup_sessions=args.warmup_sessions,
    )
    output_dir = args.output_dir / args.model_dir.name
    daily_dir = output_dir / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("baseline", *(config.name for config in configs)):
        write_csv(
            daily_dir / f"{candidate}.csv",
            [row for row in rows if row["candidate"] == candidate],
        )
    report = {
        "schema_version": 1,
        "evaluation_role": args.evaluation_role,
        "causal_contract": {
            "ordering": "mature residuals at t, then forecast from t",
            "cold_start": True,
            "closed_loop_residual": True,
            "stock_nodes_only": True,
            "temporal_target_features_only": True,
            "test_fitted_parameters": False,
            "action_outputs_fed_back": False,
        },
        "model_dir": str(args.model_dir),
        "checkpoint_sha256": sha256_file(args.model_dir / "graph_jepa_real.pt"),
        "implementation_sha256": {
            "evaluation_script": sha256_file(Path(__file__)),
            "innovation_filter": sha256_file(
                ROOT / "stock_v2" / "innovation_filter.py"
            ),
        },
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "train_data_manifest_sha256": checkpoint.get(
            "train_data_manifest", {}
        ).get("sha256"),
        "eval_start": str(features.dates[int(steps[0])].date()),
        "eval_end": str(features.dates[int(steps[-1])].date()),
        "eval_steps": int(len(steps)),
        "horizons": list(horizons),
        "selection": summary,
        "final_memory": final_memory,
        "live_orders_allowed": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = (
        "CALIBRATION_SELECTED"
        if args.evaluation_role == "calibration" and summary["selection_passed"]
        else "CALIBRATION_REJECTED"
        if args.evaluation_role == "calibration"
        else "TEST_COMPLETE"
    )
    (output_dir / marker).touch()
    print(
        json.dumps(
            {
                "evaluation_role": args.evaluation_role,
                "selected_candidate": summary["selected_candidate"],
                "selection_passed": summary["selection_passed"],
                "output": str(output_dir / "summary.json"),
                "live_orders_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
