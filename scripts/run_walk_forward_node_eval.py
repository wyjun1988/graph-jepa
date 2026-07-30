from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def parse_fold(value: str) -> tuple[str, str]:
    if ":" not in value:
        raise argparse.ArgumentTypeError("fold must be TRAIN_END:EVAL_END")
    train_end, eval_end = value.split(":", 1)
    return train_end.strip(), eval_end.strip()


def read_skill(summary_path: Path) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    current = summary["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"]
    future = {
        f"h{h}": row.get(
            "pooled_mse_skill_vs_persistence",
            row["mse_skill_vs_persistence"]["mean"],
        )
        for h, row in summary["future_rollout_by_horizon"].items()
    }
    corr = {
        f"h{h}": row["delta_corr"]["mean"]
        for h, row in summary["future_rollout_by_horizon"].items()
    }
    grouped_skill = {
        f"h{h}": {
            group: row["mse_skill_vs_persistence"]["mean"]
            for group, row in groups.items()
        }
        for h, groups in summary.get("future_rollout_by_horizon_feature_group", {}).items()
    }
    return {
        "tickers": summary["tickers"],
        "features": summary["features"],
        "eval_steps": summary["eval_steps"],
        "eval_start": summary["eval_start"],
        "eval_end": summary["eval_end"],
        "evaluation_seed": summary.get("evaluation_seed"),
        "current_skill": current,
        "future_skill_metric": summary.get(
            "future_skill_metric",
            "mean_date_level_mse_skill_vs_persistence",
        ),
        "future_skill": future,
        "future_delta_corr": corr,
        "future_feature_group_skill": grouped_skill,
    }


def read_completed_fold(
    reports_dir: Path,
    models_dir: Path,
    node_summary_path: Path,
    expected_data_sha256: str | None,
    expected_edge_sha256: str | None,
) -> dict[str, Any] | None:
    required = [
        reports_dir / "training_data_manifest.json",
        reports_dir / "training_edge_manifest.json",
        models_dir / "graph_jepa_real.pt",
        node_summary_path,
    ]
    if not all(path.is_file() for path in required):
        return None
    data_manifest = json.loads(required[0].read_text(encoding="utf-8"))
    edge_manifest = json.loads(required[1].read_text(encoding="utf-8"))
    node_summary = json.loads(node_summary_path.read_text(encoding="utf-8"))
    if node_summary.get("live_orders_allowed") is not False:
        raise ValueError("completed fold node summary is not explicitly research-only")
    for role, expected, actual in (
        ("data", expected_data_sha256, data_manifest.get("sha256")),
        ("edge", expected_edge_sha256, edge_manifest.get("sha256")),
    ):
        if expected and str(actual) != str(expected):
            raise ValueError(f"completed fold {role} manifest differs from frozen SHA")
    return read_skill(node_summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run walk-forward Graph-JEPA node-state evaluations.")
    parser.add_argument("--name", default="walk_forward")
    parser.add_argument("--fold", action="append", type=parse_fold, required=True, help="TRAIN_END:EVAL_END; repeatable")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--universe", choices=["manual", "krx"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument(
        "--training-manifest-schema-version",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
    )
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--edge-top-k", type=int, default=6)
    parser.add_argument("--edge-correlation-mode", choices=["signed", "abs", "positive", "negative", "none"], default="signed")
    parser.add_argument("--graph-neighbor-scale", type=float, default=1.0)
    parser.add_argument("--temporal-graph-neighbor-scale", type=float, default=None)
    parser.add_argument("--temporal-stock-edge-scale", type=float, default=1.0)
    parser.add_argument("--global-stock-context", action="store_true")
    parser.add_argument("--partial-corr-top-k", type=int, default=0)
    parser.add_argument("--partial-corr-min-abs", type=float, default=0.10)
    parser.add_argument("--partial-corr-mode", choices=["signed", "abs", "positive", "negative"], default="signed")
    parser.add_argument("--partial-corr-scale", type=float, default=0.50)
    parser.add_argument("--lead-lag-top-k", type=int, default=0)
    parser.add_argument("--lead-lag-days", type=int, default=1)
    parser.add_argument("--lead-lag-min-abs-corr", type=float, default=0.08)
    parser.add_argument("--lead-lag-mode", choices=["signed", "abs", "positive", "negative"], default="signed")
    parser.add_argument("--lead-lag-scale", type=float, default=0.50)
    parser.add_argument("--policy-rate-edge-scale", type=float, default=0.0)
    parser.add_argument("--ownership-edge-scale", type=float, default=0.0)
    parser.add_argument("--ownership-edge-path", default=None)
    parser.add_argument("--earnings-features", action="store_true")
    parser.add_argument("--return-lag-features", type=int, default=0)
    parser.add_argument("--sequence-window", type=int, default=0)
    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--sequence-heads", type=int, default=8)
    parser.add_argument("--sequence-residual", action="store_true")
    parser.add_argument("--event-edge-top-k", type=int, default=0)
    parser.add_argument("--event-edge-min-weight", type=float, default=0.05)
    parser.add_argument("--event-edge-scale", type=float, default=0.25)
    parser.add_argument("--industry-profile-path", action="append", default=[])
    parser.add_argument("--industry-prefix-length", type=int, default=2)
    parser.add_argument("--industry-edge-scale", type=float, default=0.20)
    parser.add_argument("--require-industry-edges", action="store_true")
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--ema-decay", type=float, default=0.9995)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.35)
    parser.add_argument("--current-imputation-loss-weight", type=float, default=0.0)
    parser.add_argument("--temporal-state-context-skip", action="store_true")
    parser.add_argument(
        "--temporal-head-input",
        choices=["context_skip", "future", "context"],
        default=None,
        help="예측 헤드 입력 (context=미래 잠재 미사용). 미지정 시 하위호환 유도.",
    )
    parser.add_argument("--state-feature-weight", action="append", default=[])
    parser.add_argument(
        "--temporal-exclude-feature-prefix",
        action="append",
        default=[],
    )
    parser.add_argument("--return-correlation-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--entry-path-correlation-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--downstream-auxiliary-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--downstream-path-weight", type=float, default=1.0)
    parser.add_argument("--downstream-mfe-weight", type=float, default=0.25)
    parser.add_argument("--downstream-mae-weight", type=float, default=0.25)
    parser.add_argument("--downstream-continuation-weight", type=float, default=1.0)
    parser.add_argument("--downstream-volatility-weight", type=float, default=1.0)
    parser.add_argument("--downstream-market-loss-weight", type=float, default=0.0)
    parser.add_argument("--downstream-market-cost-bps", type=float, default=50.0)
    parser.add_argument(
        "--downstream-transition-loss-weight", type=float, default=0.0
    )
    parser.add_argument(
        "--downstream-transition-pooling",
        choices=["mean", "robust", "robust_projected"],
        default="mean",
    )
    parser.add_argument("--temporal-impact-loss-mix", type=float, default=0.0)
    parser.add_argument("--normalize-predictor-output", action="store_true")
    parser.add_argument(
        "--temporal-state-mode",
        choices=[
            "direct",
            "residual_mixed",
            "horizon_hybrid",
            "horizon_residual_heads",
        ],
        default="direct",
    )
    parser.add_argument("--temporal-residual-short-steps", type=int, default=2)
    parser.add_argument("--hybrid-fast-direct", action="store_true")
    parser.add_argument("--pretrain-task", choices=["temporal", "masked"], default="temporal")
    parser.add_argument("--hide-ratio", type=float, default=None)
    parser.add_argument("--temporal-offset", type=int, default=10)
    parser.add_argument("--latent-rollout-steps", type=int, default=10)
    parser.add_argument("--rollout-offsets", default="1,2,3,5,10")
    parser.add_argument("--rollout-loss-weights", default="")
    parser.add_argument("--path-horizons", default="1,2,3,5,10")
    parser.add_argument("--mask-strategy", default="mixed")
    parser.add_argument("--hidden-completion-weight", type=float, default=0.0)
    parser.add_argument("--fund-hidden-target-path", default=None)
    parser.add_argument("--latent-variance-weight", type=float, default=0.0)
    parser.add_argument("--latent-covariance-weight", type=float, default=0.0)
    parser.add_argument("--latent-variance-target", type=float, default=1.0)
    parser.add_argument("--imputation-anchor", action="store_true")
    parser.add_argument("--init-encoder-from", default=None)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--fund-yoy-input-path", default=None)
    parser.add_argument("--fund-yoy-input-mode", default="own")
    parser.add_argument("--fund-hidden-min-peers", type=int, default=3)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--snapshot-workers", type=int, default=24)
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="none",
    )
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--eval-device", default="cuda")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--save-return-forecasts",
        action="store_true",
        help="write per-ticker per-day forecasts so runs can be pooled afterwards",
    )
    parser.add_argument("--checkpoint-epochs", default="")
    parser.add_argument("--event-path", action="append", default=[])
    parser.add_argument(
        "--event-coverage-mode",
        choices=["mask_uncovered", "legacy_all_observed"],
        default="mask_uncovered",
    )
    parser.add_argument("--require-event-sensors", action="store_true")
    parser.add_argument("--min-event-coverage", default="0.50")
    parser.add_argument("--fundamental-path", action="append", default=[])
    parser.add_argument("--fundamental-lag-days", default="1")
    parser.add_argument("--require-fundamental-sensors", action="store_true")
    parser.add_argument("--min-fundamental-coverage", default="0.50")
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", default="1")
    parser.add_argument("--require-investor-sensors", action="store_true")
    parser.add_argument("--min-investor-coverage", default="0.50")
    parser.add_argument(
        "--external-preset",
        choices=["none", "kr_global", "kr_global_rates"],
        default="none",
    )
    parser.add_argument("--external-symbol", action="append", default=[])
    parser.add_argument("--external-node-mode", choices=["features", "nodes", "both"], default="features")
    parser.add_argument("--external-lag-days", default="1")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--external-cache-dir", default="data/external_cache")
    parser.add_argument("--external-etf-panel", default=None)
    parser.add_argument(
        "--expected-training-manifest-sha256",
        action="append",
        default=[],
        help="Frozen training-panel SHA-256; repeat once per fold.",
    )
    parser.add_argument(
        "--expected-training-edge-manifest-sha256",
        action="append",
        default=[],
        help="Frozen causal training-edge SHA-256; repeat once per fold.",
    )
    parser.add_argument("--require-all-external-factors", action="store_true")
    parser.add_argument("--reports-root", default="reports/walk_forward")
    parser.add_argument("--models-root", default="models/walk_forward")
    parser.add_argument("--summary-output", default="reports/walk_forward/summary.json")
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Train each fold without running node-state evaluation.",
    )
    parser.add_argument(
        "--edge-manifest-only",
        action="store_true",
        help="Build each fold's causal edge manifest without training or evaluation.",
    )
    parser.add_argument(
        "--resume-complete-folds",
        action="store_true",
        help="Skip a fully verified fold when its model, manifests, and node summary exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.training_only and args.edge_manifest_only:
        raise ValueError(
            "--training-only and --edge-manifest-only are mutually exclusive"
        )
    if args.expected_training_manifest_sha256 and (
        len(args.expected_training_manifest_sha256) != len(args.fold)
    ):
        raise ValueError(
            "--expected-training-manifest-sha256 must be repeated once per fold"
        )
    if args.expected_training_edge_manifest_sha256 and (
        len(args.expected_training_edge_manifest_sha256) != len(args.fold)
    ):
        raise ValueError(
            "--expected-training-edge-manifest-sha256 must be repeated once per fold"
        )
    rows = []
    for idx, (train_end, eval_end) in enumerate(args.fold, start=1):
        fold_name = f"{args.name}_fold{idx}_{train_end}_to_{eval_end}".replace("-", "")
        reports_dir = Path(args.reports_root) / fold_name
        models_dir = Path(args.models_root) / fold_name
        node_summary_path = (
            Path(args.reports_root) / "node_eval" / models_dir.name / "summary.json"
        )
        if args.resume_complete_folds and not (
            args.training_only or args.edge_manifest_only
        ):
            completed = read_completed_fold(
                reports_dir,
                models_dir,
                node_summary_path,
                (
                    args.expected_training_manifest_sha256[idx - 1]
                    if args.expected_training_manifest_sha256
                    else None
                ),
                (
                    args.expected_training_edge_manifest_sha256[idx - 1]
                    if args.expected_training_edge_manifest_sha256
                    else None
                ),
            )
            if completed is not None:
                print(f"RESUME verified complete fold {fold_name}", flush=True)
                rows.append(
                    {
                        "fold": fold_name,
                        "train_end": train_end,
                        "eval_end": eval_end,
                        "model_dir": str(models_dir),
                        "reports_dir": str(reports_dir),
                        **completed,
                    }
                )
                continue
        cmd = [
            sys.executable, "scripts/run_real_backtest.py",
            "--start", args.start,
            "--end", eval_end,
            "--train-end", train_end,
            "--universe", args.universe,
            "--max-tickers", str(args.max_tickers),
            "--epochs", str(args.epochs),
            "--hidden-dim", str(args.hidden_dim),
            "--layers", str(args.layers),
            "--horizon", str(args.horizon),
            "--top-k", str(args.top_k),
            "--edge-top-k", str(args.edge_top_k),
            "--graph-neighbor-scale", str(args.graph_neighbor_scale),
            "--lr", str(args.lr),
            "--ema-decay", str(args.ema_decay),
            "--latent-loss-weight", str(args.latent_loss_weight),
            "--state-loss-weight", str(args.state_loss_weight),
            "--current-imputation-loss-weight", str(args.current_imputation_loss_weight),
            "--return-correlation-loss-weight", str(args.return_correlation_loss_weight),
            "--entry-path-correlation-loss-weight",
            str(args.entry_path_correlation_loss_weight),
            "--downstream-auxiliary-loss-weight",
            str(args.downstream_auxiliary_loss_weight),
            "--downstream-path-weight",
            str(args.downstream_path_weight),
            "--downstream-mfe-weight",
            str(args.downstream_mfe_weight),
            "--downstream-mae-weight",
            str(args.downstream_mae_weight),
            "--downstream-volatility-weight",
            str(args.downstream_volatility_weight),
            "--downstream-continuation-weight",
            str(args.downstream_continuation_weight),
            "--downstream-market-loss-weight",
            str(args.downstream_market_loss_weight),
            "--downstream-market-cost-bps",
            str(args.downstream_market_cost_bps),
            "--downstream-transition-loss-weight",
            str(args.downstream_transition_loss_weight),
            "--downstream-transition-pooling",
            args.downstream_transition_pooling,
            "--temporal-impact-loss-mix",
            str(args.temporal_impact_loss_mix),
            "--temporal-state-mode", args.temporal_state_mode,
            "--temporal-residual-short-steps", str(args.temporal_residual_short_steps),
            "--pretrain-task", args.pretrain_task,
            "--temporal-offset", str(args.temporal_offset),
            "--latent-rollout-steps", str(args.latent_rollout_steps),
            "--rollout-offsets", str(args.rollout_offsets),
            "--mask-strategy", args.mask_strategy,
            "--train-batch-size", str(args.train_batch_size),
            "--snapshot-workers", str(args.snapshot_workers),
            "--amp-dtype", args.amp_dtype,
            *(["--hide-ratio", str(args.hide_ratio)] if args.hide_ratio is not None else []),
            "--max-train-steps", str(args.max_train_steps),
            "--path-horizons", args.path_horizons,
            "--device", args.device,
            "--seed", str(args.seed),
            "--cache-dir", args.cache_dir,
            "--training-manifest-schema-version",
            str(args.training_manifest_schema_version),
            "--skip-return-backtest",
            "--reports-dir", str(reports_dir),
            "--models-dir", str(models_dir),
        ]
        if args.min_train_rows is not None:
            cmd.extend(["--min-train-rows", str(args.min_train_rows)])
        if args.checkpoint_epochs:
            cmd.extend(["--checkpoint-epochs", args.checkpoint_epochs])
        if args.temporal_graph_neighbor_scale is not None:
            cmd.extend([
                "--temporal-graph-neighbor-scale",
                str(args.temporal_graph_neighbor_scale),
            ])
        cmd.extend(
            ["--temporal-stock-edge-scale", str(args.temporal_stock_edge_scale)]
        )
        if args.global_stock_context:
            cmd.append("--global-stock-context")
        if args.rollout_loss_weights:
            cmd.extend(["--rollout-loss-weights", str(args.rollout_loss_weights)])
        for state_feature_weight in args.state_feature_weight:
            cmd.extend(["--state-feature-weight", state_feature_weight])
        for feature_prefix in args.temporal_exclude_feature_prefix:
            cmd.extend(["--temporal-exclude-feature-prefix", feature_prefix])
        if args.hybrid_fast_direct:
            cmd.append("--hybrid-fast-direct")
        if args.temporal_state_context_skip:
            cmd.append("--temporal-state-context-skip")
        if args.temporal_head_input:
            cmd.extend(["--temporal-head-input", args.temporal_head_input])
        if args.normalize_predictor_output:
            cmd.append("--normalize-predictor-output")
        if args.universe_manifest:
            cmd.extend(["--universe-manifest", args.universe_manifest])
        cmd.extend([
            "--edge-correlation-mode", args.edge_correlation_mode,
            "--partial-corr-top-k", str(args.partial_corr_top_k),
            "--partial-corr-min-abs", str(args.partial_corr_min_abs),
            "--partial-corr-mode", args.partial_corr_mode,
            "--partial-corr-scale", str(args.partial_corr_scale),
            "--lead-lag-top-k", str(args.lead_lag_top_k),
            "--lead-lag-days", str(args.lead_lag_days),
            "--lead-lag-min-abs-corr", str(args.lead_lag_min_abs_corr),
            "--lead-lag-mode", args.lead_lag_mode,
            "--lead-lag-scale", str(args.lead_lag_scale),
            "--policy-rate-edge-scale", str(args.policy_rate_edge_scale),
            "--ownership-edge-scale", str(args.ownership_edge_scale),
            *(["--ownership-edge-path", args.ownership_edge_path] if args.ownership_edge_path else []),
            *(["--earnings-features"] if args.earnings_features else []),
            "--return-lag-features", str(args.return_lag_features),
            "--sequence-window", str(args.sequence_window),
            "--sequence-layers", str(args.sequence_layers),
            "--sequence-heads", str(args.sequence_heads),
            *(["--sequence-residual"] if args.sequence_residual else []),
            "--event-edge-top-k", str(args.event_edge_top_k),
            "--event-edge-min-weight", str(args.event_edge_min_weight),
            "--event-edge-scale", str(args.event_edge_scale),
            "--industry-prefix-length", str(args.industry_prefix_length),
            "--industry-edge-scale", str(args.industry_edge_scale),
        ])
        for industry_profile_path in args.industry_profile_path:
            cmd.extend(["--industry-profile-path", industry_profile_path])
        if args.require_industry_edges:
            cmd.append("--require-industry-edges")
        for event_path in args.event_path:
            cmd.extend(["--event-path", event_path])
        cmd.extend(["--event-coverage-mode", args.event_coverage_mode])
        if args.require_event_sensors:
            cmd.extend([
                "--require-event-sensors",
                "--min-event-coverage", str(args.min_event_coverage),
            ])
        for fundamental_path in args.fundamental_path:
            cmd.extend(["--fundamental-path", fundamental_path])
        if args.fundamental_path:
            cmd.extend(["--fundamental-lag-days", str(args.fundamental_lag_days)])
        if args.require_fundamental_sensors:
            cmd.extend([
                "--require-fundamental-sensors",
                "--min-fundamental-coverage", str(args.min_fundamental_coverage),
            ])
        if args.investor_cache_dir:
            cmd.extend([
                "--investor-cache-dir", args.investor_cache_dir,
                "--investor-flow-lag-days", str(args.investor_flow_lag_days),
            ])
        if args.require_investor_sensors:
            cmd.extend([
                "--require-investor-sensors",
                "--min-investor-coverage", str(args.min_investor_coverage),
            ])
        if args.external_preset != "none":
            cmd.extend(["--external-preset", args.external_preset])
        if args.require_all_external_factors:
            cmd.append("--require-all-external-factors")
        for external_symbol in args.external_symbol:
            cmd.extend(["--external-symbol", external_symbol])
        cmd.extend(["--external-node-mode", args.external_node_mode])
        cmd.extend(["--external-lag-days", str(args.external_lag_days)])
        cmd.extend(["--external-cache-dir", args.external_cache_dir])
        if args.external_etf_panel:
            cmd.extend(["--external-etf-panel", args.external_etf_panel])
        if args.hidden_completion_weight > 0.0:
            cmd.extend(["--hidden-completion-weight", str(args.hidden_completion_weight)])
        if args.latent_variance_weight > 0.0:
            cmd.extend(["--latent-variance-weight", str(args.latent_variance_weight),
                        "--latent-variance-target", str(args.latent_variance_target)])
        if args.latent_covariance_weight > 0.0:
            cmd.extend(["--latent-covariance-weight", str(args.latent_covariance_weight)])
        if args.imputation_anchor:
            cmd.append("--imputation-anchor")
        if args.init_encoder_from:
            cmd.extend(["--init-encoder-from", args.init_encoder_from])
        if args.freeze_encoder:
            cmd.append("--freeze-encoder")
        if args.fund_yoy_input_path:
            cmd.extend(["--fund-yoy-input-path", args.fund_yoy_input_path,
                        "--fund-yoy-input-mode", args.fund_yoy_input_mode])
        if args.fund_hidden_target_path:
            cmd.extend(["--fund-hidden-target-path", args.fund_hidden_target_path,
                        "--fund-hidden-min-peers", str(args.fund_hidden_min_peers)])
        if args.expected_training_manifest_sha256:
            cmd.extend([
                "--expected-training-manifest-sha256",
                args.expected_training_manifest_sha256[idx - 1],
            ])
        if args.expected_training_edge_manifest_sha256:
            cmd.extend([
                "--expected-training-edge-manifest-sha256",
                args.expected_training_edge_manifest_sha256[idx - 1],
            ])
        if args.edge_manifest_only:
            cmd.append("--edge-manifest-only")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        if args.training_only or args.edge_manifest_only:
            rows.append(
                {
                    "fold": fold_name,
                    "train_end": train_end,
                    "eval_end": eval_end,
                    "model_dir": str(models_dir),
                    "reports_dir": str(reports_dir),
                    "training_only": bool(args.training_only),
                    "edge_manifest_only": bool(args.edge_manifest_only),
                }
            )
            continue
        eval_cmd = [
            sys.executable, "scripts/evaluate_node_prediction.py",
            "--model-dir", str(models_dir),
            "--output-dir", str(Path(args.reports_root) / "node_eval"),
            "--horizons", args.path_horizons,
            "--mask-strategy", args.mask_strategy,
            "--max-steps", str(args.max_steps),
            "--device", args.eval_device,
            "--seed", str(args.seed),
        ]
        if args.save_return_forecasts:
            eval_cmd.append("--save-return-forecasts")
        for event_path in args.event_path:
            eval_cmd.extend(["--event-path", event_path])
        for fundamental_path in args.fundamental_path:
            eval_cmd.extend(["--fundamental-path", fundamental_path])
        if args.fundamental_path:
            eval_cmd.extend(["--fundamental-lag-days", str(args.fundamental_lag_days)])
        if args.investor_cache_dir:
            eval_cmd.extend([
                "--investor-cache-dir", args.investor_cache_dir,
                "--investor-flow-lag-days", str(args.investor_flow_lag_days),
            ])
        if args.external_preset != "none":
            eval_cmd.extend(["--external-preset", args.external_preset])
        for external_symbol in args.external_symbol:
            eval_cmd.extend(["--external-symbol", external_symbol])
        eval_cmd.extend(["--external-lag-days", str(args.external_lag_days)])
        eval_cmd.extend(["--external-cache-dir", args.external_cache_dir])
        if args.external_etf_panel:
            eval_cmd.extend(["--external-etf-panel", args.external_etf_panel])
        print("EVAL", " ".join(eval_cmd), flush=True)
        subprocess.run(eval_cmd, cwd=ROOT, check=True)
        summary_path = node_summary_path
        row = {
            "fold": fold_name,
            "train_end": train_end,
            "eval_end": eval_end,
            "model_dir": str(models_dir),
            "reports_dir": str(reports_dir),
            **read_skill(summary_path),
        }
        rows.append(row)
    out = Path(args.summary_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "name": args.name,
        "folds": rows,
        "live_orders_allowed": False,
    }
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
