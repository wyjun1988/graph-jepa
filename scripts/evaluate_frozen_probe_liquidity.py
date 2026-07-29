from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
    rows_for_steps,
)
from scripts.benchmark_frozen_downstream import (
    ProbeInputs,
    load_or_build_latent_cache,
    predict_probe,
)
from scripts.benchmark_qlib_lgb import evaluate_signal_frame, newey_west_mean
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from stock_v2.downstream_probes import FrozenEncoderProbe, causal_probe_splits
from stock_v2.latent_path_head import sha256_file


EVALUATION_METRICS = (
    "return_path_ic",
    "return_path_rank_ic",
    "return_path_ic_top300",
    "return_path_rank_ic_top300",
    "return_path_decile_spread",
    "return_path_decile_spread_top300",
)


def parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)) or any(item < 1 for item in parsed):
        raise ValueError("values must be unique positive integers")
    return parsed


def parse_names(value: str) -> list[str]:
    allowed = {"raw", "raw_latent", "raw_shuffled_latent"}
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(parsed) - allowed)
    if not parsed or unknown:
        raise ValueError(f"invalid probe variants: {unknown or parsed}")
    return list(dict.fromkeys(parsed))


def paired_daily_metric(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    metric: str,
    horizon: int,
) -> dict[str, float | int]:
    required = {"date", metric}
    for name, frame in (("candidate", candidate), ("baseline", baseline)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} daily frame is missing columns: {sorted(missing)}")
        if frame["date"].duplicated().any():
            raise ValueError(f"{name} daily frame contains duplicate dates")
    left = candidate[["date", metric]].rename(columns={metric: "candidate"})
    right = baseline[["date", metric]].rename(columns={metric: "baseline"})
    joined = left.merge(right, on="date", how="inner", validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("paired daily frames do not contain identical dates")
    difference = joined["candidate"].to_numpy(dtype=np.float64) - joined[
        "baseline"
    ].to_numpy(dtype=np.float64)
    return newey_west_mean(difference, lag=int(horizon))


def signal_frame(
    prediction: np.ndarray,
    labels: np.ndarray,
    liquidity: np.ndarray,
    current_available: np.ndarray,
    dates: Sequence[Any],
    tickers: Sequence[str],
) -> pd.DataFrame:
    date_count = len(dates)
    stock_count = len(tickers)
    expected = (date_count, stock_count)
    arrays = {
        "prediction": np.asarray(prediction),
        "label": np.asarray(labels),
        "liquidity": np.asarray(liquidity),
        "current_available": np.asarray(current_available),
    }
    for name, values in arrays.items():
        if values.shape != expected:
            raise ValueError(f"{name} shape does not match the test panel")
    index = pd.MultiIndex.from_product(
        [pd.DatetimeIndex(dates), [str(value) for value in tickers]],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame(
        {
            "prediction": arrays["prediction"].reshape(-1),
            "label": arrays["label"].reshape(-1),
            "liquidity": arrays["liquidity"].reshape(-1),
            "current_available": arrays["current_available"].reshape(-1).astype(bool),
        },
        index=index,
    )


def load_probe(
    artifact_path: Path,
    expected_sha256: str,
    variant: str,
    horizon: int,
    parent_sha256: str,
    hidden_dim: int,
    layers: int,
    dropout: float,
    device: torch.device,
) -> FrozenEncoderProbe:
    if sha256_file(artifact_path) != str(expected_sha256):
        raise ValueError(f"probe artifact SHA-256 mismatch: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    if artifact.get("live_orders_allowed") is not False:
        raise ValueError("probe artifact must remain research-only")
    if str(artifact.get("input_variant")) != str(variant):
        raise ValueError("probe artifact input variant mismatch")
    if int(artifact.get("horizon", -1)) != int(horizon):
        raise ValueError("probe artifact horizon mismatch")
    if str(artifact.get("parent_model_sha256")) != str(parent_sha256):
        raise ValueError("probe artifact parent checkpoint mismatch")
    model = FrozenEncoderProbe(
        int(artifact["input_dim"]),
        hidden_dim=int(hidden_dim),
        layers=int(layers),
        dropout=float(dropout),
    ).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    return model


def qlib_test_rows(path: Path, horizon: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"horizon", "date", "split", *EVALUATION_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Qlib daily metrics are missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["horizon"].astype(int) == int(horizon))
        & (frame["split"].astype(str) == "test")
    ].copy()
    if selected.empty:
        raise ValueError("Qlib daily metrics contain no matching test rows")
    selected["date"] = selected["date"].astype(str)
    return selected.sort_values("date").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen probes with the exact Qlib liquidity-top-k evaluation."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--probe-report", required=True)
    parser.add_argument("--raw-context-cache", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--qlib-daily", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--split-horizons", default="5,10")
    parser.add_argument(
        "--variants", default="raw,raw_latent,raw_shuffled_latent"
    )
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    horizon = int(args.horizon)
    split_horizons = parse_ints(args.split_horizons)
    if horizon not in split_horizons:
        raise ValueError("the evaluation horizon must be included in split horizons")
    variants = parse_names(args.variants)
    if int(args.liquidity_top_k) < 3:
        raise ValueError("liquidity top-k must be at least three")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_dir = Path(args.model_dir)
    probe_report = Path(args.probe_report)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    probe_summary_path = probe_report / "summary.json"
    probe_summary = json.loads(probe_summary_path.read_text(encoding="utf-8"))
    if probe_summary.get("status") != "complete":
        raise ValueError("probe report is incomplete")

    graph_model, checkpoint = load_model(model_dir, device)
    graph_model._checkpoint_path = str(  # type: ignore[attr-defined]
        model_dir / "graph_jepa_real.pt"
    )
    parent_sha256 = sha256_file(model_dir / "graph_jepa_real.pt")
    if parent_sha256 != str(probe_summary.get("checkpoint_sha256")):
        raise ValueError("probe report parent checkpoint mismatch")
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = deepcopy(args)
    configured_horizons = checkpoint_args.get("rollout_offsets", split_horizons)
    if isinstance(configured_horizons, str):
        feature_args.horizons = configured_horizons
    else:
        feature_args.horizons = ",".join(
            str(int(value)) for value in configured_horizons
        )
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint,
        evaluator_namespace(feature_args),
    )
    splits = causal_probe_splits(
        features.dates,
        train_end=str(checkpoint_args["train_end"]),
        edge_window=int(checkpoint_args.get("edge_window", 60)),
        max_horizon=max(split_horizons),
        validation_days=int(args.validation_days),
        max_test_steps=int(args.max_test_steps),
        test_end=str(args.test_end),
    )
    all_steps = np.unique(
        np.concatenate([splits.fit_steps, splits.validation_steps, splits.test_steps])
    ).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    stock_count = int(features.tradable_count)
    layout = build_context_layout(features, splits.fit_steps)
    raw = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        checkpoint,
        checkpoint_args,
        workers=1,
        cache_path=Path(args.raw_context_cache),
    )
    context, deltas, latent_contract = load_or_build_latent_cache(
        graph_model,
        features,
        checkpoint,
        checkpoint_args,
        all_steps,
        split_horizons,
        Path(args.latent_cache_dir),
        device,
    )
    del graph_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    inputs = ProbeInputs(
        raw,
        context,
        deltas,
        stock_count=stock_count,
        seed=int(args.seed),
    )
    test_rows = rows_for_steps(splits.test_steps, step_positions, stock_count)
    labels = np.asarray(
        features.target_return_paths[horizon][splits.test_steps, :stock_count],
        dtype=np.float64,
    )
    liquidity_index = features.feature_names.index("value_ma20_log")
    liquidity = np.asarray(
        features.raw_features[
            splits.test_steps, :stock_count, liquidity_index
        ],
        dtype=np.float64,
    )
    available = np.asarray(
        features.available_mask[splits.test_steps, :stock_count]
    )
    if available.ndim == 3:
        available = available.any(axis=2)
    elif available.ndim != 2:
        raise ValueError("current availability mask has an unsupported shape")
    dates = pd.DatetimeIndex(features.dates[splits.test_steps])
    tickers = [str(value) for value in features.node_tickers[:stock_count]]

    expected_test_dates = [str(value.date()) for value in dates]
    if expected_test_dates[0] != str(probe_summary.get("test_start")):
        raise ValueError("probe report test start does not match replay split")
    if expected_test_dates[-1] != str(probe_summary.get("test_end")):
        raise ValueError("probe report test end does not match replay split")

    output_dir.mkdir(parents=True)
    variant_daily: dict[str, pd.DataFrame] = {}
    variant_metrics: dict[str, dict[str, Any]] = {}
    for variant in variants:
        result = probe_summary["results"][str(horizon)]["single"][variant]
        artifact_path = probe_report / "models" / f"h{horizon}_single_{variant}.pt"
        probe = load_probe(
            artifact_path,
            str(result["artifact_sha256"]),
            variant,
            horizon,
            parent_sha256,
            int(args.hidden_dim),
            int(args.layers),
            float(args.dropout),
            device,
        )
        continuous, _ = predict_probe(
            probe,
            inputs,
            variant,
            horizon,
            test_rows,
            int(args.batch_size),
            device,
            bool(args.amp),
        )
        prediction = continuous[:, 0].reshape(len(splits.test_steps), stock_count)
        frame = signal_frame(
            prediction,
            labels,
            liquidity,
            available,
            dates,
            tickers,
        )
        daily_rows, metrics = evaluate_signal_frame(
            frame,
            horizon=horizon,
            liquidity_top_k=int(args.liquidity_top_k),
        )
        daily = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)
        daily["variant"] = variant
        daily.to_csv(output_dir / f"daily_{variant}.csv", index=False)
        variant_daily[variant] = daily
        variant_metrics[variant] = metrics
        del probe
        if device.type == "cuda":
            torch.cuda.empty_cache()

    qlib = qlib_test_rows(Path(args.qlib_daily), horizon)
    qlib_dates = qlib["date"].tolist()
    if qlib_dates != expected_test_dates:
        raise ValueError("Qlib and replay test dates do not match")

    top_metric = "return_path_ic_top300"
    raw_latent = variant_daily["raw_latent"]
    raw = variant_daily["raw"]
    shuffled = variant_daily["raw_shuffled_latent"]
    hybrid_vs_raw = paired_daily_metric(raw_latent, raw, top_metric, horizon)
    hybrid_vs_shuffled = paired_daily_metric(
        raw_latent, shuffled, top_metric, horizon
    )
    qlib_vs_hybrid = paired_daily_metric(qlib, raw_latent, top_metric, horizon)
    qlib_superior = bool(
        float(qlib_vs_hybrid["mean"]) > 0.0
        and float(qlib_vs_hybrid["newey_west_t"]) >= 1.96
    )
    result = {
        "schema_version": 1,
        "role": "frozen_probe_exact_qlib_liquidity_challenge",
        "research_only": True,
        "live_orders_allowed": False,
        "horizon": horizon,
        "liquidity_top_k": int(args.liquidity_top_k),
        "test_start": expected_test_dates[0],
        "test_end": expected_test_dates[-1],
        "test_dates": len(expected_test_dates),
        "checkpoint_sha256": parent_sha256,
        "probe_summary_sha256": sha256_file(probe_summary_path),
        "qlib_daily_sha256": sha256_file(Path(args.qlib_daily)),
        "latent_cache_contract": latent_contract,
        "metrics": variant_metrics,
        "top300_paired": {
            "raw_latent_minus_raw": hybrid_vs_raw,
            "raw_latent_minus_shuffled_latent": hybrid_vs_shuffled,
            "qlib_minus_raw_latent": qlib_vs_hybrid,
        },
        "decision": {
            "directional_increment": bool(
                float(hybrid_vs_raw["mean"]) > 0.0
                and float(hybrid_vs_shuffled["mean"]) > 0.0
            ),
            "robust_increment": bool(
                float(hybrid_vs_raw["mean"]) > 0.0
                and float(hybrid_vs_shuffled["mean"]) > 0.0
                and float(hybrid_vs_raw["newey_west_t"]) >= 1.96
                and float(hybrid_vs_shuffled["newey_west_t"]) >= 1.96
            ),
            "qlib_significantly_superior": qlib_superior,
            "passed_qlib_noninferiority_challenge": not qlib_superior,
            "promotion_eligible_from_this_evaluation_alone": False,
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
