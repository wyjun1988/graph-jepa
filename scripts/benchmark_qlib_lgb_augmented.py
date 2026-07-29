from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_qlib_lgb import (
    build_index,
    evaluate_signal_frame,
    load_context_matrix,
    newey_west_mean,
    sha256_file,
    validate_contract,
)


AUGMENTATION_VARIANTS = ("raw_latent", "raw_shuffled_latent")
EVALUATION_METRICS = (
    "return_path_ic",
    "return_path_rank_ic",
    "return_path_ic_top300",
    "return_path_rank_ic_top300",
    "return_path_decile_spread",
    "return_path_decile_spread_top300",
)


def parse_names(value: str, allowed: Sequence[str]) -> list[str]:
    parsed = [item.strip() for item in str(value).split(",") if item.strip()]
    unknown = sorted(set(parsed) - set(allowed))
    if not parsed or unknown:
        raise ValueError(f"invalid augmentation variants: {unknown or parsed}")
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


def append_latent_features(
    context: np.ndarray,
    selected: np.ndarray,
    mode: str,
    stock_count: int,
    seed: int,
) -> np.ndarray:
    if mode not in AUGMENTATION_VARIANTS:
        raise ValueError(f"unsupported augmentation mode: {mode}")
    if context.ndim != 2 or selected.ndim != 2 or len(context) != len(selected):
        raise ValueError("raw and latent feature matrices must be aligned 2D arrays")
    if len(context) % int(stock_count):
        raise ValueError("feature rows do not contain complete date blocks")
    result = np.empty(
        (len(context), context.shape[1] + selected.shape[1]), dtype=np.float32
    )
    result[:, : context.shape[1]] = context
    destination = result[:, context.shape[1] :]
    if mode == "raw_latent":
        destination[:] = selected
        return result
    generator = np.random.default_rng(int(seed))
    for start in range(0, len(context), int(stock_count)):
        end = start + int(stock_count)
        destination[start:end] = selected[
            start + generator.permutation(int(stock_count))
        ]
    return result


def validate_latent_contract(
    metadata: dict[str, Any],
    bundle_contract: dict[str, Any],
    dates: np.ndarray,
    selected: np.ndarray,
) -> None:
    required = {
        "checkpoint_sha256",
        "rows",
        "date_count",
        "stock_count",
        "dates",
        "feature_count",
        "selected_indices",
        "selected_matrix_sha256",
    }
    missing = required.difference(metadata)
    if missing:
        raise ValueError(f"latent metadata is missing fields: {sorted(missing)}")
    if str(metadata["checkpoint_sha256"]) != str(bundle_contract["checkpoint_sha256"]):
        raise ValueError("latent and Qlib bundle checkpoint hashes do not match")
    expected_dates = [str(pd.Timestamp(value).date()) for value in dates]
    if list(metadata["dates"]) != expected_dates:
        raise ValueError("latent and Qlib bundle dates do not match")
    expected_shape = (int(bundle_contract["rows"]), int(metadata["feature_count"]))
    if selected.shape != expected_shape:
        raise ValueError("selected latent matrix shape does not match its contract")
    if int(metadata["rows"]) != expected_shape[0]:
        raise ValueError("latent metadata row count mismatch")
    if int(metadata["date_count"]) != len(dates):
        raise ValueError("latent metadata date count mismatch")
    if int(metadata["stock_count"]) != int(bundle_contract["stocks"]):
        raise ValueError("latent metadata stock count mismatch")
    if len(metadata["selected_indices"]) != expected_shape[1]:
        raise ValueError("latent selected index count mismatch")


def qlib_raw_test_rows(path: Path, horizon: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"horizon", "date", "split", *EVALUATION_METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"raw Qlib daily metrics are missing columns: {sorted(missing)}")
    selected = frame.loc[
        (frame["horizon"].astype(int) == int(horizon))
        & (frame["split"].astype(str) == "test")
    ].copy()
    if selected.empty:
        raise ValueError("raw Qlib daily metrics contain no matching test rows")
    selected["date"] = selected["date"].astype(str)
    return selected.sort_values("date").reset_index(drop=True)


def summarize_variant(
    variant: str,
    context: np.ndarray,
    selected: np.ndarray,
    feature_names: Sequence[str],
    selected_names: Sequence[str],
    index: pd.MultiIndex,
    raw_label: np.ndarray,
    meta_frame: pd.DataFrame,
    segments: dict[str, tuple[str, str]],
    output_dir: Path,
    horizon: int,
    stock_count: int,
    seed: int,
    num_threads: int,
    num_boost_round: int,
    early_stopping_rounds: int,
    liquidity_top_k: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    from qlib.data.dataset.processor import CSRankNorm, DropnaLabel
    from qlib.workflow import R

    values = append_latent_features(
        context, selected, variant, stock_count=stock_count, seed=seed
    )
    columns = pd.MultiIndex.from_tuples(
        [("feature", str(name)) for name in [*feature_names, *selected_names]]
    )
    feature_frame = pd.DataFrame(values, index=index, columns=columns, copy=False)
    panel = feature_frame.copy(deep=False)
    panel[("label", "LABEL0")] = raw_label
    panel.columns = pd.MultiIndex.from_tuples(panel.columns)
    handler = DataHandlerLP(
        data_loader=StaticDataLoader(panel),
        start_time=str(index.get_level_values("datetime").min().date()),
        end_time=str(index.get_level_values("datetime").max().date()),
        infer_processors=[],
        learn_processors=[DropnaLabel(), CSRankNorm(fields_group="label")],
    )
    dataset = DatasetH(handler=handler, segments=segments)
    model = LGBModel(
        loss="mse",
        num_boost_round=int(num_boost_round),
        early_stopping_rounds=int(early_stopping_rounds),
        colsample_bytree=0.8879,
        learning_rate=0.0421,
        subsample=0.8789,
        lambda_l1=205.6999,
        lambda_l2=580.9768,
        max_depth=8,
        num_leaves=210,
        num_threads=max(1, int(num_threads)),
        verbosity=-1,
        seed=int(seed),
        feature_fraction_seed=int(seed),
        bagging_seed=int(seed),
        data_random_seed=int(seed),
        deterministic=True,
        force_col_wise=True,
    )
    evals_result: dict[str, Any] = {}
    with R.start(
        experiment_name="stock_v2_qlib_lgb_augmented",
        recorder_name=f"horizon_{horizon}_{variant}_seed_{seed}",
    ):
        model.fit(dataset, verbose_eval=25, evals_result=evals_result)
    model_path = output_dir / f"lightgbm_h{horizon}_{variant}.txt"
    model.model.save_model(str(model_path))

    summary: dict[str, Any] = {
        "best_iteration": int(model.model.best_iteration),
        "model_sha256": sha256_file(model_path),
        "validation_used_for_early_stopping": True,
        "test_used_for_selection": False,
        "splits": {},
        "evals_result": evals_result,
    }
    all_daily: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        prediction = model.predict(dataset, segment=split).rename("prediction")
        evaluation = meta_frame.join(prediction, how="inner")
        evaluation["label"] = raw_label[index.get_indexer(evaluation.index)]
        evaluation = evaluation[
            ["prediction", "label", "liquidity", "current_available"]
        ]
        daily_rows, split_summary = evaluate_signal_frame(
            evaluation,
            horizon=int(horizon),
            liquidity_top_k=int(liquidity_top_k),
        )
        for row in daily_rows:
            row["split"] = split
            row["variant"] = variant
        all_daily.extend(daily_rows)
        summary["splits"][split] = split_summary
    daily = pd.DataFrame(all_daily)
    daily.to_csv(output_dir / f"daily_metrics_{variant}.csv", index=False)

    del model, dataset, handler, panel, feature_frame, values
    gc.collect()
    return summary, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run exact Qlib LightGBM with fit-selected JEPA latent features."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--latent-dir", required=True)
    parser.add_argument("--raw-daily", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--variants", default=",".join(AUGMENTATION_VARIANTS))
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    horizon = int(args.horizon)
    variants = parse_names(args.variants, AUGMENTATION_VARIANTS)
    bundle_dir = Path(args.bundle_dir).resolve()
    latent_dir = Path(args.latent_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    contract = validate_contract(bundle_dir)
    if horizon not in [int(value) for value in contract["horizons"]]:
        raise ValueError("requested horizon is absent from the Qlib bundle")
    metadata_path = latent_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    selected_path = latent_dir / str(metadata["selected_matrix"])
    if sha256_file(selected_path) != str(metadata["selected_matrix_sha256"]):
        raise ValueError("selected latent matrix SHA-256 mismatch")

    arrays = np.load(bundle_dir / str(contract["arrays_file"]), allow_pickle=False)
    dates = arrays["dates"]
    labels = arrays["labels"]
    liquidity = arrays["liquidity"].reshape(-1)
    current_available = arrays["current_available"].reshape(-1).astype(bool)
    tickers = json.loads(
        (bundle_dir / str(contract["tickers_file"])).read_text(encoding="utf-8")
    )
    context = load_context_matrix(bundle_dir, contract)
    selected = np.load(selected_path, mmap_mode="r")
    validate_latent_contract(metadata, contract, dates, selected)
    index = build_index(dates, tickers)
    if len(index) != int(contract["rows"]) or context.shape[0] != len(index):
        raise ValueError("bundle arrays and context rows are not aligned")
    expected_label_shape = (len(contract["horizons"]), len(dates), len(tickers))
    if labels.shape != expected_label_shape:
        raise ValueError("bundle label shape does not match its contract")

    horizon_position = [int(value) for value in contract["horizons"]].index(horizon)
    raw_label = labels[horizon_position].reshape(-1).astype(np.float32, copy=True)
    raw_label[~current_available] = np.nan
    meta_frame = pd.DataFrame(
        {"liquidity": liquidity, "current_available": current_available}, index=index
    )
    segments = {
        "train": (
            contract["splits"]["fit"]["start"],
            contract["splits"]["fit"]["end"],
        ),
        "valid": (
            contract["splits"]["validation"]["start"],
            contract["splits"]["validation"]["end"],
        ),
        "test": (
            contract["splits"]["test"]["start"],
            contract["splits"]["test"]["end"],
        ),
    }
    feature_names = [str(value) for value in contract["feature_names"]]
    hidden_dim = int(metadata["hidden_dim"])
    selected_names = [
        (
            f"jepa_context_{int(index_value)}"
            if int(index_value) < hidden_dim
            else f"jepa_h{horizon}_delta_{int(index_value) - hidden_dim}"
        )
        for index_value in metadata["selected_indices"]
    ]

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.num_threads)))
    import lightgbm
    import qlib
    from qlib.constant import REG_CN

    output_dir.mkdir(parents=True)
    workflow_dir = output_dir / ".qlib_workflow"
    workflow_dir.mkdir()
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workflow_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    provider_dir = workflow_dir / "empty_provider"
    provider_dir.mkdir()
    original_cwd = Path.cwd()
    os.chdir(workflow_dir)
    qlib.init(provider_uri=str(provider_dir), region=REG_CN)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "role": "research_only_qlib_lightgbm_jepa_augmented",
        "live_orders_allowed": False,
        "framework": "Microsoft Qlib",
        "qlib_version": str(qlib.__version__),
        "lightgbm_version": str(lightgbm.__version__),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "selection_rule": "latent coordinates selected on fit only; fixed LightGBM hyperparameters; validation-only early stopping",
        "label_processor": "DropnaLabel then cross-sectional rank normalization",
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "bundle_contract_sha256": contract["bundle_contract_sha256"],
        "latent_metadata_sha256": sha256_file(metadata_path),
        "latent_matrix_sha256": metadata["selected_matrix_sha256"],
        "horizon": horizon,
        "raw_features": int(contract["feature_count"]),
        "latent_features": int(metadata["feature_count"]),
        "segments": segments,
        "seed": int(args.seed),
        "variants": {},
    }
    daily_by_variant: dict[str, pd.DataFrame] = {}
    try:
        for variant in variants:
            variant_summary, daily = summarize_variant(
                variant,
                context,
                selected,
                feature_names,
                selected_names,
                index,
                raw_label,
                meta_frame,
                segments,
                output_dir,
                horizon,
                int(contract["stocks"]),
                int(args.seed),
                int(args.num_threads),
                int(args.num_boost_round),
                int(args.early_stopping_rounds),
                int(args.liquidity_top_k),
            )
            summary["variants"][variant] = variant_summary
            daily_by_variant[variant] = daily.loc[daily["split"] == "test"].copy()
            (output_dir / "summary.partial.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        os.chdir(original_cwd)

    raw_daily = qlib_raw_test_rows(Path(args.raw_daily), horizon)
    top_metric = "return_path_ic_top300"
    real = daily_by_variant["raw_latent"]
    shuffled = daily_by_variant["raw_shuffled_latent"]
    real_vs_raw = paired_daily_metric(real, raw_daily, top_metric, horizon)
    real_vs_shuffled = paired_daily_metric(real, shuffled, top_metric, horizon)
    raw_vs_real = paired_daily_metric(raw_daily, real, top_metric, horizon)
    summary["top300_paired"] = {
        "raw_latent_minus_raw": real_vs_raw,
        "raw_latent_minus_shuffled_latent": real_vs_shuffled,
        "raw_minus_raw_latent": raw_vs_real,
    }
    summary["decision"] = {
        "directional_increment": bool(
            float(real_vs_raw["mean"]) > 0.0
            and float(real_vs_shuffled["mean"]) > 0.0
        ),
        "robust_increment": bool(
            float(real_vs_raw["mean"]) > 0.0
            and float(real_vs_shuffled["mean"]) > 0.0
            and float(real_vs_raw["newey_west_t"]) >= 1.96
            and float(real_vs_shuffled["newey_west_t"]) >= 1.96
        ),
        "raw_significantly_superior": bool(
            float(raw_vs_real["mean"]) > 0.0
            and float(raw_vs_real["newey_west_t"]) >= 1.96
        ),
        "promotion_eligible_from_this_evaluation_alone": False,
    }
    summary["status"] = "complete"
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "summary.partial.json").unlink(missing_ok=True)
    print(json.dumps(summary["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
