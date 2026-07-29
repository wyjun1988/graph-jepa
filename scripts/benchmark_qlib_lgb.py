from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def newey_west_mean(values: Sequence[float], lag: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 3:
        return {"rows": int(len(array)), "mean": float("nan"), "newey_west_t": float("nan")}
    centered = array - array.mean()
    long_variance = float(centered @ centered / len(array))
    max_lag = min(max(0, int(lag)), len(array) - 1)
    for offset in range(1, max_lag + 1):
        weight = 1.0 - offset / (max_lag + 1.0)
        covariance = float(centered[offset:] @ centered[:-offset] / len(array))
        long_variance += 2.0 * weight * covariance
    standard_error = float(np.sqrt(max(long_variance, 0.0) / len(array)))
    mean = float(array.mean())
    return {
        "rows": int(len(array)),
        "mean": mean,
        "newey_west_lag": int(max_lag),
        "newey_west_standard_error": standard_error,
        "newey_west_t": float(mean / standard_error) if standard_error > 1e-12 else float("nan"),
        "positive_fraction": float((array > 0.0).mean()),
    }


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    x = left[valid].astype(np.float64)
    y = right[valid].astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    x = pd.Series(left[valid]).rank(method="average").to_numpy(dtype=np.float64)
    y = pd.Series(right[valid]).rank(method="average").to_numpy(dtype=np.float64)
    return _correlation(x, y)


def _decile_spread(score: np.ndarray, realized: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(realized)
    if valid.sum() < 20:
        return float("nan")
    score = score[valid]
    realized = realized[valid]
    count = max(1, len(score) // 10)
    order = np.argsort(score, kind="stable")
    return float(realized[order[-count:]].mean() - realized[order[:count]].mean())


def evaluate_signal_frame(
    frame: pd.DataFrame,
    horizon: int,
    liquidity_top_k: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    required = {"prediction", "label", "liquidity", "current_available"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"signal frame is missing columns: {sorted(missing)}")
    if not isinstance(frame.index, pd.MultiIndex) or frame.index.names != [
        "datetime",
        "instrument",
    ]:
        raise ValueError("signal frame index must be (datetime, instrument)")
    if frame.index.has_duplicates:
        raise ValueError("signal frame contains duplicate date/instrument rows")

    rows: list[dict[str, Any]] = []
    for date, daily in frame.groupby(level="datetime", sort=True):
        prediction = daily["prediction"].to_numpy(dtype=np.float64)
        label = daily["label"].to_numpy(dtype=np.float64)
        liquidity = daily["liquidity"].to_numpy(dtype=np.float64)
        available = daily["current_available"].to_numpy(dtype=bool)
        valid = available & np.isfinite(prediction) & np.isfinite(label)
        liquid_candidates = np.flatnonzero(available & np.isfinite(liquidity))
        if len(liquid_candidates) > int(liquidity_top_k):
            order = np.argsort(liquidity[liquid_candidates], kind="stable")
            liquid_candidates = liquid_candidates[order[-int(liquidity_top_k) :]]
        top_liquidity = np.zeros(len(daily), dtype=bool)
        top_liquidity[liquid_candidates] = True
        top_valid = valid & top_liquidity
        rows.append(
            {
                "horizon": int(horizon),
                "date": str(pd.Timestamp(date).date()),
                "observations": int(valid.sum()),
                "top_liquidity_observations": int(top_valid.sum()),
                "return_path_ic": _correlation(prediction[valid], label[valid]),
                "return_path_rank_ic": _rank_correlation(prediction[valid], label[valid]),
                "return_path_ic_top300": _correlation(
                    prediction[top_valid], label[top_valid]
                ),
                "return_path_rank_ic_top300": _rank_correlation(
                    prediction[top_valid], label[top_valid]
                ),
                "return_path_decile_spread": _decile_spread(
                    prediction[valid], label[valid]
                ),
                "return_path_decile_spread_top300": _decile_spread(
                    prediction[top_valid], label[top_valid]
                ),
            }
        )
    metrics = [
        "return_path_ic",
        "return_path_rank_ic",
        "return_path_ic_top300",
        "return_path_rank_ic_top300",
        "return_path_decile_spread",
        "return_path_decile_spread_top300",
    ]
    summary = {
        metric: newey_west_mean(
            [float(row[metric]) for row in rows], lag=int(horizon)
        )
        for metric in metrics
    }
    summary["dates"] = int(len(rows))
    summary["observations"] = int(sum(row["observations"] for row in rows))
    return rows, summary


def validate_contract(bundle_dir: Path) -> dict[str, Any]:
    contract_path = bundle_dir / "bundle_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1:
        raise ValueError("unsupported Qlib PIT bundle schema")
    if contract.get("test_used_for_selection") is not False:
        raise ValueError("bundle must prohibit test-set selection")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("bundle must prohibit live orders")
    expected_contract_hash = contract.pop("bundle_contract_sha256", None)
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    actual_contract_hash = hashlib.sha256(canonical).hexdigest()
    contract["bundle_contract_sha256"] = expected_contract_hash
    if expected_contract_hash != actual_contract_hash:
        raise ValueError("bundle contract hash does not match")
    for relative, expected in contract.get("artifact_sha256", {}).items():
        path = bundle_dir / relative
        if not path.exists() or sha256_file(path) != expected:
            raise ValueError(f"bundle artifact hash does not match: {relative}")
    return contract


def load_context_matrix(bundle_dir: Path, contract: dict[str, Any]) -> np.ndarray:
    cache_path = bundle_dir / str(contract["context_cache"])
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in metadata.get("shape", ()))
    if len(shape) != 2 or shape[0] != int(contract["rows"]):
        raise ValueError("context cache row count does not match bundle")
    if metadata.get("dtype") != "float32":
        raise ValueError("context cache must use float32")
    if metadata.get("format") == "single_npy":
        matrix = np.load(cache_path, mmap_mode="r")
    elif metadata.get("format") == "chunked_npy":
        matrix = np.empty(shape, dtype=np.float32)
        parts_dir = cache_path.with_suffix(cache_path.suffix + ".parts")
        cursor = 0
        for part in metadata.get("parts", []):
            start = int(part["start"])
            end = int(part["end"])
            if start != cursor or end <= start:
                raise ValueError("context cache parts are not contiguous")
            values = np.load(parts_dir / str(part["file"]), mmap_mode="r")
            if values.shape != (end - start, shape[1]):
                raise ValueError("context cache part shape does not match")
            matrix[start:end] = values
            cursor = end
        if cursor != shape[0]:
            raise ValueError("context cache parts do not cover every row")
    else:
        raise ValueError("unsupported context cache format")
    feature_count = int(contract["feature_count"])
    if matrix.shape[1] < feature_count:
        raise ValueError("context cache has fewer columns than the bundle contract")
    return matrix[:, :feature_count]


def build_index(dates: np.ndarray, tickers: list[str]) -> pd.MultiIndex:
    datetime_values = np.repeat(np.asarray(dates, dtype="datetime64[ns]"), len(tickers))
    instrument_values = np.tile(np.asarray(tickers, dtype=object), len(dates))
    index = pd.MultiIndex.from_arrays(
        [datetime_values, instrument_values], names=["datetime", "instrument"]
    )
    if index.has_duplicates:
        raise ValueError("bundle index contains duplicate rows")
    return index


def _parse_horizons(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or len(result) != len(set(result)) or any(item <= 0 for item in result):
        raise ValueError("horizons must be unique positive integers")
    return result


def _json_default(value: Any):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a fixed Qlib LightGBM baseline on the Graph-JEPA PIT panel."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--num-boost-round", type=int, default=500)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--num-threads", type=int, default=10)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons)
    contract = validate_contract(bundle_dir)
    if not set(horizons).issubset(set(int(value) for value in contract["horizons"])):
        raise ValueError("requested horizon is absent from the PIT bundle")
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    os.environ["OMP_NUM_THREADS"] = str(max(1, int(args.num_threads)))

    arrays = np.load(bundle_dir / str(contract["arrays_file"]), allow_pickle=False)
    dates = arrays["dates"]
    labels = arrays["labels"]
    liquidity = arrays["liquidity"].reshape(-1)
    current_available = arrays["current_available"].reshape(-1).astype(bool)
    tickers = json.loads(
        (bundle_dir / str(contract["tickers_file"])).read_text(encoding="utf-8")
    )
    context = load_context_matrix(bundle_dir, contract)
    index = build_index(dates, tickers)
    if len(index) != int(contract["rows"]) or context.shape[0] != len(index):
        raise ValueError("bundle arrays and context rows are not aligned")
    expected_label_shape = (len(contract["horizons"]), len(dates), len(tickers))
    if labels.shape != expected_label_shape:
        raise ValueError("bundle label shape does not match its contract")

    import lightgbm
    import qlib
    from qlib.constant import REG_CN
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader
    from qlib.data.dataset.processor import CSRankNorm, DropnaLabel
    from qlib.workflow import R

    workflow_dir = output_dir / ".qlib_workflow"
    workflow_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-q"],
        cwd=workflow_dir,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    provider_dir = workflow_dir / "empty_provider"
    provider_dir.mkdir(exist_ok=True)
    original_cwd = Path.cwd()
    os.chdir(workflow_dir)
    qlib.init(provider_uri=str(provider_dir), region=REG_CN)

    feature_names = [str(value) for value in contract["feature_names"]]
    feature_columns = pd.MultiIndex.from_tuples(
        [("feature", name) for name in feature_names]
    )
    feature_frame = pd.DataFrame(context, index=index, columns=feature_columns, copy=False)
    meta_frame = pd.DataFrame(
        {"liquidity": liquidity, "current_available": current_available}, index=index
    )
    horizon_positions = {
        int(horizon): position for position, horizon in enumerate(contract["horizons"])
    }
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
    summary: dict[str, Any] = {
        "role": "research_only_qlib_lightgbm_baseline",
        "framework": "Microsoft Qlib",
        "qlib_version": str(qlib.__version__),
        "lightgbm_version": str(lightgbm.__version__),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "test_used_for_selection": False,
        "live_orders_allowed": False,
        "selection_rule": "fixed hyperparameters; early stopping on validation only",
        "label_processor": "DropnaLabel then cross-sectional rank normalization",
        "bundle_contract_sha256": contract["bundle_contract_sha256"],
        "checkpoint_sha256": contract["checkpoint_sha256"],
        "train_data_manifest_sha256": contract.get("train_data_manifest_sha256"),
        "train_edge_manifest_sha256": contract.get("train_edge_manifest_sha256"),
        "segments": segments,
        "stocks": int(contract["stocks"]),
        "features": int(contract["feature_count"]),
        "uses_graph_neighbor_state": bool(contract["uses_graph_neighbor_state"]),
        "seed": int(args.seed),
        "horizons": {},
    }
    all_daily_rows: list[dict[str, Any]] = []

    try:
        for horizon in horizons:
            print(f"Qlib LightGBM horizon={horizon}", flush=True)
            raw_label = labels[horizon_positions[horizon]].reshape(-1).astype(
                np.float32, copy=False
            )
            raw_label = raw_label.copy()
            raw_label[~current_available] = np.nan
            panel = feature_frame.copy(deep=False)
            panel[("label", "LABEL0")] = raw_label
            panel.columns = pd.MultiIndex.from_tuples(panel.columns)
            handler = DataHandlerLP(
                data_loader=StaticDataLoader(panel),
                start_time=str(pd.Timestamp(dates[0]).date()),
                end_time=str(pd.Timestamp(dates[-1]).date()),
                infer_processors=[],
                learn_processors=[DropnaLabel(), CSRankNorm(fields_group="label")],
            )
            dataset = DatasetH(handler=handler, segments=segments)
            model = LGBModel(
                loss="mse",
                num_boost_round=int(args.num_boost_round),
                early_stopping_rounds=int(args.early_stopping_rounds),
                colsample_bytree=0.8879,
                learning_rate=0.0421,
                subsample=0.8789,
                lambda_l1=205.6999,
                lambda_l2=580.9768,
                max_depth=8,
                num_leaves=210,
                num_threads=max(1, int(args.num_threads)),
                verbosity=-1,
                seed=int(args.seed),
                feature_fraction_seed=int(args.seed),
                bagging_seed=int(args.seed),
                data_random_seed=int(args.seed),
                deterministic=True,
                force_col_wise=True,
            )
            evals_result: dict[str, Any] = {}
            with R.start(
                experiment_name="stock_v2_qlib_lgb",
                recorder_name=f"horizon_{horizon}_seed_{args.seed}",
            ):
                model.fit(dataset, verbose_eval=25, evals_result=evals_result)
            model_path = output_dir / f"lightgbm_h{horizon}.txt"
            model.model.save_model(str(model_path))

            horizon_summary: dict[str, Any] = {
                "best_iteration": int(model.model.best_iteration),
                "model_sha256": sha256_file(model_path),
                "validation_used_for_early_stopping": True,
                "test_used_for_selection": False,
                "splits": {},
            }
            for split in ("valid", "test"):
                prediction = model.predict(dataset, segment=split).rename("prediction")
                evaluation = meta_frame.join(prediction, how="inner")
                evaluation["label"] = raw_label[
                    index.get_indexer(evaluation.index)
                ]
                evaluation = evaluation[
                    ["prediction", "label", "liquidity", "current_available"]
                ]
                daily_rows, split_summary = evaluate_signal_frame(
                    evaluation,
                    horizon=int(horizon),
                    liquidity_top_k=int(args.liquidity_top_k),
                )
                for row in daily_rows:
                    row["split"] = split
                all_daily_rows.extend(daily_rows)
                horizon_summary["splits"][split] = split_summary
                evaluation.to_parquet(
                    output_dir / f"predictions_h{horizon}_{split}.parquet",
                    compression="zstd",
                )
            importance = pd.DataFrame(
                {
                    "feature": feature_names,
                    "gain": model.model.feature_importance(importance_type="gain"),
                    "split": model.model.feature_importance(importance_type="split"),
                }
            ).sort_values(["gain", "split"], ascending=False)
            importance.to_csv(
                output_dir / f"feature_importance_h{horizon}.csv", index=False
            )
            horizon_summary["top_features_by_gain"] = importance.head(30).to_dict(
                orient="records"
            )
            horizon_summary["evals_result"] = evals_result
            summary["horizons"][str(horizon)] = horizon_summary
            del dataset, handler, panel, model, raw_label
            gc.collect()
    finally:
        os.chdir(original_cwd)

    pd.DataFrame(all_daily_rows).to_csv(output_dir / "daily_metrics.csv", index=False)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(f"Qlib LightGBM baseline complete: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
