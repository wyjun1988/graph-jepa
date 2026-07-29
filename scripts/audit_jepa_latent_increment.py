from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_qlib_lgb import newey_west_mean


METRICS = (
    "return_path_ic_top300",
    "return_path_rank_ic_top300",
    "return_path_decile_spread_top300",
)
COMPARISONS = ("raw", "shuffled")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_json(path: Path, role: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe {role}: {path}")
    return payload


def load_daily(path: Path, *, horizon: int, variant: str | None = None) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"horizon", "date", "split", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily metrics schema mismatch at {path}: {sorted(missing)}")
    selected = frame.loc[
        (frame["horizon"].astype(int) == int(horizon))
        & (frame["split"].astype(str) == "test")
    ].copy()
    if variant is not None:
        if "variant" not in selected:
            raise ValueError(f"variant column missing at {path}")
        selected = selected.loc[selected["variant"].astype(str) == variant].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected.sort_values("date").reset_index(drop=True)
    if selected.empty or selected["date"].duplicated().any():
        raise ValueError(f"daily metrics are empty or duplicated at {path}")
    if not np.isfinite(selected[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"daily metrics contain non-finite values at {path}")
    return selected[["date", *METRICS]]


def paired_fold_frame(
    raw: pd.DataFrame,
    actual: pd.DataFrame,
    shuffled: pd.DataFrame,
) -> pd.DataFrame:
    dates = raw["date"].to_numpy()
    if not np.array_equal(dates, actual["date"].to_numpy()) or not np.array_equal(
        dates, shuffled["date"].to_numpy()
    ):
        raise ValueError("raw, latent, and shuffled-latent dates do not align")
    result = pd.DataFrame({"date": raw["date"]})
    for metric in METRICS:
        result[f"raw_{metric}"] = raw[metric].to_numpy(dtype=np.float64)
        result[f"latent_{metric}"] = actual[metric].to_numpy(dtype=np.float64)
        result[f"shuffled_{metric}"] = shuffled[metric].to_numpy(dtype=np.float64)
        result[f"delta_raw_{metric}"] = (
            result[f"latent_{metric}"] - result[f"raw_{metric}"]
        )
        result[f"delta_shuffled_{metric}"] = (
            result[f"latent_{metric}"] - result[f"shuffled_{metric}"]
        )
    return result


def circular_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length < 1 or block_length < 1:
        raise ValueError("bootstrap length and block length must be positive")
    blocks = int(math.ceil(length / float(block_length)))
    starts = rng.integers(0, length, size=blocks)
    return np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )[:length]


def block_bootstrap_mean(
    folds: Mapping[str, np.ndarray],
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    if samples < 100:
        raise ValueError("at least 100 bootstrap samples are required")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    arrays = {name: np.asarray(values, dtype=np.float64) for name, values in folds.items()}
    if not arrays or any(len(values) == 0 for values in arrays.values()):
        raise ValueError("bootstrap folds cannot be empty")
    if any(not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("bootstrap values must be finite")
    for draw in range(int(samples)):
        sampled = [
            values[circular_indices(len(values), int(block_length), rng)]
            for values in arrays.values()
        ]
        draws[draw] = float(np.concatenate(sampled).mean())
    return {
        "samples": int(samples),
        "block_length": int(block_length),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def summarize_fold(frame: pd.DataFrame, horizon: int) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        metrics[metric] = {
            "raw_mean": float(frame[f"raw_{metric}"].mean()),
            "latent_mean": float(frame[f"latent_{metric}"].mean()),
            "shuffled_mean": float(frame[f"shuffled_{metric}"].mean()),
            "latent_minus_raw": newey_west_mean(
                frame[f"delta_raw_{metric}"].to_numpy(dtype=np.float64),
                lag=int(horizon),
            ),
            "latent_minus_shuffled": newey_west_mean(
                frame[f"delta_shuffled_{metric}"].to_numpy(dtype=np.float64),
                lag=int(horizon),
            ),
        }
    return {"rows": int(len(frame)), "metrics": metrics}


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "jepa_latent_increment_confirmation_contract":
        raise ValueError("invalid latent increment contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe latent increment contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    horizon = int(contract["horizon"])
    confirmation_folds = set(contract["confirmation_folds"])
    fold_frames: dict[str, pd.DataFrame] = {}
    fold_results: dict[str, Any] = {}
    inputs = {"contract": sha256_file(contract_path)}
    for fold, spec in contract["folds"].items():
        raw_summary_path = Path(spec["raw_summary"])
        raw_daily_path = Path(spec["raw_daily"])
        bundle_contract_path = Path(spec["bundle_dir"]) / "bundle_contract.json"
        latent_dir = Path(spec["latent_dir"])
        augmented_dir = Path(spec["augmented_dir"])
        augmented_summary_path = augmented_dir / "summary.json"
        metadata_path = latent_dir / "metadata.json"
        expected_hashes = spec["expected_hashes"]
        pinned = {
            raw_summary_path: expected_hashes["raw_summary_sha256"],
            raw_daily_path: expected_hashes["raw_daily_sha256"],
            bundle_contract_path: expected_hashes["bundle_contract_file_sha256"],
        }
        for path, expected in pinned.items():
            if not path.is_file() or sha256_file(path) != str(expected):
                raise ValueError(f"pinned fold input mismatch: {path}")

        raw_summary = safe_json(raw_summary_path, "raw Qlib summary")
        augmented = safe_json(augmented_summary_path, "augmented Qlib summary")
        metadata = safe_json(metadata_path, "fit-only latent metadata")
        checkpoint = str(expected_hashes["checkpoint_sha256"])
        if {
            str(raw_summary.get("checkpoint_sha256")),
            str(augmented.get("checkpoint_sha256")),
            str(metadata.get("checkpoint_sha256")),
        } != {checkpoint}:
            raise ValueError(f"checkpoint mismatch for {fold}")
        if str(augmented.get("bundle_contract_sha256")) != str(
            expected_hashes["bundle_contract_content_sha256"]
        ):
            raise ValueError(f"bundle hash mismatch for {fold}")
        if int(augmented.get("horizon", -1)) != horizon or int(
            metadata.get("horizon", -1)
        ) != horizon:
            raise ValueError(f"horizon mismatch for {fold}")
        if int(metadata.get("feature_count", -1)) != int(contract["latent_features"]):
            raise ValueError(f"latent feature count mismatch for {fold}")
        if str(metadata.get("test_end")) != str(spec["test_end"]):
            raise ValueError(f"test end mismatch for {fold}")
        if sha256_file(metadata_path) != str(augmented["latent_metadata_sha256"]):
            raise ValueError(f"latent metadata hash mismatch for {fold}")

        raw = load_daily(raw_daily_path, horizon=horizon)
        actual_path = augmented_dir / "daily_metrics_raw_latent.csv"
        shuffled_path = augmented_dir / "daily_metrics_raw_shuffled_latent.csv"
        actual = load_daily(actual_path, horizon=horizon, variant="raw_latent")
        shuffled = load_daily(
            shuffled_path, horizon=horizon, variant="raw_shuffled_latent"
        )
        paired = paired_fold_frame(raw, actual, shuffled)
        if len(paired) != int(spec["expected_test_dates"]):
            raise ValueError(f"unexpected test row count for {fold}")
        paired.insert(0, "fold", fold)
        paired.insert(1, "evaluation_role", str(spec["evaluation_role"]))
        fold_frames[fold] = paired
        fold_results[fold] = {
            "evaluation_role": spec["evaluation_role"],
            "test_start": str(paired["date"].min().date()),
            "test_end": str(paired["date"].max().date()),
            **summarize_fold(paired, horizon),
        }
        for name, path in {
            "raw_summary": raw_summary_path,
            "raw_daily": raw_daily_path,
            "bundle_contract": bundle_contract_path,
            "latent_metadata": metadata_path,
            "augmented_summary": augmented_summary_path,
            "augmented_actual_daily": actual_path,
            "augmented_shuffled_daily": shuffled_path,
        }.items():
            inputs[f"{fold}.{name}"] = sha256_file(path)

    if confirmation_folds != {
        fold for fold, spec in contract["folds"].items() if spec["evaluation_role"] == "confirmation"
    }:
        raise ValueError("confirmation fold roles do not match the contract")
    confirmation = {fold: fold_frames[fold] for fold in sorted(confirmation_folds)}
    bootstrap = contract["bootstrap"]
    pooled: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        pooled[metric] = {}
        for comparison_index, comparison in enumerate(COMPARISONS):
            column = f"delta_{comparison}_{metric}"
            values = {
                fold: frame[column].to_numpy(dtype=np.float64)
                for fold, frame in confirmation.items()
            }
            concatenated = np.concatenate(list(values.values()))
            pooled[metric][f"latent_minus_{comparison}"] = {
                "rows": int(len(concatenated)),
                "mean": float(concatenated.mean()),
                "positive_fold_count": int(
                    sum(float(value.mean()) > 0.0 for value in values.values())
                ),
                "fold_count": int(len(values)),
                "block_bootstrap": block_bootstrap_mean(
                    values,
                    samples=int(bootstrap["samples"]),
                    block_length=int(bootstrap["block_length"]),
                    seed=int(bootstrap["seed"]) + metric_index * 10 + comparison_index,
                ),
            }

    gates = contract["gates"]
    primary = str(gates["primary_metric"])
    raw_primary = pooled[primary]["latent_minus_raw"]
    shuffled_primary = pooled[primary]["latent_minus_shuffled"]
    max_harm = float(gates["maximum_single_fold_primary_harm"])
    checks = {
        "primary_positive_folds_vs_raw": int(raw_primary["positive_fold_count"])
        >= int(gates["minimum_positive_confirmation_folds"]),
        "primary_positive_folds_vs_shuffled": int(
            shuffled_primary["positive_fold_count"]
        )
        >= int(gates["minimum_positive_confirmation_folds"]),
        "primary_pooled_mean_vs_raw_positive": float(raw_primary["mean"])
        > float(gates["minimum_pooled_mean_delta"]),
        "primary_pooled_mean_vs_shuffled_positive": float(shuffled_primary["mean"])
        > float(gates["minimum_pooled_mean_delta"]),
        "primary_bootstrap_lower_vs_raw_positive": float(
            raw_primary["block_bootstrap"]["lower_95"]
        )
        > float(gates["minimum_bootstrap_lower_95"]),
        "primary_bootstrap_lower_vs_shuffled_positive": float(
            shuffled_primary["block_bootstrap"]["lower_95"]
        )
        > float(gates["minimum_bootstrap_lower_95"]),
        "no_single_fold_primary_harm_vs_raw": all(
            float(frame[f"delta_raw_{primary}"].mean()) >= -max_harm
            for frame in confirmation.values()
        ),
        "rank_ic_pooled_positive_vs_both": all(
            float(pooled["return_path_rank_ic_top300"][f"latent_minus_{comparison}"]["mean"])
            > 0.0
            for comparison in COMPARISONS
        ),
        "decile_spread_pooled_positive_vs_both": all(
            float(
                pooled["return_path_decile_spread_top300"][
                    f"latent_minus_{comparison}"
                ]["mean"]
            )
            > 0.0
            for comparison in COMPARISONS
        ),
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "role": "jepa_latent_increment_confirmation_audit",
        "live_orders_allowed": False,
        "test_used_for_selection": False,
        "promotion_eligible": False,
        "contract_sha256": sha256_file(contract_path),
        "confirmation_folds": sorted(confirmation_folds),
        "disclosed_folds": list(contract["disclosed_folds"]),
        "horizon": horizon,
        "latent_features": int(contract["latent_features"]),
        "folds": fold_results,
        "confirmation_pooled": pooled,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "decision": (
            "confirm_incremental_jepa_return_information"
            if passed
            else "no_confirmed_incremental_jepa_return_information"
        ),
        "next_gate": (
            "fresh_forward_shadow_only" if passed else "do_not_use_jepa_latent_for_return_ranking"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    daily = pd.concat(fold_frames.values(), ignore_index=True)
    return summary, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit fit-only JEPA latent increment over raw Qlib features."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_deltas.csv", index=False)
    summary["daily_deltas_sha256"] = sha256_file(output_dir / "daily_deltas.csv")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "checks_passed": summary["checks_passed"],
                "checks_total": summary["checks_total"],
                "promotion_eligible": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
