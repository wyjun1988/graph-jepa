from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


HORIZONS = ("5m", "15m", "30m", "60m", "close")
BUCKETS = (
    "open_0900_0929",
    "morning_0930_1059",
    "midday_1100_1329",
    "afternoon_1330_1459",
    "close_1500_1530",
)
METRICS = ("pearson", "skill_vs_zero_mse")


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


def circular_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length < 1 or block_length < 1:
        raise ValueError("bootstrap lengths must be positive")
    block_count = int(math.ceil(length / float(block_length)))
    starts = rng.integers(0, length, size=block_count)
    return np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )[:length]


def block_bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap values must be finite and nontrivial")
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    for draw in range(int(samples)):
        indices = circular_indices(len(values), int(block_length), rng)
        draws[draw] = float(values[indices].mean())
    return {
        "samples": int(samples),
        "block_length": int(block_length),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def daily_frame(payload: Mapping[str, Any], horizon: str, bucket: str) -> pd.DataFrame:
    rows = payload["test"]["clock_bucket_daily_node_endpoint_rows"][horizon][bucket]
    if not rows:
        return pd.DataFrame(columns=["date", "count", *METRICS])
    frame = pd.DataFrame(rows)
    required = {"date", "count", *METRICS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"clock-bucket rows are missing fields: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("clock-bucket rows are empty or duplicated")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("clock-bucket metrics contain non-finite values")
    return frame[["date", "count", *METRICS]]


def paired_delta(
    actual: pd.DataFrame, comparator: pd.DataFrame, metric: str
) -> pd.DataFrame:
    left = actual[["date", metric]].rename(columns={metric: "actual"})
    right = comparator[["date", metric]].rename(columns={metric: "comparator"})
    joined = left.merge(right, on="date", validate="one_to_one")
    if len(joined) != len(left) or len(joined) != len(right):
        raise ValueError("actual and comparator dates do not align")
    joined["delta"] = joined["actual"] - joined["comparator"]
    return joined


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "post_impact_clock_bucket_localization_contract":
        raise ValueError("invalid post-impact clock-bucket contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe post-impact clock-bucket contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    payloads: dict[str, dict[str, Any]] = {}
    inputs = {"contract": sha256_file(contract_path)}
    manifests = set()
    splits = set()
    for name, spec in contract["models"].items():
        path = Path(spec["clock_report"])
        payload = safe_json(path, f"{name} clock report")
        if payload.get("adaptive_event_contract") != "causal_same_clock_rolling_upper_tail_v1":
            raise ValueError(f"adaptive event contract mismatch: {name}")
        parity = payload.get("reference_inference_parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ValueError(f"reference inference parity failed: {name}")
        if str(payload.get("variant")) != str(spec["variant"]):
            raise ValueError(f"variant mismatch: {name}")
        if bool(payload.get("shuffle_daily_context")) != bool(
            spec["shuffle_daily_context"]
        ):
            raise ValueError(f"context shuffle mismatch: {name}")
        if str(payload["inputs"]["checkpoint_sha256"]) != str(
            spec["checkpoint_sha256"]
        ):
            raise ValueError(f"checkpoint mismatch: {name}")
        if str(payload["inputs"]["reference_summary_sha256"]) != str(
            spec["reference_summary_sha256"]
        ):
            raise ValueError(f"reference summary mismatch: {name}")
        manifests.add(
            (
                payload["inputs"]["day_release_manifest_sha256"],
                payload["inputs"]["stale_cache_manifest_sha256"],
            )
        )
        splits.add(json.dumps(payload["splits"], sort_keys=True))
        payloads[name] = payload
        inputs[name] = sha256_file(path)
    if len(manifests) != 1 or len(splits) != 1:
        raise ValueError("clock reports do not share identical data and splits")

    actual_name = str(contract["actual_model"])
    comparator_names = [str(value) for value in contract["comparators"]]
    actual_payload = payloads[actual_name]
    bootstrap = contract["bootstrap"]
    cells: dict[str, Any] = {}
    daily_rows = []
    for horizon_index, horizon in enumerate(HORIZONS):
        cells[horizon] = {}
        for bucket_index, bucket in enumerate(BUCKETS):
            actual = daily_frame(actual_payload, horizon, bucket)
            if actual.empty:
                if any(
                    not daily_frame(payloads[name], horizon, bucket).empty
                    for name in comparator_names
                ):
                    raise ValueError(
                        f"clock-bucket availability differs for {horizon}/{bucket}"
                    )
                cells[horizon][bucket] = {
                    "available": False,
                    "reason": "no causally available endpoint labels in this clock bucket",
                }
                continue
            cell = {
                "available": True,
                "days": int(len(actual)),
                "actual": {
                    metric: float(actual[metric].mean()) for metric in METRICS
                },
                "comparators": {},
            }
            for comparator_index, comparator_name in enumerate(comparator_names):
                comparator = daily_frame(payloads[comparator_name], horizon, bucket)
                comparison = {}
                for metric_index, metric in enumerate(METRICS):
                    paired = paired_delta(actual, comparator, metric)
                    seed = (
                        int(bootstrap["seed"])
                        + horizon_index * 100
                        + bucket_index * 10
                        + comparator_index * 2
                        + metric_index
                    )
                    values = paired["delta"].to_numpy(dtype=np.float64)
                    comparison[metric] = {
                        "mean_delta": float(values.mean()),
                        "positive_day_fraction": float(np.mean(values > 0.0)),
                        "block_bootstrap": block_bootstrap_mean(
                            values,
                            samples=int(bootstrap["samples"]),
                            block_length=int(bootstrap["block_length"]),
                            seed=seed,
                        ),
                    }
                    for row in paired.itertuples(index=False):
                        daily_rows.append(
                            {
                                "horizon": horizon,
                                "bucket": bucket,
                                "comparator": comparator_name,
                                "metric": metric,
                                "date": row.date,
                                "actual": float(row.actual),
                                "comparator_value": float(row.comparator),
                                "delta": float(row.delta),
                            }
                        )
                cell["comparators"][comparator_name] = comparison
            cells[horizon][bucket] = cell

    gates = contract["gates"]
    checks = {}
    primary_horizon = str(gates["primary_horizon"])
    for bucket in gates["actionable_buckets"]:
        for comparator in comparator_names:
            metrics = cells[primary_horizon][bucket]["comparators"][comparator]
            prefix = f"{bucket}.{comparator}"
            checks[f"{prefix}.pearson_mean_positive"] = float(
                metrics["pearson"]["mean_delta"]
            ) > float(gates["minimum_mean_pearson_delta"])
            checks[f"{prefix}.pearson_bootstrap_lower_positive"] = float(
                metrics["pearson"]["block_bootstrap"]["lower_95"]
            ) > float(gates["minimum_pearson_bootstrap_lower_95"])
            checks[f"{prefix}.skill_mean_nonnegative"] = float(
                metrics["skill_vs_zero_mse"]["mean_delta"]
            ) >= float(gates["minimum_mean_skill_delta"])
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "role": "post_impact_clock_bucket_localization_audit",
        "retrospective_hypothesis_localization": True,
        "test_used_for_hypothesis_generation": True,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "contract_sha256": sha256_file(contract_path),
        "actual_model": actual_name,
        "comparators": comparator_names,
        "cells": cells,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "decision": (
            "eligible_as_forward_shadow_hypothesis_only"
            if passed
            else "no_actionable_early_session_latent_increment"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Localize post-impact JEPA latent increment by KRX clock bucket."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_paired_deltas.csv", index=False)
    summary["daily_paired_deltas_sha256"] = sha256_file(
        output_dir / "daily_paired_deltas.csv"
    )
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
        )
    )


if __name__ == "__main__":
    main()
