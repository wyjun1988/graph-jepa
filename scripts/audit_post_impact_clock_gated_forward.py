from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METRICS = ("pearson", "skill_vs_zero_mse")
HIGH_IMPACT_MINIMUM_KEY = (
    "adaptive_surprise_and_realized_impact_timestamps_per_primary_horizon"
)
MODEL_CONTRACTS = {
    "direct": ("direct", "none"),
    "state": ("state", "none"),
    "latent": ("latent", "none"),
    "latent_only_placebo": ("latent", "latent_only"),
}
PRIMARY_COMPARISONS = {
    "latent_vs_direct_late_session": ("latent", "direct"),
    "latent_vs_latent_only_placebo_late_session": (
        "latent",
        "latent_only_placebo",
    ),
}
DIAGNOSTIC_COMPARISONS = {
    "state_vs_direct_late_session": ("state", "direct"),
    "latent_vs_state_late_session": ("latent", "state"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def circular_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    if length < 1 or block_length < 1:
        raise ValueError("bootstrap lengths must be positive")
    block_count = int(math.ceil(length / float(block_length)))
    starts = rng.integers(0, length, size=block_count)
    return np.concatenate(
        [(start + np.arange(block_length)) % length for start in starts]
    )[:length]


def bootstrap_mean(
    values: np.ndarray,
    *,
    samples: int,
    block_length: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must be a finite vector")
    if len(values) < 2:
        return {
            "status": "insufficient_sessions",
            "sessions": int(len(values)),
            "samples": 0,
            "block_length": int(block_length),
            "lower_95": None,
            "median": None,
            "upper_95": None,
        }
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(samples), dtype=np.float64)
    effective_block = min(int(block_length), len(values))
    for draw in range(int(samples)):
        indices = circular_indices(len(values), effective_block, rng)
        draws[draw] = float(values[indices].mean())
    return {
        "status": "complete",
        "sessions": int(len(values)),
        "samples": int(samples),
        "block_length": int(effective_block),
        "lower_95": float(np.quantile(draws, 0.025)),
        "median": float(np.quantile(draws, 0.5)),
        "upper_95": float(np.quantile(draws, 0.975)),
    }


def metric_frame(
    payload: Mapping[str, Any],
    *,
    first_date: str,
    cells: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    source = payload["test"]["clock_bucket_daily_node_endpoint_rows"]
    rows: list[dict[str, Any]] = []
    for horizon, bucket in cells:
        for record in source[horizon][bucket]:
            if str(record["date"]) < first_date:
                continue
            rows.append(
                {
                    "date": str(record["date"]),
                    "horizon": horizon,
                    "bucket": bucket,
                    "count": int(record["count"]),
                    **{
                        metric: float(record[metric])
                        for metric in METRICS
                    },
                }
            )
    frame = pd.DataFrame(
        rows, columns=["date", "horizon", "bucket", "count", *METRICS]
    )
    if frame.empty:
        return frame
    keys = ["date", "horizon", "bucket"]
    if frame.duplicated(keys).any():
        raise ValueError("forward metric rows contain duplicate primary cells")
    if (frame["count"] <= 0).any():
        raise ValueError("forward metric rows contain non-positive target counts")
    if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=np.float64)).all():
        raise ValueError("forward metric rows contain non-finite metrics")
    return frame.sort_values(keys).reset_index(drop=True)


def event_frame(
    payload: Mapping[str, Any],
    *,
    first_date: str,
    horizons: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in payload["test"].get("daily_event_rows", []):
        date = str(record["date"])
        if date < first_date:
            continue
        row: dict[str, Any] = {
            "date": date,
            "observed_surprise": int(
                record["adaptive_observed_surprise"]["positive_timestamps"]
            ),
        }
        for horizon in horizons:
            row[f"high_impact_{horizon}"] = int(
                record["adaptive_surprise_and_impact"][horizon][
                    "positive_timestamps"
                ]
            )
        rows.append(row)
    columns = ["date", "observed_surprise", *[f"high_impact_{h}" for h in horizons]]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    if frame["date"].duplicated().any():
        raise ValueError("forward event rows contain duplicate dates")
    if (frame.drop(columns="date") < 0).any().any():
        raise ValueError("forward event counts cannot be negative")
    return frame.sort_values("date").reset_index(drop=True)


def paired_metric_deltas(
    actual: pd.DataFrame,
    comparator: pd.DataFrame,
    *,
    comparison: str,
) -> pd.DataFrame:
    keys = ["date", "horizon", "bucket"]
    left = actual.rename(
        columns={
            "count": "actual_count",
            **{metric: f"actual_{metric}" for metric in METRICS},
        }
    )
    right = comparator.rename(
        columns={
            "count": "comparator_count",
            **{metric: f"comparator_{metric}" for metric in METRICS},
        }
    )
    joined = left.merge(
        right, on=keys, how="outer", validate="one_to_one", indicator=True
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError(f"forward primary cells do not align for {comparison}")
    joined = joined.drop(columns="_merge")
    if not (joined["actual_count"] == joined["comparator_count"]).all():
        raise ValueError(f"forward target counts do not align for {comparison}")
    joined["comparison"] = comparison
    for metric in METRICS:
        joined[f"delta_{metric}"] = (
            joined[f"actual_{metric}"] - joined[f"comparator_{metric}"]
        )
    return joined.sort_values(keys).reset_index(drop=True)


def summarize_comparison(
    paired: pd.DataFrame,
    *,
    cells: tuple[tuple[str, str], ...],
    samples: int,
    block_length: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    expected_cells = len(cells)
    per_date_counts = paired.groupby("date", sort=True).size()
    if not (per_date_counts == expected_cells).all():
        raise ValueError(
            "a forward session is missing one or more frozen primary cells"
        )
    session = (
        paired.groupby("date", sort=True)[[f"delta_{metric}" for metric in METRICS]]
        .mean()
        .reset_index()
    )
    cell_summary: dict[str, Any] = {}
    for horizon, bucket in cells:
        cell_summary.setdefault(horizon, {})
        selected = paired[
            (paired["horizon"] == horizon) & (paired["bucket"] == bucket)
        ]
        cell_summary[horizon][bucket] = {
            "sessions": int(len(selected)),
            **{
                f"mean_delta_{metric}": float(
                    selected[f"delta_{metric}"].mean()
                )
                for metric in METRICS
            },
        }
    metric_summary: dict[str, Any] = {}
    for metric_index, metric in enumerate(METRICS):
        values = session[f"delta_{metric}"].to_numpy(dtype=np.float64)
        metric_summary[metric] = {
            "mean_session_delta": float(values.mean()),
            "positive_session_fraction": float(np.mean(values > 0.0)),
            "block_bootstrap": bootstrap_mean(
                values,
                samples=samples,
                block_length=block_length,
                seed=seed + metric_index,
            ),
        }
    return {
        "sessions": int(len(session)),
        "primary_cells_per_session": int(expected_cells),
        "metrics": metric_summary,
        "cells": cell_summary,
    }, session


def _validate_report(
    name: str,
    payload: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> None:
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe model report: {name}")
    if (
        payload.get("adaptive_event_contract")
        != "causal_same_clock_rolling_upper_tail_v1"
    ):
        raise ValueError(f"adaptive-event contract mismatch: {name}")
    expected_variant, expected_placebo = MODEL_CONTRACTS[name]
    if payload.get("variant") != expected_variant:
        raise ValueError(f"model variant mismatch: {name}")
    if payload.get("daily_context_placebo_mode") != expected_placebo:
        raise ValueError(f"daily-context placebo mismatch: {name}")
    if payload["inputs"].get("checkpoint_sha256") != model_spec["checkpoint_sha256"]:
        raise ValueError(f"checkpoint hash mismatch: {name}")


def evaluate(
    contract_path: Path,
    report_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    contract = load_json(contract_path)
    if contract.get("role") != "post_impact_clock_gated_forward_shadow_contract":
        raise ValueError("invalid clock-gated forward contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe clock-gated forward contract")
    if contract.get("broker_order_calls_allowed") is not False:
        raise ValueError("forward contract permits broker order calls")

    payloads: dict[str, dict[str, Any]] = {}
    report_hashes: dict[str, str] = {}
    data_pins: set[tuple[str, str]] = set()
    split_pins: set[str] = set()
    for name, expected in MODEL_CONTRACTS.items():
        if name not in contract["models"]:
            raise ValueError(f"contract is missing model: {name}")
        model_spec = contract["models"][name]
        checkpoint = Path(model_spec["checkpoint"])
        summary = Path(model_spec["summary"])
        if sha256_file(checkpoint) != model_spec["checkpoint_sha256"]:
            raise ValueError(f"frozen checkpoint changed: {name}")
        if sha256_file(summary) != model_spec["summary_sha256"]:
            raise ValueError(f"frozen model summary changed: {name}")
        report_path = report_dir / f"{name}.json"
        payload = load_json(report_path)
        _validate_report(name, payload, model_spec)
        if payload["splits"]["test"]["end"] < contract["first_forward_session"]:
            raise ValueError(
                f"report does not include the first forward session: {name}"
            )
        data_pins.add(
            (
                payload["inputs"]["day_release_manifest_sha256"],
                payload["inputs"]["stale_cache_manifest_sha256"],
            )
        )
        split_pins.add(json.dumps(payload["splits"], sort_keys=True))
        payloads[name] = payload
        report_hashes[name] = sha256_file(report_path)
    if len(data_pins) != 1 or len(split_pins) != 1:
        raise ValueError("forward reports do not share identical data and splits")

    endpoint = contract["primary_endpoint"]
    horizons = tuple(str(value) for value in endpoint["horizons"])
    if "cells" in endpoint:
        cells = tuple(
            (str(record["horizon"]), str(record["bucket"]))
            for record in endpoint["cells"]
        )
    else:
        buckets = tuple(str(value) for value in endpoint["clock_buckets"])
        cells = tuple((horizon, bucket) for horizon in horizons for bucket in buckets)
    if not cells or len(cells) != len(set(cells)):
        raise ValueError("primary endpoint cells must be non-empty and unique")
    if int(endpoint["cell_count"]) != len(cells):
        raise ValueError("primary endpoint cell count does not match the contract")
    if any(horizon not in horizons for horizon, _bucket in cells):
        raise ValueError("primary endpoint cell uses an undeclared horizon")
    first_date = str(contract["first_forward_session"])
    frames = {
        name: metric_frame(
            payload,
            first_date=first_date,
            cells=cells,
        )
        for name, payload in payloads.items()
    }
    if any(frame.empty for frame in frames.values()):
        missing = [name for name, frame in frames.items() if frame.empty]
        raise ValueError(f"forward reports contain no primary rows: {missing}")

    events = {
        name: event_frame(
            payload,
            first_date=first_date,
            horizons=horizons,
        )
        for name, payload in payloads.items()
    }
    direct_events = events["direct"]
    if direct_events.empty:
        raise ValueError("forward reports do not contain daily event counts")
    for name, frame in events.items():
        if not frame.equals(direct_events):
            raise ValueError(f"forward event calendars differ: {name}")

    bootstrap = endpoint["bootstrap"]
    comparisons = {**PRIMARY_COMPARISONS, **DIAGNOSTIC_COMPARISONS}
    summaries: dict[str, Any] = {}
    paired_outputs: list[pd.DataFrame] = []
    session_outputs: list[pd.DataFrame] = []
    for index, (comparison, (actual_name, comparator_name)) in enumerate(
        comparisons.items()
    ):
        paired = paired_metric_deltas(
            frames[actual_name],
            frames[comparator_name],
            comparison=comparison,
        )
        comparison_summary, session = summarize_comparison(
            paired,
            cells=cells,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["moving_block_length_sessions"]),
            seed=int(bootstrap["seed"]) + index * 10,
        )
        session.insert(1, "comparison", comparison)
        summaries[comparison] = comparison_summary
        paired_outputs.append(paired)
        session_outputs.append(session)

    dates = sorted(direct_events["date"].tolist())
    minimum = contract["minimum_evidence"]
    high_impact_counts = {
        horizon: int(direct_events[f"high_impact_{horizon}"].sum())
        for horizon in horizons
    }
    evidence_checks = {
        "minimum_forward_sessions": len(dates)
        >= int(minimum["completed_forward_sessions"]),
        **{
            f"minimum_high_impact_{horizon}": count
            >= int(minimum[HIGH_IMPACT_MINIMUM_KEY])
            for horizon, count in high_impact_counts.items()
        },
        "offline_orders_sent_zero": True,
    }
    gates = contract["promotion_gates"]
    statistical_checks: dict[str, bool] = {}
    for comparison in PRIMARY_COMPARISONS:
        prefix = (
            "latent_vs_direct"
            if comparison == "latent_vs_direct_late_session"
            else "latent_vs_placebo"
        )
        result = summaries[comparison]
        pearson = result["metrics"]["pearson"]
        skill = result["metrics"]["skill_vs_zero_mse"]
        positive_cells = sum(
            int(result["cells"][horizon][bucket]["mean_delta_pearson"] > 0.0)
            for horizon, bucket in cells
        )
        statistical_checks[f"{comparison}.pearson_mean"] = (
            pearson["mean_session_delta"]
            >= float(gates[f"{prefix}_mean_pearson_delta_minimum"])
        )
        statistical_checks[f"{comparison}.skill_mean"] = (
            skill["mean_session_delta"]
            >= float(gates[f"{prefix}_mean_skill_delta_minimum"])
        )
        statistical_checks[f"{comparison}.positive_cells"] = positive_cells >= int(
            gates["minimum_positive_primary_cells_for_each_primary_comparison"]
        )
        if pearson["block_bootstrap"]["status"] == "complete":
            statistical_checks[f"{comparison}.pearson_lower_95"] = (
                pearson["block_bootstrap"]["lower_95"]
                >= float(gates[f"{prefix}_pearson_bootstrap_lower_95_minimum"])
            )
            statistical_checks[f"{comparison}.skill_lower_95"] = (
                skill["block_bootstrap"]["lower_95"]
                >= float(gates[f"{prefix}_skill_bootstrap_lower_95_minimum"])
            )
        else:
            statistical_checks[f"{comparison}.pearson_lower_95"] = False
            statistical_checks[f"{comparison}.skill_lower_95"] = False

    evidence_complete = all(evidence_checks.values())
    statistical_pass = all(statistical_checks.values())
    if not evidence_complete:
        decision = "insufficient_forward_evidence_accumulating"
    elif statistical_pass:
        decision = "eligible_for_longer_read_only_shadow_only"
    else:
        decision = "clock_gated_latent_not_confirmed"
    day_manifest_sha, stale_manifest_sha = next(iter(data_pins))
    summary = {
        "schema_version": 1,
        "role": "post_impact_clock_gated_forward_audit",
        "contract_sha256": sha256_file(contract_path),
        "forward_period": {
            "start": dates[0],
            "end": dates[-1],
            "sessions": len(dates),
        },
        "policy": contract["policy"],
        "primary_endpoint": endpoint,
        "comparisons": summaries,
        "event_evidence": {
            "observed_surprise_timestamps": int(
                direct_events["observed_surprise"].sum()
            ),
            "high_impact_timestamps_by_horizon": high_impact_counts,
        },
        "evidence_checks": evidence_checks,
        "statistical_checks": statistical_checks,
        "evidence_complete": evidence_complete,
        "statistical_pass": statistical_pass if evidence_complete else None,
        "decision": decision,
        "shadow_stage_eligible": bool(evidence_complete and statistical_pass),
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
        "inputs": {
            "reports": report_hashes,
            "day_release_manifest_sha256": day_manifest_sha,
            "stale_cache_manifest_sha256": stale_manifest_sha,
        },
        "status": "complete",
    }
    return (
        summary,
        pd.concat(paired_outputs, ignore_index=True),
        pd.concat(session_outputs, ignore_index=True),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit the frozen clock-gated JEPA forward-shadow candidate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, paired, sessions = evaluate(
        Path(args.contract), Path(args.report_dir)
    )
    output_dir.mkdir(parents=True)
    paired.to_csv(output_dir / "paired_primary_cells.csv", index=False)
    sessions.to_csv(output_dir / "session_primary_deltas.csv", index=False)
    summary["artifacts"] = {
        "paired_primary_cells_sha256": sha256_file(
            output_dir / "paired_primary_cells.csv"
        ),
        "session_primary_deltas_sha256": sha256_file(
            output_dir / "session_primary_deltas.csv"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "sessions": summary["forward_period"]["sessions"],
                "shadow_stage_eligible": summary["shadow_stage_eligible"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
