from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_clock_bucket_increment import (
    block_bootstrap_mean,
    daily_frame,
    paired_delta,
    safe_json,
    sha256_file,
)


HORIZONS = ("5m", "15m", "30m", "60m", "close")


def summary_daily(payload: dict[str, Any], horizon: str) -> pd.DataFrame:
    frame = pd.DataFrame(payload["test"]["daily_node_endpoint_rows"][horizon])
    required = {"date", "count", "pearson", "skill_vs_zero_mse"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"training summary daily schema mismatch: {sorted(missing)}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame.empty or frame["date"].duplicated().any():
        raise ValueError("training summary daily rows are empty or duplicated")
    if not np.isfinite(
        frame[["pearson", "skill_vs_zero_mse"]].to_numpy(dtype=np.float64)
    ).all():
        raise ValueError("training summary daily metrics must be finite")
    return frame


def paired_result(
    actual: pd.DataFrame,
    comparator: pd.DataFrame,
    metric: str,
    *,
    bootstrap: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    paired = paired_delta(actual, comparator, metric)
    values = paired["delta"].to_numpy(dtype=np.float64)
    return {
        "rows": int(len(values)),
        "mean_delta": float(values.mean()),
        "positive_day_fraction": float(np.mean(values > 0.0)),
        "block_bootstrap": block_bootstrap_mean(
            values,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["block_length"]),
            seed=int(bootstrap["seed"]) + int(seed_offset),
        ),
    }


def _assert_checkpoint_args(
    checkpoint_path: Path,
    expected: dict[str, Any],
    mode: str,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = checkpoint.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"checkpoint args are missing: {checkpoint_path}")
    for name, value in expected.items():
        actual = args.get(name)
        if isinstance(value, float):
            if not np.isclose(float(actual), value, rtol=0.0, atol=1e-12):
                raise ValueError(f"checkpoint arg mismatch: {name}")
        elif actual != value:
            raise ValueError(f"checkpoint arg mismatch: {name}")
    expected_flags = {
        "disabled": (True, False),
        "causal": (False, False),
        "node_permuted_placebo": (False, True),
    }
    disabled, permuted = expected_flags[mode]
    if bool(args.get("disable_stale_graph", False)) != disabled or bool(
        args.get("permute_stale_graph_nodes", False)
    ) != permuted:
        raise ValueError(f"checkpoint graph flags mismatch: {mode}")


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "post_impact_graph_sensor_ablation_contract":
        raise ValueError("invalid graph-sensor ablation contract role")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe graph-sensor contract")
    for relative, expected in contract["source_pins"].items():
        path = Path(relative)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"source pin mismatch: {relative}")

    training: dict[str, dict[str, Any]] = {}
    clock: dict[str, dict[str, Any]] = {}
    inputs = {"contract": sha256_file(contract_path)}
    split_records = set()
    for name, spec in contract["models"].items():
        root = Path(spec["training_dir"])
        summary_path = root / "summary.json"
        checkpoint_path = root / "post_impact_reforecast.pt"
        clock_path = Path(spec["clock_report"])
        summary = safe_json(summary_path, f"{name} training summary")
        clock_payload = safe_json(clock_path, f"{name} clock report")
        if summary.get("promotion_eligible") is not False:
            raise ValueError(f"unsafe promotion field: {name}")
        if summary.get("strict_out_of_sample_stale_jepa") is not True:
            raise ValueError(f"non-strict stale JEPA input: {name}")
        if summary.get("variant") != "latent":
            raise ValueError(f"unexpected model variant: {name}")
        if summary.get("stale_stock_graph_mode") != spec["graph_mode"]:
            raise ValueError(f"training graph mode mismatch: {name}")
        if summary["inputs"].get("day_release_manifest_sha256") != contract[
            "data_pins"
        ]["day_release_manifest_sha256"]:
            raise ValueError(f"day release mismatch: {name}")
        if summary["inputs"].get("stale_cache_manifest_sha256") != contract[
            "data_pins"
        ]["stale_graph_cache_manifest_sha256"]:
            raise ValueError(f"stale graph cache mismatch: {name}")
        if sha256_file(checkpoint_path) != str(summary["checkpoint_sha256"]):
            raise ValueError(f"checkpoint checksum mismatch: {name}")
        _assert_checkpoint_args(
            checkpoint_path,
            contract["training_args"],
            str(spec["graph_mode"]),
        )
        parity = clock_payload.get("reference_inference_parity")
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ValueError(f"clock inference parity failed: {name}")
        if clock_payload["inputs"].get("checkpoint_sha256") != summary[
            "checkpoint_sha256"
        ]:
            raise ValueError(f"clock checkpoint mismatch: {name}")
        if clock_payload["inputs"].get("reference_summary_sha256") != sha256_file(
            summary_path
        ):
            raise ValueError(f"clock reference summary mismatch: {name}")
        split_records.add(json.dumps(summary["splits"], sort_keys=True))
        training[name] = summary
        clock[name] = clock_payload
        for label, path in {
            "summary": summary_path,
            "checkpoint": checkpoint_path,
            "clock": clock_path,
        }.items():
            inputs[f"{name}.{label}"] = sha256_file(path)
    if len(split_records) != 1:
        raise ValueError("graph-sensor models do not share identical splits")

    actual_name = str(contract["actual_model"])
    comparator_names = [str(value) for value in contract["comparators"]]
    bootstrap = contract["bootstrap"]
    daily_rows: list[dict[str, Any]] = []
    full_session: dict[str, Any] = {}
    for horizon_index, horizon in enumerate(HORIZONS):
        actual = summary_daily(training[actual_name], horizon)
        full_session[horizon] = {}
        for comparator_index, comparator in enumerate(comparator_names):
            compared = summary_daily(training[comparator], horizon)
            full_session[horizon][comparator] = {}
            for metric_index, metric in enumerate(("pearson", "skill_vs_zero_mse")):
                result = paired_result(
                    actual,
                    compared,
                    metric,
                    bootstrap=bootstrap,
                    seed_offset=horizon_index * 20 + comparator_index * 2 + metric_index,
                )
                full_session[horizon][comparator][metric] = result
                paired = paired_delta(actual, compared, metric)
                for row in paired.itertuples(index=False):
                    daily_rows.append(
                        {
                            "scope": "full_session",
                            "horizon": horizon,
                            "bucket": "all",
                            "comparator": comparator,
                            "metric": metric,
                            "date": row.date,
                            "actual": float(row.actual),
                            "comparator_value": float(row.comparator),
                            "delta": float(row.delta),
                        }
                    )

    early_clock: dict[str, Any] = {}
    for bucket_index, bucket in enumerate(contract["gates"]["actionable_buckets"]):
        actual = daily_frame(clock[actual_name], "close", bucket)
        early_clock[bucket] = {}
        for comparator_index, comparator in enumerate(comparator_names):
            compared = daily_frame(clock[comparator], "close", bucket)
            early_clock[bucket][comparator] = {}
            for metric_index, metric in enumerate(("pearson", "skill_vs_zero_mse")):
                result = paired_result(
                    actual,
                    compared,
                    metric,
                    bootstrap=bootstrap,
                    seed_offset=200 + bucket_index * 20 + comparator_index * 2 + metric_index,
                )
                early_clock[bucket][comparator][metric] = result
                paired = paired_delta(actual, compared, metric)
                for row in paired.itertuples(index=False):
                    daily_rows.append(
                        {
                            "scope": "early_close_endpoint",
                            "horizon": "close",
                            "bucket": bucket,
                            "comparator": comparator,
                            "metric": metric,
                            "date": row.date,
                            "actual": float(row.actual),
                            "comparator_value": float(row.comparator),
                            "delta": float(row.delta),
                        }
                    )

    systemic = {
        name: {
            horizon: float(
                payload["test"]["systemic_state_change_energy"][horizon]["pearson"]
            )
            for horizon in HORIZONS
        }
        for name, payload in training.items()
    }
    test_loss = {name: float(payload["test_loss"]) for name, payload in training.items()}
    gates = contract["gates"]
    checks: dict[str, bool] = {}
    for comparator in comparator_names:
        checks[f"test_loss_vs_{comparator}"] = test_loss[actual_name] <= test_loss[
            comparator
        ] * (1.0 + float(gates["maximum_relative_test_loss_degradation"]))
        systemic_deltas = np.asarray(
            [systemic[actual_name][h] - systemic[comparator][h] for h in HORIZONS],
            dtype=np.float64,
        )
        checks[f"systemic_mean_pearson_vs_{comparator}"] = float(
            systemic_deltas.mean()
        ) > float(gates["minimum_systemic_mean_pearson_delta"])
        checks[f"systemic_positive_horizons_vs_{comparator}"] = int(
            (systemic_deltas > 0.0).sum()
        ) >= int(gates["minimum_systemic_positive_horizons"])
        for horizon in gates["short_horizons"]:
            delta = full_session[horizon][comparator]["skill_vs_zero_mse"][
                "mean_delta"
            ]
            checks[f"{horizon}_skill_not_degraded_vs_{comparator}"] = float(delta) >= -float(
                gates["maximum_short_horizon_skill_degradation"]
            )
        for bucket in gates["actionable_buckets"]:
            metrics = early_clock[bucket][comparator]
            prefix = f"{bucket}_vs_{comparator}"
            checks[f"{prefix}_pearson_mean"] = float(
                metrics["pearson"]["mean_delta"]
            ) > float(gates["minimum_early_pearson_delta"])
            checks[f"{prefix}_pearson_lower95"] = float(
                metrics["pearson"]["block_bootstrap"]["lower_95"]
            ) > float(gates["minimum_early_pearson_bootstrap_lower_95"])
            checks[f"{prefix}_skill_mean"] = float(
                metrics["skill_vs_zero_mse"]["mean_delta"]
            ) >= float(gates["minimum_early_skill_delta"])

    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "role": "post_impact_graph_sensor_ablation_audit",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "contract_sha256": sha256_file(contract_path),
        "actual_model": actual_name,
        "comparators": comparator_names,
        "test_loss": test_loss,
        "systemic_state_change_energy_pearson": systemic,
        "full_session": full_session,
        "early_close_endpoint": early_clock,
        "checks": checks,
        "checks_passed": int(sum(checks.values())),
        "checks_total": int(len(checks)),
        "decision": (
            "graph_sensor_increment_confirmed_for_message_passing_development"
            if passed
            else "graph_sensor_increment_not_confirmed"
        ),
        "next_gate": (
            "develop_node_identity_preserving_sparse_message_passing"
            if passed
            else "do_not_promote_scalar_graph_coherence"
        ),
        "inputs": inputs,
        "status": "complete",
    }
    return summary, pd.DataFrame(daily_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit causal graph-coherence increment in the post-impact head."
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
