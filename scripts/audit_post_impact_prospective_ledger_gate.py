from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_clock_gated_forward import bootstrap_mean
from scripts.replay_post_impact_prospective_ledger import canonical_sha256
from stock_v2.prospective_ledger import file_sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit immutable D+1 reconciliations against the frozen prospective "
            "ledger promotion contract."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reconciliation-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


STATISTIC_NAMES = (
    "sum_prediction",
    "sum_actual",
    "sum_prediction_squared",
    "sum_actual_squared",
    "sum_cross",
    "sum_squared_error",
    "sum_absolute_error",
)


def combine_sufficient_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = sum(int(row["count"]) for row in rows)
    totals = {
        name: float(
            sum(float(row["sufficient_statistics"][name]) for row in rows)
        )
        for name in STATISTIC_NAMES
    }
    sign_matches = sum(
        int(row["sufficient_statistics"]["sign_matches"]) for row in rows
    )
    if count <= 0:
        raise ValueError("prospective primary cell has no available labels")
    centered_prediction = (
        totals["sum_prediction_squared"]
        - totals["sum_prediction"] ** 2 / count
    )
    centered_actual = (
        totals["sum_actual_squared"] - totals["sum_actual"] ** 2 / count
    )
    covariance = (
        totals["sum_cross"]
        - totals["sum_prediction"] * totals["sum_actual"] / count
    )
    denominator = math.sqrt(
        max(centered_prediction, 0.0) * max(centered_actual, 0.0)
    )
    pearson = None if denominator <= 1e-20 else float(covariance / denominator)
    zero_mse = totals["sum_actual_squared"] / count
    skill = (
        None
        if totals["sum_actual_squared"] <= 1e-20
        else float(1.0 - totals["sum_squared_error"] / totals["sum_actual_squared"])
    )
    return {
        "count": count,
        "mse": float(totals["sum_squared_error"] / count),
        "mae": float(totals["sum_absolute_error"] / count),
        "zero_mse": float(zero_mse),
        "skill_vs_zero_mse": skill,
        "pearson": pearson,
        "sign_accuracy": float(sign_matches / count),
        "sufficient_statistics": {**totals, "sign_matches": sign_matches},
    }


def _verify_content_hash(payload: Mapping[str, Any], field: str) -> str:
    expected = str(payload.get(field) or "")
    content = dict(payload)
    content.pop(field, None)
    if canonical_sha256(content) != expected:
        raise ValueError(f"prospective reconciliation {field} changed")
    return expected


def load_reconciliation_directory(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("role") != "post_impact_prospective_session_reconciliation_audit":
        raise ValueError("unexpected prospective reconciliation summary role")
    _verify_content_hash(summary, "audit_content_sha256")
    loaded = [
        json.loads(record.read_text(encoding="utf-8"))
        for record in (path / "records").glob("*.json")
    ]
    if len(loaded) != int(summary["records_reconciled"]):
        raise ValueError("prospective reconciliation record count changed")
    by_prediction = {
        str(record["prediction_record_sha256"]): record for record in loaded
    }
    expected_predictions = [str(value) for value in summary["prediction_record_sha256"]]
    if len(by_prediction) != len(loaded) or set(by_prediction) != set(expected_predictions):
        raise ValueError("prospective prediction record set changed")
    records = [by_prediction[value] for value in expected_predictions]
    hashes = [
        _verify_content_hash(record, "reconciliation_content_sha256")
        for record in records
    ]
    if hashes != list(summary["reconciliation_content_sha256"]):
        raise ValueError("prospective reconciliation content list changed")
    return summary, records


def _record_clock(record: Mapping[str, Any]) -> str:
    timestamp = pd.Timestamp(
        int(record["decision_timestamp_utc_ns"]), unit="ns", tz="UTC"
    ).tz_convert("Asia/Seoul")
    if str(timestamp.date()) != str(record["session"]):
        raise ValueError("prospective reconciliation timestamp crosses sessions")
    return timestamp.strftime("%H:%M")


def build_session_metrics(
    records: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    if not records:
        raise ValueError("prospective session has no reconciliation records")
    sessions = {str(record["session"]) for record in records}
    if len(sessions) != 1:
        raise ValueError("prospective reconciliation directory crosses sessions")
    session = next(iter(sessions))
    required_models = list(contract["commit_contract"]["artifact_models_exact"])
    expected_checkpoints = dict(contract["commit_contract"]["checkpoint_sha256"])
    expected_input_pins = dict(contract["commit_contract"]["required_input_pins"])
    suffix = str(contract["commit_contract"]["commit_suffix"])
    by_clock: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not str(record["prediction_commit_id"]).endswith("|" + suffix):
            continue
        if list(record["models"]) != required_models:
            raise ValueError("prospective scientific model axis changed")
        model_pins = record.get("prediction_model_pins")
        if not isinstance(model_pins, Mapping) or set(model_pins) != set(required_models):
            raise ValueError("prospective scientific model pins changed")
        for model in required_models:
            if model_pins[model].get("checkpoint_sha256") != expected_checkpoints[model]:
                raise ValueError(f"prospective checkpoint changed: {model}")
        input_pins = record.get("prediction_input_pins")
        if not isinstance(input_pins, Mapping) or any(
            input_pins.get(name) != expected for name, expected in expected_input_pins.items()
        ):
            raise ValueError("prospective inference input source pins changed")
        causality = record.get("prediction_causality")
        if not isinstance(causality, Mapping) or any(
            causality.get(name) is not True
            for name in (
                "completed_bars_only",
                "future_intraday_rows_absent_from_model_input",
                "labels_absent_from_model_input",
                "model_eval_mode",
            )
        ):
            raise ValueError("prospective inference causality claims changed")
        if record.get("counts_as_forward_evidence") is not True:
            raise ValueError("prospective reconciliation disclaims forward evidence")
        if record.get("live_orders_allowed") is not False:
            raise ValueError("prospective reconciliation permits live orders")
        if record.get("broker_order_calls_executed") != 0:
            raise ValueError("prospective reconciliation records broker calls")
        clock = _record_clock(record)
        if clock in by_clock:
            raise ValueError("prospective session contains duplicate scientific clocks")
        by_clock[clock] = record

    required_clocks = {
        clock
        for clocks in contract["clock_contract"].values()
        for clock in clocks
    }
    missing = sorted(required_clocks - set(by_clock))
    if missing:
        return {
            "session": session,
            "status": "rejected_missing_primary_clocks",
            "missing_clocks": missing,
            "available_clocks": sorted(by_clock),
            "live_orders_allowed": False,
        }

    model_cells: dict[str, dict[str, dict[str, Any]]] = {
        model: {} for model in required_models
    }
    cells = list(contract["primary_endpoint"]["cells"])
    minimum_nodes = int(contract["minimum_evidence"]["minimum_nodes_per_primary_cell"])
    for cell in cells:
        horizon = str(cell["horizon"])
        bucket = str(cell["bucket"])
        clocks = list(contract["clock_contract"][bucket])
        cell_id = f"{horizon}|{bucket}"
        for model in required_models:
            rows = []
            for clock in clocks:
                record = by_clock[clock]
                if horizon not in record["eligible_horizons"]:
                    raise ValueError(
                        f"prospective primary horizon was not eligible: {clock} {horizon}"
                    )
                rows.append(
                    record["node_metrics"][model][horizon]["endpoint_return"]
                )
            combined = combine_sufficient_statistics(rows)
            if combined["count"] < minimum_nodes:
                raise ValueError(
                    f"prospective primary cell has only {combined['count']} nodes: {cell_id}"
                )
            if combined["pearson"] is None or combined["skill_vs_zero_mse"] is None:
                raise ValueError(f"prospective primary cell metric is undefined: {cell_id}")
            model_cells[model][cell_id] = combined

    comparison_pairs = {
        "latent_vs_direct": ("latent", "direct"),
        "latent_vs_latent_only_placebo": ("latent", "latent_only_placebo"),
        "state_vs_direct": ("state", "direct"),
        "latent_vs_state": ("latent", "state"),
    }
    comparisons: dict[str, Any] = {}
    for name in [
        *contract["comparisons"]["primary"],
        *contract["comparisons"]["diagnostic"],
    ]:
        candidate, comparator = comparison_pairs[name]
        cell_rows: dict[str, Any] = {}
        for cell_id in model_cells[candidate]:
            left = model_cells[candidate][cell_id]
            right = model_cells[comparator][cell_id]
            if left["count"] != right["count"]:
                raise ValueError("prospective paired model counts changed")
            cell_rows[cell_id] = {
                "count": left["count"],
                "delta_pearson": float(left["pearson"] - right["pearson"]),
                "delta_skill_vs_zero_mse": float(
                    left["skill_vs_zero_mse"] - right["skill_vs_zero_mse"]
                ),
            }
        comparisons[name] = {
            "cells": cell_rows,
            "mean_delta_pearson": float(
                np.mean([row["delta_pearson"] for row in cell_rows.values()])
            ),
            "mean_delta_skill_vs_zero_mse": float(
                np.mean(
                    [
                        row["delta_skill_vs_zero_mse"]
                        for row in cell_rows.values()
                    ]
                )
            ),
        }
    return {
        "session": session,
        "status": "eligible_complete",
        "clocks": sorted(required_clocks),
        "model_cells": model_cells,
        "comparisons": comparisons,
        "live_orders_allowed": False,
    }


def aggregate_gate(
    sessions: Sequence[Mapping[str, Any]],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    eligible = [row for row in sessions if row["status"] == "eligible_complete"]
    minimum = int(contract["minimum_evidence"]["completed_sessions"])
    primary = list(contract["comparisons"]["primary"])
    result: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "minimum_completed_sessions": len(eligible) >= minimum,
        "zero_live_orders": all(
            row.get("live_orders_allowed") is False for row in sessions
        ),
    }
    bootstrap = contract["session_bootstrap"]
    for comparison in primary:
        values_pearson = np.asarray(
            [row["comparisons"][comparison]["mean_delta_pearson"] for row in eligible],
            dtype=np.float64,
        )
        values_skill = np.asarray(
            [
                row["comparisons"][comparison]["mean_delta_skill_vs_zero_mse"]
                for row in eligible
            ],
            dtype=np.float64,
        )
        pearson_bootstrap = bootstrap_mean(
            values_pearson,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["moving_block_length_sessions"]),
            seed=int(bootstrap["seed"]),
        )
        skill_bootstrap = bootstrap_mean(
            values_skill,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["moving_block_length_sessions"]),
            seed=int(bootstrap["seed"]) + 1,
        )
        cell_ids = list(eligible[0]["comparisons"][comparison]["cells"]) if eligible else []
        cell_means = {
            cell: float(
                np.mean(
                    [
                        row["comparisons"][comparison]["cells"][cell][
                            "delta_pearson"
                        ]
                        for row in eligible
                    ]
                )
            )
            for cell in cell_ids
        }
        positive_cells = sum(value > 0.0 for value in cell_means.values())
        prefix = (
            "latent_vs_placebo"
            if comparison == "latent_vs_latent_only_placebo"
            else comparison
        )
        gates = contract["promotion_gates"]
        mean_pearson = float(values_pearson.mean()) if len(values_pearson) else None
        mean_skill = float(values_skill.mean()) if len(values_skill) else None
        checks[f"{comparison}.mean_pearson"] = bool(
            mean_pearson is not None
            and mean_pearson >= float(gates[f"{prefix}_mean_pearson_delta_minimum"])
        )
        checks[f"{comparison}.mean_skill"] = bool(
            mean_skill is not None
            and mean_skill >= float(gates[f"{prefix}_mean_skill_delta_minimum"])
        )
        checks[f"{comparison}.pearson_lower_95"] = bool(
            pearson_bootstrap["status"] == "complete"
            and pearson_bootstrap["lower_95"]
            >= float(gates[f"{prefix}_pearson_bootstrap_lower_95_minimum"])
        )
        checks[f"{comparison}.skill_lower_95"] = bool(
            skill_bootstrap["status"] == "complete"
            and skill_bootstrap["lower_95"]
            >= float(gates[f"{prefix}_skill_bootstrap_lower_95_minimum"])
        )
        checks[f"{comparison}.positive_cells"] = positive_cells >= int(
            gates["minimum_positive_primary_cells_for_each_primary_comparison"]
        )
        result[comparison] = {
            "sessions": len(eligible),
            "mean_session_delta_pearson": mean_pearson,
            "mean_session_delta_skill_vs_zero_mse": mean_skill,
            "pearson_block_bootstrap": pearson_bootstrap,
            "skill_block_bootstrap": skill_bootstrap,
            "cell_mean_delta_pearson": cell_means,
            "positive_primary_cells": positive_cells,
        }
    promotion = len(eligible) >= minimum and all(checks.values())
    return {
        "eligible_sessions": len(eligible),
        "rejected_sessions": len(sessions) - len(eligible),
        "minimum_sessions": minimum,
        "comparisons": result,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "decision": (
            "eligible_for_longer_read_only_shadow_only"
            if promotion
            else (
                "insufficient_forward_evidence_accumulating"
                if len(eligible) < minimum
                else "prospective_ledger_gate_failed"
            )
        ),
        "promotion_scope": "read_only_shadow_only" if promotion else "none",
        "promotion_eligible": promotion,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("role") != "post_impact_prospective_ledger_gate_contract":
        raise ValueError("unexpected prospective ledger gate contract")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("prospective ledger gate contract permits live orders")
    parent = contract["parent_forward_contract"]
    parent_path = ROOT / parent["path"]
    if file_sha256(parent_path) != parent["sha256"]:
        raise ValueError("prospective ledger parent contract changed")
    for name, source in contract.get("source_pins", {}).items():
        source_path = ROOT / source["path"]
        if file_sha256(source_path) != source["sha256"]:
            raise ValueError(f"prospective ledger source pin changed: {name}")
    reports = []
    report_pins = []
    sessions: list[dict[str, Any]] = []
    for value in args.reconciliation_dir:
        path = Path(value)
        summary, records = load_reconciliation_directory(path)
        reports.append((summary, records))
        report_pins.append(
            {"path": str(path / "summary.json"), "sha256": file_sha256(path / "summary.json")}
        )
        sessions.append(build_session_metrics(records, contract))
    if len({row["session"] for row in sessions}) != len(sessions):
        raise ValueError("prospective gate received duplicate sessions")
    sessions.sort(key=lambda row: row["session"])
    aggregate = aggregate_gate(sessions, contract)
    output = {
        "schema_version": 1,
        "role": "post_impact_prospective_ledger_gate_audit",
        "status": "pass",
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "reconciliation_reports": report_pins,
        "sessions": sessions,
        "aggregate": aggregate,
        "promotion_eligible": aggregate["promotion_eligible"],
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    output["audit_content_sha256"] = canonical_sha256(output)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    print(
        json.dumps(
            {
                "status": "pass",
                "eligible_sessions": aggregate["eligible_sessions"],
                "decision": aggregate["decision"],
                "promotion_eligible": aggregate["promotion_eligible"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
