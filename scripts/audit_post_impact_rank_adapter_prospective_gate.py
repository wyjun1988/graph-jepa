from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_clock_gated_forward import bootstrap_mean
from scripts.audit_post_impact_prospective_ledger_gate import (
    combine_sufficient_statistics,
)
from scripts.replay_post_impact_prospective_ledger import canonical_sha256
from scripts.run_post_impact_rank_adapter_live_shadow import (
    RANK_MODELS,
    load_rank_contract,
    prospective_scope,
)
from stock_v2.prospective_ledger import file_sha256


ROLE = "post_impact_rank_adapter_prospective_gate_contract"
COMPARISON_PAIRS = {
    "aligned_vs_baseline": ("aligned", "baseline"),
    "aligned_vs_own_permuted": ("aligned", "own_permuted"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit immutable rank-adapter D+1 reconciliations against a frozen "
            "session-level read-only shadow gate."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reconciliation-dir", action="append", default=[])
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def _verify_content_hash(payload: Mapping[str, Any], field: str) -> str:
    expected = str(payload.get(field) or "")
    content = dict(payload)
    content.pop(field, None)
    if canonical_sha256(content) != expected:
        raise ValueError(f"rank-adapter reconciliation {field} changed")
    return expected


def load_gate_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 1 or contract.get("role") != ROLE:
        raise ValueError("invalid rank-adapter prospective gate contract")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("rank-adapter prospective gate permits live orders")
    if contract.get("broker_order_calls_allowed") is not False:
        raise ValueError("rank-adapter prospective gate permits broker order calls")
    if tuple(contract.get("models_exact") or ()) != RANK_MODELS:
        raise ValueError("rank-adapter prospective model order changed")
    if set(contract.get("comparisons") or ()) != set(COMPARISON_PAIRS):
        raise ValueError("rank-adapter prospective comparisons changed")
    parent = contract.get("parent_scope_contract")
    if not isinstance(parent, Mapping):
        raise ValueError("rank-adapter gate lacks its parent scope contract")
    parent_path = _resolve(parent.get("path"))
    if file_sha256(parent_path) != parent.get("sha256"):
        raise ValueError("rank-adapter parent scope contract changed")
    load_rank_contract(parent_path)
    for record in contract.get("source_pins", {}).values():
        source = _resolve(record.get("path"))
        if file_sha256(source) != record.get("sha256"):
            raise ValueError("rank-adapter prospective gate source pin changed")
    return contract


def load_reconciliation_directory(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if summary.get("role") != "post_impact_rank_adapter_session_reconciliation_audit":
        raise ValueError("unexpected rank-adapter reconciliation summary role")
    _verify_content_hash(summary, "audit_content_sha256")
    records = [
        json.loads(item.read_text(encoding="utf-8"))
        for item in (path / "records").glob("*.json")
    ]
    by_prediction = {
        str(record["prediction_record_sha256"]): record for record in records
    }
    expected = [str(value) for value in summary["prediction_record_sha256"]]
    if len(by_prediction) != len(records) or set(by_prediction) != set(expected):
        raise ValueError("rank-adapter reconciliation record set changed")
    ordered = [by_prediction[value] for value in expected]
    hashes = [
        _verify_content_hash(record, "reconciliation_content_sha256")
        for record in ordered
    ]
    if hashes != list(summary["reconciliation_content_sha256"]):
        raise ValueError("rank-adapter reconciliation content order changed")
    daily_path = _resolve(summary["rank_contract"])
    if file_sha256(daily_path) != summary["rank_contract_sha256"]:
        raise ValueError("rank-adapter daily contract changed")
    daily = load_rank_contract(daily_path)
    return summary, ordered, daily


def build_session_metrics(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    daily_contract: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> dict[str, Any]:
    session = str(summary["session"])
    if daily_contract.get("daily_session") != session:
        raise ValueError("rank-adapter daily contract crosses sessions")
    parent = daily_contract.get("parent_scope_contract")
    if not isinstance(parent, Mapping) or parent.get("sha256") != gate[
        "parent_scope_contract"
    ]["sha256"]:
        raise ValueError("rank-adapter daily contract parent changed")
    expected_checkpoints = dict(gate["checkpoint_sha256"])
    _first_session, expected_clocks = prospective_scope(daily_contract)
    if list(expected_clocks) != list(gate["primary_clocks_kst_minutes"]):
        raise ValueError("rank-adapter daily clocks differ from gate")
    by_clock: dict[int, Mapping[str, Any]] = {}
    for record in records:
        if record.get("role") != "post_impact_rank_adapter_prediction_reconciliation":
            raise ValueError("rank-adapter reconciliation role changed")
        if record.get("counts_as_forward_evidence") is not True:
            raise ValueError("rank-adapter reconciliation disclaims forward evidence")
        if record.get("live_orders_allowed") is not False:
            raise ValueError("rank-adapter reconciliation permits live orders")
        if record.get("broker_order_calls_executed") != 0:
            raise ValueError("rank-adapter reconciliation contains broker calls")
        if list(record.get("models") or ()) != list(RANK_MODELS):
            raise ValueError("rank-adapter reconciliation model order changed")
        model_pins = record.get("prediction_model_pins")
        if not isinstance(model_pins, Mapping):
            raise ValueError("rank-adapter reconciliation model pins are absent")
        for model, expected in expected_checkpoints.items():
            if model_pins[model].get("checkpoint_sha256") != expected:
                raise ValueError(f"rank-adapter checkpoint changed: {model}")
        for protected in record["protected_output_audit"].values():
            if (
                protected.get("status") != "pass"
                or protected.get("node_5m_maximum_absolute_difference") != 0.0
                or protected.get("systemic_maximum_absolute_difference") != 0.0
            ):
                raise ValueError("rank-adapter protected output gate failed")
        timestamp = pd.Timestamp(
            int(record["decision_timestamp_utc_ns"]), unit="ns", tz="UTC"
        ).tz_convert("Asia/Seoul")
        if str(timestamp.date()) != session:
            raise ValueError("rank-adapter reconciliation timestamp crosses sessions")
        clock = int(timestamp.hour * 60 + timestamp.minute)
        if clock in by_clock:
            raise ValueError("rank-adapter session contains duplicate clocks")
        by_clock[clock] = record
    missing = sorted(set(expected_clocks) - set(by_clock))
    extra = sorted(set(by_clock) - set(expected_clocks))
    if missing or extra:
        return {
            "session": session,
            "status": "rejected_clock_set",
            "missing_clocks": missing,
            "extra_clocks": extra,
            "live_orders_allowed": False,
        }

    minimum_nodes = int(gate["minimum_evidence"]["minimum_nodes_per_cell"])
    horizons = list(gate["primary_horizons"])
    model_cells: dict[str, dict[str, Any]] = {name: {} for name in RANK_MODELS}
    for clock in expected_clocks:
        record = by_clock[int(clock)]
        for horizon in horizons:
            if horizon not in record["eligible_horizons"]:
                raise ValueError("rank-adapter primary horizon was not eligible")
            cell = f"{clock}|{horizon}"
            for model in RANK_MODELS:
                metrics = record["node_metrics"][model][horizon]["endpoint_return"]
                combined = combine_sufficient_statistics([metrics])
                if combined["count"] < minimum_nodes:
                    raise ValueError(f"rank-adapter cell has too few nodes: {cell}")
                if combined["pearson"] is None or combined["skill_vs_zero_mse"] is None:
                    raise ValueError(f"rank-adapter cell metric is undefined: {cell}")
                model_cells[model][cell] = combined

    comparisons: dict[str, Any] = {}
    for name in gate["comparisons"]:
        candidate, comparator = COMPARISON_PAIRS[name]
        cells = {
            cell: {
                "count": model_cells[candidate][cell]["count"],
                "delta_pearson": float(
                    model_cells[candidate][cell]["pearson"]
                    - model_cells[comparator][cell]["pearson"]
                ),
                "delta_skill_vs_zero_mse": float(
                    model_cells[candidate][cell]["skill_vs_zero_mse"]
                    - model_cells[comparator][cell]["skill_vs_zero_mse"]
                ),
            }
            for cell in model_cells[candidate]
        }
        comparisons[name] = {
            "cells": cells,
            "mean_delta_pearson": float(
                np.mean([value["delta_pearson"] for value in cells.values()])
            ),
            "mean_delta_skill_vs_zero_mse": float(
                np.mean(
                    [value["delta_skill_vs_zero_mse"] for value in cells.values()]
                )
            ),
        }
    return {
        "session": session,
        "status": "eligible_complete",
        "clocks": list(expected_clocks),
        "model_cells": model_cells,
        "comparisons": comparisons,
        "protected_outputs_exact": True,
        "live_orders_allowed": False,
    }


def aggregate_gate(
    sessions: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> dict[str, Any]:
    eligible = [row for row in sessions if row["status"] == "eligible_complete"]
    minimum = int(contract["minimum_evidence"]["completed_sessions"])
    gates = contract["shadow_gates"]
    checks: dict[str, bool] = {
        "minimum_completed_sessions": len(eligible) >= minimum,
        "zero_live_orders": all(
            row.get("live_orders_allowed") is False for row in sessions
        ),
        "protected_outputs_exact": all(
            row.get("protected_outputs_exact") is True for row in eligible
        ),
    }
    comparisons: dict[str, Any] = {}
    bootstrap = contract["session_bootstrap"]
    for offset, name in enumerate(contract["comparisons"]):
        pearson = np.asarray(
            [row["comparisons"][name]["mean_delta_pearson"] for row in eligible],
            dtype=np.float64,
        )
        skill = np.asarray(
            [
                row["comparisons"][name]["mean_delta_skill_vs_zero_mse"]
                for row in eligible
            ],
            dtype=np.float64,
        )
        pearson_bootstrap = bootstrap_mean(
            pearson,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["moving_block_length_sessions"]),
            seed=int(bootstrap["seed"]) + offset * 2,
        )
        skill_bootstrap = bootstrap_mean(
            skill,
            samples=int(bootstrap["samples"]),
            block_length=int(bootstrap["moving_block_length_sessions"]),
            seed=int(bootstrap["seed"]) + offset * 2 + 1,
        )
        cell_ids = (
            list(eligible[0]["comparisons"][name]["cells"]) if eligible else []
        )
        cell_means = {
            cell: float(
                np.mean(
                    [
                        row["comparisons"][name]["cells"][cell]["delta_pearson"]
                        for row in eligible
                    ]
                )
            )
            for cell in cell_ids
        }
        positive_cells = sum(value > 0.0 for value in cell_means.values())
        mean_pearson = float(pearson.mean()) if len(pearson) else None
        mean_skill = float(skill.mean()) if len(skill) else None
        prefix = name
        checks[f"{name}.mean_pearson"] = bool(
            mean_pearson is not None
            and mean_pearson >= float(gates[f"{prefix}_mean_pearson_delta_minimum"])
        )
        checks[f"{name}.mean_skill"] = bool(
            mean_skill is not None
            and mean_skill >= float(gates[f"{prefix}_mean_skill_delta_minimum"])
        )
        checks[f"{name}.pearson_lower_95"] = bool(
            pearson_bootstrap["status"] == "complete"
            and pearson_bootstrap["lower_95"]
            >= float(gates[f"{prefix}_pearson_lower_95_minimum"])
        )
        checks[f"{name}.skill_lower_95"] = bool(
            skill_bootstrap["status"] == "complete"
            and skill_bootstrap["lower_95"]
            >= float(gates[f"{prefix}_skill_lower_95_minimum"])
        )
        checks[f"{name}.positive_cells"] = positive_cells >= int(
            gates["minimum_positive_primary_cells_per_comparison"]
        )
        comparisons[name] = {
            "sessions": len(eligible),
            "mean_session_delta_pearson": mean_pearson,
            "mean_session_delta_skill_vs_zero_mse": mean_skill,
            "pearson_block_bootstrap": pearson_bootstrap,
            "skill_block_bootstrap": skill_bootstrap,
            "cell_mean_delta_pearson": cell_means,
            "positive_primary_cells": positive_cells,
        }
    shadow_qualified = len(eligible) >= minimum and all(checks.values())
    return {
        "eligible_sessions": len(eligible),
        "rejected_sessions": len(sessions) - len(eligible),
        "minimum_sessions": minimum,
        "comparisons": comparisons,
        "checks": checks,
        "checks_passed": sum(checks.values()),
        "checks_total": len(checks),
        "decision": (
            "qualified_for_extended_read_only_shadow_not_trading"
            if shadow_qualified
            else (
                "insufficient_forward_evidence_accumulating"
                if len(eligible) < minimum
                else "rank_adapter_prospective_gate_failed"
            )
        ),
        "shadow_qualified": shadow_qualified,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = load_gate_contract(contract_path)
    loaded = [
        load_reconciliation_directory(Path(value))
        for value in args.reconciliation_dir
    ]
    sessions = [
        build_session_metrics(summary, records, daily, contract)
        for summary, records, daily in loaded
    ]
    if len({row["session"] for row in sessions}) != len(sessions):
        raise ValueError("rank-adapter gate contains duplicate sessions")
    sessions.sort(key=lambda row: row["session"])
    gate = aggregate_gate(sessions, contract)
    output = {
        "schema_version": 1,
        "role": "post_impact_rank_adapter_prospective_gate_audit",
        "status": "pass",
        "contract": str(contract_path),
        "contract_sha256": file_sha256(contract_path),
        "sessions": sessions,
        "gate": gate,
        "test_split_evaluated": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    output["audit_content_sha256"] = canonical_sha256(output)
    path = Path(args.output)
    encoded = json.dumps(
        output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise FileExistsError(f"immutable rank-adapter gate output differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "pass",
                "decision": gate["decision"],
                "eligible_sessions": gate["eligible_sessions"],
                "shadow_qualified": gate["shadow_qualified"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
