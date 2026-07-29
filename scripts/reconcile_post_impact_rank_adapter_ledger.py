from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.reconcile_post_impact_prospective_ledger import (
    _load_artifact,
    reconcile_prediction_record,
)
from scripts.replay_post_impact_prospective_ledger import canonical_sha256
from scripts.run_post_impact_rank_adapter_live_shadow import (
    RANK_MODELS,
    load_rank_contract,
    prospective_scope,
)
from scripts.train_post_impact_reforecast import DayRelease
from stock_v2.prospective_ledger import file_sha256, read_prediction_ledger


COMPARISONS = {
    "aligned_vs_baseline": ("aligned", "baseline"),
    "aligned_vs_own_permuted": ("aligned", "own_permuted"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile immutable rank-adapter shadow commits against a later "
            "completed-session label release."
        )
    )
    parser.add_argument("--rank-contract", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _clock_minute(record: Mapping[str, Any]) -> int:
    timestamp = pd.Timestamp(
        int(record["decision_timestamp_utc_ns"]), unit="ns", tz="UTC"
    ).tz_convert("Asia/Seoul")
    if str(timestamp.date()) != str(record["session"]):
        raise ValueError("rank-adapter commit timestamp crosses sessions")
    return int(timestamp.hour * 60 + timestamp.minute)


def validate_prediction_contract(
    record: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    contract_sha256: str,
) -> None:
    session = str(record.get("session"))
    if session != str(contract.get("daily_session")):
        raise ValueError("rank-adapter commit session differs from daily contract")
    _first_session, primary_clocks = prospective_scope(contract)
    if _clock_minute(record) not in primary_clocks:
        raise ValueError("rank-adapter commit clock is outside the frozen scope")
    if not str(record.get("commit_id", "")).endswith(
        "|post_impact_rank_adapter_live_v1"
    ):
        raise ValueError("unexpected rank-adapter commit suffix")
    if record.get("source_mode") != "live_read_only":
        raise ValueError("rank-adapter commit is not live read-only")
    if record.get("live_orders_allowed") is not False:
        raise ValueError("rank-adapter commit permits live orders")
    if record.get("broker_order_calls_executed") != 0:
        raise ValueError("rank-adapter commit contains broker order calls")
    causality = record.get("causality")
    if not isinstance(causality, Mapping) or any(
        causality.get(name) is not True
        for name in (
            "completed_bars_only",
            "future_intraday_rows_absent_from_model_input",
            "labels_absent_from_model_input",
            "model_eval_mode",
        )
    ):
        raise ValueError("rank-adapter commit causality claims changed")

    models = record.get("models")
    if not isinstance(models, Mapping) or set(models) != set(RANK_MODELS):
        raise ValueError("rank-adapter commit model set changed")
    for name in RANK_MODELS:
        observed = models[name]
        expected = contract["models"][name]
        for field in (
            "checkpoint_sha256",
            "graph_message_mode",
            "graph_message_fusion",
        ):
            if observed.get(field) != expected.get(field):
                raise ValueError(f"rank-adapter model pin changed: {name} {field}")

    input_pins = record.get("input_pins")
    if not isinstance(input_pins, Mapping):
        raise ValueError("rank-adapter commit input pins are absent")
    runtime = contract["runtime_inputs"]
    required = {
        "rank_shadow_contract": contract_sha256,
        "selection_audit": contract["selection_audit"]["sha256"],
        "latency_qualification": contract["latency_qualification"]["sha256"],
        "historical_day_release_manifest": runtime[
            "historical_day_release"
        ]["manifest_sha256"],
        "prospective_stale_cache_manifest": runtime[
            "prospective_stale_cache"
        ]["manifest_sha256"],
        "lifecycle_release_manifest": runtime["lifecycle_release"][
            "manifest_sha256"
        ],
        "rank_live_inference_source": contract["source_pins"][
            "scripts/run_post_impact_rank_adapter_live_shadow.py"
        ],
    }
    if any(input_pins.get(name) != expected for name, expected in required.items()):
        raise ValueError("rank-adapter commit input pin changed")


def _maximum_difference(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or not np.isfinite(left).all() or not np.isfinite(
        right
    ).all():
        raise ValueError("rank-adapter protected arrays are not finite and aligned")
    return float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))


def protected_output_audit(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    models = [str(value) for value in arrays["model_names"].tolist()]
    horizons = [str(value) for value in arrays["horizon_labels"].tolist()]
    if models != list(RANK_MODELS) or "5m" not in horizons:
        raise ValueError("rank-adapter artifact axes changed")
    baseline = models.index("baseline")
    horizon_5m = horizons.index("5m")
    node = np.asarray(arrays["node_prediction"])
    systemic = np.asarray(arrays["systemic_prediction"])
    result: dict[str, Any] = {}
    for name in ("aligned", "own_permuted"):
        index = models.index(name)
        node_difference = _maximum_difference(
            node[index, :, horizon_5m], node[baseline, :, horizon_5m]
        )
        systemic_difference = _maximum_difference(
            systemic[index], systemic[baseline]
        )
        result[f"{name}_vs_baseline"] = {
            "node_5m_maximum_absolute_difference": node_difference,
            "systemic_maximum_absolute_difference": systemic_difference,
            "status": (
                "pass"
                if node_difference == 0.0 and systemic_difference == 0.0
                else "blocked"
            ),
        }
    if any(record["status"] != "pass" for record in result.values()):
        raise ValueError("rank-adapter protected output changed")
    return result


def rank_comparisons(
    reconciliation: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    metrics = reconciliation["node_metrics"]
    horizons = list(reconciliation["eligible_horizons"])
    result: dict[str, dict[str, Any]] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        cells: dict[str, Any] = {}
        for horizon in horizons:
            left = metrics[candidate][horizon]["endpoint_return"]
            right = metrics[comparator][horizon]["endpoint_return"]
            if left["count"] != right["count"]:
                raise ValueError("rank-adapter comparison counts changed")
            deltas = {}
            for metric in ("pearson", "skill_vs_zero_mse", "sign_accuracy"):
                left_value = left[metric]
                right_value = right[metric]
                deltas[f"candidate_minus_comparator_{metric}"] = (
                    None
                    if left_value is None or right_value is None
                    else float(left_value - right_value)
                )
            cells[horizon] = {
                "candidate": candidate,
                "comparator": comparator,
                "count": int(left["count"]),
                **deltas,
            }
        result[name] = cells
    return result


def reconcile_rank_record(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    release: DayRelease,
    contract: Mapping[str, Any],
    contract_path: Path,
) -> dict[str, Any]:
    contract_sha = file_sha256(contract_path)
    validate_prediction_contract(
        record, contract=contract, contract_sha256=contract_sha
    )
    result = reconcile_prediction_record(
        record, artifact_root=artifact_root, release=release
    )
    if list(result["models"]) != list(RANK_MODELS):
        raise ValueError("rank-adapter reconciliation model order changed")
    arrays = _load_artifact(record, artifact_root)
    result["role"] = "post_impact_rank_adapter_prediction_reconciliation"
    result["rank_contract"] = str(contract_path)
    result["rank_contract_sha256"] = contract_sha
    result["protected_output_audit"] = protected_output_audit(arrays)
    result["primary_endpoint_comparisons"] = rank_comparisons(result)
    result["reconciliation_content_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in result.items()
            if key != "reconciliation_content_sha256"
        }
    )
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    contract_path = Path(args.rank_contract)
    contract = load_rank_contract(contract_path)
    session = str(pd.Timestamp(args.session).date())
    if contract.get("daily_session") != session:
        raise ValueError("rank-adapter reconciliation contract session changed")
    artifact_root = Path(args.artifact_root)
    records = read_prediction_ledger(
        Path(args.ledger), artifact_root=artifact_root
    )
    selected = [record for record in records if str(record.get("session")) == session]
    if not selected:
        raise ValueError("no rank-adapter records exist for the requested session")
    release = DayRelease(Path(args.day_release_dir), cache=False)
    if release.dates[-1] < session:
        raise ValueError("rank-adapter label release is not mature")
    reconciliations = [
        reconcile_rank_record(
            record,
            artifact_root=artifact_root,
            release=release,
            contract=contract,
            contract_path=contract_path,
        )
        for record in selected
    ]
    _first_session, primary_clocks = prospective_scope(contract)
    observed_clocks = sorted(
        int(record["decision_clock_minute_kst"]) for record in selected
    )
    if observed_clocks != list(primary_clocks):
        raise ValueError("rank-adapter reconciliation lacks exact primary clocks")
    summary = {
        "schema_version": 1,
        "role": "post_impact_rank_adapter_session_reconciliation_audit",
        "status": "pass",
        "session": session,
        "rank_contract": str(contract_path),
        "rank_contract_sha256": file_sha256(contract_path),
        "prediction_ledger": file_sha256(Path(args.ledger)),
        "prediction_ledger_head_sha256": records[-1]["record_sha256"],
        "records_reconciled": len(reconciliations),
        "primary_clocks_kst_minutes": list(primary_clocks),
        "prediction_record_sha256": [
            row["prediction_record_sha256"] for row in reconciliations
        ],
        "reconciliation_content_sha256": [
            row["reconciliation_content_sha256"] for row in reconciliations
        ],
        "counts_as_forward_session": True,
        "minimum_forward_sessions": int(
            contract["prospective_evidence"][
                "minimum_complete_sessions_before_any_policy_review"
            ]
        ),
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    summary["audit_content_sha256"] = canonical_sha256(summary)

    output = Path(args.output_dir)
    temporary = output.with_name(f".{output.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        for row in reconciliations:
            _write_json(
                temporary / "records" / f"{row['prediction_record_sha256']}.json",
                row,
            )
        _write_json(temporary / "summary.json", summary)
        if output.exists():
            existing = output / "summary.json"
            if not existing.is_file() or json.loads(
                existing.read_text(encoding="utf-8")
            ) != summary:
                raise FileExistsError(
                    f"immutable rank reconciliation differs: {output}"
                )
            shutil.rmtree(temporary)
        else:
            temporary.replace(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "pass",
                "session": session,
                "records_reconciled": len(reconciliations),
                "counts_as_forward_session": True,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
