from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping, Sequence
from uuid import uuid4

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_post_impact_prospective_ledger import canonical_sha256
from scripts.run_post_impact_live_prospective_inference import (
    prospective_horizon_eligibility,
)
from scripts.train_post_impact_reforecast import DayRelease
from stock_v2.prospective_ledger import (
    file_sha256,
    read_prediction_ledger,
)


COMPARISONS = {
    "latent_vs_direct": ("latent", "direct"),
    "latent_vs_latent_only_placebo": ("latent", "latent_only_placebo"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile precommitted live post-impact predictions against a later "
            "immutable completed-session label release."
        )
    )
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _pearson(prediction: np.ndarray, actual: np.ndarray) -> float | None:
    x = np.asarray(prediction, dtype=np.float64)
    y = np.asarray(actual, dtype=np.float64)
    if len(x) < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return None
    value = float(np.corrcoef(x, y)[0, 1])
    return value if np.isfinite(value) else None


def masked_node_metrics(
    prediction: np.ndarray,
    actual: np.ndarray,
    available: np.ndarray,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    observed = np.asarray(actual, dtype=np.float64)
    mask = np.asarray(available, dtype=bool)
    if predicted.shape != observed.shape or predicted.shape != mask.shape:
        raise ValueError("prospective node metric arrays do not align")
    mask &= np.isfinite(predicted) & np.isfinite(observed)
    count = int(np.count_nonzero(mask))
    if count == 0:
        return {
            "count": 0,
            "mse": None,
            "mae": None,
            "zero_mse": None,
            "skill_vs_zero_mse": None,
            "pearson": None,
            "sign_accuracy": None,
            "sufficient_statistics": {
                "sum_prediction": 0.0,
                "sum_actual": 0.0,
                "sum_prediction_squared": 0.0,
                "sum_actual_squared": 0.0,
                "sum_cross": 0.0,
                "sum_squared_error": 0.0,
                "sum_absolute_error": 0.0,
                "sign_matches": 0,
            },
        }
    x = predicted[mask]
    y = observed[mask]
    squared_error = np.square(x - y)
    zero_mse = float(np.mean(np.square(y)))
    skill = None if zero_mse <= 1e-20 else float(1.0 - squared_error.mean() / zero_mse)
    return {
        "count": count,
        "mse": float(squared_error.mean()),
        "mae": float(np.mean(np.abs(x - y))),
        "zero_mse": zero_mse,
        "skill_vs_zero_mse": skill,
        "pearson": _pearson(x, y),
        "sign_accuracy": float(np.mean(np.signbit(x) == np.signbit(y))),
        "sufficient_statistics": {
            "sum_prediction": float(np.sum(x)),
            "sum_actual": float(np.sum(y)),
            "sum_prediction_squared": float(np.sum(np.square(x))),
            "sum_actual_squared": float(np.sum(np.square(y))),
            "sum_cross": float(np.sum(x * y)),
            "sum_squared_error": float(np.sum(squared_error)),
            "sum_absolute_error": float(np.sum(np.abs(x - y))),
            "sign_matches": int(np.count_nonzero(np.signbit(x) == np.signbit(y))),
        },
    }


def _load_artifact(record: Mapping[str, Any], artifact_root: Path) -> dict[str, np.ndarray]:
    relative = Path(str(record["prediction_artifact"]["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("prospective prediction artifact path is unsafe")
    path = (artifact_root.resolve() / relative).resolve()
    path.relative_to(artifact_root.resolve())
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _assert_artifact_schema(
    arrays: Mapping[str, np.ndarray],
    release: DayRelease,
) -> tuple[list[str], list[str], list[str], list[str]]:
    model_names = [str(value) for value in arrays["model_names"].tolist()]
    tickers = [str(value) for value in arrays["tickers"].tolist()]
    horizons = [str(value) for value in arrays["horizon_labels"].tolist()]
    targets = [str(value) for value in arrays["target_names"].tolist()]
    systemic = [str(value) for value in arrays["systemic_target_names"].tolist()]
    if len(model_names) != len(set(model_names)):
        raise ValueError("prospective artifact contains duplicate models")
    if tuple(tickers) != tuple(release.tickers):
        raise ValueError("prospective artifact ticker order changed")
    if tuple(horizons) != tuple(release.horizon_labels):
        raise ValueError("prospective artifact horizon order changed")
    if tuple(targets) != tuple(release.target_names):
        raise ValueError("prospective artifact target order changed")
    if tuple(systemic) != tuple(release.systemic_target_names):
        raise ValueError("prospective artifact systemic target order changed")
    expected_node = (len(model_names), len(tickers), len(horizons), len(targets))
    expected_systemic = (len(model_names), len(horizons), len(systemic))
    if arrays["node_prediction"].shape != expected_node:
        raise ValueError("prospective node prediction shape changed")
    if arrays["systemic_prediction"].shape != expected_systemic:
        raise ValueError("prospective systemic prediction shape changed")
    if not np.isfinite(arrays["node_prediction"]).all():
        raise ValueError("prospective node predictions are non-finite")
    if not np.isfinite(arrays["systemic_prediction"]).all():
        raise ValueError("prospective systemic predictions are non-finite")
    return model_names, horizons, targets, systemic


def reconcile_prediction_record(
    record: Mapping[str, Any],
    *,
    artifact_root: Path,
    release: DayRelease,
) -> dict[str, Any]:
    if record.get("source_mode") != "live_read_only":
        raise ValueError("only live read-only records count as prospective evidence")
    session = str(record["session"])
    if session not in release.records:
        raise ValueError("prospective session is absent from the label release")
    arrays = _load_artifact(record, artifact_root)
    models, horizons, targets, systemic_targets = _assert_artifact_schema(
        arrays, release
    )
    decision_values = np.asarray(
        arrays["decision_timestamp_utc_ns"], dtype=np.int64
    ).reshape(-1)
    if decision_values.tolist() != [int(record["decision_timestamp_utc_ns"])]:
        raise ValueError("prospective artifact decision timestamp changed")
    day = release.load(session)
    timestamps = np.asarray(day["timestamps_utc_ns"], dtype=np.int64)
    locations = np.flatnonzero(timestamps == decision_values[0])
    if len(locations) != 1:
        raise ValueError("prospective decision timestamp does not map to one label row")
    decision_index = int(locations[0])

    generated = datetime.fromisoformat(str(record["prediction_generated_at_utc"]))
    expected_eligibility = prospective_horizon_eligibility(
        decision_timestamp_utc_ns=int(decision_values[0]),
        generated_at_utc=generated,
        horizon_labels=horizons,
        session=session,
    )
    if record.get("prospective_horizon_eligibility") != expected_eligibility:
        raise ValueError("prospective horizon eligibility changed")
    eligible_horizons = [name for name in horizons if expected_eligibility[name]]
    if not eligible_horizons:
        raise ValueError("prospective record has no eligible horizons")

    node_prediction = np.asarray(arrays["node_prediction"], dtype=np.float64)
    systemic_prediction = np.asarray(
        arrays["systemic_prediction"], dtype=np.float64
    )
    actual_node = np.asarray(day["targets"][decision_index], dtype=np.float64)
    actual_node_available = np.asarray(
        day["target_available"][decision_index], dtype=bool
    )
    actual_systemic = np.asarray(
        day["systemic_targets"][decision_index], dtype=np.float64
    )
    actual_systemic_available = np.asarray(
        day["systemic_available"][decision_index], dtype=bool
    )

    node_metrics: dict[str, Any] = {}
    systemic_metrics: dict[str, Any] = {}
    for model_index, model in enumerate(models):
        model_node: dict[str, Any] = {}
        model_systemic: dict[str, Any] = {}
        for horizon_index, horizon in enumerate(horizons):
            if horizon not in eligible_horizons:
                continue
            model_node[horizon] = {
                target: masked_node_metrics(
                    node_prediction[model_index, :, horizon_index, target_index],
                    actual_node[:, horizon_index, target_index],
                    actual_node_available[:, horizon_index, target_index],
                )
                for target_index, target in enumerate(targets)
            }
            values: dict[str, Any] = {}
            for target_index, target in enumerate(systemic_targets):
                available = bool(actual_systemic_available[horizon_index, target_index])
                actual = float(actual_systemic[horizon_index, target_index])
                prediction = float(
                    systemic_prediction[model_index, horizon_index, target_index]
                )
                if not available or not np.isfinite(actual):
                    values[target] = {
                        "available": False,
                        "prediction": prediction,
                        "actual": None,
                        "error": None,
                        "squared_error": None,
                    }
                else:
                    error = prediction - actual
                    values[target] = {
                        "available": True,
                        "prediction": prediction,
                        "actual": actual,
                        "error": error,
                        "squared_error": error * error,
                    }
            model_systemic[horizon] = values
        node_metrics[model] = model_node
        systemic_metrics[model] = model_systemic

    comparisons: dict[str, Any] = {}
    for name, (candidate, comparator) in COMPARISONS.items():
        if candidate not in node_metrics or comparator not in node_metrics:
            continue
        cells: dict[str, Any] = {}
        for horizon in eligible_horizons:
            candidate_metrics = node_metrics[candidate][horizon]["endpoint_return"]
            comparator_metrics = node_metrics[comparator][horizon]["endpoint_return"]
            if candidate_metrics["count"] != comparator_metrics["count"]:
                raise ValueError("prospective comparison target counts changed")
            deltas = {}
            for metric in ("pearson", "skill_vs_zero_mse", "sign_accuracy"):
                left = candidate_metrics[metric]
                right = comparator_metrics[metric]
                deltas[f"candidate_minus_comparator_{metric}"] = (
                    None if left is None or right is None else float(left - right)
                )
            cells[horizon] = {
                "candidate": candidate,
                "comparator": comparator,
                "count": candidate_metrics["count"],
                **deltas,
            }
        comparisons[name] = cells

    day_record = release.records[session]
    result = {
        "schema_version": 1,
        "role": "post_impact_prospective_prediction_reconciliation",
        "status": "pass",
        "session": session,
        "prediction_commit_id": record["commit_id"],
        "prediction_record_sha256": record["record_sha256"],
        "prediction_generated_at_utc": record["prediction_generated_at_utc"],
        "prediction_input_pins": dict(record["input_pins"]),
        "prediction_model_pins": {
            str(name): dict(value) for name, value in record["models"].items()
        },
        "prediction_causality": dict(record["causality"]),
        "decision_timestamp_utc_ns": int(decision_values[0]),
        "decision_row_index": decision_index,
        "eligible_horizons": eligible_horizons,
        "models": models,
        "node_metrics": node_metrics,
        "systemic_metrics": systemic_metrics,
        "primary_endpoint_comparisons": comparisons,
        "label_pins": {
            "day_release_manifest": file_sha256(release.manifest_path),
            "day_shard": str(day_record["sha256"]),
        },
        "counts_as_forward_evidence": True,
        "promotion_eligible_from_this_record_alone": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    result["reconciliation_content_sha256"] = canonical_sha256(result)
    return result


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root)
    records = read_prediction_ledger(
        Path(args.ledger), artifact_root=artifact_root
    )
    session = str(pd.Timestamp(args.session).date())
    selected = [
        record
        for record in records
        if record.get("source_mode") == "live_read_only"
        and str(record.get("session")) == session
    ]
    if not selected:
        raise ValueError("no live prospective records exist for the requested session")
    release = DayRelease(Path(args.day_release_dir), cache=False)
    if release.dates[-1] < session:
        raise ValueError("completed-session label release is not mature")
    reconciliations = [
        reconcile_prediction_record(
            record,
            artifact_root=artifact_root,
            release=release,
        )
        for record in selected
    ]
    summary = {
        "schema_version": 1,
        "role": "post_impact_prospective_session_reconciliation_audit",
        "status": "pass",
        "session": session,
        "prediction_ledger": file_sha256(Path(args.ledger)),
        "prediction_ledger_head_sha256": records[-1]["record_sha256"],
        "records_reconciled": len(reconciliations),
        "prediction_record_sha256": [
            row["prediction_record_sha256"] for row in reconciliations
        ],
        "reconciliation_content_sha256": [
            row["reconciliation_content_sha256"] for row in reconciliations
        ],
        "counts_as_forward_session": True,
        "minimum_forward_sessions": 20,
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
                    f"immutable reconciliation output already differs: {output}"
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
