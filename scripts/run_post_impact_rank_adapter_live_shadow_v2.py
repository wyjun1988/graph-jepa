from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.replay_post_impact_prospective_ledger import (
    _artifact_arrays,
    _device,
    _model_runtime,
    _predict_prefix,
    canonical_sha256,
    format_clock_hhmm,
)
from scripts.run_post_impact_live_prospective_inference import (
    _LiveRelease,
    _lifecycle_closes,
    _live_day,
    _read_snapshot,
    prospective_horizon_eligibility,
)
from scripts.train_post_impact_reforecast import DayRelease, StaleCache
from stock_v2.intraday_trajectory import (
    build_intraday_trajectory_panel,
    build_ticker_intraday_trajectory,
)
from stock_v2.kiwoom_minute import KST
from stock_v2.live_post_impact_features import (
    apply_historical_same_clock_shocks,
    synthetic_prior_close_frame,
)
from stock_v2.prospective_ledger import (
    LEDGER_ROLE,
    append_prediction_commit,
    file_sha256,
    ledger_summary,
    prediction_array_fingerprint,
    read_prediction_ledger,
    write_immutable_prediction_artifact,
)


CONTRACT_ROLE = "post_impact_rank_adapter_live_shadow_contract_v2"
EVENT_EVIDENCE_CONTRACT = "causal_observed_surprise_current_and_recent_30m_v1"
RANK_MODELS = ("baseline", "aligned", "own_permuted")
EXPECTED_MODEL_RUNTIME = {
    "baseline": ("none", "shared", False),
    "aligned": ("surprise_disabled", "long_horizon_residual", True),
    "own_permuted": (
        "surprise_own_permuted",
        "long_horizon_residual",
        True,
    ),
}
RUNTIME_INPUT_ARGS = {
    "historical_day_release": "historical_day_release_dir",
    "prospective_stale_cache": "prospective_stale_cache_dir",
    "lifecycle_release": "lifecycle_release_dir",
}


def _clock_minute(value: object) -> int:
    text = str(value)
    try:
        hour_text, minute_text = text.split(":", 1)
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid rank-adapter clock: {text}") from exc
    if (
        len(hour_text) != 2
        or len(minute_text) != 2
        or hour < 0
        or hour > 23
        or minute < 0
        or minute > 59
    ):
        raise ValueError(f"invalid rank-adapter clock: {text}")
    return hour * 60 + minute


def prospective_scope(contract: Mapping[str, Any]) -> tuple[str, tuple[int, ...]]:
    first_session = str(contract.get("first_primary_session") or "")
    try:
        first_session = str(pd.Timestamp(first_session).date())
    except (TypeError, ValueError) as exc:
        raise ValueError("rank-adapter contract lacks a valid first primary session") from exc
    evidence = contract.get("prospective_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("rank-adapter contract lacks prospective evidence scope")
    raw_clocks = evidence.get("primary_clocks_kst")
    if not isinstance(raw_clocks, list) or not raw_clocks:
        raise ValueError("rank-adapter contract lacks primary clocks")
    clocks = tuple(_clock_minute(value) for value in raw_clocks)
    if tuple(sorted(set(clocks))) != clocks:
        raise ValueError("rank-adapter primary clocks must be unique and sorted")
    raw_buckets = evidence.get("validated_clock_buckets_kst")
    if not isinstance(raw_buckets, list) or not raw_buckets:
        raise ValueError("rank-adapter contract lacks validated clock buckets")
    buckets: list[tuple[int, int]] = []
    for record in raw_buckets:
        if not isinstance(record, Mapping):
            raise ValueError("rank-adapter clock bucket is invalid")
        start = _clock_minute(record.get("start"))
        end = _clock_minute(record.get("end"))
        if start > end:
            raise ValueError("rank-adapter clock bucket is reversed")
        buckets.append((start, end))
    if any(not any(start <= clock <= end for start, end in buckets) for clock in clocks):
        raise ValueError("rank-adapter primary clock is outside validated buckets")
    return first_session, clocks


def diagnostic_scope(contract: Mapping[str, Any]) -> tuple[str, tuple[int, ...]] | None:
    scope = contract.get("diagnostic_scope")
    if scope is None:
        return None
    if not isinstance(scope, Mapping) or scope.get("enabled") is not True:
        raise ValueError("rank-adapter diagnostic scope is invalid")
    try:
        session = str(pd.Timestamp(str(scope.get("session"))).date())
    except (TypeError, ValueError) as exc:
        raise ValueError("rank-adapter diagnostic session is invalid") from exc
    raw_clocks = scope.get("clocks_kst")
    if not isinstance(raw_clocks, list) or not raw_clocks:
        raise ValueError("rank-adapter diagnostic clocks are missing")
    clocks = tuple(_clock_minute(value) for value in raw_clocks)
    if tuple(sorted(set(clocks))) != clocks:
        raise ValueError("rank-adapter diagnostic clocks must be unique and sorted")
    if (
        scope.get("counts_as_primary_forward_evidence") is not False
        or scope.get("promotion_eligible") is not False
        or scope.get("live_orders_allowed") is not False
    ):
        raise ValueError("rank-adapter diagnostic scope is unsafe")
    return session, clocks


def validate_prospective_scope(
    contract: Mapping[str, Any], *, session: str, clock_minute: int
) -> str:
    first_session, primary_clocks = prospective_scope(contract)
    current_session = str(pd.Timestamp(session).date())
    if current_session >= first_session and int(clock_minute) in primary_clocks:
        return "primary_prospective"
    diagnostic = diagnostic_scope(contract)
    if diagnostic is not None:
        diagnostic_session, diagnostic_clocks = diagnostic
        if current_session == diagnostic_session:
            if int(clock_minute) in diagnostic_clocks:
                return "diagnostic_out_of_scope"
            raise ValueError(
                "rank-adapter clock is outside the frozen diagnostic scope"
            )
    if current_session < first_session:
        raise ValueError("rank-adapter session predates the frozen primary scope")
    raise ValueError("rank-adapter clock is outside the frozen prospective scope")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Commit a zero-order rank-adapter shadow prediction from an immutable "
            "completed-bar snapshot."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--snapshot-dir", required=True)
    parser.add_argument("--historical-day-release-dir", required=True)
    parser.add_argument("--prospective-stale-cache-dir", required=True)
    parser.add_argument("--lifecycle-release-dir", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--history-context-sessions", type=int, default=100)
    parser.add_argument("--minimum-latest-nodes", type=int, default=400)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def load_rank_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema_version") != 2 or contract.get("role") != CONTRACT_ROLE:
        raise ValueError("invalid rank-adapter live shadow contract")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("rank-adapter shadow contract permits live orders")
    if contract.get("broker_order_calls_allowed") is not False:
        raise ValueError("rank-adapter shadow contract permits broker calls")
    if contract.get("promotion_eligible") is not False:
        raise ValueError("rank-adapter shadow contract permits promotion")
    if tuple(contract.get("model_order") or ()) != RANK_MODELS:
        raise ValueError("rank-adapter shadow model order changed")
    prospective_scope(contract)
    diagnostic_scope(contract)
    event_evidence = contract.get("event_evidence")
    if not isinstance(event_evidence, Mapping) or (
        event_evidence.get("contract") != EVENT_EVIDENCE_CONTRACT
        or int(event_evidence.get("lookback_minutes", -1)) != 30
        or event_evidence.get("point_in_time_observed_only") is not True
        or event_evidence.get("future_realized_labels_used_for_selection") is not False
        or event_evidence.get("checkpoint_calibration_pinned") is not True
        or event_evidence.get("artifact_flags_required") is not True
    ):
        raise ValueError("rank-adapter event evidence contract changed")
    models = contract.get("models")
    if not isinstance(models, Mapping) or set(models) != set(RANK_MODELS):
        raise ValueError("rank-adapter shadow model set changed")

    source_pins = contract.get("source_pins")
    if not isinstance(source_pins, Mapping) or not source_pins:
        raise ValueError("rank-adapter shadow contract lacks source pins")
    for relative, expected in source_pins.items():
        source = _resolve(relative)
        if not source.is_file() or file_sha256(source) != str(expected):
            raise ValueError(f"rank-adapter shadow source pin changed: {relative}")
    for label in ("selection_audit", "latency_qualification"):
        record = contract.get(label)
        if not isinstance(record, Mapping):
            raise ValueError(f"rank-adapter shadow contract lacks {label}")
        artifact = _resolve(record.get("path"))
        if not artifact.is_file() or file_sha256(artifact) != record.get("sha256"):
            raise ValueError(f"rank-adapter shadow {label} changed")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if payload.get("live_orders_allowed") is not False:
            raise ValueError(f"rank-adapter shadow {label} is unsafe")
        if label == "selection_audit" and payload.get("selected_candidate") != "aligned":
            raise ValueError("rank-adapter selection audit did not select aligned")
        if label == "latency_qualification" and payload.get("status") != "pass":
            raise ValueError("rank-adapter latency qualification did not pass")

    for name in RANK_MODELS:
        spec = models[name]
        expected_mode, expected_fusion, expected_frozen = EXPECTED_MODEL_RUNTIME[name]
        if (
            spec.get("graph_message_mode") != expected_mode
            or spec.get("graph_message_fusion") != expected_fusion
            or bool(spec.get("freeze_base_for_message_adapter")) is not expected_frozen
        ):
            raise ValueError(f"rank-adapter runtime contract changed: {name}")
        for kind in ("checkpoint", "summary"):
            artifact = _resolve(spec.get(kind))
            if not artifact.is_file() or file_sha256(artifact) != spec.get(
                f"{kind}_sha256"
            ):
                raise ValueError(f"rank-adapter {name} {kind} changed")
        summary = json.loads(_resolve(spec["summary"]).read_text(encoding="utf-8"))
        if (
            summary.get("live_orders_allowed") is not False
            or summary.get("promotion_eligible") is not False
            or summary.get("evaluation_scope") != "validation_only"
            or summary.get("test_evaluated") is not False
            or summary.get("test") is not None
        ):
            raise ValueError(f"rank-adapter {name} summary is unsafe")
    return contract


def validate_runtime_inputs(
    contract: Mapping[str, Any], args: argparse.Namespace
) -> None:
    records = contract.get("runtime_inputs")
    if not isinstance(records, Mapping) or set(records) != set(RUNTIME_INPUT_ARGS):
        raise ValueError("rank-adapter runtime input contract changed")
    for name, argument in RUNTIME_INPUT_ARGS.items():
        record = records[name]
        expected = _resolve(record.get("path")).resolve()
        actual = _resolve(getattr(args, argument)).resolve()
        if actual != expected or not actual.is_dir():
            raise ValueError(f"rank-adapter runtime path changed: {name}")
        manifest = actual / "manifest.json"
        if not manifest.is_file() or file_sha256(manifest) != record.get(
            "manifest_sha256"
        ):
            raise ValueError(f"rank-adapter runtime manifest changed: {name}")


def _runtime_event_scope(
    runtimes: Mapping[str, Mapping[str, Any]], *, prefix: int
) -> dict[str, Any]:
    flags: dict[str, tuple[bool, bool]] = {}
    for name in RANK_MODELS:
        batch = runtimes[name]["batch"]
        values: list[bool] = []
        for field in ("observed_surprise", "post_shock_recent"):
            value = batch[field]
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            array = np.asarray(value, dtype=bool)
            if array.ndim != 2 or array.shape[0] != 1 or not 0 < prefix <= array.shape[1]:
                raise ValueError(f"rank-adapter event tensor changed: {name}/{field}")
            values.append(bool(array[0, prefix - 1]))
        flags[name] = (values[0], values[1])
    if len(set(flags.values())) != 1:
        raise ValueError("rank-adapter event calibration differs across models")
    current, recent = flags["baseline"]
    if current and not recent:
        raise ValueError("current observed surprise is absent from the recent window")
    return {
        "contract": EVENT_EVIDENCE_CONTRACT,
        "lookback_minutes": 30,
        "observed_surprise_current": current,
        "observed_surprise_recent_30m": recent,
        "point_in_time_observed_only": True,
        "future_realized_labels_used_for_selection": False,
        "checkpoint_calibration_pinned": True,
    }


def _prepare_live_release(
    args: argparse.Namespace,
) -> tuple[
    dict[str, Any],
    DayRelease,
    StaleCache,
    _LiveRelease,
    Any,
    dict[str, Any],
    str,
]:
    snapshot, live_frames = _read_snapshot(Path(args.snapshot_dir))
    session = str(snapshot["session"])
    cutoff = pd.Timestamp(snapshot["common_cutoff_kst"]).tz_convert(KST)
    historical = DayRelease(Path(args.historical_day_release_dir), cache=False)
    context_date = str(historical.dates[-1])
    if context_date >= session:
        raise ValueError("historical day release is not strictly prior to the session")
    stale = StaleCache(Path(args.prospective_stale_cache_dir))
    stale.align_tickers(historical.tickers)
    if stale.dates != (session,) or stale.context_dates != (context_date,):
        raise ValueError("rank-adapter prospective stale cache date mismatch")
    prospective = stale.manifest.get("prospective_target") or {}
    if not (
        prospective.get("enabled") is True
        and prospective.get("target_observations_injected") is False
    ):
        raise ValueError("rank-adapter stale cache is not label-free")

    closes, lifecycle_manifest_sha = _lifecycle_closes(
        Path(args.lifecycle_release_dir),
        context_date=context_date,
        tickers=historical.tickers,
    )
    trajectories = {}
    for ticker, frame in live_frames.items():
        close = closes.get(ticker)
        if close is None:
            continue
        combined = pd.concat(
            [synthetic_prior_close_frame(context_date, close), frame]
        ).sort_index()
        trajectory = build_ticker_intraday_trajectory(
            combined,
            interval_minutes=int(snapshot["interval_minutes"]),
            timestamp_semantics=str(snapshot["timestamp_semantics"]),
            horizons_minutes=(5, 15, 30, 60),
            decision_start="09:15",
            decision_end="15:15",
            rolling_window=20,
            min_history=10,
        )
        if len(trajectory.timestamps):
            trajectories[ticker] = trajectory
    panel = build_intraday_trajectory_panel(
        trajectories,
        tickers=historical.tickers,
    )
    if tuple(panel.tickers) != tuple(historical.tickers):
        raise ValueError("rank-adapter live ticker order changed")
    if tuple(panel.feature_names) != tuple(historical.feature_names):
        raise ValueError("rank-adapter live feature schema changed")
    if panel.timestamps[-1] != cutoff:
        raise ValueError("rank-adapter live endpoint does not match snapshot cutoff")
    history_dates = [
        date for date in historical.dates if str(date) < session
    ][-int(args.history_context_sessions) :]
    panel, shock_diagnostics = apply_historical_same_clock_shocks(
        panel,
        historical,
        history_dates,
        rolling_window=20,
        min_history=10,
    )
    latest_nodes = int(
        np.count_nonzero(
            np.isfinite(panel.decision_price[-1]) & (panel.decision_price[-1] > 0.0)
        )
    )
    if latest_nodes < int(args.minimum_latest_nodes):
        raise ValueError(
            f"only {latest_nodes} rank-adapter nodes; require {args.minimum_latest_nodes}"
        )
    day = _live_day(panel, historical)
    release = _LiveRelease(historical, session, day)
    return (
        snapshot,
        historical,
        stale,
        release,
        panel,
        shock_diagnostics,
        lifecycle_manifest_sha,
    )


def main() -> None:
    args = parse_args()
    if int(args.history_context_sessions) < 20:
        raise ValueError("rank-adapter shadow requires at least 20 context sessions")
    contract_path = Path(args.contract)
    contract = load_rank_contract(contract_path)
    validate_runtime_inputs(contract, args)
    (
        snapshot,
        historical,
        stale,
        release,
        panel,
        shock_diagnostics,
        lifecycle_manifest_sha,
    ) = _prepare_live_release(args)
    session = str(snapshot["session"])
    cutoff = pd.Timestamp(snapshot["common_cutoff_kst"]).tz_convert(KST)
    clock = cutoff.hour * 60 + cutoff.minute
    evidence_class = validate_prospective_scope(
        contract, session=session, clock_minute=clock
    )
    device = _device(args.device)

    runtimes: dict[str, dict[str, Any]] = {}
    for name in RANK_MODELS:
        spec = dict(contract["models"][name])
        spec["checkpoint"] = str(_resolve(spec["checkpoint"]))
        runtime = dict(
            _model_runtime("latent", spec, release, stale, session, device)
        )
        checkpoint_args = runtime["checkpoint_args"]
        expected_mode, expected_fusion, expected_frozen = EXPECTED_MODEL_RUNTIME[name]
        if (
            str(getattr(checkpoint_args, "graph_message_mode", "none"))
            != expected_mode
            or str(getattr(checkpoint_args, "graph_message_fusion", "shared"))
            != expected_fusion
            or bool(
                getattr(checkpoint_args, "freeze_base_for_message_adapter", False)
            )
            is not expected_frozen
        ):
            raise ValueError(f"loaded rank-adapter runtime changed: {name}")
        runtime["name"] = name
        runtimes[name] = runtime

    prefix = len(panel.timestamps)
    event_scope = _runtime_event_scope(runtimes, prefix=prefix)
    node_predictions: list[np.ndarray] = []
    systemic_predictions: list[np.ndarray] = []
    model_records: dict[str, Any] = {}
    latencies: dict[str, float] = {}
    for name in RANK_MODELS:
        node, systemic, latency_ms, input_sha = _predict_prefix(
            runtimes[name], prefix=prefix, device=device
        )
        node_predictions.append(node)
        systemic_predictions.append(systemic)
        spec = contract["models"][name]
        model_records[name] = {
            "checkpoint_sha256": runtimes[name]["checkpoint_sha256"],
            "model_input_sha256": input_sha,
            "graph_message_mode": spec["graph_message_mode"],
            "graph_message_fusion": spec["graph_message_fusion"],
        }
        latencies[name] = latency_ms

    decision_timestamp = int(panel.timestamps[-1].tz_convert("UTC").value)
    arrays = _artifact_arrays(
        historical,
        RANK_MODELS,
        node_predictions,
        systemic_predictions,
        decision_timestamp_utc_ns=decision_timestamp,
        prefix=prefix,
    )
    arrays.update(
        {
            "latest_decision_price": np.asarray(
                panel.decision_price[-1], dtype=np.float32
            ),
            "latest_node_available": np.asarray(
                panel.available[-1], dtype=np.uint8
            ),
            "observed_surprise_current": np.asarray(
                [event_scope["observed_surprise_current"]], dtype=np.uint8
            ),
            "observed_surprise_recent_30m": np.asarray(
                [event_scope["observed_surprise_recent_30m"]], dtype=np.uint8
            ),
        }
    )
    clock_hhmm = format_clock_hhmm(clock)
    artifact_root = Path(args.artifact_root)
    artifact = write_immutable_prediction_artifact(
        Path("predictions") / session / f"{clock_hhmm}_rank_adapter_v2.npz",
        arrays,
        artifact_root=artifact_root,
    )
    generated_at = datetime.now(tz=timezone.utc)
    horizon_eligible = prospective_horizon_eligibility(
        decision_timestamp_utc_ns=decision_timestamp,
        generated_at_utc=generated_at,
        horizon_labels=historical.horizon_labels,
        session=session,
    )
    payload = {
        "schema_version": 1,
        "role": LEDGER_ROLE,
        "commit_id": f"{session}|{clock_hhmm}|post_impact_rank_adapter_live_v2",
        "session": session,
        "decision_timestamp_utc_ns": decision_timestamp,
        "decision_clock_minute_kst": int(clock),
        "prefix_timestamps": int(prefix),
        "source_mode": "live_read_only",
        "prospective_evidence_class": evidence_class,
        "counts_as_primary_forward_evidence": evidence_class == "primary_prospective",
        "prediction_generated_at_utc": generated_at.isoformat(),
        "prospective_horizon_eligibility": horizon_eligible,
        "forward_evidence": {
            "reconciliation_status": "pending_labels",
            "eligible_horizons": [
                name for name, eligible in horizon_eligible.items() if eligible
            ],
            "ineligible_horizons": [
                name for name, eligible in horizon_eligible.items() if not eligible
            ],
        },
        "input_pins": {
            "rank_shadow_contract": file_sha256(contract_path),
            "selection_audit": contract["selection_audit"]["sha256"],
            "latency_qualification": contract["latency_qualification"]["sha256"],
            "live_snapshot_manifest": file_sha256(
                Path(args.snapshot_dir) / "manifest.json"
            ),
            "historical_day_release_manifest": file_sha256(
                historical.manifest_path
            ),
            "prospective_stale_cache_manifest": file_sha256(stale.manifest_path),
            "lifecycle_release_manifest": lifecycle_manifest_sha,
            "rank_live_inference_source": file_sha256(Path(__file__)),
        },
        "models": model_records,
        "prediction_artifact": artifact,
        "prediction_array_sha256": prediction_array_fingerprint(arrays),
        "live_input": {
            "snapshot_cutoff_kst": snapshot["common_cutoff_kst"],
            "snapshot_populated_tickers": int(snapshot["populated_tickers"]),
            "latest_nodes": int(
                np.count_nonzero(
                    np.isfinite(panel.decision_price[-1])
                    & (panel.decision_price[-1] > 0.0)
                )
            ),
            "shock_context": shock_diagnostics,
        },
        "event_scope": event_scope,
        "causality": {
            "completed_bars_only": True,
            "future_intraday_rows_absent_from_model_input": True,
            "labels_absent_from_model_input": True,
            "model_eval_mode": True,
            "event_selection_uses_observed_inputs_only": True,
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    ledger_path = Path(args.ledger)
    record, appended = append_prediction_commit(
        ledger_path,
        payload,
        artifact_root=artifact_root,
    )
    records = read_prediction_ledger(ledger_path, artifact_root=artifact_root)
    summary = {
        "schema_version": 1,
        "role": "post_impact_rank_adapter_live_shadow_v2_audit",
        "status": "pass",
        "session": session,
        "clock_hhmm": clock_hhmm,
        "record_sha256": record["record_sha256"],
        "record_appended": appended,
        "ledger": ledger_summary(records),
        "models": list(RANK_MODELS),
        "inference_ms": latencies,
        "latest_nodes": payload["live_input"]["latest_nodes"],
        "shock_context": shock_diagnostics,
        "event_scope": event_scope,
        "prospective_horizon_eligibility": horizon_eligible,
        "forward_reconciliation_status": "pending_labels",
        "prospective_evidence_class": evidence_class,
        "counts_as_primary_forward_evidence": evidence_class == "primary_prospective",
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    summary["audit_content_sha256"] = canonical_sha256(summary)
    output = Path(args.summary_output)
    if output.exists():
        raise ValueError(f"refusing to overwrite rank-adapter summary: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(
        json.dumps(
            {
                "status": "pass",
                "commit_id": record["commit_id"],
                "latest_nodes": summary["latest_nodes"],
                "models": summary["models"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
