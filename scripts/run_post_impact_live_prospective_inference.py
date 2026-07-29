from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.capture_kiwoom_live_minute_snapshot import (
    collector_code_provenance,
    completed_bar_mask,
)
from scripts.replay_post_impact_prospective_ledger import (
    _artifact_arrays,
    _device,
    _load_contract,
    _model_runtime,
    _predict_prefix,
    canonical_sha256,
    format_clock_hhmm,
)
from scripts.train_post_impact_reforecast import DayRelease, StaleCache
from stock_v2.intraday_trajectory import (
    build_intraday_trajectory_panel,
    build_ticker_intraday_trajectory,
)
from stock_v2.kiwoom_minute import KST, audit_kiwoom_minute_frame
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


SCIENTIFIC_MODELS = ("direct", "state", "latent", "latent_only_placebo")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen post-impact models and their controls on a verified live "
            "completed-bar snapshot, then append a zero-order prospective commit."
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


def _safe_child(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    if not str(value or "") or path.is_absolute() or ".." in path.parts:
        raise ValueError("live snapshot contains an unsafe payload path")
    resolved = (root.resolve() / path).resolve()
    resolved.relative_to(root.resolve())
    return resolved


def _read_snapshot(snapshot_dir: Path) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    manifest_path = snapshot_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_causality = {
        "common_cutoff_fixed_before_first_ticker_request",
        "only_completed_bars_retained",
        "in_progress_bar_excluded",
        "future_bars_absent",
        "labels_absent",
    }
    if manifest.get("role") != "kiwoom_live_completed_minute_snapshot":
        raise ValueError("unsupported live minute snapshot role")
    if manifest.get("live_orders_allowed") is not False:
        raise ValueError("live minute snapshot permits live orders")
    if manifest.get("broker_order_calls_executed") != 0:
        raise ValueError("live minute snapshot contains broker order calls")
    if manifest.get("errors"):
        raise ValueError("live minute snapshot contains ticker errors")
    if manifest.get("code_provenance") != collector_code_provenance():
        raise ValueError("live minute snapshot collector provenance changed")
    causality = manifest.get("causality")
    if not isinstance(causality, Mapping) or any(
        causality.get(name) is not True for name in required_causality
    ):
        raise ValueError("live minute snapshot causality claims are incomplete")
    records = list(manifest.get("records") or [])
    if len(records) != int(manifest.get("universe_tickers", -1)):
        raise ValueError("live minute snapshot record count mismatch")
    tickers = [str(record.get("ticker")) for record in records]
    if len(tickers) != len(set(tickers)):
        raise ValueError("live minute snapshot contains duplicate tickers")
    session = pd.Timestamp(manifest["session"]).normalize()
    cutoff = pd.Timestamp(manifest["common_cutoff_kst"])
    if cutoff.tzinfo is None:
        raise ValueError("live minute snapshot cutoff is timezone naive")
    interval = int(manifest["interval_minutes"])
    semantics = str(manifest["timestamp_semantics"])
    frames: dict[str, pd.DataFrame] = {}
    output_records: list[dict[str, Any]] = []
    raw_records: dict[str, dict[str, Any]] = {}
    for record in records:
        status = str(record.get("status"))
        if status != "ok":
            if status not in {"empty", "outside_lifecycle"}:
                raise ValueError(f"unsupported live snapshot status: {status}")
            if status == "empty":
                raw_records[str(record["ticker"])] = record
            continue
        path = _safe_child(snapshot_dir, record.get("path"))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"live snapshot payload is missing: {path}")
        if path.stat().st_size != int(record["bytes"]):
            raise ValueError(f"live snapshot payload size changed: {path}")
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"live snapshot payload hash changed: {path}")
        frame = pd.read_parquet(path)
        frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
        if frame.index.tz is None:
            raise ValueError(f"live snapshot payload is timezone naive: {path}")
        frame.index.name = "Timestamp"
        audit_kiwoom_minute_frame(frame, regular_session_only=True)
        local_dates = frame.index.tz_convert(KST).tz_localize(None).normalize()
        if not np.asarray(local_dates == session, dtype=bool).all():
            raise ValueError(f"live snapshot payload crosses sessions: {path}")
        if not completed_bar_mask(
            frame.index,
            cutoff=cutoff,
            interval_minutes=interval,
            timestamp_semantics=semantics,
        ).all():
            raise ValueError(f"live snapshot payload contains an incomplete bar: {path}")
        frames[str(record["ticker"])] = frame
        output_records.append(record)
        raw_records[str(record["ticker"])] = record
    if len(frames) != int(manifest.get("populated_tickers", -1)):
        raise ValueError("live snapshot populated ticker count mismatch")
    if canonical_sha256(output_records) != manifest.get("outputs_sha256"):
        raise ValueError("live snapshot output-record fingerprint changed")
    raw_tickers = {path.parent.name for path in (snapshot_dir / "raw").glob("*/*.json.gz")}
    if raw_tickers != set(raw_records):
        raise ValueError("live snapshot raw response ticker set mismatch")
    for ticker in sorted(raw_tickers):
        raw_paths = sorted((snapshot_dir / "raw" / ticker).glob("*.json.gz"))
        record = raw_records[ticker]
        if len(raw_paths) != len(record.get("raw_page_sha256") or []):
            raise ValueError(f"live snapshot raw page count mismatch: {ticker}")
        for page_path, expected in zip(raw_paths, record["raw_page_sha256"]):
            if not page_path.is_file() or page_path.is_symlink():
                raise ValueError(f"live snapshot raw page is not immutable: {ticker}")
            with gzip.open(page_path, "rt", encoding="utf-8") as handle:
                envelope = json.load(handle)
            if canonical_sha256(envelope) != expected:
                raise ValueError(f"live snapshot raw envelope changed: {ticker}")
            if (
                envelope.get("ticker") != ticker
                or envelope.get("common_cutoff_kst") != manifest["common_cutoff_kst"]
                or envelope.get("api_id") != "ka10080"
            ):
                raise ValueError(f"live snapshot raw envelope contract mismatch: {ticker}")
    return manifest, frames


def _lifecycle_closes(
    release_dir: Path,
    *,
    context_date: str,
    tickers: Sequence[str],
) -> tuple[dict[str, float], str]:
    manifest_path = release_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("end") != context_date or int(manifest.get("output_tickers", -1)) != 500:
        raise ValueError("lifecycle release does not end at the prospective context")
    rows = {str(row["ticker"]): row for row in manifest.get("outputs") or []}
    if len(rows) != 500:
        raise ValueError("lifecycle release ticker count mismatch")
    closes: dict[str, float] = {}
    for ticker in tickers:
        row = rows.get(str(ticker))
        if row is None:
            continue
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = ROOT / path
        try:
            path.resolve().relative_to(release_dir.resolve())
        except ValueError as exc:
            raise ValueError(f"lifecycle source escapes its release: {ticker}") from exc
        if path.is_symlink():
            raise ValueError(f"lifecycle source is a symlink: {ticker}")
        if not path.is_file() or file_sha256(path) != row["sha256"]:
            raise ValueError(f"lifecycle source changed: {ticker}")
        frame = pd.read_csv(path, usecols=["Date", "RawClose"])
        selected = frame.loc[frame["Date"].astype(str) == context_date, "RawClose"]
        if len(selected) != 1:
            continue
        close = float(selected.iloc[0])
        if np.isfinite(close) and close > 0.0:
            closes[str(ticker)] = close
    return closes, file_sha256(manifest_path)


class _LiveRelease:
    def __init__(self, historical: DayRelease, date: str, day: dict[str, np.ndarray]):
        self.tickers = historical.tickers
        self.feature_names = historical.feature_names
        self.horizon_labels = historical.horizon_labels
        self.target_names = historical.target_names
        self.systemic_target_names = historical.systemic_target_names
        self.records = {date: {}}
        self._date = date
        self._day = day

    def load(self, date: str) -> dict[str, np.ndarray]:
        if str(date) != self._date:
            raise KeyError(date)
        return self._day


def _live_day(panel: Any, historical: DayRelease) -> dict[str, np.ndarray]:
    times = len(panel.timestamps)
    nodes = len(panel.tickers)
    horizons = len(historical.horizon_labels)
    targets = len(historical.target_names)
    systemic = len(historical.systemic_target_names)
    return {
        "timestamps_utc_ns": panel.timestamps.tz_convert("UTC").asi8,
        "node_values": np.asarray(panel.values, dtype=np.float32),
        "node_available": np.asarray(panel.available, dtype=np.uint8),
        "decision_price": np.asarray(panel.decision_price, dtype=np.float32),
        "targets": np.full(
            (times, nodes, horizons, targets), np.nan, dtype=np.float32
        ),
        "target_available": np.zeros(
            (times, nodes, horizons, targets), dtype=np.uint8
        ),
        "systemic_targets": np.full(
            (times, horizons, systemic), np.nan, dtype=np.float32
        ),
        "systemic_available": np.zeros(
            (times, horizons, systemic), dtype=np.uint8
        ),
    }


def prospective_horizon_eligibility(
    *,
    decision_timestamp_utc_ns: int,
    generated_at_utc: datetime,
    horizon_labels: Sequence[str],
    session: str,
) -> dict[str, bool]:
    generated = pd.Timestamp(generated_at_utc)
    if generated.tzinfo is None:
        raise ValueError("prediction generation time must be timezone aware")
    decision = pd.Timestamp(int(decision_timestamp_utc_ns), unit="ns", tz="UTC")
    close = pd.Timestamp(session).tz_localize(KST) + pd.Timedelta(
        15 * 60 + 30, unit="minute"
    )
    result: dict[str, bool] = {}
    for label in horizon_labels:
        if str(label) == "close":
            maturity = close
        elif str(label).endswith("m") and str(label)[:-1].isdigit():
            maturity = decision + pd.Timedelta(int(str(label)[:-1]), unit="minute")
        else:
            raise ValueError(f"unsupported prospective horizon label: {label}")
        result[str(label)] = bool(generated < maturity)
    return result


def main() -> None:
    args = parse_args()
    if int(args.history_context_sessions) < 20:
        raise ValueError("live inference requires at least 20 history context sessions")
    contract_path = Path(args.contract)
    contract = _load_contract(contract_path)
    snapshot_dir = Path(args.snapshot_dir)
    snapshot, live_frames = _read_snapshot(snapshot_dir)
    session = str(snapshot["session"])
    cutoff = pd.Timestamp(snapshot["common_cutoff_kst"]).tz_convert(KST)
    historical = DayRelease(Path(args.historical_day_release_dir), cache=False)
    context_date = str(historical.dates[-1])
    if context_date >= session:
        raise ValueError("historical day release is not strictly prior to the live session")
    stale = StaleCache(Path(args.prospective_stale_cache_dir))
    stale.align_tickers(historical.tickers)
    if stale.dates != (session,) or stale.context_dates != (context_date,):
        raise ValueError("prospective stale cache date contract mismatch")
    stale_prospective = stale.manifest.get("prospective_target") or {}
    if not (
        stale_prospective.get("enabled") is True
        and stale_prospective.get("target_observations_injected") is False
    ):
        raise ValueError("stale cache is not a label-free prospective release")
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
        raise ValueError("live trajectory ticker order changed")
    if tuple(panel.feature_names) != tuple(historical.feature_names):
        raise ValueError("live trajectory feature schema changed")
    if panel.timestamps[-1] != cutoff:
        raise ValueError("live trajectory endpoint does not match the snapshot cutoff")
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
            f"only {latest_nodes} nodes at live cutoff; require {args.minimum_latest_nodes}"
        )
    day = _live_day(panel, historical)
    release = _LiveRelease(historical, session, day)
    device = _device(args.device)
    runtimes = {
        name: _model_runtime(
            name,
            contract["models"][name],
            release,
            stale,
            session,
            device,
        )
        for name in SCIENTIFIC_MODELS
    }
    prefix = len(panel.timestamps)
    model_names: list[str] = []
    node_predictions: list[np.ndarray] = []
    systemic_predictions: list[np.ndarray] = []
    model_records: dict[str, Any] = {}
    latencies: dict[str, float] = {}
    for name in SCIENTIFIC_MODELS:
        node, systemic, latency_ms, input_sha = _predict_prefix(
            runtimes[name], prefix=prefix, device=device
        )
        model_names.append(name)
        node_predictions.append(node)
        systemic_predictions.append(systemic)
        model_records[name] = {
            "checkpoint_sha256": runtimes[name]["checkpoint_sha256"],
            "model_input_sha256": input_sha,
        }
        latencies[name] = latency_ms
    decision_timestamp = int(panel.timestamps[-1].tz_convert("UTC").value)
    arrays = _artifact_arrays(
        historical,
        model_names,
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
        }
    )
    artifact_root = Path(args.artifact_root)
    clock = cutoff.hour * 60 + cutoff.minute
    clock_hhmm = format_clock_hhmm(clock)
    artifact = write_immutable_prediction_artifact(
        Path("predictions") / session / f"{clock_hhmm}_scientific.npz",
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
    history_records = [historical.records[date] for date in history_dates]
    payload = {
        "schema_version": 1,
        "role": LEDGER_ROLE,
        "commit_id": f"{session}|{clock_hhmm}|post_impact_scientific_live_v1",
        "session": session,
        "decision_timestamp_utc_ns": decision_timestamp,
        "decision_clock_minute_kst": int(clock),
        "prefix_timestamps": int(prefix),
        "source_mode": "live_read_only",
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
            "forward_contract": file_sha256(contract_path),
            "live_snapshot_manifest": file_sha256(snapshot_dir / "manifest.json"),
            "live_snapshot_collector_source_tree": snapshot["code_provenance"][
                "source_tree_sha256"
            ],
            "historical_day_release_manifest": file_sha256(
                historical.manifest_path
            ),
            "historical_context_day_records": canonical_sha256(history_records),
            "prospective_stale_cache_manifest": file_sha256(stale.manifest_path),
            "lifecycle_release_manifest": lifecycle_manifest_sha,
            "live_inference_source": file_sha256(Path(__file__)),
            "live_feature_source": file_sha256(
                ROOT / "stock_v2/live_post_impact_features.py"
            ),
            "ledger_module": file_sha256(ROOT / "stock_v2/prospective_ledger.py"),
        },
        "models": model_records,
        "prediction_artifact": artifact,
        "prediction_array_sha256": prediction_array_fingerprint(arrays),
        "live_input": {
            "snapshot_cutoff_kst": snapshot["common_cutoff_kst"],
            "snapshot_populated_tickers": int(snapshot["populated_tickers"]),
            "trajectory_tickers": len(trajectories),
            "latest_nodes": latest_nodes,
            "shock_context": shock_diagnostics,
        },
        "causality": {
            "completed_bars_only": True,
            "future_intraday_rows_absent_from_model_input": True,
            "labels_absent_from_model_input": True,
            "model_eval_mode": True,
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
        "role": "post_impact_live_prospective_prediction_audit",
        "status": "pass",
        "session": session,
        "clock_hhmm": clock_hhmm,
        "record_sha256": record["record_sha256"],
        "record_appended": appended,
        "ledger": ledger_summary(records),
        "models": model_names,
        "inference_ms": latencies,
        "latest_nodes": latest_nodes,
        "shock_context": shock_diagnostics,
        "prospective_horizon_eligibility": horizon_eligible,
        "forward_reconciliation_status": "pending_labels",
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    summary["audit_content_sha256"] = canonical_sha256(summary)
    output = Path(args.summary_output)
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
                "latest_nodes": latest_nodes,
                "eligible_horizons": summary["prospective_horizon_eligibility"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
