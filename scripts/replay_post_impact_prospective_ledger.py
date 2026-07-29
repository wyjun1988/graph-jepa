from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_post_impact_reforecast_latency import (
    _input_tensors,
    parse_clocks,
    prefix_indices,
    synchronize,
)
from scripts.evaluate_post_impact_adaptive_events import _calibration, _scaler
from scripts.train_post_impact_reforecast import (
    DayRelease,
    StaleCache,
    _graph_message_feature_dim,
    _graph_message_feature_names,
    _pad_batch,
    _resolved_daily_context_placebo_mode,
)
from stock_v2.post_impact_reforecast import CausalPostImpactReforecast
from stock_v2.prospective_ledger import (
    LEDGER_ROLE,
    append_prediction_commit,
    file_sha256,
    ledger_summary,
    prediction_array_fingerprint,
    read_prediction_ledger,
    write_immutable_prediction_artifact,
)
from stock_v2.surprise_reforecast import SURPRISE_STATISTIC_NAMES


OPERATIONAL_MODELS = ("direct", "latent")
MODEL_RUNTIME_CONTRACTS = {
    "direct": ("direct", "none"),
    "state": ("state", "none"),
    "latent": ("latent", "none"),
    "latent_only_placebo": ("latent", "latent_only"),
}
DEFAULT_CLOCKS = (9 * 60 + 15, 11 * 60, 13 * 60 + 30, 15 * 60, 15 * 60 + 15)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay causal intraday prefixes through frozen post-impact models and "
            "commit their predictions to a hash-chained, zero-order ledger."
        )
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--day-release-dir", required=True)
    parser.add_argument("--stale-cache-dir", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--clocks", default=",".join(str(value) for value in DEFAULT_CLOCKS)
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--summary-output", required=True)
    return parser.parse_args()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_clock_hhmm(clock_minute: int) -> str:
    clock = int(clock_minute)
    if clock < 0 or clock >= 24 * 60:
        raise ValueError("clock minute is outside one calendar day")
    return f"{clock // 60:02d}{clock % 60:02d}"


def causal_model_input_arrays(
    batch: Mapping[str, np.ndarray],
    *,
    prefix: int,
    variant: str,
) -> dict[str, np.ndarray]:
    length = int(prefix)
    if length <= 0 or length > int(batch["node_values"].shape[1]):
        raise ValueError("causal model input prefix is outside the batch")
    arrays = {
        "node_values": np.asarray(batch["node_values"][:, :length]),
        "node_available": np.asarray(batch["node_available"][:, :length]),
        "surprise": np.asarray(batch["surprise"][:, :length]),
    }
    for name in ("graph_neighbor_values", "graph_neighbor_available"):
        if name in batch:
            arrays[name] = np.asarray(batch[name][:, :length])
    if variant != "direct":
        arrays["stale_state"] = np.asarray(batch["stale_state"])
    if variant == "latent":
        arrays["context_latent"] = np.asarray(batch["context_latent"])
        arrays["predicted_delta"] = np.asarray(batch["predicted_delta"])
    return arrays


def _load_contract(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("role") != "post_impact_clock_gated_forward_shadow_contract":
        raise ValueError("prospective replay requires a frozen forward contract")
    if payload.get("live_orders_allowed") is not False:
        raise ValueError("prospective replay contract permits live orders")
    if payload.get("broker_order_calls_allowed") is not False:
        raise ValueError("prospective replay contract permits broker order calls")
    if any(name not in payload.get("models", {}) for name in OPERATIONAL_MODELS):
        raise ValueError("prospective replay contract lacks operational models")
    return payload


def _device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _model_runtime(
    name: str,
    spec: Mapping[str, Any],
    release: DayRelease,
    stale: StaleCache,
    date: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint_path = Path(str(spec["checkpoint"]))
    if file_sha256(checkpoint_path) != spec["checkpoint_sha256"]:
        raise ValueError(f"frozen prospective checkpoint changed: {name}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = argparse.Namespace(**checkpoint["args"])
    mode = _resolved_daily_context_placebo_mode(checkpoint_args)
    checkpoint_args.daily_context_placebo_mode = mode
    checkpoint_args.shuffle_daily_context = mode == "all"
    expected = MODEL_RUNTIME_CONTRACTS.get(name)
    if expected is None:
        raise ValueError(f"unknown prospective model contract: {name}")
    if expected != (str(checkpoint_args.variant), mode):
        raise ValueError(f"prospective model contract is not operational: {name}")
    if tuple(checkpoint["feature_names"]) != release.feature_names:
        raise ValueError(f"prospective feature contract mismatch: {name}")
    expected_graph_features = _graph_message_feature_names(
        release.feature_names,
        str(getattr(checkpoint_args, "graph_message_mode", "none")),
    )
    if tuple(
        checkpoint.get("graph_message_feature_names", expected_graph_features)
    ) != expected_graph_features:
        raise ValueError(f"prospective graph-message contract mismatch: {name}")
    if tuple(checkpoint["state_feature_names"]) != stale.state_feature_names:
        raise ValueError(f"prospective stale-state contract mismatch: {name}")
    if tuple(checkpoint["horizon_labels"]) != release.horizon_labels:
        raise ValueError(f"prospective horizon contract mismatch: {name}")
    if tuple(checkpoint["target_names"]) != release.target_names:
        raise ValueError(f"prospective target contract mismatch: {name}")
    if tuple(checkpoint["systemic_target_names"]) != release.systemic_target_names:
        raise ValueError(f"prospective systemic contract mismatch: {name}")

    node_scaler = _scaler(checkpoint["node_scaler"])
    target_scaler = _scaler(checkpoint["target_scaler"])
    stale_scaler = _scaler(checkpoint["stale_scaler"])
    observed_calibration = _calibration(checkpoint["observed_surprise_calibration"])
    model_calibration = _calibration(checkpoint["model_surprise_calibration"])
    impact_thresholds = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in checkpoint["impact_thresholds"].items()
    }
    batch = _pad_batch(
        release,
        stale,
        [date],
        {date: date},
        {date: date},
        observed_calibration,
        model_calibration,
        node_scaler,
        stale_scaler,
        impact_thresholds,
        checkpoint_args,
    )
    model = CausalPostImpactReforecast(
        node_feature_dim=len(release.feature_names),
        stale_state_dim=len(stale.state_feature_names),
        latent_dim=int(stale.context.shape[-1]),
        horizons=release.horizon_labels,
        systemic_target_dim=len(release.systemic_target_names),
        variant=checkpoint_args.variant,
        hidden_dim=int(checkpoint_args.hidden_dim),
        latent_projection_dim=int(checkpoint_args.latent_projection_dim),
        temporal_layers=int(checkpoint_args.temporal_layers),
        dropout=float(checkpoint_args.dropout),
        surprise_dim=len(SURPRISE_STATISTIC_NAMES),
        graph_message_dim=_graph_message_feature_dim(
            release.feature_names,
            str(getattr(checkpoint_args, "graph_message_mode", "none")),
        ),
        graph_message_fusion=str(
            getattr(checkpoint_args, "graph_message_fusion", "shared")
        ),
        target_names=release.target_names,
        output_scales=target_scaler.scale,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    if model.training:
        raise RuntimeError(f"prospective model remained in training mode: {name}")
    return {
        "name": name,
        "checkpoint_path": checkpoint_path,
        "checkpoint_sha256": spec["checkpoint_sha256"],
        "checkpoint_args": checkpoint_args,
        "batch": batch,
        "model": model,
    }


def _predict_prefix(
    runtime: Mapping[str, Any],
    *,
    prefix: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    checkpoint_args = runtime["checkpoint_args"]
    batch = runtime["batch"]
    model_input = causal_model_input_arrays(
        batch,
        prefix=prefix,
        variant=str(checkpoint_args.variant),
    )
    input_sha256 = prediction_array_fingerprint(model_input)
    tensors = _input_tensors(batch, prefix=prefix, device=device)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        prediction = runtime["model"](
            tensors["node_values"],
            tensors["node_available"],
            stale_state=(
                tensors["stale_state"]
                if checkpoint_args.variant != "direct"
                else None
            ),
            context_latent=(
                tensors["context_latent"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            predicted_delta=(
                tensors["predicted_delta"].float()
                if checkpoint_args.variant == "latent"
                else None
            ),
            surprise_values=tensors["surprise"],
            graph_neighbor_values=tensors.get("graph_neighbor_values"),
            graph_neighbor_available=tensors.get("graph_neighbor_available"),
        )
    synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    node = prediction.node[:, -1].float().cpu().numpy()[0]
    systemic = prediction.systemic[:, -1].float().cpu().numpy()[0]
    if not np.isfinite(node).all() or not np.isfinite(systemic).all():
        raise RuntimeError(f"prospective prediction is non-finite: {runtime['name']}")
    return node, systemic, elapsed_ms, input_sha256


def _artifact_arrays(
    release: DayRelease,
    model_names: Sequence[str],
    node_predictions: Sequence[np.ndarray],
    systemic_predictions: Sequence[np.ndarray],
    *,
    decision_timestamp_utc_ns: int,
    prefix: int,
) -> dict[str, np.ndarray]:
    return {
        "model_names": np.asarray(model_names, dtype="U32"),
        "tickers": np.asarray(release.tickers, dtype="U6"),
        "horizon_labels": np.asarray(release.horizon_labels, dtype="U16"),
        "target_names": np.asarray(release.target_names, dtype="U64"),
        "systemic_target_names": np.asarray(
            release.systemic_target_names, dtype="U64"
        ),
        "decision_timestamp_utc_ns": np.asarray(
            [int(decision_timestamp_utc_ns)], dtype=np.int64
        ),
        "prefix_timestamps": np.asarray([int(prefix)], dtype=np.int16),
        "node_prediction": np.stack(node_predictions).astype(np.float32),
        "systemic_prediction": np.stack(systemic_predictions).astype(np.float32),
    }


def main() -> None:
    args = parse_args()
    contract_path = Path(args.contract)
    contract = _load_contract(contract_path)
    device = _device(args.device)
    release = DayRelease(Path(args.day_release_dir), cache=True)
    stale = StaleCache(Path(args.stale_cache_dir))
    stale.align_tickers(release.tickers)
    date = str(args.date)
    if date not in release.records or date not in stale.date_to_row:
        raise ValueError("prospective replay date is absent from frozen inputs")
    day = release.load(date)
    clocks = parse_clocks(args.clocks)
    prefixes = prefix_indices(day["timestamps_utc_ns"], clocks)
    runtimes = {
        name: _model_runtime(
            name,
            contract["models"][name],
            release,
            stale,
            date,
            device,
        )
        for name in OPERATIONAL_MODELS
    }
    artifact_root = Path(args.artifact_root)
    ledger_path = Path(args.ledger)
    source_hash = file_sha256(Path(__file__))
    ledger_module_hash = file_sha256(ROOT / "stock_v2/prospective_ledger.py")
    day_record = release.records[date]
    appended_records = 0
    replay_rows: list[dict[str, Any]] = []
    timestamps = np.asarray(day["timestamps_utc_ns"], dtype=np.int64)
    for clock, prefix in prefixes.items():
        clock_hhmm = format_clock_hhmm(clock)
        model_names: list[str] = []
        node_predictions: list[np.ndarray] = []
        systemic_predictions: list[np.ndarray] = []
        model_records: dict[str, Any] = {}
        model_latencies_ms: dict[str, float] = {}
        for name in OPERATIONAL_MODELS:
            node, systemic, latency_ms, input_sha256 = _predict_prefix(
                runtimes[name], prefix=prefix, device=device
            )
            model_names.append(name)
            node_predictions.append(node)
            systemic_predictions.append(systemic)
            model_records[name] = {
                "checkpoint_sha256": runtimes[name]["checkpoint_sha256"],
                "model_input_sha256": input_sha256,
            }
            model_latencies_ms[name] = latency_ms
        decision_timestamp = int(timestamps[prefix - 1])
        arrays = _artifact_arrays(
            release,
            model_names,
            node_predictions,
            systemic_predictions,
            decision_timestamp_utc_ns=decision_timestamp,
            prefix=prefix,
        )
        artifact = write_immutable_prediction_artifact(
            Path("predictions") / date / f"{clock_hhmm}.npz",
            arrays,
            artifact_root=artifact_root,
        )
        commit_id = f"{date}|{clock_hhmm}|post_impact_direct_latent_v1"
        payload = {
            "schema_version": 1,
            "role": LEDGER_ROLE,
            "commit_id": commit_id,
            "session": date,
            "decision_timestamp_utc_ns": decision_timestamp,
            "decision_clock_minute_kst": int(clock),
            "prefix_timestamps": int(prefix),
            "source_mode": "historical_causal_replay",
            "historical_replay": {
                "counts_as_forward_evidence": False,
                "source_day_complete_before_replay": True,
            },
            "input_pins": {
                "forward_contract": file_sha256(contract_path),
                "day_release_manifest": file_sha256(release.manifest_path),
                "day_shard": str(day_record["sha256"]),
                "stale_cache_manifest": file_sha256(stale.manifest_path),
                "replay_source": source_hash,
                "ledger_module": ledger_module_hash,
            },
            "models": model_records,
            "prediction_artifact": artifact,
            "prediction_array_sha256": prediction_array_fingerprint(arrays),
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
        record, appended = append_prediction_commit(
            ledger_path,
            payload,
            artifact_root=artifact_root,
        )
        appended_records += int(appended)
        replay_rows.append(
            {
                "clock": int(clock),
                "clock_hhmm": clock_hhmm,
                "prefix": int(prefix),
                "commit_id": commit_id,
                "record_sha256": record["record_sha256"],
                "prediction_array_sha256": payload["prediction_array_sha256"],
                "inference_ms": model_latencies_ms,
                "appended": bool(appended),
            }
        )
    records = read_prediction_ledger(ledger_path, artifact_root=artifact_root)
    summary = {
        "schema_version": 1,
        "role": "post_impact_prospective_prediction_ledger_replay_audit",
        "status": "pass",
        "date": date,
        "device": str(device),
        "stocks": len(release.tickers),
        "models": list(OPERATIONAL_MODELS),
        "clocks": list(clocks),
        "commits_requested": len(clocks),
        "commits_appended": appended_records,
        "ledger": ledger_summary(records),
        "rows": replay_rows,
        "input_pins": {
            "contract_sha256": file_sha256(contract_path),
            "day_release_manifest_sha256": file_sha256(release.manifest_path),
            "day_shard_sha256": str(day_record["sha256"]),
            "stale_cache_manifest_sha256": file_sha256(stale.manifest_path),
            "source_sha256": source_hash,
            "ledger_module_sha256": ledger_module_hash,
        },
        "historical_replay_counts_as_forward_evidence": False,
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
                "records": len(records),
                "appended": appended_records,
                "head_sha256": summary["ledger"]["head_sha256"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
