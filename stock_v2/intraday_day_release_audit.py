from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_v2.intraday_release_audit import (
    _validate_target_physics,
    file_sha256,
)
from stock_v2.intraday_trajectory import (
    INTRADAY_TRAJECTORY_FEATURE_NAMES,
    INTRADAY_TRAJECTORY_FEATURE_NAMES_V1,
    INTRADAY_TRAJECTORY_TARGET_NAMES,
    SYSTEMIC_TRAJECTORY_TARGET_NAMES,
    IntradayTrajectoryPanel,
    summarize_systemic_intraday_trajectory,
)


AUDIT_CONTRACT = "intraday_post_impact_day_release_audit_v1"
ASSEMBLY_CONTRACT = "time_major_intraday_post_impact_days_v1"
DAY_KEYS = {
    "timestamps_utc_ns",
    "node_values",
    "node_available",
    "decision_price",
    "targets",
    "target_available",
    "systemic_targets",
    "systemic_available",
}
METADATA_KEYS = {
    "tickers",
    "populated_tickers",
    "feature_names",
    "horizons_minutes",
    "horizon_labels",
    "target_names",
    "systemic_target_names",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _release_path(root: Path, value: str, label: str) -> Path:
    relative = Path(str(value))
    _require(not relative.is_absolute(), f"{label} must be release-relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the release directory") from exc
    return resolved


def _binary_mask(value: np.ndarray, label: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(bool(np.isin(raw, (0, 1)).all()), f"{label} is not binary")
    return raw.astype(bool)


def _source_index(source_root: Path, expected_manifest_sha256: str) -> tuple[np.ndarray, np.ndarray]:
    manifest_path = source_root / "manifest.json"
    _require(manifest_path.is_file(), "source trajectory manifest is missing")
    _require(file_sha256(manifest_path) == expected_manifest_sha256, "source trajectory manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output = manifest["outputs"]
    index_path = _release_path(source_root, output["timestamp_index"], "source timestamp index")
    _require(file_sha256(index_path) == output["timestamp_index_sha256"], "source timestamp index checksum mismatch")
    with np.load(index_path, allow_pickle=False) as bundle:
        timestamp_ns = np.asarray(bundle["timestamps_utc_ns"], dtype=np.int64)
        node_counts = np.asarray(bundle["node_counts"], dtype=np.int64)
    return timestamp_ns, node_counts


def audit_intraday_day_release(
    release_dir: str | Path,
    *,
    source_release_dir: str | Path | None = None,
    minimum_days: int = 1,
    target_tolerance: float = 5e-6,
    systemic_tolerance: float = 2e-6,
) -> dict[str, Any]:
    root = Path(release_dir)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "intraday day release manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "unsupported day release schema")
    _require(manifest.get("assembly_contract") == ASSEMBLY_CONTRACT, "unsupported day release contract")
    _require(manifest.get("promotion_eligible") is False, "day release enables promotion")
    _require(manifest.get("live_orders_allowed") is False, "day release enables live orders")
    _require(manifest.get("portable_payload_paths") is True, "day release does not declare portable payloads")
    _require(manifest.get("transactional_publish") is True, "day release was not transactionally published")

    metadata_record = manifest["metadata"]
    metadata_path = _release_path(root, metadata_record["path"], "day metadata")
    _require(metadata_path.is_file(), "day release metadata is missing")
    _require(file_sha256(metadata_path) == metadata_record["sha256"], "day metadata checksum mismatch")
    with np.load(metadata_path, allow_pickle=False) as metadata:
        _require(set(metadata.files) == METADATA_KEYS, "day metadata keys changed")
        tickers = tuple(str(value) for value in metadata["tickers"].tolist())
        populated_tickers = tuple(
            str(value) for value in metadata["populated_tickers"].tolist()
        )
        feature_names = tuple(str(value) for value in metadata["feature_names"].tolist())
        horizons = tuple(int(value) for value in metadata["horizons_minutes"].tolist())
        horizon_labels = tuple(str(value) for value in metadata["horizon_labels"].tolist())
        target_names = tuple(str(value) for value in metadata["target_names"].tolist())
        systemic_names = tuple(
            str(value) for value in metadata["systemic_target_names"].tolist()
        )
    _require(len(tickers) == len(set(tickers)), "day release tickers are duplicated")
    _require(len(populated_tickers) == len(set(populated_tickers)), "populated tickers are duplicated")
    _require(set(populated_tickers).issubset(tickers), "populated tickers escape the node universe")
    _require(
        feature_names
        in {
            tuple(INTRADAY_TRAJECTORY_FEATURE_NAMES_V1),
            tuple(INTRADAY_TRAJECTORY_FEATURE_NAMES),
        },
        "day feature contract changed",
    )
    _require(target_names == tuple(INTRADAY_TRAJECTORY_TARGET_NAMES), "day target contract changed")
    _require(systemic_names == tuple(SYSTEMIC_TRAJECTORY_TARGET_NAMES), "day systemic contract changed")
    _require(horizon_labels == tuple(f"{value}m" for value in horizons) + ("close",), "day horizon labels changed")
    _require(int(manifest["stocks"]) == len(tickers), "day node-axis count mismatch")
    _require(int(manifest["populated_stocks"]) == len(populated_tickers), "populated stock count mismatch")

    records = manifest["day_shards"]
    _require(len(records) >= int(minimum_days), "day release count is below audit gate")
    _require(int(manifest["days"]) == len(records), "day release manifest count mismatch")
    dates = tuple(str(record["date"]) for record in records)
    _require(dates == tuple(sorted(set(dates))), "day records are duplicated or unsorted")
    _require(manifest["first_date"] == dates[0], "day release first date mismatch")
    _require(manifest["last_date"] == dates[-1], "day release last date mismatch")
    gates = manifest["gates"]
    minimum_nodes = int(gates["minimum_nodes_per_timestamp"])
    minimum_timestamps = int(gates["minimum_timestamps_per_day"])
    systemic_min_nodes = int(gates["systemic_min_nodes"])

    source_timestamp_ns: np.ndarray | None = None
    source_node_counts: np.ndarray | None = None
    if source_release_dir is not None:
        source_timestamp_ns, source_node_counts = _source_index(
            Path(source_release_dir), str(manifest["source_manifest_sha256"])
        )

    node_count = len(tickers)
    feature_count = len(feature_names)
    target_count = len(target_names)
    horizon_count = len(horizon_labels)
    clock_index = feature_names.index("clock_fraction")
    endpoint_index = target_names.index("endpoint_return")
    total_timestamps = 0
    physical_target_rows = 0
    systemic_values = 0
    systemic_max_error = 0.0
    minimum_observed_nodes = node_count
    minimum_h5_target_nodes = node_count

    for record in records:
        date = str(record["date"])
        path = _release_path(root, record["path"], f"day shard {date}")
        _require(path.is_file(), f"day shard is missing: {date}")
        _require(path.stat().st_size == int(record["bytes"]), f"day shard byte count mismatch: {date}")
        _require(file_sha256(path) == record["sha256"], f"day shard checksum mismatch: {date}")
        with np.load(path, allow_pickle=False) as bundle:
            _require(set(bundle.files) == DAY_KEYS, f"day shard keys changed: {date}")
            timestamp_ns = np.asarray(bundle["timestamps_utc_ns"], dtype=np.int64)
            node_values = np.asarray(bundle["node_values"], dtype=np.float32)
            node_available = _binary_mask(bundle["node_available"], f"{date} node availability")
            decision_price = np.asarray(bundle["decision_price"], dtype=np.float32)
            targets = np.asarray(bundle["targets"], dtype=np.float32)
            target_available = _binary_mask(bundle["target_available"], f"{date} target availability")
            systemic_targets = np.asarray(bundle["systemic_targets"], dtype=np.float32)
            systemic_available = _binary_mask(bundle["systemic_available"], f"{date} systemic availability")

        time_count = len(timestamp_ns)
        _require(time_count >= minimum_timestamps, f"day has too few timestamps: {date}")
        _require(bool((np.diff(timestamp_ns) > 0).all()), f"day timestamps are duplicated or unsorted: {date}")
        _require(node_values.shape == node_available.shape == (time_count, node_count, feature_count), f"day node arrays are misaligned: {date}")
        _require(decision_price.shape == (time_count, node_count), f"day prices are misaligned: {date}")
        expected_target_shape = (time_count, node_count, horizon_count, target_count)
        _require(targets.shape == target_available.shape == expected_target_shape, f"day target arrays are misaligned: {date}")
        expected_systemic_shape = (time_count, horizon_count, len(systemic_names))
        _require(systemic_targets.shape == systemic_available.shape == expected_systemic_shape, f"day systemic arrays are misaligned: {date}")
        _require(bool(np.array_equal(node_available, np.isfinite(node_values))), f"day node mask differs from finite values: {date}")
        _require(bool(np.array_equal(target_available, np.isfinite(targets))), f"day target mask differs from finite values: {date}")
        _require(bool(np.array_equal(systemic_available, np.isfinite(systemic_targets))), f"day systemic mask differs from finite values: {date}")
        present = node_available[:, :, clock_index]
        _require(bool(np.array_equal(present, np.isfinite(decision_price))), f"day price presence differs from node presence: {date}")
        _require(bool((decision_price[present] > 0.0).all()), f"day has a non-positive decision price: {date}")

        timestamps = pd.to_datetime(timestamp_ns, utc=True).tz_convert("Asia/Seoul")
        local_dates = timestamps.tz_localize(None).normalize()
        _require(bool((local_dates == pd.Timestamp(date)).all()), f"day shard contains a different session date: {date}")
        clocks = np.asarray(timestamps.hour * 60 + timestamps.minute, dtype=np.int16)
        _require(bool(np.isin(clocks, np.arange(9 * 60 + 15, 15 * 60 + 16, 5)).all()), f"day shard contains an invalid decision clock: {date}")

        physical_target_rows += _validate_target_physics(
            targets[:, :, :-1],
            target_available[:, :, :-1],
            ticker=date,
            label="horizon",
            tolerance=target_tolerance,
        )
        physical_target_rows += _validate_target_physics(
            targets[:, :, -1],
            target_available[:, :, -1],
            ticker=date,
            label="close",
            tolerance=target_tolerance,
        )

        panel = IntradayTrajectoryPanel(
            timestamps=timestamps,
            session_dates=local_dates,
            decision_clock_minutes=clocks,
            tickers=tickers,
            feature_names=feature_names,
            values=node_values,
            available=node_available,
            decision_price=decision_price,
            horizons_minutes=horizons,
            target_names=target_names,
            horizon_targets=targets[:, :, :-1],
            horizon_available=target_available[:, :, :-1],
            close_targets=targets[:, :, -1],
            close_available=target_available[:, :, -1],
        )
        recomputed = summarize_systemic_intraday_trajectory(
            panel, min_nodes=systemic_min_nodes
        )
        _require(bool(np.array_equal(recomputed.available, systemic_available)), f"recomputed systemic mask mismatch: {date}")
        selected = systemic_available
        if selected.any():
            error = np.abs(
                recomputed.values[selected].astype(np.float64)
                - systemic_targets[selected].astype(np.float64)
            )
            systemic_values += int(len(error))
            systemic_max_error = max(
                systemic_max_error, float(error.max(initial=0.0))
            )
            _require(bool((error <= systemic_tolerance).all()), f"recomputed systemic values mismatch: {date}")

        observed_counts = present.sum(axis=1)
        h5_counts = target_available[:, :, 0, endpoint_index].sum(axis=1)
        minimum_observed_nodes = min(minimum_observed_nodes, int(observed_counts.min()))
        minimum_h5_target_nodes = min(minimum_h5_target_nodes, int(h5_counts.min()))
        _require(abs(float(record["median_target_nodes_h5"]) - float(np.median(h5_counts))) <= 1e-12, f"day h5 target median mismatch: {date}")
        if source_timestamp_ns is not None and source_node_counts is not None:
            positions = np.searchsorted(source_timestamp_ns, timestamp_ns)
            _require(bool((positions < len(source_timestamp_ns)).all()), f"day timestamp is absent from source: {date}")
            _require(bool(np.array_equal(source_timestamp_ns[positions], timestamp_ns)), f"day timestamp failed source alignment: {date}")
            selected_source_counts = source_node_counts[positions]
            _require(bool((selected_source_counts >= minimum_nodes).all()), f"day source-node gate failed: {date}")
            _require(abs(float(record["median_source_nodes"]) - float(np.median(selected_source_counts))) <= 1e-12, f"day source-node median mismatch: {date}")
        _require(int(record["timestamps"]) == time_count, f"day timestamp record mismatch: {date}")
        total_timestamps += time_count

    return {
        "schema_version": 1,
        "audit_contract": AUDIT_CONTRACT,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": str(root.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "source_manifest_sha256": str(manifest["source_manifest_sha256"]),
        "passed": True,
        "days": len(records),
        "stocks": node_count,
        "populated_stocks": len(populated_tickers),
        "total_timestamps": total_timestamps,
        "minimum_observed_nodes": minimum_observed_nodes,
        "minimum_h5_target_nodes": minimum_h5_target_nodes,
        "physical_target_rows": physical_target_rows,
        "systemic_values_recomputed": systemic_values,
        "systemic_max_abs_error": systemic_max_error,
        "source_release_verified": source_release_dir is not None,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
