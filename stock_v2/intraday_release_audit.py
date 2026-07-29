from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from stock_v2.intraday_trajectory import (
    INTRADAY_TRAJECTORY_FEATURE_NAMES,
    INTRADAY_TRAJECTORY_FEATURE_NAMES_V1,
    INTRADAY_TRAJECTORY_TARGET_NAMES,
)


AUDIT_CONTRACT = "portable_intraday_trajectory_release_audit_v1"
TRAJECTORY_CONTRACTS = {
    "kiwoom_raw_rolling_post_impact_trajectory_v1",
    "kiwoom_raw_rolling_post_impact_trajectory_v2",
}
SHARD_KEYS = {
    "timestamps_utc_ns",
    "feature_names",
    "values",
    "available",
    "decision_price",
    "horizons_minutes",
    "target_names",
    "horizon_targets",
    "horizon_available",
    "close_targets",
    "close_available",
}
REQUIRED_CAUSALITY = (
    "decision_bar_excluded_from_start_labelled_inputs",
    "targets_begin_strictly_after_decision",
    "same_clock_baselines_shifted_one_session",
    "missing_bars_never_filled",
    "mfe_mae_use_only_post_decision_bars",
    "close_auction_required_for_close_target",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _clock_minutes(value: str) -> int:
    try:
        hour, minute = (int(part) for part in str(value).split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid decision clock: {value}") from exc
    _require(0 <= hour <= 23 and 0 <= minute <= 59, f"invalid decision clock: {value}")
    return hour * 60 + minute


def _portable_output_path(root: Path, value: str, label: str) -> Path:
    relative = Path(str(value))
    _require(not relative.is_absolute(), f"{label} must be release-relative")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes the release directory") from exc
    return resolved


def _exact_binary_mask(value: np.ndarray, label: str) -> np.ndarray:
    raw = np.asarray(value)
    _require(bool(np.isin(raw, (0, 1)).all()), f"{label} is not a binary mask")
    return raw.astype(bool)


def _assert_close(left: float, right: float, tolerance: float, label: str) -> None:
    _require(
        bool(np.isfinite(left))
        and bool(np.isfinite(right))
        and abs(float(left) - float(right)) <= float(tolerance),
        f"{label} mismatch: {left} != {right}",
    )


def _validate_target_physics(
    target: np.ndarray,
    available: np.ndarray,
    *,
    ticker: str,
    label: str,
    tolerance: float,
) -> int:
    endpoint = INTRADAY_TRAJECTORY_TARGET_NAMES.index("endpoint_return")
    mfe = INTRADAY_TRAJECTORY_TARGET_NAMES.index("mfe")
    mae = INTRADAY_TRAJECTORY_TARGET_NAMES.index("mae")
    realized = INTRADAY_TRAJECTORY_TARGET_NAMES.index("realized_absolute_return")
    future_range = INTRADAY_TRAJECTORY_TARGET_NAMES.index("future_range")
    peak = INTRADAY_TRAJECTORY_TARGET_NAMES.index("time_to_peak_fraction")
    trough = INTRADAY_TRAJECTORY_TARGET_NAMES.index("time_to_trough_fraction")
    endpoint_available = available[..., endpoint]
    path = available[..., 1:7]
    _require(
        bool((path == path[..., :1]).all()),
        f"{ticker} {label} path target masks are inconsistent",
    )
    path_available = path[..., 0]
    _require(
        bool((~path_available | endpoint_available).all()),
        f"{ticker} {label} path target exists without endpoint return",
    )
    selected = path_available
    if not selected.any():
        return 0
    values = target[selected]
    _require(
        bool((values[:, mfe] + tolerance >= values[:, endpoint]).all()),
        f"{ticker} {label} MFE is below endpoint return",
    )
    _require(
        bool((values[:, mae] - tolerance <= values[:, endpoint]).all()),
        f"{ticker} {label} MAE is above endpoint return",
    )
    _require(
        bool((values[:, mfe] + tolerance >= values[:, mae]).all()),
        f"{ticker} {label} MFE is below MAE",
    )
    endpoint_values = values[:, endpoint]
    minimum_realized = np.where(
        endpoint_values >= 0.0,
        np.log1p(endpoint_values),
        -endpoint_values,
    )
    _require(
        bool((values[:, realized] + tolerance >= minimum_realized).all()),
        f"{ticker} {label} realized absolute return violates compound path bound",
    )
    _require(
        bool((values[:, future_range] >= -tolerance).all()),
        f"{ticker} {label} future range is negative",
    )
    for index, name in ((peak, "peak"), (trough, "trough")):
        _require(
            bool(((values[:, index] > 0.0) & (values[:, index] <= 1.0 + tolerance)).all()),
            f"{ticker} {label} {name} time fraction is outside (0, 1]",
        )
    return int(selected.sum())


def _verify_input_files(
    manifest: Mapping[str, Any], *, require_input_files: bool
) -> dict[str, Any]:
    inputs = manifest.get("inputs", {})
    records = (
        ("coverage", "coverage_sha256"),
        ("universe", "universe_sha256"),
        ("semantics_evidence", "semantics_evidence_sha256"),
    )
    result: dict[str, Any] = {}
    for path_key, hash_key in records:
        path = Path(str(inputs.get(path_key, "")))
        expected = str(inputs.get(hash_key, ""))
        exists = path.is_file()
        verified = exists and bool(expected) and file_sha256(path) == expected
        if require_input_files:
            _require(exists, f"source input is unavailable: {path_key}")
            _require(verified, f"source input checksum mismatch: {path_key}")
        result[path_key] = {
            "path": str(path),
            "exists": bool(exists),
            "verified": bool(verified),
        }
    return result


def audit_intraday_trajectory_release(
    release_dir: str | Path,
    *,
    minimum_shards: int = 400,
    minimum_price_match_ratio: float = 0.995,
    minimum_volume_contained_ratio: float = 0.995,
    minimum_median_volume_coverage: float = 0.80,
    endpoint_tolerance: float = 5e-6,
    require_input_files: bool = False,
) -> dict[str, Any]:
    root = Path(release_dir)
    manifest_path = root / "manifest.json"
    _require(manifest_path.is_file(), "trajectory release manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == 1, "unsupported trajectory schema")
    _require(
        manifest.get("trajectory_contract") in TRAJECTORY_CONTRACTS,
        "unsupported trajectory release contract",
    )
    _require(manifest.get("basis") == "raw", "trajectory basis must be raw")
    _require(manifest.get("promotion_eligible") is False, "release enables promotion")
    _require(manifest.get("live_orders_allowed") is False, "release enables live orders")
    for name in REQUIRED_CAUSALITY:
        _require(manifest.get("causality", {}).get(name) is True, f"causality claim missing: {name}")
    if manifest["trajectory_contract"].endswith("_v2"):
        for name in (
            "post_gap_decisions_require_a_fresh_completed_bar",
            "post_gap_cumulative_features_remain_masked",
            "endpoint_return_requires_exact_horizon_price_only",
            "path_targets_require_contiguous_future_bars",
        ):
            _require(
                manifest.get("causality", {}).get(name) is True,
                f"v2 causality claim missing: {name}",
            )

    interval = int(manifest["interval_minutes"])
    _require(interval > 0, "interval_minutes must be positive")
    start_clock = _clock_minutes(manifest["decision_start"])
    end_clock = _clock_minutes(manifest["decision_end"])
    _require(start_clock <= end_clock, "decision clock range is reversed")
    horizons = tuple(int(value) for value in manifest["horizons_minutes"])
    _require(
        bool(horizons)
        and tuple(sorted(set(horizons))) == horizons
        and all(value > 0 and value % interval == 0 for value in horizons),
        "horizon contract is invalid",
    )

    universe = tuple(str(value) for value in manifest["universe_tickers"])
    shard_tickers = tuple(str(value) for value in manifest["shard_tickers"])
    missing = tuple(str(value) for value in manifest.get("missing_tickers", ()))
    short = tuple(str(value) for value in manifest.get("short_tickers", ()))
    _require(len(universe) == len(set(universe)), "universe tickers are not unique")
    _require(len(shard_tickers) == len(set(shard_tickers)), "shard tickers are not unique")
    _require(len(shard_tickers) >= int(minimum_shards), "trajectory shard count is below gate")
    groups = (set(shard_tickers), set(missing), set(short))
    _require(not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]), "ticker accounting groups overlap")
    _require(set(universe) == set().union(*groups), "ticker accounting does not cover the universe")
    _require(int(manifest["universe_stocks"]) == len(universe), "universe stock count mismatch")
    _require(int(manifest["stocks"]) == len(shard_tickers), "stock count mismatch")

    outputs = manifest["outputs"]
    index_path = _portable_output_path(root, outputs["timestamp_index"], "timestamp index")
    _require(index_path.is_file(), "trajectory timestamp index is missing")
    _require(file_sha256(index_path) == outputs["timestamp_index_sha256"], "timestamp index checksum mismatch")
    with np.load(index_path, allow_pickle=False) as bundle:
        _require(set(bundle.files) == {"timestamps_utc_ns", "node_counts"}, "timestamp index keys changed")
        timestamp_ns = np.asarray(bundle["timestamps_utc_ns"], dtype=np.int64)
        node_counts = np.asarray(bundle["node_counts"], dtype=np.int64)
    _require(timestamp_ns.ndim == node_counts.ndim == 1, "timestamp index arrays must be vectors")
    _require(len(timestamp_ns) > 0 and len(timestamp_ns) == len(node_counts), "timestamp index arrays are empty or misaligned")
    _require(bool((np.diff(timestamp_ns) > 0).all()), "timestamp index is not strictly increasing")
    _require(bool(((node_counts > 0) & (node_counts <= len(shard_tickers))).all()), "timestamp node counts are outside bounds")
    local_timestamps = pd.to_datetime(timestamp_ns, utc=True).tz_convert("Asia/Seoul")
    clocks = np.asarray(local_timestamps.hour * 60 + local_timestamps.minute)
    _require(bool(((clocks >= start_clock) & (clocks <= end_clock)).all()), "timestamp index is outside decision clocks")
    _require(bool((((clocks - start_clock) % interval) == 0).all()), "timestamp index is off interval grid")
    _require(bool((local_timestamps.dayofweek < 5).all()), "timestamp index contains a weekend")

    shard_records = outputs["shards"]
    _require(len(shard_records) == len(shard_tickers), "shard records and ticker list differ")
    _require(tuple(str(record["ticker"]) for record in shard_records) == shard_tickers, "shard record order differs from ticker contract")
    _require(canonical_sha256(shard_records) == outputs["shards_sha256"], "aggregate shard record checksum mismatch")

    reconstructed_counts = np.zeros(len(timestamp_ns), dtype=np.int64)
    target_node_counts = np.zeros(
        (len(timestamp_ns), len(horizons) + 1), dtype=np.int64
    )
    total_snapshots = 0
    physics_targets = 0
    endpoint_comparisons = 0
    endpoint_max_error = 0.0
    feature_names = (
        tuple(INTRADAY_TRAJECTORY_FEATURE_NAMES_V1)
        if manifest["trajectory_contract"].endswith("_v1")
        else tuple(INTRADAY_TRAJECTORY_FEATURE_NAMES)
    )
    target_names = tuple(INTRADAY_TRAJECTORY_TARGET_NAMES)
    clock_feature = feature_names.index("clock_fraction")
    endpoint_target = target_names.index("endpoint_return")

    for record in shard_records:
        ticker = str(record["ticker"])
        path = _portable_output_path(root, record["path"], f"ticker shard {ticker}")
        _require(path.is_file(), f"ticker shard is missing: {ticker}")
        _require(file_sha256(path) == record["sha256"], f"ticker shard checksum mismatch: {ticker}")
        with np.load(path, allow_pickle=False) as bundle:
            _require(set(bundle.files) == SHARD_KEYS, f"ticker shard keys changed: {ticker}")
            ticker_timestamp_ns = np.asarray(bundle["timestamps_utc_ns"], dtype=np.int64)
            actual_features = tuple(str(value) for value in bundle["feature_names"].tolist())
            values = np.asarray(bundle["values"], dtype=np.float32)
            available = _exact_binary_mask(bundle["available"], f"{ticker} input availability")
            decision_price = np.asarray(bundle["decision_price"], dtype=np.float32)
            actual_horizons = tuple(int(value) for value in bundle["horizons_minutes"].tolist())
            actual_targets = tuple(str(value) for value in bundle["target_names"].tolist())
            horizon_targets = np.asarray(bundle["horizon_targets"], dtype=np.float32)
            horizon_available = _exact_binary_mask(bundle["horizon_available"], f"{ticker} horizon availability")
            close_targets = np.asarray(bundle["close_targets"], dtype=np.float32)
            close_available = _exact_binary_mask(bundle["close_available"], f"{ticker} close availability")

        rows = len(ticker_timestamp_ns)
        _require(rows > 0 and bool((np.diff(ticker_timestamp_ns) > 0).all()), f"{ticker} timestamps are empty, duplicated, or unsorted")
        _require(actual_features == feature_names, f"{ticker} feature contract changed")
        _require(actual_horizons == horizons, f"{ticker} horizon contract changed")
        _require(actual_targets == target_names, f"{ticker} target contract changed")
        _require(values.shape == available.shape == (rows, len(feature_names)), f"{ticker} feature arrays are misaligned")
        _require(decision_price.shape == (rows,), f"{ticker} decision prices are misaligned")
        expected_horizon_shape = (rows, len(horizons), len(target_names))
        _require(horizon_targets.shape == horizon_available.shape == expected_horizon_shape, f"{ticker} horizon targets are misaligned")
        _require(close_targets.shape == close_available.shape == (rows, len(target_names)), f"{ticker} close targets are misaligned")
        _require(bool(np.array_equal(available, np.isfinite(values))), f"{ticker} input mask does not equal finite values")
        _require(bool(np.array_equal(horizon_available, np.isfinite(horizon_targets))), f"{ticker} horizon mask does not equal finite targets")
        _require(bool(np.array_equal(close_available, np.isfinite(close_targets))), f"{ticker} close mask does not equal finite targets")
        _require(bool((np.isfinite(decision_price) & (decision_price > 0.0)).all()), f"{ticker} decision prices are non-positive or non-finite")

        positions = np.searchsorted(timestamp_ns, ticker_timestamp_ns)
        _require(bool((positions < len(timestamp_ns)).all()), f"{ticker} timestamp is absent from global index")
        _require(bool(np.array_equal(timestamp_ns[positions], ticker_timestamp_ns)), f"{ticker} timestamp failed global alignment")
        np.add.at(reconstructed_counts, positions, 1)
        for horizon_index in range(len(horizons)):
            target_rows = positions[
                horizon_available[:, horizon_index, endpoint_target]
            ]
            np.add.at(target_node_counts[:, horizon_index], target_rows, 1)
        close_rows = positions[close_available[:, endpoint_target]]
        np.add.at(target_node_counts[:, -1], close_rows, 1)
        ticker_local = pd.to_datetime(ticker_timestamp_ns, utc=True).tz_convert("Asia/Seoul")
        ticker_clocks = np.asarray(ticker_local.hour * 60 + ticker_local.minute)
        expected_clock_fraction = (ticker_clocks - 9 * 60) / float(15 * 60 + 20 - 9 * 60)
        _require(bool(available[:, clock_feature].all()), f"{ticker} clock feature is unavailable")
        _require(bool(np.allclose(values[:, clock_feature], expected_clock_fraction, atol=2e-6, rtol=0.0)), f"{ticker} clock feature is inconsistent")

        physics_targets += _validate_target_physics(horizon_targets, horizon_available, ticker=ticker, label="horizon", tolerance=endpoint_tolerance)
        physics_targets += _validate_target_physics(close_targets, close_available, ticker=ticker, label="close", tolerance=endpoint_tolerance)
        for horizon_index, horizon in enumerate(horizons):
            future_ns = ticker_timestamp_ns + int(horizon) * 60 * 1_000_000_000
            future_rows = np.searchsorted(ticker_timestamp_ns, future_ns)
            matched = future_rows < rows
            matched_indices = np.flatnonzero(matched)
            matched[matched_indices] = ticker_timestamp_ns[future_rows[matched_indices]] == future_ns[matched_indices]
            source_rows = np.flatnonzero(matched)
            if not len(source_rows):
                continue
            destination_rows = future_rows[source_rows]
            _require(bool(horizon_available[source_rows, horizon_index, endpoint_target].all()), f"{ticker} endpoint target is missing despite an exact future decision price")
            expected = decision_price[destination_rows].astype(np.float64) / decision_price[source_rows].astype(np.float64) - 1.0
            observed = horizon_targets[source_rows, horizon_index, endpoint_target].astype(np.float64)
            error = np.abs(observed - expected)
            endpoint_comparisons += int(len(error))
            endpoint_max_error = max(endpoint_max_error, float(error.max(initial=0.0)))
            _require(bool((error <= endpoint_tolerance).all()), f"{ticker} endpoint return does not match future decision prices")

        _require(int(record["snapshots"]) == rows, f"{ticker} snapshot count mismatch")
        first = pd.Timestamp(int(ticker_timestamp_ns[0]), tz="UTC").tz_convert("Asia/Seoul").isoformat()
        last = pd.Timestamp(int(ticker_timestamp_ns[-1]), tz="UTC").tz_convert("Asia/Seoul").isoformat()
        _require(record["first_timestamp"] == first, f"{ticker} first timestamp mismatch")
        _require(record["last_timestamp"] == last, f"{ticker} last timestamp mismatch")
        _assert_close(float(record["input_availability"]), float(available.mean()), 1e-12, f"{ticker} input availability")
        _assert_close(float(record["horizon_target_availability"]), float(horizon_available.mean()), 1e-12, f"{ticker} horizon availability")
        _assert_close(float(record["close_target_availability"]), float(close_available.mean()), 1e-12, f"{ticker} close availability")
        total_snapshots += rows

    _require(bool(np.array_equal(reconstructed_counts, node_counts)), "global timestamp node counts do not match ticker shards")
    _require(
        bool((target_node_counts <= reconstructed_counts[:, None]).all()),
        "target node coverage exceeds source node coverage",
    )
    _require(int(manifest["total_node_snapshots"]) == total_snapshots, "total node snapshot count mismatch")
    _require(int(manifest["graph_timestamps"]) == len(timestamp_ns), "graph timestamp count mismatch")
    expected_first = pd.Timestamp(int(timestamp_ns[0]), tz="UTC").tz_convert("Asia/Seoul").isoformat()
    expected_last = pd.Timestamp(int(timestamp_ns[-1]), tz="UTC").tz_convert("Asia/Seoul").isoformat()
    _require(manifest["first_timestamp"] == expected_first, "manifest first timestamp mismatch")
    _require(manifest["last_timestamp"] == expected_last, "manifest last timestamp mismatch")
    coverage = manifest["node_coverage"]
    _require(int(coverage["minimum"]) == int(node_counts.min()), "minimum node coverage mismatch")
    _require(float(coverage["median"]) == float(np.median(node_counts)), "median node coverage mismatch")
    _require(int(coverage["maximum"]) == int(node_counts.max()), "maximum node coverage mismatch")

    raw = manifest["raw_cross_checks"]
    _require(int(raw["price_match_count"]) > 0, "raw price cross-check has no observations")
    _require(float(raw["price_match_ratio"]) >= minimum_price_match_ratio, "raw price match gate failed")
    _require(int(raw["volume_containment_count"]) > 0, "raw volume cross-check has no observations")
    _require(float(raw["volume_contained_by_daily_ratio"]) >= minimum_volume_contained_ratio, "raw volume containment gate failed")
    _require(float(raw["regular_to_daily_volume_coverage_median"]) >= minimum_median_volume_coverage, "raw median volume coverage gate failed")
    input_files = _verify_input_files(manifest, require_input_files=require_input_files)
    target_labels = tuple(f"{value}m" for value in horizons) + ("close",)
    target_coverage = {
        label: {
            "minimum": int(target_node_counts[:, index].min()),
            "p10": float(np.quantile(target_node_counts[:, index], 0.10)),
            "median": float(np.median(target_node_counts[:, index])),
            "p90": float(np.quantile(target_node_counts[:, index], 0.90)),
            "maximum": int(target_node_counts[:, index].max()),
            "timestamps_ge_250": int(
                (target_node_counts[:, index] >= 250).sum()
            ),
        }
        for index, label in enumerate(target_labels)
    }

    return {
        "schema_version": 1,
        "audit_contract": AUDIT_CONTRACT,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "release": str(root.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "passed": True,
        "portable_output_paths": True,
        "universe_stocks": len(universe),
        "trajectory_shards": len(shard_records),
        "missing_tickers": len(missing),
        "short_tickers": len(short),
        "graph_timestamps": len(timestamp_ns),
        "total_node_snapshots": total_snapshots,
        "node_coverage": {
            "minimum": int(node_counts.min()),
            "median": float(np.median(node_counts)),
            "maximum": int(node_counts.max()),
        },
        "endpoint_target_node_coverage": target_coverage,
        "physical_target_rows": physics_targets,
        "endpoint_identity_comparisons": endpoint_comparisons,
        "endpoint_identity_max_abs_error": endpoint_max_error,
        "input_files": input_files,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
