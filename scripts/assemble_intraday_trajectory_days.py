from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_trajectory import (
    IntradayTrajectoryPanel,
    summarize_systemic_intraday_trajectory,
)


ASSEMBLY_CONTRACT = "time_major_intraday_post_impact_days_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble ticker trajectory shards into compressed trading-day shards."
    )
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--minimum-nodes-per-timestamp", type=int, default=300)
    parser.add_argument("--minimum-timestamps-per-day", type=int, default=60)
    parser.add_argument("--minimum-days", type=int, default=200)
    parser.add_argument("--systemic-min-nodes", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidate = root / path
    if candidate.exists():
        return candidate
    return ROOT / path


def _open_work_array(
    directory: Path,
    name: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
    fill: float | int,
) -> np.memmap:
    result = np.lib.format.open_memmap(
        directory / f"{name}.npy", mode="w+", dtype=dtype, shape=shape
    )
    result[...] = fill
    return result


def _atomic_write_day(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if (
        int(args.minimum_nodes_per_timestamp) < 2
        or int(args.minimum_timestamps_per_day) <= 0
        or int(args.minimum_days) <= 0
        or int(args.systemic_min_nodes) < 2
    ):
        raise ValueError("assembly minimum counts must be positive and meaningful")
    release_dir = Path(args.release_dir)
    code_provenance = {
        "files": {
            "day_release_assembler": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "intraday_trajectory": {
                "path": str((ROOT / "stock_v2/intraday_trajectory.py").resolve()),
                "sha256": file_sha256(ROOT / "stock_v2/intraday_trajectory.py"),
            },
        },
        "runtime": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    source_manifest_path = release_dir / "manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("trajectory_contract") not in {
        "kiwoom_raw_rolling_post_impact_trajectory_v1",
        "kiwoom_raw_rolling_post_impact_trajectory_v2",
    }:
        raise ValueError("unsupported trajectory release contract")
    if source_manifest.get("live_orders_allowed") is not False:
        raise ValueError("trajectory source must explicitly prohibit live orders")
    universe_tickers = tuple(str(value) for value in source_manifest["universe_tickers"])
    if len(universe_tickers) != len(set(universe_tickers)):
        raise ValueError("trajectory universe contains duplicate tickers")
    ticker_position = {ticker: index for index, ticker in enumerate(universe_tickers)}

    index_record = source_manifest["outputs"]
    index_path = _resolve(release_dir, index_record["timestamp_index"])
    if file_sha256(index_path) != index_record["timestamp_index_sha256"]:
        raise ValueError("trajectory timestamp index checksum mismatch")
    with np.load(index_path) as index_bundle:
        timestamp_ns = index_bundle["timestamps_utc_ns"].astype(np.int64)
        source_node_counts = index_bundle["node_counts"].astype(np.int64)
    timestamps = pd.to_datetime(timestamp_ns, utc=True).tz_convert("Asia/Seoul")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("trajectory timestamp index must be unique and sorted")
    if len(source_node_counts) != len(timestamps):
        raise ValueError("trajectory node counts do not match timestamps")

    shard_records = source_manifest["outputs"]["shards"]
    if not shard_records:
        raise ValueError("trajectory release has no ticker shards")
    first_path = _resolve(release_dir, shard_records[0]["path"])
    with np.load(first_path) as first:
        feature_names = tuple(str(value) for value in first["feature_names"].tolist())
        horizons = tuple(int(value) for value in first["horizons_minutes"].tolist())
        target_names = tuple(str(value) for value in first["target_names"].tolist())
    time_count = len(timestamps)
    node_count = len(universe_tickers)
    horizon_count = len(horizons) + 1
    feature_count = len(feature_names)
    target_count = len(target_names)

    output_dir = Path(args.output_dir)
    temporary_dir = Path(str(output_dir) + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    work_dir = temporary_dir / "work"
    days_dir = temporary_dir / "days"
    work_dir.mkdir(parents=True)
    values = _open_work_array(
        work_dir,
        "node_values",
        (time_count, node_count, feature_count),
        np.float32,
        np.nan,
    )
    available = _open_work_array(
        work_dir,
        "node_available",
        (time_count, node_count, feature_count),
        np.uint8,
        0,
    )
    decision_price = _open_work_array(
        work_dir,
        "decision_price",
        (time_count, node_count),
        np.float32,
        np.nan,
    )
    targets = _open_work_array(
        work_dir,
        "targets",
        (time_count, node_count, horizon_count, target_count),
        np.float32,
        np.nan,
    )
    target_available = _open_work_array(
        work_dir,
        "target_available",
        (time_count, node_count, horizon_count, target_count),
        np.uint8,
        0,
    )

    populated_tickers: list[str] = []
    for shard_number, record in enumerate(shard_records, start=1):
        ticker = str(record["ticker"])
        if ticker not in ticker_position:
            raise ValueError(f"ticker shard is absent from universe: {ticker}")
        path = _resolve(release_dir, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"trajectory shard checksum mismatch for {ticker}")
        with np.load(path) as shard:
            if tuple(str(value) for value in shard["feature_names"].tolist()) != feature_names:
                raise ValueError(f"feature contract changed in shard {ticker}")
            if tuple(int(value) for value in shard["horizons_minutes"].tolist()) != horizons:
                raise ValueError(f"horizon contract changed in shard {ticker}")
            if tuple(str(value) for value in shard["target_names"].tolist()) != target_names:
                raise ValueError(f"target contract changed in shard {ticker}")
            rows = timestamps.get_indexer(
                pd.to_datetime(shard["timestamps_utc_ns"], utc=True).tz_convert(
                    "Asia/Seoul"
                )
            )
            if (rows < 0).any() or len(np.unique(rows)) != len(rows):
                raise ValueError(f"ticker timestamps failed to align for {ticker}")
            node = ticker_position[ticker]
            values[rows, node] = shard["values"].astype(np.float32)
            available[rows, node] = shard["available"].astype(np.uint8)
            decision_price[rows, node] = shard["decision_price"].astype(np.float32)
            targets[rows, node, :-1] = shard["horizon_targets"].astype(np.float32)
            targets[rows, node, -1] = shard["close_targets"].astype(np.float32)
            target_available[rows, node, :-1] = shard["horizon_available"].astype(
                np.uint8
            )
            target_available[rows, node, -1] = shard["close_available"].astype(
                np.uint8
            )
        populated_tickers.append(ticker)
        if shard_number % 25 == 0 or shard_number == len(shard_records):
            print(f"assembled={shard_number}/{len(shard_records)}", flush=True)
    for array in (values, available, decision_price, targets, target_available):
        array.flush()

    endpoint_index = target_names.index("endpoint_return")
    actual_node_counts = np.sum(
        target_available[:, :, 0, endpoint_index] > 0, axis=1
    ).astype(np.int64)
    if (actual_node_counts > source_node_counts).any():
        raise ValueError("assembled target coverage exceeds source node coverage")
    local_dates = timestamps.tz_localize(None).normalize()
    day_records: list[dict[str, Any]] = []
    for date in pd.DatetimeIndex(local_dates.unique()).sort_values():
        rows = np.flatnonzero(local_dates == date)
        qualified = source_node_counts[rows] >= int(args.minimum_nodes_per_timestamp)
        selected = rows[qualified]
        if len(selected) < int(args.minimum_timestamps_per_day):
            continue
        day_timestamps = timestamps[selected]
        expected_clock = np.arange(9 * 60 + 15, 15 * 60 + 15 + 1, 5)
        actual_clock = day_timestamps.hour * 60 + day_timestamps.minute
        if len(actual_clock) != len(np.unique(actual_clock)):
            raise ValueError(f"duplicate decision clocks on {date.date()}")
        if not np.isin(actual_clock, expected_clock).all():
            raise ValueError(f"unexpected decision clock on {date.date()}")

        day_panel = IntradayTrajectoryPanel(
            timestamps=day_timestamps,
            session_dates=pd.DatetimeIndex([date] * len(selected)),
            decision_clock_minutes=np.asarray(actual_clock, dtype=np.int16),
            tickers=universe_tickers,
            feature_names=feature_names,
            values=np.asarray(values[selected]),
            available=np.asarray(available[selected], dtype=bool),
            decision_price=np.asarray(decision_price[selected]),
            horizons_minutes=horizons,
            target_names=target_names,
            horizon_targets=np.asarray(targets[selected, :, :-1]),
            horizon_available=np.asarray(
                target_available[selected, :, :-1], dtype=bool
            ),
            close_targets=np.asarray(targets[selected, :, -1]),
            close_available=np.asarray(
                target_available[selected, :, -1], dtype=bool
            ),
        )
        systemic = summarize_systemic_intraday_trajectory(
            day_panel, min_nodes=int(args.systemic_min_nodes)
        )
        day_path = days_dir / f"{date.date()}.npz"
        _atomic_write_day(
            day_path,
            {
                "timestamps_utc_ns": day_timestamps.tz_convert("UTC").asi8,
                "node_values": np.asarray(values[selected], dtype=np.float32),
                "node_available": np.asarray(available[selected], dtype=np.uint8),
                "decision_price": np.asarray(
                    decision_price[selected], dtype=np.float32
                ),
                "targets": np.asarray(targets[selected], dtype=np.float32),
                "target_available": np.asarray(
                    target_available[selected], dtype=np.uint8
                ),
                "systemic_targets": systemic.values.astype(np.float32),
                "systemic_available": systemic.available.astype(np.uint8),
            },
        )
        day_records.append(
            {
                "date": str(date.date()),
                "path": str(Path("days") / day_path.name),
                "sha256": file_sha256(day_path),
                "timestamps": len(selected),
                "median_source_nodes": float(np.median(source_node_counts[selected])),
                "median_target_nodes_h5": float(np.median(actual_node_counts[selected])),
                "bytes": day_path.stat().st_size,
            }
        )
    if len(day_records) < int(args.minimum_days):
        raise ValueError(f"only {len(day_records)} day shards pass assembly gates")

    metadata_path = temporary_dir / "metadata.npz"
    with metadata_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            tickers=np.asarray(universe_tickers, dtype="U6"),
            populated_tickers=np.asarray(populated_tickers, dtype="U6"),
            feature_names=np.asarray(feature_names, dtype="U64"),
            horizons_minutes=np.asarray(horizons, dtype=np.int16),
            horizon_labels=np.asarray(
                tuple(f"{value}m" for value in horizons) + ("close",), dtype="U16"
            ),
            target_names=np.asarray(target_names, dtype="U64"),
            systemic_target_names=np.asarray(systemic.target_names, dtype="U64"),
        )
    shutil.rmtree(work_dir)
    manifest = {
        "schema_version": 1,
        "assembly_contract": ASSEMBLY_CONTRACT,
        "source_manifest": str(source_manifest_path),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_trajectory_contract": source_manifest["trajectory_contract"],
        "stocks": node_count,
        "populated_stocks": len(populated_tickers),
        "features": feature_count,
        "horizons": list(horizons) + ["close"],
        "targets": list(target_names),
        "days": len(day_records),
        "first_date": day_records[0]["date"],
        "last_date": day_records[-1]["date"],
        "gates": {
            "minimum_nodes_per_timestamp": int(args.minimum_nodes_per_timestamp),
            "minimum_timestamps_per_day": int(args.minimum_timestamps_per_day),
            "minimum_days": int(args.minimum_days),
            "systemic_min_nodes": int(args.systemic_min_nodes),
        },
        "metadata": {
            "path": metadata_path.name,
            "sha256": file_sha256(metadata_path),
        },
        "code_provenance": code_provenance,
        "h5_endpoint_target_node_coverage": {
            "minimum": int(actual_node_counts.min()),
            "p10": float(np.quantile(actual_node_counts, 0.10)),
            "median": float(np.median(actual_node_counts)),
            "p90": float(np.quantile(actual_node_counts, 0.90)),
            "maximum": int(actual_node_counts.max()),
        },
        "day_shards": day_records,
        "causality": source_manifest["causality"],
        "portable_payload_paths": True,
        "transactional_publish": True,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    manifest_path = temporary_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    if output_dir.exists():
        previous = Path(str(output_dir) + ".previous")
        if previous.exists():
            shutil.rmtree(previous)
        output_dir.replace(previous)
        try:
            temporary_dir.replace(output_dir)
        except Exception:
            previous.replace(output_dir)
            raise
        shutil.rmtree(previous)
    else:
        temporary_dir.replace(output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(output_dir),
                "days": len(day_records),
                "stocks": node_count,
                "populated_stocks": len(populated_tickers),
                "first_date": day_records[0]["date"],
                "last_date": day_records[-1]["date"],
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
