from __future__ import annotations

import argparse
from collections import Counter
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

from scripts.build_intraday_sensing_release import (
    canonical_sha256,
    file_sha256,
    load_coverage,
    load_semantics_evidence,
    load_universe,
    read_daily_raw,
    read_minute_frame,
    select_coverage_record,
)
from stock_v2.intraday_sensing import summarize_session_targets
from stock_v2.intraday_trajectory import (
    TickerIntradayTrajectory,
    build_ticker_intraday_trajectory,
)


TRAJECTORY_CONTRACT = "kiwoom_raw_rolling_post_impact_trajectory_v2"


class InsufficientContextError(ValueError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable rolling post-impact trajectory shards."
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--daily-ohlcv-dir", required=True)
    parser.add_argument("--daily-close-column", default="RawClose")
    parser.add_argument("--daily-volume-column", default="RawVolume")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--output-start",
        help="First session written to the release; defaults to --start.",
    )
    parser.add_argument(
        "--output-end",
        help="Last session written to the release; defaults to --end.",
    )
    parser.add_argument(
        "--context-sessions",
        type=int,
        default=0,
        help=(
            "Retain this many observed sessions before --output-start for "
            "rolling feature computation, then write only the output interval."
        ),
    )
    parser.add_argument("--interval-minutes", type=int, required=True)
    parser.add_argument("--timestamp-semantics", choices=["start", "end"], required=True)
    parser.add_argument("--timestamp-semantics-evidence", required=True)
    parser.add_argument("--decision-start", default="09:15")
    parser.add_argument("--decision-end", default="15:15")
    parser.add_argument("--horizons-minutes", default="5,15,30,60")
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--minimum-ticker-files", type=int, default=400)
    parser.add_argument("--minimum-snapshots-per-ticker", type=int, default=1000)
    parser.add_argument("--price-relative-tolerance", type=float, default=1e-6)
    parser.add_argument("--volume-relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-price-match-ratio", type=float, default=0.995)
    parser.add_argument("--minimum-volume-contained-ratio", type=float, default=0.995)
    parser.add_argument(
        "--minimum-median-regular-volume-coverage", type=float, default=0.80
    )
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def parse_horizons(value: str, interval_minutes: int) -> tuple[int, ...]:
    try:
        horizons = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("horizons-minutes must be comma-separated integers") from exc
    if (
        not horizons
        or tuple(sorted(horizons)) != horizons
        or len(set(horizons)) != len(horizons)
        or any(item <= 0 or item % int(interval_minutes) for item in horizons)
    ):
        raise ValueError("horizons must be positive, sorted interval multiples")
    return horizons


def _matched_count(
    left: np.ndarray,
    right: np.ndarray,
    tolerance: float,
) -> tuple[int, int, list[float]]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 0.0)
    relative = np.abs(left[valid] - right[valid]) / np.abs(right[valid])
    return int((relative <= float(tolerance)).sum()), int(len(relative)), relative.tolist()


def _volume_containment(
    minute_volume: np.ndarray,
    daily_volume: np.ndarray,
    tolerance: float,
) -> tuple[int, int, list[float]]:
    minute = np.asarray(minute_volume, dtype=np.float64)
    daily = np.asarray(daily_volume, dtype=np.float64)
    valid = (
        np.isfinite(minute)
        & np.isfinite(daily)
        & (minute >= 0.0)
        & (daily > 0.0)
    )
    coverage = minute[valid] / daily[valid]
    contained = minute[valid] <= daily[valid] * (1.0 + float(tolerance))
    return int(contained.sum()), int(len(coverage)), coverage.tolist()


def _write_ticker_shard(path: Path, trajectory: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            timestamps_utc_ns=trajectory.timestamps.tz_convert("UTC").asi8,
            feature_names=np.asarray(trajectory.feature_names, dtype="U64"),
            values=trajectory.values.astype(np.float32),
            available=trajectory.available.astype(np.uint8),
            decision_price=trajectory.decision_price.astype(np.float32),
            horizons_minutes=np.asarray(trajectory.horizons_minutes, dtype=np.int16),
            target_names=np.asarray(trajectory.target_names, dtype="U64"),
            horizon_targets=trajectory.horizon_targets.astype(np.float32),
            horizon_available=trajectory.horizon_available.astype(np.uint8),
            close_targets=trajectory.close_targets.astype(np.float32),
            close_available=trajectory.close_available.astype(np.uint8),
        )
    temporary.replace(path)


def _select_context_frame(
    frame: pd.DataFrame,
    *,
    input_start: pd.Timestamp,
    input_end: pd.Timestamp,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
    context_sessions: int,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    local_dates = frame.index.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    bounded = frame.loc[(local_dates >= input_start) & (local_dates <= input_end)]
    bounded_dates = (
        bounded.index.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    )
    context_start = output_start
    if int(context_sessions) > 0:
        prior = pd.DatetimeIndex(
            bounded_dates[bounded_dates < output_start].unique()
        ).sort_values()
        if len(prior):
            retained = min(len(prior), int(context_sessions))
            context_start = pd.Timestamp(prior[-retained]).normalize()
    selected = bounded.loc[
        (bounded_dates >= context_start) & (bounded_dates <= output_end)
    ]
    selected_dates = (
        selected.index.tz_convert("Asia/Seoul").tz_localize(None).normalize()
    )
    output_rows = (selected_dates >= output_start) & (selected_dates <= output_end)
    if not bool(output_rows.any()):
        raise InsufficientContextError(
            "minute frame has no observations in the output interval"
        )
    return selected, context_start


def _slice_trajectory(
    trajectory: TickerIntradayTrajectory,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    minimum_input_snapshots: int = 0,
) -> TickerIntradayTrajectory:
    if len(trajectory.timestamps) < int(minimum_input_snapshots):
        raise InsufficientContextError(
            f"trajectory has only {len(trajectory.timestamps)} input snapshots; "
            f"require {int(minimum_input_snapshots)}"
        )
    local_dates = (
        trajectory.timestamps.tz_convert("Asia/Seoul")
        .tz_localize(None)
        .normalize()
    )
    selected = np.asarray((local_dates >= start) & (local_dates <= end), dtype=bool)
    if not selected.any():
        raise InsufficientContextError(
            "trajectory has no decision rows in the output interval"
        )
    return TickerIntradayTrajectory(
        timestamps=trajectory.timestamps[selected],
        feature_names=trajectory.feature_names,
        values=trajectory.values[selected],
        available=trajectory.available[selected],
        decision_price=trajectory.decision_price[selected],
        horizons_minutes=trajectory.horizons_minutes,
        target_names=trajectory.target_names,
        horizon_targets=trajectory.horizon_targets[selected],
        horizon_available=trajectory.horizon_available[selected],
        close_targets=trajectory.close_targets[selected],
        close_available=trajectory.close_available[selected],
    )


def _prior_same_clock_context_counts(
    trajectory: TickerIntradayTrajectory,
    output_start: pd.Timestamp,
    output_end: pd.Timestamp,
) -> dict[int, int]:
    local = trajectory.timestamps.tz_convert("Asia/Seoul")
    local_dates = local.tz_localize(None).normalize()
    output_rows = np.asarray(
        (local_dates >= output_start) & (local_dates <= output_end), dtype=bool
    )
    if not output_rows.any():
        return {}
    clocks = np.asarray(local.hour * 60 + local.minute, dtype=np.int16)
    prior_rows = np.asarray(local_dates < output_start, dtype=bool)
    return {
        int(clock): int(np.count_nonzero(prior_rows & (clocks == clock)))
        for clock in np.unique(clocks[output_rows])
    }


def main() -> int:
    args = parse_args()
    if args.minimum_ticker_files <= 0 or args.minimum_snapshots_per_ticker <= 0:
        raise ValueError("minimum ticker files and snapshots must be positive")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("end must not precede start")
    output_start = pd.Timestamp(args.output_start or args.start).normalize()
    output_end = pd.Timestamp(args.output_end or args.end).normalize()
    if output_end < output_start:
        raise ValueError("output end must not precede output start")
    if output_start < start or output_end > end:
        raise ValueError("output interval must be contained in the input interval")
    if int(args.context_sessions) < 0:
        raise ValueError("context sessions must be non-negative")
    if 0 < int(args.context_sessions) < int(args.rolling_window):
        raise ValueError("context sessions must cover the full rolling window")
    horizons = parse_horizons(args.horizons_minutes, args.interval_minutes)

    coverage_path = Path(args.coverage)
    universe_path = Path(args.universe_manifest)
    daily_dir = Path(args.daily_ohlcv_dir)
    evidence_path = Path(args.timestamp_semantics_evidence)
    output_dir = Path(args.output_dir)
    temporary_dir = Path(str(output_dir) + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    shards_dir = temporary_dir / "tickers"
    code_provenance = {
        "files": {
            "trajectory_release_builder": {
                "path": str(Path(__file__).resolve()),
                "sha256": file_sha256(Path(__file__).resolve()),
            },
            "intraday_sensing_release_reader": {
                "path": str((ROOT / "scripts/build_intraday_sensing_release.py").resolve()),
                "sha256": file_sha256(ROOT / "scripts/build_intraday_sensing_release.py"),
            },
            "intraday_trajectory": {
                "path": str((ROOT / "stock_v2/intraday_trajectory.py").resolve()),
                "sha256": file_sha256(ROOT / "stock_v2/intraday_trajectory.py"),
            },
            "intraday_sensing": {
                "path": str((ROOT / "stock_v2/intraday_sensing.py").resolve()),
                "sha256": file_sha256(ROOT / "stock_v2/intraday_sensing.py"),
            },
            "kiwoom_minute": {
                "path": str((ROOT / "stock_v2/kiwoom_minute.py").resolve()),
                "sha256": file_sha256(ROOT / "stock_v2/kiwoom_minute.py"),
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
    load_semantics_evidence(
        evidence_path,
        interval_minutes=args.interval_minutes,
        timestamp_semantics=args.timestamp_semantics,
    )
    universe = load_universe(universe_path, max(0, args.max_tickers))
    coverage = load_coverage(coverage_path)

    minute_sources: list[dict[str, Any]] = []
    daily_sources: list[dict[str, Any]] = []
    shard_records: list[dict[str, Any]] = []
    missing_tickers: list[str] = []
    short_tickers: list[str] = []
    timestamp_node_counts: Counter[int] = Counter()
    price_match = price_total = volume_contained = volume_total = 0
    price_errors: list[float] = []
    volume_errors: list[float] = []
    total_snapshots = 0

    for universe_row in universe:
        ticker = universe_row["ticker"]
        coverage_record = select_coverage_record(
            coverage,
            ticker=ticker,
            interval_minutes=args.interval_minutes,
            start=args.start,
            end=args.end,
            run_id=args.run_id,
        )
        if coverage_record is None:
            missing_tickers.append(ticker)
            continue
        minute_path = Path(str(coverage_record["output"]))
        if not minute_path.is_absolute():
            minute_path = ROOT / minute_path
        minute_sha = file_sha256(minute_path)
        if minute_sha != coverage_record["output_sha256"]:
            raise ValueError(f"minute file checksum mismatch for {ticker}")
        frame = read_minute_frame(minute_path)
        frame_dates = frame.index.tz_convert("Asia/Seoul").tz_localize(None).normalize()
        available_prior_sessions = int(
            pd.DatetimeIndex(
                frame_dates[(frame_dates >= start) & (frame_dates < output_start)]
            ).nunique()
        )
        all_prior_history_retained = (
            int(args.context_sessions) >= available_prior_sessions
        )
        try:
            frame, ticker_context_start = _select_context_frame(
                frame,
                input_start=start,
                input_end=end,
                output_start=output_start,
                output_end=output_end,
                context_sessions=int(args.context_sessions),
            )
        except InsufficientContextError:
            short_tickers.append(ticker)
            continue
        trajectory = build_ticker_intraday_trajectory(
            frame,
            interval_minutes=args.interval_minutes,
            timestamp_semantics=args.timestamp_semantics,
            horizons_minutes=horizons,
            decision_start=args.decision_start,
            decision_end=args.decision_end,
            rolling_window=args.rolling_window,
            min_history=args.min_history,
        )
        context_counts = _prior_same_clock_context_counts(
            trajectory, output_start, output_end
        )
        if (
            context_counts
            and not all_prior_history_retained
            and min(context_counts.values()) < int(args.rolling_window)
        ):
            minimum = min(context_counts.values())
            raise ValueError(
                f"context sessions are insufficient for {ticker}: only {minimum} "
                f"prior same-clock records; require {int(args.rolling_window)} or "
                "all available prior history"
            )
        try:
            trajectory = _slice_trajectory(
                trajectory,
                output_start,
                output_end,
                minimum_input_snapshots=int(args.minimum_snapshots_per_ticker),
            )
        except InsufficientContextError:
            short_tickers.append(ticker)
            continue

        daily, daily_path = read_daily_raw(
            daily_dir,
            ticker,
            output_start,
            output_end,
            close_column=args.daily_close_column,
            volume_column=args.daily_volume_column,
        )
        session = summarize_session_targets(
            frame,
            interval_minutes=args.interval_minutes,
            timestamp_semantics=args.timestamp_semantics,
        )
        session = session.loc[session["SessionComplete"].fillna(False)].copy()
        session = session.loc[
            (session.index >= output_start) & (session.index <= output_end)
        ]
        joined = session[["SessionClose", "SessionVolume"]].join(
            daily[["RawClose", "RawVolume"]], how="inner"
        )
        matched, count, errors = _matched_count(
            joined["SessionClose"].to_numpy(),
            joined["RawClose"].to_numpy(),
            args.price_relative_tolerance,
        )
        price_match += matched
        price_total += count
        price_errors.extend(errors)
        contained, count, coverage_values = _volume_containment(
            joined["SessionVolume"].to_numpy(),
            joined["RawVolume"].to_numpy(),
            args.volume_relative_tolerance,
        )
        volume_contained += contained
        volume_total += count
        volume_errors.extend(coverage_values)

        shard_path = shards_dir / f"{ticker}.npz"
        _write_ticker_shard(shard_path, trajectory)
        shard_sha = file_sha256(shard_path)
        timestamp_ns = trajectory.timestamps.tz_convert("UTC").asi8
        timestamp_node_counts.update(int(value) for value in timestamp_ns)
        total_snapshots += len(trajectory.timestamps)
        minute_sources.append(
            {
                "ticker": ticker,
                "path": str(minute_path),
                "sha256": minute_sha,
                "coverage_status": coverage_record["status"],
                "coverage_run_id": coverage_record["run_id"],
                "context_start": str(ticker_context_start.date()),
                "input_sessions_retained": int(
                    frame.index.tz_convert("Asia/Seoul").normalize().nunique()
                ),
                "available_prior_sessions": available_prior_sessions,
                "all_prior_history_retained": all_prior_history_retained,
                "minimum_prior_same_clock_records": (
                    min(context_counts.values()) if context_counts else 0
                ),
            }
        )
        daily_sources.append(
            {"ticker": ticker, "path": str(daily_path), "sha256": file_sha256(daily_path)}
        )
        shard_records.append(
            {
                "ticker": ticker,
                "path": shard_path.relative_to(temporary_dir).as_posix(),
                "sha256": shard_sha,
                "snapshots": len(trajectory.timestamps),
                "first_timestamp": trajectory.timestamps[0].isoformat(),
                "last_timestamp": trajectory.timestamps[-1].isoformat(),
                "input_availability": float(trajectory.available.mean()),
                "horizon_target_availability": float(
                    trajectory.horizon_available.mean()
                ),
                "close_target_availability": float(trajectory.close_available.mean()),
            }
        )
        print(
            f"ticker={ticker} snapshots={len(trajectory.timestamps)} "
            f"complete={len(shard_records)}/{len(universe)}",
            flush=True,
        )

    if len(shard_records) < int(args.minimum_ticker_files):
        raise ValueError(
            f"only {len(shard_records)} trajectory shards; require {args.minimum_ticker_files}"
        )
    price_ratio = float(price_match / price_total) if price_total else 0.0
    volume_ratio = float(volume_contained / volume_total) if volume_total else 0.0
    median_volume_coverage = float(np.median(volume_errors)) if volume_errors else 0.0
    if price_ratio < float(args.minimum_price_match_ratio):
        raise ValueError(f"minute/daily close match ratio {price_ratio:.6f} is below gate")
    if volume_ratio < float(args.minimum_volume_contained_ratio):
        raise ValueError(
            f"regular minute volume containment {volume_ratio:.6f} is below gate"
        )
    if median_volume_coverage < float(args.minimum_median_regular_volume_coverage):
        raise ValueError(
            "median regular-session volume coverage "
            f"{median_volume_coverage:.6f} is below gate"
        )

    timestamps_ns = np.asarray(sorted(timestamp_node_counts), dtype=np.int64)
    node_counts = np.asarray(
        [timestamp_node_counts[int(value)] for value in timestamps_ns], dtype=np.int16
    )
    index_path = temporary_dir / "timestamp_index.npz"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(index_path) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            timestamps_utc_ns=timestamps_ns,
            node_counts=node_counts,
        )
    temporary.replace(index_path)

    manifest = {
        "schema_version": 1,
        "trajectory_contract": TRAJECTORY_CONTRACT,
        "source": "kiwoom_rest_ka10080",
        "basis": "raw",
        "coverage_run_id": args.run_id,
        "interval_minutes": int(args.interval_minutes),
        "timestamp_semantics": args.timestamp_semantics,
        "decision_start": args.decision_start,
        "decision_end": args.decision_end,
        "horizons_minutes": list(horizons),
        "close_horizon": "same_session_1530_closing_auction",
        "rolling_window": int(args.rolling_window),
        "min_history": int(args.min_history),
        "input_start": str(start.date()),
        "input_end": str(end.date()),
        "output_start": str(output_start.date()),
        "output_end": str(output_end.date()),
        "context_sessions": int(args.context_sessions),
        "stocks": len(shard_records),
        "universe_stocks": len(universe),
        "universe_tickers": [row["ticker"] for row in universe],
        "shard_tickers": [row["ticker"] for row in shard_records],
        "total_node_snapshots": int(total_snapshots),
        "graph_timestamps": int(len(timestamps_ns)),
        "first_timestamp": pd.Timestamp(timestamps_ns[0], tz="UTC").tz_convert(
            "Asia/Seoul"
        ).isoformat(),
        "last_timestamp": pd.Timestamp(timestamps_ns[-1], tz="UTC").tz_convert(
            "Asia/Seoul"
        ).isoformat(),
        "node_coverage": {
            "minimum": int(node_counts.min()),
            "median": float(np.median(node_counts)),
            "maximum": int(node_counts.max()),
        },
        "missing_tickers": missing_tickers,
        "short_tickers": short_tickers,
        "raw_cross_checks": {
            "price_match_count": price_total,
            "price_match_ratio": price_ratio,
            "price_relative_tolerance": float(args.price_relative_tolerance),
            "price_relative_error_p99": float(np.quantile(price_errors, 0.99)),
            "volume_containment_count": volume_total,
            "volume_contained_by_daily_ratio": volume_ratio,
            "volume_containment_relative_tolerance": float(
                args.volume_relative_tolerance
            ),
            "regular_to_daily_volume_coverage_p10": float(
                np.quantile(volume_errors, 0.10)
            ),
            "regular_to_daily_volume_coverage_median": median_volume_coverage,
            "regular_to_daily_volume_coverage_p99": float(
                np.quantile(volume_errors, 0.99)
            ),
        },
        "causality": {
            "decision_bar_excluded_from_start_labelled_inputs": True,
            "targets_begin_strictly_after_decision": True,
            "same_clock_baselines_shifted_one_session": True,
            "missing_bars_never_filled": True,
            "mfe_mae_use_only_post_decision_bars": True,
            "close_auction_required_for_close_target": True,
            "post_gap_decisions_require_a_fresh_completed_bar": True,
            "post_gap_cumulative_features_remain_masked": True,
            "endpoint_return_requires_exact_horizon_price_only": True,
            "path_targets_require_contiguous_future_bars": True,
            "daily_volume_can_include_post_close_trades": True,
        },
        "inputs": {
            "coverage": str(coverage_path),
            "coverage_sha256": file_sha256(coverage_path),
            "universe": str(universe_path),
            "universe_sha256": file_sha256(universe_path),
            "semantics_evidence": str(evidence_path),
            "semantics_evidence_sha256": file_sha256(evidence_path),
            "minute_sources_sha256": canonical_sha256(minute_sources),
            "daily_sources_sha256": canonical_sha256(daily_sources),
            "daily_close_column": args.daily_close_column,
            "daily_volume_column": args.daily_volume_column,
        },
        "code_provenance": code_provenance,
        "outputs": {
            "timestamp_index": index_path.relative_to(temporary_dir).as_posix(),
            "timestamp_index_sha256": file_sha256(index_path),
            "shards_sha256": canonical_sha256(shard_records),
            "shards": shard_records,
        },
        "portable_output_paths": True,
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
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
