from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_sensing import (
    build_intraday_market_design,
    build_intraday_window_panel,
    remaining_session_returns,
    summarize_early_window,
    summarize_session_targets,
)
from stock_v2.kiwoom_minute import audit_kiwoom_minute_frame


SENSOR_CONTRACT = "kiwoom_raw_early_window_remaining_session_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a PIT-audited early-session sensing release from ka10080."
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument(
        "--run-id",
        default=None,
        help="Require coverage records from this exact immutable collection run.",
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--daily-ohlcv-dir", required=True)
    parser.add_argument("--daily-close-column", default="RawClose")
    parser.add_argument("--daily-volume-column", default="RawVolume")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval-minutes", type=int, required=True)
    parser.add_argument("--decision-time", default="09:15")
    parser.add_argument("--timestamp-semantics", choices=["start", "end"], required=True)
    parser.add_argument("--timestamp-semantics-evidence", required=True)
    parser.add_argument("--rolling-window", type=int, default=20)
    parser.add_argument("--min-history", type=int, default=10)
    parser.add_argument("--minimum-ticker-files", type=int, default=400)
    parser.add_argument("--minimum-date-stock-coverage", type=float, default=0.80)
    parser.add_argument(
        "--minimum-target-date-stock-coverage", type=float, default=0.80
    )
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_universe(path: Path, max_tickers: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe")
    selected = rows[: max_tickers or None]
    result = []
    for row in selected:
        ticker = str(row.get("ticker", "")).replace("A", "").strip().zfill(6)
        if len(ticker) != 6 or not ticker.isdigit():
            raise ValueError(f"invalid universe ticker: {ticker!r}")
        result.append(dict(row, ticker=ticker))
    tickers = [row["ticker"] for row in result]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe contains duplicate tickers")
    return result


def load_semantics_evidence(
    path: Path,
    *,
    interval_minutes: int,
    timestamp_semantics: str,
) -> dict[str, Any]:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    checks = {
        "verified": evidence.get("verified") is True,
        "api_id": evidence.get("api_id") == "ka10080",
        "basis": evidence.get("basis") == "raw",
        "interval": int(evidence.get("interval_minutes", -1)) == int(interval_minutes),
        "semantics": evidence.get("timestamp_semantics") == timestamp_semantics,
        "response_hash": bool(evidence.get("source_response_sha256")),
        "method": evidence.get("verification_method")
        in {"live_completed_bar_probe", "daily_close_and_live_completed_bar_probe"},
    }
    if not all(checks.values()):
        failures = [name for name, passed in checks.items() if not passed]
        raise ValueError(f"timestamp semantics evidence failed: {failures}")
    return evidence


def load_coverage(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                record = json.loads(line)
                if isinstance(record, dict):
                    records.append(record)
    if not records:
        raise ValueError("minute coverage file is empty")
    return records


def select_coverage_record(
    records: list[dict[str, Any]],
    *,
    ticker: str,
    interval_minutes: int,
    start: str,
    end: str,
    run_id: str | None = None,
) -> dict[str, Any] | None:
    matches = [
        record
        for record in records
        if str(record.get("ticker")) == ticker
        and int(record.get("interval_minutes", -1)) == int(interval_minutes)
        and record.get("basis") == "raw"
        and str(record.get("requested_start")) == start
        and str(record.get("requested_end")) == end
        and (run_id is None or str(record.get("run_id")) == str(run_id))
        and record.get("status") in {"ok", "partial"}
        and record.get("output")
        and record.get("output_sha256")
    ]
    if len(matches) > 1:
        raise ValueError(f"multiple minute coverage records for {ticker}")
    return matches[0] if matches else None


def read_minute_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"unsupported minute file format: {path}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.tz is None:
        raise ValueError(f"minute file has timezone-naive timestamps: {path}")
    frame.index.name = "Timestamp"
    audit_kiwoom_minute_frame(frame, regular_session_only=True)
    return frame


def read_daily_raw(
    directory: Path,
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    close_column: str = "RawClose",
    volume_column: str = "RawVolume",
) -> tuple[pd.DataFrame, Path]:
    close_column = str(close_column).strip()
    volume_column = str(volume_column).strip()
    if (
        not close_column
        or not volume_column
        or close_column == volume_column
        or "Date" in {close_column, volume_column}
    ):
        raise ValueError("daily close and volume columns must be distinct non-Date names")
    paths = sorted(directory.glob(f"{ticker}_*.csv"))
    if len(paths) != 1:
        raise ValueError(f"expected exactly one daily OHLCV file for {ticker}, got {len(paths)}")
    path = paths[0]
    frame = pd.read_csv(path, usecols=["Date", close_column, volume_column])
    frame = frame.rename(
        columns={close_column: "RawClose", volume_column: "RawVolume"}
    )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    if frame["Date"].duplicated().any():
        raise ValueError(f"daily OHLCV contains duplicate dates for {ticker}")
    frame = frame.set_index("Date").sort_index().loc[start:end]
    return frame, path


def _relative_match(
    left: np.ndarray,
    right: np.ndarray,
    *,
    tolerance: float,
) -> tuple[float, int, np.ndarray]:
    valid = np.isfinite(left) & np.isfinite(right) & (np.abs(right) > 0.0)
    relative = np.full(left.shape, np.nan, dtype=np.float64)
    relative[valid] = np.abs(left[valid] - right[valid]) / np.abs(right[valid])
    count = int(valid.sum())
    ratio = float((relative[valid] <= float(tolerance)).mean()) if count else 0.0
    return ratio, count, relative


def _volume_containment(
    minute_volume: np.ndarray,
    daily_volume: np.ndarray,
    *,
    tolerance: float,
) -> tuple[float, np.ndarray]:
    valid = (
        np.isfinite(minute_volume)
        & np.isfinite(daily_volume)
        & (minute_volume >= 0.0)
        & (daily_volume > 0.0)
    )
    coverage = np.full(minute_volume.shape, np.nan, dtype=np.float64)
    coverage[valid] = minute_volume[valid] / daily_volume[valid]
    ratio = (
        float(
            (
                minute_volume[valid]
                <= daily_volume[valid] * (1.0 + float(tolerance))
            ).mean()
        )
        if valid.any()
        else 0.0
    )
    return ratio, coverage


def main() -> int:
    args = parse_args()
    if not 0.0 < args.minimum_date_stock_coverage <= 1.0:
        raise ValueError("minimum-date-stock-coverage must be in (0, 1]")
    if not 0.0 < args.minimum_target_date_stock_coverage <= 1.0:
        raise ValueError("minimum-target-date-stock-coverage must be in (0, 1]")
    if args.minimum_ticker_files <= 0:
        raise ValueError("minimum-ticker-files must be positive")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("end must not precede start")

    coverage_path = Path(args.coverage)
    universe_path = Path(args.universe_manifest)
    daily_dir = Path(args.daily_ohlcv_dir)
    evidence_path = Path(args.timestamp_semantics_evidence)
    output_dir = Path(args.output_dir)
    evidence = load_semantics_evidence(
        evidence_path,
        interval_minutes=args.interval_minutes,
        timestamp_semantics=args.timestamp_semantics,
    )
    universe = load_universe(universe_path, max(0, args.max_tickers))
    tickers = [row["ticker"] for row in universe]
    market_by_ticker = {row["ticker"]: str(row.get("market", "")) for row in universe}
    coverage = load_coverage(coverage_path)

    daily_frames: dict[str, pd.DataFrame] = {}
    daily_sources = []
    all_dates: set[pd.Timestamp] = set()
    for ticker in tickers:
        daily, daily_path = read_daily_raw(
            daily_dir,
            ticker,
            start,
            end,
            close_column=args.daily_close_column,
            volume_column=args.daily_volume_column,
        )
        daily_frames[ticker] = daily
        all_dates.update(pd.DatetimeIndex(daily.index).tolist())
        daily_sources.append(
            {"ticker": ticker, "path": str(daily_path), "sha256": file_sha256(daily_path)}
        )
    dates = pd.DatetimeIndex(sorted(all_dates))
    if not len(dates):
        raise ValueError("daily source has no dates in the requested range")

    raw_close = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    raw_volume = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for ticker, frame in daily_frames.items():
        raw_close.loc[:, ticker] = frame["RawClose"].reindex(dates)
        raw_volume.loc[:, ticker] = frame["RawVolume"].reindex(dates)
    daily_close_values = raw_close.to_numpy(dtype=np.float64)
    daily_volume_values = raw_volume.to_numpy(dtype=np.float64)
    eligible = (
        np.isfinite(daily_close_values)
        & (daily_close_values > 0.0)
        & np.isfinite(daily_volume_values)
        & (daily_volume_values >= 0.0)
    )

    early_summaries: dict[str, pd.DataFrame] = {}
    session_targets: dict[str, pd.DataFrame] = {}
    minute_sources = []
    missing_tickers = []
    for ticker in tickers:
        record = select_coverage_record(
            coverage,
            ticker=ticker,
            interval_minutes=args.interval_minutes,
            start=args.start,
            end=args.end,
            run_id=args.run_id,
        )
        if record is None:
            missing_tickers.append(ticker)
            continue
        path = Path(str(record["output"]))
        if not path.is_absolute():
            path = ROOT / path
        actual_sha = file_sha256(path)
        if actual_sha != record["output_sha256"]:
            raise ValueError(f"minute file checksum mismatch for {ticker}")
        frame = read_minute_frame(path)
        local_dates = frame.index.tz_convert("Asia/Seoul").tz_localize(None).normalize()
        frame = frame.loc[(local_dates >= start) & (local_dates <= end)]
        early_summaries[ticker] = summarize_early_window(
            frame,
            interval_minutes=args.interval_minutes,
            decision_time=args.decision_time,
            timestamp_semantics=args.timestamp_semantics,
        )
        session_targets[ticker] = summarize_session_targets(
            frame,
            interval_minutes=args.interval_minutes,
            timestamp_semantics=args.timestamp_semantics,
        )
        minute_sources.append(
            {
                "ticker": ticker,
                "path": str(path),
                "sha256": actual_sha,
                "coverage_status": record["status"],
            }
        )
    if len(minute_sources) < int(args.minimum_ticker_files):
        raise ValueError(
            f"only {len(minute_sources)} minute files; require {args.minimum_ticker_files}"
        )

    panel = build_intraday_window_panel(
        early_summaries,
        dates=dates,
        tickers=tickers,
        eligible=eligible,
        rolling_window=args.rolling_window,
        min_history=args.min_history,
    )
    design = build_intraday_market_design(panel, market_by_ticker=market_by_ticker)

    session_close = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    session_volume = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for ticker, target in session_targets.items():
        aligned_target = target.reindex(dates)
        complete_target = aligned_target["SessionComplete"].fillna(False).astype(bool)
        session_close.loc[complete_target, ticker] = aligned_target.loc[
            complete_target, "SessionClose"
        ]
        session_volume.loc[complete_target, ticker] = aligned_target.loc[
            complete_target, "SessionVolume"
        ]
    minute_close_values = session_close.to_numpy(dtype=np.float64)
    minute_volume_values = session_volume.to_numpy(dtype=np.float64)

    price_match_ratio, price_match_count, price_error = _relative_match(
        minute_close_values,
        daily_close_values,
        tolerance=args.price_relative_tolerance,
    )
    volume_match_ratio, volume_match_count, volume_error = _relative_match(
        minute_volume_values,
        daily_volume_values,
        tolerance=args.volume_relative_tolerance,
    )
    volume_contained_ratio, regular_volume_coverage = _volume_containment(
        minute_volume_values,
        daily_volume_values,
        tolerance=args.volume_relative_tolerance,
    )
    finite_volume_coverage = regular_volume_coverage[
        np.isfinite(regular_volume_coverage)
    ]
    median_regular_volume_coverage = (
        float(np.median(finite_volume_coverage))
        if len(finite_volume_coverage)
        else 0.0
    )
    if price_match_ratio < args.minimum_price_match_ratio:
        raise ValueError(
            f"minute/daily raw close match ratio {price_match_ratio:.6f} is below gate"
        )
    if volume_contained_ratio < args.minimum_volume_contained_ratio:
        raise ValueError(
            "regular minute volume containment ratio "
            f"{volume_contained_ratio:.6f} is below gate"
        )
    if median_regular_volume_coverage < args.minimum_median_regular_volume_coverage:
        raise ValueError(
            "median regular-session volume coverage "
            f"{median_regular_volume_coverage:.6f} is below gate"
        )

    eligible_count = eligible.sum(axis=1)
    complete_count = (panel.window_complete & eligible).sum(axis=1)
    date_coverage = np.divide(
        complete_count,
        eligible_count,
        out=np.zeros_like(complete_count, dtype=np.float64),
        where=eligible_count > 0,
    )
    target_complete = (
        panel.window_complete
        & eligible
        & np.isfinite(minute_close_values)
        & (minute_close_values > 0.0)
        & np.isfinite(minute_volume_values)
        & (minute_volume_values >= 0.0)
    )
    target_count = target_complete.sum(axis=1)
    target_date_coverage = np.divide(
        target_count,
        eligible_count,
        out=np.zeros_like(target_count, dtype=np.float64),
        where=eligible_count > 0,
    )
    usable = (eligible_count > 0) & (
        date_coverage >= float(args.minimum_date_stock_coverage)
    ) & (
        target_date_coverage
        >= float(args.minimum_target_date_stock_coverage)
    )
    if int(usable.sum()) < max(args.min_history + 1, 20):
        raise ValueError(
            f"only {int(usable.sum())} dates pass input and target coverage gates"
        )

    remaining_return, remaining_valid = remaining_session_returns(panel, session_close)
    selected = np.flatnonzero(usable)
    output_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = output_dir / "intraday_sensing_release.npz"
    temporary = Path(str(bundle_path) + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            dates=np.asarray([str(date.date()) for date in dates[selected]], dtype="U10"),
            tickers=np.asarray(tickers, dtype="U6"),
            node_feature_names=np.asarray(panel.feature_names, dtype="U64"),
            node_values=panel.values[selected].astype(np.float32),
            node_available=panel.available[selected].astype(np.uint8),
            decision_price=panel.decision_price[selected].astype(np.float32),
            session_open=panel.session_open[selected].astype(np.float32),
            window_complete=panel.window_complete[selected].astype(np.uint8),
            market_feature_names=np.asarray(design.feature_names, dtype="U128"),
            market_values=design.values[selected].astype(np.float32),
            remaining_session_return=remaining_return[selected].astype(np.float32),
            remaining_session_return_valid=remaining_valid[selected].astype(np.uint8),
            session_close=minute_close_values[selected].astype(np.float32),
            eligible=eligible[selected].astype(np.uint8),
            date_stock_coverage=date_coverage[selected].astype(np.float32),
            target_date_stock_coverage=target_date_coverage[selected].astype(np.float32),
        )
    temporary.replace(bundle_path)

    manifest = {
        "schema_version": 1,
        "sensor_contract": SENSOR_CONTRACT,
        "source": "kiwoom_rest_ka10080",
        "basis": "raw",
        "interval_minutes": int(args.interval_minutes),
        "decision_time": args.decision_time,
        "timestamp_semantics": args.timestamp_semantics,
        "label": "decision_price_to_same_session_close_return",
        "target_close_source": "same_raw_ka10080_session",
        "dates": int(len(selected)),
        "first_date": str(dates[selected[0]].date()),
        "last_date": str(dates[selected[-1]].date()),
        "stocks": len(tickers),
        "minute_source_files": len(minute_sources),
        "missing_minute_tickers": missing_tickers,
        "node_features": len(panel.feature_names),
        "market_features": len(design.feature_names),
        "date_stock_coverage": {
            "minimum_gate": float(args.minimum_date_stock_coverage),
            "minimum": float(date_coverage[selected].min()),
            "median": float(np.median(date_coverage[selected])),
            "maximum": float(date_coverage[selected].max()),
        },
        "target_date_stock_coverage": {
            "minimum_gate": float(args.minimum_target_date_stock_coverage),
            "minimum": float(target_date_coverage[selected].min()),
            "median": float(np.median(target_date_coverage[selected])),
            "maximum": float(target_date_coverage[selected].max()),
        },
        "raw_cross_checks": {
            "price_match_count": price_match_count,
            "price_match_ratio": price_match_ratio,
            "price_relative_tolerance": float(args.price_relative_tolerance),
            "price_relative_error_p99": float(np.nanquantile(price_error, 0.99)),
            "volume_match_count": volume_match_count,
            "volume_match_ratio": volume_match_ratio,
            "volume_relative_tolerance": float(args.volume_relative_tolerance),
            "volume_relative_error_p99": float(np.nanquantile(volume_error, 0.99)),
            "volume_contained_by_daily_ratio": volume_contained_ratio,
            "regular_to_daily_volume_coverage_p10": float(
                np.quantile(finite_volume_coverage, 0.10)
            ),
            "regular_to_daily_volume_coverage_median": (
                median_regular_volume_coverage
            ),
            "regular_to_daily_volume_coverage_p99": float(
                np.quantile(finite_volume_coverage, 0.99)
            ),
        },
        "causality": {
            "rolling_volume_baselines_shifted_one_session": True,
            "decision_window_excludes_incomplete_bar": True,
            "full_session_values_absent_from_inputs": True,
            "open_to_close_label_rejected": True,
            "daily_volume_can_include_post_close_trades": True,
        },
        "inputs": {
            "coverage": str(coverage_path),
            "coverage_sha256": file_sha256(coverage_path),
            "coverage_run_id": args.run_id,
            "universe": str(universe_path),
            "universe_sha256": file_sha256(universe_path),
            "semantics_evidence": str(evidence_path),
            "semantics_evidence_sha256": file_sha256(evidence_path),
            "minute_sources_sha256": canonical_sha256(minute_sources),
            "daily_sources_sha256": canonical_sha256(daily_sources),
            "daily_close_column": args.daily_close_column,
            "daily_volume_column": args.daily_volume_column,
        },
        "output": str(bundle_path),
        "output_sha256": file_sha256(bundle_path),
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
