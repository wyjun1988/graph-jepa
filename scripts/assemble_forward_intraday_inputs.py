from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute import audit_kiwoom_minute_frame


CONTRACT = "forward_intraday_input_extension_v1"
KST = "Asia/Seoul"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a frozen intraday history with one completed forward session."
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--base-minute-dir", required=True)
    parser.add_argument("--incremental-minute-dir", required=True)
    parser.add_argument("--base-daily-dir", required=True)
    parser.add_argument("--incremental-daily-raw-dir", required=True)
    parser.add_argument(
        "--incremental-daily-start",
        help=(
            "Start date encoded in incremental daily filenames; defaults to "
            "--incremental-date for one-session inputs."
        ),
    )
    parser.add_argument("--base-start", required=True)
    parser.add_argument("--base-end", required=True)
    parser.add_argument("--incremental-date", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _compact_date(value: str) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _load_universe(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    tickers = [str(row.get("ticker", "")).replace("A", "").zfill(6) for row in rows]
    if any(not ticker.isdigit() or len(ticker) != 6 for ticker in tickers):
        raise ValueError("universe contains an invalid KRX ticker")
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe contains duplicate tickers")
    return tickers


def _one_existing(candidates: Iterable[Path], label: str) -> Path | None:
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        raise ValueError(f"multiple {label} files found: {existing}")
    return existing[0] if existing else None


def _minute_path(directory: Path, ticker: str, start: str, end: str) -> Path | None:
    stem = f"{ticker}_{_compact_date(start)}_{_compact_date(end)}"
    return _one_existing(
        (directory / f"{stem}.parquet", directory / f"{stem}.csv.gz"),
        f"minute {ticker} {start}..{end}",
    )


def _daily_path(directory: Path, ticker: str, start: str, end: str) -> Path | None:
    stem = f"{ticker}_{_compact_date(start)}_{_compact_date(end)}"
    return _one_existing((directory / f"{stem}.csv",), f"daily {ticker}")


def _read_minute(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"unsupported minute input: {path}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    if frame.index.tz is None:
        raise ValueError(f"minute timestamps are timezone-naive: {path}")
    frame.index.name = "Timestamp"
    audit_kiwoom_minute_frame(frame, regular_session_only=True)
    return frame.sort_index()


def _drop_identical_duplicate_index(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    duplicates = frame.index[frame.index.duplicated(keep=False)].unique()
    for index in duplicates:
        rows = frame.loc[[index]]
        first = rows.iloc[0]
        for _row_index, candidate in rows.iloc[1:].iterrows():
            equal = (first.eq(candidate) | (first.isna() & candidate.isna())).all()
            if not bool(equal):
                raise ValueError(f"conflicting duplicate {label}: {index}")
    return frame.loc[~frame.index.duplicated(keep="first")].sort_index()


def merge_minute_frames(
    base: pd.DataFrame | None,
    incremental: pd.DataFrame | None,
) -> pd.DataFrame | None:
    frames = [frame for frame in (base, incremental) if frame is not None]
    if not frames:
        return None
    merged = _drop_identical_duplicate_index(pd.concat(frames), label="timestamp")
    audit_kiwoom_minute_frame(merged, regular_session_only=True)
    return merged


def _read_daily_base(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Date", "RawClose", "RawVolume"])
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    frame = frame.set_index("Date").sort_index()
    if frame.index.has_duplicates:
        raise ValueError(f"base daily input has duplicate dates: {path}")
    return frame.astype(float)


def _read_daily_increment(
    path: Path,
    expected_date: str,
    *,
    requested_start: str | None = None,
) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=["Date", "Close", "Volume"]).rename(
        columns={"Close": "RawClose", "Volume": "RawVolume"}
    )
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise").dt.normalize()
    frame = frame.set_index("Date").sort_index().astype(float)
    expected = pd.Timestamp(expected_date).normalize()
    start = pd.Timestamp(requested_start or expected_date).normalize()
    if start > expected:
        raise ValueError("incremental daily start must not follow its expected date")
    if frame.index.has_duplicates:
        raise ValueError(f"incremental daily input has duplicate dates: {path}")
    if frame.empty or frame.index.min() < start or frame.index.max() > expected:
        raise ValueError(
            f"incremental daily input escapes {start.date()}..{expected.date()}: {path}"
        )
    if expected not in frame.index:
        raise ValueError(
            f"incremental daily input is missing {expected.date()}: {path}"
        )
    return frame.loc[[expected]]


def merge_daily_frames(
    base: pd.DataFrame,
    incremental: pd.DataFrame | None,
) -> pd.DataFrame:
    merged = _drop_identical_duplicate_index(
        pd.concat([base] + ([] if incremental is None else [incremental])),
        label="daily date",
    )
    close = merged["RawClose"].to_numpy(dtype=float)
    volume = merged["RawVolume"].to_numpy(dtype=float)
    present = np.isfinite(close) & np.isfinite(volume)
    missing_pair = np.isnan(close) & np.isnan(volume)
    if (
        not np.all(present | missing_pair)
        or (close[present] <= 0.0).any()
        or (volume[present] < 0.0).any()
    ):
        raise ValueError(
            "observed daily close must be positive and volume must be non-negative; "
            "missing lifecycle rows must be paired"
        )
    return merged


def _source(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    base_start = pd.Timestamp(args.base_start).normalize()
    base_end = pd.Timestamp(args.base_end).normalize()
    incremental_date = pd.Timestamp(args.incremental_date).normalize()
    incremental_daily_start = pd.Timestamp(
        args.incremental_daily_start or args.incremental_date
    ).normalize()
    if not base_start <= base_end < incremental_date:
        raise ValueError("date contract must satisfy base_start <= base_end < incremental_date")
    if not base_end <= incremental_daily_start <= incremental_date:
        raise ValueError(
            "incremental daily start must be between base end and incremental date"
        )

    universe_path = Path(args.universe_manifest)
    base_minute_dir = Path(args.base_minute_dir)
    incremental_minute_dir = Path(args.incremental_minute_dir)
    base_daily_dir = Path(args.base_daily_dir)
    incremental_daily_dir = Path(args.incremental_daily_raw_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable forward input already exists: {output_dir}")
    temporary = Path(str(output_dir) + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    minute_output = temporary / "minute" / "5min" / "raw"
    daily_output = temporary / "daily"
    minute_output.mkdir(parents=True)
    daily_output.mkdir(parents=True)

    tickers = _load_universe(universe_path)
    start_text = str(base_start.date())
    base_end_text = str(base_end.date())
    incremental_text = str(incremental_date.date())
    suffix = f"{_compact_date(start_text)}_{_compact_date(incremental_text)}"
    minute_records: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    counts = {"minute_ok": 0, "minute_partial": 0, "minute_missing": 0, "daily_extended": 0}

    try:
        for index, ticker in enumerate(tickers, start=1):
            base_minute_path = _minute_path(
                base_minute_dir, ticker, start_text, base_end_text
            )
            incremental_minute_path = _minute_path(
                incremental_minute_dir, ticker, incremental_text, incremental_text
            )
            base_minute = _read_minute(base_minute_path) if base_minute_path else None
            incremental_minute = (
                _read_minute(incremental_minute_path)
                if incremental_minute_path
                else None
            )
            merged_minute = merge_minute_frames(base_minute, incremental_minute)
            minute_status = (
                "ok"
                if base_minute is not None and incremental_minute is not None
                else "partial"
                if merged_minute is not None
                else "missing"
            )
            minute_record: dict[str, Any] = {
                "schema_version": 1,
                "contract": CONTRACT,
                "ticker": ticker,
                "interval_minutes": 5,
                "basis": "raw",
                "run_id": args.run_id,
                "status": minute_status,
                "requested_start": start_text,
                "requested_end": incremental_text,
                "sources": {
                    "base": _source(base_minute_path),
                    "incremental": _source(incremental_minute_path),
                },
                "live_orders_allowed": False,
            }
            if merged_minute is not None:
                destination = minute_output / f"{ticker}_{suffix}.parquet"
                merged_minute.to_parquet(destination, compression="zstd", index=True)
                local_dates = merged_minute.index.tz_convert(KST).tz_localize(None).normalize()
                minute_record.update(
                    {
                        "effective_start": str(local_dates.min().date()),
                        "effective_end": str(local_dates.max().date()),
                        "output": str(output_dir / "minute" / "5min" / "raw" / destination.name),
                        "output_sha256": sha256_file(destination),
                        "audit": audit_kiwoom_minute_frame(
                            merged_minute, regular_session_only=True
                        ),
                    }
                )
            else:
                minute_record.update({"output": None, "output_sha256": None})
            minute_records.append(minute_record)
            counts[f"minute_{minute_status}"] += 1

            base_daily_candidates = sorted(base_daily_dir.glob(f"{ticker}_*.csv"))
            if len(base_daily_candidates) != 1:
                raise ValueError(
                    f"expected exactly one frozen base daily file for {ticker}, "
                    f"found {len(base_daily_candidates)}"
                )
            base_daily_path = base_daily_candidates[0]
            incremental_daily_path = _daily_path(
                incremental_daily_dir,
                ticker,
                str(incremental_daily_start.date()),
                incremental_text,
            )
            base_daily = _read_daily_base(base_daily_path).loc[base_start:base_end]
            incremental_daily = (
                _read_daily_increment(
                    incremental_daily_path,
                    incremental_text,
                    requested_start=str(incremental_daily_start.date()),
                )
                if incremental_daily_path
                else None
            )
            merged_daily = merge_daily_frames(base_daily, incremental_daily)
            destination = daily_output / f"{ticker}_{suffix}.csv"
            merged_daily.rename_axis("Date").to_csv(destination)
            if incremental_daily is not None:
                counts["daily_extended"] += 1
            daily_records.append(
                {
                    "ticker": ticker,
                    "base": _source(base_daily_path),
                    "incremental": _source(incremental_daily_path),
                    "output": str(output_dir / "daily" / destination.name),
                    "output_sha256": sha256_file(destination),
                    "last_date": str(merged_daily.index.max().date()),
                }
            )
            if index % 50 == 0:
                print(f"assembled={index}/{len(tickers)}", flush=True)

        _write_jsonl(temporary / "coverage.jsonl", minute_records)
        _write_jsonl(temporary / "daily_sources.jsonl", daily_records)
        manifest = {
            "schema_version": 1,
            "contract": CONTRACT,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "run_id": args.run_id,
            "universe_manifest": str(universe_path),
            "universe_sha256": sha256_file(universe_path),
            "base_start": start_text,
            "base_end": base_end_text,
            "incremental_date": incremental_text,
            "tickers": len(tickers),
            "counts": counts,
            "outputs": {
                "coverage": "coverage.jsonl",
                "coverage_sha256": sha256_file(temporary / "coverage.jsonl"),
                "daily_sources": "daily_sources.jsonl",
                "daily_sources_sha256": sha256_file(temporary / "daily_sources.jsonl"),
                "minute_dir": "minute/5min/raw",
                "daily_dir": "daily",
            },
            "live_orders_allowed": False,
            "promotion_eligible": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
