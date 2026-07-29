from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "RawOpen",
    "RawHigh",
    "RawLow",
    "RawClose",
    "RawVolume",
    "CausalAdjustedReturn",
    "CorporateActionFlag",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_release(
    manifest_path: str | Path,
    *,
    min_tickers: int = 450,
    min_rows: int = 1500,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    blockers: list[str] = []
    warnings: list[str] = []

    outputs = list(manifest.get("outputs") or [])
    output_tickers = [str(row.get("ticker") or "") for row in outputs]
    missing_tickers = [str(ticker) for ticker in manifest.get("missing_tickers") or []]
    expected_tickers = int(manifest.get("expected_tickers", 0) or 0)
    if int(manifest.get("schema_version", 0) or 0) != 2:
        blockers.append("unsupported_manifest_schema")
    if manifest.get("source", {}).get("provider") != "kiwoom_rest_ka10081":
        blockers.append("unexpected_ohlcv_provider")
    if manifest.get("source", {}).get("execution_price_basis") != "RawOHLC columns only":
        blockers.append("missing_raw_execution_price_contract")
    if len(outputs) < int(min_tickers):
        blockers.append(f"output_tickers_below_{int(min_tickers)}")
    if len(output_tickers) != len(set(output_tickers)):
        blockers.append("duplicate_output_tickers")
    if len(missing_tickers) != len(set(missing_tickers)):
        blockers.append("duplicate_missing_tickers")
    if set(output_tickers) & set(missing_tickers):
        blockers.append("output_missing_ticker_overlap")
    if len(outputs) + len(missing_tickers) != expected_tickers:
        blockers.append("output_missing_partition_mismatch")

    universe_path = Path(str(manifest.get("universe_manifest") or ""))
    if not universe_path.exists():
        blockers.append("universe_manifest_missing")
        universe_tickers: set[str] = set()
    else:
        expected_universe_sha = str(manifest.get("universe_sha256") or "")
        if file_sha256(universe_path) != expected_universe_sha:
            blockers.append("universe_manifest_sha256_mismatch")
        universe_payload = json.loads(universe_path.read_text(encoding="utf-8"))
        universe_tickers = {
            str(row.get("ticker") or "")
            for row in universe_payload.get("universe", [])
        }
        if set(output_tickers) | set(missing_tickers) != universe_tickers:
            blockers.append("release_tickers_do_not_partition_universe")

    events_path = Path(str(manifest.get("events_path") or ""))
    if not events_path.exists():
        blockers.append("corporate_action_events_missing")
    elif file_sha256(events_path) != str(manifest.get("events_sha256") or ""):
        blockers.append("corporate_action_events_sha256_mismatch")

    verified_files = 0
    observed_action_rows = 0
    action_tickers: set[str] = set()
    max_notional_relative_error = 0.0
    for row in outputs:
        ticker = str(row.get("ticker") or "")
        output_path = Path(str(row.get("path") or ""))
        if not output_path.exists():
            blockers.append(f"missing_output:{ticker}")
            continue
        if file_sha256(output_path) != str(row.get("sha256") or ""):
            blockers.append(f"output_sha256_mismatch:{ticker}")
            continue
        try:
            frame = pd.read_csv(output_path, parse_dates=["Date"], index_col="Date")
        except Exception:
            blockers.append(f"output_parse_error:{ticker}")
            continue
        if len(frame) != int(row.get("rows", -1) or -1) or len(frame) < int(min_rows):
            blockers.append(f"output_row_count_invalid:{ticker}")
            continue
        if not REQUIRED_COLUMNS.issubset(frame.columns):
            blockers.append(f"output_columns_missing:{ticker}")
            continue
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            blockers.append(f"output_date_index_invalid:{ticker}")
            continue
        numeric = frame[list(REQUIRED_COLUMNS - {"CorporateActionFlag"})].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if not np.isfinite(numeric.to_numpy(dtype=np.float64)).all():
            blockers.append(f"output_nonfinite_required_values:{ticker}")
            continue
        if (frame[["Open", "High", "Low", "Close", "RawOpen", "RawHigh", "RawLow", "RawClose"]] <= 0.0).any().any():
            blockers.append(f"output_nonpositive_price:{ticker}")
            continue
        if (frame[["Volume", "RawVolume"]] < 0.0).any().any():
            blockers.append(f"output_negative_volume:{ticker}")
            continue
        raw_notional = frame["RawClose"].to_numpy(dtype=np.float64) * frame[
            "RawVolume"
        ].to_numpy(dtype=np.float64)
        canonical_notional = frame["Close"].to_numpy(dtype=np.float64) * frame[
            "Volume"
        ].to_numpy(dtype=np.float64)
        denominator = np.maximum(np.abs(raw_notional), 1.0)
        relative_error = float(
            np.max(np.abs(canonical_notional - raw_notional) / denominator)
        )
        max_notional_relative_error = max(max_notional_relative_error, relative_error)
        if relative_error > 1e-9:
            blockers.append(f"notional_invariant_failed:{ticker}")
            continue
        action_count = int(frame["CorporateActionFlag"].astype(bool).sum())
        if action_count != int(row.get("corporate_actions", -1) or 0):
            blockers.append(f"corporate_action_count_mismatch:{ticker}")
            continue
        observed_action_rows += action_count
        if action_count:
            action_tickers.add(ticker)
        verified_files += 1

    if observed_action_rows != int(manifest.get("corporate_action_events", -1) or 0):
        blockers.append("corporate_action_total_mismatch")
    if len(action_tickers) != int(manifest.get("corporate_action_tickers", -1) or 0):
        blockers.append("corporate_action_ticker_count_mismatch")
    if missing_tickers:
        blockers.append("frozen_universe_coverage_incomplete")
        warnings.append(
            f"{len(missing_tickers)} frozen-universe tickers lack paired raw/adjusted Kiwoom history"
        )

    blockers = sorted(set(blockers))
    return {
        "status": "pass" if not blockers else "blocked",
        "schema_version": 1,
        "release_manifest": str(path),
        "release_manifest_sha256": file_sha256(path),
        "provider": manifest.get("source", {}).get("provider"),
        "expected_tickers": expected_tickers,
        "output_tickers": len(outputs),
        "minimum_required_tickers": int(min_tickers),
        "coverage": len(outputs) / expected_tickers if expected_tickers else 0.0,
        "missing_tickers": missing_tickers,
        "verified_files": verified_files,
        "corporate_action_events": observed_action_rows,
        "corporate_action_tickers": len(action_tickers),
        "max_notional_relative_error": max_notional_relative_error,
        "blockers": blockers,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit a frozen causal Kiwoom OHLCV release and every file hash."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-tickers", type=int, default=450)
    parser.add_argument("--min-rows", type=int, default=1500)
    args = parser.parse_args()

    result = audit_release(
        args.manifest,
        min_tickers=args.min_tickers,
        min_rows=args.min_rows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output_tickers": result["output_tickers"]}))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
