from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping
import uuid

import numpy as np
import pandas as pd


HYBRID_SCHEMA_VERSION = 3
HYBRID_RELEASE_KIND = "krx500_lifecycle_hybrid"
CAUSAL_PROVIDER = "kiwoom_rest_ka10081"
PROXY_PROVIDER = "finance_data_reader_adjusted_return_index_proxy"
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]
RAW_COLUMNS = ["RawOpen", "RawHigh", "RawLow", "RawClose", "RawVolume"]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _resolve_declared_path(value: object, manifest_path: Path) -> Path:
    path = Path(str(value or "")).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, manifest_path.parent / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_ohlcv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame


def _observed_prices(frame: pd.DataFrame) -> pd.Series:
    prices = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    return prices.gt(0.0).all(axis=1) & np.isfinite(
        prices.to_numpy(dtype=np.float64)
    ).all(axis=1)


def build_proxy_return_index(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    anchor_close: float = 1_000.0,
) -> pd.DataFrame:
    """Build a scale-invariant price-state proxy without executable prices.

    FinanceDataReader prices are vendor-adjusted. Adjacent returns and intraday
    ratios are invariant to later piecewise-constant adjustment factors, so the
    output reconstructs a forward index from those quantities. Volume and raw
    exchange prices are intentionally unavailable.
    """

    missing = [column for column in [*PRICE_COLUMNS, "Volume"] if column not in frame]
    if missing:
        raise ValueError(f"proxy OHLCV is missing columns: {missing}")
    source = frame.copy()
    source.index = pd.to_datetime(source.index, errors="coerce").normalize()
    source = source.loc[
        source.index.notna()
        & (source.index >= pd.Timestamp(start))
        & (source.index <= pd.Timestamp(end))
    ]
    source = source[~source.index.duplicated(keep="last")].sort_index()
    for column in [*PRICE_COLUMNS, "Volume"]:
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source_observed = _observed_prices(source)
    if not source_observed.any():
        raise ValueError("proxy OHLCV has no observed price bars")

    output = pd.DataFrame(index=source.index)
    for column in PRICE_COLUMNS:
        output[column] = 0.0

    valid = source.loc[source_observed, PRICE_COLUMNS]
    vendor_close = valid["Close"].to_numpy(dtype=np.float64)
    adjacent = np.ones(len(valid), dtype=np.float64)
    if len(valid) > 1:
        adjacent[1:] = vendor_close[1:] / vendor_close[:-1]
    # KRX daily moves outside the price limit are reference-price resets, not
    # economic returns. Without a paired raw/adjusted source, mask the reset
    # date and continue the index from the next measurable adjacent return.
    discontinuity = (adjacent > 1.35) | (adjacent < 0.65)
    discontinuity[0] = False
    index_close = np.empty(len(valid), dtype=np.float64)
    index_close[0] = float(anchor_close)
    for position in range(1, len(valid)):
        index_close[position] = index_close[position - 1]
        if not discontinuity[position]:
            index_close[position] *= adjacent[position]
    output.loc[valid.index[~discontinuity], "Close"] = index_close[~discontinuity]
    for column in ["Open", "High", "Low"]:
        ratio = valid[column].to_numpy(dtype=np.float64) / vendor_close
        output.loc[valid.index[~discontinuity], column] = (
            index_close[~discontinuity] * ratio[~discontinuity]
        )
    observed_index = valid.index[~discontinuity]
    observed_output = output.loc[observed_index, PRICE_COLUMNS]
    high_floor = observed_output[["Open", "Close"]].max(axis=1)
    low_ceiling = observed_output[["Open", "Close"]].min(axis=1)
    repair = (observed_output["High"] < high_floor) | (
        observed_output["Low"] > low_ceiling
    )
    output.loc[observed_index, "High"] = np.maximum(
        observed_output["High"], high_floor
    )
    output.loc[observed_index, "Low"] = np.minimum(
        observed_output["Low"], low_ceiling
    )

    for column in PRICE_COLUMNS:
        output[f"VendorAdjusted{column}"] = source[column]
    output["Volume"] = 0.0
    output["VendorAdjustedVolume"] = source["Volume"]
    for column in RAW_COLUMNS:
        output[column] = np.nan
    output["TradingValueM"] = np.nan
    output["CausalPriceScale"] = np.nan
    output["VendorAdjustmentFactor"] = np.nan
    output["VendorVolumeAdjustmentFactor"] = np.nan
    output["RawReturn"] = np.nan
    output["CausalAdjustedReturn"] = np.nan
    output.loc[valid.index[~discontinuity], "CausalAdjustedReturn"] = (
        adjacent[~discontinuity] - 1.0
    )
    output["AdjustmentReturnGap"] = np.nan
    output["CorporateActionFlag"] = False
    output.loc[valid.index[discontinuity], "CorporateActionFlag"] = True
    output["ProxyOhlcRepairFlag"] = False
    output.loc[repair[repair].index, "ProxyOhlcRepairFlag"] = True
    output["ExecutionEligible"] = False
    output.index.name = "Date"

    column_order = [
        "Open",
        "RawOpen",
        "VendorAdjustedOpen",
        "High",
        "RawHigh",
        "VendorAdjustedHigh",
        "Low",
        "RawLow",
        "VendorAdjustedLow",
        "Close",
        "RawClose",
        "VendorAdjustedClose",
        "Volume",
        "RawVolume",
        "VendorAdjustedVolume",
        "TradingValueM",
        "CausalPriceScale",
        "VendorAdjustmentFactor",
        "VendorVolumeAdjustmentFactor",
        "RawReturn",
        "CausalAdjustedReturn",
        "AdjustmentReturnGap",
        "CorporateActionFlag",
        "ProxyOhlcRepairFlag",
        "ExecutionEligible",
    ]
    return output[column_order]


def _covering_cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    pattern = re.compile(rf"^{re.escape(ticker)}_(\d{{8}})_(\d{{8}})\.csv$")
    candidates: list[tuple[int, int, str, Path]] = []
    for candidate in cache_dir.glob(f"{ticker}_*.csv"):
        match = pattern.fullmatch(candidate.name)
        if match is None:
            continue
        cached_start = pd.to_datetime(match.group(1), format="%Y%m%d", errors="coerce")
        cached_end = pd.to_datetime(match.group(2), format="%Y%m%d", errors="coerce")
        if pd.isna(cached_start) or pd.isna(cached_end):
            continue
        if cached_start <= requested_start and cached_end >= requested_end:
            candidates.append(
                (-int(cached_end.value), int((requested_start - cached_start).days), candidate.name, candidate)
            )
    if not candidates:
        raise FileNotFoundError(f"no covering proxy cache for {ticker} in {cache_dir}")
    return min(candidates)[3]


def _distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"count": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    return {
        "count": int(len(array)),
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def compare_proxy_provider_overlap(
    causal_outputs: Iterable[Mapping[str, Any]],
    *,
    causal_manifest_path: Path,
    proxy_cache_dir: Path,
    start: str,
    end: str,
) -> dict[str, Any]:
    return_errors: list[float] = []
    intraday_errors: list[float] = []
    volume_log_errors: list[float] = []
    correlations: list[float] = []
    compared_tickers = 0
    for row in causal_outputs:
        ticker = str(row.get("ticker") or "")
        causal_path = _resolve_declared_path(row.get("path"), causal_manifest_path)
        try:
            proxy_path = _covering_cache_path(proxy_cache_dir, ticker, start, end)
        except FileNotFoundError:
            continue
        causal = _read_ohlcv(causal_path)
        proxy = _read_ohlcv(proxy_path)
        proxy_prices = proxy[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        proxy_observed = proxy_prices.gt(0.0).all(axis=1)
        proxy_valid = proxy_prices.loc[proxy_observed]
        proxy_return = proxy_valid["Close"].pct_change(fill_method=None)
        if "CausalAdjustedReturn" in causal:
            causal_return = pd.to_numeric(causal["CausalAdjustedReturn"], errors="coerce")
        else:
            causal_return = pd.to_numeric(causal["Close"], errors="coerce").pct_change(fill_method=None)
        corporate_action = (
            causal["CorporateActionFlag"].astype(bool)
            if "CorporateActionFlag" in causal
            else pd.Series(False, index=causal.index)
        )
        aligned = pd.concat(
            [
                causal_return.rename("causal"),
                proxy_return.rename("proxy"),
                corporate_action.rename("corporate_action"),
            ],
            axis=1,
        ).dropna(subset=["causal", "proxy"])
        aligned = aligned[
            ~aligned["corporate_action"] & aligned["proxy"].abs().le(0.35)
        ]
        if len(aligned) < 2:
            continue
        delta = (aligned["causal"] - aligned["proxy"]).abs().to_numpy(dtype=np.float64)
        return_errors.extend(delta.tolist())
        if len(aligned) >= 20 and aligned["causal"].std() > 0 and aligned["proxy"].std() > 0:
            correlations.append(float(aligned.corr().iloc[0, 1]))

        causal_ratio = pd.to_numeric(causal["Open"], errors="coerce") / pd.to_numeric(
            causal["Close"], errors="coerce"
        )
        proxy_ratio = proxy_valid["Open"] / proxy_valid["Close"]
        intraday = pd.concat([causal_ratio, proxy_ratio], axis=1).dropna()
        if len(intraday):
            intraday_errors.extend(
                (intraday.iloc[:, 0] - intraday.iloc[:, 1]).abs().to_numpy(dtype=np.float64).tolist()
            )

        causal_volume_name = "VendorAdjustedVolume" if "VendorAdjustedVolume" in causal else "Volume"
        causal_volume = pd.to_numeric(causal[causal_volume_name], errors="coerce")
        proxy_volume = pd.to_numeric(proxy["Volume"], errors="coerce")
        volume = pd.concat([causal_volume, proxy_volume], axis=1).dropna()
        volume = volume[(volume.iloc[:, 0] >= 0.0) & (volume.iloc[:, 1] >= 0.0)]
        if len(volume):
            volume_log_errors.extend(
                np.abs(np.log1p(volume.iloc[:, 0]) - np.log1p(volume.iloc[:, 1])).tolist()
            )
        compared_tickers += 1

    return_stats = _distribution(return_errors)
    intraday_stats = _distribution(intraday_errors)
    volume_stats = _distribution(volume_log_errors)
    correlation_stats = _distribution(correlations)
    blockers: list[str] = []
    if compared_tickers < 400:
        blockers.append("provider_overlap_tickers_below_400")
    if return_stats["p99"] is None or float(return_stats["p99"]) > 0.001:
        blockers.append("provider_return_error_p99_above_10bp")
    if return_stats["mean"] is None or float(return_stats["mean"]) > 0.0001:
        blockers.append("provider_return_error_mean_above_1bp")
    if correlation_stats["mean"] is None or float(correlation_stats["mean"]) < 0.999:
        blockers.append("provider_return_correlation_below_0.999")
    return {
        "schema_version": 1,
        "status": "pass" if not blockers else "blocked",
        "compared_tickers": compared_tickers,
        "daily_return_absolute_error": return_stats,
        "intraday_open_close_ratio_absolute_error": intraday_stats,
        "volume_log_absolute_error_diagnostic_only": volume_stats,
        "per_ticker_return_correlation": correlation_stats,
        "return_error_above_10bp_fraction": (
            float(np.mean(np.asarray(return_errors) > 0.001)) if return_errors else None
        ),
        "return_error_above_100bp_fraction": (
            float(np.mean(np.asarray(return_errors) > 0.01)) if return_errors else None
        ),
        "blockers": blockers,
        "decision": "price-state proxy only; volume, liquidity, notional, and execution remain unavailable",
    }


def build_lifecycle_hybrid_release(
    *,
    universe_manifest: str | Path,
    causal_manifest: str | Path,
    proxy_cache_dir: str | Path,
    output_dir: str | Path,
    start: str,
    end: str,
    expected_tickers: int = 500,
    expected_proxy_tickers: int = 47,
    validate_provider_overlap: bool = True,
) -> Path:
    universe_path = Path(universe_manifest)
    causal_manifest_path = Path(causal_manifest)
    cache_root = Path(proxy_cache_dir)
    output_root = Path(output_dir)
    if output_root.exists():
        raise FileExistsError(f"immutable release output already exists: {output_root}")

    universe_payload = _load_json(universe_path)
    universe = list(universe_payload.get("universe") or [])
    universe_by_ticker = {str(row.get("ticker") or "").zfill(6): row for row in universe}
    if len(universe) != expected_tickers or len(universe_by_ticker) != expected_tickers:
        raise ValueError(f"expected {expected_tickers} unique universe records")

    causal_payload = _load_json(causal_manifest_path)
    causal_outputs = list(causal_payload.get("outputs") or [])
    causal_by_ticker = {str(row.get("ticker") or "").zfill(6): row for row in causal_outputs}
    missing = {str(value).zfill(6) for value in causal_payload.get("missing_tickers") or []}
    universe_tickers = set(universe_by_ticker)
    if set(causal_by_ticker) | missing != universe_tickers or set(causal_by_ticker) & missing:
        raise ValueError("causal outputs and missing tickers do not partition the frozen universe")
    if len(missing) != expected_proxy_tickers:
        raise ValueError(f"expected {expected_proxy_tickers} proxy tickers, found {len(missing)}")
    if any(not universe_by_ticker[ticker].get("delisting_date") for ticker in missing):
        raise ValueError("every proxy ticker must have a frozen delisting date")

    overlap = (
        compare_proxy_provider_overlap(
            causal_outputs,
            causal_manifest_path=causal_manifest_path,
            proxy_cache_dir=cache_root,
            start=start,
            end=end,
        )
        if validate_provider_overlap
        else {
            "schema_version": 1,
            "status": "skipped",
            "blockers": [],
            "decision": "provider overlap validation disabled by caller",
        }
    )
    if overlap["status"] == "blocked":
        raise ValueError(f"proxy provider overlap validation failed: {overlap['blockers']}")

    temporary_root = output_root.with_name(f".{output_root.name}.tmp-{uuid.uuid4().hex}")
    temporary_ohlcv = temporary_root / "ohlcv"
    temporary_ohlcv.mkdir(parents=True, exist_ok=False)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    suffix = f"{start.replace('-', '')}_{end.replace('-', '')}.csv"
    outputs: list[dict[str, Any]] = []
    try:
        for record in universe:
            ticker = str(record.get("ticker") or "").zfill(6)
            final_path = output_root / "ohlcv" / f"{ticker}_{suffix}"
            temporary_path = temporary_ohlcv / final_path.name
            provider = CAUSAL_PROVIDER if ticker in causal_by_ticker else PROXY_PROVIDER
            if provider == CAUSAL_PROVIDER:
                source_row = causal_by_ticker[ticker]
                source_path = _resolve_declared_path(source_row.get("path"), causal_manifest_path)
                if file_sha256(source_path) != str(source_row.get("sha256") or ""):
                    raise ValueError(f"causal source hash mismatch: {ticker}")
                shutil.copy2(source_path, temporary_path)
                frame = _read_ohlcv(temporary_path)
                source_sha = file_sha256(source_path)
                execution_supported = True
                volume_supported = True
                ohlc_repair_rows = 0
            else:
                source_path = _covering_cache_path(cache_root, ticker, start, end)
                source = _read_ohlcv(source_path)
                delisting_date = pd.Timestamp(record["delisting_date"])
                source = source.loc[source.index <= delisting_date]
                frame = build_proxy_return_index(source, start=start, end=end)
                frame.to_csv(temporary_path)
                source_sha = file_sha256(source_path)
                execution_supported = False
                volume_supported = False
                ohlc_repair_rows = int(frame["ProxyOhlcRepairFlag"].astype(bool).sum())
            observed = _observed_prices(frame)
            outputs.append(
                {
                    "ticker": ticker,
                    "name": str(record.get("name") or ticker),
                    "provider": provider,
                    "rows": int(len(frame)),
                    "observed_price_rows": int(observed.sum()),
                    "first_date": str(frame.index.min().date()),
                    "last_date": str(frame.index.max().date()),
                    "listing_date": record.get("listing_date"),
                    "delisting_date": record.get("delisting_date"),
                    "execution_supported": execution_supported,
                    "volume_supported": volume_supported,
                    "ohlc_repair_rows": ohlc_repair_rows,
                    "path": str(final_path),
                    "sha256": file_sha256(temporary_path),
                    "source_path": str(source_path),
                    "source_sha256": source_sha,
                }
            )

        overlap_path = temporary_root / "provider_overlap_validation.json"
        overlap_path.write_text(
            json.dumps(overlap, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        provider_counts = Counter(row["provider"] for row in outputs)
        manifest = {
            "schema_version": HYBRID_SCHEMA_VERSION,
            "release_kind": HYBRID_RELEASE_KIND,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "start": start,
            "end": end,
            "universe_manifest": str(universe_path),
            "universe_sha256": file_sha256(universe_path),
            "causal_parent_manifest": str(causal_manifest_path),
            "causal_parent_manifest_sha256": file_sha256(causal_manifest_path),
            "expected_tickers": expected_tickers,
            "output_tickers": len(outputs),
            "missing_tickers": [],
            "source_counts": dict(provider_counts),
            "provider_overlap_validation": {
                "path": str(output_root / overlap_path.name),
                "sha256": file_sha256(overlap_path),
                "status": overlap["status"],
            },
            "contract": {
                "immutable": True,
                "fixed_stock_nodes": expected_tickers,
                "min_source_rows": 1,
                "price_observed_rule": "all OHLC finite and positive on the source date",
                "post_delisting_rule": "price state unavailable after frozen delisting date",
                "proxy_price_basis": "forward index from adjacent vendor-adjusted returns and intraday ratios",
                "proxy_action_rule": "reference-price discontinuities outside +/-35% are masked",
                "proxy_ohlc_repair_rule": "High/Low are clamped to contain same-row Open/Close after adjustment rounding",
                "proxy_volume_rule": "Volume is zero and all volume-derived features are unavailable",
                "proxy_execution_rule": "RawOHLC is null and execution is prohibited",
                "supported_use": "node-state representation and prediction",
                "unsupported_proxy_use": ["execution", "notional", "liquidity", "turnover"],
                "observation_checkpoints": [
                    "2020-01-02",
                    "2022-05-09",
                    "2023-03-06",
                    "2024-01-03",
                    "2024-11-05",
                    "2025-09-05",
                    "2026-07-10",
                    *([] if end == "2026-07-10" else [end]),
                ],
            },
            "framework_contract": {
                "qlib": "export from the same FeaturePanel and availability masks",
                "pyg": "GraphBatch adapter preserves feature and supervision masks",
                "jepa": "stock identity remains fixed while observations and targets are dynamically masked",
            },
            "outputs": outputs,
        }
        (temporary_root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return output_root / "manifest.json"


def audit_lifecycle_hybrid_release(
    manifest_path: str | Path,
    *,
    expected_tickers: int = 500,
    expected_proxy_tickers: int = 47,
) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = _load_json(path)
    blockers: list[str] = []
    warnings: list[str] = []
    if int(manifest.get("schema_version", 0) or 0) != HYBRID_SCHEMA_VERSION:
        blockers.append("unsupported_manifest_schema")
    if manifest.get("release_kind") != HYBRID_RELEASE_KIND:
        blockers.append("unexpected_release_kind")
    contract = manifest.get("contract") if isinstance(manifest.get("contract"), dict) else {}
    if contract.get("immutable") is not True:
        blockers.append("release_not_immutable")
    if int(contract.get("fixed_stock_nodes", 0) or 0) != expected_tickers:
        blockers.append("fixed_stock_node_count_mismatch")

    universe_path = _resolve_declared_path(manifest.get("universe_manifest"), path)
    if not universe_path.exists():
        blockers.append("universe_manifest_missing")
        universe_by_ticker: dict[str, dict[str, Any]] = {}
    else:
        if file_sha256(universe_path) != str(manifest.get("universe_sha256") or ""):
            blockers.append("universe_manifest_sha256_mismatch")
        universe_payload = _load_json(universe_path)
        universe_rows = list(universe_payload.get("universe") or [])
        universe_by_ticker = {
            str(row.get("ticker") or "").zfill(6): row for row in universe_rows
        }
        if len(universe_rows) != expected_tickers or len(universe_by_ticker) != expected_tickers:
            blockers.append("universe_ticker_count_mismatch")

    outputs = list(manifest.get("outputs") or [])
    output_tickers = [str(row.get("ticker") or "").zfill(6) for row in outputs]
    if len(outputs) != expected_tickers:
        blockers.append("output_ticker_count_mismatch")
    if len(output_tickers) != len(set(output_tickers)):
        blockers.append("duplicate_output_tickers")
    if set(output_tickers) != set(universe_by_ticker):
        blockers.append("release_tickers_do_not_equal_universe")
    if manifest.get("missing_tickers"):
        blockers.append("hybrid_release_has_missing_tickers")

    provider_counts: Counter[str] = Counter()
    verified_files = 0
    observed_by_date: defaultdict[pd.Timestamp, int] = defaultdict(int)
    lifecycle_violations = 0
    for row in outputs:
        ticker = str(row.get("ticker") or "").zfill(6)
        provider = str(row.get("provider") or "")
        provider_counts[provider] += 1
        output_path = _resolve_declared_path(row.get("path"), path)
        if not output_path.exists():
            blockers.append(f"missing_output:{ticker}")
            continue
        if file_sha256(output_path) != str(row.get("sha256") or ""):
            blockers.append(f"output_sha256_mismatch:{ticker}")
            continue
        try:
            frame = _read_ohlcv(output_path)
        except Exception:
            blockers.append(f"output_parse_error:{ticker}")
            continue
        if len(frame) != int(row.get("rows", -1) or -1) or frame.empty:
            blockers.append(f"output_row_count_invalid:{ticker}")
            continue
        if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
            blockers.append(f"output_date_index_invalid:{ticker}")
            continue
        required = {*PRICE_COLUMNS, "Volume"}
        if not required.issubset(frame.columns):
            blockers.append(f"output_columns_missing:{ticker}")
            continue
        prices = frame[PRICE_COLUMNS].apply(pd.to_numeric, errors="coerce")
        positive = prices.gt(0.0)
        observed = positive.all(axis=1) & np.isfinite(prices.to_numpy(dtype=np.float64)).all(axis=1)
        partial = positive.any(axis=1) & ~positive.all(axis=1)
        if partial.any():
            blockers.append(f"partial_nonpositive_price_bar:{ticker}")
        if observed.any():
            observed_prices = prices.loc[observed]
            bad_high = observed_prices["High"] + 1e-9 < observed_prices[["Open", "Close"]].max(axis=1)
            bad_low = observed_prices["Low"] - 1e-9 > observed_prices[["Open", "Close"]].min(axis=1)
            if bad_high.any() or bad_low.any():
                blockers.append(f"impossible_ohlc_bar:{ticker}")
            for date in frame.index[observed]:
                observed_by_date[pd.Timestamp(date)] += 1
        if int(observed.sum()) != int(row.get("observed_price_rows", -1) or -1):
            blockers.append(f"observed_price_row_count_mismatch:{ticker}")

        lifecycle = universe_by_ticker.get(ticker, {})
        delisting_value = lifecycle.get("delisting_date") or row.get("delisting_date")
        if delisting_value:
            delisting_date = pd.Timestamp(delisting_value)
            after = frame.index > delisting_date
            if after.any():
                lifecycle_violations += int(after.sum())
                blockers.append(f"post_delisting_rows:{ticker}")

        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        if provider == CAUSAL_PROVIDER:
            if row.get("execution_supported") is not True or row.get("volume_supported") is not True:
                blockers.append(f"causal_capability_contract_invalid:{ticker}")
            if not set(RAW_COLUMNS).issubset(frame.columns):
                blockers.append(f"causal_raw_columns_missing:{ticker}")
            else:
                raw = frame[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
                if not np.isfinite(raw.to_numpy(dtype=np.float64)).all():
                    blockers.append(f"causal_raw_values_nonfinite:{ticker}")
                elif (raw[["RawOpen", "RawHigh", "RawLow", "RawClose"]] <= 0.0).any().any():
                    blockers.append(f"causal_raw_price_nonpositive:{ticker}")
                else:
                    raw_notional = raw["RawClose"].to_numpy() * raw["RawVolume"].to_numpy()
                    canonical_notional = prices["Close"].to_numpy() * volume.to_numpy()
                    denominator = np.maximum(np.abs(raw_notional), 1.0)
                    if float(np.max(np.abs(canonical_notional - raw_notional) / denominator)) > 1e-9:
                        blockers.append(f"causal_notional_invariant_failed:{ticker}")
        elif provider == PROXY_PROVIDER:
            if row.get("execution_supported") is not False or row.get("volume_supported") is not False:
                blockers.append(f"proxy_capability_contract_invalid:{ticker}")
            if lifecycle.get("delisting_date") in {None, ""}:
                blockers.append(f"proxy_without_delisting_date:{ticker}")
            if not np.isfinite(volume.fillna(0.0).to_numpy()).all() or (volume.fillna(0.0) != 0.0).any():
                blockers.append(f"proxy_volume_not_masked:{ticker}")
            if not set(RAW_COLUMNS).issubset(frame.columns):
                blockers.append(f"proxy_raw_columns_missing:{ticker}")
            elif frame[RAW_COLUMNS].notna().any().any():
                blockers.append(f"proxy_raw_execution_values_present:{ticker}")
            if "CorporateActionFlag" not in frame:
                blockers.append(f"proxy_action_flag_missing:{ticker}")
            else:
                action = frame["CorporateActionFlag"].astype(bool)
                if (action & observed).any():
                    blockers.append(f"proxy_action_row_observed:{ticker}")
            if "CausalAdjustedReturn" in frame:
                proxy_return = pd.to_numeric(
                    frame["CausalAdjustedReturn"], errors="coerce"
                )
                if proxy_return.abs().gt(0.35).any():
                    blockers.append(f"proxy_return_discontinuity_unmasked:{ticker}")
            if "ProxyOhlcRepairFlag" not in frame:
                blockers.append(f"proxy_ohlc_repair_flag_missing:{ticker}")
            elif int(frame["ProxyOhlcRepairFlag"].astype(bool).sum()) != int(
                row.get("ohlc_repair_rows", -1)
            ):
                blockers.append(f"proxy_ohlc_repair_count_mismatch:{ticker}")
        else:
            blockers.append(f"unexpected_provider:{ticker}")
        verified_files += 1

    declared_counts = {
        str(key): int(value) for key, value in (manifest.get("source_counts") or {}).items()
    }
    if dict(provider_counts) != declared_counts:
        blockers.append("provider_count_manifest_mismatch")
    if provider_counts[PROXY_PROVIDER] != expected_proxy_tickers:
        blockers.append("proxy_ticker_count_mismatch")
    if provider_counts[CAUSAL_PROVIDER] != expected_tickers - expected_proxy_tickers:
        blockers.append("causal_ticker_count_mismatch")

    overlap_record = manifest.get("provider_overlap_validation") or {}
    overlap_path = _resolve_declared_path(overlap_record.get("path"), path)
    if not overlap_path.exists():
        blockers.append("provider_overlap_report_missing")
    elif file_sha256(overlap_path) != str(overlap_record.get("sha256") or ""):
        blockers.append("provider_overlap_report_sha256_mismatch")
    else:
        overlap = _load_json(overlap_path)
        if overlap.get("status") not in {"pass", "skipped"}:
            blockers.append("provider_overlap_validation_blocked")

    checkpoints = {}
    for value in contract.get("observation_checkpoints") or []:
        date = pd.Timestamp(value)
        checkpoints[str(date.date())] = int(observed_by_date.get(date, 0))
    sorted_dates = sorted(observed_by_date)
    if expected_proxy_tickers:
        warnings.append(
            f"{expected_proxy_tickers} delisted nodes are price-state proxies with execution and volume disabled"
        )
    blockers = sorted(set(blockers))
    return {
        "schema_version": 1,
        "status": "pass" if not blockers else "blocked",
        "release_manifest": str(path),
        "release_manifest_sha256": file_sha256(path),
        "expected_tickers": expected_tickers,
        "verified_files": verified_files,
        "provider_counts": dict(provider_counts),
        "lifecycle_violations": lifecycle_violations,
        "observed_nodes_at_checkpoints": checkpoints,
        "first_observation_date": str(sorted_dates[0].date()) if sorted_dates else None,
        "last_observation_date": str(sorted_dates[-1].date()) if sorted_dates else None,
        "last_observation_nodes": int(observed_by_date[sorted_dates[-1]]) if sorted_dates else 0,
        "blockers": blockers,
        "warnings": warnings,
        "live_orders_allowed": False,
    }
