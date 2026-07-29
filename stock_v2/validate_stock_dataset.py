from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.run_real_backtest import filter_history_for_training, parse_int_list
from stock_v2.data_contract import build_training_data_manifest
from stock_v2.event_features import (
    _event_payload,
    _record_ticker,
    _score_record,
    build_event_feature_frames,
    build_event_theme_exposure,
    build_event_ticker_coverage,
    clean_ticker,
    parse_event_date,
)
from stock_v2.external_factors import (
    build_external_feature_frames,
    build_external_node_feature_frames,
    fetch_external_factor_closes,
    resolve_external_factors,
)
from stock_v2.fundamental_features import (
    build_fundamental_feature_frames,
    fundamental_coverage,
    load_fundamental_observations,
)
from stock_v2.kiwoom_investor import (
    build_investor_feature_frames,
    investor_feature_coverage,
    load_investor_flow_frames,
)
from stock_v2.lifecycle_ohlcv import audit_lifecycle_hybrid_release
from stock_v2.market_data import (
    fetch_krx_ohlcv,
    load_universe_manifest,
    make_ohlcv_panel,
    select_krx_universe_from_listing,
    select_universe,
)
from stock_v2.real_features import build_feature_panel


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def summarize_array(values: list[float]) -> dict[str, float | int | None]:
    arr = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def validate_cache_files(cache_dir: Path) -> dict[str, Any]:
    csvs = sorted(cache_dir.glob("*.csv"))
    invalid_filenames: list[str] = []
    by_ticker: dict[str, list[str]] = defaultdict(list)
    filename_re = re.compile(r"^([0-9A-Z]+)_([0-9]{8})_([0-9]{8})\.csv$")
    for path in csvs:
        match = filename_re.match(path.name)
        if not match or not re.fullmatch(r"\d{6}", match.group(1)):
            invalid_filenames.append(path.name)
            continue
        by_ticker[match.group(1)].append(path.name)

    duplicate_ranges = {
        ticker: names
        for ticker, names in by_ticker.items()
        if len(names) > 1
    }
    return {
        "cache_files": len(csvs),
        "valid_ticker_files": sum(len(names) for names in by_ticker.values()),
        "unique_valid_tickers": len(by_ticker),
        "invalid_filenames": invalid_filenames[:50],
        "invalid_filename_count": len(invalid_filenames),
        "tickers_with_multiple_ranges_count": len(duplicate_ranges),
        "tickers_with_multiple_ranges_sample": dict(list(duplicate_ranges.items())[:20]),
    }


def validate_ohlcv(raw: dict[str, pd.DataFrame], train_end: str) -> dict[str, Any]:
    issues: dict[str, list[str]] = defaultdict(list)
    row_counts: list[int] = []
    train_counts: list[int] = []
    zero_volume_rates: list[float] = []
    date_min: list[pd.Timestamp] = []
    date_max: list[pd.Timestamp] = []
    train_cutoff = pd.Timestamp(train_end)

    for ticker, frame in raw.items():
        row_counts.append(len(frame))
        train_counts.append(int((frame.index <= train_cutoff).sum()))
        if not frame.index.is_monotonic_increasing:
            issues["non_monotonic_index"].append(ticker)
        if frame.index.has_duplicates:
            issues["duplicate_dates"].append(ticker)
        required = ["Open", "High", "Low", "Close", "Volume"]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            issues["missing_columns"].append(f"{ticker}:{','.join(missing)}")
            continue
        numeric = frame[required].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            issues["nonfinite_ohlcv"].append(ticker)
        positive = numeric[["Open", "High", "Low", "Close"]].gt(0.0)
        partial_nonpositive = positive.any(axis=1) & ~positive.all(axis=1)
        if partial_nonpositive.any():
            issues["partial_nonpositive_price"].append(ticker)
        if (numeric["High"] < numeric["Low"]).any():
            issues["high_below_low"].append(ticker)
        zero_volume_rates.append(float((numeric["Volume"].fillna(0.0) <= 0).mean()))
        if len(frame.index):
            date_min.append(pd.Timestamp(frame.index.min()))
            date_max.append(pd.Timestamp(frame.index.max()))

    return {
        "tickers": len(raw),
        "date_min": str(min(date_min).date()) if date_min else None,
        "date_max": str(max(date_max).date()) if date_max else None,
        "rows_per_ticker": summarize_array([float(v) for v in row_counts]),
        "train_rows_per_ticker": summarize_array([float(v) for v in train_counts]),
        "zero_volume_rate": summarize_array(zero_volume_rates),
        "issues": {key: values[:50] for key, values in issues.items()},
        "issue_counts": {key: len(values) for key, values in issues.items()},
    }


def load_jsonl_events(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    invalid_json = 0
    empty = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                empty += 1
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                invalid_json += 1
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records, {"rows": len(records), "invalid_json": invalid_json, "empty_lines": empty}


def validate_events(records: list[dict[str, Any]], selected_tickers: set[str], panel_dates: pd.DatetimeIndex) -> dict[str, Any]:
    tickers = Counter()
    invalid_tickers = 0
    outside_universe = 0
    missing_dates = 0
    before_panel = 0
    after_panel = 0
    event_dates: list[pd.Timestamp] = []
    scores: list[float] = []
    abs_scores: list[float] = []
    llm_used = 0
    qwen_exact = 0
    heuristic = 0
    node_delta_counts: list[float] = []
    edge_delta_counts: list[float] = []
    duplicate_keys = Counter()

    first_date = pd.Timestamp(panel_dates.min()).normalize()
    last_date = pd.Timestamp(panel_dates.max()).normalize()
    for record in records:
        payload = _event_payload(record)
        ticker = _record_ticker(record, payload)
        if not re.fullmatch(r"\d{6}", str(ticker)):
            invalid_tickers += 1
        else:
            tickers[ticker] += 1
            if ticker not in selected_tickers:
                outside_universe += 1
        event_date = parse_event_date(record)
        if event_date is None:
            missing_dates += 1
        else:
            event_dates.append(event_date)
            if event_date < first_date:
                before_panel += 1
            if event_date > last_date:
                after_panel += 1

        score, _pos, _neg, abs_score, _confidence = _score_record(record, payload, str(ticker))
        scores.append(float(score))
        abs_scores.append(float(abs_score))

        if bool(record.get("llm_used")) or bool(payload.get("llm_used")):
            llm_used += 1
        source = str(record.get("source") or payload.get("source") or "").lower()
        if "qwen" in source or bool(record.get("qwen_exact")):
            qwen_exact += 1
        if not (bool(record.get("llm_used")) or bool(payload.get("llm_used"))):
            heuristic += 1
        node_delta_counts.append(float(len(payload.get("node_deltas", []) or [])))
        edge_delta_counts.append(float(len(payload.get("edge_deltas", []) or [])))

        article = record.get("article")
        link = ""
        title = ""
        if isinstance(article, dict):
            link = str(article.get("link") or article.get("url") or "")
            title = str(article.get("title") or "")
        key = (str(ticker), str(event_date.date()) if event_date is not None else "", link or title)
        if key[2]:
            duplicate_keys[key] += 1

    duplicate_article_keys = sum(1 for value in duplicate_keys.values() if value > 1)
    return {
        "rows": len(records),
        "unique_tickers": len(tickers),
        "invalid_tickers": invalid_tickers,
        "outside_selected_universe": outside_universe,
        "date_min": str(min(event_dates).date()) if event_dates else None,
        "date_max": str(max(event_dates).date()) if event_dates else None,
        "missing_dates": missing_dates,
        "before_panel_dates": before_panel,
        "after_panel_dates": after_panel,
        "llm_used_rows": llm_used,
        "qwen_source_rows": qwen_exact,
        "heuristic_rows": heuristic,
        "score": summarize_array(scores),
        "abs_score": summarize_array(abs_scores),
        "node_delta_count": summarize_array(node_delta_counts),
        "edge_delta_count": summarize_array(edge_delta_counts),
        "duplicate_article_key_count": duplicate_article_keys,
        "top_tickers": tickers.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate stock-v2 KRX OHLCV/event feature dataset")
    parser.add_argument("--universe", choices=["krx", "default"], default="krx")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=100)
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", default="2023-12-29")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--event-path", action="append", default=[])
    parser.add_argument("--event-half-life-days", type=float, default=5.0)
    parser.add_argument("--event-lag-days", type=int, default=1)
    parser.add_argument("--event-max-decay-days", type=int, default=60)
    parser.add_argument("--fundamental-path", action="append", default=[])
    parser.add_argument("--fundamental-lag-days", type=int, default=1)
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", type=int, default=1)
    parser.add_argument(
        "--external-preset",
        choices=["none", "kr_global", "kr_global_rates"],
        default="none",
    )
    parser.add_argument("--external-symbol", action="append", default=[])
    parser.add_argument("--external-node-mode", choices=["features", "nodes", "both"], default="nodes")
    parser.add_argument("--external-lag-days", type=int, default=1)
    parser.add_argument("--external-cache-dir", default="data/external_cache")
    parser.add_argument("--path-horizons", default="1,3,5,10")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--edge-window", type=int, default=60)
    parser.add_argument("--min-train-rows", type=int, default=260)
    parser.add_argument("--min-train-feature-rows", type=int, default=500)
    parser.add_argument("--min-test-feature-rows", type=int, default=200)
    parser.add_argument(
        "--dump-training-arrays",
        default=None,
        help="Optional NPZ path for cross-host feature diagnostics",
    )
    parser.add_argument("--output", default="reports/dataset_validation/latest.json")
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    elif args.universe == "krx":
        universe = select_krx_universe_from_listing(args.max_tickers)
    else:
        universe = select_universe(args.max_tickers)
    selected_tickers = {ticker for ticker, _name in universe}
    names = dict(universe)

    raw_initial = fetch_krx_ohlcv(
        universe=universe,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        refresh=False,
    )
    raw = filter_history_for_training(raw_initial, train_end=args.train_end, min_train_rows=args.min_train_rows)
    panel = make_ohlcv_panel(raw, names=names)

    lifecycle_release = None
    lifecycle_manifest = cache_dir.parent / "manifest.json"
    if lifecycle_manifest.exists():
        try:
            manifest_payload = json.loads(lifecycle_manifest.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            manifest_payload = {}
        if manifest_payload.get("release_kind") == "krx500_lifecycle_hybrid":
            source_counts = manifest_payload.get("source_counts") or {}
            proxy_count = int(
                source_counts.get(
                    "finance_data_reader_adjusted_return_index_proxy",
                    0,
                )
            )
            lifecycle_release = audit_lifecycle_hybrid_release(
                lifecycle_manifest,
                expected_tickers=len(universe),
                expected_proxy_tickers=proxy_count,
            )

    event_feature_frames = None
    event_feature_names: list[str] = []
    event_ticker_coverage = None
    event_theme_exposure = None
    event_theme_names: list[str] = []
    event_reports: dict[str, Any] = {}
    fundamental_feature_frames = None
    fundamental_sensor_coverage = None
    fundamental_observed_coverage = None
    investor_feature_frames = None
    investor_sensor_coverage = None
    investor_traded_value_coverage = None
    external_node_feature_frames = None
    external_node_returns = None
    external_node_names: dict[str, str] = {}
    external_factors = []
    factor_closes: dict[str, pd.Series] = {}
    if args.event_path:
        event_feature_frames = build_event_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=args.event_path,
            half_life_days=args.event_half_life_days,
            lag_days=args.event_lag_days,
            max_decay_days=args.event_max_decay_days,
        )
        event_feature_names = list(event_feature_frames)
        event_ticker_coverage = build_event_ticker_coverage(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=args.event_path,
        )
        event_theme_exposure, event_theme_names = build_event_theme_exposure(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=args.event_path,
            half_life_days=args.event_half_life_days,
            lag_days=args.event_lag_days,
            max_decay_days=args.event_max_decay_days,
            max_themes=96,
            min_theme_count=2,
        )
        for raw_path in args.event_path:
            path = Path(raw_path)
            records, load_report = load_jsonl_events(path)
            load_report.update(validate_events(records, set(panel.tickers), panel.close.index))
            event_reports[str(path)] = load_report

    if args.fundamental_path:
        fundamental_feature_frames = build_fundamental_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            observations=load_fundamental_observations(args.fundamental_path),
            lag_days=args.fundamental_lag_days,
        )
        fundamental_sensor_coverage = fundamental_coverage(fundamental_feature_frames)
        fundamental_observed_coverage = fundamental_coverage(
            fundamental_feature_frames,
            eligible_mask=panel.price_observed,
        )

    if args.investor_cache_dir:
        investor_flow_frames = load_investor_flow_frames(
            cache_dir=args.investor_cache_dir,
            dates=panel.close.index,
            tickers=panel.tickers,
        )
        observed_close = panel.close.where(panel.price_observed)
        observed_volume = panel.volume.where(panel.price_observed)
        investor_feature_frames = build_investor_feature_frames(
            investor_flow_frames,
            traded_value=observed_close * observed_volume,
            lag_days=args.investor_flow_lag_days,
        )
        investor_sensor_coverage = investor_feature_coverage(investor_feature_frames)
        investor_traded_value_coverage = investor_feature_coverage(
            investor_feature_frames,
            eligible_mask=panel.price_observed & panel.volume.gt(0.0),
        )

    external_factors = resolve_external_factors(args.external_preset, args.external_symbol)
    if external_factors:
        factor_closes = fetch_external_factor_closes(
            external_factors,
            start=args.start,
            end=args.end,
            cache_dir=args.external_cache_dir,
            refresh=False,
        )
        if args.external_node_mode in {"features", "both"}:
            external_feature_frames = build_external_feature_frames(
                dates=panel.close.index,
                tickers=panel.tickers,
                factor_closes=factor_closes,
                lag_days=args.external_lag_days,
            )
            if external_feature_frames:
                event_feature_frames = dict(event_feature_frames or {})
                event_feature_frames.update(external_feature_frames)
        if args.external_node_mode in {"nodes", "both"}:
            external_node_feature_frames, external_node_returns, external_node_names = build_external_node_feature_frames(
                dates=panel.close.index,
                factor_closes=factor_closes,
                lag_days=args.external_lag_days,
            )

    horizons = parse_int_list(args.path_horizons)
    features = build_feature_panel(
        panel,
        horizon=args.horizon,
        train_end=args.train_end,
        require_targets=True,
        event_feature_frames=event_feature_frames,
        event_feature_names=event_feature_names,
        event_ticker_coverage=event_ticker_coverage,
        fundamental_feature_frames=fundamental_feature_frames,
        investor_feature_frames=investor_feature_frames,
        external_node_feature_frames=external_node_feature_frames,
        external_node_returns=external_node_returns,
        external_node_names=external_node_names,
        event_theme_exposure=event_theme_exposure,
        event_theme_names=event_theme_names,
        path_horizons=horizons,
        warmup_rows=80,
    )

    train_mask = features.dates <= pd.Timestamp(args.train_end)
    train_rows = int(train_mask.sum())
    test_rows = int((~train_mask).sum())
    train_features = features.features[train_mask]
    if args.dump_training_arrays:
        dump_path = Path(args.dump_training_arrays)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        theme_exposure = features.event_theme_exposure
        if theme_exposure is None:
            theme_exposure = np.empty((len(features.dates), len(features.tickers), 0), dtype=np.float32)
        np.savez_compressed(
            dump_path,
            dates=np.asarray([str(date.date()) for date in features.dates[train_mask]]),
            feature_names=np.asarray(features.feature_names),
            features=features.features[train_mask],
            raw_features=features.raw_features[train_mask],
            available_mask=features.available_mask[train_mask],
            train_mean=features.train_mean,
            train_std=features.train_std,
            event_theme_exposure=theme_exposure[train_mask],
        )
    normalized_train_mean_abs_max = float(np.max(np.abs(train_features.mean(axis=(0, 1)))))
    normalized_train_std_min = float(np.min(train_features.std(axis=(0, 1))))
    normalized_train_std_max = float(np.max(train_features.std(axis=(0, 1))))

    raw_nonfinite_by_feature = {}
    for idx, name in enumerate(features.feature_names):
        ratio = float((~np.isfinite(features.raw_features[:, :, idx])).mean())
        if ratio > 0:
            raw_nonfinite_by_feature[name] = ratio

    event_feature_summary = {}
    if event_feature_frames:
        for name, frame in event_feature_frames.items():
            arr = frame.reindex(index=panel.close.index, columns=panel.tickers).fillna(0.0).to_numpy(dtype=float)
            event_feature_summary[name] = {
                "nonzero_cells": int(np.count_nonzero(arr)),
                "abs_sum": float(np.abs(arr).sum()),
                "max_abs": float(np.abs(arr).max()) if arr.size else 0.0,
            }

    issues: list[str] = []
    warnings: list[str] = []
    if not np.isfinite(features.features).all():
        issues.append("normalized feature tensor contains nonfinite values")
    target_nonfinite_cells = int((~np.isfinite(features.target_returns)).sum())
    return_1d_index = features.feature_names.index("return_1d")
    observed_state = (
        features.available_mask[:, : len(features.tickers), return_1d_index] > 0.5
    )
    finite_target = np.isfinite(features.target_returns[:, : len(features.tickers)])
    target_finite_ratio_on_observed_state = (
        float(finite_target[observed_state].mean()) if observed_state.any() else 0.0
    )
    if target_nonfinite_cells:
        warnings.append(
            "target_returns contains some nonfinite node cells; return-head training filters invalid ticker/target pairs"
        )
    if train_rows < args.min_train_feature_rows:
        issues.append(
            f"train rows below required minimum: {train_rows} < "
            f"{args.min_train_feature_rows}"
        )
    if test_rows < args.min_test_feature_rows:
        issues.append(
            f"test rows below required minimum: {test_rows} < "
            f"{args.min_test_feature_rows}"
        )
    if args.event_lag_days < 1:
        issues.append("event lag is below 1 trading row; this risks same-day news leakage")
    if lifecycle_release is not None and lifecycle_release.get("status") != "pass":
        issues.append("lifecycle OHLCV release audit is blocked")

    report = {
        "config": vars(args),
        "training_data_manifest": build_training_data_manifest(features, args.train_end),
        "cache": validate_cache_files(cache_dir),
        "universe": {
            "requested": len(universe),
            "raw_loaded": len(raw_initial),
            "after_train_history_filter": len(raw),
            "panel_tickers": len(panel.tickers),
            "dropped_before_filter": sorted(selected_tickers - set(raw_initial)),
            "dropped_by_train_history_filter": sorted(set(raw_initial) - set(raw)),
            "panel_ticker_sample": panel.tickers[:20],
        },
        "ohlcv": validate_ohlcv(raw, args.train_end),
        "lifecycle_release": lifecycle_release,
        "panel": {
            "dates": len(panel.close.index),
            "date_min": str(panel.close.index.min().date()),
            "date_max": str(panel.close.index.max().date()),
            "tickers": len(panel.tickers),
            "close_nonfinite_cells": int((~np.isfinite(panel.close.to_numpy(dtype=float))).sum()),
            "open_nonfinite_cells": int((~np.isfinite(panel.open.to_numpy(dtype=float))).sum()),
            "volume_zero_cells": int((panel.volume.to_numpy(dtype=float) <= 0).sum()),
            "price_observed_cells": int(panel.price_observed.to_numpy(dtype=bool).sum()),
            "price_unobserved_cells": int((~panel.price_observed.to_numpy(dtype=bool)).sum()),
        },
        "features": {
            "dates": len(features.dates),
            "date_min": str(features.dates.min().date()),
            "date_max": str(features.dates.max().date()),
            "tickers": len(features.tickers),
            "feature_count": len(features.feature_names),
            "feature_names": features.feature_names,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "normalized_nonfinite_cells": int((~np.isfinite(features.features)).sum()),
            "raw_nonfinite_by_feature": raw_nonfinite_by_feature,
            "normalized_train_mean_abs_max": normalized_train_mean_abs_max,
            "normalized_train_std_min": normalized_train_std_min,
            "normalized_train_std_max": normalized_train_std_max,
            "target_nonfinite_cells": target_nonfinite_cells,
            "target_finite_ratio": float(np.isfinite(features.target_returns).mean()),
            "target_finite_ratio_on_observed_state": target_finite_ratio_on_observed_state,
            "path_target_nonfinite_cells": {
                str(h): int((~np.isfinite(arr)).sum())
                for h, arr in features.target_return_paths.items()
            },
        },
        "events": event_reports,
        "event_features": event_feature_summary,
        "fundamental_sensors": {
            "enabled": fundamental_feature_frames is not None,
            "paths": list(args.fundamental_path),
            "feature_count": len(fundamental_feature_frames or {}),
            "coverage": fundamental_sensor_coverage,
            "coverage_on_observed_prices": fundamental_observed_coverage,
            "coverage_gate_basis": "price_observed stock-date cells",
        },
        "investor_sensors": {
            "enabled": investor_feature_frames is not None,
            "cache_dir": args.investor_cache_dir,
            "feature_count": len(investor_feature_frames or {}),
            "coverage": investor_sensor_coverage,
            "coverage_on_observed_traded_value": investor_traded_value_coverage,
            "coverage_gate_basis": "price-observed cells with supported positive volume",
            "lag_days": args.investor_flow_lag_days,
        },
        "external_factors": {
            "preset": args.external_preset,
            "requested": len(external_factors),
            "loaded": len(factor_closes),
            "node_mode": args.external_node_mode,
            "node_count": (
                0 if external_node_returns is None else int(external_node_returns.shape[1])
            ),
        },
        "event_ticker_coverage": {
            "enabled": event_ticker_coverage is not None,
            "covered_tickers": (
                int(event_ticker_coverage.any(axis=0).sum())
                if event_ticker_coverage is not None
                else 0
            ),
            "uncovered_tickers": (
                int((~event_ticker_coverage.any(axis=0)).sum())
                if event_ticker_coverage is not None
                else 0
            ),
        },
        "event_theme_exposure": {
            "enabled": event_theme_exposure is not None,
            "shape": list(event_theme_exposure.shape) if event_theme_exposure is not None else None,
            "themes": len(event_theme_names),
            "theme_sample": event_theme_names[:20],
            "nonzero_cells": int(np.count_nonzero(event_theme_exposure)) if event_theme_exposure is not None else 0,
        },
        "leakage_controls": {
            "event_lag_days": args.event_lag_days,
            "normalization_fit": "train rows only",
            "target_entry": "next trading row open",
            "target_exit": "future close by configured horizon",
        },
        "issues": issues,
        "warnings": warnings,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
