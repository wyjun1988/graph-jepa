from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stock_v2.dataset_integrity import (
    load_json,
    load_jsonl,
    normalize_ticker,
    select_ohlcv_cache,
    sha256_file,
)
from stock_v2.news_dataset import load_calendar


INVESTOR_NET_COLUMNS = (
    "investor_individual_net_m",
    "investor_foreign_net_m",
    "investor_institution_net_m",
    "investor_financial_net_m",
    "investor_pension_net_m",
)
INVESTOR_OUTPUT_COLUMNS = ("investor_traded_volume", *INVESTOR_NET_COLUMNS)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _next_session(calendar: pd.DatetimeIndex, date: pd.Timestamp) -> str | None:
    values = calendar[calendar > date.normalize()]
    return str(values[0].date()) if len(values) else None


def load_universe(path: Path) -> dict[str, dict[str, Any]]:
    payload = load_json(path)
    return {
        normalize_ticker(row.get("ticker")): dict(row)
        for row in payload.get("universe", [])
        if normalize_ticker(row.get("ticker"))
    }


def curate_fundamentals(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    output_dir: Path,
) -> dict[str, Any]:
    observation_values = list(config.get("observations_paths", []))
    if not observation_values:
        observation_values = [str(config["observations_path"])]
    source_paths = [repo_root / str(value) for value in observation_values]
    profile_path = repo_root / str(config["profiles_path"])
    source_batches: list[list[dict[str, Any]]] = []
    load_reports: dict[str, Any] = {}
    for value, source_path in zip(observation_values, source_paths):
        batch, report = load_jsonl(source_path)
        if report["invalid_json"] or report["invalid_rows"]:
            raise ValueError(f"invalid fundamental observations: {source_path}")
        source_batches.append(batch)
        load_reports[str(value)] = report
    latest_source_by_ticker: dict[str, int] = {}
    for source_index, batch in enumerate(source_batches):
        for row in batch:
            ticker = normalize_ticker(row.get("ticker"))
            if ticker:
                latest_source_by_ticker[ticker] = source_index
    rows = [
        row
        for source_index, batch in enumerate(source_batches)
        for row in batch
        if latest_source_by_ticker.get(normalize_ticker(row.get("ticker"))) == source_index
    ]
    profiles, profile_load_report = load_jsonl(profile_path)
    if profile_load_report["invalid_json"] or profile_load_report["invalid_rows"]:
        raise ValueError(f"invalid fundamental profiles: {profile_path}")

    release_end = pd.Timestamp(config["release_end"]).normalize()
    unreliable_period_tickers: set[str] = set()
    for row in rows:
        ticker = normalize_ticker(row.get("ticker"))
        available = pd.to_datetime(row.get("available_at"), errors="coerce")
        period_end = pd.to_datetime(row.get("period_end"), errors="coerce")
        if ticker in universe and not pd.isna(available) and not pd.isna(period_end) and available < period_end:
            unreliable_period_tickers.add(ticker)

    accepted: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    seen: set[str] = set()
    reasons: Counter[str] = Counter()
    for row in rows:
        ticker = normalize_ticker(row.get("ticker"))
        available = pd.to_datetime(row.get("available_at"), errors="coerce")
        period_end = pd.to_datetime(row.get("period_end"), errors="coerce")
        fields = row.get("fields")
        reason = ""
        if ticker not in universe:
            reason = "outside_universe"
        elif ticker in unreliable_period_tickers:
            reason = "unreliable_fiscal_period_mapping"
        elif pd.isna(available) or pd.isna(period_end):
            reason = "invalid_date"
        elif available.normalize() > release_end:
            reason = "after_release_end"
        elif not isinstance(fields, Mapping) or not fields:
            reason = "empty_fields"
        else:
            try:
                fields_are_finite = all(math.isfinite(float(value)) for value in fields.values())
            except (TypeError, ValueError, OverflowError):
                fields_are_finite = False
            if not fields_are_finite:
                reason = "nonfinite_field"
        if reason:
            reasons[reason] += 1
            quarantine.append({"reason": reason, "record": row})
            continue
        effective_session = _next_session(calendar, pd.Timestamp(available))
        if effective_session is None:
            reasons["not_yet_effective"] += 1
            quarantine.append({"reason": "not_yet_effective", "record": row})
            continue
        canonical = {
            "schema_version": 1,
            "ticker": ticker,
            "available_at": str(pd.Timestamp(available).date()),
            "effective_session": effective_session,
            "availability_policy": "next_krx_session",
            "period_end": str(pd.Timestamp(period_end).date()),
            "source": str(row.get("source") or "opendart"),
            "fields": {str(key): float(value) for key, value in fields.items()},
        }
        if isinstance(row.get("source_lineage"), Mapping):
            canonical["source_lineage"] = dict(row["source_lineage"])
        identity = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if identity in seen:
            reasons["duplicate"] += 1
            quarantine.append({"reason": "duplicate", "record": row})
            continue
        seen.add(identity)
        accepted.append(canonical)
    accepted.sort(key=lambda row: (str(row["ticker"]), str(row["available_at"]), str(row["period_end"])))

    profile_by_ticker: dict[str, dict[str, Any]] = {}
    for row in profiles:
        ticker = normalize_ticker(row.get("ticker"))
        if ticker in universe and ticker not in profile_by_ticker:
            profile_by_ticker[ticker] = {**row, "ticker": ticker, "schema_version": 1}
    profile_rows = [profile_by_ticker[ticker] for ticker in sorted(profile_by_ticker)]

    output_dir.mkdir(parents=True, exist_ok=True)
    observation_output = output_dir / "observations.jsonl"
    profile_output = output_dir / "profiles.jsonl"
    quarantine_output = output_dir / "quarantine.jsonl"
    _write_jsonl(observation_output, accepted)
    _write_jsonl(profile_output, profile_rows)
    _write_jsonl(quarantine_output, quarantine)
    report = {
        "schema_version": 1,
        "source": {"provider": "OpenDART", "official": True},
        "source_files": {
            **{
                str(value): sha256_file(source_path)
                for value, source_path in zip(observation_values, source_paths)
            },
            str(config["profiles_path"]): sha256_file(profile_path),
        },
        "source_load_reports": load_reports,
        "replacement_tickers": sorted(
            ticker for ticker, source_index in latest_source_by_ticker.items() if source_index > 0
        ),
        "input_rows": len(rows),
        "accepted_rows": len(accepted),
        "accepted_tickers": len({row["ticker"] for row in accepted}),
        "profile_tickers": len(profile_rows),
        "unreliable_period_tickers": sorted(unreliable_period_tickers),
        "quarantine_rows": len(quarantine),
        "quarantine_reasons": dict(reasons),
        "output_files": {
            "observations.jsonl": sha256_file(observation_output),
            "profiles.jsonl": sha256_file(profile_output),
            "quarantine.jsonl": sha256_file(quarantine_output),
        },
    }
    _write_json(output_dir / "manifest.json", report)
    return report


def curate_investor(
    repo_root: Path,
    config: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
    calendar: pd.DatetimeIndex,
    output_dir: Path,
) -> dict[str, Any]:
    source_dir = repo_root / str(config["cache_dir"])
    start = pd.Timestamp(config["start"]).normalize()
    end = pd.Timestamp(config["end"]).normalize()
    valid_sessions = set(calendar)
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    output_files: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    rows_written = 0
    tickers_with_data = 0
    missing_tickers: list[str] = []

    for ticker, metadata in universe.items():
        listing = pd.to_datetime(metadata.get("listing_date"), errors="coerce")
        delisting = pd.to_datetime(metadata.get("delisting_date"), errors="coerce")
        required_start = max(start, pd.Timestamp(listing).normalize()) if not pd.isna(listing) else start
        required_end = min(end, pd.Timestamp(delisting).normalize()) if not pd.isna(delisting) else end
        source_path, _covers = select_ohlcv_cache(source_dir, ticker, required_start, required_end)
        if source_path is None:
            missing_tickers.append(ticker)
            frame = pd.DataFrame(columns=["date", *INVESTOR_OUTPUT_COLUMNS])
        else:
            source_files.append(
                {"ticker": ticker, "path": str(source_path.relative_to(repo_root)), "sha256": sha256_file(source_path)}
            )
            raw = pd.read_csv(source_path)
            if "investor_traded_volume" not in raw and "investor_traded_value_m" in raw:
                raw = raw.rename(columns={"investor_traded_value_m": "investor_traded_volume"})
            if "date" not in raw or not set(INVESTOR_OUTPUT_COLUMNS).issubset(raw.columns):
                raise ValueError(f"invalid investor schema: {source_path}")
            dates = pd.to_datetime(raw["date"], errors="coerce").dt.normalize()
            numeric = raw[list(INVESTOR_OUTPUT_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            accepted_indices: list[int] = []
            for index in raw.index:
                date = dates.loc[index]
                reason = ""
                if pd.isna(date):
                    reason = "invalid_date"
                elif not np.isfinite(numeric.loc[index].to_numpy(dtype=float)).all():
                    reason = "nonfinite_value"
                elif (numeric.loc[index, "investor_traded_volume"] < 0.0):
                    reason = "negative_volume"
                elif date < required_start or date > required_end:
                    reason = "outside_security_lifecycle"
                elif date not in valid_sessions:
                    reason = "nontrading_date"
                if reason:
                    reasons[reason] += 1
                    quarantine.append(
                        {"ticker": ticker, "date": None if pd.isna(date) else str(date.date()), "reason": reason}
                    )
                else:
                    accepted_indices.append(index)
            frame = pd.concat(
                [dates.loc[accepted_indices].rename("date"), numeric.loc[accepted_indices]],
                axis=1,
            ).sort_values("date")
            frame = frame.drop_duplicates(subset=["date"], keep="last")
        output_path = output_dir / f"{ticker}_{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.csv"
        temporary = output_path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(output_path)
        rows_written += len(frame)
        tickers_with_data += int(bool(len(frame)))
        output_files.append(
            {"ticker": ticker, "path": output_path.name, "rows": len(frame), "sha256": sha256_file(output_path)}
        )

    _write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    report = {
        "schema_version": 1,
        "source": {
            "provider": "Kiwoom REST ka10060",
            "official_broker_api": True,
            "request_mode": "amount",
            "net_flow_unit": "KRW million",
            "accumulated_field_semantics": "traded volume (empirically reconciled; legacy header corrected)",
        },
        "availability_policy": "one_session_lag_in_feature_builder",
        "expected_tickers": len(universe),
        "tickers_with_data": tickers_with_data,
        "missing_tickers": missing_tickers,
        "rows_written": rows_written,
        "quarantine_rows": len(quarantine),
        "quarantine_reasons": dict(reasons),
        "source_files": source_files,
        "output_files": output_files,
        "quarantine_sha256": sha256_file(output_dir / "quarantine.jsonl"),
    }
    _write_json(output_dir / "manifest.json", report)
    return report


def curate_structured_sources(
    repo_root: Path,
    config: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    universe = load_universe(repo_root / str(config["universe_manifest"]))
    start = pd.Timestamp(config["release_window"]["start"]).normalize()
    end = pd.Timestamp(config["release_window"]["end"]).normalize()
    calendar = load_calendar(
        [repo_root / str(path) for path in config["calendar_paths"]],
        start,
        end,
    )
    if not len(calendar):
        raise ValueError("structured-source curation requires a trading calendar")
    fundamental_report = curate_fundamentals(
        repo_root,
        config["fundamentals"],
        universe,
        calendar,
        output_dir / "fundamentals",
    )
    investor_report = curate_investor(
        repo_root,
        config["investor"],
        universe,
        calendar,
        output_dir / "investor",
    )
    report = {
        "schema_version": 1,
        "release_window": config["release_window"],
        "universe_tickers": len(universe),
        "calendar_sessions": len(calendar),
        "fundamentals": fundamental_report,
        "investor": investor_report,
    }
    _write_json(output_dir / "manifest.json", report)
    return report
