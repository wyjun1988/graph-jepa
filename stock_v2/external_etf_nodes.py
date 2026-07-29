from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PANEL_ROLE = "us_etf_cross_source_consensus_daily_panel"
EXPECTED_FEATURE_ALLOWLIST = (
    "TotalLogReturn",
    "LogVolumeShock",
    "EventFresh",
    "LogAvailabilityAgeHours",
)
FEATURE_NAMES = {
    "TotalLogReturn": "ext_etf_total_log_return",
    "LogVolumeShock": "ext_etf_log_volume_shock",
    "EventFresh": "ext_etf_event_fresh",
    "LogAvailabilityAgeHours": "ext_etf_log_availability_age_hours",
}


@dataclass(frozen=True)
class ExternalEtfNodeInputs:
    feature_frames: dict[str, pd.DataFrame]
    returns: pd.DataFrame
    names: dict[str, str]
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _validate_release(root: Path) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    contract_path = root / "contract.json"
    panel_path = root / "panel.parquet"
    summary_path = root / "summary.json"
    for path in (contract_path, panel_path, summary_path):
        if not path.is_file():
            raise ValueError(f"ETF panel artifact missing: {path}")

    contract = _load_json(contract_path)
    summary = _load_json(summary_path)
    if contract.get("role") != PANEL_ROLE or summary.get("role") != PANEL_ROLE:
        raise ValueError("invalid ETF panel role")
    if summary.get("status") != "complete":
        raise ValueError("ETF panel release is incomplete")
    for payload in (contract, summary):
        if payload.get("live_orders_allowed") is not False:
            raise ValueError("ETF panel must explicitly prohibit live orders")
        if payload.get("promotion_eligible") is not False:
            raise ValueError("ETF panel must remain diagnostic-only")
        if payload.get("external_nodes_masked_for_prediction_loss") is not False:
            raise ValueError("ETF nodes must remain input-only observations")
    if contract.get("diagnostic_model_experiment_eligible") is not True:
        raise ValueError("ETF panel is not eligible for a diagnostic model experiment")
    if tuple(contract.get("feature_allowlist", ())) != EXPECTED_FEATURE_ALLOWLIST:
        raise ValueError("ETF panel feature allowlist does not match the causal v2 contract")
    if tuple(summary.get("feature_allowlist", ())) != EXPECTED_FEATURE_ALLOWLIST:
        raise ValueError("ETF panel summary feature allowlist does not match its contract")
    if summary.get("contract_sha256") != _sha256(contract_path):
        raise ValueError("ETF panel contract hash changed")
    if summary.get("panel_sha256") != _sha256(panel_path):
        raise ValueError("ETF panel parquet hash changed")

    panel = pd.read_parquet(panel_path)
    required = {
        "Exchange",
        "Ticker",
        "Channel",
        "Date",
        "AvailableAtUTC",
        "TotalReturnValid",
        "TotalLogReturn",
        "VolumeFeatureValid",
        "LogVolumeShock",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"ETF panel columns missing: {missing}")
    panel = panel.copy()
    panel["Exchange"] = panel["Exchange"].astype(str)
    panel["Ticker"] = panel["Ticker"].astype(str)
    panel["Channel"] = panel["Channel"].astype(str)
    panel["Date"] = pd.to_datetime(panel["Date"], errors="raise").dt.normalize()
    panel["AvailableAtUTC"] = pd.to_datetime(
        panel["AvailableAtUTC"], utc=True, errors="raise"
    )
    if panel.duplicated(["Exchange", "Ticker", "Date"]).any():
        raise ValueError("ETF panel has duplicate node sessions")
    if panel[["TotalReturnValid", "VolumeFeatureValid"]].isna().any().any():
        raise ValueError("ETF panel validity flags contain missing values")
    node_count = panel[["Exchange", "Ticker"]].drop_duplicates().shape[0]
    if int(summary.get("nodes", -1)) != int(node_count):
        raise ValueError("ETF panel node count differs from its summary")
    if int(summary.get("rows", -1)) != len(panel):
        raise ValueError("ETF panel row count differs from its summary")
    return panel, contract, summary


def _krx_cutoffs(
    dates: pd.DatetimeIndex,
    cutoff_local_time: str,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    normalized = pd.DatetimeIndex(dates)
    if normalized.tz is not None:
        normalized = normalized.tz_convert("Asia/Seoul").tz_localize(None)
    normalized = normalized.normalize()
    if normalized.has_duplicates or not normalized.is_monotonic_increasing:
        raise ValueError("KRX dates must be unique and increasing")
    try:
        local = pd.DatetimeIndex(
            normalized.strftime("%Y-%m-%d") + " " + str(cutoff_local_time)
        ).tz_localize("Asia/Seoul", ambiguous="raise", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid KRX cutoff time: {cutoff_local_time}") from exc
    return normalized, local.tz_convert("UTC")


def _node_id(exchange: str, ticker: str) -> str:
    return f"EXTETF:{exchange}:{ticker}"


def _aggregate_node_events(
    source: pd.DataFrame,
    cutoffs_utc: pd.DatetimeIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    rows = len(cutoffs_utc)
    total_return = np.full(rows, np.nan, dtype=np.float32)
    volume_shock = np.full(rows, np.nan, dtype=np.float32)
    fresh = np.full(rows, np.nan, dtype=np.float32)
    age = np.full(rows, np.nan, dtype=np.float32)

    source = source.sort_values(["AvailableAtUTC", "Date"], kind="stable")
    if source["AvailableAtUTC"].duplicated().any():
        raise ValueError("ETF node has duplicate availability timestamps")
    if not source["AvailableAtUTC"].is_monotonic_increasing:
        raise ValueError("ETF availability timestamps are not increasing")

    cutoff_ns = cutoffs_utc.asi8
    availability_ns = pd.DatetimeIndex(source["AvailableAtUTC"]).asi8
    buckets = np.searchsorted(cutoff_ns, availability_ns, side="left")
    visible = buckets < rows
    source = source.loc[visible].copy()
    source["_bucket"] = buckets[visible]

    latest_availability: pd.Timestamp | None = None
    grouped = {int(key): value for key, value in source.groupby("_bucket", sort=True)}
    fresh_events = 0
    bundled_events = 0
    invalid_return_bundles = 0
    invalid_volume_bundles = 0
    warm_start_discarded_events = 0
    for row_index, cutoff in enumerate(cutoffs_utc):
        events = grouped.get(row_index)
        if events is not None:
            if row_index == 0 and len(events) > 1:
                warm_start_discarded_events += int(len(events)) - 1
                events = events.tail(1)
            fresh_events += int(len(events))
            if row_index > 0:
                bundled_events += max(0, int(len(events)) - 1)
            latest_availability = pd.Timestamp(events["AvailableAtUTC"].iloc[-1])
            fresh[row_index] = 1.0

            return_valid = events["TotalReturnValid"].astype(bool).all()
            return_values = pd.to_numeric(events["TotalLogReturn"], errors="coerce")
            if return_valid and np.isfinite(return_values.to_numpy(dtype=float)).all():
                total_return[row_index] = float(return_values.sum())
            else:
                invalid_return_bundles += 1

            latest = events.iloc[-1]
            latest_volume = pd.to_numeric(
                pd.Series([latest["LogVolumeShock"]]), errors="coerce"
            ).iloc[0]
            if bool(latest["VolumeFeatureValid"]) and np.isfinite(latest_volume):
                volume_shock[row_index] = float(latest_volume)
            else:
                invalid_volume_bundles += 1
        elif latest_availability is not None:
            total_return[row_index] = 0.0
            volume_shock[row_index] = 0.0
            fresh[row_index] = 0.0

        if latest_availability is not None:
            age_hours = (cutoff - latest_availability).total_seconds() / 3600.0
            if age_hours < -1e-9:
                raise ValueError("future ETF observation reached a KRX cutoff")
            age[row_index] = float(np.log1p(max(0.0, age_hours)))

    stats = {
        "source_events_visible": fresh_events,
        "bundled_holiday_events": bundled_events,
        "invalid_return_bundles": invalid_return_bundles,
        "invalid_volume_bundles": invalid_volume_bundles,
        "warm_start_discarded_events": warm_start_discarded_events,
    }
    return total_return, volume_shock, fresh, age, stats


def load_external_etf_node_inputs(
    panel_root: str | Path,
    dates: pd.DatetimeIndex,
    *,
    krx_cutoff_local_time: str = "15:30",
) -> ExternalEtfNodeInputs:
    """Load causal, event-once US ETF nodes for KRX daily model dates."""

    root = Path(panel_root).expanduser()
    panel, contract, summary = _validate_release(root)
    krx_dates, cutoffs_utc = _krx_cutoffs(dates, krx_cutoff_local_time)

    keys = (
        panel[["Exchange", "Ticker"]]
        .drop_duplicates()
        .sort_values(["Exchange", "Ticker"], kind="stable")
    )
    node_ids = [
        _node_id(str(row.Exchange), str(row.Ticker))
        for row in keys.itertuples(index=False)
    ]
    frames = {
        feature_name: pd.DataFrame(
            np.nan, index=krx_dates, columns=node_ids, dtype=np.float32
        )
        for feature_name in FEATURE_NAMES.values()
    }
    returns = pd.DataFrame(
        np.nan, index=krx_dates, columns=node_ids, dtype=np.float32
    )
    names: dict[str, str] = {}
    totals = {
        "source_events_visible": 0,
        "bundled_holiday_events": 0,
        "invalid_return_bundles": 0,
        "invalid_volume_bundles": 0,
        "warm_start_discarded_events": 0,
    }

    for row in keys.itertuples(index=False):
        exchange = str(row.Exchange)
        ticker = str(row.Ticker)
        node_id = _node_id(exchange, ticker)
        source = panel.loc[
            panel["Exchange"].eq(exchange) & panel["Ticker"].eq(ticker)
        ]
        channels = source["Channel"].drop_duplicates().tolist()
        if len(channels) != 1:
            raise ValueError(f"ETF node channel changed: {node_id}")
        total_return, volume, fresh, age, stats = _aggregate_node_events(
            source, cutoffs_utc
        )
        frames[FEATURE_NAMES["TotalLogReturn"]][node_id] = total_return
        frames[FEATURE_NAMES["LogVolumeShock"]][node_id] = volume
        frames[FEATURE_NAMES["EventFresh"]][node_id] = fresh
        frames[FEATURE_NAMES["LogAvailabilityAgeHours"]][node_id] = age
        returns[node_id] = total_return
        names[node_id] = f"{ticker} ({channels[0]})"
        for key, value in stats.items():
            totals[key] += int(value)

    first_cutoff = cutoffs_utc[0].isoformat() if len(cutoffs_utc) else None
    last_cutoff = cutoffs_utc[-1].isoformat() if len(cutoffs_utc) else None
    audit = {
        "role": "causal_us_etf_external_node_inputs",
        "panel_root": str(root),
        "contract_sha256": summary["contract_sha256"],
        "panel_sha256": summary["panel_sha256"],
        "nodes": len(node_ids),
        "krx_rows": len(krx_dates),
        "first_krx_cutoff_utc": first_cutoff,
        "last_krx_cutoff_utc": last_cutoff,
        "krx_cutoff_local_time": krx_cutoff_local_time,
        "feature_names": list(frames),
        "event_consumption": "once_at_first_krx_cutoff",
        "external_nodes_masked_for_prediction_loss": False,
        "counts_as_primary_forward_evidence": False,
        "counts_as_training_release": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        **totals,
    }
    return ExternalEtfNodeInputs(frames, returns, names, audit)


def merge_external_node_inputs(
    base_feature_frames: Mapping[str, pd.DataFrame] | None,
    base_returns: pd.DataFrame | None,
    base_names: Mapping[str, str] | None,
    added: ExternalEtfNodeInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Merge two external-node modalities while preserving unavailable cells."""

    base_frames = dict(base_feature_frames or {})
    base_names_dict = dict(base_names or {})
    if base_returns is None:
        base_columns: list[str] = []
        index = added.returns.index
        base_returns = pd.DataFrame(index=index, dtype=np.float32)
    else:
        base_returns = base_returns.copy()
        base_columns = [str(column) for column in base_returns.columns]
        index = base_returns.index
    if not index.equals(added.returns.index):
        raise ValueError("external node inputs use different date indices")

    added_columns = [str(column) for column in added.returns.columns]
    duplicates = sorted(set(base_columns).intersection(added_columns))
    if duplicates:
        raise ValueError(f"duplicate external node IDs: {duplicates}")
    if set(base_names_dict).difference(base_columns):
        raise ValueError("base external node names include unknown IDs")
    if set(added.names) != set(added_columns):
        raise ValueError("added external node names do not match return columns")

    combined_columns = base_columns + added_columns
    combined_frames: dict[str, pd.DataFrame] = {}
    for feature_name in list(base_frames) + [
        name for name in added.feature_frames if name not in base_frames
    ]:
        left = base_frames.get(feature_name)
        if left is None:
            left = pd.DataFrame(
                np.nan, index=index, columns=base_columns, dtype=np.float32
            )
        else:
            left = left.reindex(index=index, columns=base_columns)
        right = added.feature_frames.get(feature_name)
        if right is None:
            right = pd.DataFrame(
                np.nan, index=index, columns=added_columns, dtype=np.float32
            )
        else:
            right = right.reindex(index=index, columns=added_columns)
        combined_frames[feature_name] = pd.concat([left, right], axis=1).reindex(
            columns=combined_columns
        )

    combined_returns = pd.concat(
        [base_returns.reindex(columns=base_columns), added.returns], axis=1
    ).reindex(columns=combined_columns)
    combined_names = {**base_names_dict, **added.names}
    return combined_frames, combined_returns, combined_names
