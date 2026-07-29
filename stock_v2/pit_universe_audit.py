from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


AUDIT_CONTRACT = "point_in_time_liquidity_universe_audit_v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _parse_date(value: Any, label: str) -> date:
    try:
        return pd.Timestamp(str(value)).date()
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {label}: {value}") from exc


def audit_point_in_time_universe(
    universe_path: str | Path,
    *,
    failures_path: str | Path,
    comparison_universe_path: str | Path | None = None,
    evaluation_end: str | None = None,
) -> dict[str, Any]:
    path = Path(universe_path)
    failure_path = Path(failures_path)
    _require(path.is_file(), "point-in-time universe is missing")
    _require(failure_path.is_file(), "point-in-time universe failure file is missing")
    payload = json.loads(path.read_text(encoding="utf-8"))
    failures = json.loads(failure_path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 2, "unsupported universe schema")
    _require(isinstance(failures, dict), "universe failures must be a mapping")

    policy = payload["selection_policy"]
    _require(
        policy.get("type") == "point_in_time_trailing_turnover",
        "universe is not point-in-time trailing-turnover selected",
    )
    as_of = _parse_date(policy["as_of"], "as_of")
    rank_start = _parse_date(policy["rank_start"], "rank_start")
    rank_end = _parse_date(policy["rank_end"], "rank_end")
    _require(rank_start <= rank_end <= as_of, "universe ranking window leaks past as_of")
    top_n = int(policy["top_n"])
    minimum_observations = int(policy["min_observations"])
    _require(top_n > 0 and minimum_observations > 0, "universe gates must be positive")
    _require(policy.get("require_common_stock") is True, "universe permits non-common stock")

    rows = payload["universe"]
    _require(len(rows) == top_n, "universe row count differs from top_n")
    tickers = tuple(str(row["ticker"]) for row in rows)
    _require(len(tickers) == len(set(tickers)), "universe tickers are duplicated")
    _require(all(re.fullmatch(r"\d{6}", ticker) for ticker in tickers), "universe contains a malformed ticker")
    ranks = tuple(int(row["liquidity_rank"]) for row in rows)
    _require(ranks == tuple(range(1, top_n + 1)), "liquidity ranks are not contiguous and ordered")
    turnovers = np.asarray([float(row["trailing_turnover"]) for row in rows])
    _require(bool((np.isfinite(turnovers) & (turnovers > 0.0)).all()), "universe turnover is non-positive or non-finite")
    _require(bool((np.diff(turnovers) <= 0.0).all()), "universe turnover is not descending")
    observations = np.asarray([int(row["rank_observations"]) for row in rows])
    _require(bool((observations >= minimum_observations).all()), "universe contains an under-observed rank history")

    excluded_pattern = str(policy.get("exclude_name_pattern") or "")
    excluded = re.compile(excluded_pattern) if excluded_pattern else None
    active_at_end = 0
    end_date = _parse_date(evaluation_end, "evaluation_end") if evaluation_end else as_of
    delisted_during_evaluation: list[str] = []
    for row in rows:
        ticker = str(row["ticker"])
        listing = _parse_date(row["listing_date"], f"listing_date:{ticker}")
        _require(listing <= as_of, f"ticker listed after universe as_of: {ticker}")
        delisting_value = row.get("delisting_date")
        delisting = (
            _parse_date(delisting_value, f"delisting_date:{ticker}")
            if delisting_value
            else None
        )
        _require(delisting is None or delisting > as_of, f"ticker was inactive at universe as_of: {ticker}")
        if delisting is None or delisting > end_date:
            active_at_end += 1
        elif delisting > as_of:
            delisted_during_evaluation.append(ticker)
        if excluded is not None:
            _require(excluded.search(str(row["name"])) is None, f"excluded security name entered universe: {ticker}")

    counts = payload["source_counts"]
    _require(int(counts["eligible_as_of"]) >= top_n, "eligible universe is smaller than top_n")
    _require(int(counts["rank_history_success"]) >= top_n, "too few successful rank histories")
    _require(int(counts["rank_history_failures"]) == len(failures), "failure count does not match failure artifact")
    _require(not failures, "point-in-time universe has rank-history failures")

    comparison: dict[str, Any] | None = None
    if comparison_universe_path is not None:
        comparison_path = Path(comparison_universe_path)
        _require(comparison_path.is_file(), "comparison universe is missing")
        previous = json.loads(comparison_path.read_text(encoding="utf-8"))
        previous_tickers = {
            str(row["ticker"]) for row in previous.get("universe", ())
        }
        overlap = set(tickers) & previous_tickers
        comparison = {
            "path": str(comparison_path),
            "sha256": file_sha256(comparison_path),
            "stocks": len(previous_tickers),
            "overlap": len(overlap),
            "new_tickers": top_n - len(overlap),
            "removed_tickers": len(previous_tickers - set(tickers)),
            "jaccard": len(overlap) / len(set(tickers) | previous_tickers),
        }

    return {
        "schema_version": 1,
        "audit_contract": AUDIT_CONTRACT,
        "passed": True,
        "universe": str(path),
        "universe_sha256": file_sha256(path),
        "failures_sha256": file_sha256(failure_path),
        "as_of": as_of.isoformat(),
        "rank_start": rank_start.isoformat(),
        "rank_end": rank_end.isoformat(),
        "stocks": top_n,
        "rank_history_success": int(counts["rank_history_success"]),
        "rank_history_failures": len(failures),
        "minimum_observations": int(observations.min()),
        "evaluation_end": end_date.isoformat(),
        "active_at_evaluation_end": active_at_end,
        "delisted_during_evaluation": delisted_during_evaluation,
        "comparison": comparison,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
