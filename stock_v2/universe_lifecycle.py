from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd


LIFECYCLE_COLUMNS = [
    "ticker",
    "name",
    "market",
    "market_id",
    "security_group",
    "listing_date",
    "delisting_date",
    "source",
]


def normalize_ticker(value: Any) -> str:
    """Return a six-digit Korean equity ticker or an empty string."""

    if value is None or pd.isna(value):
        return ""
    ticker = str(value).strip()
    if ticker.startswith("A") and ticker[1:].isdigit():
        ticker = ticker[1:]
    if not ticker.isdigit() or len(ticker) > 6:
        return ""
    return ticker.zfill(6)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return pd.NaT
    return pd.Timestamp(timestamp).normalize()


def _catalog_records(
    frame: pd.DataFrame,
    *,
    source: str,
    ticker_column: str,
) -> Iterable[Dict[str, object]]:
    for row in frame.to_dict(orient="records"):
        ticker = normalize_ticker(row.get(ticker_column))
        if not ticker:
            continue
        yield {
            "ticker": ticker,
            "name": _text(row.get("Name")),
            "market": _text(row.get("Market")),
            "market_id": _text(row.get("MarketId")),
            "security_group": _text(row.get("SecuGroup")),
            "listing_date": _timestamp(row.get("ListingDate")),
            "delisting_date": _timestamp(row.get("DelistingDate")),
            "source": source,
        }


def build_lifecycle_catalog(
    current_listing: pd.DataFrame,
    delisted_listing: pd.DataFrame,
) -> pd.DataFrame:
    """Combine active and delisted KRX listings into dated security lifecycles."""

    records = [
        *_catalog_records(current_listing, source="current", ticker_column="Code"),
        *_catalog_records(delisted_listing, source="delisted", ticker_column="Symbol"),
    ]
    if not records:
        return pd.DataFrame(columns=LIFECYCLE_COLUMNS)
    catalog = pd.DataFrame.from_records(records, columns=LIFECYCLE_COLUMNS)
    catalog["listing_date"] = pd.to_datetime(catalog["listing_date"], errors="coerce")
    catalog["delisting_date"] = pd.to_datetime(catalog["delisting_date"], errors="coerce")
    return catalog.sort_values(
        ["ticker", "listing_date", "delisting_date", "source"],
        na_position="last",
    ).reset_index(drop=True)


def enrich_current_listing_dates(
    current_listing: pd.DataFrame,
    descriptive_listing: pd.DataFrame,
) -> pd.DataFrame:
    """Fill active-listing dates from the KRX descriptive listing feed."""

    if "Code" not in current_listing.columns:
        raise ValueError("current listing must include Code")
    if not {"Code", "ListingDate"}.issubset(descriptive_listing.columns):
        raise ValueError("descriptive listing must include Code and ListingDate")

    descriptions = descriptive_listing[["Code", "ListingDate"]].copy()
    descriptions["_ticker"] = descriptions["Code"].map(normalize_ticker)
    descriptions["ListingDate"] = pd.to_datetime(
        descriptions["ListingDate"],
        errors="coerce",
    )
    descriptions = descriptions.drop_duplicates(subset="_ticker", keep="last")
    dates = descriptions.set_index("_ticker")["ListingDate"]

    enriched = current_listing.copy()
    enriched["_ticker"] = enriched["Code"].map(normalize_ticker)
    existing = (
        pd.to_datetime(enriched["ListingDate"], errors="coerce")
        if "ListingDate" in enriched.columns
        else pd.Series(pd.NaT, index=enriched.index, dtype="datetime64[ns]")
    )
    enriched["ListingDate"] = existing.fillna(enriched["_ticker"].map(dates))
    return enriched.drop(columns="_ticker")


def select_lifecycle_universe(
    catalog: pd.DataFrame,
    *,
    as_of: str | pd.Timestamp,
    markets: Iterable[str] = ("KOSPI", "KOSDAQ"),
    require_common_stock: bool = True,
    exclude_name_pattern: str | None = None,
) -> pd.DataFrame:
    """Select securities that were actually active on the requested date."""

    required = set(LIFECYCLE_COLUMNS)
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"lifecycle catalog missing columns: {sorted(missing)}")
    cutoff = _timestamp(as_of)
    if pd.isna(cutoff):
        raise ValueError(f"invalid as-of date: {as_of!r}")

    selected = catalog.copy()
    selected["listing_date"] = pd.to_datetime(selected["listing_date"], errors="coerce")
    selected["delisting_date"] = pd.to_datetime(selected["delisting_date"], errors="coerce")
    selected = selected[
        selected["ticker"].astype(str).str.fullmatch(r"\d{6}", na=False)
        & selected["listing_date"].notna()
        & (selected["listing_date"] <= cutoff)
        & (selected["delisting_date"].isna() | (selected["delisting_date"] >= cutoff))
        & selected["market"].isin(set(markets))
    ].copy()

    if require_common_stock:
        is_current_common = (
            selected["source"].eq("current")
            & selected["market_id"].isin({"STK", "KSQ"})
        )
        is_delisted_common = (
            selected["source"].eq("delisted")
            & selected["security_group"].eq("\uc8fc\uad8c")
        )
        selected = selected[is_current_common | is_delisted_common].copy()

    if exclude_name_pattern:
        selected = selected[
            ~selected["name"].astype(str).str.contains(
                exclude_name_pattern,
                case=False,
                regex=True,
                na=False,
            )
        ].copy()

    selected = selected.sort_values(
        ["ticker", "listing_date", "delisting_date", "source"],
        na_position="last",
    )
    selected = selected.drop_duplicates(subset="ticker", keep="last")
    return selected.sort_values("ticker").reset_index(drop=True)


def summarize_trailing_turnover(
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> Dict[str, float]:
    """Calculate a causal liquidity score from close times volume."""

    if not {"Close", "Volume"}.issubset(frame.columns):
        return {
            "observations": 0.0,
            "mean_turnover": float("nan"),
            "median_turnover": float("nan"),
        }
    window = frame.copy()
    window.index = pd.to_datetime(window.index, errors="coerce")
    window = window.loc[
        (window.index >= _timestamp(start)) & (window.index <= _timestamp(end))
    ]
    close = pd.to_numeric(window["Close"], errors="coerce")
    volume = pd.to_numeric(window["Volume"], errors="coerce")
    turnover = (close * volume).where((close > 0.0) & (volume > 0.0))
    valid = turnover[np.isfinite(turnover)]
    return {
        "observations": float(len(valid)),
        "mean_turnover": float(valid.mean()) if len(valid) else float("nan"),
        "median_turnover": float(valid.median()) if len(valid) else float("nan"),
    }


def rank_lifecycle_universe(
    catalog: pd.DataFrame,
    scores: Mapping[str, Mapping[str, float]],
    *,
    top_n: int,
    min_observations: int,
    turnover_key: str = "median_turnover",
) -> pd.DataFrame:
    """Attach causal liquidity scores and return the highest-ranked securities."""

    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    if turnover_key not in {"mean_turnover", "median_turnover"}:
        raise ValueError("turnover_key must be 'mean_turnover' or 'median_turnover'")
    ranked = catalog.copy()
    ranked["rank_observations"] = ranked["ticker"].map(
        lambda ticker: float(scores.get(str(ticker), {}).get("observations", 0.0))
    )
    ranked["trailing_turnover"] = ranked["ticker"].map(
        lambda ticker: float(
            scores.get(str(ticker), {}).get(turnover_key, float("nan"))
        )
    )
    ranked = ranked[
        (ranked["rank_observations"] >= float(min_observations))
        & np.isfinite(ranked["trailing_turnover"])
        & (ranked["trailing_turnover"] > 0.0)
    ].copy()
    ranked = ranked.sort_values(
        ["trailing_turnover", "ticker"],
        ascending=[False, True],
    ).head(top_n)
    ranked["liquidity_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    return ranked.reset_index(drop=True)
