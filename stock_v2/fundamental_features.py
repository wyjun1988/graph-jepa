from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


BASE_FIELDS = (
    "revenue",
    "operating_income",
    "net_income",
    "assets",
    "liabilities",
    "equity",
    "cash",
    "eps",
    "shares_outstanding",
    "dividend_per_share",
)

# Flows are earned over a period and only mean something once you say which
# period. Balances (assets, equity, cash, liabilities, shares) are stocks
# measured at an instant and need no basis conversion.
FLOW_FIELDS = ("revenue", "operating_income", "net_income", "eps")


def _normalize_ticker(value: object) -> str:
    text = str(value).strip().replace("A", "")
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def _as_number(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "N/A", "nan"}:
        return float("nan")
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def load_fundamental_observations(paths: Iterable[str | Path]) -> dict[str, pd.DataFrame]:
    """Load point-in-time fundamental observations from newline-delimited JSON.

    Every record needs `ticker`, `available_at`, and a `fields` mapping. Values
    become observable only on `available_at`; no filing-period date is used as
    an availability proxy.
    """

    rows: list[dict[str, object]] = []
    for item in paths:
        path = Path(item)
        if not path.exists():
            raise FileNotFoundError(f"fundamental observation file not found: {path}")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid fundamental JSON at {path}:{line_number}") from exc
            ticker = _normalize_ticker(record.get("ticker", ""))
            available_at = pd.to_datetime(record.get("available_at"), errors="coerce")
            fields = record.get("fields", {})
            if not ticker or pd.isna(available_at) or not isinstance(fields, Mapping):
                raise ValueError(f"invalid fundamental observation at {path}:{line_number}")
            # period_end says WHICH period a flow was earned over. It was parsed
            # out of the record and dropped, which is how 3-month and 12-month
            # figures ended up stacked in one column.
            row: dict[str, object] = {
                "ticker": ticker,
                "available_at": available_at.normalize(),
                "period_end": pd.to_datetime(record.get("period_end"), errors="coerce"),
            }
            for field in BASE_FIELDS:
                row[field] = _as_number(fields.get(field))
            rows.append(row)

    if not rows:
        return {}
    frame = pd.DataFrame(rows)
    result: dict[str, pd.DataFrame] = {}
    for ticker, ticker_frame in frame.groupby("ticker", sort=False):
        ordered = ticker_frame.drop(columns="ticker").sort_values("available_at")
        ordered = ordered.groupby("available_at", as_index=True).last()

        # RATIOS FIRST, from the values exactly as filed. A margin's numerator and
        # denominator come from the SAME filing and the same window, so it is
        # scale-free in the accounting period: a three-month margin and a
        # twelve-month margin are both the margin. These never carried the basis
        # bug, and deriving them from the trailing-year levels would inherit that
        # conversion's masking for nothing.
        ordered["operating_margin"] = ordered["operating_income"] / ordered["revenue"].replace(0.0, np.nan)
        ordered["net_margin"] = ordered["net_income"] / ordered["revenue"].replace(0.0, np.nan)
        ordered["equity_to_assets"] = ordered["equity"] / ordered["assets"].replace(0.0, np.nan)
        ordered["liabilities_to_equity"] = ordered["liabilities"] / ordered["equity"].replace(0.0, np.nan)

        # LEVELS carry the period and must be put on one basis. This is where the
        # 3-month/12-month stacking lived and where its correction costs coverage.
        ordered = _discrete_quarters(ordered)
        ordered["revenue_yoy"] = _year_over_year(ordered, "revenue")
        result[str(ticker)] = ordered
    return result


def _fiscal_key(ordered: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    period_end = pd.to_datetime(ordered["period_end"], errors="coerce")
    return period_end.dt.year, (period_end.dt.month - 1) // 3 + 1


def _discrete_quarters(ordered: pd.DataFrame) -> pd.DataFrame:
    """Put every flow item on ONE accounting basis: a discrete three months.

    THE BUG. DART 분기보고서 report a DISCRETE three months; the 사업보고서
    (period_end 12-31) reports the whole year. Both were loaded into the same
    column, so the level features jumped roughly fourfold every March and fell
    back in May. Verified on 005930, whose quarters run 53-86조 against annual
    filings of 237-334조, and across the release, whose December-quarter median
    revenue is 287bn against 68-73bn for the other three.

    THE FIX. Only the annual row needs converting, and it converts to its Q4
    residual: annual - (Q1 + Q2 + Q3) of the same fiscal year. The data supports
    it because the quarterly figures are discrete -- 005930's 2020 quarters sum
    to 175.3조 against a 236.8조 annual, leaving a 61.5조 Q4 that matches the
    filing. Q1 through Q3 are already three months and are left alone.

    WHY NOT TRAILING TWELVE MONTHS, which is the more standard level. It was
    tried first and rejected on evidence. A trailing year needs four consecutive
    quarters, so the release's first year has none -- it starts in 2020 -- and
    fold r3's fundamental coverage fell to 0.781 against the 0.790 the training
    guard requires. Backfilling 2019 from DART does not rescue it: 005930's 2019
    Q1 and Q2 filings carry no revenue field at all (operating and net income are
    present, revenue is None), so the window still cannot close on the
    best-covered company in the universe. Discrete quarters need no year of
    history and cost almost no coverage.

    THE COST, STATED. A quarterly level carries seasonality that a trailing year
    would smooth. That is real information rather than an artifact, and the
    universe shares a December fiscal year, so on any given date every stock
    shows the same quarter -- the seasonality is a common factor, not a
    cross-sectional distortion.

    A residual that cannot be computed becomes NaN and is masked downstream. An
    unavailable feature is honest; one that silently changes accounting basis is
    not.
    """

    year, quarter = _fiscal_key(ordered)
    annual = quarter.eq(4)
    out = ordered.copy()

    for field in FLOW_FIELDS:
        if field not in out.columns:
            continue
        reported = out[field].astype(float)
        converted = reported.to_numpy(dtype=float).copy()

        # Q1-Q3 as filed; index them so the annual row can subtract them.
        filed: dict[tuple[int, int], float] = {}
        for position in range(len(out)):
            if pd.isna(year.iloc[position]) or pd.isna(quarter.iloc[position]) or annual.iloc[position]:
                continue
            filed[(int(year.iloc[position]), int(quarter.iloc[position]))] = float(reported.iloc[position])

        for position in range(len(out)):
            if pd.isna(year.iloc[position]) or pd.isna(quarter.iloc[position]) or not annual.iloc[position]:
                continue
            fiscal_year = int(year.iloc[position])
            earlier = np.array(
                [filed.get((fiscal_year, q), np.nan) for q in (1, 2, 3)], dtype=float
            )
            converted[position] = (
                float(reported.iloc[position]) - float(earlier.sum())
                if np.isfinite(earlier).all()
                else np.nan
            )
        out[field] = converted
    return out


def _year_over_year(ordered: pd.DataFrame, field: str) -> pd.Series:
    """Growth against the SAME fiscal quarter one year earlier.

    The previous form was `value / value.shift(4)`, which is only the same
    quarter a year ago when a ticker files exactly four times a year. In this
    release 520 of 3,329 ticker-years file once, twice or three times, so
    shift(4) reached back to whatever row happened to sit four filings earlier.
    """

    year, quarter = _fiscal_key(ordered)
    values = ordered[field].astype(float)
    lookup = {
        (int(y), int(q)): float(v)
        for y, q, v in zip(year, quarter, values)
        if not (pd.isna(y) or pd.isna(q))
    }
    out = np.full(len(ordered), np.nan)
    for position in range(len(ordered)):
        if pd.isna(year.iloc[position]) or pd.isna(quarter.iloc[position]):
            continue
        prior = lookup.get((int(year.iloc[position]) - 1, int(quarter.iloc[position])), np.nan)
        current = values.iloc[position]
        if np.isfinite(prior) and np.isfinite(current) and abs(prior) > 0.0:
            out[position] = current / prior - 1.0
    return pd.Series(out, index=ordered.index)


def _asof_series(series: pd.Series, dates: pd.DatetimeIndex, lag_days: int) -> pd.Series:
    combined_index = series.index.union(dates)
    expanded = series.reindex(combined_index).sort_index().ffill().reindex(dates)
    return expanded.shift(max(0, int(lag_days)))


def build_fundamental_feature_frames(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    observations: Mapping[str, pd.DataFrame],
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """Turn point-in-time filings into causal per-stock state features."""

    feature_names = (
        "fund_revenue_log",
        "fund_operating_income_log",
        "fund_net_income_log",
        "fund_assets_log",
        "fund_equity_log",
        "fund_cash_log",
        "fund_eps_signed_log",
        "fund_shares_log",
        "fund_dividend_per_share",
        "fund_revenue_yoy",
        "fund_operating_margin",
        "fund_net_margin",
        "fund_equity_to_assets",
        "fund_liabilities_to_equity",
        "fund_report_age_days",
    )
    frames = {
        name: pd.DataFrame(np.nan, index=dates, columns=list(tickers), dtype=np.float32)
        for name in feature_names
    }
    transformations = {
        "fund_revenue_log": ("revenue", lambda value: np.log1p(np.maximum(value, 0.0))),
        "fund_operating_income_log": ("operating_income", lambda value: np.sign(value) * np.log1p(np.abs(value))),
        "fund_net_income_log": ("net_income", lambda value: np.sign(value) * np.log1p(np.abs(value))),
        "fund_assets_log": ("assets", lambda value: np.log1p(np.maximum(value, 0.0))),
        "fund_equity_log": ("equity", lambda value: np.sign(value) * np.log1p(np.abs(value))),
        "fund_cash_log": ("cash", lambda value: np.sign(value) * np.log1p(np.abs(value))),
        "fund_eps_signed_log": ("eps", lambda value: np.sign(value) * np.log1p(np.abs(value))),
        "fund_shares_log": ("shares_outstanding", lambda value: np.log1p(np.maximum(value, 0.0))),
        "fund_dividend_per_share": ("dividend_per_share", lambda value: value),
        "fund_revenue_yoy": ("revenue_yoy", lambda value: value),
        "fund_operating_margin": ("operating_margin", lambda value: value),
        "fund_net_margin": ("net_margin", lambda value: value),
        "fund_equity_to_assets": ("equity_to_assets", lambda value: value),
        "fund_liabilities_to_equity": ("liabilities_to_equity", lambda value: value),
    }
    for ticker in tickers:
        records = observations.get(str(ticker))
        if records is None or records.empty:
            continue
        for feature_name, (source_name, transform) in transformations.items():
            if source_name not in records:
                continue
            values = _asof_series(records[source_name], dates, lag_days)
            frames[feature_name][str(ticker)] = transform(values.to_numpy(dtype=np.float64)).astype(np.float32)

        available_dates = pd.Series(records.index, index=records.index)
        last_available = _asof_series(available_dates, dates, lag_days)
        age = (pd.Series(dates, index=dates) - last_available).dt.days
        frames["fund_report_age_days"][str(ticker)] = age.to_numpy(dtype=np.float32)
    # Do not add a feature that has no observed value anywhere in the panel.
    # Such a column carries no information and destabilizes normalization.
    return {
        name: frame
        for name, frame in frames.items()
        if np.isfinite(frame.to_numpy(dtype=np.float32)).any()
    }


def fundamental_coverage(
    feature_frames: Mapping[str, pd.DataFrame],
    eligible_mask: pd.DataFrame | None = None,
) -> float:
    if not feature_frames:
        return 0.0
    frames = list(feature_frames.values())
    values = np.stack(
        [frame.to_numpy(dtype=np.float32) for frame in frames], axis=-1
    )
    finite = np.isfinite(values)
    if eligible_mask is None:
        return float(finite.mean())
    eligible = eligible_mask.reindex(
        index=frames[0].index,
        columns=frames[0].columns,
    ).eq(True).to_numpy(dtype=bool)
    if not eligible.any():
        return 0.0
    return float(finite[eligible].mean())
