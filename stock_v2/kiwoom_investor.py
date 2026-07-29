from __future__ import annotations

import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stock_v2.ops.brokers import KiwoomRestBroker


INVESTOR_COLUMNS = (
    "investor_traded_volume",
    "investor_individual_net_m",
    "investor_foreign_net_m",
    "investor_institution_net_m",
    "investor_financial_net_m",
    "investor_pension_net_m",
)

_KIWOOM_FIELD_MAP = {
    # ka10060 returns total traded volume in this field even in amount mode.
    # Investor net-flow fields remain KRW millions when amt_qty_tp="1".
    "acc_trde_prica": "investor_traded_volume",
    "ind_invsr": "investor_individual_net_m",
    "frgnr_invsr": "investor_foreign_net_m",
    "orgn": "investor_institution_net_m",
    "fnnc_invt": "investor_financial_net_m",
    "penfnd_etc": "investor_pension_net_m",
}


def parse_kiwoom_number(value: object) -> float:
    text = str(value if value is not None else "").strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return float("nan")
    if text.startswith("+"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return float("nan")


def parse_investor_chart_rows(rows: object) -> pd.DataFrame:
    """Normalize Kiwoom ka10060 rows into a dated, numeric market-data frame."""

    records: list[dict[str, object]] = []
    if not isinstance(rows, list):
        return pd.DataFrame(columns=INVESTOR_COLUMNS, dtype=np.float32)
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        date = pd.to_datetime(str(row.get("dt", "")), format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            continue
        record: dict[str, object] = {"date": pd.Timestamp(date).normalize()}
        for source_name, target_name in _KIWOOM_FIELD_MAP.items():
            record[target_name] = parse_kiwoom_number(row.get(source_name))
        records.append(record)
    if not records:
        return pd.DataFrame(columns=INVESTOR_COLUMNS, dtype=np.float32)
    frame = pd.DataFrame.from_records(records).set_index("date").sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    return frame.reindex(columns=INVESTOR_COLUMNS).astype(np.float32)


def fetch_investor_history(
    broker: KiwoomRestBroker,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    sleep_sec: float = 0.0,
    max_pages: int = 40,
) -> pd.DataFrame:
    """Fetch daily KRW-million net flows using Kiwoom ka10060 pagination."""

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    payload = {
        "dt": end_date.strftime("%Y%m%d"),
        "stk_cd": str(ticker).replace("A", "").strip(),
        "amt_qty_tp": "1",
        "trde_tp": "0",
        "unit_tp": "1000",
    }
    pages: list[pd.DataFrame] = []
    continuation = False
    next_key: str | None = None
    for _page in range(max(1, int(max_pages))):
        data, has_more, next_key = broker.post_readonly_with_continuation(
            "/api/dostk/chart",
            "ka10060",
            payload,
            continuation=continuation,
            next_key=next_key,
        )
        page = parse_investor_chart_rows(data.get("stk_invsr_orgn_chart"))
        if page.empty:
            break
        pages.append(page)
        if page.index.min() <= start_date or not has_more:
            break
        if not next_key:
            raise RuntimeError("Kiwoom ka10060 indicated continuation without a next-key")
        continuation = True
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    else:
        raise RuntimeError(f"Kiwoom ka10060 exceeded max_pages={max_pages} for {ticker}")

    if not pages:
        return pd.DataFrame(columns=INVESTOR_COLUMNS, dtype=np.float32)
    frame = pd.concat(pages).sort_index()
    frame = frame.loc[~frame.index.duplicated(keep="first")]
    return frame.loc[(frame.index >= start_date) & (frame.index <= end_date)].astype(np.float32)


def load_investor_flow_frames(
    cache_dir: str | Path,
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Load cached Kiwoom investor flows into date-by-ticker raw-value frames."""

    cache = Path(cache_dir)
    index = pd.DatetimeIndex(dates).normalize()
    ticker_list = [str(ticker).replace("A", "").zfill(6) for ticker in tickers]
    frames = {
        column: pd.DataFrame(np.nan, index=index, columns=ticker_list, dtype=np.float32)
        for column in INVESTOR_COLUMNS
    }
    if not cache.exists():
        return frames
    for ticker in ticker_list:
        parts: list[pd.DataFrame] = []
        for path in sorted(cache.glob(f"{ticker}_*.csv")):
            try:
                raw = pd.read_csv(path, index_col="date", parse_dates=["date"])
            except Exception:
                continue
            if raw.empty:
                continue
            if "investor_traded_volume" not in raw and "investor_traded_value_m" in raw:
                # Compatibility for caches written before the ka10060 field was
                # empirically reconciled against OHLCV volume.
                raw = raw.rename(columns={"investor_traded_value_m": "investor_traded_volume"})
            raw.index = pd.DatetimeIndex(raw.index).normalize()
            parts.append(raw)
        if not parts:
            continue
        combined = pd.concat(parts).sort_index()
        combined = combined.loc[~combined.index.duplicated(keep="last")]
        for column, target in frames.items():
            if column in combined:
                target.loc[:, ticker] = pd.to_numeric(combined[column], errors="coerce").reindex(index).to_numpy(dtype=np.float32)
    return frames


def build_investor_feature_frames(
    investor_flow_frames: Mapping[str, pd.DataFrame],
    traded_value: pd.DataFrame,
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """Turn net investor flows into lagged, turnover-normalized stock sensors."""

    if lag_days < 0:
        raise ValueError("lag_days must be nonnegative")
    index = traded_value.index
    columns = traded_value.columns
    denominator = (traded_value / 1_000_000.0).replace(0.0, np.nan).shift(lag_days)

    def flow(name: str) -> pd.DataFrame:
        source = investor_flow_frames.get(name)
        if source is None:
            return pd.DataFrame(np.nan, index=index, columns=columns, dtype=np.float32)
        return source.reindex(index=index, columns=columns).shift(lag_days)

    def ratio(numerator: pd.DataFrame, denominator_frame: pd.DataFrame) -> pd.DataFrame:
        values = numerator / denominator_frame.replace(0.0, np.nan)
        return values.replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0).astype(np.float32)

    foreign = flow("investor_foreign_net_m")
    institution = flow("investor_institution_net_m")
    individual = flow("investor_individual_net_m")
    pension = flow("investor_pension_net_m")
    features = {
        "investor_foreign_flow_ratio_1d": ratio(foreign, denominator),
        "investor_institution_flow_ratio_1d": ratio(institution, denominator),
        "investor_individual_flow_ratio_1d": ratio(individual, denominator),
        "investor_pension_flow_ratio_1d": ratio(pension, denominator),
    }
    for window in (5, 20):
        rolling_denominator = denominator.rolling(window, min_periods=window).sum()
        features[f"investor_foreign_flow_ratio_{window}d"] = ratio(
            foreign.rolling(window, min_periods=window).sum(),
            rolling_denominator,
        )
        features[f"investor_institution_flow_ratio_{window}d"] = ratio(
            institution.rolling(window, min_periods=window).sum(),
            rolling_denominator,
        )
    return features


def investor_feature_coverage(
    feature_frames: Mapping[str, pd.DataFrame],
    eligible_mask: pd.DataFrame | None = None,
) -> float:
    if not feature_frames:
        return 0.0
    frames = list(feature_frames.values())
    values = np.stack(
        [frame.to_numpy(dtype=np.float32) for frame in frames], axis=-1
    )
    available = np.isfinite(values).any(axis=-1)
    if eligible_mask is None:
        return float(available.mean())
    eligible = eligible_mask.reindex(
        index=frames[0].index,
        columns=frames[0].columns,
    ).eq(True).to_numpy(dtype=bool)
    if not eligible.any():
        return 0.0
    return float(available[eligible].mean())
