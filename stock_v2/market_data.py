from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
import json
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
import requests


DEFAULT_KRX_UNIVERSE: List[Tuple[str, str]] = [
    ("005930", "Samsung Electronics"),
    ("000660", "SK Hynix"),
    ("373220", "LG Energy Solution"),
    ("207940", "Samsung Biologics"),
    ("005380", "Hyundai Motor"),
    ("000270", "Kia"),
    ("068270", "Celltrion"),
    ("005490", "POSCO Holdings"),
    ("035420", "NAVER"),
    ("035720", "Kakao"),
    ("051910", "LG Chem"),
    ("006400", "Samsung SDI"),
    ("105560", "KB Financial"),
    ("055550", "Shinhan Financial"),
    ("086790", "Hana Financial"),
    ("316140", "Woori Financial"),
    ("012330", "Hyundai Mobis"),
    ("028260", "Samsung C&T"),
    ("066570", "LG Electronics"),
    ("003550", "LG"),
    ("034730", "SK"),
    ("096770", "SK Innovation"),
    ("009150", "Samsung Electro-Mechanics"),
    ("032830", "Samsung Life"),
    ("033780", "KT&G"),
    ("015760", "KEPCO"),
    ("010130", "Korea Zinc"),
    ("017670", "SK Telecom"),
    ("030200", "KT"),
    ("018260", "Samsung SDS"),
    ("011200", "HMM"),
    ("003670", "POSCO Future M"),
]


@dataclass
class OhlcvPanel:
    tickers: List[str]
    names: Dict[str, str]
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    price_observed: pd.DataFrame
    execution_close: pd.DataFrame


_NAVER_DAILY_URL = "https://fchart.stock.naver.com/sise.nhn?timeframe=day&count=6000&requestType=0&symbol="
_NAVER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120 Safari/537.36",
}


def _parse_naver_daily_chart(payload: str) -> pd.DataFrame:
    """Parse Naver's daily chart response without an unbounded HTTP request."""

    rows = re.findall(r'<item data="(.*?)"\s*/>', payload, re.DOTALL)
    if not rows:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume", "Change"])
    frame = pd.read_csv(StringIO("\n".join(rows)), delimiter="|", header=None, dtype={0: str})
    frame.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d", errors="coerce")
    frame = frame.dropna(subset=["Date"]).set_index("Date").sort_index()
    for column in ["Open", "High", "Low", "Close", "Volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["Change"] = frame["Close"].pct_change()
    return frame


def fetch_naver_ohlcv(
    ticker: str,
    start: str,
    end: str | None,
    *,
    timeout_sec: float = 20.0,
) -> pd.DataFrame:
    """Fetch one KRX ticker with a bounded Naver chart request."""

    response = requests.get(
        _NAVER_DAILY_URL + str(ticker),
        headers=_NAVER_HEADERS,
        timeout=max(float(timeout_sec), 0.1),
    )
    response.raise_for_status()
    frame = _parse_naver_daily_chart(response.text)
    if frame.empty:
        return frame
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end) if end else frame.index.max()
    return frame.loc[(frame.index >= start_at) & (frame.index <= end_at)].copy()


def _cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path:
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return cache_dir / f"{ticker}_{safe_start}_{safe_end}.csv"


def _covering_cache_path(cache_dir: Path, ticker: str, start: str, end: str) -> Path | None:
    """Return the freshest cached range that fully covers the request.

    Prefer the latest range end, then the closest range start. This makes a
    long canonical cache win over ad-hoc fold-specific downloads.
    """

    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    pattern = re.compile(rf"^{re.escape(ticker)}_(\d{{8}})_(\d{{8}})\.csv$")
    candidates: list[tuple[int, int, str, Path]] = []
    for candidate in cache_dir.glob(f"{ticker}_*.csv"):
        match = pattern.fullmatch(candidate.name)
        if match is None:
            continue
        try:
            cached_start = pd.to_datetime(match.group(1), format="%Y%m%d", errors="raise")
            cached_end = pd.to_datetime(match.group(2), format="%Y%m%d", errors="raise")
        except ValueError:
            continue
        if cached_start <= requested_start and cached_end >= requested_end:
            start_gap_days = int((requested_start - cached_start).days)
            candidates.append((-int(cached_end.value), start_gap_days, candidate.name, candidate))
    return min(candidates)[3] if candidates else None


def _immutable_cache_manifest(cache_dir: Path) -> Path | None:
    for candidate in (cache_dir / "manifest.json", cache_dir.parent / "manifest.json"):
        if not candidate.exists():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        method = payload.get("method") if isinstance(payload.get("method"), dict) else {}
        official_causal = (
            source.get("provider") == "kiwoom_rest_ka10081"
            and method.get("causal_invariance")
            and source.get("execution_price_basis") == "RawOHLC columns only"
        )
        contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
        lifecycle_hybrid = (
            payload.get("release_kind") == "krx500_lifecycle_hybrid"
            and contract.get("immutable") is True
            and contract.get("proxy_execution_rule")
            == "RawOHLC is null and execution is prohibited"
        )
        if official_causal or lifecycle_hybrid:
            return candidate
    return None


def _immutable_cache_min_rows(manifest_path: Path, requested_min_rows: int) -> int:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return requested_min_rows
    if payload.get("release_kind") != "krx500_lifecycle_hybrid":
        return requested_min_rows
    contract = payload.get("contract") if isinstance(payload.get("contract"), dict) else {}
    declared = max(1, int(contract.get("min_source_rows", 1) or 1))
    return min(requested_min_rows, declared)


def fetch_krx_ohlcv(
    universe: Sequence[Tuple[str, str]] = DEFAULT_KRX_UNIVERSE,
    start: str = "2020-01-01",
    end: str | None = None,
    cache_dir: Path | str = "data/cache",
    refresh: bool = False,
    min_rows: int = 180,
    request_timeout_sec: float = 20.0,
    request_retries: int = 2,
    request_retry_delay_sec: float = 0.5,
    cache_only: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Fetch Korean OHLCV data through bounded Naver requests with CSV caching."""

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    immutable_manifest = _immutable_cache_manifest(cache_root)
    if immutable_manifest is not None and refresh:
        raise ValueError(f"refusing to refresh immutable OHLCV cache: {immutable_manifest}")
    cache_only = bool(cache_only or immutable_manifest is not None)
    effective_min_rows = max(1, int(min_rows))
    if immutable_manifest is not None:
        effective_min_rows = _immutable_cache_min_rows(
            immutable_manifest,
            effective_min_rows,
        )
    end_value = end or pd.Timestamp.today().strftime("%Y-%m-%d")

    frames: Dict[str, pd.DataFrame] = {}
    for ticker, _name in universe:
        path = _cache_path(cache_root, ticker, start, end_value)
        cache_source = _covering_cache_path(cache_root, ticker, start, end_value)
        if cache_source is not None and not refresh:
            df = pd.read_csv(
                cache_source,
                parse_dates=["Date"],
                index_col="Date",
                float_precision="round_trip",
            )
            requested_start = pd.Timestamp(start)
            requested_end = pd.Timestamp(end_value)
            df = df.loc[(df.index >= requested_start) & (df.index <= requested_end)].copy()
        else:
            if cache_only:
                continue
            df = None
            for attempt in range(max(0, int(request_retries)) + 1):
                try:
                    df = fetch_naver_ohlcv(
                        ticker,
                        start,
                        end_value,
                        timeout_sec=request_timeout_sec,
                    )
                    break
                except (requests.RequestException, ValueError, pd.errors.ParserError):
                    if attempt >= max(0, int(request_retries)):
                        break
                    time.sleep(max(0.0, float(request_retry_delay_sec)) * (attempt + 1))
            if df is None or df.empty:
                continue
            df = df.copy()
            df.index.name = "Date"
            df.to_csv(path)

        required = ["Open", "High", "Low", "Close", "Volume"]
        if set(required).issubset(df.columns) and len(df) >= effective_min_rows:
            optional = [
                column
                for column in ("RawOpen", "RawHigh", "RawLow", "RawClose", "RawVolume")
                if column in df.columns
            ]
            frames[ticker] = df[required + optional].sort_index()

    return frames


def make_ohlcv_panel(
    data: Dict[str, pd.DataFrame],
    names: Dict[str, str] | None = None,
) -> OhlcvPanel:
    if not data:
        raise ValueError("no OHLCV data was loaded")

    cleaned: Dict[str, pd.DataFrame] = {}
    price_observed: Dict[str, pd.Series] = {}
    for ticker, frame in data.items():
        item = frame.copy()
        price_columns = ["Open", "High", "Low", "Close"]
        for column in price_columns:
            item[column] = pd.to_numeric(item[column], errors="coerce")
        if "RawClose" in item.columns:
            item["RawClose"] = pd.to_numeric(item["RawClose"], errors="coerce")
        if "Volume" in item.columns:
            item["Volume"] = pd.to_numeric(item["Volume"], errors="coerce").fillna(0.0)
            item.loc[item["Volume"] < 0, "Volume"] = 0.0

        # Two providers, two ways of spelling "this session did not trade", and
        # this panel is a hybrid of both.
        #
        #   FinanceDataReader writes a halted session as zero OHLC with a carried
        #   Close. The positivity test below catches it.
        #   Kiwoom writes it as a POSITIVE bar carried flat (O=H=L=C) with
        #   Volume=0. That passes the positivity test, so it had been entering the
        #   panel as a real, observed trading day: measured across the 500-ticker
        #   release, 12,667 such bars on 155 tickers, each handing an exact 0%
        #   return to the state losses, the downstream targets and the backtest,
        #   and each "tradable" at a carried open no one could have transacted at.
        #
        # No volume ALONE must not disqualify a bar. The 47 delisted tickers are
        # return-index proxies with volume disabled across their whole history;
        # dropping them would restore the survivorship bias this lifecycle panel
        # exists to remove. Their bars still move, so requiring BOTH no volume and
        # a flat bar keeps 27,864 of their sessions (60.4%) and removes only the
        # 89 on which they were genuinely halted. Conversely no ordinary ticker
        # has a moving zero-volume bar, so nothing real is lost on the other side.
        positive = item[price_columns].gt(0.0).all(axis=1)
        if "Volume" in item.columns:
            flat = (
                item["Open"].eq(item["High"])
                & item["High"].eq(item["Low"])
                & item["Low"].eq(item["Close"])
            )
            halted = flat & item["Volume"].le(0.0)
        else:
            halted = pd.Series(False, index=item.index)
        observed = positive & ~halted
        item.loc[~observed, price_columns] = pd.NA
        price_observed[ticker] = observed.astype(bool)
        cleaned[ticker] = item

    tickers = sorted(cleaned)
    name_map = names or {ticker: ticker for ticker in tickers}

    def collect(column: str) -> pd.DataFrame:
        frame = pd.concat(
            {ticker: cleaned[ticker][column].astype(float) for ticker in tickers},
            axis=1,
        ).sort_index()
        return frame

    open_ = collect("Open")
    high = collect("High")
    low = collect("Low")
    close = collect("Close")
    volume = collect("Volume")
    execution_close = pd.concat(
        {
            ticker: cleaned[ticker][
                "RawClose" if "RawClose" in cleaned[ticker].columns else "Close"
            ].astype(float)
            for ticker in tickers
        },
        axis=1,
    ).sort_index()
    observed = pd.concat(
        {ticker: price_observed[ticker] for ticker in tickers},
        axis=1,
    ).sort_index()

    common_index = close.index.sort_values().unique()
    open_ = open_.reindex(common_index).ffill()
    high = high.reindex(common_index).ffill()
    low = low.reindex(common_index).ffill()
    close = close.reindex(common_index).ffill()
    volume = volume.reindex(common_index).fillna(0.0)
    execution_close = execution_close.reindex(common_index).ffill()
    observed = observed.reindex(common_index).eq(True)

    valid_columns = close.columns[close.notna().any()].tolist()
    if len(valid_columns) < 4:
        raise ValueError("too few tickers have complete aligned data")

    return OhlcvPanel(
        tickers=valid_columns,
        names={ticker: name_map.get(ticker, ticker) for ticker in valid_columns},
        open=open_[valid_columns],
        high=high[valid_columns],
        low=low[valid_columns],
        close=close[valid_columns],
        volume=volume[valid_columns],
        price_observed=observed[valid_columns],
        execution_close=execution_close[valid_columns],
    )


def select_universe(max_tickers: int | None = None) -> List[Tuple[str, str]]:
    if max_tickers is None:
        return list(DEFAULT_KRX_UNIVERSE)
    return list(DEFAULT_KRX_UNIVERSE[:max_tickers])


def load_universe_manifest(path: Path | str) -> List[Tuple[str, str]]:
    """Load a frozen universe manifest for reproducible training runs."""

    manifest_path = Path(path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"universe manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = payload.get("universe", []) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty 'universe' list")
    universe: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            ticker = str(row.get("ticker", "")).strip().replace("A", "").zfill(6)
            name = str(row.get("name", ticker)).strip() or ticker
        elif isinstance(row, (list, tuple)) and len(row) == 2:
            ticker = str(row[0]).strip().replace("A", "").zfill(6)
            name = str(row[1]).strip() or ticker
        else:
            raise ValueError("universe manifest entries must be {ticker, name} records")
        if not ticker.isdigit() or len(ticker) != 6:
            raise ValueError(f"invalid ticker in universe manifest: {ticker!r}")
        if ticker in seen:
            raise ValueError(f"duplicate ticker in universe manifest: {ticker}")
        seen.add(ticker)
        universe.append((ticker, name))
    return universe


def select_krx_universe_from_listing(
    max_tickers: int = 200,
    markets: Sequence[str] = ("KOSPI", "KOSDAQ"),
    min_amount: float = 1_000_000_000.0,
) -> List[Tuple[str, str]]:
    """Select top KRX stocks from the current listing snapshot.

    This is useful for scale tests. It intentionally uses the current listing,
    so historical backtests with this universe still carry survivorship bias.
    """

    import FinanceDataReader as fdr

    listing = fdr.StockListing("KRX")
    frame = listing.copy()
    frame = frame[frame["Market"].isin(markets)]
    frame = frame[frame["MarketId"].isin(["STK", "KSQ"])]
    frame = frame[frame["Code"].astype(str).str.fullmatch(r"\d{6}")]
    frame = frame[pd.to_numeric(frame["Amount"], errors="coerce").fillna(0.0) >= min_amount]

    name = frame["Name"].astype(str)
    exclude_pattern = "스팩|리츠|우$|우B$|우선주|ETF|ETN|인버스|레버리지"
    frame = frame[~name.str.contains(exclude_pattern, regex=True)]
    frame = frame.sort_values("Marcap", ascending=False)

    selected = frame.head(max_tickers)
    return [(str(row.Code).zfill(6), str(row.Name)) for row in selected.itertuples()]
