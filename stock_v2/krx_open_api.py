from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import pandas as pd
import requests


KRX_API_BASE = "https://data-dbg.krx.co.kr/svc/apis/sto"
MARKET_API_IDS = {"KOSPI": "stk_bydd_trd", "KOSDAQ": "ksq_bydd_trd"}
REQUIRED_FIELDS = (
    "BAS_DD",
    "ISU_CD",
    "ISU_NM",
    "MKT_NM",
    "TDD_CLSPRC",
    "TDD_OPNPRC",
    "TDD_HGPRC",
    "TDD_LWPRC",
    "ACC_TRDVOL",
    "ACC_TRDVAL",
    "MKTCAP",
    "LIST_SHRS",
)


class KrxOpenApiError(RuntimeError):
    pass


def _integer(value: Any) -> int:
    text = str(value or "").strip().replace(",", "")
    if not text:
        return 0
    return int(text)


def parse_daily_rows(payload: Mapping[str, Any], expected_date: str, expected_market: str) -> list[dict[str, Any]]:
    raw_rows = payload.get("OutBlock_1")
    if not isinstance(raw_rows, list):
        raise KrxOpenApiError("KRX payload has no OutBlock_1 array")
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, Mapping) or not set(REQUIRED_FIELDS).issubset(raw):
            raise KrxOpenApiError("KRX daily row violates the documented schema")
        if str(raw["BAS_DD"]) != expected_date:
            raise KrxOpenApiError("KRX daily row date does not match the request")
        ticker = str(raw["ISU_CD"]).strip()
        market = str(raw["MKT_NM"]).strip().upper()
        if not ticker.isdigit() or len(ticker) != 6:
            continue
        if market != expected_market:
            raise KrxOpenApiError(f"KRX market mismatch: expected={expected_market} actual={market}")
        rows.append(
            {
                "Date": pd.to_datetime(expected_date, format="%Y%m%d").strftime("%Y-%m-%d"),
                "Ticker": ticker,
                "Name": str(raw["ISU_NM"]).strip(),
                "Market": market,
                "Open": _integer(raw["TDD_OPNPRC"]),
                "High": _integer(raw["TDD_HGPRC"]),
                "Low": _integer(raw["TDD_LWPRC"]),
                "Close": _integer(raw["TDD_CLSPRC"]),
                "Volume": _integer(raw["ACC_TRDVOL"]),
                "Amount": _integer(raw["ACC_TRDVAL"]),
                "MarCap": _integer(raw["MKTCAP"]),
                "ListShares": _integer(raw["LIST_SHRS"]),
            }
        )
    return rows


@dataclass(frozen=True)
class KrxRawResponse:
    market: str
    api_id: str
    date: str
    body: bytes
    rows: int

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()


class KrxOpenApiClient:
    def __init__(
        self,
        auth_key: str,
        timeout_sec: float = 30.0,
        sleep_sec: float = 0.15,
        retries: int = 3,
    ) -> None:
        if not auth_key:
            raise KrxOpenApiError("KRX_OPEN_API_KEY is required")
        self.auth_key = auth_key
        self.timeout_sec = timeout_sec
        self.sleep_sec = max(0.0, sleep_sec)
        self.retries = max(0, retries)
        self.session = requests.Session()

    def fetch_daily(self, market: str, date: str | pd.Timestamp) -> KrxRawResponse:
        normalized_market = str(market).upper()
        if normalized_market not in MARKET_API_IDS:
            raise ValueError(f"unsupported KRX market: {market}")
        bas_dd = pd.Timestamp(date).strftime("%Y%m%d")
        api_id = MARKET_API_IDS[normalized_market]
        error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self.session.get(
                    f"{KRX_API_BASE}/{api_id}",
                    params={"basDd": bas_dd},
                    headers={"AUTH_KEY": self.auth_key, "Accept": "application/json"},
                    timeout=self.timeout_sec,
                )
                response.raise_for_status()
                payload = response.json()
                rows = parse_daily_rows(payload, bas_dd, normalized_market)
                if self.sleep_sec:
                    time.sleep(self.sleep_sec)
                return KrxRawResponse(
                    market=normalized_market,
                    api_id=api_id,
                    date=bas_dd,
                    body=response.content,
                    rows=len(rows),
                )
            except (requests.RequestException, ValueError, KrxOpenApiError) as exc:
                error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2**attempt))
        raise KrxOpenApiError(f"KRX request failed for {normalized_market} {bas_dd}: {error}")


def raw_response_path(raw_dir: Path, response: KrxRawResponse) -> Path:
    return raw_dir / response.api_id / response.date[:4] / response.date[4:6] / f"{response.date}.json"


def persist_raw_response(raw_dir: Path, response: KrxRawResponse) -> tuple[Path, Path]:
    path = raw_response_path(raw_dir, response)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_bytes(response.body)
    temporary.replace(path)
    metadata_path = path.with_suffix(".meta.json")
    metadata = {
        "schema_version": 1,
        "provider": "Korea Exchange KRX Open API",
        "api_id": response.api_id,
        "market": response.market,
        "date": response.date,
        "rows": response.rows,
        "response_sha256": response.sha256,
        "fetched_at_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path, metadata_path


def load_and_validate_raw(path: Path, market: str, date: str) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return parse_daily_rows(payload, date, market)


def build_ticker_frames(
    raw_dir: Path,
    sessions: Sequence[pd.Timestamp],
    universe_tickers: set[str],
) -> dict[str, pd.DataFrame]:
    rows_by_ticker: dict[str, list[dict[str, Any]]] = {ticker: [] for ticker in universe_tickers}
    for session in sessions:
        date = pd.Timestamp(session).strftime("%Y%m%d")
        for market, api_id in MARKET_API_IDS.items():
            path = raw_dir / api_id / date[:4] / date[4:6] / f"{date}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing frozen KRX response: {path}")
            for row in load_and_validate_raw(path, market, date):
                ticker = str(row["Ticker"])
                if ticker in rows_by_ticker:
                    rows_by_ticker[ticker].append(row)
    return {
        ticker: pd.DataFrame(rows).sort_values("Date").reset_index(drop=True)
        if rows
        else pd.DataFrame(
            columns=[
                "Date",
                "Ticker",
                "Name",
                "Market",
                "Open",
                "High",
                "Low",
                "Close",
                "Volume",
                "Amount",
                "MarCap",
                "ListShares",
            ]
        )
        for ticker, rows in rows_by_ticker.items()
    }
