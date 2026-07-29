from __future__ import annotations

import os
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Protocol

import requests

from stock_v2.ops.config import KiwoomConfig
from stock_v2.ops.store import OpsStore
from stock_v2.ops.types import Account, OrderIntent, OrderResult, Position, Quote


class Broker(Protocol):
    def get_account(self, prices: dict[str, float] | None = None) -> Account: ...
    def place_order(self, intent: OrderIntent) -> OrderResult: ...


class PaperBroker:
    def __init__(
        self,
        store: OpsStore,
        initial_cash: float,
        commission_bps: float = 0.0,
        sell_tax_bps: float = 0.0,
    ):
        self.store = store
        self.commission_bps = float(commission_bps)
        self.sell_tax_bps = float(sell_tax_bps)
        if self.commission_bps < 0.0 or self.sell_tax_bps < 0.0:
            raise ValueError("paper transaction cost rates must be non-negative")
        self.store.init_cash(initial_cash, reset=False)

    @staticmethod
    def _cost(notional: int, bps: float) -> int:
        return int(math.ceil(float(notional) * float(bps) / 10_000.0))

    def get_account(self, prices: dict[str, float] | None = None) -> Account:
        return Account(cash=self.store.get_cash(), positions=self.store.get_positions(prices))

    def place_order(self, intent: OrderIntent) -> OrderResult:
        if intent.quantity <= 0 or intent.price <= 0:
            return OrderResult(intent, "REJECTED", "invalid quantity or price")

        account = self.get_account({intent.ticker: intent.price})
        current = {position.ticker: position for position in account.positions}.get(intent.ticker)

        if intent.side == "BUY":
            commission = self._cost(intent.notional, self.commission_bps)
            cost = intent.notional + commission
            if account.cash < cost:
                return OrderResult(intent, "REJECTED", f"paper cash insufficient: cash={account.cash:.0f}, cost={cost}")
            if current:
                new_qty = current.quantity + intent.quantity
                new_avg = (current.avg_price * current.quantity + cost) / new_qty
            else:
                new_qty = intent.quantity
                new_avg = cost / new_qty
            self.store.set_cash(account.cash - cost)
            self.store.upsert_position(intent.ticker, intent.name, new_qty, new_avg)
            return OrderResult(
                intent,
                "FILLED",
                "paper buy filled",
                raw={"fill_price": intent.price, "commission": commission, "sell_tax": 0},
            )

        if current is None or current.quantity < intent.quantity:
            return OrderResult(intent, "REJECTED", "paper position insufficient")
        commission = self._cost(intent.notional, self.commission_bps)
        sell_tax = self._cost(intent.notional, self.sell_tax_bps)
        revenue = intent.notional - commission - sell_tax
        remaining = current.quantity - intent.quantity
        self.store.set_cash(account.cash + revenue)
        self.store.upsert_position(intent.ticker, intent.name, remaining, current.avg_price)
        return OrderResult(
            intent,
            "FILLED",
            "paper sell filled",
            raw={"fill_price": intent.price, "commission": commission, "sell_tax": sell_tax},
        )


def build_kiwoom_order_payload(intent: OrderIntent, exchange: str) -> tuple[str, dict[str, str]]:
    if intent.quantity <= 0 or intent.price <= 0:
        raise ValueError("Kiwoom order requires positive quantity and price")
    if intent.side not in {"BUY", "SELL"}:
        raise ValueError(f"unsupported order side: {intent.side}")
    api_id = "kt10000" if intent.side == "BUY" else "kt10001"
    return api_id, {
        "dmst_stex_tp": str(exchange),
        "stk_cd": intent.ticker,
        "ord_qty": str(intent.quantity),
        "ord_uv": str(intent.price),
        "trde_tp": "0",
        "cond_uv": "",
    }


def _load_env_file(path: str | Path) -> None:
    env_path = Path(path).expanduser()
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _parse_number(value: object, *, abs_value: bool = False) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return abs(number) if abs_value else number
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--", "0.00-"}:
        return None
    if text.startswith("+"):
        text = text[1:]
    try:
        number = float(text)
    except ValueError:
        return None
    return abs(number) if abs_value else number


def _first_number(payload: dict[str, object], keys: list[str], *, abs_value: bool = False) -> float | None:
    for key in keys:
        number = _parse_number(payload.get(key), abs_value=abs_value)
        if number is not None:
            return number
    return None


def _first_text(payload: dict[str, object], keys: list[str]) -> str | None:
    for key in keys:
        value = payload.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def parse_kiwoom_quote(
    ticker: str,
    basic: dict[str, object] | None = None,
    orderbook: dict[str, object] | None = None,
    errors: list[str] | None = None,
) -> Quote | None:
    """Parse Kiwoom REST ka10001/ka10004 responses into a stable quote shape."""
    basic = basic or {}
    orderbook = orderbook or {}
    last_price = _first_number(
        basic,
        ["cur_prc", "prpr", "stck_prpr", "last_price", "현재가"],
        abs_value=True,
    )
    bid_price = _first_number(orderbook, ["buy_fpr_bid", "bid_pric", "bid_price"], abs_value=True)
    ask_price = _first_number(orderbook, ["sel_fpr_bid", "ask_pric", "ask_price"], abs_value=True)

    if last_price is None and bid_price and ask_price:
        last_price = (bid_price + ask_price) / 2.0
    if last_price is None and (bid_price or ask_price):
        last_price = bid_price or ask_price
    if last_price is None:
        return None

    volume = _first_number(basic, ["trde_qty", "acml_vol", "volume", "거래량"], abs_value=True)
    return Quote(
        ticker=ticker,
        last_price=last_price,
        bid_price=bid_price,
        ask_price=ask_price,
        open_price=_first_number(basic, ["open_pric", "stck_oprc", "open_price"], abs_value=True),
        high_price=_first_number(basic, ["high_pric", "stck_hgpr", "high_price"], abs_value=True),
        low_price=_first_number(basic, ["low_pric", "stck_lwpr", "low_price"], abs_value=True),
        volume=None if volume is None else int(volume),
        exchange_time=_first_text(orderbook, ["bid_req_base_tm", "base_tm", "stck_cntg_hour"]),
        received_at=datetime.now().isoformat(timespec="seconds"),
        source=(
            "kiwoom:ka10001+ka10004"
            if orderbook
            else "kiwoom:ka10001"
        ),
        raw={"basic": basic, "orderbook": orderbook, "errors": errors or []},
    )


class KiwoomRestBroker:
    """Minimal Kiwoom REST broker.

    Live submission is additionally guarded by env vars:
    STOCK_V2_LIVE_TRADING_ENABLED=YES
    STOCK_V2_ACKNOWLEDGE_REAL_MONEY_RISK=YES
    STOCK_V2_LIVE_ACCOUNT_LAST4=<last 4 digits of account number>
    """

    def __init__(self, config: KiwoomConfig, dry_run: bool = True):
        _load_env_file(config.env_file)
        self.config = config
        self.dry_run = dry_run
        self.app_key = os.environ.get("KIWOOM_APP_KEY", "")
        self.app_secret = os.environ.get("KIWOOM_APP_SECRET", "")
        self.account_number = os.environ.get("KIWOOM_ACCOUNT_NUMBER", "")
        self.base_url = "https://mockapi.kiwoom.com" if config.server == "mock" else "https://api.kiwoom.com"
        self.access_token: str | None = None
        self.last_auth_error: str | None = None

    def _assert_live_enabled(self) -> None:
        if self.dry_run:
            return
        checks = {
            "STOCK_V2_LIVE_TRADING_ENABLED": "YES",
            "STOCK_V2_ACKNOWLEDGE_REAL_MONEY_RISK": "YES",
            "STOCK_V2_LIVE_ACCOUNT_LAST4": self.account_number[-4:],
        }
        for key, expected in checks.items():
            actual = os.environ.get(key)
            if actual != expected:
                raise RuntimeError(f"live order blocked: {key} must be {expected!r}, got {actual!r}")

    def authenticate(self) -> bool:
        self.access_token = None
        self.last_auth_error = None
        if not self.app_key or not self.app_secret:
            self.last_auth_error = "missing KIWOOM_APP_KEY or KIWOOM_APP_SECRET"
            return False
        try:
            response = requests.post(
                f"{self.base_url}/oauth2/token",
                headers={"content-type": "application/json;charset=UTF-8"},
                json={"grant_type": "client_credentials", "appkey": self.app_key, "secretkey": self.app_secret},
                timeout=self.config.timeout_sec,
            )
        except requests.RequestException as exc:
            # Exception text can include the submitted request body, so retain only its type.
            self.last_auth_error = f"request_error={type(exc).__name__}"
            return False
        if response.status_code != 200:
            parts = [f"status={response.status_code}"]
            try:
                payload = response.json()
            except (TypeError, ValueError):
                payload = None
            if isinstance(payload, dict):
                code = payload.get("return_code")
                if code is not None:
                    parts.append(f"code={code}")
                message = payload.get("return_msg", payload.get("message"))
                if message is not None:
                    safe_message = " ".join(str(message).split())
                    for secret in (self.app_key, self.app_secret):
                        if secret:
                            safe_message = safe_message.replace(secret, "<redacted>")
                    if safe_message:
                        parts.append(f"message={safe_message[:200]}")
            self.last_auth_error = " ".join(parts)
            return False
        try:
            payload = response.json()
        except (TypeError, ValueError):
            self.last_auth_error = "status=200 invalid_json"
            return False
        if not isinstance(payload, dict):
            self.last_auth_error = "status=200 non_object_json"
            return False
        self.access_token = payload.get("token")
        if not self.access_token:
            parts = ["status=200", "token_missing"]
            code = payload.get("return_code")
            if code is not None:
                parts.append(f"code={code}")
            message = payload.get("return_msg", payload.get("message"))
            if message is not None:
                safe_message = " ".join(str(message).split())
                for secret in (self.app_key, self.app_secret):
                    if secret:
                        safe_message = safe_message.replace(secret, "<redacted>")
                if safe_message:
                    parts.append(f"message={safe_message[:200]}")
            self.last_auth_error = " ".join(parts)
            return False
        return True

    def _headers(self, api_id: str) -> dict[str, str]:
        if not self.access_token and not self.authenticate():
            detail = f": {self.last_auth_error}" if self.last_auth_error else ""
            raise RuntimeError(f"Kiwoom authentication failed{detail}")
        return {
            "content-type": "application/json;charset=UTF-8",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "api-id": api_id,
        }

    def _post(self, path: str, api_id: str, payload: dict[str, object]) -> dict[str, object]:
        response = requests.post(
            f"{self.base_url}{path}",
            headers=self._headers(api_id),
            json=payload,
            timeout=self.config.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Kiwoom {api_id} failed: status={response.status_code} body={response.text[:300]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Kiwoom {api_id} returned non-object JSON")
        return data

    def post_readonly_with_continuation(
        self,
        path: str,
        api_id: str,
        payload: dict[str, object],
        *,
        continuation: bool = False,
        next_key: str | None = None,
    ) -> tuple[dict[str, object], bool, str | None]:
        """Issue a read-only Kiwoom request and return its pagination cursor."""

        headers = self._headers(api_id)
        if continuation:
            headers["cont-yn"] = "Y"
            if next_key:
                headers["next-key"] = next_key
        response = requests.post(
            f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            timeout=self.config.timeout_sec,
        )
        if response.status_code != 200:
            raise RuntimeError(f"Kiwoom {api_id} failed: status={response.status_code} body={response.text[:300]}")
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Kiwoom {api_id} returned non-object JSON")
        return_code = data.get("return_code")
        if return_code not in (None, 0, "0"):
            raise RuntimeError(f"Kiwoom {api_id} rejected request: {data.get('return_msg', return_code)}")
        has_more = str(response.headers.get("cont-yn", "")).upper() == "Y"
        cursor = str(response.headers.get("next-key", "")).strip() or None
        return data, has_more, cursor

    def get_quote(
        self,
        ticker: str,
        *,
        include_orderbook: bool = True,
    ) -> Quote | None:
        ticker = ticker.replace("A", "").strip()
        errors: list[str] = []
        basic: dict[str, object] = {}
        orderbook: dict[str, object] = {}
        try:
            basic = self._post("/api/dostk/stkinfo", "ka10001", {"stk_cd": ticker})
        except Exception as exc:
            errors.append(f"ka10001:{exc}")
        if include_orderbook:
            try:
                orderbook = self._post(
                    "/api/dostk/mrkcond",
                    "ka10004",
                    {"stk_cd": ticker},
                )
            except Exception as exc:
                errors.append(f"ka10004:{exc}")
        quote = parse_kiwoom_quote(ticker, basic=basic, orderbook=orderbook, errors=errors)
        if quote is None and errors and not self.dry_run:
            raise RuntimeError("; ".join(errors))
        return quote

    def get_quotes(
        self,
        tickers: list[str],
        sleep_sec: float = 0.15,
        *,
        include_orderbook: bool = True,
    ) -> dict[str, Quote]:
        quotes: dict[str, Quote] = {}
        seen: set[str] = set()
        for ticker in tickers:
            normalized = ticker.replace("A", "").strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            quote = self.get_quote(
                normalized,
                include_orderbook=include_orderbook,
            )
            if quote is not None:
                quotes[normalized] = quote
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        return quotes

    def get_account(self, prices: dict[str, float] | None = None) -> Account:
        cash = 0.0
        positions: list[Position] = []
        try:
            balance = requests.post(
                f"{self.base_url}/api/dostk/acnt",
                headers=self._headers("kt00001"),
                json={"account_no": self.account_number, "password": "", "password_type": "00", "qry_tp": "0"},
                timeout=self.config.timeout_sec,
            )
            if balance.status_code == 200:
                value = str(balance.json().get("ord_alow_amt", "0")).replace(",", "")
                cash = float(value) if value.replace(".", "", 1).isdigit() else 0.0

            holdings = requests.post(
                f"{self.base_url}/api/dostk/acnt",
                headers=self._headers("kt00017"),
                json={"account_no": self.account_number, "password": "", "password_type": "00", "qry_tp": "0"},
                timeout=self.config.timeout_sec,
            )
            if holdings.status_code == 200:
                for item in holdings.json().get("output1", []):
                    ticker = str(item.get("item_cd", "")).replace("A", "")
                    quantity = int(float(str(item.get("hold_qty", "0")).replace(",", "")))
                    if quantity <= 0:
                        continue
                    avg = float(str(item.get("pchs_avg_pric", "0")).replace(",", "") or 0)
                    current = float(str(item.get("prpr", avg)).replace(",", "") or avg)
                    positions.append(
                        Position(
                            ticker=ticker,
                            name=str(item.get("item_nm", ticker)),
                            quantity=quantity,
                            avg_price=avg,
                            current_price=current,
                        )
                    )
            settled = requests.post(
                f"{self.base_url}/api/dostk/acnt",
                headers=self._headers("kt00005"),
                json={"qry_tp": "1", "dmst_stex_tp": self.config.exchange},
                timeout=self.config.timeout_sec,
            )
            if settled.status_code == 200:
                by_ticker = {position.ticker: position for position in positions}
                for item in settled.json().get("stk_cntr_remn", []):
                    ticker = str(item.get("stk_cd", "")).replace("A", "")
                    quantity = int(float(str(item.get("cur_qty", "0")).replace(",", "") or 0))
                    if quantity <= 0:
                        continue
                    avg = float(str(item.get("buy_uv", "0")).replace(",", "") or 0)
                    current = float(str(item.get("cur_prc", avg)).replace(",", "") or avg)
                    by_ticker[ticker] = Position(
                        ticker=ticker,
                        name=str(item.get("stk_nm", ticker)),
                        quantity=quantity,
                        avg_price=avg,
                        current_price=current,
                    )
                positions = list(by_ticker.values())
        except Exception:
            if not self.dry_run:
                raise
        return Account(cash=cash, positions=positions)

    def place_order(self, intent: OrderIntent) -> OrderResult:
        try:
            self._assert_live_enabled()
        except Exception as exc:
            return OrderResult(intent, "REJECTED", str(exc))

        try:
            api_id, payload = build_kiwoom_order_payload(intent, self.config.exchange)
        except Exception as exc:
            return OrderResult(intent, "REJECTED", str(exc))
        if self.dry_run:
            return OrderResult(
                intent,
                "DRY_RUN",
                "kiwoom live order dry-run only",
                raw={"api_id": api_id, "payload": payload},
            )
        try:
            response = requests.post(
                f"{self.base_url}/api/dostk/ordr",
                headers=self._headers(api_id),
                json=payload,
                timeout=self.config.timeout_sec,
            )
            try:
                body = response.json()
            except Exception:
                body = {}
            raw = {"status_code": response.status_code, "text": response.text[:1000], "json": body}
            if response.status_code == 200:
                return_code = body.get("return_code")
                if str(return_code) not in {"0", ""}:
                    message = str(body.get("return_msg") or "kiwoom order rejected by API")
                    return OrderResult(intent, "ERROR", message, raw=raw)
                return OrderResult(
                    intent,
                    "ACCEPTED",
                    "kiwoom order accepted",
                    broker_order_id=str(body.get("ord_no") or "") or None,
                    raw=raw,
                )
            return OrderResult(intent, "ERROR", "kiwoom order rejected by API", raw=raw)
        except Exception as exc:
            return OrderResult(intent, "ERROR", str(exc))
