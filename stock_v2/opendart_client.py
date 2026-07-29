from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from datetime import datetime, timezone
import calendar
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable, Mapping
import xml.etree.ElementTree as ET
import zipfile

import requests


BASE_URL = "https://opendart.fss.or.kr/api"


class OpenDartError(RuntimeError):
    pass


@dataclass(frozen=True)
class DartReport:
    business_year: int
    report_code: str
    available_at: str
    period_end: str
    report_name: str = ""
    receipt_no: str = ""
    fiscal_year_end_month: int = 12


def _number(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text or text in {"-", "--"}:
        return None
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except ValueError:
        return None


def _report_descriptor(report_name: str, received_at: str) -> tuple[int, str, str] | None:
    name = str(report_name)
    period = re.search(r"(20\d{2})\.(0[3-9]|1[0-2])", name)
    if period is None:
        return None
    year = int(period.group(1))
    month = int(period.group(2))
    if "\uc0ac\uc5c5\ubcf4\uace0\uc11c" in name:
        report_code = "11011"
    elif "\ubc18\uae30\ubcf4\uace0\uc11c" in name:
        report_code = "11012"
    elif "\ubd84\uae30\ubcf4\uace0\uc11c" in name and month == 3:
        report_code = "11013"
    elif "\ubd84\uae30\ubcf4\uace0\uc11c" in name and month == 9:
        report_code = "11014"
    else:
        return None
    try:
        available_at = date.fromisoformat(f"{received_at[:4]}-{received_at[4:6]}-{received_at[6:8]}")
    except (TypeError, ValueError):
        return None
    return year, report_code, available_at.isoformat()


def _report_listing(report_name: str, received_at: str) -> tuple[int, int, str, str] | None:
    name = str(report_name)
    period = re.search(r"(20\d{2})\.(0[1-9]|1[0-2])", name)
    if period is None:
        return None
    if "사업보고서" in name:
        kind = "annual"
    elif "반기보고서" in name:
        kind = "half"
    elif "분기보고서" in name:
        kind = "quarter"
    else:
        return None
    try:
        available_at = date.fromisoformat(f"{received_at[:4]}-{received_at[4:6]}-{received_at[6:8]}")
    except (TypeError, ValueError):
        return None
    return int(period.group(1)), int(period.group(2)), kind, available_at.isoformat()


def _period_end(year: int, report_code: str) -> str:
    month_by_code = {"11011": 12, "11012": 6, "11013": 3, "11014": 9}
    month = month_by_code[report_code]
    day = 31 if month in {3, 12} else 30
    return f"{year:04d}-{month:02d}-{day:02d}"


def _actual_period_end(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"


def _statement_period_verified(rows: Iterable[dict[str, Any]], expected_period_end: str) -> bool | None:
    expected = expected_period_end.replace("-", ".")
    saw_statement_date = False
    for row in rows:
        value = str(row.get("thstrm_dt") or "")
        saw_statement_date = saw_statement_date or bool(value.strip())
        if expected in value:
            return True
    return False if saw_statement_date else None


def _matches(row: dict[str, Any], account_ids: set[str], account_names: set[str]) -> bool:
    account_id = str(row.get("account_id", ""))
    account_name = str(row.get("account_nm", ""))
    return account_id in account_ids or account_name in account_names


def extract_canonical_fields(rows: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Extract a stable set of fields, preferring consolidated statements."""

    aliases = {
        "revenue": (
            {"ifrs-full_Revenue", "ifrs-full_SalesRevenue", "dart_Revenue"},
            {"\ub9e4\ucd9c\uc561", "\uc601\uc5c5\uc218\uc775"},
        ),
        "operating_income": (
            {"dart_OperatingIncomeLoss", "ifrs-full_OperatingProfitLoss"},
            {"\uc601\uc5c5\uc774\uc775", "\uc601\uc5c5\uc190\uc2e4"},
        ),
        "net_income": (
            {"ifrs-full_ProfitLoss", "ifrs-full_ProfitLossAttributableToOwnersOfParent"},
            {"\ub2f9\uae30\uc21c\uc774\uc775", "\ub2f9\uae30\uc21c\uc190\uc2e4"},
        ),
        "assets": ({"ifrs-full_Assets"}, {"\uc790\uc0b0\ucd1d\uacc4"}),
        "liabilities": ({"ifrs-full_Liabilities"}, {"\ubd80\ucc44\ucd1d\uacc4"}),
        "equity": ({"ifrs-full_Equity"}, {"\uc790\ubcf8\ucd1d\uacc4"}),
        "cash": ({"ifrs-full_CashAndCashEquivalents"}, {"\ud604\uae08\ubc0f\ud604\uae08\uc131\uc790\uc0b0"}),
        "eps": (
            {"dart_BasicEarningsLossPerShare", "ifrs-full_BasicEarningsLossPerShare"},
            {"\uae30\ubcf8\uc8fc\ub2f9\uc774\uc775", "\uae30\ubcf8\uc8fc\ub2f9\uc21c\uc190\uc2e4"},
        ),
    }
    result: dict[str, float] = {}
    for field, (account_ids, account_names) in aliases.items():
        candidates = [row for row in rows if _matches(row, account_ids, account_names)]
        candidates.sort(key=lambda row: 0 if row.get("fs_div") == "CFS" else 1)
        for row in candidates:
            value = _number(row.get("thstrm_amount"))
            if value is not None:
                result[field] = value
                break
    return result


class OpenDartClient:
    def __init__(
        self,
        api_key: str,
        timeout_sec: float = 30.0,
        sleep_sec: float = 0.08,
        raw_cache_dir: str | Path | None = None,
    ):
        if not api_key:
            raise OpenDartError("OpenDART API key is required")
        self.api_key = api_key
        self.timeout_sec = timeout_sec
        self.sleep_sec = max(0.0, sleep_sec)
        self.session = requests.Session()
        self.raw_cache_dir = Path(raw_cache_dir) if raw_cache_dir else None

    def _json_cache_path(self, endpoint: str, params: Mapping[str, object]) -> Path | None:
        if self.raw_cache_dir is None:
            return None
        identity = json.dumps(
            {"endpoint": endpoint, "params": params},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.raw_cache_dir / endpoint.replace(".json", "") / f"{digest}.json"

    @staticmethod
    def _persist_json_cache(path: Path, payload: Mapping[str, Any], endpoint: str, params: Mapping[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        temporary = path.with_suffix(".json.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(path)
        metadata = {
            "schema_version": 1,
            "provider": "OpenDART",
            "endpoint": endpoint,
            "params": params,
            "payload_sha256": hashlib.sha256(encoded).hexdigest(),
            "fetched_at_utc": datetime.now(tz=timezone.utc).isoformat(),
        }
        path.with_suffix(".meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _get_json(self, endpoint: str, **params: object) -> dict[str, Any]:
        cache_path = self._json_cache_path(endpoint, params)
        if cache_path is not None and cache_path.exists():
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            response = self.session.get(
                f"{BASE_URL}/{endpoint}",
                params={"crtfc_key": self.api_key, **params},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            payload = response.json()
            if cache_path is not None and payload.get("status") in {"000", "013"}:
                self._persist_json_cache(cache_path, payload, endpoint, params)
        if payload.get("status") == "013":
            return {}
        if payload.get("status") != "000":
            raise OpenDartError(f"OpenDART {endpoint}: {payload.get('status')} {payload.get('message')}")
        if self.sleep_sec:
            time.sleep(self.sleep_sec)
        return payload

    def stock_to_corp_codes(self) -> dict[str, str]:
        cache_path = self.raw_cache_dir / "corpCode" / "corpCode.zip" if self.raw_cache_dir else None
        if cache_path is not None and cache_path.exists():
            content = cache_path.read_bytes()
        else:
            response = self.session.get(
                f"{BASE_URL}/corpCode.xml",
                params={"crtfc_key": self.api_key},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            content = response.content
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                cache_path.write_bytes(content)
                cache_path.with_suffix(".meta.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "provider": "OpenDART",
                            "endpoint": "corpCode.xml",
                            "payload_sha256": hashlib.sha256(content).hexdigest(),
                            "fetched_at_utc": datetime.now(tz=timezone.utc).isoformat(),
                        },
                        ensure_ascii=False,
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            xml_name = next(name for name in archive.namelist() if name.lower().endswith(".xml"))
            root = ET.fromstring(archive.read(xml_name))
        result: dict[str, str] = {}
        for item in root.findall("list"):
            stock_code = (item.findtext("stock_code") or "").strip()
            corp_code = (item.findtext("corp_code") or "").strip()
            if stock_code.isdigit() and len(stock_code) == 6 and corp_code:
                result[stock_code] = corp_code
        return result

    def periodic_reports(self, corp_code: str, start_year: int, end_year: int) -> list[DartReport]:
        request = {
            "corp_code": corp_code,
            "bgn_de": f"{start_year - 1:04d}0101",
            "end_de": f"{end_year + 1:04d}1231",
            "pblntf_ty": "A",
            "page_count": 100,
        }
        payload = self._get_json(
            "list.json",
            page_no=1,
            **request,
        )
        rows = list(payload.get("list", []))
        for page_no in range(2, int(payload.get("total_page", 1) or 1) + 1):
            rows.extend(self._get_json("list.json", page_no=page_no, **request).get("list", []))
        parsed_rows: list[tuple[dict[str, Any], tuple[int, int, str, str]]] = []
        for row in rows:
            listing = _report_listing(str(row.get("report_nm", "")), str(row.get("rcept_dt", "")))
            if listing is None:
                continue
            parsed_rows.append((row, listing))
        annual_months = [listing[1] for _row, listing in parsed_rows if listing[2] == "annual"]
        fiscal_year_end_month = Counter(annual_months).most_common(1)[0][0] if annual_months else 12

        latest: dict[tuple[int, str], DartReport] = {}
        for row, listing in parsed_rows:
            period_year, period_month, kind, available_at = listing
            offset = (period_month - fiscal_year_end_month) % 12
            expected_offset = {"annual": {0}, "half": {6}, "quarter": {3, 9}}[kind]
            if offset not in expected_offset:
                continue
            report_code = {
                ("annual", 0): "11011",
                ("half", 6): "11012",
                ("quarter", 3): "11013",
                ("quarter", 9): "11014",
            }[(kind, offset)]
            business_year = period_year + int(period_month > fiscal_year_end_month)
            if business_year < start_year or business_year > end_year:
                continue
            key = (business_year, report_code)
            candidate = DartReport(
                business_year=business_year,
                report_code=report_code,
                available_at=available_at,
                period_end=_actual_period_end(period_year, period_month),
                report_name=str(row.get("report_nm") or ""),
                receipt_no=str(row.get("rcept_no") or ""),
                fiscal_year_end_month=fiscal_year_end_month,
            )
            if key not in latest or candidate.available_at > latest[key].available_at:
                latest[key] = candidate
        return [latest[key] for key in sorted(latest)]

    def disclosures(self, corp_code: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Return every filing revision received in the requested point-in-time window."""

        request = {
            "corp_code": corp_code,
            "bgn_de": str(start_date).replace("-", ""),
            "end_de": str(end_date).replace("-", ""),
            "page_count": 100,
            "sort": "date",
            "sort_mth": "asc",
        }
        payload = self._get_json("list.json", page_no=1, **request)
        rows = [dict(row) for row in payload.get("list", [])]
        for page_no in range(2, int(payload.get("total_page", 1) or 1) + 1):
            rows.extend(
                dict(row)
                for row in self._get_json("list.json", page_no=page_no, **request).get("list", [])
            )
        return rows

    def major_accounts(self, corp_code: str, business_year: int, report_code: str) -> list[dict[str, Any]]:
        common = {
            "corp_code": corp_code,
            "bsns_year": str(business_year),
            "reprt_code": report_code,
        }
        consolidated = self._get_json("fnlttSinglAcntAll.json", fs_div="CFS", **common).get("list", [])
        if consolidated:
            return [dict(row, fs_div=row.get("fs_div", "CFS")) for row in consolidated]
        separate = self._get_json("fnlttSinglAcntAll.json", fs_div="OFS", **common).get("list", [])
        return [dict(row, fs_div=row.get("fs_div", "OFS")) for row in separate]

    def company_info(self, corp_code: str) -> dict[str, Any]:
        """Return OpenDART's registered company profile for a corporation."""

        return self._get_json("company.json", corp_code=corp_code)


def collect_fundamental_observations(
    client: OpenDartClient,
    tickers: Iterable[str],
    start_year: int,
    end_year: int,
    on_ticker: Callable[[str, list[dict[str, object]]], None] | None = None,
) -> list[dict[str, object]]:
    corp_codes = client.stock_to_corp_codes()
    observations: list[dict[str, object]] = []
    for ticker in tickers:
        normalized = str(ticker).replace("A", "").zfill(6)
        corp_code = corp_codes.get(normalized)
        if not corp_code:
            if on_ticker is not None:
                on_ticker(normalized, [])
            continue
        ticker_observations: list[dict[str, object]] = []
        for report in client.periodic_reports(corp_code, start_year, end_year):
            statement_rows = client.major_accounts(corp_code, report.business_year, report.report_code)
            fields = extract_canonical_fields(statement_rows)
            if not fields:
                continue
            ticker_observations.append(
                {
                    "ticker": normalized,
                    "available_at": report.available_at,
                    "period_end": report.period_end,
                    "source": "opendart",
                    "source_lineage": {
                        "business_year": report.business_year,
                        "report_code": report.report_code,
                        "report_name": report.report_name,
                        "receipt_no": report.receipt_no,
                        "fiscal_year_end_month": report.fiscal_year_end_month,
                        "period_mapping_method": "report_name_actual_fiscal_month",
                        "statement_period_verified": _statement_period_verified(
                            statement_rows,
                            report.period_end,
                        ),
                    },
                    "fields": fields,
                }
            )
        if on_ticker is None:
            observations.extend(ticker_observations)
        else:
            on_ticker(normalized, ticker_observations)
    return observations


def collect_company_profiles(
    client: OpenDartClient,
    tickers: Iterable[str],
    on_ticker: Callable[[str, dict[str, object] | None], None] | None = None,
) -> list[dict[str, object]]:
    """Collect compact company metadata for static relation construction."""

    corp_codes = client.stock_to_corp_codes()
    profiles: list[dict[str, object]] = []
    for ticker in tickers:
        normalized = str(ticker).replace("A", "").zfill(6)
        corp_code = corp_codes.get(normalized)
        profile: dict[str, object] | None = None
        if corp_code:
            payload = client.company_info(corp_code)
            industry_code = str(payload.get("induty_code", "")).strip()
            profile = {
                "ticker": normalized,
                "corp_code": corp_code,
                "source": "opendart_company",
                "name": str(payload.get("corp_name", "")).strip(),
                "industry_code": industry_code,
                "industry_prefix": industry_code[:2] if industry_code.isdigit() else "",
                "market_class": str(payload.get("corp_cls", "")).strip(),
                "established_on": str(payload.get("est_dt", "")).strip(),
            }
        if on_ticker is not None:
            on_ticker(normalized, profile)
        elif profile is not None:
            profiles.append(profile)
    return profiles
