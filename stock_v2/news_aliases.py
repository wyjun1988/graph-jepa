from __future__ import annotations

from collections import defaultdict
from datetime import date
import re
from typing import Any, Mapping, Sequence

import pandas as pd

from stock_v2.dataset_integrity import clean_text, normalize_ticker


IDENTITY_ALIAS_TYPES = {
    "legal_name",
    "orthographic_variant",
    "abbreviation",
    "former_name",
}
RELATED_ALIAS_TYPES = {"brand", "subsidiary", "affiliate"}
ALIAS_TYPES = IDENTITY_ALIAS_TYPES | RELATED_ALIAS_TYPES


def compact_alias(value: Any) -> str:
    return re.sub(r"[^0-9a-z가-힣]+", "", clean_text(value).lower())


def is_lexically_ambiguous_short_name(value: Any) -> bool:
    normalized = compact_alias(value)
    return bool(re.fullmatch(r"[0-9a-z]+", normalized) and len(normalized) <= 3)


def _parse_optional_date(value: Any, field: str) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"invalid alias {field}: {raw}")
    return str(pd.Timestamp(parsed).date())


def validate_alias_registry(
    payload: Mapping[str, Any],
    universe: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    if int(payload.get("schema_version", 0) or 0) != 1:
        raise ValueError("company alias registry schema_version must be 1")
    raw_rows = payload.get("aliases")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes)):
        raise ValueError("company alias registry aliases must be an array")

    rows_by_ticker: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    owner_by_alias: defaultdict[str, set[str]] = defaultdict(set)
    seen: set[tuple[str, str, str, str | None, str | None]] = set()
    for index, value in enumerate(raw_rows):
        if not isinstance(value, Mapping):
            raise ValueError(f"company alias row {index} must be an object")
        ticker = normalize_ticker(value.get("ticker"))
        if ticker not in universe:
            raise ValueError(f"company alias row {index} has unknown ticker: {ticker}")
        alias = clean_text(value.get("alias"))
        normalized = compact_alias(alias)
        if len(normalized) < 2:
            raise ValueError(f"company alias row {index} is too short: {alias!r}")
        alias_type = str(value.get("alias_type") or "").strip()
        if alias_type not in ALIAS_TYPES:
            raise ValueError(f"company alias row {index} has invalid alias_type: {alias_type}")
        if value.get("reviewed") is not True:
            raise ValueError(f"company alias row {index} is not explicitly reviewed")
        source = clean_text(value.get("source"))
        if not source:
            raise ValueError(f"company alias row {index} is missing source")
        confidence = float(value.get("confidence", 0.0) or 0.0)
        if not 0.0 < confidence <= 1.0:
            raise ValueError(f"company alias row {index} has invalid confidence: {confidence}")
        valid_from = _parse_optional_date(value.get("valid_from"), "valid_from")
        valid_to = _parse_optional_date(value.get("valid_to"), "valid_to")
        lexically_ambiguous = value.get("lexically_ambiguous", False)
        if not isinstance(lexically_ambiguous, bool):
            raise ValueError(f"company alias row {index} has non-boolean lexically_ambiguous")
        if valid_from and valid_to and valid_from > valid_to:
            raise ValueError(f"company alias row {index} has reversed validity")
        key = (ticker, normalized, alias_type, valid_from, valid_to)
        if key in seen:
            raise ValueError(f"duplicate company alias row {index}: {ticker} {alias}")
        seen.add(key)
        owner_by_alias[normalized].add(ticker)
        rows_by_ticker[ticker].append(
            {
                "ticker": ticker,
                "alias": alias,
                "normalized_alias": normalized,
                "alias_type": alias_type,
                "relationship": "identity" if alias_type in IDENTITY_ALIAS_TYPES else "related_entity",
                "source": source,
                "confidence": confidence,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "reviewed": True,
                "lexically_ambiguous": lexically_ambiguous,
            }
        )

    ambiguous = {alias: sorted(owners) for alias, owners in owner_by_alias.items() if len(owners) > 1}
    if ambiguous:
        examples = list(sorted(ambiguous.items()))[:10]
        raise ValueError(f"ambiguous company aliases must be resolved: {examples}")
    for rows in rows_by_ticker.values():
        rows.sort(
            key=lambda row: (
                -len(str(row["normalized_alias"])),
                str(row["alias_type"]),
                str(row["alias"]),
            )
        )
    return dict(rows_by_ticker)


def alias_is_active(row: Mapping[str, Any], published_date: Any) -> bool:
    if published_date is None:
        return True
    if isinstance(published_date, date):
        candidate = published_date.isoformat()
    else:
        parsed = pd.to_datetime(published_date, errors="coerce")
        if pd.isna(parsed):
            return False
        candidate = str(pd.Timestamp(parsed).date())
    valid_from = str(row.get("valid_from") or "")
    valid_to = str(row.get("valid_to") or "")
    return (not valid_from or candidate >= valid_from) and (not valid_to or candidate <= valid_to)


def text_contains_alias(text: Any, alias: Any) -> bool:
    raw_text = clean_text(text).lower()
    raw_alias = clean_text(alias).lower()
    normalized = compact_alias(raw_alias)
    if not raw_text or not normalized:
        return False
    if re.fullmatch(r"[0-9a-z+&. -]+", raw_alias) and len(normalized) <= 4:
        escaped = re.escape(raw_alias).replace(r"\ ", r"\s+")
        return bool(re.search(rf"(?<![0-9a-z]){escaped}(?![0-9a-z])", raw_text))
    if re.fullmatch(r"[가-힣]{2,8}", raw_alias):
        particles = (
            "에게서|으로부터|에서는|에게는|으로는|에서|에게|으로|까지|부터|처럼|보다|"
            "에만|에는|이라는|라는|이라|라고|이며|이고|은|는|이|가|을|를|의|과|와|도|만|에|로|측|발"
        )
        return bool(
            re.search(
                rf"(?<![0-9a-z가-힣]){re.escape(raw_alias)}(?:{particles})?(?=$|[^0-9a-z가-힣])",
                raw_text,
            )
        )
    return normalized in compact_alias(raw_text)


def best_alias_match(
    text: Any,
    aliases: Sequence[Mapping[str, Any]],
    published_date: Any,
) -> dict[str, Any] | None:
    candidates = [
        dict(row)
        for row in aliases
        if alias_is_active(row, published_date) and text_contains_alias(text, row.get("alias"))
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            1 if row.get("relationship") == "identity" else 0,
            float(row.get("confidence", 0.0) or 0.0),
            len(str(row.get("normalized_alias") or "")),
            str(row.get("alias") or ""),
        ),
    )
