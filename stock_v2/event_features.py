from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from decimal import Decimal, ROUND_HALF_EVEN, localcontext
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_coverage_path(event_path: Path) -> Path | None:
    """Resolve a release-bound coverage sidecar and verify both file hashes."""

    manifest_path = event_path.parent / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(f"invalid event release manifest: {manifest_path}") from exc
    outputs = manifest.get("output_files")
    if not isinstance(outputs, Mapping) or event_path.name not in outputs:
        return None
    event_record = outputs.get(event_path.name) or {}
    expected_event_sha = str(event_record.get("sha256") or "")
    if len(expected_event_sha) != 64 or _file_sha256(event_path) != expected_event_sha:
        raise ValueError(f"event release hash mismatch: {event_path}")
    coverage_path = event_path.parent / "coverage.jsonl"
    coverage_record = outputs.get(coverage_path.name) or {}
    expected_coverage_sha = str(coverage_record.get("sha256") or "")
    if (
        not coverage_path.exists()
        or len(expected_coverage_sha) != 64
        or _file_sha256(coverage_path) != expected_coverage_sha
    ):
        raise ValueError(f"event coverage release hash mismatch: {coverage_path}")
    return coverage_path


def clean_ticker(value: Any) -> str:
    text = str(value or "").strip().replace("A", "")
    match = re.search(r"\d{6}", text)
    return match.group(0) if match else text


def parse_event_date(record: Mapping[str, Any]) -> pd.Timestamp | None:
    candidates = [
        record.get("published"),
        record.get("published_at"),
        record.get("date"),
        record.get("ts"),
    ]
    article = record.get("article")
    if isinstance(article, Mapping):
        candidates.extend([article.get("published"), article.get("updated")])
    for value in candidates:
        if not value:
            continue
        text = str(value)
        try:
            if "," in text and any(tz in text for tz in ["GMT", "KST", "+", "-"]):
                parsed = parsedate_to_datetime(text)
                if parsed is not None:
                    return pd.Timestamp(parsed).tz_localize(None).normalize()
        except Exception:
            pass
        try:
            parsed = pd.to_datetime(text, errors="coerce")
            if pd.notna(parsed):
                return pd.Timestamp(parsed).tz_localize(None).normalize()
        except Exception:
            continue
    return None


def _event_row(index: pd.DatetimeIndex, record: Mapping[str, Any], lag_days: int) -> int | None:
    effective = pd.to_datetime(record.get("effective_session"), errors="coerce")
    if pd.notna(effective):
        effective_date = pd.Timestamp(effective)
        if effective_date.tzinfo is not None:
            effective_date = effective_date.tz_convert(None)
        return int(index.searchsorted(effective_date.normalize(), side="left"))
    event_date = parse_event_date(record)
    if event_date is None:
        return None
    return int(index.searchsorted(event_date, side="left")) + int(lag_days)


def _iter_jsonl(paths: Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload


def _event_payload(record: Mapping[str, Any]) -> Mapping[str, Any]:
    event = record.get("event")
    return event if isinstance(event, Mapping) else record


def _is_relevant_sensor(payload: Mapping[str, Any]) -> bool:
    """Keep entity negatives and non-event content out of latent channels."""

    if "relevance" not in payload:
        return True
    try:
        relevance = float(payload["relevance"])
    except (TypeError, ValueError, OverflowError):
        return False
    if not math.isfinite(relevance) or relevance < 0.5:
        return False
    if "event_specificity" not in payload:
        return True
    try:
        event_specificity = float(payload["event_specificity"])
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(event_specificity) and event_specificity >= 0.5


def _sensor_evidence_mass(payload: Mapping[str, Any]) -> float:
    """Weight accepted event counts by entity, event, and evidence quality."""

    try:
        relevance = float(payload.get("relevance", 1.0))
        event_specificity = float(payload.get("event_specificity", 1.0))
        evidence_quality = float(payload.get("evidence_quality", 1.0))
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not all(math.isfinite(value) for value in (relevance, event_specificity, evidence_quality)):
        return 0.0
    return (
        max(0.0, min(1.0, relevance))
        * max(0.0, min(1.0, event_specificity))
        * max(0.0, min(1.0, evidence_quality))
    )


def _record_ticker(record: Mapping[str, Any], payload: Mapping[str, Any]) -> str:
    ticker = clean_ticker(record.get("ticker"))
    if re.fullmatch(r"\d{6}", ticker):
        return ticker
    for key in ["affected_nodes", "nodes"]:
        nodes = payload.get(key)
        if isinstance(nodes, Sequence) and not isinstance(nodes, (str, bytes)):
            for node in nodes:
                candidate = clean_ticker(node)
                if re.fullmatch(r"\d{6}", candidate):
                    return candidate
    for delta in payload.get("node_deltas", []) or []:
        if isinstance(delta, Mapping):
            candidate = clean_ticker(delta.get("node"))
            if re.fullmatch(r"\d{6}", candidate):
                return candidate
    return ticker


def build_event_ticker_coverage(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    event_paths: Sequence[str | Path] | None,
) -> pd.DataFrame:
    """Mark tickers with an actual event-source observation stream.

    A zero-valued event feature means "no event" only after a ticker has a
    source stream. Tickers absent from every source file stay unobserved rather
    than becoming synthetic zero-news examples.
    """

    index = pd.DatetimeIndex(dates)
    ticker_list = [clean_ticker(ticker) for ticker in tickers]
    ticker_pos = {ticker: index for index, ticker in enumerate(ticker_list)}
    values = np.zeros((len(index), len(ticker_list)), dtype=bool)
    paths = [Path(path) for path in event_paths or []]
    for path in paths:
        window_coverage_path = Path(f"{path}.windows.jsonl")
        if window_coverage_path.exists():
            for record in _iter_jsonl([window_coverage_path]):
                ticker = clean_ticker(record.get("ticker"))
                if ticker not in ticker_pos or str(record.get("status") or "") != "complete":
                    continue
                start = pd.to_datetime(record.get("start"), errors="coerce")
                end_exclusive = pd.to_datetime(record.get("end_exclusive"), errors="coerce")
                if pd.isna(start) or pd.isna(end_exclusive):
                    continue
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end_exclusive)
                if start_date.tzinfo is not None:
                    start_date = start_date.tz_convert(None)
                if end_date.tzinfo is not None:
                    end_date = end_date.tz_convert(None)
                observed = (index >= start_date.normalize()) & (index < end_date.normalize())
                values[observed, ticker_pos[ticker]] = True
            continue
        coverage_path = Path(f"{path}.coverage.jsonl")
        selected_coverage_path = (
            coverage_path
            if coverage_path.exists()
            else _release_coverage_path(path)
        )
        records = (
            [record for record in _iter_jsonl([selected_coverage_path])]
            if selected_coverage_path is not None
            else []
        )
        if records:
            # A sidecar is the backfill completion contract. Do not infer full
            # historical coverage from rows written before an interrupted run.
            for record in records:
                ticker = clean_ticker(record.get("ticker"))
                if ticker not in ticker_pos:
                    continue
                status = str(record.get("status") or "complete").strip().lower()
                if (
                    status not in {"complete", "success"}
                    or int(record.get("request_errors", 0) or 0) > 0
                    or int(record.get("saturated_leaf_windows", 0) or 0) > 0
                ):
                    continue
                start = pd.to_datetime(record.get("start"), errors="coerce")
                end = pd.to_datetime(record.get("end"), errors="coerce")
                if pd.isna(start) or pd.isna(end):
                    values[:, ticker_pos[ticker]] = True
                    continue
                start_date = pd.Timestamp(start)
                end_date = pd.Timestamp(end)
                if start_date.tzinfo is not None:
                    start_date = start_date.tz_convert(None)
                if end_date.tzinfo is not None:
                    end_date = end_date.tz_convert(None)
                observed = (index >= start_date.normalize()) & (index <= end_date.normalize())
                values[observed, ticker_pos[ticker]] = True
            continue
        if selected_coverage_path is not None:
            # An empty sidecar means the collector has not completed a ticker.
            continue
        for record in _iter_jsonl([path]):
            payload = _event_payload(record)
            ticker = _record_ticker(record, payload)
            if ticker in ticker_pos:
                values[:, ticker_pos[ticker]] = True
    return pd.DataFrame(values, index=dates, columns=tickers)


def _score_record(record: Mapping[str, Any], payload: Mapping[str, Any], ticker: str) -> tuple[float, float, float, float]:
    score = 0.0
    abs_score = 0.0
    confidence_values: list[float] = []
    deltas = payload.get("node_deltas", []) or []
    for delta in deltas:
        if not isinstance(delta, Mapping):
            continue
        node = clean_ticker(delta.get("node"))
        if node != ticker:
            continue
        try:
            raw_delta = float(delta.get("delta", 0.0))
            confidence = float(delta.get("confidence", payload.get("confidence", 1.0)))
        except Exception:
            continue
        weighted = raw_delta * confidence
        score += weighted
        abs_score += abs(weighted)
        confidence_values.append(max(0.0, min(1.0, confidence)))
    if not confidence_values:
        try:
            polarity = float(payload.get("polarity", 0.0))
            magnitude = float(payload.get("magnitude", 0.0))
            confidence = float(payload.get("confidence", 0.0))
        except Exception:
            polarity = magnitude = confidence = 0.0
        score = polarity * magnitude * confidence
        abs_score = abs(score)
        confidence_values.append(max(0.0, min(1.0, confidence)))
    if "score_contribution" in record:
        try:
            score = float(record.get("score_contribution", score))
            abs_score = max(abs_score, abs(score))
        except Exception:
            pass
    confidence_mean = float(np.mean(confidence_values)) if confidence_values else 0.0
    pos = max(0.0, score)
    neg = max(0.0, -score)
    return score, pos, neg, max(abs_score, pos + neg), confidence_mean


def _clean_theme(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().lower()
    if not text:
        return ""
    if re.fullmatch(r"\d{6}", clean_ticker(text)):
        return ""
    return text[:48]


def _record_themes(record: Mapping[str, Any], payload: Mapping[str, Any], ticker: str) -> list[str]:
    themes: list[str] = []
    for delta in payload.get("edge_deltas", []) or []:
        if not isinstance(delta, Mapping):
            continue
        dst = clean_ticker(delta.get("dst"))
        src = _clean_theme(delta.get("src"))
        edge_type = str(delta.get("edge_type", "")).lower()
        if src and (dst == ticker or edge_type == "theme_exposure"):
            themes.append(src)

    raw_llm = payload.get("raw_llm")
    if isinstance(raw_llm, Mapping):
        raw_themes = raw_llm.get("themes")
        if isinstance(raw_themes, str):
            raw_themes = [raw_themes]
        if isinstance(raw_themes, Sequence):
            themes.extend(_clean_theme(theme) for theme in raw_themes)

    for key in ["themes", "affected_nodes"]:
        values = payload.get(key)
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            continue
        for value in values:
            theme = _clean_theme(value)
            if theme:
                themes.append(theme)

    event_type = _clean_theme(payload.get("event_type"))
    if event_type and event_type != "unknown":
        themes.append(f"event:{event_type}")

    seen: set[str] = set()
    unique: list[str] = []
    for theme in themes:
        if theme and theme not in seen:
            seen.add(theme)
            unique.append(theme)
    return unique


def _record_theme_strength(payload: Mapping[str, Any], ticker: str, fallback_abs_score: float, fallback_confidence: float) -> float:
    strengths: list[float] = []
    for delta in payload.get("edge_deltas", []) or []:
        if not isinstance(delta, Mapping):
            continue
        dst = clean_ticker(delta.get("dst"))
        if dst != ticker:
            continue
        try:
            delta_weight = abs(float(delta.get("delta_weight", 0.0)))
            confidence = float(delta.get("confidence", fallback_confidence))
        except Exception:
            continue
        strengths.append(delta_weight * max(0.0, min(1.0, confidence)))
    if strengths:
        return max(strengths)
    return float(fallback_abs_score) * max(0.0, min(1.0, float(fallback_confidence)))


def _deterministic_log1p_float32(values: np.ndarray) -> np.ndarray:
    """Compute log1p identically across ARM and x86 math libraries."""

    array = np.asarray(values, dtype=np.float32)
    unique, inverse = np.unique(array, return_inverse=True)
    transformed = np.empty(unique.shape, dtype=np.float32)
    quantum = Decimal("0.00000001")
    with localcontext() as context:
        context.prec = 34
        for index, value in enumerate(unique):
            numeric = float(value)
            if not math.isfinite(numeric):
                transformed[index] = np.float32(numeric)
            elif numeric <= 0.0:
                transformed[index] = np.float32(0.0)
            else:
                logged = (Decimal(1) + Decimal.from_float(numeric)).ln()
                transformed[index] = np.float32(
                    float(logged.quantize(quantum, rounding=ROUND_HALF_EVEN))
                )
    return transformed[inverse].reshape(array.shape)


def _bounded_event_intensity(values: np.ndarray, counts: np.ndarray, *, nonnegative: bool) -> np.ndarray:
    """Keep event direction bounded while retaining volume as separate count features."""

    denominator = np.sqrt(np.maximum(np.asarray(counts, dtype=np.float32), 1.0))
    normalized = np.asarray(values, dtype=np.float32) / denominator
    lower = 0.0 if nonnegative else -1.0
    return np.clip(normalized, lower, 1.0).astype(np.float32)


def build_event_feature_frames(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    event_paths: Sequence[str | Path] | None,
    half_life_days: float = 5.0,
    lag_days: int = 1,
    max_decay_days: int = 60,
) -> dict[str, pd.DataFrame]:
    """Convert JSONL market/news events into daily node feature frames.

    Events are conservatively applied from the next trading row by default
    (`lag_days=1`) to avoid using after-close news on the same signal date.
    """

    index = pd.DatetimeIndex(dates).normalize()
    ticker_list = [clean_ticker(ticker) for ticker in tickers]
    columns = pd.Index(ticker_list)
    shape = (len(index), len(columns))
    daily_score = np.zeros(shape, dtype=np.float32)
    daily_count = np.zeros(shape, dtype=np.float32)
    daily_pos = np.zeros(shape, dtype=np.float32)
    daily_neg = np.zeros(shape, dtype=np.float32)
    daily_abs = np.zeros(shape, dtype=np.float32)
    daily_conf = np.zeros(shape, dtype=np.float32)
    col_pos = {ticker: idx for idx, ticker in enumerate(ticker_list)}

    seen_event_ids: set[tuple[str, str]] = set()
    if event_paths:
        for record in _iter_jsonl(event_paths):
            payload = _event_payload(record)
            if not _is_relevant_sensor(payload):
                continue
            ticker = _record_ticker(record, payload)
            if ticker not in col_pos:
                continue
            event_id = str(record.get("queue_id") or record.get("id") or "").strip()
            deduplication_key: tuple[str, str] | None = None
            if event_id:
                deduplication_key = (ticker, event_id)
                if deduplication_key in seen_event_ids:
                    continue
            row = _event_row(index, record, lag_days)
            if row is None:
                continue
            if row < 0 or row >= len(index):
                continue
            evidence_mass = _sensor_evidence_mass(payload)
            if evidence_mass <= 0.0:
                continue
            if deduplication_key is not None:
                seen_event_ids.add(deduplication_key)
            col = col_pos[ticker]
            score, pos, neg, abs_score, confidence = _score_record(record, payload, ticker)
            daily_score[row, col] += float(score)
            daily_count[row, col] += evidence_mass
            daily_pos[row, col] += float(pos)
            daily_neg[row, col] += float(neg)
            daily_abs[row, col] += float(abs_score)
            daily_conf[row, col] += float(confidence)

    avg_conf = np.divide(daily_conf, daily_count, out=np.zeros_like(daily_conf), where=daily_count > 0)
    daily_score = _bounded_event_intensity(daily_score, daily_count, nonnegative=False)
    daily_pos = _bounded_event_intensity(daily_pos, daily_count, nonnegative=True)
    daily_neg = _bounded_event_intensity(daily_neg, daily_count, nonnegative=True)
    daily_abs = _bounded_event_intensity(daily_abs, daily_count, nonnegative=True)
    alpha = math.exp(-math.log(2.0) / max(float(half_life_days), 1e-6))
    decayed_score = np.zeros(shape, dtype=np.float32)
    decayed_abs = np.zeros(shape, dtype=np.float32)
    decayed_count = np.zeros(shape, dtype=np.float32)
    for row in range(len(index)):
        if row == 0:
            decayed_score[row] = daily_score[row]
            decayed_abs[row] = daily_abs[row]
            decayed_count[row] = daily_count[row]
        else:
            decayed_score[row] = daily_score[row] + alpha * decayed_score[row - 1]
            decayed_abs[row] = daily_abs[row] + alpha * decayed_abs[row - 1]
            decayed_count[row] = daily_count[row] + alpha * decayed_count[row - 1]
    if max_decay_days > 0:
        # Numerical guard: very old events should not leave microscopic tails forever.
        tiny = 0.5 ** (max_decay_days / max(float(half_life_days), 1e-6))
        decayed_score[np.abs(decayed_score) < tiny * 1e-3] = 0.0
        decayed_abs[decayed_abs < tiny * 1e-3] = 0.0
        decayed_count[decayed_count < tiny * 1e-3] = 0.0

    def frame(values: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(values, index=dates, columns=tickers, dtype=np.float32)

    rolling3_count = pd.DataFrame(daily_count, index=dates, columns=tickers).rolling(3, min_periods=1).sum()
    rolling10_count = pd.DataFrame(daily_count, index=dates, columns=tickers).rolling(10, min_periods=1).sum()
    rolling3_pos = pd.DataFrame(daily_pos, index=dates, columns=tickers).rolling(3, min_periods=1).sum()
    rolling3_neg = pd.DataFrame(daily_neg, index=dates, columns=tickers).rolling(3, min_periods=1).sum()

    return {
        "news_score_1d": frame(daily_score),
        "news_score_decay": frame(decayed_score),
        "news_abs_decay": frame(decayed_abs),
        "news_count_decay": frame(_deterministic_log1p_float32(decayed_count)),
        "news_count_3d": rolling3_count.astype(np.float32),
        "news_count_10d": rolling10_count.astype(np.float32),
        "news_pos_3d": rolling3_pos.astype(np.float32),
        "news_neg_3d": rolling3_neg.astype(np.float32),
        "news_confidence_1d": frame(avg_conf),
    }


def build_event_theme_exposure(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    event_paths: Sequence[str | Path] | None,
    half_life_days: float = 5.0,
    lag_days: int = 1,
    max_decay_days: int = 60,
    max_themes: int = 96,
    min_theme_count: float = 2.0,
) -> tuple[np.ndarray | None, list[str]]:
    """Build decayed ticker-by-theme exposures for event-derived graph edges."""

    if not event_paths:
        return None, []

    index = pd.DatetimeIndex(dates).normalize()
    ticker_list = [clean_ticker(ticker) for ticker in tickers]
    ticker_pos = {ticker: idx for idx, ticker in enumerate(ticker_list)}
    prepared: list[tuple[int, int, list[str], float]] = []
    theme_counts: Counter[str] = Counter()

    seen_event_ids: set[tuple[str, str]] = set()
    for record in _iter_jsonl(event_paths):
        payload = _event_payload(record)
        if not _is_relevant_sensor(payload):
            continue
        ticker = _record_ticker(record, payload)
        if ticker not in ticker_pos:
            continue
        event_id = str(record.get("queue_id") or record.get("id") or "").strip()
        deduplication_key: tuple[str, str] | None = None
        if event_id:
            deduplication_key = (ticker, event_id)
            if deduplication_key in seen_event_ids:
                continue
        row = _event_row(index, record, lag_days)
        if row is None:
            continue
        if row < 0 or row >= len(index):
            continue
        score, _pos, _neg, abs_score, confidence = _score_record(record, payload, ticker)
        themes = _record_themes(record, payload, ticker)
        if not themes:
            continue
        evidence_mass = _sensor_evidence_mass(payload)
        if evidence_mass <= 0.0:
            continue
        if deduplication_key is not None:
            seen_event_ids.add(deduplication_key)
        strength = _record_theme_strength(
            payload,
            ticker,
            fallback_abs_score=max(abs_score, abs(score)),
            fallback_confidence=confidence,
        )
        if strength <= 0.0:
            strength = max(abs(score), abs_score, 0.05 * evidence_mass)
        strength = float(min(2.0, strength))
        prepared.append((row, ticker_pos[ticker], themes, strength))
        for theme in themes:
            theme_counts[theme] += evidence_mass

    if not prepared:
        return None, []

    eligible = [
        theme
        for theme, count in theme_counts.most_common()
        if count >= float(min_theme_count)
    ][: int(max_themes)]
    theme_pos = {theme: idx for idx, theme in enumerate(eligible)}
    if not theme_pos:
        return None, []

    daily = np.zeros((len(index), len(ticker_list), len(eligible)), dtype=np.float32)
    for row, col, themes, strength in prepared:
        matched = [theme_pos[theme] for theme in themes if theme in theme_pos]
        if not matched:
            continue
        per_theme = float(strength) / math.sqrt(len(matched))
        for theme_idx in matched:
            daily[row, col, theme_idx] += per_theme

    alpha = math.exp(-math.log(2.0) / max(float(half_life_days), 1e-6))
    decayed = np.zeros_like(daily)
    for row in range(len(index)):
        if row == 0:
            decayed[row] = daily[row]
        else:
            decayed[row] = daily[row] + alpha * decayed[row - 1]
    if max_decay_days > 0:
        tiny = 0.5 ** (max_decay_days / max(float(half_life_days), 1e-6))
        decayed[decayed < tiny * 1e-3] = 0.0
    return decayed.astype(np.float32), eligible


def event_feature_names() -> list[str]:
    return [
        "news_score_1d",
        "news_score_decay",
        "news_abs_decay",
        "news_count_decay",
        "news_count_3d",
        "news_count_10d",
        "news_pos_3d",
        "news_neg_3d",
        "news_confidence_1d",
    ]
