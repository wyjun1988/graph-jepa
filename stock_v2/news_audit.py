from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from stock_v2.dataset_integrity import clean_text, load_json, normalize_ticker, parse_timestamp, sha256_file
from stock_v2.news_dataset import (
    _mapping_evidence,
    _normalized_title,
    _selection_lineage,
    _source_provider,
    _usable_summary,
)


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any] | None, str | None]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, None, f"JSONDecodeError: {exc.msg}"
                continue
            if not isinstance(row, dict):
                yield line_number, None, "row is not an object"
                continue
            yield line_number, row, None


def _summary(values: Iterable[int]) -> dict[str, int | float | None]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "p95": None, "max": None}

    def quantile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": quantile(0.50),
        "mean": sum(ordered) / len(ordered),
        "p95": quantile(0.95),
        "max": ordered[-1],
    }


class _DeterministicSamples:
    def __init__(self, size_per_stratum: int, seed: int) -> None:
        self.size = max(0, int(size_per_stratum))
        self.seed = int(seed)
        self.heaps: defaultdict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)

    def add(self, stratum: str, identity: str, row: Mapping[str, Any]) -> None:
        if not self.size:
            return
        score = int.from_bytes(
            hashlib.sha256(f"{self.seed}|{stratum}|{identity}".encode("utf-8")).digest()[:8],
            byteorder="big",
        )
        item = (-score, identity, {"audit_stratum": stratum, **dict(row)})
        heap = self.heaps[stratum]
        if len(heap) < self.size:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    def rows(self) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for stratum in sorted(self.heaps):
            result.extend(item[2] for item in sorted(self.heaps[stratum], reverse=True))
        return result


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def audit_news_acquisition(
    repo_root: Path,
    config: Mapping[str, Any],
    *,
    sample_size_per_stratum: int = 50,
    sample_seed: int = 20260712,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    universe_payload = load_json(repo_root / str(config["universe_manifest"]))
    universe = {
        normalize_ticker(row.get("ticker")): dict(row)
        for row in universe_payload.get("universe", [])
        if normalize_ticker(row.get("ticker"))
    }
    release_start = pd.Timestamp(config["start"]).normalize()
    release_end = pd.Timestamp(config["end"]).normalize()
    counters: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    ticker_counts: Counter[str] = Counter()
    ticker_year_counts: Counter[tuple[str, int]] = Counter()
    ticker_months: defaultdict[str, set[str]] = defaultdict(set)
    mapping_counts: Counter[str] = Counter()
    query_policy_counts: Counter[str] = Counter()
    outside_query_window_tickers: Counter[str] = Counter()
    record_ids: set[str] = set()
    article_keys: set[str] = set()
    title_day_keys: set[str] = set()
    seen_tickers: set[str] = set()
    input_files: list[dict[str, Any]] = []
    samples = _DeterministicSamples(sample_size_per_stratum, sample_seed)

    for raw_value in config.get("raw_paths", []):
        path = repo_root / str(raw_value)
        input_files.append({"path": str(raw_value), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        for line_number, record, error in _iter_jsonl(path):
            counters["raw_rows"] += 1
            if error or record is None:
                counters["invalid_json_rows"] += 1
                continue
            identity = str(record.get("id") or f"{raw_value}:{line_number}")
            if identity in record_ids:
                counters["duplicate_record_ids"] += 1
            record_ids.add(identity)
            ticker = normalize_ticker(record.get("ticker"))
            if ticker not in universe:
                counters["invalid_or_outside_universe_ticker"] += 1
                continue
            payload = record.get("article") if isinstance(record.get("article"), Mapping) else record
            published_raw = payload.get("published") or record.get("published")
            published, _precision = parse_timestamp(published_raw)
            if published is None:
                counters["invalid_publication_timestamp"] += 1
                continue
            published_date = published.tz_convert("Asia/Seoul").normalize().tz_localize(None)
            if published_date < release_start or published_date > release_end:
                counters["outside_release_window"] += 1
                continue
            title = clean_text(payload.get("title") or record.get("title"))
            source = clean_text(payload.get("source") or record.get("publisher"))
            raw_summary = clean_text(
                payload.get("body")
                or payload.get("content")
                or payload.get("summary")
                or record.get("summary")
            )
            if not title:
                counters["missing_title"] += 1
                continue
            summary = _usable_summary(raw_summary, title, source)
            provider = _source_provider(record)
            acquisition = record.get("acquisition") if isinstance(record.get("acquisition"), Mapping) else {}
            query_policy_counts[str(acquisition.get("query_policy") or "legacy_unspecified")] += 1
            acquisition_mode, selection_point_in_time = _selection_lineage(
                record,
                provider,
                published,
            )
            content_tier = (
                "official_filing"
                if provider == "opendart"
                else ("title_summary" if summary else "title_only")
            )
            if provider == "opendart":
                mapping_method, mapping_confidence = "official_corp_code", 1.0
            else:
                mapping_result = _mapping_evidence(
                    ticker,
                    str(universe[ticker].get("name") or ""),
                    title,
                    raw_summary,
                    source=source,
                )
                mapping_method, mapping_confidence = mapping_result[:2]
            normalized_title = _normalized_title(title, source)
            article_key = hashlib.sha256(
                f"{normalized_title}|{_normalized_title(source)}|{published_date.date()}".encode("utf-8")
            ).hexdigest()
            title_day_key = hashlib.sha256(
                f"{normalized_title}|{published_date.date()}".encode("utf-8")
            ).hexdigest()
            duplicate_candidate = title_day_key in title_day_keys
            if article_key in article_keys:
                counters["duplicate_article_occurrences"] += 1
            if duplicate_candidate:
                counters["same_title_day_occurrences"] += 1
            article_keys.add(article_key)
            title_day_keys.add(title_day_key)
            counters["valid_rows"] += 1
            counters[content_tier] += 1
            counters[f"acquisition_mode:{acquisition_mode}"] += 1
            counters["point_in_time_selection"] += int(selection_point_in_time)
            counters["retrospective_or_unknown_selection"] += int(not selection_point_in_time)
            source_counts[source or "unknown"] += 1
            mapping_counts[mapping_method] += 1
            ticker_counts[ticker] += 1
            ticker_year_counts[(ticker, int(published_date.year))] += 1
            ticker_months[ticker].add(published_date.strftime("%Y-%m"))
            seen_tickers.add(ticker)
            query_window = record.get("query_window") if isinstance(record.get("query_window"), Mapping) else {}
            query_start = pd.to_datetime(query_window.get("start"), errors="coerce")
            query_end = pd.to_datetime(query_window.get("end"), errors="coerce")
            outside_query_window = (
                not pd.isna(query_start)
                and not pd.isna(query_end)
                and (
                    published_date < pd.Timestamp(query_start).normalize()
                    or published_date >= pd.Timestamp(query_end).normalize()
                )
            )
            if outside_query_window:
                counters["outside_query_window"] += 1
                outside_query_window_tickers[ticker] += 1

            sample_row = {
                "source_record_id": identity,
                "ticker": ticker,
                "company_name": str(universe[ticker].get("name") or ""),
                "published_date_kst": str(published_date.date()),
                "title": title,
                "source": source,
                "source_provider": provider,
                "content_tier": content_tier,
                "mapping_method": mapping_method,
                "mapping_confidence": mapping_confidence,
                "acquisition_mode": acquisition_mode,
                "selection_point_in_time": selection_point_in_time,
                "query_window": query_window,
                "human_relevant": None,
                "human_duplicate_event_id": None,
                "human_notes": "",
            }
            samples.add("random", identity, sample_row)
            if content_tier == "title_only":
                samples.add("title_only", identity, sample_row)
            if mapping_method == "source_query_only":
                samples.add("query_only_mapping", identity, sample_row)
            if duplicate_candidate:
                samples.add("same_title_day_duplicate", identity, sample_row)
            if content_tier == "official_filing":
                samples.add("official_filing", identity, sample_row)
            if outside_query_window:
                samples.add("outside_query_window", identity, sample_row)

    coverage_tickers: set[str] = set()
    complete_coverage_tickers: set[str] = set()
    complete_coverage_by_source: defaultdict[str, set[str]] = defaultdict(set)
    coverage_counts: Counter[str] = Counter()
    coverage_query_policy_counts: Counter[str] = Counter()
    coverage_files: list[dict[str, Any]] = []
    for coverage_value in config.get("coverage_paths", []):
        path = repo_root / str(coverage_value)
        coverage_files.append(
            {"path": str(coverage_value), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
        for _line_number, row, error in _iter_jsonl(path):
            coverage_counts["rows"] += 1
            if error or row is None:
                coverage_counts["invalid_rows"] += 1
                continue
            ticker = normalize_ticker(row.get("ticker"))
            if ticker not in universe:
                coverage_counts["outside_universe"] += 1
                continue
            coverage_tickers.add(ticker)
            failed = int(row.get("request_errors", 0) or 0) > 0
            saturated = int(row.get("saturated_leaf_windows", 0) or 0) > 0
            start = pd.to_datetime(row.get("start"), errors="coerce")
            end = pd.to_datetime(row.get("end"), errors="coerce")
            wrong_range = pd.isna(start) or pd.isna(end) or start > release_start or end < release_end
            status = str(row.get("status") or "complete")
            source_text = str(row.get("source") or "").lower()
            coverage_source = (
                "opendart"
                if "opendart" in source_text
                else "google_rss"
                if "google" in source_text
                else "naver_search"
                if "naver" in source_text
                else source_text or "unknown"
            )
            coverage_counts["request_errors"] += int(failed)
            coverage_counts["saturated_tickers"] += int(saturated)
            coverage_counts["wrong_range"] += int(wrong_range)
            coverage_query_policy_counts[
                str(row.get("query_policy") or "legacy_unspecified")
            ] += 1
            if not failed and not saturated and not wrong_range and status == "complete":
                complete_coverage_tickers.add(ticker)
                complete_coverage_by_source[coverage_source].add(ticker)

    valid_rows = counters["valid_rows"]
    universe_size = len(universe)
    issues: list[dict[str, Any]] = []

    def issue(severity: str, code: str, message: str, **details: Any) -> None:
        issues.append({"severity": severity, "code": code, "message": message, **details})

    if (
        counters["invalid_json_rows"]
        or counters["invalid_publication_timestamp"]
        or counters["missing_title"]
    ):
        issue("blocker", "news_raw_integrity_failure", "Raw news contains malformed or unusable records.")
    if counters["outside_query_window"]:
        issue(
            "blocker",
            "news_query_window_violation",
            "Raw search results fall outside their recorded exclusive query windows.",
            rows=counters["outside_query_window"],
            tickers=len(outside_query_window_tickers),
        )
    if counters["duplicate_record_ids"]:
        issue("blocker", "news_duplicate_record_id", "Raw acquisition record IDs are not unique.")
    if coverage_counts["invalid_rows"] or coverage_counts["request_errors"] or coverage_counts["wrong_range"]:
        issue("blocker", "news_coverage_ledger_failure", "Collection coverage ledger has errors or an incomplete range.")
    if coverage_counts["saturated_tickers"]:
        issue(
            "blocker",
            "news_search_result_saturation",
            "At least one minimum-size query window reached the provider result cap.",
            tickers=coverage_counts["saturated_tickers"],
        )
    if len(complete_coverage_tickers) < universe_size:
        issue(
            "blocker",
            "news_ticker_collection_incomplete",
            "Not every frozen-universe ticker has a complete acquisition ledger.",
            complete_tickers=len(complete_coverage_tickers),
            required=universe_size,
        )
    for required_source in config.get("required_coverage_sources", []):
        source_tickers = complete_coverage_by_source[str(required_source)]
        if len(source_tickers) < universe_size:
            issue(
                "blocker",
                "news_required_source_incomplete",
                "A required acquisition source does not cover the full frozen universe.",
                source=str(required_source),
                complete_tickers=len(source_tickers),
                required=universe_size,
            )
    title_only_ratio = counters["title_only"] / valid_rows if valid_rows else 0.0
    query_only_ratio = mapping_counts["source_query_only"] / valid_rows if valid_rows else 0.0
    if title_only_ratio > 0.50:
        issue(
            "warning",
            "news_mostly_title_only",
            "Most search-index records have no independently usable body or summary.",
            ratio=title_only_ratio,
        )
    if query_only_ratio > 0.25:
        issue(
            "warning",
            "news_mapping_requires_relevance_filter",
            "Many ticker mappings are supported only by the source query and require explicit relevance labels.",
            ratio=query_only_ratio,
        )
    issue(
        "limitation",
        "news_archive_not_exhaustive",
        "Search-index acquisition is not a licensed exhaustive news archive; absence of a record is not absence of an event.",
    )
    if counters["retrospective_or_unknown_selection"]:
        issue(
            "limitation",
            "news_retrospective_selection_not_pit",
            "Historical discovery searches reflect the index at collection time, not the feed selected on each publication date.",
            rows=counters["retrospective_or_unknown_selection"],
        )

    report = {
        "schema_version": 1,
        "policy": {
            "source_completeness_claim": "search-index sample only",
            "raw_occurrences_preserved": True,
            "irrelevant_query_results_preserved": True,
            "duplicate_events_are_clustered_downstream": True,
            "historical_google_timestamp_precision": "date-only",
            "retrospective_search_selection_is_point_in_time": False,
        },
        "release_window": {"start": str(release_start.date()), "end": str(release_end.date())},
        "universe_tickers": universe_size,
        "input_files": input_files,
        "coverage_files": coverage_files,
        "counts": dict(counters),
        "unique_record_ids": len(record_ids),
        "unique_article_keys": len(article_keys),
        "observed_tickers": len(seen_tickers),
        "observed_ticker_ratio": len(seen_tickers) / universe_size if universe_size else 0.0,
        "complete_coverage_tickers": len(complete_coverage_tickers),
        "complete_coverage_ratio": len(complete_coverage_tickers) / universe_size if universe_size else 0.0,
        "complete_coverage_by_source": {
            source: len(tickers) for source, tickers in sorted(complete_coverage_by_source.items())
        },
        "coverage_counts": dict(coverage_counts),
        "coverage_query_policies": dict(sorted(coverage_query_policy_counts.items())),
        "query_policies": dict(sorted(query_policy_counts.items())),
        "outside_query_window_by_ticker": dict(
            sorted(outside_query_window_tickers.items())
        ),
        "title_only_ratio": title_only_ratio,
        "query_only_mapping_ratio": query_only_ratio,
        "mapping_evidence": dict(mapping_counts),
        "source_count": len(source_counts),
        "top_sources": source_counts.most_common(30),
        "articles_per_ticker": _summary(ticker_counts.values()),
        "nonempty_months_per_ticker": _summary(len(value) for value in ticker_months.values()),
        "articles_per_ticker_year": _summary(ticker_year_counts.values()),
        "lowest_volume_tickers": sorted(ticker_counts.items(), key=lambda item: (item[1], item[0]))[:30],
        "highest_volume_tickers": sorted(ticker_counts.items(), key=lambda item: (-item[1], item[0]))[:30],
        "issues": issues,
        "blocker_count": sum(row["severity"] == "blocker" for row in issues),
        "warning_count": sum(row["severity"] == "warning" for row in issues),
        "sample_seed": sample_seed,
        "sample_size_per_stratum": sample_size_per_stratum,
    }
    return report, samples.rows()
