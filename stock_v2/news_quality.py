from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _iter_jsonl(paths: Sequence[str | Path]) -> Iterable[dict[str, Any]]:
    for raw_path in paths:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_number}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSON row at {path}:{line_number}")
                yield row


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _cluster_bucket(value: Any) -> str:
    try:
        size = max(0, int(value))
    except (TypeError, ValueError):
        return "invalid"
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 5:
        return "3-5"
    if size <= 10:
        return "6-10"
    return "11+"


def _relevance_bucket(value: float) -> str:
    if value < 0.3:
        return "0.0-0.3"
    if value < 0.5:
        return "0.3-0.5"
    if value < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_news_quality_report(
    *,
    occurrence_paths: Sequence[str | Path],
    article_paths: Sequence[str | Path],
    cluster_paths: Sequence[str | Path],
    mapping_paths: Sequence[str | Path],
    queue_paths: Sequence[str | Path],
    structured_paths: Sequence[str | Path] = (),
    neutral_event_paths: Sequence[str | Path] = (),
    coverage_paths: Sequence[str | Path] = (),
    coverage_window_paths: Sequence[str | Path] = (),
) -> dict[str, Any]:
    occurrence_ids: set[str] = set()
    occurrence_article_tickers: set[tuple[str, str]] = set()
    occurrence_providers: Counter[str] = Counter()
    duplicate_occurrence_ids = 0
    occurrence_rows = 0
    for row in _iter_jsonl(occurrence_paths):
        occurrence_rows += 1
        occurrence_id = str(row.get("occurrence_id") or "")
        if occurrence_id in occurrence_ids:
            duplicate_occurrence_ids += 1
        elif occurrence_id:
            occurrence_ids.add(occurrence_id)
        occurrence_article_tickers.add(
            (str(row.get("article_id") or ""), str(row.get("ticker") or ""))
        )
        occurrence_providers[str(row.get("source_provider") or "unknown")] += 1

    article_ids: set[str] = set()
    article_providers: Counter[str] = Counter()
    content_tiers: Counter[str] = Counter()
    article_rows = 0
    duplicate_article_ids = 0
    point_in_time_selection_rows = 0
    retrospective_or_unknown_selection_rows = 0
    for row in _iter_jsonl(article_paths):
        article_rows += 1
        article_id = str(row.get("article_id") or "")
        if article_id in article_ids:
            duplicate_article_ids += 1
        elif article_id:
            article_ids.add(article_id)
        article_providers[str(row.get("source_provider") or "unknown")] += 1
        content_tiers[str(row.get("content_tier") or "unknown")] += 1
        if bool(row.get("selection_point_in_time")):
            point_in_time_selection_rows += 1
        else:
            retrospective_or_unknown_selection_rows += 1

    cluster_ids: set[str] = set()
    cluster_size_buckets: Counter[str] = Counter()
    cluster_rows = 0
    clustered_article_rows = 0
    duplicate_cluster_ids = 0
    for row in _iter_jsonl(cluster_paths):
        cluster_rows += 1
        cluster_id = str(row.get("event_cluster_id") or row.get("semantic_cluster_id") or "")
        if cluster_id in cluster_ids:
            duplicate_cluster_ids += 1
        elif cluster_id:
            cluster_ids.add(cluster_id)
        try:
            article_count = int(row.get("article_count", row.get("source_count", 0)) or 0)
        except (TypeError, ValueError):
            article_count = 0
        clustered_article_rows += max(0, article_count)
        cluster_size_buckets[_cluster_bucket(article_count)] += 1

    mapping_methods: Counter[str] = Counter()
    mapping_ticker_rows: Counter[str] = Counter()
    mapping_ticker_methods: dict[str, Counter[str]] = {}
    mapping_rows = 0
    relevance_required_rows = 0
    alias_matched_rows = 0
    ambiguous_alias_rows = 0
    matched_entity_rows = 0
    ambiguous_mapping_rows = 0
    manifest_name_matched_rows = 0
    related_entity_alias_rows = 0
    for row in _iter_jsonl(mapping_paths):
        mapping_rows += 1
        ticker = str(row.get("ticker") or "")
        if ticker:
            mapping_ticker_rows[ticker] += 1
        mapping_method = str(row.get("mapping_method") or "unknown")
        reviewed_alias = mapping_method.startswith("reviewed_")
        mapping_methods[mapping_method] += 1
        if ticker:
            mapping_ticker_methods.setdefault(ticker, Counter())[mapping_method] += 1
        relevance_required_rows += int(bool(row.get("requires_relevance_classification")))
        matched_entity_rows += int(bool(str(row.get("matched_alias") or "").strip()))
        alias_matched_rows += int(reviewed_alias)
        ambiguous_mapping_rows += int(bool(row.get("matched_alias_ambiguous")))
        ambiguous_alias_rows += int(reviewed_alias and bool(row.get("matched_alias_ambiguous")))
        manifest_name_matched_rows += int(
            str(row.get("matched_alias_source") or "") == "universe_manifest"
            and str(row.get("matched_alias_type") or "") == "security_name"
        )
        related_entity_alias_rows += int(
            str(row.get("matched_alias_type") or "") in {"brand", "subsidiary", "affiliate"}
        )

    queue_ids: set[str] = set()
    queue_methods: Counter[str] = Counter()
    queue_tiers: Counter[str] = Counter()
    queue_cluster_buckets: Counter[str] = Counter()
    queue_tickers: set[str] = set()
    queue_ticker_rows: Counter[str] = Counter()
    queue_ticker_methods: dict[str, Counter[str]] = {}
    duplicate_queue_ids = 0
    queue_rows = 0
    queue_alias_matched_rows = 0
    queue_ambiguous_alias_rows = 0
    queue_matched_entity_rows = 0
    queue_ambiguous_mapping_rows = 0
    queue_manifest_name_matched_rows = 0
    queue_point_in_time_rows = 0
    for row in _iter_jsonl(queue_paths):
        queue_rows += 1
        queue_id = str(row.get("queue_id") or "")
        if queue_id in queue_ids:
            duplicate_queue_ids += 1
        elif queue_id:
            queue_ids.add(queue_id)
        mapping_method = str(row.get("mapping_method") or "unknown")
        reviewed_alias = mapping_method.startswith("reviewed_")
        queue_methods[mapping_method] += 1
        queue_tiers[str(row.get("content_tier") or "unknown")] += 1
        queue_cluster_buckets[_cluster_bucket(row.get("cluster_size", 1))] += 1
        ticker = str(row.get("ticker") or "")
        queue_tickers.add(ticker)
        if ticker:
            queue_ticker_rows[ticker] += 1
            queue_ticker_methods.setdefault(ticker, Counter())[mapping_method] += 1
        queue_matched_entity_rows += int(bool(str(row.get("matched_alias") or "").strip()))
        queue_alias_matched_rows += int(reviewed_alias)
        queue_ambiguous_mapping_rows += int(bool(row.get("matched_alias_ambiguous")))
        queue_ambiguous_alias_rows += int(
            reviewed_alias and bool(row.get("matched_alias_ambiguous"))
        )
        queue_manifest_name_matched_rows += int(
            str(row.get("matched_alias_source") or "") == "universe_manifest"
            and str(row.get("matched_alias_type") or "") == "security_name"
        )
        queue_point_in_time_rows += int(bool(row.get("selection_point_in_time")))

    structured_ids: set[str] = set()
    relevance_buckets: Counter[str] = Counter()
    event_specificity_buckets: Counter[str] = Counter()
    structured_event_types: Counter[str] = Counter()
    structured_ticker_rows: Counter[str] = Counter()
    structured_mapping_methods: Counter[str] = Counter()
    structured_content_tiers: Counter[str] = Counter()
    structured_outcomes_by_ticker: dict[str, Counter[str]] = {}
    structured_outcomes_by_mapping_method: dict[str, Counter[str]] = {}
    entity_related_rows = 0
    accepted_sensor_rows = 0
    low_relevance_delta_rows = 0
    low_specificity_delta_rows = 0
    invalid_relevance_rows = 0
    invalid_event_specificity_rows = 0
    structured_rows = 0
    for row in _iter_jsonl(structured_paths):
        structured_rows += 1
        ticker = str(row.get("ticker") or "")
        if ticker:
            structured_ticker_rows[ticker] += 1
        structured_ids.add(str(row.get("queue_id") or ""))
        event = row.get("event") if isinstance(row.get("event"), Mapping) else {}
        try:
            relevance = float(event.get("relevance"))
        except (TypeError, ValueError, OverflowError):
            invalid_relevance_rows += 1
            continue
        if not math.isfinite(relevance) or relevance < 0.0 or relevance > 1.0:
            invalid_relevance_rows += 1
            continue
        try:
            event_specificity = float(event.get("event_specificity"))
        except (TypeError, ValueError, OverflowError):
            invalid_event_specificity_rows += 1
            continue
        if (
            not math.isfinite(event_specificity)
            or event_specificity < 0.0
            or event_specificity > 1.0
        ):
            invalid_event_specificity_rows += 1
            continue
        relevance_buckets[_relevance_bucket(relevance)] += 1
        event_specificity_buckets[_relevance_bucket(event_specificity)] += 1
        structured_event_types[str(event.get("event_type") or "unknown")] += 1
        mapping_method = str(event.get("mapping_method") or "unknown")
        content_tier = str(event.get("content_tier") or "unknown")
        structured_mapping_methods[mapping_method] += 1
        structured_content_tiers[content_tier] += 1
        entity_related_rows += int(relevance >= 0.5)
        accepted = relevance >= 0.5 and event_specificity >= 0.5
        accepted_sensor_rows += int(accepted)
        outcome = {
            "rows": 1,
            "entity_related_rows": int(relevance >= 0.5),
            "accepted_sensor_rows": int(accepted),
            "entity_related_but_nonspecific_rows": int(
                relevance >= 0.5 and event_specificity < 0.5
            ),
        }
        for key, value in outcome.items():
            structured_outcomes_by_mapping_method.setdefault(
                mapping_method, Counter()
            )[key] += value
            if ticker:
                structured_outcomes_by_ticker.setdefault(ticker, Counter())[key] += value
        node_deltas = event.get("node_deltas")
        edge_deltas = event.get("edge_deltas")
        has_sensor_output = (
            isinstance(node_deltas, Sequence)
            and not isinstance(node_deltas, (str, bytes))
            and bool(node_deltas)
        ) or (
            isinstance(edge_deltas, Sequence)
            and not isinstance(edge_deltas, (str, bytes))
            and bool(edge_deltas)
        )
        low_relevance_delta_rows += int(
            relevance < 0.5 and has_sensor_output
        )
        low_specificity_delta_rows += int(
            relevance >= 0.5 and event_specificity < 0.5 and has_sensor_output
        )

    neutral_rows = sum(1 for _ in _iter_jsonl(neutral_event_paths))

    coverage_sources: Counter[str] = Counter()
    coverage_statuses: Counter[str] = Counter()
    coverage_ticker_rows: Counter[str] = Counter()
    saturated_coverage_rows = 0
    failed_coverage_rows = 0
    coverage_rows = 0
    for row in _iter_jsonl(coverage_paths):
        coverage_rows += 1
        source = str(row.get("source") or "unknown")
        status = str(row.get("status") or "unknown")
        ticker = str(row.get("ticker") or "")
        if ticker:
            coverage_ticker_rows[ticker] += 1
        coverage_sources[source] += 1
        coverage_statuses[status] += 1
        saturated_coverage_rows += int(int(row.get("saturated_leaf_windows", 0) or 0) > 0)
        failed_coverage_rows += int(
            int(row.get("request_errors", 0) or 0) > 0 or status in {"failed", "error"}
        )

    coverage_window_statuses: Counter[str] = Counter()
    coverage_window_providers: Counter[str] = Counter()
    coverage_window_tickers: set[str] = set()
    inferred_empty_window_rows = 0
    coverage_window_rows = 0
    for row in _iter_jsonl(coverage_window_paths):
        coverage_window_rows += 1
        status = str(row.get("status") or "unknown")
        provider = str(row.get("provider") or "unknown")
        coverage_window_statuses[status] += 1
        coverage_window_providers[provider] += 1
        coverage_window_tickers.add(str(row.get("ticker") or ""))
        inferred_empty_window_rows += int(bool(row.get("inferred_empty")))

    repeated_occurrences = max(0, occurrence_rows - len(occurrence_article_tickers))
    lexical_duplicate_articles = max(0, clustered_article_rows - cluster_rows)
    search_articles = article_rows - int(article_providers.get("opendart", 0))
    search_title_only = int(content_tiers.get("title_only", 0))
    matched_structured = len(structured_ids & queue_ids)
    total_sensor_rows = accepted_sensor_rows + neutral_rows

    def sensor_outcomes(values: Mapping[str, Counter[str]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                **dict(sorted(counts.items())),
                "acceptance_ratio": _ratio(
                    counts.get("accepted_sensor_rows", 0), counts.get("rows", 0)
                ),
            }
            for key, counts in sorted(values.items())
        }

    return {
        "schema_version": 3,
        "occurrences": {
            "rows": occurrence_rows,
            "unique_occurrence_ids": len(occurrence_ids),
            "duplicate_occurrence_ids": duplicate_occurrence_ids,
            "unique_article_ticker_pairs": len(occurrence_article_tickers),
            "repeated_occurrences": repeated_occurrences,
            "repetition_ratio": _ratio(repeated_occurrences, occurrence_rows),
            "providers": dict(sorted(occurrence_providers.items())),
        },
        "articles": {
            "rows": article_rows,
            "unique_article_ids": len(article_ids),
            "duplicate_article_ids": duplicate_article_ids,
            "canonical_reduction_from_occurrences": max(0, occurrence_rows - article_rows),
            "canonical_reduction_ratio": _ratio(max(0, occurrence_rows - article_rows), occurrence_rows),
            "providers": dict(sorted(article_providers.items())),
            "content_tiers": dict(sorted(content_tiers.items())),
            "search_rows": search_articles,
            "search_title_only_rows": search_title_only,
            "search_title_only_ratio": _ratio(search_title_only, search_articles),
            "point_in_time_selection_rows": point_in_time_selection_rows,
            "retrospective_or_unknown_selection_rows": retrospective_or_unknown_selection_rows,
            "point_in_time_selection_ratio": _ratio(point_in_time_selection_rows, article_rows),
        },
        "clusters": {
            "rows": cluster_rows,
            "unique_cluster_ids": len(cluster_ids),
            "duplicate_cluster_ids": duplicate_cluster_ids,
            "clustered_article_rows": clustered_article_rows,
            "duplicate_articles": lexical_duplicate_articles,
            "duplicate_article_ratio": _ratio(lexical_duplicate_articles, clustered_article_rows),
            "size_buckets": dict(sorted(cluster_size_buckets.items())),
        },
        "mappings": {
            "rows": mapping_rows,
            "ticker_rows": dict(sorted(mapping_ticker_rows.items())),
            "ticker_methods": {
                ticker: dict(sorted(methods.items()))
                for ticker, methods in sorted(mapping_ticker_methods.items())
            },
            "methods": dict(sorted(mapping_methods.items())),
            "relevance_required_rows": relevance_required_rows,
            "query_only_rows": int(mapping_methods.get("source_query_only", 0)),
            "query_only_ratio": _ratio(mapping_methods.get("source_query_only", 0), relevance_required_rows),
            "alias_matched_rows": alias_matched_rows,
            "ambiguous_alias_rows": ambiguous_alias_rows,
            "matched_entity_rows": matched_entity_rows,
            "ambiguous_mapping_rows": ambiguous_mapping_rows,
            "manifest_name_matched_rows": manifest_name_matched_rows,
            "related_entity_alias_rows": related_entity_alias_rows,
        },
        "queue": {
            "rows": queue_rows,
            "unique_queue_ids": len(queue_ids),
            "duplicate_queue_ids": duplicate_queue_ids,
            "tickers": len(queue_tickers - {""}),
            "ticker_rows": dict(sorted(queue_ticker_rows.items())),
            "ticker_methods": {
                ticker: dict(sorted(methods.items()))
                for ticker, methods in sorted(queue_ticker_methods.items())
            },
            "methods": dict(sorted(queue_methods.items())),
            "content_tiers": dict(sorted(queue_tiers.items())),
            "source_cluster_size_buckets": dict(sorted(queue_cluster_buckets.items())),
            "alias_matched_rows": queue_alias_matched_rows,
            "ambiguous_alias_rows": queue_ambiguous_alias_rows,
            "matched_entity_rows": queue_matched_entity_rows,
            "ambiguous_mapping_rows": queue_ambiguous_mapping_rows,
            "manifest_name_matched_rows": queue_manifest_name_matched_rows,
            "point_in_time_selection_rows": queue_point_in_time_rows,
            "point_in_time_selection_ratio": _ratio(queue_point_in_time_rows, queue_rows),
        },
        "structured": {
            "rows": structured_rows,
            "ticker_rows": dict(sorted(structured_ticker_rows.items())),
            "matched_queue_rows": matched_structured,
            "queue_coverage_ratio": _ratio(matched_structured, len(queue_ids)),
            "relevance_buckets": dict(sorted(relevance_buckets.items())),
            "event_specificity_buckets": dict(sorted(event_specificity_buckets.items())),
            "entity_related_rows": entity_related_rows,
            "accepted_sensor_rows": accepted_sensor_rows,
            "acceptance_ratio": _ratio(accepted_sensor_rows, structured_rows),
            "event_types": dict(sorted(structured_event_types.items())),
            "mapping_methods": dict(sorted(structured_mapping_methods.items())),
            "content_tiers": dict(sorted(structured_content_tiers.items())),
            "sensor_outcomes_by_ticker": sensor_outcomes(structured_outcomes_by_ticker),
            "sensor_outcomes_by_mapping_method": sensor_outcomes(
                structured_outcomes_by_mapping_method
            ),
            "invalid_relevance_rows": invalid_relevance_rows,
            "invalid_event_specificity_rows": invalid_event_specificity_rows,
            "low_relevance_with_node_deltas": low_relevance_delta_rows,
            "low_specificity_with_sensor_deltas": low_specificity_delta_rows,
        },
        "neutral_events": {"rows": neutral_rows},
        "sensors": {
            "accepted_directional_or_typed_rows": accepted_sensor_rows,
            "neutral_filing_count_rows": neutral_rows,
            "total_rows": total_sensor_rows,
        },
        "coverage": {
            "rows": coverage_rows,
            "ticker_rows": dict(sorted(coverage_ticker_rows.items())),
            "sources": dict(sorted(coverage_sources.items())),
            "statuses": dict(sorted(coverage_statuses.items())),
            "saturated_rows": saturated_coverage_rows,
            "failed_rows": failed_coverage_rows,
        },
        "coverage_windows": {
            "rows": coverage_window_rows,
            "tickers": len(coverage_window_tickers - {""}),
            "providers": dict(sorted(coverage_window_providers.items())),
            "statuses": dict(sorted(coverage_window_statuses.items())),
            "complete_ratio": _ratio(
                coverage_window_statuses.get("complete", 0), coverage_window_rows
            ),
            "saturated_ratio": _ratio(
                coverage_window_statuses.get("incomplete_saturated", 0), coverage_window_rows
            ),
            "inferred_empty_rows": inferred_empty_window_rows,
        },
    }


def build_ticker_balanced_queue_sample(
    *,
    queue_paths: Sequence[str | Path],
    per_ticker: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a deterministic uniform-within-ticker validation cohort."""

    if per_ticker < 1:
        raise ValueError("per_ticker must be positive")
    seen_queue_ids: set[str] = set()
    populations: Counter[str] = Counter()
    population_methods: dict[str, Counter[str]] = {}
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}

    for row in _iter_jsonl(queue_paths):
        queue_id = str(row.get("queue_id") or "")
        ticker = str(row.get("ticker") or "")
        if not queue_id or queue_id in seen_queue_ids:
            raise ValueError(f"duplicate or missing queue id: {queue_id!r}")
        if not ticker:
            raise ValueError(f"missing ticker for queue id: {queue_id}")
        seen_queue_ids.add(queue_id)
        populations[ticker] += 1
        method = str(row.get("mapping_method") or "unknown")
        population_methods.setdefault(ticker, Counter())[method] += 1
        score = int.from_bytes(
            hashlib.sha256(f"{seed}|ticker-balanced|{ticker}|{queue_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        item = (-score, queue_id, dict(row))
        heap = heaps.setdefault(ticker, [])
        if len(heap) < per_ticker:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    samples = sorted(
        (item[2] for ticker in sorted(heaps) for item in heaps[ticker]),
        key=lambda row: (str(row.get("ticker") or ""), str(row.get("queue_id") or "")),
    )
    sample_counts = Counter(str(row["ticker"]) for row in samples)
    sample_methods: dict[str, Counter[str]] = {}
    for row in samples:
        ticker = str(row["ticker"])
        method = str(row.get("mapping_method") or "unknown")
        sample_methods.setdefault(ticker, Counter())[method] += 1
    report = {
        "schema_version": 1,
        "sampling_policy": "deterministic_uniform_within_ticker_v1",
        "representative_of_global_population": False,
        "per_ticker": per_ticker,
        "seed": seed,
        "population_rows": sum(populations.values()),
        "population_by_ticker": dict(sorted(populations.items())),
        "population_methods_by_ticker": {
            ticker: dict(sorted(methods.items()))
            for ticker, methods in sorted(population_methods.items())
        },
        "sample_rows": len(samples),
        "sample_by_ticker": dict(sorted(sample_counts.items())),
        "sample_methods_by_ticker": {
            ticker: dict(sorted(methods.items()))
            for ticker, methods in sorted(sample_methods.items())
        },
        "selection_probability_by_ticker": {
            ticker: _ratio(sample_counts[ticker], population)
            for ticker, population in sorted(populations.items())
        },
        "inputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in queue_paths
        ],
    }
    return samples, report


def compare_structured_overlap(
    *,
    left_path: str | Path,
    right_path: str | Path,
    example_limit: int = 20,
) -> dict[str, Any]:
    """Compare exact deterministic outputs for queue identities shared by two runs."""

    def load(path: str | Path) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for row in _iter_jsonl([path]):
            queue_id = str(row.get("queue_id") or "")
            if not queue_id or queue_id in rows:
                raise ValueError(f"duplicate or missing queue id in {path}: {queue_id!r}")
            rows[queue_id] = row
        return rows

    left = load(left_path)
    right = load(right_path)
    overlap = sorted(set(left) & set(right))
    input_hash_mismatches = 0
    label_mismatches = 0
    event_mismatches = 0
    lineage_mismatches = 0
    mismatch_fields: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    lineage_keys = (
        "model_id",
        "model_revision",
        "prompt_version",
        "output_schema_version",
        "inference_engine",
    )

    for queue_id in overlap:
        left_row = left[queue_id]
        right_row = right[queue_id]
        input_mismatch = str(left_row.get("input_sha256") or "") != str(
            right_row.get("input_sha256") or ""
        )
        labels_left = left_row.get("labels")
        labels_right = right_row.get("labels")
        event_left = left_row.get("event")
        event_right = right_row.get("event")
        label_mismatch = labels_left != labels_right
        event_mismatch = event_left != event_right
        left_lineage = left_row.get("lineage") if isinstance(left_row.get("lineage"), Mapping) else {}
        right_lineage = (
            right_row.get("lineage") if isinstance(right_row.get("lineage"), Mapping) else {}
        )
        lineage_mismatch = any(
            left_lineage.get(key) != right_lineage.get(key) for key in lineage_keys
        )
        input_hash_mismatches += int(input_mismatch)
        label_mismatches += int(label_mismatch)
        event_mismatches += int(event_mismatch)
        lineage_mismatches += int(lineage_mismatch)
        if isinstance(labels_left, Mapping) and isinstance(labels_right, Mapping):
            for field in sorted(set(labels_left) | set(labels_right)):
                mismatch_fields[field] += int(labels_left.get(field) != labels_right.get(field))
        if (
            (input_mismatch or label_mismatch or event_mismatch or lineage_mismatch)
            and len(examples) < max(0, example_limit)
        ):
            examples.append(
                {
                    "queue_id": queue_id,
                    "input_hash_mismatch": input_mismatch,
                    "label_mismatch": label_mismatch,
                    "event_mismatch": event_mismatch,
                    "lineage_mismatch": lineage_mismatch,
                    "left_labels": labels_left,
                    "right_labels": labels_right,
                }
            )

    return {
        "schema_version": 1,
        "left": {"path": str(left_path), "sha256": _sha256_file(left_path), "rows": len(left)},
        "right": {
            "path": str(right_path),
            "sha256": _sha256_file(right_path),
            "rows": len(right),
        },
        "overlap_rows": len(overlap),
        "left_only_rows": len(set(left) - set(right)),
        "right_only_rows": len(set(right) - set(left)),
        "input_hash_mismatches": input_hash_mismatches,
        "label_mismatches": label_mismatches,
        "event_mismatches": event_mismatches,
        "lineage_mismatches": lineage_mismatches,
        "label_mismatch_fields": {
            field: count for field, count in sorted(mismatch_fields.items()) if count
        },
        "exact_reproduction_ratio": _ratio(len(overlap) - label_mismatches, len(overlap)),
        "mismatch_examples": examples,
    }


def build_relevance_audit_sample(
    *,
    queue_paths: Sequence[str | Path],
    structured_paths: Sequence[str | Path],
    per_stratum: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if per_stratum < 1:
        raise ValueError("per_stratum must be positive")
    queue: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(queue_paths):
        queue_id = str(row.get("queue_id") or "")
        if not queue_id or queue_id in queue:
            raise ValueError(f"duplicate or missing queue id: {queue_id!r}")
        queue[queue_id] = row

    populations: Counter[str] = Counter()
    heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    unmatched_structured_rows = 0
    for structured in _iter_jsonl(structured_paths):
        queue_id = str(structured.get("queue_id") or "")
        source = queue.get(queue_id)
        if source is None:
            unmatched_structured_rows += 1
            continue
        event = structured.get("event") if isinstance(structured.get("event"), Mapping) else {}
        try:
            relevance = float(event.get("relevance"))
        except (TypeError, ValueError, OverflowError):
            continue
        mapping_group = (
            "query_only"
            if str(source.get("mapping_method") or "") == "source_query_only"
            else "direct_evidence"
        )
        decision = "accepted" if relevance >= 0.5 else "rejected"
        stratum = f"{mapping_group}|{decision}"
        populations[stratum] += 1
        row = {
            "audit_id": hashlib.sha256(f"news-relevance-audit-v1|{queue_id}".encode("utf-8")).hexdigest(),
            "stratum": stratum,
            "queue_id": queue_id,
            "ticker": source.get("ticker"),
            "company_name": source.get("company_name"),
            "published_date_kst": source.get("published_date_kst"),
            "mapping_method": source.get("mapping_method"),
            "mapping_confidence": source.get("mapping_confidence"),
            "content_tier": source.get("content_tier"),
            "title": source.get("title"),
            "summary": source.get("summary"),
            "related_titles": source.get("related_titles", []),
            "model_relevance": relevance,
            "model_event_type": event.get("event_type"),
            "model_polarity": event.get("polarity"),
            "model_magnitude": event.get("magnitude"),
            "model_summary": event.get("summary"),
            "human_relevant": None,
            "human_event_type": None,
            "human_notes": "",
        }
        score = int.from_bytes(
            hashlib.sha256(f"{seed}|{stratum}|{queue_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        item = (-score, queue_id, row)
        heap = heaps.setdefault(stratum, [])
        if len(heap) < per_stratum:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    samples = [
        item[2]
        for stratum in sorted(heaps)
        for item in sorted(heaps[stratum], reverse=True)
    ]
    report = {
        "schema_version": 1,
        "queue_rows": len(queue),
        "population_by_stratum": dict(sorted(populations.items())),
        "sample_rows_by_stratum": dict(Counter(row["stratum"] for row in samples)),
        "sample_rows": len(samples),
        "per_stratum": per_stratum,
        "seed": seed,
        "unmatched_structured_rows": unmatched_structured_rows,
    }
    return samples, report


def build_sensor_audit_sample(
    *,
    queue_paths: Sequence[str | Path],
    structured_paths: Sequence[str | Path],
    population_random_size: int = 300,
    per_risk_stratum: int = 20,
    minimum_sample_size: int = 500,
    seed: int = 17,
    audit_role: str = "population_plus_risk",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build an estimable random cohort plus targeted sensor-risk cohorts."""

    if population_random_size < 0:
        raise ValueError("population_random_size must not be negative")
    if per_risk_stratum < 1:
        raise ValueError("per_risk_stratum must be positive")
    if minimum_sample_size < population_random_size:
        raise ValueError("minimum_sample_size must be at least population_random_size")
    if audit_role not in {"population_plus_risk", "targeted_risk_only"}:
        raise ValueError("unsupported audit_role")
    if audit_role == "population_plus_risk" and population_random_size < 1:
        raise ValueError("population audits require a positive random cohort")
    if audit_role == "targeted_risk_only" and population_random_size != 0:
        raise ValueError("targeted risk-only audits must disable the random cohort")

    queue: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(queue_paths):
        queue_id = str(row.get("queue_id") or "")
        if not queue_id or queue_id in queue:
            raise ValueError(f"duplicate or missing queue id: {queue_id!r}")
        queue[queue_id] = row

    global_limit = max(population_random_size, minimum_sample_size)
    global_heap: list[tuple[int, str, dict[str, Any]]] = []
    risk_heaps: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    populations: Counter[str] = Counter()
    unmatched_structured_rows = 0
    invalid_structured_rows = 0
    duplicate_structured_rows = 0
    seen_structured_ids: set[str] = set()

    def offer(
        heap: list[tuple[int, str, dict[str, Any]]],
        limit: int,
        score: int,
        queue_id: str,
        row: dict[str, Any],
    ) -> None:
        item = (-score, queue_id, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)

    for structured in _iter_jsonl(structured_paths):
        queue_id = str(structured.get("queue_id") or "")
        if queue_id in seen_structured_ids:
            duplicate_structured_rows += 1
            continue
        if queue_id:
            seen_structured_ids.add(queue_id)
        source = queue.get(queue_id)
        if source is None:
            unmatched_structured_rows += 1
            continue
        event = structured.get("event") if isinstance(structured.get("event"), Mapping) else {}
        try:
            relevance = float(event.get("relevance"))
            event_specificity = float(event.get("event_specificity"))
        except (TypeError, ValueError, OverflowError):
            invalid_structured_rows += 1
            continue
        if not all(
            math.isfinite(value) and 0.0 <= value <= 1.0
            for value in (relevance, event_specificity)
        ):
            invalid_structured_rows += 1
            continue

        method = str(source.get("mapping_method") or "unknown")
        if method == "source_query_only":
            mapping_group = "query_only"
        elif method.startswith("reviewed_"):
            mapping_group = (
                "reviewed_alias_ambiguous"
                if bool(source.get("matched_alias_ambiguous"))
                else "reviewed_alias"
            )
        else:
            mapping_group = "direct"
        content_group = (
            "title_only" if str(source.get("content_tier") or "") == "title_only" else "richer"
        )
        accepted = relevance >= 0.5 and event_specificity >= 0.5
        model_decision = "accepted" if accepted else "rejected"
        boundary = any(abs(value - 0.5) <= 0.15 for value in (relevance, event_specificity))
        risk_decision = "boundary" if boundary else model_decision
        stratum = f"{mapping_group}|{content_group}|{risk_decision}"
        populations[stratum] += 1
        rejection_reason = (
            "none"
            if accepted
            else "entity_relevance"
            if relevance < 0.5
            else "event_specificity"
        )
        row = {
            "audit_id": hashlib.sha256(
                f"news-sensor-audit-v2|{queue_id}".encode("utf-8")
            ).hexdigest(),
            "queue_id": queue_id,
            "ticker": source.get("ticker"),
            "company_name": source.get("company_name"),
            "published_date_kst": source.get("published_date_kst"),
            "mapping_method": method,
            "mapping_confidence": source.get("mapping_confidence"),
            "matched_alias": source.get("matched_alias"),
            "matched_alias_type": source.get("matched_alias_type"),
            "matched_alias_ambiguous": bool(source.get("matched_alias_ambiguous")),
            "mapping_group": mapping_group,
            "content_tier": source.get("content_tier"),
            "content_group": content_group,
            "title": source.get("title"),
            "summary": source.get("summary"),
            "related_titles": source.get("related_titles", []),
            "model_relevance": relevance,
            "model_event_specificity": event_specificity,
            "model_sensor_accepted": accepted,
            "model_rejection_reason": rejection_reason,
            "model_event_type": event.get("event_type"),
            "model_polarity": event.get("polarity"),
            "model_magnitude": event.get("magnitude"),
            "model_summary": event.get("summary"),
            "risk_stratum": stratum,
            "audit_cohorts": [],
            "human_entity_relevant": None,
            "human_event_specific": None,
            "human_event_type_correct": None,
            "human_direction": None,
            "human_duplicate_group": "",
            "human_notes": "",
        }
        global_score = int.from_bytes(
            hashlib.sha256(f"{seed}|population|{queue_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        risk_score = int.from_bytes(
            hashlib.sha256(f"{seed}|risk|{stratum}|{queue_id}".encode("utf-8")).digest()[:8],
            "big",
        )
        offer(global_heap, global_limit, global_score, queue_id, row)
        offer(risk_heaps.setdefault(stratum, []), per_risk_stratum, risk_score, queue_id, row)

    global_rows = [item[2] for item in sorted(global_heap, reverse=True)]
    selected: dict[str, dict[str, Any]] = {}

    def add(row: dict[str, Any], cohort: str) -> None:
        queue_id = str(row["queue_id"])
        selected_row = selected.setdefault(queue_id, dict(row))
        cohorts = selected_row.setdefault("audit_cohorts", [])
        if cohort not in cohorts:
            cohorts.append(cohort)

    for row in global_rows[:population_random_size]:
        add(row, "population_random")
    for stratum in sorted(risk_heaps):
        for item in sorted(risk_heaps[stratum], reverse=True):
            add(item[2], f"risk:{stratum}")
    for row in global_rows:
        if len(selected) >= minimum_sample_size:
            break
        add(row, "minimum_fill")

    samples = sorted(selected.values(), key=lambda row: str(row["audit_id"]))
    cohort_counts: Counter[str] = Counter(
        cohort for row in samples for cohort in row.get("audit_cohorts", [])
    )
    report = {
        "schema_version": 2,
        "audit_contract": "news_sensor_human_audit_v2",
        "audit_role": audit_role,
        "representative_population_estimate": audit_role == "population_plus_risk",
        "sampling_warning": (
            "Risk-targeted rows are not a population prevalence or accuracy estimate."
            if audit_role == "targeted_risk_only"
            else "Only the population_random cohort supports prevalence-weighted estimates."
        ),
        "queue_rows": len(queue),
        "valid_structured_population": sum(populations.values()),
        "population_by_risk_stratum": dict(sorted(populations.items())),
        "sample_rows": len(samples),
        "unique_sample_rows_by_risk_stratum": dict(
            sorted(Counter(row["risk_stratum"] for row in samples).items())
        ),
        "cohort_memberships": dict(sorted(cohort_counts.items())),
        "population_random_size": population_random_size,
        "per_risk_stratum": per_risk_stratum,
        "minimum_sample_size": minimum_sample_size,
        "seed": seed,
        "unmatched_structured_rows": unmatched_structured_rows,
        "invalid_structured_rows": invalid_structured_rows,
        "duplicate_structured_rows": duplicate_structured_rows,
        "queue_inputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in queue_paths
        ],
        "structured_inputs": [
            {"path": str(path), "sha256": _sha256_file(path)} for path in structured_paths
        ],
    }
    return samples, report
