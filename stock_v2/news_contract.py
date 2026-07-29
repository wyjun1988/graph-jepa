from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence


LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY = "news-structure-input-v1"
NEWS_QUEUE_INPUT_HASH_POLICY = "news-structure-input-v2"
SUPPORTED_NEWS_QUEUE_INPUT_HASH_POLICIES = {
    LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY,
    NEWS_QUEUE_INPUT_HASH_POLICY,
}
BASE_NEWS_QUEUE_ID_POLICY = "news-structure-v3"
SEMANTIC_NEWS_QUEUE_ID_POLICY = "news-semantic-structure-v2"
SUPPORTED_NEWS_QUEUE_ID_POLICIES = {
    BASE_NEWS_QUEUE_ID_POLICY,
    SEMANTIC_NEWS_QUEUE_ID_POLICY,
}


def news_queue_id(event_cluster_id: Any, ticker: Any) -> str:
    identity = f"{BASE_NEWS_QUEUE_ID_POLICY}|{event_cluster_id}|{ticker}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def expected_news_queue_id(row: Mapping[str, Any]) -> str:
    inferred_policy = (
        SEMANTIC_NEWS_QUEUE_ID_POLICY
        if row.get("semantic_cluster_id")
        else BASE_NEWS_QUEUE_ID_POLICY
    )
    policy = str(row.get("queue_identity_policy") or inferred_policy)
    if policy not in SUPPORTED_NEWS_QUEUE_ID_POLICIES:
        raise ValueError(f"unsupported news queue identity policy: {policy}")
    cluster_id = (
        row.get("semantic_cluster_id")
        if policy == SEMANTIC_NEWS_QUEUE_ID_POLICY
        else row.get("event_cluster_id")
    )
    identity = f"{policy}|{cluster_id}|{row.get('ticker')}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def news_queue_input_payload(
    row: Mapping[str, Any],
    *,
    policy: str | None = None,
) -> dict[str, Any]:
    """Return the frozen fields that define one news structuring request."""

    resolved_policy = str(
        policy or row.get("input_hash_policy") or LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY
    )
    if resolved_policy not in SUPPORTED_NEWS_QUEUE_INPUT_HASH_POLICIES:
        raise ValueError(f"unsupported news queue input hash policy: {resolved_policy}")
    acquisition_modes = row.get("acquisition_modes")
    if not isinstance(acquisition_modes, Mapping):
        raise ValueError("news queue acquisition_modes must be an object")
    legacy_payload = {
        "ticker": row.get("ticker"),
        "event_cluster_id": row.get("event_cluster_id"),
        "title": row.get("title"),
        "summary": row.get("summary"),
        "source": row.get("source"),
        "published_date_kst": row.get("published_date_kst"),
        "content_tier": row.get("content_tier"),
        "acquisition_modes": dict(sorted(acquisition_modes.items())),
        "selection_point_in_time": bool(row.get("selection_point_in_time")),
        "mapping_method": row.get("mapping_method"),
        "mapping_confidence": row.get("mapping_confidence"),
        "matched_alias": row.get("matched_alias"),
        "matched_alias_type": row.get("matched_alias_type"),
        "matched_alias_source": row.get("matched_alias_source"),
        "matched_alias_ambiguous": bool(row.get("matched_alias_ambiguous", False)),
    }
    if resolved_policy == LEGACY_NEWS_QUEUE_INPUT_HASH_POLICY:
        return legacy_payload
    related_raw = row.get("related_titles", [])
    if not isinstance(related_raw, Sequence) or isinstance(related_raw, (str, bytes)):
        raise ValueError("news queue related_titles must be an array")
    source_queue_ids = row.get("source_queue_ids", [])
    source_input_sha256 = row.get("source_input_sha256", [])
    semantic_lineage = row.get("semantic_cluster_lineage", {})
    if not isinstance(source_queue_ids, Sequence) or isinstance(
        source_queue_ids, (str, bytes)
    ):
        raise ValueError("news queue source_queue_ids must be an array")
    if not isinstance(source_input_sha256, Sequence) or isinstance(
        source_input_sha256, (str, bytes)
    ):
        raise ValueError("news queue source_input_sha256 must be an array")
    if not isinstance(semantic_lineage, Mapping):
        raise ValueError("news queue semantic_cluster_lineage must be an object")
    return {
        "input_hash_policy": resolved_policy,
        "queue_identity_policy": row.get(
            "queue_identity_policy", BASE_NEWS_QUEUE_ID_POLICY
        ),
        "article_id": row.get("article_id"),
        "event_cluster_id": row.get("event_cluster_id"),
        "cluster_size": row.get("cluster_size"),
        "ticker": row.get("ticker"),
        "company_name": row.get("company_name"),
        "mapping_method": row.get("mapping_method"),
        "mapping_confidence": row.get("mapping_confidence"),
        "matched_alias": row.get("matched_alias"),
        "matched_alias_type": row.get("matched_alias_type"),
        "matched_alias_source": row.get("matched_alias_source"),
        "matched_alias_ambiguous": bool(row.get("matched_alias_ambiguous", False)),
        "title": row.get("title"),
        "summary": row.get("summary"),
        "related_titles": list(related_raw),
        "source": row.get("source"),
        "published_date_kst": row.get("published_date_kst"),
        "published_precision": row.get("published_precision"),
        "effective_session": row.get("effective_session"),
        "content_tier": row.get("content_tier"),
        "acquisition_modes": dict(sorted(acquisition_modes.items())),
        "selection_point_in_time": bool(row.get("selection_point_in_time")),
        "semantic_cluster_id": row.get("semantic_cluster_id"),
        "semantic_cluster_size": row.get("semantic_cluster_size"),
        "source_queue_ids": list(source_queue_ids),
        "source_input_sha256": list(source_input_sha256),
        "semantic_cluster_lineage": dict(semantic_lineage),
    }


def news_queue_input_sha256(row: Mapping[str, Any], *, policy: str | None = None) -> str:
    payload = json.dumps(
        news_queue_input_payload(row, policy=policy),
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
