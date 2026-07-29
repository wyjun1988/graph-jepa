from __future__ import annotations

from collections import defaultdict
import hashlib
import re
from typing import Any, Mapping, Sequence

import numpy as np

from stock_v2.news_contract import (
    NEWS_QUEUE_INPUT_HASH_POLICY,
    SEMANTIC_NEWS_QUEUE_ID_POLICY,
    expected_news_queue_id,
    news_queue_input_sha256,
)


def _numbers(value: Any) -> set[str]:
    values = set(re.findall(r"(?<![0-9])(?:[0-9]+(?:\.[0-9]+)?)(?![0-9])", str(value or "")))
    return {value for value in values if not (len(value) == 4 and 1900 <= int(float(value)) <= 2099)}


def _event_terms(value: Any) -> set[str]:
    text = str(value or "").lower()
    families = {
        "earnings": r"실적|매출|영업이익|순이익|적자|흑자",
        "contract": r"수주|공급계약|판매계약|계약체결",
        "capital_action": r"배당|자사주|자기주식|소각|분할|병합|감자",
        "financing": r"유상증자|회사채|전환사채|차입|자금조달",
        "m_and_a": r"인수|합병|매각|기업결합",
        "litigation": r"소송|판결|수사|기소|중재",
        "regulatory": r"과징금|제재|허가|승인|규제",
        "labor": r"파업|노조|노사|임금협상",
        "product": r"출시|신제품|개발|특허",
        "supply_chain": r"생산중단|공급중단|물류|원재료",
        "market_move": r"주가|상한가|하한가|급등|급락|신고가|신저가",
    }
    return {family for family, pattern in families.items() if re.search(pattern, text)}


def compatible_titles(left: str, right: str) -> bool:
    left_numbers = _numbers(left)
    right_numbers = _numbers(right)
    if (
        left_numbers
        and right_numbers
        and not (left_numbers <= right_numbers or right_numbers <= left_numbers)
    ):
        return False
    left_events = _event_terms(left)
    right_events = _event_terms(right)
    if left_events and right_events and not (left_events & right_events):
        return False
    return True


def _representative_rank(record: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        1 if record.get("content_tier") == "title_summary" else 0,
        1 if str(record.get("mapping_method") or "").endswith(("title", "summary")) else 0,
        len(str(record.get("summary") or "")),
        str(record.get("queue_id") or ""),
    )


def semantic_cluster_queue(
    records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    threshold: float,
    model_id: str,
    model_revision: str,
    embedding_dimension: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.0 < threshold <= 1.0:
        raise ValueError("semantic clustering threshold must be in (0, 1]")
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(records):
        raise ValueError("embedding rows must align one-to-one with queue records")
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(np.isfinite(matrix)) or not np.allclose(norms, 1.0, rtol=2e-3, atol=2e-3):
        raise ValueError("semantic embeddings must be finite and L2-normalized")

    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(str(record.get("ticker") or ""), str(record.get("published_date_kst") or ""))].append(index)

    output_rows: list[dict[str, Any]] = []
    cluster_rows: list[dict[str, Any]] = []
    for (ticker, published_date), group_indices in sorted(groups.items()):
        ordered = sorted(
            group_indices,
            key=lambda index: (_representative_rank(records[index]), str(records[index].get("queue_id") or "")),
            reverse=True,
        )
        clusters: list[dict[str, Any]] = []
        for index in ordered:
            title = str(records[index].get("title") or "")
            best_cluster: dict[str, Any] | None = None
            best_similarity = -1.0
            for cluster in clusters:
                representative_index = int(cluster["representative_index"])
                representative_title = str(records[representative_index].get("title") or "")
                if not compatible_titles(title, representative_title):
                    continue
                similarity = float(matrix[index] @ matrix[representative_index])
                if similarity >= threshold and similarity > best_similarity:
                    best_cluster = cluster
                    best_similarity = similarity
            if best_cluster is None:
                clusters.append(
                    {
                        "representative_index": index,
                        "members": [index],
                        "similarities": [1.0],
                    }
                )
            else:
                best_cluster["members"].append(index)
                best_cluster["similarities"].append(best_similarity)

        for cluster in clusters:
            representative_index = int(cluster["representative_index"])
            representative = dict(records[representative_index])
            member_indices = sorted(
                cluster["members"],
                key=lambda index: str(records[index].get("queue_id") or ""),
            )
            source_queue_ids = [str(records[index]["queue_id"]) for index in member_indices]
            semantic_cluster_id = hashlib.sha256(
                f"news-semantic-v2|{ticker}|{published_date}|{source_queue_ids[0]}".encode("utf-8")
            ).hexdigest()
            member_titles = []
            for index in member_indices:
                member_title = str(records[index].get("title") or "").strip()
                if member_title and member_title != representative.get("title") and member_title not in member_titles:
                    member_titles.append(member_title)
            output_row = {
                **representative,
                "schema_version": 4,
                "event_cluster_id": semantic_cluster_id,
                "semantic_cluster_id": semantic_cluster_id,
                "semantic_cluster_size": len(member_indices),
                "cluster_size": sum(
                    int(records[index].get("cluster_size", 1) or 1)
                    for index in member_indices
                ),
                "source_queue_ids": source_queue_ids,
                "source_input_sha256": [
                    str(records[index].get("input_sha256") or "")
                    for index in member_indices
                ],
                "related_titles": member_titles[:8],
                "queue_identity_policy": SEMANTIC_NEWS_QUEUE_ID_POLICY,
                "input_hash_policy": NEWS_QUEUE_INPUT_HASH_POLICY,
                "semantic_cluster_lineage": {
                    "model_id": model_id,
                    "model_revision": model_revision,
                    "embedding_dimension": int(embedding_dimension),
                    "similarity": "cosine",
                    "threshold": float(threshold),
                    "constraints": (
                        "same_ticker_same_date+numeric_subset_v2+event_term_compatibility"
                    ),
                },
            }
            output_row["queue_id"] = expected_news_queue_id(output_row)
            output_row["input_sha256"] = news_queue_input_sha256(output_row)
            output_rows.append(output_row)
            cluster_rows.append(
                {
                    "schema_version": 1,
                    "semantic_cluster_id": semantic_cluster_id,
                    "ticker": ticker,
                    "published_date_kst": published_date,
                    "representative_queue_id": str(records[representative_index]["queue_id"]),
                    "source_queue_ids": source_queue_ids,
                    "source_count": len(source_queue_ids),
                    "minimum_similarity_to_representative": float(min(cluster["similarities"])),
                }
            )
    output_rows.sort(key=lambda row: str(row["queue_id"]))
    cluster_rows.sort(key=lambda row: str(row["semantic_cluster_id"]))
    return output_rows, cluster_rows
