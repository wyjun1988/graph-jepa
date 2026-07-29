from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def load_jsonl_by_id(path: Path, key: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identity = str(row.get(key) or "")
            if not identity or identity in result:
                raise ValueError(f"missing or duplicate {key} at {path}:{line_number}")
            result[identity] = row
    return result


def _sign(value: float, deadband: float = 0.05) -> int:
    return 1 if value > deadband else -1 if value < -deadband else 0


def _agreement(left: list[Any], right: list[Any]) -> float:
    return float(np.mean([a == b for a, b in zip(left, right)])) if left else 0.0


def _kappa(left: list[Any], right: list[Any]) -> float:
    if not left:
        return 0.0
    observed = _agreement(left, right)
    left_counts = Counter(left)
    right_counts = Counter(right)
    total = len(left)
    expected = sum(left_counts[key] * right_counts[key] for key in set(left_counts) | set(right_counts)) / total**2
    return float((observed - expected) / (1.0 - expected)) if expected < 1.0 else 1.0


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return None
    return float(np.corrcoef(np.asarray(left), np.asarray(right))[0, 1])


def _event(row: Mapping[str, Any]) -> Mapping[str, Any]:
    value = row.get("event")
    return value if isinstance(value, Mapping) else {}


def _score(event: Mapping[str, Any]) -> float:
    return float(event.get("polarity", 0.0)) * float(event.get("magnitude", 0.0)) * float(
        event.get("confidence", 0.0)
    )


def compare_structured_news(
    queue_path: Path,
    candidate_path: Path,
    reference_path: Path,
    thresholds: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    queue = load_jsonl_by_id(queue_path, "queue_id")
    candidate = load_jsonl_by_id(candidate_path, "queue_id")
    reference = load_jsonl_by_id(reference_path, "queue_id")
    common_ids = sorted(set(candidate) & set(reference) & set(queue))
    missing_candidate = sorted(set(reference) - set(candidate))
    missing_reference = sorted(set(candidate) - set(reference))
    candidate_errors = sum(not bool(row.get("llm_used")) or bool(row.get("llm_error")) for row in candidate.values())
    reference_errors = sum(not bool(row.get("llm_used")) or bool(row.get("llm_error")) for row in reference.values())

    candidate_relevance: list[bool] = []
    reference_relevance: list[bool] = []
    candidate_types: list[str] = []
    reference_types: list[str] = []
    candidate_signs: list[int] = []
    reference_signs: list[int] = []
    candidate_scores: list[float] = []
    reference_scores: list[float] = []
    relevance_differences: list[float] = []
    magnitude_differences: list[float] = []
    confidence_differences: list[float] = []
    horizon_log_differences: list[float] = []
    theme_jaccards: list[float] = []
    disagreements: list[dict[str, Any]] = []
    for queue_id in common_ids:
        candidate_event = _event(candidate[queue_id])
        reference_event = _event(reference[queue_id])
        candidate_rel_value = float(candidate_event.get("relevance", 0.0))
        reference_rel_value = float(reference_event.get("relevance", 0.0))
        candidate_rel = candidate_rel_value >= 0.5
        reference_rel = reference_rel_value >= 0.5
        candidate_type = str(candidate_event.get("event_type") or "missing")
        reference_type = str(reference_event.get("event_type") or "missing")
        candidate_score = _score(candidate_event)
        reference_score = _score(reference_event)
        candidate_relevance.append(candidate_rel)
        reference_relevance.append(reference_rel)
        candidate_types.append(candidate_type)
        reference_types.append(reference_type)
        candidate_signs.append(_sign(candidate_score))
        reference_signs.append(_sign(reference_score))
        candidate_scores.append(candidate_score)
        reference_scores.append(reference_score)
        relevance_differences.append(abs(candidate_rel_value - reference_rel_value))
        magnitude_differences.append(
            abs(float(candidate_event.get("magnitude", 0.0)) - float(reference_event.get("magnitude", 0.0)))
        )
        confidence_differences.append(
            abs(float(candidate_event.get("confidence", 0.0)) - float(reference_event.get("confidence", 0.0)))
        )
        candidate_horizon = max(1, int(candidate_event.get("horizon_days", 1)))
        reference_horizon = max(1, int(reference_event.get("horizon_days", 1)))
        horizon_log_differences.append(abs(math.log(candidate_horizon) - math.log(reference_horizon)))
        candidate_themes = set(str(value) for value in candidate_event.get("themes", []) or [])
        reference_themes = set(str(value) for value in reference_event.get("themes", []) or [])
        union = candidate_themes | reference_themes
        theme_jaccards.append(len(candidate_themes & reference_themes) / len(union) if union else 1.0)
        severity = (
            3.0 * int(candidate_rel != reference_rel)
            + 2.0 * int(_sign(candidate_score) != _sign(reference_score))
            + int(candidate_type != reference_type)
            + abs(candidate_score - reference_score)
        )
        if severity > 0:
            disagreements.append(
                {
                    "queue_id": queue_id,
                    "severity": severity,
                    "ticker": queue[queue_id].get("ticker"),
                    "title": queue[queue_id].get("title"),
                    "content_tier": queue[queue_id].get("content_tier"),
                    "candidate": {
                        "relevance": candidate_rel_value,
                        "event_type": candidate_type,
                        "score": candidate_score,
                    },
                    "reference": {
                        "relevance": reference_rel_value,
                        "event_type": reference_type,
                        "score": reference_score,
                    },
                }
            )
    disagreements.sort(key=lambda row: (-float(row["severity"]), str(row["queue_id"])))
    metrics = {
        "common_rows": len(common_ids),
        "candidate_rows": len(candidate),
        "reference_rows": len(reference),
        "candidate_errors": candidate_errors,
        "reference_errors": reference_errors,
        "missing_candidate_rows": len(missing_candidate),
        "missing_reference_rows": len(missing_reference),
        "relevance_agreement": _agreement(candidate_relevance, reference_relevance),
        "relevance_kappa": _kappa(candidate_relevance, reference_relevance),
        "event_type_agreement": _agreement(candidate_types, reference_types),
        "polarity_sign_agreement": _agreement(candidate_signs, reference_signs),
        "score_correlation": _correlation(candidate_scores, reference_scores),
        "relevance_mae": float(np.mean(relevance_differences)) if relevance_differences else None,
        "magnitude_mae": float(np.mean(magnitude_differences)) if magnitude_differences else None,
        "confidence_mae": float(np.mean(confidence_differences)) if confidence_differences else None,
        "horizon_log_mae": float(np.mean(horizon_log_differences)) if horizon_log_differences else None,
        "theme_jaccard_mean": float(np.mean(theme_jaccards)) if theme_jaccards else None,
    }
    gates = {
        "min_rows": 400.0,
        "min_relevance_agreement": 0.90,
        "min_relevance_kappa": 0.75,
        "min_event_type_agreement": 0.75,
        "min_polarity_sign_agreement": 0.85,
        "min_score_correlation": 0.80,
        **dict(thresholds or {}),
    }
    passed = (
        metrics["common_rows"] >= gates["min_rows"]
        and candidate_errors == 0
        and reference_errors == 0
        and not missing_candidate
        and not missing_reference
        and metrics["relevance_agreement"] >= gates["min_relevance_agreement"]
        and metrics["relevance_kappa"] >= gates["min_relevance_kappa"]
        and metrics["event_type_agreement"] >= gates["min_event_type_agreement"]
        and metrics["polarity_sign_agreement"] >= gates["min_polarity_sign_agreement"]
        and metrics["score_correlation"] is not None
        and metrics["score_correlation"] >= gates["min_score_correlation"]
    )
    return {
        "schema_version": 1,
        "candidate": str(candidate_path),
        "reference": str(reference_path),
        "status": "pass" if passed else "fail",
        "metrics": metrics,
        "gates": gates,
        "top_disagreements": disagreements[:100],
    }
