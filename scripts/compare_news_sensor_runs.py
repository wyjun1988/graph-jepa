from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            queue_id = str(row.get("queue_id") or "")
            if not queue_id or queue_id in seen:
                raise ValueError(f"missing or duplicate queue_id at {path}:{line_number}")
            seen.add(queue_id)
            rows.append(row)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bucket(value: float) -> str:
    if value < 0.3:
        return "0.0-0.3"
    if value < 0.5:
        return "0.3-0.5"
    if value < 0.8:
        return "0.5-0.8"
    return "0.8-1.0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare frozen news structuring runs.")
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--queue")
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-limit", type=int, default=30)
    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    candidate_path = Path(args.candidate)
    baseline_rows = _read_jsonl(baseline_path)
    candidate_rows = _read_jsonl(candidate_path)
    baseline = {str(row["queue_id"]): row for row in baseline_rows}
    candidate = {str(row["queue_id"]): row for row in candidate_rows}
    if set(baseline) != set(candidate):
        raise ValueError("baseline and candidate queue identities differ")

    queue: dict[str, dict[str, Any]] = {}
    queue_path = Path(args.queue) if args.queue else None
    if queue_path:
        queue = {str(row["queue_id"]): row for row in _read_jsonl(queue_path)}

    specificity_buckets: Counter[str] = Counter()
    specificity_scores: Counter[str] = Counter()
    relevance_buckets: Counter[str] = Counter()
    decision_changes: list[dict[str, Any]] = []
    low_specificity: list[dict[str, Any]] = []
    invalid_rows = 0
    entity_related_rows = 0
    sensor_accepted_rows = 0

    for queue_id in sorted(candidate):
        old_event = baseline[queue_id].get("event") or {}
        new_event = candidate[queue_id].get("event") or {}
        try:
            old_relevance = float(old_event["relevance"])
            relevance = float(new_event["relevance"])
            specificity = float(new_event["event_specificity"])
        except (KeyError, TypeError, ValueError, OverflowError):
            invalid_rows += 1
            continue
        if not all(math.isfinite(value) for value in (old_relevance, relevance, specificity)):
            invalid_rows += 1
            continue
        source = queue.get(queue_id, {})
        old_accepted = old_relevance >= 0.5
        entity_related = relevance >= 0.5
        sensor_accepted = entity_related and specificity >= 0.5
        entity_related_rows += int(entity_related)
        sensor_accepted_rows += int(sensor_accepted)
        relevance_buckets[_bucket(relevance)] += 1
        specificity_buckets[_bucket(specificity)] += 1
        specificity_scores[f"{specificity:.3f}"] += 1
        common = {
            "queue_id": queue_id,
            "ticker": source.get("ticker", candidate[queue_id].get("ticker")),
            "title": source.get("title"),
            "baseline_relevance": old_relevance,
            "candidate_relevance": relevance,
            "event_specificity": specificity,
            "sensor_accepted": sensor_accepted,
            "event_type": new_event.get("event_type"),
            "summary": new_event.get("summary"),
        }
        if old_accepted != entity_related:
            decision_changes.append(common)
        if entity_related and specificity < 0.5:
            low_specificity.append(common)

    decision_changes.sort(key=lambda row: str(row["queue_id"]))
    low_specificity.sort(
        key=lambda row: (float(row["event_specificity"]), str(row["queue_id"]))
    )
    candidate_lineages = Counter(
        (
            str((row.get("lineage") or {}).get("prompt_version") or "missing"),
            str((row.get("lineage") or {}).get("output_schema_version") or "missing"),
        )
        for row in candidate_rows
    )
    report = {
        "schema_version": 1,
        "rows": len(candidate_rows),
        "invalid_rows": invalid_rows,
        "entity_related_rows": entity_related_rows,
        "sensor_accepted_rows": sensor_accepted_rows,
        "entity_related_but_nonspecific_rows": len(low_specificity),
        "relevance_buckets": dict(sorted(relevance_buckets.items())),
        "event_specificity_buckets": dict(sorted(specificity_buckets.items())),
        "event_specificity_scores": dict(sorted(specificity_scores.items())),
        "entity_relevance_decision_changes": len(decision_changes),
        "decision_changes": decision_changes[: args.example_limit],
        "nonspecific_examples": low_specificity[: args.example_limit],
        "candidate_lineages": [
            {"prompt_version": key[0], "output_schema_version": key[1], "rows": count}
            for key, count in sorted(candidate_lineages.items())
        ],
        "inputs": {
            "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
            "candidate": {"path": str(candidate_path), "sha256": _sha256(candidate_path)},
            "queue": (
                {"path": str(queue_path), "sha256": _sha256(queue_path)} if queue_path else None
            ),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
