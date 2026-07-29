from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.dataset_integrity import sha256_file
from stock_v2.news_aliases import validate_alias_registry
from stock_v2.news_dataset import _mapping_evidence


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reviewed alias impact on a news queue.")
    parser.add_argument("--queue", required=True)
    parser.add_argument("--universe", required=True)
    parser.add_argument("--aliases", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--sample-output")
    parser.add_argument("--sample-per-alias", type=int, default=10)
    parser.add_argument("--exclude-sample", action="append", default=[])
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()

    queue_path = Path(args.queue)
    universe_path = Path(args.universe)
    alias_path = Path(args.aliases)
    universe_rows = _load_json(universe_path).get("universe", [])
    universe = {str(row.get("ticker")): row for row in universe_rows if isinstance(row, dict)}
    aliases_by_ticker = validate_alias_registry(_load_json(alias_path), universe)
    excluded_queue_ids: set[str] = set()
    for raw_path in args.exclude_sample:
        path = Path(raw_path)
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, dict) or not str(value.get("queue_id") or ""):
                    raise ValueError(f"invalid excluded sample row at {path}:{line_number}")
                excluded_queue_ids.add(str(value["queue_id"]))

    queue_rows = 0
    query_only_rows = 0
    upgraded_rows = 0
    methods: Counter[str] = Counter()
    aliases: Counter[str] = Counter()
    tickers: Counter[str] = Counter()
    ambiguous_rows = 0
    sample_heaps: defaultdict[str, list[tuple[int, str, dict[str, Any]]]] = defaultdict(list)
    with queue_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {queue_path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"non-object row at {queue_path}:{line_number}")
            queue_rows += 1
            if str(row.get("mapping_method")) != "source_query_only":
                continue
            query_only_rows += 1
            result = _mapping_evidence(
                str(row.get("ticker") or ""),
                str(row.get("company_name") or ""),
                str(row.get("title") or ""),
                str(row.get("summary") or ""),
                aliases_by_ticker.get(str(row.get("ticker") or ""), ()),
                row.get("published_date_kst"),
                str(row.get("source") or ""),
            )
            method, confidence, alias, alias_type, alias_source, ambiguous = result
            if method == "source_query_only":
                continue
            upgraded_rows += 1
            methods[method] += 1
            aliases[str(alias)] += 1
            tickers[str(row.get("ticker") or "")] += 1
            ambiguous_rows += int(ambiguous)
            sample = {
                "schema_version": 3,
                "queue_id": row.get("queue_id"),
                "article_id": row.get("article_id"),
                "event_cluster_id": row.get("event_cluster_id"),
                "cluster_size": row.get("cluster_size"),
                "ticker": row.get("ticker"),
                "company_name": row.get("company_name"),
                "published_date_kst": row.get("published_date_kst"),
                "published_precision": row.get("published_precision"),
                "effective_session": row.get("effective_session"),
                "content_tier": row.get("content_tier"),
                "source": row.get("source"),
                "title": row.get("title"),
                "summary": row.get("summary"),
                "related_titles": row.get("related_titles", []),
                "old_mapping_method": row.get("mapping_method"),
                "new_mapping_method": method,
                "mapping_method": method,
                "mapping_confidence": confidence,
                "matched_alias": alias,
                "matched_alias_type": alias_type,
                "matched_alias_source": alias_source,
                "matched_alias_ambiguous": ambiguous,
                "human_relevant": None,
                "human_notes": "",
            }
            sample["input_sha256"] = hashlib.sha256(
                json.dumps(
                    {
                        "policy": "news-alias-audit-v1",
                        "ticker": sample["ticker"],
                        "title": sample["title"],
                        "summary": sample["summary"],
                        "source": sample["source"],
                        "mapping_method": sample["mapping_method"],
                        "mapping_confidence": sample["mapping_confidence"],
                        "matched_alias": sample["matched_alias"],
                        "matched_alias_type": sample["matched_alias_type"],
                        "matched_alias_ambiguous": sample["matched_alias_ambiguous"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
            if str(sample.get("queue_id") or "") in excluded_queue_ids:
                continue
            sample_key = f"{alias}|{'ambiguous' if ambiguous else 'normal'}"
            digest = int(
                hashlib.sha256(
                    f"{args.seed}|{sample.get('queue_id')}|{sample_key}".encode("utf-8")
                ).hexdigest(),
                16,
            )
            heap = sample_heaps[sample_key]
            item = (-digest, str(sample.get("queue_id") or ""), sample)
            if len(heap) < args.sample_per_alias:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)

    report = {
        "schema_version": 1,
        "policy": "re-evaluate only source_query_only rows with reviewed aliases",
        "queue": {"path": str(queue_path), "sha256": sha256_file(queue_path), "rows": queue_rows},
        "universe": {"path": str(universe_path), "sha256": sha256_file(universe_path)},
        "alias_registry": {
            "path": str(alias_path),
            "sha256": sha256_file(alias_path),
            "rows": sum(len(rows) for rows in aliases_by_ticker.values()),
            "tickers": len(aliases_by_ticker),
        },
        "query_only_rows": query_only_rows,
        "upgraded_rows": upgraded_rows,
        "upgraded_ratio_of_query_only": upgraded_rows / query_only_rows if query_only_rows else 0.0,
        "ambiguous_rows": ambiguous_rows,
        "methods": dict(sorted(methods.items())),
        "aliases": dict(sorted(aliases.items(), key=lambda item: (-item[1], item[0]))),
        "tickers": dict(sorted(tickers.items())),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.sample_output:
        samples = [
            item[2]
            for key in sorted(sample_heaps)
            for item in sorted(sample_heaps[key], reverse=True)
        ]
        sample_path = Path(args.sample_output)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        sample_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in samples),
            encoding="utf-8",
        )
        report["sample"] = {
            "path": str(sample_path),
            "sha256": sha256_file(sample_path),
            "rows": len(samples),
            "per_alias_cap": args.sample_per_alias,
            "seed": args.seed,
            "excluded_queue_ids": len(excluded_queue_ids),
        }
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
