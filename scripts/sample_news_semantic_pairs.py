from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import heapq
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_semantic_cluster import compatible_titles


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    parser = argparse.ArgumentParser(description="Sample same-day news-title pairs across cosine bands.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--sample-per-band", type=int, default=40)
    parser.add_argument(
        "--max-per-ticker-band",
        type=int,
        default=0,
        help="When positive, cap each ticker's candidates before the band-level sample.",
    )
    parser.add_argument("--minimum-similarity", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    records = read_jsonl(Path(args.input))
    embeddings = np.asarray(np.load(args.embeddings, mmap_mode="r"), dtype=np.float32)
    if len(records) != len(embeddings):
        raise ValueError("queue and embedding row counts differ")
    groups: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(records):
        groups[(str(row["ticker"]), str(row["published_date_kst"]))].append(index)
    boundaries = [0.75, 0.80, 0.85, 0.90, 0.925, 0.95, 0.975, 1.000001]
    heaps: defaultdict[str, list[tuple[int, str, dict]]] = defaultdict(list)
    ticker_heaps: defaultdict[tuple[str, str], list[tuple[int, str, dict]]] = defaultdict(list)
    counts: Counter[str] = Counter()

    def band_name(similarity: float) -> str:
        for left, right in zip(boundaries, boundaries[1:]):
            if left <= similarity < right:
                return f"{left:.3f}-{min(right, 1.0):.3f}"
        return "outside"

    for (ticker, date), indices in sorted(groups.items()):
        if len(indices) < 2:
            continue
        local = embeddings[indices]
        similarities = local @ local.T
        left_positions, right_positions = np.where(
            np.triu(similarities >= args.minimum_similarity, k=1)
        )
        for left_position, right_position in zip(left_positions.tolist(), right_positions.tolist()):
            left_index = indices[left_position]
            right_index = indices[right_position]
            similarity = float(similarities[left_position, right_position])
            band = band_name(similarity)
            counts[band] += 1
            left = records[left_index]
            right = records[right_index]
            left_id = str(left["queue_id"])
            right_id = str(right["queue_id"])
            identity = "|".join(sorted((left_id, right_id)))
            score = int.from_bytes(
                hashlib.sha256(f"{args.seed}|{band}|{identity}".encode("utf-8")).digest()[:8],
                "big",
            )
            row = {
                "similarity_band": band,
                "cosine_similarity": similarity,
                "ticker": ticker,
                "published_date_kst": date,
                "left_queue_id": left_id,
                "right_queue_id": right_id,
                "left_title": left.get("title"),
                "right_title": right.get("title"),
                "rule_compatible": compatible_titles(str(left.get("title") or ""), str(right.get("title") or "")),
                "human_same_event": None,
                "human_notes": "",
            }
            item = (-score, identity, row)
            heap_key = (band, ticker)
            heap = ticker_heaps[heap_key] if args.max_per_ticker_band > 0 else heaps[band]
            heap_limit = args.max_per_ticker_band if args.max_per_ticker_band > 0 else args.sample_per_band
            if len(heap) < heap_limit:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
    if args.max_per_ticker_band > 0:
        for (band, _ticker), ticker_heap in sorted(ticker_heaps.items()):
            for item in ticker_heap:
                heap = heaps[band]
                if len(heap) < args.sample_per_band:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    samples = [
        item[2]
        for band in sorted(heaps)
        for item in sorted(heaps[band], reverse=True)
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": 1,
        "input_rows": len(records),
        "groups": len(groups),
        "minimum_similarity": args.minimum_similarity,
        "candidate_pairs_by_band": dict(sorted(counts.items())),
        "sample_rows_by_band": dict(Counter(row["similarity_band"] for row in samples)),
        "sample_tickers_by_band": {
            band: len({row["ticker"] for row in samples if row["similarity_band"] == band})
            for band in sorted(heaps)
        },
        "max_per_ticker_band": args.max_per_ticker_band,
        "sample_seed": args.seed,
    }
    Path(args.report_output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
