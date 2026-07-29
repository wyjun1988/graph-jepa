from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_semantic_cluster import semantic_cluster_queue


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Conservatively cluster a frozen news queue by title embeddings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-manifest", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--clusters-output", required=True)
    args = parser.parse_args()
    records = _read_jsonl(Path(args.input))
    embeddings = np.load(args.embeddings, mmap_mode="r")
    manifest = json.loads(Path(args.embedding_manifest).read_text(encoding="utf-8"))
    output, clusters = semantic_cluster_queue(
        records,
        embeddings,
        threshold=args.threshold,
        model_id=str(manifest["model_id"]),
        model_revision=str(manifest["model_revision"]),
        embedding_dimension=int(manifest["embedding_dimension"]),
    )
    _write_jsonl(Path(args.output), output)
    _write_jsonl(Path(args.clusters_output), clusters)
    print(
        json.dumps(
            {
                "input_rows": len(records),
                "output_rows": len(output),
                "merged_rows": len(records) - len(output),
                "multi_article_clusters": sum(int(row["source_count"]) > 1 for row in clusters),
                "threshold": args.threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

