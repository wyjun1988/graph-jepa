from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_quality import build_relevance_audit_sample


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a deterministic high-risk relevance audit sample.")
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--structured", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--queue-subset-output", default=None)
    parser.add_argument("--per-stratum", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260712)
    args = parser.parse_args()
    rows, report = build_relevance_audit_sample(
        queue_paths=args.queue,
        structured_paths=args.structured,
        per_stratum=args.per_stratum,
        seed=args.seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    if args.queue_subset_output:
        selected_ids = {str(row["queue_id"]) for row in rows}
        selected_queue: dict[str, dict] = {}
        for raw_path in args.queue:
            with Path(raw_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    queue_id = str(row.get("queue_id") or "")
                    if queue_id in selected_ids:
                        selected_queue[queue_id] = row
        if set(selected_queue) != selected_ids:
            raise ValueError("failed to recover every selected queue row")
        subset_output = Path(args.queue_subset_output)
        subset_output.parent.mkdir(parents=True, exist_ok=True)
        with subset_output.open("w", encoding="utf-8") as handle:
            for queue_id in sorted(selected_queue):
                handle.write(json.dumps(selected_queue[queue_id], ensure_ascii=False, sort_keys=True) + "\n")
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
