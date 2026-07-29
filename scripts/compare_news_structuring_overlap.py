from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_quality import compare_structured_overlap


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check exact reproducibility on queue IDs shared by two structuring runs."
    )
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--example-limit", type=int, default=20)
    parser.add_argument("--fail-on-mismatch", action="store_true")
    args = parser.parse_args()

    report = compare_structured_overlap(
        left_path=args.left,
        right_path=args.right,
        example_limit=args.example_limit,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    mismatch_count = sum(
        int(report[key])
        for key in (
            "input_hash_mismatches",
            "label_mismatches",
            "event_mismatches",
            "lineage_mismatches",
        )
    )
    return 2 if args.fail_on_mismatch and (not report["overlap_rows"] or mismatch_count) else 0


if __name__ == "__main__":
    raise SystemExit(main())
