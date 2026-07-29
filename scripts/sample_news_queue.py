from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.dataset_integrity import sha256_file
from stock_v2.news_quality import build_ticker_balanced_queue_sample


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic ticker-balanced news queue validation sample."
    )
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--per-ticker", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows, report = build_ticker_balanced_queue_sample(
        queue_paths=args.queue,
        per_ticker=args.per_ticker,
        seed=args.seed,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output_path)
    report["output"] = {
        "path": str(output_path),
        "sha256": sha256_file(output_path),
        "rows": len(rows),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
