from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_window_coverage import reconstruct_news_window_coverage


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconstruct per-window coverage from raw search acquisitions.")
    parser.add_argument("--raw", action="append", required=True)
    parser.add_argument("--coverage", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--allow-issues", action="store_true")
    args = parser.parse_args()
    rows, report = reconstruct_news_window_coverage(
        raw_paths=args.raw,
        coverage_paths=args.coverage,
    )
    _write_jsonl(Path(args.output), rows)
    report_output = Path(args.report_output)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.allow_issues or not report["issue_count"] else 2


if __name__ == "__main__":
    sys.exit(main())
