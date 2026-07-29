from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_quality import build_news_quality_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile noise and usable sensor yield in canonical news data.")
    parser.add_argument("--occurrences", action="append", default=[])
    parser.add_argument("--articles", action="append", default=[])
    parser.add_argument("--clusters", action="append", default=[])
    parser.add_argument("--mappings", action="append", default=[])
    parser.add_argument("--queue", action="append", default=[])
    parser.add_argument("--structured", action="append", default=[])
    parser.add_argument("--neutral-events", action="append", default=[])
    parser.add_argument("--coverage", action="append", default=[])
    parser.add_argument("--coverage-windows", action="append", default=[])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = build_news_quality_report(
        occurrence_paths=args.occurrences,
        article_paths=args.articles,
        cluster_paths=args.clusters,
        mapping_paths=args.mappings,
        queue_paths=args.queue,
        structured_paths=args.structured,
        neutral_event_paths=args.neutral_events,
        coverage_paths=args.coverage,
        coverage_window_paths=args.coverage_windows,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
