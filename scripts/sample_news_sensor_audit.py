from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_quality import build_sensor_audit_sample


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create deterministic population and risk-stratified news sensor audits."
    )
    parser.add_argument("--queue", action="append", required=True)
    parser.add_argument("--structured", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-output", required=True)
    parser.add_argument("--queue-subset-output")
    parser.add_argument("--population-random-size", type=int, default=300)
    parser.add_argument("--per-risk-stratum", type=int, default=20)
    parser.add_argument("--minimum-sample-size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument(
        "--audit-role",
        choices=["population_plus_risk", "targeted_risk_only"],
        default="population_plus_risk",
    )
    args = parser.parse_args()

    rows, report = build_sensor_audit_sample(
        queue_paths=args.queue,
        structured_paths=args.structured,
        population_random_size=args.population_random_size,
        per_risk_stratum=args.per_risk_stratum,
        minimum_sample_size=args.minimum_sample_size,
        seed=args.seed,
        audit_role=args.audit_role,
    )
    _write_jsonl(Path(args.output), rows)

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
        _write_jsonl(
            Path(args.queue_subset_output),
            [selected_queue[queue_id] for queue_id in sorted(selected_queue)],
        )

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
