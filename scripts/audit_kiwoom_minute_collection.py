from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute_collection_audit import (
    audit_kiwoom_minute_collection,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a Kiwoom minute collection through immutable raw pages."
    )
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--repository-root", default=str(ROOT))
    parser.add_argument("--raw-cache-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--interval-minutes", type=int, required=True)
    parser.add_argument("--basis", choices=["raw", "adjusted"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit_kiwoom_minute_collection(
        coverage_path=Path(args.coverage),
        universe_manifest_path=Path(args.universe_manifest),
        repository_root=Path(args.repository_root),
        raw_cache_dir=Path(args.raw_cache_dir),
        run_id=args.run_id,
        requested_start=args.start,
        requested_end=args.end,
        interval_minutes=args.interval_minutes,
        basis=args.basis,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["integrity_gate_passed"] else "failed",
                "coverage_records": report["coverage_records"],
                "output_files_verified": report["output_files_verified"],
                "raw_pages_verified": report["raw_pages_verified"],
                "failures": len(report["failures"]),
                "output": str(output),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if report["integrity_gate_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
