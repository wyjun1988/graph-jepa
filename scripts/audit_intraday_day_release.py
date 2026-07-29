from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_day_release_audit import audit_intraday_day_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit an intraday post-impact day release.")
    parser.add_argument("--release-dir", required=True)
    parser.add_argument("--source-release-dir")
    parser.add_argument("--minimum-days", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = audit_intraday_day_release(
            args.release_dir,
            source_release_dir=args.source_release_dir,
            minimum_days=args.minimum_days,
        )
        exit_code = 0
    except Exception as exc:
        report = {
            "schema_version": 1,
            "audit_contract": "intraday_post_impact_day_release_audit_v1",
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "promotion_eligible": False,
            "live_orders_allowed": False,
        }
        exit_code = 1
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "passed" if report["passed"] else "failed",
                "report": str(output),
                "days": report.get("days"),
                "error": report.get("error"),
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
