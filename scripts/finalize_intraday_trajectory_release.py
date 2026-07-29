from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.intraday_release_audit import file_sha256
from stock_v2.intraday_release_finalize import finalize_intraday_trajectory_release


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal code provenance and portable paths into a trajectory release."
    )
    parser.add_argument("--release-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    code_files = {
        "trajectory_release_builder": ROOT / "scripts/build_intraday_trajectory_release.py",
        "intraday_sensing_release_reader": ROOT / "scripts/build_intraday_sensing_release.py",
        "intraday_trajectory": ROOT / "stock_v2/intraday_trajectory.py",
        "intraday_sensing": ROOT / "stock_v2/intraday_sensing.py",
        "kiwoom_minute": ROOT / "stock_v2/kiwoom_minute.py",
        "release_finalizer": ROOT / "stock_v2/intraday_release_finalize.py",
    }
    manifest = finalize_intraday_trajectory_release(
        args.release_dir,
        code_files=code_files,
        legacy_path_base=ROOT,
    )
    manifest_path = Path(args.release_dir) / "manifest.json"
    print(
        json.dumps(
            {
                "status": "finalized",
                "manifest": str(manifest_path),
                "manifest_sha256": file_sha256(manifest_path),
                "stocks": manifest["stocks"],
                "portable_output_paths": True,
                "promotion_eligible": False,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
