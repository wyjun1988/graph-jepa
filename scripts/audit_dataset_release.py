from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.dataset_integrity import audit_dataset_release, render_dataset_card


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit and fingerprint a frozen dataset release.")
    parser.add_argument("--config", required=True, help="Dataset release contract JSON.")
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fail-on-blocker", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = audit_dataset_release(repo_root, config)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "reports" / "dataset_releases" / config["release_id"]
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "release_id": report["release_id"],
                "fingerprint_sha256": report["fingerprint_sha256"],
                "sources": report["sources"],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "DATASET_CARD.md").write_text(render_dataset_card(report), encoding="utf-8")

    print(
        json.dumps(
            {
                "release_id": report["release_id"],
                "status": report["status"],
                "fingerprint_sha256": report["fingerprint_sha256"],
                "issue_counts": report["issue_counts"],
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    return 2 if args.fail_on_blocker and report["status"] != "pass" else 0


if __name__ == "__main__":
    sys.exit(main())
