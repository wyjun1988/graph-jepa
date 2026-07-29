from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_audit import audit_news_acquisition


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit news acquisition volume, gaps, mappings, and saturation.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-output", required=True)
    parser.add_argument("--sample-size-per-stratum", type=int, default=50)
    parser.add_argument("--sample-seed", type=int, default=20260712)
    args = parser.parse_args()

    repo_root = Path(args.repo_root or ROOT).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    report, samples = audit_news_acquisition(
        repo_root,
        json.loads(config_path.read_text(encoding="utf-8")),
        sample_size_per_stratum=args.sample_size_per_stratum,
        sample_seed=args.sample_seed,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = repo_root / output
    sample_output = Path(args.sample_output)
    if not sample_output.is_absolute():
        sample_output = repo_root / sample_output
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with sample_output.open("w", encoding="utf-8") as handle:
        for row in samples:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["blocker_count"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())

