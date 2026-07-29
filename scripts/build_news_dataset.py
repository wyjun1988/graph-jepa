from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.news_dataset import curate_news_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a causal, deduplicated news structure queue.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--repo-root", default=None)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    repo_root = Path(args.repo_root or Path(__file__).resolve().parents[1]).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo_root / config_path
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    report = curate_news_dataset(
        repo_root,
        json.loads(config_path.read_text(encoding="utf-8")),
        output_dir,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
