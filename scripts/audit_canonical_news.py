from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.dataset_integrity import IssueLog, audit_news, load_json, normalize_ticker
from stock_v2.news_dataset import filter_universe


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit one canonical news dataset contract.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fail-on-blocker", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    config = load_json(config_path)
    universe_path = ROOT / str(config["universe_manifest"])
    universe_payload = load_json(universe_path)
    full_universe = {
        normalize_ticker(row.get("ticker")): row
        for row in universe_payload.get("universe", [])
        if normalize_ticker(row.get("ticker"))
    }
    universe = filter_universe(full_universe, config.get("include_tickers"))
    issues = IssueLog()
    files: list[dict] = []
    report = audit_news(ROOT, config, universe, issues, files)
    payload = {
        "schema_version": 1,
        "audit_id": config.get("audit_id"),
        "validation_role": config.get("validation_role"),
        "status": "pass" if not any(issue.severity == "blocker" for issue in issues.issues) else "blocked",
        "universe_manifest": str(config["universe_manifest"]),
        "source_universe_tickers": len(full_universe),
        "selected_universe_tickers": len(universe),
        "issue_counts": issues.counts(),
        "issues": [asdict(issue) for issue in issues.issues],
        "report": report,
        "files": files,
    }
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "audit_id": payload["audit_id"],
                "status": payload["status"],
                "issue_counts": payload["issue_counts"],
                "output": str(output_path),
            },
            ensure_ascii=False,
        )
    )
    return 2 if args.fail_on_blocker and payload["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
