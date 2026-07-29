from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.rolling_validation import freeze_preflight_manifests


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze data and edge manifests before rolling v6 training."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    payload = freeze_preflight_manifests(
        contract,
        Path(args.reports_root),
        args.run_name,
        source_root=ROOT,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "folds": len(payload["fold_manifests"]),
                "base_contract_sha256": payload["base_contract_sha256"],
                "source_tree_sha256": payload["source_tree_sha256"],
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
