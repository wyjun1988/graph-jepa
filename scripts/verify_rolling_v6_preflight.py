from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.rolling_validation import verify_frozen_preflight


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a frozen rolling preflight before model training."
    )
    parser.add_argument("--frozen-contract", required=True)
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frozen = json.loads(Path(args.frozen_contract).read_text(encoding="utf-8"))
    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    payload = verify_frozen_preflight(
        frozen,
        contract,
        Path(args.reports_root),
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
                "status": payload["status"],
                "verified_folds": len(payload["verified_folds"]),
                "source_tree_sha256": payload["source_tree_sha256"],
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
