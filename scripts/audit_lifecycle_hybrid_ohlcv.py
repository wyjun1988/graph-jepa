from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.lifecycle_ohlcv import audit_lifecycle_hybrid_release


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit every file and lifecycle rule in a hybrid OHLCV release."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-tickers", type=int, default=500)
    parser.add_argument("--expected-proxy-tickers", type=int, default=47)
    args = parser.parse_args()
    result = audit_lifecycle_hybrid_release(
        args.manifest,
        expected_tickers=args.expected_tickers,
        expected_proxy_tickers=args.expected_proxy_tickers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "verified_files": result["verified_files"],
                "lifecycle_violations": result["lifecycle_violations"],
            }
        )
    )
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
