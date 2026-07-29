from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a frozen ticker universe manifest from a checkpoint.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint_path = Path(args.model_dir) / "graph_jepa_real.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    tickers = list(checkpoint.get("tickers", []))
    names = dict(checkpoint.get("names", {}))
    if not tickers:
        raise ValueError("checkpoint does not contain tickers")
    payload = {
        "schema_version": 1,
        "source_model_dir": str(args.model_dir),
        "universe": [
            {"ticker": str(ticker), "name": str(names.get(ticker, ticker))}
            for ticker in tickers
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output} tickers={len(tickers)}")


if __name__ == "__main__":
    main()
