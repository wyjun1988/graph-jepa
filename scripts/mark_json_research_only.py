from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def mark_research_only(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    if payload.get("live_orders_allowed") is True:
        raise ValueError(f"refusing to rewrite a live-order artifact: {path}")
    payload["live_orders_allowed"] = False
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Atomically mark existing JSON artifacts as research-only."
    )
    parser.add_argument("path", nargs="+", type=Path)
    args = parser.parse_args()

    for path in args.path:
        mark_research_only(path)
        print(path)


if __name__ == "__main__":
    main()
