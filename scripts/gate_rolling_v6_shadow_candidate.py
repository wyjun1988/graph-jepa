from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from stock_v2.rolling_gate import evaluate_rolling_gate


def labeled_paths(values: Sequence[str], role: str) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{role} values must use LABEL=PATH")
        if label in result:
            raise ValueError(f"duplicate {role} label: {label}")
        result[label] = Path(raw_path)
    return result


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_markdown(payload: dict[str, Any]) -> str:
    failed = [row for row in payload["checks"] if not row["passed"]]
    aggregate = payload["aggregate_top300_h10"]
    lines = [
        "# Rolling v6 Shadow Qualification",
        "",
        f"- Status: `{payload['status']}`",
        f"- Approval scope: `{payload['approval_scope']}`",
        f"- Folds: `{payload['folds']}`",
        f"- Checks: `{payload['summary']['passed']}/{payload['summary']['total']}`",
        f"- Aggregate top300 h10 IC: `{aggregate['mean']:.6f}`",
        f"- Aggregate top300 h10 Newey-West t: `{aggregate['newey_west_t']:.3f}`",
        "- Live orders allowed: `false`",
        "",
        "## Failed Checks",
        "",
    ]
    if not failed:
        lines.append("None.")
    else:
        lines.extend(
            f"- `{row['id']}` ({row.get('fold', 'aggregate')}, h={row.get('horizon', '-')})"
            for row in failed
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply the predeclared five-fold Graph-JEPA shadow gate."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--frozen-verification", required=True)
    parser.add_argument("--node-summary", action="append", required=True)
    parser.add_argument("--direct-comparison", action="append", required=True)
    parser.add_argument("--head-summary", action="append", required=True)
    parser.add_argument("--head-daily", action="append", required=True)
    parser.add_argument("--qlib-comparison", action="append", required=True)
    parser.add_argument("--latency", required=True)
    parser.add_argument("--safety", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--ohlcv-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    nodes = labeled_paths(args.node_summary, "node summary")
    direct = labeled_paths(args.direct_comparison, "direct comparison")
    heads = labeled_paths(args.head_summary, "head summary")
    daily = labeled_paths(args.head_daily, "head daily")
    qlib = labeled_paths(args.qlib_comparison, "Qlib comparison")
    payload = evaluate_rolling_gate(
        load_json(args.contract),
        load_json(args.frozen_verification),
        {label: load_json(path) for label, path in nodes.items()},
        {label: load_json(path) for label, path in direct.items()},
        {label: load_json(path) for label, path in heads.items()},
        {label: pd.read_csv(path) for label, path in daily.items()},
        {label: load_json(path) for label, path in qlib.items()},
        load_json(args.latency),
        load_json(args.safety),
        load_json(args.dataset_audit),
        load_json(args.ohlcv_audit),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"]))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
