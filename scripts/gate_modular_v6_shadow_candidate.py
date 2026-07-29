from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from stock_v2.modular_shadow_gate import evaluate_modular_shadow_gate


def _labeled_paths(values: Sequence[str], role: str) -> dict[str, Path]:
    result = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError(f"{role} values must use LABEL=PATH")
        if label in result:
            raise ValueError(f"duplicate {role} label: {label}")
        result[label] = Path(raw_path)
    return result


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Modular Graph-JEPA Shadow Qualification",
        "",
        f"- Status: `{payload['status']}`",
        f"- Approval scope: `{payload['approval_scope']}`",
        f"- Checks: `{payload['summary']['passed']}/{payload['summary']['total']}`",
        f"- Shadow candidates: `{payload['shadow_candidate_count']}`",
        "- Live orders allowed: `false`",
        "",
        "## Modules",
        "",
    ]
    for name, module in payload["modules"].items():
        lines.append(
            f"- `{name}`: `{module['status']}` "
            f"({module['passed']}/{module['total']})"
        )
    lines.extend(["", "## Failed Checks", ""])
    failed = [row for row in payload["checks"] if not row["passed"]]
    if failed:
        lines.extend(
            f"- `{row['module']}.{row['id']}`: {row['requirement']}"
            for row in failed
        )
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qualify the modular JEPA-state, Qlib-return, systemic-impact stack."
    )
    parser.add_argument("--legacy-gate", required=True)
    parser.add_argument("--state-parity", required=True)
    parser.add_argument("--qlib-summary", action="append", required=True)
    parser.add_argument("--qlib-daily", action="append", required=True)
    parser.add_argument("--impact-comparison", required=True)
    parser.add_argument("--impact-stability", required=True)
    parser.add_argument("--modular-latency")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    summaries = _labeled_paths(args.qlib_summary, "Qlib summary")
    daily = _labeled_paths(args.qlib_daily, "Qlib daily")
    modular_latency = (
        _load_json(args.modular_latency) if args.modular_latency else None
    )
    payload = evaluate_modular_shadow_gate(
        _load_json(args.legacy_gate),
        _load_json(args.state_parity),
        {label: _load_json(path) for label, path in summaries.items()},
        {label: pd.read_csv(path) for label, path in daily.items()},
        _load_json(args.impact_comparison),
        _load_json(args.impact_stability),
        modular_latency,
    )
    input_paths = {
        "legacy_gate": Path(args.legacy_gate),
        "state_parity": Path(args.state_parity),
        "impact_comparison": Path(args.impact_comparison),
        "impact_stability": Path(args.impact_stability),
        **{f"qlib_summary_{label}": path for label, path in summaries.items()},
        **{f"qlib_daily_{label}": path for label, path in daily.items()},
    }
    if args.modular_latency:
        input_paths["modular_latency"] = Path(args.modular_latency)
    payload["input_sha256"] = {
        name: _sha256(path) for name, path in sorted(input_paths.items())
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "gate.md").write_text(
        _render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"]))
    raise SystemExit(0 if payload["status"] == "pass" else 1)


if __name__ == "__main__":
    main()
