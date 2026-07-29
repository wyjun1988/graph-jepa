from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


ZERO_TOP_LEVEL_FIELDS = ("paper_initial_cash", "target_weight")
ZERO_RISK_FIELDS = (
    "max_new_buys_per_run",
    "max_orders_per_day",
    "max_cash_per_order",
    "max_position_pct_equity",
    "max_total_exposure_pct",
)
SENSITIVE_KEY_PARTS = (
    "account_number",
    "api_key",
    "api_secret",
    "access_token",
    "refresh_token",
    "password",
)


def _is_zero(value: Any) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def _embedded_secret_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key)
            path = f"{prefix}.{key}" if prefix else key
            normalized = key.lower()
            if any(part in normalized for part in SENSITIVE_KEY_PARTS):
                if child not in (None, "", [], {}):
                    paths.append(path)
            paths.extend(_embedded_secret_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            paths.extend(_embedded_secret_paths(child, f"{prefix}[{index}]"))
    return paths


def evaluate_shadow_safety(config: Mapping[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, value: Any, requirement: str) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "value": value,
                "requirement": requirement,
            }
        )

    mode = config.get("mode")
    add("read_only_mode", mode == "dry_live", mode, "mode=dry_live")

    for field in ZERO_TOP_LEVEL_FIELDS:
        value = config.get(field)
        add(f"zero_{field}", _is_zero(value), value, f"{field}=0")

    risk = config.get("risk")
    if not isinstance(risk, Mapping):
        risk = {}
    for field in ZERO_RISK_FIELDS:
        value = risk.get(field)
        add(f"zero_risk_{field}", _is_zero(value), value, f"risk.{field}=0")

    embedded_secrets = sorted(set(_embedded_secret_paths(config)))
    add(
        "no_embedded_credentials",
        not embedded_secrets,
        embedded_secrets,
        "credentials must be referenced through an env file, not embedded",
    )

    failed = [row for row in checks if not row["passed"]]
    return {
        "status": "pass" if not failed else "blocked",
        "role": "read_only_shadow_safety_audit",
        "approval_scope": "read_only_shadow" if not failed else "none",
        "live_orders_allowed": False,
        "checks": checks,
        "failed_checks": failed,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Shadow Safety Audit",
        "",
        f"Status: **{payload['status']}**",
        f"Approval scope: `{payload['approval_scope']}`",
        "Live orders allowed: `false`",
        "",
        f"Checks: {payload['summary']['passed']} passed, "
        f"{payload['summary']['failed']} failed.",
    ]
    if payload["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        for row in payload["failed_checks"]:
            lines.append(
                f"- `{row['id']}`: value={row['value']}; requires {row['requirement']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit an operations config for strictly read-only shadow use."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    raw = config_path.read_bytes()
    config = json.loads(raw)
    payload = evaluate_shadow_safety(config)
    payload["config_path"] = str(config_path)
    payload["config_sha256"] = hashlib.sha256(raw).hexdigest()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "safety.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "safety.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], sort_keys=True))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
