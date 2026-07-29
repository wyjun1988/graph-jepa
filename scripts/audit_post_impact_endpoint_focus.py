from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.audit_post_impact_multifold_increment import (
    evaluate as evaluate_multifold,
    safe_json,
    sha256_file,
)


EXPERIMENT_SUBROLE = "post_impact_endpoint_focus_screen_v1"


def _validate_mode_payload(
    checkpoint_args: Mapping[str, Any],
    summary: Mapping[str, Any],
    clock: Mapping[str, Any],
    spec: Mapping[str, Any],
    objective: Mapping[str, Any],
) -> None:
    mode = str(spec["daily_context_placebo_mode"])
    variant = str(spec["variant"])
    if mode not in {"none", "latent_only"}:
        raise ValueError(f"unsupported endpoint-focus context mode: {mode}")
    if mode == "latent_only" and variant != "latent":
        raise ValueError("latent-only endpoint control must use the latent variant")
    for payload, role in (
        (checkpoint_args, "checkpoint"),
        (summary, "summary"),
        (clock, "clock"),
    ):
        if str(payload.get("daily_context_placebo_mode", "none")) != mode:
            raise ValueError(f"endpoint-focus context mode mismatch: {role}")
        if str(payload.get("variant")) != variant:
            raise ValueError(f"endpoint-focus model variant mismatch: {role}")
    if bool(checkpoint_args.get("shuffle_daily_context")):
        raise ValueError("endpoint-focus controls must not shuffle stale raw state")
    if str(checkpoint_args.get("graph_message_mode", "none")) != "none":
        raise ValueError("endpoint-focus screen must not use graph messages")
    if not bool(checkpoint_args.get("disable_stale_graph")):
        raise ValueError("endpoint-focus screen must disable scalar graph coherence")
    for name in ("endpoint_return_weight", "close_horizon_weight"):
        expected = float(objective[name])
        if not np.isclose(
            float(checkpoint_args.get(name)), expected, rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"endpoint-focus objective mismatch: {name}")
    summary_objective = summary.get("objective_weights")
    if not isinstance(summary_objective, Mapping):
        raise ValueError("endpoint-focus summary objective is missing")
    if not np.isclose(
        float(summary_objective.get("endpoint_return")),
        float(objective["endpoint_return_weight"]),
        rtol=0.0,
        atol=1e-12,
    ) or not np.isclose(
        float(summary_objective.get("close_horizon")),
        float(objective["close_horizon_weight"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("endpoint-focus summary objective mismatch")
    state_audit = summary.get("context_map_audit")
    latent_audit = summary.get("latent_context_map_audit")
    if not isinstance(state_audit, Mapping) or not isinstance(latent_audit, Mapping):
        raise ValueError("endpoint-focus context audits are missing")
    if int(state_audit.get("future_context_violations", -1)) != 0 or int(
        latent_audit.get("future_context_violations", -1)
    ) != 0:
        raise ValueError("endpoint-focus context is non-causal")
    if int(state_audit.get("same_target_date_count", -1)) != int(
        state_audit.get("dates", -2)
    ):
        raise ValueError("endpoint-focus raw state was not held fixed")
    latent_same = int(latent_audit.get("same_target_date_count", -1))
    latent_dates = int(latent_audit.get("dates", -2))
    if mode == "latent_only" and not 0 <= latent_same < latent_dates:
        raise ValueError("latent-only placebo did not replace latent context")
    if mode == "none" and latent_same != latent_dates:
        raise ValueError("aligned latent context was unexpectedly replaced")


def _validate_endpoint_focus_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("experiment_subrole") != EXPERIMENT_SUBROLE:
        raise ValueError("invalid endpoint-focus experiment subrole")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("unsafe endpoint-focus contract")
    objective = contract.get("objective")
    if not isinstance(objective, Mapping):
        raise ValueError("endpoint-focus objective contract is missing")
    expected_models = {str(contract["actual_model"]), *map(str, contract["comparators"])}
    for fold_spec in contract["folds"]:
        models = fold_spec["models"]
        if set(models) != expected_models:
            raise ValueError("endpoint-focus model set mismatch")
        for name, spec in models.items():
            root = Path(spec["training_dir"])
            checkpoint = torch.load(
                root / "post_impact_reforecast.pt",
                map_location="cpu",
                weights_only=False,
            )
            summary = safe_json(root / "summary.json", f"{name} summary")
            clock = safe_json(Path(spec["clock_report"]), f"{name} clock")
            _validate_mode_payload(
                checkpoint.get("args", {}), summary, clock, spec, objective
            )


def evaluate(contract_path: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    _validate_endpoint_focus_contract(contract)
    summary, daily = evaluate_multifold(contract_path)
    passed = all(bool(value) for value in summary["checks"].values())
    summary["base_audit_role"] = summary["role"]
    summary["role"] = "post_impact_endpoint_focus_screen_audit"
    summary["experiment_subrole"] = EXPERIMENT_SUBROLE
    summary["objective"] = contract["objective"]
    summary["decision"] = (
        "endpoint_focus_screen_passed_research_only"
        if passed
        else "endpoint_focus_screen_not_confirmed"
    )
    summary["next_gate"] = (
        "nonoverlapping_multifold_endpoint_focus_validation"
        if passed
        else "do_not_scale_endpoint_focus_candidate"
    )
    summary["promotion_eligible"] = False
    summary["live_orders_allowed"] = False
    return summary, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit endpoint-focused post-impact JEPA increment."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    summary, daily = evaluate(Path(args.contract))
    output_dir.mkdir(parents=True)
    daily.to_csv(output_dir / "daily_paired_deltas.csv", index=False)
    summary["daily_paired_deltas_sha256"] = sha256_file(
        output_dir / "daily_paired_deltas.csv"
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "decision": summary["decision"],
                "checks_passed": summary["checks_passed"],
                "checks_total": summary["checks_total"],
                "promotion_eligible": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
