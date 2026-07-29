from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_market_transition_heads import paired_rows, paired_summary


def parse_fold(value: str) -> tuple[str, Path, Path]:
    name, separator, paths = str(value).partition("=")
    jepa, path_separator, direct = paths.partition(",")
    if (
        not separator
        or not path_separator
        or not name.strip()
        or not jepa.strip()
        or not direct.strip()
    ):
        raise argparse.ArgumentTypeError(
            "folds must use NAME=JEPA_REPORT_DIR,DIRECT_REPORT_DIR"
        )
    return name.strip(), Path(jepa.strip()), Path(direct.strip())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_necessary_conditions(
    contract: Mapping[str, Any], folds: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("unsupported impact direction diagnostic contract")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("impact direction diagnostic must prohibit live orders")
    decision = contract.get("decision_contract", {})
    if decision.get("promotion_eligible_from_this_diagnostic_alone") is not False:
        raise ValueError("impact direction diagnostic cannot be promotion eligible")
    planned = [str(value) for value in contract.get("folds", [])]
    if not planned or len(planned) != len(set(planned)):
        raise ValueError("impact direction diagnostic requires unique planned folds")
    tested = [str(fold["fold"]) for fold in folds]
    if not tested or len(tested) != len(set(tested)):
        raise ValueError("impact direction diagnostic requires unique tested folds")
    if any(value not in planned for value in tested):
        raise ValueError("tested folds must be declared in the contract")

    primary_horizons = [
        int(value) for value in contract["comparison"]["primary_horizons"]
    ]
    checks = []
    for fold in folds:
        name = str(fold["fold"])
        by_horizon = fold["paired"]["by_horizon"]
        for horizon in primary_horizons:
            row = by_horizon[str(horizon)]
            family_delta = float(
                row["equal_family_log_error_delta_direct_minus_jepa"]["mean"]
            )
            signature_delta = float(
                row["signature_cosine_delta_jepa_minus_direct"]["mean"]
            )
            checks.extend(
                (
                    {
                        "id": f"{name}:h{horizon}:family_error_direct_minus_jepa_above_zero",
                        "passed": family_delta > 0.0,
                        "actual": family_delta,
                        "threshold": "> 0",
                    },
                    {
                        "id": f"{name}:h{horizon}:signature_jepa_minus_direct_above_zero",
                        "passed": signature_delta > 0.0,
                        "actual": signature_delta,
                        "threshold": "> 0",
                    },
                )
            )
    failures = [row["id"] for row in checks if not row["passed"]]
    necessary_conditions_passed = not failures
    all_fold_gate_still_possible = bool(
        necessary_conditions_passed and set(tested) != set(planned)
    )
    planned_complete = set(tested) == set(planned)
    if failures:
        status = "early_rejected_necessary_condition"
        next_action = "do_not_add_directional_trajectory_loss_to_jepa_core"
    elif planned_complete:
        status = "necessary_conditions_complete"
        next_action = "run_remaining_absolute_and_major_trajectory_checks"
    else:
        status = "partial_necessary_conditions_passed"
        next_action = "continue_predeclared_folds"
    return {
        "schema_version": 1,
        "status": status,
        "planned_folds": planned,
        "tested_folds": tested,
        "untested_folds": [value for value in planned if value not in tested],
        "planned_evaluation_complete": planned_complete,
        "necessary_conditions": {
            "passed": necessary_conditions_passed,
            "checks": checks,
            "failures": failures,
        },
        "all_fold_gate_still_possible": all_fold_gate_still_possible,
        "decision": next_action,
        "eligible_as_unbiased_promotion_evidence": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }


def _load_summary(root: Path, role: str, target_version: str) -> dict[str, Any]:
    path = root / "summary.json"
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError(f"{role} report is not complete")
    if summary.get("target_version") != target_version:
        raise ValueError(f"{role} target version differs from the contract")
    if summary.get("test_used_for_selection") is not False:
        raise ValueError(f"{role} used test rows for selection")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"{role} does not prohibit live orders")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit necessary paired conditions for impact-direction trajectory heads."
    )
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--fold", type=parse_fold, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    target_version = str(contract["target_version"])
    fold_rows = []
    paired_frames = {}
    for name, jepa_root, direct_root in args.fold:
        _load_summary(jepa_root, f"{name} JEPA", target_version)
        _load_summary(direct_root, f"{name} direct", target_version)
        frame = paired_rows(jepa_root / "daily_test.csv", direct_root / "daily_test.csv")
        summary = paired_summary(frame)
        summary.pop("daily")
        fold_rows.append(
            {
                "fold": name,
                "jepa_root": str(jepa_root),
                "direct_root": str(direct_root),
                "jepa_summary_sha256": sha256_file(jepa_root / "summary.json"),
                "direct_summary_sha256": sha256_file(direct_root / "summary.json"),
                "jepa_daily_sha256": sha256_file(jepa_root / "daily_test.csv"),
                "direct_daily_sha256": sha256_file(direct_root / "daily_test.csv"),
                "paired": summary,
            }
        )
        paired_frames[name] = frame
    report = evaluate_necessary_conditions(contract, fold_rows)
    report["contract"] = str(args.contract)
    report["contract_sha256"] = sha256_file(args.contract)
    report["folds"] = fold_rows
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in paired_frames.items():
        frame.to_csv(args.output_dir / f"paired_{name}.csv", index=False)
    output = args.output_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker = (
        "EARLY_REJECTED"
        if report["status"] == "early_rejected_necessary_condition"
        else "NECESSARY_CONDITIONS_PASSED"
    )
    (args.output_dir / marker).touch()
    print(
        json.dumps(
            {
                "status": report["status"],
                "failure_count": len(report["necessary_conditions"]["failures"]),
                "decision": report["decision"],
                "output": str(output),
                "live_orders_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
