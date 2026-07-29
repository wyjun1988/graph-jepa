from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_HORIZONS = (1, 2, 3, 5, 10)


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check(
    rows: list[dict[str, Any]],
    check_id: str,
    passed: bool,
    *,
    value: Any,
    requirement: str,
    fold: str | None = None,
    horizon: int | None = None,
) -> None:
    row: dict[str, Any] = {
        "id": check_id,
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
    }
    if fold is not None:
        row["fold"] = fold
    if horizon is not None:
        row["horizon"] = int(horizon)
    rows.append(row)


def evaluate_candidate(
    walk_forward: dict[str, Any],
    node_summaries: list[dict[str, Any]],
    direct_comparisons: list[dict[str, Any]],
    dataset_audit: dict[str, Any],
    ohlcv_audit: dict[str, Any] | None = None,
    *,
    min_stocks: int = 450,
    min_eval_steps: int = 200,
    max_h10_mean_pairwise_cosine: float = 0.98,
    min_h10_variance_participation: float = 0.10,
    min_return_nw_t: float = 1.96,
    direct_tolerance: float = 0.0,
) -> dict[str, Any]:
    folds = list(walk_forward.get("folds") or [])
    if len(folds) < 2:
        raise ValueError("at least two walk-forward folds are required")
    if len(node_summaries) != len(folds):
        raise ValueError("one node summary is required per walk-forward fold")
    if len(direct_comparisons) != len(folds):
        raise ValueError("one direct comparison is required per walk-forward fold")

    checks: list[dict[str, Any]] = []
    audit_blockers = list(dataset_audit.get("blockers") or [])
    check(
        checks,
        "dataset_integrity",
        dataset_audit.get("status") == "pass" and not audit_blockers,
        value={"status": dataset_audit.get("status"), "blockers": audit_blockers},
        requirement="status=pass and blockers empty",
    )
    if ohlcv_audit is not None:
        ohlcv_blockers = list(ohlcv_audit.get("blockers") or [])
        output_tickers = int(ohlcv_audit.get("output_tickers", 0) or 0)
        verified_files = int(ohlcv_audit.get("verified_files", 0) or 0)
        check(
            checks,
            "ohlcv_integrity",
            (
                ohlcv_audit.get("status") == "pass"
                and not ohlcv_blockers
                and output_tickers >= int(min_stocks)
                and verified_files == output_tickers
            ),
            value={
                "status": ohlcv_audit.get("status"),
                "output_tickers": output_tickers,
                "verified_files": verified_files,
                "blockers": ohlcv_blockers,
            },
            requirement=(
                f"status=pass, blockers empty, >={min_stocks} tickers, and every file verified"
            ),
        )

    for index, (fold, node, comparison) in enumerate(
        zip(folds, node_summaries, direct_comparisons), start=1
    ):
        fold_name = str(fold.get("fold") or f"fold{index}")
        check(
            checks,
            "stock_coverage",
            int(fold.get("tickers", 0)) >= int(min_stocks),
            value=int(fold.get("tickers", 0)),
            requirement=f">={min_stocks}",
            fold=fold_name,
        )
        check(
            checks,
            "evaluation_length",
            int(fold.get("eval_steps", 0)) >= int(min_eval_steps),
            value=int(fold.get("eval_steps", 0)),
            requirement=f">={min_eval_steps} trading dates",
            fold=fold_name,
        )

        current_skill = float(
            node["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"]
        )
        check(
            checks,
            "current_imputation_skill",
            current_skill > 0.0,
            value=current_skill,
            requirement=">0 versus zero-state baseline",
            fold=fold_name,
        )

        node_horizons = node.get("future_rollout_by_horizon") or {}
        direct_horizons = comparison.get("horizons") or {}
        return_significance = (
            node.get("realized_entry_path_correlation_significance") or {}
        )
        liquid_return_significance = (
            node.get("realized_entry_path_liquidity_significance", {}).get(
                "top300",
                {},
            )
        )
        rollout_dependency = node.get("rollout_dependency_by_horizon") or {}

        for horizon in REQUIRED_HORIZONS:
            raw_key = str(horizon)
            if raw_key not in node_horizons or raw_key not in direct_horizons:
                raise ValueError(f"{fold_name} is missing node/direct horizon {horizon}")
            if raw_key not in return_significance:
                raise ValueError(
                    f"{fold_name} is missing realized entry-path significance "
                    f"horizon {horizon}"
                )
            if raw_key not in liquid_return_significance:
                raise ValueError(
                    f"{fold_name} is missing top300 realized entry-path "
                    f"significance horizon {horizon}"
                )
            if raw_key not in rollout_dependency:
                raise ValueError(
                    f"{fold_name} is missing rollout-dependency horizon {horizon}"
                )

            skill = float(
                node_horizons[raw_key]["pooled_mse_skill_vs_persistence"]
            )
            delta_corr = float(
                node_horizons[raw_key]["delta_corr"]["mean"]
            )
            check(
                checks,
                "future_state_skill",
                skill > 0.0,
                value=skill,
                requirement=">0 versus persistence",
                fold=fold_name,
                horizon=horizon,
            )
            check(
                checks,
                "future_delta_correlation",
                delta_corr > 0.0,
                value=delta_corr,
                requirement=">0",
                fold=fold_name,
                horizon=horizon,
            )

            rollout_skill = float(
                rollout_dependency[raw_key]["pooled_mse_skill_vs_no_rollout"]
            )
            check(
                checks,
                "rollout_dependency",
                rollout_skill > 0.0,
                value=rollout_skill,
                requirement=">0 pooled MSE skill versus the same decoder with zero rollout innovation",
                fold=fold_name,
                horizon=horizon,
            )

            paired_delta = float(
                direct_horizons[raw_key]["state_skill"]["delta_direct_minus_jepa"][
                    "mean"
                ]
            )
            check(
                checks,
                "direct_mlp_challenge",
                paired_delta <= float(direct_tolerance),
                value=paired_delta,
                requirement=(
                    f"direct minus JEPA mean skill <= {float(direct_tolerance):.6f}"
                ),
                fold=fold_name,
                horizon=horizon,
            )

            for metric, check_id in (
                ("entry_path_ic", "direct_mlp_entry_path_challenge"),
                (
                    "entry_path_ic_top300",
                    "direct_mlp_entry_path_top300_challenge",
                ),
            ):
                if metric not in direct_horizons[raw_key]:
                    raise ValueError(
                        f"{fold_name} direct comparison is missing {metric} "
                        f"horizon {horizon}"
                    )
                direct_path_delta = float(
                    direct_horizons[raw_key][metric][
                        "delta_direct_minus_jepa"
                    ]["mean"]
                )
                check(
                    checks,
                    check_id,
                    direct_path_delta <= float(direct_tolerance),
                    value=direct_path_delta,
                    requirement=(
                        "direct minus JEPA mean entry-path IC "
                        f"<= {float(direct_tolerance):.6f}"
                    ),
                    fold=fold_name,
                    horizon=horizon,
                )

            return_row = return_significance[raw_key]
            return_corr = float(return_row["mean_target_corr"])
            return_t = float(return_row["newey_west_t_stat"])
            check(
                checks,
                "realized_entry_path_correlation",
                return_corr > 0.0 and return_t >= float(min_return_nw_t),
                value={"mean": return_corr, "newey_west_t": return_t},
                requirement=f"mean>0 and Newey-West t>={min_return_nw_t}",
                fold=fold_name,
                horizon=horizon,
            )

            liquid_return_row = liquid_return_significance[raw_key]
            liquid_return_corr = float(liquid_return_row["mean_target_corr"])
            liquid_return_t = float(liquid_return_row["newey_west_t_stat"])
            check(
                checks,
                "realized_entry_path_correlation_top300",
                (
                    liquid_return_corr > 0.0
                    and liquid_return_t >= float(min_return_nw_t)
                ),
                value={
                    "mean": liquid_return_corr,
                    "newey_west_t": liquid_return_t,
                },
                requirement=f"mean>0 and Newey-West t>={min_return_nw_t}",
                fold=fold_name,
                horizon=horizon,
            )

        latent_h10 = node.get("latent_health", {}).get("h10", {})
        cosine = float(latent_h10["mean_pairwise_cosine"]["mean"])
        participation = float(
            latent_h10["variance_participation_ratio"]["mean"]
        )
        check(
            checks,
            "h10_latent_cosine",
            cosine < float(max_h10_mean_pairwise_cosine),
            value=cosine,
            requirement=f"<{max_h10_mean_pairwise_cosine}",
            fold=fold_name,
            horizon=10,
        )
        check(
            checks,
            "h10_latent_participation",
            participation >= float(min_h10_variance_participation),
            value=participation,
            requirement=f">={min_h10_variance_participation}",
            fold=fold_name,
            horizon=10,
        )

    failed = [row for row in checks if not row["passed"]]
    return {
        "status": "pass" if not failed else "blocked",
        "approval_scope": "read_only_shadow" if not failed else "none",
        "live_orders_allowed": False,
        "folds": len(folds),
        "required_horizons": list(REQUIRED_HORIZONS),
        "checks": checks,
        "failed_checks": failed,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Shadow Candidate Gate",
        "",
        f"Status: **{result['status']}**",
        f"Approval scope: `{result['approval_scope']}`",
        "Live orders allowed: `false`",
        "",
        f"Checks: {result['summary']['passed']} passed, "
        f"{result['summary']['failed']} failed.",
    ]
    if result["failed_checks"]:
        lines.extend(["", "## Failed Checks", ""])
        for row in result["failed_checks"]:
            location = str(row.get("fold", "global"))
            if "horizon" in row:
                location += f" h{row['horizon']}"
            lines.append(
                f"- `{row['id']}` ({location}): value={row['value']}; "
                f"requires {row['requirement']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply strict read-only shadow promotion gates to a model candidate."
    )
    parser.add_argument("--walk-forward-summary", required=True)
    parser.add_argument("--node-summary", action="append", required=True)
    parser.add_argument("--direct-comparison", action="append", required=True)
    parser.add_argument("--dataset-audit", required=True)
    parser.add_argument("--ohlcv-audit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-stocks", type=int, default=450)
    parser.add_argument("--min-eval-steps", type=int, default=200)
    parser.add_argument("--max-h10-mean-pairwise-cosine", type=float, default=0.98)
    parser.add_argument("--min-h10-variance-participation", type=float, default=0.10)
    parser.add_argument("--min-return-nw-t", type=float, default=1.96)
    parser.add_argument("--direct-tolerance", type=float, default=0.0)
    args = parser.parse_args()

    result = evaluate_candidate(
        load_json(args.walk_forward_summary),
        [load_json(path) for path in args.node_summary],
        [load_json(path) for path in args.direct_comparison],
        load_json(args.dataset_audit),
        load_json(args.ohlcv_audit),
        min_stocks=args.min_stocks,
        min_eval_steps=args.min_eval_steps,
        max_h10_mean_pairwise_cosine=args.max_h10_mean_pairwise_cosine,
        min_h10_variance_participation=args.min_h10_variance_participation,
        min_return_nw_t=args.min_return_nw_t,
        direct_tolerance=args.direct_tolerance,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "gate.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "gate.md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["summary"], sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
