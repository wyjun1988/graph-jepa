from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def arg_value(args: List[str], key: str, default: str | None = None) -> str | None:
    if key not in args:
        return default
    idx = args.index(key)
    if idx + 1 >= len(args):
        return default
    return args[idx + 1]


def run_experiment(name: str, common_args: List[str], args: List[str], force: bool) -> Dict[str, Any]:
    reports_dir = Path(arg_value(args, "--reports-dir", f"reports/{name}") or f"reports/{name}")
    metrics_path = reports_dir / "real_backtest_metrics.json"
    if metrics_path.exists() and not force:
        status = "cached"
    else:
        cmd = [sys.executable, "scripts/run_real_backtest.py", *common_args, *args]
        print("RUN", name, " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        status = "ran"

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {"name": name, "status": status, "metrics": metrics, "reports_dir": str(reports_dir)}


def metric_row(result: Dict[str, Any], strategy: str) -> Dict[str, Any]:
    metrics = result["metrics"].get(strategy, {})
    return {
        "experiment": result["name"],
        "strategy": strategy,
        "periods": int(metrics.get("periods", 0)),
        "total_return": float(metrics.get("total_return", 0.0)),
        "cagr": float(metrics.get("cagr", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "hit_rate": float(metrics.get("hit_rate", 0.0)),
        "reports_dir": result["reports_dir"],
    }


def pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"


def update_docs(results: List[Dict[str, Any]], config_path: Path) -> None:
    rows = []
    for result in results:
        for strategy in [
            "jepa_only_ridge",
            "graph_jepa_ridge",
            "momentum_20d",
            "raw_ridge",
            "equal_weight_benchmark",
        ]:
            if strategy in result["metrics"]:
                rows.append(metric_row(result, strategy))

    rows_sorted = sorted(rows, key=lambda row: row["sharpe"], reverse=True)
    now = datetime.now().isoformat(timespec="seconds")

    log_lines = [
        "# Experiment Log",
        "",
        f"Last updated: {now}",
        f"Config: `{config_path}`",
        "",
        "| Experiment | Strategy | Periods | Total | CAGR | Sharpe | Max DD | Hit |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows_sorted:
        log_lines.append(
            f"| {row['experiment']} | {row['strategy']} | {row['periods']} | "
            f"{pct(row['total_return'])} | {pct(row['cagr'])} | {row['sharpe']:+.2f} | "
            f"{pct(row['max_drawdown'])} | {pct(row['hit_rate'])} |"
        )
    (ROOT / "docs" / "EXPERIMENT_LOG.md").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    best = rows_sorted[0] if rows_sorted else {}
    jepa_rows = [
        row for row in rows_sorted
        if row["strategy"] in {"graph_jepa_ridge", "jepa_only_ridge"}
    ]
    best_jepa = jepa_rows[0] if jepa_rows else {}
    handoff = {
        "last_updated": now,
        "purpose": "Continue stock-v2 Graph-JEPA experiments under M1/M1 Pro 16GB constraints.",
        "latest_config": str(config_path),
        "best_by_sharpe": best,
        "best_jepa_by_sharpe": best_jepa,
        "recent_results": rows_sorted[:12],
        "current_interpretation": [
            "Rich 26-feature node states and hidden=256/layers=4 train stably on M1 MPS.",
            "Union-date plus mean-imputed missing features is required for larger universes; date intersection was too strict.",
            "Current JEPA objectives improve masked/future state completion but do not yet beat raw factor or momentum heads.",
            "Temporal JEPA improved jepa_only over masked JEPA in KRX100, but latent loss was unstable; tune objective weights before scaling.",
            "Do not promote a JEPA model to operations just because state completion loss improves; use out-of-sample ranking metrics.",
        ],
        "hardware_notes": {
            "current_iMac": "M1 16GB, Qwen can run here; also supports torch MPS.",
            "incoming_server": "M1 Pro 16GB dedicated to Graph-JEPA experiments.",
        },
        "next_recommended_runs": [
            "Run configs/experiments.m1pro.json on the M1 Pro server.",
            "First run temporal_krx300_h384_l5_e8 after tuning temporal state_loss_weight/lr.",
            "Add an ensemble head using raw factors + momentum + graph_jepa latent; compare to each component.",
            "Try temporal pretrain with lower lr=3e-4 and state_loss_weight=0.1 to stabilize latent loss.",
            "Do not switch live trading model until paper runs are stable for several sessions.",
        ],
    }
    (ROOT / "docs" / "LLM_HANDOFF.json").write_text(
        json.dumps(handoff, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stock-v2 experiment sweep.")
    parser.add_argument("--config", default="configs/experiments.m1.json")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    common_args = list(config.get("common_args", []))
    results = []
    for experiment in config.get("experiments", []):
        results.append(
            run_experiment(
                name=experiment["name"],
                common_args=common_args,
                args=list(experiment["args"]),
                force=args.force,
            )
        )
    update_docs(results, config_path)
    print(f"updated docs/EXPERIMENT_LOG.md and docs/LLM_HANDOFF.json")


if __name__ == "__main__":
    main()
