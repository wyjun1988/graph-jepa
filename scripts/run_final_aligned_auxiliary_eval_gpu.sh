#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
DEVICE="${DEVICE:-cuda}"
CLEAN_CACHE_AFTER_EVAL="${CLEAN_CACHE_AFTER_EVAL:-1}"
RUN_NAME="final_aligned_jepa_opmask_aux_v1_seed17"
MODEL_ROOT="models/${RUN_NAME}"
CACHE_ROOT="data/cache/${RUN_NAME}_frozen_downstream"
REPORT_ROOT="reports/${RUN_NAME}_trained_auxiliary"
FOLD1="${RUN_NAME}_fold1_20231229_to_20241230"
FOLD2="${RUN_NAME}_fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$REPORT_ROOT"

evaluate_fold() {
  local fold="$1"
  local cache_name="$2"
  local test_end="$3"
  local output="$REPORT_ROOT/${fold}.json"
  if [[ ! -f "$output" ]]; then
    "$PYTHON_BIN" scripts/evaluate_trained_auxiliary_heads.py \
      --model-dir "$MODEL_ROOT/$fold" \
      --latent-cache-dir "$CACHE_ROOT/${cache_name}_latent" \
      --output "$output" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --test-end "$test_end" \
      --batch-size 8192 \
      --device "$DEVICE" \
      --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
      --external-cache-dir data/external_cache
  fi
  if [[ "$CLEAN_CACHE_AFTER_EVAL" == "1" ]]; then
    local cache_prefix="$CACHE_ROOT/$cache_name"
    rm -rf "${cache_prefix}_latent" "${cache_prefix}_raw.npy.parts"
    rm -f "${cache_prefix}_raw.npy" "${cache_prefix}_raw.npy.json"
  fi
}

evaluate_fold "$FOLD1" fold1 2024-12-30
evaluate_fold "$FOLD2" fold2 2026-07-10

"$PYTHON_BIN" - "$REPORT_ROOT" "$FOLD1" "$FOLD2" <<'PY'
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
folds = [json.loads((root / f"{name}.json").read_text(encoding="utf-8")) for name in sys.argv[2:]]
task_ics = {}
for fold_index, fold in enumerate(folds, start=1):
    for horizon, horizon_row in fold["results"].items():
        for task, task_row in horizon_row["tasks"].items():
            if "daily_ic" not in task_row:
                continue
            task_ics.setdefault(task, []).append(
                {
                    "fold": fold_index,
                    "horizon": int(horizon),
                    "mean": task_row["daily_ic"]["mean"],
                    "newey_west_t": task_row["daily_ic"]["newey_west_t"],
                    "mse_skill_vs_cross_sectional_zero": task_row[
                        "mse_skill_vs_cross_sectional_zero"
                    ],
                }
            )

aggregates = {}
for task, rows in task_ics.items():
    means = [float(row["mean"]) for row in rows]
    aggregates[task] = {
        "tests": len(rows),
        "positive_daily_ic": sum(value > 0.0 for value in means),
        "mean_fold_horizon_daily_ic": statistics.mean(means),
        "rows": rows,
    }

summary = {
    "status": "complete",
    "approval_scope": "research_only",
    "live_orders_allowed": False,
    "folds": len(folds),
    "tasks": aggregates,
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$REPORT_ROOT/AUXILIARY_EVAL_COMPLETE"
