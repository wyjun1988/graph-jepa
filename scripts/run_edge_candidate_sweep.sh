#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-../stock/venv/bin/python}"
EVENT_PATH="${EVENT_PATH:-data/events/news_backfill_qwen_calibrated_krx100_20200101_20260710.jsonl}"
SWEEP_NAME="${SWEEP_NAME:-edge_sweep_v1_krx60_h192_l3_e4}"
DEVICE="${DEVICE:-mps}"

ROOT_REPORTS="reports/${SWEEP_NAME}"
ROOT_MODELS="models/${SWEEP_NAME}"
NODE_EVAL_DIR="reports/node_prediction_eval_${SWEEP_NAME}"
LOG_DIR="ops/training/${SWEEP_NAME}"
SUMMARY_JSONL="${ROOT_REPORTS}/summary.jsonl"

mkdir -p "${ROOT_REPORTS}" "${ROOT_MODELS}" "${NODE_EVAL_DIR}" "${LOG_DIR}"
: > "${SUMMARY_JSONL}"

COMMON_ARGS=(
  --universe krx
  --max-tickers 60
  --start 2020-01-01
  --train-end 2023-12-29
  --horizon 5
  --top-k 5
  --epochs 4
  --hidden-dim 192
  --layers 3
  --hide-ratio 0.35
  --mask-strategy mixed
  --pretrain-task temporal
  --temporal-offset 5
  --latent-rollout-steps 5
  --rollout-offsets 3,5,10
  --path-horizons 1,3,5,10
  --lr 0.0003
  --state-loss-weight 0.40
  --edge-window 60
  --edge-top-k 6
  --min-abs-corr 0.20
  --event-path "${EVENT_PATH}"
  --device "${DEVICE}"
)

summarize_candidate() {
  local name="$1"
  local model_dir="${ROOT_MODELS}/${name}"
  local summary_path="${NODE_EVAL_DIR}/${name}/summary.json"
  local latency_path="${ROOT_REPORTS}/${name}/latency.json"
  "${PYTHON_BIN}" - "${name}" "${model_dir}" "${summary_path}" "${latency_path}" "${SUMMARY_JSONL}" <<'PY'
import json
import sys
from pathlib import Path

name, model_dir, summary_path, latency_path, output_path = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text())
latency = json.loads(Path(latency_path).read_text()) if Path(latency_path).exists() else {}
future = summary["future_rollout_by_horizon"]
row = {
    "candidate": name,
    "model_dir": model_dir,
    "current_mse_skill_vs_zero": summary["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"],
    "h1_mse_skill": future["1"]["mse_skill_vs_persistence"]["mean"],
    "h3_mse_skill": future["3"]["mse_skill_vs_persistence"]["mean"],
    "h5_mse_skill": future["5"]["mse_skill_vs_persistence"]["mean"],
    "h10_mse_skill": future["10"]["mse_skill_vs_persistence"]["mean"],
    "h3_delta_corr": future["3"]["delta_corr"]["mean"],
    "h5_delta_corr": future["5"]["delta_corr"]["mean"],
    "h10_delta_corr": future["10"]["delta_corr"]["mean"],
    "latency_mean_total_sec": latency.get("total_sec", {}).get("mean"),
    "latency_p95_total_sec": latency.get("total_sec", {}).get("p95"),
}
with Path(output_path).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps(row, ensure_ascii=False, indent=2), flush=True)
PY
}

run_candidate() {
  local name="$1"
  shift
  local model_dir="${ROOT_MODELS}/${name}"
  local reports_dir="${ROOT_REPORTS}/${name}"
  local log_path="${LOG_DIR}/${name}.log"
  mkdir -p "${reports_dir}" "${model_dir}"
  echo "[$(date)] candidate=${name} start" | tee -a "${log_path}"
  "${PYTHON_BIN}" scripts/run_real_backtest.py \
    "${COMMON_ARGS[@]}" \
    "$@" \
    --reports-dir "${reports_dir}" \
    --models-dir "${model_dir}" \
    >> "${log_path}" 2>&1
  "${PYTHON_BIN}" scripts/evaluate_node_prediction.py \
    --model-dir "${model_dir}" \
    --output-dir "${NODE_EVAL_DIR}" \
    --max-steps 120 \
    --device "${DEVICE}" \
    --horizons 1,3,5,10 \
    >> "${log_path}" 2>&1
  "${PYTHON_BIN}" scripts/measure_jepa_latency.py \
    --model-dir "${model_dir}" \
    --device "${DEVICE}" \
    --cycles 20 \
    --warmup 3 \
    --rollout-steps 10 \
    --output "${reports_dir}/latency.json" \
    >> "${log_path}" 2>&1
  summarize_candidate "${name}" | tee -a "${log_path}"
  echo "[$(date)] candidate=${name} done" | tee -a "${log_path}"
}

run_candidate self_only_news \
  --edge-correlation-mode none \
  --edge-top-k 0

run_candidate corr_signed_news \
  --edge-correlation-mode signed

run_candidate corr_abs_news \
  --edge-correlation-mode abs

run_candidate corr_positive_news \
  --edge-correlation-mode positive

run_candidate partial_abs_news \
  --edge-correlation-mode none \
  --edge-top-k 0 \
  --partial-corr-top-k 6 \
  --partial-corr-mode abs \
  --partial-corr-min-abs 0.08 \
  --partial-corr-scale 0.50

run_candidate lead_lag_abs_news \
  --edge-correlation-mode none \
  --edge-top-k 0 \
  --lead-lag-top-k 4 \
  --lead-lag-mode abs \
  --lead-lag-days 1 \
  --lead-lag-min-abs-corr 0.06 \
  --lead-lag-scale 0.50

run_candidate corr_abs_event_news \
  --edge-correlation-mode abs \
  --event-edge-top-k 4 \
  --event-edge-scale 0.25 \
  --event-edge-min-weight 0.05

run_candidate hybrid_positive_lead_event_news \
  --edge-correlation-mode positive \
  --lead-lag-top-k 4 \
  --lead-lag-mode abs \
  --lead-lag-days 1 \
  --lead-lag-min-abs-corr 0.06 \
  --lead-lag-scale 0.35 \
  --event-edge-top-k 4 \
  --event-edge-scale 0.20 \
  --event-edge-min-weight 0.05

echo "[$(date)] edge candidate sweep complete: ${SUMMARY_JSONL}"
