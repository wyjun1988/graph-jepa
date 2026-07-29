#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
SOURCE_RUN="final_aligned_jepa_opmask_aux_v1_seed17"
SOURCE_LOG="reports/${SOURCE_RUN}/pipeline.log"
SWEEP_NAME="${SOURCE_RUN}_batch_throughput"
REPORT_ROOT="reports/${SWEEP_NAME}"
MODEL_ROOT="models/${SWEEP_NAME}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZES="${BATCH_SIZES:-8 16 24 32}"
KEEP_MODELS="${KEEP_MODELS:-0}"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$MODEL_ROOT"

template="$(grep -m 1 '^RUN ' "$SOURCE_LOG" | sed 's/^RUN //')"
if [[ -z "$template" ]]; then
  echo "missing training command in $SOURCE_LOG" >&2
  exit 1
fi

read -r -a base_command <<< "$template"

replace_arg() {
  local name="$1"
  local value="$2"
  local index
  for index in "${!command[@]}"; do
    if [[ "${command[$index]}" == "$name" ]]; then
      command[$((index + 1))]="$value"
      return 0
    fi
  done
  echo "missing template argument: $name" >&2
  return 1
}

for batch_size in $BATCH_SIZES; do
  run_dir="$REPORT_ROOT/batch_${batch_size}"
  model_dir="$MODEL_ROOT/batch_${batch_size}"
  status_path="$run_dir/status.json"
  if [[ -f "$status_path" ]]; then
    continue
  fi
  mkdir -p "$run_dir" "$model_dir"
  command=("${base_command[@]}")
  replace_arg --epochs "$EPOCHS"
  replace_arg --train-batch-size "$batch_size"
  replace_arg --reports-dir "$run_dir"
  replace_arg --models-dir "$model_dir"

  echo "BATCH_SWEEP batch=$batch_size epochs=$EPOCHS"
  set +e
  "${command[@]}" 2>&1 | tee "$run_dir/run.log"
  status=${PIPESTATUS[0]}
  set -e
  printf '{"batch_size":%d,"epochs":%d,"exit_status":%d,"throughput_only":true,"live_orders_allowed":false}\n' \
    "$batch_size" "$EPOCHS" "$status" > "$status_path"
  if [[ "$KEEP_MODELS" == "0" ]]; then
    rm -rf "$model_dir"
  fi
done

/usr/local/bin/python - "$REPORT_ROOT" <<'PY'
import csv
import json
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for run_dir in sorted(root.glob("batch_*"), key=lambda path: int(path.name.split("_")[-1])):
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    history_path = run_dir / "pretrain_history.csv"
    row = dict(status)
    if status["exit_status"] == 0 and history_path.exists():
        with history_path.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        measured = history[1:] if len(history) > 1 else history
        row.update(
            samples_per_second_median=statistics.median(
                float(item["samples_per_second"]) for item in measured
            ),
            epoch_seconds_median=statistics.median(
                float(item["epoch_seconds"]) for item in measured
            ),
            peak_cuda_memory_mib=max(
                float(item["peak_cuda_memory_mib"]) for item in history
            ),
            optimizer_steps_per_epoch=int(float(history[-1]["optimizer_steps"])),
        )
    rows.append(row)

successful = [row for row in rows if "samples_per_second_median" in row]
best = max(successful, key=lambda row: row["samples_per_second_median"]) if successful else None
summary = {
    "scope": "throughput_only",
    "live_orders_allowed": False,
    "source_run": "final_aligned_jepa_opmask_aux_v1_seed17",
    "results": rows,
    "best_throughput_batch": best["batch_size"] if best else None,
    "note": "A larger batch requires a separate convergence test before changing production training.",
}
(root / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

touch "$REPORT_ROOT/SWEEP_COMPLETE"
