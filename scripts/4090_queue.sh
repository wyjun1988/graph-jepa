#!/usr/bin/env bash
# 4090 멀티시드 검증 큐
# 목적: 상위 후보 2개를 4시드씩 돌려서 베이스라인 대비 유의미한 차이가 있는지 확인
#
# 베이스라인 (6시드 평균): IC +0.0386, σ=0.0094
# 2σ 판정 문턱: 새 설정 4시드 평균 ≥ +0.0508
#
# 후보 A: ema_fast — ema-decay 0.99 (기존 0.9995), hidden-completion-weight 0.25
# 후보 B: vic_lat1 — latent-loss-weight 1.0, VICReg (variance 1.0, covariance 0.01, target 1.0)
#
# 예상 소요: 4090에서 ~2시간/런 × 8런 = ~16시간 (하룻밤)

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${ROOT}/venv/bin/python"
# venv 경로가 다르면 수정:
[ -x "$PY" ] || PY="$(which python3)"

cd "$ROOT"
FOLD="2025-09-05:2026-07-10"; SUF="20250905_to_20260710"

# 공통 플래그 (베이스라인과 동일)
BASE="--hidden-dim 1024 --layers 10 --train-batch-size 16 --snapshot-workers 8 \
--device cuda --eval-device cuda --max-steps 0 --lr 3e-4 --horizon 10 \
--universe krx --universe-manifest data/universes/krx500_pit_20191231.json --max-tickers 500 \
--cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
--edge-correlation-mode signed --industry-edge-scale 0.2 --edge-top-k 6 \
--external-node-mode nodes --external-preset kr_global_rates --external-cache-dir data/external_cache \
--external-lag-days 1 --require-all-external-factors \
--event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
--fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
--investor-cache-dir data/kiwoom_investor_cache \
--require-event-sensors --require-fundamental-sensors --require-investor-sensors \
--min-event-coverage 0.99 --min-fundamental-coverage 0.79 --min-investor-coverage 0.95 \
--event-coverage-mode mask_uncovered --fundamental-lag-days 1 --investor-flow-lag-days 1 \
--graph-neighbor-scale 1.0 --temporal-graph-neighbor-scale 0.0 \
--temporal-state-mode horizon_residual_heads --temporal-state-context-skip \
--mask-strategy mixed --training-manifest-schema-version 4 \
--temporal-exclude-feature-prefix fund_ --policy-rate-edge-scale 0.0 --amp-dtype bfloat16 \
--ema-decay 0.9995 --state-loss-weight 1.0 --downstream-auxiliary-loss-weight 0.25 \
--current-imputation-loss-weight 1.0 --entry-path-correlation-loss-weight 0.05 \
--sequence-window 0"

guard(){ while [ "$(pgrep -c -f 'run_real_backtest|evaluate_node_prediction')" != "0" ]; do sleep 45; done; }
have(){ [ -f "reports/walk_forward/node_eval/$1_fold1_${SUF}/future_rollout.csv" ]; }

run(){
  local N="$1"; shift
  guard
  have "$N" && { echo "  skip $N (already done)"; return 0; }
  echo "  ▶ [$(date '+%H:%M:%S')] $N"
  mkdir -p "ops/training"
  "$PY" scripts/run_walk_forward_node_eval.py --name "$N" --fold "$FOLD" --start 2020-01-01 \
    --epochs 24 --checkpoint-epochs 12 $BASE "$@" \
    >> "ops/training/${N}.log" 2>&1 \
    && echo "  ✅ $N" \
    || { echo "  ❌ $N (exit $?)"; grep -iE "error|out of memory|Traceback" "ops/training/${N}.log" | tail -3; }
}

echo "════ 4090 멀티시드 검증 큐 시작 $(date '+%Y-%m-%d %H:%M') ════"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# ───────── 후보 A: EMA fast (ema=0.99, hidden-completion=0.25) ─────────
echo "─── 후보 A: ema_fast (ema-decay=0.99) ───"
for SEED in 3 5 11 23; do
  run "ema_s${SEED}" --seed "$SEED" --latent-loss-weight 0.25 \
    --ema-decay 0.99 --hidden-completion-weight 0.25
done

# ───────── 후보 B: VICReg + 고 latent (latent=1.0, vic) ─────────
echo "─── 후보 B: vic_lat1 (latent=1.0 + VICReg) ───"
for SEED in 3 5 11 23; do
  run "vic_s${SEED}" --seed "$SEED" --latent-loss-weight 1.0 \
    --latent-variance-weight 1.0 --latent-variance-target 1.0 --latent-covariance-weight 0.01
done

# ───────── 판정 ─────────
echo ""
echo "════ 판정 $(date '+%H:%M') ════"
"$PY" - << 'PYEOF'
import csv
from pathlib import Path

NE = Path("reports/walk_forward/node_eval")
S = "20250905_to_20260710"

def ic(name):
    p = NE / f"{name}_fold1_{S}" / "future_rollout.csv"
    if not p.exists():
        return None
    vals = []
    with open(p) as f:
        for row in csv.DictReader(f):
            vals.append(float(row["realized_entry_path_ic_top100"]))
    return sum(vals) / len(vals) if vals else None

# Baseline seeds (already run on RunPod)
base_seeds = {"sig_s3": 3, "sig_s5": 5, "sig_s11": 11,
              "vf_s0.0_seed17": 17, "sig_s23": 23, "sig_s29": 29}
base_ics = []
for name in base_seeds:
    v = ic(name)
    if v is not None:
        base_ics.append(v)

if base_ics:
    base_mean = sum(base_ics) / len(base_ics)
    base_var = sum((x - base_mean)**2 for x in base_ics) / (len(base_ics) - 1)
    base_std = base_var ** 0.5
else:
    base_mean, base_std = 0.0386, 0.0094

print(f"베이스라인: mean={base_mean:+.4f} std={base_std:.4f} (n={len(base_ics)})")
print()

for label, prefix in [("A: ema_fast", "ema_s"), ("B: vic_lat1", "vic_s")]:
    ics = []
    for seed in [3, 5, 11, 23]:
        name = f"{prefix}{seed}"
        v = ic(name)
        if v is not None:
            ics.append((seed, v))
            print(f"  {name}: IC={v:+.4f}")
        else:
            print(f"  {name}: 미완료")

    if len(ics) >= 2:
        vals = [x[1] for x in ics]
        m = sum(vals) / len(vals)
        v = sum((x - m)**2 for x in vals) / (len(vals) - 1)
        s = v ** 0.5
        delta = m - base_mean
        # Two-sample t-test SE
        se = (base_std**2/len(base_ics) + s**2/len(vals)) ** 0.5
        t = delta / se if se > 0 else 0
        print(f"  >> {label}: mean={m:+.4f} std={s:.4f} delta={delta:+.4f} t={t:+.1f}")
        if t > 2:
            print(f"  >> 판정: 유의미하게 우세 (t={t:.1f} > 2)")
        elif t < -2:
            print(f"  >> 판정: 유의미하게 열세")
        else:
            print(f"  >> 판정: 유의미한 차이 없음 (|t|={abs(t):.1f} < 2)")
    print()

print("단일 폴드 결과 — 확정은 다시드x다폴드 필요")
PYEOF

echo "════ 완료 $(date '+%Y-%m-%d %H:%M') ════"
