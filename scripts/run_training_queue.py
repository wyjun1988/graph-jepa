#!/usr/bin/env python3
"""GPU 학습 큐 러너 — SSH가 닿지 않는 머신(4090 PC 등)에 넘겨서 돌리는 용도.

한 번의 실행으로 여러 학습을 순서대로 돌리고, 각각의 평가까지 끝낸 뒤 결과를
`ops/queue_results.json` 한 파일로 모은다. 그 파일과 로그만 복사해 오면 된다.

설계 의도
  - GPU 시간을 태우기 전에 막는다. 예비점검이 데이터·드라이버·코드 버전을 모두
    확인하고, 하나라도 어긋나면 학습을 시작하지 않는다. 이 프로젝트에서 잘못된
    코드로 24에폭을 돌린 뒤에야 무효를 발견한 적이 있어, 점검에는 시퀀스 병합
    수정이 실제로 들어있는지까지 포함한다.
  - 한 런이 죽어도 큐는 계속 간다. 밤새 돌리는데 3번째가 터져서 나머지가 날아가면
    안 된다.
  - 이어서 돌릴 수 있다. 이미 평가 산출물이 있는 런은 건너뛴다.
  - 표준 라이브러리만 쓴다. 학습 자체가 요구하는 것 외에 추가 의존성이 없다.

사용법
  python scripts/run_training_queue.py --check         # 예비점검만
  python scripts/run_training_queue.py                 # 기본 큐 실행
  python scripts/run_training_queue.py --queue my.json # 큐 파일 지정
  python scripts/run_training_queue.py --only base_s17,attn_s17
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Redirected to a file, Python block-buffers stdout and a queue running for hours
# shows an empty log. Whoever is watching needs to see progress as it happens.
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
FOLD = "2025-09-05:2026-07-10"
SUFFIX = "20250905_to_20260710"
EVAL_DIR = ROOT / "reports/walk_forward/node_eval"
LOG_DIR = ROOT / "ops/training"
RESULT_PATH = ROOT / "ops/queue_results.json"
VERIFY_PATH = ROOT / "ops/queue_verification.json"

# 17,045 MiB was the measured peak for the attention arm at batch 16 on a 24GB
# card. Anything under this cannot run the queue as written.
VRAM_REQUIRED_MIB = 18_000

REQUIRED_DATA = [
    "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv",
    "data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl",
    "data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl",
    "data/kiwoom_investor_cache",
    "data/external_cache",
    "data/universes/krx500_pit_20191231.json",
]

# Flags shared by every run. Identical to the flags the reference baseline was
# trained with -- changing any of them makes the arms incomparable.
BASE_ARGS = """
--epochs 24 --checkpoint-epochs 12 --hidden-dim 1024 --layers 10
--train-batch-size 16 --device cuda --eval-device cuda --max-steps 0
--lr 3e-4 --horizon 10 --universe krx
--universe-manifest data/universes/krx500_pit_20191231.json --max-tickers 500
--cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv
--edge-correlation-mode signed --industry-edge-scale 0.2 --edge-top-k 6
--external-node-mode nodes --external-preset kr_global_rates
--external-cache-dir data/external_cache --external-lag-days 1
--require-all-external-factors
--event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl
--fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl
--investor-cache-dir data/kiwoom_investor_cache
--require-event-sensors --require-fundamental-sensors --require-investor-sensors
--min-event-coverage 0.99 --min-fundamental-coverage 0.79
--min-investor-coverage 0.95 --event-coverage-mode mask_uncovered
--fundamental-lag-days 1 --investor-flow-lag-days 1
--graph-neighbor-scale 1.0 --temporal-graph-neighbor-scale 0.0
--temporal-state-mode horizon_residual_heads --temporal-state-context-skip
--mask-strategy mixed --training-manifest-schema-version 4
--temporal-exclude-feature-prefix fund_ --policy-rate-edge-scale 0.0
--amp-dtype bfloat16 --ema-decay 0.9995 --state-loss-weight 1.0
--downstream-auxiliary-loss-weight 0.25 --current-imputation-loss-weight 1.0
--entry-path-correlation-loss-weight 0.05
""".split()

SEQ_ARGS = ["--sequence-window", "20", "--sequence-layers", "2", "--sequence-heads", "8"]

# For a small or pre-Ampere card used only to shake out code errors. fp32 because
# bfloat16 needs compute 8.0, and the dimensions are cut far enough to fit 4GB.
LITE_OVERRIDE = [
    "--epochs", "2", "--checkpoint-epochs", "1", "--hidden-dim", "128",
    "--layers", "2", "--max-tickers", "40", "--train-batch-size", "2",
    "--edge-top-k", "3", "--amp-dtype", "none", "--max-train-steps", "60",
]

# Appended last so argparse's last-wins rule shrinks every run to a few minutes.
# Verification exercises the real queue entries -- same flags, same code paths --
# rather than one generic smoke run, because a failure is usually specific to one
# entry's configuration and a single smoke would not surface it.
TINY_OVERRIDE = [
    "--epochs", "2", "--checkpoint-epochs", "1", "--hidden-dim", "256",
    "--layers", "3", "--max-tickers", "60", "--train-batch-size", "4",
    "--edge-top-k", "4", "--max-train-steps", "120",
]

# Default queue: the same architecture question at several seeds. One seed cannot
# settle it -- measured seed noise is sigma=0.0094, so a single pair of runs can
# differ by more than any real effect. Baseline seeds come first so that a queue
# stopped halfway still yields a usable sigma.
DEFAULT_QUEUE = [
    {"name": "base_s17", "seed": 17, "extra": ["--latent-loss-weight", "0.25"]},
    {"name": "attn_s17", "seed": 17, "extra": ["--latent-loss-weight", "0.25"] + SEQ_ARGS},
    {"name": "base_s3", "seed": 3, "extra": ["--latent-loss-weight", "0.25"]},
    {"name": "attn_s3", "seed": 3, "extra": ["--latent-loss-weight", "0.25"] + SEQ_ARGS},
    {"name": "base_s5", "seed": 5, "extra": ["--latent-loss-weight", "0.25"]},
    {"name": "attn_s5", "seed": 5, "extra": ["--latent-loss-weight", "0.25"] + SEQ_ARGS},
]


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}GB"


def preflight(verbose: bool = True, min_vram_mib: int = VRAM_REQUIRED_MIB,
              lite: bool = False, worker_override: int | None = None) -> tuple[bool, dict]:
    """Check everything that can waste GPU hours if wrong. Returns (ok, info)."""
    info: dict = {}
    problems: list[str] = []
    warnings: list[str] = []

    def say(msg: str) -> None:
        if verbose:
            print(msg)

    say("=" * 66)
    say(" 예비점검")
    say("=" * 66)

    info["python"] = sys.version.split()[0]
    info["platform"] = f"{platform.system()} {platform.machine()}"
    say(f"  파이썬      {info['python']}   {info['platform']}")
    if sys.version_info < (3, 10):
        problems.append(f"파이썬 3.10 이상이 필요하다 (현재 {info['python']})")

    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        problems.append(f"torch 를 불러올 수 없다: {exc}")
        say(f"  torch       불러오기 실패: {exc}")
        return False, {"problems": problems, **info}

    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    say(f"  torch       {torch.__version__}  (CUDA 빌드 {torch.version.cuda})")

    if not torch.cuda.is_available():
        problems.append("CUDA 를 쓸 수 없다. GPU 드라이버와 CUDA 빌드 torch 를 확인하라")
        say("  GPU         *** 사용 불가 ***")
    else:
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        total = torch.cuda.get_device_properties(0).total_memory // 2**20
        free = (torch.cuda.mem_get_info()[0] // 2**20) if hasattr(torch.cuda, "mem_get_info") else total
        info.update({"gpu": name, "capability": f"{cap[0]}.{cap[1]}",
                     "vram_total_mib": total, "vram_free_mib": free})
        say(f"  GPU         {name}  compute {cap[0]}.{cap[1]}")
        say(f"  VRAM        총 {total:,} MiB / 여유 {free:,} MiB  (필요 {min_vram_mib:,} MiB)")
        # In lite mode these are expected -- the point is to exercise code paths on
        # whatever CUDA card is at hand, not to certify the training machine.
        bucket = warnings if lite else problems
        if cap[0] < 8:
            bucket.append(f"bfloat16 학습에는 compute 8.0 이상이 필요하다 (현재 {cap[0]}.{cap[1]})")
        if free < min_vram_mib:
            bucket.append(
                f"VRAM 여유 {free:,} MiB 로는 배치 16 을 못 돌린다 "
                f"({min_vram_mib:,} MiB 필요). 다른 프로그램이 GPU 를 쓰고 있는지 확인하라"
            )

    for mod in ("numpy", "pandas"):
        try:
            m = __import__(mod)
            info[mod] = m.__version__
            say(f"  {mod:<11} {m.__version__}")
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{mod} 를 불러올 수 없다: {exc}")

    sys.path.insert(0, str(ROOT))
    try:
        import stock_v2  # noqa: F401
        say(f"  stock_v2    불러오기 OK  ({ROOT})")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"stock_v2 패키지를 불러올 수 없다: {exc}")

    # The bug that silently invalidated a full 24-epoch run: the batch merge
    # dropped node_sequence, so the sequence encoder trained on nothing while the
    # loss looked normal. Refuse to run stale code rather than repeat that.
    gj = ROOT / "stock_v2/graph_jepa.py"
    if gj.exists():
        src = gj.read_text(encoding="utf-8")
        checks = {
            "시퀀스 병합 (node_sequences)": "node_sequences" in src,
            "노드축 청크 (node_chunk)": "node_chunk" in src,
            "폴백 가드": "received no" in src,
        }
        for label, present in checks.items():
            say(f"  {'OK ' if present else '누락'}        {label}")
            if not present:
                problems.append(
                    f"graph_jepa.py 에 '{label}' 수정이 없다. 구버전 코드다 — "
                    "이 상태로 돌리면 어텐션 실험이 무효가 된다"
                )
    else:
        problems.append(f"{gj} 가 없다")

    say("  " + "-" * 62)
    missing = []
    for rel in REQUIRED_DATA:
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            say(f"  없음        {rel}")
        else:
            size = (sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                    if p.is_dir() else p.stat().st_size)
            say(f"  OK          {rel}  ({human(size)})")
    if missing:
        problems.append(f"학습 데이터 {len(missing)}개가 없다: {', '.join(missing[:3])}"
                        + (" 외" if len(missing) > 3 else ""))

    # Snapshot building runs in threads, but the work is NumPy and torch, which
    # release the GIL, so threads do scale. The cap tracks physical cores rather
    # than SMT threads -- on a 16-core/24-thread desktop 16 is the useful ceiling.
    cpus = os.cpu_count() or 4
    workers = max(2, min(16, cpus - 2)) if worker_override is None else int(worker_override)
    info["cpu_count"] = cpus
    info["snapshot_workers"] = workers
    say(f"  CPU         {cpus} 논리코어 -> --snapshot-workers {workers}"
        + ("  (수동 지정)" if worker_override is not None else ""))

    free_disk = shutil.disk_usage(ROOT).free
    info["disk_free"] = free_disk
    say(f"  디스크      여유 {human(free_disk)}")
    if free_disk < 20 * 2**30:
        warnings.append(f"디스크 여유가 {human(free_disk)} 뿐이다. 런당 약 3GB 를 쓴다")

    say("=" * 66)
    for w in warnings:
        say(f"  주의: {w}")
    if problems:
        say("\n  *** 실행 불가 ***")
        for p in problems:
            say(f"    - {p}")
    else:
        say("\n  전부 통과 — 학습을 시작할 수 있다")
    say("=" * 66)

    info["problems"] = problems
    info["warnings"] = warnings
    return not problems, info


def vram_stress(verbose: bool = True) -> tuple[bool, str]:
    """Run the sequence encoder at the real training shape and measure the peak.

    Preflight only reads how much VRAM is free; it cannot tell whether batch 16
    actually fits. That distinction is not academic -- three separate fixes here
    all produced code that was logically correct and still died at batch 16 with
    an out-of-memory error, and a small-batch trial would have passed every time.
    So this allocates the true shape (16 snapshots x 513 nodes x 20 steps) and
    does a forward and backward pass, which is the thing that actually failed.
    Takes about half a minute and settles the question on the target card.
    """
    import torch

    sys.path.insert(0, str(ROOT))
    from stock_v2.graph_jepa import TemporalSequenceEncoder

    nodes, window, feat_in, hidden = 16 * 513, 20, 149 * 2, 1024
    if verbose:
        print(f"\n  실제 학습 형상으로 메모리 시험: nodes={nodes} window={window} hidden={hidden}")
    dev = torch.device("cuda")
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = TemporalSequenceEncoder(input_dim=feat_in, hidden_dim=hidden, window=window,
                                      num_layers=2, num_heads=8).to(dev).train()
        x = torch.randn(nodes, window, feat_in, device=dev, requires_grad=True)
        enc(x).square().mean().backward()
        peak = torch.cuda.max_memory_allocated() // 2**20
        del enc, x
        torch.cuda.empty_cache()
        msg = (f"시퀀스 인코더 피크 {peak:,} MiB. 그래프 블록과 헤드가 약 10,500 MiB 를 "
               f"더 쓰므로 학습 전체는 약 {peak + 10_500:,} MiB 로 예상된다")
        if verbose:
            print(f"  OK          {msg}")
        return True, msg
    except torch.OutOfMemoryError:
        msg = "실제 형상에서 메모리 부족. 이 카드로는 배치 16 을 못 돌린다"
        if verbose:
            print(f"  실패        {msg}")
        return False, msg
    except Exception as exc:  # noqa: BLE001
        msg = f"메모리 시험 중 오류: {type(exc).__name__}: {exc}"
        if verbose:
            print(f"  실패        {msg}")
        return False, msg


# Two epochs because checkpoint-epochs must be strictly smaller, and the sequence
# flags stay on so the path that was broken is the path being exercised.
SMOKE_ITEM = {
    "name": "smoke", "seed": 17,
    "extra": ["--latent-loss-weight", "0.25", *SEQ_ARGS, *TINY_OVERRIDE],
}


def fingerprint(queue: list[dict], info: dict) -> str:
    """Identify what was verified, so the token cannot outlive its subject.

    A pass means "this queue, this code, this GPU". Editing the queue, pulling new
    code, or moving to another card all change the answer, so each goes into the
    hash and silently invalidates the token instead of letting a stale pass wave
    through a run it never covered.
    """
    gj = ROOT / "stock_v2/graph_jepa.py"
    parts = [
        json.dumps(queue, sort_keys=True, ensure_ascii=False),
        hashlib.sha256(gj.read_bytes()).hexdigest() if gj.exists() else "-",
        str(info.get("gpu", "-")),
        str(info.get("torch", "-")),
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]


def verification_ok(queue: list[dict], info: dict) -> tuple[bool, str]:
    if not VERIFY_PATH.exists():
        return False, "검증 기록이 없다"
    try:
        rec = json.loads(VERIFY_PATH.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False, "검증 기록을 읽을 수 없다"
    want = fingerprint(queue, info)
    if rec.get("code") != want:
        return False, (f"검증 코드가 맞지 않는다 (기록 {rec.get('code')}, 현재 {want}). "
                       "큐·코드·GPU 중 무언가가 바뀌었다")
    if not rec.get("passed"):
        return False, "지난 검증이 실패로 끝났다"
    return True, f"검증 코드 {rec['code']} ({rec.get('when', '?')})"


# Seed noise measured over six identical runs. A single pair can differ by more
# than any real effect, so a lone run beating the baseline means nothing on its
# own -- the bar is two sigma, and even that only justifies a confirmation run.
SEED_SIGMA = 0.0094


def report(results: list[dict], baseline_prefix: str = "base") -> list[dict]:
    """Rank finished runs and say which are worth confirming elsewhere.

    Exploration on one card is cheap and noisy; confirmation is expensive. This
    exists to keep the second step small -- it prints every run against the
    baseline mean in units of seed noise, and returns only those clearing two
    sigma, which is the set worth paying to re-run.
    """
    done = [r for r in results if (r.get("metrics") or {}).get("ic") is not None]
    if not done:
        print("  완료된 런이 없다.")
        return []

    base = [r for r in done if r["name"].startswith(baseline_prefix)]
    base_ic = sum(r["metrics"]["ic"] for r in base) / len(base) if base else None

    print(f"{'런':<18}{'시드':>5}{'IC':>10}{'기준대비':>10}{'σ배수':>8}{'state_R2':>10}")
    print("-" * 66)
    for r in sorted(done, key=lambda x: -x["metrics"]["ic"]):
        ic = r["metrics"]["ic"]
        m = r["metrics"]
        if base_ic is None:
            print(f"{r['name']:<18}{r['seed']:>5}{ic:>10.4f}{'—':>10}{'—':>8}"
                  f"{m.get('state_r2', float('nan')):>10.4f}")
            continue
        d = ic - base_ic
        print(f"{r['name']:<18}{r['seed']:>5}{ic:>10.4f}{d:>+10.4f}{d/SEED_SIGMA:>+8.1f}"
              f"{m.get('state_r2', float('nan')):>10.4f}")

    if base_ic is None:
        print(f"\n  기준선('{baseline_prefix}'로 시작하는 런)이 없어 선별할 수 없다.")
        return []

    print(f"\n  기준선 평균 IC {base_ic:+.4f}  (런 {len(base)}건)")
    print(f"  시드 노이즈 σ={SEED_SIGMA:.4f} -> 2σ 문턱 {base_ic + 2*SEED_SIGMA:+.4f}")

    winners = [r for r in done
               if not r["name"].startswith(baseline_prefix)
               and r["metrics"]["ic"] - base_ic >= 2 * SEED_SIGMA]
    if winners:
        print(f"\n  2σ 를 넘긴 런 {len(winners)}건 — 다른 카드에서 재현 확인 대상:")
        for r in winners:
            print(f"    {r['name']}  IC {r['metrics']['ic']:+.4f}")
    else:
        print("\n  2σ 를 넘긴 런이 없다. 재현 확인에 GPU 를 더 쓸 이유가 없다.")
    return winners


def eval_csv(name: str) -> Path:
    return EVAL_DIR / f"{name}_fold1_{SUFFIX}" / "future_rollout.csv"


def read_metrics(name: str) -> dict | None:
    p = eval_csv(name)
    if not p.exists():
        return None
    try:
        import pandas as pd
        d = pd.read_csv(p)
        out = {"n": int(len(d))}
        for col, key in [("realized_entry_path_ic_top100", "ic"), ("state_r2", "state_r2")]:
            if col in d.columns:
                out[key] = float(d[col].mean())
        return out
    except Exception:  # noqa: BLE001
        return None


def run_one(item: dict, workers: int, timeout_h: float) -> dict:
    name = item["name"]
    log = LOG_DIR / f"{name}.log"
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, "scripts/run_walk_forward_node_eval.py",
           "--name", name, "--fold", FOLD, "--start", "2020-01-01",
           "--seed", str(item.get("seed", 17)),
           "--snapshot-workers", str(workers),
           *BASE_ARGS, *item.get("extra", [])]

    print(f"\n  ▶ {name}  (seed {item.get('seed', 17)})  로그 {log}")
    started = time.time()
    with log.open("a", encoding="utf-8", errors="replace") as fh:
        fh.write(f"\n===== {name} 시작 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        fh.flush()
        try:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                  timeout=timeout_h * 3600, check=False)
            code = proc.returncode
        except subprocess.TimeoutExpired:
            code = -1
            fh.write(f"\n*** {timeout_h}시간 초과로 중단 ***\n")

    elapsed = time.time() - started
    metrics = read_metrics(name)
    status = "성공" if code == 0 and metrics else ("시간초과" if code == -1 else "실패")
    print(f"    {status}  {elapsed/60:.0f}분"
          + (f"  IC {metrics['ic']:+.4f}" if metrics and "ic" in metrics else ""))

    if status != "성공":
        tail = ""
        if log.exists():
            lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in reversed(lines[-400:]):
                if any(k in line for k in ("Error", "error", "Traceback",
                                           "out of memory", "received no")):
                    tail = line[:200]
                    break
        if tail:
            print(f"    원인: {tail}")

    return {"name": name, "seed": item.get("seed", 17), "status": status,
            "returncode": code, "minutes": round(elapsed / 60, 1), "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU 학습 큐 러너")
    ap.add_argument("--check", action="store_true", help="예비점검만 하고 끝낸다")
    ap.add_argument("--report", action="store_true",
                    help="끝난 결과를 순위대로 보여주고, 2σ 를 넘겨 재현 확인할 가치가 "
                         "있는 런만 골라낸다. GPU 를 쓰지 않는다")
    ap.add_argument("--export-winners", type=str, default=None,
                    help="--report 로 골라낸 런을 큐 JSON 으로 저장한다 (다른 카드에서 그대로 실행)")
    ap.add_argument("--verify-lite", action="store_true",
                    help="작은/구형 CUDA 카드(예: 1650 Ti)에서 코드 경로만 확인한다. "
                         "fp32 로 돌리고 메모리 시험은 건너뛰므로 부분 검증이며, "
                         "실제 학습 게이트는 열리지 않는다")
    ap.add_argument("--verify", action="store_true",
                    help="큐 전체를 작은 설정으로 한 번씩 돌려 전부 통과하는지 확인하고, "
                         "통과하면 검증 코드를 남긴다. 실제 학습은 이 코드가 있어야 시작된다")
    ap.add_argument("--smoke", action="store_true",
                    help="예비점검 + 실제 형상 메모리 시험 + 짧은 학습 1건. "
                         "긴 큐를 걸기 전에 이 머신에서 되는지 확인한다 (약 10분)")
    ap.add_argument("--queue", type=str, default=None, help="큐 정의 JSON 경로")
    ap.add_argument("--only", type=str, default=None, help="쉼표로 구분한 런 이름만 실행")
    ap.add_argument("--timeout-hours", type=float, default=8.0, help="런당 제한시간")
    ap.add_argument("--force", action="store_true", help="이미 끝난 런도 다시 돌린다")
    ap.add_argument("--workers", type=int, default=None,
                    help="스냅샷 워커 수를 직접 지정한다. 기본은 논리코어에서 자동 산출")
    ap.add_argument("--min-vram-mib", type=int, default=VRAM_REQUIRED_MIB,
                    help="VRAM 하한. 작은 설정으로 배관만 점검할 때만 낮춘다")
    args = ap.parse_args()

    if args.report:
        if not RESULT_PATH.exists():
            print(f"결과 파일이 없다: {RESULT_PATH}")
            return 1
        saved = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        print("=" * 66)
        print(f" 결과  ({saved.get('host', '?')} / torch {saved.get('torch', '?')})")
        print("=" * 66)
        winners = report(saved.get("runs", []))
        if args.export_winners:
            by_name = {q["name"]: q for q in DEFAULT_QUEUE}
            out = [by_name[w["name"]] for w in winners if w["name"] in by_name]
            if out:
                Path(args.export_winners).write_text(
                    json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"\n  큐 {len(out)}건 저장 -> {args.export_winners}")
                print(f"  다른 카드에서:  python scripts/run_training_queue.py "
                      f"--queue {args.export_winners} --verify")
            else:
                print("\n  내보낼 런이 없다.")
        print("=" * 66)
        return 0

    ok, info = preflight(min_vram_mib=args.min_vram_mib, lite=args.verify_lite,
                         worker_override=args.workers)
    if args.check:
        return 0 if ok else 1
    if not ok:
        print("\n예비점검을 통과하지 못해 학습을 시작하지 않는다.")
        return 1

    queue = DEFAULT_QUEUE
    if args.queue:
        queue = json.loads(Path(args.queue).read_text(encoding="utf-8"))
    if args.only:
        wanted = {t.strip() for t in args.only.split(",")}
        queue = [q for q in queue if q["name"] in wanted]

    if args.verify_lite:
        print("\n" + "=" * 66)
        print(f" 부분 검증 — 큐 {len(queue)}건의 코드 경로만 확인 (fp32, 메모리 시험 없음)")
        print("=" * 66)
        for w in info.get("warnings", []):
            print(f"  이 카드의 한계: {w}")
        checks = []
        for item in queue:
            tiny = dict(item)
            tiny["name"] = f"lite_{item['name']}"
            tiny["extra"] = list(item.get("extra", [])) + LITE_OVERRIDE
            r = run_one(tiny, info["snapshot_workers"], timeout_h=1.0)
            checks.append({"step": item["name"], "ok": r["status"] == "성공",
                           "detail": f"{r['status']} {r['minutes']:.0f}분"})
        passed = all(c["ok"] for c in checks)
        print("\n" + "=" * 66)
        for c in checks:
            print(f"  {'통과' if c['ok'] else '실패'}   {c['step']:<16} {c['detail']}")
        print("=" * 66)
        if passed:
            print("  코드 경로는 전부 통과했다. 다만 이것은 부분 검증이다:")
            print("    확인 안 됨: bfloat16 경로, 배치 16 메모리, 처리 속도")
            print("    실제 학습 카드에서 --verify 를 다시 돌려야 게이트가 열린다.")
        else:
            print("  실패한 항목이 있다. 큰 카드로 옮기기 전에 여기서 고치는 편이 싸다.")
            print(f"    로그: {LOG_DIR}/lite_<런이름>.log")
        print("=" * 66)
        return 0 if passed else 1

    if args.verify:
        print("\n" + "=" * 66)
        print(f" 검증 — 큐 {len(queue)}건을 작은 설정으로 전부 돌린다")
        print("=" * 66)
        fits, msg = vram_stress()
        checks = [{"step": "메모리 시험", "ok": fits, "detail": msg}]
        if not fits:
            print("\n메모리 시험 실패. 이 카드로는 배치 16 을 못 돌린다.")
        else:
            for item in queue:
                tiny = dict(item)
                tiny["name"] = f"verify_{item['name']}"
                tiny["extra"] = list(item.get("extra", [])) + TINY_OVERRIDE
                r = run_one(tiny, info["snapshot_workers"], timeout_h=1.0)
                checks.append({"step": item["name"], "ok": r["status"] == "성공",
                               "detail": f"{r['status']} {r['minutes']:.0f}분"})

        passed = all(c["ok"] for c in checks)
        code = fingerprint(queue, info)
        VERIFY_PATH.parent.mkdir(parents=True, exist_ok=True)
        VERIFY_PATH.write_text(json.dumps({
            "code": code, "passed": passed,
            "when": time.strftime("%Y-%m-%d %H:%M:%S"),
            "gpu": info.get("gpu"), "torch": info.get("torch"),
            "queue": [q["name"] for q in queue], "checks": checks,
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        print("\n" + "=" * 66)
        for c in checks:
            print(f"  {'통과' if c['ok'] else '실패'}   {c['step']:<16} {c['detail']}")
        print("=" * 66)
        if passed:
            print(f"  전부 통과 — 검증 코드  {code}")
            print(f"  기록: {VERIFY_PATH}")
            print("  이제 실제 학습:  python scripts/run_training_queue.py")
        else:
            print("  통과하지 못했다. 위 실패 항목의 로그를 확인하라:")
            print(f"    {LOG_DIR}/verify_<런이름>.log")
            print("  실제 학습은 시작할 수 없다.")
        print("=" * 66)
        return 0 if passed else 1

    if args.smoke:
        print("\n" + "=" * 66)
        print(" 실기 확인 — 이 머신에서 배치 16 이 들어가는지, 코드가 끝까지 도는지")
        print("=" * 66)
        fits, msg = vram_stress()
        if not fits:
            print("\n메모리 시험을 통과하지 못했다. 긴 큐를 걸면 반드시 터진다.")
            return 1
        res = run_one(SMOKE_ITEM, info["snapshot_workers"], timeout_h=1.0)
        print("\n" + "=" * 66)
        if res["status"] == "성공":
            print("  통과 — 이 머신에서 큐를 돌려도 된다.")
            print(f"    메모리: {msg}")
            print("    다음: python scripts/run_training_queue.py")
        else:
            print(f"  실패 ({res['status']}) — 로그를 확인하라: {LOG_DIR / 'smoke.log'}")
        print("=" * 66)
        return 0 if res["status"] == "성공" else 1

    verified, why = verification_ok(queue, info)
    if not verified:
        print("\n" + "=" * 66)
        print("  검증을 통과한 기록이 없어 학습을 시작하지 않는다.")
        print(f"    이유: {why}")
        print("    먼저:  python scripts/run_training_queue.py --verify")
        print("=" * 66)
        return 1
    print(f"\n{why}")

    print(f"큐 {len(queue)}건: {', '.join(q['name'] for q in queue)}")
    results = []
    if RESULT_PATH.exists():
        try:
            results = json.loads(RESULT_PATH.read_text(encoding="utf-8")).get("runs", [])
        except Exception:  # noqa: BLE001
            results = []

    for item in queue:
        if not args.force and read_metrics(item["name"]):
            print(f"\n  건너뜀 {item['name']} (이미 평가 산출물 있음)")
            continue
        res = run_one(item, info["snapshot_workers"], args.timeout_hours)
        results = [r for r in results if r["name"] != res["name"]] + [res]
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text(json.dumps(
            {"host": info.get("gpu"), "torch": info.get("torch"),
             "platform": info.get("platform"), "runs": results},
            ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n" + "=" * 66)
    print(f"{'런':<16}{'시드':>6}{'상태':>8}{'분':>7}{'IC':>10}{'state_R2':>11}")
    print("-" * 66)
    for r in results:
        m = r.get("metrics") or {}
        print(f"{r['name']:<16}{r['seed']:>6}{r['status']:>8}{r['minutes']:>7.0f}"
              f"{m.get('ic', float('nan')):>10.4f}{m.get('state_r2', float('nan')):>11.4f}")
    print("=" * 66)
    print(f"결과 파일: {RESULT_PATH}")
    print(f"로그:      {LOG_DIR}/<런이름>.log")
    print("\n이 두 가지만 복사해 오면 된다.")
    return 0 if all(r["status"] == "성공" for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
