"""Does the availability mask hand the model a 2026 fact about 2020?

THE FINDING THIS MEASURES. All 47 delisted tickers in the lifecycle release are
FinanceDataReader return-index proxies: volume is disabled and RawClose is empty
across their ENTIRE history, from 2020-01-02, not merely near their delisting.
Every volume, value and amihud feature is therefore permanently NaN for them,
and so are all eight investor features (the traded-value denominator is NaN).
The model's `available_mask` consequently carries a perfect, permanent
fingerprint for "this name delists before the 2026-07 dataset build" -- a fact
knowable only in 2026, present in a 2020 training row.

WHY IT CANNOT SIMPLY BE FIXED. Dropping the 47 restores the survivorship bias
the lifecycle panel exists to remove, which is a worse error. The missing volume
history cannot be recovered: the names are delisted and the vendor no longer
serves them. So the question is not how to remove the fingerprint but how much
it is worth -- and that is what this measures.

WHAT IS MEASURED, all model-free:

  share            what fraction of the evaluation window's observed cells sit on
                   proxy nodes. This BOUNDS the inflation: pooled skill is a
                   ratio of summed squared error, so a node group contributing 2%
                   of the cells cannot move it by more than roughly that.
  difficulty       persistence error on proxy nodes against ordinary ones. A
                   smooth index is easier to forecast than a traded stock, and if
                   proxies are much easier their inclusion flatters intent 1.
  separability     whether the mask alone identifies them -- reported for
                   completeness, since it is true by construction.
  exploitability   whether the fingerprint carries anything about FUTURE RETURNS.
                   This is the one that matters: an identifiable group is only a
                   leak if knowing the group tells you something you should not
                   know yet.

Evidence class: research. Nothing here qualifies a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_node_prediction import build_features_from_ckpt, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from stock_v2.graph_jepa import _feature_group_map


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="read for the panel definition only")
    parser.add_argument("--output", required=True)
    # Decides the panel's path_horizons (evaluate_node_prediction.py:512,679), so
    # it must match what the model was trained on or the data contract refuses to
    # pair the two.
    parser.add_argument("--horizons", default="1,2,3,5,10", help="panel horizons -- must match training")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    args = evaluator_contract_defaults(parser.parse_args())

    ckpt = torch.load(
        Path(args.model_dir) / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    names = list(features.feature_names)
    groups = _feature_group_map(names)
    stock_count = int(features.tradable_count)
    steps = select_steps(features, ckpt_args, args)
    tickers = list(features.tickers)[:stock_count]

    mask = np.asarray(features.available_mask, dtype=bool)

    # A proxy is a node whose volume family is never observed anywhere in the
    # panel. That IS the fingerprint, written the way the model sees it.
    volume_like = sorted(set(groups.get("liquidity", [])) | set(groups.get("investor", [])))
    if not volume_like:
        raise SystemExit("no liquidity/investor features in this panel")
    ever_observed = mask[:, :stock_count, :][:, :, volume_like].any(axis=(0, 2))
    proxy = ~ever_observed
    proxy_names = [tickers[i] for i in np.where(proxy)[0]]

    eval_mask = mask[steps][:, :stock_count, :]
    proxy_cells = int(eval_mask[:, proxy, :].sum())
    total_cells = int(eval_mask.sum())

    # Difficulty: a one-step persistence error on the state target, the same
    # baseline intent 1's skill is denominated in.
    values = np.asarray(features.features[:, :stock_count, :], dtype=np.float64)
    persistence_sse: dict[str, float] = {}
    persistence_cells: dict[str, int] = {}
    for label, selector in (("proxy", proxy), ("ordinary", ~proxy)):
        errors, cells = 0.0, 0
        for step in steps:
            if int(step) + 1 >= values.shape[0]:
                continue
            current = values[int(step), selector, :]
            future = values[int(step) + 1, selector, :]
            observed = mask[int(step) + 1, :stock_count, :][selector]
            delta = (future - current)[observed]
            delta = delta[np.isfinite(delta)]
            errors += float(np.sum(delta**2))
            cells += int(delta.size)
        persistence_sse[label] = errors
        persistence_cells[label] = cells

    proxy_mse = persistence_sse["proxy"] / max(persistence_cells["proxy"], 1)
    ordinary_mse = persistence_sse["ordinary"] / max(persistence_cells["ordinary"], 1)

    # Exploitability: does membership in the fingerprinted group carry anything
    # about future returns on the evaluation window? Identifiability is only a
    # leak if it tells you something.
    forward: dict[str, list[float]] = {"proxy": [], "ordinary": []}
    for step in steps:
        if int(step) + 10 >= len(features.dates):
            continue
        start = np.asarray(features.close[int(step), :stock_count], dtype=np.float64)
        end = np.asarray(features.close[int(step) + 10, :stock_count], dtype=np.float64)
        usable = np.isfinite(start) & np.isfinite(end) & (start > 0.0)
        forward_return = np.full(stock_count, np.nan)
        forward_return[usable] = end[usable] / start[usable] - 1.0
        for label, selector in (("proxy", proxy), ("ordinary", ~proxy)):
            block = forward_return[selector]
            block = block[np.isfinite(block)]
            if block.size:
                forward[label].append(float(np.mean(block)))

    payload: dict[str, Any] = {
        "role": "research_only_delisting_fingerprint_probe",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "question": "the availability mask identifies which names delist, using 2026 knowledge in 2020 rows. How much is that worth?",
        "model_dir": str(args.model_dir),
        "train_end": str(ckpt_args.get("train_end")),
        "eval_sessions": int(len(steps)),
        "separability": {
            "proxy_nodes": int(proxy.sum()),
            "stock_nodes": stock_count,
            "rule": "no liquidity or investor feature is EVER observed for this node",
            "note": "identifiable by construction; the mask states it directly",
            "tickers": proxy_names[:60],
        },
        "share_of_evaluation_cells": {
            "proxy_cells": proxy_cells,
            "total_cells": total_cells,
            "fraction": proxy_cells / max(total_cells, 1),
            "why_this_bounds_it": "pooled skill is a ratio of summed squared error, so a group holding this fraction of the cells cannot move it much more than this",
        },
        "difficulty": {
            "proxy_persistence_mse": proxy_mse,
            "ordinary_persistence_mse": ordinary_mse,
            "ratio_proxy_over_ordinary": proxy_mse / ordinary_mse if ordinary_mse > 0 else None,
            "reading": "below 1 means proxies are EASIER than traded stocks, and their inclusion flatters intent 1",
        },
        "exploitability": {
            "mean_forward_10d_return_proxy": float(np.mean(forward["proxy"])) if forward["proxy"] else None,
            "mean_forward_10d_return_ordinary": float(np.mean(forward["ordinary"])) if forward["ordinary"] else None,
            "spread": (
                float(np.mean(forward["proxy"]) - np.mean(forward["ordinary"]))
                if forward["proxy"] and forward["ordinary"]
                else None
            ),
            "reading": "a large spread means the fingerprint sorts future returns, which is the leak that would matter",
        },
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    share = payload["share_of_evaluation_cells"]["fraction"]
    ratio = payload["difficulty"]["ratio_proxy_over_ordinary"]
    spread = payload["exploitability"]["spread"]
    print(f"프록시 노드 {int(proxy.sum())}/{stock_count}  평가 {len(steps)}세션\n")
    print(f"  평가창 관측 셀 중 프록시 몫   {share*100:.2f}%   <- 부풀림의 상한")
    print(f"  지속성 MSE 비 (프록시/일반)   {ratio:.3f}   {'프록시가 더 쉬움' if ratio and ratio < 1 else '프록시가 더 어려움'}")
    if spread is not None:
        print(f"  10일 선도수익률 격차          {spread*100:+.3f}%p   <- 지문이 수익률을 가르는가")
    print()
    if share < 0.02:
        print("  -> 프록시는 평가 셀의 2% 미만이다. 의도1의 숫자를 의미있게 부풀릴 수 없다.")
    elif ratio and ratio < 0.5:
        print("  -> 프록시가 훨씬 쉽고 몫도 작지 않다. 의도1 숫자에서 이들을 분리 보고해야 한다.")
    else:
        print("  -> 몫은 있으나 난이도가 일반 종목과 비슷하다. 부풀림은 몫에 비례하는 수준.")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
