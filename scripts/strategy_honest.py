"""Honest read on the market-neutral spread: remove the overlap inflation and
apply transaction costs. Reads the cached per-(date,stock) predictions.

The h10 long-short is evaluated every session, so consecutive periods overlap by
9 days -> autocorrelated returns understate std and inflate Sharpe. The honest
Sharpe samples NON-OVERLAPPING periods (every H sessions). Costs: each period the
book is fully re-entered and exited on both legs; net = gross - 4*c_oneway on the
traded notional (long enter+exit + short enter+exit).
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--pred-cache", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--top-frac", type=float, default=0.2)
p.add_argument("--output", default="")
a = p.parse_args()
H = a.horizon
df = pd.read_parquet(a.pred_cache)
df["pred_ret_neg"] = -df["pred_ret"]
dates = sorted(df["date"].unique())


def ls_spread(long_col, short_col, long_cut=None):
    """Per-date long-short spread series, indexed by date order."""
    out = []
    for d in dates:
        g = df[df["date"] == d]
        gl = g if long_cut is None else g[g[long_cut] <= g[long_cut].quantile(1 - a.top_frac)]
        if len(gl) < 5 or len(g) < 5:
            out.append(np.nan); continue
        lo = gl[gl[long_col] >= gl[long_col].quantile(1 - a.top_frac)]["ret"].mean()
        sh = g[g[short_col] >= g[short_col].quantile(1 - a.top_frac)]["ret"].mean()
        out.append(lo - sh)
    return np.array(out, dtype=np.float64)


def sharpe(d, periods_per_year):
    d = d[np.isfinite(d)]
    if len(d) < 3 or d.std() == 0:
        return 0.0, 0.0, 0.0
    ann = np.sqrt(periods_per_year)
    cum = np.cumprod(1 + d)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return float(d.mean() / d.std() * ann), float(d.mean()), dd


strats = {
    "LS1_return": ("pred_ret", "pred_ret_neg", None),
    "LS3_return_cutVOL/shortVOL": ("pred_ret", "pred_vol", "pred_vol"),
}
COSTS = [0.0, 0.001, 0.002, 0.003]  # one-way, applied 4x per period (both legs, enter+exit)

report = {"horizon": H, "top_frac": a.top_frac, "n_sessions": len(dates),
          "note": "overlapping = every session (inflated); non_overlap = every H sessions (honest). cost=one-way, 4x/period.",
          "strategies": {}}
print(f"h{H}, 상위 {a.top_frac:.0%}, {len(dates)} 세션. 비중첩 = 매 {H}세션 표본.\n")
for name, (lc, sc, cut) in strats.items():
    g = ls_spread(lc, sc, cut)
    ov_s, ov_m, ov_dd = sharpe(g, 252 / H)                 # overlapping (inflated)
    no = g[::H]                                              # non-overlapping sample
    no_s, no_m, no_dd = sharpe(no, 252 / H)
    print(f"[{name}]")
    print(f"  중첩(과장)    Sharpe {ov_s:5.2f}  기간수익 {ov_m*100:6.3f}%  MDD {ov_dd*100:6.1f}%")
    print(f"  비중첩(정직)  Sharpe {no_s:5.2f}  기간수익 {no_m*100:6.3f}%  MDD {no_dd*100:6.1f}%  (n={np.isfinite(no).sum()})")
    costrow = {}
    for c in COSTS:
        net = no - 4 * c
        ns_, nm_, ndd_ = sharpe(net, 252 / H)
        costrow[f"cost_{c*100:.1f}pct_oneway"] = {"sharpe": round(ns_, 2), "mean_per_period_pct": round(nm_*100, 3)}
        print(f"    +비용 {c*100:.1f}%(편도)  순Sharpe {ns_:5.2f}  순수익 {nm_*100:6.3f}%")
    report["strategies"][name] = {"overlap_sharpe": round(ov_s, 2), "nonoverlap_sharpe": round(no_s, 2),
                                  "nonoverlap_mean_pct": round(no_m*100, 3), "nonoverlap_mdd_pct": round(no_dd*100, 1),
                                  "cost_sensitivity": costrow}
    print()

if a.output:
    Path("/Users/wooyeol/work/stock-v2", a.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("->", a.output)
