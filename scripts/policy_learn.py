"""Direction C: LEARN the strategy instead of hand-picking it.

The hand-picked LS3 (return long/short + vol overlay) got test Sharpe ~2.5. Can a
small policy that LEARNS how to combine the five head predictions beat that
heuristic out-of-sample? This is the user's "그 또한 학습을 통해서 모델을 세우는" path.

Leak-free by construction: the OOS prediction window is split by TIME. The policy
trains only on the earlier dates and is evaluated only on the later dates -- it
never sees a test date during training. Inputs are the model's head predictions
(cross-sectionally z-scored per date, so the policy is scale-free). Output is a
per-stock signed weight; the book is made dollar-neutral per date, so there is no
market beta. Objective: maximize the training-window Sharpe of the realized
portfolio return (differentiable). We report train vs test to expose overfitting,
and compare against the hand-picked LS1/LS3 on the SAME test dates.

Research only. Returns are diagnostic, never a gate.
"""

import sys, json, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

p = argparse.ArgumentParser()
p.add_argument("--pred-cache", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--train-frac", type=float, default=0.6, help="earliest fraction of dates for training")
p.add_argument("--epochs", type=int, default=400)
p.add_argument("--hidden", type=int, default=16)
p.add_argument("--l2", type=float, default=1e-3)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--output", default="")
a = p.parse_args()
torch.manual_seed(a.seed)
H = a.horizon
FEATS = ["pred_ret", "pred_mfe", "pred_mae", "pred_vol"]

df = pd.read_parquet(a.pred_cache).dropna().copy()
# cross-sectional z-score of each head within each date (scale-free inputs)
for c in FEATS:
    df[c + "_z"] = df.groupby("date")[c].transform(lambda s: (s - s.mean()) / (s.std() + 1e-8))
zcols = [c + "_z" for c in FEATS]
dates = sorted(df["date"].unique())
split = int(len(dates) * a.train_frac)
train_dates, test_dates = set(dates[:split]), set(dates[split:])
print(f"dates: {len(dates)} total -> train {len(train_dates)} ({dates[0].date()}..{dates[split-1].date()}), "
      f"test {len(test_dates)} ({dates[split].date()}..{dates[-1].date()})")

# group tensors per date
def make_groups(date_set):
    g = []
    for d in dates:
        if d not in date_set:
            continue
        sub = df[df["date"] == d]
        if len(sub) < 10:
            continue
        x = torch.tensor(sub[zcols].values, dtype=torch.float32)
        r = torch.tensor(sub["ret"].values, dtype=torch.float32)
        g.append((x, r))
    return g

train_g, test_g = make_groups(train_dates), make_groups(test_dates)


class Policy(nn.Module):
    def __init__(self, d_in, hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hid), nn.GELU(), nn.Linear(hid, 1))

    def forward(self, x):
        w = torch.tanh(self.net(x).squeeze(-1))          # signed weight per stock
        w = w - w.mean()                                  # dollar-neutral (no beta)
        denom = w.abs().sum() + 1e-8
        return w / denom                                  # unit gross exposure


def port_returns(model, groups):
    return torch.stack([(model(x) * r).sum() for x, r in groups])


def sharpe(rets, ppy):
    if rets.std() < 1e-9:
        return rets.mean() * 0.0
    return rets.mean() / (rets.std() + 1e-9) * (ppy ** 0.5)


model = Policy(len(zcols), a.hidden)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=a.l2)
ppy = 252 / H
for ep in range(a.epochs):
    opt.zero_grad()
    r = port_returns(model, train_g)
    loss = -sharpe(r, ppy)                                # maximize training Sharpe
    loss.backward()
    opt.step()

# ---- evaluate: non-overlapping test dates for an honest Sharpe ----
model.eval()
with torch.no_grad():
    tr = port_returns(model, train_g).numpy()
    te_all = port_returns(model, test_g).numpy()
te_no = te_all[::H]  # non-overlapping in the test window


def stats(x, ppy):
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 3 or x.std() == 0:
        return {"n": int(len(x)), "sharpe": 0.0, "mean_pct": float(x.mean()*100), "mdd_pct": 0.0}
    cum = np.cumprod(1 + x)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {"n": int(len(x)), "sharpe": float(x.mean()/x.std()*np.sqrt(ppy)),
            "mean_pct": float(x.mean()*100), "mdd_pct": dd*100}


# ---- baselines on the SAME test dates (hand-picked, no learning) ----
def baseline(score_col, short_col, cut_col=None, frac=0.2):
    out = []
    for d in dates:
        if d not in test_dates:
            continue
        g = df[df["date"] == d]
        gl = g if cut_col is None else g[g[cut_col] <= g[cut_col].quantile(1 - frac)]
        if len(gl) < 5 or len(g) < 5:
            out.append(np.nan); continue
        lo = gl[gl[score_col] >= gl[score_col].quantile(1 - frac)]["ret"].mean()
        sh = g[g[short_col] >= g[short_col].quantile(1 - frac)]["ret"].mean()
        out.append(lo - sh)
    arr = np.array(out); return arr[np.isfinite(arr)][::H]

df["pred_ret_neg"] = -df["pred_ret"]
ls1 = baseline("pred_ret", "pred_ret_neg")
ls3 = baseline("pred_ret", "pred_vol", cut_col="pred_vol")

res = {"role": "research_direction_C_learned_policy", "promotion_eligible": False, "horizon": H,
       "train": stats(tr, ppy), "test_learned_policy": stats(te_no, ppy),
       "test_baseline_LS1_return": stats(ls1, ppy), "test_baseline_LS3_return_vol": stats(ls3, ppy),
       "note": "leak-free time split; policy trained on early dates, all eval on non-overlapping late dates. dollar-neutral."}
print(f"\n{'전략':32}{'n':>4}{'Sharpe':>9}{'기간%':>9}{'MDD%':>8}")
for k, lbl in [("test_learned_policy","학습정책(test)"),("test_baseline_LS1_return","LS1 수익(test)"),
               ("test_baseline_LS3_return_vol","LS3 수익+변동성(test)")]:
    s = res[k]; print(f"  {lbl:30}{s['n']:>4}{s['sharpe']:>9.2f}{s['mean_pct']:>9.3f}{s['mdd_pct']:>8.1f}")
print(f"  (참고) 학습정책 train Sharpe {res['train']['sharpe']:.2f}  <- test와 격차 크면 과적합")
if a.output:
    Path("/Users/wooyeol/work/stock-v2", a.output).write_text(json.dumps(res, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("->", a.output)
