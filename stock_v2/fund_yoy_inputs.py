"""GT-input channels (2026-07-26, user idea): a firm's own latest DISCLOSED
current-quarter YoY growth as INPUT features, available only from available_at.

Under staggered quarterly disclosure, peers that already disclosed carry ground
truth AT DECISION TIME while the pending firm is genuinely unobservable — so
training edges on these inputs is causal, not privileged. Modes:
  own      -> [fund_yoy_rev_dc, fund_yoy_oi_dc, fund_yoy_fresh]           (graph must aggregate peers)
  own_peer -> + [fund_yoy_peer_rev, fund_yoy_peer_oi]                      (feature-route control)
Names start with "fund_" so --temporal-exclude-feature-prefix fund_ keeps them
out of temporal state targets (quarterly-sparse), inputs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

OWN_COLS = ["yrev", "yoi", "fresh_days"]
PEER_COLS = ["peer_rev", "peer_oi"]
OWN_NAMES = ["fund_yoy_rev_dc", "fund_yoy_oi_dc", "fund_yoy_fresh"]
PEER_NAMES = ["fund_yoy_peer_rev", "fund_yoy_peer_oi"]


def fund_yoy_feature_names(mode: str) -> list[str]:
    if mode == "own":
        return list(OWN_NAMES)
    if mode == "own_peer":
        return list(OWN_NAMES) + list(PEER_NAMES)
    raise ValueError(f"unknown fund-yoy input mode: {mode}")


def augment_panel_with_fund_yoy(features, table_path: str, mode: str, train_end: str):
    """Append GT-input channels to a built FeaturePanel, in place.

    Raw values go to raw_features; normalized (train-period z, clip ±5, NaN->0)
    go to features. availability = row exists (a disclosure exists by that day).
    EXT node rows get zeros/unavailable. train_mean/std are extended so the
    checkpoint stays self-describing for evaluation-side reconstruction.
    """
    cols = OWN_COLS + (PEER_COLS if mode == "own_peer" else [])
    names = fund_yoy_feature_names(mode)
    df = pd.read_csv(table_path)
    df["fresh_days"] = np.log1p(df["fresh_days"].clip(lower=0))
    lut = {(d, str(t).zfill(6)): tuple(v) for d, t, *v in
           df[["date", "ticker"] + cols].itertuples(index=False, name=None)}

    n_dates, node_count, _ = features.features.shape
    stock_count = int(features.stock_node_count or len(features.tickers))
    tickers = [str(t).zfill(6) for t in list(features.tickers)[:stock_count]]
    date_strs = [str(d)[:10] for d in features.dates]
    k = len(cols)
    raw = np.full((n_dates, node_count, k), np.nan, dtype=np.float32)
    for di, day in enumerate(date_strs):
        for si, tk in enumerate(tickers):
            hit = lut.get((day, tk))
            if hit is not None:
                raw[di, si, :] = hit
    avail = np.isfinite(raw)

    train_mask = np.array([d <= train_end for d in date_strs])
    mean = np.zeros(k, dtype=np.float64)
    std = np.ones(k, dtype=np.float64)
    for ci in range(k):
        seg = raw[train_mask, :stock_count, ci]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 100:
            mean[ci] = float(seg.mean())
            s = float(seg.std())
            if np.isfinite(s) and s > 1e-9:
                std[ci] = s
    norm = np.clip((raw - mean[None, None, :]) / std[None, None, :], -5.0, 5.0)
    norm = np.where(avail, norm, 0.0).astype(np.float32)

    features.features = np.concatenate([features.features, norm], axis=2)
    features.raw_features = np.concatenate(
        [features.raw_features, np.where(avail, raw, 0.0).astype(features.raw_features.dtype)], axis=2)
    features.available_mask = np.concatenate(
        [features.available_mask, avail.astype(features.available_mask.dtype)], axis=2)
    features.feature_names = list(features.feature_names) + names
    features.train_mean = np.concatenate([features.train_mean, mean.astype(features.train_mean.dtype)])
    features.train_std = np.concatenate([features.train_std, std.astype(features.train_std.dtype)])
    cov = float(avail[:, :stock_count, 0].mean())
    print(f"fund-yoy inputs: mode={mode} channels={k} firm-day coverage={cov:.3f} "
          f"mean={np.round(mean, 4).tolist()}", flush=True)
    return features
