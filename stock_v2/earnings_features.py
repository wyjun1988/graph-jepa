"""Earnings-cycle features: where the stock sits in its reporting calendar, and
how it has historically reacted when it reports.

The 149 production features summarise history only through fixed-width windows
(return_20d, volatility_60d, ...). Quarterly reporting is invisible to that view:
the events are ~90 days apart, so a 20-day window never contains one and a 120-day
window drowns it in noise. Two things are therefore missing entirely:

  forward  -- nothing in the feature set looks ahead. Holding for 10 days with an
              earnings release inside the window is a different bet from holding a
              stock that just reported, and the model currently cannot tell them
              apart.
  reaction -- how *this* stock behaves on its own report days (gap size, drift
              persistence) is a stable per-name trait that fixed windows miss.

PIT discipline:
  * the next report date is *estimated* from this ticker's own past filing lags,
    never from the actual future filing;
  * a past reaction is only used once its full measurement window has closed
    strictly before the current session.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CALENDAR_NAMES = [
    "earn_days_until_next",
    "earn_within_horizon",
    "earn_quarter_phase",
]
SUE_NAMES = [
    "earn_sue_revenue",
    "earn_sue_op_income",
    "earn_sue_eps",
]
SUE_FIELDS = ["revenue", "operating_income", "eps"]
REACTION_NAMES = [
    "earn_reaction_mean_4q",
    "earn_reaction_std_4q",
    "earn_drift_5d_mean_4q",
    "earn_gap_abs_mean_4q",
]
DEFAULT_LAG_DAYS = 45.0


def _next_quarter_end(period_end: pd.Timestamp) -> pd.Timestamp:
    return (period_end + pd.offsets.QuarterEnd(1)).normalize()


def _load_disclosures(path: str | Path) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """ticker -> sorted [(available_at, period_end)]"""
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            avail = pd.Timestamp(str(row["available_at"])[:10])
            pend = pd.Timestamp(str(row["period_end"])[:10])
        except (KeyError, ValueError):
            continue
        out.setdefault(str(row.get("ticker") or "").zfill(6), []).append((avail, pend))
    for key in out:
        out[key] = sorted(set(out[key]))
    return out


def _load_fundamental_series(path: str | Path) -> dict[str, list[dict]]:
    """ticker -> filings sorted by period_end, each with the raw field values."""
    out: dict[str, list[dict]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        try:
            rec = {
                "avail": pd.Timestamp(str(row["available_at"])[:10]),
                "period": pd.Timestamp(str(row["period_end"])[:10]),
                "fields": row.get("fields") or {},
            }
        except (KeyError, ValueError):
            continue
        out.setdefault(str(row.get("ticker") or "").zfill(6), []).append(rec)
    for key in out:
        out[key] = sorted(out[key], key=lambda r: r["period"])
    return out


def _sue_series(records: list[dict], field: str) -> list[tuple[pd.Timestamp, float]]:
    """Standardised unexpected earnings per filing, using only prior filings.

    Growth of 20% is unremarkable for a firm that always grows 20% and a large
    surprise for one that usually grows 5% -- year-over-year level (which the
    panel already carries as fund_*_yoy) cannot express that difference. The
    expectation is a seasonal random walk with drift estimated from this firm's
    own history, which is the standard construction when analyst consensus is
    unavailable.
    """
    by_period = {r["period"]: r for r in records}
    out: list[tuple[pd.Timestamp, float]] = []
    surprises: list[float] = []
    for rec in records:
        prev4 = by_period.get(rec["period"] - pd.offsets.QuarterEnd(4))
        prev8 = by_period.get(rec["period"] - pd.offsets.QuarterEnd(8))
        cur = rec["fields"].get(field)
        base = prev4["fields"].get(field) if prev4 else None
        if cur is None or base is None:
            continue
        try:
            cur_v, base_v = float(cur), float(base)
        except (TypeError, ValueError):
            continue
        drift = 0.0
        if prev8 is not None:
            older = prev8["fields"].get(field)
            try:
                drift = float(base_v) - float(older)
            except (TypeError, ValueError):
                drift = 0.0
        expected = base_v + drift
        scale = max(abs(base_v), 1e-9)
        surprise = (cur_v - expected) / scale
        if not np.isfinite(surprise):
            continue
        # standardise against this firm's own past surprises (>=4 needed)
        if len(surprises) >= 4:
            sd = float(np.std(surprises))
            sue = surprise / sd if sd > 1e-9 else 0.0
        else:
            sue = np.nan
        surprises.append(surprise)
        out.append((rec["avail"], float(np.clip(sue, -10.0, 10.0)) if np.isfinite(sue) else np.nan))
    return out


def _reaction_stats(close: np.ndarray, open_: np.ndarray, idx: int, drift_days: int) -> tuple[float, float, float] | None:
    """(next-session return, drift over `drift_days`, |gap|) around disclosure at `idx`."""
    entry = idx + 1
    if entry + drift_days >= close.shape[0]:
        return None
    c0, c1, o1 = close[idx], close[entry], open_[entry]
    if not (np.isfinite(c0) and np.isfinite(c1) and c0 > 0):
        return None
    react = c1 / c0 - 1.0
    cd = close[entry + drift_days]
    drift = (cd / c1 - 1.0) if (np.isfinite(cd) and c1 > 0) else np.nan
    gap = abs(o1 / c0 - 1.0) if (np.isfinite(o1) and c0 > 0) else np.nan
    return float(react), float(drift), float(gap)


def augment_panel_with_earnings(
    features,
    fundamental_path: str | Path,
    horizon: int = 10,
    lookback_quarters: int = 4,
    drift_days: int = 5,
    train_end: str | None = None,
):
    """Append earnings-cycle channels to a built FeaturePanel, in place."""
    disclosures = _load_disclosures(fundamental_path)
    fundamentals = _load_fundamental_series(fundamental_path)
    dates = pd.DatetimeIndex([pd.Timestamp(str(d)[:10]) for d in features.dates])
    n_dates = len(dates)
    node_count = features.features.shape[1]
    stock_count = int(features.stock_node_count or len(features.tickers))
    tickers = [str(t).zfill(6) for t in list(features.tickers)[:stock_count]]

    names = CALENDAR_NAMES + REACTION_NAMES + SUE_NAMES
    raw = np.full((n_dates, node_count, len(names)), np.nan, dtype=np.float32)
    close = np.asarray(features.close, dtype=np.float64)
    open_ = np.asarray(features.open, dtype=np.float64)
    horizon_days = int(round(horizon * 7 / 5))          # trading -> calendar days

    for si, ticker in enumerate(tickers):
        events = disclosures.get(ticker) or []
        if not events:
            continue
        recs = fundamentals.get(ticker) or []
        sue_by_field = {f: _sue_series(recs, f) for f in SUE_FIELDS}
        sue_dates = {f: pd.DatetimeIndex([d for d, _ in v]) for f, v in sue_by_field.items()}
        sue_vals = {f: np.array([x for _, x in v], dtype=np.float64) for f, v in sue_by_field.items()}
        avail = pd.DatetimeIndex([a for a, _ in events])
        pends = [p for _, p in events]
        lags = np.array([(a - p).days for a, p in events], dtype=np.float64)

        # reaction measured once per disclosure, usable only after its window closes
        reactions: list[tuple[pd.Timestamp, float, float, float]] = []
        for a in avail:
            idx = int(np.searchsorted(dates, a, side="left"))
            if idx >= n_dates:
                continue
            stats = _reaction_stats(close[:, si], open_[:, si], idx, drift_days)
            if stats is None:
                continue
            ready = dates[min(idx + 1 + drift_days, n_dates - 1)]
            reactions.append((ready, *stats))
        ready_dates = pd.DatetimeIndex([r[0] for r in reactions]) if reactions else pd.DatetimeIndex([])
        react_arr = np.array([[r[1], r[2], r[3]] for r in reactions], dtype=np.float64) if reactions else np.zeros((0, 3))

        for di in range(n_dates):
            today = dates[di]
            k = int(np.searchsorted(avail, today, side="right"))   # filings known today
            if k == 0:
                continue
            last_period = pends[k - 1]
            last_avail = avail[k - 1]
            # Annual reports get 90 statutory days, quarterlies 45, so a single
            # pooled median mis-times every Q4 by roughly a month. Estimate the
            # lag from past filings of the *same* quarter only.
            nxt_period = _next_quarter_end(last_period)
            same_q = np.array(
                [lags[i] for i in range(k) if pends[i].month == nxt_period.month],
                dtype=np.float64,
            )
            pool = same_q if same_q.size else lags[:k]
            est_lag = float(np.median(pool)) if pool.size else DEFAULT_LAG_DAYS
            if not np.isfinite(est_lag):
                est_lag = DEFAULT_LAG_DAYS
            est_next = _next_quarter_end(last_period) + pd.Timedelta(days=est_lag)
            days_until = float((est_next - today).days)
            span = max(float((est_next - last_avail).days), 1.0)
            raw[di, si, 0] = np.clip(days_until, -60.0, 200.0)
            raw[di, si, 1] = 1.0 if 0.0 <= days_until <= horizon_days else 0.0
            raw[di, si, 2] = float(np.clip((today - last_avail).days / span, 0.0, 1.5))

            j = int(np.searchsorted(ready_dates, today, side="right")) if len(ready_dates) else 0
            if j > 0:
                window = react_arr[max(0, j - lookback_quarters):j]
                with np.errstate(invalid="ignore"):
                    raw[di, si, 3] = np.nanmean(window[:, 0])
                    raw[di, si, 4] = np.nanstd(window[:, 0]) if len(window) > 1 else 0.0
                    raw[di, si, 5] = np.nanmean(window[:, 1])
                    raw[di, si, 6] = np.nanmean(window[:, 2])

            for fi, field in enumerate(SUE_FIELDS):
                sd_idx = sue_dates[field]
                if not len(sd_idx):
                    continue
                m = int(np.searchsorted(sd_idx, today, side="right"))
                if m > 0 and np.isfinite(sue_vals[field][m - 1]):
                    raw[di, si, 7 + fi] = sue_vals[field][m - 1]

    avail_mask = np.isfinite(raw)
    cut = str(train_end or features.dates[-1])[:10]
    train_rows = np.array([str(d)[:10] <= cut for d in dates])
    mean = np.zeros(len(names))
    std = np.ones(len(names))
    for ci in range(len(names)):
        seg = raw[train_rows, :stock_count, ci]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 100:
            mean[ci] = float(seg.mean())
            s = float(seg.std())
            if np.isfinite(s) and s > 1e-9:
                std[ci] = s
    norm = np.clip((raw - mean[None, None, :]) / std[None, None, :], -5.0, 5.0)
    norm = np.where(avail_mask, norm, 0.0).astype(np.float32)

    features.features = np.concatenate([features.features, norm], axis=2)
    features.raw_features = np.concatenate(
        [features.raw_features, np.where(avail_mask, raw, 0.0).astype(features.raw_features.dtype)], axis=2)
    features.available_mask = np.concatenate(
        [features.available_mask, avail_mask.astype(features.available_mask.dtype)], axis=2)
    features.feature_names = list(features.feature_names) + names
    features.train_mean = np.concatenate([features.train_mean, mean.astype(features.train_mean.dtype)])
    features.train_std = np.concatenate([features.train_std, std.astype(features.train_std.dtype)])
    cov = float(avail_mask[:, :stock_count, 0].mean())
    within = float(np.nanmean(raw[:, :stock_count, 1]))
    print(f"earnings features: channels={len(names)} coverage={cov:.3f} "
          f"in-horizon_rate={within:.3f}", flush=True)
    return features


def augment_panel_with_return_lags(features, n_lags: int = 9, train_end: str | None = None):
    """Append lagged daily returns so the encoder sees path *shape*, not just the
    nested cumulative returns (r2/r3/r5/r10) that collapse different paths to the
    same value."""
    dates = features.dates
    node_count = features.features.shape[1]
    stock_count = int(features.stock_node_count or len(features.tickers))
    r1 = np.asarray(features.returns_1d, dtype=np.float64)
    names = [f"retlag_{k}" for k in range(1, n_lags + 1)]
    raw = np.full((len(dates), node_count, n_lags), np.nan, dtype=np.float32)
    for k in range(1, n_lags + 1):
        raw[k:, :r1.shape[1], k - 1] = r1[:-k, :]

    avail_mask = np.isfinite(raw)
    cut = str(train_end or dates[-1])[:10]
    train_rows = np.array([str(d)[:10] <= cut for d in dates])
    mean = np.zeros(n_lags)
    std = np.ones(n_lags)
    for ci in range(n_lags):
        seg = raw[train_rows, :stock_count, ci]
        seg = seg[np.isfinite(seg)]
        if seg.size >= 100:
            mean[ci] = float(seg.mean())
            s = float(seg.std())
            if np.isfinite(s) and s > 1e-9:
                std[ci] = s
    norm = np.clip((raw - mean[None, None, :]) / std[None, None, :], -5.0, 5.0)
    norm = np.where(avail_mask, norm, 0.0).astype(np.float32)

    features.features = np.concatenate([features.features, norm], axis=2)
    features.raw_features = np.concatenate(
        [features.raw_features, np.where(avail_mask, raw, 0.0).astype(features.raw_features.dtype)], axis=2)
    features.available_mask = np.concatenate(
        [features.available_mask, avail_mask.astype(features.available_mask.dtype)], axis=2)
    features.feature_names = list(features.feature_names) + names
    features.train_mean = np.concatenate([features.train_mean, mean.astype(features.train_mean.dtype)])
    features.train_std = np.concatenate([features.train_std, std.astype(features.train_std.dtype)])
    print(f"return-lag features: lags={n_lags} coverage={float(avail_mask[:, :stock_count, 0].mean()):.3f}", flush=True)
    return features
