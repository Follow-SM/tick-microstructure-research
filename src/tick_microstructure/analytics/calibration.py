"""Task 2a: distribution profiling on rolling 24-hour windows (p50 / p90 / p95 / p99).

Per symbol:
  * ob_toxicity_1pct and the +-1% depth (FollowSM toxicity stream)
  * vpin_percentile coverage (share of frames with a percentile)
  * per-second taker notional and its z-score vs a 5-minute EWMA (Binance ticks)
  * FollowSM delivery: arrival lag (local receive - server timestamp) and inter-frame interval
  * cross-venue prediction divergence (confluence stream): how often the Polymarket-implied 15 m
    direction (direction x prob_delta_15m) disagrees with the Binance 15 m move, their
    correlation, and the smart-money whale-sweep distribution

Windows are consecutive 24-hour blocks ending at the last sample; a trailing partial block is
reported with its real length so short samples are never mistaken for full days.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from tick_microstructure.analytics.data import Dataset, F
from tick_microstructure.analytics.features import taker_flow
from tick_microstructure.analytics.stats import quantiles

DAY_MS = 86_400_000.0


def _windows(start: float, end: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    hi = end
    while hi > start:
        lo = max(start, hi - DAY_MS)
        out.append((lo, hi))
        hi = lo
    return out[::-1]


def _slice(t: F, values: F, lo: float, hi: float) -> F:
    return values[(t >= lo) & (t < hi)]


def calibrate(ds: Dataset) -> dict[str, Any]:
    symbols = sorted(set(ds.toxicity) | set(ds.trades) | set(ds.confluence))
    report: dict[str, Any] = {
        "span_hours": ds.hours,
        "gap_hours": {k: g.total_ms() / 3_600_000 for k, g in ds.gaps.items()},
        "events": ds.event_counts,
        "symbols": {},
    }
    for sym in symbols:
        per_window: list[dict[str, Any]] = []
        tox = ds.toxicity.get(sym)
        flow = taker_flow(ds.trades[sym]) if sym in ds.trades else None
        for lo, hi in _windows(ds.start_ms, ds.end_ms):
            row: dict[str, Any] = {"from_ms": lo, "to_ms": hi, "hours": (hi - lo) / 3_600_000}
            if tox is not None and len(tox.t):
                row["ob_toxicity_1pct"] = quantiles(_slice(tox.t, tox.obt, lo, hi))
                row["bid_1pct_usd"] = quantiles(_slice(tox.t, tox.bid_1pct, lo, hi))
                row["ask_1pct_usd"] = quantiles(_slice(tox.t, tox.ask_1pct, lo, hi))
                vp = _slice(tox.t, tox.vpin_pct, lo, hi)
                row["vpin_percentile_coverage"] = float(np.isfinite(vp).mean()) if vp.size else None
                row["share_obt_gt_2"] = (
                    float((_slice(tox.t, tox.obt, lo, hi) > 2.0).mean()) if vp.size else None
                )
            if flow is not None and len(flow.t):
                row["taker_notional_per_sec"] = quantiles(_slice(flow.t, flow.total, lo, hi))
                row["taker_flow_z"] = quantiles(_slice(flow.t, flow.z, lo, hi))
            per_window.append(row)
        delivery: dict[str, Any] = {}
        conf = ds.confluence.get(sym)
        if conf is not None and conf.t.size:
            implied = conf.ev_direction * conf.ev_prob_delta_15m  # >0: Polymarket implies spot up
            spot = np.interp(conf.ev_t, conf.t, conf.spot_delta_15m)
            directional = conf.ev_direction != 0
            both = directional & (np.abs(implied) > 0) & (np.abs(spot) > 0)
            report_conf: dict[str, Any] = {
                "frames": int(conf.t.size),
                "divergence_flag_share": float(conf.divergence.mean()),
                "spot_delta_15m": quantiles(np.abs(conf.spot_delta_15m)),
                "prob_delta_15m": quantiles(np.abs(conf.ev_prob_delta_15m)),
                "whale_sweeps_1h_usdt": quantiles(conf.ev_whale_1h),
                "sign_disagreement_share": float((np.sign(implied[both]) != np.sign(spot[both])).mean())
                if both.any()
                else None,
                "implied_vs_spot_corr": float(np.corrcoef(implied[both], spot[both])[0, 1])
                if both.sum() > 2
                else None,
            }
            delivery["confluence_arrival_lag_ms"] = quantiles(conf.t - conf.ts)
        else:
            report_conf = {}
        if tox is not None and len(tox.t) > 1:
            delivery["arrival_lag_ms"] = quantiles(tox.t - tox.ts)
            delivery["frame_interval_ms"] = quantiles(np.diff(tox.t))
        if sym in ds.trades and len(ds.trades[sym].t):
            trades = ds.trades[sym]
            delivery["binance_trade_lag_ms"] = quantiles(trades.rt - trades.t)
        report["symbols"][sym] = {"windows": per_window, "delivery": delivery, "confluence": report_conf}
    return report
