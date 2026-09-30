"""Binance-implied fair value for Polymarket crypto Up/Down markets: the centre of the fair-value quoter.

    P(Up) = Φ((ln(S/K) − σ²/2) / σ)       (the hedger's digital model, zero drift)
    S  = last Binance trade at or before t (NaN if older than STALE_MS)
    K  = last Binance trade at or before the window start (the market's opening price)
    σ² = realised variance of 1-second log returns over the trailing VOL_WINDOW_S, per second,
         times the seconds left to the window end

Point-in-time: only trades before t are used. A proxy: some Up/Down markets settle on Chainlink, not
Binance, so Binance's price at the window end is also only a proxy for the settlement.
"""

from __future__ import annotations

import math

import numpy as np

from tick_microstructure.analytics.data import F, Trades

STALE_MS = 10_000.0
VOL_WINDOW_S = 1_800
MIN_VOL_SHARE = 0.5  # of VOL_WINDOW_S seconds with a return, else no fair value

_erf = np.vectorize(math.erf, otypes=[np.float64])


class FairValue:
    def __init__(self, trades: Trades) -> None:
        self.t, self.p = trades.t, trades.price
        self.t0 = int(trades.t[0] // 1000) if trades.t.size else 0
        n = int(trades.t[-1] // 1000) - self.t0 + 1 if trades.t.size else 0
        ends = (self.t0 + np.arange(n) + 1) * 1000.0  # end of each second
        px = self.p[np.searchsorted(self.t, ends, side="right") - 1]
        r2 = np.concatenate([[np.nan], np.diff(np.log(px))])[:n] ** 2
        ok = np.isfinite(r2)
        self.cum_sq = np.concatenate([[0.0], np.cumsum(np.where(ok, r2, 0.0))])
        self.cum_n = np.concatenate([[0.0], np.cumsum(ok)])

    def price(self, t_ms: F) -> F:
        if not self.t.size:
            return np.full(t_ms.shape, np.nan)
        j = np.searchsorted(self.t, t_ms, side="right") - 1
        jj = np.maximum(j, 0)
        fresh = (j >= 0) & (t_ms - self.t[jj] <= STALE_MS)
        return np.where(fresh, self.p[jj], np.nan)

    def var_per_s(self, t_ms: F) -> F:
        """Mean squared 1-second log return over the completed seconds in the trailing window."""
        n = self.cum_n.size - 1
        hi = np.clip(np.floor(t_ms / 1000).astype(np.int64) - self.t0, 0, n)
        lo = np.clip(hi - VOL_WINDOW_S, 0, n)
        obs = self.cum_n[hi] - self.cum_n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (self.cum_sq[hi] - self.cum_sq[lo]) / obs
        return np.where(obs >= MIN_VOL_SHARE * VOL_WINDOW_S, var, np.nan)

    def up(self, t_ms: F, start_s: float, end_s: float) -> F:
        """P(Up) at each time; NaN outside [start, end) or without fresh prices and volatility."""
        strike = float(self.price(np.array([start_s * 1000]))[0])
        var = self.var_per_s(t_ms) * (end_s * 1000 - t_ms) / 1000
        with np.errstate(invalid="ignore", divide="ignore"):
            sd = np.sqrt(var)
            d = (np.log(self.price(t_ms) / strike) - var / 2) / sd
        inside = (t_ms >= start_s * 1000) & (t_ms < end_s * 1000) & (sd > 0)
        return np.where(inside & np.isfinite(d), 0.5 * (1 + _erf(d / math.sqrt(2))), np.nan)

    def settle_up(self, start_s: float, end_s: float) -> float:
        """1.0 if Binance closed the window above its opening price, 0.0 below, NaN if unknown."""
        k, s = self.price(np.array([start_s * 1000, end_s * 1000]))
        if not (math.isfinite(k) and math.isfinite(s)) or s == k or self.t[-1] < end_s * 1000:
            return float("nan")
        return 1.0 if s > k else 0.0
