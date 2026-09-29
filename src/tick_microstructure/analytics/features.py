"""Tick features from Binance ground truth: taker-flow spikes, sweeps, price-move onsets."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.data import F, Trades

EWMA_HALFLIFE_SECS = 300.0  # "relative to a 5-minute EWMA"


@dataclass
class TakerFlow:
    """Per-second taker notional and its z-score vs a 5-minute EWMA (state before the update)."""

    t: F  # bucket start, ms (exchange clock)
    buy: F
    sell: F
    z: F  # NaN during the first EWMA half-life

    @property
    def total(self) -> F:
        return self.buy + self.sell


def taker_flow(trades: Trades, halflife_secs: float = EWMA_HALFLIFE_SECS) -> TakerFlow:
    if len(trades.t) == 0:
        empty = np.empty(0)
        return TakerFlow(empty, empty, empty, empty)
    sec = (trades.t // 1000).astype(np.int64)
    first = int(sec[0])
    idx = sec - first
    n = int(idx[-1]) + 1
    notional = trades.notional
    buy = np.bincount(idx, weights=np.where(trades.side > 0, notional, 0.0), minlength=n)
    sell = np.bincount(idx, weights=np.where(trades.side < 0, notional, 0.0), minlength=n)
    total = buy + sell
    alpha = 1.0 - 2.0 ** (-1.0 / halflife_secs)
    z = np.full(n, np.nan)
    mean, var = float(total[0]), 0.0
    warm = int(halflife_secs)
    for i in range(1, n):
        x = float(total[i])
        if i >= warm and var > 0:
            z[i] = (x - mean) / math.sqrt(var)
        diff = x - mean
        mean += alpha * diff
        var = (1 - alpha) * (var + alpha * diff * diff)
    t = (np.arange(n, dtype=np.float64) + first) * 1000.0
    return TakerFlow(t, buy, sell, z)


@dataclass
class Sweeps:
    """Same-side taker fills sharing one exchange timestamp (one matching event), >= a notional floor."""

    t: F  # exchange time, ms
    rt: F  # local receive time of the last fill in the group
    side: F  # +1 buy sweep (lifts asks), -1 sell sweep (hits bids)
    notional: F
    levels: F  # distinct prices consumed
    move_bps: F  # first -> last fill price, signed by side


def detect_sweeps(trades: Trades, min_notional: float) -> Sweeps:
    empty = np.empty(0)
    if len(trades.t) == 0:
        return Sweeps(empty, empty, empty, empty, empty, empty)
    boundary = np.ones(len(trades.t), dtype=bool)
    boundary[1:] = (trades.t[1:] != trades.t[:-1]) | (trades.side[1:] != trades.side[:-1])
    starts = np.flatnonzero(boundary)
    ends = np.append(starts[1:], len(trades.t))
    notional = np.add.reduceat(trades.notional, starts)
    keep = notional >= min_notional
    rows = [(s, e) for s, e, k in zip(starts, ends, keep, strict=True) if k]
    if not rows:
        return Sweeps(empty, empty, empty, empty, empty, empty)
    t, rt, side, levels, move = [], [], [], [], []
    for s, e in rows:
        t.append(trades.t[s])
        rt.append(float(np.max(trades.rt[s:e])))
        side.append(trades.side[s])
        levels.append(float(len(np.unique(trades.price[s:e]))))
        move.append(float((trades.price[e - 1] / trades.price[s] - 1) * 1e4 * trades.side[s]))
    return Sweeps(
        np.asarray(t),
        np.asarray(rt),
        np.asarray(side),
        notional[keep].astype(np.float64),
        np.asarray(levels),
        np.asarray(move),
    )


@dataclass
class Moves:
    """Onsets of |log-return over the trailing window| >= threshold, from trade prices."""

    t: F  # exchange time of the crossing trade, ms
    rt: F
    side: F  # +1 up, -1 down
    bps: F


def detect_moves(trades: Trades, window_ms: float, min_bps: float, refractory_ms: float) -> Moves:
    empty = np.empty(0)
    if len(trades.t) < 2:
        return Moves(empty, empty, empty, empty)
    ref = np.searchsorted(trades.t, trades.t - window_ms, side="left")
    ret_bps = np.log(trades.price / trades.price[ref]) * 1e4
    candidates = np.flatnonzero(np.abs(ret_bps) >= min_bps)
    chosen: list[int] = []
    last = -np.inf
    for i in candidates:
        if trades.t[i] - last >= refractory_ms:
            chosen.append(int(i))
            last = trades.t[i]
    idx = np.asarray(chosen, dtype=np.int64)
    return Moves(trades.t[idx], trades.rt[idx], np.sign(ret_bps[idx]), ret_bps[idx])


def onsets(times: F, active: npt.NDArray[np.bool_], refractory_ms: float) -> npt.NDArray[np.int64]:
    """Indices where `active` switches on, at least `refractory_ms` after the previous onset."""
    chosen: list[int] = []
    last = -np.inf
    prev = False
    for i, on in enumerate(active):
        if on and not prev and times[i] - last >= refractory_ms:
            chosen.append(i)
            last = times[i]
        prev = bool(on)
    return np.asarray(chosen, dtype=np.int64)
