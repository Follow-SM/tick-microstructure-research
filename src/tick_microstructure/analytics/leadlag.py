"""Task 2b: lead-lag between Binance microstructure events and Polymarket quote updates.

For each Binance event with a direction, find every Polymarket rolling Up/Down market of the same
coin that is open at the event, and measure the time until its Up-token mid first moves at
least `min_tick` in the implied direction (Up for buy pressure, Down for sell pressure).

Two clocks, reported separately:
    exchange_ms  Polymarket server timestamp - Binance trade time (exchange clocks; assumes both
                 venues are NTP-synced, so treat values under ~10 ms as unresolved)
    local_ms     our receive time of that Polymarket update - our receive time of the Binance
                 event. Same clock, so this is the head start a client here actually had.
Placebo: the same measurement from random gap-free times with a random direction. If Binance
events carry information, their reprice-time CDF must sit well left of the placebo's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tick_microstructure.analytics.data import Dataset, F, MarketInfo
from tick_microstructure.analytics.features import detect_moves, detect_sweeps, onsets
from tick_microstructure.analytics.stats import placebo_times, quantiles

MAX_WAIT_MS = 60_000.0
MIN_TICK = 0.01
MOVE_WINDOW_MS = 2_000.0
MOVE_BPS = 10.0
SWEEP_USD = 100_000.0
OBT_SPIKE = 2.0
REFRACTORY_MS = 30_000.0
CDF_POINTS_MS = (100, 250, 500, 1_000, 2_000, 5_000, 10_000, 30_000, 60_000)


@dataclass
class Trigger:
    t_exchange: F  # Binance exchange time (or FollowSM server time), ms
    t_local: F  # local receive time, ms
    side: F  # +1 up pressure, -1 down pressure


def _open_updown(markets: list[MarketInfo], t_ms: float) -> list[MarketInfo]:
    s = t_ms / 1000
    return [m for m in markets if m.start is not None and m.end is not None and m.start <= s < m.end - 5]


def reprice_lags(ds: Dataset, sym: str, trig: Trigger) -> list[dict[str, float]]:
    markets = [m for m in ds.markets.values() if m.sym == sym and m.kind.startswith("updown")]
    out: list[dict[str, float]] = []
    for te, tl, side in zip(trig.t_exchange, trig.t_local, trig.side, strict=True):
        for market in _open_updown(markets, te):
            quotes = ds.quotes.get(market.tokens[0])
            if quotes is None or len(quotes.t) < 2:
                continue
            i0 = int(np.searchsorted(quotes.t, te, side="right")) - 1
            if i0 < 0:
                continue
            mid0 = quotes.mid[i0]
            j_end = int(np.searchsorted(quotes.t, te + MAX_WAIT_MS, side="right"))
            moved = np.flatnonzero((quotes.mid[i0 + 1 : j_end] - mid0) * side >= MIN_TICK - 1e-9)
            row = {"kind": 5.0 if market.kind.endswith("5m") else 15.0, "responded": 0.0}
            if moved.size:
                j = i0 + 1 + int(moved[0])
                row.update(responded=1.0, exchange_ms=quotes.t[j] - te, local_ms=quotes.rt[j] - tl)
            out.append(row)
    return out


def _summary(rows: list[dict[str, float]]) -> dict[str, Any]:
    responded = [r for r in rows if r["responded"]]
    exch = np.asarray([r["exchange_ms"] for r in responded])
    local = np.asarray([r["local_ms"] for r in responded])
    n = len(rows)
    return {
        "pairs": n,
        "responded": len(responded),
        "exchange_ms": quantiles(exch),
        "local_ms": quantiles(local),
        "cdf_local": {str(x): float((local <= x).sum() / n) if n else None for x in CDF_POINTS_MS},
        "local_lags": sorted(local.tolist()),
    }


def triggers(ds: Dataset, sym: str) -> dict[str, Trigger]:
    out: dict[str, Trigger] = {}
    trades = ds.trades.get(sym)
    if trades is not None and len(trades.t):
        moves = detect_moves(trades, MOVE_WINDOW_MS, MOVE_BPS, REFRACTORY_MS)
        out["binance_move"] = Trigger(moves.t, moves.rt, moves.side)
        sweeps = detect_sweeps(trades, SWEEP_USD)
        out["binance_sweep"] = Trigger(sweeps.t, sweeps.rt, sweeps.side)
    tox = ds.toxicity.get(sym)
    if tox is not None and len(tox.t):
        idx = onsets(tox.t, tox.obt > OBT_SPIKE, REFRACTORY_MS)
        out["followsm_obt_spike"] = Trigger(tox.ts[idx], tox.t[idx], -np.ones(len(idx)))
    return out


def lead_lag(ds: Dataset, seed: int = 7) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}
    for sym in sorted({m.sym for m in ds.markets.values()}):
        per: dict[str, Any] = {}
        for name, trig in triggers(ds, sym).items():
            gap_key = "followsm" if name.startswith("followsm") else "binance"
            keep = np.array(
                [
                    not ds.gaps[gap_key].overlaps(t, t + MAX_WAIT_MS)
                    and not ds.gaps["polymarket"].overlaps(t, t + MAX_WAIT_MS)
                    for t in trig.t_local
                ],
                dtype=bool,
            )
            trig = Trigger(trig.t_exchange[keep], trig.t_local[keep], trig.side[keep])
            per[name] = {"events": len(trig.side), **_summary(reprice_lags(ds, sym, trig))}
        n_placebo = max(200, max((v["events"] for v in per.values()), default=0))
        times = placebo_times(rng, n_placebo, ds.start_ms, ds.end_ms, ds.gaps["all"], MAX_WAIT_MS)
        sides = rng.choice(np.array([-1.0, 1.0]), size=len(times))
        per["placebo"] = {
            "events": len(times),
            **_summary(reprice_lags(ds, sym, Trigger(times, times, sides))),
        }
        report[sym] = per
    return report
