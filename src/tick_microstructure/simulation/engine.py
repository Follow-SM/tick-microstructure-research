"""Task 4: quote-cancellation simulator on recorded Polymarket Up/Down books.

Maker models (deliberately simple, stated so results can be judged), run on the same prints:
  * join_best (the naive control): one unit quote joined at the best bid AND best ask of every
    tracked Up/Down token, i.e. centred on Polymarket's own mid;
  * fair_value: quotes at a Binance-implied P(Up) ± FV_HALF_SPREAD (simulation/fair_value.py),
    computed FV_QUOTE_LATENCY_MS before each print and never crossing the recorded book, so it stays
    a maker. In the equity study (research/polymarket_study.py + the hedger's backtest) the
    mid-centred maker was the one informed takers picked off; a breaker should be judged on both.
  * a recorded taker trade fills us when it prints at or through our price (taker SELL at <= our
    bid fills our bid, taker BUY at >= our ask fills our ask), for min(quote size, trade size);
    queue position is ignored, so fill counts are an upper bound;
  * markout of a fill = (mid H later - fill price) for a maker buy, (fill price - mid H later)
    for a maker sell, in USD per share; `settle` marks out against the market's outcome, taken from
    Binance's price at the window end (a proxy where the market settles on Chainlink).
Breaker: quotes are pulled while a trigger interval is active. The whole window is shifted by the
    latency budget (trigger receive time + decision latency + OMS latency): cancels land that late,
    and re-quotes after the trigger clears go out that late too. A fill is avoided if it printed
    inside the shifted window. Fill times are mapped to the local clock with the median
    Polymarket feed offset (receive - server time).
Accounting per trigger: slippage avoided (losing fills skipped), spread capture missed (winning
    fills skipped), net = avoided - missed. Placebo: the same intervals circularly shifted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.data import Dataset, F
from tick_microstructure.analytics.stats import quantiles
from tick_microstructure.research.signals import Intervals, TriggerSpec, build_triggers
from tick_microstructure.simulation.fair_value import FairValue

MARKOUT_S = (5, 30, 60)
PRIMARY_MARKOUT_S = 30
CANCEL_LATENCIES_MS = (0.0, 50.0, 100.0, 250.0, 500.0, 1_000.0, 5_000.0, 15_000.0, 60_000.0)
DEFAULT_CANCEL_MS = 100.0
QUOTE_SIZE = 100.0
PLACEBO_SHIFTS = 100
QUOTERS = ("join_best", "fair_value")
FV_HALF_SPREAD = 0.02
FV_QUOTE_LATENCY_MS = 250.0
PM_TICK = 0.001


@dataclass
class Fills:
    sym: str
    t_local: F  # fill time mapped to the local clock
    t_server: F
    side: F  # +1 maker bought, -1 maker sold
    price: F
    size: F
    spread_capture: F  # |fill price - mid at fill| per share
    markout: dict[int, F]  # horizon s -> USD per share
    settle: F  # USD per share against the outcome; NaN if the window's outcome is unknown


def collect_fills(ds: Dataset, sym: str, quoter: str = "join_best") -> Fills:
    t_loc: list[float] = []
    t_srv: list[float] = []
    side: list[float] = []
    price: list[float] = []
    size: list[float] = []
    capture: list[float] = []
    settle: list[float] = []
    marks: dict[int, list[float]] = {h: [] for h in MARKOUT_S}
    offsets = [q.rt - q.t for q in ds.quotes.values() if q.t.size]
    offset = float(np.median(np.concatenate(offsets))) if offsets else 0.0
    fv = FairValue(ds.trades[sym]) if sym in ds.trades and ds.trades[sym].t.size else None
    if quoter == "fair_value" and fv is None:
        return _fills(sym, t_loc, t_srv, side, price, size, capture, settle, marks)
    for market in ds.markets.values():
        if market.sym != sym or not market.kind.startswith("updown"):
            continue
        start, end = market.start, market.end
        up_outcome = (
            fv.settle_up(start, end) if fv is not None and start is not None and end is not None else math.nan
        )
        for outcome, tok in enumerate(market.tokens):
            q, tr = ds.quotes.get(tok), ds.pm_trades.get(tok)
            if q is None or tr is None or q.t.size < 2 or tr.t.size == 0:
                continue
            idx = np.searchsorted(q.t, tr.t, side="left") - 1  # book just before the print
            centre = np.full(tr.t.size, np.nan)
            if quoter == "fair_value":
                if fv is None or start is None or end is None:
                    continue
                p_up = fv.up(tr.t - FV_QUOTE_LATENCY_MS, start, end)
                centre = p_up if outcome == 0 else 1 - p_up
            payout = up_outcome if outcome == 0 else 1 - up_outcome
            for k in np.flatnonzero(idx >= 0):
                i = int(idx[k])
                bid, ask, mid = q.bid[i], q.ask[i], q.mid[i]
                if quoter == "fair_value":
                    c = float(centre[k])
                    if not math.isfinite(c):
                        continue
                    # never cross the recorded book, never quote at 0 or 1
                    bid = max(
                        min(math.floor((c - FV_HALF_SPREAD) / PM_TICK) * PM_TICK, ask - PM_TICK), PM_TICK
                    )
                    ask = min(
                        max(math.ceil((c + FV_HALF_SPREAD) / PM_TICK) * PM_TICK, q.bid[i] + PM_TICK),
                        1 - PM_TICK,
                    )
                if tr.side[k] < 0 and tr.price[k] <= bid + 1e-9:
                    maker_side, fill_px = 1.0, bid
                elif tr.side[k] > 0 and tr.price[k] >= ask - 1e-9:
                    maker_side, fill_px = -1.0, ask
                else:
                    continue
                ts = float(tr.t[k])
                if ds.gaps["polymarket"].overlaps(ts + offset, ts + offset + max(MARKOUT_S) * 1000):
                    continue
                t_srv.append(ts)
                t_loc.append(ts + offset)
                side.append(maker_side)
                price.append(fill_px)
                size.append(min(QUOTE_SIZE, float(tr.size[k])))
                capture.append(abs(fill_px - mid))
                settle.append(maker_side * (payout - fill_px))
                for h in MARKOUT_S:
                    j = int(np.searchsorted(q.t, ts + h * 1000, side="right")) - 1
                    later = q.mid[j] if market.end is None or ts + h * 1000 < market.end * 1000 else np.nan
                    marks[h].append(maker_side * (later - fill_px))
    return _fills(sym, t_loc, t_srv, side, price, size, capture, settle, marks)


def _fills(
    sym: str,
    t_loc: list[float],
    t_srv: list[float],
    side: list[float],
    price: list[float],
    size: list[float],
    capture: list[float],
    settle: list[float],
    marks: dict[int, list[float]],
) -> Fills:
    order = np.argsort(np.asarray(t_loc), kind="stable")

    def arr(values: list[float]) -> F:
        return np.asarray(values, dtype=np.float64)[order]

    return Fills(
        sym,
        arr(t_loc),
        arr(t_srv),
        arr(side),
        arr(price),
        arr(size),
        arr(capture),
        {h: arr(v) for h, v in marks.items()},
        arr(settle),
    )


def decision_latency_us(ds: Dataset, sym: str, spec: TriggerSpec, samples: int = 20_000) -> dict[str, Any]:
    """Measured in-process cost of the evaluator step on recorded frames (µs)."""
    tox = ds.toxicity.get(sym)
    if tox is None or tox.t.size == 0:
        return {"n": 0}
    n = min(samples, tox.t.size)
    frames = [(float(tox.vpin_pct[i]), float(tox.obt[i])) for i in range(n)]
    costs = np.empty(n)
    for i, (vp, obt) in enumerate(frames):
        t0 = time.perf_counter_ns()
        _ = (vp == vp and vp >= spec.vpin_pct) or obt > spec.obt
        costs[i] = (time.perf_counter_ns() - t0) / 1_000
    return quantiles(costs)


def _account(fills: Fills, avoided: npt.NDArray[np.bool_], h: int | str) -> dict[str, float]:
    m = (fills.settle if h == "settle" else fills.markout[int(h)]) * fills.size
    ok = np.isfinite(m)
    mm, av = m[ok], avoided[ok]
    return {
        "fills": float(ok.sum()),
        "fills_avoided": float(av.sum()),
        "static_pnl": float(mm.sum()),
        "breaker_pnl": float(mm[~av].sum()),
        "slippage_avoided": float(-mm[av & (mm < 0)].sum()),
        "spread_missed": float(mm[av & (mm > 0)].sum()),
        "net_value": float(-mm[av].sum()),
    }


def _evaluate(
    ds: Dataset,
    sym: str,
    fills: Fills,
    spec: TriggerSpec,
    decision_ms: float,
    rng: np.random.Generator,
    placebo_shifts: int,
) -> dict[str, Any]:
    span = (ds.start_ms, ds.end_ms)
    none = np.zeros(fills.t_local.size, dtype=bool)
    out: dict[str, Any] = {
        "fills": int(fills.t_local.size),
        "static_by_markout": {str(h): _account(fills, none, h)["static_pnl"] for h in (*MARKOUT_S, "settle")},
        "triggers": {},
    }
    for name, iv in build_triggers(ds, sym, spec).items():
        iv = iv.clip(*span)
        by_latency = {}
        for cancel_ms in CANCEL_LATENCIES_MS:
            eff = iv.delayed(decision_ms + cancel_ms)
            by_latency[str(cancel_ms)] = _account(fills, eff.contains(fills.t_local), PRIMARY_MARKOUT_S)
        eff = iv.delayed(decision_ms + DEFAULT_CANCEL_MS)
        avoided = eff.contains(fills.t_local)
        shifts = rng.uniform(0.05, 0.95, size=placebo_shifts) * (span[1] - span[0])
        placebo = [
            _account(
                fills,
                iv.shifted(s, *span).delayed(decision_ms + DEFAULT_CANCEL_MS).contains(fills.t_local),
                PRIMARY_MARKOUT_S,
            )
            for s in shifts
        ]
        actual = _account(fills, avoided, PRIMARY_MARKOUT_S)
        out["triggers"][name] = {
            "active_share": iv.active_ms() / (span[1] - span[0]),
            "actual": actual,
            "by_markout": {str(h): _account(fills, avoided, h) for h in (*MARKOUT_S, "settle")},
            "by_cancel_latency_ms": by_latency,
            "placebo_net_value": quantiles(np.asarray([p["net_value"] for p in placebo])),
            "placebo_share_at_least_as_good": float(
                np.mean([p["net_value"] >= actual["net_value"] for p in placebo])
            ),
        }
    return out


def simulate(
    ds: Dataset, spec: TriggerSpec | None = None, seed: int = 23, placebo_shifts: int = PLACEBO_SHIFTS
) -> dict[str, Any]:
    """Per symbol: the join_best control at the top level (as before) and the fair_value quoter under
    `fair_value`, each with static P&L by markout and the same trigger/placebo/latency evaluation."""
    spec = spec or TriggerSpec()
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {
        "quote_size": QUOTE_SIZE,
        "primary_markout_s": PRIMARY_MARKOUT_S,
        "fair_value_half_spread": FV_HALF_SPREAD,
        "fair_value_quote_latency_ms": FV_QUOTE_LATENCY_MS,
        "symbols": {},
    }
    for sym in sorted({m.sym for m in ds.markets.values()}):
        fills = collect_fills(ds, sym)
        if fills.t_local.size == 0:
            continue
        stream_lag = ds.toxicity[sym].t - ds.toxicity[sym].ts if sym in ds.toxicity else np.empty(0)
        decision = decision_latency_us(ds, sym, spec)
        decision_ms = (decision.get("p99") or 0.0) / 1000
        per: dict[str, Any] = {
            "latency_budget": {
                "stream_arrival_lag_ms": quantiles(stream_lag),
                "evaluator_decision_us": decision,
                "oms_cancel_ms": "assumed, see sensitivity (no live orders are placed)",
            },
            **_evaluate(ds, sym, fills, spec, decision_ms, rng, placebo_shifts),
        }
        fv_fills = collect_fills(ds, sym, "fair_value")
        if fv_fills.t_local.size:
            per["fair_value"] = _evaluate(ds, sym, fv_fills, spec, decision_ms, rng, placebo_shifts)
        report["symbols"][sym] = per
    return report


def pnl_paths(
    ds: Dataset,
    sym: str,
    trigger: str,
    spec: TriggerSpec | None = None,
    placebo_shifts: int = 20,
    seed: int = 29,
) -> dict[str, F] | None:
    """Cumulative static vs breaker vs median-placebo P&L (primary markout) over fill time."""
    spec = spec or TriggerSpec()
    iv = build_triggers(ds, sym, spec).get(trigger)
    fills = collect_fills(ds, sym)
    if iv is None or fills.t_local.size < 2:
        return None
    span = (ds.start_ms, ds.end_ms)
    iv = iv.clip(*span)
    m = np.nan_to_num(fills.markout[PRIMARY_MARKOUT_S] * fills.size)

    def path(intervals: Intervals) -> F:
        skipped = intervals.delayed(DEFAULT_CANCEL_MS).contains(fills.t_local)
        return np.cumsum(np.where(skipped, 0.0, m))

    rng = np.random.default_rng(seed)
    shifts = rng.uniform(0.05, 0.95, size=placebo_shifts) * (span[1] - span[0])
    placebo = np.vstack([path(iv.shifted(s, *span)) for s in shifts])
    return {
        "t": fills.t_local,
        "static": np.cumsum(m),
        "breaker": path(iv),
        "placebo": np.median(placebo, axis=0),
    }
