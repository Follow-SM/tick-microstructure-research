"""Features, intervals and the simulator on synthetic data. Offline."""

from __future__ import annotations

import numpy as np
import pytest

from tick_microstructure.analytics.data import (
    Book,
    Dataset,
    GapMask,
    MarketInfo,
    PMTrades,
    Quotes,
    Toxicity,
    Trades,
)
from tick_microstructure.analytics.features import detect_moves, detect_sweeps, taker_flow
from tick_microstructure.analytics.stats import wilson
from tick_microstructure.research.signals import Intervals, state_intervals
from tick_microstructure.simulation.engine import simulate


def _trades(t: list[float], price: list[float], qty: list[float], side: list[float]) -> Trades:
    arr = np.asarray
    return Trades(arr(t, float), arr(t, float) + 5, arr(price, float), arr(qty, float), arr(side, float))


def test_detect_sweeps_groups_one_matching_event() -> None:
    tr = _trades(
        [1000, 1000, 1000, 1000, 2000],
        [100.0, 100.1, 100.2, 100.2, 100.0],
        [500, 500, 500, 10, 2000],
        [1, 1, 1, -1, -1],
    )
    sw = detect_sweeps(tr, 100_000)
    assert sw.t.tolist() == [1000.0, 2000.0]
    assert sw.side.tolist() == [1.0, -1.0]
    assert sw.notional[0] == pytest.approx(150_150)
    assert sw.levels[0] == 3 and sw.move_bps[0] == pytest.approx(20, rel=0.01)


def test_detect_moves_onset_and_refractory() -> None:
    t = np.arange(0, 10_000, 100, dtype=float)
    price = np.full(t.size, 100.0)
    price[30:] = 99.8  # -20 bps at t = 3000
    tr = _trades(t.tolist(), price.tolist(), [1.0] * t.size, [1.0] * t.size)
    moves = detect_moves(tr, window_ms=2000, min_bps=10, refractory_ms=5000)
    assert moves.t.tolist() == [3000.0] and moves.side.tolist() == [-1.0]


def test_taker_flow_zscore_flags_spike_without_lookahead() -> None:
    rng = np.random.default_rng(0)
    n = 900
    t = np.arange(n) * 1000.0
    qty = rng.uniform(0.9, 1.1, n)
    qty[850] = 50.0
    tr = _trades(t.tolist(), [100.0] * n, qty.tolist(), [1.0] * n)
    flow = taker_flow(tr)
    assert np.isnan(flow.z[:299]).all()
    assert flow.z[850] > 20 and abs(np.nanmedian(flow.z)) < 1.5


def test_intervals_state_union_contains_shift() -> None:
    t = np.array([0.0, 1000, 2000, 3000, 4000, 9000])
    on = np.array([False, True, True, False, True, True])
    iv = state_intervals(t, on, stale_ms=3000)
    # frame gap 4000 -> 9000 exceeds stale_ms: the second interval closes at 4000 + 3000
    assert iv.starts.tolist() == [1000, 4000, 9000] and iv.ends.tolist() == [3000, 7000, 12000]
    assert iv.contains(np.array([500.0, 1500, 7500])).tolist() == [False, True, False]
    shifted = iv.shifted(5000, 0, 20000)
    assert shifted.active_ms() == pytest.approx(iv.active_ms())
    u = Intervals.merge(np.array([0.0]), np.array([10.0])).union(
        Intervals.merge(np.array([5.0]), np.array([20.0]))
    )
    assert (u.starts.tolist(), u.ends.tolist()) == ([0.0], [20.0])


def test_wilson_interval() -> None:
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(0.404, abs=1e-3) and hi == pytest.approx(0.596, abs=1e-3)


def _synthetic(signal_leads_moves: bool) -> Dataset:
    """One Up/Down token, 2 h. Every 5 min the fair price drops 5 cents: a taker SELL hits the bid
    first (adverse for the maker), then the mid reprices. Between shocks, random-side fills that
    earn the spread. The FollowSM `obt` state goes toxic 500 ms before each shock if informative;
    the adverse fill reaches us 50 ms after its server time."""
    rng = np.random.default_rng(1)
    start, hours = 1_800_000_000_000.0, 2
    end = start + hours * 3_600_000
    shocks = np.arange(start + 300_000, end - 120_000, 300_000)
    q_t: list[float] = []
    q_b: list[float] = []
    q_a: list[float] = []
    tr_t: list[float] = []
    tr_p: list[float] = []
    tr_s: list[float] = []
    mid = 0.5
    t = start
    for shock in shocks.tolist():
        while t < shock - 2_000:
            q_t.append(t)
            q_b.append(mid - 0.005)
            q_a.append(mid + 0.005)
            side = float(rng.choice([-1.0, 1.0]))
            tr_t.append(t + 1)
            tr_p.append(mid - 0.005 if side < 0 else mid + 0.005)
            tr_s.append(side)
            t += 20_000
        tr_t.append(shock)
        tr_p.append(mid - 0.005)
        tr_s.append(-1.0)
        mid -= 0.05
        q_t.append(shock + 500)
        q_b.append(mid - 0.005)
        q_a.append(mid + 0.005)
        t = shock + 5_000  # after the trigger window, so no winning fill is skipped
    q_t.append(end)
    q_b.append(mid - 0.005)
    q_a.append(mid + 0.005)
    arr = np.asarray
    quotes = Quotes(arr(q_t), arr(q_t) + 50, arr(q_b), arr(q_a))
    pm = PMTrades(arr(tr_t), arr(tr_t) + 50, arr(tr_p), np.full(len(tr_t), 100.0), arr(tr_s))
    ft = np.arange(start, end, 250.0)
    obt = np.ones(ft.size)
    toxic_at = shocks - 500 if signal_leads_moves else shocks + 150_000
    for s in toxic_at:
        obt[(ft >= s) & (ft < s + 3_000)] = 3.0
    nan = np.full(ft.size, np.nan)
    tox = Toxicity(
        ft, ft - 40, np.full(ft.size, 84_000.0), nan, nan, obt, nan, nan, nan, nan, nan, nan, nan, nan, nan
    )
    market = MarketInfo(
        "btc-updown-15m-x", "BTCUSDT", "updown_15m", ["up", "down"], start / 1000, end / 1000 + 600
    )
    empty_book = Book(np.empty(0), np.empty(0), np.empty(0), np.empty(0), np.empty(0))
    return Dataset(
        start_ms=start,
        end_ms=end,
        trades={},
        book={"BTCUSDT": empty_book},
        toxicity={"BTCUSDT": tox},
        confluence={},
        quotes={"up": quotes},
        pm_trades={"up": pm},
        markets={market.slug: market},
        gaps={k: GapMask() for k in ("followsm", "binance", "polymarket", "all")},
        event_counts={},
    )


def test_simulator_rewards_informative_trigger_over_placebo() -> None:
    good = simulate(_synthetic(True), placebo_shifts=30)["symbols"]["BTCUSDT"]["triggers"]["obt"]
    assert good["actual"]["slippage_avoided"] > 0 and good["actual"]["spread_missed"] == 0
    assert good["placebo_share_at_least_as_good"] < 0.1
    late = simulate(_synthetic(False), placebo_shifts=30)["symbols"]["BTCUSDT"]["triggers"]["obt"]
    assert late["actual"]["slippage_avoided"] == 0


def test_simulator_cancel_latency_can_miss_the_window() -> None:
    res = simulate(_synthetic(True), placebo_shifts=5)["symbols"]["BTCUSDT"]["triggers"]["obt"]
    by_latency = res["by_cancel_latency_ms"]
    assert by_latency["0.0"]["slippage_avoided"] > 0
    assert by_latency["500.0"]["slippage_avoided"] > 0  # 500 ms lead + 50 ms feed offset
    assert by_latency["1000.0"]["slippage_avoided"] == 0  # cancel lands after the fill
