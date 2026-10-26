"""Binance-implied fair value and the fair-value quoter vs the join-best control. Offline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tick_microstructure.analytics.data import (
    Dataset,
    GapMask,
    MarketInfo,
    PMTrades,
    Quotes,
    Trades,
)
from tick_microstructure.analytics.resolutions import CACHE_NAME, load_resolutions
from tick_microstructure.simulation.engine import PRIMARY_MARKOUT_S, collect_fills, settlement_check
from tick_microstructure.simulation.fair_value import VOL_WINDOW_S, FairValue

T0 = 1_800_000_000_000.0
WIN_START = T0 + VOL_WINDOW_S * 1000  # enough history for the volatility window
WIN_END = WIN_START + 900_000
JUMP = WIN_START + 300_000
REPRICE = JUMP + 20_000  # Polymarket's book stays stale for 20 s


def _binance() -> Trades:
    t = np.arange(T0, WIN_END + 60_000, 1000.0)
    logp = np.where(np.arange(t.size) % 2 == 0, 0.0, 1e-4) + np.where(t >= JUMP, 0.005, 0.0)
    n = t.size
    return Trades(t, t + 20, 100 * np.exp(logp), np.ones(n), np.ones(n))


def test_fair_value_is_a_coin_flip_at_the_strike_and_follows_the_move() -> None:
    fv = FairValue(_binance())
    before, after = fv.up(np.array([JUMP - 10_000, JUMP + 10_000]), WIN_START / 1000, WIN_END / 1000)
    assert abs(before - 0.5) < 0.05
    assert after > 0.85  # the jump itself inflates the trailing volatility
    assert np.isnan(fv.up(np.array([WIN_START - 1_000]), WIN_START / 1000, WIN_END / 1000))[0]
    assert fv.settle_up(WIN_START / 1000, WIN_END / 1000) == 1.0


def test_fair_value_needs_fresh_prices() -> None:
    tr = _binance()
    fv = FairValue(Trades(tr.t[:100], tr.rt[:100], tr.price[:100], tr.qty[:100], tr.side[:100]))
    assert np.isnan(fv.price(np.array([tr.t[99] + 60_000])))[0]


def _dataset() -> Dataset:
    rng = np.random.default_rng(3)
    q_t = [WIN_START, REPRICE, WIN_END]
    mids = [0.5, 0.95, 0.95]
    quotes = Quotes(np.array(q_t), np.array(q_t) + 50, np.array(mids) - 0.005, np.array(mids) + 0.005)
    t, p, s = [], [], []
    for ts in np.arange(WIN_START + 5_000, WIN_END - 120_000, 10_000.0):
        if JUMP <= ts < REPRICE:
            continue
        side = float(rng.choice([-1.0, 1.0]))
        mid = 0.5 if ts < REPRICE else 0.95
        t.append(ts)
        p.append(mid + 0.005 * side)
        s.append(side)
    for ts in np.arange(JUMP + 1_000, REPRICE, 2_000.0):  # informed takers lift the stale ask
        t.append(ts)
        p.append(0.505)
        s.append(1.0)
    order = np.argsort(t)
    tt = np.array(t)[order]
    pm = PMTrades(tt, tt + 50, np.array(p)[order], np.full(tt.size, 100.0), np.array(s)[order])
    market = MarketInfo(
        "btc-updown-15m-x", "BTCUSDT", "updown_15m", ["up", "down"], WIN_START / 1000, WIN_END / 1000
    )
    return Dataset(
        start_ms=T0,
        end_ms=WIN_END + 60_000,
        trades={"BTCUSDT": _binance()},
        book={},
        toxicity={},
        confluence={},
        quotes={"up": quotes},
        pm_trades={"up": pm},
        markets={market.slug: market},
        gaps={k: GapMask() for k in ("followsm", "binance", "polymarket", "all")},
        event_counts={},
        resolutions={market.slug: 1.0},
    )


def test_fair_value_quoter_is_not_picked_off_by_a_stale_book() -> None:
    ds = _dataset()
    naive, fair = collect_fills(ds, "BTCUSDT"), collect_fills(ds, "BTCUSDT", "fair_value")

    def pnl(f: np.ndarray, size: np.ndarray) -> float:
        return float(np.nansum(f * size))

    naive_30 = pnl(naive.markout[PRIMARY_MARKOUT_S], naive.size)
    fair_30 = pnl(fair.markout[PRIMARY_MARKOUT_S], fair.size)
    assert naive_30 < 0  # sold Up at 0.505 while Binance said ~0.98
    assert fair_30 > naive_30
    assert pnl(fair.settle, fair.size) > 0 > pnl(naive.settle, naive.size)
    lag = (naive.t_server >= JUMP) & (naive.t_server < REPRICE)
    assert lag.any() and not ((fair.t_server >= JUMP) & (fair.t_server < REPRICE)).any()


def test_settle_uses_polymarkets_resolution_not_binance() -> None:
    ds = _dataset()
    ds.resolutions = {"btc-updown-15m-x": 0.0}  # Chainlink said Down although Binance closed higher
    fair = collect_fills(ds, "BTCUSDT", "fair_value")
    assert float(np.nansum(fair.settle * fair.size)) < 0
    assert settlement_check(ds)["BTCUSDT"] == {
        "resolved": 1, "compared": 1, "binance_disagrees": 1, "binance_disagree_share": 1.0,
    }
    ds.resolutions = {}
    assert np.isnan(collect_fills(ds, "BTCUSDT", "fair_value").settle).all()


def test_resolution_cache_is_read_offline(tmp_path: Path) -> None:
    (tmp_path / CACHE_NAME).write_text('{"slug": "a", "up": 1.0}\n{"slug": "b", "up": 0.0}\n')
    assert load_resolutions(tmp_path, ["a", "b", "c"], fetch=False) == {"a": 1.0, "b": 0.0}
