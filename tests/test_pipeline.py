"""Full pipeline on a synthetic capture written through the real Store. Offline.

Two hours of BTC: a 1 s trade/book clock with a 30 bps drop every 10 minutes, a $150k sell sweep
at each drop, FollowSM toxicity frames whose +-1% book turns ask-heavy 3 s before each drop, and a
Polymarket Up/Down token that reprices 2 s after each drop.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tick_microstructure.analytics.calibration import calibrate
from tick_microstructure.analytics.data import load
from tick_microstructure.analytics.leadlag import lead_lag
from tick_microstructure.collector.storage import Store
from tick_microstructure.outreach.charts import render_all
from tick_microstructure.outreach.report import render
from tick_microstructure.research.hypotheses import h1_depth_depletion, h2_sweep_cascade, h3_trigger_detection
from tick_microstructure.simulation.engine import simulate

START = 1_800_000_000_000.0
HOURS = 2
DROP_EVERY_MS = 600_000.0


def _capture(root: Path) -> None:
    store = Store(root)
    rng = np.random.default_rng(3)
    end = START + HOURS * 3_600_000
    drops = np.arange(START + 300_000, end - 60_000, DROP_EVERY_MS)
    price, trade_id = 84_000.0, 1
    store.write(
        "polymarket_markets",
        {
            "rt": START,
            "slug": "btc-updown-15m-x",
            "sym": "BTCUSDT",
            "kind": "updown_15m",
            "tokens": ["up", "down"],
            "start": START / 1000,
            "end": end / 1000 + 600,
        },
    )
    up_mid = 0.5
    for t in np.arange(START, end, 1000.0).tolist():
        drop_now = bool(np.any((drops >= t) & (drops < t + 1000)))
        if drop_now:
            for k in range(3):  # sweep: one matching event across three levels
                store.write(
                    "binance_trades",
                    {
                        "rt": t + 5,
                        "T": t,
                        "sym": "BTCUSDT",
                        "id": trade_id,
                        "p": price * (1 - 0.001 * k),
                        "q": 0.6,
                        "m": True,
                    },
                )
                trade_id += 1
            price *= 0.997
        side = bool(rng.random() < 0.5)
        store.write(
            "binance_trades",
            {"rt": t + 505, "T": t + 500, "sym": "BTCUSDT", "id": trade_id, "p": price, "q": 0.01, "m": side},
        )
        trade_id += 1
        store.write(
            "binance_book",
            {
                "rt": t + 510,
                "sym": "BTCUSDT",
                "u": trade_id,
                "b": price - 0.05,
                "B": 1.0,
                "a": price + 0.05,
                "A": 1.0,
            },
        )
        warning = bool(np.any((drops > t) & (drops <= t + 3000)))
        obt = 3.0 if warning else float(rng.uniform(0.6, 1.4))
        store.write(
            "followsm_toxicity",
            {
                "rt": t + 40,
                "ts": t,
                "sym": "BTCUSDT",
                "px": price,
                "vpin": 0.3,
                "vp": 0.5,
                "obt": obt,
                "l1": 0.5,
                "vz": 0.0,
                "natr": 0.002,
                "tbr": 0.5,
                "tox": warning,
                "b1": 1e6,
                "a1": obt * 1e6,
            },
        )
        repriced = bool(np.any((drops >= t - 3000) & (drops < t - 2000)))
        if repriced:
            up_mid -= 0.05
        if int(t) % 5000 == 0:
            event = {
                "slug": "btc-updown-15m-x",
                "tok": "up",
                "dir": "bullish_if_yes",
                "conf": 1.0,
                "p": up_mid,
                "dp15": -0.02 if warning else 0.01,
                "flow": 0.5,
                "whale": 50_000.0,
            }
            store.write(
                "followsm_confluence",
                {
                    "rt": t + 45,
                    "ts": t,
                    "sym": "BTCUSDT",
                    "px": price,
                    "vp": 0.5,
                    "obt": obt,
                    "dpx15": 0.001,
                    "risk": 0.2,
                    "tox": warning,
                    "div": warning,
                    "act": "NONE",
                    "ev": [event],
                },
            )
        store.write(
            "polymarket_bbo",
            {"rt": t + 60, "ts": t + 20, "tok": "up", "b": up_mid - 0.01, "a": up_mid + 0.01},
        )
        if drop_now:  # a taker hits the stale Up bid right at the drop
            store.write(
                "polymarket_trades",
                {"rt": t + 70, "ts": t + 30, "tok": "up", "p": up_mid - 0.01, "s": 50.0, "side": "SELL"},
            )
        elif rng.random() < 0.05:
            buy = bool(rng.random() < 0.5)
            px = up_mid + 0.01 if buy else up_mid - 0.01
            store.write(
                "polymarket_trades",
                {
                    "rt": t + 70,
                    "ts": t + 30,
                    "tok": "up",
                    "p": px,
                    "s": 50.0,
                    "side": "BUY" if buy else "SELL",
                },
            )
    store.close()


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    _capture(tmp_path / "data")
    ds = load(tmp_path / "data")
    assert ds.hours > 1.9 and "BTCUSDT" in ds.toxicity and "up" in ds.quotes and "up" in ds.pm_trades

    results = {
        "calibration": calibrate(ds),
        "leadlag": lead_lag(ds),
        "hypotheses": {
            "h1": h1_depth_depletion(ds),
            "h2": h2_sweep_cascade(ds),
            "h3": h3_trigger_detection(ds),
        },
        "simulation": simulate(ds, placebo_shifts=20),
    }
    json.dumps(results, default=float)

    conf = results["calibration"]["symbols"]["BTCUSDT"]["confluence"]
    assert conf["frames"] > 1000 and conf["whale_sweeps_1h_usdt"]["p50"] == 50_000.0

    h1 = results["hypotheses"]["h1"]["BTCUSDT"]["plan_threshold_2.0_full_sample"]["10s"]
    assert h1["signal"]["down"]["rate"] == 1.0 and h1["control"]["down"]["rate"] < 0.2

    h2 = results["hypotheses"]["h2"]["BTCUSDT"]
    assert h2["sweeps"] >= 10 and h2["confirmed_from_sweep"]["events"] == 0  # book warned before, not after

    h3 = results["hypotheses"]["h3"]["BTCUSDT"]
    assert h3["obt"]["actual"]["coverage_lead_1000ms"] == 1.0
    assert h3["obt"]["placebo_mean"]["coverage_lead_1000ms"] < 0.2

    obt_sim = results["simulation"]["symbols"]["BTCUSDT"]["triggers"]["obt"]
    assert obt_sim["actual"]["slippage_avoided"] > 0 and obt_sim["placebo_share_at_least_as_good"] < 0.2
    assert "composite" in results["simulation"]["symbols"]["BTCUSDT"]["triggers"]

    report = render(results, render_all(ds, results, tmp_path / "assets", "BTCUSDT"))
    assert "INSUFFICIENT" in report and "## H1" in report
    assert sorted(p.name for p in (tmp_path / "assets").glob("*.png")) == [
        "leadlag_cdf.png",
        "obt_distribution.png",
        "pnl_drawdown.png",
    ]
