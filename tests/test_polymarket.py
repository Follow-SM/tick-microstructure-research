"""Polymarket study: history alignment (one-minute look-back bug), calibration, cooldown. Offline."""

from __future__ import annotations

import numpy as np

from tick_microstructure.research.polymarket_study import (
    DISAGREE,
    SAMPLE_AFTER_S,
    Trigger,
    _cooldown,
    sample_at,
    session_calibration,
)
from tick_microstructure.research.tradfi import MIN_MS

T0 = 1_790_000_000_000.0  # a minute boundary


def test_history_point_stamped_after_the_minute_describes_that_minute() -> None:
    # CLOB history stamps land a few seconds past each minute; the point at minute m + 4 s is the price
    # at the END of minute m - 1. Reading the last point before the minute's end would lag by one minute.
    closes = np.array([0.50, 0.55, 0.60, 0.65])
    pt = np.asarray(T0 + np.arange(1, 5) * MIN_MS + 4_000, dtype=np.float64)  # 4 s after each minute's end
    ends = np.asarray(T0 + np.arange(1, 5) * MIN_MS, dtype=np.float64)
    np.testing.assert_allclose(sample_at(pt, closes, ends), closes)


def test_history_sample_is_nan_when_the_next_point_is_too_late() -> None:
    pt = np.array([T0 + (SAMPLE_AFTER_S + 1) * 1_000, T0 + 5 * MIN_MS])
    out = sample_at(pt, np.array([0.4, 0.6]), np.array([T0, T0 + 5 * MIN_MS]))
    assert np.isnan(out[0])
    assert out[1] == 0.6


def test_session_calibration_credits_the_better_forecaster() -> None:
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.5).astype(float)
    perp = np.where(y == 1, 0.8, 0.2)
    pm = np.full(400, 0.5)  # always DISAGREE or more away from perp
    samples = {"rth": list(zip(perp, pm, y, strict=True))}
    out = session_calibration(samples, rng)["rth"]
    assert out["brier_perp"] < out["brier_pm"]
    assert out["diff_pm_minus_perp_ci95"][0] > 0
    assert out["perp_right_when_disagree"]["n"] == 400
    assert out["perp_right_when_disagree"]["rate"] == 1.0
    assert abs(0.8 - 0.5) > DISAGREE


def test_cooldown_keeps_one_trigger_per_window_per_night() -> None:
    trig = [Trigger(0, 100, 1, 1), Trigger(0, 110, 1, 1), Trigger(0, 140, -1, 1), Trigger(1, 141, 1, 1)]
    kept = _cooldown(trig)
    assert [(t.night, t.idx) for t in kept] == [(0, 100), (0, 140), (1, 141)]
