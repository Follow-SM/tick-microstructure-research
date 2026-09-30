"""TradFi study: session calendar, vectorized features, archive parsing, gap statistics. Offline."""

from __future__ import annotations

from datetime import UTC, date, datetime

import numpy as np
import pytest

from tick_microstructure.research.data.fetch import parse_agg_trades, parse_book_depth, parse_klines
from tick_microstructure.research.tradfi import (
    MIN_MS,
    bucket_sizes,
    midrank,
    trailing_percentile,
    vpin_series,
)
from tick_microstructure.research.tradfi_study import OpenDay, direction_accuracy, gap_study, xcorr
from tick_microstructure.research.utils.session import Session, SessionCalendar

CAL = SessionCalendar()


def ms(y: int, mo: int, d: int, h: int = 0, mi: int = 0, s: int = 0) -> float:
    return datetime(y, mo, d, h, mi, s, tzinfo=UTC).timestamp() * 1000


def session_at(y: int, mo: int, d: int, h: int = 0, mi: int = 0) -> Session:
    return Session(int(CAL.classify(np.array([ms(y, mo, d, h, mi)]))[0]))


# ── session calendar ─────────────────────────────────────────────────────────


def test_open_is_1330_utc_in_summer_and_1430_in_winter() -> None:
    assert CAL.open_ms(date(2026, 7, 1)) == ms(2026, 7, 1, 13, 30)
    assert CAL.open_ms(date(2026, 2, 2)) == ms(2026, 2, 2, 14, 30)
    # US moved to EDT on 8 March, Europe only on 29 March: the open is 15:30 in Eastern Europe then
    assert CAL.open_ms(date(2026, 3, 9)) == ms(2026, 3, 9, 13, 30)


def test_sessions_across_a_trading_day() -> None:
    assert session_at(2026, 7, 1, 13, 29) is Session.PRE
    assert session_at(2026, 7, 1, 13, 30) is Session.RTH
    assert session_at(2026, 7, 1, 19, 59) is Session.RTH
    assert session_at(2026, 7, 1, 20, 0) is Session.POST
    assert session_at(2026, 7, 2, 0, 0) is Session.OVERNIGHT  # 20:00 ET, Thursday is a trading day
    assert session_at(2026, 7, 2, 8, 0) is Session.PRE  # 04:00 ET


def test_weekend_and_holiday_are_closed_until_the_evening_before_the_next_session() -> None:
    assert session_at(2026, 9, 26, 12, 0) is Session.CLOSED  # Saturday
    assert session_at(2026, 9, 27, 23, 59) is Session.CLOSED  # Sunday 19:59 ET
    assert session_at(2026, 9, 28, 0, 0) is Session.OVERNIGHT  # Sunday 20:00 ET
    # Independence Day observed (Fri 3 July): Thursday 20:00 ET onwards is closed
    assert session_at(2026, 7, 3, 0, 0) is Session.CLOSED
    assert session_at(2026, 7, 3, 14, 0) is Session.CLOSED
    assert CAL.closure_hours_before(date(2026, 7, 6)) == pytest.approx(89.5)
    assert CAL.closure_hours_before(date(2026, 9, 29)) == pytest.approx(17.5)


def test_early_close_ends_rth_at_1300() -> None:
    assert session_at(2026, 11, 27, 17, 59) is Session.RTH  # 12:59 EST
    assert session_at(2026, 11, 27, 18, 0) is Session.POST


def test_calendar_verification_and_range_guard() -> None:
    days = CAL.trading_days(date(2026, 6, 29), date(2026, 7, 10))
    assert date(2026, 7, 3) not in days and len(days) == 9
    assert CAL.verify(days, date(2026, 6, 29), date(2026, 7, 10)) == []
    assert CAL.verify([*days, date(2026, 7, 3)], date(2026, 6, 29), date(2026, 7, 10)) == [
        "2026-07-03: bar on a calendar holiday"
    ]
    with pytest.raises(ValueError):
        CAL.classify(np.array([ms(2030, 1, 2)]))


# ── vectorized features ──────────────────────────────────────────────────────


def _vpin_reference(quote: list[float], buy: list[float], bucket: float, n: int) -> list[float]:
    out, imbalances, b, s, current = [], [], 0.0, 0.0, float("nan")
    for q, qb in zip(quote, buy, strict=True):
        qs = q - qb
        while q > 1e-12:
            frac = min(bucket - (b + s), q) / q
            b, s = b + qb * frac, s + qs * frac
            qb, qs, q = qb * (1 - frac), qs * (1 - frac), q * (1 - frac)
            if b + s >= bucket - 1e-9:
                imbalances.append(abs(b - s) / bucket)
                b = s = 0.0
                if len(imbalances) >= n:
                    current = sum(imbalances[-n:]) / n
        out.append(current)
    return out


def test_vectorized_vpin_matches_a_bucket_by_bucket_loop() -> None:
    rng = np.random.default_rng(1)
    quote = rng.gamma(0.6, 400.0, 3_000)
    buy = quote * rng.beta(2, 2, quote.size)
    got = vpin_series(quote, buy, np.full(quote.size, 1_000.0), n_buckets=20)
    want = np.array(_vpin_reference(quote.tolist(), buy.tolist(), 1_000.0, 20))
    both = np.isfinite(want)
    assert both.sum() > 1_000
    assert np.array_equal(np.isfinite(got), both)
    assert np.allclose(got[both], want[both], atol=1e-9)


def test_session_bucket_equals_adv_over_50_for_a_uniform_market() -> None:
    t = np.arange(0, 10 * 86_400_000, MIN_MS, dtype=np.float64)
    session = (np.arange(t.size) % 5).astype(np.int8)
    v_global, v_session = bucket_sizes(t, np.full(t.size, 10.0), session)
    day8 = t >= 8 * 86_400_000
    assert np.all(np.isnan(v_global[t < 7 * 86_400_000]))
    assert np.allclose(v_global[day8], 10.0 * 1440 / 50)
    assert np.allclose(v_session[day8], v_global[day8])


def test_session_bucket_shrinks_in_quiet_sessions() -> None:
    t = np.arange(0, 9 * 86_400_000, MIN_MS, dtype=np.float64)
    quiet = (np.arange(t.size) % 1440) < 720
    session = np.where(quiet, 3, 0).astype(np.int8)
    quote = np.where(quiet, 1.0, 99.0)
    _, v_session = bucket_sizes(t, quote, session)
    last = t >= 8 * 86_400_000
    assert np.allclose(v_session[last & quiet], 28.8)
    assert np.allclose(v_session[last & ~quiet], 99.0 * 28.8)


def test_trailing_percentile_never_sees_its_own_block() -> None:
    t = np.arange(0, 3 * 86_400_000, 3_600_000, dtype=np.float64)
    values = np.concatenate([np.zeros(48), np.full(24, 5.0)])
    p = trailing_percentile(t, values, 2 * 86_400_000, 86_400_000, min_samples=24)
    assert np.all(np.isnan(p[:24]))
    assert np.allclose(p[24:48], 0.5)  # equal to the whole reference: midrank
    assert np.allclose(p[48:], 1.0)
    assert midrank(np.array([1.0, 2.0, 3.0]), np.array([2.0]))[0] == pytest.approx(0.5)


# ── archive parsing ──────────────────────────────────────────────────────────


def test_parsers_accept_files_with_and_without_headers() -> None:
    body = b"1,380.36,27.02,10,12,1790208000708,true\n2,380.37,1.5,13,13,1790208000709,false\n"
    header = b"agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker\n"
    for raw in (body, header + body):
        a = parse_agg_trades(raw)
        assert a["side"].tolist() == [-1, 1] and a["t"][0] == 1790208000708
    d = parse_book_depth(b"timestamp,percentage,depth,notional\n2026-09-24 00:00:01,-1.00,5,1900\n")
    assert d["t"][0] == ms(2026, 9, 24, 0, 0, 1) and d["pct"][0] == -1.0
    k = parse_klines(b"1790553600000,1,2,0.5,1.5,10,1790553659999,15,3,4,6,0\n")
    assert k["quote"][0] == 15 and k["taker_buy_quote"][0] == 6


# ── gap statistics ───────────────────────────────────────────────────────────


def _day(k: int, gap: float, perp: float, flow: float) -> OpenDay:
    nan = float("nan")
    closure = 17.5 if k % 5 else 65.5
    d, prev = date(2026, 3, 2), date(2026, 2, 27)
    return OpenDay("TSLAUSDT", d, prev, 0, closure, gap, perp, perp, 0.0, flow, flow, nan, nan, nan, nan, nan)


def test_gap_study_measures_price_discovery_and_signal_accuracy() -> None:
    rng = np.random.default_rng(3)
    perp = rng.normal(0, 0.01, 200)
    gap = perp + rng.normal(0, 0.001, 200)  # the perp priced most of the gap
    flow = np.sign(gap - perp) * 0.2  # an oracle for the residual
    res = gap_study([_day(k, gap[k], perp[k], flow[k]) for k in range(200)], rng)
    pd = res["price_discovery"]["perp"]
    assert pd["slope"] == pytest.approx(1.0, abs=0.05) and pd["r2"] > 0.95
    oracle = res["direction"]["offhours_taker_imbalance"]["residual"]
    assert oracle["rate"] == 1.0 and oracle["p"] < 0.01
    acc = direction_accuracy(np.array([1.0, -1.0, 1.0]), np.array([0.01, 0.01, 0.00001]), 0.001)
    assert (acc["k"], acc["n"]) == (1, 2)


def test_xcorr_finds_the_leader() -> None:
    rng = np.random.default_rng(5)
    a = rng.normal(size=2_000)
    b = np.concatenate([[0.0, 0.0], a[:-2]]) + rng.normal(scale=0.1, size=2_000)  # b follows a by 2
    cc = xcorr(a, b, 4)
    assert max(cc, key=lambda k: cc[k]) == 2
