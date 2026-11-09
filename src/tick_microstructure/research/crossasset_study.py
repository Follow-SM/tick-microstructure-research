"""Crypto vs US stocks, minute by minute: how tightly BTC/ETH move with the equity and commodity perps
Binance lists next to them, in which US session, who moves first, and how far the two drift apart.

All from 1m USD-M klines (data.binance.vision, SHA-256 checked) on one UTC minute grid:

    basket      equal-weight mean of the US single-stock perps' log returns (the "stock market" built
                only from pairs FollowSM serves); QQQUSDT is reported as a cross-check
    corr        5-minute log-return correlation by US session; 95% CI from a day-block bootstrap
    leadlag     1-minute cross-correlation at lags -5..+5 by session; asymmetry = corr(crypto leads
                by 1 min) - corr(stocks lead by 1 min), with a day-block bootstrap CI
    shocks      5-minute moves beyond 4 trailing sigmas in one asset: the other asset's move before,
                during and in the 15 minutes after, signed by the shock; placebo = the same windows at
                random times of the same session
    daily       US-close-to-close returns (perp at 15:59 ET): beta, R^2, the residual of each day, and
                where 4-5 Nov 2026 rank against the trailing 60 trading days
    cash        the perps against the cash market on 4-6 Nov: Yahoo official closes, Yahoo 1m QQQ at
                the 5 Nov open (Yahoo keeps 1m bars for ~30 days only), VIX, crypto UTC-day returns

Nothing is tuned on the data it is scored on; every rate or correlation carries its interval.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from tick_microstructure.research.data.fetch import (
    YAHOO_HEADERS,
    FetchReport,
    _bar_table,
    _yahoo_1m,
    _yahoo_daily,
    listing_dates,
    ticker,
)
from tick_microstructure.research.data.store import TradfiStore
from tick_microstructure.research.utils.session import NY, Session, SessionCalendar

log = logging.getLogger(__name__)

F = npt.NDArray[np.float64]
MIN_MS = 60_000
CRYPTO = ("BTCUSDT", "ETHUSDT")
STOCKS = (
    "TSLAUSDT",
    "AMZNUSDT",
    "METAUSDT",
    "GOOGLUSDT",
    "AAPLUSDT",
    "MSFTUSDT",
    "NVDAUSDT",
    "MUUSDT",
    "TSMUSDT",
    "AVGOUSDT",
    "INTCUSDT",
    "PLTRUSDT",
)
OTHER_TRADFI = ("BABAUSDT", "XAUUSDT", "XAGUSDT", "CLUSDT", "COPPERUSDT")
CROSS_CHECK = ("QQQUSDT", "SPYUSDT")
ALL_SYMBOLS = (*CRYPTO, *STOCKS, *OTHER_TRADFI, *CROSS_CHECK)
SESSIONS = {s.value: s.name.lower() for s in Session}
BOOTSTRAP = 1_000
LAGS = range(-5, 6)
SHOCK_SIGMAS = 4.0
SHOCK_TRAIL_WINDOWS = 2_016  # trailing same-session 5-minute windows for the shock sigma
EVENT_DAYS = (date(2026, 11, 4), date(2026, 11, 5))
CASH_DAYS = (date(2026, 11, 3), *EVENT_DAYS, date(2026, 11, 6))
CASH_TICKERS = tuple(ticker(s) for s in (*CROSS_CHECK, *STOCKS, "BABAUSDT"))
VIX = "^VIX"
DAILY_TRAIL = 60


async def fetch(store: TradfiStore, start: date, end: date) -> FetchReport:
    report = FetchReport()
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient(timeout=60, headers=YAHOO_HEADERS, follow_redirects=True) as client:
        listed = await listing_dates(client)
        for tk in (*CASH_TICKERS, VIX):
            await _yahoo_daily(client, store, tk, start)
        for tk in CASH_TICKERS:
            await _yahoo_1m(client, store, tk)
        await asyncio.gather(
            *(
                _bar_table(client, store, "klines", s, max(start, listed.get(s, start)), end, sem, report)
                for s in ALL_SYMBOLS
            )
        )
    return report


# ───────────────────────────── data ─────────────────────────────


@dataclass
class Grid:
    t: F  # minute open, UTC ms
    log_close: dict[str, F]  # NaN where the perp has no bar
    quote: dict[str, F]
    session: npt.NDArray[np.int8]
    ny_day: npt.NDArray[np.int64]  # New York local date as an ordinal (bootstrap blocks)

    def ret(self, name: str, step: int) -> F:
        lc = self.log_close[name]
        out = np.full(lc.shape, np.nan)
        out[step:] = lc[step:] - lc[:-step]
        return out


def load_grid(store: TradfiStore, symbols: tuple[str, ...], start_ms: float, end_ms: float) -> Grid:
    t = np.arange(start_ms, end_ms, MIN_MS, dtype=np.float64)
    log_close: dict[str, F] = {}
    quote: dict[str, F] = {}
    for s in symbols:
        b = store.bars("klines", s)
        idx = ((b.t - start_ms) // MIN_MS).astype(np.int64)
        ok = (idx >= 0) & (idx < t.size)
        lc = np.full(t.size, np.nan)
        q = np.zeros(t.size)
        lc[idx[ok]] = np.log(b.close[ok])
        q[idx[ok]] = b.quote[ok]
        log_close[s], quote[s] = lc, q
    stocks = [s for s in STOCKS if s in symbols]
    # Equal-weight geometric basket: its log return over any interval is the members' mean log
    # return. A minute where any member has no bar is NaN rather than a partial basket.
    log_close["BASKET"] = np.vstack([log_close[s] for s in stocks]).mean(axis=0)
    quote["BASKET"] = np.sum([quote[s] for s in stocks], axis=0)
    cal = SessionCalendar()
    offset = cal.utc_offset_ms(t)
    return Grid(
        t=t,
        log_close=log_close,
        quote=quote,
        session=cal.classify(t),
        ny_day=np.floor_divide(t + offset, 86_400_000).astype(np.int64),
    )


# ───────────────────────────── statistics ─────────────────────────────


def _day_sums(day: npt.NDArray[np.int64], x: F, y: F) -> F:
    """Per-day sufficient statistics (n, Sx, Sy, Sxx, Syy, Sxy) for a block bootstrap."""
    _, inv = np.unique(day, return_inverse=True)
    out = np.zeros((inv.max() + 1 if inv.size else 0, 6))
    for k, v in enumerate((np.ones_like(x), x, y, x * x, y * y, x * y)):
        np.add.at(out[:, k], inv, v)
    return out


def _corr_from(s: F) -> F:
    n, sx, sy, sxx, syy, sxy = (s[..., k] for k in range(6))
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy / n - sx * sy / n**2
        vx = sxx / n - (sx / n) ** 2
        vy = syy / n - (sy / n) ** 2
        return np.asarray(cov / np.sqrt(vx * vy), dtype=np.float64)


def block_corr(day: npt.NDArray[np.int64], x: F, y: F, rng: np.random.Generator) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 30:
        return {"n": int(ok.sum()), "corr": None, "ci95": None}
    sums = _day_sums(day[ok], x[ok], y[ok])
    point = float(_corr_from(sums.sum(axis=0)))
    draws = rng.integers(0, sums.shape[0], size=(BOOTSTRAP, sums.shape[0]))
    boot = _corr_from(sums[draws].sum(axis=1))
    return {
        "n": int(ok.sum()),
        "days": int(sums.shape[0]),
        "corr": point,
        "ci95": [float(np.nanquantile(boot, 0.025)), float(np.nanquantile(boot, 0.975))],
    }


def _five_minute(g: Grid, name: str) -> tuple[F, npt.NDArray[np.int8], npt.NDArray[np.int64]]:
    """Non-overlapping 5-minute log returns, labelled with the session of the window's start."""
    r = g.ret(name, 5)
    ends = np.arange(5, g.t.size, 5)
    return r[ends], g.session[ends - 5], g.ny_day[ends - 5]


def correlation_by_session(g: Grid, a: str, b: str, rng: np.random.Generator) -> dict[str, Any]:
    ra, sess, day = _five_minute(g, a)
    rb, _, _ = _five_minute(g, b)
    out = {"all": block_corr(day, ra, rb, rng)}
    for code, name in SESSIONS.items():
        m = sess == code
        out[name] = block_corr(day[m], ra[m], rb[m], rng)
    return out


def lead_lag(g: Grid, crypto: str, stock: str, rng: np.random.Generator) -> dict[str, Any]:
    """corr(crypto[t], stock[t + k]) on 1-minute returns: k > 0 means crypto moved first."""
    x, y = g.ret(crypto, 1), g.ret(stock, 1)
    out: dict[str, Any] = {}
    for code, name in [(None, "all"), *SESSIONS.items()]:
        per: dict[str, Any] = {}
        sums_by_lag: dict[int, F] = {}
        days = None
        for k in LAGS:
            if k >= 0:
                xs, ys, sess, day = x[: x.size - k], y[k:], g.session[: x.size - k], g.ny_day[: x.size - k]
            else:
                xs, ys, sess, day = x[-k:], y[: y.size + k], g.session[-k:], g.ny_day[-k:]
            m = np.isfinite(xs) & np.isfinite(ys) & (True if code is None else sess == code)
            if m.sum() < 100:
                continue
            sums = _day_sums(day[m], xs[m], ys[m])
            if days is None:
                days = np.unique(day[m])
            per[str(k)] = float(_corr_from(sums.sum(axis=0)))
            # Align per-day rows on the same day set across lags for a paired bootstrap.
            u = np.unique(day[m])
            full = np.zeros((days.size, 6))
            pos = np.searchsorted(days, u)
            keep = (pos < days.size) & (days[np.minimum(pos, days.size - 1)] == u)
            full[pos[keep]] = sums[keep]
            sums_by_lag[k] = full
        if not per or 1 not in sums_by_lag or -1 not in sums_by_lag:
            continue
        draws = rng.integers(0, days.size, size=(BOOTSTRAP, days.size))
        lead = _corr_from(sums_by_lag[1][draws].sum(axis=1))
        follow = _corr_from(sums_by_lag[-1][draws].sum(axis=1))
        asym = lead - follow
        out[name] = {
            "corr_by_lag": per,
            "asymmetry": per["1"] - per["-1"],
            "asymmetry_ci95": [float(np.nanquantile(asym, 0.025)), float(np.nanquantile(asym, 0.975))],
        }
    return out


def shock_response(g: Grid, leader: str, follower: str, rng: np.random.Generator) -> dict[str, Any]:
    """When `leader` makes a > 4-sigma 5-minute move, what does `follower` do around it?"""
    r5 = g.ret(leader, 5)
    lf = g.log_close[follower]
    ends = np.arange(5, g.t.size, 5)
    rl = r5[ends]
    sess = g.session[ends - 5]
    sigma = np.full(rl.size, np.nan)
    trail = SHOCK_TRAIL_WINDOWS
    # Trailing sigma per session from strictly earlier windows (no look-ahead).
    for code in SESSIONS:
        idx = np.flatnonzero((sess == code) & np.isfinite(rl))
        v = rl[idx]
        c1 = np.concatenate([[0.0], np.cumsum(v)])
        c2 = np.concatenate([[0.0], np.cumsum(v * v)])
        j = np.arange(v.size)
        lo = np.maximum(j - trail, 0)
        n = j - lo
        ok = n >= 200
        mean = np.where(ok, (c1[j] - c1[lo]) / np.maximum(n, 1), np.nan)
        sigma[idx] = np.sqrt(
            np.maximum(np.where(ok, (c2[j] - c2[lo]) / np.maximum(n, 1) - mean**2, np.nan), 0)
        )
    shock = np.isfinite(sigma) & (np.abs(rl) > SHOCK_SIGMAS * sigma)

    def windows(end_idx: npt.NDArray[np.int64], sign: F) -> dict[str, F]:
        def move(a: int, b: int) -> F:
            i0, i1 = end_idx + a, end_idx + b
            ok = (i0 >= 0) & (i1 < lf.size)
            out = np.full(end_idx.size, np.nan)
            out[ok] = (lf[i1[ok]] - lf[i0[ok]]) * sign[ok] * 1e4
            return out

        return {"before_15m": move(-20, -5), "during": move(-5, 0), "after_15m": move(0, 15)}

    out: dict[str, Any] = {}
    for code, name in [(None, "all"), *SESSIONS.items()]:
        m = shock & (True if code is None else sess == code)
        e = ends[m]
        if e.size < 5:
            out[name] = {"n": int(e.size)}
            continue
        sign = np.sign(rl[m])
        real = windows(e, sign)
        # Placebo: random windows of the same session, random sign.
        pool = ends[(sess == code) if code is not None else np.ones(ends.size, bool)]
        reps = []
        for _ in range(200):
            pe = rng.choice(pool, size=e.size)
            reps.append(
                {k: float(np.nanmean(v)) for k, v in windows(pe, rng.choice([-1.0, 1.0], e.size)).items()}
            )
        row: dict[str, Any] = {"n": int(e.size), "leader_move_bps": float(np.nanmedian(np.abs(rl[m])) * 1e4)}
        for k, v in real.items():
            v = v[np.isfinite(v)]
            boot = [float(np.mean(rng.choice(v, v.size))) for _ in range(BOOTSTRAP)]
            row[k] = {
                "mean_bps": float(np.mean(v)),
                "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                "same_sign": float(np.mean(v > 0)),
                "placebo_mean_bps_p95": float(np.quantile([abs(r[k]) for r in reps], 0.95)),
            }
        out[name] = row
    return out


def daily_moves(g: Grid, cal: SessionCalendar, names: tuple[str, ...]) -> dict[str, Any]:
    """US-close to US-close returns (perp at the last minute before the close)."""
    first = cal.local_dates(g.t[:1])[0]
    last = cal.local_dates(g.t[-1:])[0]
    days = cal.trading_days(first, last)
    idx = np.array([int((cal.close_ms(d) - MIN_MS - g.t[0]) // MIN_MS) for d in days])
    keep = (idx >= 0) & (idx < g.t.size)
    days = [d for d, k in zip(days, keep, strict=True) if k]
    idx = idx[keep]
    rets = {n: np.diff(g.log_close[n][idx]) for n in names}
    return {"days": days[1:], "returns": rets}


def daily_study(d: dict[str, Any], x: str, y: str) -> dict[str, Any]:
    days: list[date] = d["days"]
    rx, ry = d["returns"][x], d["returns"][y]
    ok = np.isfinite(rx) & np.isfinite(ry)
    slope, intercept = np.polyfit(rx[ok], ry[ok], 1)
    resid = ry - (slope * rx + intercept)
    corr = float(np.corrcoef(rx[ok], ry[ok])[0, 1])
    events: dict[str, Any] = {}
    for ev in EVENT_DAYS:
        if ev not in days:
            continue
        k = days.index(ev)
        trail = slice(max(0, k - DAILY_TRAIL), k)
        sx, sy = np.nanstd(rx[trail]), np.nanstd(ry[trail])
        tr_ok = ok[trail]
        tr_corr = float(np.corrcoef(rx[trail][tr_ok], ry[trail][tr_ok])[0, 1])
        events[ev.isoformat()] = {
            f"{x}_pct": float(np.expm1(rx[k]) * 100),
            f"{y}_pct": float(np.expm1(ry[k]) * 100),
            f"{x}_sigmas": float(rx[k] / sx),
            f"{y}_sigmas": float(ry[k] / sy),
            "residual_pct": float(resid[k] * 100),
            "residual_rank_of_all": float(np.mean(np.abs(resid[ok]) <= abs(resid[k]))),
            "trailing_60d_corr": tr_corr,
        }
    # Rolling 20-day correlation (for the chart and its range)
    roll = np.full(rx.size, np.nan)
    for k in range(20, rx.size):
        w = slice(k - 20, k)
        m = ok[w]
        if m.sum() >= 15:
            roll[k] = np.corrcoef(rx[w][m], ry[w][m])[0, 1]
    return {
        "n_days": int(ok.sum()),
        "corr": corr,
        "beta": float(slope),
        "r2": corr**2,
        "abs_residual_pct": {
            "p50": float(np.median(np.abs(resid[ok])) * 100),
            "p90": float(np.quantile(np.abs(resid[ok]), 0.9) * 100),
        },
        "rolling20_corr_range": [float(np.nanmin(roll)), float(np.nanmax(roll))],
        "events": events,
        "_rolling": roll,
    }


def event_table(d: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
    """Each perp's 4 Nov and 5 Nov close-to-close move, in % and in trailing-60-day sigmas."""
    days: list[date] = d["days"]
    out: dict[str, Any] = {}
    for n in names:
        r = d["returns"][n]
        row = {}
        for ev in EVENT_DAYS:
            if ev not in days:
                continue
            k = days.index(ev)
            trail = r[max(0, k - DAILY_TRAIL) : k]
            sd = float(np.nanstd(trail)) if np.isfinite(trail).sum() >= 20 else float("nan")
            row[ev.isoformat()] = {"pct": float(np.expm1(r[k]) * 100), "sigmas": float(r[k] / sd)}
        out[n] = row
    return out


def run(store: TradfiStore, start: date, end: date, seed: int = 11) -> tuple[dict[str, Any], dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cal = SessionCalendar()
    start_ms = datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000
    end_ms = datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp() * 1000
    g = load_grid(store, ALL_SYMBOLS, start_ms, end_ms)
    results: dict[str, Any] = {"period": [start.isoformat(), end.isoformat()], "symbols": list(ALL_SYMBOLS)}

    log.info("Correlation by session ...")
    results["corr"] = {
        c: {s: correlation_by_session(g, c, s, rng) for s in ("BASKET", *CROSS_CHECK, *STOCKS, *OTHER_TRADFI)}
        for c in CRYPTO
    }
    results["basket_vs_qqq"] = correlation_by_session(g, "BASKET", "QQQUSDT", rng)
    log.info("Lead-lag ...")
    results["leadlag"] = {
        f"{c}~{s}": lead_lag(g, c, s, rng)
        for c in CRYPTO
        for s in ("BASKET", "QQQUSDT", "NVDAUSDT", "XAUUSDT")
    }
    # Split-sample check: a lead that only exists in one half is not reported as a finding.
    mid = g.t[0] + (g.t[-1] - g.t[0]) / 2
    halves = {}
    for name, keep in (("first_half", g.t < mid), ("second_half", g.t >= mid)):
        sub = Grid(
            t=g.t[keep],
            log_close={k: v[keep] for k, v in g.log_close.items()},
            quote={k: v[keep] for k, v in g.quote.items()},
            session=g.session[keep],
            ny_day=g.ny_day[keep],
        )
        halves[name] = {
            "from": ny_time(float(sub.t[0])),
            "to": ny_time(float(sub.t[-1])),
            **{f"BTCUSDT~{s}": lead_lag(sub, "BTCUSDT", s, rng) for s in ("BASKET", "QQQUSDT")},
        }
    results["leadlag_split"] = halves
    log.info("Shocks ...")
    results["shocks"] = {
        "stocks_to_btc": shock_response(g, "BASKET", "BTCUSDT", rng),
        "btc_to_stocks": shock_response(g, "BTCUSDT", "BASKET", rng),
    }
    log.info("Daily ...")
    d = daily_moves(g, cal, ("BTCUSDT", "ETHUSDT", "BASKET", *CROSS_CHECK, *STOCKS, *OTHER_TRADFI))
    daily = {f"BTCUSDT~{y}": daily_study(d, "BTCUSDT", y) for y in ("BASKET", "QQQUSDT")}
    daily["ETHUSDT~BASKET"] = daily_study(d, "ETHUSDT", "BASKET")
    chart = {"grid": g, "daily_days": d["days"], "rolling": {k: v.pop("_rolling") for k, v in daily.items()}}
    results["daily"] = daily
    results["events"] = event_table(d, ("BTCUSDT", "ETHUSDT", "BASKET", *CROSS_CHECK, *STOCKS, *OTHER_TRADFI))
    results["session_volume_share"] = {
        s: {
            SESSIONS[c]: float(g.quote[s][g.session == c].sum() / max(g.quote[s].sum(), 1.0))
            for c in SESSIONS
        }
        for s in ("BTCUSDT", "BASKET", "QQQUSDT", "XAUUSDT")
    }
    results["event_paths"] = event_paths(g, cal)
    results["follow_through"] = {
        "stocks_then_btc": follow_through(g, cal, rng, "BASKET", "BTCUSDT"),
        "btc_then_stocks": follow_through(g, cal, rng, "BTCUSDT", "BASKET"),
    }
    results["cash"] = cash_check(store, g, cal)
    return results, chart


def cash_check(store: TradfiStore, g: Grid, cal: SessionCalendar) -> dict[str, Any]:
    """Are the perps the stock market? Official closes, VIX, the 5 Nov open, and crypto by UTC day."""

    def perp_at(name: str, t_ms: float) -> float:
        return float(np.exp(g.log_close[name][int((t_ms - MIN_MS - g.t[0]) // MIN_MS)]))

    out: dict[str, Any] = {"close_to_close": {}, "vix": {}, "utc_day": {}, "open_5nov_qqq_bps": {}}
    for tk in CASH_TICKERS:
        if not store.table_path("yahoo_daily", tk).exists():
            continue
        y = store.yahoo_daily(tk)
        k = {d: i for i, d in enumerate(y.days)}
        rows = {}
        for prev, d in pairwise(CASH_DAYS):
            if prev in k and d in k:
                cash = y.close[k[d]] / y.close[k[prev]] - 1
                perp = perp_at(f"{tk}USDT", cal.close_ms(d)) / perp_at(f"{tk}USDT", cal.close_ms(prev)) - 1
                rows[d.isoformat()] = {"cash_pct": float(cash * 100), "perp_pct": float(perp * 100)}
        out["close_to_close"][tk] = rows
    diffs = [abs(r["cash_pct"] - r["perp_pct"]) for v in out["close_to_close"].values() for r in v.values()]
    out["max_abs_diff_pct_points"] = max(diffs) if diffs else None
    if store.table_path("yahoo_daily", VIX).exists():
        v = store.yahoo_daily(VIX)
        out["vix"] = {d.isoformat(): float(c) for d, c in zip(v.days, v.close, strict=True) if d in CASH_DAYS}
    for sym in (*CRYPTO, "XAUUSDT", "CLUSDT"):
        rows = {}
        for d in CASH_DAYS[1:]:
            t0 = datetime.combine(d, datetime.min.time(), UTC).timestamp() * 1000
            rows[d.isoformat()] = float((perp_at(sym, t0 + 86_400_000) / perp_at(sym, t0) - 1) * 100)
        out["utc_day"][sym] = rows
    y1 = store.yahoo_1m("QQQ")
    o = cal.open_ms(date(2026, 11, 5))
    stamps = {int(t): i for i, t in enumerate(y1.t)}
    if int(o - MIN_MS) in stamps:
        ref_cash, ref_perp = y1.close[stamps[int(o - MIN_MS)]], perp_at("QQQUSDT", o)
        for m in (6, 7, 30, 60):  # Yahoo stamps a bar at its open; compare at the bar's close
            t = int(o + (m - 1) * MIN_MS)
            if t in stamps:
                out["open_5nov_qqq_bps"][f"+{m}m"] = {
                    "cash": float((y1.close[stamps[t]] / ref_cash - 1) * 1e4),
                    "perp": float((perp_at("QQQUSDT", t + MIN_MS) / ref_perp - 1) * 1e4),
                }
    return out


EVENT_NAMES = ("BTCUSDT", "ETHUSDT", "BASKET", "QQQUSDT", "MUUSDT", "MSFTUSDT", "XAUUSDT", "CLUSDT")
FOLLOW_FIRST_MIN = 5
FOLLOW_AFTER_MIN = 15
PLACEBO_CLOCK = (12, 0)  # a mid-session minute with no scheduled event, same weekday set


def follow_through(
    g: Grid, cal: SessionCalendar, rng: np.random.Generator, leader: str, follower: str
) -> dict[str, Any]:
    """Does `follower` keep moving after `leader`'s first 5 minutes past the US open?

    Hypothesis taken from 5 Nov (stocks gapped at 09:30, BTC caught up over ~10 minutes), so 5 Nov
    is excluded and the test runs on every other open. Control: the same measurement at 12:00 ET.
    """
    first = cal.local_dates(g.t[:1])[0]
    last = cal.local_dates(g.t[-1:])[0]
    out: dict[str, Any] = {}
    for name, clock in (("open", None), ("placebo_1200", PLACEBO_CLOCK)):
        xs, ys, same = [], [], []
        for d in cal.trading_days(first, last):
            if d == date(2026, 11, 5):
                continue
            t0 = (
                cal.open_ms(d)
                if clock is None
                else datetime(d.year, d.month, d.day, *clock, tzinfo=NY).timestamp() * 1000
            )
            i0 = int((t0 - MIN_MS - g.t[0]) // MIN_MS)  # bar ending at t0
            i1, i2 = i0 + FOLLOW_FIRST_MIN, i0 + FOLLOW_FIRST_MIN + FOLLOW_AFTER_MIN
            if i0 < 0 or i2 >= g.t.size:
                continue
            ll, lf = g.log_close[leader], g.log_close[follower]
            x, y, z = ll[i1] - ll[i0], lf[i2] - lf[i1], lf[i1] - lf[i0]
            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                xs.append(x)
                ys.append(y)
                same.append(z)
        x, y, z = np.array(xs), np.array(ys), np.array(same)
        rho = float(np.corrcoef(x, y)[0, 1])
        null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(5_000)])
        slope = float(np.polyfit(x, y, 1)[0])
        big = np.abs(x) >= np.quantile(np.abs(x), 0.8)
        out[name] = {
            "n_days": int(x.size),
            "corr_first5_vs_next15": rho,
            "p_permutation": float((np.sum(np.abs(null) >= abs(rho)) + 1) / (null.size + 1)),
            "slope_next15_per_first5": slope,
            "corr_first5_same_window": float(np.corrcoef(x, z)[0, 1]),
            "top_quintile_days": int(big.sum()),
            "top_quintile_follow_same_sign": float(np.mean(np.sign(y[big]) == np.sign(x[big]))),
            "top_quintile_mean_follow_bps": float(np.mean(y[big] * np.sign(x[big])) * 1e4),
            "median_abs_first5_bps": float(np.median(np.abs(x)) * 1e4),
        }
    return out


def event_paths(g: Grid, cal: SessionCalendar) -> dict[str, Any]:
    """Cumulative bps paths: hourly from the 3 Nov close, minute-level around the 5 Nov open."""

    def path(t_ref: float, t_from: float, t_to: float, step_min: int) -> dict[str, Any]:
        i_ref = int((t_ref - MIN_MS - g.t[0]) // MIN_MS)  # bar ending at t_ref
        i0 = int((t_from - g.t[0]) // MIN_MS)
        i1 = int((t_to - g.t[0]) // MIN_MS)
        idx = np.arange(i0, i1, step_min)
        return {
            "t": g.t[idx] + MIN_MS,  # bar close times
            "session": g.session[idx],
            **{n: (g.log_close[n][idx] - g.log_close[n][i_ref]) * 1e4 for n in EVENT_NAMES},
            "BTCUSDT_quote": g.quote["BTCUSDT"][idx],
            "BASKET_quote": g.quote["BASKET"][idx],
        }

    close_3 = cal.close_ms(date(2026, 11, 3))
    open_5 = cal.open_ms(date(2026, 11, 5))
    return {
        "hourly": path(close_3, close_3 - 4 * 3_600_000, cal.close_ms(date(2026, 11, 6)), 60),
        "open_5nov": path(open_5, open_5 - 30 * MIN_MS, open_5 + 60 * MIN_MS, 1),
    }


def ny_time(t_ms: float) -> str:
    return datetime.fromtimestamp(t_ms / 1000, UTC).astimezone(NY).strftime("%a %d %b %H:%M ET")
