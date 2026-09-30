"""TradFi study: tokenized-equity perps across US sessions vs the 24/7 crypto control.

    structure   volume share and per-minute activity by session (Part 1)
    vpin        global-V vs session-V VPIN by session; profile across the US open (Task 2a)
    volume_z    how often the naive volume z-score fakes a "sweep" at the open (Task 1)
    depth       +-1% depth thinning in the 15 minutes before the open, gap days vs normal (Task 2b)
    spread      effective half-spread from the tick tape by minute around the open (Task 4 drag)
    gap         official open gap vs what the perp priced before the bell; do off-hours VPIN,
                taker flow or book imbalance predict the gap or the part the perp missed? (Task 3)
    leadlag     perp vs Binance index (full period) and vs Yahoo 1m incl. pre-market (last ~30 d)

Every p-value is a permutation test and every rate carries a Wilson 95% interval, as in the main
study. Small samples are reported as small; nothing here is tuned on the data it is scored on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.stats import rate
from tick_microstructure.research.data.fetch import CONTROL_SYMBOL, ticker
from tick_microstructure.research.data.store import TradfiStore
from tick_microstructure.research.tradfi import MIN_MS, MinuteFeatures, load_features
from tick_microstructure.research.utils.session import Session, SessionCalendar

log = logging.getLogger(__name__)

F = npt.NDArray[np.float64]
PROFILE_MIN = 180  # minutes either side of the open
PREOPEN_MIN = 15
SWEEP_Z = 6.0  # the hedger's liquidity-sweep volume threshold
PERMUTATIONS = 2_000
SESSION_NAMES = {s.value: s.name.lower() for s in Session}
CORE = ("TSLAUSDT", "NVDAUSDT", "AAPLUSDT", "SPYUSDT", "QQQUSDT")

ASSUMPTIONS = [
    "Signals are this study's own reimplementation from Binance public data (1m klines with exact "
    "taker-buy notional, 30 s bookDepth, aggTrades), not FollowSM's production feed.",
    "Session calendar: NYSE regular hours 09:30-16:00 America/New_York, pre-market from 04:00, "
    "post-market to 20:00, overnight 20:00-04:00 into a trading day; weekends and NYSE holidays are "
    "CLOSED. Checked against the days Yahoo printed a daily bar.",
    "The official open/close are Yahoo daily bars; the perp reference is the last 1m bar before "
    "the bell (09:29) and the bar ending at the prior close.",
    "Depth uses the latest bookDepth snapshot no older than 90 s; missing snapshots are NaN, never "
    "carried forward further.",
    "Effective half-spread = (taker-buy VWAP - taker-sell VWAP) / 2 / mid within each minute: a "
    "tape-based proxy, not a quoted spread.",
]


@dataclass
class OpenDay:
    sym: str
    day: date
    prev_day: date
    open_ms: int
    closure_h: float
    gap: float  # official open / previous official close - 1
    perp_overnight: float  # perp 09:29 close / perp close at the previous 16:00 - 1
    index_overnight: float
    basis_at_close: float  # perp / official close - 1 at the previous close
    offhours_imbalance: float  # (taker buy - taker sell) / volume, previous close -> 09:29
    preopen_imbalance: float  # the same over the last 60 minutes before the bell
    vpin_pctl: float  # session-V VPIN percentile at 09:29
    ob_log_imbalance: float  # ln(ask_1% / bid_1%) at 09:29
    ob_pctl: float
    preopen_depth_ratio: float  # +-1% depth 09:15-09:29 / its 08:00-09:00 median
    post_open_vol: float  # std of 1m perp log returns 09:30-09:45

    @property
    def residual(self) -> float:
        """Part of the gap the perp had not priced by 09:29."""
        return (1 + self.gap) / (1 + self.perp_overnight) - 1


def _minute_value(f: MinuteFeatures, series: F, t_ms: float) -> float:
    i = int(f.at(np.array([t_ms]))[0])
    return float(series[i]) if i >= 0 and not f.missing[i] else float("nan")


def _window(f: MinuteFeatures, t0: float, t1: float) -> slice:
    i0 = int(np.clip((t0 - f.t[0]) // MIN_MS, 0, f.t.size))
    i1 = int(np.clip((t1 - f.t[0]) // MIN_MS, 0, f.t.size))
    return slice(i0, i1)


def open_days(
    store: TradfiStore, cal: SessionCalendar, f: MinuteFeatures, sym: str
) -> tuple[list[OpenDay], list[str]]:
    daily = store.yahoo_daily(ticker(sym))
    first = cal.local_dates(f.t[:1])[0]
    last = cal.local_dates(f.t[-1:])[0]
    mismatches = cal.verify(daily.days, max(first, daily.days[0]), min(last, daily.days[-1]))
    by_day = {d: k for k, d in enumerate(daily.days)}
    depth1 = f.bid1 + f.ask1
    out: list[OpenDay] = []
    for d in cal.trading_days(first, last):
        prev = cal.previous_trading_day(d)
        if d not in by_day or prev not in by_day:
            continue
        o, c_prev = cal.open_ms(d), cal.close_ms(prev)
        if c_prev - 60 * MIN_MS < f.t[0]:
            continue
        perp_prev = _minute_value(f, f.close, c_prev - MIN_MS)
        perp_pre = _minute_value(f, f.close, o - MIN_MS)
        idx_prev = _minute_value(f, f.index_close, c_prev - MIN_MS)
        idx_pre = _minute_value(f, f.index_close, o - MIN_MS)
        if not (np.isfinite(perp_prev) and np.isfinite(perp_pre)):
            continue
        k, kp = by_day[d], by_day[prev]
        off = _window(f, c_prev, o)
        pre = _window(f, o - 60 * MIN_MS, o)
        vol_off, vol_pre = f.quote[off].sum(), f.quote[pre].sum()
        base = depth1[_window(f, o - 90 * MIN_MS, o - 30 * MIN_MS)]
        late = depth1[_window(f, o - PREOPEN_MIN * MIN_MS, o)]
        post = f.close[_window(f, o, o + 16 * MIN_MS)]
        with np.errstate(invalid="ignore", divide="ignore"):
            depth_ratio = (
                float(np.nanmedian(late) / np.nanmedian(base)) if base.size and late.size else np.nan
            )
            post_vol = float(np.std(np.diff(np.log(post)))) if post.size > 2 else np.nan
        out.append(
            OpenDay(
                sym=sym,
                day=d,
                prev_day=prev,
                open_ms=o,
                closure_h=cal.closure_hours_before(d),
                gap=float(daily.open[k] / daily.close[kp] - 1),
                perp_overnight=perp_pre / perp_prev - 1,
                index_overnight=idx_pre / idx_prev - 1 if np.isfinite(idx_pre * idx_prev) else np.nan,
                basis_at_close=perp_prev / float(daily.close[kp]) - 1,
                offhours_imbalance=float((2 * f.taker_buy[off].sum() - vol_off) / vol_off)
                if vol_off
                else np.nan,
                preopen_imbalance=float((2 * f.taker_buy[pre].sum() - vol_pre) / vol_pre)
                if vol_pre
                else np.nan,
                vpin_pctl=_minute_value(f, f.pctl_session, o - MIN_MS),
                ob_log_imbalance=_minute_value(f, f.log_imbalance, o - MIN_MS),
                ob_pctl=_minute_value(f, f.ob_pctl, o - MIN_MS),
                preopen_depth_ratio=depth_ratio,
                post_open_vol=post_vol,
            )
        )
    return out, mismatches


# ───────────────────────────── statistics ─────────────────────────────


def spearman(x: F, y: F) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_test(x: F, y: F, rng: np.random.Generator, perms: int = PERMUTATIONS) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 10:
        return {"n": int(x.size), "rho": None, "p": None}
    rho = spearman(x, y)
    null = np.array([spearman(x, rng.permutation(y)) for _ in range(perms)])
    p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (perms + 1))
    return {"n": int(x.size), "rho": rho, "p": p}


def direction_accuracy(signal: F, target: F, min_abs_target: float) -> dict[str, Any]:
    ok = np.isfinite(signal) & np.isfinite(target) & (np.abs(target) >= min_abs_target) & (signal != 0)
    hits = int(np.sum(np.sign(signal[ok]) == np.sign(target[ok])))
    return rate(hits, int(ok.sum()))


def ols(x: F, y: F) -> dict[str, Any]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"n": int(x.size)}
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    r2 = 1 - float(np.var(resid) / np.var(y)) if np.var(y) > 0 else float("nan")
    return {"n": int(x.size), "slope": float(slope), "intercept_bps": float(intercept * 1e4), "r2": r2}


def _bps_quantiles(v: F) -> dict[str, float | int | None]:
    v = np.abs(v[np.isfinite(v)]) * 1e4
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "p50": float(np.median(v)), "p90": float(np.quantile(v, 0.9))}


def gap_study(days: list[OpenDay], rng: np.random.Generator) -> dict[str, Any]:
    if not days:
        return {"n": 0}
    col = {
        name: np.array([getattr(d, name) for d in days], dtype=np.float64)
        for name in (
            "gap",
            "perp_overnight",
            "index_overnight",
            "basis_at_close",
            "offhours_imbalance",
            "preopen_imbalance",
            "vpin_pctl",
            "ob_log_imbalance",
            "ob_pctl",
            "preopen_depth_ratio",
            "post_open_vol",
            "closure_h",
        )
    }
    residual = np.array([d.residual for d in days])
    weekend = col["closure_h"] > 30
    out: dict[str, Any] = {
        "n_opens": len(days),
        "n_after_weekend_or_holiday": int(weekend.sum()),
        "abs_gap_bps": _bps_quantiles(col["gap"]),
        "abs_residual_bps": _bps_quantiles(residual),
        "abs_basis_at_close_bps": _bps_quantiles(col["basis_at_close"]),
        "price_discovery": {
            "perp": ols(col["perp_overnight"], col["gap"]),
            "index": ols(col["index_overnight"], col["gap"]),
            "perp_after_overnight": ols(col["perp_overnight"][~weekend], col["gap"][~weekend]),
            "perp_after_weekend_or_holiday": ols(col["perp_overnight"][weekend], col["gap"][weekend]),
            "perp_sign_matches_gap": direction_accuracy(col["perp_overnight"], col["gap"], 10e-4),
        },
    }
    signals = {
        "offhours_taker_imbalance": col["offhours_imbalance"],
        "preopen_taker_imbalance": col["preopen_imbalance"],
        "book_log_imbalance_0929": -col["ob_log_imbalance"],  # ask-heavy book = bearish
    }
    out["direction"] = {
        name: {
            "gap": {**direction_accuracy(s, col["gap"], 10e-4), **spearman_test(s, col["gap"], rng)},
            "residual": {**direction_accuracy(s, residual, 5e-4), **spearman_test(s, residual, rng)},
        }
        for name, s in signals.items()
    }
    out["magnitude"] = {
        name: {
            "abs_gap": spearman_test(s, np.abs(col["gap"]), rng),
            "abs_residual": spearman_test(s, np.abs(residual), rng),
            "post_open_vol": spearman_test(s, col["post_open_vol"], rng),
        }
        for name, s in (
            ("vpin_pctl_0929", col["vpin_pctl"]),
            ("book_imbalance_extremity_0929", np.abs(col["ob_pctl"] - 0.5)),
            ("preopen_depth_ratio", col["preopen_depth_ratio"]),
        )
    }
    return out


# ───────────────────────────── structure, VPIN, depth, spread ─────────────────────────────


def session_structure(f: MinuteFeatures) -> dict[str, Any]:
    total = f.quote.sum()
    out: dict[str, Any] = {}
    for s in Session:
        m = f.session == s
        out[SESSION_NAMES[s]] = {
            "share_of_time": float(m.mean()),
            "share_of_volume": float(f.quote[m].sum() / total) if total else None,
            "median_usd_per_min": float(np.median(f.quote[m])) if m.any() else None,
        }
    return out


def vpin_by_session(f: MinuteFeatures) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for mode, v, p in (
        ("global_v", f.vpin_global, f.pctl_global),
        ("session_v", f.vpin_session, f.pctl_session),
    ):
        per: dict[str, Any] = {}
        for s in Session:
            m = (f.session == s) & np.isfinite(p)
            per[SESSION_NAMES[s]] = {
                "minutes": int(m.sum()),
                "median_vpin": float(np.nanmedian(v[m])) if m.any() else None,
                "share_pctl_ge_095": float(np.mean(p[m] >= 0.95)) if m.any() else None,
            }
        out[mode] = per
    return out


def open_profiles(cal: SessionCalendar, f: MinuteFeatures, series: dict[str, F]) -> dict[str, F]:
    """[trading days x (2 * PROFILE_MIN + 1)] matrices of each series around 09:30 ET (NaN outside data)."""
    first, last = cal.local_dates(f.t[:1])[0], cal.local_dates(f.t[-1:])[0]
    opens = np.array([cal.open_ms(d) for d in cal.trading_days(first, last)], dtype=np.float64)
    offsets = np.arange(-PROFILE_MIN, PROFILE_MIN + 1) * MIN_MS
    idx = f.at((opens[:, None] + offsets[None, :]).ravel()).reshape(opens.size, offsets.size)
    out = {}
    for name, s in series.items():
        m = np.where(idx >= 0, s[np.maximum(idx, 0)], np.nan)
        m[(idx >= 0) & f.missing[np.maximum(idx, 0)]] = np.nan
        out[name] = m
    return out


def depth_profile(cal: SessionCalendar, f: MinuteFeatures) -> F:
    """+-1% depth around the open, normalised by each day's 08:00-09:00 ET median."""
    prof = open_profiles(cal, f, {"d": f.bid1 + f.ask1})["d"]
    base = np.nanmedian(prof[:, PROFILE_MIN - 90 : PROFILE_MIN - 30], axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return prof / base[:, None]


def spread_profile(
    store: TradfiStore, cal: SessionCalendar, f: MinuteFeatures, sym: str
) -> tuple[F, dict[str, float]]:
    """Effective half-spread (bps) per minute from the tick tape: open profile + median by session."""
    first, last = cal.local_dates(f.t[:1])[0], cal.local_dates(f.t[-1:])[0]
    days = store.days_with("trades", sym)
    hs = np.full(f.t.size, np.nan)
    for d in days:
        if not first <= d <= last:
            continue
        tr = store.trades(sym, d, d)
        if tr.t.size == 0:
            continue
        i = f.at(tr.t)
        ok = i >= 0
        i, px, qty, side = i[ok], tr.price[ok], tr.qty[ok], tr.side[ok]
        n = f.t.size
        nb = np.bincount(i, weights=px * qty * (side > 0), minlength=n)
        qb = np.bincount(i, weights=qty * (side > 0), minlength=n)
        ns = np.bincount(i, weights=px * qty * (side < 0), minlength=n)
        qs = np.bincount(i, weights=qty * (side < 0), minlength=n)
        both = (qb > 0) & (qs > 0)
        vb, vs = nb[both] / qb[both], ns[both] / qs[both]
        hs[both] = (vb - vs) / (vb + vs) * 1e4  # (buy - sell) / 2 / mid
    prof = open_profiles(cal, f, {"hs": hs})["hs"]
    by_session = {
        SESSION_NAMES[s]: float(np.nanmedian(hs[f.session == s]))
        for s in Session
        if np.isfinite(hs[f.session == s]).any()
    }
    return prof, by_session


def false_sweeps_at_open(cal: SessionCalendar, f: MinuteFeatures) -> dict[str, Any]:
    """Minutes 09:30-09:45 where the volume z-score alone crosses the sweep threshold."""
    prof = open_profiles(cal, f, {"naive": f.vol_z, "session": f.vol_z_session})
    window = slice(PROFILE_MIN, PROFILE_MIN + 15)
    out = {}
    for name, m in prof.items():
        w = m[:, window]
        ok = np.isfinite(w)
        out[name] = rate(int(np.sum(w[ok] >= SWEEP_Z)), int(ok.sum()))
        out[f"{name}_days_with_any"] = rate(
            int(np.sum(np.nanmax(np.where(ok, w, -np.inf), axis=1) >= SWEEP_Z)), int(ok.any(axis=1).sum())
        )
    return out


# ───────────────────────────── lead-lag ─────────────────────────────


def xcorr(a: F, b: F, max_lag: int) -> dict[int, float]:
    """corr(a[t], b[t + lag]); a positive peak lag means `a` moves first."""
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        x = a[max(0, -lag) : a.size - max(0, lag)]
        y = b[max(0, lag) : b.size - max(0, -lag)]
        ok = np.isfinite(x) & np.isfinite(y)
        out[lag] = float(np.corrcoef(x[ok], y[ok])[0, 1]) if ok.sum() > 30 else float("nan")
    return out


def _log_returns(price: F) -> F:
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.diff(np.log(price))
    return np.concatenate([[np.nan], r])


def lead_lag(store: TradfiStore, f: MinuteFeatures, sym: str, max_lag: int = 5) -> dict[str, Any]:
    perp = _log_returns(np.where(f.missing, np.nan, f.close))
    idx = _log_returns(f.index_close)
    out: dict[str, Any] = {"perp_vs_index": {}, "perp_vs_yahoo": {}}
    for s in (Session.PRE, Session.RTH, Session.POST, Session.OVERNIGHT, Session.CLOSED):
        m = f.session == s
        cc = xcorr(np.where(m, perp, np.nan), np.where(m, idx, np.nan), max_lag)
        out["perp_vs_index"][SESSION_NAMES[s]] = _peak(cc)
    y = store.yahoo_1m(ticker(sym))
    if y.t.size:
        yc = np.full(f.t.size, np.nan)
        i = f.at(y.t)
        ok = i >= 0
        yc[i[ok]] = y.close[ok]
        yr = _log_returns(yc)
        for s in (Session.PRE, Session.RTH, Session.POST):
            m = f.session == s
            cc = xcorr(np.where(m, perp, np.nan), np.where(m, yr, np.nan), max_lag)
            out["perp_vs_yahoo"][SESSION_NAMES[s]] = _peak(cc) | {"minutes": int((m & np.isfinite(yr)).sum())}
    return out


def _peak(cc: dict[int, float]) -> dict[str, Any]:
    finite = {k: v for k, v in cc.items() if np.isfinite(v)}
    if not finite:
        return {"peak_lag_min": None}
    lag = max(finite, key=lambda k: finite[k])
    return {"peak_lag_min": lag, "peak_corr": finite[lag], "corr_at_0": cc.get(0), "xcorr": finite}


# ───────────────────────────── run ─────────────────────────────


def run(store: TradfiStore, symbols: tuple[str, ...], seed: int = 7) -> tuple[dict[str, Any], dict[str, Any]]:
    """(results, chart data)."""
    cal = SessionCalendar()
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {"assumptions": ASSUMPTIONS, "symbols": {}, "data_quality": {}}
    charts: dict[str, Any] = {
        "vpin_profiles": {},
        "depth_profiles": {},
        "spread_profiles": {},
        "profile_days": {},
        "open_days": [],
    }
    all_days: list[OpenDay] = []
    for sym in (*symbols, CONTROL_SYMBOL):
        f = load_features(store, cal, sym)
        first, last = cal.local_dates(f.t[:1])[0], cal.local_dates(f.t[-1:])[0]
        sym_res: dict[str, Any] = {
            "period": [first.isoformat(), last.isoformat()],
            "missing_minutes_share": float(f.missing.mean()),
            "depth_coverage": float(np.isfinite(f.bid1).mean()),
            "structure": session_structure(f),
            "vpin": vpin_by_session(f),
            "false_sweeps_0930_0945": false_sweeps_at_open(cal, f),
        }
        charts["vpin_profiles"][sym] = open_profiles(
            cal, f, {"global_v": f.pctl_global, "session_v": f.pctl_session, "quote": f.quote}
        )
        charts["depth_profiles"][sym] = depth_profile(cal, f)
        charts["profile_days"][sym] = cal.trading_days(first, last)
        if sym != CONTROL_SYMBOL:
            days, mismatches = open_days(store, cal, f, sym)
            results["data_quality"][sym] = {"calendar_vs_yahoo_mismatches": mismatches}
            sym_res["gap"] = gap_study(days, rng)
            sym_res["leadlag"] = lead_lag(store, f, sym)
            all_days += days
            if sym in CORE:
                prof, by_session = spread_profile(store, cal, f, sym)
                charts["spread_profiles"][sym] = prof
                sym_res["half_spread_bps_by_session"] = by_session
        results["symbols"][sym] = sym_res
        log.info("TradFi %s: %s -> %s", sym, first, last)
    results["pooled_gap"] = gap_study(all_days, rng)
    charts["open_days"] = all_days
    results["depth_preopen"] = _depth_summary(charts["depth_profiles"], all_days)
    return results, charts


def _depth_summary(profiles: dict[str, F], days: list[OpenDay]) -> dict[str, Any]:
    """Median +-1% depth in 09:15-09:29 relative to 08:00-09:00, equities vs the crypto control."""
    window = slice(PROFILE_MIN - PREOPEN_MIN, PROFILE_MIN)
    rth = slice(PROFILE_MIN + 30, PROFILE_MIN + 180)
    out: dict[str, Any] = {}
    for sym, prof in profiles.items():
        pre = np.nanmedian(prof[:, window], axis=1)
        mid = np.nanmedian(prof[:, rth], axis=1)
        ok = np.isfinite(pre)
        out[sym] = {
            "days": int(ok.sum()),
            "median_preopen_vs_0800": float(np.nanmedian(pre)) if ok.any() else None,
            "share_days_thinner": float(np.mean(pre[ok] < 1)) if ok.any() else None,
            "median_rth_1000_1230_vs_0800": float(np.nanmedian(mid)) if np.isfinite(mid).any() else None,
        }
    ratio = np.array([d.preopen_depth_ratio for d in days])
    big = np.array([abs(d.gap) for d in days])
    ok = np.isfinite(ratio) & np.isfinite(big)
    if ok.sum() > 20:
        cut = np.quantile(big[ok], 0.9)
        out["top_decile_gap_days"] = {
            "gap_cut_bps": float(cut * 1e4),
            "median_ratio_top_decile": float(np.median(ratio[ok & (big >= cut)])),
            "median_ratio_rest": float(np.median(ratio[ok & (big < cut)])),
        }
    return out
