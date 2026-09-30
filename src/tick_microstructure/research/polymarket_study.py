"""Does Polymarket lead the tokenized-equity perps? Off-hours flow vs the unpriced gap.

For every daily "<TICKER> Up or Down" market we lay Polymarket onto the perp's minute grid from the
previous official close to the resolving close ("a night"), then ask, with the perp as the control:

    leadlag     1m / 5m cross-correlation of Polymarket probability changes and perp returns
    events      signed perp path after Polymarket whale sweeps and probability jumps (vs 20 timing
                placebos), and Polymarket's path after perp jumps: who reprices first, by how long
    gap         do off-hours Polymarket signals predict the part of the open gap the perp had not
                priced by 09:29:00 ET, or the rest of the day the market actually resolves on?
    brier       Polymarket probability vs a perp-implied probability for the same Up/Down outcome
    earnings    EPS-beat markets around the release: probability path vs the perp's reaction

Point-in-time throughout: sweep thresholds, flow z-score baselines and the perp-implied volatility
use only markets that resolved before the night being scored.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.stats import rate
from tick_microstructure.research.data.fetch import ticker
from tick_microstructure.research.data.polymarket import PmMarket, load_markets, load_prices, load_trades
from tick_microstructure.research.data.store import TradfiStore
from tick_microstructure.research.tradfi import MIN_MS, MinuteFeatures, load_features
from tick_microstructure.research.tradfi_study import direction_accuracy, spearman_test, xcorr
from tick_microstructure.research.utils.session import Session, SessionCalendar

log = logging.getLogger(__name__)

F = npt.NDArray[np.float64]
HISTORY_NIGHTS = 20
MIN_HISTORY_NIGHTS = 5
SWEEP_FLOOR_USD = 1_000.0
SWEEP_QUANTILE = 0.99
PM_JUMP = 0.10  # |Δp over 5 min|
PERP_JUMP_QUANTILE = 0.99  # |5m perp return| among off-hours minutes, per symbol
EVENT_PRE, EVENT_POST = 30, 60  # minutes
EVENT_COOLDOWN = 30
PLACEBO_SEEDS = 20
BOOTSTRAP = 2_000
BRIER_ANCHORS = {"04:00": -330, "08:30": -60, "09:28": -2, "10:00": 30, "13:30": 240}
SESSION_NAMES = {s.value: s.name.lower() for s in Session}
SAMPLE_AFTER_S = 30
CALIBRATION_STEP = 30  # minutes between calibration anchors
DISAGREE = 0.05  # |p_perp - p_pm| for the "who is right" count

ASSUMPTIONS = [
    "Polymarket: public data-api taker fills (each order arrives as one print, with wallet and tx hash) "
    "and CLOB 1-minute price history of the Up token. History points are stamped ~3-15 s past the minute, "
    f"so a minute's probability is the first point within {SAMPLE_AFTER_S} s after the perp bar closes "
    "(a few seconds of look-ahead in Polymarket's favour); NaN if there is none.",
    f"Whale sweep: one taker order >= max(${SWEEP_FLOOR_USD:,.0f}, the {SWEEP_QUANTILE:.0%} quantile of "
    f"order notional over the symbol's previous {HISTORY_NIGHTS} nights). Direction +1 = buys Up or sells "
    "Down.",
    f"Polymarket flow z: log(1 + 15-minute taker notional) vs the same session and quarter-hour over the "
    f"previous {HISTORY_NIGHTS} nights (median / 1.4826 MAD); NaN until {MIN_HISTORY_NIGHTS} nights exist.",
    "Perp-implied probability: Φ(ln(S/K) / σ), S = perp / (1 + perp-vs-official basis at the previous "
    f"close), K = the previous official close, σ = std of ln(close / S) at the same anchor over the "
    f"previous {HISTORY_NIGHTS} nights of that symbol.",
    "Residual gap = official open / previous close vs the perp's move to 09:29:00 ET; rest of day = "
    "ln(official close / S at 09:29:00), which is what an Up/Down market resolves on. Pre-open signals "
    "use Polymarket samples stamped no later than 09:28:30 and flow up to 09:28:00 (no look-ahead).",
    f"Event curves: {EVENT_COOLDOWN}-minute cooldown per trigger type and night; placebos keep each "
    "trigger's direction and night and draw a uniform minute in the same off-hours window.",
    "All p-values are permutation tests; the report lists a Bonferroni threshold for each family.",
]


def sample_at(pt: F, pp: F, ends: F) -> F:
    """Probability at each minute end: the first history point at/after it, if within SAMPLE_AFTER_S."""
    j = np.minimum(np.searchsorted(pt, ends, side="left"), pt.size - 1)
    ok = (pt[j] >= ends) & (pt[j] - ends <= SAMPLE_AFTER_S * 1000)
    return np.where(ok, pp[j], np.nan)


def norm_cdf(x: F) -> F:
    return np.asarray([0.5 * (1 + math.erf(v / math.sqrt(2))) if np.isfinite(v) else np.nan for v in x])


@dataclass
class Night:
    sym: str
    day: date
    i0: int  # grid index of the first minute after the previous close
    i1: int  # grid index of the minute that ends at the resolving close (exclusive)
    offhours_end: int  # grid index of 09:25 ET (exclusive): the off-hours window is [i0, offhours_end)
    open_idx: int  # grid index of the 09:30 bar
    p: F  # Up probability at the end of each minute
    flow_signed: F  # taker notional toward Up per minute
    flow_total: F
    sweep_signed: F  # signed notional of whale-sweep orders per minute
    sweep_threshold: float  # NaN in warm-up
    flow_z: F  # 15-minute flow z-score per minute (NaN in warm-up)
    payout_up: float
    close_prev: float
    close_day: float
    open_day: float
    basis: float  # perp / official close - 1 at the previous close

    @property
    def n(self) -> int:
        return self.i1 - self.i0

    def spot(self, f: MinuteFeatures) -> F:
        return f.close[self.i0 : self.i1] / (1 + self.basis)


def _slot_keys(cal: SessionCalendar, f: MinuteFeatures, i0: int, i1: int) -> npt.NDArray[np.int64]:
    local = cal.minutes_from_open(f.t[i0:i1])  # minutes from that day's 09:30 (negative before)
    quarter = np.floor_divide(np.mod(local + 570, 1440), 15).astype(np.int64)  # ET quarter-hour of day
    return np.asarray(f.session[i0:i1].astype(np.int64) * 100 + quarter, dtype=np.int64)


def build_nights(
    store: TradfiStore, cal: SessionCalendar, f: MinuteFeatures, sym: str, markets: list[PmMarket]
) -> tuple[list[Night], dict[str, int]]:
    daily = store.yahoo_daily(ticker(sym))
    bars = {d: k for k, d in enumerate(daily.days)}
    skipped: dict[str, int] = defaultdict(int)
    nights: list[Night] = []
    fill_hist: list[F] = []
    slot_hist: list[dict[int, list[float]]] = []
    for m in sorted((m for m in markets if m.kind == "updown" and m.sym == sym), key=lambda m: m.day):
        prev = cal.previous_trading_day(m.day)
        if m.day not in bars or prev not in bars:
            skipped["no official bar"] += 1
            continue
        if (daily.close[bars[m.day]] > daily.close[bars[prev]]) != (m.payout_yes == 1.0):
            skipped["resolution disagrees with Yahoo closes"] += 1
            continue
        c_prev, o, c_day = cal.close_ms(prev), cal.open_ms(m.day), cal.close_ms(m.day)
        i0, i_open, i1 = (int(x) for x in f.at(np.array([c_prev, o, c_day], dtype=np.float64)))
        if min(i0, i_open, i1) < 0:
            skipped["no perp bars"] += 1
            continue
        pt, pp = load_prices(store, m.slug)
        tr = load_trades(store, m.slug)
        if pt.size < 60:
            skipped["thin Polymarket history"] += 1
            continue
        p = sample_at(pt, pp, f.t[i0:i1] + MIN_MS)
        n = i1 - i0
        k = f.at(tr.t) - i0
        inside = (k >= 0) & (k < n)
        kk = k[inside]
        flow_signed = np.bincount(kk, weights=(tr.toward_yes * tr.notional)[inside], minlength=n)
        flow_total = np.bincount(kk, weights=tr.notional[inside], minlength=n)

        recent = fill_hist[-HISTORY_NIGHTS:]
        threshold = float("nan")
        sweep_signed = np.zeros(n)
        if len(recent) >= MIN_HISTORY_NIGHTS:
            pool = np.concatenate(recent)
            threshold = max(SWEEP_FLOOR_USD, float(np.quantile(pool, SWEEP_QUANTILE)) if pool.size else 0.0)
            big = inside & (tr.notional >= threshold)
            sweep_signed = np.bincount(k[big], weights=(tr.toward_yes * tr.notional)[big], minlength=n)

        roll = np.convolve(flow_total, np.ones(15), mode="full")[:n]
        keys = _slot_keys(cal, f, i0, i1)
        flow_z = np.full(n, np.nan)
        if len(slot_hist) >= MIN_HISTORY_NIGHTS:
            merged: dict[int, list[float]] = defaultdict(list)
            for h in slot_hist[-HISTORY_NIGHTS:]:
                for key, vals in h.items():
                    merged[key] += vals
            stats = {}
            for key, vals in merged.items():
                v = np.log1p(np.asarray(vals))
                med = float(np.median(v))
                stats[key] = (med, 1.4826 * float(np.median(np.abs(v - med))))
            logs = np.log1p(roll)
            for q in range(n):
                s = stats.get(int(keys[q]))
                if s and s[1] > 0:
                    flow_z[q] = (logs[q] - s[0]) / s[1]
        fill_hist.append(tr.notional[inside])
        hist: dict[int, list[float]] = defaultdict(list)
        for q in range(14, n):
            hist[int(keys[q])].append(float(roll[q]))
        slot_hist.append(hist)

        perp_prev = float(f.close[i0 - 1]) if i0 > 0 else float("nan")
        nights.append(
            Night(
                sym=sym,
                day=m.day,
                i0=i0,
                i1=i1,
                offhours_end=i_open - 5,
                open_idx=i_open,
                p=p,
                flow_signed=flow_signed,
                flow_total=flow_total,
                sweep_signed=sweep_signed,
                sweep_threshold=threshold,
                flow_z=flow_z,
                payout_up=m.payout_yes,
                close_prev=float(daily.close[bars[prev]]),
                close_day=float(daily.close[bars[m.day]]),
                open_day=float(daily.open[bars[m.day]]),
                basis=perp_prev / float(daily.close[bars[prev]]) - 1,
            )
        )
    return nights, dict(skipped)


# ───────────────────────────── lead-lag ─────────────────────────────


def _stitched(nights: list[Night], f: MinuteFeatures, step: int, pad: int) -> tuple[F, F, F]:
    """Concatenated (Δp, perp log return, session) per `step` minutes, NaN-padded between nights."""
    dps, rs, ss = [], [], []
    gap = np.full(pad, np.nan)
    for nt in nights:
        p = nt.p[::step]
        c = f.close[nt.i0 : nt.i1][::step]
        miss = f.missing[nt.i0 : nt.i1][::step]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.concatenate([[np.nan], np.diff(np.log(c))])
        dp = np.concatenate([[np.nan], np.diff(p)])
        mid = (p > 0.05) & (p < 0.95)
        dp[~mid] = np.nan
        r[miss] = np.nan
        dps += [dp, gap]
        rs += [r, gap]
        ss += [f.session[nt.i0 : nt.i1][::step].astype(np.float64), gap]
    if not dps:
        e = np.empty(0)
        return e, e, e
    return np.concatenate(dps), np.concatenate(rs), np.concatenate(ss)


def lead_lag(nights: list[Night], f: MinuteFeatures) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step, max_lag in ((1, 10), (5, 6)):
        dp, r, sess = _stitched(nights, f, step, max_lag + 1)
        per: dict[str, Any] = {}
        groups = {
            "offhours": np.isin(sess, [Session.POST, Session.OVERNIGHT, Session.PRE, Session.CLOSED]),
            "rth": sess == Session.RTH,
        }
        for name, m in groups.items():
            cc = xcorr(np.where(m, dp, np.nan), np.where(m, r, np.nan), max_lag)
            finite = {k: v for k, v in cc.items() if np.isfinite(v)}
            per[name] = {
                "xcorr": finite,
                "peak_lag": max(finite, key=lambda k: abs(finite[k])) if finite else None,
                "pm_leads_sum": float(sum(v for k, v in finite.items() if k > 0)),
                "perp_leads_sum": float(sum(v for k, v in finite.items() if k < 0)),
                "minutes": int(np.sum(m & np.isfinite(dp) & np.isfinite(r))),
            }
        out[f"{step}m"] = per
    return out


# ───────────────────────────── event curves ─────────────────────────────


@dataclass
class Trigger:
    night: int
    idx: int  # grid index of the trigger minute (decision at its end)
    direction: float
    size: float


def _cooldown(cands: list[Trigger]) -> list[Trigger]:
    out: list[Trigger] = []
    for c in cands:
        if not out or c.night != out[-1].night or c.idx - out[-1].idx >= EVENT_COOLDOWN:
            out.append(c)
    return out


def sweep_triggers(nights: list[Night]) -> list[Trigger]:
    cands = []
    for n_i, nt in enumerate(nights):
        for q in np.flatnonzero(nt.sweep_signed[: nt.offhours_end - nt.i0] != 0):
            cands.append(
                Trigger(
                    n_i, nt.i0 + int(q), float(np.sign(nt.sweep_signed[q])), float(abs(nt.sweep_signed[q]))
                )
            )
    return _cooldown(cands)


def pm_jump_triggers(nights: list[Night]) -> list[Trigger]:
    cands = []
    for n_i, nt in enumerate(nights):
        stop = nt.offhours_end - nt.i0
        dp5 = np.full(nt.n, np.nan)
        dp5[5:] = nt.p[5:] - nt.p[:-5]
        start_ok = np.zeros(nt.n, dtype=bool)
        start_ok[5:] = (nt.p[:-5] > 0.1) & (nt.p[:-5] < 0.9)
        for q in np.flatnonzero((np.abs(dp5) >= PM_JUMP) & start_ok)[:]:
            if q < stop:
                cands.append(Trigger(n_i, nt.i0 + int(q), float(np.sign(dp5[q])), float(abs(dp5[q]))))
    return _cooldown(cands)


def perp_jump_triggers(nights: list[Night], f: MinuteFeatures) -> tuple[list[Trigger], float]:
    r5s = []
    for nt in nights:
        c = f.close[nt.i0 : nt.offhours_end]
        with np.errstate(invalid="ignore", divide="ignore"):
            r5s.append(np.log(c[5:] / c[:-5]))
    pool = np.abs(np.concatenate(r5s)) if r5s else np.empty(0)
    pool = pool[np.isfinite(pool)]
    if not pool.size:
        return [], float("nan")
    cut = float(np.quantile(pool, PERP_JUMP_QUANTILE))
    cands = []
    for n_i, (nt, r5) in enumerate(zip(nights, r5s, strict=True)):
        for q in np.flatnonzero(np.abs(r5) >= cut):
            cands.append(Trigger(n_i, nt.i0 + int(q) + 5, float(np.sign(r5[q])), float(abs(r5[q]))))
    return _cooldown(cands), cut


def _perp_curve(f: MinuteFeatures, t: Trigger) -> F:
    lo, hi = t.idx - EVENT_PRE, t.idx + EVENT_POST
    if lo < 0 or hi >= f.t.size or f.missing[t.idx]:
        return np.full(EVENT_PRE + EVENT_POST + 1, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.asarray(t.direction * np.log(f.close[lo : hi + 1] / f.close[t.idx]) * 1e4, dtype=np.float64)


def _pm_curve(nights: list[Night], t: Trigger) -> F:
    nt = nights[t.night]
    out = np.full(EVENT_PRE + EVENT_POST + 1, np.nan)
    for k, off in enumerate(range(-EVENT_PRE, EVENT_POST + 1)):
        q = t.idx - nt.i0 + off
        if 0 <= q < nt.n:
            out[k] = t.direction * (nt.p[q] - nt.p[t.idx - nt.i0])
    return out


def _mean_ci(curves: F, rng: np.random.Generator) -> dict[str, Any]:
    ok = curves[np.isfinite(curves[:, EVENT_PRE + EVENT_POST])] if curves.size else curves
    if ok.shape[0] < 5:
        return {"n": int(ok.shape[0])}
    mean = np.nanmean(ok, axis=0)
    boots = np.array(
        [np.nanmean(ok[rng.integers(0, ok.shape[0], ok.shape[0])], axis=0) for _ in range(BOOTSTRAP // 4)]
    )
    lo, hi = np.nanquantile(boots, [0.025, 0.975], axis=0)
    return {"n": int(ok.shape[0]), "mean": mean, "lo": lo, "hi": hi}


def _at(curve: dict[str, Any], minute: int) -> float | None:
    return float(curve["mean"][EVENT_PRE + minute]) if "mean" in curve else None


def _half_life(curve: dict[str, Any]) -> int | None:
    """First minute after the trigger at which the mean path reaches half of its +EVENT_POST value."""
    if "mean" not in curve:
        return None
    post = curve["mean"][EVENT_PRE:]
    final = post[-1]
    if not np.isfinite(final) or final <= 0:
        return None
    hit = np.flatnonzero(post >= final / 2)
    return int(hit[0]) if hit.size else None


def _placebo(
    nights: list[Night], triggers: list[Trigger], f: MinuteFeatures, rng: np.random.Generator
) -> list[float]:
    """Mean signed perp return at +EVENT_POST for each seed: same nights and directions, random minutes."""
    out = []
    for _ in range(PLACEBO_SEEDS):
        vals = []
        for t in triggers:
            nt = nights[t.night]
            q = int(rng.integers(nt.i0 + EVENT_PRE, max(nt.i0 + EVENT_PRE + 1, nt.offhours_end)))
            v = _perp_curve(f, Trigger(t.night, q, t.direction, t.size))[-1]
            if np.isfinite(v):
                vals.append(v)
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


def event_study(
    nights_by_sym: dict[str, list[Night]], feats: dict[str, MinuteFeatures], rng: np.random.Generator
) -> tuple[dict[str, Any], dict[str, Any]]:
    curves: dict[str, list[F]] = defaultdict(list)
    placebo_seeds: dict[str, list[list[float]]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)
    cuts = {}
    for sym, nights in nights_by_sym.items():
        f = feats[sym]
        sweeps, pm_jumps = sweep_triggers(nights), pm_jump_triggers(nights)
        perp_jumps, cuts[sym] = perp_jump_triggers(nights, f)
        for name, trig in (("sweep", sweeps), ("pm_jump", pm_jumps)):
            counts[name] += len(trig)
            curves[f"{name}->perp"] += [_perp_curve(f, t) for t in trig]
            curves[f"{name}->pm"] += [_pm_curve(nights, t) for t in trig]
            if trig:
                placebo_seeds[name].append(_placebo(nights, trig, f, rng))
        counts["perp_jump"] += len(perp_jumps)
        curves["perp_jump->perp"] += [_perp_curve(f, t) for t in perp_jumps]
        curves["perp_jump->pm"] += [_pm_curve(nights, t) for t in perp_jumps]
    summary: dict[str, Any] = {
        "triggers": dict(counts),
        "perp_jump_cut_bps": {k: v * 1e4 for k, v in cuts.items()},
    }
    chart: dict[str, Any] = {}
    for name, cs in curves.items():
        c = _mean_ci(np.array(cs) if cs else np.empty((0, EVENT_PRE + EVENT_POST + 1)), rng)
        chart[name] = c
        unit = "bps" if name.endswith("perp") else "prob"
        summary[name] = {
            "n": c["n"],
            "unit": unit,
            "pre_-30_to_0": None if "mean" not in c else -_at(c, -EVENT_PRE),  # type: ignore[operator]
            "post_5": _at(c, 5),
            "post_15": _at(c, 15),
            "post_60": _at(c, EVENT_POST),
            "post_60_ci": None if "lo" not in c else [float(c["lo"][-1]), float(c["hi"][-1])],
            "half_life_min": _half_life(c),
        }
    for name, per_sym in placebo_seeds.items():
        seeds = np.nanmean(np.array(per_sym), axis=0)
        actual = summary[f"{name}->perp"]["post_60"]
        summary[f"{name}->perp"]["placebo_post_60"] = {
            "seeds": seeds.tolist(),
            "share_at_least_actual": float(np.mean(seeds >= actual)) if actual is not None else None,
        }
        chart[f"{name}->perp"]["placebo"] = seeds
    return summary, chart


# ───────────────────────────── gap, rest of day, Brier ─────────────────────────────


@dataclass
class NightRow:
    sym: str
    day: date
    residual: float
    rest_of_day: float
    p_pm_0928: float
    p_perp_0928: float
    pm_change: float
    flow_imbalance: float
    sweep_net: float
    flow_z_max: float
    vpin_pctl: float
    up: float


def _anchor_values(nt: Night, f: MinuteFeatures, offset: int) -> tuple[float, float, float]:
    """(Polymarket p, ln(S/K), ln(close/S)) at the minute ending `offset` minutes from the open."""
    q = nt.open_idx + offset - nt.i0
    if not 0 <= q < nt.n or f.missing[nt.i0 + q]:
        return float("nan"), float("nan"), float("nan")
    s = float(f.close[nt.i0 + q]) / (1 + nt.basis)
    return float(nt.p[q]), math.log(s / nt.close_prev), math.log(nt.close_day / s)


def night_rows(nights: list[Night], f: MinuteFeatures) -> tuple[list[NightRow], dict[str, dict[str, F]]]:
    rest_hist: dict[str, list[float]] = defaultdict(list)
    brier: dict[str, dict[str, list[float]]] = {a: defaultdict(list) for a in BRIER_ANCHORS}
    rows = []
    for nt in nights:
        sig: dict[str, float] = {}
        for name, off in BRIER_ANCHORS.items():
            p_pm, lsk, rest = _anchor_values(nt, f, off)
            hist = rest_hist[name][-HISTORY_NIGHTS:]
            sigma = float(np.std(hist)) if len(hist) >= MIN_HISTORY_NIGHTS else float("nan")
            p_perp = float(norm_cdf(np.array([lsk / sigma]))[0]) if sigma > 0 else float("nan")
            if np.isfinite(p_pm) and np.isfinite(p_perp):
                brier[name]["pm"].append(p_pm)
                brier[name]["perp"].append(p_perp)
                brier[name]["up"].append(nt.payout_up)
            if np.isfinite(rest):
                rest_hist[name].append(rest)
            sig[name] = p_perp
            if name == "09:28":
                # direction tests read Polymarket one sample earlier (stamped <= 09:28:30): no look-ahead
                q = nt.open_idx + off - nt.i0
                sig["p_pm"] = float(nt.p[q - 1]) if 0 < q <= nt.n else float("nan")
                sig["rest"], sig["lsk"] = rest, lsk
        q_end = nt.offhours_end - nt.i0 + 2  # the 09:27 bar: flow up to 09:28:00
        tot = float(nt.flow_total[: q_end + 1].sum())
        first_p = nt.p[np.isfinite(nt.p)][:1]
        rows.append(
            NightRow(
                sym=nt.sym,
                day=nt.day,
                residual=(nt.open_day / nt.close_prev) / math.exp(sig["lsk"]) - 1
                if np.isfinite(sig["lsk"])
                else np.nan,
                rest_of_day=sig["rest"],
                p_pm_0928=sig["p_pm"],
                p_perp_0928=sig["09:28"],
                pm_change=sig["p_pm"] - float(first_p[0]) if first_p.size else np.nan,
                flow_imbalance=float(nt.flow_signed[: q_end + 1].sum()) / tot if tot else np.nan,
                sweep_net=float(nt.sweep_signed[: q_end + 1].sum())
                if np.isfinite(nt.sweep_threshold)
                else np.nan,
                flow_z_max=float(np.nanmax(nt.flow_z[: q_end + 1]))
                if np.isfinite(nt.flow_z[: q_end + 1]).any()
                else np.nan,
                vpin_pctl=float(f.pctl_session[nt.open_idx - 2]),
                up=nt.payout_up,
            )
        )
    arrays = {a: {k: np.array(v) for k, v in d.items()} for a, d in brier.items()}
    return rows, arrays


def brier_table(arrays: dict[str, dict[str, F]], rng: np.random.Generator) -> dict[str, Any]:
    out = {}
    for anchor, d in arrays.items():
        if "up" not in d or d["up"].size < 20:
            continue
        e_pm, e_perp = (d["pm"] - d["up"]) ** 2, (d["perp"] - d["up"]) ** 2
        diff = e_pm - e_perp
        boots = [float(np.mean(diff[rng.integers(0, diff.size, diff.size)])) for _ in range(BOOTSTRAP)]
        out[anchor] = {
            "n": int(diff.size),
            "brier_pm": float(e_pm.mean()),
            "brier_perp": float(e_perp.mean()),
            "diff_pm_minus_perp": float(diff.mean()),
            "diff_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "brier_coinflip": 0.25,
        }
    return out


def gap_tests(rows: list[NightRow], rng: np.random.Generator) -> dict[str, Any]:
    col = {
        k: np.array([getattr(r, k) for r in rows], dtype=np.float64)
        for k in NightRow.__dataclass_fields__
        if k not in ("sym", "day")
    }
    divergence = col["p_pm_0928"] - col["p_perp_0928"]
    direction = {
        "pm_divergence_0928": divergence,
        "pm_prob_change_offhours": col["pm_change"],
        "pm_flow_imbalance_offhours": col["flow_imbalance"],
        "pm_sweep_net_offhours": col["sweep_net"],
    }
    magnitude = {
        "pm_flow_z_max_offhours": col["flow_z_max"],
        "stock_vpin_pctl_0928": col["vpin_pctl"],
        "abs_pm_divergence_0928": np.abs(divergence),
    }
    targets = {"residual_gap": col["residual"], "rest_of_day": col["rest_of_day"]}
    out: dict[str, Any] = {"n_nights": len(rows), "direction": {}, "magnitude": {}}
    for name, s in direction.items():
        out["direction"][name] = {
            t: {**direction_accuracy(s, y, 5e-4), **spearman_test(s, y, rng)} for t, y in targets.items()
        }
    for name, s in magnitude.items():
        out["magnitude"][name] = {t: spearman_test(s, np.abs(y), rng) for t, y in targets.items()}
    tests = len(direction) * len(targets) + len(magnitude) * len(targets)
    out["bonferroni_alpha"] = 0.05 / tests
    out["n_tests"] = tests
    return out


# ───────────────────────────── earnings ─────────────────────────────


def session_samples(nights: list[Night], f: MinuteFeatures) -> dict[str, list[tuple[float, float, float]]]:
    """(p_perp, p_pm, outcome) every CALIBRATION_STEP minutes, grouped by the session of that minute.

    σ for each anchor (fixed offset from the open) is the std of ln(close / S) at the same anchor over
    the symbol's previous HISTORY_NIGHTS nights: point-in-time.
    """
    rest_hist: dict[int, list[float]] = defaultdict(list)
    out: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for nt in nights:
        first = nt.i0 - nt.open_idx
        for off in range(first - first % CALIBRATION_STEP, nt.i1 - nt.open_idx - 5, CALIBRATION_STEP):
            p_pm, lsk, rest = _anchor_values(nt, f, off)
            hist = rest_hist[off][-HISTORY_NIGHTS:]
            if np.isfinite(rest):
                rest_hist[off].append(rest)
            if len(hist) < MIN_HISTORY_NIGHTS or not (np.isfinite(p_pm) and np.isfinite(lsk)):
                continue
            sigma = float(np.std(hist))
            if sigma <= 0 or not 0.03 < p_pm < 0.97:
                continue
            p_perp = float(norm_cdf(np.array([lsk / sigma]))[0])
            out[SESSION_NAMES[int(f.session[nt.open_idx + off])]].append((p_perp, p_pm, nt.payout_up))
    return out


def session_calibration(
    samples: dict[str, list[tuple[float, float, float]]], rng: np.random.Generator
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for sess, rows in samples.items():
        a = np.array(rows)
        if a.shape[0] < 50:
            continue
        perp, pm, y = a[:, 0], a[:, 1], a[:, 2]
        diff = (pm - y) ** 2 - (perp - y) ** 2
        boots = [float(np.mean(diff[rng.integers(0, diff.size, diff.size)])) for _ in range(BOOTSTRAP // 4)]
        big = np.abs(perp - pm) > DISAGREE
        perp_right = ((y > 0.5) == (perp > pm))[big]
        out[sess] = {
            "n": int(a.shape[0]),
            "brier_perp": float(np.mean((perp - y) ** 2)),
            "brier_pm": float(np.mean((pm - y) ** 2)),
            "diff_pm_minus_perp_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "perp_right_when_disagree": rate(int(perp_right.sum()), int(big.sum())),
        }
    return out


def earnings_cases(
    store: TradfiStore, cal: SessionCalendar, markets: list[PmMarket], feats: dict[str, MinuteFeatures]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases, paths = [], {}
    for m in sorted((m for m in markets if m.kind == "earnings"), key=lambda m: (m.day, m.sym)):
        f = feats.get(m.sym)
        if f is None or m.payout_yes not in (0.0, 1.0):
            continue
        # release: the largest 5-minute perp move outside regular hours, release-day 04:00 ET -> next open
        start = cal.open_ms(m.day) - 330 * MIN_MS
        nxt = m.day + timedelta(days=1)
        while not cal.is_trading_day(nxt):
            nxt += timedelta(days=1)
        stop = cal.open_ms(nxt)
        a, b = (int(x) for x in f.at(np.array([start, stop], dtype=np.float64)))
        if a < 0 or b < 0 or b - a < 60:
            continue
        with np.errstate(invalid="ignore", divide="ignore"):
            r5 = np.log(f.close[a + 5 : b] / f.close[a : b - 5])
        r5[f.session[a + 5 : b] == Session.RTH] = np.nan
        if not np.isfinite(r5).any():
            continue
        star = a + 5 + int(np.nanargmax(np.abs(r5))) - 5  # minute the move starts
        pt, pp = load_prices(store, m.slug)
        if pt.size < 30 or star - 120 < 0 or star + 120 >= f.t.size:
            continue
        offs = np.arange(-120, 121)
        p = sample_at(pt, pp, f.t[star + offs] + MIN_MS)
        with np.errstate(invalid="ignore", divide="ignore"):
            perp = np.log(f.close[star + offs] / f.close[star - 1]) * 1e4
        outcome = m.payout_yes
        react = float(perp[120 + 60])
        pm_toward = np.where(outcome == 1, p, 1 - p)
        reached = np.flatnonzero(pm_toward[120:] >= 0.9)
        perp_half = np.flatnonzero(np.sign(react) * perp[120:] >= abs(react) / 2)
        cases.append(
            {
                "sym": m.sym,
                "release_day": m.day.isoformat(),
                "question": m.question,
                "beat": bool(outcome == 1),
                "t_star_utc_ms": float(f.t[star]),
                "session_at_release": SESSION_NAMES[int(f.session[star])],
                "p_minus_60": float(p[60]),
                "p_minus_1": float(p[119]),
                "p_plus_5": float(p[125]),
                "p_plus_30": float(p[150]),
                "perp_pre_60_bps": float(perp[119] - perp[60]),
                "perp_reaction_60_bps": react,
                "pm_minutes_to_90pct": int(reached[0]) if reached.size else None,
                "perp_minutes_to_half_reaction": int(perp_half[0]) if perp_half.size else None,
                "pm_right_side_at_minus_60": bool((p[60] > 0.5) == (outcome == 1))
                if np.isfinite(p[60])
                else None,
                "reaction_agrees_with_beat": bool((react > 0) == (outcome == 1)),
            }
        )
        paths[f"{m.sym} {m.day}"] = {
            "offsets": offs,
            "p_toward_outcome": pm_toward,
            "perp_toward_reaction_bps": np.sign(react) * perp,
        }
    return cases, paths


# ───────────────────────────── run ─────────────────────────────


def run(
    store: TradfiStore, symbols: tuple[str, ...], seed: int = 11
) -> tuple[dict[str, Any], dict[str, Any]]:
    cal = SessionCalendar()
    rng = np.random.default_rng(seed)
    markets = load_markets(store)
    feats: dict[str, MinuteFeatures] = {}
    nights_by_sym: dict[str, list[Night]] = {}
    results: dict[str, Any] = {"assumptions": ASSUMPTIONS, "symbols": {}, "skipped": {}}
    rows: list[NightRow] = []
    brier_parts: list[dict[str, dict[str, F]]] = []
    calib: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    leadlag_nights: list[tuple[list[Night], MinuteFeatures]] = []
    for sym in symbols:
        if not any(m.sym == sym for m in markets):
            continue
        f = load_features(store, cal, sym)
        feats[sym] = f
        nights, skipped = build_nights(store, cal, f, sym, markets)
        results["skipped"][sym] = skipped
        if not nights:
            continue
        nights_by_sym[sym] = nights
        r, b = night_rows(nights, f)
        rows += r
        brier_parts.append(b)
        for sess, samples in session_samples(nights, f).items():
            calib[sess] += samples
        leadlag_nights.append((nights, f))
        vol = [float(nt.flow_total.sum()) for nt in nights]
        thresholds = [nt.sweep_threshold for nt in nights if np.isfinite(nt.sweep_threshold)]
        results["symbols"][sym] = {
            "nights": len(nights),
            "median_night_taker_usd": float(np.median(vol)),
            "median_sweep_threshold_usd": float(np.median(thresholds)) if thresholds else None,
            "offhours_share_of_pm_volume": float(
                np.sum([nt.flow_total[: nt.offhours_end - nt.i0].sum() for nt in nights]) / max(sum(vol), 1.0)
            ),
            "leadlag": lead_lag(nights, f),
        }
        log.info("Polymarket %s: %d nights", sym, len(nights))
    pooled_ll = _pooled_leadlag(leadlag_nights)
    results["leadlag_pooled"] = pooled_ll
    events, event_chart = event_study(nights_by_sym, feats, rng)
    results["events"] = events
    merged: dict[str, dict[str, F]] = {}
    for anchor in BRIER_ANCHORS:
        parts = [b[anchor] for b in brier_parts if "up" in b[anchor]]
        if parts:
            merged[anchor] = {k: np.concatenate([p[k] for p in parts]) for k in ("pm", "perp", "up")}
    results["brier"] = brier_table(merged, rng)
    results["session_calibration"] = session_calibration(calib, rng)
    results["gap"] = gap_tests(rows, rng)
    cases, paths = earnings_cases(store, cal, markets, feats)
    results["earnings"] = {"n": len(cases), "cases": cases, **_earnings_summary(cases)}
    results["n_nights"] = len(rows)
    charts = {"leadlag": pooled_ll, "events": event_chart, "earnings": paths}
    return results, charts


def _pooled_leadlag(parts: list[tuple[list[Night], MinuteFeatures]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for step, max_lag in ((1, 10), (5, 6)):
        dps, rs, ss = [], [], []
        for nights, f in parts:
            dp, r, s = _stitched(nights, f, step, max_lag + 1)
            dps.append(dp)
            rs.append(r)
            ss.append(s)
        if not dps:
            continue
        dp, r, s = np.concatenate(dps), np.concatenate(rs), np.concatenate(ss)
        per = {}
        for name, m in (
            ("offhours", np.isin(s, [Session.POST, Session.OVERNIGHT, Session.PRE, Session.CLOSED])),
            ("rth", s == Session.RTH),
        ):
            cc = xcorr(np.where(m, dp, np.nan), np.where(m, r, np.nan), max_lag)
            finite = {k: v for k, v in cc.items() if np.isfinite(v)}
            per[name] = {
                "xcorr": finite,
                "peak_lag": max(finite, key=lambda k: abs(finite[k])) if finite else None,
                "minutes": int(np.sum(m & np.isfinite(dp) & np.isfinite(r))),
            }
        out[f"{step}m"] = per
    return out


def _earnings_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if not cases:
        return {}
    right = [c["pm_right_side_at_minus_60"] for c in cases if c["pm_right_side_at_minus_60"] is not None]
    pm_lag = [c["pm_minutes_to_90pct"] for c in cases if c["pm_minutes_to_90pct"] is not None]
    perp_lag = [
        c["perp_minutes_to_half_reaction"] for c in cases if c["perp_minutes_to_half_reaction"] is not None
    ]
    return {
        "beat_rate": rate(sum(c["beat"] for c in cases), len(cases)),
        "pm_right_side_1h_before": rate(sum(right), len(right)),
        "reaction_agrees_with_beat": rate(sum(c["reaction_agrees_with_beat"] for c in cases), len(cases)),
        "median_pm_minutes_to_90pct": statistics.median(pm_lag) if pm_lag else None,
        "median_perp_minutes_to_half_reaction": statistics.median(perp_lag) if perp_lag else None,
    }
