"""Task 3: hypothesis tests. Every result is reported next to a control, with Wilson 95% intervals.

H1  L1/L2 depth depletion: does ob_toxicity_1pct > 2 (ask depth >= 2x bid depth within +-1%)
    predict a Binance mid move of <= -10 bps within 10 / 30 / 60 s?
    Control: random FollowSM frames where the condition was false. A directional signal must
    raise down-moves more than up-moves; both are reported. The threshold sweep is walk-forward:
    chosen on the first half of the sample, scored on the second.
H2  Whale sweep cascade: does a Binance taker sweep >= $100k confirmed by a same-direction book
    spike (next FollowSM frames, within 5 s) make Polymarket reprice sooner/more often?
    Controls: unconfirmed sweeps, and random times with a random direction.
H3  Composite vs individual triggers as detectors of Binance adverse-move onsets (>= 10 bps in
    2 s): time active (cost), coverage at a lead (active L ms before the onset), precision
    (onsets within 30 s of a trigger onset). Placebo: the same intervals circularly shifted.
    The P&L side of H3 is in simulation/.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.data import Book, Dataset, F, Toxicity
from tick_microstructure.analytics.features import detect_moves, detect_sweeps, onsets
from tick_microstructure.analytics.leadlag import Trigger, reprice_lags
from tick_microstructure.analytics.stats import placebo_times, rate
from tick_microstructure.research.signals import Intervals, TriggerSpec, build_triggers

H1_HORIZONS_S = (10, 30, 60)
H1_MOVE_BPS = 10.0
H1_THRESHOLDS = (1.5, 2.0, 3.0, 5.0)
H1_MIN_EVENTS = 30
REFRACTORY_MS = 60_000.0
H2_CONFIRM_MS = 5_000.0
H2_WINDOWS_S = (5, 10, 30)
H3_LEADS_MS = (0, 1_000, 5_000)
H3_PRECISION_MS = 30_000.0
H3_MOVE_WINDOW_MS = 2_000.0
H3_MOVE_BPS = 10.0
PLACEBO_SHIFTS = 50


def _path_extremes(book: Book, t0: F, horizon_ms: float) -> tuple[F, F, F]:
    """(min, max, end) log-return in bps of the Binance mid over [t0, t0 + horizon]."""
    i0 = book.index_at(t0)
    i1 = book.index_at(t0 + horizon_ms)
    mid = book.mid
    lo = np.full(t0.shape, np.nan)
    hi = np.full(t0.shape, np.nan)
    end = np.full(t0.shape, np.nan)
    for k in range(t0.size):
        a, b = int(i0[k]), int(i1[k])
        if a < 0 or b <= a:
            continue
        seg = np.log(mid[a : b + 1] / mid[a]) * 1e4
        lo[k], hi[k], end[k] = seg.min(), seg.max(), seg[-1]
    return lo, hi, end


def _clean(ds: Dataset, times: F, horizon_ms: float) -> F:
    keep = [
        not ds.gaps["followsm"].overlaps(t, t + horizon_ms)
        and not ds.gaps["binance"].overlaps(t, t + horizon_ms)
        for t in times
    ]
    return times[np.asarray(keep, dtype=bool)] if times.size else times


def _outcomes(book: Book, times: F, horizon_ms: float) -> dict[str, Any]:
    lo, hi, end = _path_extremes(book, times, horizon_ms)
    ok = np.isfinite(lo)
    n = int(ok.sum())
    return {
        "down": rate(int((lo[ok] <= -H1_MOVE_BPS).sum()), n),
        "up": rate(int((hi[ok] >= H1_MOVE_BPS).sum()), n),
        "mean_end_bps": float(end[ok].mean()) if n else None,
    }


def _h1_for(
    ds: Dataset, tox: Toxicity, book: Book, threshold: float, lo: float, hi: float, rng: np.random.Generator
) -> dict[str, Any]:
    window = (tox.t >= lo) & (tox.t < hi)
    t, obt = tox.t[window], tox.obt[window]
    events = t[onsets(t, obt > threshold, REFRACTORY_MS)]
    calm = t[obt <= threshold]
    control = (
        np.sort(rng.choice(calm, size=min(calm.size, max(2_000, 5 * events.size)), replace=False))
        if calm.size
        else calm
    )
    out: dict[str, Any] = {"threshold": threshold}
    for h in H1_HORIZONS_S:
        hms = h * 1000.0
        sig = _outcomes(book, _clean(ds, events, hms), hms)
        ctl = _outcomes(book, _clean(ds, control, hms), hms)
        base = ctl["down"]["rate"]
        sig["lift_down"] = sig["down"]["rate"] / base if base and sig["down"]["rate"] is not None else None
        sig["false_positive_rate"] = 1 - sig["down"]["rate"] if sig["down"]["rate"] is not None else None
        out[f"{h}s"] = {"signal": sig, "control": ctl}
    return out


def h1_depth_depletion(ds: Dataset, seed: int = 11) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}
    for sym, tox in sorted(ds.toxicity.items()):
        book = ds.book.get(sym)
        if book is None or tox.t.size == 0:
            continue
        mid_t = ds.start_ms + (ds.end_ms - ds.start_ms) / 2
        full = _h1_for(ds, tox, book, 2.0, ds.start_ms, ds.end_ms, rng)
        train = {thr: _h1_for(ds, tox, book, thr, ds.start_ms, mid_t, rng) for thr in H1_THRESHOLDS}

        def score(res: dict[str, Any]) -> float:
            sig = res["30s"]["signal"]
            return float(sig["lift_down"]) if sig["down"]["n"] >= H1_MIN_EVENTS and sig["lift_down"] else -1.0

        best = max(H1_THRESHOLDS, key=lambda thr: score(train[thr]))
        chosen = best if score(train[best]) > 0 else None
        report[sym] = {
            "plan_threshold_2.0_full_sample": full,
            "walk_forward": {
                "train_until_ms": mid_t,
                "train_scores_30s_lift": {str(k): score(v) for k, v in train.items()},
                "chosen_threshold": chosen,
                "test": _h1_for(ds, tox, book, chosen, mid_t, ds.end_ms, rng) if chosen else None,
            },
        }
    return report


def h2_sweep_cascade(ds: Dataset, spec: TriggerSpec | None = None, seed: int = 13) -> dict[str, Any]:
    spec = spec or TriggerSpec()
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}
    for sym in sorted({m.sym for m in ds.markets.values()}):
        trades, tox = ds.trades.get(sym), ds.toxicity.get(sym)
        if trades is None or trades.t.size == 0:
            continue
        sweeps = detect_sweeps(trades, spec.sweep_usd)
        clean = np.asarray(
            [not ds.gaps["all"].overlaps(t - H2_CONFIRM_MS, t + 60_000) for t in sweeps.rt], dtype=bool
        )
        confirmed = np.zeros(sweeps.t.size, dtype=bool)
        confirm_at = sweeps.rt.copy()
        if tox is not None and tox.t.size:
            for k, (rt, side) in enumerate(zip(sweeps.rt, sweeps.side, strict=True)):
                j0 = int(np.searchsorted(tox.t, rt, side="right"))
                j1 = int(np.searchsorted(tox.t, rt + H2_CONFIRM_MS, side="right"))
                obt = tox.obt[j0:j1]
                # a sell sweep is confirmed by an ask-heavy book, a buy sweep by a bid-heavy one
                hit = np.flatnonzero(obt > spec.obt if side < 0 else obt < 1 / spec.obt)
                if hit.size:
                    confirmed[k], confirm_at[k] = True, tox.t[j0 + int(hit[0])]
        placebo_t = placebo_times(
            rng, max(200, sweeps.t.size), ds.start_ms, ds.end_ms, ds.gaps["all"], 60_000
        )
        placebo = Trigger(placebo_t, placebo_t, rng.choice(np.array([-1.0, 1.0]), size=placebo_t.size))
        report[sym] = {
            "sweeps": int(sweeps.t.size),
            "confirmed_from_sweep": _sweep_group(
                ds, sym, sweeps.t, sweeps.rt, sweeps.side, clean & confirmed
            ),
            "confirmed_from_confirmation": _sweep_group(
                ds, sym, sweeps.t + (confirm_at - sweeps.rt), confirm_at, sweeps.side, clean & confirmed
            ),
            "unconfirmed": _sweep_group(ds, sym, sweeps.t, sweeps.rt, sweeps.side, clean & ~confirmed),
            "placebo": _response(reprice_lags(ds, sym, placebo), sweeps=int(placebo_t.size)),
        }
    return report


def _sweep_group(
    ds: Dataset, sym: str, t_exchange: F, t_local: F, side: F, mask: npt.NDArray[np.bool_]
) -> dict[str, Any]:
    trig = Trigger(t_exchange[mask], t_local[mask], side[mask])
    return _response(reprice_lags(ds, sym, trig), sweeps=int(mask.sum()))


def _response(rows: list[dict[str, float]], sweeps: int) -> dict[str, Any]:
    n = len(rows)
    lags = np.asarray([r["local_ms"] for r in rows if r["responded"]])
    out: dict[str, Any] = {"events": sweeps, "market_pairs": n}
    for w in H2_WINDOWS_S:
        out[f"reprice_within_{w}s"] = rate(int((lags <= w * 1000).sum()), n)
    out["median_lag_ms"] = float(np.median(lags)) if lags.size else None
    return out


def h3_trigger_detection(ds: Dataset, spec: TriggerSpec | None = None, seed: int = 17) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    report: dict[str, Any] = {}
    for sym in sorted(ds.trades):
        moves = detect_moves(ds.trades[sym], H3_MOVE_WINDOW_MS, H3_MOVE_BPS, H3_PRECISION_MS)
        targets = moves.rt
        if targets.size == 0:
            continue
        span = (ds.start_ms, ds.end_ms)
        per: dict[str, Any] = {"move_onsets": int(targets.size)}
        for name, iv in build_triggers(ds, sym, spec).items():
            iv = iv.clip(*span)
            actual = _detect_scores(iv, targets, span)
            shifts = rng.uniform(0.05, 0.95, size=PLACEBO_SHIFTS) * (span[1] - span[0])
            placebo = [_detect_scores(iv.shifted(s, *span), targets, span) for s in shifts]
            per[name] = {
                "actual": actual,
                "placebo_mean": {k: float(np.mean([p[k] for p in placebo])) for k in actual},
                "placebo_share_at_least_as_good": {
                    k: float(np.mean([p[k] >= actual[k] for p in placebo]))
                    for k in actual
                    if k != "active_share"
                },
            }
        report[sym] = per
    return report


def _detect_scores(iv: Intervals, targets: F, span: tuple[float, float]) -> dict[str, float]:
    scores = {"active_share": iv.active_ms() / (span[1] - span[0]), "onsets": float(len(iv))}
    for lead in H3_LEADS_MS:
        scores[f"coverage_lead_{lead}ms"] = float(iv.contains(targets - lead).mean())
    if len(iv):
        follow = np.searchsorted(targets, iv.starts, side="left")
        ok = follow < targets.size
        within = np.zeros(iv.starts.size, dtype=bool)
        within[ok] = targets[follow[ok]] - iv.starts[ok] <= H3_PRECISION_MS
        scores["precision_30s"] = float(within.mean())
    else:
        scores["precision_30s"] = 0.0
    return scores
