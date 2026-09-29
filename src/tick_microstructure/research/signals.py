"""Trigger signals as on/off intervals on the local clock (ms), built from captured streams.

Native resolution is kept: a FollowSM state turns on at the receive time of the first frame that
satisfies it; an event signal (sweep, move) is on for `hold_ms` from its receive time. That keeps
the millisecond latency budget meaningful in the simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.data import Dataset, F
from tick_microstructure.analytics.features import detect_moves, detect_sweeps

STALE_MS = 3_000.0  # FollowSM toxicity frames arrive about once a second


@dataclass
class Intervals:
    """Sorted, non-overlapping [start, end] intervals (ms)."""

    starts: F
    ends: F

    @classmethod
    def empty(cls) -> Intervals:
        return cls(np.empty(0), np.empty(0))

    @classmethod
    def merge(cls, starts: F, ends: F) -> Intervals:
        if starts.size == 0:
            return cls.empty()
        order = np.argsort(starts, kind="stable")
        s, e = starts[order], ends[order]
        out_s, out_e = [float(s[0])], [float(e[0])]
        for a, b in zip(s[1:], e[1:], strict=True):
            if a <= out_e[-1]:
                out_e[-1] = max(out_e[-1], float(b))
            else:
                out_s.append(float(a))
                out_e.append(float(b))
        return cls(np.asarray(out_s), np.asarray(out_e))

    def union(self, other: Intervals) -> Intervals:
        return Intervals.merge(
            np.concatenate([self.starts, other.starts]), np.concatenate([self.ends, other.ends])
        )

    def delayed(self, latency_ms: float) -> Intervals:
        """Effective protection window: cancels land and re-quotes go out `latency_ms` late."""
        return Intervals(self.starts + latency_ms, self.ends + latency_ms)

    def contains(self, times: F) -> npt.NDArray[np.bool_]:
        if self.starts.size == 0:
            return np.zeros(times.shape, dtype=bool)
        i = np.searchsorted(self.starts, times, side="right") - 1
        ok = i >= 0
        out = np.zeros(times.shape, dtype=bool)
        out[ok] = times[ok] <= self.ends[i[ok]]
        return out

    def clip(self, lo: float, hi: float) -> Intervals:
        s, e = np.maximum(self.starts, lo), np.minimum(self.ends, hi)
        keep = e > s
        return Intervals(s[keep], e[keep])

    def active_ms(self) -> float:
        return float((self.ends - self.starts).sum())

    def shifted(self, shift_ms: float, lo: float, hi: float) -> Intervals:
        """Circular shift within [lo, hi): same count, durations and clustering, new timing."""
        span = hi - lo
        s = (self.starts - lo + shift_ms) % span + lo
        e = s + (self.ends - self.starts)
        wrap = e > hi
        starts = np.concatenate([s, np.full(int(wrap.sum()), lo)])
        ends = np.concatenate([np.minimum(e, hi), lo + (e[wrap] - hi)])
        return Intervals.merge(starts, ends)

    def __len__(self) -> int:
        return int(self.starts.size)


def state_intervals(t: F, condition: npt.NDArray[np.bool_], stale_ms: float = STALE_MS) -> Intervals:
    """On from the first frame where `condition` holds until the next frame where it doesn't
    (or `stale_ms` after the last frame if the stream went quiet)."""
    if t.size == 0:
        return Intervals.empty()
    starts: list[float] = []
    ends: list[float] = []
    on_since: float | None = None
    for i in range(t.size):
        if on_since is not None and i > 0 and t[i] - t[i - 1] > stale_ms:
            starts.append(on_since)
            ends.append(float(t[i - 1]) + stale_ms)
            on_since = None
        if condition[i] and on_since is None:
            on_since = float(t[i])
        elif not condition[i] and on_since is not None:
            starts.append(on_since)
            ends.append(float(t[i]))
            on_since = None
    if on_since is not None:
        starts.append(on_since)
        ends.append(float(t[-1]) + stale_ms)
    return Intervals.merge(np.asarray(starts), np.asarray(ends))


def event_intervals(times: F, hold_ms: float) -> Intervals:
    return Intervals.merge(times, times + hold_ms)


@dataclass(frozen=True)
class TriggerSpec:
    vpin_pct: float = 0.95
    obt: float = 2.0
    sweep_usd: float = 100_000.0
    sweep_hold_ms: float = 5_000.0
    move_window_ms: float = 2_000.0
    move_bps: float = 10.0
    move_hold_ms: float = 5_000.0


def build_triggers(ds: Dataset, sym: str, spec: TriggerSpec | None = None) -> dict[str, Intervals]:
    """Individual triggers plus the composite `vpin_pct | sweep | obt` from the research plan.

    `binance_move` is a public-data reactive baseline (no FollowSM): a trigger has to beat it to
    justify a paid feed.
    """
    spec = spec or TriggerSpec()
    out: dict[str, Intervals] = {}
    tox = ds.toxicity.get(sym)
    if tox is not None and tox.t.size:
        vp = np.nan_to_num(tox.vpin_pct, nan=-1.0)
        out["vpin_pct"] = state_intervals(tox.t, vp >= spec.vpin_pct)
        out["obt"] = state_intervals(tox.t, tox.obt > spec.obt)
    trades = ds.trades.get(sym)
    if trades is not None and trades.t.size:
        sweeps = detect_sweeps(trades, spec.sweep_usd)
        out["sweep"] = event_intervals(sweeps.rt, spec.sweep_hold_ms)
        moves = detect_moves(trades, spec.move_window_ms, spec.move_bps, spec.move_window_ms)
        out["binance_move"] = event_intervals(moves.rt, spec.move_hold_ms)
    if all(k in out for k in ("vpin_pct", "sweep", "obt")):
        out["composite"] = out["vpin_pct"].union(out["sweep"]).union(out["obt"])
    return out
