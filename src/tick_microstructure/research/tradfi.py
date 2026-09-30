"""Session-aware minute features for tokenized-equity perps (and the 24/7 crypto control).

Everything is computed on a regular 1-minute grid of the perp, from Binance 1m klines (exact taker-buy
notional per minute) and 30-second `bookDepth` snapshots, with NumPy only:

    VPIN            50-bucket window; buckets filled in fractional-bucket units, so a bucket
                    boundary inside a minute splits that minute's flow proportionally
      global V      trailing 7-day ADV / 50 (one size for every hour)
      session V     trailing 7-day volume *per minute of the same session* x 28.8 min, i.e. ADV/50
                    if the whole day traded like this session. A bucket then takes about as long to
                    fill at 03:00 on a Sunday as at 10:00 on a Tuesday.
    pctl            midrank among the trailing 7 days (5-minute samples, refreshed at 00:00 UTC)
    natr_15m        ATR(14) of completed 15m candles / close
    vol_z           last-15m volume vs the previous 96 candles (naive, as in the crypto backtest)
    vol_z_session   the same volume vs the same local 15-minute slot on the previous 20 days of
                    the same kind (trading day or not), so the open's routine burst is not a spike
    depth           +-1% / +-0.2% cumulative notional from the latest snapshot <= 90 s old
    ob_pctl         midrank of ln(ask_1% / bid_1%) among the trailing 24 h of snapshots (hourly
                    refresh, at least 720), the backend's `ob_imbalance_percentile` rule

Nothing looks ahead: every trailing statistic uses data strictly before the minute it is applied to.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import timedelta
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tick_microstructure.research.data.store import TradfiStore, read_columns, write_table
from tick_microstructure.research.utils.session import DAY_MS, HOUR_MS, Session, SessionCalendar

F = npt.NDArray[np.float64]
B = npt.NDArray[np.bool_]
MIN_MS = 60_000
VPIN_BUCKETS = 50
BUCKET_MINUTES = 1440 / VPIN_BUCKETS  # 28.8: ADV / 50 expressed as minutes of typical flow
TRAILING_DAYS = 7
PCTL_SAMPLE_EVERY = 5
PCTL_MIN_SAMPLES = 1_000  # ~3.5 days of 5-minute samples
OB_PCTL_WINDOW_MS = 24 * HOUR_MS
OB_PCTL_MIN_SAMPLES = 720
DEPTH_MAX_AGE_MS = 90_000
Z_SLOT_DAYS = 20
N_SESSIONS = len(Session)


def ffill_index(valid: B) -> npt.NDArray[np.int64]:
    """Index of the last valid element at or before each position (0 before the first)."""
    idx = np.where(valid, np.arange(valid.size), 0)
    return np.maximum.accumulate(idx)


def vpin_series(quote: F, taker_buy: F, bucket: F, n_buckets: int = VPIN_BUCKETS) -> F:
    """VPIN after each minute; NaN until `n_buckets` buckets have closed.

    Minutes with no usable bucket size (NaN / <= 0) add no volume, so VPIN holds its last value.
    """
    ok = np.isfinite(bucket) & (bucket > 0) & (quote > 0)
    u = np.where(ok, quote / np.where(ok, bucket, 1.0), 0.0)
    b = np.where(ok, taker_buy / np.where(ok, bucket, 1.0), 0.0)
    cum_u = np.cumsum(u)
    out = np.full(quote.shape, np.nan)
    n_closed = int(np.floor(cum_u[-1] + 1e-9)) if cum_u.size else 0
    if n_closed < n_buckets:
        return out
    pos = u > 0  # interpolate on strictly increasing cumulative units
    xs = np.concatenate([[0.0], cum_u[pos]])
    ys = np.concatenate([[0.0], np.cumsum(b)[pos]])
    buy_at_edge = np.interp(np.arange(n_closed + 1, dtype=np.float64), xs, ys)
    imbalance = np.abs(2 * np.diff(buy_at_edge) - 1)
    window = np.cumsum(np.concatenate([[0.0], imbalance]))
    vpin_by_bucket = (window[n_buckets:] - window[:-n_buckets]) / n_buckets  # bucket k >= n-1
    closed_by = np.floor(cum_u + 1e-9).astype(np.int64)
    have = closed_by >= n_buckets
    out[have] = vpin_by_bucket[closed_by[have] - n_buckets]
    return out


def bucket_sizes(t: F, quote: F, session: npt.NDArray[np.int8]) -> tuple[F, F]:
    """(global V, session V) per minute from the trailing 7 UTC days (excluding the current one)."""
    day = np.floor_divide(t, DAY_MS).astype(np.int64)
    di = day - day[0]
    nd = int(di[-1]) + 1
    q = np.zeros((nd, N_SESSIONS))
    m = np.zeros((nd, N_SESSIONS))
    np.add.at(q, (di, session), quote)
    np.add.at(m, (di, session), 1.0)
    cq = np.vstack([np.zeros(N_SESSIONS), np.cumsum(q, axis=0)])
    cm = np.vstack([np.zeros(N_SESSIONS), np.cumsum(m, axis=0)])
    trail_q = np.full((nd, N_SESSIONS), np.nan)
    trail_m = np.full((nd, N_SESSIONS), np.nan)
    trail_q[TRAILING_DAYS:] = cq[TRAILING_DAYS:nd] - cq[: nd - TRAILING_DAYS]
    trail_m[TRAILING_DAYS:] = cm[TRAILING_DAYS:nd] - cm[: nd - TRAILING_DAYS]
    adv = trail_q.sum(axis=1) / TRAILING_DAYS
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(trail_m > 0, trail_q / trail_m, np.nan)
    return adv[di] / VPIN_BUCKETS, rate[di, session] * BUCKET_MINUTES


def midrank(reference_sorted: F, values: F) -> F:
    n = reference_sorted.size
    lo = np.searchsorted(reference_sorted, values, side="left")
    hi = np.searchsorted(reference_sorted, values, side="right")
    return np.asarray((lo + hi) / 2 / n, dtype=np.float64)


def trailing_percentile(
    t: F,
    values: F,
    window_ms: float,
    refresh_ms: float,
    min_samples: int,
    sample_every: int = 1,
) -> F:
    """Midrank of each value among samples in [block start - window, block start), where blocks are
    `refresh_ms` long. The reference never contains the block being ranked (no look-ahead)."""
    out = np.full(values.shape, np.nan)
    st, sv = t[::sample_every], values[::sample_every]
    keep = np.isfinite(sv)
    st, sv = st[keep], sv[keep]
    block = np.floor_divide(t, refresh_ms).astype(np.int64)
    starts = np.flatnonzero(np.concatenate([[True], np.diff(block) != 0]))
    ends = np.concatenate([starts[1:], [t.size]])
    for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
        b0 = float(block[a]) * refresh_ms
        lo, hi = np.searchsorted(st, b0 - window_ms, side="left"), np.searchsorted(st, b0, side="left")
        if hi - lo < min_samples:
            continue
        ref = np.sort(sv[lo:hi])
        seg = values[a:b]
        fin = np.isfinite(seg)
        res = np.full(seg.shape, np.nan)
        res[fin] = midrank(ref, seg[fin])
        out[a:b] = res
    return out


def candles_15m(t: F, high: F, low: F, close: F, quote: F) -> tuple[F, F, F, F, F]:
    """15m candles aligned to UTC quarter-hours: (start, high, low, close, quote)."""
    k = np.floor_divide(t, 15 * MIN_MS).astype(np.int64)
    starts = np.flatnonzero(np.concatenate([[True], np.diff(k) != 0]))
    ends = np.concatenate([starts[1:], [t.size]]) - 1
    return (
        k[starts].astype(np.float64) * 15 * MIN_MS,
        np.maximum.reduceat(high, starts),
        np.minimum.reduceat(low, starts),
        close[ends],
        np.add.reduceat(quote, starts),
    )


def natr_by_minute(t: F, high: F, low: F, close: F) -> F:
    cs, ch, cl, cc, _ = candles_15m(t, high, low, close, np.zeros_like(close))
    prev_close = np.concatenate([[np.nan], cc[:-1]])
    tr = np.fmax(ch - cl, np.fmax(np.abs(ch - prev_close), np.abs(cl - prev_close)))
    ctr = np.concatenate([[0.0], np.cumsum(np.nan_to_num(tr))])
    natr = np.full(cs.size, np.nan)
    j = np.arange(15, cs.size)  # candle j uses candles j-14 .. j-1 (completed before it)
    natr[j] = (ctr[j] - ctr[j - 14]) / 14 / cc[j - 1]
    ci = np.floor_divide(t - cs[0], 15 * MIN_MS).astype(np.int64)
    return np.asarray(natr[ci], dtype=np.float64)


def volume_z(t: F, quote: F, cal: SessionCalendar) -> tuple[F, F]:
    """(naive z vs the previous 96 candles, session z vs the same local slot on similar days)."""
    rolling = np.convolve(quote, np.ones(15), mode="full")[: quote.size]
    cs, _, _, _, cq = candles_15m(t, quote, quote, quote, quote)
    ci = np.floor_divide(t - cs[0], 15 * MIN_MS).astype(np.int64)

    c1 = np.concatenate([[0.0], np.cumsum(cq)])
    c2 = np.concatenate([[0.0], np.cumsum(cq * cq)])
    mean = np.full(cs.size, np.nan)
    sd = np.full(cs.size, np.nan)
    j = np.arange(96, cs.size)
    mean[j] = (c1[j] - c1[j - 96]) / 96
    sd[j] = np.sqrt(np.maximum((c2[j] - c2[j - 96]) / 96 - mean[j] ** 2, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        naive = (rolling - mean[ci]) / sd[ci]

    offset = cal.utc_offset_ms(cs)
    slot = np.floor_divide(np.mod(cs + offset, DAY_MS), 15 * MIN_MS).astype(np.int64)
    local_day = np.floor_divide(cs + offset, DAY_MS).astype(np.int64)
    key = cal.classify(cs).astype(np.int64) * 96 + slot  # e.g. Monday 02:00 overnight != Saturday 02:00
    order = np.lexsort((local_day, key))
    s_mean = np.full(cs.size, np.nan)
    s_sd = np.full(cs.size, np.nan)
    sk, sv = key[order], cq[order]
    starts = np.flatnonzero(np.concatenate([[True], np.diff(sk) != 0]))
    ends = np.concatenate([starts[1:], [sk.size]])
    for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
        v = sv[a:b]
        p1 = np.concatenate([[0.0], np.cumsum(v)])
        p2 = np.concatenate([[0.0], np.cumsum(v * v)])
        i = np.arange(v.size)
        n = np.minimum(i, Z_SLOT_DAYS)
        ok = n >= 5
        lo = i - n
        m = np.where(ok, (p1[i] - p1[lo]) / np.maximum(n, 1), np.nan)
        var = np.where(ok, (p2[i] - p2[lo]) / np.maximum(n, 1) - m**2, np.nan)
        s_mean[order[a:b]] = m
        s_sd[order[a:b]] = np.sqrt(np.maximum(var, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        session_z = (rolling - s_mean[ci]) / s_sd[ci]
    return naive, session_z


@dataclass
class MinuteFeatures:
    t: F  # minute open, ms
    close: F
    high: F
    low: F
    quote: F
    taker_buy: F
    index_close: F
    missing: B  # no perp bar this minute: fail closed downstream
    session: npt.NDArray[np.int8]
    vpin_global: F
    vpin_session: F
    pctl_global: F
    pctl_session: F
    natr_15m: F
    vol_z: F
    vol_z_session: F
    move15: F
    bid1: F
    ask1: F
    bid02: F
    ask02: F
    ob_pctl: F

    @property
    def obt(self) -> F:
        with np.errstate(invalid="ignore", divide="ignore"):
            return self.ask1 / self.bid1

    @property
    def log_imbalance(self) -> F:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.log(self.ask1 / self.bid1)

    def at(self, times_ms: F) -> npt.NDArray[np.int64]:
        """Grid index of the minute bar containing each time (-1 outside the grid)."""
        i = np.floor_divide(times_ms - self.t[0], MIN_MS).astype(np.int64)
        i[(i < 0) | (i >= self.t.size)] = -1
        return np.asarray(i, dtype=np.int64)


def _on_grid(t_grid: F, t: F, values: F) -> tuple[F, B]:
    pos = np.floor_divide(t - t_grid[0], MIN_MS).astype(np.int64)
    ok = (pos >= 0) & (pos < t_grid.size)
    out = np.full(t_grid.shape, np.nan)
    out[pos[ok]] = values[ok]
    have = np.zeros(t_grid.shape, dtype=bool)
    have[pos[ok]] = True
    return out, have


def build_minute_features(store: TradfiStore, cal: SessionCalendar, sym: str) -> MinuteFeatures:
    k = store.bars("klines", sym)
    grid = np.arange(k.t[0], k.t[-1] + MIN_MS, MIN_MS, dtype=np.float64)
    close, have = _on_grid(grid, k.t, k.close)
    fill = ffill_index(have)
    close = close[fill]
    high = np.where(have, _on_grid(grid, k.t, k.high)[0], close)
    low = np.where(have, _on_grid(grid, k.t, k.low)[0], close)
    quote = np.nan_to_num(_on_grid(grid, k.t, k.quote)[0])
    taker_buy = np.nan_to_num(_on_grid(grid, k.t, k.taker_buy_quote)[0])
    idx = store.bars("index", sym)
    index_close, have_idx = _on_grid(grid, idx.t, idx.close)
    index_close = index_close[ffill_index(have_idx)]

    session = cal.classify(grid)
    v_global, v_session = bucket_sizes(grid, quote, session)
    vpin_g = vpin_series(quote, taker_buy, v_global)
    vpin_s = vpin_series(quote, taker_buy, v_session)
    pctl = {
        name: trailing_percentile(
            grid, v, TRAILING_DAYS * DAY_MS, DAY_MS, PCTL_MIN_SAMPLES, PCTL_SAMPLE_EVERY
        )
        for name, v in (("g", vpin_g), ("s", vpin_s))
    }
    naive_z, session_z = volume_z(grid, quote, cal)
    move15 = np.full(grid.shape, np.nan)
    move15[15:] = close[15:] / close[:-15] - 1

    first_day = cal.local_dates(grid[:1])[0]
    last_day = cal.local_dates(grid[-1:])[0]
    depth = store.depth(sym, first_day - timedelta(days=1), last_day + timedelta(days=1))
    levels = {}
    ob_pctl = np.full(grid.shape, np.nan)
    if depth.t.size:
        j = np.searchsorted(depth.t, grid + MIN_MS, side="right") - 1
        fresh = (j >= 0) & (grid + MIN_MS - depth.t[np.maximum(j, 0)] <= DEPTH_MAX_AGE_MS)
        for name, pct in (("bid1", -1.0), ("ask1", 1.0), ("bid02", -0.2), ("ask02", 0.2)):
            series = depth.level(pct)
            levels[name] = np.where(fresh, series[np.maximum(j, 0)], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            snap_imb = np.log(depth.level(1.0) / depth.level(-1.0))
        snap_imb[~np.isfinite(snap_imb)] = np.nan
        snap_pctl = trailing_percentile(depth.t, snap_imb, OB_PCTL_WINDOW_MS, HOUR_MS, OB_PCTL_MIN_SAMPLES)
        ob_pctl = np.where(fresh, snap_pctl[np.maximum(j, 0)], np.nan)
    nan = np.full(grid.shape, np.nan)
    return MinuteFeatures(
        t=grid,
        close=close,
        high=high,
        low=low,
        quote=quote,
        taker_buy=taker_buy,
        index_close=index_close,
        missing=~have,
        session=session,
        vpin_global=vpin_g,
        vpin_session=vpin_s,
        pctl_global=pctl["g"],
        pctl_session=pctl["s"],
        natr_15m=natr_by_minute(grid, high, low, close),
        vol_z=naive_z,
        vol_z_session=session_z,
        move15=move15,
        bid1=levels.get("bid1", nan),
        ask1=levels.get("ask1", nan),
        bid02=levels.get("bid02", nan),
        ask02=levels.get("ask02", nan),
        ob_pctl=ob_pctl,
    )


def features_path(store: TradfiStore, sym: str) -> Path:
    return store.table_path("features", sym)


def save_features(store: TradfiStore, sym: str, f: MinuteFeatures) -> Path:
    path = features_path(store, sym)
    write_table(path, {fld.name: getattr(f, fld.name) for fld in fields(f)})
    return path


def load_features(
    store: TradfiStore, cal: SessionCalendar, sym: str, rebuild: bool = False
) -> MinuteFeatures:
    """Cached Parquet artifact of `build_minute_features` (rebuilt when the klines are newer)."""
    path = features_path(store, sym)
    klines = store.table_path("klines", sym)
    if not rebuild and path.exists() and path.stat().st_mtime >= klines.stat().st_mtime:
        names = tuple(fld.name for fld in fields(MinuteFeatures))
        c = read_columns(path, names)
        return MinuteFeatures(**{n: c[n] for n in names})
    f = build_minute_features(store, cal, sym)
    save_features(store, sym, f)
    return f
