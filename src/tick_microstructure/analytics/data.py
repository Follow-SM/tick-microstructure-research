"""Captured streams -> per-symbol / per-token numpy arrays, plus the gap mask.

Clocks: Binance trade `T` and Polymarket `ts` are exchange clocks; `rt` everywhere is the local
receive clock. Signals are evaluated at `rt` (the moment they were knowable here), outcomes on
the Binance book (local clock), cross-venue lead-lag on exchange clocks.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

from tick_microstructure.collector.storage import iter_records

F = npt.NDArray[np.float64]


def _arr(values: list[float]) -> F:
    return np.asarray(values, dtype=np.float64)


class _Columns:
    """Row-wise float accumulator flushed to numpy chunks, so multi-week loads stay compact."""

    CHUNK = 100_000

    def __init__(self, width: int) -> None:
        self._rows: list[tuple[float, ...]] = []
        self._chunks: list[F] = []
        self.width = width

    def add(self, *values: float) -> None:
        self._rows.append(values)
        if len(self._rows) >= self.CHUNK:
            self._flush()

    def _flush(self) -> None:
        if self._rows:
            self._chunks.append(np.asarray(self._rows, dtype=np.float64).reshape(-1, self.width))
            self._rows = []

    def arrays(self) -> list[F]:
        self._flush()
        block = np.concatenate(self._chunks) if self._chunks else np.empty((0, self.width))
        return [np.ascontiguousarray(block[:, k]) for k in range(self.width)]


def _sorted_by(t: F, *cols: F) -> tuple[F, ...]:
    order = np.argsort(t, kind="stable")
    return (t[order], *(c[order] for c in cols))


@dataclass
class Trades:
    """Binance aggregate trades. `side` is +1 for a taker buy, -1 for a taker sell."""

    t: F  # exchange time, ms
    rt: F
    price: F
    qty: F
    side: F

    @property
    def notional(self) -> F:
        return self.price * self.qty


@dataclass
class Book:
    t: F  # local receive time, ms
    bid: F
    ask: F
    bid_qty: F
    ask_qty: F

    @property
    def mid(self) -> F:
        return (self.bid + self.ask) / 2

    def index_at(self, times: F) -> npt.NDArray[np.int64]:
        """Index of the last update at or before each time (-1 if none)."""
        return np.searchsorted(self.t, times, side="right").astype(np.int64) - 1

    def mid_at(self, times: F) -> F:
        idx = self.index_at(times)
        out = np.full(times.shape, np.nan)
        ok = idx >= 0
        out[ok] = self.mid[idx[ok]]
        return out


@dataclass
class Toxicity:
    """FollowSM /ws/v1/toxicity for one symbol. `t` = local receive time (when knowable)."""

    t: F
    ts: F
    price: F
    vpin: F
    vpin_pct: F  # NaN while warming up
    obt: F  # ob_toxicity_1pct = ask/bid notional within +-1%
    l1: F
    vol_z: F
    taker_buy_ratio: F
    bid_1pct: F
    ask_1pct: F
    seq: F  # NaN on frames from the top-10-level backend (before the local order book)
    coverage: F  # depth_coverage_pct; NaN when absent or the whole book is known
    log_imbalance: F  # ob_log_imbalance_1pct
    imbalance_pct: F  # ob_imbalance_percentile; NaN while warming up or before it existed


@dataclass
class Quotes:
    """Polymarket top of book for one token. `t` = server time, ms."""

    t: F
    rt: F
    bid: F
    ask: F

    @property
    def mid(self) -> F:
        return (self.bid + self.ask) / 2


@dataclass
class PMTrades:
    """Polymarket trades on one token. `side` is the taker side: +1 BUY (lifts asks), -1 SELL."""

    t: F  # server time, ms
    rt: F
    price: F
    size: F
    side: F


@dataclass
class Confluence:
    """FollowSM /ws/v1/confluence: symbol-level frames plus one row per (frame, Polymarket event)."""

    t: F  # local receive time
    ts: F
    spot_delta_15m: F  # price_delta_15m_pct (fraction)
    divergence: F  # cross_market_divergence_flag as 0/1
    risk: F
    ev_t: F
    ev_token: npt.NDArray[np.int64]  # index into `tokens`
    ev_direction: F  # +1 bullish_if_yes, -1 bearish_if_yes, 0 neutral
    ev_prob: F
    ev_prob_delta_15m: F
    ev_flow: F
    ev_whale_1h: F
    tokens: list[str]


@dataclass
class MarketInfo:
    slug: str
    sym: str
    kind: str
    tokens: list[str]
    start: float | None  # seconds
    end: float | None


@dataclass
class GapMask:
    """Sorted, merged [from, to] intervals (ms) during which a stream is unreliable."""

    starts: list[float] = field(default_factory=list)
    ends: list[float] = field(default_factory=list)

    @classmethod
    def build(cls, intervals: list[tuple[float, float]]) -> GapMask:
        merged: list[list[float]] = []
        for a, b in sorted(intervals):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        return cls([m[0] for m in merged], [m[1] for m in merged])

    def overlaps(self, t0: float, t1: float) -> bool:
        i = bisect.bisect_right(self.starts, t1) - 1
        return i >= 0 and self.ends[i] >= t0

    def clean(self, times: F, horizon_ms: float, lookback_ms: float = 0.0) -> npt.NDArray[np.bool_]:
        return np.array([not self.overlaps(t - lookback_ms, t + horizon_ms) for t in times], dtype=bool)

    def total_ms(self) -> float:
        return float(sum(b - a for a, b in zip(self.starts, self.ends, strict=True)))


@dataclass
class Dataset:
    start_ms: float
    end_ms: float
    trades: dict[str, Trades]
    book: dict[str, Book]
    toxicity: dict[str, Toxicity]
    confluence: dict[str, Confluence]
    quotes: dict[str, Quotes]
    pm_trades: dict[str, PMTrades]
    markets: dict[str, MarketInfo]
    gaps: dict[str, GapMask]  # "followsm" | "binance" | "polymarket" | "all"
    event_counts: dict[str, int]
    # Receive time of the first frame from the local-order-book backend. Depth metrics mean
    # something different before it (top 10 levels), so analyses must not mix the two.
    book_model_switch_ms: float | None = None

    def token_market(self) -> dict[str, tuple[MarketInfo, int]]:
        """token id -> (market, outcome index) with index 0 = YES / Up."""
        return {tok: (m, i) for m in self.markets.values() for i, tok in enumerate(m.tokens)}

    @property
    def hours(self) -> float:
        return (self.end_ms - self.start_ms) / 3_600_000


def find_book_model_switch(root: Path) -> float | None:
    """Receive time of the first toxicity frame carrying `seq` (local-order-book backend)."""
    for r in iter_records(root, "followsm_toxicity"):
        if "seq" in r:
            return float(r["rt"])
    return None


def load(root: Path, start_ms: float | None = None, end_ms: float | None = None) -> Dataset:
    trade_cols: dict[str, _Columns] = defaultdict(lambda: _Columns(6))
    for r in iter_records(root, "binance_trades", start_ms, end_ms):
        trade_cols[r["sym"]].add(r["T"], r["rt"], r["p"], r["q"], -1.0 if r["m"] else 1.0, r["id"])
    trades: dict[str, Trades] = {}
    for sym, cols in trade_cols.items():
        t, rt, p, q, side, ids = cols.arrays()
        _, first = np.unique(ids, return_index=True)  # live + back-filled duplicates
        keep = np.sort(first)
        trades[sym] = Trades(*_sorted_by(t[keep], rt[keep], p[keep], q[keep], side[keep]))

    book_cols: dict[str, _Columns] = defaultdict(lambda: _Columns(5))
    for r in iter_records(root, "binance_book", start_ms, end_ms):
        book_cols[r["sym"]].add(r["rt"], r["b"], r["a"], r["B"], r["A"])
    book = {s: Book(*_sorted_by(*c.arrays())) for s, c in book_cols.items()}

    tox_keys = (
        "rt",
        "ts",
        "px",
        "vpin",
        "vp",
        "obt",
        "l1",
        "vz",
        "tbr",
        "b1",
        "a1",
        "seq",
        "cov",
        "oli",
        "obp",
    )
    tox_cols: dict[str, _Columns] = defaultdict(lambda: _Columns(len(tox_keys)))
    for r in iter_records(root, "followsm_toxicity", start_ms, end_ms):
        tox_cols[r["sym"]].add(*(np.nan if r.get(k) is None else float(r[k]) for k in tox_keys))
    toxicity = {s: Toxicity(*_sorted_by(*c.arrays())) for s, c in tox_cols.items()}
    switch = [float(t.t[np.isfinite(t.seq)][0]) for t in toxicity.values() if np.isfinite(t.seq).any()]

    confluence = _load_confluence(root, start_ms, end_ms)

    quote_cols: dict[str, _Columns] = defaultdict(lambda: _Columns(4))
    for r in iter_records(root, "polymarket_bbo", start_ms, end_ms):
        if r["b"] > 0 and r["a"] > 0 and r["a"] >= r["b"]:
            quote_cols[r["tok"]].add(r["ts"], r["rt"], r["b"], r["a"])
    quotes = {tok: Quotes(*_sorted_by(*c.arrays())) for tok, c in quote_cols.items()}

    pm_cols: dict[str, _Columns] = defaultdict(lambda: _Columns(5))
    for r in iter_records(root, "polymarket_trades", start_ms, end_ms):
        pm_cols[r["tok"]].add(r["ts"], r["rt"], r["p"], r["s"], 1.0 if r.get("side") == "BUY" else -1.0)
    pm_trades = {tok: PMTrades(*_sorted_by(*c.arrays())) for tok, c in pm_cols.items()}

    markets: dict[str, MarketInfo] = {}
    for r in iter_records(root, "polymarket_markets"):
        markets.setdefault(
            r["slug"], MarketInfo(r["slug"], r["sym"], r["kind"], r["tokens"], r["start"], r["end"])
        )

    intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    event_counts: dict[str, int] = defaultdict(int)
    first_rt, last_rt = float("inf"), 0.0
    for r in iter_records(root, "events", start_ms, end_ms):
        event_counts[f"{r['src']}:{r['kind']}"] += 1
        first_rt, last_rt = min(first_rt, r["rt"]), max(last_rt, r["rt"])
        if r["kind"] == "gap" and "from_rt" in r:
            group = r["src"].split("_")[0]
            intervals[group].append((r["from_rt"], r["to_rt"]))
            intervals["all"].append((r["from_rt"], r["to_rt"]))
        elif r["kind"] == "unrecovered":
            intervals["binance"].append((r["rt"] - 60_000, r["rt"]))
            intervals["all"].append((r["rt"] - 60_000, r["rt"]))
    gaps = {k: GapMask.build(v) for k, v in intervals.items()}
    for key in ("followsm", "binance", "polymarket", "all"):
        gaps.setdefault(key, GapMask())

    spans = [a.t[[0, -1]] for a in [*book.values()] if len(a.t)]
    spans += [a.t[[0, -1]] for a in toxicity.values() if len(a.t)]
    lo = min((float(s[0]) for s in spans), default=first_rt)
    hi = max((float(s[1]) for s in spans), default=last_rt)
    return Dataset(
        start_ms=start_ms if start_ms is not None else lo,
        end_ms=end_ms if end_ms is not None else hi,
        trades=trades,
        book=book,
        toxicity=toxicity,
        confluence=confluence,
        quotes=quotes,
        pm_trades=pm_trades,
        markets=markets,
        gaps=gaps,
        event_counts=dict(event_counts),
        book_model_switch_ms=min(switch) if switch else None,
    )


_DIRECTION = {"bullish_if_yes": 1.0, "bearish_if_yes": -1.0}


def _load_confluence(root: Path, start_ms: float | None, end_ms: float | None) -> dict[str, Confluence]:
    frames: dict[str, _Columns] = defaultdict(lambda: _Columns(5))
    events: dict[str, _Columns] = defaultdict(lambda: _Columns(7))
    token_ids: dict[str, int] = {}
    for r in iter_records(root, "followsm_confluence", start_ms, end_ms):
        sym, rt = r["sym"], r["rt"]
        frames[sym].add(rt, r["ts"], r["dpx15"], 1.0 if r["div"] else 0.0, r["risk"])
        for e in r["ev"]:
            code = token_ids.setdefault(e["tok"], len(token_ids))
            events[sym].add(rt, code, _DIRECTION.get(e["dir"], 0.0), e["p"], e["dp15"], e["flow"], e["whale"])
    tokens = list(token_ids)
    out: dict[str, Confluence] = {}
    for sym, cols in frames.items():
        t, ts, dpx, div, risk = _sorted_by(*cols.arrays())
        et, codes, direction, prob, dp15, flow, whale = _sorted_by(*events[sym].arrays())
        out[sym] = Confluence(
            t, ts, dpx, div, risk, et, codes.astype(np.int64), direction, prob, dp15, flow, whale, tokens
        )
    return out
