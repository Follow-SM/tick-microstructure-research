"""Parquet layout of the TradFi capture and its NumPy readers.

    <root>/trades/<SYM>/<YYYY-MM-DD>.parquet   t (ms), price, qty, side (+1 taker buy, -1 taker sell)
    <root>/depth/<SYM>/<YYYY-MM-DD>.parquet    t (ms), pct (-5..+5), notional (cumulative USDT)
    <root>/klines/<SYM>.parquet                1m perp bars: t, open, high, low, close, quote, taker_buy_quote
    <root>/index/<SYM>.parquet                 1m index-price bars: t, close
    <root>/funding/<SYM>.parquet               t, rate
    <root>/yahoo_daily/<TICKER>.parquet        day (YYYY-MM-DD), open, high, low, close, volume
    <root>/yahoo_1m/<TICKER>.parquet           t, open, high, low, close, volume (pre/post included)

Daily partitions keep multi-month tick loads chunked: readers take a day range and concatenate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]


def pyarrow_modules() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('The TradFi pipeline needs pyarrow: pip install -e ".[parquet]"') from exc
    return pa, pq


def write_table(path: Path, columns: dict[str, npt.NDArray[Any] | list[str]]) -> None:
    pa, pq = pyarrow_modules()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    pq.write_table(pa.table(columns), tmp, compression="zstd")
    tmp.replace(path)


def read_columns(path: Path, names: tuple[str, ...]) -> dict[str, npt.NDArray[Any]]:
    _, pq = pyarrow_modules()
    table = pq.read_table(path, columns=list(names))
    return {n: table.column(n).to_numpy() for n in names}


def day_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=k) for k in range((end - start).days + 1)]


@dataclass
class TickTrades:
    t: F
    price: F
    qty: F
    side: F

    @property
    def notional(self) -> F:
        return self.price * self.qty


@dataclass
class DepthSnapshots:
    """Cumulative notional within +-pct of mid, one row per (snapshot, pct)."""

    t: F  # unique snapshot times, ms
    pcts: F  # sorted pct levels, e.g. [-5, ..., -0.2, 0.2, ..., 5]
    notional: F  # shape (len(t), len(pcts)); NaN where a level is missing

    def level(self, pct: float) -> F:
        k = int(np.argmin(np.abs(self.pcts - pct)))
        if abs(self.pcts[k] - pct) > 1e-9:
            raise KeyError(pct)
        return self.notional[:, k]


@dataclass
class Bars:
    t: F  # bar open, ms
    open: F
    high: F
    low: F
    close: F
    quote: F
    taker_buy_quote: F


@dataclass
class DailyBars:
    days: list[date]
    open: F
    high: F
    low: F
    close: F


class TradfiStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def trades_path(self, sym: str, day: date) -> Path:
        return self.root / "trades" / sym / f"{day.isoformat()}.parquet"

    def depth_path(self, sym: str, day: date) -> Path:
        return self.root / "depth" / sym / f"{day.isoformat()}.parquet"

    def table_path(self, kind: str, name: str) -> Path:
        return self.root / kind / f"{name}.parquet"

    def days_with(self, kind: str, sym: str) -> list[date]:
        folder = self.root / kind / sym
        return sorted(date.fromisoformat(p.stem) for p in folder.glob("*.parquet")) if folder.exists() else []

    def trades(self, sym: str, start: date, end: date) -> TickTrades:
        parts = [
            read_columns(p, ("t", "price", "qty", "side"))
            for d in day_range(start, end)
            if (p := self.trades_path(sym, d)).exists()
        ]
        cols = {
            k: np.concatenate([p[k] for p in parts]).astype(np.float64) if parts else np.empty(0)
            for k in (
                "t",
                "price",
                "qty",
                "side",
            )
        }
        return TickTrades(cols["t"], cols["price"], cols["qty"], cols["side"])

    def depth(self, sym: str, start: date, end: date) -> DepthSnapshots:
        parts = [
            read_columns(p, ("t", "pct", "notional"))
            for d in day_range(start, end)
            if (p := self.depth_path(sym, d)).exists()
        ]
        if not parts:
            return DepthSnapshots(np.empty(0), np.empty(0), np.empty((0, 0)))
        t = np.concatenate([p["t"] for p in parts]).astype(np.float64)
        pct = np.round(np.concatenate([p["pct"] for p in parts]).astype(np.float64), 2)
        notional = np.concatenate([p["notional"] for p in parts]).astype(np.float64)
        times, ti = np.unique(t, return_inverse=True)
        pcts, pi = np.unique(pct, return_inverse=True)
        grid = np.full((times.size, pcts.size), np.nan)
        grid[ti, pi] = notional
        return DepthSnapshots(times, pcts, grid)

    def bars(self, kind: str, sym: str) -> Bars:
        path = self.table_path(kind, sym)
        if kind == "index":
            c = read_columns(path, ("t", "close"))
            nan = np.full(c["t"].size, np.nan)
            close = c["close"].astype(np.float64)
            return Bars(c["t"].astype(np.float64), close, close, close, close, nan, nan)
        c = read_columns(path, ("t", "open", "high", "low", "close", "quote", "taker_buy_quote"))
        return Bars(
            *(
                c[k].astype(np.float64)
                for k in ("t", "open", "high", "low", "close", "quote", "taker_buy_quote")
            )
        )

    def funding(self, sym: str) -> tuple[F, F]:
        c = read_columns(self.table_path("funding", sym), ("t", "rate"))
        return c["t"].astype(np.float64), c["rate"].astype(np.float64)

    def yahoo_daily(self, ticker: str) -> DailyBars:
        c = read_columns(self.table_path("yahoo_daily", ticker), ("day", "open", "high", "low", "close"))
        days = [date.fromisoformat(str(d)) for d in c["day"].tolist()]
        return DailyBars(days, *(c[k].astype(np.float64) for k in ("open", "high", "low", "close")))

    def yahoo_1m(self, ticker: str) -> Bars:
        path = self.table_path("yahoo_1m", ticker)
        if not path.exists():
            e = np.empty(0)
            return Bars(e, e, e, e, e, e, e)
        c = read_columns(path, ("t", "open", "high", "low", "close", "volume"))
        col = {k: v.astype(np.float64) for k, v in c.items()}
        nan = np.full(col["t"].size, np.nan)
        return Bars(col["t"], col["open"], col["high"], col["low"], col["close"], col["volume"], nan)
