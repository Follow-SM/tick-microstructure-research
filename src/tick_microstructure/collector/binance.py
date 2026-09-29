"""Binance spot ground truth: every aggregate trade (gap-free) plus top-of-book.

aggTrade ids are consecutive per symbol, so any hole (dropped frames, reconnects) is detected
exactly and back-filled from REST `/api/v3/aggTrades?fromId=`; back-filled rows carry `bf: 1`.
bookTicker has no sequence to repair: every best-price change is stored, size-only updates are
thinned to one per 100 ms per symbol, and reconnect windows are logged as gap events.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from typing import Any

import httpx
from websockets.asyncio.client import connect
from websockets.exceptions import WebSocketException

from tick_microstructure.collector.common import Backoff, Health
from tick_microstructure.collector.storage import Store, now_ms

SIZE_ONLY_THROTTLE_MS = 100.0
STALE_SECS = 30.0
MAX_BACKFILL_PAGES = 200  # 200k trades; beyond that the hole is logged as unrecovered


def trade_record(d: dict[str, Any], sym: str, rt: float, backfilled: bool = False) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rt": rt,
        "T": d["T"],
        "sym": sym,
        "id": d["a"],
        "p": float(d["p"]),
        "q": float(d["q"]),
        "m": d["m"],  # buyer is maker -> the taker SOLD
    }
    if backfilled:
        record["bf"] = 1
    return record


class TradeSequencer:
    """Per-symbol aggTrade id bookkeeping: which ids are new, and which holes need back-filling."""

    def __init__(self) -> None:
        self.last_id: dict[str, int] = {}

    def check(self, sym: str, trade_id: int) -> tuple[bool, tuple[int, int] | None]:
        """(is_new, missing id range [first, last] or None)."""
        last = self.last_id.get(sym)
        if last is not None and trade_id <= last:
            return False, None
        hole = (last + 1, trade_id - 1) if last is not None and trade_id > last + 1 else None
        return True, hole

    def advance(self, sym: str, trade_id: int) -> None:
        if trade_id > self.last_id.get(sym, -1):
            self.last_id[sym] = trade_id


async def backfill(
    http: httpx.AsyncClient, sym: str, first: int, last: int, store: Store, health: Health
) -> int:
    fetched, cursor = 0, first
    for _ in range(MAX_BACKFILL_PAGES):
        resp = await http.get("/api/v3/aggTrades", params={"symbol": sym, "fromId": cursor, "limit": 1000})
        resp.raise_for_status()
        rows = [r for r in resp.json() if r["a"] <= last]
        rt = now_ms()
        for row in rows:
            store.write("binance_trades", trade_record(row, sym, rt, backfilled=True))
        fetched += len(rows)
        if not rows or rows[-1]["a"] >= last:
            break
        cursor = rows[-1]["a"] + 1
    missing = last - first + 1 - fetched
    health.event("backfill", sym=sym, first=first, last=last, fetched=fetched, missing=missing)
    if missing > 0:
        health.event("unrecovered", sym=sym, first=first + fetched, last=last)
    return fetched


async def run_binance(
    ws_url: str, rest_url: str, symbols: Iterable[str], store: Store, health: Health
) -> None:
    syms = sorted(set(symbols))
    streams = "/".join(f"{s.lower()}@aggTrade/{s.lower()}@bookTicker" for s in syms)
    url = f"{ws_url}?streams={streams}"
    sequencer = TradeSequencer()
    last_book: dict[str, tuple[str, str, float]] = {}
    backoff = Backoff()
    down_since: float | None = None
    async with httpx.AsyncClient(base_url=rest_url, timeout=15.0) as http:
        while True:
            try:
                async with connect(url, max_size=2**22, open_timeout=15) as ws:
                    if down_since is not None:
                        health.event("gap", stream="binance_book", from_rt=down_since, to_rt=now_ms())
                        down_since = None
                    health.event("connect", streams=len(syms) * 2)
                    backoff.reset()
                    while True:
                        raw = await asyncio.wait_for(ws.recv(), timeout=STALE_SECS)
                        rt = now_ms()
                        msg = json.loads(raw)
                        data, stream = msg["data"], msg["stream"]
                        sym = data["s"]
                        if stream.endswith("@aggTrade"):
                            is_new, hole = sequencer.check(sym, data["a"])
                            if hole is not None:
                                health.event(
                                    "gap", stream="binance_trades", sym=sym, first=hole[0], last=hole[1]
                                )
                                await backfill(http, sym, hole[0], hole[1], store, health)
                            if is_new:
                                store.write("binance_trades", trade_record(data, sym, rt))
                                sequencer.advance(sym, data["a"])
                        else:
                            bid, ask = data["b"], data["a"]
                            prev = last_book.get(sym)
                            if (
                                prev is None
                                or prev[0] != bid
                                or prev[1] != ask
                                or rt - prev[2] >= SIZE_ONLY_THROTTLE_MS
                            ):
                                store.write(
                                    "binance_book",
                                    {
                                        "rt": rt,
                                        "sym": sym,
                                        "u": data["u"],
                                        "b": float(bid),
                                        "B": float(data["B"]),
                                        "a": float(ask),
                                        "A": float(data["A"]),
                                    },
                                )
                                last_book[sym] = (bid, ask, rt)
            except asyncio.CancelledError:
                raise
            except (TimeoutError, OSError, WebSocketException, httpx.HTTPError, json.JSONDecodeError) as exc:
                down_since = down_since or now_ms()
                health.event("disconnect", error=f"{type(exc).__name__}: {exc}"[:200])
                await asyncio.sleep(backoff.next())
