"""Polymarket CLOB ground truth: best bid/ask changes and trades for crypto markets.

Markets tracked:
  * rolling "<coin>-updown-{5m,15m}-<slot>" markets for each tracked symbol (current + next slot),
  * any market the FollowSM confluence stream maps to a tracked symbol.
New markets are subscribed on the open connection; ended ones are pruned. The CLOB market
channel has no sequence numbers, so disconnect windows are logged as gap events.

Streams: "polymarket_markets" (metadata), "polymarket_bbo" (server-stamped top of book changes),
"polymarket_trades" (last_trade_price events).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import WebSocketException

from tick_microstructure.collector.common import Backoff, Health
from tick_microstructure.collector.storage import Store, now_ms
from tick_microstructure.config import POLYMARKET_COINS, UPDOWN_WINDOWS

PING_SECS = 9.0  # the server closes the socket after ~10 s without a text "PING"
STALE_SECS = 45.0
DISCOVERY_SECS = 30.0
PRUNE_AFTER_SECS = 120.0


@dataclass
class Market:
    slug: str
    sym: str
    kind: str  # "updown_5m" | "updown_15m" | "followsm"
    tokens: tuple[str, ...]
    start_ts: float | None
    end_ts: float | None


def _iso_ts(value: str | None) -> float | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def parse_messages(raw: str, rt: float) -> list[tuple[str, dict[str, Any]]]:
    """Market-channel frame -> [(stream, record)] for BBO changes and trades."""
    if not raw.startswith(("[", "{")):
        return []  # "PONG"
    payload = json.loads(raw)
    out: list[tuple[str, dict[str, Any]]] = []
    for item in payload if isinstance(payload, list) else [payload]:
        kind = item.get("event_type")
        if kind == "best_bid_ask":
            out.append(
                (
                    "polymarket_bbo",
                    {
                        "rt": rt,
                        "ts": int(item["timestamp"]),
                        "tok": item["asset_id"],
                        "b": float(item["best_bid"] or 0),
                        "a": float(item["best_ask"] or 0),
                    },
                )
            )
        elif kind == "book":
            bids = [float(level["price"]) for level in item.get("bids", [])]
            asks = [float(level["price"]) for level in item.get("asks", [])]
            out.append(
                (
                    "polymarket_bbo",
                    {
                        "rt": rt,
                        "ts": int(item["timestamp"]),
                        "tok": item["asset_id"],
                        "b": max(bids, default=0.0),
                        "a": min(asks, default=0.0),
                        "snap": 1,
                    },
                )
            )
        elif kind == "last_trade_price":
            out.append(
                (
                    "polymarket_trades",
                    {
                        "rt": rt,
                        "ts": int(item["timestamp"]),
                        "tok": item["asset_id"],
                        "p": float(item["price"]),
                        "s": float(item["size"]),
                        "side": item.get("side"),
                    },
                )
            )
    return out


class PolymarketCollector:
    def __init__(self, ws_url: str, gamma_url: str, symbols: list[str], store: Store, health: Health) -> None:
        self.ws_url, self.gamma_url = ws_url, gamma_url
        self.symbols = [s for s in symbols if s in POLYMARKET_COINS]
        self.store, self.health = store, health
        self.markets: dict[str, Market] = {}
        self._pending: set[str] = set()
        self._missing_slugs: dict[str, float] = {}
        self._ws: ClientConnection | None = None

    # ── market registry ──

    def add_market(self, market: Market) -> None:
        if market.slug in self.markets:
            return
        self.markets[market.slug] = market
        self._pending.update(market.tokens)
        self.store.write(
            "polymarket_markets",
            {
                "rt": now_ms(),
                "slug": market.slug,
                "sym": market.sym,
                "kind": market.kind,
                "tokens": list(market.tokens),
                "start": market.start_ts,
                "end": market.end_ts,
            },
        )

    def on_confluence(self, record: dict[str, Any]) -> None:
        for event in record.get("ev", []):
            if event["slug"] not in self.markets:
                self.add_market(Market(event["slug"], record["sym"], "followsm", (event["tok"],), None, None))

    def active_tokens(self) -> list[str]:
        return sorted({t for m in self.markets.values() for t in m.tokens})

    def prune(self, now: float) -> None:
        for slug in [s for s, m in self.markets.items() if m.end_ts and now - m.end_ts > PRUNE_AFTER_SECS]:
            del self.markets[slug]

    async def discover(self, http: httpx.AsyncClient) -> None:
        now = time.time()
        for sym in self.symbols:
            coin = POLYMARKET_COINS[sym]
            for label, secs in UPDOWN_WINDOWS.items():
                slot = int(now // secs * secs)
                for start in (slot, slot + secs):
                    slug = f"{coin}-updown-{label}-{start}"
                    if slug in self.markets or now - self._missing_slugs.get(slug, 0.0) < 60:
                        continue
                    resp = await http.get(f"{self.gamma_url}/markets", params={"slug": slug})
                    resp.raise_for_status()
                    rows = resp.json()
                    if not rows:
                        self._missing_slugs[slug] = now
                        continue
                    tokens = tuple(json.loads(rows[0]["clobTokenIds"]))
                    self.add_market(
                        Market(
                            slug,
                            sym,
                            f"updown_{label}",
                            tokens,
                            float(start),
                            _iso_ts(rows[0].get("endDate")),
                        )
                    )
        self.prune(now)

    # ── tasks ──

    async def discovery_loop(self) -> None:
        async with httpx.AsyncClient(timeout=15.0) as http:
            while True:
                try:
                    await self.discover(http)
                except (httpx.HTTPError, ValueError, KeyError) as exc:
                    self.health.event("error", where="discovery", error=f"{type(exc).__name__}: {exc}"[:200])
                await self._flush_subscriptions()
                await asyncio.sleep(DISCOVERY_SECS)

    async def _flush_subscriptions(self) -> None:
        if self._ws is None or not self._pending:
            return
        tokens, self._pending = sorted(self._pending), set()
        with contextlib.suppress(WebSocketException, OSError):
            await self._ws.send(
                json.dumps({"assets_ids": tokens, "operation": "subscribe", "custom_feature_enabled": True})
            )
            self.health.event("subscribe", tokens=len(tokens))

    async def run(self) -> None:
        backoff = Backoff()
        down_since: float | None = None
        while True:
            tokens = self.active_tokens()
            if not tokens:
                await asyncio.sleep(2)
                continue
            try:
                async with connect(self.ws_url, max_size=2**24, open_timeout=15, ping_interval=None) as ws:
                    await ws.send(
                        json.dumps({"assets_ids": tokens, "type": "market", "custom_feature_enabled": True})
                    )
                    self._ws, self._pending = ws, set(self.active_tokens()) - set(tokens)
                    if down_since is not None:
                        self.health.event("gap", stream="polymarket", from_rt=down_since, to_rt=now_ms())
                        down_since = None
                    self.health.event("connect", tokens=len(tokens))
                    backoff.reset()
                    last_ping = last_msg = time.monotonic()
                    while True:
                        if time.monotonic() - last_ping >= PING_SECS:
                            await ws.send("PING")
                            last_ping = time.monotonic()
                        await self._flush_subscriptions()
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=PING_SECS / 3)
                        except TimeoutError:
                            if time.monotonic() - last_msg > STALE_SECS:
                                raise
                            continue
                        last_msg = time.monotonic()
                        for stream, record in parse_messages(str(raw), now_ms()):
                            self.store.write(stream, record)
            except asyncio.CancelledError:
                raise
            except (
                TimeoutError,
                OSError,
                WebSocketException,
                json.JSONDecodeError,
                KeyError,
                ValueError,
            ) as exc:
                self._ws = None
                down_since = down_since or now_ms()
                self.health.event("disconnect", error=f"{type(exc).__name__}: {exc}"[:200])
                await asyncio.sleep(backoff.next())
