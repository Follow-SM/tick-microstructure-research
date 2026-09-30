"""Polymarket history for the tokenized-equity tickers: markets, taker prints and 1m probabilities.

    <root>/polymarket/markets.parquet           one row per market (daily Up/Down + earnings EPS beats)
    <root>/polymarket/trades/<slug>.parquet     t (ms), outcome (0 Up/Yes, 1 Down/No), side (+1 buy,
                                                -1 sell), price, size, wallet, tx: every taker fill
    <root>/polymarket/prices/<slug>.parquet     t (ms), p: CLOB price history of the Up/Yes token

Sources (public, no key): gamma-api (market metadata), data-api `/trades` (taker fills with wallet and
transaction hash, so multi-level sweeps can be re-assembled) and clob `/prices-history` (fidelity 1 min,
at most 7 days per request). Earnings markets are found three ways because none is complete on its
own: the per-ticker `<tk>-earnings` series, the `earnings` tag and full-text search.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from tick_microstructure.research.data.fetch import ticker
from tick_microstructure.research.data.store import TradfiStore, read_columns, write_table
from tick_microstructure.research.utils.session import SessionCalendar

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com/trades"
CLOB_HISTORY = "https://clob.polymarket.com/prices-history"
PAGE = 500
MAX_OFFSET = 10_000  # data-api pagination ceiling; a market that reaches it is flagged incomplete
HISTORY_CHUNK_S = 6 * 86_400  # prices-history returns nothing beyond ~7 days per request
EARNINGS_SLUG = re.compile(r"^([a-z]+)-quarterly-earnings-(?:non)?gaap-eps-(\d\d)-(\d\d)-(\d{4})-")
MARKET_COLUMNS = (
    "kind",
    "sym",
    "day",
    "slug",
    "condition_id",
    "yes_token",
    "payout_yes",
    "start_ms",
    "created_ms",
    "closed_ms",
    "question",
    "trades_complete",
)


@dataclass
class PmMarket:
    kind: str  # "updown" | "earnings"
    sym: str
    day: date  # Up/Down: the session whose close resolves it; earnings: the expected release date
    slug: str
    condition_id: str
    yes_token: str  # "Up" for Up/Down markets
    payout_yes: float
    start_ms: float
    created_ms: float
    closed_ms: float
    question: str
    trades_complete: bool


@dataclass
class PmTrades:
    t: npt.NDArray[np.float64]
    toward_yes: npt.NDArray[np.float64]  # +1 buys Yes/Up or sells No/Down, -1 the opposite
    notional: npt.NDArray[np.float64]  # USDC
    price_yes: npt.NDArray[np.float64]  # the fill expressed as a Yes/Up price
    order: npt.NDArray[np.int64]  # id of the (tx, wallet) order the fill belongs to
    shares: npt.NDArray[np.float64]  # outcome tokens; an Up buy and a Down sell match in one book


def _iso_ms(value: str | None) -> float:
    if not value:
        return float("nan")
    return datetime.fromisoformat(value.replace("Z", "+00:00").replace(" ", "T")).timestamp() * 1000


async def _json(client: httpx.AsyncClient, sem: asyncio.Semaphore, url: str, **params: Any) -> Any:
    async with sem:
        for attempt in range(7):
            try:
                resp = await client.get(url, params=params)
            except httpx.TransportError:
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(2**attempt)
                continue
            if resp.status_code >= 400:
                return None
            return resp.json()
    raise RuntimeError(f"giving up on {url} {params}")


def _market_row(kind: str, sym: str, day: date, event: dict[str, Any]) -> PmMarket | None:
    m = (event.get("markets") or [None])[0]
    if not m:
        return None
    tokens = json.loads(m.get("clobTokenIds") or "[]")
    prices = json.loads(m.get("outcomePrices") or "[]")
    outcomes = json.loads(m.get("outcomes") or "[]")
    if len(tokens) < 2 or len(prices) < 2 or outcomes[:1] not in (["Up"], ["Yes"]):
        return None
    return PmMarket(
        kind=kind,
        sym=sym,
        day=day,
        slug=event["slug"],
        condition_id=m["conditionId"],
        yes_token=tokens[0],
        payout_yes=float(prices[0]),
        start_ms=_iso_ms(m.get("startDate") or event.get("startDate")),
        created_ms=_iso_ms(m.get("createdAt") or event.get("createdAt")),
        closed_ms=_iso_ms(m.get("closedTime") or event.get("closedTime")),
        question=m.get("question") or "",
        trades_complete=False,
    )


async def _updown_markets(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, symbols: tuple[str, ...], days: list[date]
) -> list[PmMarket]:
    async def one(sym: str, d: date) -> PmMarket | None:
        slug = f"{ticker(sym).lower()}-up-or-down-on-{d:%B}-{d.day}-{d.year}".lower()
        events = await _json(client, sem, f"{GAMMA}/events", slug=slug)
        return _market_row("updown", sym, d, events[0]) if events else None

    found = await asyncio.gather(*(one(s, d) for s in symbols for d in days))
    return [m for m in found if m]


async def _earnings_markets(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, symbols: tuple[str, ...], start: date, end: date
) -> list[PmMarket]:
    by_tk = {ticker(s).lower(): s for s in symbols}
    events: dict[str, dict[str, Any]] = {}

    def keep(batch: list[dict[str, Any]] | None) -> None:
        for e in batch or []:
            if EARNINGS_SLUG.match(e.get("slug", "")):
                events.setdefault(e["slug"], e)

    for tk in by_tk:
        series = await _json(client, sem, f"{GAMMA}/series", slug=f"{tk}-earnings")
        keep((series or [{}])[0].get("events") if series else None)
        found = await _json(
            client, sem, f"{GAMMA}/public-search", q=f"{tk} quarterly earnings", events_status="all"
        )
        keep((found or {}).get("events"))
    for offset in range(0, 5_000, 100):
        batch = await _json(
            client,
            sem,
            f"{GAMMA}/events",
            tag_slug="earnings",
            closed="true",
            limit=100,
            offset=offset,
            end_date_min=start.isoformat(),
            end_date_max=(end + timedelta(days=1)).isoformat(),
        )
        if not batch:
            break
        keep(batch)
    out = []
    for slug in sorted(events):
        match = EARNINGS_SLUG.match(slug)
        assert match is not None
        tk, mm, dd, yyyy = match.groups()
        day = date(int(yyyy), int(mm), int(dd))
        if tk not in by_tk or not start <= day <= end:
            continue
        full = await _json(client, sem, f"{GAMMA}/events", slug=slug)  # search hits lack token ids
        row = _market_row("earnings", by_tk[tk], day, full[0]) if full else None
        if row:
            out.append(row)
    return out


async def _trades(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, m: PmMarket
) -> tuple[dict[str, Any], bool]:
    rows: list[dict[str, Any]] = []
    complete = False
    for offset in range(0, MAX_OFFSET, PAGE):
        page = await _json(
            client, sem, DATA_API, market=m.condition_id, limit=PAGE, offset=offset, takerOnly="true"
        )
        if page is None:
            break  # the API refuses deeper offsets: keep what we have, flagged incomplete
        rows += page
        if len(page) < PAGE:
            complete = True
            break
    rows.sort(key=lambda r: (r["timestamp"], r.get("transactionHash", "")))
    return (
        {
            "t": np.array([r["timestamp"] * 1000 for r in rows], dtype=np.int64),
            "outcome": np.array([int(r.get("outcomeIndex", 0)) for r in rows], dtype=np.int8),
            "side": np.array([1 if r["side"] == "BUY" else -1 for r in rows], dtype=np.int8),
            "price": np.array([float(r["price"]) for r in rows]),
            "size": np.array([float(r["size"]) for r in rows]),
            "wallet": [str(r.get("proxyWallet", "")) for r in rows],
            "tx": [str(r.get("transactionHash", "")) for r in rows],
        },
        complete,
    )


async def _prices(
    client: httpx.AsyncClient, sem: asyncio.Semaphore, token: str, start_s: int, end_s: int
) -> dict[str, Any]:
    points: dict[int, float] = {}
    for a in range(start_s, end_s, HISTORY_CHUNK_S):
        body = await _json(
            client,
            sem,
            CLOB_HISTORY,
            market=token,
            startTs=a,
            endTs=min(end_s, a + HISTORY_CHUNK_S),
            fidelity=1,
        )
        for p in (body or {}).get("history", []):
            points[int(p["t"]) * 1000] = float(p["p"])
    ts = sorted(points)
    return {"t": np.array(ts, dtype=np.int64), "p": np.array([points[t] for t in ts])}


def _price_window(cal: SessionCalendar, m: PmMarket) -> tuple[int, int]:
    if m.kind == "updown":
        return cal.close_ms(cal.previous_trading_day(m.day)) // 1000 - 3_600, cal.close_ms(m.day) // 1000
    t0 = datetime.combine(m.day - timedelta(days=7), datetime.min.time(), UTC)
    return int(t0.timestamp()), int((t0 + timedelta(days=10)).timestamp())


def markets_path(store: TradfiStore) -> Path:
    return store.root / "polymarket" / "markets.parquet"


def trades_path(store: TradfiStore, slug: str) -> Path:
    return store.root / "polymarket" / "trades" / f"{slug}.parquet"


def prices_path(store: TradfiStore, slug: str) -> Path:
    return store.root / "polymarket" / "prices" / f"{slug}.parquet"


def _write_markets(store: TradfiStore, markets: list[PmMarket]) -> None:
    cols: dict[str, Any] = {}
    for name in MARKET_COLUMNS:
        values = [getattr(m, name) for m in markets]
        if name == "day":
            cols[name] = [v.isoformat() for v in values]
        elif name in ("kind", "sym", "slug", "condition_id", "yes_token", "question"):
            cols[name] = [str(v) for v in values]
        else:
            cols[name] = np.array(values, dtype=bool if name == "trades_complete" else np.float64)
    write_table(markets_path(store), cols)


async def fetch_polymarket(
    store: TradfiStore, symbols: tuple[str, ...], start: date, end: date, concurrency: int = 6
) -> dict[str, int]:
    cal = SessionCalendar()
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        log.info("Polymarket markets ...")
        markets = await _updown_markets(client, sem, symbols, cal.trading_days(start, end))
        markets += await _earnings_markets(client, sem, symbols, start, end)
        log.info(
            "  %d Up/Down, %d earnings",
            sum(m.kind == "updown" for m in markets),
            sum(m.kind == "earnings" for m in markets),
        )
        known = {m.slug: m for m in load_markets(store)} if markets_path(store).exists() else {}

        async def one(m: PmMarket) -> None:
            tp, pp = trades_path(store, m.slug), prices_path(store, m.slug)
            if tp.exists() and m.slug in known:
                m.trades_complete = known[m.slug].trades_complete
            else:
                cols, m.trades_complete = await _trades(client, sem, m)
                write_table(tp, cols)
            if not pp.exists():
                write_table(pp, await _prices(client, sem, m.yes_token, *_price_window(cal, m)))

        for done, fut in enumerate(asyncio.as_completed([one(m) for m in markets]), 1):
            await fut
            if done % 200 == 0:
                log.info("  %d / %d markets", done, len(markets))
    _write_markets(store, markets)
    return {
        "updown": sum(m.kind == "updown" for m in markets),
        "earnings": sum(m.kind == "earnings" for m in markets),
        "incomplete_trades": sum(not m.trades_complete for m in markets),
    }


# ───────────────────────────── readers ─────────────────────────────


def load_markets(store: TradfiStore) -> list[PmMarket]:
    c = read_columns(markets_path(store), MARKET_COLUMNS)
    return [
        PmMarket(
            kind=str(c["kind"][i]),
            sym=str(c["sym"][i]),
            day=date.fromisoformat(str(c["day"][i])),
            slug=str(c["slug"][i]),
            condition_id=str(c["condition_id"][i]),
            yes_token=str(c["yes_token"][i]),
            payout_yes=float(c["payout_yes"][i]),
            start_ms=float(c["start_ms"][i]),
            created_ms=float(c["created_ms"][i]),
            closed_ms=float(c["closed_ms"][i]),
            question=str(c["question"][i]),
            trades_complete=bool(c["trades_complete"][i]),
        )
        for i in range(len(c["slug"]))
    ]


def load_trades(store: TradfiStore, slug: str) -> PmTrades:
    path = trades_path(store, slug)
    if not path.exists():
        e = np.empty(0)
        return PmTrades(e, e, e, e, np.empty(0, dtype=np.int64), e)
    c = read_columns(path, ("t", "outcome", "side", "price", "size", "wallet", "tx"))
    outcome = c["outcome"].astype(np.float64)
    side = c["side"].astype(np.float64)
    price = c["price"].astype(np.float64)
    keys = [f"{tx}:{w}" for tx, w in zip(c["tx"].tolist(), c["wallet"].tolist(), strict=True)]
    _, order = np.unique(np.array(keys, dtype=object), return_inverse=True)
    return PmTrades(
        t=c["t"].astype(np.float64),
        toward_yes=np.where(outcome == 0, side, -side),
        notional=price * c["size"].astype(np.float64),
        price_yes=np.where(outcome == 0, price, 1 - price),
        order=np.asarray(order, dtype=np.int64),
        shares=c["size"].astype(np.float64),
    )


def load_prices(store: TradfiStore, slug: str) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    path = prices_path(store, slug)
    if not path.exists():
        return np.empty(0), np.empty(0)
    c = read_columns(path, ("t", "p"))
    return c["t"].astype(np.float64), c["p"].astype(np.float64)
