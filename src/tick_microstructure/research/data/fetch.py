"""Historical ingestion for the TradFi study: data.binance.vision archives + Yahoo bars -> Parquet.

Binance (USD-M, `TRADIFI_PERPETUAL` equity perps and the BTCUSDT control):
    aggTrades        daily zip, every taker trade          -> trades/<SYM>/<day>.parquet
    bookDepth        daily zip, +-0.2/1..5% depth, 30 s     -> depth/<SYM>/<day>.parquet
    klines 1m        monthly zips + daily for the last month -> klines/<SYM>.parquet
    indexPriceKlines same                                   -> index/<SYM>.parquet
    fundingRate      REST                                   -> funding/<SYM>.parquet
Every archive is checked against its published SHA-256 before it is parsed.

Yahoo Finance (official US prints; no key): daily bars over the whole period, and 1m bars with
pre/post-market. Yahoo only serves 1m data for the last 30 days, so each run merges what it gets
into the existing file and the 1m history grows with repeated runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx
import numpy as np

from tick_microstructure.research.data.store import TradfiStore, day_range, pyarrow_modules, write_table
from tick_microstructure.research.utils.session import NY

log = logging.getLogger(__name__)

VISION = "https://data.binance.vision/data/futures/um"
FAPI = "https://fapi.binance.com/fapi/v1"
YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart"
YAHOO_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) tick-microstructure-research"}
EQUITY_SYMBOLS = (
    "TSLAUSDT",
    "NVDAUSDT",
    "AAPLUSDT",
    "SPYUSDT",
    "QQQUSDT",
    "AMZNUSDT",
    "METAUSDT",
    "GOOGLUSDT",
    "MSFTUSDT",
    "COINUSDT",
    "PLTRUSDT",
    "HOODUSDT",
)
CONTROL_SYMBOL = "BTCUSDT"  # 24/7 crypto control: klines + depth only (its tick tape is ~100x larger)
KLINE_COLUMNS = ("t", "open", "high", "low", "close", "quote", "taker_buy_quote")


def ticker(sym: str) -> str:
    return sym.removesuffix("USDT")


@dataclass
class FetchReport:
    downloaded: dict[str, int] = field(default_factory=dict)
    missing: dict[str, list[str]] = field(default_factory=dict)

    def add(self, kind: str) -> None:
        self.downloaded[kind] = self.downloaded.get(kind, 0) + 1

    def miss(self, kind: str, what: str) -> None:
        self.missing.setdefault(kind, []).append(what)


class ChecksumError(RuntimeError):
    pass


async def _get(client: httpx.AsyncClient, url: str, **params: Any) -> httpx.Response | None:
    for attempt in range(6):
        try:
            resp = await client.get(url, params=params or None)
        except httpx.TransportError:
            await asyncio.sleep(2**attempt)
            continue
        if resp.status_code == 404:
            return None
        if resp.status_code in (418, 429) or resp.status_code >= 500:
            await asyncio.sleep(2**attempt)
            continue
        return resp.raise_for_status()
    raise RuntimeError(f"giving up on {url}")


async def _archive_csv(client: httpx.AsyncClient, path: str) -> bytes | None:
    """The CSV inside a data.binance.vision zip, SHA-256 verified; None if it doesn't exist."""
    blob = await _get(client, f"{VISION}/{path}.zip")
    if blob is None:
        return None
    check = await _get(client, f"{VISION}/{path}.zip.CHECKSUM")
    if check is not None:
        expected = check.text.split()[0]
        if hashlib.sha256(blob.content).hexdigest() != expected:
            raise ChecksumError(path)
    with zipfile.ZipFile(io.BytesIO(blob.content)) as zf:
        return zf.read(zf.namelist()[0])


def _csv_columns(raw: bytes, names: list[str]) -> dict[str, Any]:
    """Parse a Binance CSV that may or may not carry a header row, under our column names."""
    pyarrow_modules()
    import pyarrow.csv as pacsv

    has_header = not raw[:1].isdigit()
    table = pacsv.read_csv(
        io.BytesIO(raw), read_options=pacsv.ReadOptions(column_names=names, skip_rows=int(has_header))
    )
    return {n: table.column(n).to_numpy() for n in names}


AGG_NAMES = ["id", "price", "qty", "first_id", "last_id", "t", "is_buyer_maker"]
DEPTH_NAMES = ["timestamp", "pct", "depth", "notional"]
KLINE_NAMES = [
    "t",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote",
    "count",
    "taker_buy_volume",
    "taker_buy_quote",
    "ignore",
]


def parse_agg_trades(raw: bytes) -> dict[str, Any]:
    c = _csv_columns(raw, AGG_NAMES)
    maker = c["is_buyer_maker"]
    if maker.dtype != np.bool_:
        maker = np.char.lower(maker.astype(str)) == "true"
    return {
        "t": c["t"].astype(np.int64),
        "price": c["price"].astype(np.float64),
        "qty": c["qty"].astype(np.float64),
        "side": np.where(maker, -1, 1).astype(np.int8),  # buyer is maker -> the taker sold
    }


def parse_book_depth(raw: bytes) -> dict[str, Any]:
    c = _csv_columns(raw, DEPTH_NAMES)
    stamps = c["timestamp"]
    t = np.asarray(stamps, dtype="datetime64[ms]").astype(np.int64)
    return {
        "t": t,
        "pct": c["pct"].astype(np.float64),
        "notional": c["notional"].astype(np.float64),
    }


def parse_klines(raw: bytes) -> dict[str, Any]:
    c = _csv_columns(raw, KLINE_NAMES)
    out: dict[str, Any] = {"t": c["t"].astype(np.int64)}
    for k in KLINE_COLUMNS[1:]:
        out[k] = c[k].astype(np.float64)
    return out


async def _daily_partition(
    client: httpx.AsyncClient,
    store: TradfiStore,
    kind: str,
    sym: str,
    day: date,
    sem: asyncio.Semaphore,
    report: FetchReport,
) -> None:
    folder, parse = {"trades": ("aggTrades", parse_agg_trades), "depth": ("bookDepth", parse_book_depth)}[
        kind
    ]
    path = store.trades_path(sym, day) if kind == "trades" else store.depth_path(sym, day)
    if path.exists():
        return
    async with sem:
        raw = await _archive_csv(client, f"daily/{folder}/{sym}/{sym}-{folder}-{day.isoformat()}")
    if raw is None:
        report.miss(kind, f"{sym} {day}")
        return
    write_table(path, parse(raw))
    report.add(kind)


def _months(start: date, end: date) -> list[date]:
    out, m = [], date(start.year, start.month, 1)
    while m <= end:
        out.append(m)
        m = _next_month(m)
    return out


def _next_month(m: date) -> date:
    return date(m.year + (m.month == 12), m.month % 12 + 1, 1)


async def _bar_table(
    client: httpx.AsyncClient,
    store: TradfiStore,
    kind: str,
    sym: str,
    start: date,
    end: date,
    sem: asyncio.Semaphore,
    report: FetchReport,
) -> None:
    """1m klines / index klines: monthly archives, daily ones for months not yet archived."""
    folder = {"klines": "klines", "index": "indexPriceKlines"}[kind]
    out = store.table_path(kind, sym)
    if out.exists():
        c = pyarrow_modules()[1].read_table(out, columns=["t"]).column("t").to_numpy()
        if c.size and c.max() >= datetime.combine(end, datetime.min.time(), UTC).timestamp() * 1000:
            return

    async def month(m: date) -> list[dict[str, Any]]:
        async with sem:
            raw = await _archive_csv(client, f"monthly/{folder}/{sym}/1m/{sym}-1m-{m:%Y-%m}")
        if raw is not None:
            return [parse_klines(raw)]
        parts = []
        for d in day_range(max(m, start), min(end, _next_month(m) - timedelta(days=1))):
            async with sem:
                raw = await _archive_csv(client, f"daily/{folder}/{sym}/1m/{sym}-1m-{d.isoformat()}")
            if raw is None:
                report.miss(kind, f"{sym} {d}")
            else:
                parts.append(parse_klines(raw))
        return parts

    chunks = [p for parts in await asyncio.gather(*(month(m) for m in _months(start, end))) for p in parts]
    if not chunks:
        report.miss(kind, sym)
        return
    cols = {k: np.concatenate([c[k] for c in chunks]) for k in KLINE_COLUMNS}
    order = np.argsort(cols["t"], kind="stable")
    t_sorted = cols["t"][order]
    keep = np.concatenate([[True], np.diff(t_sorted) > 0])
    write_table(out, {k: v[order][keep] for k, v in cols.items()})
    report.add(kind)


async def _funding(client: httpx.AsyncClient, store: TradfiStore, sym: str, start: date, end: date) -> None:
    t = int(datetime.combine(start, datetime.min.time(), UTC).timestamp() * 1000)
    stop = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), UTC).timestamp() * 1000)
    rows: list[tuple[int, float]] = []
    while t < stop:
        resp = await _get(client, f"{FAPI}/fundingRate", symbol=sym, startTime=t, endTime=stop, limit=1000)
        batch = resp.json() if resp is not None else []
        if not batch:
            break
        rows += [(int(r["fundingTime"]), float(r["fundingRate"])) for r in batch]
        t = rows[-1][0] + 1
    write_table(
        store.table_path("funding", sym),
        {"t": np.array([r[0] for r in rows], dtype=np.int64), "rate": np.array([r[1] for r in rows])},
    )


def _yahoo_rows(payload: dict[str, Any]) -> tuple[list[int], dict[str, list[float]]]:
    result = (payload.get("chart", {}).get("result") or [None])[0]
    if not result or not result.get("timestamp"):
        return [], {}
    quote = result["indicators"]["quote"][0]
    return result["timestamp"], {k: quote[k] for k in ("open", "high", "low", "close", "volume")}


async def _yahoo_daily(client: httpx.AsyncClient, store: TradfiStore, tk: str, start: date) -> None:
    p1 = int(datetime.combine(start - timedelta(days=10), datetime.min.time(), UTC).timestamp())
    resp = await _get(
        client, f"{YAHOO}/{tk}", interval="1d", period1=p1, period2=int(datetime.now(UTC).timestamp())
    )
    stamps, q = _yahoo_rows(resp.json() if resp is not None else {})
    ok = [i for i, c in enumerate(q.get("close", [])) if c is not None and q["open"][i] is not None]
    today = datetime.now(NY).date()
    days = [datetime.fromtimestamp(stamps[i], NY).date() for i in ok]
    keep = [k for k, d in enumerate(days) if d < today]  # today's bar is still forming
    write_table(
        store.table_path("yahoo_daily", tk),
        {"day": [days[k].isoformat() for k in keep]}
        | {
            f: np.array([float(q[f][ok[k]]) for k in keep])
            for f in ("open", "high", "low", "close", "volume")
        },
    )


async def _yahoo_1m(client: httpx.AsyncClient, store: TradfiStore, tk: str) -> None:
    now = int(datetime.now(UTC).timestamp())
    rows: dict[int, list[float]] = {}
    path = store.table_path("yahoo_1m", tk)
    if path.exists():
        old = store.yahoo_1m(tk)
        for i in range(old.t.size):
            rows[int(old.t[i])] = [old.open[i], old.high[i], old.low[i], old.close[i], old.quote[i]]
    for back in range(29, 0, -7):
        p1, p2 = now - back * 86_400, min(now, now - (back - 7) * 86_400)
        resp = await _get(
            client, f"{YAHOO}/{tk}", interval="1m", period1=p1, period2=p2, includePrePost="true"
        )
        stamps, q = _yahoo_rows(resp.json() if resp is not None else {})
        for i, s in enumerate(stamps):
            vals = [q[f][i] for f in ("open", "high", "low", "close", "volume")]
            if all(v is not None for v in vals):
                rows[s * 1000] = [float(v) for v in vals]
    ts = np.array(sorted(rows), dtype=np.int64)
    block = np.array([rows[int(t)] for t in ts]) if ts.size else np.empty((0, 5))
    names = ("open", "high", "low", "close", "volume")
    columns: dict[str, Any] = {"t": ts}
    for k, n in enumerate(names):
        columns[n] = block[:, k]
    write_table(path, columns)


async def listing_dates(client: httpx.AsyncClient) -> dict[str, date]:
    resp = await _get(client, f"{FAPI}/exchangeInfo")
    assert resp is not None
    return {
        s["symbol"]: datetime.fromtimestamp(s["onboardDate"] / 1000, UTC).date()
        for s in resp.json()["symbols"]
    }


async def fetch_all(
    store: TradfiStore,
    symbols: tuple[str, ...],
    start: date,
    end: date,
    control: str | None = CONTROL_SYMBOL,
    concurrency: int = 8,
) -> FetchReport:
    report = FetchReport()
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=60, headers=YAHOO_HEADERS, follow_redirects=True) as client:
        listed = await listing_dates(client)
        log.info("Yahoo daily + 1m bars ...")
        for sym in symbols:
            await _yahoo_daily(client, store, ticker(sym), start)
            await _yahoo_1m(client, store, ticker(sym))
        tasks = []
        for sym in (*symbols, *((control,) if control else ())):
            first = max(start, listed.get(sym, start))
            days = day_range(first, end)
            kinds = ("depth",) if sym == control else ("trades", "depth")
            tasks += [_daily_partition(client, store, k, sym, d, sem, report) for k in kinds for d in days]
            tasks += [_bar_table(client, store, k, sym, first, end, sem, report) for k in ("klines", "index")]
            tasks.append(_funding(client, store, sym, first, end))
        log.info("Binance archives: %d tasks ...", len(tasks))
        for done, fut in enumerate(asyncio.as_completed(tasks), 1):
            await fut
            if done % 250 == 0:
                log.info("  %d / %d", done, len(tasks))
    return report
