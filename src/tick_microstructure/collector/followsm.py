"""FollowSM Enterprise streams -> flattened per-symbol records.

    /ws/v1/toxicity    full fresh cache about once a second      -> stream "followsm_toxicity"
    /ws/v1/confluence  changed symbols about once a second       -> stream "followsm_confluence"

Every record carries `rt` (local receive time, ms) next to the server timestamp, so arrival
latency and gaps are measurable. FollowSM has no replay API: a gap cannot be back-filled, so it
is logged as an event and the analytics exclude it.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Iterable
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException

from tick_microstructure.collector.common import Backoff, FatalStreamError, Health
from tick_microstructure.collector.storage import Store, now_ms

PRICING_URL = "https://follow-sm.com/pricing"
BANDS = (("0.5%", "05"), ("1.0%", "1"), ("2.0%", "2"))
# (seconds without any frame before forcing a reconnect, inter-frame gap worth logging in ms)
CADENCE = {"toxicity": (10.0, 3_000.0), "confluence": (60.0, None)}


def flatten_toxicity(item: dict[str, Any], rt: float) -> dict[str, Any]:
    record: dict[str, Any] = {
        "rt": rt,
        "ts": round(float(item["timestamp"]) * 1000, 1),
        "sym": item["symbol"],
        "px": item["price"],
        "vpin": item["vpin"],
        "vp": item.get("vpin_percentile"),
        "obt": item["ob_toxicity_1pct"],
        "l1": item["ob_imbalance_l1"],
        "vz": item["volume_z_score"],
        "natr": item["natr_15m"],
        "tbr": item["taker_buy_ratio"],
        "tox": item["is_toxic_alert"],
    }
    # Fields added with the local-order-book backend; absent from older frames.
    for key, tag in (
        ("seq", "seq"),
        ("depth_coverage_pct", "cov"),
        ("ob_log_imbalance_1pct", "oli"),
        ("ob_imbalance_percentile", "obp"),
    ):
        if key in item:
            record[tag] = item[key]
    bands = item.get("depth_bands") or {}
    for key, tag in BANDS:
        band = bands.get(key)
        if band:
            record[f"b{tag}"] = band["bid_notional"]
            record[f"a{tag}"] = band["ask_notional"]
    return record


def flatten_confluence(item: dict[str, Any], rt: float) -> dict[str, Any]:
    micro = item["binance_microstructure"]
    composite = item["composite_signals"]
    events = item["polymarket_confluence"]["active_events"]
    return {
        "rt": rt,
        "ts": item["timestamp_ms"],
        "sym": item["symbol"],
        "px": micro["price"],
        "vp": micro.get("vpin_percentile"),
        "obt": micro["ob_toxicity_1pct"],
        "dpx15": micro["price_delta_15m_pct"],
        "risk": item["polymarket_confluence"]["macro_event_risk_score"],
        "tox": composite["is_toxic_alert"],
        "div": composite["cross_market_divergence_flag"],
        "act": composite["recommended_action"],
        "ev": [
            {
                "slug": e["market_slug"],
                "tok": e["yes_token_id"],
                "dir": e["direction"],
                "conf": e["direction_confidence"],
                "p": e["implied_probability"],
                "dp15": e["prob_delta_15m"],
                "flow": e["clob_order_flow_imbalance"],
                "whale": e["smart_money_whale_sweeps_1h_usdt"],
            }
            for e in events
        ],
    }


def check_seq(
    health: Health, stream: str, last_seq: dict[str, tuple[int, float]], record: dict[str, Any]
) -> None:
    """A per-symbol `seq` jump means frames were missed; a drop means the backend restarted."""
    sym, seq, rt = record["sym"], int(record["seq"]), record["rt"]
    previous = last_seq.get(sym)
    if previous is not None:
        prev_seq, prev_rt = previous
        if seq > prev_seq + 1:
            health.event(
                "gap",
                stream=stream,
                sym=sym,
                from_rt=prev_rt,
                to_rt=rt,
                cause="seq",
                missed=seq - prev_seq - 1,
            )
        elif seq < prev_seq:
            health.event("gap", stream=stream, sym=sym, from_rt=prev_rt, to_rt=rt, cause="backend_restart")
    last_seq[sym] = (seq, rt)


def _rejected(exc: BaseException) -> bool:
    if isinstance(exc, InvalidStatus):
        return exc.response.status_code in (401, 403)
    if isinstance(exc, ConnectionClosed) and exc.rcvd is not None:
        return exc.rcvd.code in (4001, 4003)
    return False


async def run_followsm(
    kind: str,
    base_url: str,
    api_key: str,
    symbols: Iterable[str],
    store: Store,
    health: Health,
    on_confluence: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Consume one FollowSM stream forever; raises FatalStreamError if the key is refused."""
    wanted = set(symbols)
    stale_secs, gap_ms = CADENCE[kind]
    flatten = flatten_toxicity if kind == "toxicity" else flatten_confluence
    stream = f"followsm_{kind}"
    backoff = Backoff()
    last_frame: float | None = None
    down_since: float | None = None
    last_seq: dict[str, tuple[int, float]] = {}  # symbol -> (seq, receive time)
    while True:
        try:
            async with connect(f"{base_url}/{kind}?api_key={api_key}", max_size=2**24, open_timeout=15) as ws:
                if down_since is not None:
                    health.event("gap", stream=stream, from_rt=down_since, to_rt=now_ms(), cause="reconnect")
                    down_since = None
                health.event("connect", stream=stream)
                backoff.reset()
                while True:
                    raw = await asyncio.wait_for(ws.recv(), timeout=stale_secs)
                    rt = now_ms()
                    payload = json.loads(raw)
                    if not isinstance(payload, list):
                        continue  # {"type": "ping"} heartbeat
                    if gap_ms is not None and last_frame is not None and rt - last_frame > gap_ms:
                        health.event("gap", stream=stream, from_rt=last_frame, to_rt=rt, cause="cadence")
                    last_frame = rt
                    for item in payload:
                        if item.get("symbol") in wanted:
                            record = flatten(item, rt)
                            store.write(stream, record)
                            if "seq" in record:
                                check_seq(health, stream, last_seq, record)
                            if on_confluence is not None:
                                on_confluence(record)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if _rejected(exc):
                raise FatalStreamError(
                    f"FollowSM refused /ws/v1/{kind}: the key is missing, rotated or not on ENTERPRISE. "
                    f"Set FOLLOWSM_API_KEY in .env (see {PRICING_URL})."
                ) from exc
            if not isinstance(exc, (TimeoutError, OSError, WebSocketException, json.JSONDecodeError)):
                raise
            down_since = down_since or last_frame or now_ms()
            health.event("disconnect", stream=stream, error=f"{type(exc).__name__}: {exc}"[:200])
            await asyncio.sleep(backoff.next())
