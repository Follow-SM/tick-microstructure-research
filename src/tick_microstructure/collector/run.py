"""`tmr collect`: run every collector until interrupted (or for --minutes)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import Coroutine
from typing import Any

from tick_microstructure.collector.binance import run_binance
from tick_microstructure.collector.common import FatalStreamError, Health
from tick_microstructure.collector.followsm import run_followsm
from tick_microstructure.collector.polymarket import PolymarketCollector
from tick_microstructure.collector.storage import Store
from tick_microstructure.config import Settings

log = logging.getLogger("tick_microstructure.collector")
STATUS_SECS = 30.0


async def _status_loop(store: Store, healths: list[Health], pm: PolymarketCollector) -> None:
    started, previous = time.monotonic(), store.counts()
    while True:
        await asyncio.sleep(STATUS_SECS)
        counts = store.counts()
        rates = {k: (v - previous.get(k, 0)) / STATUS_SECS for k, v in counts.items() if k != "events"}
        previous = counts
        store.flush()
        problems = {
            f"{h.source}:{k}": n
            for h in healths
            for k, n in h.counters.items()
            if k not in ("connect", "subscribe")
        }
        log.info(
            "up %.0fm | msg/s %s | markets %d | %s",
            (time.monotonic() - started) / 60,
            " ".join(f"{k}={v:.1f}" for k, v in sorted(rates.items())),
            len(pm.markets),
            problems or "no gaps",
        )


async def collect(settings: Settings, minutes: float | None, venues_only: bool) -> None:
    if not venues_only and not settings.followsm_api_key:
        raise FatalStreamError(
            "FOLLOWSM_API_KEY is not set. Put an ENTERPRISE key in .env, or run with --venues-only "
            "to capture Binance/Polymarket ground truth alone."
        )
    store = Store(settings.data_dir)
    symbols = list(settings.symbols)
    h_bn = Health("binance", store)
    h_pm = Health("polymarket", store)
    pm = PolymarketCollector(settings.polymarket_ws, settings.gamma_api, symbols, store, h_pm)
    jobs: list[Coroutine[Any, Any, None]] = [
        run_binance(settings.binance_ws, settings.binance_rest, symbols, store, h_bn),
        pm.discovery_loop(),
        pm.run(),
    ]
    healths = [h_bn, h_pm]
    if not venues_only:
        assert settings.followsm_api_key is not None
        for kind in ("toxicity", "confluence"):
            health = Health(f"followsm_{kind}", store)
            healths.append(health)
            jobs.append(
                run_followsm(
                    kind,
                    settings.followsm_ws_base,
                    settings.followsm_api_key,
                    symbols,
                    store,
                    health,
                    on_confluence=pm.on_confluence if kind == "confluence" else None,
                )
            )
    jobs.append(_status_loop(store, healths, pm))
    log.info(
        "Collecting %s into %s (%s)",
        ",".join(symbols),
        settings.data_dir,
        "venues only" if venues_only else "FollowSM + venues",
    )
    tasks = [asyncio.create_task(job) for job in jobs]
    try:
        done, _ = await asyncio.wait(
            tasks, timeout=minutes * 60 if minutes else None, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            if task.exception() is not None:
                raise task.exception()  # type: ignore[misc]
    finally:
        for task in tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(*tasks, return_exceptions=True)
        store.close()
        log.info("Stopped. Records written: %s", store.counts())
