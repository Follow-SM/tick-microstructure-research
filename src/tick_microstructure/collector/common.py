"""Shared plumbing for the stream collectors: reconnect backoff, fatal errors, health events."""

from __future__ import annotations

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from tick_microstructure.collector.storage import Store, now_ms

log = logging.getLogger("tick_microstructure.collector")


class FatalStreamError(RuntimeError):
    """Retrying cannot help (e.g. the FollowSM key is not ENTERPRISE)."""


class Backoff:
    """Exponential backoff with full jitter, reset after a healthy connection."""

    def __init__(self, base: float = 1.0, cap: float = 60.0) -> None:
        self.base, self.cap, self.attempt = base, cap, 0

    def next(self) -> float:
        delay = min(self.cap, self.base * 2**self.attempt)
        self.attempt += 1
        return random.uniform(delay / 2, delay)

    def reset(self) -> None:
        self.attempt = 0


@dataclass
class Health:
    """Per-source counters plus an append-only `events` stream for gaps and reconnects."""

    source: str
    store: Store
    counters: Counter[str] = field(default_factory=Counter)

    def event(self, kind: str, **fields: Any) -> None:
        self.counters[kind] += 1
        self.store.write("events", {"rt": now_ms(), "src": self.source, "kind": kind, **fields})
        level = (
            logging.WARNING
            if kind in ("gap", "disconnect", "stale", "error", "unrecovered")
            else logging.INFO
        )
        log.log(level, "[%s] %s %s", self.source, kind, fields or "")
