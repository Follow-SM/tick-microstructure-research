"""Polymarket's own resolution of each Up/Down market, from the public gamma API, cached in the data dir.

Crypto Up/Down markets resolve on a Chainlink TWAP against the window's opening price, not on any
exchange's last trade, so fills must be settled against these outcomes rather than a Binance proxy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx

GAMMA_MARKETS = "https://gamma-api.polymarket.com/markets"
CACHE_NAME = "polymarket_resolutions.jsonl"
BATCH = 40


def _parse(market: dict[str, Any]) -> float | None:
    """1.0 if Up won, 0.0 if Down won, None while unresolved."""
    prices = json.loads(market.get("outcomePrices") or "[]")
    if not market.get("closed") or not prices or float(prices[0]) not in (0.0, 1.0):
        return None
    return float(prices[0])


def load_resolutions(root: Path, slugs: list[str], fetch: bool = True) -> dict[str, float]:
    """slug -> Up outcome (1.0 / 0.0) for each resolved market in `slugs`; new ones are cached."""
    path = root / CACHE_NAME
    known: dict[str, float] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                known[row["slug"]] = float(row["up"])
    missing = [s for s in dict.fromkeys(slugs) if s not in known]
    if fetch and missing:
        with httpx.Client(timeout=30) as client, path.open("a") as fh:
            for i in range(0, len(missing), BATCH):
                batch = [("slug", s) for s in missing[i : i + BATCH]]
                params = httpx.QueryParams([*batch, ("closed", "true"), ("limit", BATCH)])
                for market in client.get(GAMMA_MARKETS, params=params).raise_for_status().json():
                    up = _parse(market)
                    if up is not None and market.get("slug") in missing:
                        known[market["slug"]] = up
                        fh.write(json.dumps({"slug": market["slug"], "up": up}) + "\n")
    return {s: known[s] for s in slugs if s in known}
