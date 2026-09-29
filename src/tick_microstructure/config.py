"""Settings from the environment (a local .env is loaded if present)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Binance symbol -> Polymarket slug prefix of its rolling "Up or Down" markets.
POLYMARKET_COINS = {"BTCUSDT": "btc", "ETHUSDT": "eth", "SOLUSDT": "sol", "XRPUSDT": "xrp"}
UPDOWN_WINDOWS = {"5m": 300, "15m": 900}


@dataclass(frozen=True)
class Settings:
    followsm_api_key: str | None = None
    symbols: tuple[str, ...] = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT")
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    outreach_dir: Path = Path("outreach")
    followsm_ws_base: str = "wss://follow-sm.com/ws/v1"
    binance_ws: str = "wss://stream.binance.com:9443/stream"
    binance_rest: str = "https://api.binance.com"
    polymarket_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polymarket_clob: str = "https://clob.polymarket.com"
    gamma_api: str = "https://gamma-api.polymarket.com"


def load_settings() -> Settings:
    load_dotenv()
    symbols = tuple(s.strip().upper() for s in os.getenv("TMR_SYMBOLS", "").split(",") if s.strip())
    return Settings(
        followsm_api_key=os.getenv("FOLLOWSM_API_KEY") or None,
        symbols=symbols or Settings.symbols,
        data_dir=Path(os.getenv("TMR_DATA_DIR", "data")),
        results_dir=Path(os.getenv("TMR_RESULTS_DIR", "results")),
        outreach_dir=Path(os.getenv("TMR_OUTREACH_DIR", "outreach")),
    )
