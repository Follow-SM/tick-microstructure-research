"""Collector plumbing: storage, trade sequencing, message parsing. Offline."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

from tick_microstructure.collector.binance import TradeSequencer, trade_record
from tick_microstructure.collector.followsm import flatten_confluence, flatten_toxicity
from tick_microstructure.collector.polymarket import parse_messages
from tick_microstructure.collector.storage import HOUR_MS, JsonlWriter, Store, iter_records

H0 = 1_790_690_400_000.0  # an hour boundary


def test_writer_rotates_hourly_and_gzips_closed_hours(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "s")
    writer.write({"rt": H0 + 1, "v": 1})
    writer.write({"rt": H0 + HOUR_MS + 1, "v": 2})
    writer.close()
    deadline = time.time() + 5
    while not list(tmp_path.glob("s/*/*.jsonl.gz")) and time.time() < deadline:
        time.sleep(0.05)
    assert len(list(tmp_path.glob("s/*/*.jsonl.gz"))) == 1
    assert [r["v"] for r in iter_records(tmp_path, "s")] == [1, 2]


def test_reader_filters_by_time_and_skips_torn_line(tmp_path: Path) -> None:
    writer = JsonlWriter(tmp_path, "s")
    for i in range(5):
        writer.write({"rt": H0 + i * 1000, "v": i})
    writer.close()
    path = next(tmp_path.glob("s/*/*.jsonl"))
    with path.open("a") as fh:
        fh.write('{"rt": 1')  # crash mid-write
    assert [r["v"] for r in iter_records(tmp_path, "s", H0 + 1000, H0 + 4000)] == [1, 2, 3]


def test_store_compresses_leftover_hours_on_start(tmp_path: Path) -> None:
    old = tmp_path / "s" / "20260101" / "00.jsonl"
    old.parent.mkdir(parents=True)
    old.write_text(json.dumps({"rt": 1.0}) + "\n")
    Store(tmp_path).close()
    gz = old.with_suffix(".jsonl.gz")
    assert gz.exists() and not old.exists()
    with gzip.open(gz, "rt") as fh:
        assert json.loads(fh.readline()) == {"rt": 1.0}


def test_trade_sequencer_detects_holes_and_duplicates() -> None:
    seq = TradeSequencer()
    assert seq.check("BTCUSDT", 10) == (True, None)
    seq.advance("BTCUSDT", 10)
    assert seq.check("BTCUSDT", 10) == (False, None)
    assert seq.check("BTCUSDT", 14) == (True, (11, 13))
    seq.advance("BTCUSDT", 14)
    assert seq.check("ETHUSDT", 3) == (True, None)


def test_trade_record_side_convention() -> None:
    raw = {"a": 5, "T": 1000, "p": "100.5", "q": "2", "m": True}
    record = trade_record(raw, "BTCUSDT", 1001.0, backfilled=True)
    assert record == {
        "rt": 1001.0,
        "T": 1000,
        "sym": "BTCUSDT",
        "id": 5,
        "p": 100.5,
        "q": 2.0,
        "m": True,
        "bf": 1,
    }


def test_flatten_followsm_frames() -> None:
    band = {"bid_notional": 1e6, "ask_notional": 2.5e6, "imbalance_ratio": 0.29}
    tox = {
        "symbol": "BTCUSDT",
        "timestamp": 1790690400.5,
        "price": 84000.0,
        "vpin": 0.4,
        "vpin_percentile": None,
        "ob_toxicity_1pct": 2.5,
        "ob_imbalance_l1": 0.4,
        "depth_bands": {"1.0%": band},
        "volume_z_score": 1.2,
        "natr_15m": 0.002,
        "taker_buy_ratio": 0.45,
        "is_toxic_alert": True,
    }
    flat = flatten_toxicity(tox, 1790690400600.0)
    assert flat["ts"] == 1790690400500.0 and flat["vp"] is None
    assert (flat["b1"], flat["a1"]) == (1e6, 2.5e6) and "b05" not in flat
    conf = {
        "symbol": "BTCUSDT",
        "timestamp_ms": 1790690400500,
        "binance_microstructure": {**tox, "price_delta_15m_pct": -0.003},
        "polymarket_confluence": {
            "macro_event_risk_score": 0.3,
            "active_events": [
                {
                    "market_slug": "btc-updown-15m-1790690400",
                    "question": "q",
                    "condition_id": "c",
                    "yes_token_id": "tok",
                    "direction": "bullish_if_yes",
                    "direction_confidence": 1.0,
                    "implied_probability": 0.4,
                    "prob_delta_15m": -0.1,
                    "clob_order_flow_imbalance": 0.3,
                    "smart_money_whale_sweeps_1h_usdt": 120000.0,
                }
            ],
        },
        "composite_signals": {
            "is_toxic_alert": True,
            "cross_market_divergence_flag": False,
            "recommended_action": "WIDEN_SPREAD_2X",
        },
    }
    flat_c = flatten_confluence(conf, 1790690400700.0)
    assert flat_c["dpx15"] == -0.003 and flat_c["act"] == "WIDEN_SPREAD_2X"
    assert flat_c["ev"][0]["whale"] == 120000.0 and flat_c["ev"][0]["dp15"] == -0.1


def test_parse_polymarket_messages() -> None:
    frame = json.dumps(
        [
            {
                "event_type": "best_bid_ask",
                "asset_id": "t",
                "best_bid": "0.41",
                "best_ask": "0.43",
                "timestamp": "5",
            },
            {"event_type": "price_change", "price_changes": [], "timestamp": "6"},
            {
                "event_type": "last_trade_price",
                "asset_id": "t",
                "price": "0.43",
                "size": "50",
                "side": "BUY",
                "timestamp": "7",
            },
            {
                "event_type": "book",
                "asset_id": "t",
                "timestamp": "8",
                "bids": [{"price": "0.40", "size": "1"}, {"price": "0.41", "size": "1"}],
                "asks": [{"price": "0.44", "size": "1"}],
            },
        ]
    )
    out = parse_messages(frame, 10.0)
    assert [s for s, _ in out] == ["polymarket_bbo", "polymarket_trades", "polymarket_bbo"]
    assert out[0][1] == {"rt": 10.0, "ts": 5, "tok": "t", "b": 0.41, "a": 0.43}
    assert out[2][1]["b"] == 0.41 and out[2][1]["a"] == 0.44
    assert parse_messages("PONG", 1.0) == []
