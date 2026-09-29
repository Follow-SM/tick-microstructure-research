"""Parquet export of captured JSONL (optional `parquet` extra)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tick_microstructure.collector.storage import iter_records

STREAMS = (
    "binance_trades",
    "binance_book",
    "followsm_toxicity",
    "followsm_confluence",
    "polymarket_bbo",
    "polymarket_trades",
    "polymarket_markets",
    "events",
)


def export_parquet(data_dir: Path, out_dir: Path) -> dict[str, int]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit('Parquet export needs pyarrow: pip install -e ".[parquet]"') from exc
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, int] = {}
    for stream in STREAMS:
        rows: list[dict[str, Any]] = list(iter_records(data_dir, stream))
        if not rows:
            continue
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_dir / f"{stream}.parquet", compression="zstd")
        written[stream] = len(rows)
    return written
