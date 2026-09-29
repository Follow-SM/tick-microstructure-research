"""Hourly-rotated JSONL storage: <root>/<stream>/<YYYYMMDD>/<HH>.jsonl, gzipped once the hour closes.

The open hour stays plain JSONL (append-safe; a crash loses at most the unflushed tail).
Closed hours are compressed on a background thread so the event loop never blocks.
"""

from __future__ import annotations

import calendar
import gzip
import json
import shutil
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any

HOUR_MS = 3_600_000


def now_ms() -> float:
    return time.time_ns() / 1e6


def _hour_path(root: Path, stream: str, hour: int) -> Path:
    stamp = time.gmtime(hour * 3600)
    return root / stream / time.strftime("%Y%m%d", stamp) / f"{time.strftime('%H', stamp)}.jsonl"


def _gzip_in_place(path: Path) -> None:
    target = path.with_suffix(".jsonl.gz")
    tmp = path.with_suffix(".jsonl.gz.tmp")
    with path.open("rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
        shutil.copyfileobj(src, dst)
    tmp.replace(target)  # atomic: readers see either the plain file or a complete .gz
    path.unlink()


def compress_closed_hours(root: Path, current_hour: int | None = None) -> int:
    """Gzip every plain .jsonl except the current hour's (left over by a previous crash)."""
    current = {_hour_path(root, s.name, current_hour) for s in root.glob("*") if current_hour is not None}
    done = 0
    for path in root.glob("*/*/*.jsonl"):
        if path not in current:
            _gzip_in_place(path)
            done += 1
    return done


class JsonlWriter:
    """One stream's writer. `write` is synchronous and cheap; flushes at most every `flush_secs`."""

    def __init__(self, root: Path, stream: str, flush_secs: float = 1.0) -> None:
        self.root, self.stream, self.flush_secs = root, stream, flush_secs
        self.count = 0
        self._hour: int | None = None
        self._fh: IO[str] | None = None
        self._last_flush = time.monotonic()

    def write(self, record: dict[str, Any]) -> None:
        hour = int(record.get("rt", now_ms()) // HOUR_MS)
        if hour != self._hour:
            self._rotate(hour)
        assert self._fh is not None
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.count += 1
        if time.monotonic() - self._last_flush >= self.flush_secs:
            self._fh.flush()
            self._last_flush = time.monotonic()

    def _rotate(self, hour: int) -> None:
        previous = self._close_file()
        path = _hour_path(self.root, self.stream, hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("a", encoding="utf-8")
        self._hour = hour
        if previous is not None and previous != path:
            threading.Thread(target=_gzip_in_place, args=(previous,), daemon=False).start()

    def _close_file(self) -> Path | None:
        if self._fh is None or self._hour is None:
            return None
        self._fh.close()
        self._fh = None
        return _hour_path(self.root, self.stream, self._hour)

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> None:
        self._close_file()


class Store:
    """Lazily created writers keyed by stream name."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.writers: dict[str, JsonlWriter] = {}
        root.mkdir(parents=True, exist_ok=True)
        compress_closed_hours(root, int(now_ms() // HOUR_MS))

    def write(self, stream: str, record: dict[str, Any]) -> None:
        writer = self.writers.get(stream)
        if writer is None:
            writer = self.writers[stream] = JsonlWriter(self.root, stream)
        writer.write(record)

    def counts(self) -> dict[str, int]:
        return {name: w.count for name, w in self.writers.items()}

    def flush(self) -> None:
        for writer in self.writers.values():
            writer.flush()

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()


def iter_records(
    root: Path, stream: str, start_ms: float | None = None, end_ms: float | None = None
) -> Iterator[dict[str, Any]]:
    """Records of one stream in file (hour) order, filtered on receive time `rt`."""
    hours = sorted({(p.parent, p.name.split(".")[0]) for p in (root / stream).glob("*/*.jsonl*")})
    for folder, hour in hours:
        hour_start = calendar.timegm(time.strptime(folder.name + hour, "%Y%m%d%H"))
        if end_ms is not None and hour_start * 1000 >= end_ms:
            continue
        if start_ms is not None and (hour_start + 3600) * 1000 <= start_ms:
            continue
        yield from _read_hour(folder / f"{hour}.jsonl", start_ms, end_ms)


def _open_hour(plain: Path) -> IO[str]:
    gz = plain.with_suffix(".jsonl.gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    try:
        return plain.open(encoding="utf-8")
    except FileNotFoundError:  # compressed between the check and the open
        return gzip.open(gz, "rt", encoding="utf-8")


def _read_hour(plain: Path, start_ms: float | None, end_ms: float | None) -> Iterator[dict[str, Any]]:
    with _open_hour(plain) as fh:
        for line in fh:
            if not line.endswith("\n"):
                break  # torn final line of a crashed hour
            record: dict[str, Any] = json.loads(line)
            rt = record.get("rt", 0.0)
            if (start_ms is None or rt >= start_ms) and (end_ms is None or rt < end_ms):
                yield record
