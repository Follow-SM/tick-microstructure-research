"""US equity session calendar for tokenized-equity research (NYSE hours, America/New_York).

Every timestamp maps to one session of the *underlying* stock, while its perp keeps trading 24/7:

    PRE        04:00 to 09:30 ET on a trading day
    RTH        09:30 to the close (16:00, or 13:00 on early-close days)
    POST       the close to 20:00 ET
    OVERNIGHT  20:00 to 04:00 ET, when it leads into a trading day (Sunday 20:00 opens the week)
    CLOSED     weekends and holidays: from 20:00 before a non-trading day until 20:00 before the
               next trading day. No US venue prices the underlying then.

The calendar is anchored on America/New_York, so DST is handled by the time-zone database. The
US open is 16:30 in Eastern Europe (EET/EEST) except in the weeks when US and EU daylight-saving
dates differ, when it is 15:30. Dates outside the covered years raise instead of guessing.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, date, datetime, time, timedelta
from enum import IntEnum
from zoneinfo import ZoneInfo

import numpy as np
import numpy.typing as npt

F = npt.NDArray[np.float64]

NY = ZoneInfo("America/New_York")
HOUR_MS = 3_600_000
DAY_MS = 24 * HOUR_MS
PRE_OPEN = time(4, 0)
OPEN = time(9, 30)
CLOSE = time(16, 0)
POST_CLOSE = time(20, 0)

# NYSE full-day closures and 13:00 early closes (published calendars).
NYSE_HOLIDAYS = frozenset(
    {
        date(2025, 1, 1),
        date(2025, 1, 9),  # national day of mourning (President Carter)
        date(2025, 1, 20),
        date(2025, 2, 17),
        date(2025, 4, 18),
        date(2025, 5, 26),
        date(2025, 6, 19),
        date(2025, 7, 4),
        date(2025, 9, 1),
        date(2025, 11, 27),
        date(2025, 12, 25),
        date(2026, 1, 1),
        date(2026, 1, 19),
        date(2026, 2, 16),
        date(2026, 4, 3),
        date(2026, 5, 25),
        date(2026, 6, 19),
        date(2026, 7, 3),  # Independence Day observed
        date(2026, 9, 7),
        date(2026, 11, 26),
        date(2026, 12, 25),
    }
)
NYSE_EARLY_CLOSES = {
    date(2025, 7, 3): time(13, 0),
    date(2025, 11, 28): time(13, 0),
    date(2025, 12, 24): time(13, 0),
    date(2026, 11, 27): time(13, 0),
    date(2026, 12, 24): time(13, 0),
}
COVERED_YEARS = frozenset({2025, 2026})
EPOCH = date(1970, 1, 1)


class Session(IntEnum):
    RTH = 0
    PRE = 1
    POST = 2
    OVERNIGHT = 3
    CLOSED = 4


def _ms(t: time) -> int:
    return (t.hour * 60 + t.minute) * 60_000


def _utc_offset_s(instant: datetime) -> float:
    offset = instant.astimezone(NY).utcoffset()
    assert offset is not None
    return offset.total_seconds()


def _local_ms(d: date, t: time) -> int:
    """UTC epoch ms of local New York wall-clock time `t` on `d`."""
    return int(datetime.combine(d, t, NY).timestamp() * 1000)


class SessionCalendar:
    def __init__(
        self,
        holidays: Iterable[date] = NYSE_HOLIDAYS,
        early_closes: dict[date, time] | None = None,
        years: Iterable[int] = COVERED_YEARS,
    ) -> None:
        self.holidays = frozenset(holidays)
        self.early_closes = dict(NYSE_EARLY_CLOSES if early_closes is None else early_closes)
        self.years = frozenset(years)
        first, last = date(min(self.years), 1, 1), date(max(self.years), 12, 31)
        self._day0 = (first - EPOCH).days
        n = (last - first).days + 1
        days = [first + timedelta(days=k) for k in range(n)]
        self._trading = np.array([self._is_trading(d) for d in days], dtype=bool)
        self._close_tod = np.array([_ms(self.early_closes.get(d, CLOSE)) for d in days], dtype=np.float64)

    def _check(self, d: date) -> None:
        if d.year not in self.years:
            raise ValueError(f"{d} is outside the session calendar ({sorted(self.years)})")

    def _is_trading(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.holidays

    def is_trading_day(self, d: date) -> bool:
        self._check(d)
        return self._is_trading(d)

    def trading_days(self, start: date, end: date) -> list[date]:
        """Trading days in [start, end]."""
        days = (start + timedelta(days=k) for k in range((end - start).days + 1))
        return [d for d in days if self.is_trading_day(d)]

    def previous_trading_day(self, d: date) -> date:
        prev = d - timedelta(days=1)
        while not self.is_trading_day(prev):
            prev -= timedelta(days=1)
        return prev

    def open_ms(self, d: date) -> int:
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day")
        return _local_ms(d, OPEN)

    def close_ms(self, d: date) -> int:
        if not self.is_trading_day(d):
            raise ValueError(f"{d} is not a trading day")
        return _local_ms(d, self.early_closes.get(d, CLOSE))

    def closure_hours_before(self, d: date) -> float:
        """Hours from the previous regular close to this open: 17.5 overnight, 65.5 over a weekend."""
        return (self.open_ms(d) - self.close_ms(self.previous_trading_day(d))) / HOUR_MS

    @staticmethod
    def utc_offset_ms(t_ms: F) -> F:
        """New York UTC offset at each time (exact across DST: resolved per UTC hour)."""
        hours = np.floor_divide(t_ms, HOUR_MS)
        uniq, inverse = np.unique(hours, return_inverse=True)
        offsets = np.array(
            [_utc_offset_s(datetime.fromtimestamp(h * 3600, UTC)) * 1000 for h in uniq.tolist()],
            dtype=np.float64,
        )
        return offsets[inverse.reshape(-1)].reshape(t_ms.shape)

    def _local(self, t_ms: F) -> tuple[npt.NDArray[np.int64], F]:
        """(local epoch day index into the calendar arrays, local time of day in ms)."""
        local = t_ms + self.utc_offset_ms(t_ms)
        day = np.floor_divide(local, DAY_MS).astype(np.int64)
        idx = day - self._day0
        if idx.size and (idx.min() < 0 or idx.max() + 1 >= self._trading.size):
            raise ValueError(f"timestamps outside the session calendar ({sorted(self.years)})")
        return idx, local - day * DAY_MS

    def classify(self, t_ms: F) -> npt.NDArray[np.int8]:
        """Session code (a `Session` value) of the underlying at each UTC epoch-ms time."""
        idx, tod = self._local(t_ms)
        trading, next_trading = self._trading[idx], self._trading[idx + 1]
        close = self._close_tod[idx]
        out = np.full(t_ms.shape, Session.CLOSED, dtype=np.int8)
        day_part = trading & (tod >= _ms(PRE_OPEN)) & (tod < _ms(POST_CLOSE))
        out[day_part & (tod < _ms(OPEN))] = Session.PRE
        out[day_part & (tod >= _ms(OPEN)) & (tod < close)] = Session.RTH
        out[day_part & (tod >= close)] = Session.POST
        out[trading & (tod < _ms(PRE_OPEN))] = Session.OVERNIGHT
        out[next_trading & (tod >= _ms(POST_CLOSE))] = Session.OVERNIGHT
        return out

    def minutes_from_open(self, t_ms: F) -> F:
        """Signed minutes from that local day's 09:30 open; NaN on non-trading days."""
        idx, tod = self._local(t_ms)
        out = (tod - _ms(OPEN)) / 60_000
        out[~self._trading[idx]] = np.nan
        return out

    def local_dates(self, t_ms: F) -> list[date]:
        idx, _ = self._local(t_ms)
        return [date.fromordinal(EPOCH.toordinal() + self._day0 + int(i)) for i in idx.tolist()]

    def verify(self, dates_with_bars: Iterable[date], start: date, end: date) -> list[str]:
        """Disagreements between this calendar and the days an exchange actually printed a bar."""
        seen = {d for d in dates_with_bars if start <= d <= end}
        expected = set(self.trading_days(start, end))
        return [f"{d}: bar on a calendar holiday" for d in sorted(seen - expected)] + [
            f"{d}: no bar on a calendar trading day" for d in sorted(expected - seen)
        ]
