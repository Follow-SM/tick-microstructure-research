"""Small statistics helpers: Wilson intervals, placebo draws, quantile summaries."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt

from tick_microstructure.analytics.data import F, GapMask

QUANTILES = (0.5, 0.9, 0.95, 0.99)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


def rate(successes: int, n: int) -> dict[str, Any]:
    lo, hi = wilson(successes, n)
    return {"n": n, "k": successes, "rate": successes / n if n else None, "ci95": [lo, hi]}


def quantiles(values: F) -> dict[str, float | int | None]:
    finite = values[np.isfinite(values)]
    out: dict[str, float | int | None] = {"n": int(finite.size)}
    for q in QUANTILES:
        out[f"p{round(q * 100)}"] = float(np.quantile(finite, q)) if finite.size else None
    return out


def placebo_times(
    rng: np.random.Generator, n: int, start_ms: float, end_ms: float, gaps: GapMask, horizon_ms: float
) -> F:
    """`n` uniform times in [start, end - horizon] whose [t, t + horizon] avoids gaps."""
    out: list[float] = []
    tries = 0
    while len(out) < n and tries < n * 50:
        tries += 1
        t = float(rng.uniform(start_ms, end_ms - horizon_ms))
        if not gaps.overlaps(t, t + horizon_ms):
            out.append(t)
    return np.asarray(sorted(out), dtype=np.float64)


def circular_shift(times: F, shift_ms: float, start_ms: float, end_ms: float) -> F:
    span = end_ms - start_ms
    return np.sort((times - start_ms + shift_ms) % span + start_ms)


def as_bool(values: npt.ArrayLike) -> npt.NDArray[np.bool_]:
    return np.asarray(values, dtype=bool)
