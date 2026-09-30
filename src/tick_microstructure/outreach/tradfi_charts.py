"""TradFi study charts (matplotlib, 150 dpi PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from tick_microstructure.research.data.fetch import CONTROL_SYMBOL
from tick_microstructure.research.tradfi_study import CORE, PROFILE_MIN, SESSION_NAMES, OpenDay

F = npt.NDArray[np.float64]
EQUITY = "#7e57c2"
EQUITY_ALT = "#b39ddb"
CRYPTO = "#ef6c00"
X = np.arange(-PROFILE_MIN, PROFILE_MIN + 1)


def _pooled(profiles: dict[str, F], symbols: tuple[str, ...]) -> F:
    rows = [profiles[s] for s in symbols if s in profiles]
    return np.vstack(rows) if rows else np.empty((0, X.size))


def _median(m: F) -> F:
    with np.errstate(all="ignore"):
        ok = np.isfinite(m).sum(axis=0) >= 5
        return np.where(ok, np.nanmedian(np.where(np.isfinite(m), m, np.nan), axis=0), np.nan)


def _mark_open(ax: Any) -> None:
    ax.axvline(0, color="black", lw=0.8)
    ax.axvspan(-15, 0, color="#ffe0b2", alpha=0.5, lw=0)
    ax.grid(alpha=0.3)


def vpin_open(charts: dict[str, Any], path: Path) -> bool:
    prof = charts["vpin_profiles"]
    eq = {k: _pooled({s: prof[s][k] for s in prof}, CORE) for k in ("global_v", "session_v", "quote")}
    if not eq["global_v"].size:
        return False
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=(3, 2))
    top.plot(X, _median(eq["global_v"]), color=EQUITY, lw=1.8, label="equity perps, global V (ADV/50)")
    top.plot(X, _median(eq["session_v"]), color=EQUITY_ALT, lw=1.8, label="equity perps, session-aware V")
    if CONTROL_SYMBOL in prof:
        c = prof[CONTROL_SYMBOL]
        top.plot(
            X, _median(c["global_v"]), color=CRYPTO, lw=1.4, ls="--", label="BTCUSDT (24/7 control), global V"
        )
    top.axhline(0.5, color="grey", lw=0.6, ls=":")
    top.set_ylabel("median VPIN percentile\n(vs own trailing 7 days)")
    top.set_title("VPIN around the US equity open (09:30 ET = 16:30 EEST); shaded: 15-minute pre-open")
    top.legend(fontsize=8)
    _mark_open(top)
    with np.errstate(all="ignore"):
        norm = eq["quote"] / np.nanmedian(eq["quote"][:, : PROFILE_MIN - 60], axis=1)[:, None]
    bottom.semilogy(X, _median(norm), color=EQUITY, lw=1.5, label="equity perps")
    if CONTROL_SYMBOL in prof:
        q = prof[CONTROL_SYMBOL]["quote"]
        with np.errstate(all="ignore"):
            norm = q / np.nanmedian(q[:, : PROFILE_MIN - 60], axis=1)[:, None]
        bottom.semilogy(X, _median(norm), color=CRYPTO, lw=1.2, ls="--", label="BTCUSDT")
    bottom.set_ylabel("volume per minute\n/ 06:30-08:30 median")
    bottom.set_xlabel("minutes from the US open")
    bottom.legend(fontsize=8)
    _mark_open(bottom)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def depth_open(charts: dict[str, Any], path: Path) -> bool:
    prof = charts["depth_profiles"]
    rows = charts["profile_days"]
    eq = _pooled(prof, CORE)
    if not eq.size:
        return False
    days: list[OpenDay] = charts["open_days"]
    gaps = {(d.sym, d.day): abs(d.gap) for d in days}
    cut = float(np.quantile(list(gaps.values()), 0.9)) if gaps else np.inf
    big_rows = [
        prof[s][k] for s in CORE if s in prof for k, d in enumerate(rows[s]) if gaps.get((s, d), 0.0) >= cut
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    with np.errstate(all="ignore"):
        lo, hi = np.nanquantile(eq, 0.25, axis=0), np.nanquantile(eq, 0.75, axis=0)
    ax.fill_between(X, lo, hi, color=EQUITY, alpha=0.15, lw=0, label="equity perps, interquartile range")
    ax.plot(X, _median(eq), color=EQUITY, lw=1.8, label=f"equity perps, all days (n={eq.shape[0]})")
    if big_rows:
        ax.plot(
            X,
            _median(np.vstack(big_rows)),
            color="#c62828",
            lw=1.6,
            label=f"top-decile |gap| days (|gap| >= {cut * 1e4:.0f} bps, n={len(big_rows)})",
        )
    if CONTROL_SYMBOL in prof:
        ax.plot(
            X, _median(prof[CONTROL_SYMBOL]), color=CRYPTO, lw=1.3, ls="--", label="BTCUSDT (24/7 control)"
        )
    ax.axhline(1, color="grey", lw=0.6, ls=":")
    ax.set_ylim(0, None)
    ax.set_xlabel("minutes from the US open")
    ax.set_ylabel("+-1% book depth / same day's 08:00-09:00 ET median")
    ax.set_title("Order-book depth within 1% of mid around the US open")
    ax.legend(fontsize=8)
    _mark_open(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def gap_discovery(charts: dict[str, Any], results: dict[str, Any], path: Path) -> bool:
    days: list[OpenDay] = charts["open_days"]
    if not days:
        return False
    x = np.array([d.perp_overnight for d in days]) * 1e4
    y = np.array([d.gap for d in days]) * 1e4
    weekend = np.array([d.closure_h > 30 for d in days])
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(x[~weekend], y[~weekend], s=10, color=EQUITY, alpha=0.5, label="after an overnight close")
    ax.scatter(x[weekend], y[weekend], s=14, color="#c62828", alpha=0.6, label="after a weekend / holiday")
    lim = float(np.nanquantile(np.abs(np.concatenate([x, y])), 0.995))
    ax.plot([-lim, lim], [-lim, lim], color="black", lw=0.8, ls="--", label="gap fully priced by the perp")
    pd = results["pooled_gap"]["price_discovery"]["perp"]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("perp move from the previous 16:00 to 09:29 ET, bps")
    ax.set_ylabel("official opening gap (open / previous close - 1), bps")
    ax.set_title(
        "Opening gap vs what the perp priced before the bell\n"
        f"slope {pd['slope']:.2f}, R² {pd['r2']:.2f}, n={pd['n']}"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def session_volume(results: dict[str, Any], path: Path) -> bool:
    syms = [s for s in (*CORE, CONTROL_SYMBOL) if s in results["symbols"]]
    if not syms:
        return False
    sessions = list(SESSION_NAMES.values())
    colours = ["#7e57c2", "#90caf9", "#26a69a", "#ffb74d", "#bdbdbd"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(syms))
    for s, colour in zip(sessions, colours, strict=True):
        vals = np.array([results["symbols"][sym]["structure"][s]["share_of_volume"] or 0 for sym in syms])
        ax.bar(syms, vals, bottom=bottom, color=colour, label=s)
        bottom += vals
    time_share = [results["symbols"][syms[0]]["structure"][s]["share_of_time"] for s in sessions]
    ax.set_ylabel("share of perp volume")
    ax.set_title(
        "Where the volume trades, by US session of the underlying\n(share of clock time: "
        + ", ".join(f"{s} {v:.0%}" for s, v in zip(sessions, time_share, strict=True))
        + ")"
    )
    ax.legend(fontsize=8, ncols=5, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def spread_open(charts: dict[str, Any], path: Path) -> bool:
    eq = _pooled(charts["spread_profiles"], CORE)
    if not eq.size:
        return False
    fig, ax = plt.subplots(figsize=(10, 4.5))
    with np.errstate(all="ignore"):
        lo, hi = np.nanquantile(eq, 0.25, axis=0), np.nanquantile(eq, 0.75, axis=0)
    ax.fill_between(X, lo, hi, color=EQUITY, alpha=0.15, lw=0)
    ax.plot(X, _median(eq), color=EQUITY, lw=1.6, label="equity perps, median (IQR shaded)")
    ax.set_xlabel("minutes from the US open")
    ax.set_ylabel("effective half-spread, bps")
    ax.set_title("Execution cost from the tick tape: (taker-buy VWAP - taker-sell VWAP) / 2 per minute")
    ax.legend(fontsize=8)
    _mark_open(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def render_tradfi_charts(results: dict[str, Any], charts: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, ok in (
        ("tradfi_fig1_vpin_open.png", vpin_open(charts, out_dir / "tradfi_fig1_vpin_open.png")),
        ("tradfi_fig2_depth_open.png", depth_open(charts, out_dir / "tradfi_fig2_depth_open.png")),
        ("tradfi_gap_discovery.png", gap_discovery(charts, results, out_dir / "tradfi_gap_discovery.png")),
        ("tradfi_session_volume.png", session_volume(results, out_dir / "tradfi_session_volume.png")),
        ("tradfi_spread_open.png", spread_open(charts, out_dir / "tradfi_spread_open.png")),
    ):
        if ok:
            written.append(out_dir / name)
    return written
