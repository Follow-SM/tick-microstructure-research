"""Task 5a: charts (matplotlib, 150 dpi PNG) from a loaded dataset + computed results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tick_microstructure.analytics.data import Dataset
from tick_microstructure.simulation.engine import PRIMARY_MARKOUT_S, pnl_paths

COLOURS = {
    "static": "#9e9e9e",
    "breaker": "#7e57c2",
    "placebo": "#ef6c00",
    "binance_move": "#26a69a",
    "binance_sweep": "#26a69a",
    "followsm_obt_spike": "#7e57c2",
}


def pnl_drawdown(ds: Dataset, sym: str, trigger: str, path: Path) -> bool:
    paths = pnl_paths(ds, sym, trigger)
    if paths is None:
        return False
    hours = (paths["t"] - paths["t"][0]) / 3_600_000
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True, height_ratios=(3, 2))
    series_labels = (
        ("static", "static quoting", "-"),
        ("placebo", "same trigger, random timing (median)", "--"),
        ("breaker", f"circuit breaker ({trigger})", "-"),
    )
    for key, label, style in series_labels:
        series = paths[key]
        top.plot(hours, series, color=COLOURS[key], ls=style, label=label, lw=1.5)
        bottom.plot(hours, series - np.maximum.accumulate(series), color=COLOURS[key], ls=style, lw=1.2)
    top.axhline(0, color="black", lw=0.6)
    top.set_ylabel(f"cumulative {PRIMARY_MARKOUT_S}s markout, USD")
    top.set_title(f"{sym} Polymarket Up/Down maker: static quoting vs circuit breaker")
    top.legend()
    top.grid(alpha=0.3)
    bottom.set_ylabel("drawdown, USD")
    bottom.set_xlabel("hours since first fill")
    bottom.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def leadlag_cdf(leadlag: dict[str, Any], sym: str, path: Path) -> bool:
    per = leadlag.get(sym)
    if not per:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for name, res in per.items():
        lags = np.asarray(res.get("local_lags", []))
        pairs = res.get("pairs", 0)
        if pairs == 0:
            continue
        xs = np.sort(lags)
        ys = np.arange(1, xs.size + 1) / pairs  # non-responders keep the curve below 1
        ax.step(
            xs,
            ys,
            where="post",
            color=COLOURS.get(name, "#455a64"),
            ls="--" if name == "placebo" else "-",
            label=f"{name} (n={pairs})",
        )
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xscale("symlog", linthresh=10)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel(
        "ms from Binance event to Polymarket reprice, local clock (< 0: Polymarket reached us first)"
    )
    ax.set_ylabel("share of event-market pairs repriced")
    ax.set_title(f"{sym}: Binance -> Polymarket lead-lag CDF (placebo = random times)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def obt_distribution(ds: Dataset, path: Path) -> bool:
    series = {s: t.obt[np.isfinite(t.obt)] for s, t in ds.toxicity.items() if t.obt.size}
    if not series:
        return False
    fig, ax = plt.subplots(figsize=(9, 5))
    for sym, values in sorted(series.items()):
        xs = np.sort(np.clip(values, 1e-3, None))
        ax.plot(xs, 1 - np.arange(xs.size) / xs.size, label=f"{sym} (n={xs.size})")
    ax.axvline(2.0, color="black", ls="--", lw=0.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ob_toxicity_1pct (ask / bid notional within +-1% of mid)")
    ax.set_ylabel("share of frames above")
    ax.set_title("How often the +-1% book is ask-heavy (dashed: the 2.0 trigger)")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def render_all(ds: Dataset, results: dict[str, Any], out_dir: Path, sym: str) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    trigger = "composite" if sym in ds.toxicity else "sweep"  # venues-only captures have no FollowSM
    if pnl_drawdown(ds, sym, trigger, out_dir / "pnl_drawdown.png"):
        made.append("pnl_drawdown.png")
    if leadlag_cdf(results.get("leadlag", {}), sym, out_dir / "leadlag_cdf.png"):
        made.append("leadlag_cdf.png")
    if obt_distribution(ds, out_dir / "obt_distribution.png"):
        made.append("obt_distribution.png")
    return made
