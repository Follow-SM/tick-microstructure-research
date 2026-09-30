"""Polymarket-vs-perp study charts (matplotlib, 150 dpi PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tick_microstructure.research.polymarket_study import EVENT_POST, EVENT_PRE

PERP = "#7e57c2"
PM = "#26a69a"
SESSIONS = ("closed", "overnight", "pre", "rth", "post")


def leadlag(results: dict[str, Any], path: Path) -> bool:
    ll = results.get("leadlag_pooled", {}).get("1m")
    if not ll:
        return False
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.4
    for shift, (name, colour) in zip((-width / 2, width / 2), (("rth", PERP), ("offhours", PM)), strict=True):
        cc = {int(k): v for k, v in ll[name]["xcorr"].items() if abs(int(k)) <= 5}
        lags = sorted(cc)
        ax.bar(
            np.array(lags) + shift,
            [cc[k] for k in lags],
            width,
            color=colour,
            label=f"{name} (n={ll[name]['minutes']:,})",
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("lag, minutes (positive = Polymarket moves first, negative = perp moves first)")
    ax.set_ylabel("correlation: Polymarket Δp vs perp return")
    ax.set_title("Who moves first? 1-minute cross-correlation, all equity Up/Down market-nights")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def session_calibration(results: dict[str, Any], path: Path) -> bool:
    sc = results.get("session_calibration", {})
    names = [s for s in SESSIONS if s in sc]
    if not names:
        return False
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    x = np.arange(len(names))
    rates = [sc[s]["perp_right_when_disagree"] for s in names]
    left.bar(x, [r["rate"] for r in rates], color=PERP)
    left.errorbar(
        x,
        [r["rate"] for r in rates],
        yerr=np.array([[r["rate"] - r["ci95"][0] for r in rates], [r["ci95"][1] - r["rate"] for r in rates]]),
        fmt="none",
        color="black",
        capsize=4,
    )
    left.axhline(0.5, color="grey", lw=0.8, ls=":")
    left.set_xticks(x, [f"{s}\nn={r['n']:,}" for s, r in zip(names, rates, strict=True)], fontsize=8)
    left.set_ylim(0.4, 0.7)
    left.set_ylabel("share resolved on the perp's side")
    left.set_title("When perp-implied and Polymarket probabilities differ by > 5¢")
    left.grid(alpha=0.3, axis="y")
    diff = [(sc[s]["brier_pm"] - sc[s]["brier_perp"]) * 1e3 for s in names]
    lo = [
        (sc[s]["brier_pm"] - sc[s]["brier_perp"]) * 1e3 - sc[s]["diff_pm_minus_perp_ci95"][0] * 1e3
        for s in names
    ]
    hi = [
        sc[s]["diff_pm_minus_perp_ci95"][1] * 1e3 - (sc[s]["brier_pm"] - sc[s]["brier_perp"]) * 1e3
        for s in names
    ]
    right.bar(x, diff, color=PM)
    right.errorbar(x, diff, yerr=np.array([lo, hi]), fmt="none", color="black", capsize=4)
    right.axhline(0, color="black", lw=0.8)
    right.set_xticks(x, names, fontsize=8)
    right.set_ylabel("Brier(Polymarket) - Brier(perp), x1000")
    right.set_title("Positive = the perp-implied probability is better calibrated")
    right.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def events(charts: dict[str, Any], results: dict[str, Any], path: Path) -> bool:
    ev = charts.get("events", {})
    if "perp_jump->pm" not in ev or "mean" not in ev["perp_jump->pm"]:
        return False
    x = np.arange(-EVENT_PRE, EVENT_POST + 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))
    c = ev["perp_jump->pm"]
    left.fill_between(x, c["lo"], c["hi"], color=PM, alpha=0.2, lw=0)
    left.plot(x, c["mean"], color=PM, lw=1.8)
    left.axvline(0, color="black", lw=0.8)
    left.axhline(0, color="black", lw=0.5)
    n = results["events"]["perp_jump->pm"]["n"]
    left.set_title(f"Polymarket after a perp jump (off-hours |5m| ≥ p99, n={n:,})")
    left.set_xlabel("minutes from the end of the perp's 5-minute jump")
    left.set_ylabel("Polymarket Up probability change, in the jump's direction")
    left.grid(alpha=0.3)
    for name, colour, label in (
        ("sweep->perp", PERP, "whale sweep"),
        ("pm_jump->perp", PM, "Polymarket jump ≥ 10¢"),
    ):
        c = ev.get(name, {})
        if "mean" not in c:
            continue
        right.fill_between(x, c["lo"], c["hi"], color=colour, alpha=0.15, lw=0)
        right.plot(x, c["mean"], color=colour, lw=1.6, label=f"{label} (n={c['n']:,})")
        if "placebo" in c:
            pl = np.asarray(c["placebo"])
            right.errorbar(
                [EVENT_POST + 2],
                [float(np.mean(pl))],
                yerr=[[np.mean(pl) - pl.min()], [pl.max() - np.mean(pl)]],
                fmt="o",
                color=colour,
                ms=3,
                capsize=3,
            )
    right.axvline(0, color="black", lw=0.8)
    right.axhline(0, color="black", lw=0.5)
    right.set_title("Perp after Polymarket triggers (dots: 20 placebo seeds at +60)")
    right.set_xlabel("minutes from the Polymarket trigger")
    right.set_ylabel("signed perp return, bps")
    right.legend(fontsize=8)
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def render_polymarket_charts(results: dict[str, Any], charts: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, ok in (
        ("polymarket_leadlag.png", leadlag(results, out_dir / "polymarket_leadlag.png")),
        (
            "polymarket_session_calibration.png",
            session_calibration(results, out_dir / "polymarket_session_calibration.png"),
        ),
        ("polymarket_events.png", events(charts, results, out_dir / "polymarket_events.png")),
    ):
        if ok:
            written.append(out_dir / name)
    return written
