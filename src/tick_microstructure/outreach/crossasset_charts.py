"""Crypto vs US stocks charts (matplotlib, 150 dpi PNG)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from tick_microstructure.research.utils.session import NY, Session

CRYPTO = "#ef6c00"
CRYPTO_ALT = "#ffb74d"
EQUITY = "#7e57c2"
EQUITY_ALT = "#b39ddb"
GOLD = "#c9a227"
OIL = "#455a64"
SESSION_SHADE = {Session.RTH: "#e3f2fd", Session.CLOSED: "#eeeeee"}
LINES = {
    "BTCUSDT": ("BTC", CRYPTO, "-"),
    "ETHUSDT": ("ETH", CRYPTO_ALT, "-"),
    "BASKET": ("12 US stock perps (equal weight)", EQUITY, "-"),
    "QQQUSDT": ("QQQ perp", EQUITY_ALT, "--"),
    "MUUSDT": ("MU", "#c62828", ":"),
    "MSFTUSDT": ("MSFT", "#2e7d32", ":"),
    "XAUUSDT": ("gold (XAU)", GOLD, "-"),
    "CLUSDT": ("crude (CL)", OIL, "-"),
}


def _ny(t_ms: np.ndarray) -> list[datetime]:
    return [datetime.fromtimestamp(t / 1000, UTC).astimezone(NY).replace(tzinfo=None) for t in t_ms.tolist()]


def _shade(ax: Any, x: list[datetime], session: np.ndarray) -> None:
    for code, colour in SESSION_SHADE.items():
        on = session == code
        start = None
        for k, flag in enumerate([*on.tolist(), False]):
            if flag and start is None:
                start = k
            elif not flag and start is not None:
                ax.axvspan(x[start], x[min(k, len(x) - 1)], color=colour, lw=0, zorder=0)
                start = None


def event_hourly(results: dict[str, Any], path: Path) -> None:
    p = results["event_paths"]["hourly"]
    x = _ny(np.asarray(p["t"]))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    _shade(ax, x, np.asarray(p["session"]))
    for name in ("BTCUSDT", "ETHUSDT", "BASKET", "QQQUSDT", "MUUSDT", "MSFTUSDT", "XAUUSDT"):
        label, colour, ls = LINES[name]
        lw = 2.2 if name in ("BTCUSDT", "BASKET") else 1.3
        ax.plot(x, np.asarray(p[name]) / 100, color=colour, ls=ls, lw=lw, label=label)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("% since the 3 Nov US close")
    ax.set_title("3-6 Nov 2026, hourly. Blue: US regular hours. All lines are Binance USD-M perps")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M ET"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def event_open(results: dict[str, Any], path: Path) -> None:
    p = results["event_paths"]["open_5nov"]
    x = _ny(np.asarray(p["t"]))
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, height_ratios=(3, 1.4))
    for name in ("BTCUSDT", "ETHUSDT", "BASKET", "QQQUSDT", "XAUUSDT"):
        label, colour, ls = LINES[name]
        top.plot(
            x,
            np.asarray(p[name]),
            color=colour,
            ls=ls,
            lw=2.0 if name in ("BTCUSDT", "BASKET") else 1.3,
            label=label,
            drawstyle="steps-post",
        )
    for ax in (top, bottom):
        ax.axvline(x[29] if len(x) > 29 else x[-1], color="black", lw=0.8)  # close of the 09:29 bar
        ax.grid(alpha=0.3)
    top.axhline(0, color="black", lw=0.6)
    top.set_ylabel("bps since 09:29 ET")
    top.set_title("5 Nov 2026 around the US open (09:30 ET), 1-minute closes")
    top.legend(fontsize=8, ncol=3, loc="lower left")
    btc = np.asarray(p["BTCUSDT_quote"]) / 1e6
    stocks = np.asarray(p["BASKET_quote"]) / 1e6
    bottom.bar(x, btc, width=1 / 1440, color=CRYPTO, alpha=0.7, label="BTC perp volume, $M/min")
    bottom.bar(x, stocks, width=1 / 1440, color=EQUITY, alpha=0.9, label="12 stock perps, $M/min")
    bottom.set_ylabel("$M per minute")
    bottom.legend(fontsize=8)
    bottom.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def session_corr(results: dict[str, Any], path: Path) -> None:
    order = ["rth", "pre", "post", "overnight", "closed"]
    rows = [("BASKET", EQUITY), ("QQQUSDT", EQUITY_ALT), ("XAUUSDT", GOLD), ("CLUSDT", OIL)]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    width = 0.2
    for k, (sym, colour) in enumerate(rows):
        vals = [results["corr"]["BTCUSDT"][sym][s] for s in order]
        y = np.array([v["corr"] for v in vals])
        lo = y - np.array([v["ci95"][0] for v in vals])
        hi = np.array([v["ci95"][1] for v in vals]) - y
        ax.bar(
            np.arange(5) + (k - 1.5) * width,
            y,
            width,
            color=colour,
            yerr=[lo, hi],
            capsize=2,
            label=LINES[sym][0],
        )
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(
        np.arange(5), ["regular hours", "pre-market", "post-market", "overnight", "weekend/holiday"]
    )
    ax.set_ylabel("correlation with BTC\n(5-minute returns, 95% CI)")
    ax.set_title("How tightly each perp moves with BTC, by US session (21 Apr - 7 Nov 2026)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lead_lag_profile(results: dict[str, Any], path: Path) -> None:
    ll = results["leadlag"]["BTCUSDT~BASKET"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for ax, (sess, title) in zip(
        axes,
        (("rth", "US regular hours"), ("overnight", "overnight"), ("closed", "weekends and holidays")),
        strict=True,
    ):
        lags = sorted(ll[sess]["corr_by_lag"], key=int)
        k = np.array([int(v) for v in lags])
        y = np.array([ll[sess]["corr_by_lag"][v] for v in lags])
        ax.bar(k, np.where(k == 0, np.nan, y), color=np.where(k > 0, CRYPTO, EQUITY))
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(f"{title}\nlag 0: ρ = {y[k == 0][0]:.2f} (off scale)", fontsize=10)
        ax.set_xlabel("lag, minutes (+ = BTC moves first)")
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("corr(BTC 1m return, stock basket\n1m return k minutes later)")
    axes[0].set_ylim(-0.03, 0.13)
    fig.suptitle("Who moves first? BTC vs the 12 US stock perps, 1-minute returns", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def rolling_corr(chart: dict[str, Any], results: dict[str, Any], path: Path) -> None:
    days = chart["daily_days"]
    roll = chart["rolling"]["BTCUSDT~BASKET"]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(days, roll, color=EQUITY, lw=1.6, label="BTC vs 12 stock perps")
    ax.plot(
        days, chart["rolling"]["BTCUSDT~QQQUSDT"], color=EQUITY_ALT, lw=1.0, ls="--", label="BTC vs QQQ perp"
    )
    full = results["daily"]["BTCUSDT~BASKET"]["corr"]
    ax.axhline(full, color="grey", lw=0.8, ls=":", label=f"whole period: {full:.2f}")
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("20-day correlation of\nUS close-to-close returns")
    ax.set_title("The relationship is not stable: rolling 20-trading-day correlation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_crossasset_charts(results: dict[str, Any], chart: dict[str, Any], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in (
        ("crossasset_fig1_nov3_6.png", lambda p: event_hourly(results, p)),
        ("crossasset_fig2_open_5nov.png", lambda p: event_open(results, p)),
        ("crossasset_fig3_session_corr.png", lambda p: session_corr(results, p)),
        ("crossasset_fig4_leadlag.png", lambda p: lead_lag_profile(results, p)),
        ("crossasset_fig5_rolling_corr.png", lambda p: rolling_corr(chart, results, p)),
    ):
        fn(out / name)
        paths.append(out / name)
    return paths
