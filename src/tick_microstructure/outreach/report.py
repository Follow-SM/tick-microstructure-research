"""Task 5b: technical report (markdown) generated from results.json.

Numbers are only ever copied from results; every claim sits next to its control and sample size,
and the report opens with a data-sufficiency verdict so a short capture cannot read as a finding.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

MIN_HOURS = 24 * 14  # two weeks before any threshold is recommended
MIN_EVENTS = 100
X = "\u00d7"  # multiplication sign


def _pct(x: float | None, digits: int = 1) -> str:
    return "n/a" if x is None else f"{x * 100:.{digits}f}%"


def _num(x: float | None, fmt: str = ",.2f") -> str:
    return "n/a" if x is None else format(x + 0.0, fmt).replace("-0.00", "0.00")


def _usd(x: float) -> str:
    return f"${_num(x)}"


def _ci(r: dict[str, Any]) -> str:
    if not r or r.get("rate") is None:
        return "n/a (n=0)"
    lo, hi = r["ci95"]
    return f"{_pct(r['rate'])} [{_pct(lo)}-{_pct(hi)}] (n={r['n']})"


def _row(*cells: str) -> str:
    return "| " + " | ".join(cells) + " |"


def _header(*cells: str) -> list[str]:
    return [_row(*cells), _row(*("---" for _ in cells))]


def _strength(n: int) -> str:
    if n < 30:
        return f"\nEvidence strength: **anecdotal** (n={n}); do not act on it."
    if n < MIN_EVENTS:
        return f"\nEvidence strength: **weak** (n={n}); direction only."
    return f"\nEvidence strength: n={n}; read the intervals."


NO_FOLLOWSM = "_No FollowSM toxicity frames in this capture; this test needs them._"


def sufficiency(results: dict[str, Any]) -> tuple[bool, str]:
    cal = results["calibration"]
    hours = cal["span_hours"]
    if not any(s["delivery"].get("arrival_lag_ms") for s in cal["symbols"].values()):
        return False, "No FollowSM frames in this capture: only the venue ground truth was recorded."
    if hours < MIN_HOURS:
        return False, (
            f"This capture spans {hours:.1f} h. Thresholds and effect sizes need at least "
            f"{MIN_HOURS // 24} days of clean data; everything below is a pipeline check, not a result."
        )
    return True, f"Capture spans {hours / 24:.1f} days."


def _capture(cal: dict[str, Any]) -> list[str]:
    gaps = ", ".join(f"{k} {v * 60:.1f} min" for k, v in cal["gap_hours"].items())
    lines = [
        "## Capture",
        "",
        f"- Span: {cal['span_hours']:.2f} h. Gap time: {gaps}.",
        f"- Collector events: {cal['events']}",
        "",
        *_header(
            "Symbol",
            "ob_toxicity_1pct p50 / p90 / p95 / p99",
            "share > 2",
            "taker z p99",
            "FollowSM arrival lag p50 / p99",
            "frame interval p50",
        ),
    ]
    for sym, s in cal["symbols"].items():
        w = s["windows"][-1] if s["windows"] else {}
        obt, z = w.get("ob_toxicity_1pct", {}), w.get("taker_flow_z", {})
        lag = s["delivery"].get("arrival_lag_ms", {})
        interval = s["delivery"].get("frame_interval_ms", {})
        obt_q = " / ".join(_num(obt.get(k)) for k in ("p50", "p90", "p95", "p99"))
        lines.append(
            _row(
                sym,
                obt_q,
                _pct(w.get("share_obt_gt_2")),
                _num(z.get("p99")),
                f"{_num(lag.get('p50'), ',.0f')} / {_num(lag.get('p99'), ',.0f')} ms",
                f"{_num(interval.get('p50'), ',.0f')} ms",
            )
        )
    conf_rows = [(sym, s["confluence"]) for sym, s in cal["symbols"].items() if s.get("confluence")]
    if conf_rows:
        lines += [
            "",
            "Cross-venue prediction divergence (confluence stream): Polymarket-implied 15 m direction "
            f"(direction {X} `prob_delta_15m`) against the Binance 15 m move.",
            "",
            *_header(
                "Symbol", "frames", "divergence flag", "sign disagreement", "corr", "whale 1h p50 / p99"
            ),
        ]
        for sym, c in conf_rows:
            whale = c["whale_sweeps_1h_usdt"]
            lines.append(
                _row(
                    sym,
                    str(c["frames"]),
                    _pct(c["divergence_flag_share"]),
                    _pct(c["sign_disagreement_share"]),
                    _num(c["implied_vs_spot_corr"]),
                    f"${_num(whale.get('p50'), ',.0f')} / ${_num(whale.get('p99'), ',.0f')}",
                )
            )
    return [*lines, ""]


def _h1(h1: dict[str, Any]) -> list[str]:
    lines = ["## H1: ±1% depth depletion (`ob_toxicity_1pct > 2`) → Binance down-move ≥ 10 bps", ""]
    if not h1:
        return [*lines, NO_FOLLOWSM, ""]
    for sym, res in h1.items():
        full = res["plan_threshold_2.0_full_sample"]
        lines += [
            f"**{sym}**",
            "",
            *_header("Horizon", "P(down) signal", "P(down) control", "lift", "P(up) signal", "P(up) control"),
        ]
        for h in ("10s", "30s", "60s"):
            s, c = full[h]["signal"], full[h]["control"]
            lift = f"{_num(s.get('lift_down'))}{X}"
            lines.append(_row(h, _ci(s["down"]), _ci(c["down"]), lift, _ci(s["up"]), _ci(c["up"])))
        wf = res["walk_forward"]
        chosen = wf["chosen_threshold"]
        lines += ["", f"Walk-forward threshold (chosen on the first half by 30 s lift): **{chosen}**."]
        if chosen and wf["test"]:
            s = wf["test"]["30s"]["signal"]
            lines.append(
                f"Out-of-sample 30 s: down-move rate {_ci(s['down'])}, lift {_num(s.get('lift_down'))}{X}, "
                f"false-positive rate {_pct(s.get('false_positive_rate'))}."
            )
        lines += [_strength(full["30s"]["signal"]["down"]["n"]), ""]
    return lines


def _h2(h2: dict[str, Any]) -> list[str]:
    lines = [
        "## H2: whale sweep (≥ $100k, one matching event) + book confirmation → Polymarket repricing",
        "",
    ]
    groups = ("confirmed_from_sweep", "confirmed_from_confirmation", "unconfirmed", "placebo")
    for sym, res in h2.items():
        lines += [
            f"**{sym}**: {res['sweeps']} sweeps.",
            "",
            *_header("Group", "events", "pairs", "reprice ≤ 5 s", "≤ 10 s", "≤ 30 s", "median lag"),
        ]
        for group in groups:
            g = res[group]
            lines.append(
                _row(
                    group,
                    str(g["events"]),
                    str(g["market_pairs"]),
                    _ci(g["reprice_within_5s"]),
                    _ci(g["reprice_within_10s"]),
                    _ci(g["reprice_within_30s"]),
                    f"{_num(g['median_lag_ms'], ',.0f')} ms",
                )
            )
        lines += [_strength(res["confirmed_from_sweep"]["market_pairs"]), ""]
    return lines


def _h3(h3: dict[str, Any]) -> list[str]:
    lines = [
        "## H3: individual vs composite triggers as detectors of Binance move onsets (≥ 10 bps in 2 s)",
        "",
        "Coverage = share of onsets the trigger was already active for (at the lead). Precision = share of "
        "trigger onsets followed by a move within 30 s. Placebo = the same intervals circularly shifted.",
        "",
    ]
    if not h3:
        return [*lines, "_No Binance move onsets in this capture._", ""]
    for sym, res in h3.items():
        lines += [
            f"**{sym}**: {res['move_onsets']} move onsets.",
            "",
            *_header(
                "Trigger",
                "active",
                "coverage lead 0 / 1 s / 5 s",
                "placebo coverage 1 s",
                "precision",
                "placebo precision",
                "placebo ≥ actual (cov / prec)",
            ),
        ]
        for name, r in res.items():
            if name == "move_onsets":
                continue
            a, p, share = r["actual"], r["placebo_mean"], r["placebo_share_at_least_as_good"]
            coverage = " / ".join(_pct(a[f"coverage_lead_{k}ms"]) for k in (0, 1000, 5000))
            lines.append(
                _row(
                    name,
                    _pct(a["active_share"]),
                    coverage,
                    _pct(p["coverage_lead_1000ms"]),
                    _pct(a["precision_30s"]),
                    _pct(p["precision_30s"]),
                    f"{_pct(share['coverage_lead_1000ms'], 0)} / {_pct(share['precision_30s'], 0)}",
                )
            )
        lines += [_strength(res["move_onsets"]), ""]
    return lines


def _sim(sim: dict[str, Any]) -> list[str]:
    lines = [
        f"## Simulation: static quoting vs circuit breaker ({sim['primary_markout_s']} s markout, "
        f"{sim['quote_size']:.0f}-share quotes)",
        "",
        "Fills are an upper bound (queue position ignored). Net value = slippage avoided − spread capture "
        "missed. Placebo ≥ actual = share of shifted placebos that did at least as well.",
        "",
    ]
    for sym, res in sim["symbols"].items():
        lag = res["latency_budget"]["stream_arrival_lag_ms"]
        dec = res["latency_budget"]["evaluator_decision_us"]
        lines += [
            f"**{sym}**: {res['fills']} fills. Latency budget: stream arrival lag p50 "
            f"{_num(lag.get('p50'), ',.0f')} ms / p99 {_num(lag.get('p99'), ',.0f')} ms; evaluator p50 "
            f"{_num(dec.get('p50'))} µs / p99 {_num(dec.get('p99'))} µs; "
            "OMS cancel assumed (see sensitivity).",
            "",
            *_header(
                "Trigger",
                "active",
                "fills avoided",
                "slippage avoided",
                "spread missed",
                "net value",
                "placebo net p50",
                "placebo ≥ actual",
            ),
        ]
        for name, r in res["triggers"].items():
            a = r["actual"]
            lines.append(
                _row(
                    name,
                    _pct(r["active_share"]),
                    f"{a['fills_avoided']:.0f}/{a['fills']:.0f}",
                    _usd(a["slippage_avoided"]),
                    _usd(a["spread_missed"]),
                    _usd(a["net_value"]),
                    f"${_num(r['placebo_net_value'].get('p50'))}",
                    _pct(r["placebo_share_at_least_as_good"], 0),
                )
            )
        composite = res["triggers"].get("composite")
        if composite:
            sens = ", ".join(
                f"{k} ms: ${v['net_value']:,.2f}" for k, v in composite["by_cancel_latency_ms"].items()
            )
            lines += ["", f"Composite net value by OMS cancel latency: {sens}."]
        lines.append("")
    return lines


def render(results: dict[str, Any], charts: list[str]) -> str:
    ok, verdict = sufficiency(results)
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Tick Microstructure Research: Signal Evaluation Report",
        "",
        f"*Generated {generated} from `results/results.json`.*",
        "",
        f"> **Data sufficiency: {'OK' if ok else 'INSUFFICIENT'}.** {verdict}",
        "",
        *_capture(results["calibration"]),
        *_h1(results["hypotheses"]["h1"]),
        *_h2(results["hypotheses"]["h2"]),
        *_h3(results["hypotheses"]["h3"]),
        *_sim(results["simulation"]),
    ]
    if charts:
        lines += ["## Charts", "", *(f"![{c}](assets/{c})" for c in charts), ""]
    lines += [
        "## Method notes",
        "",
        "- Signals are evaluated at local receive time (when a client here could act); outcomes on the "
        "Binance book; cross-venue lags on both clocks.",
        "- Windows touching a logged gap (reconnects, cadence breaks, unrecovered trades) are excluded.",
        "- Thresholds are only recommended from walk-forward splits, and only once the sufficiency "
        "check passes.",
        "",
    ]
    return "\n".join(lines)
