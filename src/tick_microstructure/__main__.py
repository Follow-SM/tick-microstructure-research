"""CLI: `tmr collect | analyze | export`."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from tick_microstructure.analytics.calibration import calibrate
from tick_microstructure.analytics.data import load
from tick_microstructure.analytics.leadlag import lead_lag
from tick_microstructure.collector.common import FatalStreamError
from tick_microstructure.collector.export import export_parquet
from tick_microstructure.collector.run import collect
from tick_microstructure.config import load_settings
from tick_microstructure.outreach.charts import render_all
from tick_microstructure.outreach.report import render
from tick_microstructure.research.hypotheses import h1_depth_depletion, h2_sweep_cascade, h3_trigger_detection
from tick_microstructure.simulation.engine import simulate

log = logging.getLogger("tick_microstructure")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def analyze(hours: float | None, chart_symbol: str) -> Path:
    settings = load_settings()
    end = time.time() * 1000 if hours else None
    start = end - hours * 3_600_000 if end is not None and hours else None
    t0 = time.monotonic()
    ds = load(settings.data_dir, start, end)
    log.info("Loaded %.2f h in %.1fs", ds.hours, time.monotonic() - t0)
    results = {
        "calibration": calibrate(ds),
        "leadlag": lead_lag(ds),
        "hypotheses": {
            "h1": h1_depth_depletion(ds),
            "h2": h2_sweep_cascade(ds),
            "h3": h3_trigger_detection(ds),
        },
        "simulation": simulate(ds),
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    out = settings.results_dir / "results.json"
    out.write_text(json.dumps(results, indent=1, default=_json_default))
    charts = render_all(ds, results, settings.outreach_dir / "assets", chart_symbol)
    report = settings.outreach_dir / "REPORT.md"
    report.write_text(render(results, charts))
    log.info("Wrote %s, %s and %d charts in %.1fs", out, report, len(charts), time.monotonic() - t0)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(prog="tmr", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="capture FollowSM + Binance + Polymarket streams")
    c.add_argument("--minutes", type=float, default=None, help="stop after this long (default: run forever)")
    c.add_argument("--venues-only", action="store_true", help="skip FollowSM (no key needed)")
    a = sub.add_parser("analyze", help="calibration, lead-lag, hypotheses, simulation, report")
    a.add_argument("--hours", type=float, default=None, help="only the last N hours (default: everything)")
    a.add_argument("--chart-symbol", default="BTCUSDT")
    e = sub.add_parser("export", help="convert captured JSONL to Parquet")
    e.add_argument("--out", type=Path, default=Path("parquet"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.cmd == "collect":
        try:
            asyncio.run(collect(load_settings(), args.minutes, args.venues_only))
        except FatalStreamError as exc:
            log.error("%s", exc)
            sys.exit(2)
        except KeyboardInterrupt:
            log.info("Interrupted.")
    elif args.cmd == "analyze":
        analyze(args.hours, args.chart_symbol)
    else:
        written = export_parquet(load_settings().data_dir, args.out)
        log.info("Parquet: %s", written)


if __name__ == "__main__":
    main()
