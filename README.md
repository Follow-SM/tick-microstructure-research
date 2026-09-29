# tick-microstructure-research

**Do FollowSM's live tick signals actually protect a market maker, and by how much?** This repo records the live streams, tests the signals against a placebo, and simulates quote cancellation.

- **Capture.** It records FollowSM's Enterprise streams (`/ws/v1/toxicity`, `/ws/v1/confluence`) alongside the venues' own public tick data, which serves as ground truth.
- **Calibrate.** It profiles the order-book signals.
- **Test.** It tests three hypotheses, each against a control group.
- **Simulate.** It simulates quote cancellation on real Polymarket books and compares every result with a **placebo** that fires the same trigger at random times.

Written as a follow-up to the [dynamic-inventory-hedger backtest](https://github.com/Follow-SM/dynamic-inventory-hedger#honest-backtest--methodology). That backtest could only rebuild VPIN from 1-minute public bars, and it found no timing value. The question left open was whether **tick-level book and sweep signals** do better. This repo collects the data needed to answer it.

> **Status: pipeline complete, data collection pending.** Every number in a report is generated from captured data. The report states **INSUFFICIENT** until at least 14 days of clean capture exist, and it never recommends a threshold before then.

---

## Architecture

```mermaid
flowchart LR
    subgraph collector/
        F1["FollowSM /ws/v1/toxicity<br/>~1 Hz, all symbols"] --> S[("data/&lt;stream&gt;/&lt;day&gt;/&lt;hour&gt;.jsonl(.gz)")]
        F2["FollowSM /ws/v1/confluence<br/>changed symbols"] --> S
        B["Binance aggTrade + bookTicker<br/>(ground truth, gap-free trades)"] --> S
        P["Polymarket CLOB market WS<br/>BBO + trades, rolling Up/Down markets"] --> S
    end
    S --> A["analytics/<br/>calibration · lead-lag"]
    S --> R["research/<br/>H1 · H2 · H3"]
    S --> M["simulation/<br/>cancel simulator + placebo"]
    A & R & M --> O["outreach/<br/>results.json · REPORT.md · charts"]
```

| Package | Plan task | What it does |
|---|---|---|
| `collector/` | 1 | Async WebSocket consumers with reconnect and backoff; hourly JSONL, gzipped once the hour closes; optional Parquet export; gap detection |
| `analytics/` | 2 | Rolling 24 h p50/p90/p95/p99 profiles; taker-flow spikes vs a 5-minute EWMA; cross-venue prediction divergence; Binance → Polymarket lead-lag on two clocks |
| `research/` | 3 | H1 depth depletion, H2 whale-sweep cascade, H3 composite vs individual triggers, each against a control |
| `simulation/` | 4 | Paper quote-cancellation simulator: latency budget, adverse selection avoided vs spread missed, placebo benchmark |
| `outreach/` | 5 | P&L/drawdown, lead-lag CDF and depth-distribution charts; a markdown technical report generated from results |

### Why the venues' own data is recorded too

FollowSM streams **1-second snapshots** of order-book and flow metrics. Two of the tests need things those snapshots don't contain:

- **Per-tick taker trades** for H2's single-sweep test. These come from Binance `aggTrade`.
- **Sub-second price paths** for the 10-second outcomes and the latency budget. These come from Binance `bookTicker` and Polymarket's best bid/ask.

So FollowSM provides the **signals**, and the venues provide the **outcomes**. That also keeps the test independent: FollowSM data is never scored against FollowSM data.

### Data integrity

- **Binance trades are gap-free by construction.** Aggregate-trade IDs are consecutive. Any hole (dropped frames or a reconnect) is detected and back-filled from REST `aggTrades?fromId=`, and back-filled rows are tagged `bf: 1`. Holes that can't be recovered are logged as `unrecovered`.
- **Streams without sequence numbers** (FollowSM, Binance `bookTicker`, Polymarket) have reconnect windows and cadence breaks logged as `gap` events. The analytics **exclude any window that touches a gap**, so an outage can't create or hide an effect.
- **Two timestamps on every record:**
  - `rt` is the local receive time, which is when a client here could act;
  - the venue or server timestamp is kept alongside it.
  Signals are evaluated at `rt`. Cross-venue lead-lag is reported on both clocks.
- **Storage.** Files are written with append-only JSONL and rotated hourly. Compression runs on a background thread and swaps files atomically, and readers ignore a torn final line. A crash loses at most the last second.

## The tests

**H1: depth depletion.**
- *Question:* does `ob_toxicity_1pct > 2` predict a Binance mid move of ≤ −10 bps within 10 s, 30 s and 60 s? `ob_toxicity_1pct` is the ask/bid notional ratio within ±1% of mid, so values above 2 mean ask depth is at least twice bid depth.
- *Reported:* down-move and up-move rates with Wilson 95% intervals, against random frames where the condition was false. Up-moves are shown too, so a volatility effect can't pass as a directional one.
- *Thresholds:* walk-forward. A threshold is chosen on the first half and reported on the second.

**H2: whale-sweep cascade.**
- *Question:* does a ≥ $100k taker sweep (one matching event on Binance), confirmed by a same-direction book spike within 5 s, make the matching Polymarket Up/Down market reprice sooner or more often?
- *Controls:* unconfirmed sweeps, and random times with a random direction.

**H3: composite vs individual triggers.**
- *Triggers compared:* `vpin_percentile ≥ 0.95`, sweep, `ob_toxicity_1pct > 2`, and their union. Also a **public-data-only reactive baseline** (a ≥ 10 bps Binance move in 2 s), which a paid signal must beat.
- *What each trigger detects:* Binance move onsets.
- *Scored on:* time active (the cost), coverage at a 0 / 1 / 5 s lead, and precision.
- *Placebo:* the same intervals, circularly shifted.

**Simulation.**
- *Setup:* a maker joins the best bid and best ask on every tracked Polymarket Up/Down token. Recorded Polymarket trades at or through the quote fill it, and each fill is marked out 5, 30 and 60 s later.
- *Breaker:* quotes are pulled while a trigger is active. The whole window is shifted by the **latency budget**, stream arrival → evaluator decision → OMS cancel:
  - arrival lag is measured;
  - evaluator latency is measured in-process;
  - OMS latency is a parameter swept from 0 to 1,000 ms, because no live orders are placed.
- *Result:* slippage avoided, spread capture missed, and net value, against the placebo distribution.
- *Limits:* the model is deliberately simple and says so. Queue position is ignored (fills are an upper bound), and the Polymarket mid stands in for fair value.

## Quickstart

```bash
git clone https://github.com/Follow-SM/tick-microstructure-research.git
cd tick-microstructure-research
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,parquet]"
cp .env.example .env          # set FOLLOWSM_API_KEY (ENTERPRISE)

tmr collect                   # runs until stopped; add --minutes N for a test run
tmr analyze                   # results/results.json, outreach/REPORT.md, outreach/assets/*.png
tmr export --out parquet/     # optional Parquet copy of every stream
```

- **No Enterprise key?** `tmr collect --venues-only` records the Binance and Polymarket data on its own. Lead-lag and the sweep/move triggers work; H1 and the FollowSM triggers need the Enterprise streams.
- **Status lines.** Every 30 s the collector logs message rates, tracked markets and gap counters.
- **Storage.** Measured at about **210 MB per day, gzipped, for two symbols**, around 9× compression.

### Long captures

The tests need weeks of data, so run the collector as a service. For example, a systemd user unit:

```ini
# ~/.config/systemd/user/tmr-collect.service
[Unit]
Description=tick-microstructure-research collector
After=network-online.target

[Service]
WorkingDirectory=%h/tick-microstructure-research
ExecStart=%h/tick-microstructure-research/.venv/bin/tmr collect
Restart=always
RestartSec=10

[Install]
WantedBy=default.target
```

```bash
systemctl --user enable --now tmr-collect
loginctl enable-linger "$USER"     # keep it running after logout
```

If `systemctl --user` fails with *"Failed to connect to bus"*, there is no user session bus (common over SSH). Use a system unit instead. It runs as your user but needs `sudo` to install:

```ini
# /etc/systemd/system/tmr-collect.service
[Unit]
Description=tick-microstructure-research collector
After=network-online.target
Wants=network-online.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/tick-microstructure-research
ExecStart=/home/ubuntu/tick-microstructure-research/.venv/bin/tmr collect
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now tmr-collect
systemctl status tmr-collect                  # state
journalctl -u tmr-collect -f                  # live log; a status line every 30 s
```

Restarts are safe. The collector appends to the current hour, compresses any hours left open by a crash, and the gap between runs is logged and excluded.

### Configuration (`.env`)

| Variable | Default | |
|---|---|---|
| `FOLLOWSM_API_KEY` | | ENTERPRISE key for the WebSocket streams |
| `TMR_SYMBOLS` | `BTCUSDT,ETHUSDT,SOLUSDT,XRPUSDT` | Binance symbols; the ones with Polymarket Up/Down markets get cross-venue tests |
| `TMR_DATA_DIR` / `TMR_RESULTS_DIR` / `TMR_OUTREACH_DIR` | `data` / `results` / `outreach` | Output locations |

## Development

```bash
pytest -q && ruff check src tests && mypy
```

- **Offline tests:**
  - storage rotation, compression and crash recovery;
  - trade-hole detection;
  - stream parsing;
  - sweep and move detection;
  - interval logic.
- **Synthetic simulator:** an informative trigger has to beat its placebo, a late one must not, and a slow cancel must miss.
- **End-to-end:** a full pipeline run on a synthetic capture written through the real storage layer.
- **Static checks:** mypy runs in strict mode.

## Plans

| | Free | DEVELOPER ($199/mo) | **ENTERPRISE** ($499/mo) |
|---|---|---|---|
| REST requests/min | 30 (per IP) | 300 | 1,000 |
| `/ws/v1/toxicity` + `/ws/v1/confluence` (needed by `tmr collect`) | ❌ | ❌ | ✅ |

### 👉 [Get your API key at follow-sm.com/pricing](https://follow-sm.com/pricing)

## Disclaimer

Research software, MIT-licensed, provided as-is. Not financial advice. Simulated fills ignore queue position and market impact; paper results overstate what live quoting achieves.
