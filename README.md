# Autonomous Crypto Trading Agent

A modular, config-switchable crypto trading agent for MEXC. It combines a
price-action **market-structure** parser (swing-based HH/HL/LH/LL) with a
zero-shot **Chronos** time-series forecast, and only trades when both engines
agree. Runs spot (long-only) or futures (long + short) from a single config
flag, with a full simulated PnL ledger for dry-run validation before any real
money is committed.

> **Reality check.** This is a tool, not a money printer. Most retail strategies
> are net-negative after fees. Run it in `DRY_RUN` until the simulated equity
> curve is convincingly net-positive after fees and slippage, then risk only
> what you can lose. There is no guaranteed monthly yield.

---

## How it decides

1. **Market structure** (`core/structure.py`) — detects *confirmed* swing highs
   and lows with a symmetric fractal window (no lookahead) and classifies the
   trend as bullish, bearish, or consolidating.
2. **Forecast bias** (`core/forecast.py`) — Amazon Chronos forecasts the median
   terminal price over a short horizon and derives a directional bias against a
   buffer `alpha`. Best used for higher-timeframe bias, not tick timing.
3. **Congruence gate** (`core/strategy.py`) — an order fires only when structure
   AND forecast point the same way. Bearish signals trade only when shorting is
   enabled (futures); on spot they mean "go to cash".
4. **Risk** (`core/risk.py`) — position size is solved from the stop distance so
   each trade risks a fixed fraction of equity, capped by budget. Bracket is
   1 : `REWARD_RISK_RATIO`.
5. **Portfolio/ledger** (`core/portfolio.py`) — holds open-position state (no
   double entries) and, in dry-run, simulates fills, fees, slippage, and
   stop/target exits into SQLite with a running equity curve.

```
market_data ─► structure ─┐
                          ├─► strategy ─► risk ─► executor ─► portfolio (SQLite)
market_data ─► forecast  ─┘
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # omit torch/chronos to run structure-only
cp .env.example .env                     # then fill in keys and tune params
```

## Run

```bash
# Validate on history first (seconds, not weeks):
python backtest.py BTC/USDT 1500

# Live-data dry run (simulated ledger, no orders sent):
python main.py

# Go live only after dry-run is net-positive. In .env set TRADE_MODE=LIVE.
```

Switch markets entirely from `.env`:

| Goal | `MARKET_TYPE` | `ALLOW_SHORTS` | Notes |
|---|---|---|---|
| Spot, long-only | `spot` | `false` | Works today, no KYC gate |
| Futures, long + short | `futures` | `true` | Needs MEXC Futures API (KYC + application) |

## Safety

- Use a MEXC API key with **withdrawals disabled**.
- `.env` and `*.db` are git-ignored. Never commit secrets.
- Start at the `$10` tier, scale capital only after a proven, fee-positive edge.
- Stop the agent any time with `Ctrl-C`; it finishes the current cycle and exits
  cleanly.

## Tests

```bash
python -m pytest tests/ -q
```

Covers structure detection, risk sizing/caps, and the no-double-entry +
stop-resolution path. They run without torch, chronos, or network.
