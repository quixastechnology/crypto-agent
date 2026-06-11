# Autonomous Crypto Trading Agent

A modular, config-switchable crypto trading agent for MEXC. It **assesses every
signal first**, then opens the single path (long, short, or nothing) with the
best expected value. Signals: price-action **market structure** (swing-based
HH/HL/LH/LL), a zero-shot **Chronos** forecast, **momentum** (RSI + MACD),
**sentiment** (Fear & Greed, plus optional news-ML), and a **volatility**
regime veto. Runs spot (long-only) or futures (long + short) from a single
config flag, with a full simulated PnL ledger for dry-run validation before any
real money is committed.

> **Reality check.** This is a tool, not a money printer. Most retail strategies
> are net-negative after fees. Run it in `DRY_RUN` until the simulated equity
> curve is convincingly net-positive after fees and slippage, then risk only
> what you can lose. There is no guaranteed monthly yield.

---

## How it decides

The agent assesses first, then trades the best path. Each signal provider
returns a direction score `[-1,+1]`, a confidence `[0,1]`, and can raise a veto.

1. **Signal providers** (`core/signals/`):
   - `structure` — confirmed swing highs/lows (no lookahead) → trend.
   - `forecast` — Amazon Chronos median path → directional bias.
   - `momentum` — RSI + MACD histogram.
   - `fear_greed` — free Crypto Fear & Greed Index (contrarian at extremes).
   - `news_sentiment` — optional CryptoBERT/FinBERT over headlines (opt-in).
   - `volatility` — ATR regime; vetoes trading when too chaotic.
2. **Decision engine** (`core/decision.py`) — blends signals by configurable
   weight × confidence into a composite view, converts it to a directional
   probability, computes the **expected value of long vs short** net of
   round-trip fees, and picks the higher-EV path only if it clears
   `MIN_EXPECTED_VALUE` and `MIN_CONVICTION` and isn't vetoed. By default
   structure must agree with the chosen direction (`REQUIRE_STRUCTURE_ALIGNMENT`).
3. **Risk** (`core/risk.py`) — position size is solved from the stop distance so
   each trade risks a fixed fraction of equity, capped by budget. Bracket is
   1 : `REWARD_RISK_RATIO`.
4. **Portfolio/ledger** (`core/portfolio.py`) — holds open-position state (no
   double entries) and, in dry-run, simulates fills, fees, slippage, and
   stop/target exits into SQLite with a running equity curve.

```
              structure ┐
              forecast  │
market_data ─►momentum  ├─► DecisionEngine ─► risk ─► executor ─► portfolio (SQLite)
              sentiment │      (weighted EV)
              volatility┘   LONG / SHORT / FLAT
```

Tune the mix in `.env`: `WEIGHT_STRUCTURE`, `WEIGHT_FORECAST`, `WEIGHT_MOMENTUM`,
`WEIGHT_SENTIMENT`, `WEIGHT_NEWS`, plus `MIN_CONVICTION` / `MIN_EXPECTED_VALUE`.

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

## Analytics

```bash
python analytics.py                # report on the live/dry-run ledger
python analytics.py backtest.db    # report on the last backtest
python analytics.py backtest.db --csv   # also export trades.csv + equity_curve.csv
```

Reports profit factor, expectancy, max drawdown, per-trade Sharpe, longest
losing streak, and breakdowns by exit reason and symbol. The go-live bar is not
"win rate > 50%" — it is profit factor > 1 with a drawdown you can stomach,
sustained across hundreds of trades.

## Risk guardrails

- `MAX_OPEN_POSITIONS` caps total concurrent positions (correlated exposure).
- `DAILY_MAX_LOSS_PCT` halts new entries after a bad day (kill switch).
- `COOLDOWN_SECONDS` blocks immediate re-entry after a stop-out (anti-chop).
- `USE_ATR_STOPS` switches the fixed % stop to a volatility-adaptive ATR stop.

## Tests

```bash
python -m pytest tests/ -q
```

Covers structure detection, risk sizing/caps, and the no-double-entry +
stop-resolution path. They run without torch, chronos, or network.
