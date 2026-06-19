# Crypto Agent — Full Audit (read-only, pre-implementation)

Date: 2026-06-19. Method: read every module, ran the unit tests, ran the
backtest on BTC/ETH/SOL/DOGE, ran a read-only per-candle decision tally, and ran
the real `main.py` entry point for two live cycles in DRY_RUN. Findings are
ranked by severity with evidence. No code was changed.

---

## The one-line conclusion

The dry-run pipeline runs, but the system is **mostly gated off and partly
unverified**. The multi-signal "pick the best-EV path" engine does not actually
decide — a hard structure gate overrides it ~60% of the time, the output
misreports why, the flagship ML signal (Chronos) is not running at all, and the
entire LIVE order path has never executed. We have been adding features faster
than we have been running them.

---

## P0 — breaks the core value or actively misleads

### P0-1. The structure gate neuters the whole ensemble
Evidence (live run, BTC/USDT):
```
EV(L/S) 1.717%/0.883%  conv 0.35  momentum +0.42  fear_greed +0.72 (bullish)
-> action FLAT, because structure = "consolidation"
```
`decision._allowed_directions` returns an empty set whenever structure is NEUTRAL
(and on spot, whenever it is BEARISH). With an empty set there are no candidates,
so the carefully computed EV and conviction are discarded. Across 300 assessed
candles per coin: ~46% NEUTRAL + ~17% BEARISH = **~63% of candles auto-FLAT
regardless of every other signal**. The "decide which path makes more money"
design is effectively disabled; structure alone decides.

### P0-2. The GO/NO-GO output is misleading
Same run: the engine logs `EV(L/S) 1.717%/0.883%` but the human line says
`FLAT chosen: EV +0.000%`. The reason string prints `best_ev` (which is 0 when
structure gated out all candidates), not the real `ev_long`/`ev_short`. So the
operator sees "+0.000%" and concludes there is no edge, when the model actually
saw +1.7%. There is also no explicit GO / NO-GO label and no plain-English reason
("consolidating", "bearish + shorts off", "EV below bar"). This is literally the
"not giving clear go/no-go" complaint.

### P0-3. Backtest data is silently capped at 500 candles
`fetch_ohlcv_paginated(total=N)` returns 500 for any N (verified: asked 1500 and
5000, both returned 500). Root cause: it breaks when `len(batch) < per_call`
(per_call=1000) but MEXC returns 500 rows per page, so it exits after the first
page. Every backtest therefore tests ~5 days of 15m data and produces 1-5
trades. **No backtest result so far is statistically meaningful.**

---

## P1 — headline features that are not actually working

### P1-1. The Chronos forecast never runs
On this machine torch/chronos are not installed, so `predict_bias` returns None
every cycle (`forecast unavailable, conf 0.00`). The flagship ML signal
contributes nothing. It has never executed end to end, anywhere, in this project.

### P1-2. Fear & Greed is not per-coin sentiment
It is a single global market number (F&G 14 for BTC, ETH, and SOL identically in
the live run). It applies the same tilt to every asset and has no historical
endpoint, so it contributes **nothing in backtests** and only a constant nudge
live. It is not the asset-specific sentiment the design implies.

### P1-3. News sentiment (CryptoBERT) is untested
Needs `transformers`; never executed. RSS parsing, headline filtering, and label
mapping have not been run once.

---

## P2 — the LIVE path is unverified and risky

### P2-1. No live order has ever been placed
`executor._enter_live`, `_attach_exits`, and `close_market` have never run. MEXC
spot very likely rejects `stop_market` / `take_profit_market` order types through
ccxt, in which case `_attach_exits` silently falls back to "loop manages exits" —
meaning the only stop protection is the 15-minute poll. Unverified and dangerous
for real money.

### P2-2. Decisions act on the forming candle
`strategy.run_symbol` uses `df.iloc[-1]`, the current in-progress candle, which
repaints. Live entries, exits, and structure all read incomplete data. Should use
the last *closed* candle (`iloc[-2]`).

### P2-3. Live exits are coarse and mis-recorded
Exits are only checked once per poll (15m); a stop can be blown through between
polls if native brackets failed. And `close_market` records the ledger at the
theoretical SL/TP level, not the actual fill price, so live PnL will drift from
reality.

---

## P3 — testing and coverage gaps

- **Backtest != live logic.** The backtest reimplements its own loop and does NOT
  use `strategy.run_symbol` or the guardrails (`max_open_positions`, cooldown,
  daily-loss). So it validates a different code path than what runs live.
- `strategy.run_symbol` and `_entry_allowed` have no integration test.
- Sentiment and forecast cannot be meaningfully exercised by the current backtest.

---

## What actually works (verified)

- `main.py` runs the integrated multi-symbol loop without crashing; graceful
  forecast fallback works; live Fear & Greed fetch works; clean shutdown works.
- The DRY_RUN sim pipeline (fetch -> assess -> size -> open -> resolve exits) runs.
- Risk sizing math, stop resolution, the no-double-entry rule, the live-exit
  ordering fix, and analytics are unit-tested (11/11 pass).

---

## Recommended fix order (proposed — not yet implemented)

1. **P0-3 pagination** — get real data volume first; nothing can be judged until
   backtests pull thousands of candles.
2. **P0-2 clear GO/NO-GO output** — report the real EV, add an explicit verdict
   and plain reason in both the loop and a backtest signal summary.
3. **P0-1 structure gate** — make structure a *weighted signal* (let the EV engine
   decide) instead of a hard veto, OR enable futures shorts. This is the biggest
   lever on whether the ensemble actually functions.
4. **P2-2 use closed candles** — correctness fix for live.
5. **P1-1 forecast** — install torch+chronos and verify Chronos runs once.
6. **P2-1 live path** — paper-trade a single tiny live order to verify MEXC order
   types and bracket support before trusting it.
7. **P3 align backtest with live** — route the backtest through `strategy` so we
   test the real path.

Decision needed before any code: do we keep the structure hard-gate (safer, fewer
trades) or let the EV ensemble decide (more trades, what the design intends)?
