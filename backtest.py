"""Historical backtester.

Drives the REAL `strategy.run_symbol` code path over historical candles via a
ReplayMarketData feed. This is the same logic that runs live (guardrails,
closed-candle handling, decision engine, sizing, executor, exit resolution), so
the backtest validates the actual system instead of a reimplementation.

Uses a separate SQLite file so it never pollutes the live ledger.

Usage:
    python backtest.py BTC/USDT 1500
"""

from __future__ import annotations

import logging
import os
import sys

from analytics import compute_metrics, print_report
from config.settings import load_config
from core.decision import DecisionEngine
from core.executor import Executor
from core.forecast import ForecastEngine
from core.market_data import MarketData, ReplayMarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.signals import build_providers
from core.strategy import Strategy

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backtest")
log.setLevel(logging.INFO)


def run(symbol: str, bars: int) -> None:
    config = load_config()
    config.trade_mode = "DRY_RUN"
    config.db_path = "backtest.db"
    # Fresh ledger every run so results are reproducible.
    if os.path.exists(config.db_path):
        os.remove(config.db_path)

    # Load the full history once with a real client, then replay it.
    loader = MarketData(config)
    loader.load_markets()
    full = loader.fetch_ohlcv_paginated(symbol, total=bars)

    market = ReplayMarketData(config, full)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    portfolio = Portfolio(config)
    executor = Executor(config, market, portfolio)
    engine = DecisionEngine(config, build_providers(config))
    strategy = Strategy(config, market, forecast, risk, portfolio, executor, engine)

    cost_rate = config.taker_fee + config.slippage
    log.info("Backtesting %s over %d candles via strategy.run_symbol (signals=%s)...",
             symbol, len(full), [p.name for p in engine.providers])

    # Step the cursor one candle at a time and run the real per-symbol logic.
    for cursor in range(config.context_length, len(full)):
        market.set_cursor(cursor)
        strategy.run_symbol(symbol)

    portfolio.close()
    log.info("Round-trip cost assumption: ~%.3f%%", cost_rate * 2 * 100)
    metrics = compute_metrics(config.db_path, initial_equity=config.initial_equity)
    print_report(metrics)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(sym, n)
