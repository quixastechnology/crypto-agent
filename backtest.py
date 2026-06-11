"""Historical backtester.

Replays past candles through the exact structure + forecast + risk + portfolio
stack used live, so you can validate the edge (after fees) in seconds instead of
waiting weeks of real-time DRY_RUN. Uses a separate SQLite file so it never
pollutes the live ledger.

Usage:
    python backtest.py BTC/USDT 1500
"""

from __future__ import annotations

import logging
import os
import sys

from analytics import compute_metrics, print_report
from config.settings import load_config
from core.strategy import _atr
from core.decision import DecisionEngine
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.signals import build_providers
from core.signals.base import MarketContext
from core.structure import extract_market_structure

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backtest")
log.setLevel(logging.INFO)


def run(symbol: str, bars: int) -> None:
    config = load_config()
    config.trade_mode = "DRY_RUN"
    config.db_path = "backtest.db"
    # Fresh ledger every run. The old version reloaded the last equity row and
    # summed ALL historical trades, so a second run started from the previous
    # run's equity and polluted every stat. Backtests must be reproducible.
    if os.path.exists(config.db_path):
        os.remove(config.db_path)

    market = MarketData(config)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    portfolio = Portfolio(config)
    engine = DecisionEngine(config, build_providers(config))

    df = market.fetch_ohlcv_paginated(symbol, total=bars)
    window = config.context_length
    cost_rate = config.taker_fee + config.slippage

    log.info("Backtesting %s over %d candles (window=%d)...", symbol, len(df), window)

    for i in range(window, len(df)):
        slice_df = df.iloc[i - window : i + 1].reset_index(drop=True)
        last = slice_df.iloc[-1]
        ts = int(last["timestamp"])
        price = float(last["close"])

        if portfolio.has_position(symbol):
            portfolio.check_exits(symbol, float(last["high"]), float(last["low"]), ts)
            continue

        structure = extract_market_structure(slice_df, config.structure_window)
        fc = forecast.predict_bias(slice_df) if config.weight_forecast > 0 else None
        ctx = MarketContext(symbol=symbol, df=slice_df, price=price,
                            config=config, structure=structure, forecast=fc)
        assessment = engine.assess(ctx)
        side = {"LONG": "buy", "SHORT": "sell"}.get(assessment.action)
        if side is None:
            continue

        atr = _atr(slice_df, config.atr_period) if config.use_atr_stops else None
        bracket = risk.size_position(side, price, portfolio.equity, atr)
        if bracket is None:
            continue
        fill = price * (1 + config.slippage) if side == "buy" else price * (1 - config.slippage)
        bracket.entry = fill
        bracket.stop_loss, bracket.take_profit = risk.bracket_levels(side, fill, atr)
        portfolio.open_position(bracket, symbol, ts)

    portfolio.close()
    log.info("Round-trip cost assumption: ~%.3f%%", cost_rate * 2 * 100)
    metrics = compute_metrics(config.db_path, initial_equity=config.initial_equity)
    print_report(metrics)


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(sym, n)
