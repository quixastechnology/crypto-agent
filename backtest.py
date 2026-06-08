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
import sys

from config.settings import load_config
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.structure import Structure, extract_market_structure
from core.forecast import Bias

logging.basicConfig(level=logging.WARNING, format="%(message)s")
log = logging.getLogger("backtest")
log.setLevel(logging.INFO)


def run(symbol: str, bars: int) -> None:
    config = load_config()
    config.trade_mode = "DRY_RUN"
    config.db_path = "backtest.db"

    market = MarketData(config)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    portfolio = Portfolio(config)

    df = market.fetch_ohlcv(symbol, limit=bars)
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

        structure = extract_market_structure(slice_df, config.structure_window).state
        fc = forecast.predict_bias(slice_df)

        side = None
        if structure == Structure.BULLISH and fc.bias == Bias.BULLISH:
            side = "buy"
        elif structure == Structure.BEARISH and fc.bias == Bias.BEARISH and config.allow_shorts:
            side = "sell"
        if side is None:
            continue

        bracket = risk.size_position(side, price, portfolio.equity)
        if bracket is None:
            continue
        fill = price * (1 + config.slippage) if side == "buy" else price * (1 - config.slippage)
        bracket.entry = fill
        bracket.stop_loss, bracket.take_profit = risk.bracket_levels(side, fill)
        portfolio.open_position(bracket, symbol, ts)

    stats = portfolio.stats()
    start_eq = config.initial_equity
    ret = (stats["equity"] - start_eq) / start_eq * 100
    log.info("-" * 50)
    log.info("Trades: %d | Win rate: %.1f%% | Net PnL: %.4f USDT",
             stats["trades"], stats["win_rate"] * 100, stats["net_pnl"])
    log.info("Equity: %.4f -> %.4f (%.2f%%) | round-trip cost ~%.3f%%",
             start_eq, stats["equity"], ret, cost_rate * 2 * 100)
    portfolio.close()


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "BTC/USDT"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    run(sym, n)
