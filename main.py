"""Entry point for the autonomous crypto trading agent.

Run modes are controlled entirely by .env (TRADE_MODE, MARKET_TYPE, SYMBOLS...).
Start in DRY_RUN, confirm the simulated equity curve is net-positive after fees,
then switch to LIVE with a withdrawal-disabled API key.
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from config.settings import load_config
from core.decision import DecisionEngine
from core.executor import Executor
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.signals import build_providers
from core.strategy import Strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("crypto-agent")

_running = True


def _handle_stop(signum, frame) -> None:
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


def build_strategy(config) -> tuple[Strategy, Portfolio, MarketData]:
    market = MarketData(config)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    portfolio = Portfolio(config)
    executor = Executor(config, market, portfolio)
    engine = DecisionEngine(config, build_providers(config))
    logger.info("Decision engine signals: %s", [p.name for p in engine.providers])
    strategy = Strategy(config, market, forecast, risk, portfolio, executor, engine)
    return strategy, portfolio, market


def main() -> int:
    config = load_config()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    logger.info("=" * 60)
    logger.info("AUTONOMOUS CRYPTO AGENT")
    logger.info("mode=%s market=%s shorts=%s symbols=%s tf=%s",
                config.trade_mode, config.market_type, config.allow_shorts,
                config.symbols, config.timeframe)
    logger.info("=" * 60)

    if config.trade_mode == "LIVE" and (not config.api_key or not config.api_secret):
        logger.error("LIVE mode requires MEXC_API_KEY and MEXC_API_SECRET")
        return 1

    strategy, portfolio, _ = build_strategy(config)

    try:
        while _running:
            for symbol in config.symbols:
                try:
                    strategy.run_symbol(symbol)
                except Exception as exc:  # one symbol failing must not kill the loop
                    logger.exception("Error processing %s: %s", symbol, exc)
            stats = portfolio.stats()
            logger.info("Cycle done | trades=%d win_rate=%.1f%% net_pnl=%.4f equity=%.4f",
                        stats["trades"], stats["win_rate"] * 100, stats["net_pnl"], stats["equity"])
            if not _running:
                break
            time.sleep(config.poll_seconds)
    finally:
        portfolio.close()
        logger.info("Stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
