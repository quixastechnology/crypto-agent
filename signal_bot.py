"""Signal bot entry point — analysis and alerts only, never trades.

Runs the same multi-signal engine as the trading bot, but instead of placing
orders it sends you a full trade setup (pair, direction, entry, stop, target,
conviction, reasoning) via Telegram (and the console). No MEXC API key needed:
all market data comes from MEXC's public endpoints.

    python signal_bot.py
"""

from __future__ import annotations

import logging
import signal
import sys
import time

from config.settings import load_config
from core.decision import DecisionEngine
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.notifier import Notifier
from core.risk import RiskManager
from core.signaler import SignalService
from core.signals import build_providers

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("signal-bot")

_running = True


def _handle_stop(signum, frame) -> None:
    global _running
    logger.info("Shutdown signal received, finishing current cycle...")
    _running = False


def main() -> int:
    config = load_config()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    market = MarketData(config)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    engine = DecisionEngine(config, build_providers(config))
    notifier = Notifier(config)
    service = SignalService(config, market, forecast, risk, engine, notifier)

    logger.info("=" * 60)
    logger.info("CRYPTO SIGNAL BOT (advisory only, no trading)")
    logger.info("symbols=%s tf=%s signals=%s telegram=%s",
                config.symbols, config.timeframe, [p.name for p in engine.providers],
                notifier.telegram_enabled)
    logger.info("=" * 60)

    try:
        while _running:
            for symbol in config.symbols:
                try:
                    service.run_symbol(symbol)
                except Exception as exc:
                    logger.exception("Error on %s: %s", symbol, exc)
            if not _running:
                break
            time.sleep(config.poll_seconds)
    finally:
        logger.info("Signal bot stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
