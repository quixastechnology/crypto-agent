"""Daily signal digest — one combined Telegram message for all pairs, then exits.

Designed to be run once per day from cron. Assesses every configured symbol and
sends a single summary (GO LONG/SHORT setups first, then NO-GO pairs). No
execution, no MEXC key. Pass a date label as argv[1] for the header.

    python signal_digest.py 2026-06-19
"""

from __future__ import annotations

import datetime
import logging
import sys

from config.settings import load_config
from core.decision import DecisionEngine
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.notifier import Notifier
from core.risk import RiskManager
from core.signaler import SignalService
from core.signals import build_providers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger("signal-digest")


def main() -> int:
    config = load_config()
    date_label = sys.argv[1] if len(sys.argv) > 1 else datetime.datetime.utcnow().strftime("%Y-%m-%d")

    market = MarketData(config)
    market.load_markets()
    forecast = ForecastEngine(config.chronos_model, config.forecast_horizon, config.forecast_alpha)
    risk = RiskManager(config)
    engine = DecisionEngine(config, build_providers(config))
    notifier = Notifier(config)
    service = SignalService(config, market, forecast, risk, engine, notifier)

    logger.info("Building daily digest for %s", config.symbols)
    digest = service.build_digest(config.symbols, date_label)
    notifier.send(digest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
