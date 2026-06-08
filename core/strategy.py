"""Strategy orchestration for one symbol per cycle.

Order of operations each cycle:
  1. Resolve any open position against the latest candle (stop/target).
  2. If flat, evaluate structure AND forecast. Trade only when both agree
     (the congruence gate). Bearish trades are taken only when shorting is
     enabled (futures); otherwise the bearish signal means "stay in cash".
"""

from __future__ import annotations

import logging

from config.settings import Config
from core.executor import Executor
from core.forecast import Bias, ForecastEngine
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.structure import Structure, extract_market_structure

logger = logging.getLogger(__name__)


class Strategy:
    def __init__(
        self,
        config: Config,
        market: MarketData,
        forecast: ForecastEngine,
        risk: RiskManager,
        portfolio: Portfolio,
        executor: Executor,
    ) -> None:
        self.config = config
        self.market = market
        self.forecast = forecast
        self.risk = risk
        self.portfolio = portfolio
        self.executor = executor

    def run_symbol(self, symbol: str) -> None:
        df = self.market.fetch_ohlcv(symbol)
        if len(df) < self.config.structure_window * 2 + 2:
            logger.info("%s: not enough candles yet", symbol)
            return

        last = df.iloc[-1]
        ts = int(last["timestamp"])
        price = float(last["close"])

        # 1. Manage an existing position first.
        if self.portfolio.has_position(symbol):
            reason = self.portfolio.check_exits(symbol, float(last["high"]), float(last["low"]), ts)
            if reason and self.config.trade_mode == "LIVE":
                # In live mode, mirror the simulated exit with a market close in
                # case native bracket orders are not supported by the venue.
                self.executor.close_market(symbol, price, reason, ts)
            logger.info("%s: holding position, no new entry", symbol)
            return

        # 2. Flat: evaluate signals.
        structure = extract_market_structure(df, self.config.structure_window).state
        fc = self.forecast.predict_bias(df)
        logger.info("%s: structure=%s bias=%s price=%.4f pred=%.4f",
                    symbol, structure.value, fc.bias.value, price, fc.predicted_price)

        side = self._congruent_side(structure, fc.bias)
        if side is None:
            logger.info("%s: no congruent signal, staying flat", symbol)
            return

        bracket = self.risk.size_position(side, price, self.portfolio.equity)
        if bracket is None:
            logger.warning("%s: could not size position", symbol)
            return

        self.executor.enter(symbol, bracket, ts)

    def _congruent_side(self, structure: Structure, bias: Bias) -> str | None:
        """Return an order side only when structure and forecast agree."""
        if structure == Structure.BULLISH and bias == Bias.BULLISH:
            return "buy"
        if structure == Structure.BEARISH and bias == Bias.BEARISH:
            return "sell" if self.config.allow_shorts else None
        return None
