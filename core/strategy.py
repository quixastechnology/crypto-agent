"""Strategy orchestration for one symbol per cycle.

Order of operations each cycle:
  1. Resolve any open position against the latest candle (stop/target).
  2. If flat, ASSESS: build the market context, run the decision engine over all
     signal providers, and act only on the chosen high-EV path (LONG/SHORT).
     FLAT means no path cleared the expected-value / conviction bar, or a veto
     fired (capital preservation).
"""

from __future__ import annotations

import logging

from config.settings import Config
from core.decision import DecisionEngine
from core.executor import Executor
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import RiskManager
from core.signals.base import MarketContext
from core.structure import extract_market_structure

logger = logging.getLogger(__name__)

_ACTION_TO_SIDE = {"LONG": "buy", "SHORT": "sell"}


class Strategy:
    def __init__(
        self,
        config: Config,
        market: MarketData,
        forecast: ForecastEngine,
        risk: RiskManager,
        portfolio: Portfolio,
        executor: Executor,
        engine: DecisionEngine,
    ) -> None:
        self.config = config
        self.market = market
        self.forecast = forecast
        self.risk = risk
        self.portfolio = portfolio
        self.executor = executor
        self.engine = engine

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
                self.executor.close_market(symbol, price, reason, ts)
            logger.info("%s: holding position, no new entry", symbol)
            return

        # 2. Assess: structure (cheap) + forecast (Chronos) feed the engine.
        structure = extract_market_structure(df, self.config.structure_window)
        forecast = self.forecast.predict_bias(df)
        ctx = MarketContext(
            symbol=symbol, df=df, price=price, config=self.config,
            structure=structure, forecast=forecast,
        )

        assessment = self.engine.assess(ctx)
        logger.info("%s: %s | %s", symbol, assessment.action, assessment.reason)
        for s in assessment.breakdown:
            logger.info("    - %-14s score=%+.2f conf=%.2f %s", s.name, s.score, s.confidence, s.rationale)

        side = _ACTION_TO_SIDE.get(assessment.action)
        if side is None:
            return

        bracket = self.risk.size_position(side, price, self.portfolio.equity)
        if bracket is None:
            logger.warning("%s: could not size position", symbol)
            return

        self.executor.enter(symbol, bracket, ts)
