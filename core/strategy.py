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
        if len(df) < self.config.structure_window * 2 + 3:
            logger.info("%s: not enough candles yet", symbol)
            return

        # Decide on the last CLOSED candle, not the in-progress one (which
        # repaints). Drop the final forming candle before any analysis.
        df = df.iloc[:-1].reset_index(drop=True)
        last = df.iloc[-1]
        ts = int(last["timestamp"])
        price = float(last["close"])
        high, low = float(last["high"]), float(last["low"])

        # 1. Manage an existing position first.
        #    Order matters in LIVE: send the exchange order BEFORE touching the
        #    ledger. (The old flow closed the ledger first, so close_market
        #    found no position and the exchange leg was never closed.)
        if self.portfolio.has_position(symbol):
            hit = self.portfolio.peek_exits(symbol, high, low)
            if hit:
                reason, exit_price = hit
                if self.config.trade_mode == "LIVE":
                    self.executor.close_market(symbol, exit_price, reason, ts)
                else:
                    self.portfolio.close_position(symbol, exit_price, reason, ts)
            else:
                logger.info("%s: holding position, no new entry", symbol)
            return

        # 2. Entry guardrails (capital preservation before any assessment).
        if not self._entry_allowed(symbol, ts):
            return

        # 3. Assess: structure (cheap) + forecast (Chronos) feed the engine.
        structure = extract_market_structure(df, self.config.structure_window)
        forecast = self.forecast.predict_bias(df) if self.config.weight_forecast > 0 else None
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

        atr = _atr(df, self.config.atr_period) if self.config.use_atr_stops else None
        bracket = self.risk.size_position(side, price, self.portfolio.equity, atr)
        if bracket is None:
            logger.warning("%s: could not size position", symbol)
            return

        self.executor.enter(symbol, bracket, ts)

    def _entry_allowed(self, symbol: str, ts: int) -> bool:
        """Portfolio-level gates that fire before any signal is computed."""
        cfg = self.config

        # Daily loss kill switch: stop opening new risk after a bad day.
        day_start = ts - (ts % 86_400_000)
        day_pnl = self.portfolio.realized_pnl_since(day_start)
        if day_pnl <= -abs(cfg.daily_max_loss_pct) * self.portfolio.equity:
            logger.warning("%s: daily loss limit hit (%.4f) — no new entries today", symbol, day_pnl)
            return False

        # Concurrency cap: crypto pairs are highly correlated; N open positions
        # is closer to one big position than N independent bets.
        if self.portfolio.open_positions_count() >= cfg.max_open_positions:
            logger.info("%s: max open positions (%d) reached", symbol, cfg.max_open_positions)
            return False

        # Cooldown after a stop-out: don't immediately re-enter the same chop.
        last_stop = self.portfolio.last_stop_ts(symbol)
        if last_stop is not None and ts - last_stop < cfg.cooldown_seconds * 1000:
            logger.info("%s: in post-stop cooldown", symbol)
            return False
        return True


def _atr(df, period: int) -> float:
    """ATR in price units over the dataframe (EWM, matching the volatility signal)."""
    import pandas as pd

    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return float(tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1])
