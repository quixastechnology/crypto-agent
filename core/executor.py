"""Order execution: dry-run simulation and live MEXC orders.

In DRY_RUN nothing is sent to the exchange; the entry is booked into the
Portfolio at a slippage-adjusted price and the simulator resolves the exit on
later candles.

In LIVE mode the market entry is placed, the quantity is rounded to the market's
precision and checked against minimum notional, and protective stop/target
orders are attached where the venue supports them. The position is always
recorded so the loop never double-enters.
"""

from __future__ import annotations

import logging

from config.settings import Config
from core.market_data import MarketData
from core.portfolio import Portfolio
from core.risk import Bracket

logger = logging.getLogger(__name__)


class Executor:
    def __init__(self, config: Config, market: MarketData, portfolio: Portfolio) -> None:
        self.config = config
        self.market = market
        self.portfolio = portfolio

    # --- helpers --------------------------------------------------------
    def _slip(self, side: str, price: float) -> float:
        """Apply assumed adverse slippage to a fill price."""
        s = self.config.slippage
        return price * (1 + s) if side == "buy" else price * (1 - s)

    def _validate(self, symbol: str, bracket: Bracket) -> bool:
        m = self.market.market(symbol)
        limits = m.get("limits", {})
        min_cost = (limits.get("cost") or {}).get("min")
        min_amount = (limits.get("amount") or {}).get("min")
        if min_cost and bracket.notional < min_cost:
            logger.warning("%s notional %.4f below min cost %.4f; skipping", symbol, bracket.notional, min_cost)
            return False
        if min_amount and bracket.quantity < min_amount:
            logger.warning("%s qty %.8f below min amount %.8f; skipping", symbol, bracket.quantity, min_amount)
            return False
        return True

    # --- entry ----------------------------------------------------------
    def enter(self, symbol: str, bracket: Bracket, ts: int) -> bool:
        if not self._validate(symbol, bracket):
            return False

        if self.config.trade_mode == "DRY_RUN":
            fill = self._slip(bracket.side, bracket.entry)
            simulated = Bracket(
                side=bracket.side, entry=fill,
                stop_loss=bracket.stop_loss, take_profit=bracket.take_profit,
                quantity=bracket.quantity, notional=bracket.notional,
            )
            self.portfolio.open_position(simulated, symbol, ts)
            return True

        return self._enter_live(symbol, bracket, ts)

    def _enter_live(self, symbol: str, bracket: Bracket, ts: int) -> bool:
        ex = self.market.exchange
        amount = float(ex.amount_to_precision(symbol, bracket.quantity))
        try:
            if self.config.market_type == "futures" and self.config.leverage > 1:
                try:
                    ex.set_leverage(self.config.leverage, symbol)
                except Exception as exc:  # not fatal; some accounts preset leverage
                    logger.warning("set_leverage failed for %s: %s", symbol, exc)

            order = ex.create_order(symbol=symbol, type="market", side=bracket.side, amount=amount)
            fill = float(order.get("average") or order.get("price") or bracket.entry)
            logger.info("LIVE entry %s %s amount=%s id=%s", bracket.side, symbol, amount, order.get("id"))

            booked = Bracket(
                side=bracket.side, entry=fill, stop_loss=bracket.stop_loss,
                take_profit=bracket.take_profit, quantity=amount, notional=bracket.notional,
            )
            self.portfolio.open_position(booked, symbol, ts)
            self._attach_exits(symbol, booked)
            return True
        except Exception as exc:
            logger.error("LIVE entry failed for %s: %s", symbol, exc)
            return False

    def _attach_exits(self, symbol: str, bracket: Bracket) -> None:
        """Attach protective stop and take-profit orders for a live position.

        MEXC spot supports stop_market but NOT take_profit_market (verified via
        ccxt.has), so the take-profit is placed as a plain limit order at the
        target price. Each leg is attached independently: if one order type is
        rejected, the other still stands and the poll loop manages the rest.
        """
        ex = self.market.exchange
        exit_side = "sell" if bracket.side == "buy" else "buy"
        amount = bracket.quantity
        reduce_only = self.config.market_type == "futures"

        # Stop loss (trigger order).
        try:
            ex.create_order(
                symbol=symbol, type="stop_market", side=exit_side, amount=amount,
                params={"stopPrice": bracket.stop_loss, "reduceOnly": reduce_only},
            )
            logger.info("Attached stop_market SL %.6f for %s", bracket.stop_loss, symbol)
        except Exception as exc:
            logger.warning("Could not attach stop for %s (%s); loop will manage exit", symbol, exc)

        # Take profit (limit order at target; take_profit_market is unsupported on MEXC spot).
        try:
            ex.create_order(
                symbol=symbol, type="limit", side=exit_side, amount=amount,
                price=bracket.take_profit, params={"reduceOnly": reduce_only},
            )
            logger.info("Attached limit TP %.6f for %s", bracket.take_profit, symbol)
        except Exception as exc:
            logger.warning("Could not attach take-profit for %s (%s); loop will manage exit", symbol, exc)

    # --- exit (live fallback) ------------------------------------------
    def close_market(self, symbol: str, price: float, reason: str, ts: int) -> None:
        """Close a live position at market (used when native brackets are unavailable)."""
        pos = self.portfolio.get_position(symbol)
        if pos is None:
            return
        if self.config.trade_mode == "LIVE":
            ex = self.market.exchange
            exit_side = "sell" if pos.side == "buy" else "buy"
            try:
                ex.create_order(symbol=symbol, type="market", side=exit_side, amount=pos.quantity,
                                params={"reduceOnly": self.config.market_type == "futures"})
            except Exception as exc:
                logger.error("LIVE exit failed for %s: %s", symbol, exc)
                return
        self.portfolio.close_position(symbol, price, reason, ts)
