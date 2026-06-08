"""Risk manager: position sizing and bracket levels.

Sizing is driven by the stop distance, not the raw budget. The amount of equity
put at risk on a single trade is fixed (`risk_per_trade`), and the position size
is solved from it:

    risk_amount   = equity * risk_per_trade
    stop_distance = entry * stop_loss_pct
    notional      = risk_amount / stop_loss_pct     (= risk_amount * entry / stop_distance)
    quantity      = notional / entry

The notional is then capped by available budget (spot) or budget * leverage
(futures), and validated against the market's minimum notional and precision.
"""

from __future__ import annotations

from dataclasses import dataclass

from config.settings import Config
from core.structure import Structure


@dataclass
class Bracket:
    side: str          # "buy" or "sell"
    entry: float
    stop_loss: float
    take_profit: float
    quantity: float
    notional: float


class RiskManager:
    def __init__(self, config: Config) -> None:
        self.config = config

    def bracket_levels(self, side: str, entry: float) -> tuple[float, float]:
        """Return (stop_loss, take_profit) for a long or short entry."""
        stop_dist = entry * self.config.stop_loss_pct
        tp_dist = stop_dist * self.config.reward_risk_ratio
        if side == "buy":
            return entry - stop_dist, entry + tp_dist
        return entry + stop_dist, entry - tp_dist

    def size_position(self, side: str, entry: float, equity: float) -> Bracket | None:
        """Compute a fully sized bracket, or None if it cannot be placed."""
        risk_amount = equity * self.config.risk_per_trade
        # notional such that a full stop loses exactly risk_amount
        notional = risk_amount / self.config.stop_loss_pct

        # Cap by deployable capital.
        max_notional = min(self.config.max_budget, equity)
        if self.config.market_type == "futures":
            max_notional = min(self.config.max_budget, equity) * max(1, self.config.leverage)
        notional = min(notional, max_notional)

        if notional <= 0 or entry <= 0:
            return None

        quantity = notional / entry
        stop_loss, take_profit = self.bracket_levels(side, entry)
        return Bracket(
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            quantity=quantity,
            notional=notional,
        )

    @staticmethod
    def signal_to_side(structure: Structure, allow_shorts: bool) -> str | None:
        """Map a confirmed structure state to an order side, honouring shorts."""
        if structure == Structure.BULLISH:
            return "buy"
        if structure == Structure.BEARISH:
            return "sell" if allow_shorts else None
        return None
