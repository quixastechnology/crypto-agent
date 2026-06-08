"""Structure provider: wraps the price-action swing parser as a signal."""

from __future__ import annotations

from core.signals.base import MarketContext, SignalProvider, SignalScore
from core.structure import Structure


class StructureSignal(SignalProvider):
    name = "structure"
    weight_key = "weight_structure"

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        state = ctx.structure.state
        if state == Structure.BULLISH:
            return SignalScore(self.name, 1.0, 0.9, "HH + HL uptrend")
        if state == Structure.BEARISH:
            return SignalScore(self.name, -1.0, 0.9, "LH + LL downtrend")
        return SignalScore(self.name, 0.0, 0.3, "consolidation / no clear structure")
