"""Signal provider interface.

Every analysis source (structure, forecast, momentum, sentiment, volatility)
implements `SignalProvider.evaluate(context)` and returns a `SignalScore`:

  - score:      directional view in [-1, +1]  (-1 = strong short, +1 = strong long)
  - confidence: how sure this provider is, in [0, 1]
  - veto:       hard "do not trade" flag (e.g. regime too chaotic)
  - rationale:  short human-readable explanation, stored for transparency

The DecisionEngine blends these into one expected-value decision. Providers must
be failsafe: any error should degrade to a neutral, zero-confidence score rather
than raise, so one bad data source never stops the bot.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config.settings import Config
from core.forecast import ForecastResult
from core.structure import StructureState


@dataclass
class MarketContext:
    """Everything a provider may need for one symbol on one cycle."""

    symbol: str
    df: pd.DataFrame
    price: float
    config: Config
    structure: StructureState
    forecast: ForecastResult | None = None


@dataclass
class SignalScore:
    name: str
    score: float            # [-1, +1]
    confidence: float       # [0, 1]
    rationale: str = ""
    veto: bool = False

    def clamped(self) -> "SignalScore":
        self.score = max(-1.0, min(1.0, self.score))
        self.confidence = max(0.0, min(1.0, self.confidence))
        return self


class SignalProvider:
    """Base class. Subclasses set `name` and implement `_evaluate`."""

    name: str = "base"
    weight_key: str = ""     # which config weight applies to this provider

    def evaluate(self, ctx: MarketContext) -> SignalScore:
        try:
            return self._evaluate(ctx).clamped()
        except Exception as exc:  # never let a provider crash the loop
            return SignalScore(self.name, 0.0, 0.0, rationale=f"error: {exc}")

    def _evaluate(self, ctx: MarketContext) -> SignalScore:  # pragma: no cover
        raise NotImplementedError
