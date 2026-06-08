"""Forecast provider: turns the Chronos median path into a directional score.

Score scales with how far the predicted move exceeds the alpha buffer, so a
forecast that barely clears the threshold contributes weakly and a large
predicted move contributes strongly.
"""

from __future__ import annotations

from core.forecast import Bias
from core.signals.base import MarketContext, SignalProvider, SignalScore


class ForecastSignal(SignalProvider):
    name = "forecast"
    weight_key = "weight_forecast"

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        fc = ctx.forecast
        if fc is None:
            return SignalScore(self.name, 0.0, 0.0, "forecast unavailable")

        move = (fc.predicted_price - fc.current_price) / fc.current_price
        alpha = ctx.config.forecast_alpha or 0.005
        # Normalise the move against a few multiples of the buffer.
        score = max(-1.0, min(1.0, move / (alpha * 3)))
        confidence = min(1.0, abs(move) / (alpha * 3))

        if fc.bias == Bias.NEUTRAL:
            confidence *= 0.3
        return SignalScore(
            self.name, score, confidence,
            f"pred {move * 100:+.2f}% over horizon ({fc.bias.value})",
        )
