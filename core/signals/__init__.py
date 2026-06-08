"""Signal providers package and factory."""

from __future__ import annotations

from config.settings import Config
from core.signals.base import SignalProvider
from core.signals.forecast_signal import ForecastSignal
from core.signals.momentum_signal import MomentumSignal
from core.signals.sentiment_signal import FearGreedSignal, NewsSentimentSignal
from core.signals.structure_signal import StructureSignal
from core.signals.volatility_signal import VolatilitySignal


def build_providers(config: Config) -> list[SignalProvider]:
    """Assemble the active signal providers based on config weights.

    A provider with weight 0 is dropped so it costs nothing to evaluate. News
    sentiment is only added when explicitly enabled (it pulls a HF model).
    """
    candidates: list[SignalProvider] = [
        StructureSignal(),
        ForecastSignal(),
        MomentumSignal(),
        VolatilitySignal(),
        FearGreedSignal(),
    ]
    if config.enable_news_sentiment and config.weight_news > 0:
        candidates.append(NewsSentimentSignal())

    # Keep volatility even at weight 0: it only votes via veto, not weight.
    active = [p for p in candidates if getattr(config, p.weight_key, 0.0) > 0 or p.name == "volatility"]
    return active
