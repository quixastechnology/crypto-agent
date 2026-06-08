"""Chronos zero-shot directional bias.

Wraps Amazon's Chronos time-series foundation model. The pipeline is loaded
lazily on first use so the rest of the system (structure parsing, simulation,
tests) can run without torch/chronos installed.

Note on horizon: Chronos adds little over a naive forecast at very short
horizons, where price is close to a random walk. Use this layer for
higher-timeframe directional *bias*, not for tick-level entry timing. The
structure parser handles timing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Bias(str, Enum):
    BULLISH = "BULLISH_BIAS"
    BEARISH = "BEARISH_BIAS"
    NEUTRAL = "NEUTRAL_BIAS"


@dataclass
class ForecastResult:
    bias: Bias
    current_price: float
    predicted_price: float


class ForecastEngine:
    """Lazy-loaded Chronos forecaster."""

    def __init__(self, model_name: str, horizon: int, alpha: float) -> None:
        self.model_name = model_name
        self.horizon = horizon
        self.alpha = alpha
        self._pipeline = None

    def _ensure_pipeline(self) -> None:
        if self._pipeline is not None:
            return
        import torch  # imported here so the dependency is optional
        from chronos import ChronosPipeline

        use_cuda = torch.cuda.is_available()
        logger.info("Loading Chronos model %s (cuda=%s)", self.model_name, use_cuda)
        self._pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map="cuda" if use_cuda else "cpu",
            # bfloat16 cannot be converted to numpy; keep float32 so the
            # quantile extraction below never crashes on the GPU path.
            torch_dtype=torch.float32,
        )

    def predict_bias(self, df: pd.DataFrame) -> ForecastResult:
        """Forecast the median terminal price and derive a directional bias."""
        import torch

        self._ensure_pipeline()
        context = torch.tensor(df["close"].values, dtype=torch.float32)
        forecast = self._pipeline.predict(context, self.horizon)

        # forecast shape: [num_series, num_samples, horizon].
        samples = forecast[0].float().cpu().numpy()
        median_path = np.quantile(samples, 0.5, axis=0)
        predicted = float(median_path[-1])
        current = float(df["close"].iloc[-1])

        if predicted > current * (1.0 + self.alpha):
            bias = Bias.BULLISH
        elif predicted < current * (1.0 - self.alpha):
            bias = Bias.BEARISH
        else:
            bias = Bias.NEUTRAL

        return ForecastResult(bias=bias, current_price=current, predicted_price=predicted)
