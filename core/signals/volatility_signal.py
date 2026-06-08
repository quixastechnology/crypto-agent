"""Volatility regime provider: ATR-based capital-preservation veto.

This provider is directionally neutral. Its job is to detect when volatility is
too high to trade safely (whipsaw / chaotic regime) and raise a veto, putting
the agent into the No-Trade / capital-preservation state your spec calls for.
"""

from __future__ import annotations

import pandas as pd

from core.signals.base import MarketContext, SignalProvider, SignalScore


def _atr_pct(df: pd.DataFrame, period: int) -> float:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(atr) / float(close.iloc[-1])


class VolatilitySignal(SignalProvider):
    name = "volatility"
    weight_key = "weight_volatility"

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        cfg = ctx.config
        atr_pct = _atr_pct(ctx.df, cfg.atr_period)
        if atr_pct > cfg.atr_max_pct:
            return SignalScore(
                self.name, 0.0, 1.0, veto=True,
                rationale=f"ATR {atr_pct * 100:.2f}% > max {cfg.atr_max_pct * 100:.2f}% -> no trade",
            )
        # Calmer markets get slightly higher confidence weighting downstream.
        confidence = max(0.0, 1.0 - atr_pct / cfg.atr_max_pct)
        return SignalScore(self.name, 0.0, confidence, f"ATR {atr_pct * 100:.2f}% (ok)")
