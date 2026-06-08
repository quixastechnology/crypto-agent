"""Momentum provider: RSI + MACD, pure pandas (no extra ML, fast, free).

Combines two classic momentum reads:
  - MACD histogram sign gives the trend direction.
  - RSI distance from 50 gives strength, with overbought/oversold tempering.
"""

from __future__ import annotations

import pandas as pd

from core.signals.base import MarketContext, SignalProvider, SignalScore


def _rsi(close: pd.Series, period: int) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    rsi = 100 - 100 / (1 + rs)
    return float(rsi.iloc[-1])


def _macd_hist(close: pd.Series, fast: int, slow: int, signal: int) -> tuple[float, float]:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - signal_line
    # Normalise histogram by price so it is comparable across assets.
    return float(hist.iloc[-1]), float(close.iloc[-1])


class MomentumSignal(SignalProvider):
    name = "momentum"
    weight_key = "weight_momentum"

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        cfg = ctx.config
        close = ctx.df["close"]
        rsi = _rsi(close, cfg.rsi_period)
        hist, price = _macd_hist(close, cfg.macd_fast, cfg.macd_slow, cfg.macd_signal)

        rsi_norm = (rsi - 50) / 50                       # [-1, +1]
        hist_norm = max(-1.0, min(1.0, (hist / price) / 0.002))
        score = 0.5 * hist_norm + 0.5 * rsi_norm
        confidence = min(1.0, 0.5 * abs(hist_norm) + 0.5 * abs(rsi_norm))
        return SignalScore(
            self.name, score, confidence,
            f"RSI {rsi:.0f}, MACD-hist {hist / price * 100:+.3f}%",
        )
