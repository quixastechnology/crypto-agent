"""Price-action market structure parser.

Detects confirmed swing highs and lows using a symmetric fractal window, then
classifies the trend as bullish (HH + HL), bearish (LH + LL), or consolidating.

A swing at index i is only *confirmed* once `window` bars exist on both sides of
it. We never act on an unconfirmed swing, so there is no lookahead leak into the
trading decision: the most recent confirmed swing is always at least `window`
bars in the past, which is the honest, lagging nature of structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Structure(str, Enum):
    BULLISH = "BULLISH_STRUCTURE"
    BEARISH = "BEARISH_STRUCTURE"
    NEUTRAL = "CONSOLIDATION_NEUTRAL"


@dataclass
class StructureState:
    state: Structure
    last_high: float | None = None
    prev_high: float | None = None
    last_low: float | None = None
    prev_low: float | None = None


def _confirmed_swings(values: pd.Series, window: int, kind: str) -> list[float]:
    """Return confirmed swing values in chronological order.

    A point is a swing high if it is the strict maximum of the window on each
    side; a swing low if it is the strict minimum. Only points with a full
    window on both sides are evaluated, so the result excludes the last `window`
    bars (which cannot yet be confirmed).
    """
    n = len(values)
    swings: list[float] = []
    for i in range(window, n - window):
        center = values.iloc[i]
        left = values.iloc[i - window : i]
        right = values.iloc[i + 1 : i + 1 + window]
        if kind == "high":
            if center >= left.max() and center > right.max():
                swings.append(float(center))
        else:  # low
            if center <= left.min() and center < right.min():
                swings.append(float(center))
    return swings


def extract_market_structure(df: pd.DataFrame, window: int = 5) -> StructureState:
    """Classify the trend from the two most recent confirmed swings."""
    highs = _confirmed_swings(df["high"], window, "high")
    lows = _confirmed_swings(df["low"], window, "low")

    if len(highs) < 2 or len(lows) < 2:
        return StructureState(Structure.NEUTRAL)

    last_high, prev_high = highs[-1], highs[-2]
    last_low, prev_low = lows[-1], lows[-2]

    if last_high > prev_high and last_low > prev_low:
        state = Structure.BULLISH
    elif last_high < prev_high and last_low < prev_low:
        state = Structure.BEARISH
    else:
        state = Structure.NEUTRAL

    return StructureState(state, last_high, prev_high, last_low, prev_low)
