"""Decision engine: assess all signals, then pick the highest-EV path.

This is the "decide which path makes more money" core. It:

  1. Runs every enabled signal provider (the assessment phase).
  2. Blends their scores into one composite directional view, weighted by both
     the configured provider weight and the provider's own confidence.
  3. Converts the composite into a directional probability and computes the
     expected value of going long vs short, net of round-trip costs.
  4. Returns the better path only if it clears the minimum EV and conviction,
     respects any veto, and (by default) aligns with market structure.

Expected value is in "fraction of price" units:
    win  = stop_loss_pct * reward_risk_ratio
    loss = stop_loss_pct
    cost = 2 * (taker_fee + slippage)
    EV_long  = p_up * win - (1 - p_up) * loss - cost
    EV_short = (1 - p_up) * win - p_up * loss - cost
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config.settings import Config
from core.signals.base import MarketContext, SignalProvider, SignalScore
from core.structure import Structure

logger = logging.getLogger(__name__)


@dataclass
class Assessment:
    action: str                       # "LONG", "SHORT", or "FLAT"
    composite: float                  # [-1, +1]
    p_up: float                       # [0, 1]
    conviction: float                 # [0, 1]
    ev_long: float
    ev_short: float
    reason: str
    breakdown: list[SignalScore] = field(default_factory=list)


class DecisionEngine:
    def __init__(self, config: Config, providers: list[SignalProvider]) -> None:
        self.config = config
        self.providers = providers

    def _weight(self, provider: SignalProvider) -> float:
        return float(getattr(self.config, provider.weight_key, 0.0)) if provider.weight_key else 0.0

    def assess(self, ctx: MarketContext) -> Assessment:
        scores = [p.evaluate(ctx) for p in self.providers]

        # Hard veto wins immediately (capital preservation).
        vetoed = [s for s in scores if s.veto]
        if vetoed:
            reason = "; ".join(s.rationale for s in vetoed)
            return Assessment("FLAT", 0.0, 0.5, 0.0, 0.0, 0.0, f"NO-GO (veto): {reason}", scores)

        # Weighted, confidence-scaled blend.
        num = 0.0
        denom = 0.0
        conf_sum = 0.0
        conf_wsum = 0.0
        for provider, s in zip(self.providers, scores):
            w = self._weight(provider) * s.confidence
            num += w * s.score
            denom += self._weight(provider)
            conf_sum += self._weight(provider) * s.confidence
            conf_wsum += self._weight(provider)
        composite = (num / denom) if denom > 0 else 0.0
        conviction = (conf_sum / conf_wsum) if conf_wsum > 0 else 0.0

        p_up = max(0.05, min(0.95, 0.5 + 0.5 * composite))

        win = self.config.stop_loss_pct * self.config.reward_risk_ratio
        loss = self.config.stop_loss_pct
        cost = 2 * (self.config.taker_fee + self.config.slippage)
        ev_long = p_up * win - (1 - p_up) * loss - cost
        ev_short = (1 - p_up) * win - p_up * loss - cost

        # Directions allowed (structure gate is optional; by default the EV
        # ensemble decides and structure only contributes via its weight).
        allowed = self._allowed_directions(ctx)

        candidates = []
        if "LONG" in allowed:
            candidates.append(("LONG", ev_long))
        if "SHORT" in allowed:
            candidates.append(("SHORT", ev_short))

        # Pick the highest-EV direction available, then apply the GO bar.
        if not candidates:
            action, best_ev = "FLAT", 0.0
            reason = "NO-GO: no direction allowed (structure gate blocked all paths)"
        else:
            best_dir, best_ev = max(candidates, key=lambda c: c[1])
            if best_ev < self.config.min_expected_value:
                action = "FLAT"
                reason = (f"NO-GO: best path {best_dir} EV {best_ev * 100:+.3f}% "
                          f"below min {self.config.min_expected_value * 100:.3f}%")
            elif conviction < self.config.min_conviction:
                action = "FLAT"
                reason = (f"NO-GO: {best_dir} EV {best_ev * 100:+.3f}% ok, but conviction "
                          f"{conviction:.2f} below min {self.config.min_conviction:.2f}")
            else:
                action = best_dir
                reason = (f"GO {best_dir}: EV {best_ev * 100:+.3f}%, conviction {conviction:.2f}, "
                          f"p_up {p_up:.2f}")

        logger.info("assessment %s | composite %.2f p_up %.2f conv %.2f EV(L/S) %.3f%%/%.3f%% | %s",
                    action, composite, p_up, conviction, ev_long * 100, ev_short * 100, reason)
        return Assessment(action, composite, p_up, conviction, ev_long, ev_short, reason, scores)

    def _allowed_directions(self, ctx: MarketContext) -> set[str]:
        if not self.config.require_structure_alignment:
            allowed = {"LONG"}
            if self.config.allow_shorts:
                allowed.add("SHORT")
            return allowed

        state = ctx.structure.state
        if state == Structure.BULLISH:
            return {"LONG"}
        if state == Structure.BEARISH:
            return {"SHORT"} if self.config.allow_shorts else set()
        return set()
