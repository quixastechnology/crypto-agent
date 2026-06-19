"""Signal/advisory service: analyse a pair and emit a trade alert.

This is the no-execution mode. It runs the exact same assessment engine the
trading bot uses, then formats a full setup (pair, direction, entry, stop,
target, conviction, EV, and the per-signal reasoning) and sends it via the
Notifier. It never places an order. You act on the alert yourself.

De-duplication: it only alerts when the call for a symbol *changes* (e.g. flips
to GO LONG, or a previous GO turns to NO-GO), so you are not pinged with the
same setup every cycle. Set SIGNAL_REPEAT_MINUTES > 0 to also re-send an active
GO on an interval.
"""

from __future__ import annotations

import logging

from config.settings import Config
from core.decision import DecisionEngine
from core.forecast import ForecastEngine
from core.market_data import MarketData
from core.notifier import Notifier
from core.risk import RiskManager
from core.signals.base import MarketContext
from core.structure import extract_market_structure

logger = logging.getLogger(__name__)

_ARROW = {"LONG": "🟢 GO LONG", "SHORT": "🔴 GO SHORT", "FLAT": "⚪ NO-GO"}


def _fmt_price(p: float) -> str:
    """Readable price: more decimals for cheap coins, fewer for expensive ones."""
    decimals = 2 if p >= 100 else 4 if p >= 1 else 6
    return f"{p:,.{decimals}f}"


class SignalService:
    def __init__(
        self,
        config: Config,
        market: MarketData,
        forecast: ForecastEngine,
        risk: RiskManager,
        engine: DecisionEngine,
        notifier: Notifier,
    ) -> None:
        self.config = config
        self.market = market
        self.forecast = forecast
        self.risk = risk
        self.engine = engine
        self.notifier = notifier
        self._last_action: dict[str, str] = {}
        self._last_sent_ms: dict[str, int] = {}

    def _assess(self, symbol: str):
        """Run the engine on the last closed candle. Returns (ts, price, Assessment) or None."""
        df = self.market.fetch_ohlcv(symbol)
        if len(df) < self.config.structure_window * 2 + 3:
            return None
        # Decide on the last CLOSED candle.
        df = df.iloc[:-1].reset_index(drop=True)
        last = df.iloc[-1]
        ts = int(last["timestamp"])
        price = float(last["close"])
        structure = extract_market_structure(df, self.config.structure_window)
        forecast = self.forecast.predict_bias(df) if self.config.weight_forecast > 0 else None
        ctx = MarketContext(symbol=symbol, df=df, price=price, config=self.config,
                            structure=structure, forecast=forecast)
        return ts, price, self.engine.assess(ctx)

    def run_symbol(self, symbol: str) -> None:
        res = self._assess(symbol)
        if res is None:
            return
        ts, price, assessment = res
        if self._should_alert(symbol, assessment.action, ts):
            self.notifier.send(self._format(symbol, price, assessment))
            self._last_action[symbol] = assessment.action
            self._last_sent_ms[symbol] = ts

    def build_digest(self, symbols: list[str], date_label: str = "") -> str:
        """Assess every symbol once and return a single combined daily message."""
        header = f"📊 Crypto Signals {date_label} ({self.config.timeframe})".rstrip()
        gos: list[str] = []
        nogos: list[str] = []
        for symbol in symbols:
            res = self._assess(symbol)
            if res is None:
                continue
            _, price, a = res
            if a.action in ("LONG", "SHORT"):
                side = "buy" if a.action == "LONG" else "sell"
                stop, target = self.risk.bracket_levels(side, price)
                sp = (stop - price) / price * 100
                tp = (target - price) / price * 100
                gos.append(
                    f"{_ARROW[a.action]}  {symbol}\n"
                    f"   entry {_fmt_price(price)} | stop {_fmt_price(stop)} ({sp:+.1f}%) | "
                    f"tgt {_fmt_price(target)} ({tp:+.1f}%) | conv {a.conviction:.2f}"
                )
            else:
                nogos.append(f"⚪ NO-GO  {symbol}")
        body = gos + ([""] + nogos if nogos else [])
        if not gos and not nogos:
            body = ["No data available."]
        return header + "\n\n" + "\n".join(body) + "\n\nNot financial advice. You place the trade."

    def _should_alert(self, symbol: str, action: str, ts: int) -> bool:
        prev = self._last_action.get(symbol)
        changed = action != prev

        if self.config.signal_alert_on == "go":
            # Only ping for actionable setups, or when an active GO turns off.
            actionable = action in ("LONG", "SHORT")
            went_flat = action == "FLAT" and prev in ("LONG", "SHORT")
            if not (actionable or went_flat):
                # still record so the first NO-GO baseline doesn't spam later
                self._last_action.setdefault(symbol, action)
                return False
        # "all" mode alerts on any change.

        if changed:
            return True
        # Same call as before: re-alert only if a repeat interval is set.
        if self.config.signal_repeat_minutes > 0 and action in ("LONG", "SHORT"):
            elapsed_min = (ts - self._last_sent_ms.get(symbol, 0)) / 60000
            return elapsed_min >= self.config.signal_repeat_minutes
        return False

    def _format(self, symbol: str, price: float, a) -> str:
        tf = self.config.timeframe
        header = _ARROW.get(a.action, a.action)
        lines = [f"{header} — {symbol} ({tf})"]

        if a.action in ("LONG", "SHORT"):
            side = "buy" if a.action == "LONG" else "sell"
            stop, target = self.risk.bracket_levels(side, price)
            stop_pct = (stop - price) / price * 100
            tp_pct = (target - price) / price * 100
            lines += [
                f"Entry:  {_fmt_price(price)}",
                f"Stop:   {_fmt_price(stop)} ({stop_pct:+.2f}%)",
                f"Target: {_fmt_price(target)} ({tp_pct:+.2f}%)  R:R {self.config.reward_risk_ratio:g}",
                f"Conviction {a.conviction:.2f} | EV "
                f"{(a.ev_long if a.action == 'LONG' else a.ev_short) * 100:+.2f}% | p_up {a.p_up:.2f}",
            ]
        else:
            lines.append(a.reason)

        lines.append("Why:")
        for s in a.breakdown:
            if s.confidence > 0:
                lines.append(f" • {s.name}: {s.rationale} ({s.score:+.2f})")
        lines.append("Not financial advice. You place the trade.")
        return "\n".join(lines)
