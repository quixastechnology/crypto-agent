"""Lightweight sanity tests that run without torch/chronos or network.

Run: python -m pytest tests/ -q
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd

from config.settings import Config
from core.risk import RiskManager
from core.structure import Structure, extract_market_structure
from core.portfolio import Portfolio
from core.risk import Bracket
from core.signals.base import SignalScore


def _candles(highs, lows, closes):
    n = len(closes)
    return pd.DataFrame(
        {
            "timestamp": list(range(n)),
            "open": closes,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [1.0] * n,
        }
    )


def test_bullish_structure_detected():
    # Clear zig-zag with rising swing highs and rising swing lows -> HH + HL.
    highs = [11, 14, 11, 17, 12, 20, 13, 23, 14]
    lows = [10, 8, 12, 10, 14, 12, 16, 14, 18]
    df = _candles(highs, lows, lows)
    state = extract_market_structure(df, window=1).state
    assert state == Structure.BULLISH


def test_bearish_structure_detected():
    # Clear zig-zag with falling swing highs and falling swing lows -> LH + LL.
    highs = [22, 23, 12, 20, 11, 17, 10, 14, 9]
    lows = [18, 14, 16, 12, 14, 10, 12, 8, 10]
    df = _candles(highs, lows, lows)
    state = extract_market_structure(df, window=1).state
    assert state == Structure.BEARISH


def _config(tmpdb, **overrides):
    base = dict(
        api_key="", api_secret="", trade_mode="DRY_RUN", market_type="spot",
        allow_shorts=False, symbols=["BTC/USDT"], timeframe="15m",
        initial_equity=10.0, max_budget=10.0, risk_per_trade=0.02,
        stop_loss_pct=0.02, reward_risk_ratio=2.5, leverage=1,
        chronos_model="x", forecast_horizon=5, forecast_alpha=0.005,
        structure_window=5, context_length=200, taker_fee=0.0005,
        slippage=0.0005,
        weight_structure=0.30, weight_forecast=0.25, weight_momentum=0.20,
        weight_volatility=0.05, weight_sentiment=0.15, weight_news=0.0,
        require_structure_alignment=True, min_conviction=0.35,
        min_expected_value=0.001, rsi_period=14, macd_fast=12, macd_slow=26,
        macd_signal=9, atr_period=14, atr_max_pct=0.05,
        fear_greed_url="", fear_greed_contrarian=True, enable_news_sentiment=False,
        news_model="x", news_rss_urls=[], news_max_headlines=20,
        poll_seconds=900, db_path=tmpdb,
    )
    base.update(overrides)
    return Config(**base)


def test_risk_sizing_caps_at_budget():
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(os.path.join(d, "t.db"))
        rm = RiskManager(cfg)
        b = rm.size_position("buy", entry=100.0, equity=10.0)
        assert b is not None
        # notional must never exceed equity on spot
        assert b.notional <= 10.0 + 1e-9
        assert b.stop_loss < 100.0 < b.take_profit
        # reward:risk respected
        assert abs((b.take_profit - 100.0) / (100.0 - b.stop_loss) - 2.5) < 1e-6


def _ctx(cfg, structure_state):
    from core.signals.base import MarketContext
    from core.structure import StructureState
    df = _candles([1, 2, 3], [1, 2, 3], [1, 2, 3])
    return MarketContext(symbol="BTC/USDT", df=df, price=100.0, config=cfg,
                         structure=StructureState(structure_state))


class _StubSignal:
    def __init__(self, name, weight_key, score, conf, veto=False):
        self.name, self.weight_key = name, weight_key
        self._s = SignalScore(name, score, conf, veto=veto)

    def evaluate(self, ctx):
        return self._s


def test_engine_picks_long_on_bullish_alignment():
    from core.decision import DecisionEngine
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(os.path.join(d, "t.db"), min_conviction=0.2, min_expected_value=-1.0)
        providers = [
            _StubSignal("structure", "weight_structure", 1.0, 0.9),
            _StubSignal("momentum", "weight_momentum", 1.0, 0.9),
        ]
        eng = DecisionEngine(cfg, providers)
        a = eng.assess(_ctx(cfg, Structure.BULLISH))
        assert a.action == "LONG"
        assert a.p_up > 0.5


def test_engine_veto_forces_flat():
    from core.decision import DecisionEngine
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(os.path.join(d, "t.db"))
        providers = [
            _StubSignal("structure", "weight_structure", 1.0, 0.9),
            _StubSignal("volatility", "weight_volatility", 0.0, 1.0, veto=True),
        ]
        eng = DecisionEngine(cfg, providers)
        a = eng.assess(_ctx(cfg, Structure.BULLISH))
        assert a.action == "FLAT"
        assert "veto" in a.reason


def test_engine_structure_gate_blocks_counter_trend_long():
    from core.decision import DecisionEngine
    with tempfile.TemporaryDirectory() as d:
        # spot (no shorts): bearish structure means no allowed direction -> FLAT
        cfg = _config(os.path.join(d, "t.db"), min_conviction=0.0, min_expected_value=-1.0)
        providers = [_StubSignal("momentum", "weight_momentum", 1.0, 0.9)]
        eng = DecisionEngine(cfg, providers)
        a = eng.assess(_ctx(cfg, Structure.BEARISH))
        assert a.action == "FLAT"


def test_portfolio_blocks_double_entry_and_resolves_stop():
    with tempfile.TemporaryDirectory() as d:
        cfg = _config(os.path.join(d, "t.db"))
        pf = Portfolio(cfg)
        b = Bracket(side="buy", entry=100.0, stop_loss=98.0, take_profit=105.0,
                    quantity=0.1, notional=10.0)
        pf.open_position(b, "BTC/USDT", ts=1)
        assert pf.has_position("BTC/USDT")
        # a candle that pierces the stop must close it as a loss
        reason = pf.check_exits("BTC/USDT", candle_high=101.0, candle_low=97.0, ts=2)
        assert reason == "STOP_LOSS"
        assert not pf.has_position("BTC/USDT")
        assert pf.equity < 10.0  # took a loss + fees
        pf.close()
