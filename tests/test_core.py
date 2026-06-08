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


def _config(tmpdb):
    return Config(
        api_key="", api_secret="", trade_mode="DRY_RUN", market_type="spot",
        allow_shorts=False, symbols=["BTC/USDT"], timeframe="15m",
        initial_equity=10.0, max_budget=10.0, risk_per_trade=0.02,
        stop_loss_pct=0.02, reward_risk_ratio=2.5, leverage=1,
        chronos_model="x", forecast_horizon=5, forecast_alpha=0.005,
        structure_window=5, context_length=200, taker_fee=0.0005,
        slippage=0.0005, poll_seconds=900, db_path=tmpdb,
    )


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
