"""Exchange connectivity and OHLCV retrieval via ccxt.

A thin wrapper around ccxt so the rest of the system never touches the raw
client. The same wrapper serves spot and futures by toggling defaultType.
"""

from __future__ import annotations

import logging

import ccxt
import pandas as pd

from config.settings import Config

logger = logging.getLogger(__name__)


class MarketData:
    """Loads markets and fetches candles from MEXC."""

    def __init__(self, config: Config) -> None:
        self.config = config
        default_type = "swap" if config.market_type == "futures" else "spot"
        self.exchange = ccxt.mexc(
            {
                "apiKey": config.api_key,
                "secret": config.api_secret,
                "enableRateLimit": True,
                "options": {"defaultType": default_type},
            }
        )
        self._markets_loaded = False

    def load_markets(self) -> None:
        """Load market metadata once (precision, limits, min notional)."""
        if not self._markets_loaded:
            self.exchange.load_markets()
            self._markets_loaded = True
            logger.info("Loaded %d markets from MEXC (%s)", len(self.exchange.markets), self.config.market_type)

    def market(self, symbol: str) -> dict:
        self.load_markets()
        return self.exchange.market(symbol)

    def fetch_ohlcv(self, symbol: str, timeframe: str | None = None, limit: int | None = None) -> pd.DataFrame:
        """Return a DataFrame of OHLCV candles indexed chronologically."""
        timeframe = timeframe or self.config.timeframe
        limit = limit or self.config.context_length
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    def last_price(self, df: pd.DataFrame) -> float:
        return float(df["close"].iloc[-1])

    def fetch_ohlcv_paginated(self, symbol: str, timeframe: str | None = None, total: int = 1000) -> pd.DataFrame:
        """Fetch `total` candles by paging backwards past the exchange's
        per-request cap (MEXC returns at most ~1000 klines per call).

        Without this, `backtest.py BTC/USDT 5000` silently tested on far fewer
        candles than requested.
        """
        timeframe = timeframe or self.config.timeframe
        tf_ms = self.exchange.parse_timeframe(timeframe) * 1000
        per_call = 1000
        since = self.exchange.milliseconds() - total * tf_ms
        rows: list[list] = []
        while len(rows) < total:
            batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=per_call)
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + tf_ms
            if len(batch) < per_call:
                break
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df.tail(total).reset_index(drop=True)
