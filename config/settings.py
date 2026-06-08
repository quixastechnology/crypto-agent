"""Central configuration loader.

All tunable parameters live here and are loaded from environment variables
(.env) so nothing sensitive is hard-coded. Import `load_config()` and pass the
returned `Config` object through the system.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass
class Config:
    """Immutable runtime configuration for the trading agent."""

    # --- Exchange credentials ---
    api_key: str
    api_secret: str

    # --- Mode and market ---
    trade_mode: str            # "DRY_RUN" or "LIVE"
    market_type: str           # "spot" or "futures"
    allow_shorts: bool         # only honoured on futures
    symbols: list[str]
    timeframe: str

    # --- Capital and risk ---
    initial_equity: float      # starting equity for the simulated ledger
    max_budget: float          # hard cap on capital deployed at once (USDT)
    risk_per_trade: float      # fraction of equity risked per trade, e.g. 0.02
    stop_loss_pct: float       # stop distance as a fraction of entry price
    reward_risk_ratio: float   # take-profit multiple of the stop distance
    leverage: int              # futures only

    # --- Forecast model ---
    chronos_model: str
    forecast_horizon: int
    forecast_alpha: float      # bias threshold buffer, e.g. 0.005 (0.5%)
    structure_window: int      # swing detection radius (bars)
    context_length: int        # candles fed to the model / structure parser

    # --- Costs (used by the simulator and for edge checks) ---
    taker_fee: float           # per-side taker fee fraction, e.g. 0.0005
    slippage: float            # assumed slippage per side, fraction

    # --- Decision engine: signal weights ---
    weight_structure: float
    weight_forecast: float
    weight_momentum: float
    weight_volatility: float
    weight_sentiment: float
    weight_news: float
    require_structure_alignment: bool   # structure gates trade direction
    min_conviction: float               # min aggregate confidence to trade
    min_expected_value: float           # min EV (fraction of price) to trade

    # --- Momentum / volatility params ---
    rsi_period: int
    macd_fast: int
    macd_slow: int
    macd_signal: int
    atr_period: int
    atr_max_pct: float                  # ATR above this -> no-trade veto

    # --- Sentiment ---
    fear_greed_url: str
    fear_greed_contrarian: bool
    enable_news_sentiment: bool
    news_model: str
    news_rss_urls: list[str]
    news_max_headlines: int

    # --- Loop ---
    poll_seconds: int
    db_path: str

    derived: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.trade_mode = self.trade_mode.upper()
        self.market_type = self.market_type.lower()
        if self.trade_mode not in {"DRY_RUN", "LIVE"}:
            raise ValueError(f"TRADE_MODE must be DRY_RUN or LIVE, got {self.trade_mode}")
        if self.market_type not in {"spot", "futures"}:
            raise ValueError(f"MARKET_TYPE must be spot or futures, got {self.market_type}")
        # Shorting is impossible on spot regardless of the flag.
        if self.market_type == "spot":
            self.allow_shorts = False


def _get_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _get_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _get_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


def load_config() -> Config:
    """Read .env and return a validated Config."""
    load_dotenv()

    symbols_raw = os.getenv("SYMBOLS", "BTC/USDT")
    symbols = [s.strip() for s in symbols_raw.split(",") if s.strip()]

    news_rss = os.getenv(
        "NEWS_RSS_URLS",
        "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss",
    )
    news_rss_urls = [u.strip() for u in news_rss.split(",") if u.strip()]

    return Config(
        api_key=os.getenv("MEXC_API_KEY", ""),
        api_secret=os.getenv("MEXC_API_SECRET", ""),
        trade_mode=os.getenv("TRADE_MODE", "DRY_RUN"),
        market_type=os.getenv("MARKET_TYPE", "spot"),
        allow_shorts=_get_bool("ALLOW_SHORTS", False),
        symbols=symbols,
        timeframe=os.getenv("TIMEFRAME", "15m"),
        initial_equity=_get_float("INITIAL_TEST_BUDGET", 10.0),
        max_budget=_get_float("MAX_BUDGET", 10.0),
        risk_per_trade=_get_float("MAX_RISK_PER_TRADE", 0.02),
        stop_loss_pct=_get_float("STOP_LOSS_PCT", 0.02),
        reward_risk_ratio=_get_float("REWARD_RISK_RATIO", 2.5),
        leverage=_get_int("LEVERAGE", 1),
        chronos_model=os.getenv("CHRONOS_MODEL", "amazon/chronos-t5-tiny"),
        forecast_horizon=_get_int("FORECAST_HORIZON", 5),
        forecast_alpha=_get_float("FORECAST_ALPHA", 0.005),
        structure_window=_get_int("STRUCTURE_WINDOW", 5),
        context_length=_get_int("CONTEXT_LENGTH", 200),
        taker_fee=_get_float("TAKER_FEE", 0.0005),
        slippage=_get_float("SLIPPAGE", 0.0005),
        weight_structure=_get_float("WEIGHT_STRUCTURE", 0.30),
        weight_forecast=_get_float("WEIGHT_FORECAST", 0.25),
        weight_momentum=_get_float("WEIGHT_MOMENTUM", 0.20),
        weight_volatility=_get_float("WEIGHT_VOLATILITY", 0.05),
        weight_sentiment=_get_float("WEIGHT_SENTIMENT", 0.15),
        weight_news=_get_float("WEIGHT_NEWS", 0.05),
        require_structure_alignment=_get_bool("REQUIRE_STRUCTURE_ALIGNMENT", True),
        min_conviction=_get_float("MIN_CONVICTION", 0.35),
        min_expected_value=_get_float("MIN_EXPECTED_VALUE", 0.001),
        rsi_period=_get_int("RSI_PERIOD", 14),
        macd_fast=_get_int("MACD_FAST", 12),
        macd_slow=_get_int("MACD_SLOW", 26),
        macd_signal=_get_int("MACD_SIGNAL", 9),
        atr_period=_get_int("ATR_PERIOD", 14),
        atr_max_pct=_get_float("ATR_MAX_PCT", 0.05),
        fear_greed_url=os.getenv("FEAR_GREED_URL", "https://api.alternative.me/fng/?limit=1"),
        fear_greed_contrarian=_get_bool("FEAR_GREED_CONTRARIAN", True),
        enable_news_sentiment=_get_bool("ENABLE_NEWS_SENTIMENT", False),
        news_model=os.getenv("NEWS_MODEL", "ElKulako/cryptobert"),
        news_rss_urls=news_rss_urls,
        news_max_headlines=_get_int("NEWS_MAX_HEADLINES", 20),
        poll_seconds=_get_int("POLL_SECONDS", 900),
        db_path=os.getenv("DB_PATH", "crypto_agent.db"),
    )
