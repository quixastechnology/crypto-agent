"""Sentiment providers.

1. FearGreedSignal  - free market-wide Crypto Fear & Greed Index (alternative.me,
   no API key). Always on. Treated as trend-confirming in the mid-range and
   contrarian at extremes (extreme greed/fear often precedes reversals), which
   is configurable.

2. NewsSentimentSignal - optional. Runs a HuggingFace crypto/finance sentiment
   model (CryptoBERT or FinBERT) over recent headlines, like Chronos it loads
   lazily and is opt-in via ENABLE_NEWS_SENTIMENT so the base bot stays light.

Both are failsafe: network/model errors degrade to neutral, zero confidence.
"""

from __future__ import annotations

import json
import logging
import urllib.request
import xml.etree.ElementTree as ET

from core.signals.base import MarketContext, SignalProvider, SignalScore

logger = logging.getLogger(__name__)

_HTTP_TIMEOUT = 10


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "crypto-agent/1.0"})
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
        return resp.read()


class FearGreedSignal(SignalProvider):
    name = "fear_greed"
    weight_key = "weight_sentiment"

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        raw = _get(ctx.config.fear_greed_url)
        data = json.loads(raw)["data"][0]
        value = int(data["value"])               # 0..100
        label = data.get("value_classification", "")

        centered = (value - 50) / 50.0           # [-1, +1], greed positive
        confidence = min(1.0, abs(centered))

        # Contrarian flip at extremes if configured.
        if ctx.config.fear_greed_contrarian and (value >= 80 or value <= 20):
            centered = -centered
            label += " (contrarian)"
        return SignalScore(self.name, centered, confidence, f"F&G {value} {label}")


class NewsSentimentSignal(SignalProvider):
    name = "news_sentiment"
    weight_key = "weight_news"

    def __init__(self) -> None:
        self._pipeline = None

    def _ensure_pipeline(self, model_name: str) -> None:
        if self._pipeline is not None:
            return
        from transformers import pipeline  # optional dependency
        logger.info("Loading news sentiment model %s", model_name)
        self._pipeline = pipeline("sentiment-analysis", model=model_name, truncation=True)

    def _headlines(self, ctx: MarketContext) -> list[str]:
        base = ctx.symbol.split("/")[0]
        titles: list[str] = []
        for url in ctx.config.news_rss_urls:
            try:
                root = ET.fromstring(_get(url))
                for item in root.iter("item"):
                    title = item.findtext("title") or ""
                    if title:
                        titles.append(title)
            except Exception as exc:
                logger.warning("news feed %s failed: %s", url, exc)
        # Keep headlines that mention the asset (or all, if none match).
        relevant = [t for t in titles if base.lower() in t.lower()]
        pool = relevant or titles
        return pool[: ctx.config.news_max_headlines]

    def _evaluate(self, ctx: MarketContext) -> SignalScore:
        if not ctx.config.enable_news_sentiment:
            return SignalScore(self.name, 0.0, 0.0, "news sentiment disabled")

        headlines = self._headlines(ctx)
        if not headlines:
            return SignalScore(self.name, 0.0, 0.0, "no headlines")

        self._ensure_pipeline(ctx.config.news_model)
        results = self._pipeline(headlines)
        total = 0.0
        for r in results:
            label = r["label"].lower()
            sign = 1.0 if ("pos" in label or "bull" in label) else (-1.0 if ("neg" in label or "bear" in label) else 0.0)
            total += sign * float(r["score"])
        score = total / len(results)
        confidence = min(1.0, abs(score))
        return SignalScore(self.name, score, confidence, f"{len(headlines)} headlines, mean {score:+.2f}")
