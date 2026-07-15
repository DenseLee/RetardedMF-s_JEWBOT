"""
Alpha Vantage crypto sentiment filter.
Blocks trades that go against extreme market sentiment.

Uses Alpha Vantage's NEWS_SENTIMENT endpoint for crypto news.
Disabled by default (sentiment_enabled=False in config).
"""
import time
import logging
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    score: float            # -1.0 (extreme fear) to +1.0 (extreme greed)
    label: str              # 'bullish', 'bearish', 'neutral'
    block_long: bool        # True if sentiment is too bearish for longs
    block_short: bool       # True if sentiment is too bullish for shorts
    source: str = "alpha_vantage"
    articles_count: int = 0


class SentimentFilter:
    """
    Fetches crypto sentiment from Alpha Vantage API.
    Caches results to avoid rate limits (free tier: 25 requests/day).
    """

    def __init__(self, api_key: Optional[str] = None,
                 cache_minutes: int = 60,
                 block_threshold: float = 0.7,
                 enabled: bool = False):
        self.api_key = api_key
        self.cache_minutes = cache_minutes
        self.block_threshold = block_threshold
        self.enabled = enabled
        self._cached_result: Optional[SentimentResult] = None
        self._cache_time: float = 0.0

    def get_sentiment(self) -> SentimentResult:
        """Get current BTC sentiment. Returns cached result if fresh."""
        if not self.enabled:
            return SentimentResult(
                score=0.0, label="neutral",
                block_long=False, block_short=False,
                source="disabled")

        if self._is_cache_fresh():
            return self._cached_result

        if self.api_key:
            result = self._fetch_alpha_vantage()
        else:
            result = self._fallback_neutral()

        self._cached_result = result
        self._cache_time = time.time()
        return result

    def _is_cache_fresh(self) -> bool:
        if self._cached_result is None:
            return False
        return (time.time() - self._cache_time) < (self.cache_minutes * 60)

    def _fetch_alpha_vantage(self) -> SentimentResult:
        """Fetch crypto sentiment from Alpha Vantage API."""
        import urllib.request
        import json

        url = (f"https://www.alphavantage.co/query?"
               f"function=NEWS_SENTIMENT&tickers=CRYPTO:BTC&apikey={self.api_key}")

        try:
            with urllib.request.urlopen(url, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            articles = data.get("feed", [])
            if not articles:
                logger.warning("Alpha Vantage: no articles returned")
                return self._fallback_neutral()

            # Average sentiment from recent articles
            scores = []
            for article in articles[:20]:  # top 20 most recent
                sentiment = article.get("overall_sentiment_score", 0)
                scores.append(float(sentiment))

            if not scores:
                return self._fallback_neutral()

            avg_score = sum(scores) / len(scores)

            # Normalize: Alpha Vantage scores are roughly -1 to +1
            # Conservative: only block at extremes
            block_long = avg_score < -self.block_threshold
            block_short = avg_score > self.block_threshold

            label = "neutral"
            if avg_score > 0.3:
                label = "bullish"
            elif avg_score < -0.3:
                label = "bearish"

            logger.info(f"Sentiment: {label} (score={avg_score:.3f}, "
                       f"articles={len(scores)})")

            return SentimentResult(
                score=avg_score, label=label,
                block_long=block_long, block_short=block_short,
                articles_count=len(scores))

        except Exception as e:
            logger.error(f"Alpha Vantage API error: {e}")
            return self._fallback_neutral()

    def _fallback_neutral(self) -> SentimentResult:
        """Return neutral sentiment when API is unavailable."""
        return SentimentResult(
            score=0.0, label="neutral",
            block_long=False, block_short=False,
            source="fallback")

    def is_trade_allowed(self, direction: int) -> tuple:
        """
        Check if a trade in the given direction is allowed.

        Args:
            direction: 1 = long, -1 = short

        Returns:
            (allowed: bool, reason: str)
        """
        if not self.enabled:
            return True, "sentiment disabled"

        sentiment = self.get_sentiment()

        if direction == 1 and sentiment.block_long:
            return False, f"blocked: sentiment too bearish ({sentiment.score:.2f})"
        if direction == -1 and sentiment.block_short:
            return False, f"blocked: sentiment too bullish ({sentiment.score:.2f})"

        return True, f"allowed (sentiment: {sentiment.label})"
