"""
Market Data Cache.

Provides caching for market data including tickers, orderbooks, and prices.
"""

from decimal import Decimal
from typing import Any, Optional

from src.core import get_logger
from src.core.models import Ticker

from .redis_client import RedisManager

logger = get_logger(__name__)


# Default TTL values (in seconds)
TICKER_TTL = 5
ORDERBOOK_TTL = 2
PRICE_TTL = 5
KLINE_TTL = 60


class MarketCache:
    """
    Market data cache using Redis.

    Caches tickers, orderbooks, prices, and klines with appropriate TTLs
    for fast access to frequently requested market data.

    Example:
        >>> cache = MarketCache(redis_manager)
        >>> await cache.set_ticker("BTCUSDT", ticker)
        >>> ticker = await cache.get_ticker("BTCUSDT")
    """

    def __init__(self, redis: RedisManager):
        """
        Initialize MarketCache.

        Args:
            redis: RedisManager instance
        """
        self._redis = redis

    # =========================================================================
    # Ticker Cache
    # =========================================================================

    async def set_ticker(
        self,
        symbol: str,
        ticker: Ticker | dict[str, Any],
        ttl: int = TICKER_TTL,
    ) -> bool:
        """
        Cache ticker data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            ticker: Ticker object or dict
            ttl: Time to live in seconds (default 5)

        Returns:
            True if successful
        """
        key = f"ticker:{symbol.upper()}"

        if isinstance(ticker, Ticker):
            # Convert Ticker to dict for serialization
            data = {
                "symbol": ticker.symbol,
                "price": str(ticker.price),
                "bid": str(ticker.bid),
                "ask": str(ticker.ask),
                "high_24h": str(ticker.high_24h),
                "low_24h": str(ticker.low_24h),
                "volume_24h": str(ticker.volume_24h),
                "change_24h": str(ticker.change_24h),
                "timestamp": ticker.timestamp.isoformat(),
            }
        else:
            data = ticker

        return await self._redis.set(key, data, ttl=ttl)

    async def get_ticker(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Get cached ticker data.

        Args:
            symbol: Trading pair

        Returns:
            Ticker data dict or None if not cached
        """
        key = f"ticker:{symbol.upper()}"
        data = await self._redis.get(key)

        if data and isinstance(data, dict):
            # Convert string decimals back
            for field in ["price", "bid", "ask", "high_24h", "low_24h", "volume_24h", "change_24h"]:
                if field in data and data[field]:
                    data[field] = Decimal(data[field])

        return data

    # =========================================================================
    # Orderbook Cache
    # =========================================================================

    async def set_orderbook(
        self,
        symbol: str,
        orderbook: dict[str, Any],
        ttl: int = ORDERBOOK_TTL,
    ) -> bool:
        """
        Cache orderbook data.

        Args:
            symbol: Trading pair
            orderbook: Orderbook dict with 'bids' and 'asks'
            ttl: Time to live in seconds (default 2)

        Returns:
            True if successful
        """
        key = f"orderbook:{symbol.upper()}"

        # Convert Decimals to strings for serialization
        data = {
            "bids": [[str(p), str(q)] for p, q in orderbook.get("bids", [])],
            "asks": [[str(p), str(q)] for p, q in orderbook.get("asks", [])],
            "lastUpdateId": orderbook.get("lastUpdateId"),
        }

        return await self._redis.set(key, data, ttl=ttl)

    async def get_orderbook(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Get cached orderbook data.

        Args:
            symbol: Trading pair

        Returns:
            Orderbook dict or None if not cached
        """
        key = f"orderbook:{symbol.upper()}"
        data = await self._redis.get(key)

        if data and isinstance(data, dict):
            # Convert string decimals back
            data["bids"] = [[Decimal(p), Decimal(q)] for p, q in data.get("bids", [])]
            data["asks"] = [[Decimal(p), Decimal(q)] for p, q in data.get("asks", [])]

        return data

    # =========================================================================
    # Price Cache
    # =========================================================================

    async def set_price(
        self,
        symbol: str,
        price: Decimal | str,
        ttl: int = PRICE_TTL,
    ) -> bool:
        """
        Cache latest price.

        Args:
            symbol: Trading pair
            price: Current price
            ttl: Time to live in seconds (default 5)

        Returns:
            True if successful
        """
        key = f"price:{symbol.upper()}"
        return await self._redis.set(key, str(price), ttl=ttl)

    async def get_price(self, symbol: str) -> Optional[Decimal]:
        """
        Get cached price.

        Args:
            symbol: Trading pair

        Returns:
            Price as Decimal or None if not cached
        """
        key = f"price:{symbol.upper()}"
        value = await self._redis.get(key)

        if value:
            return Decimal(str(value))
        return None

    async def set_prices(
        self,
        prices: dict[str, Decimal | str],
        ttl: int = PRICE_TTL,
    ) -> bool:
        """
        Cache multiple prices.

        Args:
            prices: Dict of symbol -> price
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        mapping = {
            f"price:{symbol.upper()}": str(price)
            for symbol, price in prices.items()
        }
        result = await self._redis.mset(mapping)

        # Set TTL for each key
        for symbol in prices:
            await self._redis.expire(f"price:{symbol.upper()}", ttl)

        return result

    async def get_prices(self, symbols: list[str]) -> dict[str, Optional[Decimal]]:
        """
        Get multiple cached prices.

        Args:
            symbols: List of trading pairs

        Returns:
            Dict of symbol -> price (None for uncached)
        """
        keys = [f"price:{s.upper()}" for s in symbols]
        values = await self._redis.mget(keys)

        result = {}
        for symbol, value in zip(symbols, values):
            if value:
                result[symbol] = Decimal(str(value))
            else:
                result[symbol] = None

        return result

    # =========================================================================
    # Kline Cache
    # =========================================================================

    async def set_kline(
        self,
        symbol: str,
        interval: str,
        kline: dict[str, Any],
        ttl: int = KLINE_TTL,
    ) -> bool:
        """
        Cache latest kline.

        Args:
            symbol: Trading pair
            interval: Kline interval (e.g., "1h")
            kline: Kline data dict
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        key = f"kline:{symbol.upper()}:{interval}"

        # Convert Decimals to strings
        data = {}
        for k, v in kline.items():
            if isinstance(v, Decimal):
                data[k] = str(v)
            elif hasattr(v, "isoformat"):  # datetime
                data[k] = v.isoformat()
            else:
                data[k] = v

        return await self._redis.set(key, data, ttl=ttl)

    async def get_kline(
        self,
        symbol: str,
        interval: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get cached kline.

        Args:
            symbol: Trading pair
            interval: Kline interval

        Returns:
            Kline dict or None if not cached
        """
        key = f"kline:{symbol.upper()}:{interval}"
        data = await self._redis.get(key)

        if data and isinstance(data, dict):
            # Convert string decimals back
            for field in ["open", "high", "low", "close", "volume", "quote_volume"]:
                if field in data and data[field]:
                    data[field] = Decimal(data[field])

        return data

    # =========================================================================
    # Pub/Sub for Market Updates
    # =========================================================================

    async def publish_ticker(self, symbol: str, ticker: dict[str, Any]) -> int:
        """
        Publish ticker update.

        Args:
            symbol: Trading pair
            ticker: Ticker data

        Returns:
            Number of subscribers
        """
        channel = f"ticker:{symbol.upper()}"
        return await self._redis.publish(channel, ticker)

    async def subscribe_ticker(
        self,
        symbol: str,
        callback,
    ) -> None:
        """
        Subscribe to ticker updates.

        Args:
            symbol: Trading pair
            callback: Callback function(channel, ticker_data)
        """
        channel = f"ticker:{symbol.upper()}"
        await self._redis.subscribe(channel, callback)

    async def unsubscribe_ticker(self, symbol: str) -> None:
        """
        Unsubscribe from ticker updates.

        Args:
            symbol: Trading pair
        """
        channel = f"ticker:{symbol.upper()}"
        await self._redis.unsubscribe(channel)

    # =========================================================================
    # Cache Invalidation
    # =========================================================================

    async def invalidate_ticker(self, symbol: str) -> bool:
        """
        Invalidate cached ticker.

        Args:
            symbol: Trading pair

        Returns:
            True if deleted
        """
        key = f"ticker:{symbol.upper()}"
        return await self._redis.delete(key)

    async def invalidate_orderbook(self, symbol: str) -> bool:
        """
        Invalidate cached orderbook.

        Args:
            symbol: Trading pair

        Returns:
            True if deleted
        """
        key = f"orderbook:{symbol.upper()}"
        return await self._redis.delete(key)

    async def invalidate_price(self, symbol: str) -> bool:
        """
        Invalidate cached price.

        Args:
            symbol: Trading pair

        Returns:
            True if deleted
        """
        key = f"price:{symbol.upper()}"
        return await self._redis.delete(key)
