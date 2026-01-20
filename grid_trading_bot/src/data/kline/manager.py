"""
K-line Manager.

Provides unified K-line data access with three-tier data source integration:
1. Redis Cache (fastest)
2. PostgreSQL Database (persistent)
3. Exchange API (most recent)
"""

from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

from sqlalchemy import select

from src.core import get_logger
from src.core.models import Kline, KlineInterval
from src.data.cache import MarketCache, RedisManager
from src.data.database import DatabaseManager, KlineModel
from src.exchange import ExchangeClient

logger = get_logger(__name__)


# Default cache TTL for klines (in seconds)
KLINE_CACHE_TTL = 60


class KlineManager:
    """
    K-line data manager with three-tier data source.

    Integrates Redis cache, PostgreSQL database, and exchange API
    to provide efficient K-line data access with automatic fallback.

    Data retrieval order:
    1. Redis cache (fastest, short-lived)
    2. PostgreSQL database (persistent storage)
    3. Exchange API (most up-to-date)

    Example:
        >>> manager = KlineManager(redis, db, exchange)
        >>> klines = await manager.get_klines("BTCUSDT", "1h", limit=100)
        >>> latest = await manager.get_latest_kline("BTCUSDT", "1h")
    """

    def __init__(
        self,
        redis: RedisManager,
        database: DatabaseManager,
        exchange: ExchangeClient,
    ):
        """
        Initialize KlineManager.

        Args:
            redis: RedisManager instance for caching
            database: DatabaseManager instance for persistence
            exchange: ExchangeClient instance for API access
        """
        self._redis = redis
        self._db = database
        self._exchange = exchange
        self._market_cache = MarketCache(redis)

        # Subscription callbacks
        self._subscriptions: dict[str, list[Callable]] = {}

    # =========================================================================
    # Public Methods
    # =========================================================================

    async def get_klines(
        self,
        symbol: str,
        interval: str | KlineInterval,
        limit: int = 500,
    ) -> list[Kline]:
        """
        Get K-lines with three-tier fallback.

        Retrieval order:
        1. Check Redis cache
        2. Query PostgreSQL if cache insufficient
        3. Fetch from exchange if still insufficient
        4. Update cache and database

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: K-line interval (e.g., "1h" or KlineInterval.h1)
            limit: Maximum number of K-lines to return

        Returns:
            List of Kline objects (oldest first)
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        logger.debug(f"Getting {limit} klines for {symbol} {interval_str}")

        # Step 1: Try to get from cache
        cached = await self._get_from_cache(symbol, interval_str, limit)
        if len(cached) >= limit:
            logger.debug(f"Cache hit: {len(cached)} klines")
            return cached[-limit:]

        # Step 2: Try to get from database
        db_klines = await self._get_from_database(symbol, interval_str, limit)
        if len(db_klines) >= limit:
            logger.debug(f"Database hit: {len(db_klines)} klines")
            # Update cache with database data
            await self._update_cache(symbol, interval_str, db_klines)
            return db_klines[-limit:]

        # Step 3: Fetch from exchange
        exchange_klines = await self._get_from_exchange(symbol, interval_str, limit)
        if exchange_klines:
            logger.debug(f"Exchange fetch: {len(exchange_klines)} klines")
            # Update both cache and database
            await self._update_cache(symbol, interval_str, exchange_klines)
            await self._save_to_database(symbol, interval_str, exchange_klines)
            return exchange_klines[-limit:]

        # Return whatever we have
        combined = self._merge_klines(db_klines, cached)
        return combined[-limit:] if combined else []

    async def get_klines_range(
        self,
        symbol: str,
        interval: str | KlineInterval,
        start: datetime,
        end: datetime,
    ) -> list[Kline]:
        """
        Get K-lines within a time range.

        Args:
            symbol: Trading pair
            interval: K-line interval
            start: Start time (inclusive)
            end: End time (inclusive)

        Returns:
            List of Kline objects within the range
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        logger.debug(f"Getting klines for {symbol} {interval_str} from {start} to {end}")

        # Try database first for range queries
        db_klines = await self._get_from_database_range(
            symbol, interval_str, start, end
        )

        if db_klines:
            return db_klines

        # Fallback to exchange
        exchange_klines = await self._get_from_exchange_range(
            symbol, interval_str, start, end
        )

        if exchange_klines:
            await self._save_to_database(symbol, interval_str, exchange_klines)

        return exchange_klines

    async def get_latest_kline(
        self,
        symbol: str,
        interval: str | KlineInterval,
    ) -> Optional[Kline]:
        """
        Get the most recent K-line.

        Args:
            symbol: Trading pair
            interval: K-line interval

        Returns:
            Latest Kline or None if not available
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        # Try cache first
        cached = await self._market_cache.get_kline(symbol, interval_str)
        if cached:
            return self._dict_to_kline(cached, symbol, interval_str)

        # Fetch from exchange
        klines = await self._get_from_exchange(symbol, interval_str, 1)
        if klines:
            await self._update_cache(symbol, interval_str, klines)
            return klines[-1]

        return None

    async def sync_klines(
        self,
        symbol: str,
        interval: str | KlineInterval,
        days: int = 30,
    ) -> int:
        """
        Sync historical K-lines to database.

        Fetches historical data from exchange and stores in database
        for offline access and faster queries.

        Args:
            symbol: Trading pair
            interval: K-line interval
            days: Number of days of history to sync

        Returns:
            Number of K-lines synced
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        logger.info(f"Syncing {days} days of {symbol} {interval_str} klines")

        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        # Fetch from exchange
        klines = await self._get_from_exchange_range(
            symbol, interval_str, start_time, end_time
        )

        if klines:
            await self._save_to_database(symbol, interval_str, klines)
            logger.info(f"Synced {len(klines)} klines for {symbol} {interval_str}")
            return len(klines)

        return 0

    async def subscribe_kline(
        self,
        symbol: str,
        interval: str | KlineInterval,
        callback: Callable[[Kline], None],
    ) -> None:
        """
        Subscribe to K-line updates.

        Args:
            symbol: Trading pair
            interval: K-line interval
            callback: Callback function(kline)
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        key = f"{symbol}:{interval_str}"

        if key not in self._subscriptions:
            self._subscriptions[key] = []

            # Subscribe to WebSocket kline stream
            async def on_kline(data: dict):
                kline = Kline.from_binance_ws(
                    data["k"],
                    symbol,
                    KlineInterval(interval_str),
                )
                # Update cache
                await self._market_cache.set_kline(
                    symbol, interval_str, self._kline_to_dict(kline)
                )
                # Call all callbacks
                for cb in self._subscriptions.get(key, []):
                    try:
                        cb(kline)
                    except Exception as e:
                        logger.error(f"Kline callback error: {e}")

            await self._exchange.ws.subscribe_kline(symbol, interval_str, on_kline)

        self._subscriptions[key].append(callback)
        logger.debug(f"Subscribed to {symbol} {interval_str} klines")

    async def unsubscribe_kline(
        self,
        symbol: str,
        interval: str | KlineInterval,
        callback: Optional[Callable] = None,
    ) -> None:
        """
        Unsubscribe from K-line updates.

        Args:
            symbol: Trading pair
            interval: K-line interval
            callback: Specific callback to remove (None removes all)
        """
        symbol = symbol.upper()
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        key = f"{symbol}:{interval_str}"

        if key in self._subscriptions:
            if callback:
                self._subscriptions[key].remove(callback)
            else:
                self._subscriptions[key] = []

            if not self._subscriptions[key]:
                del self._subscriptions[key]
                # Unsubscribe from WebSocket
                if self._exchange.ws:
                    await self._exchange.ws.unsubscribe_kline(symbol, interval_str)

    # =========================================================================
    # Private Methods - Data Sources
    # =========================================================================

    async def _get_from_cache(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[Kline]:
        """Get K-lines from Redis cache."""
        try:
            # Get cached klines list
            key = f"klines:{symbol}:{interval}"
            data = await self._redis.get(key)

            if data and isinstance(data, list):
                klines = [
                    self._dict_to_kline(d, symbol, interval) for d in data[-limit:]
                ]
                return klines
        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return []

    async def _get_from_database(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[Kline]:
        """Get K-lines from PostgreSQL database."""
        try:
            async with self._db.get_session() as session:
                query = (
                    select(KlineModel)
                    .where(KlineModel.symbol == symbol)
                    .where(KlineModel.interval == interval)
                    .order_by(KlineModel.open_time.desc())
                    .limit(limit)
                )
                result = await session.execute(query)
                models = result.scalars().all()

                # Convert to domain models and reverse to oldest first
                klines = [m.to_domain() for m in reversed(models)]
                return klines

        except Exception as e:
            logger.warning(f"Database read error: {e}")
            return []

    async def _get_from_database_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Kline]:
        """Get K-lines from database within time range."""
        try:
            async with self._db.get_session() as session:
                query = (
                    select(KlineModel)
                    .where(KlineModel.symbol == symbol)
                    .where(KlineModel.interval == interval)
                    .where(KlineModel.open_time >= start)
                    .where(KlineModel.open_time <= end)
                    .order_by(KlineModel.open_time.asc())
                )
                result = await session.execute(query)
                models = result.scalars().all()

                return [m.to_domain() for m in models]

        except Exception as e:
            logger.warning(f"Database range query error: {e}")
            return []

    async def _get_from_exchange(
        self,
        symbol: str,
        interval: str,
        limit: int,
    ) -> list[Kline]:
        """Get K-lines from exchange API."""
        try:
            klines = await self._exchange.spot.get_klines(
                symbol, KlineInterval(interval), limit
            )
            return klines
        except Exception as e:
            logger.warning(f"Exchange API error: {e}")
            return []

    async def _get_from_exchange_range(
        self,
        symbol: str,
        interval: str,
        start: datetime,
        end: datetime,
    ) -> list[Kline]:
        """Get K-lines from exchange API within time range."""
        try:
            # Convert to timestamps
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)

            all_klines: list[Kline] = []
            current_start = start_ms

            # Fetch in batches (max 1000 per request)
            while current_start < end_ms:
                klines = await self._exchange.spot.get_klines(
                    symbol,
                    KlineInterval(interval),
                    limit=1000,
                    start_time=current_start,
                    end_time=end_ms,
                )

                if not klines:
                    break

                all_klines.extend(klines)

                # Move to next batch
                last_time = klines[-1].close_time
                current_start = int(last_time.timestamp() * 1000) + 1

                # Prevent infinite loop
                if len(klines) < 1000:
                    break

            return all_klines

        except Exception as e:
            logger.warning(f"Exchange range query error: {e}")
            return []

    # =========================================================================
    # Private Methods - Data Storage
    # =========================================================================

    async def _update_cache(
        self,
        symbol: str,
        interval: str,
        klines: list[Kline],
    ) -> None:
        """Update Redis cache with K-lines."""
        try:
            key = f"klines:{symbol}:{interval}"
            data = [self._kline_to_dict(k) for k in klines]
            await self._redis.set(key, data, ttl=KLINE_CACHE_TTL)

            # Also cache latest kline separately
            if klines:
                await self._market_cache.set_kline(
                    symbol, interval, self._kline_to_dict(klines[-1])
                )
        except Exception as e:
            logger.warning(f"Cache update error: {e}")

    async def _save_to_database(
        self,
        symbol: str,
        interval: str,
        klines: list[Kline],
    ) -> None:
        """Save K-lines to PostgreSQL database."""
        try:
            async with self._db.get_session() as session:
                for kline in klines:
                    # Check if exists (upsert logic)
                    query = (
                        select(KlineModel)
                        .where(KlineModel.symbol == symbol)
                        .where(KlineModel.interval == interval)
                        .where(KlineModel.open_time == kline.open_time)
                    )
                    result = await session.execute(query)
                    existing = result.scalar_one_or_none()

                    if existing:
                        # Update existing
                        existing.open = kline.open
                        existing.high = kline.high
                        existing.low = kline.low
                        existing.close = kline.close
                        existing.volume = kline.volume
                        existing.close_time = kline.close_time
                    else:
                        # Insert new
                        model = KlineModel.from_domain(kline)
                        session.add(model)

        except Exception as e:
            logger.warning(f"Database save error: {e}")

    # =========================================================================
    # Private Methods - Utilities
    # =========================================================================

    def _kline_to_dict(self, kline: Kline) -> dict:
        """Convert Kline to dict for caching."""
        return {
            "open_time": kline.open_time.isoformat(),
            "open": str(kline.open),
            "high": str(kline.high),
            "low": str(kline.low),
            "close": str(kline.close),
            "volume": str(kline.volume),
            "close_time": kline.close_time.isoformat(),
            "quote_volume": str(kline.quote_volume),
            "trades_count": kline.trades_count,
        }

    def _dict_to_kline(self, data: dict, symbol: str, interval: str) -> Kline:
        """Convert dict to Kline."""
        from decimal import Decimal

        return Kline(
            symbol=symbol,
            interval=KlineInterval(interval),
            open_time=datetime.fromisoformat(data["open_time"]),
            open=Decimal(data["open"]),
            high=Decimal(data["high"]),
            low=Decimal(data["low"]),
            close=Decimal(data["close"]),
            volume=Decimal(data["volume"]),
            close_time=datetime.fromisoformat(data["close_time"]),
            quote_volume=Decimal(data.get("quote_volume", "0")),
            trades_count=data.get("trades_count", 0),
        )

    def _merge_klines(
        self,
        list1: list[Kline],
        list2: list[Kline],
    ) -> list[Kline]:
        """Merge two kline lists, removing duplicates by open_time."""
        seen = set()
        merged = []

        for k in list1 + list2:
            key = k.open_time
            if key not in seen:
                seen.add(key)
                merged.append(k)

        # Sort by open_time
        merged.sort(key=lambda x: x.open_time)
        return merged
