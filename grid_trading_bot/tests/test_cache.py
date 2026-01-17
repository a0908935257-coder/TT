"""
Tests for Redis Cache Layer.

Tests RedisManager, MarketCache, and AccountCache functionality.
"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import Balance, Position, Ticker
from data.cache import AccountCache, MarketCache, RedisManager


# =============================================================================
# RedisManager Tests
# =============================================================================


class TestRedisManagerInit:
    """Test RedisManager initialization."""

    def test_default_init(self):
        """Test default initialization."""
        manager = RedisManager()
        assert manager._host == "localhost"
        assert manager._port == 6379
        assert manager._db == 0
        assert manager._password is None
        assert manager._key_prefix == "trading:"
        assert manager._connected is False

    def test_custom_init(self):
        """Test custom initialization."""
        manager = RedisManager(
            host="redis.example.com",
            port=6380,
            db=1,
            password="secret",
            key_prefix="bot:",
            max_connections=20,
        )
        assert manager._host == "redis.example.com"
        assert manager._port == 6380
        assert manager._db == 1
        assert manager._password == "secret"
        assert manager._key_prefix == "bot:"
        assert manager._max_connections == 20


class TestRedisManagerProperties:
    """Test RedisManager properties."""

    def test_is_connected_false(self):
        """Test is_connected when not connected."""
        manager = RedisManager()
        assert manager.is_connected is False

    def test_key_prefix(self):
        """Test key_prefix property."""
        manager = RedisManager(key_prefix="test:")
        assert manager.key_prefix == "test:"

    def test_client_none(self):
        """Test client property when not connected."""
        manager = RedisManager()
        assert manager.client is None


class TestRedisManagerConnect:
    """Test RedisManager connection."""

    @pytest.mark.asyncio
    async def test_connect_success(self):
        """Test successful connection."""
        manager = RedisManager()

        with patch("data.cache.redis_client.redis.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            result = await manager.connect()

            assert result is True
            assert manager.is_connected is True
            mock_client.ping.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_already_connected(self):
        """Test connect when already connected."""
        manager = RedisManager()
        manager._connected = True

        result = await manager.connect()
        assert result is True

    @pytest.mark.asyncio
    async def test_connect_failure(self):
        """Test connection failure."""
        manager = RedisManager()

        with patch("data.cache.redis_client.redis.Redis") as mock_redis:
            mock_redis.side_effect = Exception("Connection refused")

            result = await manager.connect()

            assert result is False
            assert manager.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_health_check_fail(self):
        """Test connection with failed health check."""
        manager = RedisManager()

        with patch("data.cache.redis_client.redis.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = False
            mock_redis.return_value = mock_client

            result = await manager.connect()

            assert result is False


class TestRedisManagerDisconnect:
    """Test RedisManager disconnect."""

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test disconnect."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._connected = True

        await manager.disconnect()

        assert manager._connected is False
        assert manager._client is None

    @pytest.mark.asyncio
    async def test_disconnect_with_pubsub(self):
        """Test disconnect with active pubsub."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._pubsub = AsyncMock()
        manager._connected = True

        # Create a real task that raises CancelledError
        async def cancelled_coro():
            raise asyncio.CancelledError()

        manager._pubsub_task = asyncio.create_task(cancelled_coro())

        await manager.disconnect()

        assert manager._connected is False
        assert manager._pubsub is None


class TestRedisManagerContextManager:
    """Test RedisManager context manager."""

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        with patch("data.cache.redis_client.redis.Redis") as mock_redis:
            mock_client = AsyncMock()
            mock_client.ping.return_value = True
            mock_redis.return_value = mock_client

            async with RedisManager() as manager:
                assert manager.is_connected is True

            mock_client.close.assert_awaited_once()


class TestRedisManagerKeyManagement:
    """Test RedisManager key management."""

    def test_make_key(self):
        """Test key prefix creation."""
        manager = RedisManager(key_prefix="trading:")
        assert manager._make_key("test") == "trading:test"
        assert manager._make_key("market:btc") == "trading:market:btc"


class TestRedisManagerBasicOperations:
    """Test RedisManager basic operations."""

    @pytest.fixture
    def connected_manager(self):
        """Create connected manager mock."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._connected = True
        return manager

    @pytest.mark.asyncio
    async def test_get_string(self, connected_manager):
        """Test get with string value."""
        connected_manager._client.get.return_value = "test_value"

        result = await connected_manager.get("test_key")

        assert result == "test_value"
        connected_manager._client.get.assert_awaited_once_with("trading:test_key")

    @pytest.mark.asyncio
    async def test_get_json(self, connected_manager):
        """Test get with JSON value."""
        connected_manager._client.get.return_value = '{"key": "value"}'

        result = await connected_manager.get("test_key")

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_get_none(self, connected_manager):
        """Test get with missing key."""
        connected_manager._client.get.return_value = None

        result = await connected_manager.get("missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_not_connected(self):
        """Test get when not connected."""
        manager = RedisManager()

        with pytest.raises(RuntimeError, match="Redis not connected"):
            await manager.get("test")

    @pytest.mark.asyncio
    async def test_set_string(self, connected_manager):
        """Test set with string value."""
        connected_manager._client.set.return_value = True

        result = await connected_manager.set("key", "value")

        assert result is True
        connected_manager._client.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_dict(self, connected_manager):
        """Test set with dict value (JSON serialized)."""
        connected_manager._client.set.return_value = True

        result = await connected_manager.set("key", {"data": "value"})

        assert result is True
        # Verify JSON serialization
        call_args = connected_manager._client.set.call_args
        assert '"data"' in call_args[0][1]

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, connected_manager):
        """Test set with TTL."""
        connected_manager._client.setex.return_value = True

        result = await connected_manager.set("key", "value", ttl=60)

        assert result is True
        connected_manager._client.setex.assert_awaited_once_with(
            "trading:key", 60, "value"
        )

    @pytest.mark.asyncio
    async def test_delete(self, connected_manager):
        """Test delete."""
        connected_manager._client.delete.return_value = 1

        result = await connected_manager.delete("key")

        assert result is True
        connected_manager._client.delete.assert_awaited_once_with("trading:key")

    @pytest.mark.asyncio
    async def test_delete_missing(self, connected_manager):
        """Test delete missing key."""
        connected_manager._client.delete.return_value = 0

        result = await connected_manager.delete("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_exists(self, connected_manager):
        """Test exists."""
        connected_manager._client.exists.return_value = 1

        result = await connected_manager.exists("key")

        assert result is True

    @pytest.mark.asyncio
    async def test_not_exists(self, connected_manager):
        """Test exists for missing key."""
        connected_manager._client.exists.return_value = 0

        result = await connected_manager.exists("missing")

        assert result is False

    @pytest.mark.asyncio
    async def test_expire(self, connected_manager):
        """Test expire."""
        connected_manager._client.expire.return_value = True

        result = await connected_manager.expire("key", 120)

        assert result is True
        connected_manager._client.expire.assert_awaited_once_with("trading:key", 120)

    @pytest.mark.asyncio
    async def test_ttl(self, connected_manager):
        """Test ttl."""
        connected_manager._client.ttl.return_value = 55

        result = await connected_manager.ttl("key")

        assert result == 55

    @pytest.mark.asyncio
    async def test_keys(self, connected_manager):
        """Test keys pattern matching."""
        connected_manager._client.keys.return_value = [
            "trading:price:BTC",
            "trading:price:ETH",
        ]

        result = await connected_manager.keys("price:*")

        assert result == ["price:BTC", "price:ETH"]

    @pytest.mark.asyncio
    async def test_mget(self, connected_manager):
        """Test mget."""
        connected_manager._client.mget.return_value = ["value1", '{"a": 1}', None]

        result = await connected_manager.mget(["k1", "k2", "k3"])

        assert result == ["value1", {"a": 1}, None]

    @pytest.mark.asyncio
    async def test_mset(self, connected_manager):
        """Test mset."""
        connected_manager._client.mset.return_value = True

        result = await connected_manager.mset({"k1": "v1", "k2": {"a": 1}})

        assert result is True


class TestRedisManagerPubSub:
    """Test RedisManager Pub/Sub operations."""

    @pytest.fixture
    def connected_manager(self):
        """Create connected manager mock."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._connected = True
        return manager

    @pytest.mark.asyncio
    async def test_publish(self, connected_manager):
        """Test publish."""
        connected_manager._client.publish.return_value = 3

        result = await connected_manager.publish("channel", {"msg": "test"})

        assert result == 3
        connected_manager._client.publish.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_subscribe(self, connected_manager):
        """Test subscribe."""
        mock_pubsub = AsyncMock()
        # pubsub() is a sync method that returns a PubSub object
        connected_manager._client.pubsub = MagicMock(return_value=mock_pubsub)

        callback = MagicMock()
        await connected_manager.subscribe("channel", callback)

        assert "trading:channel" in connected_manager._subscriptions
        mock_pubsub.subscribe.assert_awaited_once_with("trading:channel")

    @pytest.mark.asyncio
    async def test_unsubscribe(self, connected_manager):
        """Test unsubscribe."""
        mock_pubsub = AsyncMock()
        connected_manager._pubsub = mock_pubsub
        connected_manager._subscriptions["trading:channel"] = MagicMock()

        await connected_manager.unsubscribe("channel")

        mock_pubsub.unsubscribe.assert_awaited_once_with("trading:channel")
        assert "trading:channel" not in connected_manager._subscriptions


class TestRedisManagerHashOperations:
    """Test RedisManager hash operations."""

    @pytest.fixture
    def connected_manager(self):
        """Create connected manager mock."""
        manager = RedisManager()
        manager._client = AsyncMock()
        manager._connected = True
        return manager

    @pytest.mark.asyncio
    async def test_hget(self, connected_manager):
        """Test hget."""
        connected_manager._client.hget.return_value = '{"value": 123}'

        result = await connected_manager.hget("hash", "field")

        assert result == {"value": 123}

    @pytest.mark.asyncio
    async def test_hget_none(self, connected_manager):
        """Test hget missing field."""
        connected_manager._client.hget.return_value = None

        result = await connected_manager.hget("hash", "missing")

        assert result is None

    @pytest.mark.asyncio
    async def test_hset(self, connected_manager):
        """Test hset."""
        connected_manager._client.hset.return_value = 1

        result = await connected_manager.hset("hash", "field", {"data": "value"})

        assert result == 1

    @pytest.mark.asyncio
    async def test_hgetall(self, connected_manager):
        """Test hgetall."""
        connected_manager._client.hgetall.return_value = {
            "f1": "v1",
            "f2": '{"a": 1}',
        }

        result = await connected_manager.hgetall("hash")

        assert result == {"f1": "v1", "f2": {"a": 1}}

    @pytest.mark.asyncio
    async def test_hdel(self, connected_manager):
        """Test hdel."""
        connected_manager._client.hdel.return_value = 2

        result = await connected_manager.hdel("hash", "f1", "f2")

        assert result == 2


# =============================================================================
# MarketCache Tests
# =============================================================================


class TestMarketCacheInit:
    """Test MarketCache initialization."""

    def test_init(self):
        """Test initialization."""
        redis = MagicMock()
        cache = MarketCache(redis)
        assert cache._redis is redis


class TestMarketCacheTicker:
    """Test MarketCache ticker operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_set_ticker_dict(self, cache):
        """Test set_ticker with dict."""
        cache._redis.set.return_value = True

        result = await cache.set_ticker("BTCUSDT", {"price": "50000"})

        assert result is True
        cache._redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_ticker_object(self, cache):
        """Test set_ticker with Ticker object."""
        cache._redis.set.return_value = True

        ticker = Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            bid=Decimal("49999"),
            ask=Decimal("50001"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("1000"),
            change_24h=Decimal("2.5"),
            timestamp=datetime.now(timezone.utc),
        )

        result = await cache.set_ticker("BTCUSDT", ticker, ttl=10)

        assert result is True
        call_args = cache._redis.set.call_args
        assert call_args[0][0] == "ticker:BTCUSDT"
        assert call_args[1]["ttl"] == 10

    @pytest.mark.asyncio
    async def test_get_ticker(self, cache):
        """Test get_ticker."""
        cache._redis.get.return_value = {
            "symbol": "BTCUSDT",
            "price": "50000",
            "bid": "49999",
            "ask": "50001",
            "high_24h": "51000",
            "low_24h": "49000",
            "volume_24h": "1000",
            "change_24h": "2.5",
        }

        result = await cache.get_ticker("BTCUSDT")

        assert result["symbol"] == "BTCUSDT"
        assert result["price"] == Decimal("50000")
        assert result["bid"] == Decimal("49999")

    @pytest.mark.asyncio
    async def test_get_ticker_none(self, cache):
        """Test get_ticker with missing data."""
        cache._redis.get.return_value = None

        result = await cache.get_ticker("BTCUSDT")

        assert result is None


class TestMarketCacheOrderbook:
    """Test MarketCache orderbook operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_set_orderbook(self, cache):
        """Test set_orderbook."""
        cache._redis.set.return_value = True

        orderbook = {
            "bids": [[Decimal("49999"), Decimal("1.5")]],
            "asks": [[Decimal("50001"), Decimal("2.0")]],
            "lastUpdateId": 12345,
        }

        result = await cache.set_orderbook("BTCUSDT", orderbook)

        assert result is True
        call_args = cache._redis.set.call_args
        assert call_args[0][0] == "orderbook:BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_orderbook(self, cache):
        """Test get_orderbook."""
        cache._redis.get.return_value = {
            "bids": [["49999", "1.5"]],
            "asks": [["50001", "2.0"]],
            "lastUpdateId": 12345,
        }

        result = await cache.get_orderbook("BTCUSDT")

        assert result["bids"] == [[Decimal("49999"), Decimal("1.5")]]
        assert result["asks"] == [[Decimal("50001"), Decimal("2.0")]]


class TestMarketCachePrice:
    """Test MarketCache price operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_set_price(self, cache):
        """Test set_price."""
        cache._redis.set.return_value = True

        result = await cache.set_price("BTCUSDT", Decimal("50000"))

        assert result is True
        cache._redis.set.assert_awaited_once_with("price:BTCUSDT", "50000", ttl=5)

    @pytest.mark.asyncio
    async def test_get_price(self, cache):
        """Test get_price."""
        cache._redis.get.return_value = "50000"

        result = await cache.get_price("BTCUSDT")

        assert result == Decimal("50000")

    @pytest.mark.asyncio
    async def test_get_price_none(self, cache):
        """Test get_price with missing data."""
        cache._redis.get.return_value = None

        result = await cache.get_price("BTCUSDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_prices(self, cache):
        """Test set_prices."""
        cache._redis.mset.return_value = True
        cache._redis.expire.return_value = True

        prices = {
            "BTCUSDT": Decimal("50000"),
            "ETHUSDT": Decimal("3000"),
        }

        result = await cache.set_prices(prices)

        assert result is True
        cache._redis.mset.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_get_prices(self, cache):
        """Test get_prices."""
        cache._redis.mget.return_value = ["50000", "3000", None]

        result = await cache.get_prices(["BTCUSDT", "ETHUSDT", "XRPUSDT"])

        assert result["BTCUSDT"] == Decimal("50000")
        assert result["ETHUSDT"] == Decimal("3000")
        assert result["XRPUSDT"] is None


class TestMarketCacheKline:
    """Test MarketCache kline operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_set_kline(self, cache):
        """Test set_kline."""
        cache._redis.set.return_value = True

        kline = {
            "open": Decimal("49000"),
            "high": Decimal("51000"),
            "low": Decimal("48500"),
            "close": Decimal("50000"),
            "volume": Decimal("100"),
        }

        result = await cache.set_kline("BTCUSDT", "1h", kline)

        assert result is True
        call_args = cache._redis.set.call_args
        assert call_args[0][0] == "kline:BTCUSDT:1h"

    @pytest.mark.asyncio
    async def test_get_kline(self, cache):
        """Test get_kline."""
        cache._redis.get.return_value = {
            "open": "49000",
            "high": "51000",
            "low": "48500",
            "close": "50000",
            "volume": "100",
        }

        result = await cache.get_kline("BTCUSDT", "1h")

        assert result["open"] == Decimal("49000")
        assert result["close"] == Decimal("50000")


class TestMarketCachePubSub:
    """Test MarketCache Pub/Sub operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_publish_ticker(self, cache):
        """Test publish_ticker."""
        cache._redis.publish.return_value = 2

        result = await cache.publish_ticker("BTCUSDT", {"price": "50000"})

        assert result == 2
        cache._redis.publish.assert_awaited_once_with(
            "ticker:BTCUSDT", {"price": "50000"}
        )

    @pytest.mark.asyncio
    async def test_subscribe_ticker(self, cache):
        """Test subscribe_ticker."""
        callback = MagicMock()

        await cache.subscribe_ticker("BTCUSDT", callback)

        cache._redis.subscribe.assert_awaited_once_with("ticker:BTCUSDT", callback)

    @pytest.mark.asyncio
    async def test_unsubscribe_ticker(self, cache):
        """Test unsubscribe_ticker."""
        await cache.unsubscribe_ticker("BTCUSDT")

        cache._redis.unsubscribe.assert_awaited_once_with("ticker:BTCUSDT")


class TestMarketCacheInvalidation:
    """Test MarketCache cache invalidation."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return MarketCache(redis)

    @pytest.mark.asyncio
    async def test_invalidate_ticker(self, cache):
        """Test invalidate_ticker."""
        cache._redis.delete.return_value = True

        result = await cache.invalidate_ticker("BTCUSDT")

        assert result is True
        cache._redis.delete.assert_awaited_once_with("ticker:BTCUSDT")

    @pytest.mark.asyncio
    async def test_invalidate_orderbook(self, cache):
        """Test invalidate_orderbook."""
        cache._redis.delete.return_value = True

        result = await cache.invalidate_orderbook("BTCUSDT")

        assert result is True
        cache._redis.delete.assert_awaited_once_with("orderbook:BTCUSDT")

    @pytest.mark.asyncio
    async def test_invalidate_price(self, cache):
        """Test invalidate_price."""
        cache._redis.delete.return_value = True

        result = await cache.invalidate_price("BTCUSDT")

        assert result is True
        cache._redis.delete.assert_awaited_once_with("price:BTCUSDT")


# =============================================================================
# AccountCache Tests
# =============================================================================


class TestAccountCacheInit:
    """Test AccountCache initialization."""

    def test_init(self):
        """Test initialization."""
        redis = MagicMock()
        cache = AccountCache(redis)
        assert cache._redis is redis


class TestAccountCacheBalance:
    """Test AccountCache balance operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return AccountCache(redis)

    @pytest.mark.asyncio
    async def test_set_balance_dict(self, cache):
        """Test set_balance with dict."""
        cache._redis.set.return_value = True

        result = await cache.set_balance(
            "bot_1", "USDT", {"free": "1000", "locked": "0", "total": "1000"}
        )

        assert result is True
        cache._redis.set.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_set_balance_object(self, cache):
        """Test set_balance with Balance object."""
        cache._redis.set.return_value = True

        balance = Balance(
            asset="USDT",
            free=Decimal("1000"),
            locked=Decimal("100"),
            total=Decimal("1100"),
        )

        result = await cache.set_balance("bot_1", "USDT", balance)

        assert result is True
        call_args = cache._redis.set.call_args
        assert call_args[0][0] == "balance:bot_1:USDT"

    @pytest.mark.asyncio
    async def test_get_balance(self, cache):
        """Test get_balance."""
        cache._redis.get.return_value = {
            "asset": "USDT",
            "free": "1000",
            "locked": "100",
            "total": "1100",
        }

        result = await cache.get_balance("bot_1", "USDT")

        assert result["asset"] == "USDT"
        assert result["free"] == Decimal("1000")
        assert result["locked"] == Decimal("100")
        assert result["total"] == Decimal("1100")

    @pytest.mark.asyncio
    async def test_get_balance_none(self, cache):
        """Test get_balance with missing data."""
        cache._redis.get.return_value = None

        result = await cache.get_balance("bot_1", "USDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_balances(self, cache):
        """Test set_balances."""
        cache._redis.set.return_value = True

        balances = {
            "USDT": {"free": "1000", "locked": "0", "total": "1000"},
            "BTC": {"free": "1.5", "locked": "0.5", "total": "2.0"},
        }

        result = await cache.set_balances("bot_1", balances)

        assert result is True
        assert cache._redis.set.await_count == 2

    @pytest.mark.asyncio
    async def test_get_balances(self, cache):
        """Test get_balances."""
        cache._redis.get.side_effect = [
            {"asset": "USDT", "free": "1000", "locked": "0", "total": "1000"},
            {"asset": "BTC", "free": "1.5", "locked": "0.5", "total": "2.0"},
        ]

        result = await cache.get_balances("bot_1", ["USDT", "BTC"])

        assert "USDT" in result
        assert "BTC" in result
        assert result["USDT"]["free"] == Decimal("1000")

    @pytest.mark.asyncio
    async def test_invalidate_balance(self, cache):
        """Test invalidate_balance."""
        cache._redis.delete.return_value = True

        result = await cache.invalidate_balance("bot_1", "USDT")

        assert result is True
        cache._redis.delete.assert_awaited_once_with("balance:bot_1:USDT")

    @pytest.mark.asyncio
    async def test_invalidate_all_balances(self, cache):
        """Test invalidate_all_balances."""
        cache._redis.keys.return_value = ["balance:bot_1:USDT", "balance:bot_1:BTC"]
        cache._redis.delete.return_value = True

        result = await cache.invalidate_all_balances("bot_1")

        assert result == 2


class TestAccountCachePosition:
    """Test AccountCache position operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return AccountCache(redis)

    @pytest.mark.asyncio
    async def test_set_position_dict(self, cache):
        """Test set_position with dict."""
        cache._redis.set.return_value = True

        position_data = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": "1.0",
            "entry_price": "50000",
        }

        result = await cache.set_position("bot_1", "BTCUSDT", position_data)

        assert result is True

    @pytest.mark.asyncio
    async def test_set_position_object(self, cache):
        """Test set_position with Position object."""
        cache._redis.set.return_value = True

        position = Position(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("51000"),
            liquidation_price=Decimal("40000"),
            leverage=10,
            margin=Decimal("5000"),
            unrealized_pnl=Decimal("1000"),
            margin_type="CROSSED",
            updated_at=datetime.now(timezone.utc),
        )

        result = await cache.set_position("bot_1", "BTCUSDT", position)

        assert result is True
        call_args = cache._redis.set.call_args
        assert call_args[0][0] == "position:bot_1:BTCUSDT"

    @pytest.mark.asyncio
    async def test_get_position(self, cache):
        """Test get_position."""
        cache._redis.get.return_value = {
            "symbol": "BTCUSDT",
            "side": "LONG",
            "quantity": "1.0",
            "entry_price": "50000",
            "mark_price": "51000",
            "margin": "5000",
            "unrealized_pnl": "1000",
        }

        result = await cache.get_position("bot_1", "BTCUSDT")

        assert result["symbol"] == "BTCUSDT"
        assert result["quantity"] == Decimal("1.0")
        assert result["entry_price"] == Decimal("50000")

    @pytest.mark.asyncio
    async def test_get_position_none(self, cache):
        """Test get_position with missing data."""
        cache._redis.get.return_value = None

        result = await cache.get_position("bot_1", "BTCUSDT")

        assert result is None

    @pytest.mark.asyncio
    async def test_invalidate_position(self, cache):
        """Test invalidate_position."""
        cache._redis.delete.return_value = True

        result = await cache.invalidate_position("bot_1", "BTCUSDT")

        assert result is True
        cache._redis.delete.assert_awaited_once_with("position:bot_1:BTCUSDT")

    @pytest.mark.asyncio
    async def test_invalidate_all_positions(self, cache):
        """Test invalidate_all_positions."""
        cache._redis.keys.return_value = [
            "position:bot_1:BTCUSDT",
            "position:bot_1:ETHUSDT",
        ]
        cache._redis.delete.return_value = True

        result = await cache.invalidate_all_positions("bot_1")

        assert result == 2


class TestAccountCachePubSub:
    """Test AccountCache Pub/Sub operations."""

    @pytest.fixture
    def cache(self):
        """Create cache with mock redis."""
        redis = AsyncMock()
        return AccountCache(redis)

    @pytest.mark.asyncio
    async def test_publish_order_update(self, cache):
        """Test publish_order_update."""
        cache._redis.publish.return_value = 1

        order_data = {"order_id": "123", "status": "FILLED"}
        result = await cache.publish_order_update("bot_1", order_data)

        assert result == 1
        cache._redis.publish.assert_awaited_once_with("order:bot_1", order_data)

    @pytest.mark.asyncio
    async def test_subscribe_order_updates(self, cache):
        """Test subscribe_order_updates."""
        callback = MagicMock()

        await cache.subscribe_order_updates("bot_1", callback)

        cache._redis.subscribe.assert_awaited_once_with("order:bot_1", callback)

    @pytest.mark.asyncio
    async def test_publish_position_update(self, cache):
        """Test publish_position_update."""
        cache._redis.publish.return_value = 1

        position_data = {"symbol": "BTCUSDT", "pnl": "1000"}
        result = await cache.publish_position_update("bot_1", position_data)

        assert result == 1
        cache._redis.publish.assert_awaited_once_with("position:bot_1", position_data)

    @pytest.mark.asyncio
    async def test_subscribe_position_updates(self, cache):
        """Test subscribe_position_updates."""
        callback = MagicMock()

        await cache.subscribe_position_updates("bot_1", callback)

        cache._redis.subscribe.assert_awaited_once_with("position:bot_1", callback)

    @pytest.mark.asyncio
    async def test_publish_alert(self, cache):
        """Test publish_alert."""
        cache._redis.publish.return_value = 3

        alert_data = {"type": "MARGIN_CALL", "level": "warning"}
        result = await cache.publish_alert("bot_1", alert_data)

        assert result == 3
        cache._redis.publish.assert_awaited_once_with("alert:bot_1", alert_data)

    @pytest.mark.asyncio
    async def test_subscribe_alerts(self, cache):
        """Test subscribe_alerts."""
        callback = MagicMock()

        await cache.subscribe_alerts("bot_1", callback)

        cache._redis.subscribe.assert_awaited_once_with("alert:bot_1", callback)


# =============================================================================
# Integration Tests (require Redis)
# =============================================================================


@pytest.fixture
async def redis_manager():
    """Create and connect RedisManager for integration tests."""
    manager = RedisManager(key_prefix="test:")
    connected = await manager.connect()
    if not connected:
        pytest.skip("Redis not available")
    yield manager
    # Cleanup
    keys = await manager.keys("*")
    for key in keys:
        await manager.delete(key)
    await manager.disconnect()


class TestRedisIntegration:
    """Integration tests requiring actual Redis."""

    @pytest.mark.asyncio
    async def test_set_get_roundtrip(self, redis_manager):
        """Test set/get roundtrip."""
        await redis_manager.set("test_key", {"data": "value"}, ttl=10)
        result = await redis_manager.get("test_key")

        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, redis_manager):
        """Test TTL expiry."""
        await redis_manager.set("expiring", "value", ttl=1)

        # Should exist immediately
        assert await redis_manager.exists("expiring") is True

        # Wait for expiry
        await asyncio.sleep(1.5)

        # Should be gone
        assert await redis_manager.exists("expiring") is False

    @pytest.mark.asyncio
    async def test_pubsub_messaging(self, redis_manager):
        """Test Pub/Sub messaging."""
        received_messages = []

        async def callback(channel, message):
            received_messages.append((channel, message))

        await redis_manager.subscribe("test_channel", callback)

        # Give subscription time to establish
        await asyncio.sleep(0.1)

        # Publish message
        await redis_manager.publish("test_channel", {"msg": "hello"})

        # Wait for message
        await asyncio.sleep(0.2)

        # Verify
        assert len(received_messages) > 0
        assert received_messages[0][0] == "test_channel"
        assert received_messages[0][1] == {"msg": "hello"}
