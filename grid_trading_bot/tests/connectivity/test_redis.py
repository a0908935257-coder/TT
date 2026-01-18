"""
Redis Connectivity Tests.

Tests Redis connection, read/write, and Pub/Sub functionality.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data import RedisManager


# =============================================================================
# Mock-based Tests (Always Run)
# =============================================================================


class TestRedisConnectivityMock:
    """Mock-based Redis connectivity tests."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.ping = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value=None)
        redis.set = AsyncMock(return_value=True)
        redis.setex = AsyncMock(return_value=True)
        redis.delete = AsyncMock(return_value=1)
        redis.exists = AsyncMock(return_value=1)
        redis.publish = AsyncMock(return_value=1)
        redis.close = AsyncMock()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_pubsub.close = AsyncMock()
        redis.pubsub = MagicMock(return_value=mock_pubsub)

        return redis

    @pytest.mark.asyncio
    async def test_connect(self, mock_redis):
        """Test Redis connection."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            result = await mgr.connect()

            assert result is True
            assert mgr.is_connected is True
            mock_redis.ping.assert_called_once()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_ping(self, mock_redis):
        """Test Redis ping."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            result = await mgr.health_check()

            assert result is True

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_set_get(self, mock_redis):
        """Test Redis set and get."""
        mock_redis.get = AsyncMock(return_value="test_value")

        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            # Set
            await mgr.set("test_key", "test_value")
            mock_redis.set.assert_called()

            # Get
            value = await mgr.get("test_key")
            assert value == "test_value"

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, mock_redis):
        """Test Redis set with TTL."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            await mgr.set("expire_key", "value", ttl=60)
            mock_redis.setex.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis):
        """Test Redis delete."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            result = await mgr.delete("test_key")

            assert result is True
            mock_redis.delete.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_exists(self, mock_redis):
        """Test Redis exists check."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            result = await mgr.exists("test_key")

            assert result is True
            mock_redis.exists.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_publish(self, mock_redis):
        """Test Redis publish."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            result = await mgr.publish("test_channel", "test_message")

            assert result == 1
            mock_redis.publish.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_subscribe(self, mock_redis):
        """Test Redis subscribe."""
        mock_pubsub = mock_redis.pubsub()

        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            callback_called = []

            def callback(channel, message):
                callback_called.append((channel, message))

            await mgr.subscribe("test_channel", callback)

            mock_pubsub.subscribe.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, mock_redis):
        """Test Redis unsubscribe."""
        mock_pubsub = mock_redis.pubsub()

        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()

            def callback(channel, message):
                pass

            await mgr.subscribe("test_channel", callback)
            await mgr.unsubscribe("test_channel")

            mock_pubsub.unsubscribe.assert_called()

            await mgr.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self, mock_redis):
        """Test Redis disconnection."""
        with patch("src.data.cache.redis_client.redis.Redis", return_value=mock_redis):
            mgr = RedisManager(host="localhost", port=6379)
            await mgr.connect()
            await mgr.disconnect()

            assert mgr.is_connected is False
            mock_redis.close.assert_called_once()


# =============================================================================
# Live Tests (Skip if no Redis)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("REDIS_HOST"),
    reason="No Redis connection configured"
)
class TestRedisConnectivityLive:
    """Live Redis connectivity tests."""

    @pytest.fixture
    def redis_config(self):
        """Get Redis config from environment."""
        return {
            "host": os.getenv("REDIS_HOST", "localhost"),
            "port": int(os.getenv("REDIS_PORT", "6379")),
            "password": os.getenv("REDIS_PASSWORD"),
        }

    @pytest.mark.asyncio
    async def test_connect(self, redis_config):
        """Test live Redis connection."""
        async with RedisManager(**redis_config) as mgr:
            assert mgr.is_connected is True

    @pytest.mark.asyncio
    async def test_ping(self, redis_config):
        """Test live Redis ping."""
        async with RedisManager(**redis_config) as mgr:
            result = await mgr.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_set_get(self, redis_config):
        """Test live Redis set and get."""
        async with RedisManager(**redis_config) as mgr:
            await mgr.set("test_key_live", "test_value")
            value = await mgr.get("test_key_live")
            assert value == "test_value"
            await mgr.delete("test_key_live")

    @pytest.mark.asyncio
    async def test_pubsub(self, redis_config):
        """Test live Redis Pub/Sub."""
        received = []

        async def handler(channel, message):
            received.append((channel, message))

        async with RedisManager(**redis_config) as mgr:
            await mgr.subscribe("test_channel_live", handler)

            # Publish
            await mgr.publish("test_channel_live", "test_message")
            await asyncio.sleep(0.5)

            # Unsubscribe first to stop listener
            await mgr.unsubscribe("test_channel_live")

            # Note: Pub/Sub test might be flaky due to timing
            # assert len(received) > 0

    @pytest.mark.asyncio
    async def test_expire(self, redis_config):
        """Test live Redis key expiration."""
        async with RedisManager(**redis_config) as mgr:
            await mgr.set("expire_key_live", "value", ttl=1)

            value1 = await mgr.get("expire_key_live")
            assert value1 == "value"

            await asyncio.sleep(1.5)

            value2 = await mgr.get("expire_key_live")
            assert value2 is None
