"""
Redis Connection Manager.

Provides async Redis connection management with support for
caching, Pub/Sub messaging, and key prefix management.
"""

import asyncio
import json
from typing import Any, Callable, Optional, Union

import redis.asyncio as redis
from redis.asyncio.client import PubSub

from core import get_logger

logger = get_logger(__name__)


class RedisManager:
    """
    Async Redis connection manager.

    Provides connection management, key-value operations, and Pub/Sub support
    with automatic key prefix management.

    Example:
        >>> redis_mgr = RedisManager(host="localhost", port=6379)
        >>> await redis_mgr.connect()
        >>>
        >>> await redis_mgr.set("my_key", {"data": "value"}, ttl=60)
        >>> data = await redis_mgr.get("my_key")
        >>>
        >>> await redis_mgr.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        key_prefix: str = "trading:",
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
        max_connections: int = 10,
    ):
        """
        Initialize RedisManager.

        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password (optional)
            key_prefix: Prefix for all keys
            socket_timeout: Socket timeout in seconds
            socket_connect_timeout: Connection timeout in seconds
            max_connections: Maximum connections in pool
        """
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._key_prefix = key_prefix
        self._socket_timeout = socket_timeout
        self._socket_connect_timeout = socket_connect_timeout
        self._max_connections = max_connections

        self._client: Optional[redis.Redis] = None
        self._pubsub: Optional[PubSub] = None
        self._connected = False

        # Subscription tracking
        self._subscriptions: dict[str, Callable] = {}
        self._pubsub_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self._connected and self._client is not None

    @property
    def key_prefix(self) -> str:
        """Get the key prefix."""
        return self._key_prefix

    @property
    def client(self) -> Optional[redis.Redis]:
        """Get the Redis client."""
        return self._client

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish Redis connection.

        Returns:
            True if connection successful
        """
        if self._connected:
            logger.debug("Already connected to Redis")
            return True

        try:
            logger.info(f"Connecting to Redis at {self._host}:{self._port}")

            self._client = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                socket_timeout=self._socket_timeout,
                socket_connect_timeout=self._socket_connect_timeout,
                max_connections=self._max_connections,
                decode_responses=True,  # Return strings instead of bytes
            )

            # Test connection
            if await self.health_check():
                self._connected = True
                logger.info("Redis connection established")
                return True
            else:
                logger.error("Redis health check failed")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close Redis connection."""
        logger.info("Disconnecting from Redis")

        # Stop pubsub listener if running
        if self._pubsub_task:
            self._pubsub_task.cancel()
            try:
                await self._pubsub_task
            except asyncio.CancelledError:
                pass

        # Close pubsub
        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        # Close client
        if self._client:
            await self._client.close()
            self._client = None

        self._connected = False
        self._subscriptions.clear()
        logger.info("Redis disconnected")

    async def __aenter__(self) -> "RedisManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def health_check(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if Redis is healthy
        """
        try:
            if not self._client:
                return False
            result = await self._client.ping()
            return result is True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    # =========================================================================
    # Key Management
    # =========================================================================

    def _make_key(self, key: str) -> str:
        """
        Create full key with prefix.

        Args:
            key: Key without prefix

        Returns:
            Full key with prefix
        """
        return f"{self._key_prefix}{key}"

    # =========================================================================
    # Basic Operations
    # =========================================================================

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value by key.

        Args:
            key: Key (without prefix)

        Returns:
            Deserialized value or None if not found
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)
        value = await self._client.get(full_key)

        if value is None:
            return None

        # Try to deserialize JSON
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Set value by key.

        Args:
            key: Key (without prefix)
            value: Value to store (will be JSON serialized if dict/list)
            ttl: Time to live in seconds (optional)

        Returns:
            True if successful
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)

        # Serialize to JSON if needed
        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        elif not isinstance(value, str):
            value = str(value)

        if ttl:
            result = await self._client.setex(full_key, ttl, value)
        else:
            result = await self._client.set(full_key, value)

        return bool(result)

    async def delete(self, key: str) -> bool:
        """
        Delete key.

        Args:
            key: Key (without prefix)

        Returns:
            True if key was deleted
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)
        result = await self._client.delete(full_key)
        return result > 0

    async def exists(self, key: str) -> bool:
        """
        Check if key exists.

        Args:
            key: Key (without prefix)

        Returns:
            True if key exists
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)
        result = await self._client.exists(full_key)
        return result > 0

    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time on key.

        Args:
            key: Key (without prefix)
            ttl: Time to live in seconds

        Returns:
            True if timeout was set
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)
        result = await self._client.expire(full_key, ttl)
        return bool(result)

    async def ttl(self, key: str) -> int:
        """
        Get remaining TTL for key.

        Args:
            key: Key (without prefix)

        Returns:
            TTL in seconds, -1 if no expiry, -2 if key doesn't exist
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_key = self._make_key(key)
        return await self._client.ttl(full_key)

    async def keys(self, pattern: str = "*") -> list[str]:
        """
        Get keys matching pattern.

        Args:
            pattern: Pattern to match (without prefix)

        Returns:
            List of keys (without prefix)
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_pattern = self._make_key(pattern)
        keys = await self._client.keys(full_pattern)

        # Remove prefix from returned keys
        prefix_len = len(self._key_prefix)
        return [k[prefix_len:] for k in keys]

    async def mget(self, keys: list[str]) -> list[Optional[Any]]:
        """
        Get multiple values by keys.

        Args:
            keys: List of keys (without prefix)

        Returns:
            List of values (None for missing keys)
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_keys = [self._make_key(k) for k in keys]
        values = await self._client.mget(full_keys)

        results = []
        for value in values:
            if value is None:
                results.append(None)
            else:
                try:
                    results.append(json.loads(value))
                except (json.JSONDecodeError, TypeError):
                    results.append(value)

        return results

    async def mset(self, mapping: dict[str, Any]) -> bool:
        """
        Set multiple key-value pairs.

        Args:
            mapping: Dict of key-value pairs (keys without prefix)

        Returns:
            True if successful
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_mapping = {}
        for key, value in mapping.items():
            full_key = self._make_key(key)
            if isinstance(value, (dict, list)):
                full_mapping[full_key] = json.dumps(value, default=str)
            elif not isinstance(value, str):
                full_mapping[full_key] = str(value)
            else:
                full_mapping[full_key] = value

        result = await self._client.mset(full_mapping)
        return bool(result)

    # =========================================================================
    # Pub/Sub Operations
    # =========================================================================

    async def publish(self, channel: str, message: Any) -> int:
        """
        Publish message to channel.

        Args:
            channel: Channel name (without prefix)
            message: Message to publish (will be JSON serialized if dict/list)

        Returns:
            Number of subscribers that received the message
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_channel = self._make_key(channel)

        # Serialize to JSON if needed
        if isinstance(message, (dict, list)):
            message = json.dumps(message, default=str)
        elif not isinstance(message, str):
            message = str(message)

        return await self._client.publish(full_channel, message)

    async def subscribe(
        self,
        channel: str,
        callback: Callable[[str, Any], None],
    ) -> None:
        """
        Subscribe to channel with callback.

        Args:
            channel: Channel name (without prefix)
            callback: Callback function(channel, message)
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_channel = self._make_key(channel)

        # Initialize pubsub if needed
        if self._pubsub is None:
            self._pubsub = self._client.pubsub()

        # Store callback
        self._subscriptions[full_channel] = callback

        # Subscribe to channel
        await self._pubsub.subscribe(full_channel)
        logger.debug(f"Subscribed to channel: {channel}")

        # Start listener task if not running
        if self._pubsub_task is None or self._pubsub_task.done():
            self._pubsub_task = asyncio.create_task(self._pubsub_listener())

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from channel.

        Args:
            channel: Channel name (without prefix)
        """
        if self._pubsub is None:
            return

        full_channel = self._make_key(channel)

        await self._pubsub.unsubscribe(full_channel)
        self._subscriptions.pop(full_channel, None)
        logger.debug(f"Unsubscribed from channel: {channel}")

    async def _pubsub_listener(self) -> None:
        """Listen for pubsub messages and dispatch to callbacks."""
        try:
            while self._pubsub and self._subscriptions:
                message = await self._pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message is None:
                    continue

                channel = message.get("channel")
                data = message.get("data")

                if channel and channel in self._subscriptions:
                    # Try to deserialize JSON
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass

                    # Get original channel name (without prefix)
                    prefix_len = len(self._key_prefix)
                    original_channel = channel[prefix_len:]

                    # Call callback
                    callback = self._subscriptions[channel]
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(original_channel, data)
                        else:
                            callback(original_channel, data)
                    except Exception as e:
                        logger.error(f"Callback error for {original_channel}: {e}")

        except asyncio.CancelledError:
            logger.debug("Pubsub listener cancelled")
            raise
        except Exception as e:
            logger.error(f"Pubsub listener error: {e}")

    # =========================================================================
    # Hash Operations (useful for structured data)
    # =========================================================================

    async def hget(self, name: str, key: str) -> Optional[Any]:
        """
        Get value from hash.

        Args:
            name: Hash name (without prefix)
            key: Field key

        Returns:
            Value or None
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_name = self._make_key(name)
        value = await self._client.hget(full_name, key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value

    async def hset(self, name: str, key: str, value: Any) -> int:
        """
        Set value in hash.

        Args:
            name: Hash name (without prefix)
            key: Field key
            value: Value to store

        Returns:
            1 if new field, 0 if updated
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_name = self._make_key(name)

        if isinstance(value, (dict, list)):
            value = json.dumps(value, default=str)
        elif not isinstance(value, str):
            value = str(value)

        return await self._client.hset(full_name, key, value)

    async def hgetall(self, name: str) -> dict[str, Any]:
        """
        Get all fields from hash.

        Args:
            name: Hash name (without prefix)

        Returns:
            Dict of all fields and values
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_name = self._make_key(name)
        data = await self._client.hgetall(full_name)

        result = {}
        for key, value in data.items():
            try:
                result[key] = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                result[key] = value

        return result

    async def hdel(self, name: str, *keys: str) -> int:
        """
        Delete fields from hash.

        Args:
            name: Hash name (without prefix)
            keys: Fields to delete

        Returns:
            Number of fields deleted
        """
        if not self._client:
            raise RuntimeError("Redis not connected")

        full_name = self._make_key(name)
        return await self._client.hdel(full_name, *keys)
