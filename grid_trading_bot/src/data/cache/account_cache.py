"""
Account Data Cache.

Provides caching for account data including balances and positions.
"""

from decimal import Decimal
from typing import Any, Optional

from core import get_logger
from core.models import Balance, Position

from .redis_client import RedisManager

logger = get_logger(__name__)


# Default TTL values (in seconds)
BALANCE_TTL = 60
POSITION_TTL = 30


class AccountCache:
    """
    Account data cache using Redis.

    Caches balances and positions with appropriate TTLs
    for fast access to account state.

    Example:
        >>> cache = AccountCache(redis_manager)
        >>> await cache.set_balance("bot_1", "USDT", balance)
        >>> balance = await cache.get_balance("bot_1", "USDT")
    """

    def __init__(self, redis: RedisManager):
        """
        Initialize AccountCache.

        Args:
            redis: RedisManager instance
        """
        self._redis = redis

    # =========================================================================
    # Balance Cache
    # =========================================================================

    async def set_balance(
        self,
        bot_id: str,
        asset: str,
        balance: Balance | dict[str, Any],
        ttl: int = BALANCE_TTL,
    ) -> bool:
        """
        Cache balance data.

        Args:
            bot_id: Bot identifier
            asset: Asset name (e.g., "USDT")
            balance: Balance object or dict
            ttl: Time to live in seconds (default 60)

        Returns:
            True if successful
        """
        key = f"balance:{bot_id}:{asset.upper()}"

        if isinstance(balance, Balance):
            data = {
                "asset": balance.asset,
                "free": str(balance.free),
                "locked": str(balance.locked),
                "total": str(balance.total),
            }
        else:
            data = balance

        return await self._redis.set(key, data, ttl=ttl)

    async def get_balance(
        self,
        bot_id: str,
        asset: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get cached balance data.

        Args:
            bot_id: Bot identifier
            asset: Asset name

        Returns:
            Balance data dict or None if not cached
        """
        key = f"balance:{bot_id}:{asset.upper()}"
        data = await self._redis.get(key)

        if data and isinstance(data, dict):
            # Convert string decimals back
            for field in ["free", "locked", "total"]:
                if field in data and data[field]:
                    data[field] = Decimal(data[field])

        return data

    async def set_balances(
        self,
        bot_id: str,
        balances: dict[str, Balance | dict[str, Any]],
        ttl: int = BALANCE_TTL,
    ) -> bool:
        """
        Cache multiple balances.

        Args:
            bot_id: Bot identifier
            balances: Dict of asset -> balance
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        for asset, balance in balances.items():
            await self.set_balance(bot_id, asset, balance, ttl)
        return True

    async def get_balances(
        self,
        bot_id: str,
        assets: list[str],
    ) -> dict[str, Optional[dict[str, Any]]]:
        """
        Get multiple cached balances.

        Args:
            bot_id: Bot identifier
            assets: List of asset names

        Returns:
            Dict of asset -> balance data
        """
        result = {}
        for asset in assets:
            result[asset] = await self.get_balance(bot_id, asset)
        return result

    async def invalidate_balance(self, bot_id: str, asset: str) -> bool:
        """
        Invalidate cached balance.

        Args:
            bot_id: Bot identifier
            asset: Asset name

        Returns:
            True if deleted
        """
        key = f"balance:{bot_id}:{asset.upper()}"
        return await self._redis.delete(key)

    async def invalidate_all_balances(self, bot_id: str) -> int:
        """
        Invalidate all cached balances for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Number of keys deleted
        """
        pattern = f"balance:{bot_id}:*"
        keys = await self._redis.keys(pattern)

        count = 0
        for key in keys:
            if await self._redis.delete(key):
                count += 1
        return count

    # =========================================================================
    # Position Cache
    # =========================================================================

    async def set_position(
        self,
        bot_id: str,
        symbol: str,
        position: Position | dict[str, Any],
        ttl: int = POSITION_TTL,
    ) -> bool:
        """
        Cache position data.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair (e.g., "BTCUSDT")
            position: Position object or dict
            ttl: Time to live in seconds (default 30)

        Returns:
            True if successful
        """
        key = f"position:{bot_id}:{symbol.upper()}"

        if isinstance(position, Position):
            data = {
                "symbol": position.symbol,
                "side": position.side if isinstance(position.side, str) else position.side.value,
                "quantity": str(position.quantity),
                "entry_price": str(position.entry_price),
                "mark_price": str(position.mark_price),
                "liquidation_price": str(position.liquidation_price) if position.liquidation_price else None,
                "leverage": position.leverage,
                "margin": str(position.margin),
                "unrealized_pnl": str(position.unrealized_pnl),
                "margin_type": position.margin_type,
                "updated_at": position.updated_at.isoformat(),
            }
        else:
            data = position

        return await self._redis.set(key, data, ttl=ttl)

    async def get_position(
        self,
        bot_id: str,
        symbol: str,
    ) -> Optional[dict[str, Any]]:
        """
        Get cached position data.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair

        Returns:
            Position data dict or None if not cached
        """
        key = f"position:{bot_id}:{symbol.upper()}"
        data = await self._redis.get(key)

        if data and isinstance(data, dict):
            # Convert string decimals back
            for field in ["quantity", "entry_price", "mark_price", "margin", "unrealized_pnl"]:
                if field in data and data[field]:
                    data[field] = Decimal(data[field])
            if "liquidation_price" in data and data["liquidation_price"]:
                data["liquidation_price"] = Decimal(data["liquidation_price"])

        return data

    async def set_positions(
        self,
        bot_id: str,
        positions: dict[str, Position | dict[str, Any]],
        ttl: int = POSITION_TTL,
    ) -> bool:
        """
        Cache multiple positions.

        Args:
            bot_id: Bot identifier
            positions: Dict of symbol -> position
            ttl: Time to live in seconds

        Returns:
            True if successful
        """
        for symbol, position in positions.items():
            await self.set_position(bot_id, symbol, position, ttl)
        return True

    async def get_positions(
        self,
        bot_id: str,
        symbols: list[str],
    ) -> dict[str, Optional[dict[str, Any]]]:
        """
        Get multiple cached positions.

        Args:
            bot_id: Bot identifier
            symbols: List of trading pairs

        Returns:
            Dict of symbol -> position data
        """
        result = {}
        for symbol in symbols:
            result[symbol] = await self.get_position(bot_id, symbol)
        return result

    async def invalidate_position(self, bot_id: str, symbol: str) -> bool:
        """
        Invalidate cached position.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair

        Returns:
            True if deleted
        """
        key = f"position:{bot_id}:{symbol.upper()}"
        return await self._redis.delete(key)

    async def invalidate_all_positions(self, bot_id: str) -> int:
        """
        Invalidate all cached positions for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Number of keys deleted
        """
        pattern = f"position:{bot_id}:*"
        keys = await self._redis.keys(pattern)

        count = 0
        for key in keys:
            if await self._redis.delete(key):
                count += 1
        return count

    # =========================================================================
    # Pub/Sub for Account Updates
    # =========================================================================

    async def publish_order_update(
        self,
        bot_id: str,
        order_data: dict[str, Any],
    ) -> int:
        """
        Publish order update.

        Args:
            bot_id: Bot identifier
            order_data: Order data

        Returns:
            Number of subscribers
        """
        channel = f"order:{bot_id}"
        return await self._redis.publish(channel, order_data)

    async def subscribe_order_updates(
        self,
        bot_id: str,
        callback,
    ) -> None:
        """
        Subscribe to order updates.

        Args:
            bot_id: Bot identifier
            callback: Callback function(channel, order_data)
        """
        channel = f"order:{bot_id}"
        await self._redis.subscribe(channel, callback)

    async def publish_position_update(
        self,
        bot_id: str,
        position_data: dict[str, Any],
    ) -> int:
        """
        Publish position update.

        Args:
            bot_id: Bot identifier
            position_data: Position data

        Returns:
            Number of subscribers
        """
        channel = f"position:{bot_id}"
        return await self._redis.publish(channel, position_data)

    async def subscribe_position_updates(
        self,
        bot_id: str,
        callback,
    ) -> None:
        """
        Subscribe to position updates.

        Args:
            bot_id: Bot identifier
            callback: Callback function(channel, position_data)
        """
        channel = f"position:{bot_id}"
        await self._redis.subscribe(channel, callback)

    async def publish_alert(
        self,
        bot_id: str,
        alert_data: dict[str, Any],
    ) -> int:
        """
        Publish alert notification.

        Args:
            bot_id: Bot identifier
            alert_data: Alert data

        Returns:
            Number of subscribers
        """
        channel = f"alert:{bot_id}"
        return await self._redis.publish(channel, alert_data)

    async def subscribe_alerts(
        self,
        bot_id: str,
        callback,
    ) -> None:
        """
        Subscribe to alert notifications.

        Args:
            bot_id: Bot identifier
            callback: Callback function(channel, alert_data)
        """
        channel = f"alert:{bot_id}"
        await self._redis.subscribe(channel, callback)
