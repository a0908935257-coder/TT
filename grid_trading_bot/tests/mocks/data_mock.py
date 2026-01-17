"""
Mock Data Manager and Notifier for testing.

Provides mock implementations of data services that store data in memory.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from core.models import Kline, MarketType, Order


class MockDataManager:
    """
    Mock Data Manager.

    Stores data in memory for testing:
    - Price data
    - Kline data
    - Order storage
    - Bot state persistence

    Example:
        >>> data_manager = MockDataManager()
        >>> data_manager.set_price("BTCUSDT", 50000)
        >>> price = await data_manager.get_price("BTCUSDT")
    """

    def __init__(self):
        """Initialize mock data manager."""
        # Price storage: symbol -> price
        self._prices: dict[str, Decimal] = {}

        # Kline storage: symbol -> list of klines
        self._klines: dict[str, list[Kline]] = {}

        # Order storage: order_id -> order data
        self._orders: dict[str, dict[str, Any]] = {}

        # Bot state storage: bot_id -> state data
        self._bot_states: dict[str, dict[str, Any]] = {}

        # Connection status
        self._connected: bool = True

        # Simulated latency
        self._latency: float = 0.01

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if data manager is connected."""
        return self._connected

    # =========================================================================
    # Price Data
    # =========================================================================

    def set_price(
        self,
        symbol: str,
        price: Decimal | float | str,
        market_type: MarketType = MarketType.SPOT,
    ) -> None:
        """
        Set price for a symbol.

        Args:
            symbol: Trading pair symbol
            price: Price value
            market_type: Market type
        """
        key = f"{symbol}_{market_type.value}"
        self._prices[key] = Decimal(str(price))

    async def get_price(
        self,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Decimal]:
        """
        Get price for a symbol.

        Args:
            symbol: Trading pair symbol
            market_type: Market type

        Returns:
            Price if available
        """
        await asyncio.sleep(self._latency)
        key = f"{symbol}_{market_type.value}"
        return self._prices.get(key)

    # =========================================================================
    # Kline Data
    # =========================================================================

    def set_klines(
        self,
        symbol: str,
        klines: list[Kline],
        market_type: MarketType = MarketType.SPOT,
    ) -> None:
        """
        Set kline data for a symbol.

        Args:
            symbol: Trading pair symbol
            klines: List of Kline objects
            market_type: Market type
        """
        key = f"{symbol}_{market_type.value}"
        self._klines[key] = klines

    async def get_klines(
        self,
        symbol: str,
        timeframe: str = "4h",
        limit: int = 100,
        market_type: MarketType = MarketType.SPOT,
    ) -> list[Kline]:
        """
        Get kline data for a symbol.

        Args:
            symbol: Trading pair symbol
            timeframe: Kline timeframe
            limit: Maximum number of klines
            market_type: Market type

        Returns:
            List of Kline objects
        """
        await asyncio.sleep(self._latency)
        key = f"{symbol}_{market_type.value}"
        klines = self._klines.get(key, [])
        return klines[-limit:] if len(klines) > limit else klines

    # =========================================================================
    # Order Storage
    # =========================================================================

    async def save_order(
        self,
        order: Order,
        bot_id: str = "",
        market_type: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Save order to storage.

        Args:
            order: Order to save
            bot_id: Bot identifier
            market_type: Market type

        Returns:
            True if saved successfully
        """
        await asyncio.sleep(self._latency)

        self._orders[order.order_id] = {
            "order": order,
            "bot_id": bot_id,
            "market_type": market_type.value,
            "saved_at": datetime.now(timezone.utc),
        }
        return True

    async def update_order(
        self,
        order: Order,
        bot_id: str = "",
    ) -> bool:
        """
        Update order in storage.

        Args:
            order: Order to update
            bot_id: Bot identifier

        Returns:
            True if updated successfully
        """
        await asyncio.sleep(self._latency)

        if order.order_id in self._orders:
            self._orders[order.order_id]["order"] = order
            self._orders[order.order_id]["updated_at"] = datetime.now(timezone.utc)
            return True
        return False

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order if found
        """
        await asyncio.sleep(self._latency)

        data = self._orders.get(order_id)
        if data:
            return data["order"]
        return None

    async def get_open_orders(
        self,
        symbol: str,
        bot_id: str = "",
    ) -> list[Order]:
        """
        Get open orders for a symbol.

        Args:
            symbol: Trading pair symbol
            bot_id: Bot identifier (optional filter)

        Returns:
            List of open orders
        """
        await asyncio.sleep(self._latency)

        orders = []
        for data in self._orders.values():
            order = data["order"]
            if order.symbol == symbol and order.status.value in ("NEW", "PARTIALLY_FILLED"):
                if not bot_id or data.get("bot_id") == bot_id:
                    orders.append(order)
        return orders

    # =========================================================================
    # Bot State Persistence
    # =========================================================================

    async def save_bot_state(
        self,
        bot_id: str,
        state_data: dict[str, Any],
    ) -> bool:
        """
        Save bot state to storage.

        Args:
            bot_id: Bot identifier
            state_data: State data to save

        Returns:
            True if saved successfully
        """
        await asyncio.sleep(self._latency)

        self._bot_states[bot_id] = {
            **state_data,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        return True

    async def load_bot_state(
        self,
        bot_id: str,
    ) -> Optional[dict[str, Any]]:
        """
        Load bot state from storage.

        Args:
            bot_id: Bot identifier

        Returns:
            State data if found
        """
        await asyncio.sleep(self._latency)
        return self._bot_states.get(bot_id)

    async def delete_bot_state(self, bot_id: str) -> bool:
        """
        Delete bot state from storage.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted
        """
        await asyncio.sleep(self._latency)

        if bot_id in self._bot_states:
            del self._bot_states[bot_id]
            return True
        return False

    # =========================================================================
    # Test Helpers
    # =========================================================================

    def set_connected(self, connected: bool) -> None:
        """Set connection status."""
        self._connected = connected

    def set_latency(self, latency: float) -> None:
        """Set simulated latency in seconds."""
        self._latency = latency

    def reset(self) -> None:
        """Reset all mock state."""
        self._prices.clear()
        self._klines.clear()
        self._orders.clear()
        self._bot_states.clear()
        self._connected = True

    def get_stored_orders(self) -> dict[str, dict[str, Any]]:
        """Get all stored orders."""
        return self._orders.copy()

    def get_stored_bot_states(self) -> dict[str, dict[str, Any]]:
        """Get all stored bot states."""
        return self._bot_states.copy()


class MockNotifier:
    """
    Mock Notification Manager.

    Records all notifications for verification in tests.
    Does not actually send any notifications.

    Example:
        >>> notifier = MockNotifier()
        >>> await notifier.send_info("Test", "Hello")
        >>> assert len(notifier.messages) == 1
        >>> assert notifier.get_last_message()["title"] == "Test"
    """

    def __init__(self):
        """Initialize mock notifier."""
        self.messages: list[dict[str, Any]] = []

    async def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        **kwargs: Any,
    ) -> bool:
        """
        Record a notification.

        Args:
            title: Notification title
            message: Notification message
            level: Notification level
            **kwargs: Additional parameters

        Returns:
            True (always succeeds)
        """
        self.messages.append({
            "title": title,
            "message": message,
            "level": level,
            "timestamp": datetime.now(timezone.utc),
            **kwargs,
        })
        return True

    async def send_info(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """Send info level notification."""
        return await self.send(title, message, level="info", **kwargs)

    async def send_success(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """Send success level notification."""
        return await self.send(title, message, level="success", **kwargs)

    async def send_warning(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """Send warning level notification."""
        return await self.send(title, message, level="warning", **kwargs)

    async def send_error(
        self,
        title: str,
        message: str,
        **kwargs: Any,
    ) -> bool:
        """Send error level notification."""
        return await self.send(title, message, level="error", **kwargs)

    def get_last_message(self) -> Optional[dict[str, Any]]:
        """
        Get the last notification message.

        Returns:
            Last message dict or None if no messages
        """
        if self.messages:
            return self.messages[-1]
        return None

    def get_messages_by_level(self, level: str) -> list[dict[str, Any]]:
        """
        Get all messages of a specific level.

        Args:
            level: Notification level to filter

        Returns:
            List of messages matching the level
        """
        return [m for m in self.messages if m.get("level") == level]

    def get_messages_by_title(self, title: str) -> list[dict[str, Any]]:
        """
        Get all messages with a specific title.

        Args:
            title: Title to filter

        Returns:
            List of messages matching the title
        """
        return [m for m in self.messages if m.get("title") == title]

    def clear(self) -> None:
        """Clear all recorded messages."""
        self.messages = []

    def count(self) -> int:
        """Get total message count."""
        return len(self.messages)

    def has_message_containing(self, text: str) -> bool:
        """
        Check if any message contains the specified text.

        Args:
            text: Text to search for

        Returns:
            True if any message contains the text
        """
        for msg in self.messages:
            if text in msg.get("message", "") or text in msg.get("title", ""):
                return True
        return False
