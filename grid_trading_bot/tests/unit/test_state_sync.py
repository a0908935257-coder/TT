"""
Unit tests for State Synchronizer.

Tests for:
- State cache with TTL
- Order/Position/Balance state management
- WebSocket update handling
- Conflict detection
- Periodic synchronization
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from src.core.models import MarketType
from src.exchange.state_sync import (
    StateSynchronizer,
    StateCache,
    SyncConfig,
    SyncState,
    SyncEvent,
    OrderState,
    PositionState,
    BalanceState,
    CacheEntry,
    CacheEventType,
    ConflictResolution,
)


# =============================================================================
# Mock Exchange Provider
# =============================================================================


class MockExchangeProvider:
    """Mock exchange data provider for testing."""

    def __init__(self):
        self.orders: List[Dict[str, Any]] = []
        self.positions: List[Dict[str, Any]] = []
        self.balances: Dict[str, Dict[str, Any]] = {}

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> List[Dict[str, Any]]:
        if symbol:
            return [o for o in self.orders if o.get("symbol") == symbol]
        return self.orders

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Dict[str, Any]]:
        for order in self.orders:
            if str(order.get("orderId")) == order_id:
                return order
        return None

    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if symbol:
            return [p for p in self.positions if p.get("symbol") == symbol]
        return self.positions

    async def get_balance(
        self,
        asset: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> Dict[str, Dict[str, Any]]:
        if asset:
            return {asset: self.balances.get(asset, {})}
        return self.balances


# =============================================================================
# State Cache Tests
# =============================================================================


class TestStateCache:
    """Tests for StateCache."""

    @pytest.fixture
    def cache(self):
        """Create test cache."""
        return StateCache[str](default_ttl=60.0)

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache):
        """Test basic set and get."""
        await cache.set("key1", "value1")

        result = await cache.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_cache_get_nonexistent(self, cache):
        """Test getting non-existent key."""
        result = await cache.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_cache_expiry(self):
        """Test cache entry expiration."""
        cache = StateCache[str](default_ttl=0.1)

        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        # Wait for expiry
        await asyncio.sleep(0.15)

        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_custom_ttl(self, cache):
        """Test custom TTL per entry."""
        await cache.set("key1", "value1", ttl=0.1)
        await cache.set("key2", "value2", ttl=10.0)

        await asyncio.sleep(0.15)

        assert await cache.get("key1") is None
        assert await cache.get("key2") == "value2"

    @pytest.mark.asyncio
    async def test_cache_delete(self, cache):
        """Test cache deletion."""
        await cache.set("key1", "value1")
        assert await cache.get("key1") == "value1"

        result = await cache.delete("key1")

        assert result is True
        assert await cache.get("key1") is None

    @pytest.mark.asyncio
    async def test_cache_delete_nonexistent(self, cache):
        """Test deleting non-existent key."""
        result = await cache.delete("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_cache_get_all(self, cache):
        """Test getting all values."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        result = await cache.get_all()

        assert len(result) == 3
        assert result["key1"] == "value1"
        assert result["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_cache_clear(self, cache):
        """Test clearing cache."""
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        count = await cache.clear()

        assert count == 2
        assert cache.size == 0

    @pytest.mark.asyncio
    async def test_cache_version_tracking(self, cache):
        """Test version increments on update."""
        await cache.set("key1", "value1")
        entry1 = await cache.get_entry("key1")
        assert entry1.version == 1

        await cache.set("key1", "value2")
        entry2 = await cache.get_entry("key1")
        assert entry2.version == 2

    @pytest.mark.asyncio
    async def test_cache_event_callback(self, cache):
        """Test event callbacks are triggered."""
        events = []

        def callback(event: SyncEvent):
            events.append(event)

        cache.add_event_callback(callback)

        await cache.set("key1", "value1")
        await cache.set("key1", "value2")
        await cache.delete("key1")

        assert len(events) == 3
        assert events[0].event_type == CacheEventType.ADDED
        assert events[1].event_type == CacheEventType.UPDATED
        assert events[2].event_type == CacheEventType.REMOVED

    @pytest.mark.asyncio
    async def test_cache_source_tracking(self, cache):
        """Test source tracking on entries."""
        await cache.set("key1", "value1", source="rest")
        await cache.set("key2", "value2", source="websocket")

        entry1 = await cache.get_entry("key1")
        entry2 = await cache.get_entry("key2")

        assert entry1.source == "rest"
        assert entry2.source == "websocket"


# =============================================================================
# Order State Tests
# =============================================================================


class TestOrderState:
    """Tests for OrderState."""

    def test_order_is_open(self):
        """Test is_open property."""
        order = OrderState(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status="NEW",
        )

        assert order.is_open is True

        order.status = "FILLED"
        assert order.is_open is False

    def test_order_is_closed(self):
        """Test is_closed property."""
        order = OrderState(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            status="CANCELED",
        )

        assert order.is_closed is True

    def test_order_remaining_quantity(self):
        """Test remaining_quantity calculation."""
        order = OrderState(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
            status="PARTIALLY_FILLED",
            filled_quantity=Decimal("3.0"),
        )

        assert order.remaining_quantity == Decimal("7.0")


# =============================================================================
# State Synchronizer Tests
# =============================================================================


class TestStateSynchronizer:
    """Tests for StateSynchronizer."""

    @pytest.fixture
    def provider(self):
        """Create mock exchange provider."""
        return MockExchangeProvider()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return SyncConfig(
            order_ttl_seconds=300.0,
            sync_interval_seconds=30.0,
            stale_threshold_seconds=120.0,
        )

    @pytest.fixture
    def sync(self, provider, config):
        """Create test synchronizer."""
        return StateSynchronizer(provider, config)

    def test_sync_initialization(self, sync, config):
        """Test synchronizer initializes correctly."""
        assert sync.config == config
        assert sync.sync_state == SyncState.STALE
        assert sync.last_sync is None

    @pytest.mark.asyncio
    async def test_sync_all_empty(self, sync):
        """Test sync with no data."""
        result = await sync.sync_all()

        assert result is True
        assert sync.sync_state == SyncState.SYNCED
        assert sync.last_sync is not None

    @pytest.mark.asyncio
    async def test_sync_orders(self, sync, provider):
        """Test order synchronization."""
        provider.orders = [
            {
                "orderId": "123",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "1.0",
                "price": "50000",
                "status": "NEW",
                "executedQty": "0",
            },
            {
                "orderId": "456",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "MARKET",
                "origQty": "5.0",
                "status": "NEW",
                "executedQty": "0",
            },
        ]

        await sync.sync_all()

        orders = await sync.get_open_orders()

        assert len(orders) == 2

    @pytest.mark.asyncio
    async def test_sync_balances(self, sync, provider):
        """Test balance synchronization."""
        provider.balances = {
            "BTC": {"free": "1.5", "locked": "0.5"},
            "USDT": {"free": "10000", "locked": "0"},
        }

        await sync.sync_all()

        btc = await sync.get_balance("BTC")
        usdt = await sync.get_balance("USDT")

        assert btc is not None
        assert btc.free == Decimal("1.5")
        assert btc.locked == Decimal("0.5")
        assert btc.total == Decimal("2.0")

        assert usdt is not None
        assert usdt.free == Decimal("10000")

    @pytest.mark.asyncio
    async def test_get_open_orders_by_symbol(self, sync, provider):
        """Test getting orders by symbol."""
        provider.orders = [
            {
                "orderId": "123",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "1.0",
                "status": "NEW",
                "executedQty": "0",
            },
            {
                "orderId": "456",
                "symbol": "ETHUSDT",
                "side": "SELL",
                "type": "LIMIT",
                "origQty": "5.0",
                "status": "NEW",
                "executedQty": "0",
            },
        ]

        await sync.sync_all()

        btc_orders = await sync.get_open_orders("BTCUSDT")
        eth_orders = await sync.get_open_orders("ETHUSDT")

        assert len(btc_orders) == 1
        assert len(eth_orders) == 1
        assert btc_orders[0].symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_handle_order_update(self, sync, provider):
        """Test WebSocket order update handling."""
        # Initial order
        provider.orders = [
            {
                "orderId": "123",
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "LIMIT",
                "origQty": "1.0",
                "status": "NEW",
                "executedQty": "0",
            },
        ]
        await sync.sync_all()

        # Simulate WebSocket update
        update_data = {
            "i": "123",
            "s": "BTCUSDT",
            "S": "BUY",
            "o": "LIMIT",
            "q": "1.0",
            "X": "PARTIALLY_FILLED",
            "z": "0.5",
        }

        sync.handle_order_update(update_data)
        await asyncio.sleep(0.1)  # Wait for async handler

        order = await sync.get_order("123")

        assert order is not None
        assert order.status == "PARTIALLY_FILLED"
        assert order.filled_quantity == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_record_order_sent(self, sync):
        """Test recording locally sent order."""
        await sync.record_order_sent(
            order_id="new_123",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        order = await sync.get_order("new_123")

        assert order is not None
        assert order.symbol == "BTCUSDT"
        assert order.status == "NEW"

    @pytest.mark.asyncio
    async def test_record_order_cancelled(self, sync):
        """Test recording order cancellation."""
        await sync.record_order_sent(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        await sync.record_order_cancelled("123")

        order = await sync.get_order("123")

        assert order is not None
        assert order.status == "CANCELED"

    @pytest.mark.asyncio
    async def test_is_stale(self, sync):
        """Test stale detection."""
        # Initially stale (no sync)
        assert sync.is_stale is True

        await sync.sync_all()

        # After sync, not stale
        assert sync.is_stale is False

    @pytest.mark.asyncio
    async def test_event_callbacks(self, sync, provider):
        """Test event callbacks are triggered."""
        events = []

        def callback(event: SyncEvent):
            events.append(event)

        sync.add_event_callback(callback)

        provider.balances = {"BTC": {"free": "1.0", "locked": "0"}}
        await sync.sync_all()

        # Should have events for synced data
        assert len(events) > 0

    @pytest.mark.asyncio
    async def test_track_symbol(self, sync):
        """Test symbol tracking."""
        sync.track_symbol("BTCUSDT")
        sync.track_symbol("ETHUSDT")

        assert "BTCUSDT" in sync._tracked_symbols
        assert "ETHUSDT" in sync._tracked_symbols

        sync.untrack_symbol("BTCUSDT")

        assert "BTCUSDT" not in sync._tracked_symbols

    @pytest.mark.asyncio
    async def test_get_statistics(self, sync, provider):
        """Test statistics retrieval."""
        provider.orders = [{"orderId": "123", "symbol": "BTCUSDT",
                           "side": "BUY", "type": "LIMIT",
                           "origQty": "1", "status": "NEW", "executedQty": "0"}]

        await sync.sync_all()

        stats = sync.get_statistics()

        assert stats["sync_state"] == "synced"
        assert stats["order_cache_size"] == 1
        assert stats["is_stale"] is False

    @pytest.mark.asyncio
    async def test_conflict_detection(self, sync, provider):
        """Test conflict detection between local and exchange."""
        # Set up local order
        await sync.record_order_sent(
            order_id="123",
            client_order_id=None,
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        # Local says filled more
        order = await sync.get_order("123")
        order.filled_quantity = Decimal("1.0")
        order.status = "FILLED"
        await sync._order_cache.set("123", order, source="local")

        # Exchange says less filled
        provider.orders = [{
            "orderId": "123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "origQty": "1.0",
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.5",
        }]

        await sync.sync_all()

        # Should detect conflict
        conflicts = sync.get_conflicts()
        assert len(conflicts) > 0


# =============================================================================
# Position State Tests
# =============================================================================


class TestPositionState:
    """Tests for PositionState."""

    def test_position_is_open(self):
        """Test is_open property."""
        position = PositionState(
            symbol="BTCUSDT",
            side="LONG",
            quantity=Decimal("1.0"),
            entry_price=Decimal("50000"),
        )

        assert position.is_open is True

        position.quantity = Decimal("0")
        assert position.is_open is False


# =============================================================================
# Balance State Tests
# =============================================================================


class TestBalanceState:
    """Tests for BalanceState."""

    def test_balance_total(self):
        """Test total calculation."""
        balance = BalanceState(
            asset="BTC",
            free=Decimal("1.5"),
            locked=Decimal("0.5"),
        )

        assert balance.total == Decimal("2.0")


# =============================================================================
# Cache Entry Tests
# =============================================================================


class TestCacheEntry:
    """Tests for CacheEntry."""

    def test_entry_expiry(self):
        """Test entry expiration check."""
        # Not expired
        entry1 = CacheEntry(
            data="value",
            expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        )
        assert entry1.is_expired is False

        # Expired
        entry2 = CacheEntry(
            data="value",
            expires_at=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        assert entry2.is_expired is True

        # No expiry
        entry3 = CacheEntry(data="value", expires_at=None)
        assert entry3.is_expired is False

    def test_entry_age(self):
        """Test age calculation."""
        entry = CacheEntry(data="value")

        # Just created, age should be ~0
        assert entry.age_seconds < 1.0
