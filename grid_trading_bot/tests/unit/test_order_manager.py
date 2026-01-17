"""
Unit tests for Grid Order Manager.

Tests order placement, fill handling, and profit calculation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import MarketType, Order, OrderSide, OrderStatus, OrderType
from src.strategy.grid import (
    FilledRecord,
    GridConfig,
    GridLevel,
    GridOrderManager,
    GridSetup,
    LevelSide,
    LevelState,
    create_grid_with_manual_range,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


@pytest.fixture
def order_manager(mock_exchange, mock_data_manager, mock_notifier, grid_setup):
    """Create an order manager for testing."""
    manager = GridOrderManager(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
        bot_id="test_bot",
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
    )
    manager.initialize(grid_setup)
    return manager


@pytest.fixture
def initialized_manager(order_manager, mock_data_manager):
    """Order manager with price set in data manager."""
    mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
    return order_manager


class TestOrderManagerInitialization:
    """Tests for order manager initialization."""

    def test_initialize(self, mock_exchange, mock_data_manager, mock_notifier, grid_setup):
        """Test order manager initialization."""
        manager = GridOrderManager(
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
            bot_id="test_bot",
            symbol="BTCUSDT",
        )

        manager.initialize(grid_setup)

        assert manager.setup is not None
        assert manager.symbol == "BTCUSDT"
        assert manager.bot_id == "test_bot"
        assert manager.active_order_count == 0

    def test_initialize_clears_previous_state(self, order_manager, grid_setup):
        """Test that initialize clears previous state."""
        # Simulate some state
        order_manager._level_order_map[0] = "old_order"
        order_manager._filled_history.append(MagicMock())
        order_manager._total_profit = Decimal("100")

        # Re-initialize
        order_manager.initialize(grid_setup)

        assert len(order_manager._level_order_map) == 0
        assert len(order_manager._filled_history) == 0
        assert order_manager._total_profit == Decimal("0")


class TestInitialOrderPlacement:
    """Tests for initial order placement."""

    @pytest.mark.asyncio
    async def test_place_initial_orders(self, initialized_manager):
        """Test placing initial orders."""
        placed = await initialized_manager.place_initial_orders()

        # Should place orders at levels not near current price
        assert placed > 0
        assert initialized_manager.active_order_count > 0

    @pytest.mark.asyncio
    async def test_initial_sides_correct(self, initialized_manager):
        """Test that initial order sides are correct."""
        await initialized_manager.place_initial_orders()

        current_price = Decimal("50000")

        for level_index, order_id in initialized_manager._level_order_map.items():
            level = initialized_manager.get_level(level_index)
            order = initialized_manager._orders.get(order_id)

            if level and order:
                if level.price < current_price:
                    assert order.side == OrderSide.BUY
                else:
                    assert order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_skips_levels_near_current_price(self, initialized_manager):
        """Test that levels near current price are skipped."""
        await initialized_manager.place_initial_orders()

        current_price = Decimal("50000")
        tolerance = initialized_manager.DEFAULT_PRICE_TOLERANCE

        for level in initialized_manager._setup.levels:
            diff_ratio = abs(level.price - current_price) / current_price

            if diff_ratio <= tolerance:
                # This level should not have an order
                assert level.index not in initialized_manager._level_order_map


class TestSingleLevelOperations:
    """Tests for single level order operations."""

    @pytest.mark.asyncio
    async def test_place_order_at_level(self, initialized_manager):
        """Test placing order at a specific level."""
        order = await initialized_manager.place_order_at_level(0, OrderSide.BUY)

        assert order is not None
        assert order.side == OrderSide.BUY
        assert 0 in initialized_manager._level_order_map

    @pytest.mark.asyncio
    async def test_cancel_order_at_level(self, initialized_manager):
        """Test cancelling order at a specific level."""
        # First place an order
        await initialized_manager.place_order_at_level(0, OrderSide.BUY)
        assert 0 in initialized_manager._level_order_map

        # Cancel it
        result = await initialized_manager.cancel_order_at_level(0)

        assert result is True
        assert 0 not in initialized_manager._level_order_map

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_order(self, initialized_manager):
        """Test cancelling order that doesn't exist."""
        result = await initialized_manager.cancel_order_at_level(99)

        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, initialized_manager):
        """Test cancelling all orders."""
        await initialized_manager.place_initial_orders()
        initial_count = initialized_manager.active_order_count
        assert initial_count > 0

        cancelled = await initialized_manager.cancel_all_orders()

        assert cancelled == initial_count
        assert initialized_manager.active_order_count == 0


class TestOrderFillHandling:
    """Tests for order fill handling."""

    @pytest.mark.asyncio
    async def test_on_buy_filled(self, initialized_manager, mock_exchange):
        """Test handling of buy order fill."""
        # Place a buy order
        order = await initialized_manager.place_order_at_level(0, OrderSide.BUY)
        original_order_id = order.order_id

        # Simulate fill
        filled_order = Order(
            order_id=original_order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=order.quantity * order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        reverse_order = await initialized_manager.on_order_filled(filled_order)

        # Verify fill was recorded
        assert len(initialized_manager._filled_history) == 1
        assert initialized_manager._filled_history[0].side == OrderSide.BUY

        # Verify reverse order was placed (SELL at upper level)
        # Note: reverse may be None if at boundary
        if reverse_order:
            assert reverse_order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_on_sell_filled(self, initialized_manager):
        """Test handling of sell order fill."""
        # Place a sell order at a higher level
        last_level = len(initialized_manager._setup.levels) - 2  # Not the very last
        order = await initialized_manager.place_order_at_level(last_level, OrderSide.SELL)

        # Simulate fill
        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=order.quantity * order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        reverse_order = await initialized_manager.on_order_filled(filled_order)

        # Verify fill was recorded
        assert len(initialized_manager._filled_history) == 1
        assert initialized_manager._filled_history[0].side == OrderSide.SELL

        # Verify reverse order was placed (BUY at lower level)
        if reverse_order:
            assert reverse_order.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_reverse_order_placed(self, initialized_manager):
        """Test that reverse orders are placed correctly."""
        # BUY at level 2 -> SELL at level 3
        order = await initialized_manager.place_order_at_level(2, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("0.1"),
            created_at=datetime.now(timezone.utc),
        )

        reverse = await initialized_manager.on_order_filled(filled_order)

        if reverse:
            # Reverse should be at level 3
            reverse_level = initialized_manager.get_level_by_order_id(reverse.order_id)
            assert reverse_level == 3


class TestProfitCalculation:
    """Tests for profit calculation."""

    def test_profit_calculation_basic(self, order_manager):
        """Test basic profit calculation."""
        buy_record = FilledRecord(
            level_index=5,
            side=OrderSide.BUY,
            price=Decimal("49000"),
            quantity=Decimal("0.1"),
            fee=Decimal("4.9"),  # 0.1%
            timestamp=MagicMock(),
            order_id="buy_001",
        )

        sell_record = FilledRecord(
            level_index=6,
            side=OrderSide.SELL,
            price=Decimal("51000"),
            quantity=Decimal("0.1"),
            fee=Decimal("5.1"),  # 0.1%
            timestamp=MagicMock(),
            order_id="sell_001",
        )

        profit = order_manager.calculate_profit(buy_record, sell_record)

        # Profit = (51000 - 49000) * 0.1 - (4.9 + 5.1) = 200 - 10 = 190
        expected = (Decimal("51000") - Decimal("49000")) * Decimal("0.1") - Decimal("10")
        assert profit == expected

    def test_profit_with_different_quantities(self, order_manager):
        """Test profit calculation with different quantities (uses minimum)."""
        buy_record = FilledRecord(
            level_index=5,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.15"),  # More
            fee=Decimal("7.5"),
            timestamp=MagicMock(),
            order_id="buy_001",
        )

        sell_record = FilledRecord(
            level_index=6,
            side=OrderSide.SELL,
            price=Decimal("52000"),
            quantity=Decimal("0.10"),  # Less
            fee=Decimal("5.2"),
            timestamp=MagicMock(),
            order_id="sell_001",
        )

        profit = order_manager.calculate_profit(buy_record, sell_record)

        # Uses min quantity = 0.10
        # Gross = (52000 - 50000) * 0.10 = 200
        # Proportional fees: buy_fee * (0.10/0.15) + sell_fee * (0.10/0.10)
        assert profit > 0


class TestBoundaryLevels:
    """Tests for boundary level handling."""

    @pytest.mark.asyncio
    async def test_no_reverse_at_highest_level(self, initialized_manager):
        """Test that no reverse order is placed at highest level."""
        highest_index = len(initialized_manager._setup.levels) - 1

        # Place buy at highest level
        order = await initialized_manager.place_order_at_level(highest_index, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("0.1"),
            created_at=datetime.now(timezone.utc),
        )

        reverse = await initialized_manager.on_order_filled(filled_order)

        # No reverse because we're at the highest level
        assert reverse is None

    @pytest.mark.asyncio
    async def test_no_reverse_at_lowest_level(self, initialized_manager):
        """Test that no reverse order is placed at lowest level."""
        # Place sell at lowest level
        order = await initialized_manager.place_order_at_level(0, OrderSide.SELL)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("0.1"),
            created_at=datetime.now(timezone.utc),
        )

        reverse = await initialized_manager.on_order_filled(filled_order)

        # No reverse because we're at the lowest level
        assert reverse is None


class TestOrderSynchronization:
    """Tests for order synchronization."""

    @pytest.mark.asyncio
    async def test_sync_orders(self, initialized_manager, mock_exchange):
        """Test order synchronization with exchange."""
        # Place some orders
        await initialized_manager.place_initial_orders()

        # Sync should return statistics
        stats = await initialized_manager.sync_orders()

        assert "synced" in stats
        assert "filled" in stats
        assert "external" in stats


class TestStatistics:
    """Tests for statistics calculation."""

    @pytest.mark.asyncio
    async def test_get_statistics_empty(self, order_manager):
        """Test statistics with no trades."""
        stats = order_manager.get_statistics()

        assert stats["total_profit"] == Decimal("0")
        assert stats["trade_count"] == 0
        assert stats["buy_filled_count"] == 0
        assert stats["sell_filled_count"] == 0

    @pytest.mark.asyncio
    async def test_get_statistics_with_fills(self, initialized_manager):
        """Test statistics after some fills."""
        # Place and fill a buy order
        order = await initialized_manager.place_order_at_level(0, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("0.1"),
            created_at=datetime.now(timezone.utc),
        )

        await initialized_manager.on_order_filled(filled_order)

        stats = initialized_manager.get_statistics()

        assert stats["buy_filled_count"] == 1
        assert stats["sell_filled_count"] == 0


class TestOrderMappingPersistence:
    """Tests for order mapping persistence (reconnection support)."""

    @pytest.mark.asyncio
    async def test_get_order_mapping_empty(self, order_manager):
        """Test get_order_mapping with no orders."""
        mapping = order_manager.get_order_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) == 0

    @pytest.mark.asyncio
    async def test_get_order_mapping_with_orders(self, initialized_manager):
        """Test get_order_mapping returns correct mapping after placing orders."""
        await initialized_manager.place_initial_orders()

        mapping = initialized_manager.get_order_mapping()

        assert isinstance(mapping, dict)
        assert len(mapping) > 0

        # Verify mapping structure: order_id -> level_index
        for order_id, level_index in mapping.items():
            assert isinstance(order_id, str)
            assert isinstance(level_index, int)
            # Verify the order exists
            assert order_id in initialized_manager._orders

    @pytest.mark.asyncio
    async def test_restore_order_mapping_empty(self, order_manager):
        """Test restore_order_mapping with empty mapping."""
        restored = order_manager.restore_order_mapping({})

        assert restored == 0
        assert len(order_manager._order_level_map) == 0

    @pytest.mark.asyncio
    async def test_restore_order_mapping_success(self, initialized_manager):
        """Test successful order mapping restoration."""
        # Place orders and get mapping
        await initialized_manager.place_initial_orders()
        original_mapping = initialized_manager.get_order_mapping()

        # Clear the internal maps to simulate restart
        initialized_manager._order_level_map.clear()
        initialized_manager._level_order_map.clear()

        # Restore the mapping
        restored = initialized_manager.restore_order_mapping(original_mapping)

        assert restored == len(original_mapping)
        assert len(initialized_manager._order_level_map) == len(original_mapping)

        # Verify bidirectional mapping is restored
        for order_id, level_index in original_mapping.items():
            assert initialized_manager._order_level_map.get(order_id) == level_index
            assert initialized_manager._level_order_map.get(level_index) == order_id

    @pytest.mark.asyncio
    async def test_restore_order_mapping_invalid_level(self, order_manager):
        """Test restore_order_mapping skips invalid level indices."""
        # Try to restore with invalid level indices
        invalid_mapping = {
            "order_123": 999,  # Level doesn't exist
            "order_456": 1000,
        }

        restored = order_manager.restore_order_mapping(invalid_mapping)

        # Should not restore any because levels don't exist
        assert restored == 0

    @pytest.mark.asyncio
    async def test_order_mapping_round_trip(self, initialized_manager, mock_exchange):
        """Test complete round trip: place -> save -> restore -> sync."""
        # Place initial orders
        await initialized_manager.place_initial_orders()
        initial_count = initialized_manager.active_order_count

        # Get the mapping
        mapping = initialized_manager.get_order_mapping()
        assert len(mapping) == initial_count

        # Simulate restart by clearing internal state
        initialized_manager._order_level_map.clear()
        initialized_manager._level_order_map.clear()
        assert initialized_manager.active_order_count == 0

        # Restore mapping
        restored = initialized_manager.restore_order_mapping(mapping)
        assert restored == initial_count

        # Sync with exchange to verify orders still exist
        sync_result = await initialized_manager.sync_orders()
        assert "synced" in sync_result

    @pytest.mark.asyncio
    async def test_get_level_by_order_id_after_restore(self, initialized_manager):
        """Test get_level_by_order_id works correctly after restoration."""
        # Place orders
        await initialized_manager.place_initial_orders()
        mapping = initialized_manager.get_order_mapping()

        # Get expected level for first order
        first_order_id = list(mapping.keys())[0]
        expected_level = mapping[first_order_id]

        # Clear and restore
        initialized_manager._order_level_map.clear()
        initialized_manager._level_order_map.clear()
        initialized_manager.restore_order_mapping(mapping)

        # Verify lookup works
        level = initialized_manager.get_level_by_order_id(first_order_id)
        assert level == expected_level
