"""
Integration tests for order flow.

Tests complete order cycles including fills, reverses, and profit tracking.
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.core.models import MarketType, Order, OrderSide, OrderStatus, OrderType
from src.bots.grid import (
    FilledRecord,
    GridOrderManager,
    LevelState,
    create_grid_with_manual_range,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


@pytest.fixture
def order_flow_setup(mock_exchange, mock_data_manager, mock_notifier):
    """Create setup for order flow tests."""
    # Create grid: 10 levels from 45000 to 55000
    setup = create_grid_with_manual_range(
        symbol="BTCUSDT",
        investment=10000,
        upper_price=55000,
        lower_price=45000,
        grid_count=10,
        current_price=50000,
    )

    # Set price in mocks
    mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
    mock_exchange.set_price(Decimal("50000"))

    # Create order manager
    manager = GridOrderManager(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
        bot_id="flow_test",
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
    )
    manager.initialize(setup)

    return {
        "manager": manager,
        "setup": setup,
        "exchange": mock_exchange,
        "data_manager": mock_data_manager,
        "notifier": mock_notifier,
    }


class TestCompleteOrderCycle:
    """Tests for complete order cycles."""

    @pytest.mark.asyncio
    async def test_complete_grid_trading_cycle(self, order_flow_setup):
        """
        Test complete grid trading cycle:

        Initial: Price 50000, 10 levels 45000-55000

        Round 1:
            Price drops to 49000 -> L4 buy filled
            Verify: L5 sell order placed

        Round 2:
            Price rises to 51000 -> L5 sell filled
            Verify: Profit calculated
            Verify: L4 buy order placed

        Continue for multiple rounds...
        """
        manager = order_flow_setup["manager"]
        exchange = order_flow_setup["exchange"]
        setup = order_flow_setup["setup"]

        # Place initial orders
        await manager.place_initial_orders()
        initial_order_count = manager.active_order_count

        # Find buy order at a specific level (below 50000)
        buy_level_index = None
        buy_order_id = None

        for level in setup.levels:
            if level.price < Decimal("50000") and level.index in manager._level_order_map:
                buy_level_index = level.index
                buy_order_id = manager._level_order_map[level.index]
                break

        if buy_level_index is None:
            pytest.skip("No suitable buy level found")

        buy_level = setup.levels[buy_level_index]
        buy_order = manager._orders[buy_order_id]

        # Round 1: Simulate price drop -> buy fill
        filled_buy = Order(
            order_id=buy_order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=buy_order.order_type,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=buy_order.quantity * buy_order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        reverse_sell = await manager.on_order_filled(filled_buy)

        # Verify buy was recorded
        assert len(manager._filled_history) == 1
        assert manager._filled_history[0].side == OrderSide.BUY
        assert manager._filled_history[0].level_index == buy_level_index

        # Verify reverse sell was placed at upper level
        if reverse_sell:
            sell_level_index = manager.get_level_by_order_id(reverse_sell.order_id)
            assert sell_level_index == buy_level_index + 1
            assert reverse_sell.side == OrderSide.SELL

            # Round 2: Simulate price rise -> sell fill
            filled_sell = Order(
                order_id=reverse_sell.order_id,
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=reverse_sell.order_type,
                status=OrderStatus.FILLED,
                price=reverse_sell.price,
                quantity=reverse_sell.quantity,
                filled_qty=reverse_sell.quantity,
                avg_price=reverse_sell.price,
                fee=reverse_sell.quantity * reverse_sell.price * Decimal("0.001"),
                created_at=datetime.now(timezone.utc),
            )

            reverse_buy = await manager.on_order_filled(filled_sell)

            # Verify sell was recorded
            assert len(manager._filled_history) == 2
            assert manager._filled_history[1].side == OrderSide.SELL

            # Verify profit was calculated
            assert manager._total_profit > 0
            assert manager._trade_count == 1

            # Verify reverse buy was placed
            if reverse_buy:
                assert reverse_buy.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_multiple_round_trips(self, order_flow_setup):
        """Test multiple complete round trips."""
        manager = order_flow_setup["manager"]
        setup = order_flow_setup["setup"]

        await manager.place_initial_orders()

        # Find a buy level
        buy_level = None
        for level in setup.levels:
            if level.price < Decimal("50000") and level.index in manager._level_order_map:
                buy_level = level
                break

        if buy_level is None:
            pytest.skip("No suitable buy level found")

        total_profit = Decimal("0")
        num_round_trips = 3

        for round_num in range(num_round_trips):
            # Get current buy order at this level
            if buy_level.index not in manager._level_order_map:
                # Re-place if not present
                await manager.place_order_at_level(buy_level.index, OrderSide.BUY)

            buy_order_id = manager._level_order_map.get(buy_level.index)
            if not buy_order_id:
                continue

            buy_order = manager._orders[buy_order_id]

            # Simulate buy fill
            filled_buy = Order(
                order_id=buy_order_id,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=buy_order.order_type,
                status=OrderStatus.FILLED,
                price=buy_order.price,
                quantity=buy_order.quantity,
                filled_qty=buy_order.quantity,
                avg_price=buy_order.price,
                fee=buy_order.quantity * buy_order.price * Decimal("0.001"),
                created_at=datetime.now(timezone.utc),
            )

            reverse_sell = await manager.on_order_filled(filled_buy)

            if reverse_sell:
                # Simulate sell fill
                filled_sell = Order(
                    order_id=reverse_sell.order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.SELL,
                    order_type=reverse_sell.order_type,
                    status=OrderStatus.FILLED,
                    price=reverse_sell.price,
                    quantity=reverse_sell.quantity,
                    filled_qty=reverse_sell.quantity,
                    avg_price=reverse_sell.price,
                    fee=reverse_sell.quantity * reverse_sell.price * Decimal("0.001"),
                    created_at=datetime.now(timezone.utc),
                )

                await manager.on_order_filled(filled_sell)

        # Verify multiple round trips
        stats = manager.get_statistics()
        assert stats["trade_count"] >= 1  # At least some completed trades
        assert stats["total_profit"] > 0


class TestProfitCalculation:
    """Tests for profit calculation in order flow."""

    @pytest.mark.asyncio
    async def test_profit_calculation_accuracy(self, order_flow_setup):
        """Test accurate profit calculation."""
        manager = order_flow_setup["manager"]

        # Manual fill records with known values
        buy_price = Decimal("49000")
        sell_price = Decimal("51000")
        quantity = Decimal("0.1")
        buy_fee = buy_price * quantity * Decimal("0.001")  # 4.9
        sell_fee = sell_price * quantity * Decimal("0.001")  # 5.1

        buy_record = FilledRecord(
            level_index=4,
            side=OrderSide.BUY,
            price=buy_price,
            quantity=quantity,
            fee=buy_fee,
            timestamp=datetime.now(timezone.utc),
            order_id="buy_001",
        )

        sell_record = FilledRecord(
            level_index=5,
            side=OrderSide.SELL,
            price=sell_price,
            quantity=quantity,
            fee=sell_fee,
            timestamp=datetime.now(timezone.utc),
            order_id="sell_001",
        )

        profit = manager.calculate_profit(buy_record, sell_record)

        # Expected: (51000 - 49000) * 0.1 - (4.9 + 5.1) = 200 - 10 = 190
        expected_gross = (sell_price - buy_price) * quantity
        expected_fees = buy_fee + sell_fee
        expected_profit = expected_gross - expected_fees

        assert profit == expected_profit

    @pytest.mark.asyncio
    async def test_cumulative_profit_tracking(self, order_flow_setup):
        """Test cumulative profit tracking over multiple trades."""
        manager = order_flow_setup["manager"]
        setup = order_flow_setup["setup"]

        await manager.place_initial_orders()

        # Simulate multiple fills and track profit
        initial_profit = manager._total_profit
        assert initial_profit == Decimal("0")

        # Find two adjacent levels
        for i in range(len(setup.levels) - 1):
            level = setup.levels[i]
            if level.price < Decimal("50000") and i in manager._level_order_map:
                buy_order_id = manager._level_order_map[i]
                buy_order = manager._orders[buy_order_id]

                # Fill buy
                filled_buy = Order(
                    order_id=buy_order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=buy_order.order_type,
                    status=OrderStatus.FILLED,
                    price=buy_order.price,
                    quantity=buy_order.quantity,
                    filled_qty=buy_order.quantity,
                    avg_price=buy_order.price,
                    fee=Decimal("5"),
                    created_at=datetime.now(timezone.utc),
                )

                reverse = await manager.on_order_filled(filled_buy)

                if reverse:
                    # Fill sell
                    filled_sell = Order(
                        order_id=reverse.order_id,
                        symbol="BTCUSDT",
                        side=OrderSide.SELL,
                        order_type=reverse.order_type,
                        status=OrderStatus.FILLED,
                        price=reverse.price,
                        quantity=reverse.quantity,
                        filled_qty=reverse.quantity,
                        avg_price=reverse.price,
                        fee=Decimal("5"),
                        created_at=datetime.now(timezone.utc),
                    )

                    await manager.on_order_filled(filled_sell)

                    # Verify profit increased
                    assert manager._total_profit > initial_profit

                break


class TestOrderStateTracking:
    """Tests for order state tracking through flow."""

    @pytest.mark.asyncio
    async def test_level_state_transitions(self, order_flow_setup):
        """Test level state transitions through order flow."""
        manager = order_flow_setup["manager"]
        setup = order_flow_setup["setup"]

        # Initially all levels are EMPTY
        for level in setup.levels:
            assert level.state == LevelState.EMPTY

        # Place initial orders
        await manager.place_initial_orders()

        # Levels with orders should be PENDING_BUY or PENDING_SELL
        for level_index in manager._level_order_map:
            level = setup.levels[level_index]
            order = manager._orders[manager._level_order_map[level_index]]

            if order.side == OrderSide.BUY:
                assert level.state == LevelState.PENDING_BUY
            else:
                assert level.state == LevelState.PENDING_SELL

    @pytest.mark.asyncio
    async def test_order_to_level_mapping_consistency(self, order_flow_setup):
        """Test bidirectional mapping consistency."""
        manager = order_flow_setup["manager"]

        await manager.place_initial_orders()

        # Verify bidirectional consistency
        for level_index, order_id in manager._level_order_map.items():
            # Reverse lookup should match
            assert manager._order_level_map[order_id] == level_index

            # Order should exist
            assert order_id in manager._orders


class TestEdgeCases:
    """Tests for edge cases in order flow."""

    @pytest.mark.asyncio
    async def test_boundary_level_fills(self, order_flow_setup):
        """Test fills at boundary levels."""
        manager = order_flow_setup["manager"]
        setup = order_flow_setup["setup"]

        # Place order at lowest level
        lowest_level = setup.levels[0]
        order = await manager.place_order_at_level(0, OrderSide.SELL)

        # Fill sell at lowest level
        filled = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=order.order_type,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        reverse = await manager.on_order_filled(filled)

        # At lowest level, a BUY reverse is placed at the same level
        assert reverse is not None
        assert reverse.side == OrderSide.BUY

    @pytest.mark.asyncio
    async def test_rapid_fills(self, order_flow_setup):
        """Test handling of rapid consecutive fills."""
        manager = order_flow_setup["manager"]
        setup = order_flow_setup["setup"]

        await manager.place_initial_orders()

        # Simulate rapid fills at multiple levels
        fill_count = 0
        for level in setup.levels:
            if level.index in manager._level_order_map:
                order_id = manager._level_order_map[level.index]
                order = manager._orders[order_id]

                filled = Order(
                    order_id=order_id,
                    symbol="BTCUSDT",
                    side=order.side,
                    order_type=order.order_type,
                    status=OrderStatus.FILLED,
                    price=order.price,
                    quantity=order.quantity,
                    filled_qty=order.quantity,
                    avg_price=order.price,
                    fee=Decimal("5"),
                    created_at=datetime.now(timezone.utc),
                )

                await manager.on_order_filled(filled)
                fill_count += 1

                if fill_count >= 3:
                    break

        # All fills should be recorded
        assert len(manager._filled_history) == fill_count

    @pytest.mark.asyncio
    async def test_fill_with_unknown_order(self, order_flow_setup):
        """Test handling fill for unknown order."""
        manager = order_flow_setup["manager"]

        # Try to fill an order that doesn't exist
        unknown_order = Order(
            order_id="unknown_12345",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            filled_qty=Decimal("0.1"),
            avg_price=Decimal("50000"),
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        result = await manager.on_order_filled(unknown_order)

        # Should return None and not crash
        assert result is None
        assert len(manager._filled_history) == 0
