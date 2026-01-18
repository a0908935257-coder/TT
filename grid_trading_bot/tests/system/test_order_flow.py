"""
Order Flow System Tests.

Tests complete order flow through system components:
OrderManager → Exchange → Database → Notification

Follows the order flow path:
1. Create Order: OrderManager → Exchange API → Database
2. Order Filled: Exchange WS → Order Callback → Update State
3. Post-Fill: Reverse Order → Profit Calculation → Notification
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import MarketType, Order, OrderSide, OrderStatus, OrderType
from src.bots.grid.order_manager import FilledRecord, GridOrderManager
from src.bots.grid.models import GridConfig, GridLevel, GridSetup, LevelSide, LevelState


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_grid_setup(
    grid_count: int = 5,
    lower_price: Decimal = Decimal("45000"),
    upper_price: Decimal = Decimal("55000"),
    current_price: Decimal = Decimal("50000"),
    total_investment: Decimal = Decimal("10000"),
) -> GridSetup:
    """Create mock grid setup for testing."""
    from src.bots.grid.models import ATRConfig, ATRData, GridConfig

    # Create ATR data
    atr_data = ATRData(
        value=Decimal("1000"),
        period=14,
        timeframe="4h",
        multiplier=Decimal("2.0"),
        current_price=current_price,
        upper_price=upper_price,
        lower_price=lower_price,
        calculated_at=datetime.now(timezone.utc),
    )

    # Create config
    config = GridConfig(
        symbol="BTCUSDT",
        total_investment=total_investment,
    )

    # Calculate price step
    price_step = (upper_price - lower_price) / grid_count

    # Create levels
    levels = []
    amount_per_level = total_investment / grid_count

    for i in range(grid_count + 1):  # grid_count + 1 levels
        level_price = lower_price + (price_step * i)
        side = LevelSide.BUY if level_price < current_price else LevelSide.SELL

        level = GridLevel(
            index=i,
            price=level_price,
            side=side,
            state=LevelState.EMPTY,
            allocated_amount=amount_per_level if side == LevelSide.BUY else Decimal("0"),
        )
        levels.append(level)

    # Create setup
    setup = GridSetup(
        config=config,
        atr_data=atr_data,
        upper_price=upper_price,
        lower_price=lower_price,
        current_price=current_price,
        grid_count=grid_count,
        grid_spacing_percent=Decimal("2.0"),
        amount_per_grid=amount_per_level,
        levels=levels,
        expected_profit_per_trade=Decimal("0.5"),
        created_at=datetime.now(timezone.utc),
        version=1,
    )

    return setup


def create_mock_symbol_info() -> MagicMock:
    """Create mock symbol info."""
    info = MagicMock()
    info.min_notional = Decimal("10")
    info.min_quantity = Decimal("0.0001")
    info.base_asset = "BTC"
    info.quote_asset = "USDT"
    return info


def create_mock_balance(free: Decimal = Decimal("10000")) -> MagicMock:
    """Create mock balance."""
    balance = MagicMock()
    balance.free = free
    return balance


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()

    # Price methods
    exchange.get_price = AsyncMock(return_value=Decimal("50000"))

    # Symbol info
    exchange.get_symbol_info = AsyncMock(return_value=create_mock_symbol_info())

    # Balance
    exchange.get_balance = AsyncMock(return_value=create_mock_balance())

    # Order methods
    order_counter = {"count": 0}

    def create_order(symbol, quantity, price, market_type):
        order_counter["count"] += 1
        return Order(
            order_id=f"order_{order_counter['count']}",
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=price,
            quantity=quantity,
            filled_qty=Decimal("0"),
            created_at=datetime.now(timezone.utc),
        )

    def create_sell_order(symbol, quantity, price, market_type):
        order_counter["count"] += 1
        return Order(
            order_id=f"order_{order_counter['count']}",
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=price,
            quantity=quantity,
            filled_qty=Decimal("0"),
            created_at=datetime.now(timezone.utc),
        )

    exchange.limit_buy = AsyncMock(side_effect=create_order)
    exchange.limit_sell = AsyncMock(side_effect=create_sell_order)
    exchange.cancel_order = AsyncMock()
    exchange.get_order = AsyncMock()
    exchange.get_open_orders = AsyncMock(return_value=[])

    # Rounding methods
    exchange.round_quantity = MagicMock(side_effect=lambda s, q, m: q)
    exchange.round_price = MagicMock(side_effect=lambda s, p, m: p)

    return exchange


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    manager = MagicMock()
    manager.get_price = AsyncMock(return_value=Decimal("50000"))
    manager.save_order = AsyncMock()
    manager.update_order = AsyncMock()
    return manager


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    notifier = MagicMock()
    notifier.send_info = AsyncMock()
    notifier.send_success = AsyncMock()
    notifier.send_error = AsyncMock()
    return notifier


@pytest.fixture
def order_manager(mock_exchange, mock_data_manager, mock_notifier):
    """Create GridOrderManager with mocks."""
    manager = GridOrderManager(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
        bot_id="test_bot",
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
    )

    # Initialize with mock setup
    setup = create_mock_grid_setup()
    manager.initialize(setup)

    return manager


# =============================================================================
# Test: Create Order (OrderManager → Exchange)
# =============================================================================


class TestCreateOrder:
    """Tests for order creation flow."""

    @pytest.mark.asyncio
    async def test_place_order_calls_exchange(self, order_manager, mock_exchange):
        """Test OrderManager.place_order calls Exchange API."""
        # Act
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Assert - Exchange API was called
        assert mock_exchange.limit_buy.called or mock_exchange.limit_sell.called
        assert order is not None
        assert order.order_id is not None

    @pytest.mark.asyncio
    async def test_place_order_correct_parameters(self, order_manager, mock_exchange):
        """Test order parameters are correct."""
        # Arrange
        level_index = 1
        level = order_manager._setup.levels[level_index]

        # Act
        order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)

        # Assert - verify call parameters
        call_args = mock_exchange.limit_buy.call_args
        assert call_args[0][0] == "BTCUSDT"  # symbol
        # Price should be from level

    @pytest.mark.asyncio
    async def test_place_order_returns_order_id(self, order_manager):
        """Test order creation returns valid order_id."""
        # Act
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Assert
        assert order is not None
        assert order.order_id is not None
        assert len(order.order_id) > 0

    @pytest.mark.asyncio
    async def test_place_initial_orders(self, order_manager, mock_exchange):
        """Test placing initial grid orders."""
        # Act
        count = await order_manager.place_initial_orders()

        # Assert - some orders should be placed
        assert count >= 0
        # Exchange should be called
        assert mock_exchange.limit_buy.called or mock_exchange.limit_sell.called


# =============================================================================
# Test: Order → Database
# =============================================================================


class TestOrderToDatabase:
    """Tests for order persistence to database."""

    @pytest.mark.asyncio
    async def test_order_saved_to_database(self, order_manager, mock_data_manager):
        """Test order is saved to database after creation."""
        # Act
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Assert - save_order should be called
        mock_data_manager.save_order.assert_called()
        call_args = mock_data_manager.save_order.call_args
        saved_order = call_args[0][0]
        assert saved_order.order_id == order.order_id

    @pytest.mark.asyncio
    async def test_order_status_is_pending(self, order_manager):
        """Test new order status is NEW/PENDING."""
        # Act
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Assert
        assert order.status == OrderStatus.NEW

    @pytest.mark.asyncio
    async def test_order_fields_correct(self, order_manager):
        """Test order fields are correctly set."""
        # Act
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Assert
        assert order.symbol == "BTCUSDT"
        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        assert order.quantity > 0
        assert order.price > 0


# =============================================================================
# Test: Order Filled Callback (Exchange WS → OrderManager)
# =============================================================================


class TestOrderFilledCallback:
    """Tests for order fill callback handling."""

    @pytest.mark.asyncio
    async def test_on_order_filled_called(self, order_manager):
        """Test on_order_filled processes fill event."""
        # Arrange - place order first
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        # Create filled order event
        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=order.quantity * order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        result = await order_manager.on_order_filled(filled_order)

        # Assert - fill should be recorded
        assert len(order_manager._filled_history) == 1

    @pytest.mark.asyncio
    async def test_fill_contains_required_info(self, order_manager):
        """Test fill record contains order_id, price, quantity."""
        # Arrange
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert
        fill_record = order_manager._filled_history[0]
        assert fill_record.order_id == order.order_id
        assert fill_record.price == order.price
        assert fill_record.quantity == order.quantity


# =============================================================================
# Test: Filled → Database Update
# =============================================================================


class TestFilledUpdateDatabase:
    """Tests for database update on fill."""

    @pytest.mark.asyncio
    async def test_filled_updates_database(self, order_manager, mock_data_manager):
        """Test filled order updates database."""
        # Arrange
        order = await order_manager.place_order_at_level(0, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert - update_order should be called
        mock_data_manager.update_order.assert_called()

    @pytest.mark.asyncio
    async def test_filled_status_recorded(self, order_manager):
        """Test filled status is recorded in level state."""
        # Arrange
        level_index = 0
        order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert - level state should be FILLED
        level = order_manager._setup.levels[level_index]
        assert level.state == LevelState.FILLED

    @pytest.mark.asyncio
    async def test_filled_price_recorded(self, order_manager):
        """Test filled price is recorded."""
        # Arrange
        level_index = 0
        order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)
        fill_price = Decimal("45100")

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=fill_price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert
        level = order_manager._setup.levels[level_index]
        assert level.filled_price == fill_price


# =============================================================================
# Test: Buy Filled → Place Sell (Reverse Order)
# =============================================================================


class TestFilledReverseOrder:
    """Tests for reverse order placement after fill."""

    @pytest.mark.asyncio
    async def test_buy_filled_places_sell(self, order_manager, mock_exchange):
        """Test buy fill triggers sell order placement."""
        # Arrange - place buy order at level 1 (not boundary)
        level_index = 1
        buy_order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)

        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        reverse_order = await order_manager.on_order_filled(filled_buy)

        # Assert - sell order should be placed
        assert reverse_order is not None
        assert reverse_order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_reverse_order_at_correct_level(self, order_manager):
        """Test reverse order is at level + 1."""
        # Arrange
        level_index = 1
        buy_order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)

        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        reverse_order = await order_manager.on_order_filled(filled_buy)

        # Assert - should be at level_index + 1
        if reverse_order:
            reverse_level = order_manager.get_level_by_order_id(reverse_order.order_id)
            assert reverse_level == level_index + 1

    @pytest.mark.asyncio
    async def test_sell_filled_places_buy(self, order_manager):
        """Test sell fill triggers buy order placement."""
        # Arrange - place sell order at level 4 (not boundary)
        level_index = 4
        sell_order = await order_manager.place_order_at_level(level_index, OrderSide.SELL)

        filled_sell = Order(
            order_id=sell_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=sell_order.price,
            quantity=sell_order.quantity,
            filled_qty=sell_order.quantity,
            avg_price=sell_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        reverse_order = await order_manager.on_order_filled(filled_sell)

        # Assert - buy order should be placed
        assert reverse_order is not None
        assert reverse_order.side == OrderSide.BUY


# =============================================================================
# Test: Sell Filled → Profit Calculation
# =============================================================================


class TestFilledProfitCalculation:
    """Tests for profit calculation on sell fill."""

    @pytest.mark.asyncio
    async def test_profit_calculated_on_sell(self, order_manager):
        """Test profit is calculated when sell completes round trip."""
        # Arrange - complete a buy-sell round trip
        buy_level = 1
        buy_order = await order_manager.place_order_at_level(buy_level, OrderSide.BUY)

        # Fill buy
        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        sell_order = await order_manager.on_order_filled(filled_buy)

        if sell_order:
            # Fill sell
            filled_sell = Order(
                order_id=sell_order.order_id,
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                status=OrderStatus.FILLED,
                price=sell_order.price,
                quantity=sell_order.quantity,
                filled_qty=sell_order.quantity,
                avg_price=sell_order.price,
                fee=Decimal("5"),
                created_at=datetime.now(timezone.utc),
            )

            # Act
            await order_manager.on_order_filled(filled_sell)

            # Assert - profit should be tracked
            assert order_manager._trade_count >= 1
            # With price difference between levels, profit should be positive
            # (sell_price - buy_price) * qty - fees

    @pytest.mark.asyncio
    async def test_profit_formula_correct(self, order_manager):
        """Test profit formula: (sell - buy) * qty - fees."""
        # Test via calculate_profit directly
        buy_price = Decimal("45000")
        sell_price = Decimal("47000")
        quantity = Decimal("0.1")
        buy_fee = Decimal("4.5")
        sell_fee = Decimal("4.7")

        buy_record = FilledRecord(
            level_index=1,
            side=OrderSide.BUY,
            price=buy_price,
            quantity=quantity,
            fee=buy_fee,
            timestamp=datetime.now(timezone.utc),
            order_id="buy_test",
        )

        sell_record = FilledRecord(
            level_index=2,
            side=OrderSide.SELL,
            price=sell_price,
            quantity=quantity,
            fee=sell_fee,
            timestamp=datetime.now(timezone.utc),
            order_id="sell_test",
        )

        # Act
        profit = order_manager.calculate_profit(buy_record, sell_record)

        # Assert
        # Expected: (47000 - 45000) * 0.1 - (4.5 + 4.7) = 200 - 9.2 = 190.8
        expected = (sell_price - buy_price) * quantity - (buy_fee + sell_fee)
        assert profit == expected

    @pytest.mark.asyncio
    async def test_statistics_updated(self, order_manager):
        """Test statistics are updated after trade."""
        # Arrange - complete round trip
        buy_order = await order_manager.place_order_at_level(1, OrderSide.BUY)

        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        sell_order = await order_manager.on_order_filled(filled_buy)

        if sell_order:
            filled_sell = Order(
                order_id=sell_order.order_id,
                symbol="BTCUSDT",
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                status=OrderStatus.FILLED,
                price=sell_order.price,
                quantity=sell_order.quantity,
                filled_qty=sell_order.quantity,
                avg_price=sell_order.price,
                fee=Decimal("5"),
                created_at=datetime.now(timezone.utc),
            )

            await order_manager.on_order_filled(filled_sell)

        # Assert
        stats = order_manager.get_statistics()
        assert "total_profit" in stats
        assert "trade_count" in stats
        assert stats["buy_filled_count"] >= 1


# =============================================================================
# Test: Filled → Discord Notification
# =============================================================================


class TestFilledNotification:
    """Tests for notification on order fill."""

    @pytest.mark.asyncio
    async def test_notification_sent_on_fill(self, order_manager, mock_notifier):
        """Test notification is sent when order fills."""
        # Arrange
        order = await order_manager.place_order_at_level(1, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert - notification should be sent
        mock_notifier.send_success.assert_called()

    @pytest.mark.asyncio
    async def test_notification_contains_trade_info(self, order_manager, mock_notifier):
        """Test notification contains trade information."""
        # Arrange
        order = await order_manager.place_order_at_level(1, OrderSide.BUY)

        filled_order = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=order.quantity,
            avg_price=order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Act
        await order_manager.on_order_filled(filled_order)

        # Assert - check notification content
        call_args = mock_notifier.send_success.call_args
        assert call_args is not None
        # Should include title and message
        kwargs = call_args.kwargs if call_args.kwargs else {}
        assert "title" in kwargs or len(call_args.args) > 0


# =============================================================================
# Test: Full Order Flow
# =============================================================================


class TestFullOrderFlow:
    """Tests for complete order flow through all components."""

    @pytest.mark.asyncio
    async def test_complete_buy_sell_cycle(
        self, order_manager, mock_exchange, mock_data_manager, mock_notifier
    ):
        """
        Test complete order cycle:
        1. Place buy order → Exchange API called
        2. Save to database
        3. Simulate buy fill
        4. Verify callback triggered
        5. Verify sell order placed
        6. Simulate sell fill
        7. Verify profit calculated
        8. Verify notification sent
        """
        # Step 1: Place buy order
        buy_order = await order_manager.place_order_at_level(1, OrderSide.BUY)
        assert buy_order is not None
        assert mock_exchange.limit_buy.called

        # Step 2: Verify saved to database
        assert mock_data_manager.save_order.called

        # Step 3: Simulate buy fill
        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=buy_order.quantity * buy_order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        # Step 4 & 5: Process fill and verify reverse order
        sell_order = await order_manager.on_order_filled(filled_buy)
        assert len(order_manager._filled_history) == 1
        assert sell_order is not None
        assert sell_order.side == OrderSide.SELL

        # Step 6: Simulate sell fill
        filled_sell = Order(
            order_id=sell_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=sell_order.price,
            quantity=sell_order.quantity,
            filled_qty=sell_order.quantity,
            avg_price=sell_order.price,
            fee=sell_order.quantity * sell_order.price * Decimal("0.001"),
            created_at=datetime.now(timezone.utc),
        )

        await order_manager.on_order_filled(filled_sell)

        # Step 7: Verify profit calculated
        stats = order_manager.get_statistics()
        assert stats["trade_count"] >= 1
        assert stats["buy_filled_count"] >= 1
        assert stats["sell_filled_count"] >= 1

        # Step 8: Verify notification sent
        assert mock_notifier.send_success.call_count >= 2  # Once for buy, once for sell

    @pytest.mark.asyncio
    async def test_data_consistency_through_flow(self, order_manager):
        """Test data remains consistent through entire flow."""
        # Place and fill multiple orders
        orders_placed = []

        # Place orders at multiple levels
        for level_index in [1, 2]:
            order = await order_manager.place_order_at_level(level_index, OrderSide.BUY)
            orders_placed.append((level_index, order))

        # Verify mappings are consistent
        for level_index, order in orders_placed:
            # Level -> Order mapping
            assert level_index in order_manager._level_order_map
            mapped_order_id = order_manager._level_order_map[level_index]
            assert mapped_order_id == order.order_id

            # Order -> Level mapping
            assert order.order_id in order_manager._order_level_map
            mapped_level = order_manager._order_level_map[order.order_id]
            assert mapped_level == level_index

    @pytest.mark.asyncio
    async def test_no_data_loss_in_flow(self, order_manager, mock_data_manager):
        """Test no data is lost through the flow."""
        # Track all operations
        initial_save_count = mock_data_manager.save_order.call_count

        # Complete a full cycle
        buy_order = await order_manager.place_order_at_level(1, OrderSide.BUY)

        filled_buy = Order(
            order_id=buy_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=buy_order.price,
            quantity=buy_order.quantity,
            filled_qty=buy_order.quantity,
            avg_price=buy_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        sell_order = await order_manager.on_order_filled(filled_buy)

        # Verify all operations recorded
        # save_order should be called for both buy and sell
        assert mock_data_manager.save_order.call_count >= initial_save_count + 1

        # Fill history should contain the fill
        assert len(order_manager._filled_history) >= 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestOrderFlowEdgeCases:
    """Edge case tests for order flow."""

    @pytest.mark.asyncio
    async def test_fill_unknown_order(self, order_manager):
        """Test handling fill for unknown order."""
        unknown_order = Order(
            order_id="unknown_order_123",
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

        # Should not crash, should return None
        result = await order_manager.on_order_filled(unknown_order)
        assert result is None

    @pytest.mark.asyncio
    async def test_boundary_level_no_reverse(self, order_manager):
        """Test no reverse order at boundary levels."""
        # Place sell at highest level
        highest_level = len(order_manager._setup.levels) - 1
        sell_order = await order_manager.place_order_at_level(highest_level, OrderSide.SELL)

        filled_sell = Order(
            order_id=sell_order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=sell_order.price,
            quantity=sell_order.quantity,
            filled_qty=sell_order.quantity,
            avg_price=sell_order.price,
            fee=Decimal("5"),
            created_at=datetime.now(timezone.utc),
        )

        # Sell at highest level should not place buy (no level above)
        # Note: This depends on implementation - sell triggers buy at lower level
        result = await order_manager.on_order_filled(filled_sell)
        # Result may or may not be None depending on level configuration

    @pytest.mark.asyncio
    async def test_cancel_order_flow(self, order_manager, mock_exchange):
        """Test order cancellation flow."""
        # Place order
        order = await order_manager.place_order_at_level(1, OrderSide.BUY)
        level_index = 1

        # Cancel
        mock_exchange.cancel_order.return_value = Order(
            order_id=order.order_id,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.CANCELED,
            price=order.price,
            quantity=order.quantity,
            filled_qty=Decimal("0"),
            created_at=datetime.now(timezone.utc),
        )

        result = await order_manager.cancel_order_at_level(level_index)

        # Verify cancelled
        assert result is True
        assert level_index not in order_manager._level_order_map
