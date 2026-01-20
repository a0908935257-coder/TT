"""
Tests for Order Executor.

Tests order execution including:
- Placing entry limit orders
- Placing take profit and stop loss orders
- Cancelling orders
- Market close position
- Order fill callbacks
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, MagicMock
from dataclasses import dataclass

from src.bots.bollinger.order_executor import OrderExecutor, OrderNotFoundError
from src.bots.bollinger.models import (
    BollingerConfig,
    Signal,
    SignalType,
    Position,
    PositionSide,
    BollingerBands,
    BBWData,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockOrder:
    """Mock order for testing."""
    order_id: str
    status: str = "NEW"
    side: str = "BUY"
    avg_price: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")


@pytest.fixture
def config() -> BollingerConfig:
    """Create default config."""
    return BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        leverage=2,
        position_size_pct=Decimal("0.1"),
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=20,
        stop_loss_pct=Decimal("0.015"),
        max_hold_bars=16,
    )


@pytest.fixture
def mock_exchange() -> Mock:
    """Create mock exchange."""
    mock = Mock()
    mock.futures_create_order = AsyncMock(return_value=MockOrder(order_id="order_001"))
    mock.futures_cancel_order = AsyncMock(return_value={})
    mock.futures_cancel_algo_order = AsyncMock(return_value={})  # For stop loss (Algo orders)
    mock.futures_get_order = AsyncMock(return_value=MockOrder(order_id="order_001"))
    return mock


@pytest.fixture
def mock_notifier() -> Mock:
    """Create mock notifier."""
    mock = Mock()
    mock.send_trade_notification = AsyncMock()
    return mock


@pytest.fixture
def order_executor(
    config: BollingerConfig,
    mock_exchange: Mock,
    mock_notifier: Mock,
) -> OrderExecutor:
    """Create order executor."""
    return OrderExecutor(config, mock_exchange, mock_notifier)


def create_signal(signal_type: SignalType, entry_price: float) -> Signal:
    """Create a signal for testing."""
    return Signal(
        signal_type=signal_type,
        entry_price=Decimal(str(entry_price)),
        take_profit=Decimal("50000"),
        stop_loss=Decimal("48500") if signal_type == SignalType.LONG else Decimal("51500"),
        bands=BollingerBands(
            upper=Decimal("51000"),
            middle=Decimal("50000"),
            lower=Decimal("49000"),
            std=Decimal("500"),
        ),
        bbw=BBWData(
            bbw=Decimal("0.04"),
            bbw_percentile=50,
            is_squeeze=False,
            threshold=Decimal("0.02"),
        ),
    )


def create_position(side: PositionSide, entry_price: float, quantity: float) -> Position:
    """Create a position for testing."""
    return Position(
        symbol="BTCUSDT",
        side=side,
        entry_price=Decimal(str(entry_price)),
        quantity=Decimal(str(quantity)),
        leverage=2,
        unrealized_pnl=Decimal("0"),
        entry_time=datetime.now(timezone.utc),
        entry_bar=1,
        take_profit_price=Decimal("50000"),
        stop_loss_price=Decimal("48500") if side == PositionSide.LONG else Decimal("51500"),
    )


# =============================================================================
# Test Place Entry Limit Order
# =============================================================================


class TestPlaceEntryLimitOrder:
    """Test placing entry limit orders."""

    @pytest.mark.asyncio
    async def test_place_entry_limit_order_long(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test placing entry limit order for long."""
        signal = create_signal(SignalType.LONG, entry_price=49000)
        quantity = Decimal("0.01")

        order_id = await order_executor.place_entry_order(signal, quantity)

        assert order_id == "order_001"
        mock_exchange.futures_create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=quantity,
            price=Decimal("49000"),
            time_in_force="GTC",
        )

    @pytest.mark.asyncio
    async def test_place_entry_limit_order_short(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test placing entry limit order for short."""
        signal = create_signal(SignalType.SHORT, entry_price=51000)
        quantity = Decimal("0.01")

        order_id = await order_executor.place_entry_order(signal, quantity)

        mock_exchange.futures_create_order.assert_called_once_with(
            symbol="BTCUSDT",
            side="SELL",
            order_type="LIMIT",
            quantity=quantity,
            price=Decimal("51000"),
            time_in_force="GTC",
        )

    @pytest.mark.asyncio
    async def test_entry_order_tracked(
        self,
        order_executor: OrderExecutor,
    ):
        """Test that entry order is tracked."""
        signal = create_signal(SignalType.LONG, entry_price=49000)
        await order_executor.place_entry_order(signal, Decimal("0.01"))

        assert order_executor.has_pending_entry is True
        assert order_executor.pending_entry_order == "order_001"


# =============================================================================
# Test Place Take Profit Order
# =============================================================================


class TestPlaceTakeProfitOrder:
    """Test placing take profit orders."""

    @pytest.mark.asyncio
    async def test_place_take_profit_order_long(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test placing take profit order for long position."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)

        await order_executor.place_exit_orders(position)

        # Check TP order was placed
        calls = mock_exchange.futures_create_order.call_args_list
        tp_call = calls[0]
        assert tp_call.kwargs["side"] == "SELL"  # Close long = SELL
        assert tp_call.kwargs["order_type"] == "LIMIT"
        assert tp_call.kwargs["price"] == Decimal("50000")
        assert tp_call.kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_place_take_profit_order_short(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test placing take profit order for short position."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        position = create_position(PositionSide.SHORT, entry_price=51000, quantity=0.01)

        await order_executor.place_exit_orders(position)

        calls = mock_exchange.futures_create_order.call_args_list
        tp_call = calls[0]
        assert tp_call.kwargs["side"] == "BUY"  # Close short = BUY


# =============================================================================
# Test Place Stop Loss Order
# =============================================================================


class TestPlaceStopLossOrder:
    """Test placing stop loss orders."""

    @pytest.mark.asyncio
    async def test_place_stop_loss_order_long(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test placing stop loss order for long position."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)

        await order_executor.place_exit_orders(position)

        # Check SL order was placed
        calls = mock_exchange.futures_create_order.call_args_list
        sl_call = calls[1]
        assert sl_call.kwargs["side"] == "SELL"
        assert sl_call.kwargs["order_type"] == "STOP_MARKET"
        assert sl_call.kwargs["stop_price"] == Decimal("48500")
        assert sl_call.kwargs["reduce_only"] is True

    @pytest.mark.asyncio
    async def test_stop_loss_order_tracked(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test that stop loss order is tracked."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        assert order_executor.stop_loss_order == "sl_001"
        assert order_executor.take_profit_order == "tp_001"


# =============================================================================
# Test Cancel Entry Order
# =============================================================================


class TestCancelEntryOrder:
    """Test cancelling entry orders."""

    @pytest.mark.asyncio
    async def test_cancel_entry_order(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test cancelling pending entry order."""
        # Place entry order first
        signal = create_signal(SignalType.LONG, entry_price=49000)
        await order_executor.place_entry_order(signal, Decimal("0.01"))

        # Cancel it
        result = await order_executor.cancel_entry_order()

        assert result is True
        mock_exchange.futures_cancel_order.assert_called_once_with(
            symbol="BTCUSDT",
            order_id="order_001",
        )
        assert order_executor.has_pending_entry is False

    @pytest.mark.asyncio
    async def test_cancel_entry_order_no_order(
        self,
        order_executor: OrderExecutor,
    ):
        """Test cancelling when no entry order exists."""
        result = await order_executor.cancel_entry_order()

        assert result is False


# =============================================================================
# Test Cancel Exit Orders
# =============================================================================


class TestCancelExitOrders:
    """Test cancelling exit orders."""

    @pytest.mark.asyncio
    async def test_cancel_exit_orders(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test cancelling both TP and SL orders."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        # Place exit orders
        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        # Cancel them
        await order_executor.cancel_exit_orders()

        # Should have called regular cancel for TP, algo cancel for SL
        assert mock_exchange.futures_cancel_order.call_count == 1  # TP only
        assert mock_exchange.futures_cancel_algo_order.call_count == 1  # SL only
        assert order_executor.take_profit_order is None
        assert order_executor.stop_loss_order is None


# =============================================================================
# Test Market Close Position
# =============================================================================


class TestMarketClosePosition:
    """Test market closing positions."""

    @pytest.mark.asyncio
    async def test_market_close_position_long(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test market closing a long position."""
        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)

        await order_executor.close_position_market(position)

        mock_exchange.futures_create_order.assert_called_with(
            symbol="BTCUSDT",
            side="SELL",
            order_type="MARKET",
            quantity=Decimal("0.01"),
            reduce_only=True,
        )

    @pytest.mark.asyncio
    async def test_market_close_position_short(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test market closing a short position."""
        position = create_position(PositionSide.SHORT, entry_price=51000, quantity=0.01)

        await order_executor.close_position_market(position)

        mock_exchange.futures_create_order.assert_called_with(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.01"),
            reduce_only=True,
        )

    @pytest.mark.asyncio
    async def test_market_close_cancels_exit_orders(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test that market close cancels existing exit orders."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
                MockOrder(order_id="market_001"),
            ]
        )

        # Place exit orders
        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        # Market close
        await order_executor.close_position_market(position)

        # Should have cancelled exit orders (TP: regular, SL: algo)
        assert mock_exchange.futures_cancel_order.call_count == 1  # TP only
        assert mock_exchange.futures_cancel_algo_order.call_count == 1  # SL only


# =============================================================================
# Test On Entry Filled
# =============================================================================


class TestOnEntryFilled:
    """Test entry fill callback."""

    @pytest.mark.asyncio
    async def test_on_entry_filled(
        self,
        order_executor: OrderExecutor,
        mock_notifier: Mock,
    ):
        """Test handling entry order fill."""
        # Place entry order
        signal = create_signal(SignalType.LONG, entry_price=49000)
        await order_executor.place_entry_order(signal, Decimal("0.01"))

        # Create filled order
        filled_order = MockOrder(
            order_id="order_001",
            status="FILLED",
            side="BUY",
            avg_price=Decimal("49000"),
            filled_quantity=Decimal("0.01"),
        )

        result = await order_executor.on_order_filled(filled_order)

        assert result == "entry_filled"
        assert order_executor.has_pending_entry is False
        mock_notifier.send_trade_notification.assert_called_once()


# =============================================================================
# Test On Take Profit Filled
# =============================================================================


class TestOnTakeProfitFilled:
    """Test take profit fill callback."""

    @pytest.mark.asyncio
    async def test_on_take_profit_filled(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test handling take profit order fill."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        # Place exit orders
        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        # Create filled TP order
        filled_order = MockOrder(
            order_id="tp_001",
            status="FILLED",
            side="SELL",
            avg_price=Decimal("50000"),
            filled_quantity=Decimal("0.01"),
        )

        result = await order_executor.on_order_filled(filled_order)

        assert result == "take_profit_filled"
        assert order_executor.take_profit_order is None
        # Should cancel stop loss (Algo order)
        mock_exchange.futures_cancel_algo_order.assert_called()


# =============================================================================
# Test On Stop Loss Filled
# =============================================================================


class TestOnStopLossFilled:
    """Test stop loss fill callback."""

    @pytest.mark.asyncio
    async def test_on_stop_loss_filled(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test handling stop loss order fill."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        # Place exit orders
        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        # Create filled SL order
        filled_order = MockOrder(
            order_id="sl_001",
            status="FILLED",
            side="SELL",
            avg_price=Decimal("48500"),
            filled_quantity=Decimal("0.01"),
        )

        result = await order_executor.on_order_filled(filled_order)

        assert result == "stop_loss_filled"
        assert order_executor.stop_loss_order is None
        # Should cancel take profit
        mock_exchange.futures_cancel_order.assert_called()


# =============================================================================
# Test Unknown Order Fill
# =============================================================================


class TestUnknownOrderFill:
    """Test handling unknown order fills."""

    @pytest.mark.asyncio
    async def test_unknown_order_returns_none(
        self,
        order_executor: OrderExecutor,
    ):
        """Test that unknown order ID returns None."""
        unknown_order = MockOrder(
            order_id="unknown_001",
            status="FILLED",
        )

        result = await order_executor.on_order_filled(unknown_order)

        assert result is None


# =============================================================================
# Test Entry Timeout
# =============================================================================


class TestEntryTimeout:
    """Test entry order timeout settings."""

    def test_get_entry_timeout_bars(
        self,
        order_executor: OrderExecutor,
    ):
        """Test getting entry timeout bars."""
        timeout = order_executor.get_entry_timeout_bars()

        assert timeout == 3  # Default

    def test_set_entry_timeout_bars(
        self,
        order_executor: OrderExecutor,
    ):
        """Test setting entry timeout bars."""
        order_executor.set_entry_timeout_bars(5)

        assert order_executor.get_entry_timeout_bars() == 5


# =============================================================================
# Test Clear All Orders
# =============================================================================


class TestClearAllOrders:
    """Test clearing all tracked orders."""

    @pytest.mark.asyncio
    async def test_clear_all_orders(
        self,
        order_executor: OrderExecutor,
        mock_exchange: Mock,
    ):
        """Test clearing all tracked order IDs."""
        mock_exchange.futures_create_order = AsyncMock(
            side_effect=[
                MockOrder(order_id="entry_001"),
                MockOrder(order_id="tp_001"),
                MockOrder(order_id="sl_001"),
            ]
        )

        # Place orders
        signal = create_signal(SignalType.LONG, entry_price=49000)
        await order_executor.place_entry_order(signal, Decimal("0.01"))

        position = create_position(PositionSide.LONG, entry_price=49000, quantity=0.01)
        await order_executor.place_exit_orders(position)

        # Clear all
        order_executor.clear_all_orders()

        assert order_executor.pending_entry_order is None
        assert order_executor.take_profit_order is None
        assert order_executor.stop_loss_order is None
