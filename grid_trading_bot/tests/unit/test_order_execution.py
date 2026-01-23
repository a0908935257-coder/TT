"""
Unit tests for order execution - limit orders, partial fills, OrderBook.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.backtest.config import BacktestConfig
from src.backtest.order import (
    OrderType,
    OrderSide,
    OrderTimeInForce,
    OrderStatus,
    PendingOrder,
    Fill,
    OrderBook,
)
from src.core.models import Kline, KlineInterval


def create_test_kline(
    open_price: Decimal = Decimal("100"),
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("95"),
    close: Decimal = Decimal("102"),
) -> Kline:
    """Create a test kline."""
    return Kline(
        symbol="BTCUSDT",
        interval=KlineInterval.h1,
        open_time=datetime(2024, 1, 1, 0, 0),
        close_time=datetime(2024, 1, 1, 0, 59),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=Decimal("1000"),
        quote_volume=Decimal("100000"),
        trades_count=100,
    )


class TestPendingOrder:
    """Tests for PendingOrder dataclass."""

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("10"),
            limit_price=Decimal("100"),
            filled_quantity=Decimal("3"),
        )
        assert order.remaining_quantity == Decimal("7")

    def test_is_buy_property(self):
        """Test is_buy property."""
        buy_order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        sell_order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=Decimal("1"),
        )
        assert buy_order.is_buy is True
        assert sell_order.is_buy is False

    def test_is_limit_property(self):
        """Test is_limit property."""
        market = PendingOrder(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        limit = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        stop_limit = PendingOrder(
            order_type=OrderType.STOP_LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        assert market.is_limit is False
        assert limit.is_limit is True
        assert stop_limit.is_limit is True

    def test_is_expired_gtd(self):
        """Test GTD expiration."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            time_in_force=OrderTimeInForce.GTD,
            expiry_bar=10,
        )
        assert order.is_expired(5) is False
        assert order.is_expired(10) is True
        assert order.is_expired(15) is True

    def test_gtc_never_expires(self):
        """Test GTC orders never expire."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            time_in_force=OrderTimeInForce.GTC,
        )
        assert order.is_expired(1000000) is False


class TestOrderBook:
    """Tests for OrderBook class."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BacktestConfig()

    @pytest.fixture
    def order_book(self, config):
        """Create test order book."""
        return OrderBook(config)

    def test_add_order(self, order_book):
        """Test adding an order."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("100"),
        )
        order_id = order_book.add_order(order)
        assert order_id == order.order_id
        assert order_book.order_count == 1

    def test_cancel_order(self, order_book):
        """Test cancelling an order."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("100"),
        )
        order_id = order_book.add_order(order)
        assert order_book.cancel_order(order_id) is True
        assert order.status == OrderStatus.CANCELLED
        assert order_book.order_count == 0

    def test_cancel_nonexistent_order(self, order_book):
        """Test cancelling nonexistent order returns False."""
        assert order_book.cancel_order("fake-id") is False

    def test_get_order(self, order_book):
        """Test getting an order by ID."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("100"),
        )
        order_id = order_book.add_order(order)
        retrieved = order_book.get_order(order_id)
        assert retrieved is order

    def test_market_order_fills_immediately(self, order_book):
        """Test market orders fill at close price."""
        order = PendingOrder(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        order_book.add_order(order)

        kline = create_test_kline(close=Decimal("102"))
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("102")
        assert fills[0].fill_quantity == Decimal("1")
        assert order.status == OrderStatus.FILLED

    def test_buy_limit_fills_below_limit(self, order_book):
        """Test buy limit fills when price touches limit."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("98"),
        )
        order_book.add_order(order)

        # Price goes down to 95, should fill at limit 98
        kline = create_test_kline(
            open_price=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
        )
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("98")
        assert order.status == OrderStatus.FILLED

    def test_buy_limit_not_filled_above_limit(self, order_book):
        """Test buy limit doesn't fill when price stays above limit."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("90"),
        )
        order_book.add_order(order)

        # Low is 95, above limit of 90
        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 0
        assert order.status == OrderStatus.PENDING

    def test_sell_limit_fills_above_limit(self, order_book):
        """Test sell limit fills when price touches limit."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            limit_price=Decimal("103"),
        )
        order_book.add_order(order)

        # Price goes up to 105, should fill at limit 103
        kline = create_test_kline(
            open_price=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
        )
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("103")
        assert order.status == OrderStatus.FILLED

    def test_stop_market_buy_triggers(self, order_book):
        """Test stop market buy triggers when price goes above stop."""
        order = PendingOrder(
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            stop_price=Decimal("103"),
        )
        order_book.add_order(order)

        kline = create_test_kline(
            open_price=Decimal("100"),
            high=Decimal("105"),
        )
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("103")
        assert order.status == OrderStatus.FILLED

    def test_stop_market_sell_triggers(self, order_book):
        """Test stop market sell triggers when price goes below stop."""
        order = PendingOrder(
            order_type=OrderType.STOP_MARKET,
            side=OrderSide.SELL,
            quantity=Decimal("1"),
            stop_price=Decimal("97"),
        )
        order_book.add_order(order)

        kline = create_test_kline(
            open_price=Decimal("100"),
            low=Decimal("95"),
        )
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].fill_price == Decimal("97")
        assert order.status == OrderStatus.FILLED

    def test_order_expiration(self, order_book):
        """Test GTD order expires."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("90"),
            time_in_force=OrderTimeInForce.GTD,
            expiry_bar=5,
        )
        order_book.add_order(order)

        kline = create_test_kline()
        fills = order_book.process_bar(kline, bar_idx=5)

        assert len(fills) == 0
        assert order.status == OrderStatus.EXPIRED

    def test_fok_all_or_nothing(self, config):
        """Test FOK order cancels if can't fill completely."""
        order_book = OrderBook(config, enable_partial_fills=True)

        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1000000"),  # Very large order
            limit_price=Decimal("98"),
            time_in_force=OrderTimeInForce.FOK,
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(
            kline, bar_idx=0, avg_volume=Decimal("100")  # Low volume
        )

        assert len(fills) == 0
        assert order.status == OrderStatus.CANCELLED

    def test_ioc_partial_fill_then_cancel(self, config):
        """Test IOC order fills partially then cancels rest."""
        order_book = OrderBook(config, enable_partial_fills=True)

        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # Large but not huge
            limit_price=Decimal("98"),
            time_in_force=OrderTimeInForce.IOC,
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(
            kline, bar_idx=0, avg_volume=Decimal("10")  # Low volume
        )

        assert len(fills) == 1
        assert order.status == OrderStatus.CANCELLED  # Cancelled after partial fill
        assert fills[0].fill_quantity < Decimal("100")  # Partial fill

    def test_limit_order_maker_fee(self, order_book):
        """Test limit orders get maker fee rate."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("98"),
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].is_maker is True

    def test_market_order_taker_fee(self, order_book):
        """Test market orders get taker fee rate."""
        order = PendingOrder(
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
        )
        order_book.add_order(order)

        kline = create_test_kline()
        fills = order_book.process_bar(kline, bar_idx=0)

        assert len(fills) == 1
        assert fills[0].is_maker is False

    def test_reset(self, order_book):
        """Test order book reset."""
        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("1"),
            limit_price=Decimal("100"),
        )
        order_book.add_order(order)
        assert order_book.order_count == 1

        order_book.reset()
        assert order_book.order_count == 0


class TestPartialFills:
    """Tests for partial fill logic."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return BacktestConfig()

    def test_small_order_full_fill(self, config):
        """Test small orders (< 1% volume) get full fill."""
        order_book = OrderBook(config, enable_partial_fills=True)

        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),  # Very small
            limit_price=Decimal("100"),
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(
            kline, bar_idx=0, avg_volume=Decimal("1000")
        )

        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal("0.1")
        assert order.status == OrderStatus.FILLED

    def test_large_order_partial_fill(self, config):
        """Test large orders get partial fills."""
        order_book = OrderBook(config, enable_partial_fills=True)

        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("100"),  # Large relative to volume
            limit_price=Decimal("100"),
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"), close=Decimal("100"))
        fills = order_book.process_bar(
            kline, bar_idx=0, avg_volume=Decimal("10")  # Low volume
        )

        assert len(fills) == 1
        assert fills[0].fill_quantity < Decimal("100")  # Partial fill
        assert order.status == OrderStatus.PARTIALLY_FILLED

    def test_without_partial_fills_enabled(self, config):
        """Test full fills when partial fills disabled."""
        order_book = OrderBook(config, enable_partial_fills=False)

        order = PendingOrder(
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=Decimal("100"),
            limit_price=Decimal("100"),
        )
        order_book.add_order(order)

        kline = create_test_kline(low=Decimal("95"))
        fills = order_book.process_bar(
            kline, bar_idx=0, avg_volume=Decimal("10")
        )

        assert len(fills) == 1
        assert fills[0].fill_quantity == Decimal("100")  # Full fill
        assert order.status == OrderStatus.FILLED


class TestFill:
    """Tests for Fill dataclass."""

    def test_fill_creation(self):
        """Test Fill dataclass creation."""
        fill = Fill(
            order_id="test-123",
            fill_price=Decimal("100"),
            fill_quantity=Decimal("1"),
            fill_bar=5,
            fill_time=datetime(2024, 1, 1, 12, 0),
            fee=Decimal("0.04"),
            is_maker=True,
        )

        assert fill.order_id == "test-123"
        assert fill.fill_price == Decimal("100")
        assert fill.fill_quantity == Decimal("1")
        assert fill.fill_bar == 5
        assert fill.fee == Decimal("0.04")
        assert fill.is_maker is True

    def test_fill_default_values(self):
        """Test Fill default values."""
        fill = Fill(
            order_id="test",
            fill_price=Decimal("100"),
            fill_quantity=Decimal("1"),
            fill_bar=0,
            fill_time=datetime.now(),
        )

        assert fill.fee == Decimal("0")
        assert fill.is_maker is False
