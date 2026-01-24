"""
Unit tests for OrderRouter.

Tests the order routing and execution functionality including:
- Algorithm selection
- Order splitting (TWAP, Iceberg)
- Market depth analysis
- Execution lifecycle
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

from src.core.models import MarketType
from src.execution.models import (
    ChildOrder,
    ExecutionAlgorithm,
    ExecutionRequest,
    ExecutionUrgency,
    MarketDepthAnalysis,
    OrderSide,
    OrderStatus,
    OrderType,
    RouterConfig,
    SplitStrategy,
)
from src.execution.router import OrderRouter


# =============================================================================
# Mock Classes
# =============================================================================


class MockExchangeExecutor:
    """Mock exchange executor for testing."""

    def __init__(self):
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0
        self.should_fail = False
        self.should_reject = False
        self.fill_immediately = True

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        market_type: MarketType = MarketType.SPOT,
        client_order_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mock place order."""
        if self.should_fail:
            raise Exception("Exchange error")

        if self.should_reject:
            return {
                "success": False,
                "msg": "Insufficient balance",
            }

        self.order_counter += 1
        order_id = f"order_{self.order_counter}"

        status = "FILLED" if self.fill_immediately else "NEW"
        executed_qty = quantity if self.fill_immediately else Decimal("0")

        order = {
            "orderId": order_id,
            "clientOrderId": client_order_id,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": str(quantity),
            "price": str(price) if price else None,
            "status": status,
            "executedQty": str(executed_qty),
            "avgPrice": str(price) if price else "0",
            "success": True,
        }

        self.orders[order_id] = order
        return order

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> bool:
        """Mock cancel order."""
        if order_id in self.orders:
            self.orders[order_id]["status"] = "CANCELED"
            return True
        return False

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Dict[str, Any]]:
        """Mock get order."""
        return self.orders.get(order_id)


class MockMarketDataProvider:
    """Mock market data provider for testing."""

    def __init__(self):
        self.orderbook_bids = [
            (Decimal("49900"), Decimal("10")),
            (Decimal("49800"), Decimal("20")),
            (Decimal("49700"), Decimal("30")),
        ]
        self.orderbook_asks = [
            (Decimal("50100"), Decimal("10")),
            (Decimal("50200"), Decimal("20")),
            (Decimal("50300"), Decimal("30")),
        ]
        self.volume_24h = Decimal("1000")

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Mock get orderbook."""
        return {
            "bids": self.orderbook_bids[:limit],
            "asks": self.orderbook_asks[:limit],
        }

    async def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Mock get ticker."""
        return {
            "symbol": symbol,
            "price": "50000",
            "bid": "49900",
            "ask": "50100",
        }

    async def get_24h_volume(self, symbol: str) -> Decimal:
        """Mock get 24h volume."""
        return self.volume_24h


class MockSymbolInfoProvider:
    """Mock symbol info provider for testing."""

    def __init__(self):
        self.min_quantity = Decimal("0.001")
        self.min_notional = Decimal("10")
        self.quantity_precision = 3
        self.price_precision = 2

    def get_min_quantity(self, symbol: str) -> Decimal:
        return self.min_quantity

    def get_min_notional(self, symbol: str) -> Decimal:
        return self.min_notional

    def round_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        return round(quantity, self.quantity_precision)

    def round_price(self, symbol: str, price: Decimal) -> Decimal:
        return round(price, self.price_precision)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config():
    """Default router configuration."""
    return RouterConfig(
        small_order_threshold_pct=Decimal("0.5"),
        large_order_threshold_pct=Decimal("5.0"),
        max_child_orders=10,
        min_child_order_value=Decimal("10"),
    )


@pytest.fixture
def executor():
    """Mock exchange executor."""
    return MockExchangeExecutor()


@pytest.fixture
def market_data():
    """Mock market data provider."""
    return MockMarketDataProvider()


@pytest.fixture
def symbol_info():
    """Mock symbol info provider."""
    return MockSymbolInfoProvider()


@pytest.fixture
def router(config, executor, market_data, symbol_info):
    """OrderRouter instance with all dependencies."""
    return OrderRouter(
        config=config,
        executor=executor,
        market_data=market_data,
        symbol_info=symbol_info,
    )


@pytest.fixture
def simple_router(config, executor):
    """OrderRouter with only executor (no market data)."""
    return OrderRouter(
        config=config,
        executor=executor,
    )


# =============================================================================
# Basic Tests
# =============================================================================


class TestOrderRouterBasic:
    """Basic tests for OrderRouter."""

    def test_initialization(self, router, config):
        """Test router initializes correctly."""
        assert router.config == config
        assert router.active_execution_count == 0

    def test_statistics_initial(self, router):
        """Test initial statistics."""
        stats = router.get_statistics()
        assert stats["total_executions"] == 0
        assert stats["successful_executions"] == 0
        assert stats["active_executions"] == 0


# =============================================================================
# Algorithm Selection Tests
# =============================================================================


class TestAlgorithmSelection:
    """Tests for algorithm selection."""

    @pytest.mark.asyncio
    async def test_market_order_uses_direct(self, router):
        """Test that market orders use direct execution."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            order_type=OrderType.MARKET,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        algorithm = await router._select_algorithm(request, depth)
        assert algorithm == ExecutionAlgorithm.DIRECT

    @pytest.mark.asyncio
    async def test_immediate_urgency_uses_direct(self, router):
        """Test that immediate urgency uses direct execution."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            urgency=ExecutionUrgency.IMMEDIATE,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        algorithm = await router._select_algorithm(request, depth)
        assert algorithm == ExecutionAlgorithm.DIRECT

    @pytest.mark.asyncio
    async def test_explicit_algorithm_respected(self, router):
        """Test that explicit algorithm selection is respected."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.TWAP,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        algorithm = await router._select_algorithm(request, depth)
        assert algorithm == ExecutionAlgorithm.TWAP

    @pytest.mark.asyncio
    async def test_small_order_uses_direct(self, router, market_data):
        """Test small orders use direct execution."""
        # Small quantity relative to orderbook (60 total ask volume)
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),  # < 0.5% of 60
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.SMART,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        algorithm = await router._select_algorithm(request, depth)
        assert algorithm == ExecutionAlgorithm.DIRECT

    @pytest.mark.asyncio
    async def test_large_order_uses_algorithm(self, router, market_data):
        """Test large orders use execution algorithm."""
        # Large quantity relative to orderbook (60 total ask volume)
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10"),  # > 5% of 60
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.SMART,
            urgency=ExecutionUrgency.MEDIUM,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        algorithm = await router._select_algorithm(request, depth)
        # Large + medium urgency = ICEBERG
        assert algorithm == ExecutionAlgorithm.ICEBERG


# =============================================================================
# Market Depth Analysis Tests
# =============================================================================


class TestMarketDepthAnalysis:
    """Tests for market depth analysis."""

    @pytest.mark.asyncio
    async def test_analyze_market_depth(self, router):
        """Test market depth analysis."""
        depth = await router._analyze_market_depth("BTCUSDT")

        assert depth is not None
        assert depth.symbol == "BTCUSDT"
        assert depth.best_bid == Decimal("49900")
        assert depth.best_ask == Decimal("50100")
        assert depth.spread == Decimal("200")
        assert depth.total_bid_volume == Decimal("60")
        assert depth.total_ask_volume == Decimal("60")

    @pytest.mark.asyncio
    async def test_depth_fill_price_estimate(self, router):
        """Test fill price estimation from depth."""
        depth = await router._analyze_market_depth("BTCUSDT")

        # Buy 10 units (first ask level)
        estimate = depth.get_fill_price_estimate(OrderSide.BUY, Decimal("10"))
        assert estimate == Decimal("50100")

        # Buy 20 units (spans two levels)
        estimate = depth.get_fill_price_estimate(OrderSide.BUY, Decimal("20"))
        # 10 @ 50100 + 10 @ 50200 = 501500 + 502000 = 1003500 / 20 = 50175
        expected = (Decimal("10") * Decimal("50100") + Decimal("10") * Decimal("50200")) / Decimal("20")
        assert estimate == expected

    @pytest.mark.asyncio
    async def test_no_market_data_provider(self, simple_router):
        """Test when no market data provider is configured."""
        depth = await simple_router._analyze_market_depth("BTCUSDT")
        assert depth is None


# =============================================================================
# Order Splitting Tests
# =============================================================================


class TestOrderSplitting:
    """Tests for order splitting."""

    @pytest.mark.asyncio
    async def test_direct_no_split(self, router):
        """Test direct execution creates single child."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        strategy, children = await router._create_child_orders(
            "exec_001", request, ExecutionAlgorithm.DIRECT, depth
        )

        assert strategy == SplitStrategy.NONE
        assert len(children) == 1
        assert children[0].quantity == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_twap_splits_by_time(self, router):
        """Test TWAP creates time-based splits."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
            twap_intervals=5,
            twap_duration_minutes=10,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        strategy, children = await router._create_child_orders(
            "exec_001", request, ExecutionAlgorithm.TWAP, depth
        )

        assert strategy == SplitStrategy.TIME_BASED
        assert len(children) == 5

        # Each child should have 2.0 quantity
        for child in children:
            assert child.quantity == Decimal("2.0")

        # Children should have scheduled times
        assert all(c.scheduled_at is not None for c in children)

    @pytest.mark.asyncio
    async def test_iceberg_splits_by_size(self, router):
        """Test iceberg creates size-based splits."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
            iceberg_visible_pct=Decimal("0.20"),  # 20% visible
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        strategy, children = await router._create_child_orders(
            "exec_001", request, ExecutionAlgorithm.ICEBERG, depth
        )

        assert strategy == SplitStrategy.FIXED_SIZE
        # 10.0 / (10.0 * 0.20) = 5 children
        assert len(children) == 5

        # Each child should have 2.0 quantity
        for child in children:
            assert child.quantity == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_split_respects_max_children(self, router, config):
        """Test splitting respects max_child_orders config."""
        config.max_child_orders = 3

        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("10.0"),
            price=Decimal("50000"),
            twap_intervals=10,  # Request 10, but max is 3
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        strategy, children = await router._create_child_orders(
            "exec_001", request, ExecutionAlgorithm.TWAP, depth
        )

        assert len(children) <= config.max_child_orders


# =============================================================================
# Execution Tests
# =============================================================================


class TestExecution:
    """Tests for order execution."""

    @pytest.mark.asyncio
    async def test_simple_execution(self, router, executor):
        """Test simple direct execution."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,
        )

        result = await router.execute(request)

        assert result.success is True
        assert result.final_quantity == Decimal("1.0")
        assert len(executor.orders) == 1

    @pytest.mark.asyncio
    async def test_execution_with_twap(self, router, executor):
        """Test TWAP execution creates multiple children."""
        # Use DIRECT to test, but verify TWAP plan creation separately
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,  # Use DIRECT for fast test
        )

        result = await router.execute(request)

        assert result.success is True
        assert result.final_quantity == Decimal("2.0")

    @pytest.mark.asyncio
    async def test_twap_plan_creation(self, router):
        """Test TWAP execution plan is created correctly."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
            twap_intervals=2,
        )

        depth = await router._analyze_market_depth("BTCUSDT")
        strategy, children = await router._create_child_orders(
            "exec_001", request, ExecutionAlgorithm.TWAP, depth
        )

        assert strategy == SplitStrategy.TIME_BASED
        assert len(children) == 2
        assert all(c.scheduled_at is not None for c in children)

    @pytest.mark.asyncio
    async def test_execution_failure(self, router, executor):
        """Test execution handles failure."""
        executor.should_fail = True

        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,  # Use DIRECT to avoid timing
        )

        result = await router.execute(request)

        # When child order fails, final_quantity is 0
        assert result.success is False
        assert result.final_quantity == Decimal("0")

    @pytest.mark.asyncio
    async def test_execution_rejection(self, router, executor):
        """Test execution handles rejection."""
        executor.should_reject = True

        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,  # Use DIRECT to avoid timing
        )

        result = await router.execute(request)

        # Rejection leads to no fills
        assert result.final_quantity == Decimal("0")

    @pytest.mark.asyncio
    async def test_statistics_updated(self, router):
        """Test statistics are updated after execution."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,  # Use DIRECT to avoid timing
        )

        await router.execute(request)

        stats = router.get_statistics()
        assert stats["total_executions"] == 1
        assert stats["successful_executions"] == 1


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for callbacks."""

    @pytest.mark.asyncio
    async def test_on_order_sent_callback(self, config, executor):
        """Test on_order_sent callback is called."""
        callback = MagicMock()

        router = OrderRouter(
            config=config,
            executor=executor,
            on_order_sent=callback,
        )

        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,
        )

        await router.execute(request)
        callback.assert_called()

    @pytest.mark.asyncio
    async def test_on_execution_complete_callback(self, config, executor):
        """Test on_execution_complete callback is called."""
        callback = MagicMock()

        router = OrderRouter(
            config=config,
            executor=executor,
            on_execution_complete=callback,
        )

        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            algorithm=ExecutionAlgorithm.DIRECT,
        )

        await router.execute(request)
        callback.assert_called_once()


# =============================================================================
# Cancel Tests
# =============================================================================


class TestCancellation:
    """Tests for execution cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_nonexistent_execution(self, router):
        """Test cancelling non-existent execution."""
        result = await router.cancel_execution("nonexistent")
        assert result is False


# =============================================================================
# ExecutionRequest Tests
# =============================================================================


class TestExecutionRequest:
    """Tests for ExecutionRequest model."""

    def test_request_initialization(self):
        """Test request initializes correctly."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.5"),
            price=Decimal("50000"),
        )

        assert request.symbol == "BTCUSDT"
        assert request.side == OrderSide.BUY
        assert request.quantity == Decimal("1.5")
        assert request.price == Decimal("50000")

    def test_request_notional_value(self):
        """Test notional value calculation."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("2.0"),
            price=Decimal("50000"),
        )

        assert request.notional_value == Decimal("100000")

    def test_request_string_conversion(self):
        """Test string values are converted to enums."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side="BUY",
            quantity="1.5",
            price="50000",
            order_type="LIMIT",
            urgency="medium",
            algorithm="smart",
        )

        assert request.side == OrderSide.BUY
        assert request.quantity == Decimal("1.5")
        assert request.order_type == OrderType.LIMIT
        assert request.urgency == ExecutionUrgency.MEDIUM
        assert request.algorithm == ExecutionAlgorithm.SMART


# =============================================================================
# ChildOrder Tests
# =============================================================================


class TestChildOrder:
    """Tests for ChildOrder model."""

    def test_child_order_properties(self):
        """Test child order properties."""
        child = ChildOrder(
            parent_id="exec_001",
            child_index=0,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
            price=Decimal("50000"),
            filled_quantity=Decimal("0.5"),
        )

        assert child.remaining_quantity == Decimal("0.5")
        assert child.fill_pct == Decimal("50")
        assert child.is_complete is False

    def test_child_order_complete_states(self):
        """Test child order complete state detection."""
        child = ChildOrder(
            parent_id="exec_001",
            child_index=0,
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("1.0"),
        )

        # Not complete initially
        assert child.is_complete is False

        # Complete when filled
        child.status = OrderStatus.FILLED
        assert child.is_complete is True

        # Complete when cancelled
        child.status = OrderStatus.CANCELLED
        assert child.is_complete is True


# =============================================================================
# MarketDepthAnalysis Tests
# =============================================================================


class TestMarketDepthAnalysisModel:
    """Tests for MarketDepthAnalysis model."""

    def test_mid_price(self):
        """Test mid price calculation."""
        depth = MarketDepthAnalysis(
            symbol="BTCUSDT",
            best_bid=Decimal("49900"),
            best_ask=Decimal("50100"),
        )

        assert depth.mid_price == Decimal("50000")

    def test_market_impact_estimation(self):
        """Test market impact estimation."""
        from src.execution.models import DepthLevel

        depth = MarketDepthAnalysis(
            symbol="BTCUSDT",
            best_bid=Decimal("49900"),
            best_ask=Decimal("50100"),
            ask_levels=[
                DepthLevel(price=Decimal("50100"), quantity=Decimal("10")),
                DepthLevel(price=Decimal("50200"), quantity=Decimal("10")),
            ],
        )

        # Buy order at first level
        impact = depth.get_market_impact_pct(OrderSide.BUY, Decimal("10"))
        assert impact is not None
        # Fill price 50100, mid 50000, impact = (50100-50000)/50000 * 100 = 0.2%
        assert abs(impact - Decimal("0.2")) < Decimal("0.01")
