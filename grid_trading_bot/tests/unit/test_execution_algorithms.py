"""
Unit tests for Execution Algorithms.

Tests TWAP, VWAP, and Iceberg algorithms including:
- Slice generation
- Execution flow
- Progress tracking
- Pause/resume/cancel operations
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

from src.execution.models import (
    ExecutionAlgorithm,
    ExecutionRequest,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.execution.algorithms import (
    AlgorithmProgress,
    AlgorithmState,
    BaseExecutionAlgorithm,
    IcebergAlgorithm,
    IcebergConfig,
    TWAPAlgorithm,
    TWAPConfig,
    VWAPAlgorithm,
    VWAPConfig,
    create_algorithm,
)


# =============================================================================
# Mock Classes
# =============================================================================


class MockOrderExecutor:
    """Mock order executor for testing."""

    def __init__(self):
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.order_counter = 0
        self.fill_immediately = True
        self.should_reject = False

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Mock place order."""
        if self.should_reject:
            return {"success": False, "msg": "Rejected"}

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
    ) -> Optional[Dict[str, Any]]:
        """Mock get order."""
        return self.orders.get(order_id)


class MockPriceProvider:
    """Mock price provider for testing."""

    def __init__(self, price: Decimal = Decimal("50000")):
        self.price = price

    async def get_price(self, symbol: str) -> Decimal:
        return self.price


class MockVolumeProvider:
    """Mock volume provider for testing."""

    def __init__(self):
        self.current_volume = Decimal("100")
        self.volume_profile = {hour: Decimal("1") / Decimal("24") for hour in range(24)}

    async def get_current_volume(self, symbol: str) -> Decimal:
        return self.current_volume

    async def get_historical_volume_profile(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> Dict[int, Decimal]:
        return self.volume_profile


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def executor():
    """Mock order executor."""
    return MockOrderExecutor()


@pytest.fixture
def price_provider():
    """Mock price provider."""
    return MockPriceProvider()


@pytest.fixture
def volume_provider():
    """Mock volume provider."""
    return MockVolumeProvider()


@pytest.fixture
def basic_request():
    """Basic execution request."""
    return ExecutionRequest(
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=Decimal("10.0"),
        price=Decimal("50000"),
        order_type=OrderType.LIMIT,
    )


# =============================================================================
# AlgorithmProgress Tests
# =============================================================================


class TestAlgorithmProgress:
    """Tests for AlgorithmProgress."""

    def test_progress_initialization(self):
        """Test progress initializes correctly."""
        progress = AlgorithmProgress(total_quantity=Decimal("100"))

        assert progress.total_quantity == Decimal("100")
        assert progress.filled_quantity == Decimal("0")
        assert progress.fill_pct == Decimal("0")
        assert progress.state == AlgorithmState.IDLE

    def test_fill_pct_calculation(self):
        """Test fill percentage calculation."""
        progress = AlgorithmProgress(
            total_quantity=Decimal("100"),
            filled_quantity=Decimal("25"),
        )

        assert progress.fill_pct == Decimal("25")

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        progress = AlgorithmProgress(
            total_quantity=Decimal("100"),
            filled_quantity=Decimal("40"),
        )

        assert progress.remaining_quantity == Decimal("60")


# =============================================================================
# TWAPAlgorithm Tests
# =============================================================================


class TestTWAPAlgorithm:
    """Tests for TWAP algorithm."""

    def test_twap_initialization(self, basic_request, executor):
        """Test TWAP algorithm initializes correctly."""
        config = TWAPConfig(duration_minutes=10, num_slices=5)
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        assert algo.get_algorithm_type() == ExecutionAlgorithm.TWAP
        assert algo.progress.state == AlgorithmState.IDLE
        assert not algo.is_running

    def test_twap_slice_generation(self, basic_request, executor):
        """Test TWAP generates correct slices."""
        config = TWAPConfig(
            duration_minutes=10,
            num_slices=5,
            randomize_timing=False,
            randomize_size=False,
        )
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        slices = algo._generate_slices()

        assert len(slices) == 5
        # Each slice should have 2.0 quantity (10.0 / 5)
        for s in slices:
            assert s.quantity == Decimal("2.0")
        # All slices should have scheduled times
        assert all(s.scheduled_at is not None for s in slices)

    def test_twap_slice_generation_with_randomization(self, basic_request, executor):
        """Test TWAP with randomized slices."""
        config = TWAPConfig(
            duration_minutes=10,
            num_slices=5,
            randomize_timing=True,
            randomize_size=True,
            size_variation_pct=Decimal("0.10"),
        )
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        slices = algo._generate_slices()

        assert len(slices) == 5
        # Total quantity should still equal original
        total = sum(s.quantity for s in slices)
        assert total == basic_request.quantity

    @pytest.mark.asyncio
    async def test_twap_execution(self, basic_request, executor):
        """Test TWAP execution completes."""
        config = TWAPConfig(
            duration_minutes=0.05,  # 3 seconds total for testing
            num_slices=2,
            randomize_timing=False,
            randomize_size=False,
        )
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        # Execute (3 seconds duration with 2 slices = 1.5s interval)
        progress = await algo.execute()

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.filled_quantity == basic_request.quantity
        assert len(executor.orders) == 2

    @pytest.mark.asyncio
    async def test_twap_with_price_limit(self, basic_request, executor, price_provider):
        """Test TWAP respects price limits."""
        # Set price provider to return a price above limit
        price_provider.price = Decimal("60000")  # Way above 50000

        basic_request.max_slippage_pct = Decimal("1.0")  # 1% max slippage

        config = TWAPConfig(
            duration_minutes=0.05,  # 3 seconds for testing
            num_slices=2,
            randomize_timing=False,
        )
        algo = TWAPAlgorithm(
            basic_request,
            executor,
            config=config,
            price_provider=price_provider,
        )

        # The price check should fail, but algorithm still completes
        progress = await algo.execute()

        assert progress.state == AlgorithmState.COMPLETED

    def test_twap_pause_resume(self, basic_request, executor):
        """Test TWAP pause/resume."""
        config = TWAPConfig(duration_minutes=10, num_slices=5)
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        # Start state
        assert algo.progress.state == AlgorithmState.IDLE

        # Manually set to running then pause
        algo._progress.state = AlgorithmState.RUNNING
        algo.pause()
        assert algo.progress.state == AlgorithmState.PAUSED

        algo.resume()
        assert algo.progress.state == AlgorithmState.RUNNING


# =============================================================================
# VWAPAlgorithm Tests
# =============================================================================


class TestVWAPAlgorithm:
    """Tests for VWAP algorithm."""

    def test_vwap_initialization(self, basic_request, executor):
        """Test VWAP algorithm initializes correctly."""
        config = VWAPConfig(duration_minutes=10)
        algo = VWAPAlgorithm(basic_request, executor, config=config)

        assert algo.get_algorithm_type() == ExecutionAlgorithm.VWAP
        assert algo.progress.state == AlgorithmState.IDLE

    @pytest.mark.asyncio
    async def test_vwap_load_volume_profile(self, basic_request, executor, volume_provider):
        """Test VWAP loads volume profile."""
        config = VWAPConfig(duration_minutes=10)
        algo = VWAPAlgorithm(
            basic_request,
            executor,
            config=config,
            volume_provider=volume_provider,
        )

        await algo._load_volume_profile()

        assert len(algo._volume_profile) == 24
        assert all(hour in algo._volume_profile for hour in range(24))

    @pytest.mark.asyncio
    async def test_vwap_fallback_profile(self, basic_request, executor):
        """Test VWAP uses flat profile when no volume provider."""
        config = VWAPConfig(duration_minutes=10)
        algo = VWAPAlgorithm(basic_request, executor, config=config)

        await algo._load_volume_profile()

        # Should use flat profile
        assert len(algo._volume_profile) == 24
        expected_share = Decimal("1") / Decimal("24")
        for hour in range(24):
            assert algo._volume_profile[hour] == expected_share

    @pytest.mark.asyncio
    async def test_vwap_calculate_slice_quantity(
        self, basic_request, executor, volume_provider
    ):
        """Test VWAP slice quantity calculation."""
        config = VWAPConfig(
            duration_minutes=10,
            participation_rate=Decimal("0.10"),
        )
        algo = VWAPAlgorithm(
            basic_request,
            executor,
            config=config,
            volume_provider=volume_provider,
        )

        # Initialize progress
        algo._progress.start_time = datetime.now(timezone.utc)
        await algo._load_volume_profile()

        qty = await algo._calculate_slice_quantity()

        # Should be current_volume * participation = 100 * 0.10 = 10
        # But capped at remaining quantity
        assert qty > 0
        assert qty <= basic_request.quantity


# =============================================================================
# IcebergAlgorithm Tests
# =============================================================================


class TestIcebergAlgorithm:
    """Tests for Iceberg algorithm."""

    def test_iceberg_initialization(self, basic_request, executor):
        """Test Iceberg algorithm initializes correctly."""
        config = IcebergConfig(visible_quantity_pct=Decimal("0.20"))
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        assert algo.get_algorithm_type() == ExecutionAlgorithm.ICEBERG
        assert algo.progress.state == AlgorithmState.IDLE

    def test_iceberg_visible_quantity_calculation(self, basic_request, executor):
        """Test Iceberg calculates visible quantity correctly."""
        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.20"),
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        qty = algo._calculate_visible_quantity()

        # 10.0 * 0.20 = 2.0
        assert qty == Decimal("2.0")

    def test_iceberg_visible_quantity_with_minimum(self, basic_request, executor):
        """Test Iceberg respects minimum visible quantity."""
        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.01"),  # Very small
            min_visible_quantity=Decimal("1.0"),   # But minimum is 1.0
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        qty = algo._calculate_visible_quantity()

        # Should be min_visible_quantity since 10.0 * 0.01 = 0.1 < 1.0
        assert qty == Decimal("1.0")

    def test_iceberg_visible_quantity_capped_at_remaining(self, basic_request, executor):
        """Test Iceberg caps visible quantity at remaining."""
        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.50"),
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        # Simulate most of order filled
        algo._progress.filled_quantity = Decimal("9.5")

        qty = algo._calculate_visible_quantity()

        # Remaining is 0.5, so visible should be capped
        assert qty == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_iceberg_execution(self, basic_request, executor):
        """Test Iceberg execution completes."""
        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.50"),  # 50% visible = 2 slices
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        progress = await algo.execute()

        assert progress.state == AlgorithmState.COMPLETED
        assert progress.filled_quantity == basic_request.quantity
        # Should have 2 slices (50% * 2 = 100%)
        assert len(executor.orders) == 2

    @pytest.mark.asyncio
    async def test_iceberg_with_price_improvement(
        self, basic_request, executor, price_provider
    ):
        """Test Iceberg with price improvement."""
        # Set current price lower than request price (good for buy)
        price_provider.price = Decimal("49000")

        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.50"),
            price_improvement=True,
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(
            basic_request,
            executor,
            config=config,
            price_provider=price_provider,
        )

        progress = await algo.execute()

        assert progress.state == AlgorithmState.COMPLETED


# =============================================================================
# Factory Tests
# =============================================================================


class TestCreateAlgorithm:
    """Tests for algorithm factory function."""

    def test_create_twap(self, basic_request, executor):
        """Test creating TWAP algorithm."""
        algo = create_algorithm(
            ExecutionAlgorithm.TWAP,
            basic_request,
            executor,
        )

        assert isinstance(algo, TWAPAlgorithm)
        assert algo.get_algorithm_type() == ExecutionAlgorithm.TWAP

    def test_create_vwap(self, basic_request, executor):
        """Test creating VWAP algorithm."""
        algo = create_algorithm(
            ExecutionAlgorithm.VWAP,
            basic_request,
            executor,
        )

        assert isinstance(algo, VWAPAlgorithm)
        assert algo.get_algorithm_type() == ExecutionAlgorithm.VWAP

    def test_create_iceberg(self, basic_request, executor):
        """Test creating Iceberg algorithm."""
        algo = create_algorithm(
            ExecutionAlgorithm.ICEBERG,
            basic_request,
            executor,
        )

        assert isinstance(algo, IcebergAlgorithm)
        assert algo.get_algorithm_type() == ExecutionAlgorithm.ICEBERG

    def test_create_unsupported(self, basic_request, executor):
        """Test creating unsupported algorithm raises error."""
        with pytest.raises(ValueError):
            create_algorithm(
                ExecutionAlgorithm.DIRECT,  # DIRECT is not a complex algorithm
                basic_request,
                executor,
            )


# =============================================================================
# Cancel Tests
# =============================================================================


class TestAlgorithmCancel:
    """Tests for algorithm cancellation."""

    @pytest.mark.asyncio
    async def test_twap_cancel(self, basic_request, executor):
        """Test TWAP can be cancelled."""
        config = TWAPConfig(
            duration_minutes=1,  # 60 seconds, but we'll cancel quickly
            num_slices=10,
            randomize_timing=False,
        )
        algo = TWAPAlgorithm(basic_request, executor, config=config)

        # Start execution in background
        task = asyncio.create_task(algo.execute())

        # Wait a bit then cancel (cancellable_sleep checks every 0.5s)
        await asyncio.sleep(0.2)
        await algo.cancel()

        # Wait for task to complete (should be fast due to cancel)
        progress = await task

        assert progress.state == AlgorithmState.CANCELLED

    @pytest.mark.asyncio
    async def test_iceberg_cancel(self, basic_request, executor):
        """Test Iceberg can be cancelled."""
        # Make executor slow to fill
        executor.fill_immediately = False

        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.10"),
            fill_timeout_seconds=60,  # Long timeout
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        # Start execution in background
        task = asyncio.create_task(algo.execute())

        # Wait a bit then cancel
        await asyncio.sleep(0.1)
        await algo.cancel()

        # Wait for task to complete
        progress = await task

        assert progress.state == AlgorithmState.CANCELLED


# =============================================================================
# Callback Tests
# =============================================================================


class TestAlgorithmCallbacks:
    """Tests for algorithm callbacks."""

    @pytest.mark.asyncio
    async def test_on_slice_complete_callback(self, basic_request, executor):
        """Test on_slice_complete callback is called."""
        callback = MagicMock()

        config = TWAPConfig(
            duration_minutes=0.05,  # 3 seconds for testing
            num_slices=2,
            randomize_timing=False,
        )
        algo = TWAPAlgorithm(
            basic_request,
            executor,
            config=config,
            on_slice_complete=callback,
        )

        await algo.execute()

        # Callback should be called for each slice
        assert callback.call_count == 2

    @pytest.mark.asyncio
    async def test_on_progress_callback(self, basic_request, executor):
        """Test on_progress callback is called."""
        callback = MagicMock()

        config = TWAPConfig(
            duration_minutes=0.05,  # 3 seconds for testing
            num_slices=2,
            randomize_timing=False,
        )
        algo = TWAPAlgorithm(
            basic_request,
            executor,
            config=config,
            on_progress=callback,
        )

        await algo.execute()

        # Callback should be called after each slice
        assert callback.call_count >= 2


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_zero_quantity_slice(self, basic_request, executor):
        """Test handling of zero remaining quantity."""
        config = IcebergConfig(
            visible_quantity_pct=Decimal("0.20"),
            random_visible_size=False,
        )
        algo = IcebergAlgorithm(basic_request, executor, config=config)

        # Simulate all filled
        algo._progress.filled_quantity = basic_request.quantity

        qty = algo._calculate_visible_quantity()
        assert qty == Decimal("0")

    def test_small_quantity_rounding(self, executor):
        """Test small quantity doesn't cause issues."""
        request = ExecutionRequest(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            quantity=Decimal("0.001"),
            price=Decimal("50000"),
        )

        config = TWAPConfig(
            duration_minutes=1,
            num_slices=5,
            randomize_timing=False,
            randomize_size=False,
        )
        algo = TWAPAlgorithm(request, executor, config=config)

        slices = algo._generate_slices()

        # Total should still equal original
        total = sum(s.quantity for s in slices)
        assert total == request.quantity
