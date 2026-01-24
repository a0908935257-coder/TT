"""
Execution Algorithms.

Provides sophisticated execution algorithms for large order execution:
- TWAP (Time-Weighted Average Price)
- VWAP (Volume-Weighted Average Price)
- Iceberg (Hidden quantity)

These algorithms help minimize market impact and achieve better execution prices.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.core import get_logger
from src.execution.models import (
    ChildOrder,
    ExecutionAlgorithm,
    ExecutionRequest,
    MarketDepthAnalysis,
    OrderSide,
    OrderStatus,
    OrderType,
    SplitStrategy,
)

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class PriceProvider(Protocol):
    """Protocol for getting current price."""

    async def get_price(self, symbol: str) -> Decimal:
        """Get current market price."""
        ...


class VolumeProvider(Protocol):
    """Protocol for volume data."""

    async def get_current_volume(self, symbol: str) -> Decimal:
        """Get current period volume."""
        ...

    async def get_historical_volume_profile(
        self,
        symbol: str,
        lookback_days: int = 30,
    ) -> Dict[int, Decimal]:
        """
        Get historical volume profile by hour.

        Returns:
            Dict mapping hour (0-23) to average volume
        """
        ...


class OrderExecutor(Protocol):
    """Protocol for order execution."""

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
        """Place an order."""
        ...

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Cancel an order."""
        ...

    async def get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Get order status."""
        ...


# =============================================================================
# Algorithm State
# =============================================================================


class AlgorithmState(str, Enum):
    """Execution algorithm state."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class AlgorithmProgress:
    """Progress tracking for algorithm execution."""

    total_quantity: Decimal
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    total_slices: int = 0
    completed_slices: int = 0
    average_fill_price: Optional[Decimal] = None
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    state: AlgorithmState = AlgorithmState.IDLE

    @property
    def fill_pct(self) -> Decimal:
        """Get fill percentage."""
        if self.total_quantity <= 0:
            return Decimal("0")
        return (self.filled_quantity / self.total_quantity) * Decimal("100")

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity."""
        return self.total_quantity - self.filled_quantity

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        if not self.start_time:
            return 0.0
        end = self.end_time or datetime.now(timezone.utc)
        return (end - self.start_time).total_seconds()


# =============================================================================
# Base Algorithm
# =============================================================================


class BaseExecutionAlgorithm(ABC):
    """
    Base class for execution algorithms.

    Provides common functionality for all algorithms:
    - Progress tracking
    - Price limit enforcement
    - Pause/resume/cancel operations
    - Statistics and metrics
    """

    def __init__(
        self,
        request: ExecutionRequest,
        executor: OrderExecutor,
        price_provider: Optional[PriceProvider] = None,
        on_slice_complete: Optional[Callable[[ChildOrder], None]] = None,
        on_progress: Optional[Callable[[AlgorithmProgress], None]] = None,
    ):
        """
        Initialize algorithm.

        Args:
            request: Execution request
            executor: Order executor
            price_provider: Price provider for limit checks
            on_slice_complete: Callback when a slice completes
            on_progress: Callback for progress updates
        """
        self._request = request
        self._executor = executor
        self._price_provider = price_provider
        self._on_slice_complete = on_slice_complete
        self._on_progress = on_progress

        # State
        self._progress = AlgorithmProgress(total_quantity=request.quantity)
        self._children: List[ChildOrder] = []
        self._cancelled = False
        self._paused = False

        # Price limits
        self._price_limit_upper: Optional[Decimal] = None
        self._price_limit_lower: Optional[Decimal] = None
        self._setup_price_limits()

    def _setup_price_limits(self) -> None:
        """Setup price limits based on max slippage."""
        if self._request.price and self._request.max_slippage_pct:
            slippage = self._request.max_slippage_pct / Decimal("100")

            if self._request.side == OrderSide.BUY:
                # For buys, limit how high we pay
                self._price_limit_upper = self._request.price * (1 + slippage)
            else:
                # For sells, limit how low we accept
                self._price_limit_lower = self._request.price * (1 - slippage)

    @property
    def progress(self) -> AlgorithmProgress:
        """Get current progress."""
        return self._progress

    @property
    def children(self) -> List[ChildOrder]:
        """Get child orders."""
        return self._children

    @property
    def is_running(self) -> bool:
        """Check if algorithm is running."""
        return self._progress.state == AlgorithmState.RUNNING

    @property
    def is_complete(self) -> bool:
        """Check if algorithm is complete."""
        return self._progress.state in (
            AlgorithmState.COMPLETED,
            AlgorithmState.CANCELLED,
            AlgorithmState.ERROR,
        )

    # =========================================================================
    # Abstract Methods
    # =========================================================================

    @abstractmethod
    async def execute(self) -> AlgorithmProgress:
        """
        Execute the algorithm.

        Returns:
            Final progress
        """
        pass

    @abstractmethod
    def get_algorithm_type(self) -> ExecutionAlgorithm:
        """Get algorithm type."""
        pass

    # =========================================================================
    # Control Methods
    # =========================================================================

    def pause(self) -> None:
        """Pause execution."""
        if self._progress.state == AlgorithmState.RUNNING:
            self._paused = True
            self._progress.state = AlgorithmState.PAUSED
            logger.info("Algorithm paused")

    def resume(self) -> None:
        """Resume execution."""
        if self._progress.state == AlgorithmState.PAUSED:
            self._paused = False
            self._progress.state = AlgorithmState.RUNNING
            logger.info("Algorithm resumed")

    async def cancel(self) -> None:
        """Cancel execution."""
        self._cancelled = True
        self._progress.state = AlgorithmState.CANCELLED

        # Cancel any pending orders
        for child in self._children:
            if child.exchange_order_id and not child.is_complete:
                try:
                    await self._executor.cancel_order(
                        symbol=child.symbol,
                        order_id=child.exchange_order_id,
                    )
                    child.status = OrderStatus.CANCELLED
                except Exception as e:
                    logger.warning(f"Failed to cancel order: {e}")

        logger.info("Algorithm cancelled")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    async def _check_price_limit(self) -> bool:
        """
        Check if current price is within limits.

        Returns:
            True if within limits, False otherwise
        """
        if not self._price_provider:
            return True

        try:
            current_price = await self._price_provider.get_price(self._request.symbol)

            if self._price_limit_upper and current_price > self._price_limit_upper:
                logger.warning(
                    f"Price {current_price} exceeds upper limit {self._price_limit_upper}"
                )
                return False

            if self._price_limit_lower and current_price < self._price_limit_lower:
                logger.warning(
                    f"Price {current_price} below lower limit {self._price_limit_lower}"
                )
                return False

            return True

        except Exception as e:
            logger.warning(f"Failed to check price: {e}")
            return True  # Continue if price check fails

    async def _execute_slice(
        self,
        child: ChildOrder,
        wait_for_fill: bool = True,
        fill_timeout: int = 30,
    ) -> bool:
        """
        Execute a single slice (child order).

        Args:
            child: Child order to execute
            wait_for_fill: Whether to wait for fill
            fill_timeout: Timeout for fill wait

        Returns:
            True if executed successfully
        """
        try:
            child.status = OrderStatus.SUBMITTED
            child.sent_at = datetime.now(timezone.utc)

            result = await self._executor.place_order(
                symbol=child.symbol,
                side=child.side.value,
                order_type=child.order_type.value,
                quantity=child.quantity,
                price=child.price,
                client_order_id=child.client_order_id,
            )

            if result.get("success", False) or result.get("orderId"):
                child.exchange_order_id = str(result.get("orderId", ""))
                child.status = OrderStatus.ACCEPTED

                # Check immediate fill
                status = result.get("status", "NEW")
                if status == "FILLED":
                    child.status = OrderStatus.FILLED
                    child.filled_quantity = Decimal(str(result.get("executedQty", child.quantity)))
                    child.average_price = Decimal(str(result.get("avgPrice", result.get("price", child.price or 0))))
                    child.filled_at = datetime.now(timezone.utc)
                elif status == "PARTIALLY_FILLED":
                    child.status = OrderStatus.PARTIALLY_FILLED
                    child.filled_quantity = Decimal(str(result.get("executedQty", 0)))

                # Wait for fill if needed
                if wait_for_fill and child.status not in (OrderStatus.FILLED, OrderStatus.CANCELLED):
                    await self._wait_for_fill(child, fill_timeout)

                return True

            else:
                child.status = OrderStatus.REJECTED
                child.error_message = result.get("msg", "Order rejected")
                return False

        except Exception as e:
            child.status = OrderStatus.FAILED
            child.error_message = str(e)
            logger.error(f"Slice execution failed: {e}")
            return False

    async def _wait_for_fill(
        self,
        child: ChildOrder,
        timeout_seconds: int = 30,
        poll_interval: float = 1.0,
    ) -> None:
        """Wait for order fill."""
        start_time = datetime.now(timezone.utc)
        timeout = timedelta(seconds=timeout_seconds)

        while datetime.now(timezone.utc) - start_time < timeout:
            if child.is_complete or self._cancelled:
                break

            try:
                order_info = await self._executor.get_order(
                    symbol=child.symbol,
                    order_id=child.exchange_order_id or "",
                )

                if order_info:
                    status = order_info.get("status", "")

                    if status == "FILLED":
                        child.status = OrderStatus.FILLED
                        child.filled_quantity = Decimal(str(order_info.get("executedQty", child.quantity)))
                        child.average_price = Decimal(str(order_info.get("avgPrice", order_info.get("price", child.price or 0))))
                        child.filled_at = datetime.now(timezone.utc)
                        break

                    elif status == "PARTIALLY_FILLED":
                        child.status = OrderStatus.PARTIALLY_FILLED
                        child.filled_quantity = Decimal(str(order_info.get("executedQty", 0)))

                    elif status in ("CANCELED", "CANCELLED"):
                        child.status = OrderStatus.CANCELLED
                        break

            except Exception as e:
                logger.warning(f"Error polling order: {e}")

            await asyncio.sleep(poll_interval)

    def _update_progress(self) -> None:
        """Update progress from child orders."""
        self._progress.filled_quantity = sum(c.filled_quantity for c in self._children)
        self._progress.completed_slices = sum(1 for c in self._children if c.is_complete)
        self._progress.total_fees = sum(c.fee for c in self._children)

        # Calculate average price
        filled_orders = [(c.filled_quantity, c.average_price)
                         for c in self._children
                         if c.filled_quantity > 0 and c.average_price]
        if filled_orders:
            total_qty = sum(qty for qty, _ in filled_orders)
            if total_qty > 0:
                weighted_sum = sum(qty * price for qty, price in filled_orders)
                self._progress.average_fill_price = weighted_sum / total_qty

        # Callback
        if self._on_progress:
            self._on_progress(self._progress)

    async def _cancellable_sleep(
        self,
        seconds: float,
        check_interval: float = 0.5,
    ) -> None:
        """
        Sleep with periodic cancellation checks.

        Args:
            seconds: Total seconds to sleep
            check_interval: Interval between cancel checks
        """
        remaining = seconds
        while remaining > 0 and not self._cancelled:
            sleep_time = min(remaining, check_interval)
            await asyncio.sleep(sleep_time)
            remaining -= sleep_time

    def _create_child(
        self,
        index: int,
        quantity: Decimal,
        scheduled_at: Optional[datetime] = None,
    ) -> ChildOrder:
        """Create a child order."""
        return ChildOrder(
            parent_id=self._request.client_order_id or "algo",
            child_index=index,
            symbol=self._request.symbol,
            side=self._request.side,
            quantity=quantity,
            price=self._request.price,
            order_type=self._request.order_type,
            scheduled_at=scheduled_at,
            client_order_id=f"{self._request.client_order_id or 'algo'}-{index}",
        )


# =============================================================================
# TWAP Algorithm
# =============================================================================


@dataclass
class TWAPConfig:
    """Configuration for TWAP algorithm."""

    duration_minutes: float = 30         # Total execution duration (supports fractional)
    num_slices: int = 10                 # Number of slices
    randomize_timing: bool = True        # Add random jitter to timing
    randomize_size: bool = True          # Add random variation to size
    timing_jitter_pct: Decimal = field(default_factory=lambda: Decimal("0.10"))  # 10%
    size_variation_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))  # 5%
    adaptive_pricing: bool = True        # Adjust price based on market
    price_chase_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))    # Chase price by 0.05%
    min_slice_value: Decimal = field(default_factory=lambda: Decimal("10"))


class TWAPAlgorithm(BaseExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) Algorithm.

    Splits an order into equal-sized slices executed at regular intervals.
    Features:
    - Randomized timing to avoid detection
    - Randomized slice sizes
    - Adaptive pricing based on market conditions
    - Price limit enforcement
    - Pause/resume/cancel support

    Example:
        >>> config = TWAPConfig(duration_minutes=30, num_slices=10)
        >>> algo = TWAPAlgorithm(request, executor, config=config)
        >>> progress = await algo.execute()
    """

    def __init__(
        self,
        request: ExecutionRequest,
        executor: OrderExecutor,
        config: Optional[TWAPConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        on_slice_complete: Optional[Callable[[ChildOrder], None]] = None,
        on_progress: Optional[Callable[[AlgorithmProgress], None]] = None,
    ):
        """
        Initialize TWAP algorithm.

        Args:
            request: Execution request
            executor: Order executor
            config: TWAP configuration
            price_provider: Price provider
            on_slice_complete: Slice completion callback
            on_progress: Progress callback
        """
        super().__init__(
            request=request,
            executor=executor,
            price_provider=price_provider,
            on_slice_complete=on_slice_complete,
            on_progress=on_progress,
        )

        self._config = config or TWAPConfig(
            duration_minutes=request.twap_duration_minutes,
            num_slices=request.twap_intervals,
        )

        self._interval_seconds = (self._config.duration_minutes * 60) / self._config.num_slices

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        return ExecutionAlgorithm.TWAP

    async def execute(self) -> AlgorithmProgress:
        """Execute TWAP algorithm."""
        self._progress.state = AlgorithmState.RUNNING
        self._progress.start_time = datetime.now(timezone.utc)
        self._progress.total_slices = self._config.num_slices

        logger.info(
            f"Starting TWAP: {self._request.quantity} {self._request.symbol} "
            f"over {self._config.duration_minutes}m in {self._config.num_slices} slices"
        )

        try:
            # Generate slice schedule
            slices = self._generate_slices()
            self._children = slices
            self._progress.total_slices = len(slices)

            # Execute slices
            for i, child in enumerate(slices):
                if self._cancelled:
                    break

                # Wait while paused
                while self._paused and not self._cancelled:
                    await asyncio.sleep(0.5)

                # Wait until scheduled time (with cancel check)
                if child.scheduled_at:
                    now = datetime.now(timezone.utc)
                    if child.scheduled_at > now:
                        wait_seconds = (child.scheduled_at - now).total_seconds()
                        logger.debug(f"Waiting {wait_seconds:.1f}s for slice {i+1}")
                        # Break sleep into chunks to check for cancel
                        await self._cancellable_sleep(wait_seconds)

                # Check price limits
                if not await self._check_price_limit():
                    logger.warning(f"Skipping slice {i+1} due to price limit")
                    continue

                # Adaptive pricing
                if self._config.adaptive_pricing and self._price_provider:
                    child.price = await self._get_adaptive_price()

                # Execute slice
                logger.info(f"Executing TWAP slice {i+1}/{len(slices)}: {child.quantity}")
                success = await self._execute_slice(child)

                if success and self._on_slice_complete:
                    self._on_slice_complete(child)

                self._update_progress()

            # Finalize
            self._progress.end_time = datetime.now(timezone.utc)

            if self._cancelled:
                self._progress.state = AlgorithmState.CANCELLED
            else:
                self._progress.state = AlgorithmState.COMPLETED

            logger.info(
                f"TWAP complete: filled {self._progress.filled_quantity}/{self._request.quantity} "
                f"({self._progress.fill_pct:.1f}%)"
            )

        except Exception as e:
            self._progress.state = AlgorithmState.ERROR
            self._progress.end_time = datetime.now(timezone.utc)
            logger.error(f"TWAP execution error: {e}")

        return self._progress

    def _generate_slices(self) -> List[ChildOrder]:
        """Generate TWAP slices with optional randomization."""
        import random

        num_slices = self._config.num_slices
        base_quantity = self._request.quantity / Decimal(str(num_slices))

        slices = []
        now = datetime.now(timezone.utc)
        remaining = self._request.quantity

        for i in range(num_slices):
            # Calculate timing
            base_time = now + timedelta(seconds=i * self._interval_seconds)

            if self._config.randomize_timing and i > 0:
                # Add jitter
                jitter_max = float(self._interval_seconds * float(self._config.timing_jitter_pct))
                jitter = random.uniform(-jitter_max, jitter_max)
                scheduled_at = base_time + timedelta(seconds=jitter)
            else:
                scheduled_at = base_time

            # Calculate quantity
            if i == num_slices - 1:
                # Last slice gets remaining
                quantity = remaining
            else:
                quantity = base_quantity

                if self._config.randomize_size:
                    # Add variation
                    variation = float(self._config.size_variation_pct)
                    factor = Decimal(str(1 + random.uniform(-variation, variation)))
                    quantity = quantity * factor

                # Don't exceed remaining
                quantity = min(quantity, remaining)

            remaining -= quantity

            child = self._create_child(i, quantity, scheduled_at)
            slices.append(child)

        return slices

    async def _get_adaptive_price(self) -> Optional[Decimal]:
        """Get adaptive price based on market."""
        if not self._price_provider or not self._request.price:
            return self._request.price

        try:
            current_price = await self._price_provider.get_price(self._request.symbol)
            base_price = self._request.price

            # Chase price if moved away
            if self._request.side == OrderSide.BUY:
                if current_price > base_price:
                    # Market moved up, chase it a bit
                    chase = base_price * (1 + self._config.price_chase_pct / Decimal("100"))
                    return min(current_price, chase)
            else:
                if current_price < base_price:
                    # Market moved down, chase it a bit
                    chase = base_price * (1 - self._config.price_chase_pct / Decimal("100"))
                    return max(current_price, chase)

            return base_price

        except Exception:
            return self._request.price


# =============================================================================
# VWAP Algorithm
# =============================================================================


@dataclass
class VWAPConfig:
    """Configuration for VWAP algorithm."""

    duration_minutes: float = 60         # Total execution duration (supports fractional)
    participation_rate: Decimal = field(default_factory=lambda: Decimal("0.10"))  # Target 10% of volume
    min_slice_interval_seconds: int = 30   # Minimum time between slices
    max_slice_interval_seconds: int = 300  # Maximum time between slices
    volume_lookback_days: int = 30         # Days for volume profile
    adaptive_participation: bool = True    # Adjust rate based on actual volume
    max_participation_rate: Decimal = field(default_factory=lambda: Decimal("0.25"))  # Cap at 25%
    min_slice_value: Decimal = field(default_factory=lambda: Decimal("10"))


class VWAPAlgorithm(BaseExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) Algorithm.

    Executes orders proportionally to market volume, aiming to achieve
    a fill price close to the market VWAP.

    Features:
    - Historical volume profile analysis
    - Real-time volume tracking
    - Adaptive participation rate
    - Price limit enforcement

    Example:
        >>> config = VWAPConfig(duration_minutes=60, participation_rate=Decimal("0.10"))
        >>> algo = VWAPAlgorithm(request, executor, config=config, volume_provider=provider)
        >>> progress = await algo.execute()
    """

    def __init__(
        self,
        request: ExecutionRequest,
        executor: OrderExecutor,
        config: Optional[VWAPConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        volume_provider: Optional[VolumeProvider] = None,
        on_slice_complete: Optional[Callable[[ChildOrder], None]] = None,
        on_progress: Optional[Callable[[AlgorithmProgress], None]] = None,
    ):
        """
        Initialize VWAP algorithm.

        Args:
            request: Execution request
            executor: Order executor
            config: VWAP configuration
            price_provider: Price provider
            volume_provider: Volume data provider
            on_slice_complete: Slice completion callback
            on_progress: Progress callback
        """
        super().__init__(
            request=request,
            executor=executor,
            price_provider=price_provider,
            on_slice_complete=on_slice_complete,
            on_progress=on_progress,
        )

        self._config = config or VWAPConfig()
        self._volume_provider = volume_provider

        # Volume profile (hour -> expected volume share)
        self._volume_profile: Dict[int, Decimal] = {}
        self._slice_index = 0

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        return ExecutionAlgorithm.VWAP

    async def execute(self) -> AlgorithmProgress:
        """Execute VWAP algorithm."""
        self._progress.state = AlgorithmState.RUNNING
        self._progress.start_time = datetime.now(timezone.utc)

        logger.info(
            f"Starting VWAP: {self._request.quantity} {self._request.symbol} "
            f"over {self._config.duration_minutes}m with {self._config.participation_rate*100}% participation"
        )

        try:
            # Load volume profile
            await self._load_volume_profile()

            # Calculate execution schedule
            schedule = self._generate_schedule()

            end_time = datetime.now(timezone.utc) + timedelta(minutes=self._config.duration_minutes)

            # Execute based on schedule
            while datetime.now(timezone.utc) < end_time:
                if self._cancelled:
                    break

                # Wait while paused
                while self._paused and not self._cancelled:
                    await asyncio.sleep(0.5)

                # Check if we're done
                if self._progress.remaining_quantity <= 0:
                    break

                # Check price limits
                if not await self._check_price_limit():
                    await asyncio.sleep(self._config.min_slice_interval_seconds)
                    continue

                # Calculate slice size based on volume
                slice_qty = await self._calculate_slice_quantity()

                if slice_qty <= 0:
                    await asyncio.sleep(self._config.min_slice_interval_seconds)
                    continue

                # Create and execute slice
                child = self._create_child(self._slice_index, slice_qty)
                self._children.append(child)

                logger.info(f"Executing VWAP slice {self._slice_index+1}: {slice_qty}")
                success = await self._execute_slice(child)

                if success and self._on_slice_complete:
                    self._on_slice_complete(child)

                self._update_progress()
                self._slice_index += 1

                # Wait before next slice
                wait_time = self._calculate_wait_time()
                await asyncio.sleep(wait_time)

            # Finalize
            self._progress.end_time = datetime.now(timezone.utc)
            self._progress.total_slices = len(self._children)

            if self._cancelled:
                self._progress.state = AlgorithmState.CANCELLED
            else:
                self._progress.state = AlgorithmState.COMPLETED

            logger.info(
                f"VWAP complete: filled {self._progress.filled_quantity}/{self._request.quantity} "
                f"({self._progress.fill_pct:.1f}%)"
            )

        except Exception as e:
            self._progress.state = AlgorithmState.ERROR
            self._progress.end_time = datetime.now(timezone.utc)
            logger.error(f"VWAP execution error: {e}")

        return self._progress

    async def _load_volume_profile(self) -> None:
        """Load historical volume profile."""
        if self._volume_provider:
            try:
                self._volume_profile = await self._volume_provider.get_historical_volume_profile(
                    self._request.symbol,
                    self._config.volume_lookback_days,
                )
                logger.debug(f"Loaded volume profile with {len(self._volume_profile)} hours")
            except Exception as e:
                logger.warning(f"Failed to load volume profile: {e}")

        # Use flat profile if not available
        if not self._volume_profile:
            for hour in range(24):
                self._volume_profile[hour] = Decimal("1") / Decimal("24")

    def _generate_schedule(self) -> List[Tuple[datetime, Decimal]]:
        """
        Generate execution schedule based on volume profile.

        Returns:
            List of (time, expected_volume_share) tuples
        """
        schedule = []
        now = datetime.now(timezone.utc)
        end_time = now + timedelta(minutes=self._config.duration_minutes)

        current = now
        while current < end_time:
            hour = current.hour
            volume_share = self._volume_profile.get(hour, Decimal("1") / Decimal("24"))
            schedule.append((current, volume_share))
            current += timedelta(minutes=5)  # 5-minute intervals

        return schedule

    async def _calculate_slice_quantity(self) -> Decimal:
        """Calculate quantity for next slice based on volume."""
        remaining = self._progress.remaining_quantity

        if remaining <= 0:
            return Decimal("0")

        # Get current volume if available
        if self._volume_provider:
            try:
                current_volume = await self._volume_provider.get_current_volume(self._request.symbol)

                # Calculate participation
                participation = self._config.participation_rate

                if self._config.adaptive_participation:
                    # Adjust based on how much we've filled vs expected
                    elapsed_pct = self._progress.elapsed_seconds / (self._config.duration_minutes * 60)
                    expected_fill_pct = elapsed_pct * Decimal("100")
                    actual_fill_pct = self._progress.fill_pct

                    if actual_fill_pct < expected_fill_pct * Decimal("0.8"):
                        # Behind schedule, increase participation
                        participation = min(
                            participation * Decimal("1.5"),
                            self._config.max_participation_rate
                        )

                # Calculate slice quantity
                slice_qty = current_volume * participation

                # Cap at remaining
                slice_qty = min(slice_qty, remaining)

                return slice_qty

            except Exception as e:
                logger.warning(f"Failed to get volume: {e}")

        # Fallback: simple time-based calculation
        elapsed_pct = self._progress.elapsed_seconds / (self._config.duration_minutes * 60)
        expected_remaining_pct = Decimal("1") - Decimal(str(elapsed_pct))

        if expected_remaining_pct <= 0:
            return remaining

        # Estimate slices remaining
        time_remaining = self._config.duration_minutes * 60 - self._progress.elapsed_seconds
        slices_remaining = max(1, int(time_remaining / self._config.min_slice_interval_seconds))

        return remaining / Decimal(str(slices_remaining))

    def _calculate_wait_time(self) -> float:
        """Calculate wait time until next slice."""
        # Use current hour's volume profile
        hour = datetime.now(timezone.utc).hour
        volume_share = self._volume_profile.get(hour, Decimal("1") / Decimal("24"))

        # Higher volume = more frequent slices
        base_wait = (
            self._config.min_slice_interval_seconds +
            self._config.max_slice_interval_seconds
        ) / 2

        # Scale inversely with volume share
        avg_share = Decimal("1") / Decimal("24")
        scale = float(avg_share / max(volume_share, Decimal("0.01")))
        wait = base_wait * scale

        # Clamp to bounds
        return max(
            self._config.min_slice_interval_seconds,
            min(self._config.max_slice_interval_seconds, wait)
        )


# =============================================================================
# Iceberg Algorithm
# =============================================================================


@dataclass
class IcebergConfig:
    """Configuration for Iceberg algorithm."""

    visible_quantity_pct: Decimal = field(default_factory=lambda: Decimal("0.10"))  # 10% visible
    min_visible_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    replenish_on_fill: bool = True        # Replenish immediately after fill
    random_visible_size: bool = True      # Randomize visible size
    size_variation_pct: Decimal = field(default_factory=lambda: Decimal("0.20"))  # 20% variation
    price_improvement: bool = True        # Try to improve price between slices
    fill_timeout_seconds: int = 60        # How long to wait for fill


class IcebergAlgorithm(BaseExecutionAlgorithm):
    """
    Iceberg Order Algorithm.

    Shows only a portion of the total order quantity at a time,
    replenishing as fills occur. Used to hide large order sizes.

    Features:
    - Configurable visible quantity
    - Randomized visible size
    - Immediate replenishment on fill
    - Price improvement between slices

    Example:
        >>> config = IcebergConfig(visible_quantity_pct=Decimal("0.15"))
        >>> algo = IcebergAlgorithm(request, executor, config=config)
        >>> progress = await algo.execute()
    """

    def __init__(
        self,
        request: ExecutionRequest,
        executor: OrderExecutor,
        config: Optional[IcebergConfig] = None,
        price_provider: Optional[PriceProvider] = None,
        on_slice_complete: Optional[Callable[[ChildOrder], None]] = None,
        on_progress: Optional[Callable[[AlgorithmProgress], None]] = None,
    ):
        """
        Initialize Iceberg algorithm.

        Args:
            request: Execution request
            executor: Order executor
            config: Iceberg configuration
            price_provider: Price provider
            on_slice_complete: Slice completion callback
            on_progress: Progress callback
        """
        super().__init__(
            request=request,
            executor=executor,
            price_provider=price_provider,
            on_slice_complete=on_slice_complete,
            on_progress=on_progress,
        )

        self._config = config or IcebergConfig(
            visible_quantity_pct=request.iceberg_visible_pct,
        )

        self._slice_index = 0

    def get_algorithm_type(self) -> ExecutionAlgorithm:
        return ExecutionAlgorithm.ICEBERG

    async def execute(self) -> AlgorithmProgress:
        """Execute Iceberg algorithm."""
        self._progress.state = AlgorithmState.RUNNING
        self._progress.start_time = datetime.now(timezone.utc)

        logger.info(
            f"Starting Iceberg: {self._request.quantity} {self._request.symbol} "
            f"with {self._config.visible_quantity_pct*100}% visible"
        )

        try:
            while self._progress.remaining_quantity > 0:
                if self._cancelled:
                    break

                # Wait while paused
                while self._paused and not self._cancelled:
                    await asyncio.sleep(0.5)

                # Check price limits
                if not await self._check_price_limit():
                    await asyncio.sleep(5)
                    continue

                # Calculate visible quantity
                visible_qty = self._calculate_visible_quantity()

                if visible_qty <= 0:
                    break

                # Get price (with improvement if enabled)
                price = self._request.price
                if self._config.price_improvement and self._price_provider:
                    price = await self._get_improved_price()

                # Create and execute slice
                child = self._create_child(self._slice_index, visible_qty)
                if price:
                    child.price = price

                self._children.append(child)

                logger.info(f"Executing Iceberg slice {self._slice_index+1}: {visible_qty}")
                success = await self._execute_slice(
                    child,
                    wait_for_fill=True,
                    fill_timeout=self._config.fill_timeout_seconds,
                )

                if success and self._on_slice_complete:
                    self._on_slice_complete(child)

                self._update_progress()
                self._slice_index += 1

                # Small delay between slices to avoid pattern detection
                await asyncio.sleep(0.5)

            # Finalize
            self._progress.end_time = datetime.now(timezone.utc)
            self._progress.total_slices = len(self._children)

            if self._cancelled:
                self._progress.state = AlgorithmState.CANCELLED
            else:
                self._progress.state = AlgorithmState.COMPLETED

            logger.info(
                f"Iceberg complete: filled {self._progress.filled_quantity}/{self._request.quantity} "
                f"({self._progress.fill_pct:.1f}%)"
            )

        except Exception as e:
            self._progress.state = AlgorithmState.ERROR
            self._progress.end_time = datetime.now(timezone.utc)
            logger.error(f"Iceberg execution error: {e}")

        return self._progress

    def _calculate_visible_quantity(self) -> Decimal:
        """Calculate visible quantity for next slice."""
        import random

        remaining = self._progress.remaining_quantity

        if remaining <= 0:
            return Decimal("0")

        # Base visible quantity
        base_qty = self._request.quantity * self._config.visible_quantity_pct

        # Apply minimum
        if self._config.min_visible_quantity > 0:
            base_qty = max(base_qty, self._config.min_visible_quantity)

        # Randomize if enabled
        if self._config.random_visible_size:
            variation = float(self._config.size_variation_pct)
            factor = Decimal(str(1 + random.uniform(-variation, variation)))
            base_qty = base_qty * factor

        # Don't exceed remaining
        return min(base_qty, remaining)

    async def _get_improved_price(self) -> Optional[Decimal]:
        """Try to get a better price than the base."""
        if not self._price_provider or not self._request.price:
            return self._request.price

        try:
            current_price = await self._price_provider.get_price(self._request.symbol)
            base_price = self._request.price

            # For buys, try to get a lower price
            if self._request.side == OrderSide.BUY:
                if current_price < base_price:
                    return current_price
            # For sells, try to get a higher price
            else:
                if current_price > base_price:
                    return current_price

            return base_price

        except Exception:
            return self._request.price


# =============================================================================
# Factory Function
# =============================================================================


def create_algorithm(
    algorithm_type: ExecutionAlgorithm,
    request: ExecutionRequest,
    executor: OrderExecutor,
    price_provider: Optional[PriceProvider] = None,
    volume_provider: Optional[VolumeProvider] = None,
    **kwargs: Any,
) -> BaseExecutionAlgorithm:
    """
    Factory function to create execution algorithm.

    Args:
        algorithm_type: Type of algorithm
        request: Execution request
        executor: Order executor
        price_provider: Price provider
        volume_provider: Volume provider (for VWAP)
        **kwargs: Additional arguments passed to algorithm

    Returns:
        Configured algorithm instance
    """
    if algorithm_type == ExecutionAlgorithm.TWAP:
        config = TWAPConfig(
            duration_minutes=request.twap_duration_minutes,
            num_slices=request.twap_intervals,
        )
        return TWAPAlgorithm(
            request=request,
            executor=executor,
            config=config,
            price_provider=price_provider,
            **kwargs,
        )

    elif algorithm_type == ExecutionAlgorithm.VWAP:
        config = VWAPConfig()
        return VWAPAlgorithm(
            request=request,
            executor=executor,
            config=config,
            price_provider=price_provider,
            volume_provider=volume_provider,
            **kwargs,
        )

    elif algorithm_type == ExecutionAlgorithm.ICEBERG:
        config = IcebergConfig(
            visible_quantity_pct=request.iceberg_visible_pct,
        )
        return IcebergAlgorithm(
            request=request,
            executor=executor,
            config=config,
            price_provider=price_provider,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported algorithm type: {algorithm_type}")
