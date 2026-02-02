"""
Order Router.

Intelligent order routing system that:
- Selects the best execution algorithm based on order characteristics
- Splits large orders into smaller child orders
- Routes orders to appropriate exchanges
- Manages execution lifecycle

Part of the Execution Layer.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from src.core import get_logger
from src.core.models import MarketType
from src.execution.models import (
    ChildOrder,
    DepthLevel,
    ExecutionAlgorithm,
    ExecutionPlan,
    ExecutionRequest,
    ExecutionResult,
    ExecutionUrgency,
    MarketDepthAnalysis,
    OnExecutionComplete,
    OnOrderCancelled,
    OnOrderError,
    OnOrderFilled,
    OnOrderSent,
    OrderSide,
    OrderStatus,
    OrderType,
    RouterConfig,
    SplitStrategy,
)

logger = get_logger(__name__)


# =============================================================================
# Protocols for Dependency Injection
# =============================================================================


class ExchangeExecutor(Protocol):
    """Protocol for exchange order execution."""

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
        """Place an order on the exchange."""
        ...

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> bool:
        """Cancel an order."""
        ...

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Dict[str, Any]]:
        """Get order status."""
        ...


class MarketDataProvider(Protocol):
    """Protocol for market data access."""

    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """Get order book with bids and asks."""
        ...

    async def get_ticker(
        self,
        symbol: str,
    ) -> Dict[str, Any]:
        """Get ticker data."""
        ...

    async def get_24h_volume(
        self,
        symbol: str,
    ) -> Decimal:
        """Get 24h trading volume."""
        ...


class SymbolInfoProvider(Protocol):
    """Protocol for symbol information."""

    def get_min_quantity(self, symbol: str) -> Decimal:
        """Get minimum order quantity."""
        ...

    def get_min_notional(self, symbol: str) -> Decimal:
        """Get minimum order notional value."""
        ...

    def round_quantity(self, symbol: str, quantity: Decimal) -> Decimal:
        """Round quantity to valid precision."""
        ...

    def round_price(self, symbol: str, price: Decimal) -> Decimal:
        """Round price to valid precision."""
        ...


# =============================================================================
# Order Router
# =============================================================================


class OrderRouter:
    """
    Intelligent order routing system.

    Responsible for:
    1. Analyzing order characteristics and market conditions
    2. Selecting the optimal execution algorithm
    3. Splitting large orders into child orders
    4. Managing the execution lifecycle
    5. Tracking execution quality metrics

    Example:
        >>> config = RouterConfig()
        >>> router = OrderRouter(
        ...     config=config,
        ...     executor=exchange_executor,
        ...     market_data=market_data_provider,
        ...     symbol_info=symbol_info_provider,
        ... )
        >>>
        >>> request = ExecutionRequest(
        ...     symbol="BTCUSDT",
        ...     side=OrderSide.BUY,
        ...     quantity=Decimal("1.5"),
        ...     price=Decimal("50000"),
        ... )
        >>>
        >>> result = await router.execute(request)
    """

    def __init__(
        self,
        config: RouterConfig,
        executor: ExchangeExecutor,
        market_data: Optional[MarketDataProvider] = None,
        symbol_info: Optional[SymbolInfoProvider] = None,
        on_order_sent: Optional[OnOrderSent] = None,
        on_order_filled: Optional[OnOrderFilled] = None,
        on_order_cancelled: Optional[OnOrderCancelled] = None,
        on_order_error: Optional[OnOrderError] = None,
        on_execution_complete: Optional[OnExecutionComplete] = None,
    ):
        """
        Initialize OrderRouter.

        Args:
            config: Router configuration
            executor: Exchange execution interface
            market_data: Market data provider (optional)
            symbol_info: Symbol info provider (optional)
            on_order_sent: Callback when child order is sent
            on_order_filled: Callback when child order fills
            on_order_cancelled: Callback when child order cancelled
            on_order_error: Callback on order error
            on_execution_complete: Callback when execution completes
        """
        self._config = config
        self._executor = executor
        self._market_data = market_data
        self._symbol_info = symbol_info

        # Callbacks
        self._on_order_sent = on_order_sent
        self._on_order_filled = on_order_filled
        self._on_order_cancelled = on_order_cancelled
        self._on_order_error = on_order_error
        self._on_execution_complete = on_execution_complete

        # Active executions
        self._active_executions: Dict[str, ExecutionPlan] = {}

        # Current execution context
        self._current_market_type = MarketType.SPOT

        # Statistics
        self._total_executions = 0
        self._successful_executions = 0
        self._total_volume = Decimal("0")
        self._total_fees = Decimal("0")

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> RouterConfig:
        """Get configuration."""
        return self._config

    @property
    def active_execution_count(self) -> int:
        """Get number of active executions."""
        return len(self._active_executions)

    # =========================================================================
    # Main Execution Method
    # =========================================================================

    async def execute(
        self,
        request: ExecutionRequest,
    ) -> ExecutionResult:
        """
        Execute an order request.

        This is the main entry point for order execution. It will:
        1. Analyze market conditions
        2. Select execution algorithm
        3. Create execution plan with child orders
        4. Execute child orders
        5. Return execution result

        Args:
            request: Execution request

        Returns:
            ExecutionResult with details of the execution
        """
        start_time = datetime.now(timezone.utc)
        execution_id = self._generate_execution_id()

        logger.info(
            f"Starting execution {execution_id}: "
            f"{request.side.value} {request.quantity} {request.symbol}"
        )

        try:
            # Step 1: Analyze market depth
            depth_analysis = await self._analyze_market_depth(request.symbol)

            # Step 2: Select algorithm
            algorithm = await self._select_algorithm(request, depth_analysis)

            # Step 3: Create execution plan
            plan = await self._create_execution_plan(
                execution_id, request, algorithm, depth_analysis
            )

            # Store active execution
            self._active_executions[execution_id] = plan

            # Step 4: Execute the plan
            await self._execute_plan(plan)

            # Step 5: Create result
            end_time = datetime.now(timezone.utc)
            duration = (end_time - start_time).total_seconds()

            result = self._create_result(plan, duration)

            # Update statistics
            self._total_executions += 1
            if result.success:
                self._successful_executions += 1
                self._total_volume += result.final_quantity
                self._total_fees += result.total_fee

            logger.info(
                f"Execution {execution_id} complete: "
                f"filled {result.final_quantity}/{request.quantity} "
                f"({result.filled_pct:.1f}%) in {duration:.2f}s"
            )

            # Callback
            if self._on_execution_complete:
                self._on_execution_complete(result)

            return result

        except Exception as e:
            logger.error(f"Execution {execution_id} failed: {e}")

            # Create failed result
            plan = ExecutionPlan(
                execution_id=execution_id,
                request=request,
                algorithm=ExecutionAlgorithm.DIRECT,
                split_strategy=SplitStrategy.NONE,
            )

            return ExecutionResult(
                execution_id=execution_id,
                success=False,
                plan=plan,
                error_message=str(e),
            )

        finally:
            # Remove from active executions
            self._active_executions.pop(execution_id, None)

    # =========================================================================
    # Market Analysis
    # =========================================================================

    async def _analyze_market_depth(
        self,
        symbol: str,
        levels: int = 20,
    ) -> Optional[MarketDepthAnalysis]:
        """
        Analyze market depth for execution planning.

        Args:
            symbol: Trading symbol
            levels: Number of levels to analyze

        Returns:
            MarketDepthAnalysis or None if unavailable
        """
        if not self._market_data:
            return None

        try:
            orderbook = await self._market_data.get_orderbook(symbol, levels)

            bids = orderbook.get("bids", [])
            asks = orderbook.get("asks", [])

            # Convert to DepthLevel objects with cumulative quantities
            bid_levels = []
            cumulative = Decimal("0")
            for price, qty in bids:
                price = Decimal(str(price))
                qty = Decimal(str(qty))
                cumulative += qty
                bid_levels.append(DepthLevel(
                    price=price,
                    quantity=qty,
                    cumulative_quantity=cumulative,
                ))

            ask_levels = []
            cumulative = Decimal("0")
            for price, qty in asks:
                price = Decimal(str(price))
                qty = Decimal(str(qty))
                cumulative += qty
                ask_levels.append(DepthLevel(
                    price=price,
                    quantity=qty,
                    cumulative_quantity=cumulative,
                ))

            # Calculate metrics
            best_bid = bid_levels[0].price if bid_levels else None
            best_ask = ask_levels[0].price if ask_levels else None

            spread = None
            spread_pct = None
            if best_bid and best_ask:
                spread = best_ask - best_bid
                mid_price = (best_bid + best_ask) / 2
                spread_pct = (spread / mid_price) * Decimal("100")

            total_bid_volume = sum(l.quantity for l in bid_levels)
            total_ask_volume = sum(l.quantity for l in ask_levels)

            # Imbalance: positive = more bids (bullish), negative = more asks
            total_volume = total_bid_volume + total_ask_volume
            if total_volume > 0:
                imbalance = (total_bid_volume - total_ask_volume) / total_volume
            else:
                imbalance = Decimal("0")

            return MarketDepthAnalysis(
                symbol=symbol,
                bid_levels=bid_levels,
                ask_levels=ask_levels,
                best_bid=best_bid,
                best_ask=best_ask,
                spread=spread,
                spread_pct=spread_pct,
                total_bid_volume=total_bid_volume,
                total_ask_volume=total_ask_volume,
                imbalance=imbalance,
            )

        except Exception as e:
            logger.warning(f"Failed to analyze market depth for {symbol}: {e}")
            return None

    # =========================================================================
    # Algorithm Selection
    # =========================================================================

    async def _select_algorithm(
        self,
        request: ExecutionRequest,
        depth: Optional[MarketDepthAnalysis],
    ) -> ExecutionAlgorithm:
        """
        Select the best execution algorithm.

        Args:
            request: Execution request
            depth: Market depth analysis

        Returns:
            Selected algorithm
        """
        # If algorithm is specified (not SMART), use it
        if request.algorithm != ExecutionAlgorithm.SMART:
            return request.algorithm

        # Market orders always use DIRECT
        if request.order_type == OrderType.MARKET:
            return ExecutionAlgorithm.DIRECT

        # Immediate urgency uses DIRECT
        if request.urgency == ExecutionUrgency.IMMEDIATE:
            return ExecutionAlgorithm.DIRECT

        # Get order size relative to market
        order_size_pct = await self._get_order_size_pct(request, depth)

        # Small orders: direct execution
        if order_size_pct < self._config.small_order_threshold_pct:
            logger.debug(f"Order size {order_size_pct:.2f}% < threshold, using DIRECT")
            return ExecutionAlgorithm.DIRECT

        # Large orders: use algorithm based on urgency
        if order_size_pct > self._config.large_order_threshold_pct:
            if request.urgency == ExecutionUrgency.HIGH:
                # High urgency large order: TWAP is faster
                logger.debug("Large order with high urgency, using TWAP")
                return ExecutionAlgorithm.TWAP
            else:
                # Low/medium urgency: ICEBERG to hide size
                logger.debug("Large order with low/medium urgency, using ICEBERG")
                return ExecutionAlgorithm.ICEBERG

        # Medium orders
        if request.urgency == ExecutionUrgency.LOW:
            # Low urgency: VWAP to match market
            return ExecutionAlgorithm.VWAP
        else:
            # Medium/high urgency: TWAP for predictable timing
            return ExecutionAlgorithm.TWAP

    async def _get_order_size_pct(
        self,
        request: ExecutionRequest,
        depth: Optional[MarketDepthAnalysis],
    ) -> Decimal:
        """
        Calculate order size as percentage of market liquidity.

        Args:
            request: Execution request
            depth: Market depth analysis

        Returns:
            Order size as percentage of available liquidity
        """
        # Use depth if available
        if depth:
            if request.side == OrderSide.BUY:
                available = depth.total_ask_volume
            else:
                available = depth.total_bid_volume

            if available > 0:
                return (request.quantity / available) * Decimal("100")

        # Fallback: try to get 24h volume
        if self._market_data:
            try:
                volume_24h = await self._market_data.get_24h_volume(request.symbol)
                if volume_24h > 0:
                    # Assume orderbook is ~1% of 24h volume
                    estimated_depth = volume_24h * Decimal("0.01")
                    return (request.quantity / estimated_depth) * Decimal("100")
            except Exception:
                pass

        # Default to medium (will use TWAP)
        return Decimal("2.5")

    # =========================================================================
    # Execution Plan Creation
    # =========================================================================

    async def _create_execution_plan(
        self,
        execution_id: str,
        request: ExecutionRequest,
        algorithm: ExecutionAlgorithm,
        depth: Optional[MarketDepthAnalysis],
    ) -> ExecutionPlan:
        """
        Create execution plan with child orders.

        Args:
            execution_id: Execution ID
            request: Original request
            algorithm: Selected algorithm
            depth: Market depth analysis

        Returns:
            ExecutionPlan with child orders
        """
        # Determine split strategy
        split_strategy, child_orders = await self._create_child_orders(
            execution_id, request, algorithm, depth
        )

        # Estimate duration
        estimated_duration = self._estimate_duration(algorithm, request, len(child_orders))

        return ExecutionPlan(
            execution_id=execution_id,
            request=request,
            algorithm=algorithm,
            split_strategy=split_strategy,
            child_orders=child_orders,
            estimated_duration_seconds=estimated_duration,
        )

    async def _create_child_orders(
        self,
        execution_id: str,
        request: ExecutionRequest,
        algorithm: ExecutionAlgorithm,
        depth: Optional[MarketDepthAnalysis],
    ) -> Tuple[SplitStrategy, List[ChildOrder]]:
        """
        Create child orders based on algorithm.

        Args:
            execution_id: Parent execution ID
            request: Original request
            algorithm: Selected algorithm
            depth: Market depth analysis

        Returns:
            Tuple of (split strategy, list of child orders)
        """
        if algorithm == ExecutionAlgorithm.DIRECT:
            # Single order
            child = self._create_single_child(execution_id, request, 0)
            return SplitStrategy.NONE, [child]

        elif algorithm == ExecutionAlgorithm.TWAP:
            return self._create_twap_children(execution_id, request)

        elif algorithm == ExecutionAlgorithm.ICEBERG:
            return self._create_iceberg_children(execution_id, request, depth)

        elif algorithm == ExecutionAlgorithm.VWAP:
            return self._create_vwap_children(execution_id, request)

        else:
            # Default to direct
            child = self._create_single_child(execution_id, request, 0)
            return SplitStrategy.NONE, [child]

    def _create_single_child(
        self,
        execution_id: str,
        request: ExecutionRequest,
        index: int,
        quantity: Optional[Decimal] = None,
        scheduled_at: Optional[datetime] = None,
    ) -> ChildOrder:
        """Create a single child order."""
        qty = quantity or request.quantity

        # Round quantity if symbol info available
        if self._symbol_info:
            qty = self._symbol_info.round_quantity(request.symbol, qty)

        # Round price if symbol info available
        price = request.price
        if price and self._symbol_info:
            price = self._symbol_info.round_price(request.symbol, price)

        return ChildOrder(
            parent_id=execution_id,
            child_index=index,
            symbol=request.symbol,
            side=request.side,
            quantity=qty,
            price=price,
            order_type=request.order_type,
            scheduled_at=scheduled_at,
            client_order_id=f"{execution_id}-{index}",
        )

    def _create_twap_children(
        self,
        execution_id: str,
        request: ExecutionRequest,
    ) -> Tuple[SplitStrategy, List[ChildOrder]]:
        """
        Create TWAP child orders.

        Splits order into equal-sized pieces at regular intervals.
        """
        intervals = request.twap_intervals or self._config.twap_default_intervals
        intervals = min(intervals, self._config.max_child_orders)

        duration_minutes = request.twap_duration_minutes
        interval_seconds = (duration_minutes * 60) / intervals

        quantity_per_child = request.quantity / Decimal(str(intervals))

        # Ensure minimum order value
        if request.price:
            min_qty = self._config.min_child_order_value / request.price
            if quantity_per_child < min_qty:
                # Reduce number of intervals
                intervals = int(request.quantity / min_qty)
                intervals = max(1, intervals)
                quantity_per_child = request.quantity / Decimal(str(intervals))

        children = []
        now = datetime.now(timezone.utc)

        for i in range(intervals):
            scheduled_at = now + timedelta(seconds=i * interval_seconds)

            # Last child gets remaining quantity to handle rounding
            if i == intervals - 1:
                filled_so_far = quantity_per_child * Decimal(str(i))
                qty = request.quantity - filled_so_far
            else:
                qty = quantity_per_child

            child = self._create_single_child(
                execution_id, request, i,
                quantity=qty,
                scheduled_at=scheduled_at,
            )
            children.append(child)

        return SplitStrategy.TIME_BASED, children

    def _create_iceberg_children(
        self,
        execution_id: str,
        request: ExecutionRequest,
        depth: Optional[MarketDepthAnalysis],
    ) -> Tuple[SplitStrategy, List[ChildOrder]]:
        """
        Create iceberg child orders.

        Only shows a small portion of the order at a time.
        """
        visible_pct = request.iceberg_visible_pct or self._config.iceberg_default_visible_pct
        visible_qty = request.quantity * visible_pct

        # Ensure minimum order value
        if request.price:
            min_qty = self._config.min_child_order_value / request.price
            visible_qty = max(visible_qty, min_qty)

        # Calculate number of children
        num_children = int(request.quantity / visible_qty)
        num_children = max(1, min(num_children, self._config.max_child_orders))

        children = []
        remaining = request.quantity

        for i in range(num_children):
            qty = min(visible_qty, remaining)
            if qty <= 0:
                break

            child = self._create_single_child(
                execution_id, request, i,
                quantity=qty,
            )
            children.append(child)
            remaining -= qty

        return SplitStrategy.FIXED_SIZE, children

    def _create_vwap_children(
        self,
        execution_id: str,
        request: ExecutionRequest,
    ) -> Tuple[SplitStrategy, List[ChildOrder]]:
        """
        Create VWAP child orders.

        Note: True VWAP requires historical volume data.
        This is a simplified version that uses TWAP as base.
        """
        # For now, use TWAP as fallback
        # TODO: Implement true VWAP with volume profile
        return self._create_twap_children(execution_id, request)

    def _estimate_duration(
        self,
        algorithm: ExecutionAlgorithm,
        request: ExecutionRequest,
        num_children: int,
    ) -> int:
        """Estimate execution duration in seconds."""
        if algorithm == ExecutionAlgorithm.DIRECT:
            return 5  # Quick execution

        elif algorithm == ExecutionAlgorithm.TWAP:
            return request.twap_duration_minutes * 60

        elif algorithm == ExecutionAlgorithm.ICEBERG:
            # Estimate based on fills (assume 30s per child)
            return num_children * 30

        elif algorithm == ExecutionAlgorithm.VWAP:
            return request.twap_duration_minutes * 60

        return 60  # Default 1 minute

    # =========================================================================
    # Execution
    # =========================================================================

    async def _execute_plan(self, plan: ExecutionPlan) -> None:
        """
        Execute the plan by sending child orders.

        Args:
            plan: Execution plan to execute
        """
        # Store market_type for child execution
        self._current_market_type = plan.request.market_type

        if plan.algorithm == ExecutionAlgorithm.DIRECT:
            # Execute single order immediately
            if not plan.child_orders:
                raise ValueError("No child orders to execute in DIRECT algorithm")
            await self._execute_child(plan.child_orders[0])

        elif plan.algorithm in (ExecutionAlgorithm.TWAP, ExecutionAlgorithm.VWAP):
            # Execute with timing
            await self._execute_timed_children(plan)

        elif plan.algorithm == ExecutionAlgorithm.ICEBERG:
            # Execute sequentially (wait for fill before next)
            await self._execute_sequential_children(plan)

        else:
            # Default: parallel execution
            await self._execute_parallel_children(plan)

    async def _execute_child(self, child: ChildOrder) -> None:
        """
        Execute a single child order.

        Args:
            child: Child order to execute
        """
        try:
            child.status = OrderStatus.SUBMITTED
            child.sent_at = datetime.now(timezone.utc)

            # Place order
            result = await self._executor.place_order(
                symbol=child.symbol,
                side=child.side.value,
                order_type=child.order_type.value,
                quantity=child.quantity,
                price=child.price,
                market_type=self._current_market_type,
                client_order_id=child.client_order_id,
            )

            # Update child with result
            if result.get("success", False) or result.get("orderId"):
                child.exchange_order_id = str(result.get("orderId", ""))
                child.status = OrderStatus.ACCEPTED

                # Check if already filled
                status = result.get("status", "NEW")
                if status == "FILLED":
                    child.status = OrderStatus.FILLED
                    child.filled_quantity = Decimal(str(result.get("executedQty", child.quantity)))
                    child.average_price = Decimal(str(result.get("avgPrice", result.get("price", child.price or 0))))
                    child.filled_at = datetime.now(timezone.utc)
                elif status == "PARTIALLY_FILLED":
                    child.status = OrderStatus.PARTIALLY_FILLED
                    child.filled_quantity = Decimal(str(result.get("executedQty", 0)))
                    child.average_price = Decimal(str(result.get("avgPrice", result.get("price", child.price or 0))))

                # Callback
                if self._on_order_sent:
                    self._on_order_sent(child)

                # Wait for fill if limit order (and not already terminal)
                if child.status not in (
                    OrderStatus.FILLED,
                    OrderStatus.CANCELLED,
                    OrderStatus.REJECTED,
                    OrderStatus.FAILED,
                ):
                    await self._wait_for_fill(child)

            else:
                child.status = OrderStatus.REJECTED
                child.error_message = result.get("msg", "Order rejected")

                if self._on_order_error:
                    self._on_order_error(child, child.error_message or "Unknown error")

        except Exception as e:
            child.status = OrderStatus.FAILED
            child.error_message = str(e)
            logger.error(f"Failed to execute child order: {e}")

            if self._on_order_error:
                self._on_order_error(child, str(e))

    async def _wait_for_fill(
        self,
        child: ChildOrder,
        timeout_seconds: int = 30,
        poll_interval: float = 1.0,
    ) -> None:
        """
        Wait for a child order to be filled.

        Args:
            child: Child order to monitor
            timeout_seconds: Maximum wait time
            poll_interval: Polling interval
        """
        start_time = datetime.now(timezone.utc)
        timeout = timedelta(seconds=timeout_seconds)

        while datetime.now(timezone.utc) - start_time < timeout:
            if child.is_complete:
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

                        if self._on_order_filled:
                            self._on_order_filled(child)
                        break

                    elif status == "PARTIALLY_FILLED":
                        child.status = OrderStatus.PARTIALLY_FILLED
                        child.filled_quantity = Decimal(str(order_info.get("executedQty", 0)))
                        # Also update average price for partial fills
                        avg_price = order_info.get("avgPrice", order_info.get("price"))
                        if avg_price:
                            child.average_price = Decimal(str(avg_price))

                    elif status == "CANCELED":
                        child.status = OrderStatus.CANCELLED
                        if self._on_order_cancelled:
                            self._on_order_cancelled(child)
                        break

                    elif status == "REJECTED":
                        child.status = OrderStatus.REJECTED
                        break

            except Exception as e:
                logger.warning(f"Error polling order status: {e}")

            await asyncio.sleep(poll_interval)

    async def _execute_timed_children(self, plan: ExecutionPlan) -> None:
        """
        Execute children at scheduled times (TWAP/VWAP).

        Args:
            plan: Execution plan
        """
        for child in plan.child_orders:
            if child.scheduled_at:
                # Wait until scheduled time
                now = datetime.now(timezone.utc)
                if child.scheduled_at > now:
                    wait_seconds = (child.scheduled_at - now).total_seconds()
                    await asyncio.sleep(wait_seconds)

            await self._execute_child(child)

    async def _execute_sequential_children(self, plan: ExecutionPlan) -> None:
        """
        Execute children sequentially (wait for fill).

        Args:
            plan: Execution plan
        """
        for child in plan.child_orders:
            await self._execute_child(child)

            # Wait for this child to complete before next (with timeout)
            wait_elapsed = 0
            max_wait = 300  # 5 minutes max per child order
            while not child.is_complete:
                await asyncio.sleep(0.5)
                wait_elapsed += 0.5
                if wait_elapsed >= max_wait:
                    logger.error(f"Child order timed out after {max_wait}s: {child}")
                    break

    async def _execute_parallel_children(
        self,
        plan: ExecutionPlan,
        max_parallel: Optional[int] = None,
    ) -> None:
        """
        Execute children in parallel with concurrency limit.

        Args:
            plan: Execution plan
            max_parallel: Maximum parallel executions
        """
        max_parallel = max_parallel or self._config.max_parallel_children
        semaphore = asyncio.Semaphore(max_parallel)

        async def execute_with_limit(child: ChildOrder):
            async with semaphore:
                await self._execute_child(child)

        tasks = [execute_with_limit(child) for child in plan.child_orders]
        await asyncio.gather(*tasks)

    # =========================================================================
    # Result Creation
    # =========================================================================

    def _create_result(
        self,
        plan: ExecutionPlan,
        duration_seconds: float,
    ) -> ExecutionResult:
        """
        Create execution result from completed plan.

        Args:
            plan: Completed execution plan
            duration_seconds: Execution duration

        Returns:
            ExecutionResult
        """
        # Calculate totals
        final_quantity = sum(c.filled_quantity for c in plan.child_orders)
        total_fee = sum(c.fee for c in plan.child_orders)

        # Calculate average price
        average_price = plan.average_fill_price

        # Calculate slippage
        slippage_pct = None
        if average_price and plan.request.price:
            if plan.request.side == OrderSide.BUY:
                slippage_pct = ((average_price - plan.request.price) / plan.request.price) * Decimal("100")
            else:
                slippage_pct = ((plan.request.price - average_price) / plan.request.price) * Decimal("100")

        # Determine success
        success = final_quantity > 0

        return ExecutionResult(
            execution_id=plan.execution_id,
            success=success,
            plan=plan,
            final_quantity=final_quantity,
            average_price=average_price,
            total_fee=total_fee,
            slippage_pct=slippage_pct,
            duration_seconds=duration_seconds,
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def _generate_execution_id(self) -> str:
        """Generate unique execution ID."""
        return f"exec_{uuid.uuid4().hex[:12]}"

    # =========================================================================
    # Management Methods
    # =========================================================================

    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an active execution.

        Args:
            execution_id: Execution to cancel

        Returns:
            True if cancellation was initiated
        """
        plan = self._active_executions.get(execution_id)
        if not plan:
            return False

        for child in plan.pending_children:
            if child.exchange_order_id:
                try:
                    await self._executor.cancel_order(
                        symbol=child.symbol,
                        order_id=child.exchange_order_id,
                    )
                    child.status = OrderStatus.CANCELLED

                    if self._on_order_cancelled:
                        self._on_order_cancelled(child)

                except Exception as e:
                    logger.warning(f"Failed to cancel child order: {e}")

        return True

    def get_execution_status(
        self,
        execution_id: str,
    ) -> Optional[ExecutionPlan]:
        """
        Get status of an execution.

        Args:
            execution_id: Execution ID

        Returns:
            ExecutionPlan if found, else None
        """
        return self._active_executions.get(execution_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get router statistics."""
        success_rate = Decimal("0")
        if self._total_executions > 0:
            success_rate = Decimal(str(self._successful_executions)) / Decimal(str(self._total_executions)) * Decimal("100")

        return {
            "total_executions": self._total_executions,
            "successful_executions": self._successful_executions,
            "success_rate_pct": float(success_rate),
            "total_volume": str(self._total_volume),
            "total_fees": str(self._total_fees),
            "active_executions": self.active_execution_count,
        }

    def reset_statistics(self) -> None:
        """Reset statistics."""
        self._total_executions = 0
        self._successful_executions = 0
        self._total_volume = Decimal("0")
        self._total_fees = Decimal("0")
