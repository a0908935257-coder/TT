"""
Fund Manager.

Main class for centralized fund allocation and distribution across trading bots.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from src.core import get_logger, get_timeout_config, with_timeout
from src.core.models import MarketType
from src.core.timeout import TimeoutError as OperationTimeout

from .core.allocator import BaseAllocator, create_allocator
from .core.atomic_allocator import AtomicAllocationManager
from .core.dispatcher import Dispatcher
from .core.fund_pool import FundPool
from .models.config import FundManagerConfig
from .models.records import AllocationRecord, AllocationTransaction, DispatchResult

if TYPE_CHECKING:
    from src.master.registry import BotRegistry
    from src.notification.manager import NotificationManager

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_account(self, market: MarketType = MarketType.SPOT) -> Any:
        """Get account information."""
        ...


class BotRegistryProtocol(Protocol):
    """Protocol for bot registry interface."""

    def get_all(self) -> List[Any]:
        """Get all registered bots."""
        ...

    def get(self, bot_id: str) -> Optional[Any]:
        """Get a specific bot."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager interface."""

    async def notify_fund_allocated(
        self,
        bot_id: str,
        amount: Decimal,
        total_allocated: Decimal,
    ) -> None:
        """Notify about fund allocation."""
        ...

    async def notify_deposit_detected(
        self,
        amount: Decimal,
        total_balance: Decimal,
    ) -> None:
        """Notify about deposit detection."""
        ...

    async def send_info(self, title: str, message: str) -> bool:
        """Send info notification."""
        ...


class FundManager:
    """
    Fund Manager (Singleton).

    Provides centralized fund allocation and distribution across trading bots.
    Monitors total capital pool and automatically allocates funds based on
    configured strategies.

    Example:
        >>> config = FundManagerConfig.from_yaml(yaml_config)
        >>> manager = FundManager(exchange, registry, notifier, config)
        >>> await manager.start()
        >>> result = await manager.dispatch_funds()
        >>> await manager.stop()
    """

    _instance: Optional["FundManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "FundManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        exchange: Optional[ExchangeProtocol] = None,
        registry: Optional[BotRegistryProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        config: Optional[FundManagerConfig] = None,
        market_type: MarketType = MarketType.FUTURES,
    ):
        """
        Initialize FundManager.

        Args:
            exchange: Exchange client for fetching account data
            registry: Bot registry for getting bot information
            notifier: Notification manager for alerts
            config: Fund manager configuration
            market_type: Market type to monitor (SPOT or FUTURES)
        """
        # Only initialize once
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._exchange = exchange
        self._registry = registry
        self._notifier = notifier
        self._config = config or FundManagerConfig()
        self._market_type = market_type

        # Core components
        self._fund_pool = FundPool(exchange, self._config, market_type)
        self._allocator: BaseAllocator = create_allocator(self._config)
        self._dispatcher = Dispatcher(registry, notifier)

        # Atomic allocation manager for transactional dispatch
        self._atomic_allocator = AtomicAllocationManager(
            fund_pool=self._fund_pool,
            dispatcher=self._dispatcher,
            notifier=notifier,
            rollback_threshold=0.5,  # Rollback if > 50% fail
        )

        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._allocation_records: List[AllocationRecord] = []
        self._max_records: int = 1000

        # Callbacks for dispatch notifications
        self._dispatch_callbacks: List[Any] = []

        self._initialized = True
        logger.info("FundManager initialized")

    @classmethod
    def get_instance(cls) -> Optional["FundManager"]:
        """Get singleton instance if exists."""
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance._initialized = False
        cls._instance = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Check if fund manager is running."""
        return self._running

    @property
    def config(self) -> FundManagerConfig:
        """Get configuration."""
        return self._config

    @property
    def fund_pool(self) -> FundPool:
        """Get fund pool."""
        return self._fund_pool

    @property
    def dispatcher(self) -> Dispatcher:
        """Get dispatcher."""
        return self._dispatcher

    @property
    def atomic_allocator(self) -> AtomicAllocationManager:
        """Get atomic allocation manager."""
        return self._atomic_allocator

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """
        Start the fund manager.

        Begins monitoring balance and auto-dispatching funds if enabled.
        """
        if self._running:
            logger.warning("FundManager is already running")
            return

        if not self._config.enabled:
            logger.info("FundManager is disabled in config")
            return

        logger.info("Starting FundManager...")

        try:
            # Initial balance update
            await self._fund_pool.update()

            # Start monitoring loop
            self._running = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())

            logger.info("FundManager started successfully")

        except Exception as e:
            logger.error(f"Failed to start FundManager: {e}")
            self._running = False
            raise

    async def stop(self) -> None:
        """
        Stop the fund manager.

        Stops monitoring and saves state.
        """
        if not self._running:
            logger.warning("FundManager is not running")
            return

        logger.info("Stopping FundManager...")

        try:
            self._running = False

            # Cancel monitor task
            if self._monitor_task:
                self._monitor_task.cancel()
                try:
                    await self._monitor_task
                except asyncio.CancelledError:
                    pass
                self._monitor_task = None

            logger.info("FundManager stopped")

        except Exception as e:
            logger.error(f"Error stopping FundManager: {e}")

    async def _monitor_loop(self) -> None:
        """
        Background monitoring loop.

        Periodically checks balance and dispatches funds on deposit.
        """
        while self._running:
            try:
                await asyncio.sleep(self._config.poll_interval)

                if not self._running:
                    break

                # Update balance
                await self._fund_pool.update()

                # Check for deposit
                if self._fund_pool.detect_deposit():
                    deposit_amount = self._fund_pool.get_deposit_amount()
                    logger.info(f"Deposit detected: {deposit_amount} USDT")

                    # Notify deposit
                    await self._notify_deposit(deposit_amount)

                    # Auto-dispatch if enabled
                    if self._config.auto_dispatch:
                        await self.dispatch_funds(trigger="deposit")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def dispatch_funds(self, trigger: str = "manual") -> DispatchResult:
        """
        Execute fund allocation and dispatch to bots.

        Thread-safe: Uses allocation lock to prevent race conditions
        when multiple triggers occur simultaneously. Protected by timeout
        to prevent infinite hangs.

        Args:
            trigger: What triggered the dispatch (manual, deposit, rebalance)

        Returns:
            DispatchResult with allocation details
        """
        result = DispatchResult(trigger=trigger)
        timeout_config = get_timeout_config()

        # Use allocation lock to prevent concurrent dispatches
        # Wrap entire dispatch operation with timeout
        try:
            async with asyncio.timeout(timeout_config.fund_allocation):
                async with self._fund_pool.allocation_lock:
                    try:
                        # Get active bots
                        bot_ids = self._get_active_bot_ids()
                        if not bot_ids:
                            logger.warning("No active bots to dispatch funds to")
                            result.errors.append("No active bots found")
                            return result

                        # Get available funds (inside lock to ensure consistency)
                        available = self._fund_pool.get_unallocated()
                        if available <= 0:
                            logger.info("No unallocated funds available for dispatch")
                            return result

                        # Calculate allocations
                        current_allocations = self._fund_pool.allocations
                        allocations = self._allocator.calculate(
                            available_funds=available,
                            bot_allocations=self._config.allocations,
                            current_allocations=current_allocations,
                            bot_ids=bot_ids,
                        )

                        if not allocations:
                            logger.info("No allocations calculated")
                            return result

                        # Execute allocations (still inside lock)
                        for bot_id, amount in allocations.items():
                            record = await self._allocate_to_bot(bot_id, amount, trigger)
                            result.add_allocation(record)

                        # Update overall success status
                        result.success = result.failed_count == 0

                        # Log summary
                        logger.info(
                            f"Dispatch complete: {result.successful_count} successful, "
                            f"{result.failed_count} failed, "
                            f"total dispatched: {result.total_dispatched}"
                        )

                    except Exception as e:
                        logger.error(f"Error during dispatch: {e}")
                        result.success = False
                        result.errors.append(str(e))

        except asyncio.TimeoutError:
            logger.error(
                f"Fund dispatch timed out after {timeout_config.fund_allocation}s"
            )
            result.success = False
            result.errors.append(
                f"Dispatch operation timed out after {timeout_config.fund_allocation}s"
            )
            return result

        # Notify dispatch (outside lock to avoid blocking)
        await self._notify_dispatch(result)

        return result

    async def dispatch_funds_atomic(
        self,
        trigger: str = "manual",
        auto_rollback: bool = True,
    ) -> AllocationTransaction:
        """
        Execute fund allocation with atomic transaction semantics.

        Provides all-or-nothing allocation with automatic rollback
        on high failure rate.

        Args:
            trigger: What triggered the dispatch (manual, deposit, rebalance)
            auto_rollback: Whether to auto-rollback on high failure rate

        Returns:
            AllocationTransaction with complete transaction details
        """
        timeout_config = get_timeout_config()

        try:
            async with asyncio.timeout(timeout_config.fund_allocation):
                async with self._fund_pool.allocation_lock:
                    # Get active bots
                    bot_ids = self._get_active_bot_ids()
                    if not bot_ids:
                        logger.warning("No active bots to dispatch funds to")
                        tx = AllocationTransaction(trigger=trigger)
                        tx.mark_failed("No active bots found")
                        return tx

                    # Get available funds
                    available = self._fund_pool.get_unallocated()
                    if available <= 0:
                        logger.info("No unallocated funds available for dispatch")
                        tx = AllocationTransaction(trigger=trigger)
                        tx.mark_committed()  # Not a failure, just nothing to do
                        return tx

                    # Calculate allocations
                    current_allocations = self._fund_pool.allocations
                    allocations = self._allocator.calculate(
                        available_funds=available,
                        bot_allocations=self._config.allocations,
                        current_allocations=current_allocations,
                        bot_ids=bot_ids,
                    )

                    if not allocations:
                        logger.info("No allocations calculated")
                        tx = AllocationTransaction(trigger=trigger)
                        tx.mark_committed()
                        return tx

                    # Execute atomically with potential rollback
                    tx = await self._atomic_allocator.execute_atomic(
                        planned_allocations=allocations,
                        trigger=trigger,
                        auto_rollback=auto_rollback,
                    )

                    # Store executed records
                    for record in tx.executed_allocations:
                        self._allocation_records.append(record)
                    if len(self._allocation_records) > self._max_records:
                        self._allocation_records = self._allocation_records[-self._max_records:]

                    # Log summary
                    logger.info(
                        f"Atomic dispatch {tx.status.value}: "
                        f"{tx.successful_count} successful, {tx.failed_count} failed, "
                        f"total allocated: {tx.total_allocated}"
                    )

                    return tx

        except asyncio.TimeoutError:
            logger.error(
                f"Atomic dispatch timed out after {timeout_config.fund_allocation}s"
            )
            tx = AllocationTransaction(trigger=trigger)
            tx.mark_failed(f"Timed out after {timeout_config.fund_allocation}s")
            return tx

    async def _allocate_to_bot(
        self,
        bot_id: str,
        amount: Decimal,
        trigger: str,
    ) -> AllocationRecord:
        """
        Allocate funds to a specific bot.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
            trigger: Allocation trigger

        Returns:
            AllocationRecord with result
        """
        previous = self._fund_pool.get_allocation(bot_id)
        new_total = previous + amount

        record = AllocationRecord(
            bot_id=bot_id,
            amount=amount,
            trigger=trigger,
            previous_allocation=previous,
            new_allocation=new_total,
        )

        try:
            # Notify bot via dispatcher (sends update_capital to bot)
            dispatch_record = await self._dispatcher.notify_bot(
                bot_id=bot_id,
                amount=new_total,  # Send total allocation, not just increment
                trigger=trigger,
            )

            if dispatch_record.success:
                # Update fund pool allocation tracking
                self._fund_pool.add_allocation(bot_id, amount)
                record.success = True
                logger.info(
                    f"Allocated {amount} to {bot_id} "
                    f"(previous: {previous}, new: {new_total})"
                )
            else:
                record.success = False
                record.error_message = dispatch_record.error_message
                logger.warning(
                    f"Bot {bot_id} rejected allocation: {record.error_message}"
                )

            # Store record
            self._allocation_records.append(record)
            if len(self._allocation_records) > self._max_records:
                self._allocation_records = self._allocation_records[-self._max_records:]

        except Exception as e:
            logger.error(f"Failed to allocate to {bot_id}: {e}")
            record.success = False
            record.error_message = str(e)

        return record

    def _get_active_bot_ids(self) -> List[str]:
        """
        Get list of active bot IDs.

        Returns:
            List of bot IDs that are running
        """
        if not self._registry:
            return []

        bot_ids = []
        for bot_info in self._registry.get_all():
            # Check if bot matches any allocation pattern and is in running state
            alloc_config = self._config.get_allocation_for_bot(bot_info.bot_id)
            if alloc_config and bot_info.state.value == "running":
                bot_ids.append(bot_info.bot_id)

        return bot_ids

    # =========================================================================
    # Manual Operations
    # =========================================================================

    def set_allocation(
        self,
        bot_id: str,
        amount: Decimal,
    ) -> None:
        """
        Manually set allocation for a bot.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
        """
        self._fund_pool.set_allocation(bot_id, amount)
        logger.info(f"Manual allocation set for {bot_id}: {amount}")

    def adjust_ratio(
        self,
        bot_pattern: str,
        new_ratio: Decimal,
    ) -> bool:
        """
        Adjust allocation ratio for a bot pattern.

        Args:
            bot_pattern: Bot pattern to adjust
            new_ratio: New ratio value

        Returns:
            True if successful
        """
        for alloc in self._config.allocations:
            if alloc.bot_pattern == bot_pattern:
                alloc.ratio = new_ratio
                logger.info(f"Adjusted ratio for {bot_pattern}: {new_ratio}")
                return True

        logger.warning(f"No allocation found for pattern: {bot_pattern}")
        return False

    async def recall_funds(
        self,
        bot_id: str,
        amount: Optional[Decimal] = None,
    ) -> AllocationRecord:
        """
        Recall funds from a bot back to the pool.

        Args:
            bot_id: Bot identifier
            amount: Amount to recall (None = recall all allocated funds)

        Returns:
            AllocationRecord with recall result
        """
        current_allocation = self._fund_pool.get_allocation(bot_id)

        if current_allocation <= 0:
            record = AllocationRecord(
                bot_id=bot_id,
                amount=Decimal("0"),
                trigger="recall",
                previous_allocation=Decimal("0"),
                new_allocation=Decimal("0"),
                success=False,
                error_message="No funds allocated to this bot",
            )
            logger.warning(f"No funds to recall from {bot_id}")
            return record

        # Determine recall amount
        if amount is None:
            recall_amount = current_allocation
        else:
            recall_amount = min(amount, current_allocation)

        new_allocation = current_allocation - recall_amount

        record = AllocationRecord(
            bot_id=bot_id,
            amount=-recall_amount,  # Negative to indicate recall
            trigger="recall",
            previous_allocation=current_allocation,
            new_allocation=new_allocation,
        )

        try:
            # Update fund pool allocation tracking
            self._fund_pool.set_allocation(bot_id, new_allocation)

            # Notify bot about reduced allocation
            dispatch_record = await self._dispatcher.notify_bot(
                bot_id=bot_id,
                amount=new_allocation,
                trigger="recall",
                previous_allocation=current_allocation,
            )

            if dispatch_record.success:
                record.success = True
                logger.info(
                    f"Recalled {recall_amount} from {bot_id} "
                    f"(previous: {current_allocation}, new: {new_allocation})"
                )
            else:
                # Even if notification fails, allocation is updated
                record.success = True
                record.error_message = (
                    f"Funds recalled but notification failed: "
                    f"{dispatch_record.error_message}"
                )
                logger.warning(
                    f"Recalled {recall_amount} from {bot_id} but notification failed"
                )

            # Store record
            self._allocation_records.append(record)
            if len(self._allocation_records) > self._max_records:
                self._allocation_records = self._allocation_records[-self._max_records:]

            # Send notification
            await self._notify_recall(bot_id, recall_amount, new_allocation)

        except Exception as e:
            logger.error(f"Failed to recall funds from {bot_id}: {e}")
            record.success = False
            record.error_message = str(e)

        return record

    async def recall_all_funds(self) -> List[AllocationRecord]:
        """
        Recall all funds from all bots back to the pool.

        Returns:
            List of AllocationRecord for each recall attempt
        """
        records: List[AllocationRecord] = []
        allocations = self._fund_pool.allocations

        for bot_id in list(allocations.keys()):
            record = await self.recall_funds(bot_id)
            records.append(record)

        # Log summary
        successful = sum(1 for r in records if r.success)
        total_recalled = sum(
            abs(r.amount) for r in records if r.success
        )
        logger.info(
            f"Recalled funds from {successful}/{len(records)} bots, "
            f"total: {total_recalled} USDT"
        )

        return records

    async def _notify_recall(
        self,
        bot_id: str,
        amount: Decimal,
        new_allocation: Decimal,
    ) -> None:
        """
        Send recall notification.

        Args:
            bot_id: Bot identifier
            amount: Amount recalled
            new_allocation: New allocation after recall
        """
        if self._notifier:
            try:
                await self._notifier.send_info(
                    "Fund Recall",
                    f"Recalled {amount} USDT from {bot_id}. "
                    f"New allocation: {new_allocation} USDT",
                )
            except Exception as e:
                logger.warning(f"Failed to send recall notification: {e}")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get fund manager status.

        Returns:
            Dictionary with current status
        """
        return {
            "running": self._running,
            "enabled": self._config.enabled,
            "strategy": self._config.strategy.value,
            "auto_dispatch": self._config.auto_dispatch,
            "poll_interval": self._config.poll_interval,
            "fund_pool": self._fund_pool.get_status(),
            "allocation_count": len(self._allocation_records),
        }

    def get_allocation_history(
        self,
        bot_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AllocationRecord]:
        """
        Get allocation history.

        Args:
            bot_id: Filter by bot ID (optional)
            limit: Maximum records to return

        Returns:
            List of AllocationRecord
        """
        records = self._allocation_records
        if bot_id:
            records = [r for r in records if r.bot_id == bot_id]
        return records[-limit:]

    # =========================================================================
    # Notification Methods
    # =========================================================================

    async def _notify_deposit(self, amount: Decimal) -> None:
        """
        Send deposit notification.

        Args:
            amount: Deposit amount
        """
        if self._notifier:
            try:
                if hasattr(self._notifier, "notify_deposit_detected"):
                    await self._notifier.notify_deposit_detected(
                        amount=amount,
                        total_balance=self._fund_pool.total_balance,
                    )
                else:
                    await self._notifier.send_info(
                        "Deposit Detected",
                        f"Deposit of {amount} USDT detected. "
                        f"Total balance: {self._fund_pool.total_balance} USDT",
                    )
            except Exception as e:
                logger.warning(f"Failed to send deposit notification: {e}")

    async def _notify_dispatch(self, result: DispatchResult) -> None:
        """
        Send dispatch notification.

        Args:
            result: Dispatch result
        """
        if self._notifier:
            try:
                message = (
                    f"Dispatched {result.total_dispatched} USDT to "
                    f"{result.successful_count} bots"
                )
                if result.failed_count > 0:
                    message += f" ({result.failed_count} failed)"

                await self._notifier.send_info("Fund Dispatch", message)
            except Exception as e:
                logger.warning(f"Failed to send dispatch notification: {e}")

    # =========================================================================
    # Callback Registration
    # =========================================================================

    def on_dispatch(self, callback: Any) -> None:
        """
        Register callback for dispatch events.

        Args:
            callback: Callback function
        """
        self._dispatch_callbacks.append(callback)

    def remove_dispatch_callback(self, callback: Any) -> None:
        """
        Remove dispatch callback.

        Args:
            callback: Callback to remove
        """
        if callback in self._dispatch_callbacks:
            self._dispatch_callbacks.remove(callback)
