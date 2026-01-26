"""
Atomic Allocation Manager.

Provides transaction-like semantics for batch fund allocations
with automatic rollback on failure.
"""

import asyncio
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

from src.core import get_logger, get_timeout_config, with_timeout
from src.core.timeout import TimeoutError as OperationTimeout

from ..models.records import (
    AllocationRecord,
    AllocationTransaction,
    TransactionStatus,
)

if TYPE_CHECKING:
    from .dispatcher import Dispatcher
    from .fund_pool import FundPool

logger = get_logger(__name__)


class NotifierProtocol(Protocol):
    """Protocol for notification manager interface."""

    async def send_error(self, title: str, message: str) -> bool: ...
    async def send_warning(self, title: str, message: str) -> bool: ...
    async def send_info(self, title: str, message: str) -> bool: ...


class AtomicAllocationManager:
    """
    Manages atomic batch allocation operations.

    Provides transaction-like semantics:
    - Begin: Create snapshot of current state
    - Execute: Perform allocations
    - Commit: Finalize successful transaction
    - Rollback: Restore previous state on failure

    Example:
        >>> manager = AtomicAllocationManager(fund_pool, dispatcher, notifier)
        >>> tx = await manager.begin_transaction(allocations, "deposit")
        >>> tx = await manager.execute_transaction(tx)
        >>> if tx.failed_count > 0:
        ...     await manager.rollback_transaction(tx)
        ... else:
        ...     await manager.commit_transaction(tx)
    """

    def __init__(
        self,
        fund_pool: "FundPool",
        dispatcher: "Dispatcher",
        notifier: Optional[NotifierProtocol] = None,
        rollback_threshold: float = 0.5,
    ):
        """
        Initialize AtomicAllocationManager.

        Args:
            fund_pool: FundPool for managing allocations
            dispatcher: Dispatcher for notifying bots
            notifier: Optional notification manager
            rollback_threshold: Failure ratio threshold to trigger rollback (0.5 = 50%)
        """
        self._fund_pool = fund_pool
        self._dispatcher = dispatcher
        self._notifier = notifier
        self._rollback_threshold = rollback_threshold

        # Track active transactions
        self._active_transactions: Dict[str, AllocationTransaction] = {}
        self._transaction_history: List[AllocationTransaction] = []
        self._max_history: int = 100

    async def begin_transaction(
        self,
        planned_allocations: Dict[str, Decimal],
        trigger: str = "manual",
    ) -> AllocationTransaction:
        """
        Begin a new allocation transaction.

        Creates a snapshot of current state for potential rollback.

        Args:
            planned_allocations: Dict mapping bot_id to allocation amount
            trigger: What triggered this allocation

        Returns:
            New AllocationTransaction in PENDING state
        """
        # Create transaction
        tx = AllocationTransaction(
            trigger=trigger,
            planned_allocations=planned_allocations,
            pre_state_snapshot=self._fund_pool.create_allocation_snapshot(),
        )

        # Track active transaction
        self._active_transactions[tx.transaction_id] = tx

        logger.info(
            f"Transaction {tx.transaction_id[:8]} started: "
            f"{len(planned_allocations)} allocations planned"
        )

        return tx

    async def execute_transaction(
        self,
        tx: AllocationTransaction,
        timeout: Optional[float] = None,
    ) -> AllocationTransaction:
        """
        Execute the allocations in a transaction.

        Args:
            tx: Transaction to execute
            timeout: Optional timeout in seconds (uses config default if not provided)

        Returns:
            Updated transaction with execution results
        """
        if not tx.is_pending:
            logger.warning(f"Transaction {tx.transaction_id[:8]} not in pending state")
            return tx

        tx.mark_executing()
        timeout_config = get_timeout_config()
        effective_timeout = timeout or timeout_config.fund_allocation

        logger.info(f"Executing transaction {tx.transaction_id[:8]}")

        try:
            # Execute allocations with timeout
            async with asyncio.timeout(effective_timeout):
                for bot_id, amount in tx.planned_allocations.items():
                    record = await self._execute_single_allocation(
                        bot_id, amount, tx.trigger, tx.pre_state_snapshot
                    )
                    tx.add_executed(record)

        except asyncio.TimeoutError:
            logger.error(
                f"Transaction {tx.transaction_id[:8]} timed out after {effective_timeout}s"
            )
            tx.mark_failed(f"Transaction timed out after {effective_timeout}s")
            return tx

        except Exception as e:
            logger.error(f"Transaction {tx.transaction_id[:8]} failed: {e}")
            tx.mark_failed(str(e))
            return tx

        # Check if we should auto-rollback based on failure ratio
        total = len(tx.executed_allocations)
        if total > 0:
            failure_ratio = tx.failed_count / total
            if failure_ratio >= self._rollback_threshold:
                logger.warning(
                    f"Transaction {tx.transaction_id[:8]} failure ratio "
                    f"{failure_ratio:.1%} >= threshold {self._rollback_threshold:.1%}"
                )
                # Don't auto-rollback here, let caller decide

        return tx

    async def _execute_single_allocation(
        self,
        bot_id: str,
        amount: Decimal,
        trigger: str,
        pre_snapshot: Dict[str, Decimal],
    ) -> AllocationRecord:
        """
        Execute a single allocation with bot notification.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
            trigger: Allocation trigger
            pre_snapshot: Pre-transaction allocation snapshot

        Returns:
            AllocationRecord with result
        """
        previous = pre_snapshot.get(bot_id, Decimal("0"))
        new_total = previous + amount

        record = AllocationRecord(
            bot_id=bot_id,
            amount=amount,
            trigger=trigger,
            previous_allocation=previous,
            new_allocation=new_total,
        )

        try:
            # Notify bot via dispatcher
            dispatch_record = await self._dispatcher.notify_bot(
                bot_id=bot_id,
                amount=new_total,
                trigger=trigger,
                previous_allocation=previous,
            )

            if dispatch_record.success:
                # Update fund pool allocation tracking
                self._fund_pool.add_allocation(bot_id, amount)
                record.success = True
                logger.debug(f"Allocated {amount} to {bot_id} (new total: {new_total})")
            else:
                record.success = False
                record.error_message = dispatch_record.error_message
                logger.warning(f"Failed to allocate to {bot_id}: {record.error_message}")

        except Exception as e:
            record.success = False
            record.error_message = str(e)
            logger.error(f"Exception allocating to {bot_id}: {e}")

        return record

    async def commit_transaction(self, tx: AllocationTransaction) -> AllocationTransaction:
        """
        Commit a successful transaction.

        Args:
            tx: Transaction to commit

        Returns:
            Updated transaction
        """
        if tx.is_completed:
            logger.warning(f"Transaction {tx.transaction_id[:8]} already completed")
            return tx

        tx.mark_committed()

        # Remove from active, add to history
        self._active_transactions.pop(tx.transaction_id, None)
        self._transaction_history.append(tx)
        if len(self._transaction_history) > self._max_history:
            self._transaction_history = self._transaction_history[-self._max_history:]

        logger.info(
            f"Transaction {tx.transaction_id[:8]} committed: "
            f"{tx.successful_count} successful, {tx.failed_count} failed"
        )

        return tx

    async def rollback_transaction(
        self,
        tx: AllocationTransaction,
        reason: Optional[str] = None,
    ) -> AllocationTransaction:
        """
        Rollback a transaction, restoring previous state.

        Args:
            tx: Transaction to rollback
            reason: Optional reason for rollback

        Returns:
            Updated transaction
        """
        if tx.is_completed:
            logger.warning(f"Transaction {tx.transaction_id[:8]} already completed")
            return tx

        logger.warning(
            f"Rolling back transaction {tx.transaction_id[:8]}: {reason or 'no reason'}"
        )

        try:
            # Restore fund pool state
            self._fund_pool.restore_allocation_snapshot(tx.pre_state_snapshot)

            # Notify bots of rollback (restore previous allocations)
            for record in tx.executed_allocations:
                if record.success:
                    # This bot received allocation, notify it of rollback
                    try:
                        previous = tx.pre_state_snapshot.get(record.bot_id, Decimal("0"))
                        await self._dispatcher.notify_bot(
                            bot_id=record.bot_id,
                            amount=previous,  # Restore to previous amount
                            trigger="rollback",
                            previous_allocation=record.new_allocation,
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to notify {record.bot_id} of rollback: {e}"
                        )

            tx.mark_rolled_back(reason)

            # Send notification
            if self._notifier:
                await self._notifier.send_warning(
                    "Transaction Rolled Back",
                    f"Allocation transaction rolled back.\n"
                    f"Reason: {reason or 'Failure threshold exceeded'}\n"
                    f"Successful: {tx.successful_count}, Failed: {tx.failed_count}",
                )

        except Exception as e:
            logger.error(f"Rollback failed for {tx.transaction_id[:8]}: {e}")
            tx.mark_failed(f"Rollback failed: {e}")

        # Remove from active, add to history
        self._active_transactions.pop(tx.transaction_id, None)
        self._transaction_history.append(tx)
        if len(self._transaction_history) > self._max_history:
            self._transaction_history = self._transaction_history[-self._max_history:]

        return tx

    async def execute_atomic(
        self,
        planned_allocations: Dict[str, Decimal],
        trigger: str = "manual",
        auto_rollback: bool = True,
    ) -> AllocationTransaction:
        """
        Execute allocations atomically with automatic rollback on failure.

        Convenience method that handles the full transaction lifecycle.

        Args:
            planned_allocations: Dict mapping bot_id to allocation amount
            trigger: What triggered this allocation
            auto_rollback: Whether to auto-rollback on high failure rate

        Returns:
            Completed transaction
        """
        # Begin transaction
        tx = await self.begin_transaction(planned_allocations, trigger)

        # Execute
        tx = await self.execute_transaction(tx)

        # Check result
        if tx.status == TransactionStatus.FAILED:
            # Already failed, try to rollback
            if auto_rollback:
                tx = await self.rollback_transaction(tx, "Execution failed")
            return tx

        # Check failure ratio
        total = len(tx.executed_allocations)
        if total > 0 and auto_rollback:
            failure_ratio = tx.failed_count / total
            if failure_ratio >= self._rollback_threshold:
                tx = await self.rollback_transaction(
                    tx,
                    f"Failure ratio {failure_ratio:.1%} exceeded threshold",
                )
                return tx

        # All good, commit
        tx = await self.commit_transaction(tx)
        return tx

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_transactions(self) -> List[AllocationTransaction]:
        """Get all active (non-completed) transactions."""
        return list(self._active_transactions.values())

    def get_transaction(self, transaction_id: str) -> Optional[AllocationTransaction]:
        """Get a transaction by ID."""
        # Check active first
        if transaction_id in self._active_transactions:
            return self._active_transactions[transaction_id]

        # Check history
        for tx in self._transaction_history:
            if tx.transaction_id == transaction_id:
                return tx

        return None

    def get_transaction_history(
        self,
        limit: int = 50,
        status: Optional[TransactionStatus] = None,
    ) -> List[AllocationTransaction]:
        """
        Get transaction history.

        Args:
            limit: Maximum transactions to return
            status: Filter by status

        Returns:
            List of transactions (newest first)
        """
        history = self._transaction_history
        if status:
            history = [tx for tx in history if tx.status == status]
        return history[-limit:][::-1]

    def get_status(self) -> Dict[str, Any]:
        """Get manager status."""
        return {
            "active_transactions": len(self._active_transactions),
            "history_count": len(self._transaction_history),
            "rollback_threshold": self._rollback_threshold,
        }
