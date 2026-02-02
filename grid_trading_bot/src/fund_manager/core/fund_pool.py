"""
Fund Pool Manager.

Monitors account balance, detects deposits, and tracks fund allocations.
Thread-safe with asyncio.Lock for concurrent access protection.
"""

import asyncio
import sqlite3
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from src.core import get_logger
from src.core.models import AccountInfo, MarketType

from ..models.config import FundManagerConfig
from ..models.records import BalanceSnapshot

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_account(self, market: MarketType = MarketType.SPOT) -> AccountInfo:
        """Get account information."""
        ...


class FundPool:
    """
    Fund pool manager for monitoring and allocating capital.

    Tracks account balance, detects deposits, and manages fund allocations
    across trading bots.

    Example:
        >>> pool = FundPool(exchange, config)
        >>> await pool.update()
        >>> if pool.detect_deposit():
        ...     unallocated = pool.get_unallocated()
    """

    def __init__(
        self,
        exchange: Optional[ExchangeProtocol] = None,
        config: Optional[FundManagerConfig] = None,
        market_type: MarketType = MarketType.FUTURES,
        db_path: Optional[str] = None,
    ):
        """
        Initialize FundPool.

        Args:
            exchange: Exchange client for fetching account data
            config: Fund manager configuration
            market_type: Market type to monitor (SPOT or FUTURES)
            db_path: Path to SQLite database for allocation persistence
        """
        self._exchange = exchange
        self._config = config or FundManagerConfig()
        self._market_type = market_type
        self._db_path = db_path

        # =======================================================================
        # Thread Safety: Allocation Lock
        # =======================================================================
        # Prevents race conditions when multiple bots request funds concurrently
        self._allocation_lock = asyncio.Lock()

        # Balance tracking
        self._current_snapshot: Optional[BalanceSnapshot] = None
        self._previous_snapshot: Optional[BalanceSnapshot] = None
        self._snapshots: List[BalanceSnapshot] = []
        self._max_snapshots: int = 1000

        # Allocation tracking (bot_id -> allocated amount)
        self._allocations: Dict[str, Decimal] = {}

        # Leverage tracking (bot_id -> leverage multiplier)
        self._leverage_map: Dict[str, int] = {}

        # Deposit detection
        self._last_deposit_amount: Decimal = Decimal("0")
        self._deposit_detected: bool = False

        # Initialize persistence and restore state
        if self._db_path:
            self._init_db()
            self._restore_allocations()

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def current_snapshot(self) -> Optional[BalanceSnapshot]:
        """Get current balance snapshot."""
        return self._current_snapshot

    @property
    def total_balance(self) -> Decimal:
        """Get total account balance."""
        if self._current_snapshot:
            return self._current_snapshot.total_balance
        return Decimal("0")

    @property
    def available_balance(self) -> Decimal:
        """Get available balance."""
        if self._current_snapshot:
            return self._current_snapshot.available_balance
        return Decimal("0")

    @property
    def allocated_balance(self) -> Decimal:
        """Get total allocated balance.

        Note: This is a snapshot and may be inconsistent during concurrent
        modifications. For thread-safe access, use get_allocated_balance_safe().
        """
        # Create a copy of values to avoid iteration errors during modification
        return sum(list(self._allocations.values()))

    async def get_allocated_balance_safe(self) -> Decimal:
        """Get total allocated balance with lock protection.

        Use this method when accurate balance is critical during concurrent operations.
        """
        async with self._allocation_lock:
            return sum(self._allocations.values())

    @property
    def reserved_balance(self) -> Decimal:
        """Get reserved balance (not for allocation).

        Based on total_balance to ensure stable reserves regardless of
        funds tied up in open orders.
        """
        return self.total_balance * self._config.reserve_ratio

    @property
    def allocations(self) -> Dict[str, Decimal]:
        """Get current allocations by bot_id."""
        return self._allocations.copy()

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def update(self) -> BalanceSnapshot:
        """
        Update balance snapshot from exchange.

        Fetches account data and creates a new balance snapshot.

        Returns:
            New BalanceSnapshot

        Raises:
            RuntimeError: If no exchange client configured
        """
        if self._exchange is None:
            raise RuntimeError("No exchange client configured")

        # Get account info
        account = await self._exchange.get_account(self._market_type)

        # Use Binance futures account-level fields if available
        # These reflect the actual margin balance (USDT-M) from the API
        if (
            self._market_type == MarketType.FUTURES
            and account.total_wallet_balance is not None
            and account.available_balance is not None
        ):
            total_balance = account.total_wallet_balance
            available_balance = account.available_balance
        else:
            # Fallback: sum USDT balances (for spot or missing fields)
            total_balance = Decimal("0")
            available_balance = Decimal("0")
            for balance in account.balances:
                if balance.asset == "USDT":
                    total_balance += balance.total
                    available_balance += balance.free

        # Create snapshot with lock protection for consistent allocated balance
        async with self._allocation_lock:
            allocated = sum(self._allocations.values())
            reserved = total_balance * self._config.reserve_ratio
            snapshot = BalanceSnapshot(
                timestamp=datetime.now(timezone.utc),
                total_balance=total_balance,
                available_balance=available_balance,
                allocated_balance=allocated,
                reserved_balance=reserved,
            )

        # Update internal state
        self._update_snapshot(snapshot)

        return snapshot

    def update_from_values(
        self,
        total_balance: Decimal,
        available_balance: Decimal,
    ) -> BalanceSnapshot:
        """
        Update balance snapshot from provided values.

        Useful for testing or when not using exchange client directly.

        Args:
            total_balance: Total account balance
            available_balance: Available balance

        Returns:
            New BalanceSnapshot
        """
        snapshot = BalanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=total_balance,
            available_balance=available_balance,
            allocated_balance=self.allocated_balance,
            reserved_balance=self.reserved_balance,
        )

        self._update_snapshot(snapshot)
        return snapshot

    def _update_snapshot(self, snapshot: BalanceSnapshot) -> None:
        """
        Update internal state with new snapshot.

        Args:
            snapshot: New balance snapshot
        """
        # Check for deposit
        self._detect_deposit_internal(snapshot)

        # Store snapshot
        self._previous_snapshot = self._current_snapshot
        self._current_snapshot = snapshot
        self._snapshots.append(snapshot)

        # Trim old snapshots
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots:]

        logger.debug(
            f"Balance updated: {snapshot.total_balance} "
            f"(available: {snapshot.available_balance})"
        )

    def _detect_deposit_internal(self, new_snapshot: BalanceSnapshot) -> None:
        """
        Internal deposit detection.

        Args:
            new_snapshot: New balance snapshot
        """
        self._deposit_detected = False
        self._last_deposit_amount = Decimal("0")

        if self._current_snapshot is None:
            return

        # Calculate balance change
        change = new_snapshot.total_balance - self._current_snapshot.total_balance

        # Check if change exceeds deposit threshold
        if change >= self._config.deposit_threshold:
            self._deposit_detected = True
            self._last_deposit_amount = change
            logger.info(f"Deposit detected: {change} USDT")

    # =========================================================================
    # Query Methods
    # =========================================================================

    def detect_deposit(self) -> bool:
        """
        Check if a deposit was detected in the last update.

        Returns:
            True if deposit was detected
        """
        return self._deposit_detected

    def get_deposit_amount(self) -> Decimal:
        """
        Get the last detected deposit amount.

        Returns:
            Deposit amount or 0 if no deposit
        """
        return self._last_deposit_amount

    def get_unallocated(self) -> Decimal:
        """
        Get unallocated funds available for distribution.

        Returns:
            Unallocated balance (available - allocated - reserved)
        """
        unallocated = (
            self.available_balance
            - self.allocated_balance
            - self.reserved_balance
        )
        return max(Decimal("0"), unallocated)

    def get_allocation(self, bot_id: str) -> Decimal:
        """
        Get current allocation for a bot (sync version).

        Note: For thread-safe access, use get_allocation_async().

        Args:
            bot_id: Bot identifier

        Returns:
            Allocated amount
        """
        return self._allocations.get(bot_id, Decimal("0"))

    async def get_allocation_async(self, bot_id: str) -> Decimal:
        """
        Get current allocation for a bot with lock protection.

        Args:
            bot_id: Bot identifier

        Returns:
            Allocated amount
        """
        async with self._allocation_lock:
            return self._allocations.get(bot_id, Decimal("0"))

    # =========================================================================
    # Allocation Management
    # =========================================================================

    def set_allocation(self, bot_id: str, amount: Decimal) -> None:
        """
        Set allocation for a bot (sync version).

        Note: For thread-safe access, use set_allocation_async() or call
        within allocation_lock context.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
        """
        if amount <= 0:
            self._allocations.pop(bot_id, None)
        else:
            self._allocations[bot_id] = amount
        self._persist_allocation(bot_id, amount)
        logger.debug(f"Allocation set for {bot_id}: {amount}")

    async def set_allocation_async(self, bot_id: str, amount: Decimal) -> None:
        """
        Set allocation for a bot with lock protection.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
        """
        async with self._allocation_lock:
            if amount <= 0:
                self._allocations.pop(bot_id, None)
            else:
                self._allocations[bot_id] = amount
            self._persist_allocation(bot_id, amount)
            logger.debug(f"Allocation set for {bot_id}: {amount}")

    def add_allocation(self, bot_id: str, amount: Decimal) -> Decimal:
        """
        Add to existing allocation for a bot (sync version).

        Persists updated allocation to database if configured.

        Note: For thread-safe access, use add_allocation_async() or call
        within allocation_lock context.

        Args:
            bot_id: Bot identifier
            amount: Amount to add

        Returns:
            New total allocation
        """
        current = self._allocations.get(bot_id, Decimal("0"))
        new_amount = current + amount
        self.set_allocation(bot_id, new_amount)
        return new_amount

    async def add_allocation_async(self, bot_id: str, amount: Decimal) -> Decimal:
        """
        Add to existing allocation for a bot with lock protection.

        Args:
            bot_id: Bot identifier
            amount: Amount to add

        Returns:
            New total allocation
        """
        async with self._allocation_lock:
            current = self._allocations.get(bot_id, Decimal("0"))
            new_amount = current + amount
            if new_amount <= 0:
                self._allocations.pop(bot_id, None)
            else:
                self._allocations[bot_id] = new_amount
            self._persist_allocation(bot_id, new_amount)
            logger.debug(f"Allocation added for {bot_id}: {amount} (total: {new_amount})")
            return new_amount

    def remove_allocation(self, bot_id: str) -> Decimal:
        """
        Remove allocation for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Previous allocation amount
        """
        amount = self._allocations.pop(bot_id, Decimal("0"))
        self._persist_allocation(bot_id, Decimal("0"))
        return amount

    def clear_allocations(self) -> None:
        """Clear all allocations."""
        bot_ids = list(self._allocations.keys())
        self._allocations.clear()
        for bot_id in bot_ids:
            self._persist_allocation(bot_id, Decimal("0"))
        logger.info("All allocations cleared")

    # =========================================================================
    # Atomic Allocation Methods (Thread-Safe)
    # =========================================================================

    async def atomic_allocate(
        self,
        bot_id: str,
        amount: Decimal,
        check_available: bool = True,
    ) -> tuple[bool, str]:
        """
        Atomically allocate funds to a bot with lock protection.

        This method ensures that:
        1. Balance check and allocation happen atomically
        2. No race condition between multiple bots
        3. Funds cannot be over-allocated

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
            check_available: Whether to check available balance

        Returns:
            Tuple of (success, message)
        """
        async with self._allocation_lock:
            # Check available funds
            if check_available:
                unallocated = self.get_unallocated()
                if amount > unallocated:
                    return False, (
                        f"Insufficient funds: requested {amount}, "
                        f"available {unallocated}"
                    )

            # Check exposure limit before allocating
            if self.check_exposure_limit():
                return False, (
                    f"Allocation rejected: system exposure exceeds safety limit"
                )

            # Perform allocation
            current = self._allocations.get(bot_id, Decimal("0"))
            new_amount = current + amount
            self._allocations[bot_id] = new_amount
            self._persist_allocation(bot_id, new_amount)

            logger.info(
                f"Atomic allocation: {bot_id} += {amount} "
                f"(total: {new_amount})"
            )
            return True, f"Allocated {amount} to {bot_id}"

    async def atomic_deallocate(
        self,
        bot_id: str,
        amount: Optional[Decimal] = None,
    ) -> tuple[bool, Decimal]:
        """
        Atomically deallocate funds from a bot with lock protection.

        Also clears related bot metadata (leverage tracking) when fully
        deallocated, preventing stale data on bot restart.

        Args:
            bot_id: Bot identifier
            amount: Amount to deallocate (None = all)

        Returns:
            Tuple of (success, amount_deallocated)
        """
        async with self._allocation_lock:
            current = self._allocations.get(bot_id, Decimal("0"))

            if current <= 0:
                return False, Decimal("0")

            if amount is None:
                dealloc_amount = current
            else:
                dealloc_amount = min(amount, current)

            new_amount = current - dealloc_amount

            if new_amount <= 0:
                self._allocations.pop(bot_id, None)
                # Clear associated metadata to prevent stale data on restart
                self._leverage_map.pop(bot_id, None)
            else:
                self._allocations[bot_id] = new_amount
            self._persist_allocation(bot_id, new_amount)

            logger.info(
                f"Atomic deallocation: {bot_id} -= {dealloc_amount} "
                f"(remaining: {new_amount})"
            )
            return True, dealloc_amount

    async def atomic_transfer(
        self,
        from_bot_id: str,
        to_bot_id: str,
        amount: Decimal,
    ) -> tuple[bool, str]:
        """
        Atomically transfer funds between bots with lock protection.

        Args:
            from_bot_id: Source bot identifier
            to_bot_id: Destination bot identifier
            amount: Amount to transfer

        Returns:
            Tuple of (success, message)
        """
        async with self._allocation_lock:
            from_current = self._allocations.get(from_bot_id, Decimal("0"))

            if amount > from_current:
                return False, (
                    f"Insufficient allocation: {from_bot_id} has {from_current}, "
                    f"requested {amount}"
                )

            # Perform transfer
            from_new = from_current - amount
            self._allocations[from_bot_id] = from_new
            to_current = self._allocations.get(to_bot_id, Decimal("0"))
            to_new = to_current + amount
            self._allocations[to_bot_id] = to_new

            # Clean up zero allocations
            if self._allocations[from_bot_id] <= 0:
                self._allocations.pop(from_bot_id, None)

            self._persist_allocation(from_bot_id, from_new)
            self._persist_allocation(to_bot_id, to_new)

            logger.info(
                f"Atomic transfer: {from_bot_id} -> {to_bot_id}: {amount}"
            )
            return True, f"Transferred {amount} from {from_bot_id} to {to_bot_id}"

    @property
    def allocation_lock(self) -> asyncio.Lock:
        """Get the allocation lock for external use."""
        return self._allocation_lock

    # =========================================================================
    # Leverage & Exposure Tracking
    # =========================================================================

    def set_leverage(self, bot_id: str, leverage: int) -> None:
        """Set leverage multiplier for a bot."""
        self._leverage_map[bot_id] = leverage

    @property
    def total_notional_exposure(self) -> Decimal:
        """Calculate total notional exposure across all bots."""
        total = Decimal("0")
        for bot_id, allocated in self._allocations.items():
            lev = self._leverage_map.get(bot_id, 1)
            total += allocated * lev
        return total

    def check_exposure_limit(self, max_ratio: Decimal = Decimal("12.0")) -> bool:
        """
        Check if system-wide exposure exceeds the safety limit.

        Args:
            max_ratio: Maximum allowed exposure ratio (notional / total_balance)

        Returns:
            True if exposure exceeds limit (warning condition)
        """
        if self.total_balance <= 0:
            return True  # No balance = exceeds limit (safety: block allocation)
        ratio = self.total_notional_exposure / self.total_balance
        if ratio > max_ratio:
            logger.warning(
                f"System exposure {ratio:.1f}x exceeds limit {max_ratio}x"
            )
            return True
        return False

    # =========================================================================
    # Persistence Methods
    # =========================================================================

    def _init_db(self) -> None:
        """Initialize SQLite database for allocation persistence."""
        if not self._db_path:
            return
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS allocations ("
                "  bot_id TEXT PRIMARY KEY,"
                "  amount TEXT NOT NULL,"
                "  updated_at TEXT NOT NULL"
                ")"
            )
            conn.commit()
        finally:
            conn.close()

    def _persist_allocation(self, bot_id: str, amount: Decimal) -> None:
        """Persist a single allocation to SQLite."""
        if not self._db_path:
            return
        try:
            self._persist_allocation_sync(bot_id, amount)
        except Exception as e:
            logger.warning(f"Failed to persist allocation for {bot_id}: {e}")

    def _persist_allocation_sync(self, bot_id: str, amount: Decimal) -> None:
        """Synchronous SQLite persist (can be offloaded to executor)."""
        conn = sqlite3.connect(self._db_path)
        try:
            now = datetime.now(timezone.utc).isoformat()
            if amount <= 0:
                conn.execute(
                    "DELETE FROM allocations WHERE bot_id = ?", (bot_id,)
                )
            else:
                conn.execute(
                    "INSERT OR REPLACE INTO allocations (bot_id, amount, updated_at) "
                    "VALUES (?, ?, ?)",
                    (bot_id, str(amount), now),
                )
            conn.commit()
        finally:
            conn.close()

    def _restore_allocations(self) -> None:
        """Restore allocations from SQLite on startup."""
        if not self._db_path or not Path(self._db_path).exists():
            return
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT bot_id, amount FROM allocations"
                ).fetchall()
                for bot_id, amount_str in rows:
                    self._allocations[bot_id] = Decimal(amount_str)
                if rows:
                    logger.info(
                        f"Restored {len(rows)} allocations from database"
                    )
            finally:
                conn.close()
        except Exception as e:
            logger.warning(f"Failed to restore allocations from DB: {e}")

    # =========================================================================
    # Status Methods
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get fund pool status.

        Returns:
            Dictionary with current status
        """
        return {
            "total_balance": str(self.total_balance),
            "available_balance": str(self.available_balance),
            "allocated_balance": str(self.allocated_balance),
            "reserved_balance": str(self.reserved_balance),
            "unallocated_balance": str(self.get_unallocated()),
            "allocation_count": len(self._allocations),
            "deposit_detected": self._deposit_detected,
            "last_deposit_amount": str(self._last_deposit_amount),
            "last_update": (
                self._current_snapshot.timestamp.isoformat()
                if self._current_snapshot
                else None
            ),
        }

    def get_snapshots_since(self, since: datetime) -> List[BalanceSnapshot]:
        """
        Get snapshots since a specific time.

        Args:
            since: Start time

        Returns:
            List of snapshots since the given time
        """
        return [s for s in self._snapshots if s.timestamp >= since]

    # =========================================================================
    # State Snapshot Methods (for Atomic Transactions)
    # =========================================================================

    def create_allocation_snapshot(self) -> Dict[str, Decimal]:
        """
        Create a snapshot of current allocations.

        Used for transaction rollback support.

        Returns:
            Copy of current allocations
        """
        return self._allocations.copy()

    def restore_allocation_snapshot(self, snapshot: Dict[str, Decimal]) -> None:
        """
        Restore allocations from a snapshot.

        Used for transaction rollback.

        Args:
            snapshot: Allocation snapshot to restore
        """
        # Find bots to delete (in current allocations but not in snapshot)
        bots_to_delete = set(self._allocations.keys()) - set(snapshot.keys())

        self._allocations = snapshot.copy()
        # Persist all restored allocations to database
        for bot_id, amount in self._allocations.items():
            self._persist_allocation(bot_id, amount)
        # Remove stale allocations from database
        for bot_id in bots_to_delete:
            self._persist_allocation(bot_id, Decimal("0"))
        logger.info(
            f"Restored allocation snapshot with {len(snapshot)} bots, "
            f"removed {len(bots_to_delete)} stale allocations"
        )

    def get_allocation_diff(
        self,
        snapshot: Dict[str, Decimal],
    ) -> Dict[str, tuple[Decimal, Decimal]]:
        """
        Get differences between current state and snapshot.

        Args:
            snapshot: Previous allocation snapshot

        Returns:
            Dict mapping bot_id to (old_value, new_value) for changed allocations
        """
        diff: Dict[str, tuple[Decimal, Decimal]] = {}

        # Check for changes and additions
        for bot_id, current in self._allocations.items():
            previous = snapshot.get(bot_id, Decimal("0"))
            if current != previous:
                diff[bot_id] = (previous, current)

        # Check for removals
        for bot_id, previous in snapshot.items():
            if bot_id not in self._allocations:
                diff[bot_id] = (previous, Decimal("0"))

        return diff
