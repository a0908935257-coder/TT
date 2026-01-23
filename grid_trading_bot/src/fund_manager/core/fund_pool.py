"""
Fund Pool Manager.

Monitors account balance, detects deposits, and tracks fund allocations.
"""

from datetime import datetime, timezone
from decimal import Decimal
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
    ):
        """
        Initialize FundPool.

        Args:
            exchange: Exchange client for fetching account data
            config: Fund manager configuration
            market_type: Market type to monitor (SPOT or FUTURES)
        """
        self._exchange = exchange
        self._config = config or FundManagerConfig()
        self._market_type = market_type

        # Balance tracking
        self._current_snapshot: Optional[BalanceSnapshot] = None
        self._previous_snapshot: Optional[BalanceSnapshot] = None
        self._snapshots: List[BalanceSnapshot] = []
        self._max_snapshots: int = 1000

        # Allocation tracking (bot_id -> allocated amount)
        self._allocations: Dict[str, Decimal] = {}

        # Deposit detection
        self._last_deposit_amount: Decimal = Decimal("0")
        self._deposit_detected: bool = False

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
        """Get total allocated balance."""
        return sum(self._allocations.values())

    @property
    def reserved_balance(self) -> Decimal:
        """Get reserved balance (not for allocation)."""
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

        # Calculate totals from balances
        total_balance = Decimal("0")
        available_balance = Decimal("0")

        for balance in account.balances:
            # Sum USDT-like stablecoin balances
            if balance.asset in ("USDT", "BUSD", "USDC"):
                total_balance += balance.total
                available_balance += balance.free

        # Create snapshot
        snapshot = BalanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=total_balance,
            available_balance=available_balance,
            allocated_balance=self.allocated_balance,
            reserved_balance=self.reserved_balance,
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
        Get current allocation for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Allocated amount
        """
        return self._allocations.get(bot_id, Decimal("0"))

    # =========================================================================
    # Allocation Management
    # =========================================================================

    def set_allocation(self, bot_id: str, amount: Decimal) -> None:
        """
        Set allocation for a bot.

        Args:
            bot_id: Bot identifier
            amount: Amount to allocate
        """
        if amount <= 0:
            self._allocations.pop(bot_id, None)
        else:
            self._allocations[bot_id] = amount
        logger.debug(f"Allocation set for {bot_id}: {amount}")

    def add_allocation(self, bot_id: str, amount: Decimal) -> Decimal:
        """
        Add to existing allocation for a bot.

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

    def remove_allocation(self, bot_id: str) -> Decimal:
        """
        Remove allocation for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Previous allocation amount
        """
        return self._allocations.pop(bot_id, Decimal("0"))

    def clear_allocations(self) -> None:
        """Clear all allocations."""
        self._allocations.clear()
        logger.info("All allocations cleared")

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
