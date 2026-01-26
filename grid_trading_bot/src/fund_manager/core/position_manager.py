"""
Shared Position Manager.

Centralized position tracking across all trading bots to prevent:
- Position conflicts between bots
- Over-exposure on single symbols
- Inconsistent position state
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.core import get_logger
from src.core.models import MarketType

logger = get_logger(__name__)


class PositionSide(str, Enum):
    """Position side."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # Hedge mode


@dataclass
class SharedPosition:
    """
    Shared position state tracked across all bots.

    Attributes:
        symbol: Trading symbol
        side: Position side (long/short/both)
        quantity: Total position quantity
        entry_price: Average entry price
        unrealized_pnl: Unrealized profit/loss
        bot_contributions: Quantity contributed by each bot
        market_type: Spot or Futures
        updated_at: Last update timestamp
        leverage: Position leverage (futures only)
    """
    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    unrealized_pnl: Decimal = Decimal("0")
    bot_contributions: Dict[str, Decimal] = field(default_factory=dict)
    market_type: MarketType = MarketType.FUTURES
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    leverage: int = 1

    @property
    def total_contributed(self) -> Decimal:
        """Total quantity contributed by all bots."""
        return sum(self.bot_contributions.values())

    @property
    def bot_count(self) -> int:
        """Number of bots with contributions."""
        return len([q for q in self.bot_contributions.values() if q > 0])

    def get_bot_share(self, bot_id: str) -> Decimal:
        """Get a bot's share of this position."""
        return self.bot_contributions.get(bot_id, Decimal("0"))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "quantity": str(self.quantity),
            "entry_price": str(self.entry_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "bot_contributions": {k: str(v) for k, v in self.bot_contributions.items()},
            "market_type": self.market_type.value,
            "updated_at": self.updated_at.isoformat(),
            "leverage": self.leverage,
        }


@dataclass
class PositionChange:
    """Record of a position change."""
    symbol: str
    bot_id: str
    side: PositionSide
    quantity_change: Decimal  # Positive = add, Negative = reduce
    price: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get positions from exchange."""
        ...


class SharedPositionManager:
    """
    Centralized position manager for all trading bots.

    Ensures consistent position tracking and prevents conflicts:
    - Single source of truth for all positions
    - Tracks which bot owns which portion of a position
    - Prevents over-exposure through position limits
    - Synchronizes with exchange state

    Example:
        >>> manager = SharedPositionManager(exchange)
        >>> await manager.start()
        >>>
        >>> # Bot requests to open position
        >>> success, msg = await manager.request_position(
        ...     bot_id="grid_bot_1",
        ...     symbol="BTCUSDT",
        ...     side=PositionSide.LONG,
        ...     quantity=Decimal("0.01"),
        ...     price=Decimal("50000"),
        ... )
        >>>
        >>> # Get bot's position share
        >>> share = manager.get_bot_position("grid_bot_1", "BTCUSDT")
    """

    _instance: Optional["SharedPositionManager"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SharedPositionManager":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        exchange: Optional[ExchangeProtocol] = None,
        max_position_per_symbol: Optional[Decimal] = None,
        max_total_exposure: Optional[Decimal] = None,
        sync_interval: float = 30.0,
    ):
        """
        Initialize SharedPositionManager.

        Args:
            exchange: Exchange client for syncing positions
            max_position_per_symbol: Maximum position size per symbol
            max_total_exposure: Maximum total exposure across all symbols
            sync_interval: Seconds between exchange syncs
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._exchange = exchange
        self._max_position_per_symbol = max_position_per_symbol
        self._max_total_exposure = max_total_exposure
        self._sync_interval = sync_interval

        # Position tracking
        self._positions: Dict[str, SharedPosition] = {}  # symbol -> position
        self._position_lock = asyncio.Lock()

        # Change history
        self._changes: List[PositionChange] = []
        self._max_changes: int = 1000

        # Sync state
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_position_change: List[Callable[[PositionChange], None]] = []

        self._initialized = True
        logger.info("SharedPositionManager initialized")

    @classmethod
    def get_instance(cls) -> Optional["SharedPositionManager"]:
        """Get singleton instance if exists."""
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance._initialized = False
        cls._instance = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start the position manager."""
        if self._running:
            return

        logger.info("Starting SharedPositionManager...")

        # Initial sync
        await self.sync_with_exchange()

        # Start sync loop
        self._running = True
        self._sync_task = asyncio.create_task(self._sync_loop())

        logger.info("SharedPositionManager started")

    async def stop(self) -> None:
        """Stop the position manager."""
        if not self._running:
            return

        logger.info("Stopping SharedPositionManager...")

        self._running = False
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        logger.info("SharedPositionManager stopped")

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval)
                if self._running:
                    await self.sync_with_exchange()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")

    # =========================================================================
    # Position Operations (Thread-Safe)
    # =========================================================================

    async def request_position(
        self,
        bot_id: str,
        symbol: str,
        side: PositionSide,
        quantity: Decimal,
        price: Decimal,
        reason: str = "",
    ) -> tuple[bool, str]:
        """
        Request to open/add to a position.

        Thread-safe: Uses position lock to prevent race conditions.

        Args:
            bot_id: Requesting bot's ID
            symbol: Trading symbol
            side: Position side
            quantity: Quantity to add
            price: Entry price
            reason: Reason for the position

        Returns:
            Tuple of (success, message)
        """
        async with self._position_lock:
            # Check position limits
            if self._max_position_per_symbol:
                current = self._positions.get(symbol)
                current_qty = current.quantity if current else Decimal("0")
                if current_qty + quantity > self._max_position_per_symbol:
                    return False, (
                        f"Exceeds max position: {current_qty + quantity} > "
                        f"{self._max_position_per_symbol}"
                    )

            # Check total exposure
            if self._max_total_exposure:
                total_exposure = sum(
                    p.quantity * p.entry_price for p in self._positions.values()
                )
                new_exposure = quantity * price
                if total_exposure + new_exposure > self._max_total_exposure:
                    return False, (
                        f"Exceeds max exposure: "
                        f"{total_exposure + new_exposure} > {self._max_total_exposure}"
                    )

            # Update or create position
            if symbol in self._positions:
                pos = self._positions[symbol]
                # Update average entry price
                old_value = pos.quantity * pos.entry_price
                new_value = quantity * price
                pos.quantity += quantity
                if pos.quantity > 0:
                    pos.entry_price = (old_value + new_value) / pos.quantity
                pos.bot_contributions[bot_id] = (
                    pos.bot_contributions.get(bot_id, Decimal("0")) + quantity
                )
                pos.updated_at = datetime.now(timezone.utc)
            else:
                self._positions[symbol] = SharedPosition(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=price,
                    bot_contributions={bot_id: quantity},
                )

            # Record change
            change = PositionChange(
                symbol=symbol,
                bot_id=bot_id,
                side=side,
                quantity_change=quantity,
                price=price,
                reason=reason,
            )
            self._record_change(change)

            logger.info(
                f"Position opened: {bot_id} {side.value} {quantity} {symbol} @ {price}"
            )
            return True, f"Position opened: {quantity} {symbol}"

    async def release_position(
        self,
        bot_id: str,
        symbol: str,
        quantity: Optional[Decimal] = None,
        price: Decimal = Decimal("0"),
        reason: str = "",
    ) -> tuple[bool, Decimal]:
        """
        Release/reduce a position.

        Thread-safe: Uses position lock.

        Args:
            bot_id: Releasing bot's ID
            symbol: Trading symbol
            quantity: Quantity to release (None = all)
            price: Exit price
            reason: Reason for release

        Returns:
            Tuple of (success, quantity_released)
        """
        async with self._position_lock:
            if symbol not in self._positions:
                return False, Decimal("0")

            pos = self._positions[symbol]
            bot_qty = pos.bot_contributions.get(bot_id, Decimal("0"))

            if bot_qty <= 0:
                return False, Decimal("0")

            # Determine release amount
            if quantity is None:
                release_qty = bot_qty
            else:
                release_qty = min(quantity, bot_qty)

            # Update position
            pos.quantity -= release_qty
            pos.bot_contributions[bot_id] = bot_qty - release_qty
            pos.updated_at = datetime.now(timezone.utc)

            # Clean up zero contributions
            if pos.bot_contributions[bot_id] <= 0:
                del pos.bot_contributions[bot_id]

            # Remove position if fully closed
            if pos.quantity <= 0:
                del self._positions[symbol]

            # Record change
            change = PositionChange(
                symbol=symbol,
                bot_id=bot_id,
                side=pos.side,
                quantity_change=-release_qty,
                price=price,
                reason=reason,
            )
            self._record_change(change)

            logger.info(
                f"Position released: {bot_id} {release_qty} {symbol} @ {price}"
            )
            return True, release_qty

    async def transfer_position(
        self,
        from_bot_id: str,
        to_bot_id: str,
        symbol: str,
        quantity: Decimal,
    ) -> tuple[bool, str]:
        """
        Transfer position ownership between bots.

        Thread-safe: Uses position lock.

        Args:
            from_bot_id: Source bot
            to_bot_id: Destination bot
            symbol: Trading symbol
            quantity: Quantity to transfer

        Returns:
            Tuple of (success, message)
        """
        async with self._position_lock:
            if symbol not in self._positions:
                return False, f"No position for {symbol}"

            pos = self._positions[symbol]
            from_qty = pos.bot_contributions.get(from_bot_id, Decimal("0"))

            if quantity > from_qty:
                return False, (
                    f"Insufficient position: {from_bot_id} has {from_qty}, "
                    f"requested {quantity}"
                )

            # Transfer
            pos.bot_contributions[from_bot_id] = from_qty - quantity
            pos.bot_contributions[to_bot_id] = (
                pos.bot_contributions.get(to_bot_id, Decimal("0")) + quantity
            )
            pos.updated_at = datetime.now(timezone.utc)

            # Clean up zero contributions
            if pos.bot_contributions[from_bot_id] <= 0:
                del pos.bot_contributions[from_bot_id]

            logger.info(
                f"Position transferred: {quantity} {symbol} "
                f"from {from_bot_id} to {to_bot_id}"
            )
            return True, f"Transferred {quantity} {symbol}"

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_position(self, symbol: str) -> Optional[SharedPosition]:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def get_bot_position(
        self,
        bot_id: str,
        symbol: str,
    ) -> Decimal:
        """Get a bot's share of a position."""
        pos = self._positions.get(symbol)
        if pos:
            return pos.bot_contributions.get(bot_id, Decimal("0"))
        return Decimal("0")

    def get_bot_positions(self, bot_id: str) -> Dict[str, Decimal]:
        """Get all positions for a bot."""
        result: Dict[str, Decimal] = {}
        for symbol, pos in self._positions.items():
            qty = pos.bot_contributions.get(bot_id, Decimal("0"))
            if qty > 0:
                result[symbol] = qty
        return result

    def get_all_positions(self) -> Dict[str, SharedPosition]:
        """Get all positions."""
        return self._positions.copy()

    def get_total_exposure(self) -> Decimal:
        """Get total exposure across all positions."""
        return sum(
            p.quantity * p.entry_price for p in self._positions.values()
        )

    # =========================================================================
    # Exchange Sync
    # =========================================================================

    async def sync_with_exchange(self) -> None:
        """Synchronize positions with exchange."""
        if not self._exchange:
            return

        try:
            exchange_positions = await self._exchange.get_positions()

            async with self._position_lock:
                # Update from exchange
                for ep in exchange_positions:
                    symbol = ep.symbol
                    if symbol in self._positions:
                        pos = self._positions[symbol]
                        # Update from exchange but keep bot contributions
                        pos.quantity = ep.quantity
                        pos.entry_price = ep.entry_price
                        pos.unrealized_pnl = getattr(ep, "unrealized_pnl", Decimal("0"))
                        pos.leverage = getattr(ep, "leverage", 1)
                        pos.updated_at = datetime.now(timezone.utc)
                    else:
                        # New position from exchange (not tracked by any bot)
                        self._positions[symbol] = SharedPosition(
                            symbol=symbol,
                            side=PositionSide.LONG if ep.quantity > 0 else PositionSide.SHORT,
                            quantity=abs(ep.quantity),
                            entry_price=ep.entry_price,
                            unrealized_pnl=getattr(ep, "unrealized_pnl", Decimal("0")),
                            leverage=getattr(ep, "leverage", 1),
                            bot_contributions={"_untracked": abs(ep.quantity)},
                        )

                # Remove positions that no longer exist on exchange
                exchange_symbols = {ep.symbol for ep in exchange_positions if ep.quantity != 0}
                for symbol in list(self._positions.keys()):
                    if symbol not in exchange_symbols:
                        logger.warning(
                            f"Position {symbol} closed on exchange, removing"
                        )
                        del self._positions[symbol]

            logger.debug(f"Synced {len(exchange_positions)} positions from exchange")

        except Exception as e:
            logger.error(f"Failed to sync positions: {e}")

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _record_change(self, change: PositionChange) -> None:
        """Record a position change."""
        self._changes.append(change)
        if len(self._changes) > self._max_changes:
            self._changes = self._changes[-self._max_changes:]

        # Notify callbacks (copy list to allow modifications during iteration)
        callbacks = list(self._on_position_change)
        for callback in callbacks:
            try:
                callback(change)
            except Exception as e:
                logger.error(f"Position change callback error: {e}")

    def on_position_change(self, callback: Callable[[PositionChange], None]) -> None:
        """Register callback for position changes."""
        self._on_position_change.append(callback)

    # =========================================================================
    # Status Methods
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get position manager status."""
        return {
            "running": self._running,
            "position_count": len(self._positions),
            "total_exposure": str(self.get_total_exposure()),
            "positions": {
                symbol: pos.to_dict()
                for symbol, pos in self._positions.items()
            },
            "max_position_per_symbol": (
                str(self._max_position_per_symbol)
                if self._max_position_per_symbol else None
            ),
            "max_total_exposure": (
                str(self._max_total_exposure)
                if self._max_total_exposure else None
            ),
        }

    def get_change_history(
        self,
        bot_id: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[PositionChange]:
        """Get position change history."""
        changes = self._changes
        if bot_id:
            changes = [c for c in changes if c.bot_id == bot_id]
        if symbol:
            changes = [c for c in changes if c.symbol == symbol]
        return changes[-limit:]
