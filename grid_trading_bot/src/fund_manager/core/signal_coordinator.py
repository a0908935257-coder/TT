"""
Signal Coordinator.

Centralized signal coordination to prevent multi-bot conflicts:
- Detects when bots want to trade opposite directions on same symbol
- Provides coordination strategies (block, warn, hedge)
- Tracks signal history for conflict analysis
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.core import get_logger

logger = get_logger(__name__)


class SignalDirection(str, Enum):
    """Signal direction."""
    LONG = "long"
    SHORT = "short"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    NEUTRAL = "neutral"


class ConflictResolution(str, Enum):
    """How to resolve signal conflicts."""
    BLOCK_NEWER = "block_newer"  # Block the newer conflicting signal
    BLOCK_SMALLER = "block_smaller"  # Block signal with smaller size
    WARN_ONLY = "warn_only"  # Log warning but allow both
    ALLOW_HEDGE = "allow_hedge"  # Allow hedging (both long and short)
    PRIORITY_BASED = "priority_based"  # Use bot priority to decide


class ConflictType(str, Enum):
    """Type of signal conflict."""
    OPPOSITE_DIRECTION = "opposite_direction"  # Long vs Short
    SAME_DIRECTION_OVERLAP = "same_direction_overlap"  # Multiple bots same direction
    CLOSE_WHILE_OPENING = "close_while_opening"  # One closing while another opening
    NONE = "none"


@dataclass
class SignalRequest:
    """
    A signal request from a bot.

    Attributes:
        request_id: Unique request ID
        bot_id: Bot making the request
        symbol: Trading symbol
        direction: Signal direction
        quantity: Intended quantity
        price: Target price
        timestamp: Request timestamp
        priority: Bot priority (higher = more important)
        reason: Signal reason/description
        expires_at: When this signal expires
    """
    request_id: str
    bot_id: str
    symbol: str
    direction: SignalDirection
    quantity: Decimal
    price: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0
    reason: str = ""
    expires_at: Optional[datetime] = None

    def is_expired(self) -> bool:
        """Check if signal has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def is_opening(self) -> bool:
        """Check if this is an opening signal."""
        return self.direction in [SignalDirection.LONG, SignalDirection.SHORT]

    def is_closing(self) -> bool:
        """Check if this is a closing signal."""
        return self.direction in [SignalDirection.CLOSE_LONG, SignalDirection.CLOSE_SHORT]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "bot_id": self.bot_id,
            "symbol": self.symbol,
            "direction": self.direction.value,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "reason": self.reason,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }


@dataclass
class ConflictRecord:
    """Record of a detected conflict."""
    conflict_id: str
    conflict_type: ConflictType
    symbol: str
    bot_a: str
    bot_b: str
    signal_a: SignalRequest
    signal_b: SignalRequest
    resolution: ConflictResolution
    blocked_bot: Optional[str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "symbol": self.symbol,
            "bot_a": self.bot_a,
            "bot_b": self.bot_b,
            "signal_a": self.signal_a.to_dict(),
            "signal_b": self.signal_b.to_dict(),
            "resolution": self.resolution.value,
            "blocked_bot": self.blocked_bot,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class CoordinationResult:
    """Result of signal coordination check."""
    approved: bool
    conflict_type: ConflictType
    conflicting_bot: Optional[str] = None
    conflicting_signal: Optional[SignalRequest] = None
    message: str = ""

    @classmethod
    def approved_result(cls, message: str = "Signal approved") -> "CoordinationResult":
        """Create an approved result."""
        return cls(approved=True, conflict_type=ConflictType.NONE, message=message)

    @classmethod
    def blocked_result(
        cls,
        conflict_type: ConflictType,
        conflicting_bot: str,
        conflicting_signal: SignalRequest,
        message: str,
    ) -> "CoordinationResult":
        """Create a blocked result."""
        return cls(
            approved=False,
            conflict_type=conflict_type,
            conflicting_bot=conflicting_bot,
            conflicting_signal=conflicting_signal,
            message=message,
        )


class SignalCoordinator:
    """
    Centralized signal coordinator for all trading bots.

    Prevents multi-strategy signal conflicts by:
    - Tracking active signals from all bots
    - Detecting conflicting intentions (long vs short on same symbol)
    - Applying conflict resolution strategies
    - Recording conflict history for analysis

    Example:
        >>> coordinator = SignalCoordinator(resolution=ConflictResolution.BLOCK_NEWER)
        >>>
        >>> # Bot A wants to go long
        >>> result = await coordinator.request_signal(
        ...     bot_id="grid_bot",
        ...     symbol="BTCUSDT",
        ...     direction=SignalDirection.LONG,
        ...     quantity=Decimal("0.01"),
        ...     price=Decimal("50000"),
        ... )
        >>> if result.approved:
        ...     # Execute the trade
        ...     pass
        >>>
        >>> # Bot B wants to go short on same symbol - conflict!
        >>> result = await coordinator.request_signal(
        ...     bot_id="bollinger_bot",
        ...     symbol="BTCUSDT",
        ...     direction=SignalDirection.SHORT,
        ...     ...
        ... )
        >>> # result.approved = False, result.conflict_type = OPPOSITE_DIRECTION
    """

    _instance: Optional["SignalCoordinator"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "SignalCoordinator":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        resolution: ConflictResolution = ConflictResolution.BLOCK_NEWER,
        signal_ttl_seconds: float = 60.0,
        bot_priorities: Optional[Dict[str, int]] = None,
        allowed_hedge_symbols: Optional[Set[str]] = None,
    ):
        """
        Initialize SignalCoordinator.

        Args:
            resolution: Default conflict resolution strategy
            signal_ttl_seconds: Time-to-live for active signals
            bot_priorities: Bot priorities (higher = more important)
            allowed_hedge_symbols: Symbols that allow hedging
        """
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._resolution = resolution
        self._signal_ttl = signal_ttl_seconds
        self._bot_priorities = bot_priorities or {}
        self._allowed_hedge_symbols = allowed_hedge_symbols or set()

        # Active signals: symbol -> {bot_id -> SignalRequest}
        self._active_signals: Dict[str, Dict[str, SignalRequest]] = {}
        self._signal_lock = asyncio.Lock()

        # Conflict history
        self._conflicts: List[ConflictRecord] = []
        self._max_conflicts: int = 500

        # Statistics
        self._stats = {
            "total_requests": 0,
            "approved": 0,
            "blocked": 0,
            "conflicts_detected": 0,
        }

        # Callbacks
        self._on_conflict: List[Callable[[ConflictRecord], None]] = []

        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        self._initialized = True
        logger.info(f"SignalCoordinator initialized with resolution={resolution.value}")

    @classmethod
    def get_instance(cls) -> Optional["SignalCoordinator"]:
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
        """Start the coordinator."""
        if self._running:
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SignalCoordinator started")

    async def stop(self) -> None:
        """Stop the coordinator."""
        if not self._running:
            return

        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("SignalCoordinator stopped")

    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired signals."""
        while self._running:
            try:
                await asyncio.sleep(self._signal_ttl / 2)
                await self._cleanup_expired_signals()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    async def _cleanup_expired_signals(self) -> None:
        """Remove expired signals."""
        async with self._signal_lock:
            now = datetime.now(timezone.utc)
            for symbol in list(self._active_signals.keys()):
                bot_signals = self._active_signals[symbol]
                expired_bots = [
                    bot_id for bot_id, signal in bot_signals.items()
                    if signal.is_expired()
                ]
                for bot_id in expired_bots:
                    del bot_signals[bot_id]
                    logger.debug(f"Expired signal cleaned: {bot_id} {symbol}")

                # Remove empty symbol entries
                if not bot_signals:
                    del self._active_signals[symbol]

    # =========================================================================
    # Signal Coordination
    # =========================================================================

    async def request_signal(
        self,
        bot_id: str,
        symbol: str,
        direction: SignalDirection,
        quantity: Decimal,
        price: Decimal,
        reason: str = "",
        priority: Optional[int] = None,
    ) -> CoordinationResult:
        """
        Request approval for a trading signal.

        Thread-safe: Uses signal lock to prevent race conditions.

        Args:
            bot_id: Bot making the request
            symbol: Trading symbol
            direction: Signal direction (long/short/close)
            quantity: Intended quantity
            price: Target price
            reason: Signal reason
            priority: Override bot priority

        Returns:
            CoordinationResult indicating if signal is approved
        """
        self._stats["total_requests"] += 1

        # Create signal request
        request_id = f"{bot_id}_{symbol}_{datetime.now(timezone.utc).timestamp()}"
        bot_priority = priority if priority is not None else self._bot_priorities.get(bot_id, 0)

        signal = SignalRequest(
            request_id=request_id,
            bot_id=bot_id,
            symbol=symbol,
            direction=direction,
            quantity=quantity,
            price=price,
            priority=bot_priority,
            reason=reason,
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=self._signal_ttl),
        )

        async with self._signal_lock:
            # Check for conflicts
            result = self._check_conflicts(signal)

            if result.approved:
                # Register the signal
                if symbol not in self._active_signals:
                    self._active_signals[symbol] = {}
                self._active_signals[symbol][bot_id] = signal
                self._stats["approved"] += 1
                logger.info(f"Signal approved: {bot_id} {direction.value} {symbol}")
            else:
                self._stats["blocked"] += 1
                logger.warning(
                    f"Signal blocked: {bot_id} {direction.value} {symbol} - "
                    f"{result.message}"
                )

            return result

    def _check_conflicts(self, new_signal: SignalRequest) -> CoordinationResult:
        """
        Check for conflicts with existing signals.

        Args:
            new_signal: The new signal to check

        Returns:
            CoordinationResult
        """
        symbol = new_signal.symbol

        # No existing signals for this symbol
        if symbol not in self._active_signals:
            return CoordinationResult.approved_result("No existing signals")

        # Check each existing signal
        for existing_bot_id, existing_signal in self._active_signals[symbol].items():
            # Skip expired signals
            if existing_signal.is_expired():
                continue

            # Skip self
            if existing_bot_id == new_signal.bot_id:
                continue

            # Detect conflict type
            conflict_type = self._detect_conflict_type(existing_signal, new_signal)

            if conflict_type != ConflictType.NONE:
                # Resolve conflict
                result = self._resolve_conflict(
                    existing_signal,
                    new_signal,
                    conflict_type,
                )

                if not result.approved:
                    # Record conflict
                    self._record_conflict(
                        conflict_type=conflict_type,
                        signal_a=existing_signal,
                        signal_b=new_signal,
                        blocked_bot=new_signal.bot_id,
                    )

                return result

        return CoordinationResult.approved_result("No conflicts detected")

    def _detect_conflict_type(
        self,
        existing: SignalRequest,
        new: SignalRequest,
    ) -> ConflictType:
        """Detect the type of conflict between two signals."""
        # Both opening in opposite directions
        if existing.is_opening() and new.is_opening():
            if (existing.direction == SignalDirection.LONG and
                new.direction == SignalDirection.SHORT):
                return ConflictType.OPPOSITE_DIRECTION
            if (existing.direction == SignalDirection.SHORT and
                new.direction == SignalDirection.LONG):
                return ConflictType.OPPOSITE_DIRECTION

        # One closing while other opening
        if existing.is_opening() and new.is_closing():
            return ConflictType.CLOSE_WHILE_OPENING
        if existing.is_closing() and new.is_opening():
            return ConflictType.CLOSE_WHILE_OPENING

        # Same direction overlap (might want to track for exposure limits)
        if existing.is_opening() and new.is_opening():
            if existing.direction == new.direction:
                return ConflictType.SAME_DIRECTION_OVERLAP

        return ConflictType.NONE

    def _resolve_conflict(
        self,
        existing: SignalRequest,
        new: SignalRequest,
        conflict_type: ConflictType,
    ) -> CoordinationResult:
        """
        Resolve a signal conflict based on resolution strategy.

        Args:
            existing: The existing signal
            new: The new signal
            conflict_type: Type of conflict

        Returns:
            CoordinationResult
        """
        symbol = new.symbol

        # Check if hedging is allowed for this symbol
        if (conflict_type == ConflictType.OPPOSITE_DIRECTION and
            symbol in self._allowed_hedge_symbols):
            return CoordinationResult.approved_result(
                f"Hedging allowed for {symbol}"
            )

        # Same direction overlap is usually OK (exposure checked elsewhere)
        if conflict_type == ConflictType.SAME_DIRECTION_OVERLAP:
            return CoordinationResult.approved_result(
                "Same direction, exposure checked separately"
            )

        # Apply resolution strategy for opposite direction conflicts
        if conflict_type == ConflictType.OPPOSITE_DIRECTION:
            return self._apply_resolution_strategy(existing, new, conflict_type)

        # Close while opening - usually allow the close
        if conflict_type == ConflictType.CLOSE_WHILE_OPENING:
            if new.is_closing():
                return CoordinationResult.approved_result(
                    "Close signal takes priority"
                )
            else:
                # New is opening while existing is closing - might block
                return CoordinationResult.blocked_result(
                    conflict_type=conflict_type,
                    conflicting_bot=existing.bot_id,
                    conflicting_signal=existing,
                    message=f"Another bot ({existing.bot_id}) is closing position",
                )

        return CoordinationResult.approved_result("No blocking conflict")

    def _apply_resolution_strategy(
        self,
        existing: SignalRequest,
        new: SignalRequest,
        conflict_type: ConflictType,
    ) -> CoordinationResult:
        """Apply the configured resolution strategy."""

        if self._resolution == ConflictResolution.WARN_ONLY:
            logger.warning(
                f"CONFLICT WARNING: {new.bot_id} wants {new.direction.value} "
                f"but {existing.bot_id} has {existing.direction.value} on {new.symbol}"
            )
            return CoordinationResult.approved_result(
                f"Warning logged - conflict with {existing.bot_id}"
            )

        elif self._resolution == ConflictResolution.ALLOW_HEDGE:
            return CoordinationResult.approved_result(
                f"Hedging allowed with {existing.bot_id}"
            )

        elif self._resolution == ConflictResolution.BLOCK_NEWER:
            return CoordinationResult.blocked_result(
                conflict_type=conflict_type,
                conflicting_bot=existing.bot_id,
                conflicting_signal=existing,
                message=f"Blocked: {existing.bot_id} already has {existing.direction.value} position",
            )

        elif self._resolution == ConflictResolution.BLOCK_SMALLER:
            if new.quantity < existing.quantity:
                return CoordinationResult.blocked_result(
                    conflict_type=conflict_type,
                    conflicting_bot=existing.bot_id,
                    conflicting_signal=existing,
                    message=f"Blocked: smaller quantity than {existing.bot_id}",
                )
            else:
                # Cancel existing signal, allow new
                self._cancel_signal(existing.bot_id, existing.symbol)
                return CoordinationResult.approved_result(
                    f"Approved: larger quantity, cancelled {existing.bot_id}"
                )

        elif self._resolution == ConflictResolution.PRIORITY_BASED:
            if new.priority < existing.priority:
                return CoordinationResult.blocked_result(
                    conflict_type=conflict_type,
                    conflicting_bot=existing.bot_id,
                    conflicting_signal=existing,
                    message=f"Blocked: lower priority than {existing.bot_id}",
                )
            elif new.priority > existing.priority:
                self._cancel_signal(existing.bot_id, existing.symbol)
                return CoordinationResult.approved_result(
                    f"Approved: higher priority, cancelled {existing.bot_id}"
                )
            else:
                # Same priority - block newer
                return CoordinationResult.blocked_result(
                    conflict_type=conflict_type,
                    conflicting_bot=existing.bot_id,
                    conflicting_signal=existing,
                    message=f"Blocked: same priority as {existing.bot_id}, newer blocked",
                )

        # Default: block
        return CoordinationResult.blocked_result(
            conflict_type=conflict_type,
            conflicting_bot=existing.bot_id,
            conflicting_signal=existing,
            message=f"Blocked by default resolution",
        )

    # =========================================================================
    # Signal Management
    # =========================================================================

    def _cancel_signal(self, bot_id: str, symbol: str) -> bool:
        """Cancel an active signal (internal use)."""
        if symbol in self._active_signals:
            if bot_id in self._active_signals[symbol]:
                del self._active_signals[symbol][bot_id]
                logger.info(f"Signal cancelled: {bot_id} {symbol}")
                return True
        return False

    async def cancel_signal(self, bot_id: str, symbol: str) -> bool:
        """
        Cancel an active signal (public API).

        Args:
            bot_id: Bot ID
            symbol: Trading symbol

        Returns:
            True if signal was cancelled
        """
        async with self._signal_lock:
            return self._cancel_signal(bot_id, symbol)

    async def complete_signal(self, bot_id: str, symbol: str) -> bool:
        """
        Mark a signal as completed (trade executed).

        Args:
            bot_id: Bot ID
            symbol: Trading symbol

        Returns:
            True if signal was found and completed
        """
        async with self._signal_lock:
            if symbol in self._active_signals:
                if bot_id in self._active_signals[symbol]:
                    signal = self._active_signals[symbol][bot_id]
                    del self._active_signals[symbol][bot_id]
                    logger.debug(f"Signal completed: {bot_id} {signal.direction.value} {symbol}")
                    return True
        return False

    # =========================================================================
    # Conflict Recording
    # =========================================================================

    def _record_conflict(
        self,
        conflict_type: ConflictType,
        signal_a: SignalRequest,
        signal_b: SignalRequest,
        blocked_bot: Optional[str],
    ) -> None:
        """Record a conflict for analysis."""
        conflict_id = f"conflict_{datetime.now(timezone.utc).timestamp()}"

        record = ConflictRecord(
            conflict_id=conflict_id,
            conflict_type=conflict_type,
            symbol=signal_a.symbol,
            bot_a=signal_a.bot_id,
            bot_b=signal_b.bot_id,
            signal_a=signal_a,
            signal_b=signal_b,
            resolution=self._resolution,
            blocked_bot=blocked_bot,
        )

        self._conflicts.append(record)
        self._stats["conflicts_detected"] += 1

        # Trim history
        if len(self._conflicts) > self._max_conflicts:
            self._conflicts = self._conflicts[-self._max_conflicts:]

        # Notify callbacks
        for callback in self._on_conflict:
            try:
                callback(record)
            except Exception as e:
                logger.error(f"Conflict callback error: {e}")

    def on_conflict(self, callback: Callable[[ConflictRecord], None]) -> None:
        """Register callback for conflict events."""
        self._on_conflict.append(callback)

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_active_signals(self, symbol: Optional[str] = None) -> Dict[str, Dict[str, SignalRequest]]:
        """Get active signals, optionally filtered by symbol."""
        if symbol:
            return {symbol: self._active_signals.get(symbol, {})}
        return self._active_signals.copy()

    def get_bot_signals(self, bot_id: str) -> List[SignalRequest]:
        """Get all active signals for a bot."""
        signals = []
        for symbol_signals in self._active_signals.values():
            if bot_id in symbol_signals:
                signals.append(symbol_signals[bot_id])
        return signals

    def has_conflicting_signal(self, bot_id: str, symbol: str, direction: SignalDirection) -> Optional[str]:
        """
        Check if there's a conflicting signal from another bot.

        Returns:
            Bot ID of conflicting bot, or None if no conflict
        """
        if symbol not in self._active_signals:
            return None

        opposite = SignalDirection.SHORT if direction == SignalDirection.LONG else SignalDirection.LONG

        for other_bot_id, signal in self._active_signals[symbol].items():
            if other_bot_id != bot_id and signal.direction == opposite:
                if not signal.is_expired():
                    return other_bot_id

        return None

    def get_conflict_history(
        self,
        symbol: Optional[str] = None,
        bot_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[ConflictRecord]:
        """Get conflict history."""
        conflicts = self._conflicts
        if symbol:
            conflicts = [c for c in conflicts if c.symbol == symbol]
        if bot_id:
            conflicts = [c for c in conflicts if c.bot_a == bot_id or c.bot_b == bot_id]
        return conflicts[-limit:]

    # =========================================================================
    # Configuration
    # =========================================================================

    def set_resolution(self, resolution: ConflictResolution) -> None:
        """Change the conflict resolution strategy."""
        self._resolution = resolution
        logger.info(f"Resolution strategy changed to: {resolution.value}")

    def set_bot_priority(self, bot_id: str, priority: int) -> None:
        """Set a bot's priority."""
        self._bot_priorities[bot_id] = priority
        logger.info(f"Bot priority set: {bot_id} = {priority}")

    def allow_hedge_symbol(self, symbol: str) -> None:
        """Allow hedging for a symbol."""
        self._allowed_hedge_symbols.add(symbol)
        logger.info(f"Hedging allowed for: {symbol}")

    def disallow_hedge_symbol(self, symbol: str) -> None:
        """Disallow hedging for a symbol."""
        self._allowed_hedge_symbols.discard(symbol)
        logger.info(f"Hedging disallowed for: {symbol}")

    # =========================================================================
    # Status
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get coordinator status."""
        return {
            "running": self._running,
            "resolution": self._resolution.value,
            "signal_ttl_seconds": self._signal_ttl,
            "active_signal_count": sum(
                len(signals) for signals in self._active_signals.values()
            ),
            "active_symbols": list(self._active_signals.keys()),
            "bot_priorities": self._bot_priorities,
            "allowed_hedge_symbols": list(self._allowed_hedge_symbols),
            "stats": self._stats.copy(),
            "recent_conflicts": [c.to_dict() for c in self._conflicts[-10:]],
        }

    def get_stats(self) -> Dict[str, int]:
        """Get coordination statistics."""
        return self._stats.copy()
