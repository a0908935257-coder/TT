"""
Bot Registry.

Centralized registration and lifecycle management for trading bots.
Implements singleton pattern for global access.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional, Protocol, TypeVar

from src.core import get_logger

from .models import (
    BotAlreadyExistsError,
    BotInfo,
    BotNotFoundError,
    BotState,
    BotType,
    InvalidStateTransitionError,
    MarketType,
    RegistryEvent,
    VALID_STATE_TRANSITIONS,
)

logger = get_logger(__name__)

# Type variable for BaseBot (to avoid circular import)
T = TypeVar("T")


class BaseBotProtocol(Protocol):
    """Protocol for bot instances."""

    @property
    def bot_id(self) -> str: ...

    async def stop(self) -> bool: ...


class DatabaseManagerProtocol(Protocol):
    """Protocol for database manager."""

    async def get_all_bots(self) -> list[dict[str, Any]]: ...
    async def upsert_bot(self, data: dict[str, Any]) -> None: ...
    async def delete_bot(self, bot_id: str) -> bool: ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def notify_bot_registered(self, bot_id: str, bot_type: str, symbol: str) -> None: ...
    async def notify_bot_state_changed(self, bot_id: str, old_state: str, new_state: str, message: str) -> None: ...


class BotRegistry:
    """
    Bot Registration Center (Singleton).

    Manages registration, state tracking, and lifecycle of trading bots.

    Example:
        >>> registry = BotRegistry.get_instance()
        >>> info = await registry.register("bot_001", BotType.GRID, config)
        >>> await registry.update_state("bot_001", BotState.INITIALIZING)
        >>> bot = registry.get("bot_001")
    """

    _instance: Optional["BotRegistry"] = None
    _lock: Optional[asyncio.Lock] = None  # Lazy-initialized in event loop context

    def __new__(cls, *args: Any, **kwargs: Any) -> "BotRegistry":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        db_manager: Optional[DatabaseManagerProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize BotRegistry.

        Args:
            db_manager: Optional database manager for persistence
            notifier: Optional notification manager
        """
        # Only initialize once
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._bots: dict[str, BotInfo] = {}
        self._instances: dict[str, BaseBotProtocol] = {}
        self._events: list[RegistryEvent] = []
        self._async_lock = asyncio.Lock()
        self._db = db_manager
        self._notifier = notifier
        self._initialized = True

    @classmethod
    def get_instance(
        cls,
        db_manager: Optional[DatabaseManagerProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
    ) -> "BotRegistry":
        """
        Get singleton instance.

        Args:
            db_manager: Optional database manager (only used on first call)
            notifier: Optional notification manager (only used on first call)

        Returns:
            BotRegistry singleton instance
        """
        if cls._instance is None:
            cls._instance = cls(db_manager, notifier)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    # =========================================================================
    # Core Registration Methods
    # =========================================================================

    async def register(
        self,
        bot_id: str,
        bot_type: BotType,
        config: dict[str, Any],
        symbol: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> BotInfo:
        """
        Register a new bot.

        Args:
            bot_id: Unique bot identifier
            bot_type: Type of bot (grid, dca, etc.)
            config: Bot configuration dictionary
            symbol: Trading symbol (can be from config)
            market_type: Market type (spot/futures)

        Returns:
            BotInfo for registered bot

        Raises:
            BotAlreadyExistsError: If bot_id already exists
            ValueError: If config is invalid
        """
        async with self._async_lock:
            # Validate bot_id uniqueness
            if bot_id in self._bots:
                raise BotAlreadyExistsError(bot_id)

            # Validate and extract config
            if not config:
                raise ValueError("Config cannot be empty")

            # Get symbol from config if not provided
            if symbol is None:
                symbol = config.get("symbol")
                if not symbol:
                    raise ValueError("Symbol is required (in config or as parameter)")

            # Get market_type from config if available
            if "market_type" in config:
                market_type = MarketType(config["market_type"].upper())

            # Create BotInfo
            bot_info = BotInfo(
                bot_id=bot_id,
                bot_type=bot_type,
                symbol=symbol,
                market_type=market_type,
                state=BotState.REGISTERED,
                config=config,
            )

            # Store in registry
            self._bots[bot_id] = bot_info

            # Record event
            event = self._record_event(
                bot_id=bot_id,
                event_type="registered",
                old_state=None,
                new_state=BotState.REGISTERED,
                message=f"Bot registered: {bot_type.value} for {symbol}",
            )

            # Persist to database
            if self._db:
                try:
                    await self._db.upsert_bot(bot_info.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to persist bot registration: {e}")

            # Send notification
            if self._notifier:
                try:
                    await self._notifier.notify_bot_registered(
                        bot_id=bot_id,
                        bot_type=bot_type.value,
                        symbol=symbol,
                    )
                except Exception as e:
                    logger.warning(f"Failed to send registration notification: {e}")

            logger.info(f"Bot registered: {bot_id} ({bot_type.value}) for {symbol}")
            return bot_info

    async def unregister(self, bot_id: str) -> bool:
        """
        Unregister a bot.

        Bot must be in STOPPED or REGISTERED state.
        If instance exists, it will be stopped first.

        Args:
            bot_id: Bot identifier to unregister

        Returns:
            True if unregistered successfully

        Raises:
            BotNotFoundError: If bot not found
            ValueError: If bot is not in valid state for unregistration
        """
        async with self._async_lock:
            # Check bot exists
            if bot_id not in self._bots:
                raise BotNotFoundError(bot_id)

            bot_info = self._bots[bot_id]

            # Check state allows unregistration (also allow ERROR state)
            if bot_info.state not in (BotState.STOPPED, BotState.REGISTERED, BotState.ERROR):
                raise ValueError(
                    f"Cannot unregister bot in state {bot_info.state.value}. "
                    "Bot must be STOPPED, REGISTERED, or ERROR."
                )

            # Stop instance if exists
            if bot_id in self._instances:
                instance = self._instances[bot_id]
                try:
                    await instance.stop()
                except Exception as e:
                    logger.warning(f"Error stopping instance during unregister: {e}")
                del self._instances[bot_id]

            # Record event before removal
            self._record_event(
                bot_id=bot_id,
                event_type="unregistered",
                old_state=bot_info.state,
                new_state=BotState.STOPPED,
                message="Bot unregistered",
            )

            # Remove from registry
            del self._bots[bot_id]

            # Remove from database
            if self._db:
                try:
                    await self._db.delete_bot(bot_id)
                except Exception as e:
                    logger.warning(f"Failed to delete bot from database: {e}")

            logger.info(f"Bot unregistered: {bot_id}")
            return True

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get(self, bot_id: str) -> Optional[BotInfo]:
        """
        Get bot information by ID.

        Args:
            bot_id: Bot identifier

        Returns:
            BotInfo if found, None otherwise
        """
        return self._bots.get(bot_id)

    def get_all(self) -> list[BotInfo]:
        """
        Get all registered bots.

        Returns:
            List of all BotInfo
        """
        return list(self._bots.values())

    def get_by_state(self, state: BotState) -> list[BotInfo]:
        """
        Get bots filtered by state.

        Args:
            state: BotState to filter by

        Returns:
            List of BotInfo matching the state
        """
        return [info for info in self._bots.values() if info.state == state]

    def get_by_type(self, bot_type: BotType) -> list[BotInfo]:
        """
        Get bots filtered by type.

        Args:
            bot_type: BotType to filter by

        Returns:
            List of BotInfo matching the type
        """
        return [info for info in self._bots.values() if info.bot_type == bot_type]

    def get_by_symbol(self, symbol: str) -> list[BotInfo]:
        """
        Get bots filtered by trading symbol.

        Args:
            symbol: Trading symbol to filter by

        Returns:
            List of BotInfo matching the symbol
        """
        return [info for info in self._bots.values() if info.symbol == symbol]

    # =========================================================================
    # State Management
    # =========================================================================

    async def update_state(
        self,
        bot_id: str,
        new_state: BotState,
        message: str = "",
    ) -> None:
        """
        Update bot state.

        Validates state transition and updates timestamps accordingly.

        Args:
            bot_id: Bot identifier
            new_state: Target state
            message: Optional message (used for ERROR state)

        Raises:
            BotNotFoundError: If bot not found
            InvalidStateTransitionError: If transition is not valid
        """
        async with self._async_lock:
            # Get bot info
            if bot_id not in self._bots:
                raise BotNotFoundError(bot_id)

            bot_info = self._bots[bot_id]
            old_state = bot_info.state

            # Validate transition
            if not self.validate_transition(old_state, new_state):
                raise InvalidStateTransitionError(old_state, new_state, bot_id)

            # Update state
            bot_info.state = new_state

            # Update timestamps based on new state
            now = datetime.now(timezone.utc)
            if new_state == BotState.RUNNING:
                bot_info.started_at = now
            elif new_state == BotState.STOPPED:
                bot_info.stopped_at = now
            elif new_state == BotState.ERROR:
                bot_info.error_message = message

            # Update heartbeat
            bot_info.last_heartbeat = now

            # Record event
            self._record_event(
                bot_id=bot_id,
                event_type="state_changed",
                old_state=old_state,
                new_state=new_state,
                message=message,
            )

            # Persist to database
            if self._db:
                try:
                    await self._db.upsert_bot(bot_info.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to persist state change: {e}")

            # Send notification
            if self._notifier:
                try:
                    await self._notifier.notify_bot_state_changed(
                        bot_id=bot_id,
                        old_state=old_state.value,
                        new_state=new_state.value,
                        message=message,
                    )
                except Exception as e:
                    logger.warning(f"Failed to send state change notification: {e}")

            logger.info(f"Bot {bot_id} state changed: {old_state.value} -> {new_state.value}")

    def validate_transition(self, current: BotState, target: BotState) -> bool:
        """
        Validate if state transition is allowed.

        Args:
            current: Current state
            target: Target state

        Returns:
            True if transition is valid
        """
        return target in VALID_STATE_TRANSITIONS.get(current, [])

    async def update_heartbeat(self, bot_id: str) -> None:
        """
        Update bot heartbeat timestamp.

        Args:
            bot_id: Bot identifier
        """
        if bot_id in self._bots:
            self._bots[bot_id].last_heartbeat = datetime.now(timezone.utc)

    # =========================================================================
    # Instance Management
    # =========================================================================

    def bind_instance(self, bot_id: str, instance: BaseBotProtocol) -> None:
        """
        Bind a bot instance to the registry.

        Args:
            bot_id: Bot identifier
            instance: Bot instance

        Raises:
            BotNotFoundError: If bot not registered
        """
        if bot_id not in self._bots:
            raise BotNotFoundError(bot_id)

        self._instances[bot_id] = instance
        logger.debug(f"Instance bound for bot: {bot_id}")

    def unbind_instance(self, bot_id: str) -> None:
        """
        Unbind a bot instance from the registry.

        Args:
            bot_id: Bot identifier
        """
        if bot_id in self._instances:
            del self._instances[bot_id]
            logger.debug(f"Instance unbound for bot: {bot_id}")

    def get_bot_instance(self, bot_id: str) -> Optional[BaseBotProtocol]:
        """
        Get bot instance by ID.

        Args:
            bot_id: Bot identifier

        Returns:
            Bot instance if bound, None otherwise
        """
        return self._instances.get(bot_id)

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_summary(self) -> dict[str, Any]:
        """
        Get registry summary statistics.

        Returns:
            Dict with total_bots, by_state, by_type, by_symbol counts
        """
        by_state: dict[str, int] = defaultdict(int)
        by_type: dict[str, int] = defaultdict(int)
        by_symbol: dict[str, int] = defaultdict(int)

        for bot_info in self._bots.values():
            by_state[bot_info.state.value] += 1
            by_type[bot_info.bot_type.value] += 1
            by_symbol[bot_info.symbol] += 1

        return {
            "total_bots": len(self._bots),
            "by_state": dict(by_state),
            "by_type": dict(by_type),
            "by_symbol": dict(by_symbol),
        }

    def get_events(
        self,
        bot_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[RegistryEvent]:
        """
        Get registry events.

        Args:
            bot_id: Optional bot ID to filter by
            limit: Maximum number of events to return

        Returns:
            List of RegistryEvent (newest first)
        """
        events = self._events
        if bot_id:
            events = [e for e in events if e.bot_id == bot_id]

        # Return newest first, limited
        return list(reversed(events[-limit:]))

    # =========================================================================
    # Persistence
    # =========================================================================

    async def load_from_db(self) -> int:
        """
        Load registered bots from database.

        Returns:
            Number of bots loaded
        """
        if not self._db:
            logger.warning("No database manager configured, cannot load bots")
            return 0

        try:
            records = await self._db.get_all_bots()
            loaded = 0

            async with self._async_lock:
                for record in records:
                    try:
                        bot_info = BotInfo.from_dict(record)
                        self._bots[bot_info.bot_id] = bot_info
                        loaded += 1
                    except Exception as e:
                        logger.warning(f"Failed to load bot record: {e}")

            logger.info(f"Loaded {loaded} bots from database")
            return loaded

        except Exception as e:
            logger.error(f"Failed to load bots from database: {e}")
            return 0

    async def _save_to_db(self, bot_info: BotInfo) -> None:
        """
        Save bot info to database.

        Args:
            bot_info: BotInfo to save
        """
        if not self._db:
            return

        try:
            await self._db.upsert_bot(bot_info.to_dict())
        except Exception as e:
            logger.warning(f"Failed to save bot to database: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _record_event(
        self,
        bot_id: str,
        event_type: str,
        old_state: Optional[BotState],
        new_state: BotState,
        message: str = "",
    ) -> RegistryEvent:
        """
        Record a registry event.

        Args:
            bot_id: Bot identifier
            event_type: Type of event
            old_state: Previous state
            new_state: New state
            message: Event message

        Returns:
            Created RegistryEvent
        """
        event = RegistryEvent.create(
            bot_id=bot_id,
            event_type=event_type,
            old_state=old_state,
            new_state=new_state,
            message=message,
        )
        self._events.append(event)

        # Keep event list bounded (max 10000 events)
        if len(self._events) > 10000:
            self._events = self._events[-5000:]

        return event

    @property
    def bot_count(self) -> int:
        """Get total number of registered bots."""
        return len(self._bots)

    @property
    def running_count(self) -> int:
        """Get number of running bots."""
        return len(self.get_by_state(BotState.RUNNING))

    def __contains__(self, bot_id: str) -> bool:
        """Check if bot is registered."""
        return bot_id in self._bots

    def __len__(self) -> int:
        """Get number of registered bots."""
        return len(self._bots)
