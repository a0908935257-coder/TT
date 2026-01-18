"""
Bot Registry Data Models.

Defines data structures for bot registration, state tracking,
and lifecycle management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
import uuid


class BotType(str, Enum):
    """Bot type enumeration."""

    GRID = "grid"                    # Grid trading bot
    DCA = "dca"                      # Dollar-cost averaging bot
    TRAILING_STOP = "trailing_stop"  # Trailing stop bot
    SIGNAL = "signal"                # Signal following bot


class BotState(str, Enum):
    """
    Bot operational state.

    State transitions:
    REGISTERED -> INITIALIZING (start)
    INITIALIZING -> RUNNING (init complete) | ERROR (init failed)
    RUNNING -> PAUSED (pause) | STOPPING (stop) | ERROR (runtime error)
    PAUSED -> RUNNING (resume) | STOPPING (stop)
    STOPPING -> STOPPED (stop complete)
    STOPPED -> INITIALIZING (restart)
    ERROR -> STOPPED (acknowledge)
    """

    REGISTERED = "registered"        # Registered but not started
    INITIALIZING = "initializing"    # Bot is initializing
    RUNNING = "running"              # Bot is running normally
    PAUSED = "paused"                # Bot is paused (can resume)
    STOPPING = "stopping"            # Bot is stopping
    STOPPED = "stopped"              # Bot is stopped
    ERROR = "error"                  # Bot encountered error


class MarketType(str, Enum):
    """Market type enumeration."""

    SPOT = "spot"
    FUTURES = "futures"


# Valid state transitions mapping
VALID_STATE_TRANSITIONS: dict[BotState, list[BotState]] = {
    BotState.REGISTERED: [BotState.INITIALIZING],
    BotState.INITIALIZING: [BotState.RUNNING, BotState.ERROR],
    BotState.RUNNING: [BotState.PAUSED, BotState.STOPPING, BotState.ERROR],
    BotState.PAUSED: [BotState.RUNNING, BotState.STOPPING],
    BotState.STOPPING: [BotState.STOPPED],
    BotState.STOPPED: [BotState.INITIALIZING],
    BotState.ERROR: [BotState.STOPPED],
}


@dataclass
class BotInfo:
    """
    Bot registration information.

    Holds metadata and state for a registered bot.
    """

    bot_id: str
    bot_type: BotType
    symbol: str
    market_type: MarketType
    state: BotState = BotState.REGISTERED
    config: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    stopped_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "bot_type": self.bot_type.value,
            "symbol": self.symbol,
            "market_type": self.market_type.value,
            "state": self.state.value,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "stopped_at": self.stopped_at.isoformat() if self.stopped_at else None,
            "last_heartbeat": self.last_heartbeat.isoformat() if self.last_heartbeat else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotInfo":
        """Create from dictionary."""
        return cls(
            bot_id=data["bot_id"],
            bot_type=BotType(data["bot_type"]),
            symbol=data["symbol"],
            market_type=MarketType(data["market_type"]),
            state=BotState(data.get("state", "registered")),
            config=data.get("config", {}),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(timezone.utc),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            stopped_at=datetime.fromisoformat(data["stopped_at"]) if data.get("stopped_at") else None,
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]) if data.get("last_heartbeat") else None,
            error_message=data.get("error_message"),
        )


@dataclass
class RegistryEvent:
    """
    Registry event record.

    Tracks state changes and lifecycle events for bots.
    """

    event_id: str
    bot_id: str
    event_type: str
    old_state: Optional[BotState]
    new_state: BotState
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def create(
        cls,
        bot_id: str,
        event_type: str,
        old_state: Optional[BotState],
        new_state: BotState,
        message: str = "",
    ) -> "RegistryEvent":
        """Create a new event with generated ID."""
        return cls(
            event_id=str(uuid.uuid4()),
            bot_id=bot_id,
            event_type=event_type,
            old_state=old_state,
            new_state=new_state,
            message=message,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_id": self.event_id,
            "bot_id": self.bot_id,
            "event_type": self.event_type,
            "old_state": self.old_state.value if self.old_state else None,
            "new_state": self.new_state.value,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
        }


class InvalidStateTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""

    def __init__(self, current: BotState, target: BotState, bot_id: str = ""):
        self.current = current
        self.target = target
        self.bot_id = bot_id
        super().__init__(
            f"Invalid state transition for bot '{bot_id}': "
            f"{current.value} -> {target.value}"
        )


class BotNotFoundError(Exception):
    """Raised when a bot is not found in registry."""

    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        super().__init__(f"Bot not found: {bot_id}")


class BotAlreadyExistsError(Exception):
    """Raised when trying to register a bot that already exists."""

    def __init__(self, bot_id: str):
        self.bot_id = bot_id
        super().__init__(f"Bot already exists: {bot_id}")
