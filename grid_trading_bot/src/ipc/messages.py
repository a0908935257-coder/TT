"""
IPC Message Models.

Defines message formats for inter-process communication between
Master and Bot processes via Redis Pub/Sub.
"""

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar("T", bound="BaseMessage")


class CommandType(Enum):
    """Types of commands Master can send to Bot."""

    START = "start"
    STOP = "stop"
    PAUSE = "pause"
    RESUME = "resume"
    STATUS = "status"
    SHUTDOWN = "shutdown"
    FUND_UPDATE = "fund_update"  # Update bot capital allocation


class EventType(Enum):
    """Types of events Bot can send to Master."""

    STARTED = "started"
    STOPPED = "stopped"
    ERROR = "error"
    TRADE = "trade"
    ALERT = "alert"


@dataclass
class BaseMessage:
    """Base class for all IPC messages with serialization support."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            elif isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result

    def to_json(self) -> str:
        """Serialize message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create message from dictionary. Subclasses should override."""
        raise NotImplementedError("Subclasses must implement from_dict")

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        """Deserialize message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)


@dataclass
class Command(BaseMessage):
    """
    Command message from Master to Bot.

    Used to control bot lifecycle and query status.

    Attributes:
        id: Unique command identifier (UUID)
        type: Command type (start, stop, pause, resume, status, shutdown)
        params: Optional parameters for the command
        timestamp: When the command was created
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: CommandType = CommandType.STATUS
    params: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Command":
        """Create Command from dictionary."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            type=CommandType(data["type"]) if isinstance(data.get("type"), str) else data.get("type", CommandType.STATUS),
            params=data.get("params", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
        )


@dataclass
class Response(BaseMessage):
    """
    Response message from Bot to Master.

    Sent in reply to a Command.

    Attributes:
        command_id: ID of the command being responded to
        success: Whether the command executed successfully
        data: Response data (status info, etc.)
        error: Error message if success is False
        timestamp: When the response was created
    """

    command_id: str = ""
    success: bool = True
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Response":
        """Create Response from dictionary."""
        return cls(
            command_id=data.get("command_id", ""),
            success=data.get("success", True),
            data=data.get("data", {}),
            error=data.get("error"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
        )

    @classmethod
    def success_response(cls, command_id: str, data: Optional[Dict[str, Any]] = None) -> "Response":
        """Create a successful response."""
        return cls(
            command_id=command_id,
            success=True,
            data=data or {},
        )

    @classmethod
    def error_response(cls, command_id: str, error: str) -> "Response":
        """Create an error response."""
        return cls(
            command_id=command_id,
            success=False,
            error=error,
        )


@dataclass
class Heartbeat(BaseMessage):
    """
    Heartbeat message from Bot to Master.

    Sent periodically to indicate bot is alive and report metrics.

    Attributes:
        bot_id: Unique bot identifier
        state: Current bot state (registered, running, paused, etc.)
        pid: Process ID of the bot
        metrics: Performance metrics (trades, profit, etc.)
        timestamp: When the heartbeat was created
    """

    bot_id: str = ""
    state: str = "unknown"
    pid: int = field(default_factory=lambda: os.getpid())
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Heartbeat":
        """Create Heartbeat from dictionary."""
        return cls(
            bot_id=data.get("bot_id", ""),
            state=data.get("state", "unknown"),
            pid=data.get("pid", 0),
            metrics=data.get("metrics", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
        )


@dataclass
class Event(BaseMessage):
    """
    Event message from Bot to Master.

    Used to notify Master of significant events.

    Attributes:
        bot_id: Unique bot identifier
        type: Event type (started, stopped, error, trade, alert)
        data: Event-specific data
        timestamp: When the event occurred
    """

    bot_id: str = ""
    type: EventType = EventType.ALERT
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Event":
        """Create Event from dictionary."""
        return cls(
            bot_id=data.get("bot_id", ""),
            type=EventType(data["type"]) if isinstance(data.get("type"), str) else data.get("type", EventType.ALERT),
            data=data.get("data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"]) if isinstance(data.get("timestamp"), str) else data.get("timestamp", datetime.now(timezone.utc)),
        )

    @classmethod
    def started(cls, bot_id: str, data: Optional[Dict[str, Any]] = None) -> "Event":
        """Create a STARTED event."""
        return cls(bot_id=bot_id, type=EventType.STARTED, data=data or {})

    @classmethod
    def stopped(cls, bot_id: str, reason: str = "") -> "Event":
        """Create a STOPPED event."""
        return cls(bot_id=bot_id, type=EventType.STOPPED, data={"reason": reason})

    @classmethod
    def error(cls, bot_id: str, error: str, details: Optional[Dict[str, Any]] = None) -> "Event":
        """Create an ERROR event."""
        return cls(
            bot_id=bot_id,
            type=EventType.ERROR,
            data={"error": error, **(details or {})},
        )

    @classmethod
    def trade(cls, bot_id: str, trade_data: Dict[str, Any]) -> "Event":
        """Create a TRADE event."""
        return cls(bot_id=bot_id, type=EventType.TRADE, data=trade_data)

    @classmethod
    def alert(cls, bot_id: str, message: str, level: str = "info") -> "Event":
        """Create an ALERT event."""
        return cls(
            bot_id=bot_id,
            type=EventType.ALERT,
            data={"message": message, "level": level},
        )
