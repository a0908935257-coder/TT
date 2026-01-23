"""
Bot Notifier.

Handles notifications to trading bots about fund allocation changes.
Supports file-based and API-based notification methods.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class NotificationMessage:
    """
    Notification message for bot fund updates.

    Follows the specification format for bot notifications.

    Attributes:
        bot_id: Target bot identifier
        timestamp: When the notification was created
        event_type: Type of event (allocation_update, rebalance, etc.)
        new_allocation: Amount newly allocated
        total_allocation: Total allocation after update
        trigger: What triggered the allocation
        message: Human-readable message
    """

    bot_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = "allocation_update"
    new_allocation: Decimal = Decimal("0")
    total_allocation: Decimal = Decimal("0")
    trigger: str = "manual"
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format for JSON serialization.

        Returns:
            Dictionary matching the specification format
        """
        return {
            "bot_id": self.bot_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "data": {
                "new_allocation": float(self.new_allocation),
                "total_allocation": float(self.total_allocation),
                "trigger": self.trigger,
                "message": self.message,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """
        Convert to JSON string.

        Args:
            indent: JSON indentation level

        Returns:
            JSON string
        """
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NotificationMessage":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)

        inner_data = data.get("data", {})
        return cls(
            bot_id=data.get("bot_id", ""),
            timestamp=timestamp or datetime.now(timezone.utc),
            event_type=data.get("event_type", "allocation_update"),
            new_allocation=Decimal(str(inner_data.get("new_allocation", "0"))),
            total_allocation=Decimal(str(inner_data.get("total_allocation", "0"))),
            trigger=inner_data.get("trigger", "manual"),
            message=inner_data.get("message", ""),
        )


class NotifierProtocol(Protocol):
    """Protocol for notifier implementations."""

    async def notify(self, message: NotificationMessage) -> bool:
        """
        Send notification to bot.

        Args:
            message: Notification message

        Returns:
            True if successful
        """
        ...


class BaseNotifier(ABC):
    """Base class for notifier implementations."""

    def __init__(self, retry_count: int = 3, retry_interval: int = 60):
        """
        Initialize notifier.

        Args:
            retry_count: Number of retry attempts on failure
            retry_interval: Seconds between retries
        """
        self._retry_count = retry_count
        self._retry_interval = retry_interval
        self._pending_notifications: List[NotificationMessage] = []

    @abstractmethod
    async def _send(self, message: NotificationMessage) -> bool:
        """
        Internal send implementation.

        Args:
            message: Notification message

        Returns:
            True if successful
        """
        pass

    async def notify(self, message: NotificationMessage) -> bool:
        """
        Send notification with retry logic.

        Args:
            message: Notification message

        Returns:
            True if successful
        """
        for attempt in range(self._retry_count):
            try:
                success = await self._send(message)
                if success:
                    logger.info(f"Notification sent to {message.bot_id}")
                    return True

                logger.warning(
                    f"Notification to {message.bot_id} failed "
                    f"(attempt {attempt + 1}/{self._retry_count})"
                )

            except Exception as e:
                logger.error(
                    f"Notification error for {message.bot_id}: {e} "
                    f"(attempt {attempt + 1}/{self._retry_count})"
                )

            if attempt < self._retry_count - 1:
                await asyncio.sleep(self._retry_interval)

        # Add to pending for later retry
        self._pending_notifications.append(message)
        logger.error(
            f"Notification to {message.bot_id} failed after {self._retry_count} attempts"
        )
        return False

    async def retry_pending(self) -> int:
        """
        Retry pending notifications.

        Returns:
            Number of successful retries
        """
        if not self._pending_notifications:
            return 0

        successful = 0
        still_pending = []

        for message in self._pending_notifications:
            try:
                if await self._send(message):
                    successful += 1
                    logger.info(f"Pending notification sent to {message.bot_id}")
                else:
                    still_pending.append(message)
            except Exception as e:
                logger.error(f"Retry failed for {message.bot_id}: {e}")
                still_pending.append(message)

        self._pending_notifications = still_pending
        return successful

    @property
    def pending_count(self) -> int:
        """Get count of pending notifications."""
        return len(self._pending_notifications)


class FileNotifier(BaseNotifier):
    """
    File-based notifier.

    Writes JSON notification files for bots to read.
    Each bot has its own notification file at a configured path.
    """

    def __init__(
        self,
        default_path: str = "data/fund_notifications",
        retry_count: int = 3,
        retry_interval: int = 60,
    ):
        """
        Initialize file notifier.

        Args:
            default_path: Default directory for notification files
            retry_count: Number of retry attempts
            retry_interval: Seconds between retries
        """
        super().__init__(retry_count, retry_interval)
        self._default_path = Path(default_path)
        self._default_path.mkdir(parents=True, exist_ok=True)

        # Bot-specific file paths
        self._bot_paths: Dict[str, Path] = {}

    def set_bot_path(self, bot_id: str, path: str) -> None:
        """
        Set notification file path for a specific bot.

        Args:
            bot_id: Bot identifier
            path: File path for this bot's notifications
        """
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        self._bot_paths[bot_id] = file_path
        logger.debug(f"Set notification path for {bot_id}: {path}")

    def get_bot_path(self, bot_id: str) -> Path:
        """
        Get notification file path for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            File path for the bot's notifications
        """
        if bot_id in self._bot_paths:
            return self._bot_paths[bot_id]
        return self._default_path / f"{bot_id}.json"

    async def _send(self, message: NotificationMessage) -> bool:
        """
        Write notification to file.

        Args:
            message: Notification message

        Returns:
            True if file was written successfully
        """
        try:
            file_path = self.get_bot_path(message.bot_id)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write notification to file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(message.to_json())

            logger.debug(f"Wrote notification to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to write notification file: {e}")
            return False

    def read_notification(self, bot_id: str) -> Optional[NotificationMessage]:
        """
        Read notification file for a bot.

        Useful for bots to read their notifications.

        Args:
            bot_id: Bot identifier

        Returns:
            NotificationMessage if file exists and is valid
        """
        try:
            file_path = self.get_bot_path(bot_id)
            if not file_path.exists():
                return None

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return NotificationMessage.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to read notification for {bot_id}: {e}")
            return None

    def clear_notification(self, bot_id: str) -> bool:
        """
        Clear notification file for a bot.

        Called by bot after processing notification.

        Args:
            bot_id: Bot identifier

        Returns:
            True if file was cleared
        """
        try:
            file_path = self.get_bot_path(bot_id)
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Cleared notification for {bot_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to clear notification for {bot_id}: {e}")
            return False


class BotNotifier:
    """
    Main bot notifier that manages multiple notification methods.

    Routes notifications to appropriate notifier based on bot configuration.

    Example:
        >>> notifier = BotNotifier()
        >>> notifier.register_bot("grid_btc", "file", "data/notifications/grid_btc.json")
        >>> await notifier.notify_allocation("grid_btc", Decimal("100"), Decimal("500"))
    """

    def __init__(
        self,
        default_notification_path: str = "data/fund_notifications",
        retry_count: int = 3,
        retry_interval: int = 60,
    ):
        """
        Initialize bot notifier.

        Args:
            default_notification_path: Default path for file notifications
            retry_count: Number of retry attempts
            retry_interval: Seconds between retries
        """
        self._retry_count = retry_count
        self._retry_interval = retry_interval

        # Initialize notifiers
        self._file_notifier = FileNotifier(
            default_path=default_notification_path,
            retry_count=retry_count,
            retry_interval=retry_interval,
        )

        # Bot notification method mapping
        self._bot_methods: Dict[str, str] = {}  # bot_id -> method

    def register_bot(
        self,
        bot_id: str,
        method: str = "file",
        target: Optional[str] = None,
    ) -> None:
        """
        Register a bot with its notification method.

        Args:
            bot_id: Bot identifier
            method: Notification method (file, api, none)
            target: Target path or endpoint
        """
        self._bot_methods[bot_id] = method

        if method == "file" and target:
            self._file_notifier.set_bot_path(bot_id, target)

        logger.info(f"Registered bot {bot_id} with {method} notification")

    def unregister_bot(self, bot_id: str) -> None:
        """
        Unregister a bot.

        Args:
            bot_id: Bot identifier
        """
        self._bot_methods.pop(bot_id, None)
        logger.info(f"Unregistered bot {bot_id}")

    async def notify_allocation(
        self,
        bot_id: str,
        new_allocation: Decimal,
        total_allocation: Decimal,
        trigger: str = "manual",
        message: str = "",
    ) -> bool:
        """
        Send allocation notification to a bot.

        Args:
            bot_id: Bot identifier
            new_allocation: Amount newly allocated
            total_allocation: Total allocation after update
            trigger: What triggered the allocation
            message: Optional human-readable message

        Returns:
            True if notification was successful
        """
        method = self._bot_methods.get(bot_id, "file")

        if method == "none":
            logger.debug(f"Notifications disabled for {bot_id}")
            return True

        notification = NotificationMessage(
            bot_id=bot_id,
            event_type="allocation_update",
            new_allocation=new_allocation,
            total_allocation=total_allocation,
            trigger=trigger,
            message=message or f"Fund allocation updated: +{new_allocation} USDT",
        )

        if method == "file":
            return await self._file_notifier.notify(notification)
        elif method == "api":
            # TODO: Implement API notifier when needed
            logger.warning(f"API notification not implemented for {bot_id}")
            return False
        else:
            logger.warning(f"Unknown notification method {method} for {bot_id}")
            return False

    async def notify_fund_allocated(
        self,
        bot_id: str,
        amount: Decimal,
        total_allocated: Decimal,
    ) -> None:
        """
        Notify about fund allocation (protocol method).

        Args:
            bot_id: Bot identifier
            amount: Amount allocated
            total_allocated: Total allocated after this
        """
        await self.notify_allocation(
            bot_id=bot_id,
            new_allocation=amount,
            total_allocation=total_allocated,
            trigger="dispatch",
        )

    async def retry_pending(self) -> int:
        """
        Retry all pending notifications.

        Returns:
            Number of successful retries
        """
        return await self._file_notifier.retry_pending()

    @property
    def pending_count(self) -> int:
        """Get total pending notification count."""
        return self._file_notifier.pending_count

    def get_status(self) -> Dict[str, Any]:
        """
        Get notifier status.

        Returns:
            Dictionary with status information
        """
        return {
            "registered_bots": len(self._bot_methods),
            "pending_notifications": self.pending_count,
            "bot_methods": dict(self._bot_methods),
        }
