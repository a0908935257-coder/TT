"""
Base Notification Classes.

Provides base classes and enums for the notification system.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Optional


class NotificationLevel(str, Enum):
    """Notification severity level."""

    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class BaseNotifier(ABC):
    """
    Abstract base class for notification providers.

    Provides a common interface for sending notifications
    through various channels (Discord, Telegram, Email, etc.).

    Example:
        >>> class MyNotifier(BaseNotifier):
        ...     async def send(self, message, level):
        ...         # Implementation
        ...         pass
    """

    @abstractmethod
    async def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> bool:
        """
        Send a plain text notification.

        Args:
            message: Message content
            level: Notification severity level

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def send_embed(self, embed: dict[str, Any]) -> bool:
        """
        Send a rich text/embed notification.

        Args:
            embed: Embed data dictionary

        Returns:
            True if sent successfully
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the notifier and release resources."""
        pass

    async def __aenter__(self) -> "BaseNotifier":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # Convenience methods with default implementations

    async def debug(self, message: str) -> bool:
        """Send debug notification."""
        return await self.send(message, NotificationLevel.DEBUG)

    async def info(self, message: str) -> bool:
        """Send info notification."""
        return await self.send(message, NotificationLevel.INFO)

    async def success(self, message: str) -> bool:
        """Send success notification."""
        return await self.send(message, NotificationLevel.SUCCESS)

    async def warning(self, message: str) -> bool:
        """Send warning notification."""
        return await self.send(message, NotificationLevel.WARNING)

    async def error(self, message: str) -> bool:
        """Send error notification."""
        return await self.send(message, NotificationLevel.ERROR)

    async def critical(self, message: str) -> bool:
        """Send critical notification."""
        return await self.send(message, NotificationLevel.CRITICAL)
