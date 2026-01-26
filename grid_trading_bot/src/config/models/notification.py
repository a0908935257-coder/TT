"""
Notification Configuration Model.

Provides configuration for notification services.
"""

from typing import Literal, Optional

from pydantic import Field, field_validator

from .base import BaseConfig


class NotificationConfig(BaseConfig):
    """
    Notification service configuration.

    Example:
        >>> config = NotificationConfig(
        ...     enabled=True,
        ...     discord_webhook_url="${DISCORD_WEBHOOK_URL}",
        ...     min_level="info",
        ... )
    """

    enabled: bool = Field(
        default=True,
        description="Enable notifications",
    )
    discord_webhook_url: Optional[str] = Field(
        default=None,
        description="Discord webhook URL",
    )
    discord_username: str = Field(
        default="Trading Bot",
        description="Discord bot display name",
    )
    min_level: Literal["debug", "info", "success", "warning", "error", "critical"] = Field(
        default="info",
        description="Minimum notification level",
    )
    rate_limit: int = Field(
        default=25,
        ge=1,
        le=60,
        description="Max notifications per minute",
    )

    # Notification toggles
    notify_on_order: bool = Field(
        default=True,
        description="Notify on order events",
    )
    notify_on_trade: bool = Field(
        default=True,
        description="Notify on trade events",
    )
    notify_on_position: bool = Field(
        default=True,
        description="Notify on position changes",
    )
    notify_on_error: bool = Field(
        default=True,
        description="Notify on errors",
    )
    notify_on_risk: bool = Field(
        default=True,
        description="Notify on risk alerts",
    )

    @field_validator("min_level")
    @classmethod
    def validate_min_level(cls, v: str) -> str:
        """Validate and normalize min_level."""
        return v.lower().strip()

    @property
    def is_discord_configured(self) -> bool:
        """Check if Discord is configured."""
        return bool(self.discord_webhook_url)

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and configured."""
        return self.enabled and self.is_discord_configured

    def should_notify(self, level: str) -> bool:
        """
        Check if a notification should be sent based on level.

        Args:
            level: Notification level

        Returns:
            True if notification should be sent
        """
        if not self.enabled:
            return False

        level_order = ["debug", "info", "success", "warning", "error", "critical"]

        try:
            min_index = level_order.index(self.min_level)
        except ValueError:
            # Invalid min_level configured, default to info
            min_index = level_order.index("info")

        try:
            level_index = level_order.index(level.lower())
        except ValueError:
            # Invalid level requested, reject unknown levels
            return False

        return level_index >= min_index
