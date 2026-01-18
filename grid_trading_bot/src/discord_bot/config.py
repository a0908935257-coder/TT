"""
Discord Bot Configuration.

Provides configuration models for Discord bot settings.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class ChannelConfig:
    """Configuration for Discord channels."""

    control: int = 0
    notifications: int = 0
    dashboard: int = 0
    alerts: int = 0
    logs: int = 0

    @classmethod
    def from_env(cls) -> "ChannelConfig":
        """Load channel configuration from environment variables."""
        return cls(
            control=int(os.getenv("DISCORD_CONTROL_CHANNEL_ID", "0")),
            notifications=int(os.getenv("DISCORD_NOTIFICATION_CHANNEL_ID", "0")),
            dashboard=int(os.getenv("DISCORD_DASHBOARD_CHANNEL_ID", "0")),
            alerts=int(os.getenv("DISCORD_ALERT_CHANNEL_ID", "0")),
            logs=int(os.getenv("DISCORD_LOG_CHANNEL_ID", "0")),
        )


@dataclass
class DiscordConfig:
    """Configuration for Discord bot."""

    # Bot credentials
    token: str = ""

    # Server and channels
    guild_id: int = 0
    channels: ChannelConfig = field(default_factory=ChannelConfig)

    # Legacy channel IDs (for backward compatibility)
    control_channel_id: int = 0
    notification_channel_id: int = 0
    dashboard_channel_id: int = 0

    # Role IDs for permissions
    admin_role_id: int = 0
    user_role_id: int = 0

    # Command settings
    command_prefix: str = "/"

    # Features
    sync_commands_on_start: bool = True
    send_online_notification: bool = True

    def __post_init__(self):
        """Initialize derived values after dataclass init."""
        # If legacy channel IDs are set but ChannelConfig is not, copy them
        if self.control_channel_id and not self.channels.control:
            self.channels.control = self.control_channel_id
        if self.notification_channel_id and not self.channels.notifications:
            self.channels.notifications = self.notification_channel_id
        if self.dashboard_channel_id and not self.channels.dashboard:
            self.channels.dashboard = self.dashboard_channel_id

    @classmethod
    def from_env(cls) -> "DiscordConfig":
        """Load configuration from environment variables."""
        channels = ChannelConfig.from_env()

        return cls(
            token=os.getenv("DISCORD_BOT_TOKEN", ""),
            guild_id=int(os.getenv("DISCORD_GUILD_ID", "0")),
            channels=channels,
            control_channel_id=channels.control,
            notification_channel_id=channels.notifications,
            dashboard_channel_id=channels.dashboard,
            admin_role_id=int(os.getenv("DISCORD_ADMIN_ROLE_ID", "0")),
            user_role_id=int(os.getenv("DISCORD_USER_ROLE_ID", "0")),
            command_prefix=os.getenv("DISCORD_COMMAND_PREFIX", "/"),
            sync_commands_on_start=os.getenv("DISCORD_SYNC_COMMANDS", "true").lower() == "true",
            send_online_notification=os.getenv("DISCORD_ONLINE_NOTIFICATION", "true").lower() == "true",
        )

    def validate(self) -> list[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not self.token:
            errors.append("DISCORD_BOT_TOKEN is required")

        if not self.guild_id:
            errors.append("DISCORD_GUILD_ID is required")

        return errors

    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


def load_discord_config() -> DiscordConfig:
    """
    Load Discord configuration from environment.

    Returns:
        DiscordConfig instance

    Raises:
        ValueError: If configuration is invalid
    """
    config = DiscordConfig.from_env()

    errors = config.validate()
    if errors:
        for error in errors:
            logger.warning(f"Discord config warning: {error}")

    return config
