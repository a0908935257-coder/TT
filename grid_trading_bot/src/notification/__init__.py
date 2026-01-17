# Notification module - Multi-channel notification system
from .base import BaseNotifier, NotificationLevel
from .discord import DiscordEmbed, DiscordNotifier, EmbedColor
from .manager import (
    NotificationManager,
    close_notification_manager,
    get_notification_manager,
    init_notification_manager,
)
from .templates import NotificationTemplates

__all__ = [
    # Base
    "NotificationLevel",
    "BaseNotifier",
    # Discord
    "DiscordNotifier",
    "DiscordEmbed",
    "EmbedColor",
    # Templates
    "NotificationTemplates",
    # Manager
    "NotificationManager",
    "init_notification_manager",
    "get_notification_manager",
    "close_notification_manager",
]
