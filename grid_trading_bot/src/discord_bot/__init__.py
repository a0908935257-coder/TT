"""
Discord Bot Module.

Provides Discord-based control console for the trading system.
Includes slash commands, interactive components, and notifications.
"""

from .bot import TradingBot
from .config import ChannelConfig, DiscordConfig, load_discord_config
from .permissions import (
    PermissionChecker,
    PermissionConfig,
    PermissionLevel,
    allowed_channel_only,
    check_admin,
    check_owner,
    control_channel_only,
    get_permission_checker,
    load_permission_config_from_env,
    require_admin,
    require_owner,
    require_user,
    set_permission_checker,
)

__all__ = [
    # Bot
    "TradingBot",
    # Config
    "DiscordConfig",
    "ChannelConfig",
    "load_discord_config",
    # Permissions
    "PermissionLevel",
    "PermissionConfig",
    "PermissionChecker",
    "require_owner",
    "require_admin",
    "require_user",
    "allowed_channel_only",
    "control_channel_only",
    "check_admin",
    "check_owner",
    "get_permission_checker",
    "set_permission_checker",
    "load_permission_config_from_env",
]
