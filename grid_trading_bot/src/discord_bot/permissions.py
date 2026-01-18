"""
Discord Bot Permissions.

Permission control system for the Discord trading bot.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from typing import Callable, List, Optional, Set, Union

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Permission Level Enum
# =============================================================================


class PermissionLevel(Enum):
    """Permission levels from lowest to highest."""

    NONE = 0
    USER = 1
    ADMIN = 2
    OWNER = 3

    def __ge__(self, other: "PermissionLevel") -> bool:
        return self.value >= other.value

    def __gt__(self, other: "PermissionLevel") -> bool:
        return self.value > other.value

    def __le__(self, other: "PermissionLevel") -> bool:
        return self.value <= other.value

    def __lt__(self, other: "PermissionLevel") -> bool:
        return self.value < other.value


# =============================================================================
# Permission Config
# =============================================================================


@dataclass
class PermissionConfig:
    """Configuration for permission system."""

    # Owner IDs (highest privilege)
    owner_ids: Set[int] = field(default_factory=set)

    # Admin role ID
    admin_role_id: Optional[int] = None

    # User role ID (basic access)
    user_role_id: Optional[int] = None

    # Allowed channel IDs (if empty, all channels allowed)
    allowed_channels: Set[int] = field(default_factory=set)

    # Control channel ID (for admin commands)
    control_channel_id: Optional[int] = None

    # Whether to allow server administrators as bot admins
    allow_server_admins: bool = True

    def add_owner(self, user_id: int) -> None:
        """Add an owner ID."""
        self.owner_ids.add(user_id)

    def remove_owner(self, user_id: int) -> None:
        """Remove an owner ID."""
        self.owner_ids.discard(user_id)

    def add_allowed_channel(self, channel_id: int) -> None:
        """Add an allowed channel."""
        self.allowed_channels.add(channel_id)


# =============================================================================
# Permission Checker
# =============================================================================


class PermissionChecker:
    """Check user permissions for bot commands."""

    def __init__(self, config: PermissionConfig):
        self._config = config

    @property
    def config(self) -> PermissionConfig:
        """Get permission config."""
        return self._config

    def get_level(self, member: Union[discord.Member, discord.User]) -> PermissionLevel:
        """
        Get the permission level for a member.

        Args:
            member: Discord member or user

        Returns:
            PermissionLevel for the member
        """
        # Check if owner
        if member.id in self._config.owner_ids:
            return PermissionLevel.OWNER

        # For User objects (DMs) without roles, only check owner
        # Use duck typing to support both discord.Member and mock objects
        if not hasattr(member, "roles") or not hasattr(member, "guild_permissions"):
            return PermissionLevel.NONE

        # Check server administrator (if allowed)
        if self._config.allow_server_admins and member.guild_permissions.administrator:
            return PermissionLevel.ADMIN

        # Check admin role
        if self._config.admin_role_id:
            for role in member.roles:
                if role.id == self._config.admin_role_id:
                    return PermissionLevel.ADMIN

        # Check user role
        if self._config.user_role_id:
            for role in member.roles:
                if role.id == self._config.user_role_id:
                    return PermissionLevel.USER

        # Default: no special permissions, but allow basic access
        return PermissionLevel.USER

    def check(self, member: Union[discord.Member, discord.User], required: PermissionLevel) -> bool:
        """
        Check if member has required permission level.

        Args:
            member: Discord member or user
            required: Required permission level

        Returns:
            True if member has required level or higher
        """
        return self.get_level(member) >= required

    def is_owner(self, member: Union[discord.Member, discord.User]) -> bool:
        """Check if member is an owner."""
        return member.id in self._config.owner_ids

    def is_admin(self, member: Union[discord.Member, discord.User]) -> bool:
        """Check if member is admin or higher."""
        return self.get_level(member) >= PermissionLevel.ADMIN

    def is_user(self, member: Union[discord.Member, discord.User]) -> bool:
        """Check if member is user or higher."""
        return self.get_level(member) >= PermissionLevel.USER

    def is_allowed_channel(self, channel_id: int) -> bool:
        """
        Check if channel is allowed for bot commands.

        Args:
            channel_id: Discord channel ID

        Returns:
            True if allowed (or if no channel restrictions)
        """
        if not self._config.allowed_channels:
            return True
        return channel_id in self._config.allowed_channels

    def is_control_channel(self, channel_id: int) -> bool:
        """
        Check if channel is the control channel.

        Args:
            channel_id: Discord channel ID

        Returns:
            True if this is the control channel
        """
        if not self._config.control_channel_id:
            return True
        return channel_id == self._config.control_channel_id


# =============================================================================
# Global Permission Checker Instance
# =============================================================================

_permission_checker: Optional[PermissionChecker] = None


def set_permission_checker(checker: PermissionChecker) -> None:
    """Set the global permission checker."""
    global _permission_checker
    _permission_checker = checker


def get_permission_checker() -> Optional[PermissionChecker]:
    """Get the global permission checker."""
    return _permission_checker


# =============================================================================
# Permission Check Decorators
# =============================================================================


def require_owner():
    """
    Decorator to require owner permission.

    Usage:
        @bot_group.command(name="shutdown")
        @require_owner()
        async def shutdown(self, interaction):
            ...
    """
    async def predicate(interaction: discord.Interaction) -> bool:
        # Check using bot's permission checker if available
        bot = interaction.client
        if hasattr(bot, "permission_checker"):
            return bot.permission_checker.is_owner(interaction.user)

        # Fallback to global checker
        checker = get_permission_checker()
        if checker:
            return checker.is_owner(interaction.user)

        # Fallback to checking guild owner
        if interaction.guild and interaction.guild.owner_id == interaction.user.id:
            return True

        return False

    return app_commands.check(predicate)


def require_admin():
    """
    Decorator to require admin permission.

    Usage:
        @bot_group.command(name="create")
        @require_admin()
        async def bot_create(self, interaction):
            ...
    """
    async def predicate(interaction: discord.Interaction) -> bool:
        # Check using bot's permission checker if available
        bot = interaction.client
        if hasattr(bot, "permission_checker"):
            return bot.permission_checker.is_admin(interaction.user)

        # Check using bot's is_admin method
        if hasattr(bot, "is_admin"):
            return bot.is_admin(interaction.user)

        # Fallback to global checker
        checker = get_permission_checker()
        if checker:
            return checker.is_admin(interaction.user)

        # Fallback to server administrator permission
        if isinstance(interaction.user, discord.Member):
            return interaction.user.guild_permissions.administrator

        return False

    return app_commands.check(predicate)


def require_user():
    """
    Decorator to require user permission (minimum access level).

    Usage:
        @bot_group.command(name="list")
        @require_user()
        async def bot_list(self, interaction):
            ...
    """
    async def predicate(interaction: discord.Interaction) -> bool:
        # Check using bot's permission checker if available
        bot = interaction.client
        if hasattr(bot, "permission_checker"):
            return bot.permission_checker.is_user(interaction.user)

        # Check using bot's is_user method
        if hasattr(bot, "is_user"):
            return bot.is_user(interaction.user)

        # Fallback to global checker
        checker = get_permission_checker()
        if checker:
            return checker.is_user(interaction.user)

        # Default: allow all users
        return True

    return app_commands.check(predicate)


def allowed_channel_only():
    """
    Decorator to restrict command to allowed channels only.

    Usage:
        @bot_group.command(name="status")
        @allowed_channel_only()
        async def status(self, interaction):
            ...
    """
    async def predicate(interaction: discord.Interaction) -> bool:
        # Check using bot's permission checker if available
        bot = interaction.client
        if hasattr(bot, "permission_checker"):
            return bot.permission_checker.is_allowed_channel(interaction.channel_id)

        # Fallback to global checker
        checker = get_permission_checker()
        if checker:
            return checker.is_allowed_channel(interaction.channel_id)

        # Default: allow all channels
        return True

    return app_commands.check(predicate)


def control_channel_only():
    """
    Decorator to restrict command to control channel only.

    Usage:
        @bot_group.command(name="create")
        @control_channel_only()
        async def bot_create(self, interaction):
            ...
    """
    async def predicate(interaction: discord.Interaction) -> bool:
        # Check using bot's permission checker if available
        bot = interaction.client
        if hasattr(bot, "permission_checker"):
            return bot.permission_checker.is_control_channel(interaction.channel_id)

        # Fallback to global checker
        checker = get_permission_checker()
        if checker:
            return checker.is_control_channel(interaction.channel_id)

        # Default: allow all channels
        return True

    return app_commands.check(predicate)


# =============================================================================
# Error Handler for Permission Errors
# =============================================================================


async def handle_permission_error(
    interaction: discord.Interaction,
    error: app_commands.CheckFailure,
) -> None:
    """
    Handle permission check failures with friendly messages.

    Args:
        interaction: Discord interaction
        error: The check failure error
    """
    # Determine error message based on the check that failed
    message = "You don't have permission to use this command."

    # Try to determine which check failed
    if hasattr(error, "missing_permissions"):
        message = f"Missing permissions: {', '.join(error.missing_permissions)}"
    elif hasattr(error, "missing_role"):
        message = f"You need the {error.missing_role} role to use this command."

    # Send ephemeral message
    if interaction.response.is_done():
        await interaction.followup.send(message, ephemeral=True)
    else:
        await interaction.response.send_message(message, ephemeral=True)


# =============================================================================
# Permission Check Functions (for use in code)
# =============================================================================


def check_admin(interaction: discord.Interaction) -> bool:
    """
    Check if user has admin permission (non-decorator version).

    Args:
        interaction: Discord interaction

    Returns:
        True if user is admin or higher
    """
    bot = interaction.client
    if hasattr(bot, "is_admin"):
        return bot.is_admin(interaction.user)

    checker = get_permission_checker()
    if checker:
        return checker.is_admin(interaction.user)

    if isinstance(interaction.user, discord.Member):
        return interaction.user.guild_permissions.administrator

    return False


def check_owner(interaction: discord.Interaction) -> bool:
    """
    Check if user is owner (non-decorator version).

    Args:
        interaction: Discord interaction

    Returns:
        True if user is owner
    """
    bot = interaction.client
    if hasattr(bot, "permission_checker"):
        return bot.permission_checker.is_owner(interaction.user)

    checker = get_permission_checker()
    if checker:
        return checker.is_owner(interaction.user)

    if interaction.guild and interaction.guild.owner_id == interaction.user.id:
        return True

    return False


# =============================================================================
# Utility Functions
# =============================================================================


def load_permission_config_from_env() -> PermissionConfig:
    """
    Load permission config from environment variables.

    Environment variables:
        DISCORD_OWNER_IDS: Comma-separated owner IDs
        DISCORD_ADMIN_ROLE_ID: Admin role ID
        DISCORD_USER_ROLE_ID: User role ID
        DISCORD_ALLOWED_CHANNELS: Comma-separated channel IDs
        DISCORD_CONTROL_CHANNEL_ID: Control channel ID
        DISCORD_ALLOW_SERVER_ADMINS: Whether to allow server admins (true/false)

    Returns:
        PermissionConfig
    """
    import os

    config = PermissionConfig()

    # Owner IDs
    owner_ids_str = os.getenv("DISCORD_OWNER_IDS", "")
    if owner_ids_str:
        for id_str in owner_ids_str.split(","):
            try:
                config.owner_ids.add(int(id_str.strip()))
            except ValueError:
                pass

    # Admin role ID
    admin_role_str = os.getenv("DISCORD_ADMIN_ROLE_ID", "")
    if admin_role_str:
        try:
            config.admin_role_id = int(admin_role_str)
        except ValueError:
            pass

    # User role ID
    user_role_str = os.getenv("DISCORD_USER_ROLE_ID", "")
    if user_role_str:
        try:
            config.user_role_id = int(user_role_str)
        except ValueError:
            pass

    # Allowed channels
    channels_str = os.getenv("DISCORD_ALLOWED_CHANNELS", "")
    if channels_str:
        for id_str in channels_str.split(","):
            try:
                config.allowed_channels.add(int(id_str.strip()))
            except ValueError:
                pass

    # Control channel ID
    control_channel_str = os.getenv("DISCORD_CONTROL_CHANNEL_ID", "")
    if control_channel_str:
        try:
            config.control_channel_id = int(control_channel_str)
        except ValueError:
            pass

    # Allow server admins
    allow_admins = os.getenv("DISCORD_ALLOW_SERVER_ADMINS", "true").lower()
    config.allow_server_admins = allow_admins in ("true", "1", "yes")

    return config
