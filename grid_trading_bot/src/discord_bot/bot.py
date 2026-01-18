"""
Discord Trading Bot.

Main bot class that integrates with the trading system.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional, Protocol

import discord
from discord.ext import commands

from src.core import get_logger

from .config import DiscordConfig

logger = get_logger(__name__)


class MasterProtocol(Protocol):
    """Protocol for Master interface."""

    async def start(self) -> None:
        """Start the master."""
        ...

    async def stop(self) -> None:
        """Stop the master."""
        ...

    def get_all_bots(self) -> list:
        """Get all bots."""
        ...

    def get_dashboard_data(self) -> Any:
        """Get dashboard data."""
        ...


class RiskEngineProtocol(Protocol):
    """Protocol for RiskEngine interface."""

    def get_status(self) -> Any:
        """Get risk status."""
        ...

    def get_statistics(self) -> dict:
        """Get risk statistics."""
        ...


class TradingBot(commands.Bot):
    """
    Discord bot for trading system control.

    Provides slash commands for:
    - Bot management (create, start, stop, pause, resume, delete)
    - Status monitoring (dashboard, bot details)
    - Risk management (risk status, emergency controls)
    - Profit tracking

    Example:
        >>> config = DiscordConfig.from_env()
        >>> bot = TradingBot(config, master, risk_engine)
        >>> await bot.start_bot()
    """

    def __init__(
        self,
        config: DiscordConfig,
        master: Optional[MasterProtocol] = None,
        risk_engine: Optional[RiskEngineProtocol] = None,
    ):
        """
        Initialize TradingBot.

        Args:
            config: Discord configuration
            master: Master control instance
            risk_engine: Risk engine instance
        """
        # Set up intents
        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        intents.members = True

        # Initialize bot
        super().__init__(
            command_prefix=config.command_prefix if config.command_prefix != "/" else "!",
            intents=intents,
        )

        # Store references
        self._config = config
        self._master = master
        self._risk_engine = risk_engine

        # Channel cache
        self._channels: Dict[str, discord.TextChannel] = {}

        # State
        self._ready = False
        self._guild: Optional[discord.Guild] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> DiscordConfig:
        """Get Discord configuration."""
        return self._config

    @property
    def master(self) -> Optional[MasterProtocol]:
        """Get Master instance."""
        return self._master

    @property
    def risk_engine(self) -> Optional[RiskEngineProtocol]:
        """Get RiskEngine instance."""
        return self._risk_engine

    @property
    def is_ready(self) -> bool:
        """Check if bot is ready."""
        return self._ready

    # =========================================================================
    # Setup and Lifecycle
    # =========================================================================

    async def setup_hook(self) -> None:
        """
        Called when bot is starting up.

        Loads all cogs and syncs slash commands.
        """
        logger.info("Setting up Discord bot...")

        # Load cogs
        cogs = [
            "src.discord_bot.cogs.bot_commands",
            "src.discord_bot.cogs.status_commands",
            "src.discord_bot.cogs.risk_commands",
            "src.discord_bot.cogs.admin_commands",
        ]

        for cog in cogs:
            try:
                await self.load_extension(cog)
                logger.info(f"Loaded cog: {cog}")
            except Exception as e:
                logger.warning(f"Failed to load cog {cog}: {e}")

        # Sync slash commands to guild
        if self._config.sync_commands_on_start and self._config.guild_id:
            try:
                guild = discord.Object(id=self._config.guild_id)
                self.tree.copy_global_to(guild=guild)
                synced = await self.tree.sync(guild=guild)
                logger.info(f"Synced {len(synced)} slash commands to guild {self._config.guild_id}")
            except Exception as e:
                logger.error(f"Failed to sync commands: {e}")

    async def on_ready(self) -> None:
        """Called when bot is ready."""
        logger.info(f"Discord Bot logged in as {self.user}")

        # Get guild
        if self._config.guild_id:
            self._guild = self.get_guild(self._config.guild_id)
            if self._guild:
                logger.info(f"Connected to guild: {self._guild.name}")

                # Cache channels
                await self._cache_channels()
            else:
                logger.warning(f"Guild {self._config.guild_id} not found")

        # Set presence
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching,
                name="Trading System",
            )
        )

        self._ready = True

        # Send online notification
        if self._config.send_online_notification:
            await self._send_online_notification()

    async def _cache_channels(self) -> None:
        """Cache configured channels."""
        if not self._guild:
            return

        channel_mapping = {
            "control": self._config.channels.control,
            "notifications": self._config.channels.notifications,
            "dashboard": self._config.channels.dashboard,
            "alerts": self._config.channels.alerts,
            "logs": self._config.channels.logs,
        }

        for name, channel_id in channel_mapping.items():
            if channel_id:
                channel = self._guild.get_channel(channel_id)
                if channel and isinstance(channel, discord.TextChannel):
                    self._channels[name] = channel
                    logger.debug(f"Cached channel: {name} -> #{channel.name}")
                else:
                    logger.warning(f"Channel not found: {name} (ID: {channel_id})")

    async def _send_online_notification(self) -> None:
        """Send online notification to control channel."""
        channel = self._channels.get("control")
        if not channel:
            return

        embed = discord.Embed(
            title="Trading Console Online",
            description="Trading system control console is now online.",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc),
        )

        # Add system info
        if self._master:
            try:
                bots = self._master.get_all_bots()
                embed.add_field(
                    name="Registered Bots",
                    value=str(len(bots)),
                    inline=True,
                )
            except Exception:
                pass

        embed.set_footer(text=f"Bot: {self.user}")

        try:
            await channel.send(embed=embed)
        except Exception as e:
            logger.error(f"Failed to send online notification: {e}")

    # =========================================================================
    # Channel Methods
    # =========================================================================

    def get_channel_by_name(self, name: str) -> Optional[discord.TextChannel]:
        """
        Get cached channel by name.

        Args:
            name: Channel name (control, notifications, dashboard, alerts, logs)

        Returns:
            TextChannel or None
        """
        return self._channels.get(name)

    async def send_notification(
        self,
        embed: discord.Embed,
        channel_name: str = "notifications",
    ) -> bool:
        """
        Send notification to specified channel.

        Args:
            embed: Discord embed to send
            channel_name: Target channel name

        Returns:
            True if sent successfully
        """
        channel = self._channels.get(channel_name)
        if not channel:
            logger.warning(f"Channel not found: {channel_name}")
            return False

        try:
            await channel.send(embed=embed)
            return True
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    async def send_alert(
        self,
        embed: discord.Embed,
        mention_role: bool = False,
    ) -> bool:
        """
        Send alert to alerts channel.

        Args:
            embed: Discord embed to send
            mention_role: Whether to mention admin role

        Returns:
            True if sent successfully
        """
        channel = self._channels.get("alerts") or self._channels.get("notifications")
        if not channel:
            logger.warning("No alert channel configured")
            return False

        try:
            content = None
            if mention_role and self._config.admin_role_id:
                content = f"<@&{self._config.admin_role_id}>"

            await channel.send(content=content, embed=embed)
            return True
        except Exception as e:
            logger.error(f"Failed to send alert: {e}")
            return False

    async def send_log(self, message: str) -> bool:
        """
        Send log message to logs channel.

        Args:
            message: Log message

        Returns:
            True if sent successfully
        """
        channel = self._channels.get("logs")
        if not channel:
            return False

        try:
            # Use code block for logs
            await channel.send(f"```\n{message}\n```")
            return True
        except Exception as e:
            logger.error(f"Failed to send log: {e}")
            return False

    # =========================================================================
    # Bot Control
    # =========================================================================

    async def start_bot(self) -> None:
        """Start the Discord bot."""
        if not self._config.token:
            raise ValueError("Discord bot token not configured")

        logger.info("Starting Discord bot...")
        await self.start(self._config.token)

    async def stop_bot(self) -> None:
        """Stop the Discord bot gracefully."""
        logger.info("Stopping Discord bot...")

        # Send offline notification
        channel = self._channels.get("control")
        if channel:
            try:
                embed = discord.Embed(
                    title="Trading Console Offline",
                    description="Trading system control console is shutting down.",
                    color=discord.Color.orange(),
                    timestamp=datetime.now(timezone.utc),
                )
                await channel.send(embed=embed)
            except Exception:
                pass

        await self.close()

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_admin(self, member: discord.Member) -> bool:
        """
        Check if member has admin role.

        Args:
            member: Discord member

        Returns:
            True if member has admin role
        """
        if not self._config.admin_role_id:
            # If no admin role configured, check for administrator permission
            return member.guild_permissions.administrator

        return any(role.id == self._config.admin_role_id for role in member.roles)

    def is_user(self, member: discord.Member) -> bool:
        """
        Check if member has user role.

        Args:
            member: Discord member

        Returns:
            True if member has user role (or admin role)
        """
        if self.is_admin(member):
            return True

        if not self._config.user_role_id:
            # If no user role configured, allow all
            return True

        return any(role.id == self._config.user_role_id for role in member.roles)
