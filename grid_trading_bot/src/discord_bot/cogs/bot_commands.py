"""
Bot Management Commands.

Slash commands for managing trading bots.
"""

from datetime import datetime, timezone
from typing import List, Optional

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_status_emoji(state) -> str:
    """Get emoji for bot state."""
    state_str = state.value if hasattr(state, "value") else str(state)
    mapping = {
        "registered": "⚪",
        "initializing": "🔄",
        "running": "🟢",
        "paused": "🟡",
        "stopping": "🟠",
        "stopped": "🔴",
        "error": "❌",
    }
    return mapping.get(state_str.lower(), "❓")


def get_state_color(state) -> discord.Color:
    """Get color for bot state."""
    state_str = state.value if hasattr(state, "value") else str(state)
    mapping = {
        "registered": discord.Color.light_grey(),
        "initializing": discord.Color.blue(),
        "running": discord.Color.green(),
        "paused": discord.Color.yellow(),
        "stopping": discord.Color.orange(),
        "stopped": discord.Color.red(),
        "error": discord.Color.dark_red(),
    }
    return mapping.get(state_str.lower(), discord.Color.default())


def is_admin():
    """Check if user has admin permission."""
    async def predicate(interaction: discord.Interaction) -> bool:
        trading_bot = interaction.client
        if hasattr(trading_bot, "is_admin"):
            return trading_bot.is_admin(interaction.user)
        return interaction.user.guild_permissions.administrator
    return app_commands.check(predicate)


# =============================================================================
# Views
# =============================================================================


class BotControlView(discord.ui.View):
    """Control buttons for a bot."""

    def __init__(self, bot_id: str, state: str, master):
        super().__init__(timeout=180)
        self.bot_id = bot_id
        self.state = state
        self.master = master

        # Add buttons based on state
        state_lower = state.lower() if isinstance(state, str) else state.value.lower()

        if state_lower in ("stopped", "registered", "error"):
            self.add_item(StartButton(bot_id, master))
        elif state_lower == "running":
            self.add_item(PauseButton(bot_id, master))
            self.add_item(StopButton(bot_id, master))
        elif state_lower == "paused":
            self.add_item(ResumeButton(bot_id, master))
            self.add_item(StopButton(bot_id, master))


class StartButton(discord.ui.Button):
    """Start bot button."""

    def __init__(self, bot_id: str, master):
        super().__init__(label="Start", style=discord.ButtonStyle.green, emoji="▶️")
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        result = await self.master.start_bot(self.bot_id)
        if result.success:
            await interaction.followup.send(f"✅ Bot `{self.bot_id}` started")
        else:
            await interaction.followup.send(f"❌ Failed: {result.message}")


class StopButton(discord.ui.Button):
    """Stop bot button."""

    def __init__(self, bot_id: str, master):
        super().__init__(label="Stop", style=discord.ButtonStyle.danger, emoji="⏹️")
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        result = await self.master.stop_bot(self.bot_id)
        if result.success:
            await interaction.followup.send(f"🛑 Bot `{self.bot_id}` stopped")
        else:
            await interaction.followup.send(f"❌ Failed: {result.message}")


class PauseButton(discord.ui.Button):
    """Pause bot button."""

    def __init__(self, bot_id: str, master):
        super().__init__(label="Pause", style=discord.ButtonStyle.secondary, emoji="⏸️")
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        result = await self.master.pause_bot(self.bot_id)
        if result.success:
            await interaction.followup.send(f"⏸️ Bot `{self.bot_id}` paused")
        else:
            await interaction.followup.send(f"❌ Failed: {result.message}")


class ResumeButton(discord.ui.Button):
    """Resume bot button."""

    def __init__(self, bot_id: str, master):
        super().__init__(label="Resume", style=discord.ButtonStyle.green, emoji="▶️")
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()
        result = await self.master.resume_bot(self.bot_id)
        if result.success:
            await interaction.followup.send(f"▶️ Bot `{self.bot_id}` resumed")
        else:
            await interaction.followup.send(f"❌ Failed: {result.message}")


class ConfirmDeleteView(discord.ui.View):
    """Confirmation view for deleting a bot."""

    def __init__(self, bot_id: str, master):
        super().__init__(timeout=60)
        self.bot_id = bot_id
        self.master = master

    @discord.ui.button(label="Confirm Delete", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        result = await self.master.delete_bot(self.bot_id)
        if result.success:
            embed = discord.Embed(
                title="🗑️ Bot Deleted",
                description=f"Bot `{self.bot_id}` has been deleted",
                color=discord.Color.red(),
            )
        else:
            embed = discord.Embed(
                title="❌ Delete Failed",
                description=result.message,
                color=discord.Color.red(),
            )
        await interaction.followup.send(embed=embed)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message("Delete cancelled", ephemeral=True)
        self.stop()


class CreateBotModal(discord.ui.Modal):
    """Modal for creating a new bot."""

    symbol = discord.ui.TextInput(
        label="Trading Pair",
        placeholder="e.g., BTCUSDT",
        default="BTCUSDT",
        max_length=20,
    )

    max_capital = discord.ui.TextInput(
        label="Max Capital (USDT)",
        placeholder="e.g., 100",
        default="100",
        max_length=10,
    )

    def __init__(self, master, bot_type: str):
        self.bot_type_str = bot_type
        # Set title based on bot type
        titles = {
            "bollinger": "Create Bollinger Bot",
            "rsi": "Create RSI Bot",
            "grid_futures": "Create Grid Futures Bot",
            "grid": "Create Grid Bot (Spot)",
        }
        super().__init__(title=titles.get(bot_type, "Create Bot"))
        self.master = master

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            import os
            from src.master import BotType

            symbol = self.symbol.value.upper()
            max_capital = self.max_capital.value

            # Build config based on bot type
            if self.bot_type_str == "bollinger":
                bot_type = BotType.BOLLINGER
                bot_config = {
                    "symbol": symbol,
                    "timeframe": os.getenv('BOLLINGER_TIMEFRAME', '15m'),
                    "leverage": int(os.getenv('BOLLINGER_LEVERAGE', '20')),
                    "position_size_pct": os.getenv('BOLLINGER_POSITION_SIZE', '0.1'),
                    "bb_period": int(os.getenv('BOLLINGER_BB_PERIOD', '20')),
                    "bb_std": os.getenv('BOLLINGER_BB_STD', '3.25'),
                    "bbw_lookback": int(os.getenv('BOLLINGER_BBW_LOOKBACK', '200')),
                    "bbw_threshold_pct": int(os.getenv('BOLLINGER_BBW_THRESHOLD', '20')),
                    "stop_loss_pct": os.getenv('BOLLINGER_STOP_LOSS_PCT', '0.015'),
                    "max_hold_bars": int(os.getenv('BOLLINGER_MAX_HOLD_BARS', '48')),
                    "max_capital": max_capital,
                    # Trend filter settings
                    "use_trend_filter": os.getenv('BOLLINGER_USE_TREND_FILTER', 'false').lower() == 'true',
                    "trend_period": int(os.getenv('BOLLINGER_TREND_PERIOD', '50')),
                    # ATR stop loss settings
                    "use_atr_stop": os.getenv('BOLLINGER_USE_ATR_STOP', 'true').lower() == 'true',
                    "atr_period": int(os.getenv('BOLLINGER_ATR_PERIOD', '14')),
                    "atr_multiplier": os.getenv('BOLLINGER_ATR_MULTIPLIER', '2.0'),
                    # Trailing stop settings
                    "use_trailing_stop": os.getenv('BOLLINGER_USE_TRAILING_STOP', 'true').lower() == 'true',
                    "trailing_atr_mult": os.getenv('BOLLINGER_TRAILING_ATR_MULT', '2.0'),
                }
                type_info = f"Leverage: 20x | Timeframe: 15m"

            elif self.bot_type_str == "rsi":
                bot_type = BotType.RSI
                bot_config = {
                    "symbol": symbol,
                    "timeframe": os.getenv('RSI_TIMEFRAME', '15m'),
                    "rsi_period": int(os.getenv('RSI_PERIOD', '14')),
                    "oversold": int(os.getenv('RSI_OVERSOLD', '20')),
                    "overbought": int(os.getenv('RSI_OVERBOUGHT', '80')),
                    "exit_level": int(os.getenv('RSI_EXIT_LEVEL', '50')),
                    "leverage": int(os.getenv('RSI_LEVERAGE', '7')),
                    "margin_type": os.getenv('RSI_MARGIN_TYPE', 'ISOLATED'),
                    "max_capital": max_capital,
                    "position_size_pct": os.getenv('RSI_POSITION_SIZE', '0.1'),
                    "stop_loss_pct": os.getenv('RSI_STOP_LOSS_PCT', '0.02'),
                    "take_profit_pct": os.getenv('RSI_TAKE_PROFIT_PCT', '0.03'),
                }
                type_info = f"Leverage: 7x | RSI: 20/80"

            elif self.bot_type_str == "grid_futures":
                bot_type = BotType.GRID_FUTURES
                bot_config = {
                    "symbol": symbol,
                    "timeframe": os.getenv('GRID_FUTURES_TIMEFRAME', '1h'),
                    "leverage": int(os.getenv('GRID_FUTURES_LEVERAGE', '3')),
                    "margin_type": os.getenv('GRID_FUTURES_MARGIN_TYPE', 'ISOLATED'),
                    "grid_count": int(os.getenv('GRID_FUTURES_COUNT', '12')),
                    "direction": os.getenv('GRID_FUTURES_DIRECTION', 'neutral'),
                    "use_trend_filter": os.getenv('GRID_FUTURES_USE_TREND_FILTER', 'false').lower() == 'true',
                    "trend_period": int(os.getenv('GRID_FUTURES_TREND_PERIOD', '20')),
                    "use_atr_range": os.getenv('GRID_FUTURES_USE_ATR_RANGE', 'true').lower() == 'true',
                    "atr_period": int(os.getenv('GRID_FUTURES_ATR_PERIOD', '14')),
                    "atr_multiplier": os.getenv('GRID_FUTURES_ATR_MULTIPLIER', '2.0'),
                    "fallback_range_pct": os.getenv('GRID_FUTURES_RANGE_PCT', '0.08'),
                    "max_capital": max_capital,
                    "position_size_pct": os.getenv('GRID_FUTURES_POSITION_SIZE', '0.1'),
                    "max_position_pct": os.getenv('GRID_FUTURES_MAX_POSITION', '0.5'),
                    "stop_loss_pct": os.getenv('GRID_FUTURES_STOP_LOSS', '0.05'),
                    "rebuild_threshold_pct": os.getenv('GRID_FUTURES_REBUILD_THRESHOLD', '0.02'),
                }
                type_info = f"Leverage: 3x | Grids: 12"

            else:  # grid (spot)
                bot_type = BotType.GRID
                bot_config = {
                    "symbol": symbol,
                    "market_type": "spot",
                    "total_investment": max_capital,
                    "risk_level": "moderate",
                }
                type_info = "Spot | Risk: Moderate"

            result = await self.master.create_bot(bot_type, bot_config)

            if result.success:
                embed = discord.Embed(
                    title=f"✅ {self.bot_type_str.replace('_', ' ').title()} Bot Created",
                    description=f"Bot `{result.bot_id}` created successfully",
                    color=discord.Color.green(),
                )
                embed.add_field(name="Symbol", value=symbol, inline=True)
                embed.add_field(name="Max Capital", value=f"{max_capital} USDT", inline=True)
                embed.add_field(name="Settings", value=type_info, inline=True)
                embed.add_field(
                    name="Next Step",
                    value=f"Use `/bot start {result.bot_id}` to start trading",
                    inline=False,
                )
            else:
                embed = discord.Embed(
                    title="❌ Create Failed",
                    description=result.message,
                    color=discord.Color.red(),
                )

            await interaction.followup.send(embed=embed)

        except Exception as e:
            logger.error(f"Error creating bot: {e}")
            await interaction.followup.send(f"Error: {e}")


# =============================================================================
# Cog
# =============================================================================


class BotCommands(commands.Cog):
    """Commands for bot management."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def master(self):
        """Get master instance."""
        return getattr(self.bot, "_master", None)

    # Bot command group (variable name cannot start with 'bot_' due to discord.py restriction)
    trading_group = app_commands.Group(name="bot", description="Bot management commands")

    # =========================================================================
    # /bot list
    # =========================================================================

    @trading_group.command(name="list", description="List all trading bots")
    async def bot_list(self, interaction: discord.Interaction):
        """List all trading bots."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="📋 Trading Bots",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        if not self.master:
            embed.description = "Master not connected"
            embed.color = discord.Color.orange()
            await interaction.followup.send(embed=embed)
            return

        try:
            bots = self.master.get_all_bots()
            if not bots:
                embed.description = "No bots registered\n\nUse `/bot create` to create a new bot"
                await interaction.followup.send(embed=embed)
                return

            for bot_info in bots[:12]:  # Limit to 12 for embed
                state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
                emoji = get_status_emoji(bot_info.state)
                bot_type = bot_info.bot_type.value if hasattr(bot_info.bot_type, "value") else str(bot_info.bot_type)

                embed.add_field(
                    name=f"{emoji} {bot_info.bot_id}",
                    value=f"**Type:** {bot_type}\n"
                          f"**Symbol:** {bot_info.symbol}\n"
                          f"**State:** {state}",
                    inline=True,
                )

            embed.set_footer(text=f"Total: {len(bots)} bots")

        except Exception as e:
            logger.error(f"Error listing bots: {e}")
            embed.description = f"Error: {e}"
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /bot create
    # =========================================================================

    @trading_group.command(name="create", description="Create a new trading bot")
    @app_commands.describe(bot_type="Type of bot to create")
    @app_commands.choices(bot_type=[
        app_commands.Choice(name="🔷 Bollinger Bot (合約 20x)", value="bollinger"),
        app_commands.Choice(name="📉 RSI Bot (合約 7x)", value="rsi"),
        app_commands.Choice(name="📊 Grid Futures Bot (合約 3x)", value="grid_futures"),
        app_commands.Choice(name="🟢 Grid Bot (現貨)", value="grid"),
    ])
    @is_admin()
    async def bot_create(
        self,
        interaction: discord.Interaction,
        bot_type: app_commands.Choice[str],
    ):
        """Create a new trading bot."""
        if not self.master:
            await interaction.response.send_message("Master not connected", ephemeral=True)
            return

        modal = CreateBotModal(self.master, bot_type.value)
        await interaction.response.send_modal(modal)

    # =========================================================================
    # /bot start
    # =========================================================================

    @trading_group.command(name="start", description="Start a trading bot")
    @app_commands.describe(bot_id="Bot ID to start")
    @is_admin()
    async def bot_start(self, interaction: discord.Interaction, bot_id: str):
        """Start a trading bot."""
        await interaction.response.defer()

        if not self.master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await self.master.start_bot(bot_id)

            if result.success:
                embed = discord.Embed(
                    title="✅ Bot Started",
                    description=f"Bot `{bot_id}` started successfully",
                    color=discord.Color.green(),
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                embed = discord.Embed(
                    title="❌ Start Failed",
                    description=f"Error: {result.message}",
                    color=discord.Color.red(),
                )

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /bot stop
    # =========================================================================

    @trading_group.command(name="stop", description="Stop a trading bot")
    @app_commands.describe(
        bot_id="Bot ID to stop",
        clear_position="Whether to close all positions",
    )
    @is_admin()
    async def bot_stop(
        self,
        interaction: discord.Interaction,
        bot_id: str,
        clear_position: bool = False,
    ):
        """Stop a trading bot."""
        await interaction.response.defer()

        if not self.master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await self.master.stop_bot(bot_id)

            if result.success:
                embed = discord.Embed(
                    title="🛑 Bot Stopped",
                    description=f"Bot `{bot_id}` has been stopped",
                    color=discord.Color.orange(),
                    timestamp=datetime.now(timezone.utc),
                )
                embed.add_field(
                    name="Clear Position",
                    value="Yes" if clear_position else "No (positions retained)",
                    inline=True,
                )

                # Show final stats if available
                if result.data:
                    stats = result.data
                    if "total_trades" in stats:
                        embed.add_field(
                            name="Final Statistics",
                            value=f"Total Trades: {stats.get('total_trades', 0)}\n"
                                  f"Total Profit: {stats.get('total_profit', 0):.2f} USDT",
                            inline=False,
                        )
            else:
                embed = discord.Embed(
                    title="❌ Stop Failed",
                    description=f"Error: {result.message}",
                    color=discord.Color.red(),
                )

        except Exception as e:
            logger.error(f"Error stopping bot: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /bot pause
    # =========================================================================

    @trading_group.command(name="pause", description="Pause a trading bot")
    @app_commands.describe(bot_id="Bot ID to pause")
    @is_admin()
    async def bot_pause(self, interaction: discord.Interaction, bot_id: str):
        """Pause a trading bot."""
        await interaction.response.defer()

        if not self.master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await self.master.pause_bot(bot_id)

            if result.success:
                embed = discord.Embed(
                    title="⏸️ Bot Paused",
                    description=f"Bot `{bot_id}` has been paused\n"
                                f"Orders cancelled, positions retained",
                    color=discord.Color.yellow(),
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                embed = discord.Embed(
                    title="❌ Pause Failed",
                    description=f"Error: {result.message}",
                    color=discord.Color.red(),
                )

        except Exception as e:
            logger.error(f"Error pausing bot: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /bot resume
    # =========================================================================

    @trading_group.command(name="resume", description="Resume a paused bot")
    @app_commands.describe(bot_id="Bot ID to resume")
    @is_admin()
    async def bot_resume(self, interaction: discord.Interaction, bot_id: str):
        """Resume a paused bot."""
        await interaction.response.defer()

        if not self.master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await self.master.resume_bot(bot_id)

            if result.success:
                embed = discord.Embed(
                    title="▶️ Bot Resumed",
                    description=f"Bot `{bot_id}` has resumed trading",
                    color=discord.Color.green(),
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                embed = discord.Embed(
                    title="❌ Resume Failed",
                    description=f"Error: {result.message}",
                    color=discord.Color.red(),
                )

        except Exception as e:
            logger.error(f"Error resuming bot: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /bot detail
    # =========================================================================

    @trading_group.command(name="detail", description="Show bot details")
    @app_commands.describe(bot_id="Bot ID to view")
    async def bot_detail(self, interaction: discord.Interaction, bot_id: str):
        """Show detailed bot information."""
        await interaction.response.defer()

        if not self.master:
            await interaction.followup.send("Master not connected")
            return

        try:
            bot_info = self.master.get_bot(bot_id)
            if not bot_info:
                await interaction.followup.send(f"Bot not found: `{bot_id}`")
                return

            state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
            bot_type = bot_info.bot_type.value if hasattr(bot_info.bot_type, "value") else str(bot_info.bot_type)
            emoji = get_status_emoji(bot_info.state)
            color = get_state_color(bot_info.state)

            embed = discord.Embed(
                title=f"{emoji} Bot: {bot_id}",
                color=color,
                timestamp=datetime.now(timezone.utc),
            )

            embed.add_field(name="Type", value=bot_type, inline=True)
            embed.add_field(name="Symbol", value=bot_info.symbol, inline=True)
            embed.add_field(name="State", value=state, inline=True)

            if bot_info.created_at:
                embed.add_field(
                    name="Created",
                    value=bot_info.created_at.strftime("%Y-%m-%d %H:%M"),
                    inline=True,
                )

            # Try to get more details from instance
            instance = self.master.registry.get_bot_instance(bot_id)
            if instance:
                if hasattr(instance, "get_status"):
                    status = instance.get_status()
                    if status.get("lower_price") and status.get("upper_price"):
                        embed.add_field(
                            name="Grid Range",
                            value=f"${float(status['lower_price']):.2f} - ${float(status['upper_price']):.2f}",
                            inline=False,
                        )
                    if status.get("grid_count"):
                        embed.add_field(name="Grid Count", value=str(status["grid_count"]), inline=True)
                    if "pending_buy_orders" in status:
                        embed.add_field(name="Buy Orders", value=str(status["pending_buy_orders"]), inline=True)
                    if "pending_sell_orders" in status:
                        embed.add_field(name="Sell Orders", value=str(status["pending_sell_orders"]), inline=True)

                if hasattr(instance, "get_statistics"):
                    stats = instance.get_statistics()
                    if stats.get("total_profit") is not None:
                        profit = float(stats["total_profit"])
                        profit_emoji = "🟢" if profit >= 0 else "🔴"
                        embed.add_field(
                            name=f"{profit_emoji} Total Profit",
                            value=f"{profit:.4f} USDT",
                            inline=True,
                        )
                    if stats.get("trade_count"):
                        embed.add_field(name="Trades", value=str(stats["trade_count"]), inline=True)

            # Add control buttons
            view = BotControlView(bot_id, state, self.master)
            await interaction.followup.send(embed=embed, view=view)

        except Exception as e:
            logger.error(f"Error getting bot detail: {e}")
            await interaction.followup.send(f"Error: {e}")

    # =========================================================================
    # /bot delete
    # =========================================================================

    @trading_group.command(name="delete", description="Delete a trading bot")
    @app_commands.describe(bot_id="Bot ID to delete")
    @is_admin()
    async def bot_delete(self, interaction: discord.Interaction, bot_id: str):
        """Delete a trading bot."""
        if not self.master:
            await interaction.response.send_message("Master not connected", ephemeral=True)
            return

        # Check if bot exists
        bot_info = self.master.get_bot(bot_id)
        if not bot_info:
            await interaction.response.send_message(f"Bot not found: `{bot_id}`", ephemeral=True)
            return

        # Show confirmation
        embed = discord.Embed(
            title="⚠️ Confirm Delete",
            description=f"Are you sure you want to delete bot `{bot_id}`?\n\n"
                        f"**This action cannot be undone.**",
            color=discord.Color.red(),
        )
        embed.add_field(name="Symbol", value=bot_info.symbol, inline=True)

        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
        embed.add_field(name="State", value=state, inline=True)

        view = ConfirmDeleteView(bot_id, self.master)
        await interaction.response.send_message(embed=embed, view=view)

    # =========================================================================
    # Autocomplete
    # =========================================================================

    async def bot_id_autocomplete(
        self,
        interaction: discord.Interaction,
        current: str,
    ) -> List[app_commands.Choice[str]]:
        """Autocomplete for bot_id parameter."""
        if not self.master:
            return []

        try:
            bots = self.master.get_all_bots()
            choices = []

            for bot_info in bots:
                if current.lower() in bot_info.bot_id.lower():
                    state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
                    choices.append(
                        app_commands.Choice(
                            name=f"{bot_info.bot_id} ({bot_info.symbol}) - {state}",
                            value=bot_info.bot_id,
                        )
                    )

            return choices[:25]  # Discord limit

        except Exception:
            return []

    # Register autocomplete
    @bot_start.autocomplete("bot_id")
    async def start_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    @bot_stop.autocomplete("bot_id")
    async def stop_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    @bot_pause.autocomplete("bot_id")
    async def pause_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    @bot_resume.autocomplete("bot_id")
    async def resume_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    @bot_detail.autocomplete("bot_id")
    async def detail_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    @bot_delete.autocomplete("bot_id")
    async def delete_autocomplete(self, interaction: discord.Interaction, current: str):
        return await self.bot_id_autocomplete(interaction, current)

    # =========================================================================
    # Error Handler
    # =========================================================================

    @bot_create.error
    @bot_start.error
    @bot_stop.error
    @bot_pause.error
    @bot_resume.error
    @bot_delete.error
    async def admin_command_error(self, interaction: discord.Interaction, error: Exception):
        """Handle admin command errors."""
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message(
                "You don't have permission to use this command",
                ephemeral=True,
            )
        else:
            logger.error(f"Bot command error: {error}")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"An error occurred: {error}",
                    ephemeral=True,
                )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(BotCommands(bot))
