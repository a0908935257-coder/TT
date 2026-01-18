"""
Bot Control Views.

Interactive buttons for controlling trading bots.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord

from src.core import get_logger

if TYPE_CHECKING:
    from src.master import Master

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


async def build_bot_embed(master, bot_id: str) -> discord.Embed:
    """Build embed for bot details."""
    bot_info = master.get_bot(bot_id)
    if not bot_info:
        return discord.Embed(
            title="Bot Not Found",
            description=f"Bot `{bot_id}` not found",
            color=discord.Color.red(),
        )

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
    try:
        instance = master.registry.get_bot_instance(bot_id)
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
    except Exception as e:
        logger.debug(f"Could not get bot instance details: {e}")

    return embed


# =============================================================================
# Button Classes
# =============================================================================


class StartButton(discord.ui.Button):
    """Start bot button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Start",
            style=discord.ButtonStyle.success,
            emoji="▶️",
            custom_id=f"start_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            result = await self.master.start_bot(self.bot_id)

            if result.success:
                # Update embed and view
                embed = await build_bot_embed(self.master, self.bot_id)
                bot_info = self.master.get_bot(self.bot_id)
                state = bot_info.state.value if bot_info and hasattr(bot_info.state, "value") else "running"
                view = BotControlView(self.bot_id, state, self.master)
                await interaction.edit_original_response(embed=embed, view=view)
            else:
                await interaction.followup.send(f"❌ Failed to start: {result.message}", ephemeral=True)

        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


class StopButton(discord.ui.Button):
    """Stop bot button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Stop",
            style=discord.ButtonStyle.danger,
            emoji="🛑",
            custom_id=f"stop_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        # Import here to avoid circular imports
        from .confirm_views import ConfirmStopView

        embed = discord.Embed(
            title="⚠️ Confirm Stop",
            description=f"Are you sure you want to stop bot `{self.bot_id}`?\n\n"
                        f"This will cancel all pending orders.",
            color=discord.Color.orange(),
        )

        view = ConfirmStopView(self.bot_id, self.master)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


class PauseButton(discord.ui.Button):
    """Pause bot button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Pause",
            style=discord.ButtonStyle.secondary,
            emoji="⏸️",
            custom_id=f"pause_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            result = await self.master.pause_bot(self.bot_id)

            if result.success:
                embed = await build_bot_embed(self.master, self.bot_id)
                bot_info = self.master.get_bot(self.bot_id)
                state = bot_info.state.value if bot_info and hasattr(bot_info.state, "value") else "paused"
                view = BotControlView(self.bot_id, state, self.master)
                await interaction.edit_original_response(embed=embed, view=view)
            else:
                await interaction.followup.send(f"❌ Failed to pause: {result.message}", ephemeral=True)

        except Exception as e:
            logger.error(f"Error pausing bot: {e}")
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


class ResumeButton(discord.ui.Button):
    """Resume bot button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Resume",
            style=discord.ButtonStyle.success,
            emoji="▶️",
            custom_id=f"resume_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            result = await self.master.resume_bot(self.bot_id)

            if result.success:
                embed = await build_bot_embed(self.master, self.bot_id)
                bot_info = self.master.get_bot(self.bot_id)
                state = bot_info.state.value if bot_info and hasattr(bot_info.state, "value") else "running"
                view = BotControlView(self.bot_id, state, self.master)
                await interaction.edit_original_response(embed=embed, view=view)
            else:
                await interaction.followup.send(f"❌ Failed to resume: {result.message}", ephemeral=True)

        except Exception as e:
            logger.error(f"Error resuming bot: {e}")
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


class RefreshButton(discord.ui.Button):
    """Refresh bot info button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Refresh",
            style=discord.ButtonStyle.secondary,
            emoji="🔄",
            custom_id=f"refresh_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        await interaction.response.defer()

        try:
            embed = await build_bot_embed(self.master, self.bot_id)
            bot_info = self.master.get_bot(self.bot_id)

            if bot_info:
                state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
                view = BotControlView(self.bot_id, state, self.master)
                await interaction.edit_original_response(embed=embed, view=view)
            else:
                await interaction.edit_original_response(embed=embed, view=None)

        except Exception as e:
            logger.error(f"Error refreshing bot: {e}")
            await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


class DeleteButton(discord.ui.Button):
    """Delete bot button."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(
            label="Delete",
            style=discord.ButtonStyle.danger,
            emoji="🗑️",
            custom_id=f"delete_{bot_id}",
        )
        self.bot_id = bot_id
        self.master = master

    async def callback(self, interaction: discord.Interaction):
        from .confirm_views import ConfirmDeleteView

        bot_info = self.master.get_bot(self.bot_id)
        if not bot_info:
            await interaction.response.send_message(f"Bot `{self.bot_id}` not found", ephemeral=True)
            return

        embed = discord.Embed(
            title="⚠️ Confirm Delete",
            description=f"Are you sure you want to delete bot `{self.bot_id}`?\n\n"
                        f"**This action cannot be undone.**",
            color=discord.Color.red(),
        )
        embed.add_field(name="Symbol", value=bot_info.symbol, inline=True)

        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
        embed.add_field(name="State", value=state, inline=True)

        view = ConfirmDeleteView(self.bot_id, self.master)
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


# =============================================================================
# View Classes
# =============================================================================


class BotControlView(discord.ui.View):
    """View with control buttons for a bot."""

    def __init__(self, bot_id: str, state: str, master: "Master"):
        super().__init__(timeout=300)
        self.bot_id = bot_id
        self.state = state
        self.master = master

        self._setup_buttons(state)

    def _setup_buttons(self, state: str):
        """Setup buttons based on bot state."""
        state_lower = state.lower() if isinstance(state, str) else state.value.lower()

        if state_lower in ("stopped", "registered", "error"):
            self.add_item(StartButton(self.bot_id, self.master))
        elif state_lower == "running":
            self.add_item(PauseButton(self.bot_id, self.master))
            self.add_item(StopButton(self.bot_id, self.master))
        elif state_lower == "paused":
            self.add_item(ResumeButton(self.bot_id, self.master))
            self.add_item(StopButton(self.bot_id, self.master))

        # Always add refresh button
        self.add_item(RefreshButton(self.bot_id, self.master))

    async def on_timeout(self):
        """Handle view timeout."""
        # Disable all buttons on timeout
        for item in self.children:
            if isinstance(item, discord.ui.Button):
                item.disabled = True


class BotControlWithDeleteView(BotControlView):
    """Bot control view with delete button (for admin use)."""

    def _setup_buttons(self, state: str):
        """Setup buttons including delete."""
        super()._setup_buttons(state)
        self.add_item(DeleteButton(self.bot_id, self.master))
