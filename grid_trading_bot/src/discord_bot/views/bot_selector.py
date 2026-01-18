"""
Bot Selector Views.

Dropdown select menus for selecting bots.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable, List, Optional

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


# =============================================================================
# Selector Classes
# =============================================================================


class BotSelector(discord.ui.Select):
    """Dropdown selector for bots."""

    def __init__(
        self,
        bots: List,
        placeholder: str = "Select a bot...",
        on_select: Optional[Callable] = None,
    ):
        options = []
        for bot in bots[:25]:  # Discord limit
            state = bot.state.value if hasattr(bot.state, "value") else str(bot.state)
            emoji = get_status_emoji(bot.state)

            options.append(
                discord.SelectOption(
                    label=bot.bot_id[:100],  # Discord label limit
                    description=f"{bot.symbol} - {state}"[:100],  # Discord desc limit
                    emoji=emoji,
                    value=bot.bot_id,
                )
            )

        if not options:
            options.append(
                discord.SelectOption(
                    label="No bots available",
                    value="none",
                    description="Create a bot with /bot create",
                )
            )

        super().__init__(
            placeholder=placeholder,
            options=options,
            min_values=1,
            max_values=1,
        )
        self._on_select = on_select

    async def callback(self, interaction: discord.Interaction):
        """Handle selection."""
        if self.values[0] == "none":
            await interaction.response.send_message(
                "No bots available. Use `/bot create` to create one.",
                ephemeral=True,
            )
            return

        if self._on_select:
            await self._on_select(interaction, self.values[0])
        else:
            await interaction.response.send_message(
                f"Selected bot: `{self.values[0]}`",
                ephemeral=True,
            )


class BotSelectorByState(discord.ui.Select):
    """Dropdown selector filtered by bot state."""

    def __init__(
        self,
        bots: List,
        states: List[str],
        placeholder: str = "Select a bot...",
        on_select: Optional[Callable] = None,
    ):
        # Filter bots by state
        filtered_bots = []
        for bot in bots:
            state = bot.state.value if hasattr(bot.state, "value") else str(bot.state)
            if state.lower() in [s.lower() for s in states]:
                filtered_bots.append(bot)

        options = []
        for bot in filtered_bots[:25]:
            state = bot.state.value if hasattr(bot.state, "value") else str(bot.state)
            emoji = get_status_emoji(bot.state)

            options.append(
                discord.SelectOption(
                    label=bot.bot_id[:100],
                    description=f"{bot.symbol} - {state}"[:100],
                    emoji=emoji,
                    value=bot.bot_id,
                )
            )

        if not options:
            options.append(
                discord.SelectOption(
                    label="No bots with selected state",
                    value="none",
                    description=f"States: {', '.join(states)}",
                )
            )

        super().__init__(
            placeholder=placeholder,
            options=options,
            min_values=1,
            max_values=1,
        )
        self._on_select = on_select

    async def callback(self, interaction: discord.Interaction):
        """Handle selection."""
        if self.values[0] == "none":
            await interaction.response.send_message(
                "No bots available with the required state.",
                ephemeral=True,
            )
            return

        if self._on_select:
            await self._on_select(interaction, self.values[0])
        else:
            await interaction.response.send_message(
                f"Selected bot: `{self.values[0]}`",
                ephemeral=True,
            )


class MultipleBotsSelector(discord.ui.Select):
    """Dropdown for selecting multiple bots."""

    def __init__(
        self,
        bots: List,
        placeholder: str = "Select bots...",
        max_values: int = 5,
        on_select: Optional[Callable] = None,
    ):
        options = []
        for bot in bots[:25]:
            state = bot.state.value if hasattr(bot.state, "value") else str(bot.state)
            emoji = get_status_emoji(bot.state)

            options.append(
                discord.SelectOption(
                    label=bot.bot_id[:100],
                    description=f"{bot.symbol} - {state}"[:100],
                    emoji=emoji,
                    value=bot.bot_id,
                )
            )

        if not options:
            options.append(
                discord.SelectOption(
                    label="No bots available",
                    value="none",
                )
            )
            max_values = 1

        super().__init__(
            placeholder=placeholder,
            options=options,
            min_values=1,
            max_values=min(max_values, len(options)),
        )
        self._on_select = on_select

    async def callback(self, interaction: discord.Interaction):
        """Handle selection."""
        if "none" in self.values:
            await interaction.response.send_message(
                "No bots available.",
                ephemeral=True,
            )
            return

        if self._on_select:
            await self._on_select(interaction, self.values)
        else:
            await interaction.response.send_message(
                f"Selected bots: {', '.join(self.values)}",
                ephemeral=True,
            )


# =============================================================================
# View Classes
# =============================================================================


class BotSelectorView(discord.ui.View):
    """View with bot selector dropdown."""

    def __init__(
        self,
        bots: List,
        master: "Master",
        placeholder: str = "Select a bot...",
    ):
        super().__init__(timeout=180)
        self.master = master

        async def on_select(interaction: discord.Interaction, bot_id: str):
            await self._show_bot_detail(interaction, bot_id)

        self.add_item(BotSelector(bots, placeholder, on_select))

    async def _show_bot_detail(self, interaction: discord.Interaction, bot_id: str):
        """Show bot detail when selected."""
        from .bot_controls import BotControlView, build_bot_embed

        await interaction.response.defer()

        embed = await build_bot_embed(self.master, bot_id)
        bot_info = self.master.get_bot(bot_id)

        if bot_info:
            state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
            view = BotControlView(bot_id, state, self.master)
            await interaction.followup.send(embed=embed, view=view)
        else:
            await interaction.followup.send(embed=embed)


class BotActionSelectorView(discord.ui.View):
    """View for selecting bots and performing an action."""

    def __init__(
        self,
        bots: List,
        master: "Master",
        action: str,  # "start", "stop", "pause", "resume"
        placeholder: str = "Select bots to {action}...",
    ):
        super().__init__(timeout=180)
        self.master = master
        self.action = action

        # Filter bots based on action
        state_filters = {
            "start": ["stopped", "registered", "error"],
            "stop": ["running", "paused"],
            "pause": ["running"],
            "resume": ["paused"],
        }

        async def on_select(interaction: discord.Interaction, bot_ids: List[str]):
            await self._perform_action(interaction, bot_ids)

        self.add_item(
            BotSelectorByState(
                bots,
                states=state_filters.get(action, []),
                placeholder=placeholder.format(action=action),
                on_select=lambda i, bid: on_select(i, [bid]),
            )
        )

    async def _perform_action(self, interaction: discord.Interaction, bot_ids: List[str]):
        """Perform the action on selected bots."""
        await interaction.response.defer()

        results = []
        for bot_id in bot_ids:
            try:
                if self.action == "start":
                    result = await self.master.start_bot(bot_id)
                elif self.action == "stop":
                    result = await self.master.stop_bot(bot_id)
                elif self.action == "pause":
                    result = await self.master.pause_bot(bot_id)
                elif self.action == "resume":
                    result = await self.master.resume_bot(bot_id)
                else:
                    result = type("Result", (), {"success": False, "message": "Unknown action"})()

                emoji = "✅" if result.success else "❌"
                results.append(f"{emoji} `{bot_id}`: {result.message if not result.success else 'OK'}")

            except Exception as e:
                results.append(f"❌ `{bot_id}`: {e}")

        embed = discord.Embed(
            title=f"{self.action.title()} Results",
            description="\n".join(results),
            color=discord.Color.green() if all("✅" in r for r in results) else discord.Color.orange(),
            timestamp=datetime.now(timezone.utc),
        )

        await interaction.followup.send(embed=embed)
