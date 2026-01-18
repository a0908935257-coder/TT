"""
Bot Management Commands.

Slash commands for managing trading bots.
"""

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


class BotCommands(commands.Cog):
    """Commands for bot management."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    # Bot command group
    bot_group = app_commands.Group(name="bot", description="Bot management commands")

    @bot_group.command(name="list", description="List all trading bots")
    async def bot_list(self, interaction: discord.Interaction):
        """List all trading bots."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="Trading Bots",
            color=discord.Color.blue(),
        )

        master = getattr(self.bot, "_master", None)
        if master:
            try:
                bots = master.get_all_bots()
                if bots:
                    for bot_info in bots[:10]:  # Limit to 10
                        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
                        embed.add_field(
                            name=bot_info.bot_id,
                            value=f"Symbol: {bot_info.symbol}\nState: {state}",
                            inline=True,
                        )
                else:
                    embed.description = "No bots registered"
            except Exception as e:
                embed.description = f"Error: {e}"
                embed.color = discord.Color.red()
        else:
            embed.description = "Master not connected"
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)

    @bot_group.command(name="start", description="Start a trading bot")
    @app_commands.describe(bot_id="Bot ID to start")
    async def bot_start(self, interaction: discord.Interaction, bot_id: str):
        """Start a trading bot."""
        await interaction.response.defer()

        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await master.start_bot(bot_id)
            if result.success:
                embed = discord.Embed(
                    title="Bot Started",
                    description=f"Bot `{bot_id}` started successfully",
                    color=discord.Color.green(),
                )
            else:
                embed = discord.Embed(
                    title="Start Failed",
                    description=result.message,
                    color=discord.Color.red(),
                )
        except Exception as e:
            embed = discord.Embed(
                title="Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    @bot_group.command(name="stop", description="Stop a trading bot")
    @app_commands.describe(bot_id="Bot ID to stop")
    async def bot_stop(self, interaction: discord.Interaction, bot_id: str):
        """Stop a trading bot."""
        await interaction.response.defer()

        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.followup.send("Master not connected")
            return

        try:
            result = await master.stop_bot(bot_id)
            if result.success:
                embed = discord.Embed(
                    title="Bot Stopped",
                    description=f"Bot `{bot_id}` stopped",
                    color=discord.Color.orange(),
                )
            else:
                embed = discord.Embed(
                    title="Stop Failed",
                    description=result.message,
                    color=discord.Color.red(),
                )
        except Exception as e:
            embed = discord.Embed(
                title="Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    @bot_group.command(name="pause", description="Pause a trading bot")
    @app_commands.describe(bot_id="Bot ID to pause")
    async def bot_pause(self, interaction: discord.Interaction, bot_id: str):
        """Pause a trading bot."""
        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.response.send_message("Master not connected")
            return

        try:
            result = await master.pause_bot(bot_id)
            if result.success:
                await interaction.response.send_message(f"Bot `{bot_id}` paused")
            else:
                await interaction.response.send_message(f"Failed: {result.message}")
        except Exception as e:
            await interaction.response.send_message(f"Error: {e}")

    @bot_group.command(name="resume", description="Resume a paused bot")
    @app_commands.describe(bot_id="Bot ID to resume")
    async def bot_resume(self, interaction: discord.Interaction, bot_id: str):
        """Resume a paused bot."""
        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.response.send_message("Master not connected")
            return

        try:
            result = await master.resume_bot(bot_id)
            if result.success:
                await interaction.response.send_message(f"Bot `{bot_id}` resumed")
            else:
                await interaction.response.send_message(f"Failed: {result.message}")
        except Exception as e:
            await interaction.response.send_message(f"Error: {e}")


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(BotCommands(bot))
