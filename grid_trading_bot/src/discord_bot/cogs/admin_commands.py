"""
Admin Commands.

Slash commands for system administration.
"""

from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


def is_admin():
    """Check if user is admin."""

    async def predicate(interaction: discord.Interaction) -> bool:
        trading_bot = interaction.client
        if hasattr(trading_bot, "is_admin"):
            return trading_bot.is_admin(interaction.user)
        # Fallback to administrator permission
        return interaction.user.guild_permissions.administrator

    return app_commands.check(predicate)


class AdminCommands(commands.Cog):
    """Commands for system administration."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="sync", description="Sync slash commands")
    @is_admin()
    async def sync_commands(self, interaction: discord.Interaction):
        """Sync slash commands to guild."""
        await interaction.response.defer(ephemeral=True)

        try:
            config = getattr(self.bot, "_config", None)
            if config and config.guild_id:
                guild = discord.Object(id=config.guild_id)
                self.bot.tree.copy_global_to(guild=guild)
                synced = await self.bot.tree.sync(guild=guild)
                await interaction.followup.send(
                    f"Synced {len(synced)} commands",
                    ephemeral=True,
                )
            else:
                synced = await self.bot.tree.sync()
                await interaction.followup.send(
                    f"Synced {len(synced)} commands globally",
                    ephemeral=True,
                )
        except Exception as e:
            await interaction.followup.send(f"Error: {e}", ephemeral=True)

    @app_commands.command(name="health", description="Run health check on all bots")
    @is_admin()
    async def health_check(self, interaction: discord.Interaction):
        """Run health check on all bots."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="Health Check",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        master = getattr(self.bot, "_master", None)
        if master:
            try:
                results = await master.health_check_all()
                if results:
                    for result in results[:10]:  # Limit to 10
                        status = result.status.value if hasattr(result.status, "value") else str(result.status)
                        embed.add_field(
                            name=result.bot_id,
                            value=f"Status: {status}\n{result.message[:50]}",
                            inline=True,
                        )
                else:
                    embed.description = "No bots to check"
            except Exception as e:
                embed.description = f"Error: {e}"
                embed.color = discord.Color.red()
        else:
            embed.description = "Master not connected"
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="restart", description="Restart a bot")
    @app_commands.describe(bot_id="Bot ID to restart")
    @is_admin()
    async def restart_bot(self, interaction: discord.Interaction, bot_id: str):
        """Restart a bot (stop then start)."""
        await interaction.response.defer()

        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.followup.send("Master not connected")
            return

        try:
            # Stop first
            stop_result = await master.stop_bot(bot_id)
            if not stop_result.success:
                await interaction.followup.send(f"Failed to stop: {stop_result.message}")
                return

            # Then start
            start_result = await master.start_bot(bot_id)
            if start_result.success:
                embed = discord.Embed(
                    title="Bot Restarted",
                    description=f"Bot `{bot_id}` has been restarted",
                    color=discord.Color.green(),
                )
            else:
                embed = discord.Embed(
                    title="Restart Failed",
                    description=f"Stopped but failed to start: {start_result.message}",
                    color=discord.Color.red(),
                )
        except Exception as e:
            embed = discord.Embed(
                title="Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="stopall", description="Stop all running bots")
    @is_admin()
    async def stop_all(self, interaction: discord.Interaction):
        """Stop all running bots."""
        await interaction.response.defer()

        master = getattr(self.bot, "_master", None)
        if not master:
            await interaction.followup.send("Master not connected")
            return

        try:
            from src.master import BotCommander

            commander = BotCommander(master.registry)
            stopped = await commander.stop_all()

            embed = discord.Embed(
                title="All Bots Stopped",
                description=f"Stopped {len(stopped)} bots",
                color=discord.Color.orange(),
                timestamp=datetime.now(timezone.utc),
            )

            if stopped:
                embed.add_field(
                    name="Stopped Bots",
                    value="\n".join(stopped[:10]),
                    inline=False,
                )
        except Exception as e:
            embed = discord.Embed(
                title="Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)

    @sync_commands.error
    @health_check.error
    @restart_bot.error
    @stop_all.error
    async def admin_error(self, interaction: discord.Interaction, error: Exception):
        """Handle admin command errors."""
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message(
                "You don't have permission to use this command",
                ephemeral=True,
            )
        else:
            logger.error(f"Admin command error: {error}")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"An error occurred: {error}",
                    ephemeral=True,
                )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(AdminCommands(bot))
