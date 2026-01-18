"""
Status Commands.

Slash commands for checking system status.
"""

from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


class StatusCommands(commands.Cog):
    """Commands for status queries."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="status", description="Show system status")
    async def status(self, interaction: discord.Interaction):
        """Show overall system status."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="System Status",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        master = getattr(self.bot, "_master", None)
        if master:
            try:
                bots = master.get_all_bots()
                running = sum(1 for b in bots if hasattr(b, "state") and str(b.state.value).lower() == "running")

                embed.add_field(name="Total Bots", value=str(len(bots)), inline=True)
                embed.add_field(name="Running", value=str(running), inline=True)
                embed.add_field(name="Stopped", value=str(len(bots) - running), inline=True)
            except Exception as e:
                embed.add_field(name="Error", value=str(e), inline=False)
        else:
            embed.add_field(name="Status", value="Master not connected", inline=False)

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="dashboard", description="Show dashboard summary")
    async def dashboard(self, interaction: discord.Interaction):
        """Show dashboard summary."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="Dashboard",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc),
        )

        master = getattr(self.bot, "_master", None)
        if master:
            try:
                dashboard = master.get_dashboard_data()
                summary = dashboard.summary

                embed.add_field(name="Total Bots", value=str(summary.total_bots), inline=True)
                embed.add_field(name="Running", value=str(summary.running_bots), inline=True)
                embed.add_field(name="Paused", value=str(summary.paused_bots), inline=True)
                embed.add_field(
                    name="Total Investment",
                    value=f"{summary.total_investment} USDT",
                    inline=True,
                )
                embed.add_field(
                    name="Today's Profit",
                    value=f"{summary.today_profit} USDT",
                    inline=True,
                )
                embed.add_field(
                    name="Today's Trades",
                    value=str(summary.today_trades),
                    inline=True,
                )
            except Exception as e:
                embed.add_field(name="Error", value=str(e), inline=False)
                embed.color = discord.Color.red()
        else:
            embed.add_field(name="Status", value="Master not connected", inline=False)
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)

    @app_commands.command(name="profit", description="Show profit statistics")
    @app_commands.describe(days="Number of days (default: 7)")
    async def profit(self, interaction: discord.Interaction, days: int = 7):
        """Show profit statistics."""
        await interaction.response.defer()

        embed = discord.Embed(
            title=f"Profit Statistics ({days} days)",
            color=discord.Color.gold(),
            timestamp=datetime.now(timezone.utc),
        )

        master = getattr(self.bot, "_master", None)
        if master:
            try:
                dashboard = master.get_dashboard_data()
                summary = dashboard.summary

                embed.add_field(
                    name="Today's Profit",
                    value=f"{summary.today_profit} USDT",
                    inline=True,
                )
                embed.add_field(
                    name="Today's Trades",
                    value=str(summary.today_trades),
                    inline=True,
                )
                embed.add_field(
                    name="Total Investment",
                    value=f"{summary.total_investment} USDT",
                    inline=True,
                )
            except Exception as e:
                embed.add_field(name="Error", value=str(e), inline=False)
                embed.color = discord.Color.red()
        else:
            embed.add_field(name="Status", value="Master not connected", inline=False)
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(StatusCommands(bot))
