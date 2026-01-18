"""
Risk Commands.

Slash commands for risk management.
"""

from datetime import datetime, timezone

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


class RiskCommands(commands.Cog):
    """Commands for risk management."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @app_commands.command(name="risk", description="Show risk status")
    async def risk(self, interaction: discord.Interaction):
        """Show current risk status."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="Risk Status",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        risk_engine = getattr(self.bot, "_risk_engine", None)
        if risk_engine:
            try:
                status = risk_engine.get_status()
                if status:
                    level = status.level.name if hasattr(status.level, "name") else str(status.level)

                    # Color based on level
                    level_colors = {
                        "NORMAL": discord.Color.green(),
                        "WARNING": discord.Color.yellow(),
                        "RISK": discord.Color.orange(),
                        "DANGER": discord.Color.red(),
                        "CIRCUIT_BREAK": discord.Color.dark_red(),
                    }
                    embed.color = level_colors.get(level, discord.Color.blue())

                    embed.add_field(name="Risk Level", value=level, inline=True)

                    if status.capital:
                        embed.add_field(
                            name="Total Capital",
                            value=f"{status.capital.total_capital} USDT",
                            inline=True,
                        )

                    if status.drawdown:
                        embed.add_field(
                            name="Drawdown",
                            value=f"{float(status.drawdown.drawdown_pct) * 100:.2f}%",
                            inline=True,
                        )

                    if status.active_alerts:
                        alerts_text = "\n".join(
                            [f"- {a.message}" for a in status.active_alerts[:5]]
                        )
                        embed.add_field(
                            name="Active Alerts",
                            value=alerts_text or "None",
                            inline=False,
                        )
                else:
                    embed.add_field(name="Status", value="No status available", inline=False)
            except Exception as e:
                embed.add_field(name="Error", value=str(e), inline=False)
                embed.color = discord.Color.red()
        else:
            embed.add_field(name="Status", value="Risk engine not connected", inline=False)
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)

    # Emergency command group
    emergency_group = app_commands.Group(name="emergency", description="Emergency controls")

    @emergency_group.command(name="stop", description="Trigger emergency stop")
    async def emergency_stop(self, interaction: discord.Interaction):
        """Trigger emergency stop."""
        # Check admin permission
        trading_bot = self.bot
        if hasattr(trading_bot, "is_admin") and not trading_bot.is_admin(interaction.user):
            await interaction.response.send_message(
                "You don't have permission to use this command",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        risk_engine = getattr(self.bot, "_risk_engine", None)
        if risk_engine:
            try:
                await risk_engine.trigger_emergency("Discord emergency command")
                embed.title = "EMERGENCY STOP ACTIVATED"
                embed.description = "All trading has been stopped"
                embed.color = discord.Color.red()
            except Exception as e:
                embed.title = "Error"
                embed.description = str(e)
                embed.color = discord.Color.red()
        else:
            embed.title = "Error"
            embed.description = "Risk engine not connected"
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)

    @emergency_group.command(name="reset", description="Reset circuit breaker")
    async def emergency_reset(self, interaction: discord.Interaction):
        """Reset circuit breaker."""
        # Check admin permission
        trading_bot = self.bot
        if hasattr(trading_bot, "is_admin") and not trading_bot.is_admin(interaction.user):
            await interaction.response.send_message(
                "You don't have permission to use this command",
                ephemeral=True,
            )
            return

        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        risk_engine = getattr(self.bot, "_risk_engine", None)
        if risk_engine:
            try:
                result = await risk_engine.reset_circuit_breaker(force=True)
                if result:
                    embed.title = "Circuit Breaker Reset"
                    embed.description = "Circuit breaker has been reset"
                    embed.color = discord.Color.green()
                else:
                    embed.title = "Reset Failed"
                    embed.description = "Could not reset circuit breaker"
                    embed.color = discord.Color.red()
            except Exception as e:
                embed.title = "Error"
                embed.description = str(e)
                embed.color = discord.Color.red()
        else:
            embed.title = "Error"
            embed.description = "Risk engine not connected"
            embed.color = discord.Color.orange()

        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(RiskCommands(bot))
