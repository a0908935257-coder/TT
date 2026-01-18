"""
Status Commands.

Slash commands for checking system status.
"""

from datetime import datetime, timezone
from typing import Optional

import discord
from discord import app_commands
from discord.ext import commands

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_risk_color(level) -> discord.Color:
    """Get color for risk level."""
    level_str = level.name if hasattr(level, "name") else str(level).upper()
    mapping = {
        "NORMAL": discord.Color.green(),
        "WARNING": discord.Color.yellow(),
        "RISK": discord.Color.orange(),
        "DANGER": discord.Color.red(),
        "CIRCUIT_BREAK": discord.Color.dark_red(),
    }
    return mapping.get(level_str, discord.Color.blue())


def get_risk_emoji(level) -> str:
    """Get emoji for risk level."""
    level_str = level.name if hasattr(level, "name") else str(level).upper()
    mapping = {
        "NORMAL": "🟢",
        "WARNING": "🟡",
        "RISK": "🟠",
        "DANGER": "🔴",
        "CIRCUIT_BREAK": "⛔",
    }
    return mapping.get(level_str, "❓")


# =============================================================================
# Views
# =============================================================================


class DashboardView(discord.ui.View):
    """Dashboard view with refresh button."""

    def __init__(self, master):
        super().__init__(timeout=300)
        self.master = master

    @discord.ui.button(label="Refresh", style=discord.ButtonStyle.primary, emoji="🔄")
    async def refresh(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Refresh dashboard data."""
        await interaction.response.defer()

        try:
            embeds = build_dashboard_embeds(self.master)
            await interaction.edit_original_response(embeds=embeds, view=self)
        except Exception as e:
            await interaction.followup.send(f"Error refreshing: {e}", ephemeral=True)


# =============================================================================
# Embed Builders
# =============================================================================


def build_dashboard_embeds(master) -> list:
    """Build dashboard embeds."""
    embeds = []

    try:
        dashboard = master.get_dashboard_data()
        summary = dashboard.summary

        # Main dashboard embed
        main_embed = discord.Embed(
            title="📊 Dashboard",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        # Bot statistics
        main_embed.add_field(
            name="🤖 Bot Status",
            value=f"Running: {summary.running_bots}\n"
                  f"Paused: {summary.paused_bots}\n"
                  f"Error: {getattr(summary, 'error_bots', 0)}\n"
                  f"Total: {summary.total_bots}",
            inline=True,
        )

        # Fund status
        total_value = getattr(summary, "total_value", summary.total_investment)
        total_profit = getattr(summary, "total_profit", summary.today_profit)
        profit_rate = getattr(summary, "total_profit_rate", 0)

        main_embed.add_field(
            name="💰 Fund Status",
            value=f"Total Value: {float(total_value):,.2f} USDT\n"
                  f"Investment: {float(summary.total_investment):,.2f} USDT\n"
                  f"Total Profit: {float(total_profit):+,.2f} USDT\n"
                  f"Profit Rate: {float(profit_rate) * 100:+.2f}%",
            inline=True,
        )

        # Today's statistics
        main_embed.add_field(
            name="📈 Today",
            value=f"Profit: {float(summary.today_profit):+,.2f} USDT\n"
                  f"Trades: {summary.today_trades}",
            inline=True,
        )

        embeds.append(main_embed)

        # Bot list embed
        bots = getattr(dashboard, "bots", [])
        if bots:
            bots_embed = discord.Embed(
                title="🤖 Active Bots",
                color=discord.Color.green(),
            )

            for bot_info in bots[:9]:  # Limit to 9 for embed
                state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
                emoji = "🟢" if state.lower() == "running" else "🟡" if state.lower() == "paused" else "🔴"

                profit = getattr(bot_info, "profit", 0)
                profit_display = f"{float(profit):+.4f}" if profit else "0.00"

                bots_embed.add_field(
                    name=f"{emoji} {bot_info.bot_id}",
                    value=f"Symbol: {bot_info.symbol}\n"
                          f"State: {state}\n"
                          f"Profit: {profit_display} USDT",
                    inline=True,
                )

            embeds.append(bots_embed)

        # Alerts embed
        alerts = getattr(dashboard, "alerts", [])
        if alerts:
            alerts_embed = discord.Embed(
                title="⚠️ Active Alerts",
                color=discord.Color.orange(),
            )

            alert_text = "\n".join([f"• {a.message}" for a in alerts[:10]])
            alerts_embed.description = alert_text
            embeds.append(alerts_embed)

    except Exception as e:
        error_embed = discord.Embed(
            title="❌ Error",
            description=str(e),
            color=discord.Color.red(),
        )
        embeds.append(error_embed)

    return embeds


# =============================================================================
# Cog
# =============================================================================


class StatusCommands(commands.Cog):
    """Commands for status queries."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def master(self):
        """Get master instance."""
        return getattr(self.bot, "_master", None)

    @property
    def risk_engine(self):
        """Get risk engine instance."""
        return getattr(self.bot, "_risk_engine", None)

    # =========================================================================
    # /status
    # =========================================================================

    @app_commands.command(name="status", description="Show system status")
    async def status(self, interaction: discord.Interaction):
        """Show overall system status."""
        await interaction.response.defer()

        # Default color
        embed_color = discord.Color.blue()

        # Get risk status for color
        risk_status = None
        if self.risk_engine:
            try:
                risk_status = self.risk_engine.get_status()
                if risk_status:
                    embed_color = get_risk_color(risk_status.level)
            except Exception:
                pass

        embed = discord.Embed(
            title="📊 System Status",
            color=embed_color,
            timestamp=datetime.now(timezone.utc),
        )

        if not self.master:
            embed.description = "Master not connected"
            embed.color = discord.Color.orange()
            await interaction.followup.send(embed=embed)
            return

        try:
            # Get dashboard data
            dashboard = self.master.get_dashboard_data()
            summary = dashboard.summary

            # Bot statistics
            error_bots = getattr(summary, "error_bots", 0)
            embed.add_field(
                name="🤖 Bot Status",
                value=f"Running: {summary.running_bots}\n"
                      f"Paused: {summary.paused_bots}\n"
                      f"Error: {error_bots}\n"
                      f"Total: {summary.total_bots}",
                inline=True,
            )

            # Fund status
            total_value = getattr(summary, "total_value", summary.total_investment)
            total_profit = getattr(summary, "total_profit", summary.today_profit)
            profit_rate = getattr(summary, "total_profit_rate", 0)

            embed.add_field(
                name="💰 Fund Status",
                value=f"Total: {float(total_value):,.2f} USDT\n"
                      f"Investment: {float(summary.total_investment):,.2f} USDT\n"
                      f"Profit: {float(total_profit):+,.2f} USDT\n"
                      f"Rate: {float(profit_rate) * 100:+.2f}%",
                inline=True,
            )

            # Today's statistics
            embed.add_field(
                name="📈 Today",
                value=f"Profit: {float(summary.today_profit):+,.2f} USDT\n"
                      f"Trades: {summary.today_trades}",
                inline=True,
            )

            # Risk status
            if risk_status:
                level_name = risk_status.level.name if hasattr(risk_status.level, "name") else str(risk_status.level)
                risk_emoji = get_risk_emoji(risk_status.level)

                drawdown_pct = 0
                if risk_status.drawdown:
                    drawdown_pct = float(risk_status.drawdown.drawdown_pct) * 100

                circuit_status = "🟢 Normal"
                if risk_status.circuit_breaker and risk_status.circuit_breaker.is_triggered:
                    circuit_status = "🔴 Triggered"

                embed.add_field(
                    name="🛡️ Risk Status",
                    value=f"Level: {risk_emoji} {level_name}\n"
                          f"Drawdown: {drawdown_pct:.2f}%\n"
                          f"Circuit: {circuit_status}",
                    inline=True,
                )

        except Exception as e:
            logger.error(f"Error getting status: {e}")
            embed.add_field(name="Error", value=str(e), inline=False)
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /dashboard
    # =========================================================================

    @app_commands.command(name="dashboard", description="Show full dashboard")
    async def dashboard(self, interaction: discord.Interaction):
        """Show full dashboard with refresh button."""
        await interaction.response.defer()

        if not self.master:
            embed = discord.Embed(
                title="Dashboard",
                description="Master not connected",
                color=discord.Color.orange(),
            )
            await interaction.followup.send(embed=embed)
            return

        try:
            embeds = build_dashboard_embeds(self.master)
            view = DashboardView(self.master)
            await interaction.followup.send(embeds=embeds, view=view)

        except Exception as e:
            logger.error(f"Error building dashboard: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )
            await interaction.followup.send(embed=embed)

    # =========================================================================
    # /profit
    # =========================================================================

    @app_commands.command(name="profit", description="Show profit statistics")
    @app_commands.describe(days="Number of days (default: 7)")
    async def profit(self, interaction: discord.Interaction, days: int = 7):
        """Show profit statistics."""
        await interaction.response.defer()

        if not self.master:
            embed = discord.Embed(
                title="Profit Statistics",
                description="Master not connected",
                color=discord.Color.orange(),
            )
            await interaction.followup.send(embed=embed)
            return

        try:
            # Get dashboard for current stats
            dashboard = self.master.get_dashboard_data()
            summary = dashboard.summary

            total_profit = getattr(summary, "total_profit", summary.today_profit)

            embed = discord.Embed(
                title=f"📈 Profit Statistics ({days} days)",
                color=discord.Color.green() if float(total_profit) >= 0 else discord.Color.red(),
                timestamp=datetime.now(timezone.utc),
            )

            # Overview
            trade_count = getattr(summary, "total_trades", summary.today_trades)
            win_rate = getattr(summary, "win_rate", 0)

            embed.add_field(
                name="📊 Overview",
                value=f"Total Profit: {float(total_profit):+,.2f} USDT\n"
                      f"Total Trades: {trade_count}\n"
                      f"Win Rate: {float(win_rate) * 100:.1f}%",
                inline=False,
            )

            # Today's stats
            embed.add_field(
                name="📅 Today",
                value=f"Profit: {float(summary.today_profit):+,.2f} USDT\n"
                      f"Trades: {summary.today_trades}",
                inline=True,
            )

            # Investment info
            embed.add_field(
                name="💰 Investment",
                value=f"Total: {float(summary.total_investment):,.2f} USDT\n"
                      f"Active Bots: {summary.running_bots}",
                inline=True,
            )

            # Try to get profit history if available
            if hasattr(self.master, "get_profit_history"):
                try:
                    history = await self.master.get_profit_history(days)
                    if history:
                        # Daily breakdown
                        daily = getattr(history, "daily", [])
                        if daily:
                            daily_text = ""
                            for day in daily[:7]:  # Last 7 days
                                pnl = getattr(day, "pnl", 0)
                                trades = getattr(day, "trades", 0)
                                date_str = getattr(day, "date", "")
                                emoji = "📈" if pnl >= 0 else "📉"
                                daily_text += f"{emoji} {date_str}: {float(pnl):+,.2f} ({trades})\n"

                            if daily_text:
                                embed.add_field(
                                    name="📅 Daily Breakdown",
                                    value=daily_text,
                                    inline=False,
                                )

                        # Best/worst day
                        best_day = getattr(history, "best_day", None)
                        worst_day = getattr(history, "worst_day", None)

                        if best_day:
                            embed.add_field(
                                name="🏆 Best Day",
                                value=f"{best_day.date}: {float(best_day.pnl):+,.2f}",
                                inline=True,
                            )

                        if worst_day:
                            embed.add_field(
                                name="📉 Worst Day",
                                value=f"{worst_day.date}: {float(worst_day.pnl):+,.2f}",
                                inline=True,
                            )
                except Exception:
                    pass  # History not available

        except Exception as e:
            logger.error(f"Error getting profit stats: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(StatusCommands(bot))
