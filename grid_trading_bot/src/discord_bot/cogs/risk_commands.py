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


class ConfirmEmergencyView(discord.ui.View):
    """Confirmation view for emergency stop."""

    def __init__(self, risk_engine, master=None):
        super().__init__(timeout=60)
        self.risk_engine = risk_engine
        self.master = master

    @discord.ui.button(label="Confirm Emergency Stop", style=discord.ButtonStyle.danger, emoji="🚨")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm emergency stop."""
        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        try:
            if self.risk_engine:
                await self.risk_engine.trigger_emergency("Discord emergency command")

            # Also stop all bots if master is available
            if self.master:
                try:
                    from src.master import BotCommander
                    commander = BotCommander(self.master.registry)
                    await commander.stop_all()
                except Exception as e:
                    logger.warning(f"Could not stop all bots: {e}")

            embed.title = "🚨 EMERGENCY STOP ACTIVATED"
            embed.description = (
                "**All trading has been stopped**\n\n"
                "• All pending orders cancelled\n"
                "• All positions closed\n"
                "• All bots stopped\n\n"
                "Use `/emergency reset` to recover after cooldown"
            )
            embed.color = discord.Color.red()

        except Exception as e:
            embed.title = "❌ Emergency Stop Failed"
            embed.description = str(e)
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel emergency stop."""
        await interaction.response.send_message("Emergency stop cancelled", ephemeral=True)
        self.stop()


# =============================================================================
# Cog
# =============================================================================


class RiskCommands(commands.Cog):
    """Commands for risk management."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot

    @property
    def risk_engine(self):
        """Get risk engine instance."""
        return getattr(self.bot, "_risk_engine", None)

    @property
    def master(self):
        """Get master instance."""
        return getattr(self.bot, "_master", None)

    # =========================================================================
    # /risk
    # =========================================================================

    @app_commands.command(name="risk", description="Show risk status")
    async def risk(self, interaction: discord.Interaction):
        """Show current risk status."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="🛡️ Risk Status",
            color=discord.Color.blue(),
            timestamp=datetime.now(timezone.utc),
        )

        if not self.risk_engine:
            embed.description = "Risk engine not connected"
            embed.color = discord.Color.orange()
            await interaction.followup.send(embed=embed)
            return

        try:
            status = self.risk_engine.get_status()
            if not status:
                embed.description = "No status available"
                await interaction.followup.send(embed=embed)
                return

            # Get level info
            level_name = status.level.name if hasattr(status.level, "name") else str(status.level)
            level_emoji = get_risk_emoji(status.level)
            embed.color = get_risk_color(status.level)

            # Risk level
            embed.add_field(
                name="Risk Level",
                value=f"{level_emoji} {level_name}",
                inline=True,
            )

            # Capital info
            if status.capital:
                total_capital = float(status.capital.total_capital)
                initial_capital = float(getattr(status.capital, "initial_capital", total_capital))
                available = float(getattr(status.capital, "available_balance", total_capital))

                # Calculate change percentage
                if initial_capital > 0:
                    change_pct = (total_capital - initial_capital) / initial_capital
                else:
                    change_pct = 0

                embed.add_field(
                    name="Capital",
                    value=f"Total: {total_capital:,.2f} USDT\n"
                          f"Available: {available:,.2f} USDT\n"
                          f"Change: {change_pct:+.2%}",
                    inline=True,
                )

            # Drawdown info
            if status.drawdown:
                current_dd = float(status.drawdown.drawdown_pct) * 100
                max_dd = float(getattr(status.drawdown, "max_drawdown_pct", status.drawdown.drawdown_pct)) * 100
                peak = float(getattr(status.drawdown, "peak_value", 0))

                embed.add_field(
                    name="Drawdown",
                    value=f"Current: {current_dd:.2f}%\n"
                          f"Max: {max_dd:.2f}%\n"
                          f"Peak: {peak:,.2f} USDT",
                    inline=True,
                )

            # Daily P&L
            daily_pnl = getattr(status, "daily_pnl", None)
            if daily_pnl:
                pnl = float(getattr(daily_pnl, "pnl", 0))
                pnl_pct = float(getattr(daily_pnl, "pnl_pct", 0))
                pnl_emoji = "📈" if pnl >= 0 else "📉"

                embed.add_field(
                    name=f"{pnl_emoji} Today's P&L",
                    value=f"{pnl:+,.2f} USDT\n({pnl_pct:+.2%})",
                    inline=True,
                )

            # Circuit breaker status
            if status.circuit_breaker:
                cb = status.circuit_breaker
                if cb.is_triggered:
                    reason = getattr(cb, "trigger_reason", "Unknown")
                    cooldown_until = getattr(cb, "cooldown_until", None)
                    cooldown_str = cooldown_until.strftime("%H:%M:%S") if cooldown_until else "N/A"

                    cb_text = (
                        f"🔴 **TRIGGERED**\n"
                        f"Reason: {reason}\n"
                        f"Cooldown: {cooldown_str}"
                    )
                else:
                    cb_text = "🟢 Normal"

                embed.add_field(
                    name="Circuit Breaker",
                    value=cb_text,
                    inline=True,
                )

            # Statistics
            stats = getattr(status, "statistics", None)
            if stats:
                checks = getattr(stats, "total_checks", 0)
                violations = getattr(stats, "violations", 0)
                triggers = getattr(stats, "circuit_breaker_triggers", 0)

                embed.add_field(
                    name="📊 Statistics",
                    value=f"Checks: {checks}\n"
                          f"Violations: {violations}\n"
                          f"CB Triggers: {triggers}",
                    inline=True,
                )

            # Active alerts
            if status.active_alerts:
                alerts_text = "\n".join([f"• {a.message}" for a in status.active_alerts[:5]])
                embed.add_field(
                    name="⚠️ Active Alerts",
                    value=alerts_text,
                    inline=False,
                )

        except Exception as e:
            logger.error(f"Error getting risk status: {e}")
            embed.description = f"Error: {e}"
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # /emergency command group
    # =========================================================================

    emergency_group = app_commands.Group(name="emergency", description="Emergency controls")

    @emergency_group.command(name="stop", description="Trigger emergency stop")
    @is_admin()
    async def emergency_stop(self, interaction: discord.Interaction):
        """Trigger emergency stop with confirmation."""
        if not self.risk_engine and not self.master:
            await interaction.response.send_message(
                "Neither risk engine nor master is connected",
                ephemeral=True,
            )
            return

        # Show confirmation dialog
        embed = discord.Embed(
            title="🚨 Emergency Stop Confirmation",
            description=(
                "**Are you sure you want to trigger emergency stop?**\n\n"
                "This action will:\n"
                "• Cancel all pending orders\n"
                "• Close all positions\n"
                "• Stop all trading bots\n\n"
                "**⚠️ This action cannot be automatically reversed**"
            ),
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc),
        )

        view = ConfirmEmergencyView(self.risk_engine, self.master)
        await interaction.response.send_message(embed=embed, view=view)

    @emergency_group.command(name="reset", description="Reset circuit breaker")
    @is_admin()
    async def emergency_reset(self, interaction: discord.Interaction):
        """Reset circuit breaker."""
        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        if not self.risk_engine:
            embed.title = "❌ Error"
            embed.description = "Risk engine not connected"
            embed.color = discord.Color.orange()
            await interaction.followup.send(embed=embed)
            return

        try:
            result = await self.risk_engine.reset_circuit_breaker(force=True)

            if result:
                embed.title = "✅ Circuit Breaker Reset"
                embed.description = (
                    "Circuit breaker has been reset.\n\n"
                    "The system is now back to normal operation.\n"
                    "You can restart bots with `/bot start`"
                )
                embed.color = discord.Color.green()
            else:
                embed.title = "❌ Reset Failed"
                embed.description = (
                    "Could not reset circuit breaker.\n\n"
                    "Possible reasons:\n"
                    "• Cooldown period not finished\n"
                    "• Risk conditions still active"
                )
                embed.color = discord.Color.red()

        except Exception as e:
            error_msg = str(e)
            if "cooldown" in error_msg.lower():
                embed.title = "⏳ Cooldown Active"
                embed.description = f"Cannot reset: {error_msg}"
            else:
                embed.title = "❌ Error"
                embed.description = str(e)
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)

    @emergency_group.command(name="status", description="Show emergency status")
    async def emergency_status(self, interaction: discord.Interaction):
        """Show current emergency/circuit breaker status."""
        await interaction.response.defer()

        embed = discord.Embed(
            title="🚨 Emergency Status",
            timestamp=datetime.now(timezone.utc),
        )

        if not self.risk_engine:
            embed.description = "Risk engine not connected"
            embed.color = discord.Color.orange()
            await interaction.followup.send(embed=embed)
            return

        try:
            status = self.risk_engine.get_status()

            if status and status.circuit_breaker:
                cb = status.circuit_breaker

                if cb.is_triggered:
                    embed.color = discord.Color.red()
                    embed.description = "**⚠️ CIRCUIT BREAKER ACTIVE**"

                    reason = getattr(cb, "trigger_reason", "Unknown")
                    triggered_at = getattr(cb, "triggered_at", None)
                    cooldown_until = getattr(cb, "cooldown_until", None)

                    embed.add_field(name="Status", value="🔴 Triggered", inline=True)
                    embed.add_field(name="Reason", value=reason, inline=True)

                    if triggered_at:
                        embed.add_field(
                            name="Triggered At",
                            value=triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
                            inline=True,
                        )

                    if cooldown_until:
                        now = datetime.now(timezone.utc)
                        if cooldown_until > now:
                            remaining = cooldown_until - now
                            mins = int(remaining.total_seconds() // 60)
                            secs = int(remaining.total_seconds() % 60)
                            embed.add_field(
                                name="Cooldown Remaining",
                                value=f"{mins}m {secs}s",
                                inline=True,
                            )
                        else:
                            embed.add_field(
                                name="Cooldown",
                                value="✅ Finished - can reset",
                                inline=True,
                            )
                else:
                    embed.color = discord.Color.green()
                    embed.description = "**✅ System Normal**"
                    embed.add_field(name="Status", value="🟢 Normal", inline=True)
                    embed.add_field(name="Circuit Breaker", value="Not triggered", inline=True)
            else:
                embed.description = "No circuit breaker status available"
                embed.color = discord.Color.blue()

        except Exception as e:
            embed.description = f"Error: {e}"
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)

    # =========================================================================
    # Error Handlers
    # =========================================================================

    @emergency_stop.error
    @emergency_reset.error
    async def emergency_error(self, interaction: discord.Interaction, error: Exception):
        """Handle emergency command errors."""
        if isinstance(error, app_commands.CheckFailure):
            await interaction.response.send_message(
                "You don't have permission to use this command",
                ephemeral=True,
            )
        else:
            logger.error(f"Emergency command error: {error}")
            if not interaction.response.is_done():
                await interaction.response.send_message(
                    f"An error occurred: {error}",
                    ephemeral=True,
                )


async def setup(bot: commands.Bot):
    """Setup function for loading the cog."""
    await bot.add_cog(RiskCommands(bot))
