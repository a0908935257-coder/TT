"""
Confirmation Views.

Views for confirming dangerous operations.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord

from src.core import get_logger

if TYPE_CHECKING:
    from src.master import Master
    from src.risk import RiskEngine

logger = get_logger(__name__)


# =============================================================================
# Confirm Stop View
# =============================================================================


class ConfirmStopView(discord.ui.View):
    """Confirmation view for stopping a bot."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(timeout=60)
        self.bot_id = bot_id
        self.master = master

    @discord.ui.button(label="Confirm Stop", style=discord.ButtonStyle.danger, emoji="🛑")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm stop action."""
        await interaction.response.defer()

        try:
            result = await self.master.stop_bot(self.bot_id)

            if result.success:
                embed = discord.Embed(
                    title="🛑 Bot Stopped",
                    description=f"Bot `{self.bot_id}` has been stopped successfully",
                    color=discord.Color.orange(),
                    timestamp=datetime.now(timezone.utc),
                )

                # Show final stats if available
                if result.data:
                    stats = result.data
                    if "total_trades" in stats:
                        embed.add_field(
                            name="Final Statistics",
                            value=f"Total Trades: {stats.get('total_trades', 0)}\n"
                                  f"Total Profit: {stats.get('total_profit', 0):.4f} USDT",
                            inline=False,
                        )
            else:
                embed = discord.Embed(
                    title="❌ Stop Failed",
                    description=result.message,
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
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel stop action."""
        await interaction.response.send_message("Stop cancelled", ephemeral=True)
        self.stop()


# =============================================================================
# Confirm Delete View
# =============================================================================


class ConfirmDeleteView(discord.ui.View):
    """Confirmation view for deleting a bot."""

    def __init__(self, bot_id: str, master: "Master"):
        super().__init__(timeout=60)
        self.bot_id = bot_id
        self.master = master

    @discord.ui.button(label="Confirm Delete", style=discord.ButtonStyle.danger, emoji="🗑️")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm delete action."""
        await interaction.response.defer()

        try:
            result = await self.master.delete_bot(self.bot_id)

            if result.success:
                embed = discord.Embed(
                    title="🗑️ Bot Deleted",
                    description=f"Bot `{self.bot_id}` has been deleted",
                    color=discord.Color.red(),
                    timestamp=datetime.now(timezone.utc),
                )
            else:
                embed = discord.Embed(
                    title="❌ Delete Failed",
                    description=result.message,
                    color=discord.Color.red(),
                )

        except Exception as e:
            logger.error(f"Error deleting bot: {e}")
            embed = discord.Embed(
                title="❌ Error",
                description=str(e),
                color=discord.Color.red(),
            )

        await interaction.followup.send(embed=embed)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel delete action."""
        await interaction.response.send_message("Delete cancelled", ephemeral=True)
        self.stop()


# =============================================================================
# Confirm Emergency View
# =============================================================================


class ConfirmEmergencyView(discord.ui.View):
    """Confirmation view for emergency stop."""

    def __init__(
        self,
        risk_engine: Optional["RiskEngine"] = None,
        master: Optional["Master"] = None,
    ):
        super().__init__(timeout=30)  # Short timeout for emergency actions
        self.risk_engine = risk_engine
        self.master = master

    @discord.ui.button(
        label="CONFIRM EMERGENCY STOP",
        style=discord.ButtonStyle.danger,
        emoji="🚨",
    )
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm emergency stop."""
        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        try:
            # Trigger emergency stop
            if self.risk_engine:
                await self.risk_engine.trigger_emergency("Discord emergency command")

            # Stop all bots
            stopped_bots = []
            if self.master:
                try:
                    from src.master import BotCommander

                    commander = BotCommander(self.master.registry)
                    stopped_bots = await commander.stop_all()
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

            if stopped_bots:
                embed.add_field(
                    name="Stopped Bots",
                    value="\n".join(stopped_bots[:10]) if stopped_bots else "None",
                    inline=False,
                )

        except Exception as e:
            logger.error(f"Error in emergency stop: {e}")
            embed.title = "❌ Emergency Stop Failed"
            embed.description = str(e)
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel emergency stop."""
        await interaction.response.send_message(
            "Emergency stop cancelled",
            ephemeral=True,
        )
        self.stop()


# =============================================================================
# Confirm Stop All View
# =============================================================================


class ConfirmStopAllView(discord.ui.View):
    """Confirmation view for stopping all bots."""

    def __init__(self, master: "Master"):
        super().__init__(timeout=60)
        self.master = master

    @discord.ui.button(label="Confirm Stop All", style=discord.ButtonStyle.danger, emoji="⛔")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm stop all action."""
        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

        try:
            from src.master import BotCommander

            commander = BotCommander(self.master.registry)
            stopped = await commander.stop_all()

            embed.title = "⛔ All Bots Stopped"
            embed.description = f"Stopped {len(stopped)} bots"
            embed.color = discord.Color.orange()

            if stopped:
                embed.add_field(
                    name="Stopped Bots",
                    value="\n".join(stopped[:15]),
                    inline=False,
                )

        except Exception as e:
            logger.error(f"Error stopping all bots: {e}")
            embed.title = "❌ Error"
            embed.description = str(e)
            embed.color = discord.Color.red()

        await interaction.followup.send(embed=embed)
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel stop all action."""
        await interaction.response.send_message("Stop all cancelled", ephemeral=True)
        self.stop()


# =============================================================================
# Confirm Reset Circuit Breaker View
# =============================================================================


class ConfirmResetCircuitBreakerView(discord.ui.View):
    """Confirmation view for resetting circuit breaker."""

    def __init__(self, risk_engine: "RiskEngine"):
        super().__init__(timeout=60)
        self.risk_engine = risk_engine

    @discord.ui.button(label="Confirm Reset", style=discord.ButtonStyle.primary, emoji="🔓")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Confirm reset action."""
        await interaction.response.defer()

        embed = discord.Embed(timestamp=datetime.now(timezone.utc))

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
        self.stop()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        """Cancel reset action."""
        await interaction.response.send_message("Reset cancelled", ephemeral=True)
        self.stop()
