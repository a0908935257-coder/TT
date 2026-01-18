"""
Create Bot Modal.

Modal forms for creating new trading bots.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

import discord

from src.core import get_logger

if TYPE_CHECKING:
    from src.master import Master

logger = get_logger(__name__)


# =============================================================================
# Create Bot Modal
# =============================================================================


class CreateBotModal(discord.ui.Modal, title="Create New Trading Bot"):
    """Modal for creating a new grid trading bot."""

    symbol = discord.ui.TextInput(
        label="Trading Pair",
        placeholder="e.g., BTCUSDT, ETHUSDT",
        default="BTCUSDT",
        max_length=20,
        required=True,
    )

    investment = discord.ui.TextInput(
        label="Investment (USDT)",
        placeholder="Amount to invest, e.g., 100",
        default="100",
        max_length=15,
        required=True,
    )

    grid_count = discord.ui.TextInput(
        label="Grid Count",
        placeholder="Number of grid levels (5-100)",
        default="20",
        max_length=5,
        required=False,
    )

    atr_multiplier = discord.ui.TextInput(
        label="ATR Multiplier",
        placeholder="Grid width multiplier (1.0-5.0)",
        default="2.0",
        max_length=5,
        required=False,
    )

    def __init__(self, master: "Master"):
        super().__init__()
        self.master = master

    async def on_submit(self, interaction: discord.Interaction):
        """Handle form submission."""
        await interaction.response.defer()

        try:
            # Parse and validate inputs
            symbol = self.symbol.value.upper().strip()
            if not symbol:
                await interaction.followup.send("Symbol is required", ephemeral=True)
                return

            try:
                investment = float(self.investment.value.strip())
                if investment <= 0:
                    raise ValueError("Investment must be positive")
            except ValueError as e:
                await interaction.followup.send(
                    f"Invalid investment amount: {e}",
                    ephemeral=True,
                )
                return

            try:
                grid_count = int(self.grid_count.value.strip()) if self.grid_count.value else 20
                if grid_count < 5 or grid_count > 100:
                    raise ValueError("Grid count must be between 5 and 100")
            except ValueError as e:
                await interaction.followup.send(
                    f"Invalid grid count: {e}",
                    ephemeral=True,
                )
                return

            try:
                atr_multiplier = float(self.atr_multiplier.value.strip()) if self.atr_multiplier.value else 2.0
                if atr_multiplier < 1.0 or atr_multiplier > 5.0:
                    raise ValueError("ATR multiplier must be between 1.0 and 5.0")
            except ValueError as e:
                await interaction.followup.send(
                    f"Invalid ATR multiplier: {e}",
                    ephemeral=True,
                )
                return

            # Create bot configuration
            from src.master import BotType

            bot_config = {
                "symbol": symbol,
                "market_type": "spot",
                "total_investment": str(investment),
                "grid_count": grid_count,
                "atr_multiplier": atr_multiplier,
            }

            # Create the bot
            result = await self.master.create_bot(BotType.GRID, bot_config)

            if result.success:
                embed = discord.Embed(
                    title="✅ Bot Created",
                    description=f"Bot `{result.bot_id}` created successfully",
                    color=discord.Color.green(),
                    timestamp=datetime.now(timezone.utc),
                )
                embed.add_field(name="Symbol", value=symbol, inline=True)
                embed.add_field(name="Investment", value=f"{investment} USDT", inline=True)
                embed.add_field(name="Grid Count", value=str(grid_count), inline=True)
                embed.add_field(name="ATR Multiplier", value=str(atr_multiplier), inline=True)
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
            await interaction.followup.send(
                f"Error creating bot: {e}",
                ephemeral=True,
            )

    async def on_error(self, interaction: discord.Interaction, error: Exception):
        """Handle modal errors."""
        logger.error(f"Modal error: {error}")
        await interaction.response.send_message(
            f"An error occurred: {error}",
            ephemeral=True,
        )


# =============================================================================
# Quick Create Bot Modal (Simpler version)
# =============================================================================


class QuickCreateBotModal(discord.ui.Modal, title="Quick Create Bot"):
    """Simplified modal for quick bot creation."""

    symbol = discord.ui.TextInput(
        label="Trading Pair",
        placeholder="e.g., BTCUSDT",
        default="BTCUSDT",
        max_length=20,
        required=True,
    )

    investment = discord.ui.TextInput(
        label="Investment (USDT)",
        placeholder="e.g., 100",
        default="100",
        max_length=15,
        required=True,
    )

    risk_level = discord.ui.TextInput(
        label="Risk Level",
        placeholder="conservative / moderate / aggressive",
        default="moderate",
        max_length=20,
        required=False,
    )

    def __init__(self, master: "Master"):
        super().__init__()
        self.master = master

    async def on_submit(self, interaction: discord.Interaction):
        """Handle form submission."""
        await interaction.response.defer()

        try:
            symbol = self.symbol.value.upper().strip()
            investment = float(self.investment.value.strip())
            risk_level = self.risk_level.value.lower().strip() or "moderate"

            # Map risk level to grid parameters
            risk_params = {
                "conservative": {"grid_count": 30, "atr_multiplier": 1.5},
                "moderate": {"grid_count": 20, "atr_multiplier": 2.0},
                "aggressive": {"grid_count": 10, "atr_multiplier": 3.0},
            }

            params = risk_params.get(risk_level, risk_params["moderate"])

            from src.master import BotType

            bot_config = {
                "symbol": symbol,
                "market_type": "spot",
                "total_investment": str(investment),
                "grid_count": params["grid_count"],
                "atr_multiplier": params["atr_multiplier"],
            }

            result = await self.master.create_bot(BotType.GRID, bot_config)

            if result.success:
                embed = discord.Embed(
                    title="✅ Bot Created",
                    description=f"Bot `{result.bot_id}` created with {risk_level} risk profile",
                    color=discord.Color.green(),
                    timestamp=datetime.now(timezone.utc),
                )
                embed.add_field(name="Symbol", value=symbol, inline=True)
                embed.add_field(name="Investment", value=f"{investment} USDT", inline=True)
                embed.add_field(name="Risk Level", value=risk_level.title(), inline=True)
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

        except ValueError as e:
            await interaction.followup.send(f"Invalid input: {e}", ephemeral=True)
        except Exception as e:
            logger.error(f"Error creating bot: {e}")
            await interaction.followup.send(f"Error: {e}", ephemeral=True)


# =============================================================================
# Edit Bot Modal
# =============================================================================


class EditBotModal(discord.ui.Modal, title="Edit Bot Configuration"):
    """Modal for editing bot configuration."""

    grid_count = discord.ui.TextInput(
        label="Grid Count",
        placeholder="Number of grid levels (5-100)",
        max_length=5,
        required=False,
    )

    atr_multiplier = discord.ui.TextInput(
        label="ATR Multiplier",
        placeholder="Grid width multiplier (1.0-5.0)",
        max_length=5,
        required=False,
    )

    def __init__(self, bot_id: str, master: "Master", current_config: Optional[dict] = None):
        super().__init__()
        self.bot_id = bot_id
        self.master = master

        # Pre-fill with current values if available
        if current_config:
            if "grid_count" in current_config:
                self.grid_count.default = str(current_config["grid_count"])
            if "atr_multiplier" in current_config:
                self.atr_multiplier.default = str(current_config["atr_multiplier"])

    async def on_submit(self, interaction: discord.Interaction):
        """Handle form submission."""
        await interaction.response.defer()

        try:
            updates = {}

            if self.grid_count.value:
                grid_count = int(self.grid_count.value.strip())
                if 5 <= grid_count <= 100:
                    updates["grid_count"] = grid_count
                else:
                    await interaction.followup.send(
                        "Grid count must be between 5 and 100",
                        ephemeral=True,
                    )
                    return

            if self.atr_multiplier.value:
                atr_multiplier = float(self.atr_multiplier.value.strip())
                if 1.0 <= atr_multiplier <= 5.0:
                    updates["atr_multiplier"] = atr_multiplier
                else:
                    await interaction.followup.send(
                        "ATR multiplier must be between 1.0 and 5.0",
                        ephemeral=True,
                    )
                    return

            if not updates:
                await interaction.followup.send("No changes to apply", ephemeral=True)
                return

            # Apply updates (if method exists)
            if hasattr(self.master, "update_bot_config"):
                result = await self.master.update_bot_config(self.bot_id, updates)

                if result.success:
                    embed = discord.Embed(
                        title="✅ Bot Updated",
                        description=f"Bot `{self.bot_id}` configuration updated",
                        color=discord.Color.green(),
                        timestamp=datetime.now(timezone.utc),
                    )
                    for key, value in updates.items():
                        embed.add_field(name=key, value=str(value), inline=True)
                else:
                    embed = discord.Embed(
                        title="❌ Update Failed",
                        description=result.message,
                        color=discord.Color.red(),
                    )
            else:
                embed = discord.Embed(
                    title="⚠️ Not Supported",
                    description="Bot configuration update is not supported yet",
                    color=discord.Color.orange(),
                )

            await interaction.followup.send(embed=embed)

        except ValueError as e:
            await interaction.followup.send(f"Invalid input: {e}", ephemeral=True)
        except Exception as e:
            logger.error(f"Error updating bot: {e}")
            await interaction.followup.send(f"Error: {e}", ephemeral=True)
