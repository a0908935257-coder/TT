"""
Bot Embeds.

Embed builders for bot-related displays.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

import discord

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def get_state_color(state) -> discord.Color:
    """Get color for bot state."""
    state_str = state.value if hasattr(state, "value") else str(state)
    mapping = {
        "registered": discord.Color.light_grey(),
        "initializing": discord.Color.blue(),
        "running": discord.Color.green(),
        "paused": discord.Color.yellow(),
        "stopping": discord.Color.orange(),
        "stopped": discord.Color.greyple(),
        "error": discord.Color.red(),
    }
    return mapping.get(state_str.lower(), discord.Color.default())


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


def format_number(value, decimals: int = 2) -> str:
    """Format number with thousand separators."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
        if decimals == 0:
            return f"{num:,.0f}"
        return f"{num:,.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_profit(value) -> str:
    """Format profit with sign and color indicator."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
        sign = "+" if num >= 0 else ""
        return f"{sign}{num:,.4f}"
    except (ValueError, TypeError):
        return str(value)


def format_percentage(value) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    try:
        num = float(value) * 100
        sign = "+" if num >= 0 else ""
        return f"{sign}{num:.2f}%"
    except (ValueError, TypeError):
        return str(value)


def format_duration(seconds: Optional[int]) -> str:
    """Format duration from seconds."""
    if seconds is None:
        return "N/A"
    try:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, secs = divmod(remainder, 60)
        if hours > 24:
            days = hours // 24
            hours = hours % 24
            return f"{days}d {hours}h {minutes}m"
        return f"{hours}h {minutes}m {secs}s"
    except (ValueError, TypeError):
        return "N/A"


# =============================================================================
# Bot Detail Embed
# =============================================================================


def build_bot_detail_embed(bot_info, instance_status: Optional[Dict] = None, stats: Optional[Dict] = None) -> discord.Embed:
    """
    Build detailed embed for a single bot.

    Args:
        bot_info: Bot info object with bot_id, symbol, state, etc.
        instance_status: Optional status from bot instance
        stats: Optional statistics from bot instance
    """
    state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
    bot_type = bot_info.bot_type.value if hasattr(bot_info.bot_type, "value") else str(bot_info.bot_type)
    emoji = get_status_emoji(bot_info.state)
    color = get_state_color(bot_info.state)

    embed = discord.Embed(
        title=f"{emoji} Bot: {bot_info.bot_id}",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Basic info
    embed.add_field(name="Type", value=bot_type.title(), inline=True)
    embed.add_field(name="Symbol", value=bot_info.symbol, inline=True)
    embed.add_field(name="State", value=state.title(), inline=True)

    # Created time
    if hasattr(bot_info, "created_at") and bot_info.created_at:
        embed.add_field(
            name="Created",
            value=bot_info.created_at.strftime("%Y-%m-%d %H:%M"),
            inline=True,
        )

    # Running time
    if hasattr(bot_info, "started_at") and bot_info.started_at:
        running_time = (datetime.now(timezone.utc) - bot_info.started_at).total_seconds()
        embed.add_field(name="Running Time", value=format_duration(int(running_time)), inline=True)

    # Grid settings from instance status
    if instance_status:
        lower_price = instance_status.get("lower_price")
        upper_price = instance_status.get("upper_price")
        grid_count = instance_status.get("grid_count")
        grid_version = instance_status.get("grid_version", 1)

        if lower_price and upper_price:
            embed.add_field(
                name="📊 Grid Settings",
                value=f"Range: ${float(lower_price):,.2f} - ${float(upper_price):,.2f}\n"
                      f"Grids: {grid_count or 'N/A'}\n"
                      f"Version: {grid_version}",
                inline=False,
            )

        # Position and orders
        pending_buy = instance_status.get("pending_buy_orders", 0)
        pending_sell = instance_status.get("pending_sell_orders", 0)
        base_position = instance_status.get("base_position", 0)

        embed.add_field(
            name="📋 Orders",
            value=f"Buy Orders: {pending_buy}\n"
                  f"Sell Orders: {pending_sell}\n"
                  f"Position: {format_number(base_position, 6)}",
            inline=True,
        )

    # Statistics
    if stats:
        total_profit = stats.get("total_profit", 0)
        trade_count = stats.get("trade_count", 0)
        profit_rate = stats.get("profit_rate", 0)
        win_rate = stats.get("win_rate", 0)

        profit_emoji = "📈" if float(total_profit or 0) >= 0 else "📉"

        embed.add_field(
            name=f"{profit_emoji} Statistics",
            value=f"Total Profit: {format_profit(total_profit)} USDT\n"
                  f"Total Trades: {trade_count}\n"
                  f"Profit Rate: {format_percentage(profit_rate)}\n"
                  f"Win Rate: {format_percentage(win_rate)}",
            inline=True,
        )

        # Last trade
        last_trade = stats.get("last_trade")
        if last_trade:
            trade_time = last_trade.get("time", "N/A")
            trade_side = last_trade.get("side", "N/A")
            trade_price = last_trade.get("price", "N/A")
            trade_profit = last_trade.get("profit")

            embed.add_field(
                name="🔄 Last Trade",
                value=f"Time: {trade_time}\n"
                      f"Side: {trade_side}\n"
                      f"Price: ${format_number(trade_price)}\n"
                      f"Profit: {format_profit(trade_profit)} USDT" if trade_profit else "",
                inline=True,
            )

    # Health status
    health = instance_status.get("health") if instance_status else None
    if health:
        health_emoji = "✅" if health == "healthy" else "⚠️" if health == "warning" else "❌"
        embed.add_field(name="Health", value=f"{health_emoji} {health.title()}", inline=True)

    return embed


# =============================================================================
# Bot List Embed
# =============================================================================


def build_bot_list_embed(bots: List, title: str = "Trading Bots", max_bots: int = 12) -> discord.Embed:
    """
    Build embed showing list of bots.

    Args:
        bots: List of bot info objects
        title: Embed title
        max_bots: Maximum bots to show
    """
    embed = discord.Embed(
        title=f"📋 {title}",
        color=discord.Color.blue(),
        timestamp=datetime.now(timezone.utc),
    )

    if not bots:
        embed.description = "No bots registered\n\nUse `/bot create` to create a new bot"
        return embed

    # Count by state
    running = 0
    paused = 0
    stopped = 0
    error = 0

    for bot_info in bots:
        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
        state_lower = state.lower()
        if state_lower == "running":
            running += 1
        elif state_lower == "paused":
            paused += 1
        elif state_lower in ("stopped", "registered"):
            stopped += 1
        elif state_lower == "error":
            error += 1

    # Summary
    embed.description = f"🟢 Running: {running}  🟡 Paused: {paused}  🔴 Stopped: {stopped}  ❌ Error: {error}"

    # Add bot fields
    for bot_info in bots[:max_bots]:
        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
        bot_type = bot_info.bot_type.value if hasattr(bot_info.bot_type, "value") else str(bot_info.bot_type)
        emoji = get_status_emoji(bot_info.state)

        # Get profit if available
        profit_str = ""
        if hasattr(bot_info, "profit") and bot_info.profit is not None:
            profit = float(bot_info.profit)
            profit_emoji = "📈" if profit >= 0 else "📉"
            profit_str = f"\n{profit_emoji} {format_profit(profit)}"

        embed.add_field(
            name=f"{emoji} {bot_info.bot_id}",
            value=f"**Type:** {bot_type.title()}\n"
                  f"**Symbol:** {bot_info.symbol}\n"
                  f"**State:** {state.title()}{profit_str}",
            inline=True,
        )

    # Footer with total count
    if len(bots) > max_bots:
        embed.set_footer(text=f"Showing {max_bots} of {len(bots)} bots")
    else:
        embed.set_footer(text=f"Total: {len(bots)} bots")

    return embed


# =============================================================================
# Bot Summary Embed
# =============================================================================


def build_bot_summary_embed(bot_info, action: str, result: Optional[Any] = None) -> discord.Embed:
    """
    Build embed for bot action result.

    Args:
        bot_info: Bot info object
        action: Action performed (started, stopped, paused, resumed, created, deleted)
        result: Optional result data
    """
    action_configs = {
        "started": ("✅ Bot Started", discord.Color.green()),
        "stopped": ("🛑 Bot Stopped", discord.Color.orange()),
        "paused": ("⏸️ Bot Paused", discord.Color.yellow()),
        "resumed": ("▶️ Bot Resumed", discord.Color.green()),
        "created": ("✅ Bot Created", discord.Color.green()),
        "deleted": ("🗑️ Bot Deleted", discord.Color.red()),
        "error": ("❌ Error", discord.Color.red()),
    }

    title, color = action_configs.get(action, ("Bot Action", discord.Color.blue()))

    embed = discord.Embed(
        title=title,
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Bot info
    embed.add_field(name="Bot ID", value=f"`{bot_info.bot_id}`", inline=True)
    embed.add_field(name="Symbol", value=bot_info.symbol, inline=True)

    state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)
    embed.add_field(name="State", value=state.title(), inline=True)

    # Additional result data
    if result and hasattr(result, "data") and result.data:
        data = result.data
        if "total_trades" in data:
            embed.add_field(
                name="📊 Final Stats",
                value=f"Trades: {data['total_trades']}\n"
                      f"Profit: {format_profit(data.get('total_profit', 0))} USDT",
                inline=False,
            )

    return embed
