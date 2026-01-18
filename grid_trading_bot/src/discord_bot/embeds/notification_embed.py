"""
Notification Embeds.

Embed builders for trade notifications and reports.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import discord

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


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
    """Format profit with sign."""
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


# =============================================================================
# Trade Embed
# =============================================================================


def build_trade_embed(trade) -> discord.Embed:
    """
    Build embed for trade notification.

    Args:
        trade: Trade notification object
    """
    # Determine color based on side
    side = getattr(trade, "side", "").upper()
    if side == "BUY":
        color = discord.Color.green()
        emoji = "🟢"
        title = "Buy Order Executed"
    elif side == "SELL":
        color = discord.Color.red()
        emoji = "🔴"
        title = "Sell Order Executed"
    else:
        color = discord.Color.blue()
        emoji = "🔄"
        title = "Trade Executed"

    embed = discord.Embed(
        title=f"{emoji} {title}",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Bot info
    bot_id = getattr(trade, "bot_id", "N/A")
    symbol = getattr(trade, "symbol", "N/A")

    embed.add_field(name="Bot", value=f"`{bot_id}`", inline=True)
    embed.add_field(name="Symbol", value=symbol, inline=True)
    embed.add_field(name="Side", value=f"{emoji} {side}", inline=True)

    # Trade details
    price = getattr(trade, "price", None)
    quantity = getattr(trade, "quantity", None)
    amount = getattr(trade, "amount", None)

    if amount is None and price and quantity:
        try:
            amount = float(price) * float(quantity)
        except (ValueError, TypeError):
            amount = None

    embed.add_field(name="Price", value=f"${format_number(price)}", inline=True)
    embed.add_field(name="Quantity", value=format_number(quantity, 6), inline=True)
    embed.add_field(name="Amount", value=f"{format_number(amount)} USDT", inline=True)

    # Profit (for sell orders)
    profit = getattr(trade, "profit", None)
    if profit is not None:
        profit_emoji = "📈" if float(profit) >= 0 else "📉"
        embed.add_field(
            name=f"{profit_emoji} Profit",
            value=f"{format_profit(profit)} USDT",
            inline=True,
        )

    # Fee
    fee = getattr(trade, "fee", None)
    if fee is not None:
        embed.add_field(name="Fee", value=f"{format_number(fee, 6)} USDT", inline=True)

    # Trade ID
    trade_id = getattr(trade, "trade_id", None) or getattr(trade, "order_id", None)
    if trade_id:
        embed.set_footer(text=f"Trade ID: {trade_id}")

    return embed


# =============================================================================
# Daily Report Embed
# =============================================================================


def build_daily_report_embed(report) -> discord.Embed:
    """
    Build embed for daily report.

    Args:
        report: Daily report object
    """
    total_profit = getattr(report, "total_profit", 0)
    color = discord.Color.green() if float(total_profit or 0) >= 0 else discord.Color.red()

    embed = discord.Embed(
        title="📊 Daily Trading Report",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Date
    date = getattr(report, "date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    embed.description = f"**Date:** {date}"

    # P&L Summary
    profit_emoji = "📈" if float(total_profit or 0) >= 0 else "📉"
    embed.add_field(
        name=f"{profit_emoji} P&L Summary",
        value=f"Total Profit: {format_profit(total_profit)} USDT\n"
              f"Profit Rate: {format_percentage(getattr(report, 'profit_rate', 0))}",
        inline=True,
    )

    # Trade Statistics
    total_trades = getattr(report, "total_trades", 0)
    winning_trades = getattr(report, "winning_trades", 0)
    losing_trades = getattr(report, "losing_trades", 0)
    win_rate = getattr(report, "win_rate", 0)

    embed.add_field(
        name="📈 Trade Statistics",
        value=f"Total Trades: {total_trades}\n"
              f"Winning: {winning_trades}\n"
              f"Losing: {losing_trades}\n"
              f"Win Rate: {format_percentage(win_rate)}",
        inline=True,
    )

    # Best/Worst Trade
    best_trade = getattr(report, "best_trade", None)
    worst_trade = getattr(report, "worst_trade", None)

    if best_trade or worst_trade:
        trade_text = ""
        if best_trade:
            trade_text += f"🏆 Best: {format_profit(getattr(best_trade, 'profit', 0))} USDT\n"
        if worst_trade:
            trade_text += f"📉 Worst: {format_profit(getattr(worst_trade, 'profit', 0))} USDT"

        embed.add_field(
            name="📊 Notable Trades",
            value=trade_text,
            inline=True,
        )

    # Bot Statistics
    active_bots = getattr(report, "active_bots", 0)
    total_position = getattr(report, "total_position", 0)

    embed.add_field(
        name="🤖 Bot Statistics",
        value=f"Active Bots: {active_bots}\n"
              f"Total Position: {format_number(total_position)} USDT",
        inline=True,
    )

    # Top performing bots
    bot_details = getattr(report, "bot_details", [])
    if bot_details:
        # Sort by profit and get top 5
        sorted_bots = sorted(
            bot_details,
            key=lambda x: float(getattr(x, "profit", 0) or 0),
            reverse=True,
        )[:5]

        bot_text = ""
        for i, bot in enumerate(sorted_bots, 1):
            bot_id = getattr(bot, "bot_id", "N/A")
            profit = getattr(bot, "profit", 0)
            trades = getattr(bot, "trades", 0)
            profit_emoji = "📈" if float(profit or 0) >= 0 else "📉"
            bot_text += f"{i}. `{bot_id}`: {profit_emoji} {format_profit(profit)} ({trades} trades)\n"

        embed.add_field(
            name="🏆 Top Bots",
            value=bot_text or "No data",
            inline=False,
        )

    return embed


# =============================================================================
# Bot Event Embed
# =============================================================================


def build_bot_event_embed(event_type: str, bot_id: str, symbol: str, details: Optional[str] = None) -> discord.Embed:
    """
    Build embed for bot events (started, stopped, error, etc.).

    Args:
        event_type: Type of event
        bot_id: Bot ID
        symbol: Trading symbol
        details: Optional additional details
    """
    event_configs = {
        "started": ("▶️ Bot Started", discord.Color.green()),
        "stopped": ("⏹️ Bot Stopped", discord.Color.orange()),
        "paused": ("⏸️ Bot Paused", discord.Color.yellow()),
        "resumed": ("▶️ Bot Resumed", discord.Color.green()),
        "error": ("❌ Bot Error", discord.Color.red()),
        "warning": ("⚠️ Bot Warning", discord.Color.yellow()),
        "grid_updated": ("🔄 Grid Updated", discord.Color.blue()),
    }

    title, color = event_configs.get(event_type, ("🔔 Bot Event", discord.Color.blue()))

    embed = discord.Embed(
        title=title,
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    embed.add_field(name="Bot", value=f"`{bot_id}`", inline=True)
    embed.add_field(name="Symbol", value=symbol, inline=True)

    if details:
        embed.add_field(name="Details", value=details, inline=False)

    return embed


# =============================================================================
# System Notification Embed
# =============================================================================


def build_system_notification_embed(
    title: str,
    message: str,
    level: str = "info",
    details: Optional[Dict] = None,
) -> discord.Embed:
    """
    Build embed for system notifications.

    Args:
        title: Notification title
        message: Notification message
        level: Notification level (info, warning, error, success)
        details: Optional additional details
    """
    level_configs = {
        "info": ("ℹ️", discord.Color.blue()),
        "warning": ("⚠️", discord.Color.yellow()),
        "error": ("❌", discord.Color.red()),
        "success": ("✅", discord.Color.green()),
    }

    emoji, color = level_configs.get(level, ("🔔", discord.Color.blue()))

    embed = discord.Embed(
        title=f"{emoji} {title}",
        description=message,
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    if details:
        for key, value in details.items():
            embed.add_field(name=key, value=str(value), inline=True)

    return embed


# =============================================================================
# Startup/Shutdown Embed
# =============================================================================


def build_startup_embed(
    version: str = "1.0.0",
    bots_loaded: int = 0,
    risk_engine: bool = False,
) -> discord.Embed:
    """Build embed for system startup."""
    embed = discord.Embed(
        title="🚀 Trading System Online",
        description="System has started successfully",
        color=discord.Color.green(),
        timestamp=datetime.now(timezone.utc),
    )

    embed.add_field(name="Version", value=version, inline=True)
    embed.add_field(name="Bots Loaded", value=str(bots_loaded), inline=True)
    embed.add_field(name="Risk Engine", value="✅ Active" if risk_engine else "⚠️ Inactive", inline=True)

    return embed


def build_shutdown_embed(reason: str = "Manual shutdown") -> discord.Embed:
    """Build embed for system shutdown."""
    embed = discord.Embed(
        title="🔴 Trading System Offline",
        description=f"System is shutting down\n\n**Reason:** {reason}",
        color=discord.Color.red(),
        timestamp=datetime.now(timezone.utc),
    )

    return embed
