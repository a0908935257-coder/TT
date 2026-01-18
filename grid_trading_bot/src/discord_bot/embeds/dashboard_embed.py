"""
Dashboard Embeds.

Embed builders for dashboard and status displays.
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
        return f"{sign}{num:,.2f}"
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
# Dashboard Embed
# =============================================================================


def build_dashboard_embed(dashboard_data) -> discord.Embed:
    """
    Build main dashboard embed.

    Args:
        dashboard_data: Dashboard data with summary
    """
    summary = dashboard_data.summary

    # Determine color based on profit
    total_profit = getattr(summary, "total_profit", summary.today_profit)
    color = discord.Color.green() if float(total_profit or 0) >= 0 else discord.Color.red()

    embed = discord.Embed(
        title="📊 Dashboard",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Bot statistics
    error_bots = getattr(summary, "error_bots", 0)
    embed.add_field(
        name="🤖 Bot Status",
        value=f"🟢 Running: {summary.running_bots}\n"
              f"🟡 Paused: {summary.paused_bots}\n"
              f"❌ Error: {error_bots}\n"
              f"📊 Total: {summary.total_bots}",
        inline=True,
    )

    # Fund overview
    total_value = getattr(summary, "total_value", summary.total_investment)
    profit_rate = getattr(summary, "total_profit_rate", 0)

    embed.add_field(
        name="💰 Fund Overview",
        value=f"Investment: {format_number(summary.total_investment)} USDT\n"
              f"Current: {format_number(total_value)} USDT\n"
              f"Profit: {format_profit(total_profit)} USDT\n"
              f"Rate: {format_percentage(profit_rate)}",
        inline=True,
    )

    # Today's statistics
    embed.add_field(
        name="📈 Today",
        value=f"Profit: {format_profit(summary.today_profit)} USDT\n"
              f"Trades: {summary.today_trades}\n"
              f"Orders: {getattr(summary, 'pending_orders', 'N/A')}",
        inline=True,
    )

    return embed


# =============================================================================
# Dashboard Bots Embed
# =============================================================================


def build_dashboard_bots_embed(bots: List, max_bots: int = 9) -> discord.Embed:
    """
    Build bots section of dashboard.

    Args:
        bots: List of bot info objects
        max_bots: Maximum bots to show
    """
    embed = discord.Embed(
        title="🤖 Active Bots",
        color=discord.Color.blue(),
    )

    if not bots:
        embed.description = "No active bots"
        return embed

    for bot_info in bots[:max_bots]:
        state = bot_info.state.value if hasattr(bot_info.state, "value") else str(bot_info.state)

        # Status emoji
        state_lower = state.lower()
        if state_lower == "running":
            emoji = "🟢"
        elif state_lower == "paused":
            emoji = "🟡"
        elif state_lower == "error":
            emoji = "❌"
        else:
            emoji = "⚪"

        # Get profit if available
        profit = getattr(bot_info, "profit", None)
        profit_str = format_profit(profit) if profit is not None else "N/A"

        embed.add_field(
            name=f"{emoji} {bot_info.bot_id}",
            value=f"**{bot_info.symbol}**\n"
                  f"State: {state.title()}\n"
                  f"Profit: {profit_str}",
            inline=True,
        )

    if len(bots) > max_bots:
        embed.set_footer(text=f"Showing {max_bots} of {len(bots)} bots")

    return embed


# =============================================================================
# Status Embed
# =============================================================================


def build_status_embed(dashboard_data, risk_status=None) -> discord.Embed:
    """
    Build system status embed.

    Args:
        dashboard_data: Dashboard data with summary
        risk_status: Optional risk engine status
    """
    summary = dashboard_data.summary

    # Determine color from risk status or profit
    if risk_status:
        color = get_risk_color(risk_status.level)
    else:
        total_profit = getattr(summary, "total_profit", summary.today_profit)
        color = discord.Color.green() if float(total_profit or 0) >= 0 else discord.Color.red()

    embed = discord.Embed(
        title="📊 System Status",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Bot status
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
        value=f"Total: {format_number(total_value)} USDT\n"
              f"Investment: {format_number(summary.total_investment)} USDT\n"
              f"Profit: {format_profit(total_profit)} USDT\n"
              f"Rate: {format_percentage(profit_rate)}",
        inline=True,
    )

    # Today's statistics
    embed.add_field(
        name="📈 Today",
        value=f"Profit: {format_profit(summary.today_profit)} USDT\n"
              f"Trades: {summary.today_trades}",
        inline=True,
    )

    # Risk status
    if risk_status:
        level_name = risk_status.level.name if hasattr(risk_status.level, "name") else str(risk_status.level)
        risk_emoji = get_risk_emoji(risk_status.level)

        # Drawdown
        drawdown_pct = 0
        if risk_status.drawdown:
            drawdown_pct = float(risk_status.drawdown.drawdown_pct) * 100

        # Circuit breaker
        cb_status = "🟢 Normal"
        if risk_status.circuit_breaker and risk_status.circuit_breaker.is_triggered:
            cb_status = "🔴 Triggered"

        embed.add_field(
            name="🛡️ Risk Status",
            value=f"Level: {risk_emoji} {level_name}\n"
                  f"Drawdown: {drawdown_pct:.2f}%\n"
                  f"Circuit: {cb_status}",
            inline=True,
        )

    return embed


# =============================================================================
# Alerts Embed
# =============================================================================


def build_alerts_embed(alerts: List, max_alerts: int = 10) -> discord.Embed:
    """
    Build active alerts embed.

    Args:
        alerts: List of alert objects
        max_alerts: Maximum alerts to show
    """
    embed = discord.Embed(
        title="⚠️ Active Alerts",
        color=discord.Color.orange(),
        timestamp=datetime.now(timezone.utc),
    )

    if not alerts:
        embed.description = "No active alerts"
        embed.color = discord.Color.green()
        return embed

    alert_text = []
    for alert in alerts[:max_alerts]:
        level = getattr(alert, "level", "WARNING")
        level_str = level.name if hasattr(level, "name") else str(level)

        level_emoji = {
            "WARNING": "🟡",
            "RISK": "🟠",
            "DANGER": "🔴",
            "CIRCUIT_BREAK": "⛔",
        }.get(level_str.upper(), "⚠️")

        message = getattr(alert, "message", str(alert))
        alert_text.append(f"{level_emoji} {message}")

    embed.description = "\n".join(alert_text)

    if len(alerts) > max_alerts:
        embed.set_footer(text=f"Showing {max_alerts} of {len(alerts)} alerts")

    return embed


# =============================================================================
# Profit Summary Embed
# =============================================================================


def build_profit_summary_embed(
    summary,
    days: int = 7,
    daily_breakdown: Optional[List] = None,
    best_day=None,
    worst_day=None,
) -> discord.Embed:
    """
    Build profit statistics embed.

    Args:
        summary: Dashboard summary
        days: Number of days for stats
        daily_breakdown: Optional list of daily stats
        best_day: Optional best day stats
        worst_day: Optional worst day stats
    """
    total_profit = getattr(summary, "total_profit", summary.today_profit)
    color = discord.Color.green() if float(total_profit or 0) >= 0 else discord.Color.red()

    embed = discord.Embed(
        title=f"📈 Profit Statistics ({days} days)",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Overview
    trade_count = getattr(summary, "total_trades", summary.today_trades)
    win_rate = getattr(summary, "win_rate", 0)

    embed.add_field(
        name="📊 Overview",
        value=f"Total Profit: {format_profit(total_profit)} USDT\n"
              f"Total Trades: {trade_count}\n"
              f"Win Rate: {format_percentage(win_rate)}",
        inline=False,
    )

    # Today's stats
    embed.add_field(
        name="📅 Today",
        value=f"Profit: {format_profit(summary.today_profit)} USDT\n"
              f"Trades: {summary.today_trades}",
        inline=True,
    )

    # Investment info
    embed.add_field(
        name="💰 Investment",
        value=f"Total: {format_number(summary.total_investment)} USDT\n"
              f"Active Bots: {summary.running_bots}",
        inline=True,
    )

    # Daily breakdown
    if daily_breakdown:
        daily_text = ""
        for day in daily_breakdown[:7]:
            pnl = getattr(day, "pnl", 0)
            trades = getattr(day, "trades", 0)
            date_str = getattr(day, "date", "")
            emoji = "📈" if float(pnl or 0) >= 0 else "📉"
            daily_text += f"{emoji} {date_str}: {format_profit(pnl)} ({trades})\n"

        if daily_text:
            embed.add_field(
                name="📅 Daily Breakdown",
                value=daily_text,
                inline=False,
            )

    # Best/worst day
    if best_day:
        embed.add_field(
            name="🏆 Best Day",
            value=f"{best_day.date}: {format_profit(best_day.pnl)}",
            inline=True,
        )

    if worst_day:
        embed.add_field(
            name="📉 Worst Day",
            value=f"{worst_day.date}: {format_profit(worst_day.pnl)}",
            inline=True,
        )

    return embed
