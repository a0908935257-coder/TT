"""
Risk Embeds.

Embed builders for risk alerts and circuit breaker notifications.
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


def format_percentage(value, as_decimal: bool = True) -> str:
    """Format percentage."""
    if value is None:
        return "N/A"
    try:
        num = float(value)
        if as_decimal:
            num = num * 100
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
# Risk Alert Embed
# =============================================================================


def build_alert_embed(alert) -> discord.Embed:
    """
    Build embed for risk alert.

    Args:
        alert: Risk alert object
    """
    level = getattr(alert, "level", None)
    level_str = level.name if hasattr(level, "name") else str(level).upper()

    color = get_risk_color(level)
    emoji = get_risk_emoji(level)

    # Title based on level
    level_titles = {
        "NORMAL": "Risk Status Normal",
        "WARNING": "Warning Alert",
        "RISK": "Risk Alert",
        "DANGER": "Danger Alert",
        "CIRCUIT_BREAK": "Circuit Breaker Triggered",
    }
    title = level_titles.get(level_str, "Risk Alert")

    embed = discord.Embed(
        title=f"{emoji} {title}",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Message
    message = getattr(alert, "message", str(alert))
    embed.description = message

    # Level
    embed.add_field(name="Level", value=f"{emoji} {level_str}", inline=True)

    # Metric info
    metric = getattr(alert, "metric", None)
    if metric:
        embed.add_field(name="Metric", value=str(metric), inline=True)

    # Current value
    current_value = getattr(alert, "current_value", None)
    if current_value is not None:
        embed.add_field(name="Current Value", value=format_percentage(current_value), inline=True)

    # Threshold
    threshold = getattr(alert, "threshold", None)
    if threshold is not None:
        embed.add_field(name="Threshold", value=format_percentage(threshold), inline=True)

    # Action taken
    action = getattr(alert, "action", None)
    if action:
        embed.add_field(name="Action", value=str(action), inline=False)

    # Alert time
    alert_time = getattr(alert, "timestamp", None) or getattr(alert, "created_at", None)
    if alert_time:
        embed.set_footer(text=f"Alert time: {alert_time.strftime('%Y-%m-%d %H:%M:%S')}")

    return embed


# =============================================================================
# Circuit Breaker Embed
# =============================================================================


def build_circuit_break_embed(
    triggered: bool,
    reason: Optional[str] = None,
    cooldown_until: Optional[datetime] = None,
    triggered_at: Optional[datetime] = None,
) -> discord.Embed:
    """
    Build embed for circuit breaker notification.

    Args:
        triggered: Whether circuit breaker is triggered
        reason: Trigger reason
        cooldown_until: Cooldown end time
        triggered_at: When it was triggered
    """
    if triggered:
        embed = discord.Embed(
            title="⛔ CIRCUIT BREAKER TRIGGERED",
            description="**All trading has been automatically stopped**",
            color=discord.Color.dark_red(),
            timestamp=datetime.now(timezone.utc),
        )

        if reason:
            embed.add_field(name="Reason", value=reason, inline=False)

        if triggered_at:
            embed.add_field(
                name="Triggered At",
                value=triggered_at.strftime("%Y-%m-%d %H:%M:%S"),
                inline=True,
            )

        if cooldown_until:
            embed.add_field(
                name="Cooldown Until",
                value=cooldown_until.strftime("%Y-%m-%d %H:%M:%S"),
                inline=True,
            )

            # Calculate remaining time
            now = datetime.now(timezone.utc)
            if cooldown_until > now:
                remaining = cooldown_until - now
                mins = int(remaining.total_seconds() // 60)
                secs = int(remaining.total_seconds() % 60)
                embed.add_field(
                    name="Remaining",
                    value=f"{mins}m {secs}s",
                    inline=True,
                )

        embed.add_field(
            name="Next Steps",
            value="• Wait for cooldown to finish\n"
                  "• Review risk metrics\n"
                  "• Use `/emergency reset` to manually reset (admin only)",
            inline=False,
        )

    else:
        embed = discord.Embed(
            title="✅ Circuit Breaker Reset",
            description="Circuit breaker has been reset. Trading can resume.",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc),
        )

    return embed


# =============================================================================
# Risk Status Embed
# =============================================================================


def build_risk_status_embed(status) -> discord.Embed:
    """
    Build embed for risk engine status.

    Args:
        status: Risk engine status object
    """
    level = status.level
    level_name = level.name if hasattr(level, "name") else str(level)
    color = get_risk_color(level)
    emoji = get_risk_emoji(level)

    embed = discord.Embed(
        title="🛡️ Risk Status",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    # Risk level
    embed.add_field(name="Risk Level", value=f"{emoji} {level_name}", inline=True)

    # Capital info
    if status.capital:
        total_capital = float(status.capital.total_capital)
        initial_capital = float(getattr(status.capital, "initial_capital", total_capital))
        available = float(getattr(status.capital, "available_balance", total_capital))

        change_pct = 0
        if initial_capital > 0:
            change_pct = (total_capital - initial_capital) / initial_capital

        embed.add_field(
            name="💰 Capital",
            value=f"Total: {format_number(total_capital)} USDT\n"
                  f"Available: {format_number(available)} USDT\n"
                  f"Change: {format_percentage(change_pct)}",
            inline=True,
        )

    # Drawdown info
    if status.drawdown:
        current_dd = float(status.drawdown.drawdown_pct) * 100
        max_dd = float(getattr(status.drawdown, "max_drawdown_pct", status.drawdown.drawdown_pct)) * 100
        peak = float(getattr(status.drawdown, "peak_value", 0))

        embed.add_field(
            name="📉 Drawdown",
            value=f"Current: {current_dd:.2f}%\n"
                  f"Max: {max_dd:.2f}%\n"
                  f"Peak: {format_number(peak)} USDT",
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
            value=f"{format_profit(pnl)} USDT\n({format_percentage(pnl_pct)})",
            inline=True,
        )

    # Circuit breaker status
    if status.circuit_breaker:
        cb = status.circuit_breaker
        if cb.is_triggered:
            reason = getattr(cb, "trigger_reason", "Unknown")
            cooldown_until = getattr(cb, "cooldown_until", None)
            cooldown_str = cooldown_until.strftime("%H:%M:%S") if cooldown_until else "N/A"

            cb_text = f"🔴 **TRIGGERED**\nReason: {reason}\nCooldown: {cooldown_str}"
        else:
            cb_text = "🟢 Normal"

        embed.add_field(name="Circuit Breaker", value=cb_text, inline=True)

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
        embed.add_field(name="⚠️ Active Alerts", value=alerts_text, inline=False)

    return embed


# =============================================================================
# Emergency Status Embed
# =============================================================================


def build_emergency_status_embed(
    is_active: bool,
    reason: Optional[str] = None,
    triggered_at: Optional[datetime] = None,
    cooldown_until: Optional[datetime] = None,
) -> discord.Embed:
    """
    Build embed for emergency status.

    Args:
        is_active: Whether emergency is active
        reason: Emergency reason
        triggered_at: When emergency was triggered
        cooldown_until: Cooldown end time
    """
    if is_active:
        embed = discord.Embed(
            title="🚨 Emergency Status",
            description="**⚠️ EMERGENCY STOP ACTIVE**",
            color=discord.Color.red(),
            timestamp=datetime.now(timezone.utc),
        )

        embed.add_field(name="Status", value="🔴 Active", inline=True)

        if reason:
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
        embed = discord.Embed(
            title="🚨 Emergency Status",
            description="**✅ System Normal**",
            color=discord.Color.green(),
            timestamp=datetime.now(timezone.utc),
        )

        embed.add_field(name="Status", value="🟢 Normal", inline=True)
        embed.add_field(name="Emergency", value="Not active", inline=True)

    return embed


# =============================================================================
# Drawdown Alert Embed
# =============================================================================


def build_drawdown_alert_embed(
    current_drawdown: float,
    max_drawdown: float,
    threshold: float,
    capital: float,
    peak: float,
) -> discord.Embed:
    """
    Build embed for drawdown alert.

    Args:
        current_drawdown: Current drawdown percentage (as decimal)
        max_drawdown: Maximum drawdown percentage (as decimal)
        threshold: Alert threshold (as decimal)
        capital: Current capital
        peak: Peak capital
    """
    severity = "WARNING" if current_drawdown < 0.1 else "DANGER" if current_drawdown < 0.15 else "CIRCUIT_BREAK"
    color = get_risk_color(type("Level", (), {"name": severity})())
    emoji = get_risk_emoji(type("Level", (), {"name": severity})())

    embed = discord.Embed(
        title=f"{emoji} Drawdown Alert",
        description=f"Current drawdown ({current_drawdown * 100:.2f}%) has exceeded threshold ({threshold * 100:.2f}%)",
        color=color,
        timestamp=datetime.now(timezone.utc),
    )

    embed.add_field(
        name="Drawdown",
        value=f"Current: {current_drawdown * 100:.2f}%\n"
              f"Max: {max_drawdown * 100:.2f}%\n"
              f"Threshold: {threshold * 100:.2f}%",
        inline=True,
    )

    embed.add_field(
        name="Capital",
        value=f"Current: {format_number(capital)} USDT\n"
              f"Peak: {format_number(peak)} USDT\n"
              f"Loss: {format_profit(capital - peak)} USDT",
        inline=True,
    )

    return embed
