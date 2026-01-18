"""
Discord Bot Embeds.

Embed builders for various Discord displays.
"""

from .bot_embed import (
    build_bot_detail_embed,
    build_bot_list_embed,
    build_bot_summary_embed,
    format_duration,
    format_number,
    format_percentage,
    format_profit,
    get_state_color,
    get_status_emoji,
)
from .dashboard_embed import (
    build_alerts_embed,
    build_dashboard_bots_embed,
    build_dashboard_embed,
    build_profit_summary_embed,
    build_status_embed,
    get_risk_color,
    get_risk_emoji,
)
from .notification_embed import (
    build_bot_event_embed,
    build_daily_report_embed,
    build_shutdown_embed,
    build_startup_embed,
    build_system_notification_embed,
    build_trade_embed,
)
from .risk_embed import (
    build_alert_embed,
    build_circuit_break_embed,
    build_drawdown_alert_embed,
    build_emergency_status_embed,
    build_risk_status_embed,
)

__all__ = [
    # Bot embeds
    "build_bot_detail_embed",
    "build_bot_list_embed",
    "build_bot_summary_embed",
    # Dashboard embeds
    "build_dashboard_embed",
    "build_dashboard_bots_embed",
    "build_status_embed",
    "build_alerts_embed",
    "build_profit_summary_embed",
    # Notification embeds
    "build_trade_embed",
    "build_daily_report_embed",
    "build_bot_event_embed",
    "build_system_notification_embed",
    "build_startup_embed",
    "build_shutdown_embed",
    # Risk embeds
    "build_alert_embed",
    "build_circuit_break_embed",
    "build_risk_status_embed",
    "build_emergency_status_embed",
    "build_drawdown_alert_embed",
    # Helper functions
    "get_state_color",
    "get_status_emoji",
    "get_risk_color",
    "get_risk_emoji",
    "format_number",
    "format_profit",
    "format_percentage",
    "format_duration",
]
