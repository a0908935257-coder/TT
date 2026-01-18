"""
Discord Bot Module.

Provides Discord-based control console for the trading system.
Includes slash commands, interactive components, and notifications.
"""

from .bot import TradingBot
from .config import ChannelConfig, DiscordConfig, load_discord_config

__all__ = [
    "TradingBot",
    "DiscordConfig",
    "ChannelConfig",
    "load_discord_config",
]
