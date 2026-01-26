"""
Notification Manager.

Provides a unified notification manager with deduplication, muting,
rate limiting, and business-level notification methods.

Enhanced with:
- Rate limiting to prevent notification storms
- Critical notification bypass for urgent alerts
"""

import hashlib
import os
import time
from collections import deque
from decimal import Decimal
from typing import Any, Deque, Optional, Tuple

from src.core import get_logger

from .base import NotificationLevel
from .discord import DiscordNotifier
from .templates import NotificationTemplates

logger = get_logger(__name__)


class NotificationRateLimiter:
    """
    Rate limiter for notifications with critical bypass.

    Prevents notification storms while ensuring critical notifications
    are never blocked.
    """

    def __init__(
        self,
        max_per_minute: int = 25,
        burst_limit: int = 10,
        burst_window_seconds: int = 5,
    ):
        """
        Initialize rate limiter.

        Args:
            max_per_minute: Maximum notifications per minute
            burst_limit: Maximum notifications in burst window
            burst_window_seconds: Burst detection window
        """
        self._max_per_minute = max_per_minute
        self._burst_limit = burst_limit
        self._burst_window = burst_window_seconds

        # Sliding window: (timestamp, level)
        self._window: Deque[Tuple[float, str]] = deque()

        # Statistics
        self._stats = {
            "total_allowed": 0,
            "total_rate_limited": 0,
            "total_burst_limited": 0,
            "critical_bypassed": 0,
        }

    def should_allow(self, level: NotificationLevel) -> bool:
        """
        Check if notification should be allowed.

        Args:
            level: Notification level

        Returns:
            True if notification should be sent
        """
        now = time.time()
        minute_ago = now - 60
        burst_ago = now - self._burst_window

        # Clean old entries
        while self._window and self._window[0][0] < minute_ago:
            self._window.popleft()

        # Critical notifications always bypass
        if level == NotificationLevel.CRITICAL:
            self._window.append((now, level.value))
            self._stats["critical_bypassed"] += 1
            self._stats["total_allowed"] += 1
            return True

        # Check burst limit
        burst_count = sum(1 for t, _ in self._window if t >= burst_ago)
        if burst_count >= self._burst_limit:
            self._stats["total_burst_limited"] += 1
            return False

        # Check per-minute limit
        if len(self._window) >= self._max_per_minute:
            self._stats["total_rate_limited"] += 1
            return False

        self._window.append((now, level.value))
        self._stats["total_allowed"] += 1
        return True

    def get_stats(self) -> dict[str, int]:
        """Get rate limiter statistics."""
        return self._stats.copy()

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._window.clear()


class NotificationManager:
    """
    Unified notification manager with deduplication and muting support.

    Provides business-level notification methods for trading events,
    alerts, and reports. Supports deduplication within a time window
    to prevent spam.

    Example:
        >>> manager = NotificationManager.from_env()
        >>> await manager.notify_order_filled(
        ...     symbol="BTCUSDT",
        ...     side="BUY",
        ...     price=Decimal("50000"),
        ...     quantity=Decimal("0.1"),
        ... )
    """

    def __init__(
        self,
        discord: Optional[DiscordNotifier] = None,
        enabled: bool = True,
        min_level: NotificationLevel = NotificationLevel.INFO,
        dedup_window: int = 60,
        rate_limit: int = 25,
        enable_rate_limiting: bool = True,
    ):
        """
        Initialize NotificationManager.

        Args:
            discord: DiscordNotifier instance (optional)
            enabled: Enable/disable notifications
            min_level: Minimum notification level to send
            dedup_window: Deduplication time window in seconds
            rate_limit: Maximum notifications per minute
            enable_rate_limiting: Enable rate limiting
        """
        self._discord = discord
        self._enabled = enabled
        self._min_level = min_level
        self._dedup_window = dedup_window
        self._muted = False
        self._enable_rate_limiting = enable_rate_limiting

        # Rate limiter
        self._rate_limiter = NotificationRateLimiter(
            max_per_minute=rate_limit,
            burst_limit=10,
            burst_window_seconds=5,
        )

        # Deduplication cache: hash -> timestamp
        self._dedup_cache: dict[str, float] = {}

        # Statistics
        self._stats = {
            "total_sent": 0,
            "total_deduplicated": 0,
            "total_muted": 0,
            "total_rate_limited": 0,
            "by_level": {level.value: 0 for level in NotificationLevel},
        }

        # Level ordering for comparison
        self._level_order = [
            NotificationLevel.DEBUG,
            NotificationLevel.INFO,
            NotificationLevel.SUCCESS,
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL,
        ]

    # =========================================================================
    # Factory Methods
    # =========================================================================

    @classmethod
    def from_env(cls) -> "NotificationManager":
        """
        Create NotificationManager from environment variables.

        Environment variables:
            DISCORD_WEBHOOK_URL: Discord webhook URL
            NOTIFICATION_ENABLED: Enable notifications (default: true)
            NOTIFICATION_MIN_LEVEL: Minimum level (default: info)
            NOTIFICATION_DEDUP_WINDOW: Dedup window in seconds (default: 60)

        Returns:
            NotificationManager instance
        """
        webhook_url = os.environ.get("DISCORD_WEBHOOK_URL")
        enabled = os.environ.get("NOTIFICATION_ENABLED", "true").lower() == "true"
        min_level_str = os.environ.get("NOTIFICATION_MIN_LEVEL", "info").lower()
        dedup_window = int(os.environ.get("NOTIFICATION_DEDUP_WINDOW", "60"))

        # Parse min level
        try:
            min_level = NotificationLevel(min_level_str)
        except ValueError:
            min_level = NotificationLevel.INFO

        # Create Discord notifier if webhook URL provided
        discord = None
        if webhook_url:
            discord = DiscordNotifier(webhook_url=webhook_url)

        return cls(
            discord=discord,
            enabled=enabled,
            min_level=min_level,
            dedup_window=dedup_window,
        )

    @classmethod
    def from_config(cls, config: Any) -> "NotificationManager":
        """
        Create NotificationManager from configuration object.

        Args:
            config: Configuration object with notification settings.
                   Expected to have a 'notification' attribute with:
                   - enabled: bool
                   - discord_webhook_url: str
                   - min_level: str
                   - rate_limit: int (max notifications per minute)

        Returns:
            NotificationManager instance
        """
        # Handle both AppConfig and NotificationConfig
        notif_config = getattr(config, "notification", config)

        webhook_url = getattr(notif_config, "discord_webhook_url", None)
        enabled = getattr(notif_config, "enabled", True)
        min_level_str = getattr(notif_config, "min_level", "info")
        rate_limit = getattr(notif_config, "rate_limit", 25)
        dedup_window = getattr(notif_config, "dedup_window", 60)

        # Parse min level
        try:
            min_level = NotificationLevel(min_level_str)
        except ValueError:
            min_level = NotificationLevel.INFO

        # Create Discord notifier if webhook URL provided
        discord = None
        if webhook_url:
            discord = DiscordNotifier(webhook_url=webhook_url)

        return cls(
            discord=discord,
            enabled=enabled,
            min_level=min_level,
            dedup_window=dedup_window,
            rate_limit=rate_limit,
        )

    # =========================================================================
    # Control Methods
    # =========================================================================

    def mute(self) -> None:
        """Mute all notifications."""
        self._muted = True
        logger.info("Notifications muted")

    def unmute(self) -> None:
        """Unmute notifications."""
        self._muted = False
        logger.info("Notifications unmuted")

    def set_min_level(self, level: NotificationLevel) -> None:
        """
        Set minimum notification level.

        Args:
            level: Minimum level to send
        """
        self._min_level = level
        logger.info(f"Notification min level set to {level.value}")

    @property
    def is_enabled(self) -> bool:
        """Check if notifications are enabled and not muted."""
        return self._enabled and not self._muted and self._discord is not None

    @property
    def stats(self) -> dict[str, Any]:
        """Get notification statistics."""
        return self._stats.copy()

    def get_stats(self) -> dict[str, Any]:
        """
        Get notification statistics summary.

        Returns:
            Statistics dictionary
        """
        return {
            "total_sent": self._stats["total_sent"],
            "total_deduplicated": self._stats["total_deduplicated"],
            "total_muted": self._stats["total_muted"],
            "total_rate_limited": self._stats["total_rate_limited"],
            "by_level": self._stats["by_level"].copy(),
            "enabled": self._enabled,
            "muted": self._muted,
            "min_level": self._min_level.value,
            "rate_limiter": self._rate_limiter.get_stats(),
        }

    # =========================================================================
    # Internal Methods
    # =========================================================================

    def _should_send(self, level: NotificationLevel) -> bool:
        """
        Check if notification should be sent based on level.

        Args:
            level: Notification level

        Returns:
            True if should send
        """
        if not self.is_enabled:
            return False

        try:
            current_index = self._level_order.index(level)
            min_index = self._level_order.index(self._min_level)
            return current_index >= min_index
        except ValueError:
            return True

    def _is_duplicate(self, content_hash: str) -> bool:
        """
        Check if content is a duplicate within the dedup window.

        Args:
            content_hash: Hash of the content

        Returns:
            True if duplicate
        """
        now = time.time()

        # Clean expired entries
        self._dedup_cache = {
            k: v for k, v in self._dedup_cache.items()
            if now - v < self._dedup_window
        }

        if content_hash in self._dedup_cache:
            return True

        self._dedup_cache[content_hash] = now
        return False

    def _hash_content(self, *args: Any) -> str:
        """
        Create hash from content for deduplication.

        Args:
            *args: Content to hash

        Returns:
            Hash string
        """
        content = "|".join(str(a) for a in args)
        return hashlib.md5(content.encode()).hexdigest()

    async def _send_embed(
        self,
        embed: Any,
        level: NotificationLevel = NotificationLevel.INFO,
        dedup_key: Optional[str] = None,
    ) -> bool:
        """
        Send embed notification with deduplication and rate limiting.

        Args:
            embed: DiscordEmbed or dict
            level: Notification level
            dedup_key: Optional deduplication key

        Returns:
            True if sent successfully
        """
        if not self._should_send(level):
            if self._muted:
                self._stats["total_muted"] += 1
            return False

        # Check deduplication
        if dedup_key and self._is_duplicate(dedup_key):
            self._stats["total_deduplicated"] += 1
            logger.debug(f"Notification deduplicated: {dedup_key[:16]}...")
            return False

        # Check rate limiting (CRITICAL always bypasses)
        if self._enable_rate_limiting and not self._rate_limiter.should_allow(level):
            self._stats["total_rate_limited"] += 1
            logger.warning(
                f"Notification rate limited (level={level.value}). "
                f"Stats: {self._rate_limiter.get_stats()}"
            )
            return False

        # Get embed dict
        embed_dict = embed.build() if hasattr(embed, "build") else embed

        try:
            if self._discord:
                result = await self._discord.send_embed(embed_dict)
                if result:
                    self._stats["total_sent"] += 1
                    self._stats["by_level"][level.value] += 1
                return result
            return False
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
            return False

    # =========================================================================
    # Order Notifications
    # =========================================================================

    async def notify_order_placed(
        self,
        symbol: str,
        side: str,
        order_type: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
    ) -> bool:
        """
        Send order placed notification.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: Order type
            price: Order price
            quantity: Order quantity

        Returns:
            True if sent
        """
        embed = NotificationTemplates.order_placed_embed(
            symbol=symbol,
            side=side,
            order_type=order_type,
            price=price,
            quantity=quantity,
        )
        dedup_key = self._hash_content("order_placed", symbol, side, price, quantity)
        return await self._send_embed(embed, NotificationLevel.INFO, dedup_key)

    async def notify_order_filled(
        self,
        symbol: str,
        side: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
        pnl: Optional[Decimal | float | str] = None,
    ) -> bool:
        """
        Send order filled notification.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Fill price
            quantity: Fill quantity
            pnl: Realized PnL

        Returns:
            True if sent
        """
        embed = NotificationTemplates.order_filled_embed(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            pnl=pnl,
        )
        dedup_key = self._hash_content("order_filled", symbol, side, price, quantity)
        return await self._send_embed(embed, NotificationLevel.INFO, dedup_key)

    async def notify_order_canceled(
        self,
        symbol: str,
        side: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
        reason: Optional[str] = None,
    ) -> bool:
        """
        Send order canceled notification.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Order price
            quantity: Order quantity
            reason: Cancellation reason

        Returns:
            True if sent
        """
        embed = NotificationTemplates.order_canceled_embed(
            symbol=symbol,
            side=side,
            price=price,
            quantity=quantity,
            reason=reason,
        )
        dedup_key = self._hash_content("order_canceled", symbol, side, price)
        return await self._send_embed(embed, NotificationLevel.INFO, dedup_key)

    # =========================================================================
    # Position Notifications
    # =========================================================================

    async def notify_position_opened(
        self,
        symbol: str,
        side: str,
        quantity: Decimal | float | str,
        entry_price: Decimal | float | str,
        leverage: Optional[int] = None,
    ) -> bool:
        """
        Send position opened notification.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            quantity: Position size
            entry_price: Entry price
            leverage: Leverage

        Returns:
            True if sent
        """
        embed = NotificationTemplates.position_opened_embed(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            leverage=leverage,
        )
        dedup_key = self._hash_content("position_opened", symbol, side, entry_price)
        return await self._send_embed(embed, NotificationLevel.INFO, dedup_key)

    async def notify_position_closed(
        self,
        symbol: str,
        side: str,
        quantity: Decimal | float | str,
        entry_price: Decimal | float | str,
        exit_price: Decimal | float | str,
        pnl: Decimal | float | str,
        holding_time: int | float,
    ) -> bool:
        """
        Send position closed notification.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            quantity: Position size
            entry_price: Entry price
            exit_price: Exit price
            pnl: Realized PnL
            holding_time: Holding time in seconds

        Returns:
            True if sent
        """
        embed = NotificationTemplates.position_closed_embed(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            holding_time=holding_time,
        )
        # Position closed is important, use SUCCESS level
        level = NotificationLevel.SUCCESS
        dedup_key = self._hash_content("position_closed", symbol, exit_price, pnl)
        return await self._send_embed(embed, level, dedup_key)

    # =========================================================================
    # Risk Alerts
    # =========================================================================

    async def alert_risk(
        self,
        alert_type: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> bool:
        """
        Send generic risk alert.

        Args:
            alert_type: Type of alert
            message: Alert message
            details: Additional details

        Returns:
            True if sent
        """
        embed = NotificationTemplates.risk_alert_embed(
            alert_type=alert_type,
            message=message,
            details=details,
        )
        dedup_key = self._hash_content("risk_alert", alert_type, message)
        return await self._send_embed(embed, NotificationLevel.WARNING, dedup_key)

    async def alert_drawdown(
        self,
        current_drawdown: Decimal | float | str,
        max_threshold: Decimal | float | str,
        peak_balance: Decimal | float | str,
        current_balance: Decimal | float | str,
    ) -> bool:
        """
        Send drawdown alert.

        Args:
            current_drawdown: Current drawdown percentage
            max_threshold: Maximum allowed drawdown
            peak_balance: Peak balance value
            current_balance: Current balance value

        Returns:
            True if sent
        """
        embed = NotificationTemplates.drawdown_alert_embed(
            current_drawdown=current_drawdown,
            max_threshold=max_threshold,
            peak_balance=peak_balance,
            current_balance=current_balance,
        )
        dedup_key = self._hash_content("drawdown_alert", current_drawdown)
        return await self._send_embed(embed, NotificationLevel.ERROR, dedup_key)

    async def alert_daily_loss(
        self,
        current_loss: Decimal | float | str,
        max_threshold: Decimal | float | str,
        remaining: Decimal | float | str,
    ) -> bool:
        """
        Send daily loss alert.

        Args:
            current_loss: Current daily loss percentage
            max_threshold: Maximum allowed daily loss
            remaining: Remaining budget before limit

        Returns:
            True if sent
        """
        embed = NotificationTemplates.daily_loss_alert_embed(
            current_loss=current_loss,
            max_threshold=max_threshold,
            remaining=remaining,
        )
        dedup_key = self._hash_content("daily_loss_alert", current_loss)
        return await self._send_embed(embed, NotificationLevel.ERROR, dedup_key)

    async def alert_liquidation(
        self,
        symbol: str,
        side: str,
        distance_percent: Decimal | float | str,
        liquidation_price: Decimal | float | str,
        current_price: Decimal | float | str,
    ) -> bool:
        """
        Send liquidation warning.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            distance_percent: Distance to liquidation
            liquidation_price: Liquidation price
            current_price: Current market price

        Returns:
            True if sent
        """
        embed = NotificationTemplates.liquidation_warning_embed(
            symbol=symbol,
            side=side,
            distance_percent=distance_percent,
            liquidation_price=liquidation_price,
            current_price=current_price,
        )
        dedup_key = self._hash_content("liquidation_alert", symbol, distance_percent)
        return await self._send_embed(embed, NotificationLevel.CRITICAL, dedup_key)

    # =========================================================================
    # System Notifications
    # =========================================================================

    async def notify_bot_started(
        self,
        bot_name: str,
        bot_type: str,
        config_summary: dict[str, Any],
    ) -> bool:
        """
        Send bot started notification.

        Args:
            bot_name: Bot name/identifier
            bot_type: Type of bot
            config_summary: Key configuration parameters

        Returns:
            True if sent
        """
        embed = NotificationTemplates.bot_started_embed(
            bot_name=bot_name,
            bot_type=bot_type,
            config_summary=config_summary,
        )
        # Don't deduplicate bot start
        return await self._send_embed(embed, NotificationLevel.SUCCESS)

    async def notify_bot_stopped(
        self,
        bot_name: str,
        reason: str,
        runtime: int | float,
        total_pnl: Decimal | float | str,
    ) -> bool:
        """
        Send bot stopped notification.

        Args:
            bot_name: Bot name/identifier
            reason: Stop reason
            runtime: Total runtime in seconds
            total_pnl: Total PnL during runtime

        Returns:
            True if sent
        """
        embed = NotificationTemplates.bot_stopped_embed(
            bot_name=bot_name,
            reason=reason,
            runtime=runtime,
            total_pnl=total_pnl,
        )
        # Don't deduplicate bot stop
        return await self._send_embed(embed, NotificationLevel.WARNING)

    async def notify_error(
        self,
        bot_name: str,
        error_type: str,
        error_message: str,
    ) -> bool:
        """
        Send error notification.

        Args:
            bot_name: Bot name/identifier
            error_type: Type of error
            error_message: Error message

        Returns:
            True if sent
        """
        embed = NotificationTemplates.bot_error_embed(
            bot_name=bot_name,
            error_type=error_type,
            error_message=error_message,
        )
        dedup_key = self._hash_content("error", bot_name, error_type, error_message[:50])
        return await self._send_embed(embed, NotificationLevel.ERROR, dedup_key)

    async def notify_connection_lost(
        self,
        service_name: str,
        error: Optional[str] = None,
    ) -> bool:
        """
        Send connection lost notification.

        Args:
            service_name: Name of the disconnected service
            error: Error message

        Returns:
            True if sent
        """
        embed = NotificationTemplates.connection_lost_embed(
            service_name=service_name,
            error=error,
        )
        dedup_key = self._hash_content("connection_lost", service_name)
        return await self._send_embed(embed, NotificationLevel.ERROR, dedup_key)

    async def notify_connection_restored(
        self,
        service_name: str,
        downtime: int | float,
    ) -> bool:
        """
        Send connection restored notification.

        Args:
            service_name: Name of the reconnected service
            downtime: Downtime duration in seconds

        Returns:
            True if sent
        """
        embed = NotificationTemplates.connection_restored_embed(
            service_name=service_name,
            downtime=downtime,
        )
        # Don't deduplicate connection restored
        return await self._send_embed(embed, NotificationLevel.SUCCESS)

    # =========================================================================
    # Report Methods
    # =========================================================================

    async def send_daily_report(
        self,
        bot_name: str,
        date: str,
        starting_balance: Decimal | float | str,
        ending_balance: Decimal | float | str,
        total_pnl: Decimal | float | str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        **kwargs: Any,
    ) -> bool:
        """
        Send daily performance report.

        Args:
            bot_name: Bot name/identifier
            date: Report date string
            starting_balance: Starting balance
            ending_balance: Ending balance
            total_pnl: Total PnL for the day
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            **kwargs: Additional report fields

        Returns:
            True if sent
        """
        embed = NotificationTemplates.daily_report_embed(
            bot_name=bot_name,
            date=date,
            starting_balance=starting_balance,
            ending_balance=ending_balance,
            total_pnl=total_pnl,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            **kwargs,
        )
        # Reports shouldn't be deduplicated
        return await self._send_embed(embed, NotificationLevel.INFO)

    # =========================================================================
    # Generic Send Methods (for compatibility)
    # =========================================================================

    async def send_info(self, title: str, message: str) -> bool:
        """Send an info notification."""
        embed = {"title": title, "description": message, "color": 0x3498DB}
        return await self._send_embed(embed, NotificationLevel.INFO)

    async def send_success(self, title: str, message: str) -> bool:
        """Send a success notification."""
        embed = {"title": title, "description": message, "color": 0x2ECC71}
        return await self._send_embed(embed, NotificationLevel.SUCCESS)

    async def send_warning(self, title: str, message: str) -> bool:
        """Send a warning notification."""
        embed = {"title": title, "description": message, "color": 0xF39C12}
        return await self._send_embed(embed, NotificationLevel.WARNING)

    async def send_error(self, title: str, message: str) -> bool:
        """Send an error notification."""
        embed = {"title": title, "description": message, "color": 0xE74C3C}
        return await self._send_embed(embed, NotificationLevel.ERROR)

    # =========================================================================
    # Master Control Console Integration
    # =========================================================================

    async def send(self, message: str, **kwargs: Any) -> bool:
        """
        Send a simple text message notification.

        This method is used by Master Control Console for generic notifications.

        Args:
            message: Message text to send
            **kwargs: Additional parameters (ignored)

        Returns:
            True if sent successfully
        """
        embed = {
            "title": "📢 通知",
            "description": message,
            "color": 0x3498DB,  # Blue
        }
        return await self._send_embed(embed, NotificationLevel.INFO)

    async def notify_bot_registered(
        self,
        bot_id: str,
        bot_type: str,
        symbol: str,
    ) -> None:
        """
        Send bot registered notification.

        Args:
            bot_id: Bot identifier
            bot_type: Type of bot (grid, dca, etc.)
            symbol: Trading symbol
        """
        embed = {
            "title": "🤖 機器人已註冊",
            "description": f"新機器人已加入系統",
            "color": 0x3498DB,  # Blue
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "類型", "value": bot_type.upper(), "inline": True},
                {"name": "交易對", "value": symbol, "inline": True},
            ],
        }
        await self._send_embed(embed, NotificationLevel.INFO)

    async def notify_bot_state_changed(
        self,
        bot_id: str,
        old_state: str,
        new_state: str,
        message: str = "",
    ) -> None:
        """
        Send bot state changed notification.

        Args:
            bot_id: Bot identifier
            old_state: Previous state
            new_state: New state
            message: Optional message
        """
        # Choose color based on new state
        state_colors = {
            "running": 0x2ECC71,    # Green
            "paused": 0xF39C12,     # Orange
            "stopped": 0x95A5A6,    # Gray
            "error": 0xE74C3C,      # Red
            "initializing": 0x3498DB,  # Blue
            "stopping": 0xF39C12,   # Orange
        }
        color = state_colors.get(new_state.lower(), 0x3498DB)

        # Choose emoji based on new state
        state_emojis = {
            "running": "▶️",
            "paused": "⏸️",
            "stopped": "⏹️",
            "error": "❌",
            "initializing": "🔄",
            "stopping": "⏳",
        }
        emoji = state_emojis.get(new_state.lower(), "📋")

        embed = {
            "title": f"{emoji} 狀態變更",
            "description": message if message else f"機器人狀態已更新",
            "color": color,
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "狀態", "value": f"{old_state} → {new_state}", "inline": True},
            ],
        }
        await self._send_embed(embed, NotificationLevel.INFO)

    async def notify_bot_timeout(
        self,
        bot_id: str,
        message: str = "",
    ) -> None:
        """
        Send bot timeout notification.

        Args:
            bot_id: Bot identifier
            message: Timeout message
        """
        embed = {
            "title": "⚠️ 心跳超時",
            "description": message if message else f"機器人 {bot_id} 心跳超時",
            "color": 0xE74C3C,  # Red
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "狀態", "value": "無回應", "inline": True},
            ],
        }
        await self._send_embed(embed, NotificationLevel.WARNING)

    async def notify_bot_recovered(self, bot_id: str) -> None:
        """
        Send bot recovered notification.

        Args:
            bot_id: Bot identifier
        """
        embed = {
            "title": "✅ 心跳恢復",
            "description": f"機器人 {bot_id} 已恢復正常",
            "color": 0x2ECC71,  # Green
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "狀態", "value": "正常", "inline": True},
            ],
        }
        await self._send_embed(embed, NotificationLevel.SUCCESS)

    async def notify_circuit_breaker_open(
        self,
        bot_id: str,
        consecutive_failures: int,
        duration: int,
    ) -> None:
        """
        Send circuit breaker open notification.

        Args:
            bot_id: Bot identifier
            consecutive_failures: Number of consecutive failures
            duration: Duration circuit will stay open (seconds)
        """
        embed = {
            "title": "🔴 熔斷器開啟",
            "description": f"機器人 {bot_id} 自動重啟已暫停",
            "color": 0xE74C3C,  # Red
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "連續失敗次數", "value": str(consecutive_failures), "inline": True},
                {"name": "熔斷時間", "value": f"{duration // 60} 分鐘", "inline": True},
            ],
        }
        # Circuit breaker is critical - always send
        await self._send_embed(embed, NotificationLevel.CRITICAL)

    async def notify_circuit_breaker_closed(self, bot_id: str) -> None:
        """
        Send circuit breaker closed notification.

        Args:
            bot_id: Bot identifier
        """
        embed = {
            "title": "🟢 熔斷器關閉",
            "description": f"機器人 {bot_id} 自動重啟已恢復",
            "color": 0x2ECC71,  # Green
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "狀態", "value": "可重啟", "inline": True},
            ],
        }
        await self._send_embed(embed, NotificationLevel.SUCCESS)

    # =========================================================================
    # Fund Manager Notifications
    # =========================================================================

    async def notify_deposit_detected(
        self,
        amount: Decimal | float | str,
        total_balance: Decimal | float | str,
    ) -> bool:
        """
        Send deposit detected notification.

        Args:
            amount: Deposit amount
            total_balance: New total balance

        Returns:
            True if sent
        """
        embed = {
            "title": "💰 入金偵測",
            "description": f"偵測到新的入金",
            "color": 0x2ECC71,  # Green
            "fields": [
                {"name": "入金金額", "value": f"{amount} USDT", "inline": True},
                {"name": "總餘額", "value": f"{total_balance} USDT", "inline": True},
            ],
        }
        return await self._send_embed(embed, NotificationLevel.SUCCESS)

    async def notify_fund_allocated(
        self,
        bot_id: str,
        amount: Decimal | float | str,
        total_allocated: Decimal | float | str,
    ) -> bool:
        """
        Send fund allocated notification.

        Args:
            bot_id: Bot identifier
            amount: Amount allocated
            total_allocated: Total allocation for this bot

        Returns:
            True if sent
        """
        embed = {
            "title": "📊 資金分配",
            "description": f"已分配資金到機器人",
            "color": 0x3498DB,  # Blue
            "fields": [
                {"name": "Bot ID", "value": bot_id, "inline": True},
                {"name": "分配金額", "value": f"{amount} USDT", "inline": True},
                {"name": "總配額", "value": f"{total_allocated} USDT", "inline": True},
            ],
        }
        dedup_key = self._hash_content("fund_allocated", bot_id, amount)
        return await self._send_embed(embed, NotificationLevel.INFO, dedup_key)

    async def notify_fund_dispatch_complete(
        self,
        total_dispatched: Decimal | float | str,
        bot_count: int,
        trigger: str = "manual",
    ) -> bool:
        """
        Send fund dispatch complete notification.

        Args:
            total_dispatched: Total amount dispatched
            bot_count: Number of bots that received funds
            trigger: What triggered the dispatch

        Returns:
            True if sent
        """
        trigger_text = {
            "manual": "手動觸發",
            "deposit": "入金偵測",
            "rebalance": "重新平衡",
        }.get(trigger, trigger)

        embed = {
            "title": "✅ 資金調度完成",
            "description": f"資金已成功分配到 {bot_count} 個機器人",
            "color": 0x2ECC71,  # Green
            "fields": [
                {"name": "總分配金額", "value": f"{total_dispatched} USDT", "inline": True},
                {"name": "機器人數量", "value": str(bot_count), "inline": True},
                {"name": "觸發來源", "value": trigger_text, "inline": True},
            ],
        }
        return await self._send_embed(embed, NotificationLevel.SUCCESS)

    async def notify_fund_dispatch_failed(
        self,
        error_message: str,
        failed_bots: int = 0,
    ) -> bool:
        """
        Send fund dispatch failed notification.

        Args:
            error_message: Error description
            failed_bots: Number of failed allocations

        Returns:
            True if sent
        """
        embed = {
            "title": "❌ 資金調度失敗",
            "description": error_message,
            "color": 0xE74C3C,  # Red
            "fields": [
                {"name": "失敗數量", "value": str(failed_bots), "inline": True},
            ],
        }
        return await self._send_embed(embed, NotificationLevel.ERROR)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def close(self) -> None:
        """Close the notification manager and release resources."""
        if self._discord:
            await self._discord.close()
            self._discord = None


# =========================================================================
# Global Instance
# =========================================================================

_manager: Optional[NotificationManager] = None


def get_notification_manager() -> NotificationManager:
    """
    Get the global notification manager instance.

    Returns:
        NotificationManager instance

    Raises:
        RuntimeError: If not initialized
    """
    if _manager is None:
        raise RuntimeError(
            "NotificationManager not initialized. Call init_notification_manager() first."
        )
    return _manager


def init_notification_manager(config: Optional[Any] = None) -> NotificationManager:
    """
    Initialize the global notification manager.

    Args:
        config: Configuration object (optional). If None, uses environment variables.

    Returns:
        NotificationManager instance
    """
    global _manager

    if config is not None:
        _manager = NotificationManager.from_config(config)
    else:
        _manager = NotificationManager.from_env()

    return _manager


async def close_notification_manager() -> None:
    """Close the global notification manager."""
    global _manager

    if _manager is not None:
        await _manager.close()
        _manager = None
