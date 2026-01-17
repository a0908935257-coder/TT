"""
Notification Templates.

Provides pre-built notification templates for trading events, alerts, and reports.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from .discord.embed import DiscordEmbed, EmbedColor


class NotificationTemplates:
    """
    Collection of notification templates for trading bot events.

    All methods are static and return DiscordEmbed objects ready to be sent.

    Example:
        >>> embed = NotificationTemplates.order_filled_embed(
        ...     symbol="BTCUSDT",
        ...     side="BUY",
        ...     price=Decimal("50000"),
        ...     quantity=Decimal("0.1"),
        ... )
        >>> await notifier.send_embed(embed.build())
    """

    # =========================================================================
    # Formatting Utilities
    # =========================================================================

    @staticmethod
    def format_price(price: Decimal | float | str) -> str:
        """
        Format price with thousands separator.

        Args:
            price: Price value

        Returns:
            Formatted price string (e.g., "50,000.00")
        """
        try:
            value = Decimal(str(price))
            # Determine decimal places based on magnitude
            if value >= 1000:
                return f"{value:,.2f}"
            elif value >= 1:
                return f"{value:,.4f}"
            else:
                return f"{value:.8f}"
        except Exception:
            return str(price)

    @staticmethod
    def format_quantity(qty: Decimal | float | str) -> str:
        """
        Format quantity with appropriate precision.

        Args:
            qty: Quantity value

        Returns:
            Formatted quantity string (e.g., "0.12345")
        """
        try:
            value = Decimal(str(qty))
            # Use up to 8 decimal places, strip trailing zeros
            formatted = f"{value:.8f}".rstrip("0").rstrip(".")
            return formatted
        except Exception:
            return str(qty)

    @staticmethod
    def format_pnl(pnl: Decimal | float | str) -> str:
        """
        Format PnL with sign.

        Args:
            pnl: PnL value

        Returns:
            Formatted PnL string (e.g., "+350.00" or "-120.00")
        """
        try:
            value = Decimal(str(pnl))
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:,.2f}"
        except Exception:
            return str(pnl)

    @staticmethod
    def format_percent(pct: Decimal | float | str) -> str:
        """
        Format percentage with sign.

        Args:
            pct: Percentage value

        Returns:
            Formatted percentage string (e.g., "+5.25%")
        """
        try:
            value = Decimal(str(pct))
            sign = "+" if value >= 0 else ""
            return f"{sign}{value:.2f}%"
        except Exception:
            return str(pct)

    @staticmethod
    def format_time(dt: datetime | None = None) -> str:
        """
        Format datetime.

        Args:
            dt: Datetime object (default: now)

        Returns:
            Formatted time string (e.g., "2024-01-15 10:30")
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")

    @staticmethod
    def format_duration(seconds: int | float) -> str:
        """
        Format duration in human-readable format.

        Args:
            seconds: Duration in seconds

        Returns:
            Formatted duration (e.g., "2h 30m" or "45m 30s")
        """
        seconds = int(seconds)
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    @staticmethod
    def get_side_emoji(side: str) -> str:
        """
        Get emoji for order side.

        Args:
            side: BUY, SELL, LONG, or SHORT

        Returns:
            Emoji string
        """
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return "🟢"
        elif side_upper in ("SELL", "SHORT"):
            return "🔴"
        return "⚪"

    @staticmethod
    def get_pnl_emoji(pnl: Decimal | float | str) -> str:
        """
        Get emoji for PnL.

        Args:
            pnl: PnL value

        Returns:
            Emoji string
        """
        try:
            value = Decimal(str(pnl))
            if value > 0:
                return "📈"
            elif value < 0:
                return "📉"
            return "➖"
        except Exception:
            return "❓"

    @staticmethod
    def get_side_color(side: str) -> EmbedColor:
        """
        Get color for order side.

        Args:
            side: BUY, SELL, LONG, or SHORT

        Returns:
            EmbedColor
        """
        side_upper = side.upper()
        if side_upper in ("BUY", "LONG"):
            return EmbedColor.BUY
        elif side_upper in ("SELL", "SHORT"):
            return EmbedColor.SELL
        return EmbedColor.NEUTRAL

    # =========================================================================
    # Order Templates
    # =========================================================================

    @classmethod
    def order_placed_embed(
        cls,
        symbol: str,
        side: str,
        order_type: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
    ) -> DiscordEmbed:
        """
        Create order placed notification embed.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            order_type: LIMIT, MARKET, etc.
            price: Order price
            quantity: Order quantity

        Returns:
            DiscordEmbed object
        """
        emoji = cls.get_side_emoji(side)
        color = cls.get_side_color(side)

        return (
            DiscordEmbed()
            .set_title(f"{emoji} Order Placed - {symbol}")
            .set_color(color)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Type", order_type.upper(), inline=True)
            .add_field("Price", cls.format_price(price), inline=True)
            .add_field("Quantity", cls.format_quantity(quantity), inline=True)
            .set_timestamp()
        )

    @classmethod
    def order_filled_embed(
        cls,
        symbol: str,
        side: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
        pnl: Optional[Decimal | float | str] = None,
    ) -> DiscordEmbed:
        """
        Create order filled notification embed.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Fill price
            quantity: Fill quantity
            pnl: Realized PnL (optional)

        Returns:
            DiscordEmbed object
        """
        emoji = cls.get_side_emoji(side)
        color = cls.get_side_color(side)

        embed = (
            DiscordEmbed()
            .set_title(f"{emoji} Order Filled - {symbol}")
            .set_color(color)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Price", cls.format_price(price), inline=True)
            .add_field("Quantity", cls.format_quantity(quantity), inline=True)
        )

        if pnl is not None:
            pnl_emoji = cls.get_pnl_emoji(pnl)
            embed.add_field(f"{pnl_emoji} Realized PnL", cls.format_pnl(pnl), inline=True)

        return embed.set_timestamp()

    @classmethod
    def order_canceled_embed(
        cls,
        symbol: str,
        side: str,
        price: Decimal | float | str,
        quantity: Decimal | float | str,
        reason: Optional[str] = None,
    ) -> DiscordEmbed:
        """
        Create order canceled notification embed.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Order price
            quantity: Order quantity
            reason: Cancellation reason (optional)

        Returns:
            DiscordEmbed object
        """
        embed = (
            DiscordEmbed()
            .set_title(f"❌ Order Canceled - {symbol}")
            .set_color(EmbedColor.NEUTRAL)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Price", cls.format_price(price), inline=True)
            .add_field("Quantity", cls.format_quantity(quantity), inline=True)
        )

        if reason:
            embed.add_field("Reason", reason, inline=False)

        return embed.set_timestamp()

    # =========================================================================
    # Position Templates
    # =========================================================================

    @classmethod
    def position_opened_embed(
        cls,
        symbol: str,
        side: str,
        quantity: Decimal | float | str,
        entry_price: Decimal | float | str,
        leverage: Optional[int] = None,
    ) -> DiscordEmbed:
        """
        Create position opened notification embed.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            quantity: Position size
            entry_price: Entry price
            leverage: Leverage (optional)

        Returns:
            DiscordEmbed object
        """
        emoji = cls.get_side_emoji(side)
        color = cls.get_side_color(side)

        embed = (
            DiscordEmbed()
            .set_title(f"{emoji} Position Opened - {symbol}")
            .set_color(color)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Entry Price", cls.format_price(entry_price), inline=True)
            .add_field("Size", cls.format_quantity(quantity), inline=True)
        )

        if leverage:
            embed.add_field("Leverage", f"{leverage}x", inline=True)

        return embed.set_timestamp()

    @classmethod
    def position_closed_embed(
        cls,
        symbol: str,
        side: str,
        quantity: Decimal | float | str,
        entry_price: Decimal | float | str,
        exit_price: Decimal | float | str,
        pnl: Decimal | float | str,
        holding_time: int | float,
    ) -> DiscordEmbed:
        """
        Create position closed notification embed.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            quantity: Position size
            entry_price: Entry price
            exit_price: Exit price
            pnl: Realized PnL
            holding_time: Holding time in seconds

        Returns:
            DiscordEmbed object
        """
        pnl_emoji = cls.get_pnl_emoji(pnl)
        pnl_decimal = Decimal(str(pnl))
        color = EmbedColor.SUCCESS if pnl_decimal >= 0 else EmbedColor.ERROR

        return (
            DiscordEmbed()
            .set_title(f"{pnl_emoji} Position Closed - {symbol}")
            .set_color(color)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Entry", cls.format_price(entry_price), inline=True)
            .add_field("Exit", cls.format_price(exit_price), inline=True)
            .add_field("Size", cls.format_quantity(quantity), inline=True)
            .add_field("PnL", cls.format_pnl(pnl), inline=True)
            .add_field("Duration", cls.format_duration(holding_time), inline=True)
            .set_timestamp()
        )

    # =========================================================================
    # Risk Alert Templates
    # =========================================================================

    @classmethod
    def risk_alert_embed(
        cls,
        alert_type: str,
        message: str,
        details: Optional[dict[str, Any]] = None,
    ) -> DiscordEmbed:
        """
        Create generic risk alert embed.

        Args:
            alert_type: Type of alert (e.g., "Margin Warning", "Circuit Breaker")
            message: Alert message
            details: Additional details as key-value pairs

        Returns:
            DiscordEmbed object
        """
        embed = (
            DiscordEmbed()
            .set_title(f"⚠️ Risk Alert: {alert_type}")
            .set_description(message)
            .set_color(EmbedColor.WARNING)
        )

        if details:
            for key, value in details.items():
                embed.add_field(key, str(value), inline=True)

        return embed.set_timestamp()

    @classmethod
    def drawdown_alert_embed(
        cls,
        current_drawdown: Decimal | float | str,
        max_threshold: Decimal | float | str,
        peak_balance: Decimal | float | str,
        current_balance: Decimal | float | str,
    ) -> DiscordEmbed:
        """
        Create drawdown alert embed.

        Args:
            current_drawdown: Current drawdown percentage
            max_threshold: Maximum allowed drawdown
            peak_balance: Peak balance value
            current_balance: Current balance value

        Returns:
            DiscordEmbed object
        """
        return (
            DiscordEmbed()
            .set_title("🚨 Drawdown Alert")
            .set_description(
                f"Current drawdown ({cls.format_percent(current_drawdown)}) "
                f"approaching limit ({cls.format_percent(max_threshold)})"
            )
            .set_color(EmbedColor.ERROR)
            .add_field("Current Drawdown", cls.format_percent(current_drawdown), inline=True)
            .add_field("Max Threshold", cls.format_percent(max_threshold), inline=True)
            .add_field("Peak Balance", cls.format_price(peak_balance), inline=True)
            .add_field("Current Balance", cls.format_price(current_balance), inline=True)
            .set_timestamp()
        )

    @classmethod
    def daily_loss_alert_embed(
        cls,
        current_loss: Decimal | float | str,
        max_threshold: Decimal | float | str,
        remaining: Decimal | float | str,
    ) -> DiscordEmbed:
        """
        Create daily loss alert embed.

        Args:
            current_loss: Current daily loss percentage
            max_threshold: Maximum allowed daily loss
            remaining: Remaining budget before limit

        Returns:
            DiscordEmbed object
        """
        return (
            DiscordEmbed()
            .set_title("🚨 Daily Loss Alert")
            .set_description(
                f"Daily loss ({cls.format_percent(current_loss)}) "
                f"approaching limit ({cls.format_percent(max_threshold)})"
            )
            .set_color(EmbedColor.ERROR)
            .add_field("Current Loss", cls.format_percent(current_loss), inline=True)
            .add_field("Max Threshold", cls.format_percent(max_threshold), inline=True)
            .add_field("Remaining", cls.format_price(remaining), inline=True)
            .set_timestamp()
        )

    @classmethod
    def liquidation_warning_embed(
        cls,
        symbol: str,
        side: str,
        distance_percent: Decimal | float | str,
        liquidation_price: Decimal | float | str,
        current_price: Decimal | float | str,
    ) -> DiscordEmbed:
        """
        Create liquidation warning embed.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            distance_percent: Distance to liquidation in percent
            liquidation_price: Liquidation price
            current_price: Current market price

        Returns:
            DiscordEmbed object
        """
        return (
            DiscordEmbed()
            .set_title(f"🚨 Liquidation Warning - {symbol}")
            .set_description(
                f"Position is {cls.format_percent(distance_percent)} away from liquidation!"
            )
            .set_color(EmbedColor.ERROR)
            .add_field("Side", side.upper(), inline=True)
            .add_field("Distance", cls.format_percent(distance_percent), inline=True)
            .add_field("Current Price", cls.format_price(current_price), inline=True)
            .add_field("Liquidation Price", cls.format_price(liquidation_price), inline=True)
            .set_timestamp()
        )

    # =========================================================================
    # Bot Status Templates
    # =========================================================================

    @classmethod
    def bot_started_embed(
        cls,
        bot_name: str,
        bot_type: str,
        config_summary: dict[str, Any],
    ) -> DiscordEmbed:
        """
        Create bot started notification embed.

        Args:
            bot_name: Bot name/identifier
            bot_type: Type of bot (e.g., "Grid Bot", "DCA Bot")
            config_summary: Key configuration parameters

        Returns:
            DiscordEmbed object
        """
        embed = (
            DiscordEmbed()
            .set_title(f"🚀 {bot_name} Started")
            .set_description(f"Trading bot ({bot_type}) is now running")
            .set_color(EmbedColor.SUCCESS)
        )

        for key, value in config_summary.items():
            embed.add_field(key, str(value), inline=True)

        return embed.set_timestamp()

    @classmethod
    def bot_stopped_embed(
        cls,
        bot_name: str,
        reason: str,
        runtime: int | float,
        total_pnl: Decimal | float | str,
    ) -> DiscordEmbed:
        """
        Create bot stopped notification embed.

        Args:
            bot_name: Bot name/identifier
            reason: Stop reason
            runtime: Total runtime in seconds
            total_pnl: Total PnL during runtime

        Returns:
            DiscordEmbed object
        """
        pnl_emoji = cls.get_pnl_emoji(total_pnl)

        return (
            DiscordEmbed()
            .set_title(f"🛑 {bot_name} Stopped")
            .set_description(f"Reason: {reason}")
            .set_color(EmbedColor.WARNING)
            .add_field("Runtime", cls.format_duration(runtime), inline=True)
            .add_field(f"{pnl_emoji} Total PnL", cls.format_pnl(total_pnl), inline=True)
            .set_timestamp()
        )

    @classmethod
    def bot_error_embed(
        cls,
        bot_name: str,
        error_type: str,
        error_message: str,
    ) -> DiscordEmbed:
        """
        Create bot error notification embed.

        Args:
            bot_name: Bot name/identifier
            error_type: Type of error
            error_message: Error message

        Returns:
            DiscordEmbed object
        """
        return (
            DiscordEmbed()
            .set_title(f"❌ {bot_name} Error")
            .set_description(f"**{error_type}**\n```{error_message[:1000]}```")
            .set_color(EmbedColor.ERROR)
            .set_timestamp()
        )

    @classmethod
    def connection_lost_embed(
        cls,
        service_name: str,
        error: Optional[str] = None,
    ) -> DiscordEmbed:
        """
        Create connection lost notification embed.

        Args:
            service_name: Name of the disconnected service
            error: Error message (optional)

        Returns:
            DiscordEmbed object
        """
        embed = (
            DiscordEmbed()
            .set_title(f"🔌 Connection Lost - {service_name}")
            .set_color(EmbedColor.ERROR)
        )

        if error:
            embed.set_description(f"Error: {error}")

        return embed.set_timestamp()

    @classmethod
    def connection_restored_embed(
        cls,
        service_name: str,
        downtime: int | float,
    ) -> DiscordEmbed:
        """
        Create connection restored notification embed.

        Args:
            service_name: Name of the reconnected service
            downtime: Downtime duration in seconds

        Returns:
            DiscordEmbed object
        """
        return (
            DiscordEmbed()
            .set_title(f"✅ Connection Restored - {service_name}")
            .set_description(f"Downtime: {cls.format_duration(downtime)}")
            .set_color(EmbedColor.SUCCESS)
            .set_timestamp()
        )

    # =========================================================================
    # Report Templates
    # =========================================================================

    @classmethod
    def daily_report_embed(
        cls,
        bot_name: str,
        date: str,
        starting_balance: Decimal | float | str,
        ending_balance: Decimal | float | str,
        total_pnl: Decimal | float | str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        win_rate: Optional[Decimal | float | str] = None,
        largest_win: Optional[Decimal | float | str] = None,
        largest_loss: Optional[Decimal | float | str] = None,
        avg_trade_pnl: Optional[Decimal | float | str] = None,
    ) -> DiscordEmbed:
        """
        Create daily performance report embed.

        Args:
            bot_name: Bot name/identifier
            date: Report date string
            starting_balance: Starting balance
            ending_balance: Ending balance
            total_pnl: Total PnL for the day
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            win_rate: Win rate percentage (optional)
            largest_win: Largest winning trade (optional)
            largest_loss: Largest losing trade (optional)
            avg_trade_pnl: Average PnL per trade (optional)

        Returns:
            DiscordEmbed object
        """
        pnl_decimal = Decimal(str(total_pnl))
        pnl_emoji = cls.get_pnl_emoji(total_pnl)
        color = EmbedColor.SUCCESS if pnl_decimal >= 0 else EmbedColor.ERROR

        # Calculate win rate if not provided
        if win_rate is None and total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100

        embed = (
            DiscordEmbed()
            .set_title(f"📊 Daily Report - {bot_name}")
            .set_description(f"**Date:** {date}")
            .set_color(color)
            .add_field("Starting Balance", cls.format_price(starting_balance), inline=True)
            .add_field("Ending Balance", cls.format_price(ending_balance), inline=True)
            .add_field(f"{pnl_emoji} Total PnL", cls.format_pnl(total_pnl), inline=True)
            .add_field("Total Trades", str(total_trades), inline=True)
            .add_field("Wins / Losses", f"{winning_trades} / {losing_trades}", inline=True)
        )

        if win_rate is not None:
            embed.add_field("Win Rate", cls.format_percent(win_rate), inline=True)

        if largest_win is not None:
            embed.add_field("Largest Win", cls.format_pnl(largest_win), inline=True)

        if largest_loss is not None:
            embed.add_field("Largest Loss", cls.format_pnl(largest_loss), inline=True)

        if avg_trade_pnl is not None:
            embed.add_field("Avg Trade PnL", cls.format_pnl(avg_trade_pnl), inline=True)

        return embed.set_timestamp()
