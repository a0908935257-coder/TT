"""
Discord Embed Builder.

Provides builder pattern for creating Discord embed messages.
"""

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Optional


class EmbedColor(IntEnum):
    """Discord embed colors."""

    SUCCESS = 0x00FF00   # Green - Success/Buy
    ERROR = 0xFF0000     # Red - Error
    WARNING = 0xFFA500   # Orange - Warning
    INFO = 0x0099FF      # Blue - Info
    BUY = 0x00FF00       # Green - Buy order
    SELL = 0xFF6B6B      # Light red - Sell order
    NEUTRAL = 0x808080   # Gray - Neutral
    PURPLE = 0x9B59B6    # Purple - Special
    GOLD = 0xFFD700      # Gold - Premium/Important


class DiscordEmbed:
    """
    Discord Embed builder using the Builder pattern.

    Allows fluent construction of Discord embed messages
    with method chaining.

    Example:
        >>> embed = (
        ...     DiscordEmbed()
        ...     .set_title("訂單成交")
        ...     .set_color(EmbedColor.SUCCESS)
        ...     .add_field("交易對", "BTCUSDT", inline=True)
        ...     .add_field("價格", "50,000", inline=True)
        ...     .set_timestamp()
        ...     .build()
        ... )
    """

    def __init__(self):
        """Initialize empty embed."""
        self._title: Optional[str] = None
        self._description: Optional[str] = None
        self._color: Optional[int] = None
        self._fields: list[dict[str, Any]] = []
        self._timestamp: Optional[str] = None
        self._footer: Optional[dict[str, str]] = None
        self._thumbnail: Optional[dict[str, str]] = None
        self._image: Optional[dict[str, str]] = None
        self._author: Optional[dict[str, str]] = None
        self._url: Optional[str] = None

    def set_title(self, title: str) -> "DiscordEmbed":
        """
        Set embed title.

        Args:
            title: Title text (max 256 characters)

        Returns:
            Self for chaining
        """
        self._title = title[:256]
        return self

    def set_description(self, description: str) -> "DiscordEmbed":
        """
        Set embed description.

        Args:
            description: Description text (max 4096 characters)

        Returns:
            Self for chaining
        """
        self._description = description[:4096]
        return self

    def set_color(self, color: EmbedColor | int) -> "DiscordEmbed":
        """
        Set embed color.

        Args:
            color: EmbedColor or hex integer

        Returns:
            Self for chaining
        """
        self._color = int(color)
        return self

    def set_url(self, url: str) -> "DiscordEmbed":
        """
        Set embed URL (makes title clickable).

        Args:
            url: URL string

        Returns:
            Self for chaining
        """
        self._url = url
        return self

    def add_field(
        self,
        name: str,
        value: str,
        inline: bool = False,
    ) -> "DiscordEmbed":
        """
        Add a field to the embed.

        Args:
            name: Field name (max 256 characters)
            value: Field value (max 1024 characters)
            inline: Display inline with other fields

        Returns:
            Self for chaining
        """
        if len(self._fields) >= 25:
            # Discord limit: max 25 fields
            return self

        self._fields.append({
            "name": str(name)[:256],
            "value": str(value)[:1024],
            "inline": inline,
        })
        return self

    def set_timestamp(
        self,
        timestamp: Optional[datetime] = None,
    ) -> "DiscordEmbed":
        """
        Set embed timestamp.

        Args:
            timestamp: Datetime (default: now)

        Returns:
            Self for chaining
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        self._timestamp = timestamp.isoformat()
        return self

    def set_footer(
        self,
        text: str,
        icon_url: Optional[str] = None,
    ) -> "DiscordEmbed":
        """
        Set embed footer.

        Args:
            text: Footer text (max 2048 characters)
            icon_url: Footer icon URL (optional)

        Returns:
            Self for chaining
        """
        self._footer = {"text": text[:2048]}
        if icon_url:
            self._footer["icon_url"] = icon_url
        return self

    def set_thumbnail(self, url: str) -> "DiscordEmbed":
        """
        Set embed thumbnail.

        Args:
            url: Thumbnail image URL

        Returns:
            Self for chaining
        """
        self._thumbnail = {"url": url}
        return self

    def set_image(self, url: str) -> "DiscordEmbed":
        """
        Set embed image.

        Args:
            url: Image URL

        Returns:
            Self for chaining
        """
        self._image = {"url": url}
        return self

    def set_author(
        self,
        name: str,
        url: Optional[str] = None,
        icon_url: Optional[str] = None,
    ) -> "DiscordEmbed":
        """
        Set embed author.

        Args:
            name: Author name (max 256 characters)
            url: Author URL (optional)
            icon_url: Author icon URL (optional)

        Returns:
            Self for chaining
        """
        self._author = {"name": name[:256]}
        if url:
            self._author["url"] = url
        if icon_url:
            self._author["icon_url"] = icon_url
        return self

    def build(self) -> dict[str, Any]:
        """
        Build the embed as a dictionary.

        Returns:
            Embed data dictionary ready for Discord API
        """
        embed: dict[str, Any] = {}

        if self._title:
            embed["title"] = self._title
        if self._description:
            embed["description"] = self._description
        if self._color is not None:
            embed["color"] = self._color
        if self._url:
            embed["url"] = self._url
        if self._fields:
            embed["fields"] = self._fields
        if self._timestamp:
            embed["timestamp"] = self._timestamp
        if self._footer:
            embed["footer"] = self._footer
        if self._thumbnail:
            embed["thumbnail"] = self._thumbnail
        if self._image:
            embed["image"] = self._image
        if self._author:
            embed["author"] = self._author

        return embed

    def to_dict(self) -> dict[str, Any]:
        """Alias for build()."""
        return self.build()

    # =========================================================================
    # Factory Methods for Common Embeds
    # =========================================================================

    @classmethod
    def order_filled(
        cls,
        symbol: str,
        side: str,
        price: str,
        quantity: str,
        total: Optional[str] = None,
    ) -> "DiscordEmbed":
        """
        Create an order filled embed.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Order price
            quantity: Order quantity
            total: Total value (optional)

        Returns:
            Configured DiscordEmbed
        """
        color = EmbedColor.BUY if side.upper() == "BUY" else EmbedColor.SELL
        emoji = "🟢" if side.upper() == "BUY" else "🔴"

        embed = (
            cls()
            .set_title(f"{emoji} 訂單成交 - {symbol}")
            .set_color(color)
            .add_field("方向", side.upper(), inline=True)
            .add_field("價格", price, inline=True)
            .add_field("數量", quantity, inline=True)
        )

        if total:
            embed.add_field("總額", total, inline=True)

        return embed.set_timestamp()

    @classmethod
    def position_update(
        cls,
        symbol: str,
        side: str,
        entry_price: str,
        mark_price: str,
        pnl: str,
        pnl_percent: str,
    ) -> "DiscordEmbed":
        """
        Create a position update embed.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            entry_price: Entry price
            mark_price: Current mark price
            pnl: Unrealized PnL
            pnl_percent: PnL percentage

        Returns:
            Configured DiscordEmbed
        """
        # Determine color based on PnL
        try:
            is_profit = float(pnl.replace(",", "").replace("+", "")) >= 0
        except (ValueError, AttributeError):
            is_profit = True

        color = EmbedColor.SUCCESS if is_profit else EmbedColor.ERROR
        emoji = "📈" if is_profit else "📉"

        return (
            cls()
            .set_title(f"{emoji} 持倉更新 - {symbol}")
            .set_color(color)
            .add_field("方向", side.upper(), inline=True)
            .add_field("開倉價", entry_price, inline=True)
            .add_field("標記價", mark_price, inline=True)
            .add_field("未實現盈虧", pnl, inline=True)
            .add_field("收益率", pnl_percent, inline=True)
            .set_timestamp()
        )

    @classmethod
    def alert(
        cls,
        title: str,
        message: str,
        level: str = "info",
    ) -> "DiscordEmbed":
        """
        Create an alert embed.

        Args:
            title: Alert title
            message: Alert message
            level: Alert level (info, warning, error, critical)

        Returns:
            Configured DiscordEmbed
        """
        level_config = {
            "info": (EmbedColor.INFO, "ℹ️"),
            "success": (EmbedColor.SUCCESS, "✅"),
            "warning": (EmbedColor.WARNING, "⚠️"),
            "error": (EmbedColor.ERROR, "❌"),
            "critical": (EmbedColor.ERROR, "🚨"),
        }

        color, emoji = level_config.get(level.lower(), (EmbedColor.INFO, "ℹ️"))

        return (
            cls()
            .set_title(f"{emoji} {title}")
            .set_description(message)
            .set_color(color)
            .set_timestamp()
        )
