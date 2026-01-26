"""
Discord Webhook Client.

Provides Discord webhook integration with rate limiting and retry logic.
"""

import asyncio
import time
from typing import Any, Optional

import aiohttp

from src.core import get_logger
from src.notification.base import BaseNotifier, NotificationLevel

from .embed import DiscordEmbed, EmbedColor

logger = get_logger(__name__)


# Level to color mapping
LEVEL_COLORS = {
    NotificationLevel.DEBUG: EmbedColor.NEUTRAL,
    NotificationLevel.INFO: EmbedColor.INFO,
    NotificationLevel.SUCCESS: EmbedColor.SUCCESS,
    NotificationLevel.WARNING: EmbedColor.WARNING,
    NotificationLevel.ERROR: EmbedColor.ERROR,
    NotificationLevel.CRITICAL: EmbedColor.ERROR,
}

# Level to emoji mapping
LEVEL_EMOJIS = {
    NotificationLevel.DEBUG: "🔍",
    NotificationLevel.INFO: "ℹ️",
    NotificationLevel.SUCCESS: "✅",
    NotificationLevel.WARNING: "⚠️",
    NotificationLevel.ERROR: "❌",
    NotificationLevel.CRITICAL: "🚨",
}


class DiscordNotifier(BaseNotifier):
    """
    Discord webhook notification client.

    Supports:
    - Plain text messages
    - Rich embed messages
    - Rate limiting (25 requests per minute default)
    - Automatic retry with exponential backoff
    - 429 rate limit handling

    Example:
        >>> async with DiscordNotifier(webhook_url) as notifier:
        ...     await notifier.send("Hello!", NotificationLevel.INFO)
        ...     await notifier.success("訂單成交", "BTCUSDT 買入 0.01 BTC")
    """

    def __init__(
        self,
        webhook_url: str,
        username: str = "Trading Bot",
        avatar_url: Optional[str] = None,
        rate_limit: int = 25,
        max_retries: int = 3,
    ):
        """
        Initialize Discord notifier.

        Args:
            webhook_url: Discord webhook URL
            username: Bot display name
            avatar_url: Bot avatar URL (optional)
            rate_limit: Max requests per minute (default 25)
            max_retries: Max retry attempts (default 3)
        """
        self._webhook_url = webhook_url
        self._username = username
        self._avatar_url = avatar_url
        self._rate_limit = rate_limit
        self._max_retries = max_retries

        # Rate limiting
        self._request_times: list[float] = []
        self._rate_limit_lock = asyncio.Lock()

        # HTTP session
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _check_rate_limit(self) -> None:
        """Check and enforce rate limiting."""
        async with self._rate_limit_lock:
            now = time.time()
            # Remove requests older than 60 seconds
            self._request_times = [t for t in self._request_times if now - t < 60]

            if len(self._request_times) >= self._rate_limit:
                # Wait until oldest request is 60 seconds old
                wait_time = 60 - (now - self._request_times[0])
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.1f}s")
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    async def _send_request(
        self,
        payload: dict[str, Any],
    ) -> bool:
        """
        Send request to webhook with retry logic.

        Args:
            payload: Request payload

        Returns:
            True if successful
        """
        await self._check_rate_limit()

        session = await self._get_session()
        last_error: Optional[Exception] = None

        for attempt in range(self._max_retries):
            try:
                async with session.post(
                    self._webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status == 204:
                        # Success - no content
                        return True

                    if response.status == 200:
                        # Success with content
                        return True

                    if response.status == 429:
                        # Rate limited by Discord
                        data = await response.json()
                        retry_after = data.get("retry_after", 1)
                        # Ensure minimum wait time to prevent infinite fast retry loop
                        retry_after = max(float(retry_after), 1.0)
                        logger.warning(
                            f"Discord rate limited, retry after {retry_after}s"
                        )
                        await asyncio.sleep(retry_after)
                        continue

                    if response.status >= 500:
                        # Server error - retry
                        logger.warning(
                            f"Discord server error {response.status}, "
                            f"attempt {attempt + 1}/{self._max_retries}"
                        )
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
                        continue

                    # Client error - don't retry
                    error_text = await response.text()
                    logger.error(
                        f"Discord webhook error {response.status}: {error_text}"
                    )
                    return False

            except asyncio.TimeoutError:
                logger.warning(
                    f"Discord request timeout, attempt {attempt + 1}/{self._max_retries}"
                )
                last_error = asyncio.TimeoutError("Request timeout")
                await asyncio.sleep(2 ** attempt)

            except aiohttp.ClientError as e:
                logger.warning(
                    f"Discord request error: {e}, "
                    f"attempt {attempt + 1}/{self._max_retries}"
                )
                last_error = e
                await asyncio.sleep(2 ** attempt)

        if last_error:
            logger.error(f"Discord request failed after {self._max_retries} retries")
        return False

    async def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
    ) -> bool:
        """
        Send a plain text notification.

        Args:
            message: Message content
            level: Notification level

        Returns:
            True if sent successfully
        """
        emoji = LEVEL_EMOJIS.get(level, "")
        content = f"{emoji} {message}" if emoji else message

        payload = {
            "username": self._username,
            "content": content[:2000],  # Discord limit
        }

        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        return await self._send_request(payload)

    async def send_embed(self, embed: dict[str, Any]) -> bool:
        """
        Send an embed notification.

        Args:
            embed: Embed data dictionary

        Returns:
            True if sent successfully
        """
        payload = {
            "username": self._username,
            "embeds": [embed],
        }

        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        return await self._send_request(payload)

    async def send_embeds(self, embeds: list[dict[str, Any]]) -> bool:
        """
        Send multiple embeds in one message.

        Args:
            embeds: List of embed data dictionaries (max 10)

        Returns:
            True if sent successfully
        """
        payload = {
            "username": self._username,
            "embeds": embeds[:10],  # Discord limit
        }

        if self._avatar_url:
            payload["avatar_url"] = self._avatar_url

        return await self._send_request(payload)

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # =========================================================================
    # Convenience Methods
    # =========================================================================

    async def success(self, title: str, message: str) -> bool:
        """
        Send a success notification with embed.

        Args:
            title: Embed title
            message: Embed description

        Returns:
            True if sent successfully
        """
        embed = (
            DiscordEmbed()
            .set_title(f"✅ {title}")
            .set_description(message)
            .set_color(EmbedColor.SUCCESS)
            .set_timestamp()
            .build()
        )
        return await self.send_embed(embed)

    async def error(self, title: str, message: str) -> bool:
        """
        Send an error notification with embed.

        Args:
            title: Embed title
            message: Embed description

        Returns:
            True if sent successfully
        """
        embed = (
            DiscordEmbed()
            .set_title(f"❌ {title}")
            .set_description(message)
            .set_color(EmbedColor.ERROR)
            .set_timestamp()
            .build()
        )
        return await self.send_embed(embed)

    async def warning(self, title: str, message: str) -> bool:
        """
        Send a warning notification with embed.

        Args:
            title: Embed title
            message: Embed description

        Returns:
            True if sent successfully
        """
        embed = (
            DiscordEmbed()
            .set_title(f"⚠️ {title}")
            .set_description(message)
            .set_color(EmbedColor.WARNING)
            .set_timestamp()
            .build()
        )
        return await self.send_embed(embed)

    async def info(self, title: str, message: str) -> bool:
        """
        Send an info notification with embed.

        Args:
            title: Embed title
            message: Embed description

        Returns:
            True if sent successfully
        """
        embed = (
            DiscordEmbed()
            .set_title(f"ℹ️ {title}")
            .set_description(message)
            .set_color(EmbedColor.INFO)
            .set_timestamp()
            .build()
        )
        return await self.send_embed(embed)

    # =========================================================================
    # Trading-Specific Methods
    # =========================================================================

    async def order_filled(
        self,
        symbol: str,
        side: str,
        price: str,
        quantity: str,
        total: Optional[str] = None,
    ) -> bool:
        """
        Send order filled notification.

        Args:
            symbol: Trading pair
            side: BUY or SELL
            price: Order price
            quantity: Order quantity
            total: Total value (optional)

        Returns:
            True if sent successfully
        """
        embed = DiscordEmbed.order_filled(symbol, side, price, quantity, total)
        return await self.send_embed(embed.build())

    async def position_update(
        self,
        symbol: str,
        side: str,
        entry_price: str,
        mark_price: str,
        pnl: str,
        pnl_percent: str,
    ) -> bool:
        """
        Send position update notification.

        Args:
            symbol: Trading pair
            side: LONG or SHORT
            entry_price: Entry price
            mark_price: Current mark price
            pnl: Unrealized PnL
            pnl_percent: PnL percentage

        Returns:
            True if sent successfully
        """
        embed = DiscordEmbed.position_update(
            symbol, side, entry_price, mark_price, pnl, pnl_percent
        )
        return await self.send_embed(embed.build())

    async def alert(
        self,
        title: str,
        message: str,
        level: str = "info",
    ) -> bool:
        """
        Send alert notification.

        Args:
            title: Alert title
            message: Alert message
            level: Alert level

        Returns:
            True if sent successfully
        """
        embed = DiscordEmbed.alert(title, message, level)
        return await self.send_embed(embed.build())
