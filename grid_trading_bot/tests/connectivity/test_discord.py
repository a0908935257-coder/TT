"""
Discord Connectivity Tests.

Tests Discord webhook validation and message sending.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.notification import DiscordNotifier, NotificationLevel


# =============================================================================
# Mock-based Tests (Always Run)
# =============================================================================


class TestDiscordConnectivityMock:
    """Mock-based Discord connectivity tests."""

    @pytest.fixture
    def mock_session(self):
        """Create mock aiohttp session."""
        session = AsyncMock()

        mock_response = AsyncMock()
        mock_response.status = 204
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=None)

        session.post = MagicMock(return_value=mock_response)
        session.get = MagicMock(return_value=mock_response)
        session.closed = False
        session.close = AsyncMock()

        return session

    @pytest.mark.asyncio
    async def test_send_message(self, mock_session):
        """Test sending a message."""
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
        )
        notifier._session = mock_session

        result = await notifier.send("Test message", NotificationLevel.INFO)

        assert result is True
        mock_session.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_embed(self, mock_session):
        """Test sending an embed."""
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
        )
        notifier._session = mock_session

        # Use the success() convenience method which sends an embed
        result = await notifier.success("Test Title", "Test description")

        assert result is True

    @pytest.mark.asyncio
    async def test_rate_limiting(self, mock_session):
        """Test rate limiting behavior."""
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
            rate_limit=5,
        )
        notifier._session = mock_session

        # Send multiple messages
        for i in range(5):
            await notifier.send(f"Message {i}", NotificationLevel.INFO)

        # All should succeed
        assert mock_session.post.call_count == 5

    @pytest.mark.asyncio
    async def test_retry_on_failure(self, mock_session):
        """Test retry logic on failure."""
        # First call fails, second succeeds
        mock_response_fail = AsyncMock()
        mock_response_fail.status = 500
        mock_response_fail.__aenter__ = AsyncMock(return_value=mock_response_fail)
        mock_response_fail.__aexit__ = AsyncMock(return_value=None)

        mock_response_success = AsyncMock()
        mock_response_success.status = 204
        mock_response_success.__aenter__ = AsyncMock(return_value=mock_response_success)
        mock_response_success.__aexit__ = AsyncMock(return_value=None)

        mock_session.post = MagicMock(
            side_effect=[mock_response_fail, mock_response_success]
        )

        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
            max_retries=3,
        )
        notifier._session = mock_session

        result = await notifier.send("Test", NotificationLevel.INFO)

        # Should retry and succeed
        assert mock_session.post.call_count == 2

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_session):
        """Test async context manager."""
        with patch("aiohttp.ClientSession", return_value=mock_session):
            async with DiscordNotifier(
                webhook_url="https://discord.com/api/webhooks/test/test",
            ) as notifier:
                result = await notifier.send("Test", NotificationLevel.INFO)
                assert result is True

    @pytest.mark.asyncio
    async def test_notification_levels(self, mock_session):
        """Test different notification levels."""
        notifier = DiscordNotifier(
            webhook_url="https://discord.com/api/webhooks/test/test",
        )
        notifier._session = mock_session

        levels = [
            NotificationLevel.DEBUG,
            NotificationLevel.INFO,
            NotificationLevel.SUCCESS,
            NotificationLevel.WARNING,
            NotificationLevel.ERROR,
            NotificationLevel.CRITICAL,
        ]

        for level in levels:
            result = await notifier.send(f"Test {level.value}", level)
            assert result is True


# =============================================================================
# Live Tests (Skip if no webhook)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("DISCORD_WEBHOOK_URL"),
    reason="No Discord webhook configured"
)
class TestDiscordConnectivityLive:
    """Live Discord connectivity tests."""

    @pytest.fixture
    def webhook_url(self):
        """Get webhook URL from environment."""
        return os.getenv("DISCORD_WEBHOOK_URL", "")

    @pytest.mark.asyncio
    async def test_send_message(self, webhook_url):
        """Test live message sending."""
        async with DiscordNotifier(webhook_url=webhook_url) as notifier:
            result = await notifier.send(
                "Test connectivity message (please ignore)",
                NotificationLevel.INFO,
            )
            assert result is True

    @pytest.mark.asyncio
    async def test_send_embed(self, webhook_url):
        """Test live embed sending."""
        async with DiscordNotifier(webhook_url=webhook_url) as notifier:
            result = await notifier.success(
                "Connectivity Test",
                "This is a test embed (please ignore)",
            )
            assert result is True
