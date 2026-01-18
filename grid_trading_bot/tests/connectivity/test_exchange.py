"""
Exchange Connectivity Tests.

Tests REST API and WebSocket connectivity to Binance exchange.
"""

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exchange import ExchangeClient


# =============================================================================
# Mock-based Tests (Always Run)
# =============================================================================


class TestExchangeConnectivityMock:
    """Mock-based exchange connectivity tests."""

    @pytest.fixture
    def mock_spot_api(self):
        """Create mock Spot API."""
        api = AsyncMock()
        api.connect = AsyncMock()
        api.close = AsyncMock()
        api.sync_time = AsyncMock()
        api.get_price = AsyncMock(return_value=50000.0)
        api.get_ticker = AsyncMock()
        api.get_account = AsyncMock()
        return api

    @pytest.fixture
    def mock_futures_api(self):
        """Create mock Futures API."""
        api = AsyncMock()
        api.connect = AsyncMock()
        api.close = AsyncMock()
        return api

    @pytest.fixture
    def mock_ws(self):
        """Create mock WebSocket."""
        ws = AsyncMock()
        ws.connect = AsyncMock(return_value=True)
        ws.disconnect = AsyncMock()
        ws.subscribe_ticker = AsyncMock(return_value=True)
        ws.subscribe_kline = AsyncMock(return_value=True)
        ws.unsubscribe = AsyncMock(return_value=True)
        return ws

    @pytest.mark.asyncio
    async def test_client_connect(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test client connection."""
        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            result = await client.connect()

            assert result is True
            assert client.is_connected is True
            mock_spot_api.connect.assert_called_once()
            mock_futures_api.connect.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_get_price(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test getting price."""
        mock_spot_api.get_price = AsyncMock(return_value=50000.0)

        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()

            price = await client.get_price("BTCUSDT")

            assert price == 50000.0
            mock_spot_api.get_price.assert_called_once_with("BTCUSDT")

            await client.close()

    @pytest.mark.asyncio
    async def test_get_ticker(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test getting ticker."""
        mock_ticker = MagicMock()
        mock_ticker.symbol = "BTCUSDT"
        mock_ticker.price = 50000.0
        mock_spot_api.get_ticker = AsyncMock(return_value=mock_ticker)

        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()

            ticker = await client.get_ticker("BTCUSDT")

            assert ticker.symbol == "BTCUSDT"
            assert ticker.price == 50000.0

            await client.close()

    @pytest.mark.asyncio
    async def test_ws_connect(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test WebSocket connection."""
        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()

            # WebSocket should be connected during client.connect()
            mock_ws.connect.assert_called()

            await client.close()

    @pytest.mark.asyncio
    async def test_ws_subscribe_ticker(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test WebSocket ticker subscription."""
        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()

            callback = MagicMock()
            result = await client.subscribe_ticker("BTCUSDT", callback)

            assert result is True
            mock_ws.subscribe_ticker.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_ws_subscribe_kline(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test WebSocket kline subscription."""
        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()

            callback = MagicMock()
            result = await client.subscribe_kline("BTCUSDT", "1m", callback)

            assert result is True
            mock_ws.subscribe_kline.assert_called_once()

            await client.close()

    @pytest.mark.asyncio
    async def test_client_close(self, mock_spot_api, mock_futures_api, mock_ws):
        """Test client close."""
        with patch("src.exchange.client.BinanceSpotAPI", return_value=mock_spot_api), \
             patch("src.exchange.client.BinanceFuturesAPI", return_value=mock_futures_api), \
             patch("src.exchange.client.BinanceWebSocket", return_value=mock_ws):

            client = ExchangeClient(testnet=True)
            await client.connect()
            await client.close()

            assert client.is_connected is False
            mock_ws.disconnect.assert_called()
            mock_spot_api.close.assert_called_once()
            mock_futures_api.close.assert_called_once()


# =============================================================================
# Live Tests (Skip if no credentials)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("BINANCE_TESTNET_API_KEY"),
    reason="No Binance testnet credentials"
)
class TestExchangeConnectivityLive:
    """Live exchange connectivity tests (requires testnet credentials)."""

    @pytest.fixture
    def api_key(self):
        """Get API key from environment."""
        return os.getenv("BINANCE_TESTNET_API_KEY", "")

    @pytest.fixture
    def api_secret(self):
        """Get API secret from environment."""
        return os.getenv("BINANCE_TESTNET_API_SECRET", "")

    @pytest.mark.asyncio
    async def test_rest_ping(self, api_key, api_secret):
        """Test REST API ping."""
        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            # If we got here without error, ping succeeded
            assert client.is_connected is True

    @pytest.mark.asyncio
    async def test_rest_get_price(self, api_key, api_secret):
        """Test getting price from REST API."""
        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            price = await client.get_price("BTCUSDT")
            assert price > 0

    @pytest.mark.asyncio
    async def test_rest_get_ticker(self, api_key, api_secret):
        """Test getting ticker from REST API."""
        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            ticker = await client.get_ticker("BTCUSDT")
            assert ticker is not None
            assert ticker.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_rest_account(self, api_key, api_secret):
        """Test getting account info."""
        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            account = await client.get_account()
            assert account is not None

    @pytest.mark.asyncio
    async def test_ws_receive_ticker(self, api_key, api_secret):
        """Test WebSocket receives ticker data."""
        received = []

        async def on_ticker(ticker):
            received.append(ticker)

        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            await client.subscribe_ticker("BTCUSDT", on_ticker)
            await asyncio.sleep(5)  # Wait for data

            assert len(received) > 0, "Should receive ticker data"

    @pytest.mark.asyncio
    async def test_ws_receive_kline(self, api_key, api_secret):
        """Test WebSocket receives kline data."""
        received = []

        async def on_kline(kline):
            received.append(kline)

        async with ExchangeClient(api_key, api_secret, testnet=True) as client:
            await client.subscribe_kline("BTCUSDT", "1m", on_kline)
            await asyncio.sleep(5)  # Wait for data

            assert len(received) > 0, "Should receive kline data"
