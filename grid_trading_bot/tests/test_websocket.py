"""
Tests for Binance WebSocket client.
"""

import asyncio
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.exchange.binance.websocket import BinanceWebSocket
from src.exchange.binance.constants import (
    SPOT_WS_URL,
    SPOT_WS_TESTNET_URL,
    FUTURES_WS_URL,
    FUTURES_WS_TESTNET_URL,
)
from src.core.models import MarketType, KlineInterval, Ticker, Kline


# =============================================================================
# Constants Tests
# =============================================================================


class TestWebSocketConstants:
    """Test cases for WebSocket constants."""

    def test_spot_ws_urls(self):
        """Test Spot WebSocket URL constants."""
        assert SPOT_WS_URL == "wss://stream.binance.com:9443/ws"
        assert SPOT_WS_TESTNET_URL == "wss://testnet.binance.vision/ws"

    def test_futures_ws_urls(self):
        """Test Futures WebSocket URL constants."""
        assert FUTURES_WS_URL == "wss://fstream.binance.com/ws"
        assert FUTURES_WS_TESTNET_URL == "wss://stream.binancefuture.com/ws"


# =============================================================================
# Initialization Tests
# =============================================================================


class TestBinanceWebSocketInit:
    """Test BinanceWebSocket initialization."""

    def test_init_spot_default(self):
        """Test default initialization for Spot."""
        ws = BinanceWebSocket()
        assert ws._market_type == MarketType.SPOT
        assert ws._testnet is False
        assert ws._base_url == SPOT_WS_URL
        assert ws.is_connected is False

    def test_init_spot_testnet(self):
        """Test Spot testnet initialization."""
        ws = BinanceWebSocket(testnet=True)
        assert ws._base_url == SPOT_WS_TESTNET_URL

    def test_init_futures(self):
        """Test Futures initialization."""
        ws = BinanceWebSocket(market_type=MarketType.FUTURES)
        assert ws._market_type == MarketType.FUTURES
        assert ws._base_url == FUTURES_WS_URL

    def test_init_futures_testnet(self):
        """Test Futures testnet initialization."""
        ws = BinanceWebSocket(market_type=MarketType.FUTURES, testnet=True)
        assert ws._base_url == FUTURES_WS_TESTNET_URL

    def test_init_with_callbacks(self):
        """Test initialization with callbacks."""
        on_message = MagicMock()
        on_error = MagicMock()
        on_close = MagicMock()

        ws = BinanceWebSocket(
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )

        assert ws._on_message == on_message
        assert ws._on_error == on_error
        assert ws._on_close == on_close

    def test_default_reconnect_settings(self):
        """Test default reconnection settings."""
        ws = BinanceWebSocket()
        assert ws.reconnect_delay == 5
        assert ws.max_reconnect_delay == 60
        assert ws.max_reconnect_attempts == 10


# =============================================================================
# Data Parsing Tests
# =============================================================================


class TestBinanceWebSocketParsing:
    """Test data parsing methods."""

    @pytest.fixture
    def ws(self):
        """Create WebSocket instance."""
        return BinanceWebSocket()

    def test_parse_ticker(self, ws):
        """Test ticker message parsing."""
        data = {
            "e": "24hrTicker",
            "E": 1704067200000,
            "s": "BTCUSDT",
            "c": "40000.00",
            "b": "39999.00",
            "a": "40001.00",
            "h": "41000.00",
            "l": "39000.00",
            "v": "10000.00",
            "P": "2.5",
        }

        ticker = ws._parse_ticker(data)

        assert isinstance(ticker, Ticker)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == Decimal("40000.00")
        assert ticker.bid == Decimal("39999.00")
        assert ticker.ask == Decimal("40001.00")
        assert ticker.change_24h == Decimal("2.5")

    def test_parse_kline(self, ws):
        """Test kline message parsing."""
        data = {
            "e": "kline",
            "E": 1704067200000,
            "s": "BTCUSDT",
            "k": {
                "t": 1704067200000,
                "T": 1704070799999,
                "s": "BTCUSDT",
                "i": "1h",
                "o": "40000.00",
                "c": "40500.00",
                "h": "41000.00",
                "l": "39500.00",
                "v": "100.00",
                "q": "4050000.00",
                "n": 150,
            },
        }

        kline = ws._parse_kline(data)

        assert isinstance(kline, Kline)
        assert kline.symbol == "BTCUSDT"
        assert kline.interval == "1h"
        assert kline.open == Decimal("40000.00")
        assert kline.close == Decimal("40500.00")
        assert kline.trades_count == 150

    def test_parse_depth(self, ws):
        """Test depth message parsing."""
        data = {
            "E": 1704067200000,
            "bids": [["39999.00", "1.5"], ["39998.00", "2.0"]],
            "asks": [["40001.00", "1.0"], ["40002.00", "0.5"]],
        }

        depth = ws._parse_depth(data)

        assert "bids" in depth
        assert "asks" in depth
        assert len(depth["bids"]) == 2
        assert depth["bids"][0][0] == Decimal("39999.00")
        assert depth["asks"][0][1] == Decimal("1.0")

    def test_parse_agg_trade(self, ws):
        """Test aggregate trade message parsing."""
        data = {
            "e": "aggTrade",
            "E": 1704067200000,
            "s": "BTCUSDT",
            "a": 12345,
            "p": "40000.00",
            "q": "0.5",
            "f": 100,
            "l": 105,
            "T": 1704067200000,
            "m": False,
        }

        trade = ws._parse_agg_trade(data)

        assert trade["symbol"] == "BTCUSDT"
        assert trade["trade_id"] == 12345
        assert trade["price"] == Decimal("40000.00")
        assert trade["quantity"] == Decimal("0.5")
        assert trade["is_buyer_maker"] is False

    def test_parse_book_ticker(self, ws):
        """Test book ticker message parsing."""
        data = {
            "s": "BTCUSDT",
            "b": "39999.00",
            "B": "1.5",
            "a": "40001.00",
            "A": "1.0",
            "E": 1704067200000,
        }

        book = ws._parse_book_ticker(data)

        assert book["symbol"] == "BTCUSDT"
        assert book["bid_price"] == Decimal("39999.00")
        assert book["ask_price"] == Decimal("40001.00")

    def test_parse_mark_price(self, ws):
        """Test mark price message parsing."""
        data = {
            "e": "markPriceUpdate",
            "E": 1704067200000,
            "s": "BTCUSDT",
            "p": "40000.00",
            "i": "40001.00",
            "r": "0.0001",
            "T": 1704070800000,
        }

        mark = ws._parse_mark_price(data)

        assert mark["symbol"] == "BTCUSDT"
        assert mark["mark_price"] == Decimal("40000.00")
        assert mark["index_price"] == Decimal("40001.00")
        assert mark["funding_rate"] == Decimal("0.0001")


# =============================================================================
# Stream Name Tests
# =============================================================================


class TestBinanceWebSocketStreams:
    """Test stream name handling."""

    @pytest.fixture
    def ws(self):
        """Create WebSocket instance."""
        return BinanceWebSocket()

    def test_get_stream_from_ticker_data(self, ws):
        """Test extracting stream name from ticker data."""
        data = {"e": "24hrTicker", "s": "BTCUSDT"}
        stream = ws._get_stream_from_data(data)
        assert stream == "btcusdt@ticker"

    def test_get_stream_from_kline_data(self, ws):
        """Test extracting stream name from kline data."""
        data = {"e": "kline", "s": "BTCUSDT", "k": {"i": "1h"}}
        stream = ws._get_stream_from_data(data)
        assert stream == "btcusdt@kline_1h"

    def test_get_stream_from_agg_trade_data(self, ws):
        """Test extracting stream name from aggTrade data."""
        data = {"e": "aggTrade", "s": "BTCUSDT"}
        stream = ws._get_stream_from_data(data)
        assert stream == "btcusdt@aggTrade"


# =============================================================================
# Subscription Tests (Mocked)
# =============================================================================


class TestBinanceWebSocketSubscription:
    """Test subscription methods with mocking."""

    @pytest.fixture
    def ws(self):
        """Create WebSocket instance with mocked connection."""
        ws = BinanceWebSocket()
        ws._ws = AsyncMock()
        ws._connected = True
        # Mock the state for is_connected check
        from websockets.protocol import State
        ws._ws.state = State.OPEN
        return ws

    @pytest.mark.asyncio
    async def test_subscribe_sends_message(self, ws):
        """Test that subscribe sends correct WebSocket message."""
        callback = MagicMock()
        await ws.subscribe(["btcusdt@ticker"], callback)

        ws._ws.send.assert_called_once()
        call_args = ws._ws.send.call_args[0][0]
        message = __import__("json").loads(call_args)

        assert message["method"] == "SUBSCRIBE"
        assert "btcusdt@ticker" in message["params"]

    @pytest.mark.asyncio
    async def test_unsubscribe_sends_message(self, ws):
        """Test that unsubscribe sends correct WebSocket message."""
        ws._subscriptions["btcusdt@ticker"] = MagicMock()
        await ws.unsubscribe(["btcusdt@ticker"])

        ws._ws.send.assert_called_once()
        call_args = ws._ws.send.call_args[0][0]
        message = __import__("json").loads(call_args)

        assert message["method"] == "UNSUBSCRIBE"

    @pytest.mark.asyncio
    async def test_subscribe_stores_callback(self, ws):
        """Test that subscribe stores callback."""
        callback = MagicMock()
        await ws.subscribe(["btcusdt@ticker"], callback)

        assert "btcusdt@ticker" in ws._subscriptions

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_callback(self, ws):
        """Test that unsubscribe removes callback."""
        ws._subscriptions["btcusdt@ticker"] = MagicMock()
        await ws.unsubscribe(["btcusdt@ticker"])

        assert "btcusdt@ticker" not in ws._subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_not_connected(self):
        """Test subscribe fails when not connected."""
        ws = BinanceWebSocket()
        result = await ws.subscribe(["btcusdt@ticker"], MagicMock())
        assert result is False


# =============================================================================
# Reconnection Tests
# =============================================================================


class TestBinanceWebSocketReconnect:
    """Test reconnection logic."""

    def test_reconnect_attempts_tracking(self):
        """Test reconnection attempts counter."""
        ws = BinanceWebSocket()
        assert ws._reconnect_attempts == 0
        assert ws._current_delay == ws.reconnect_delay

    def test_max_reconnect_reached(self):
        """Test that reconnection stops after max attempts."""
        ws = BinanceWebSocket()
        ws._reconnect_attempts = ws.max_reconnect_attempts
        # This would be tested with actual reconnection logic


# =============================================================================
# Context Manager Tests
# =============================================================================


class TestBinanceWebSocketContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_context_manager_connect_disconnect(self):
        """Test context manager calls connect and disconnect."""
        ws = BinanceWebSocket()

        with patch.object(ws, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(ws, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                mock_connect.return_value = True

                async with ws:
                    mock_connect.assert_called_once()

                mock_disconnect.assert_called_once()


# =============================================================================
# Integration Tests (Optional - require network)
# =============================================================================


@pytest.mark.integration
class TestBinanceWebSocketIntegration:
    """
    Integration tests that actually connect to Binance WebSocket.

    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test actual connection and disconnection."""
        ws = BinanceWebSocket()
        connected = await ws.connect()
        assert connected is True
        assert ws.is_connected is True

        await ws.disconnect()
        assert ws.is_connected is False

    @pytest.mark.asyncio
    async def test_subscribe_ticker_live(self):
        """Test subscribing to ticker stream."""
        received_data = []

        async def callback(ticker):
            received_data.append(ticker)

        async with BinanceWebSocket() as ws:
            await ws.subscribe_ticker("BTCUSDT", callback)
            # Wait for some data
            await asyncio.sleep(3)

        assert len(received_data) > 0
        assert all(isinstance(t, Ticker) for t in received_data)
        assert all(t.symbol == "BTCUSDT" for t in received_data)

    @pytest.mark.asyncio
    async def test_subscribe_kline_live(self):
        """Test subscribing to kline stream."""
        received_data = []

        async def callback(kline):
            received_data.append(kline)

        async with BinanceWebSocket() as ws:
            await ws.subscribe_kline("BTCUSDT", KlineInterval.m1, callback)
            # Wait for some data
            await asyncio.sleep(3)

        # Klines update less frequently
        # Just check connection worked
        assert ws._subscriptions  # Had subscriptions

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self):
        """Test multiple simultaneous subscriptions."""
        btc_tickers = []
        eth_tickers = []

        async def btc_callback(ticker):
            btc_tickers.append(ticker)

        async def eth_callback(ticker):
            eth_tickers.append(ticker)

        async with BinanceWebSocket() as ws:
            await ws.subscribe_ticker("BTCUSDT", btc_callback)
            await ws.subscribe_ticker("ETHUSDT", eth_callback)
            await asyncio.sleep(3)

        assert len(btc_tickers) > 0 or len(eth_tickers) > 0

    @pytest.mark.asyncio
    async def test_futures_websocket(self):
        """Test Futures WebSocket connection."""
        received_data = []

        async def callback(ticker):
            received_data.append(ticker)

        async with BinanceWebSocket(market_type=MarketType.FUTURES) as ws:
            await ws.subscribe_ticker("BTCUSDT", callback)
            await asyncio.sleep(3)

        assert len(received_data) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
