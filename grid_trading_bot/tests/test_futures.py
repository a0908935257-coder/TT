"""
Tests for Binance Futures API client.
"""

import sys
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.exchange.binance.futures_api import BinanceFuturesAPI
from src.exchange.binance.constants import (
    FUTURES_REST_URL,
    FUTURES_TESTNET_URL,
    FUTURES_PUBLIC_ENDPOINTS,
    FUTURES_PRIVATE_ENDPOINTS,
)
from src.core.exceptions import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    OrderError,
    RateLimitError,
)
from src.core.models import KlineInterval, PositionSide


# =============================================================================
# Constants Tests
# =============================================================================


class TestFuturesConstants:
    """Test cases for futures constants."""

    def test_futures_urls(self):
        """Test futures URL constants."""
        assert FUTURES_REST_URL == "https://fapi.binance.com"
        assert FUTURES_TESTNET_URL == "https://testnet.binancefuture.com"

    def test_futures_public_endpoints(self):
        """Test futures public endpoints."""
        assert "PING" in FUTURES_PUBLIC_ENDPOINTS
        assert FUTURES_PUBLIC_ENDPOINTS["PING"]["path"] == "/fapi/v1/ping"
        assert FUTURES_PUBLIC_ENDPOINTS["MARK_PRICE"]["path"] == "/fapi/v1/premiumIndex"
        assert FUTURES_PUBLIC_ENDPOINTS["FUNDING_RATE"]["path"] == "/fapi/v1/fundingRate"

    def test_futures_private_endpoints(self):
        """Test futures private endpoints."""
        assert "ACCOUNT" in FUTURES_PRIVATE_ENDPOINTS
        assert FUTURES_PRIVATE_ENDPOINTS["ACCOUNT"]["path"] == "/fapi/v2/account"
        assert FUTURES_PRIVATE_ENDPOINTS["LEVERAGE"]["path"] == "/fapi/v1/leverage"
        assert FUTURES_PRIVATE_ENDPOINTS["POSITION"]["path"] == "/fapi/v2/positionRisk"


# =============================================================================
# BinanceFuturesAPI Init Tests
# =============================================================================


class TestBinanceFuturesAPIInit:
    """Test BinanceFuturesAPI initialization."""

    def test_init_default(self):
        """Test default initialization."""
        api = BinanceFuturesAPI()
        assert api._base_url == FUTURES_REST_URL
        assert api._auth is None
        assert api._testnet is False

    def test_init_testnet(self):
        """Test testnet initialization."""
        api = BinanceFuturesAPI(testnet=True)
        assert api._base_url == FUTURES_TESTNET_URL
        assert api._testnet is True

    def test_init_with_auth(self):
        """Test initialization with API credentials."""
        api = BinanceFuturesAPI(api_key="key", api_secret="secret")
        assert api._auth is not None
        assert api._auth.api_key == "key"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestBinanceFuturesAPIErrorHandling:
    """Test error handling in BinanceFuturesAPI."""

    @pytest.fixture
    def api(self):
        """Create API instance."""
        return BinanceFuturesAPI()

    def test_raise_exception_auth_error(self, api):
        """Test authentication error handling."""
        with pytest.raises(AuthenticationError):
            api._raise_exception(-1002, "Unauthorized")

    def test_raise_exception_rate_limit(self, api):
        """Test rate limit error handling."""
        with pytest.raises(RateLimitError):
            api._raise_exception(-1003, "Too many requests")

    def test_raise_exception_insufficient_balance(self, api):
        """Test insufficient balance error handling."""
        with pytest.raises(InsufficientBalanceError):
            api._raise_exception(-2010, "Insufficient balance")

    def test_raise_exception_margin_error(self, api):
        """Test margin insufficient error handling."""
        with pytest.raises(InsufficientBalanceError):
            api._raise_exception(-2019, "Margin insufficient")

    def test_raise_exception_order_error(self, api):
        """Test order error handling."""
        with pytest.raises(OrderError):
            api._raise_exception(-2021, "Order would trigger liquidation")


# =============================================================================
# Mocked API Tests
# =============================================================================


class TestBinanceFuturesAPIMocked:
    """Test BinanceFuturesAPI methods with mocked responses."""

    @pytest.fixture
    def api(self):
        """Create API instance."""
        return BinanceFuturesAPI()

    @pytest.mark.asyncio
    async def test_ping_success(self, api):
        """Test ping with mocked response."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}

            result = await api.ping()

            assert result is True
            mock_request.assert_called_once_with("GET", "/fapi/v1/ping")

    @pytest.mark.asyncio
    async def test_get_server_time(self, api):
        """Test get_server_time with mocked response."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"serverTime": 1704067200000}

            result = await api.get_server_time()

            assert isinstance(result, datetime)
            assert result.year == 2024

    @pytest.mark.asyncio
    async def test_get_klines(self, api):
        """Test get_klines with mocked response."""
        mock_kline_data = [
            [
                1704067200000,
                "40000.00",
                "41000.00",
                "39500.00",
                "40500.00",
                "100.00",
                1704070799999,
                "4050000.00",
                150,
            ]
        ]

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_kline_data

            result = await api.get_klines("BTCUSDT", KlineInterval.h1, limit=1)

            assert len(result) == 1
            assert result[0].symbol == "BTCUSDT"
            assert result[0].open == Decimal("40000.00")

    @pytest.mark.asyncio
    async def test_get_mark_price(self, api):
        """Test get_mark_price with mocked response."""
        mock_data = {
            "symbol": "BTCUSDT",
            "markPrice": "40000.00",
            "indexPrice": "40001.00",
            "estimatedSettlePrice": "40000.50",
            "lastFundingRate": "0.0001",
            "nextFundingTime": 1704067200000,
            "interestRate": "0.0003",
            "time": 1704067200000,
        }

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            result = await api.get_mark_price("BTCUSDT")

            assert result["symbol"] == "BTCUSDT"
            assert result["mark_price"] == Decimal("40000.00")
            assert result["last_funding_rate"] == Decimal("0.0001")

    @pytest.mark.asyncio
    async def test_get_funding_rate(self, api):
        """Test get_funding_rate with mocked response."""
        mock_data = [
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingTime": 1704067200000,
                "markPrice": "40000.00",
            }
        ]

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            result = await api.get_funding_rate("BTCUSDT", limit=1)

            assert len(result) == 1
            assert result[0]["funding_rate"] == Decimal("0.0001")

    @pytest.mark.asyncio
    async def test_set_leverage(self, api):
        """Test set_leverage with mocked response."""
        api._auth = AsyncMock()
        api._auth.sign_params = lambda p: {**p, "timestamp": 123, "signature": "sig"}
        api._auth.get_headers = lambda: {"X-MBX-APIKEY": "key"}

        mock_data = {
            "symbol": "BTCUSDT",
            "leverage": 10,
            "maxNotionalValue": "1000000",
        }

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            result = await api.set_leverage("BTCUSDT", 10)

            assert result["symbol"] == "BTCUSDT"
            assert result["leverage"] == 10
            assert result["max_notional_value"] == Decimal("1000000")

    @pytest.mark.asyncio
    async def test_get_position_mode(self, api):
        """Test get_position_mode with mocked response."""
        api._auth = AsyncMock()

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"dualSidePosition": True}

            result = await api.get_position_mode()

            assert result["dual_side_position"] is True

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, api):
        """Test get_positions with no positions."""
        api._auth = AsyncMock()

        mock_data = [
            {
                "symbol": "BTCUSDT",
                "positionAmt": "0",
                "entryPrice": "0",
                "markPrice": "40000",
                "unRealizedProfit": "0",
                "liquidationPrice": "0",
                "leverage": "10",
                "maxNotionalValue": "1000000",
                "marginType": "isolated",
                "isolatedMargin": "0",
                "positionSide": "BOTH",
                "updateTime": 0,
            }
        ]

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_data

            result = await api.get_positions()

            assert len(result) == 0  # Zero position filtered out


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestBinanceFuturesAPILifecycle:
    """Test BinanceFuturesAPI lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_creates_session(self):
        """Test connect creates aiohttp session."""
        api = BinanceFuturesAPI()

        assert api._session is None
        await api.connect()
        assert api._session is not None
        await api.close()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with BinanceFuturesAPI() as api:
            assert api._session is not None
        assert api._session is None


# =============================================================================
# Integration Tests (Optional - require network)
# =============================================================================


@pytest.mark.integration
class TestBinanceFuturesAPIIntegration:
    """
    Integration tests that actually call Binance Futures API.

    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_ping_live(self):
        """Test ping against live API."""
        async with BinanceFuturesAPI() as api:
            result = await api.ping()
            assert result is True

    @pytest.mark.asyncio
    async def test_get_server_time_live(self):
        """Test server time against live API."""
        async with BinanceFuturesAPI() as api:
            result = await api.get_server_time()
            assert isinstance(result, datetime)

    @pytest.mark.asyncio
    async def test_get_klines_live(self):
        """Test klines against live API."""
        async with BinanceFuturesAPI() as api:
            klines = await api.get_klines("BTCUSDT", KlineInterval.h1, limit=5)
            assert len(klines) == 5
            assert all(k.symbol == "BTCUSDT" for k in klines)

    @pytest.mark.asyncio
    async def test_get_mark_price_live(self):
        """Test mark price against live API."""
        async with BinanceFuturesAPI() as api:
            result = await api.get_mark_price("BTCUSDT")
            assert result["symbol"] == "BTCUSDT"
            assert result["mark_price"] > 0

    @pytest.mark.asyncio
    async def test_get_funding_rate_live(self):
        """Test funding rate against live API."""
        async with BinanceFuturesAPI() as api:
            result = await api.get_funding_rate("BTCUSDT", limit=5)
            assert len(result) > 0
            assert "funding_rate" in result[0]

    @pytest.mark.asyncio
    async def test_get_ticker_live(self):
        """Test ticker against live API."""
        async with BinanceFuturesAPI() as api:
            ticker = await api.get_ticker("BTCUSDT")
            assert ticker.symbol == "BTCUSDT"
            assert ticker.price > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
