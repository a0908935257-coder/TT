"""
Tests for exchange module: Binance API client.
"""

import hashlib
import hmac
import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from exchange.binance.auth import BinanceAuth
from exchange.binance.constants import (
    BINANCE_ERROR_CODES,
    PRIVATE_ENDPOINTS,
    PUBLIC_ENDPOINTS,
    SPOT_REST_URL,
    SPOT_TESTNET_URL,
    Endpoint,
    PrivateEndpoints,
    PublicEndpoints,
)
from exchange.binance.spot_api import BinanceSpotAPI
from core.exceptions import (
    AuthenticationError,
    ExchangeError,
    InsufficientBalanceError,
    OrderError,
    RateLimitError,
)
from core.models import KlineInterval


# =============================================================================
# Constants Tests
# =============================================================================


class TestConstants:
    """Test cases for constants module."""

    def test_base_urls(self):
        """Test base URL constants."""
        assert SPOT_REST_URL == "https://api.binance.com"
        assert SPOT_TESTNET_URL == "https://testnet.binance.vision"

    def test_endpoint_dataclass(self):
        """Test Endpoint dataclass."""
        ep = Endpoint("/api/v3/ping", "GET")
        assert ep.path == "/api/v3/ping"
        assert ep.method == "GET"

    def test_public_endpoints_enum(self):
        """Test PublicEndpoints enum."""
        assert PublicEndpoints.PING.value.path == "/api/v3/ping"
        assert PublicEndpoints.KLINES.value.path == "/api/v3/klines"

    def test_private_endpoints_enum(self):
        """Test PrivateEndpoints enum."""
        assert PrivateEndpoints.ACCOUNT.value.path == "/api/v3/account"
        assert PrivateEndpoints.ORDER.value.path == "/api/v3/order"

    def test_public_endpoints_dict(self):
        """Test PUBLIC_ENDPOINTS dictionary."""
        assert "PING" in PUBLIC_ENDPOINTS
        assert PUBLIC_ENDPOINTS["PING"]["path"] == "/api/v3/ping"
        assert PUBLIC_ENDPOINTS["KLINES"]["method"] == "GET"

    def test_private_endpoints_dict(self):
        """Test PRIVATE_ENDPOINTS dictionary."""
        assert "ACCOUNT" in PRIVATE_ENDPOINTS
        assert PRIVATE_ENDPOINTS["ORDER"]["path"] == "/api/v3/order"
        assert PRIVATE_ENDPOINTS["ORDER"]["method"] == "POST"

    def test_error_codes(self):
        """Test BINANCE_ERROR_CODES mapping."""
        assert BINANCE_ERROR_CODES[-1002] == "UNAUTHORIZED"
        assert BINANCE_ERROR_CODES[-1003] == "TOO_MANY_REQUESTS"
        assert BINANCE_ERROR_CODES[-2010] == "INSUFFICIENT_BALANCE"


# =============================================================================
# Auth Tests
# =============================================================================


class TestBinanceAuth:
    """Test cases for BinanceAuth class."""

    @pytest.fixture
    def auth(self):
        """Create auth instance with test credentials."""
        return BinanceAuth("test_api_key", "test_api_secret")

    def test_init(self, auth):
        """Test auth initialization."""
        assert auth.api_key == "test_api_key"
        assert auth.api_secret == "test_api_secret"

    def test_get_headers(self, auth):
        """Test get_headers returns correct header."""
        headers = auth.get_headers()
        assert headers == {"X-MBX-APIKEY": "test_api_key"}

    def test_sign_params_adds_timestamp(self, auth):
        """Test that sign_params adds timestamp."""
        params = {"symbol": "BTCUSDT"}
        signed = auth.sign_params(params)

        assert "timestamp" in signed
        assert isinstance(signed["timestamp"], int)
        # Timestamp should be recent (within last minute)
        import time
        assert abs(signed["timestamp"] - int(time.time() * 1000)) < 60000

    def test_sign_params_adds_signature(self, auth):
        """Test that sign_params adds signature."""
        params = {"symbol": "BTCUSDT"}
        signed = auth.sign_params(params)

        assert "signature" in signed
        # Signature should be 64-char hex string (SHA256)
        assert len(signed["signature"]) == 64
        assert all(c in "0123456789abcdef" for c in signed["signature"])

    def test_sign_params_signature_correctness(self, auth):
        """Test signature is calculated correctly."""
        # Use fixed timestamp for reproducible test
        params = {"symbol": "BTCUSDT", "side": "BUY"}

        # Patch time.time() directly since auth.py uses time.time()
        with patch("time.time", return_value=1234567890.0):
            signed = auth.sign_params(params)

        # Manually calculate expected signature (includes recvWindow, preserves dict order)
        query_string = "symbol=BTCUSDT&side=BUY&timestamp=1234567890000&recvWindow=60000"
        expected_sig = hmac.new(
            b"test_api_secret",
            query_string.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert signed["signature"] == expected_sig

    def test_sign_params_does_not_modify_original(self, auth):
        """Test that original params dict is not modified."""
        original = {"symbol": "BTCUSDT"}
        auth.sign_params(original)

        assert "timestamp" not in original
        assert "signature" not in original

    def test_sign_params_empty_dict(self, auth):
        """Test signing empty params."""
        signed = auth.sign_params({})

        assert "timestamp" in signed
        assert "signature" in signed

    def test_sign_params_none(self, auth):
        """Test signing None params."""
        signed = auth.sign_params(None)

        assert "timestamp" in signed
        assert "signature" in signed

    def test_sign_query_string(self, auth):
        """Test sign_query_string method."""
        query = "symbol=BTCUSDT&timestamp=1234567890000"
        signature = auth.sign_query_string(query)

        expected = hmac.new(
            b"test_api_secret",
            query.encode(),
            hashlib.sha256,
        ).hexdigest()

        assert signature == expected


# =============================================================================
# SpotAPI Tests - Unit Tests with Mocking
# =============================================================================


class TestBinanceSpotAPIInit:
    """Test BinanceSpotAPI initialization."""

    def test_init_default(self):
        """Test default initialization."""
        api = BinanceSpotAPI()
        assert api._base_url == SPOT_REST_URL
        assert api._auth is None
        assert api._testnet is False

    def test_init_testnet(self):
        """Test testnet initialization."""
        api = BinanceSpotAPI(testnet=True)
        assert api._base_url == SPOT_TESTNET_URL
        assert api._testnet is True

    def test_init_with_auth(self):
        """Test initialization with API credentials."""
        api = BinanceSpotAPI(api_key="key", api_secret="secret")
        assert api._auth is not None
        assert api._auth.api_key == "key"

    def test_init_partial_auth_ignored(self):
        """Test that partial auth (only key or secret) creates no auth."""
        api1 = BinanceSpotAPI(api_key="key")
        api2 = BinanceSpotAPI(api_secret="secret")

        assert api1._auth is None
        assert api2._auth is None


class TestBinanceSpotAPIErrorHandling:
    """Test error handling in BinanceSpotAPI."""

    @pytest.fixture
    def api(self):
        """Create API instance."""
        return BinanceSpotAPI()

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

    def test_raise_exception_order_error(self, api):
        """Test order error handling."""
        with pytest.raises(OrderError):
            api._raise_exception(-2011, "Order not found")

    def test_raise_exception_generic(self, api):
        """Test generic error handling."""
        with pytest.raises(ExchangeError):
            api._raise_exception(-1000, "Unknown error")


class TestBinanceSpotAPIMocked:
    """Test BinanceSpotAPI methods with mocked responses."""

    @pytest.fixture
    def api(self):
        """Create API instance with mocked session."""
        api = BinanceSpotAPI()
        return api

    @pytest.mark.asyncio
    async def test_ping_success(self, api):
        """Test ping with mocked response."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {}

            result = await api.ping()

            assert result is True
            mock_request.assert_called_once_with("GET", "/api/v3/ping")

    @pytest.mark.asyncio
    async def test_get_server_time(self, api):
        """Test get_server_time with mocked response."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"serverTime": 1704067200000}

            result = await api.get_server_time()

            assert isinstance(result, datetime)
            assert result.year == 2024
            assert result.month == 1
            assert result.day == 1

    @pytest.mark.asyncio
    async def test_get_klines(self, api):
        """Test get_klines with mocked response."""
        mock_kline_data = [
            [
                1704067200000,  # open_time
                "40000.00",     # open
                "41000.00",     # high
                "39500.00",     # low
                "40500.00",     # close
                "100.00",       # volume
                1704070799999,  # close_time
                "4050000.00",   # quote_volume
                150,            # trades
            ]
        ]

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_kline_data

            result = await api.get_klines("BTCUSDT", KlineInterval.h1, limit=1)

            assert len(result) == 1
            assert result[0].symbol == "BTCUSDT"
            assert result[0].open == Decimal("40000.00")
            assert result[0].close == Decimal("40500.00")

    @pytest.mark.asyncio
    async def test_get_ticker(self, api):
        """Test get_ticker with mocked response."""
        mock_ticker = {
            "symbol": "BTCUSDT",
            "lastPrice": "40000.00",
            "bidPrice": "39999.00",
            "askPrice": "40001.00",
            "highPrice": "41000.00",
            "lowPrice": "39000.00",
            "volume": "10000.00",
            "priceChangePercent": "2.5",
            "closeTime": 1704067200000,
        }

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_ticker

            result = await api.get_ticker("BTCUSDT")

            assert result.symbol == "BTCUSDT"
            assert result.price == Decimal("40000.00")

    @pytest.mark.asyncio
    async def test_get_price(self, api):
        """Test get_price with mocked response."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = {"symbol": "BTCUSDT", "price": "40000.50"}

            result = await api.get_price("BTCUSDT")

            assert result == Decimal("40000.50")

    @pytest.mark.asyncio
    async def test_get_orderbook(self, api):
        """Test get_orderbook with mocked response."""
        mock_depth = {
            "lastUpdateId": 12345,
            "bids": [["39999.00", "1.5"], ["39998.00", "2.0"]],
            "asks": [["40001.00", "1.0"], ["40002.00", "0.5"]],
        }

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.return_value = mock_depth

            result = await api.get_orderbook("BTCUSDT")

            assert len(result["bids"]) == 2
            assert result["bids"][0][0] == Decimal("39999.00")
            assert result["asks"][0][1] == Decimal("1.0")

    @pytest.mark.asyncio
    async def test_get_account_requires_auth(self, api):
        """Test get_account raises error without auth."""
        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = AuthenticationError("API key required")

            with pytest.raises(AuthenticationError):
                await api.get_account()

    @pytest.mark.asyncio
    async def test_create_order_requires_auth(self):
        """Test create_order raises error without auth."""
        api = BinanceSpotAPI()  # No auth

        with patch.object(api, "_request", new_callable=AsyncMock) as mock_request:
            mock_request.side_effect = AuthenticationError("API key required")

            with pytest.raises(AuthenticationError):
                await api.create_order(
                    "BTCUSDT", "BUY", "MARKET", Decimal("0.001")
                )


class TestBinanceSpotAPILifecycle:
    """Test BinanceSpotAPI lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_creates_session(self):
        """Test connect creates aiohttp session."""
        api = BinanceSpotAPI()

        assert api._session is None
        await api.connect()
        assert api._session is not None
        await api.close()

    @pytest.mark.asyncio
    async def test_close_closes_session(self):
        """Test close properly closes session."""
        api = BinanceSpotAPI()

        await api.connect()
        session = api._session
        await api.close()

        assert api._session is None
        assert session.closed

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager."""
        async with BinanceSpotAPI() as api:
            assert api._session is not None
        assert api._session is None

    @pytest.mark.asyncio
    async def test_double_connect_safe(self):
        """Test calling connect twice is safe."""
        api = BinanceSpotAPI()

        await api.connect()
        session1 = api._session
        await api.connect()
        session2 = api._session

        # Should be same session
        assert session1 is session2
        await api.close()


# =============================================================================
# Integration Tests (Optional - require network)
# =============================================================================


@pytest.mark.integration
class TestBinanceSpotAPIIntegration:
    """
    Integration tests that actually call Binance API.

    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_ping_live(self):
        """Test ping against live API."""
        async with BinanceSpotAPI() as api:
            result = await api.ping()
            assert result is True

    @pytest.mark.asyncio
    async def test_get_server_time_live(self):
        """Test server time against live API."""
        async with BinanceSpotAPI() as api:
            result = await api.get_server_time()
            assert isinstance(result, datetime)
            # Should be within 30 seconds of current time
            import time
            diff = abs(result.timestamp() - time.time())
            assert diff < 30

    @pytest.mark.asyncio
    async def test_get_klines_live(self):
        """Test klines against live API."""
        async with BinanceSpotAPI() as api:
            klines = await api.get_klines("BTCUSDT", KlineInterval.h1, limit=5)
            assert len(klines) == 5
            assert all(k.symbol == "BTCUSDT" for k in klines)

    @pytest.mark.asyncio
    async def test_get_ticker_live(self):
        """Test ticker against live API."""
        async with BinanceSpotAPI() as api:
            ticker = await api.get_ticker("BTCUSDT")
            assert ticker.symbol == "BTCUSDT"
            assert ticker.price > 0

    @pytest.mark.asyncio
    async def test_get_price_live(self):
        """Test price against live API."""
        async with BinanceSpotAPI() as api:
            price = await api.get_price("BTCUSDT")
            assert price > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
