"""
Tests for unified ExchangeClient.
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

from exchange.client import ExchangeClient
from core.models import (
    AccountInfo,
    Balance,
    Kline,
    KlineInterval,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    SymbolInfo,
    Ticker,
)


# =============================================================================
# Test Fixtures - Complete Model Instances
# =============================================================================


def make_ticker(symbol: str = "BTCUSDT", price: Decimal = Decimal("45000")) -> Ticker:
    """Create a complete Ticker instance for testing."""
    return Ticker(
        symbol=symbol,
        price=price,
        bid=price - Decimal("1"),
        ask=price + Decimal("1"),
        high_24h=price + Decimal("1000"),
        low_24h=price - Decimal("1000"),
        volume_24h=Decimal("10000"),
        change_24h=Decimal("2.5"),
        timestamp=datetime.now(timezone.utc),
    )


def make_kline(symbol: str = "BTCUSDT", interval: KlineInterval = KlineInterval.h1) -> Kline:
    """Create a complete Kline instance for testing."""
    return Kline(
        symbol=symbol,
        interval=interval,
        open_time=datetime.now(timezone.utc),
        open=Decimal("44000"),
        high=Decimal("46000"),
        low=Decimal("43000"),
        close=Decimal("45000"),
        volume=Decimal("1000"),
        close_time=datetime.now(timezone.utc),
        quote_volume=Decimal("45000000"),
        trades_count=5000,
    )


def make_order(
    order_id: str = "12345",
    symbol: str = "BTCUSDT",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    status: OrderStatus = OrderStatus.NEW,
) -> Order:
    """Create a complete Order instance for testing."""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        status=status,
        price=Decimal("45000"),
        quantity=Decimal("0.001"),
        filled_qty=Decimal("0"),
        created_at=datetime.now(timezone.utc),
    )


def make_position(symbol: str = "BTCUSDT") -> Position:
    """Create a complete Position instance for testing."""
    return Position(
        symbol=symbol,
        side=PositionSide.LONG,
        quantity=Decimal("0.1"),
        entry_price=Decimal("44000"),
        mark_price=Decimal("45000"),
        leverage=10,
        margin=Decimal("440"),
        unrealized_pnl=Decimal("100"),
        updated_at=datetime.now(timezone.utc),
    )


def make_balance(asset: str = "USDT") -> Balance:
    """Create a complete Balance instance for testing."""
    return Balance(asset=asset, free=Decimal("1000"), locked=Decimal("100"))


def make_account_info(market: MarketType = MarketType.SPOT) -> AccountInfo:
    """Create a complete AccountInfo instance for testing."""
    return AccountInfo(
        market_type=market,
        balances=[make_balance("USDT"), make_balance("BTC")],
        positions=[],
        updated_at=datetime.now(timezone.utc),
    )


def make_symbol_info(symbol: str = "BTCUSDT") -> SymbolInfo:
    """Create a complete SymbolInfo instance for testing."""
    return SymbolInfo(
        symbol=symbol,
        base_asset="BTC",
        quote_asset="USDT",
        price_precision=2,
        quantity_precision=5,
        min_quantity=Decimal("0.00001"),
        tick_size=Decimal("0.01"),
        step_size=Decimal("0.00001"),
        min_notional=Decimal("10"),
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestExchangeClientInit:
    """Test ExchangeClient initialization."""

    def test_init_default(self):
        """Test default initialization."""
        client = ExchangeClient()
        assert client._api_key == ""
        assert client._api_secret == ""
        assert client._testnet is False
        assert client.is_connected is False

    def test_init_with_credentials(self):
        """Test initialization with API credentials."""
        client = ExchangeClient(
            api_key="test_key",
            api_secret="test_secret",
            testnet=True,
        )
        assert client._api_key == "test_key"
        assert client._api_secret == "test_secret"
        assert client._testnet is True

    def test_spot_api_initialized(self):
        """Test Spot API is initialized."""
        client = ExchangeClient()
        assert client.spot is not None
        assert client._spot is not None

    def test_futures_api_initialized(self):
        """Test Futures API is initialized."""
        client = ExchangeClient()
        assert client.futures is not None
        assert client._futures is not None

    def test_ws_is_none_before_connect(self):
        """Test WebSocket is None before connect."""
        client = ExchangeClient()
        assert client.ws is None
        assert client._spot_ws is None
        assert client._futures_ws is None


# =============================================================================
# Connection Tests
# =============================================================================


class TestExchangeClientConnection:
    """Test connection management."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient instance."""
        return ExchangeClient()

    @pytest.mark.asyncio
    async def test_connect_success(self, client):
        """Test successful connection."""
        with patch.object(client._spot, "connect", new_callable=AsyncMock) as mock_spot:
            with patch.object(client._spot, "sync_time", new_callable=AsyncMock) as mock_sync:
                with patch.object(client._futures, "connect", new_callable=AsyncMock) as mock_futures:
                    with patch("exchange.client.BinanceWebSocket") as mock_ws_class:
                        mock_ws_instance = MagicMock()
                        mock_ws_instance.connect = AsyncMock(return_value=True)
                        mock_ws_class.return_value = mock_ws_instance

                        result = await client.connect()

                        assert result is True
                        assert client.is_connected is True
                        mock_spot.assert_called_once()
                        mock_futures.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, client):
        """Test closing connections."""
        # Setup mocks
        client._spot = MagicMock()
        client._spot.close = AsyncMock()
        client._futures = MagicMock()
        client._futures.close = AsyncMock()
        client._spot_ws = MagicMock()
        client._spot_ws.disconnect = AsyncMock()
        client._futures_ws = MagicMock()
        client._futures_ws.disconnect = AsyncMock()
        client._connected = True

        await client.close()

        assert client.is_connected is False
        client._spot.close.assert_called_once()
        client._futures.close.assert_called_once()
        client._spot_ws.disconnect.assert_called_once()
        client._futures_ws.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager(self, client):
        """Test async context manager."""
        with patch.object(client, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(client, "close", new_callable=AsyncMock) as mock_close:
                mock_connect.return_value = True

                async with client as ctx:
                    assert ctx is client
                    mock_connect.assert_called_once()

                mock_close.assert_called_once()


# =============================================================================
# Market Data Tests
# =============================================================================


class TestExchangeClientMarketData:
    """Test market data methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient with mocked APIs."""
        client = ExchangeClient()
        client._spot = MagicMock()
        client._futures = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_get_price_spot(self, client):
        """Test get_price for spot market."""
        client._spot.get_price = AsyncMock(return_value=Decimal("45000.00"))

        price = await client.get_price("BTCUSDT", MarketType.SPOT)

        assert price == Decimal("45000.00")
        client._spot.get_price.assert_called_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_price_futures(self, client):
        """Test get_price for futures market."""
        client._futures.get_price = AsyncMock(return_value=Decimal("45100.00"))

        price = await client.get_price("BTCUSDT", MarketType.FUTURES)

        assert price == Decimal("45100.00")
        client._futures.get_price.assert_called_once_with("BTCUSDT")

    @pytest.mark.asyncio
    async def test_get_price_default_spot(self, client):
        """Test get_price defaults to spot market."""
        client._spot.get_price = AsyncMock(return_value=Decimal("45000.00"))

        price = await client.get_price("BTCUSDT")

        assert price == Decimal("45000.00")
        client._spot.get_price.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_ticker(self, client):
        """Test get_ticker."""
        mock_ticker = make_ticker("BTCUSDT", Decimal("45000"))
        client._spot.get_ticker = AsyncMock(return_value=mock_ticker)

        ticker = await client.get_ticker("BTCUSDT")

        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == Decimal("45000")

    @pytest.mark.asyncio
    async def test_get_klines(self, client):
        """Test get_klines."""
        mock_klines = [make_kline(), make_kline()]
        client._spot.get_klines = AsyncMock(return_value=mock_klines)

        klines = await client.get_klines("BTCUSDT", KlineInterval.h1, limit=100)

        assert len(klines) == 2
        client._spot.get_klines.assert_called_once_with("BTCUSDT", KlineInterval.h1, 100)


# =============================================================================
# Account Tests
# =============================================================================


class TestExchangeClientAccount:
    """Test account methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient with mocked APIs."""
        client = ExchangeClient()
        client._spot = MagicMock()
        client._futures = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_get_balance_spot(self, client):
        """Test get_balance for spot."""
        mock_balance = make_balance("USDT")
        client._spot.get_balance = AsyncMock(return_value=mock_balance)

        balance = await client.get_balance("USDT", MarketType.SPOT)

        assert balance.asset == "USDT"
        assert balance.free == Decimal("1000")

    @pytest.mark.asyncio
    async def test_get_balance_futures(self, client):
        """Test get_balance for futures."""
        mock_balance = Balance(asset="USDT", free=Decimal("2000"), locked=Decimal("0"))
        client._futures.get_balance = AsyncMock(return_value=mock_balance)

        balance = await client.get_balance("USDT", MarketType.FUTURES)

        assert balance.asset == "USDT"
        assert balance.free == Decimal("2000")

    @pytest.mark.asyncio
    async def test_get_account(self, client):
        """Test get_account."""
        mock_account = make_account_info(MarketType.SPOT)
        client._spot.get_account = AsyncMock(return_value=mock_account)

        account = await client.get_account(MarketType.SPOT)

        assert account.market_type == MarketType.SPOT
        assert len(account.balances) == 2

    @pytest.mark.asyncio
    async def test_get_positions(self, client):
        """Test get_positions (futures only)."""
        mock_positions = [make_position("BTCUSDT")]
        client._futures.get_positions = AsyncMock(return_value=mock_positions)

        positions = await client.get_positions("BTCUSDT")

        assert len(positions) == 1
        assert positions[0].symbol == "BTCUSDT"
        client._futures.get_positions.assert_called_once_with("BTCUSDT")


# =============================================================================
# Order Tests
# =============================================================================


class TestExchangeClientOrders:
    """Test order methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient with mocked APIs."""
        client = ExchangeClient()
        client._spot = MagicMock()
        client._futures = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_create_order(self, client):
        """Test create_order."""
        mock_order = make_order()
        client._spot.create_order = AsyncMock(return_value=mock_order)

        order = await client.create_order(
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("0.001"),
            price=Decimal("45000"),
        )

        assert order.order_id == "12345"
        assert order.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_cancel_order(self, client):
        """Test cancel_order."""
        mock_order = make_order(status=OrderStatus.CANCELED)
        client._spot.cancel_order = AsyncMock(return_value=mock_order)

        order = await client.cancel_order("BTCUSDT", "12345")

        assert order.status == OrderStatus.CANCELED
        client._spot.cancel_order.assert_called_once_with("BTCUSDT", order_id="12345")

    @pytest.mark.asyncio
    async def test_get_order(self, client):
        """Test get_order."""
        mock_order = make_order()
        client._spot.get_order = AsyncMock(return_value=mock_order)

        order = await client.get_order("BTCUSDT", "12345")

        assert order.order_id == "12345"

    @pytest.mark.asyncio
    async def test_get_open_orders(self, client):
        """Test get_open_orders."""
        mock_order = make_order()
        client._spot.get_open_orders = AsyncMock(return_value=[mock_order])

        orders = await client.get_open_orders("BTCUSDT")

        assert len(orders) == 1
        assert orders[0].order_id == "12345"

    @pytest.mark.asyncio
    async def test_market_buy(self, client):
        """Test market_buy."""
        mock_order = make_order(order_type=OrderType.MARKET)
        client._spot.market_buy = AsyncMock(return_value=mock_order)

        order = await client.market_buy("BTCUSDT", Decimal("0.001"))

        assert order.side == OrderSide.BUY
        client._spot.market_buy.assert_called_once_with("BTCUSDT", Decimal("0.001"))

    @pytest.mark.asyncio
    async def test_market_sell(self, client):
        """Test market_sell."""
        mock_order = make_order(side=OrderSide.SELL, order_type=OrderType.MARKET)
        client._spot.market_sell = AsyncMock(return_value=mock_order)

        order = await client.market_sell("BTCUSDT", Decimal("0.001"))

        assert order.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_limit_buy(self, client):
        """Test limit_buy."""
        mock_order = make_order()
        client._spot.limit_buy = AsyncMock(return_value=mock_order)

        order = await client.limit_buy("BTCUSDT", Decimal("0.001"), Decimal("45000"))

        assert order.side == OrderSide.BUY
        assert order.order_type == OrderType.LIMIT
        client._spot.limit_buy.assert_called_once_with("BTCUSDT", Decimal("0.001"), Decimal("45000"))

    @pytest.mark.asyncio
    async def test_limit_sell(self, client):
        """Test limit_sell."""
        mock_order = make_order(side=OrderSide.SELL)
        client._spot.limit_sell = AsyncMock(return_value=mock_order)

        order = await client.limit_sell("BTCUSDT", Decimal("0.001"), Decimal("46000"))

        assert order.side == OrderSide.SELL


# =============================================================================
# Subscription Tests
# =============================================================================


class TestExchangeClientSubscriptions:
    """Test subscription methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient with mocked WebSocket."""
        client = ExchangeClient()
        client._spot_ws = MagicMock()
        client._futures_ws = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_subscribe_ticker_spot(self, client):
        """Test subscribe_ticker for spot."""
        client._spot_ws.subscribe_ticker = AsyncMock(return_value=True)
        callback = MagicMock()

        result = await client.subscribe_ticker("BTCUSDT", callback, MarketType.SPOT)

        assert result is True
        client._spot_ws.subscribe_ticker.assert_called_once_with("BTCUSDT", callback)

    @pytest.mark.asyncio
    async def test_subscribe_ticker_futures(self, client):
        """Test subscribe_ticker for futures."""
        client._futures_ws.subscribe_ticker = AsyncMock(return_value=True)
        callback = MagicMock()

        result = await client.subscribe_ticker("BTCUSDT", callback, MarketType.FUTURES)

        assert result is True
        client._futures_ws.subscribe_ticker.assert_called_once_with("BTCUSDT", callback)

    @pytest.mark.asyncio
    async def test_subscribe_kline(self, client):
        """Test subscribe_kline."""
        client._spot_ws.subscribe_kline = AsyncMock(return_value=True)
        callback = MagicMock()

        result = await client.subscribe_kline("BTCUSDT", KlineInterval.h1, callback)

        assert result is True
        client._spot_ws.subscribe_kline.assert_called_once_with("BTCUSDT", KlineInterval.h1, callback)

    @pytest.mark.asyncio
    async def test_subscribe_without_ws_connection(self):
        """Test subscribe fails without WebSocket connection."""
        client = ExchangeClient()
        callback = MagicMock()

        result = await client.subscribe_ticker("BTCUSDT", callback)

        assert result is False

    @pytest.mark.asyncio
    async def test_unsubscribe_all(self, client):
        """Test unsubscribe_all."""
        client._spot_ws._subscriptions = {"btcusdt@ticker": MagicMock()}
        client._spot_ws.unsubscribe = AsyncMock(return_value=True)
        client._futures_ws._subscriptions = {"btcusdt@ticker": MagicMock()}
        client._futures_ws.unsubscribe = AsyncMock(return_value=True)

        await client.unsubscribe_all()

        client._spot_ws.unsubscribe.assert_called_once()
        client._futures_ws.unsubscribe.assert_called_once()


# =============================================================================
# Symbol Info & Precision Tests
# =============================================================================


class TestExchangeClientPrecision:
    """Test symbol info and precision methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient with mocked APIs."""
        client = ExchangeClient()
        client._spot = MagicMock()
        client._futures = MagicMock()
        return client

    @pytest.mark.asyncio
    async def test_get_symbol_info(self, client):
        """Test get_symbol_info."""
        mock_info = make_symbol_info()
        client._spot.get_exchange_info = AsyncMock(return_value=mock_info)

        info = await client.get_symbol_info("BTCUSDT")

        assert info.symbol == "BTCUSDT"
        assert info.price_precision == 2
        assert info.quantity_precision == 5

    @pytest.mark.asyncio
    async def test_get_symbol_info_caching(self, client):
        """Test symbol info is cached."""
        mock_info = make_symbol_info()
        client._spot.get_exchange_info = AsyncMock(return_value=mock_info)

        # First call
        await client.get_symbol_info("BTCUSDT")
        # Second call should use cache
        await client.get_symbol_info("BTCUSDT")

        # API should only be called once
        assert client._spot.get_exchange_info.call_count == 1

    def test_get_price_precision_from_cache(self, client):
        """Test get_price_precision from cache."""
        client._symbol_cache["BTCUSDT"] = {MarketType.SPOT: make_symbol_info()}

        precision = client.get_price_precision("BTCUSDT")

        assert precision == 2

    def test_get_price_precision_default(self, client):
        """Test get_price_precision default value."""
        precision = client.get_price_precision("UNKNOWN")

        assert precision == 8

    def test_get_quantity_precision_from_cache(self, client):
        """Test get_quantity_precision from cache."""
        client._symbol_cache["BTCUSDT"] = {MarketType.SPOT: make_symbol_info()}

        precision = client.get_quantity_precision("BTCUSDT")

        assert precision == 5

    def test_get_quantity_precision_default(self, client):
        """Test get_quantity_precision default value."""
        precision = client.get_quantity_precision("UNKNOWN")

        assert precision == 8

    def test_round_price(self, client):
        """Test round_price."""
        client._symbol_cache["BTCUSDT"] = {MarketType.SPOT: make_symbol_info()}

        rounded = client.round_price("BTCUSDT", Decimal("45123.456789"))

        assert rounded == Decimal("45123.45")

    def test_round_quantity(self, client):
        """Test round_quantity."""
        client._symbol_cache["BTCUSDT"] = {MarketType.SPOT: make_symbol_info()}

        rounded = client.round_quantity("BTCUSDT", Decimal("0.123456789"))

        assert rounded == Decimal("0.12345")

    def test_round_decimal_with_precision_zero(self, client):
        """Test rounding with precision 0."""
        result = ExchangeClient._round_decimal(Decimal("123.456"), 0)
        assert result == Decimal("123")

    def test_round_decimal_rounds_down(self, client):
        """Test that rounding uses ROUND_DOWN."""
        result = ExchangeClient._round_decimal(Decimal("0.9999"), 2)
        assert result == Decimal("0.99")


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestExchangeClientHelpers:
    """Test helper methods."""

    @pytest.fixture
    def client(self):
        """Create ExchangeClient instance."""
        return ExchangeClient()

    def test_get_api_spot(self, client):
        """Test _get_api returns spot API."""
        api = client._get_api(MarketType.SPOT)
        assert api is client._spot

    def test_get_api_futures(self, client):
        """Test _get_api returns futures API."""
        api = client._get_api(MarketType.FUTURES)
        assert api is client._futures

    def test_get_ws_spot(self, client):
        """Test _get_ws returns spot WebSocket."""
        client._spot_ws = MagicMock()
        ws = client._get_ws(MarketType.SPOT)
        assert ws is client._spot_ws

    def test_get_ws_futures(self, client):
        """Test _get_ws returns futures WebSocket."""
        client._futures_ws = MagicMock()
        ws = client._get_ws(MarketType.FUTURES)
        assert ws is client._futures_ws


# =============================================================================
# Integration Tests (Optional)
# =============================================================================


@pytest.mark.integration
class TestExchangeClientIntegration:
    """
    Integration tests that actually connect to Binance.

    Run with: pytest -m integration
    """

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test actual connection and disconnection."""
        async with ExchangeClient(testnet=True) as client:
            assert client.is_connected is True

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_get_price_live(self):
        """Test getting live price."""
        async with ExchangeClient(testnet=True) as client:
            price = await client.get_price("BTCUSDT")
            assert price > 0

    @pytest.mark.asyncio
    async def test_get_symbol_info_live(self):
        """Test getting live symbol info."""
        async with ExchangeClient(testnet=True) as client:
            info = await client.get_symbol_info("BTCUSDT")
            assert info.symbol == "BTCUSDT"
            assert info.price_precision >= 0
            assert info.quantity_precision >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
