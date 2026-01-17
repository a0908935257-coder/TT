"""
Tests for core models module: enums and Pydantic models.
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.models import (
    # Enums
    OrderSide,
    OrderType,
    OrderStatus,
    PositionSide,
    MarketType,
    KlineInterval,
    # Models
    Kline,
    Ticker,
    Order,
    Position,
    Balance,
    AccountInfo,
    Trade,
    SymbolInfo,
)


# =============================================================================
# Enum Tests
# =============================================================================


class TestEnums:
    """Test cases for enum definitions."""

    def test_order_side_values(self):
        """Test OrderSide enum values."""
        assert OrderSide.BUY.value == "BUY"
        assert OrderSide.SELL.value == "SELL"

    def test_order_type_values(self):
        """Test OrderType enum values."""
        assert OrderType.MARKET.value == "MARKET"
        assert OrderType.LIMIT.value == "LIMIT"
        assert OrderType.STOP_MARKET.value == "STOP_MARKET"
        assert OrderType.STOP_LIMIT.value == "STOP_LIMIT"
        assert OrderType.TAKE_PROFIT_MARKET.value == "TAKE_PROFIT_MARKET"

    def test_order_status_values(self):
        """Test OrderStatus enum values."""
        assert OrderStatus.NEW.value == "NEW"
        assert OrderStatus.PARTIALLY_FILLED.value == "PARTIALLY_FILLED"
        assert OrderStatus.FILLED.value == "FILLED"
        assert OrderStatus.CANCELED.value == "CANCELED"
        assert OrderStatus.REJECTED.value == "REJECTED"
        assert OrderStatus.EXPIRED.value == "EXPIRED"

    def test_position_side_values(self):
        """Test PositionSide enum values."""
        assert PositionSide.LONG.value == "LONG"
        assert PositionSide.SHORT.value == "SHORT"
        assert PositionSide.BOTH.value == "BOTH"

    def test_market_type_values(self):
        """Test MarketType enum values."""
        assert MarketType.SPOT.value == "SPOT"
        assert MarketType.FUTURES.value == "FUTURES"

    def test_kline_interval_values(self):
        """Test KlineInterval enum values."""
        assert KlineInterval.m1.value == "1m"
        assert KlineInterval.m5.value == "5m"
        assert KlineInterval.m15.value == "15m"
        assert KlineInterval.m30.value == "30m"
        assert KlineInterval.h1.value == "1h"
        assert KlineInterval.h4.value == "4h"
        assert KlineInterval.d1.value == "1d"
        assert KlineInterval.w1.value == "1w"

    def test_enum_string_serialization(self):
        """Test that enums serialize to strings."""
        assert str(OrderSide.BUY) == "OrderSide.BUY"
        assert OrderSide.BUY.value == "BUY"


# =============================================================================
# Kline Model Tests
# =============================================================================


class TestKline:
    """Test cases for Kline model."""

    @pytest.fixture
    def bullish_kline(self):
        """Create a bullish (green) kline."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=Decimal("40000"),
            high=Decimal("41000"),
            low=Decimal("39500"),
            close=Decimal("40500"),
            volume=Decimal("100"),
            close_time=datetime(2024, 1, 1, 0, 59, 59, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def bearish_kline(self):
        """Create a bearish (red) kline."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            open=Decimal("40500"),
            high=Decimal("41000"),
            low=Decimal("39500"),
            close=Decimal("40000"),
            volume=Decimal("100"),
            close_time=datetime(2024, 1, 1, 0, 59, 59, tzinfo=timezone.utc),
        )

    def test_kline_creation(self, bullish_kline):
        """Test basic kline creation."""
        assert bullish_kline.symbol == "BTCUSDT"
        assert bullish_kline.interval == "1h"  # use_enum_values=True
        assert bullish_kline.open == Decimal("40000")

    def test_kline_is_bullish(self, bullish_kline, bearish_kline):
        """Test is_bullish computed property."""
        assert bullish_kline.is_bullish is True
        assert bearish_kline.is_bullish is False

    def test_kline_is_bearish(self, bullish_kline, bearish_kline):
        """Test is_bearish computed property."""
        assert bullish_kline.is_bearish is False
        assert bearish_kline.is_bearish is True

    def test_kline_body(self, bullish_kline, bearish_kline):
        """Test body computed property."""
        assert bullish_kline.body == Decimal("500")  # |40500 - 40000|
        assert bearish_kline.body == Decimal("500")  # |40000 - 40500|

    def test_kline_range(self, bullish_kline):
        """Test range computed property."""
        assert bullish_kline.range == Decimal("1500")  # 41000 - 39500

    def test_kline_body_ratio(self, bullish_kline):
        """Test body_ratio computed property."""
        # body = 500, range = 1500
        expected = Decimal("500") / Decimal("1500")
        assert bullish_kline.body_ratio == expected

    def test_kline_body_ratio_zero_range(self):
        """Test body_ratio when range is zero (doji)."""
        doji = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            open=Decimal("40000"),
            high=Decimal("40000"),
            low=Decimal("40000"),
            close=Decimal("40000"),
            volume=Decimal("0"),
            close_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )
        assert doji.body_ratio == Decimal("0")

    def test_kline_from_binance(self):
        """Test from_binance factory method."""
        binance_data = [
            1704067200000,  # open_time
            "40000.00",     # open
            "41000.00",     # high
            "39500.00",     # low
            "40500.00",     # close
            "100.00",       # volume
            1704070799999,  # close_time
            "4050000.00",   # quote_volume
            150,            # trades_count
        ]
        kline = Kline.from_binance(binance_data, "BTCUSDT", KlineInterval.h1)
        assert kline.symbol == "BTCUSDT"
        assert kline.open == Decimal("40000.00")
        assert kline.high == Decimal("41000.00")
        assert kline.trades_count == 150

    def test_kline_defaults(self, bullish_kline):
        """Test default values."""
        assert bullish_kline.quote_volume == Decimal("0")
        assert bullish_kline.trades_count == 0


# =============================================================================
# Ticker Model Tests
# =============================================================================


class TestTicker:
    """Test cases for Ticker model."""

    @pytest.fixture
    def ticker(self):
        """Create a sample ticker."""
        return Ticker(
            symbol="BTCUSDT",
            price=Decimal("40000"),
            bid=Decimal("39999"),
            ask=Decimal("40001"),
            high_24h=Decimal("41000"),
            low_24h=Decimal("39000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("2.5"),
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_ticker_creation(self, ticker):
        """Test basic ticker creation."""
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == Decimal("40000")

    def test_ticker_spread(self, ticker):
        """Test spread computed property."""
        assert ticker.spread == Decimal("2")  # 40001 - 39999

    def test_ticker_spread_percent(self, ticker):
        """Test spread_percent computed property."""
        # spread = 2, price = 40000
        expected = (Decimal("2") / Decimal("40000")) * Decimal("100")
        assert ticker.spread_percent == expected

    def test_ticker_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
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
        ticker = Ticker.from_binance(binance_data)
        assert ticker.symbol == "BTCUSDT"
        assert ticker.price == Decimal("40000.00")
        assert ticker.change_24h == Decimal("2.5")


# =============================================================================
# Order Model Tests
# =============================================================================


class TestOrder:
    """Test cases for Order model."""

    @pytest.fixture
    def new_order(self):
        """Create a new (unfilled) order."""
        return Order(
            order_id="12345",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=Decimal("40000"),
            quantity=Decimal("1.0"),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def partial_order(self):
        """Create a partially filled order."""
        return Order(
            order_id="12346",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.PARTIALLY_FILLED,
            price=Decimal("40000"),
            quantity=Decimal("1.0"),
            filled_qty=Decimal("0.5"),
            avg_price=Decimal("40000"),
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def filled_order(self):
        """Create a filled order."""
        return Order(
            order_id="12347",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.FILLED,
            price=Decimal("40000"),
            quantity=Decimal("1.0"),
            filled_qty=Decimal("1.0"),
            avg_price=Decimal("40000"),
            fee=Decimal("0.001"),
            fee_asset="BTC",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_order_creation(self, new_order):
        """Test basic order creation."""
        assert new_order.order_id == "12345"
        assert new_order.symbol == "BTCUSDT"
        assert new_order.side == "BUY"  # use_enum_values=True

    def test_order_is_active(self, new_order, partial_order, filled_order):
        """Test is_active computed property."""
        assert new_order.is_active is True
        assert partial_order.is_active is True
        assert filled_order.is_active is False

    def test_order_is_filled(self, new_order, filled_order):
        """Test is_filled computed property."""
        assert new_order.is_filled is False
        assert filled_order.is_filled is True

    def test_order_filled_percent(self, new_order, partial_order, filled_order):
        """Test filled_percent computed property."""
        assert new_order.filled_percent == Decimal("0")
        assert partial_order.filled_percent == Decimal("50")
        assert filled_order.filled_percent == Decimal("100")

    def test_order_remaining_qty(self, new_order, partial_order, filled_order):
        """Test remaining_qty computed property."""
        assert new_order.remaining_qty == Decimal("1.0")
        assert partial_order.remaining_qty == Decimal("0.5")
        assert filled_order.remaining_qty == Decimal("0")

    def test_order_filled_value(self, partial_order, new_order):
        """Test filled_value computed property."""
        assert partial_order.filled_value == Decimal("20000")  # 0.5 * 40000
        assert new_order.filled_value == Decimal("0")  # No avg_price

    def test_order_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
            "orderId": 12345,
            "clientOrderId": "BOT_123",
            "symbol": "BTCUSDT",
            "side": "BUY",
            "type": "LIMIT",
            "status": "FILLED",
            "price": "40000.00",
            "origQty": "1.00",
            "executedQty": "1.00",
            "avgPrice": "40000.00",
            "time": 1704067200000,
            "updateTime": 1704067260000,
        }
        order = Order.from_binance(binance_data)
        assert order.order_id == "12345"
        assert order.client_order_id == "BOT_123"
        assert order.status == "FILLED"

    def test_order_defaults(self, new_order):
        """Test default values."""
        assert new_order.filled_qty == Decimal("0")
        assert new_order.fee == Decimal("0")
        assert new_order.client_order_id is None


# =============================================================================
# Position Model Tests
# =============================================================================


class TestPosition:
    """Test cases for Position model."""

    @pytest.fixture
    def long_position(self):
        """Create a long position."""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("1.0"),
            entry_price=Decimal("40000"),
            mark_price=Decimal("42000"),
            leverage=10,
            margin=Decimal("4000"),
            unrealized_pnl=Decimal("2000"),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    @pytest.fixture
    def short_position(self):
        """Create a short position."""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            quantity=Decimal("1.0"),
            entry_price=Decimal("42000"),
            mark_price=Decimal("40000"),
            leverage=10,
            margin=Decimal("4200"),
            unrealized_pnl=Decimal("2000"),
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_position_creation(self, long_position):
        """Test basic position creation."""
        assert long_position.symbol == "BTCUSDT"
        assert long_position.leverage == 10

    def test_position_notional_value(self, long_position):
        """Test notional_value computed property."""
        assert long_position.notional_value == Decimal("42000")  # 1.0 * 42000

    def test_position_roe(self, long_position):
        """Test roe computed property."""
        # unrealized_pnl = 2000, margin = 4000
        expected = (Decimal("2000") / Decimal("4000")) * Decimal("100")
        assert long_position.roe == expected  # 50%

    def test_position_is_long(self, long_position, short_position):
        """Test is_long computed property."""
        assert long_position.is_long is True
        assert short_position.is_long is False

    def test_position_is_short(self, long_position, short_position):
        """Test is_short computed property."""
        assert long_position.is_short is False
        assert short_position.is_short is True

    def test_position_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
            "symbol": "BTCUSDT",
            "positionSide": "LONG",
            "positionAmt": "1.0",
            "entryPrice": "40000.00",
            "markPrice": "42000.00",
            "liquidationPrice": "36000.00",
            "leverage": "10",
            "isolatedMargin": "4000.00",
            "unRealizedProfit": "2000.00",
            "marginType": "isolated",
            "updateTime": 1704067200000,
        }
        position = Position.from_binance(binance_data)
        assert position.symbol == "BTCUSDT"
        assert position.side == "LONG"
        assert position.quantity == Decimal("1.0")

    def test_position_defaults(self, long_position):
        """Test default values."""
        assert long_position.margin_type == "isolated"
        assert long_position.liquidation_price is None


# =============================================================================
# Balance Model Tests
# =============================================================================


class TestBalance:
    """Test cases for Balance model."""

    @pytest.fixture
    def balance(self):
        """Create a sample balance."""
        return Balance(
            asset="USDT",
            free=Decimal("10000"),
            locked=Decimal("5000"),
        )

    def test_balance_creation(self, balance):
        """Test basic balance creation."""
        assert balance.asset == "USDT"
        assert balance.free == Decimal("10000")

    def test_balance_total(self, balance):
        """Test total computed property."""
        assert balance.total == Decimal("15000")  # 10000 + 5000

    def test_balance_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
            "asset": "BTC",
            "free": "1.5",
            "locked": "0.5",
        }
        balance = Balance.from_binance(binance_data)
        assert balance.asset == "BTC"
        assert balance.total == Decimal("2.0")


# =============================================================================
# AccountInfo Model Tests
# =============================================================================


class TestAccountInfo:
    """Test cases for AccountInfo model."""

    @pytest.fixture
    def account_info(self):
        """Create sample account info."""
        return AccountInfo(
            market_type=MarketType.SPOT,
            balances=[
                Balance(asset="USDT", free=Decimal("10000"), locked=Decimal("0")),
                Balance(asset="BTC", free=Decimal("1.0"), locked=Decimal("0.5")),
            ],
            updated_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_account_info_creation(self, account_info):
        """Test basic account info creation."""
        assert account_info.market_type == "SPOT"
        assert len(account_info.balances) == 2

    def test_get_balance_found(self, account_info):
        """Test get_balance when asset exists."""
        usdt = account_info.get_balance("USDT")
        assert usdt is not None
        assert usdt.free == Decimal("10000")

    def test_get_balance_not_found(self, account_info):
        """Test get_balance when asset doesn't exist."""
        eth = account_info.get_balance("ETH")
        assert eth is None

    def test_account_info_from_binance_spot(self):
        """Test from_binance_spot factory method."""
        binance_data = {
            "balances": [
                {"asset": "USDT", "free": "10000.00", "locked": "0.00"},
                {"asset": "BTC", "free": "1.5", "locked": "0.5"},
                {"asset": "ETH", "free": "0.00", "locked": "0.00"},  # Should be filtered
            ],
            "updateTime": 1704067200000,
        }
        account = AccountInfo.from_binance_spot(binance_data)
        assert account.market_type == "SPOT"
        assert len(account.balances) == 2  # ETH filtered out

    def test_account_info_defaults(self, account_info):
        """Test default values."""
        assert account_info.positions == []


# =============================================================================
# Trade Model Tests
# =============================================================================


class TestTrade:
    """Test cases for Trade model."""

    @pytest.fixture
    def trade(self):
        """Create a sample trade."""
        return Trade(
            trade_id="98765",
            order_id="12345",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("40000"),
            quantity=Decimal("0.5"),
            fee=Decimal("0.0005"),
            fee_asset="BTC",
            is_maker=True,
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        )

    def test_trade_creation(self, trade):
        """Test basic trade creation."""
        assert trade.trade_id == "98765"
        assert trade.order_id == "12345"

    def test_trade_value(self, trade):
        """Test value computed property."""
        assert trade.value == Decimal("20000")  # 40000 * 0.5

    def test_trade_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
            "id": 98765,
            "orderId": 12345,
            "symbol": "BTCUSDT",
            "isBuyer": True,
            "price": "40000.00",
            "qty": "0.5",
            "commission": "0.0005",
            "commissionAsset": "BTC",
            "isMaker": True,
            "time": 1704067200000,
        }
        trade = Trade.from_binance(binance_data)
        assert trade.trade_id == "98765"
        assert trade.side == "BUY"

    def test_trade_defaults(self, trade):
        """Test default values."""
        assert trade.realized_pnl is None


# =============================================================================
# SymbolInfo Model Tests
# =============================================================================


class TestSymbolInfo:
    """Test cases for SymbolInfo model."""

    @pytest.fixture
    def symbol_info(self):
        """Create sample symbol info."""
        return SymbolInfo(
            symbol="BTCUSDT",
            base_asset="BTC",
            quote_asset="USDT",
            price_precision=2,
            quantity_precision=5,
            min_quantity=Decimal("0.00001"),
            min_notional=Decimal("10"),
            tick_size=Decimal("0.01"),
            step_size=Decimal("0.00001"),
        )

    def test_symbol_info_creation(self, symbol_info):
        """Test basic symbol info creation."""
        assert symbol_info.symbol == "BTCUSDT"
        assert symbol_info.base_asset == "BTC"
        assert symbol_info.quote_asset == "USDT"

    def test_symbol_info_from_binance(self):
        """Test from_binance factory method."""
        binance_data = {
            "symbol": "BTCUSDT",
            "baseAsset": "BTC",
            "quoteAsset": "USDT",
            "pricePrecision": 2,
            "quantityPrecision": 5,
            "filters": [
                {
                    "filterType": "PRICE_FILTER",
                    "tickSize": "0.01",
                },
                {
                    "filterType": "LOT_SIZE",
                    "minQty": "0.00001",
                    "stepSize": "0.00001",
                },
                {
                    "filterType": "NOTIONAL",
                    "minNotional": "10.00",
                },
            ],
        }
        info = SymbolInfo.from_binance(binance_data)
        assert info.symbol == "BTCUSDT"
        assert info.tick_size == Decimal("0.01")
        assert info.min_notional == Decimal("10.00")


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Test Pydantic validation behavior."""

    def test_invalid_quantity_type(self):
        """Test that invalid types raise validation error."""
        with pytest.raises(ValidationError):
            Balance(
                asset="USDT",
                free="not_a_number",  # Invalid
                locked=Decimal("0"),
            )

    def test_validate_assignment(self):
        """Test that validate_assignment works."""
        balance = Balance(
            asset="USDT",
            free=Decimal("100"),
            locked=Decimal("0"),
        )
        # Should be able to assign valid value
        balance.free = Decimal("200")
        assert balance.free == Decimal("200")

    def test_use_enum_values(self):
        """Test that use_enum_values converts to string."""
        order = Order(
            order_id="123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.NEW,
            quantity=Decimal("1.0"),
            created_at=datetime.now(timezone.utc),
        )
        # Should be string, not enum
        assert order.side == "BUY"
        assert order.order_type == "MARKET"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
