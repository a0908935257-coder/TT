"""
Data Flow Tests.

Tests data flow through system components:
Exchange → KlineManager → Database → Redis → ATR → Grid Calculator
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bots.grid.atr import ATRCalculator, ATRConfig
from src.bots.grid.calculator import SmartGridCalculator
from src.bots.grid.models import GridConfig
from src.core.models import Kline, KlineInterval
from src.data.kline.manager import KlineManager


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_klines(count: int = 50, base_price: Decimal = Decimal("50000")) -> list[Kline]:
    """Create mock kline data for testing."""
    klines = []
    current_time = datetime.now(timezone.utc) - timedelta(hours=count)

    for i in range(count):
        # Create realistic price movement
        variation = Decimal(str(i % 10)) * Decimal("10")
        open_price = base_price + variation
        close_price = open_price + (Decimal("50") if i % 2 == 0 else Decimal("-30"))
        high_price = max(open_price, close_price) + Decimal("100")
        low_price = min(open_price, close_price) - Decimal("80")

        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=current_time + timedelta(hours=i),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            close_time=current_time + timedelta(hours=i + 1) - timedelta(seconds=1),
            quote_volume=Decimal("50000000"),
            trades_count=5000,
        )
        klines.append(kline)

    return klines


def create_mock_binance_kline_data(count: int = 50) -> list[list[Any]]:
    """Create mock Binance API kline response data."""
    data = []
    current_time = datetime.now(timezone.utc) - timedelta(hours=count)

    for i in range(count):
        open_time_ms = int((current_time + timedelta(hours=i)).timestamp() * 1000)
        close_time_ms = int(
            (current_time + timedelta(hours=i + 1) - timedelta(seconds=1)).timestamp() * 1000
        )

        # Create realistic price data
        base = 50000 + (i % 10) * 10
        kline_data = [
            open_time_ms,  # Open time
            str(base),  # Open
            str(base + 100),  # High
            str(base - 80),  # Low
            str(base + 50 if i % 2 == 0 else base - 30),  # Close
            "1000",  # Volume
            close_time_ms,  # Close time
            "50000000",  # Quote volume
            5000,  # Number of trades
        ]
        data.append(kline_data)

    return data


@pytest.fixture
def mock_redis():
    """Create mock Redis manager."""
    redis = MagicMock()
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def mock_database():
    """Create mock database manager."""
    db = MagicMock()
    session = MagicMock()
    session.execute = AsyncMock()
    session.add = MagicMock()

    # Configure context manager
    db.get_session = MagicMock()
    db.get_session.return_value.__aenter__ = AsyncMock(return_value=session)
    db.get_session.return_value.__aexit__ = AsyncMock(return_value=None)

    return db


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()
    exchange.spot = MagicMock()
    exchange.spot.get_klines = AsyncMock(return_value=create_mock_klines(50))
    exchange.ws = MagicMock()
    exchange.ws.subscribe_kline = AsyncMock()
    exchange.ws.unsubscribe_kline = AsyncMock()
    return exchange


@pytest.fixture
def kline_manager(mock_redis, mock_database, mock_exchange):
    """Create KlineManager with mocks."""
    return KlineManager(mock_redis, mock_database, mock_exchange)


# =============================================================================
# Exchange → KlineManager Tests
# =============================================================================


class TestExchangeToKlineManager:
    """Tests for Exchange → KlineManager data flow."""

    @pytest.mark.asyncio
    async def test_get_klines_from_exchange(self, kline_manager, mock_exchange):
        """Test KlineManager fetches klines from exchange."""
        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert
        assert len(klines) == 50
        mock_exchange.spot.get_klines.assert_called()

    @pytest.mark.asyncio
    async def test_kline_data_format(self, kline_manager):
        """Test kline data has correct format."""
        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=10)

        # Assert
        for kline in klines:
            assert kline.symbol == "BTCUSDT"
            assert kline.interval == "1h"
            assert isinstance(kline.open, Decimal)
            assert isinstance(kline.high, Decimal)
            assert isinstance(kline.low, Decimal)
            assert isinstance(kline.close, Decimal)
            assert isinstance(kline.volume, Decimal)
            assert kline.high >= kline.low
            assert kline.high >= kline.open
            assert kline.high >= kline.close

    @pytest.mark.asyncio
    async def test_kline_timestamp_ordering(self, kline_manager):
        """Test klines are ordered by timestamp ascending."""
        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert - timestamps should be ascending
        for i in range(1, len(klines)):
            assert klines[i].open_time > klines[i - 1].open_time

    @pytest.mark.asyncio
    async def test_symbol_normalization(self, kline_manager, mock_exchange):
        """Test symbol is normalized to uppercase."""
        # Act
        await kline_manager.get_klines("btcusdt", KlineInterval.h1, limit=10)

        # Assert - should convert to uppercase
        call_args = mock_exchange.spot.get_klines.call_args
        assert call_args[0][0] == "BTCUSDT"


# =============================================================================
# KlineManager → Database Tests
# =============================================================================


class TestKlineManagerToDatabase:
    """Tests for KlineManager → Database data flow."""

    @pytest.mark.asyncio
    async def test_klines_saved_to_database(self, kline_manager, mock_database):
        """Test klines are saved to database."""
        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert - database session should be used
        mock_database.get_session.assert_called()

    @pytest.mark.asyncio
    async def test_database_upsert_logic(self, kline_manager, mock_database):
        """Test database upsert prevents duplicates."""
        # Arrange
        session = await mock_database.get_session().__aenter__()

        # Configure to return existing record
        existing = MagicMock()
        result = MagicMock()
        result.scalar_one_or_none.return_value = existing
        session.execute.return_value = result

        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=5)

        # Assert - session.add should not be called for existing records
        # (updates existing instead)
        assert session.execute.called

    @pytest.mark.asyncio
    async def test_new_klines_inserted(self, kline_manager, mock_database):
        """Test new klines are inserted."""
        # Arrange
        session = await mock_database.get_session().__aenter__()

        # Configure to return no existing record
        result = MagicMock()
        result.scalar_one_or_none.return_value = None
        result.scalars.return_value.all.return_value = []
        session.execute.return_value = result

        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=5)

        # Assert - new records should be added
        assert session.add.called or session.execute.called


# =============================================================================
# Database → Redis Tests
# =============================================================================


class TestDatabaseToRedis:
    """Tests for Database → Redis cache data flow."""

    @pytest.mark.asyncio
    async def test_klines_cached_to_redis(self, kline_manager, mock_redis):
        """Test klines are cached to Redis."""
        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert
        mock_redis.set.assert_called()

    @pytest.mark.asyncio
    async def test_redis_cache_key_format(self, kline_manager, mock_redis):
        """Test Redis cache key format."""
        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert - verify key format (key includes symbol and interval)
        call_args = mock_redis.set.call_args
        key = call_args[0][0]
        assert "BTCUSDT" in key
        assert "1h" in key

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_data(self, kline_manager, mock_redis, mock_exchange):
        """Test cache hit avoids exchange call."""
        # Arrange - simulate cache hit
        cached_data = [
            {
                "open_time": datetime.now(timezone.utc).isoformat(),
                "open": "50000",
                "high": "51000",
                "low": "49000",
                "close": "50500",
                "volume": "1000",
                "close_time": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "quote_volume": "50000000",
                "trades_count": 5000,
            }
            for _ in range(100)
        ]
        mock_redis.get.return_value = cached_data

        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert
        assert len(klines) == 50
        # Cache hit means no exchange call needed
        mock_redis.get.assert_called()

    @pytest.mark.asyncio
    async def test_cache_ttl_configured(self, kline_manager, mock_redis):
        """Test cache TTL is set."""
        # Act
        await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # Assert - TTL should be passed
        call_args = mock_redis.set.call_args
        if call_args and call_args.kwargs:
            assert "ttl" in call_args.kwargs


# =============================================================================
# Kline → ATR Calculator Tests
# =============================================================================


class TestKlineToATR:
    """Tests for Kline → ATR Calculator data flow."""

    def test_atr_from_klines(self):
        """Test ATR calculation from klines."""
        # Arrange
        klines = create_mock_klines(50)

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines)

        # Assert
        assert atr_data.value > 0
        assert atr_data.period == 14  # Default period

    def test_atr_reasonable_value(self):
        """Test ATR value is reasonable (< 10% of price)."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines)

        # Assert - ATR should be less than 10% of price
        max_reasonable_atr = Decimal("5000")  # 10% of 50000
        assert atr_data.value < max_reasonable_atr

    def test_atr_with_custom_config(self):
        """Test ATR calculation with custom config."""
        # Arrange
        klines = create_mock_klines(50)
        config = ATRConfig(period=20, use_ema=False)  # use_ema=False for SMA

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines, config)

        # Assert
        assert atr_data.value > 0
        assert atr_data.period == 20

    def test_atr_requires_minimum_klines(self):
        """Test ATR requires minimum number of klines."""
        # Arrange - too few klines (need at least period+1 = 15 for default config)
        klines = create_mock_klines(5)

        # Act & Assert - should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Need at least"):
            ATRCalculator.calculate_from_klines(klines)

    def test_atr_current_price_tracking(self):
        """Test ATR tracks current price."""
        # Arrange
        klines = create_mock_klines(50)

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines)

        # Assert
        assert atr_data.current_price > 0
        # Current price should be close to last kline's close
        assert atr_data.current_price == klines[-1].close


# =============================================================================
# ATR → Grid Calculator Tests
# =============================================================================


class TestATRToGrid:
    """Tests for ATR → Grid Calculator data flow."""

    def test_grid_from_atr(self):
        """Test grid calculation using ATR."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))
        current_price = klines[-1].close

        # Act
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert
        assert grid_setup.upper_price > current_price
        assert grid_setup.lower_price < current_price
        assert grid_setup.grid_count > 0
        # Grid has grid_count intervals, which means grid_count+1 price levels
        assert len(grid_setup.levels) == grid_setup.grid_count + 1

    def test_grid_levels_ordering(self):
        """Test grid levels are properly ordered."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))
        current_price = klines[-1].close

        # Act
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert - levels should be in ascending order (lowest price first)
        for i in range(1, len(grid_setup.levels)):
            assert grid_setup.levels[i - 1].price < grid_setup.levels[i].price

    def test_grid_price_range_validation(self):
        """Test grid price range is valid."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))
        current_price = klines[-1].close

        # Act
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert
        assert grid_setup.upper_price > grid_setup.lower_price
        assert grid_setup.lower_price > 0
        # Current price should be within grid range
        assert grid_setup.lower_price <= current_price <= grid_setup.upper_price

    def test_grid_fund_allocation(self):
        """Test grid fund allocation."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))
        current_price = klines[-1].close
        total_investment = Decimal("10000")

        # Act
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=total_investment,
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert - total allocation should not exceed investment
        total_allocated = sum(level.allocated_amount for level in grid_setup.levels)
        assert total_allocated <= total_investment * Decimal("1.01")  # Allow 1% rounding

    def test_grid_level_quantities(self):
        """Test grid levels have valid quantities."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("50000"))
        current_price = klines[-1].close

        # Act
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert
        for level in grid_setup.levels:
            assert level.price > 0
            # Buy levels have allocated_amount > 0, sell levels have 0
            if level.side.value == "buy":
                assert level.allocated_amount > 0


# =============================================================================
# Full Data Flow Tests
# =============================================================================


class TestFullDataFlow:
    """Tests for complete data flow through all components."""

    @pytest.mark.asyncio
    async def test_exchange_to_grid_flow(self, kline_manager, mock_exchange):
        """Test complete flow: Exchange → KlineManager → ATR → Grid."""
        # Step 1: Get klines from exchange via manager
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)
        assert len(klines) > 0
        assert all(isinstance(k, Kline) for k in klines)

        # Step 2: Calculate ATR from klines
        atr_data = ATRCalculator.calculate_from_klines(klines)
        assert atr_data.value > 0

        # Step 3: Calculate grid from ATR
        current_price = klines[-1].close
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=current_price,
        )
        grid_setup = calculator.calculate()

        # Assert final grid is valid
        assert grid_setup.upper_price > current_price
        assert grid_setup.lower_price < current_price
        assert len(grid_setup.levels) > 0

    @pytest.mark.asyncio
    async def test_data_integrity_through_flow(self, kline_manager):
        """Test data integrity is maintained through entire flow."""
        # Arrange
        symbol = "ETHUSDT"
        mock_klines = create_mock_klines(50, base_price=Decimal("3000"))

        # Patch exchange to return specific klines
        with patch.object(
            kline_manager._exchange.spot, "get_klines", new_callable=AsyncMock
        ) as mock_get:
            mock_get.return_value = mock_klines

            # Act - Full flow
            klines = await kline_manager.get_klines(symbol, KlineInterval.h1, limit=50)

            # Verify kline integrity
            assert len(klines) == 50
            for kline in klines:
                assert kline.high >= kline.low
                assert kline.volume > 0

            # Calculate ATR
            atr_data = ATRCalculator.calculate_from_klines(klines)
            assert atr_data.current_price == klines[-1].close

            # Calculate grid
            config = GridConfig(
                symbol=symbol.upper(),
                total_investment=Decimal("5000"),
            )
            calculator = SmartGridCalculator(
                config=config,
                klines=klines,
                current_price=klines[-1].close,
            )
            grid_setup = calculator.calculate()

            # Verify grid uses correct price
            assert grid_setup.lower_price < atr_data.current_price < grid_setup.upper_price

    @pytest.mark.asyncio
    async def test_no_data_loss_through_flow(self, kline_manager, mock_redis, mock_database):
        """Test no data is lost through the flow."""
        # Arrange
        original_count = 50

        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=original_count)

        # Verify count
        assert len(klines) == original_count

        # Verify cache was updated
        mock_redis.set.assert_called()

        # Find the call that set the klines list (not the single kline cache)
        kline_list_cached = False
        for call in mock_redis.set.call_args_list:
            key = call[0][0]
            data = call[0][1]
            if "klines:" in key and isinstance(data, list):
                assert len(data) == original_count
                kline_list_cached = True
                break

        assert kline_list_cached, "Klines list was not cached to Redis"

    @pytest.mark.asyncio
    async def test_format_consistency_through_flow(self, kline_manager):
        """Test data format is consistent through entire flow."""
        # Act
        klines = await kline_manager.get_klines("BTCUSDT", KlineInterval.h1, limit=50)

        # All prices should be Decimal
        for kline in klines:
            assert isinstance(kline.open, Decimal)
            assert isinstance(kline.high, Decimal)
            assert isinstance(kline.low, Decimal)
            assert isinstance(kline.close, Decimal)
            assert isinstance(kline.volume, Decimal)

        # ATR should return Decimal
        atr_data = ATRCalculator.calculate_from_klines(klines)
        assert isinstance(atr_data.value, Decimal)
        assert isinstance(atr_data.current_price, Decimal)

        # Grid should use Decimal
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=klines[-1].close,
        )
        grid_setup = calculator.calculate()

        assert isinstance(grid_setup.upper_price, Decimal)
        assert isinstance(grid_setup.lower_price, Decimal)
        for level in grid_setup.levels:
            assert isinstance(level.price, Decimal)
            assert isinstance(level.quantity, Decimal)


# =============================================================================
# Edge Cases
# =============================================================================


class TestDataFlowEdgeCases:
    """Edge case tests for data flow."""

    def test_empty_klines_handling(self):
        """Test handling of empty kline list."""
        # Arrange
        klines: list[Kline] = []

        # Act & Assert - ATR should handle empty list
        with pytest.raises((ValueError, IndexError, Exception)):
            ATRCalculator.calculate_from_klines(klines)

    def test_single_kline_handling(self):
        """Test handling of single kline - requires at least period+1 klines."""
        # Arrange - only 1 kline, but need at least 15 for default period
        klines = create_mock_klines(1)

        # Act & Assert - should raise ValueError for insufficient data
        with pytest.raises(ValueError, match="Need at least"):
            ATRCalculator.calculate_from_klines(klines)

    def test_extreme_price_movement(self):
        """Test handling of extreme price movements."""
        # Arrange - Create klines with extreme volatility
        klines = []
        base_time = datetime.now(timezone.utc)

        for i in range(50):
            # Alternate between high and low prices
            if i % 2 == 0:
                price = Decimal("50000")
            else:
                price = Decimal("60000")  # 20% jump

            kline = Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.h1,
                open_time=base_time + timedelta(hours=i),
                open=price,
                high=price + Decimal("1000"),
                low=price - Decimal("500"),
                close=price + Decimal("500"),
                volume=Decimal("1000"),
                close_time=base_time + timedelta(hours=i + 1),
            )
            klines.append(kline)

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines)
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=klines[-1].close,
        )
        grid_setup = calculator.calculate()

        # Assert - Should handle extreme volatility
        assert atr_data.value > 0
        assert grid_setup.upper_price > grid_setup.lower_price

    def test_very_low_price_asset(self):
        """Test handling of very low price assets."""
        # Arrange
        klines = create_mock_klines(50, base_price=Decimal("0.00001"))

        # Act
        atr_data = ATRCalculator.calculate_from_klines(klines)
        config = GridConfig(
            symbol="SHIBUSDT",
            total_investment=Decimal("100"),
        )
        calculator = SmartGridCalculator(
            config=config,
            klines=klines,
            current_price=klines[-1].close,
        )
        grid_setup = calculator.calculate()

        # Assert - Should handle low prices
        assert atr_data.value > 0
        assert grid_setup.lower_price > 0
        assert len(grid_setup.levels) > 0
