"""
Tests for Position Manager.

Tests position management including:
- Position size calculation
- Opening long/short positions
- Closing positions with profit/loss
- Position existence checks
- Leverage settings
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, MagicMock
from dataclasses import dataclass

from src.bots.bollinger.position_manager import (
    PositionManager,
    PositionExistsError,
    NoPositionError,
)
from src.bots.bollinger.models import (
    BollingerConfig,
    Signal,
    SignalType,
    Position,
    PositionSide,
    BollingerBands,
    BBWData,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockAccount:
    """Mock account for testing."""
    available_balance: Decimal = Decimal("1000")


@dataclass
class MockTicker:
    """Mock ticker for testing."""
    last_price: Decimal = Decimal("50000")


@dataclass
class MockPosition:
    """Mock exchange position for testing."""
    symbol: str = "BTCUSDT"
    quantity: Decimal = Decimal("0.01")
    entry_price: Decimal = Decimal("50000")
    leverage: int = 2
    unrealized_pnl: Decimal = Decimal("0")
    margin: Decimal = Decimal("250")
    notional_value: Decimal = Decimal("500")
    liquidation_price: Decimal = Decimal("45000")


@pytest.fixture
def config() -> BollingerConfig:
    """Create default config."""
    return BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        leverage=3,
        position_size_pct=Decimal("0.1"),  # 10%
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=Decimal("0.25"),
        stop_loss_pct=Decimal("0.015"),
        timeout_bars=16,
    )


@pytest.fixture
def mock_exchange() -> Mock:
    """Create mock exchange."""
    mock = Mock()
    mock.futures_set_leverage = AsyncMock(return_value={})
    mock.futures_set_margin_type = AsyncMock(return_value={})
    mock.futures_get_account = AsyncMock(return_value=MockAccount())
    mock.futures_get_positions = AsyncMock(return_value=[])
    mock.get_ticker = AsyncMock(return_value=MockTicker())
    return mock


@pytest.fixture
def mock_data_manager() -> Mock:
    """Create mock data manager."""
    mock = Mock()
    mock.save_trade = AsyncMock()
    return mock


@pytest.fixture
def position_manager(
    config: BollingerConfig,
    mock_exchange: Mock,
    mock_data_manager: Mock,
) -> PositionManager:
    """Create position manager."""
    return PositionManager(config, mock_exchange, mock_data_manager)


def create_signal(
    signal_type: SignalType,
    entry_price: float,
    take_profit: float,
    stop_loss: float,
) -> Signal:
    """Create a signal for testing."""
    return Signal(
        signal_type=signal_type,
        entry_price=Decimal(str(entry_price)),
        take_profit=Decimal(str(take_profit)),
        stop_loss=Decimal(str(stop_loss)),
        bands=BollingerBands(
            upper=Decimal("51000"),
            middle=Decimal("50000"),
            lower=Decimal("49000"),
            std=Decimal("500"),
        ),
        bbw=BBWData(
            bbw=Decimal("0.04"),
            percentile=Decimal("0.5"),
            is_squeeze=False,
        ),
    )


# =============================================================================
# Test Position Size Calculation
# =============================================================================


class TestPositionSizeCalculation:
    """Test position size calculation."""

    @pytest.mark.asyncio
    async def test_position_size_calculation(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """
        Test position size calculation.

        Setup:
        - Available balance: 1000 USDT
        - position_size_pct: 10%
        - Leverage: 3x
        - Entry price: 50000

        Calculation:
        - Used capital = 1000 * 10% = 100 USDT
        - Notional value = 100 * 3 = 300 USDT
        - Quantity = 300 / 50000 = 0.006 BTC
        """
        entry_price = Decimal("50000")

        quantity = await position_manager.calculate_position_size(entry_price)

        # Expected: 0.006 BTC (rounded to 3 decimal places)
        assert quantity == Decimal("0.006")

    @pytest.mark.asyncio
    async def test_position_size_with_different_balance(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test position size with different balance."""
        # Change balance to 5000
        mock_exchange.futures_get_account.return_value = MockAccount(
            available_balance=Decimal("5000")
        )

        entry_price = Decimal("50000")
        quantity = await position_manager.calculate_position_size(entry_price)

        # 5000 * 0.1 * 3 / 50000 = 0.03
        assert quantity == Decimal("0.030")

    @pytest.mark.asyncio
    async def test_position_size_with_different_price(
        self,
        position_manager: PositionManager,
    ):
        """Test position size with different entry price."""
        entry_price = Decimal("25000")

        quantity = await position_manager.calculate_position_size(entry_price)

        # 1000 * 0.1 * 3 / 25000 = 0.012
        assert quantity == Decimal("0.012")


# =============================================================================
# Test Open Long Position
# =============================================================================


class TestOpenLongPosition:
    """Test opening long positions."""

    @pytest.mark.asyncio
    async def test_open_long_position(
        self,
        position_manager: PositionManager,
    ):
        """Test opening a long position."""
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )

        position = await position_manager.open_position(signal)

        assert position is not None
        assert position.side == PositionSide.LONG
        assert position.entry_price == Decimal("49000")
        assert position.symbol == "BTCUSDT"
        assert position.leverage == 3
        assert position.take_profit_price == Decimal("50000")
        assert position.stop_loss_price == Decimal("48265")

    @pytest.mark.asyncio
    async def test_open_long_position_quantity(
        self,
        position_manager: PositionManager,
    ):
        """Test that long position has correct quantity."""
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=50000,
            take_profit=51000,
            stop_loss=49250,
        )

        position = await position_manager.open_position(signal)

        # 1000 * 0.1 * 3 / 50000 = 0.006
        assert position.quantity == Decimal("0.006")


# =============================================================================
# Test Open Short Position
# =============================================================================


class TestOpenShortPosition:
    """Test opening short positions."""

    @pytest.mark.asyncio
    async def test_open_short_position(
        self,
        position_manager: PositionManager,
    ):
        """Test opening a short position."""
        signal = create_signal(
            signal_type=SignalType.SHORT,
            entry_price=51000,
            take_profit=50000,
            stop_loss=51765,
        )

        position = await position_manager.open_position(signal)

        assert position is not None
        assert position.side == PositionSide.SHORT
        assert position.entry_price == Decimal("51000")
        assert position.take_profit_price == Decimal("50000")
        assert position.stop_loss_price == Decimal("51765")


# =============================================================================
# Test Close Position with Profit
# =============================================================================


class TestClosePositionProfit:
    """Test closing positions with profit."""

    @pytest.mark.asyncio
    async def test_close_position_profit_long(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test closing long position with profit."""
        # Open position
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )
        await position_manager.open_position(signal)

        # Set exit price higher (profit)
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("50000"))

        # Close position
        record = await position_manager.close_position("止盈")

        assert record is not None
        assert record.pnl > 0
        assert record.exit_reason == "止盈"
        assert position_manager.get_position() is None

    @pytest.mark.asyncio
    async def test_close_position_profit_short(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test closing short position with profit."""
        # Open short position
        signal = create_signal(
            signal_type=SignalType.SHORT,
            entry_price=51000,
            take_profit=50000,
            stop_loss=51765,
        )
        await position_manager.open_position(signal)

        # Set exit price lower (profit for short)
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("50000"))

        # Close position
        record = await position_manager.close_position("止盈")

        assert record is not None
        assert record.pnl > 0


# =============================================================================
# Test Close Position with Loss
# =============================================================================


class TestClosePositionLoss:
    """Test closing positions with loss."""

    @pytest.mark.asyncio
    async def test_close_position_loss_long(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test closing long position with loss."""
        # Open position
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )
        await position_manager.open_position(signal)

        # Set exit price lower (loss)
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("48000"))

        # Close position
        record = await position_manager.close_position("止損")

        assert record is not None
        assert record.pnl < 0
        assert record.exit_reason == "止損"

    @pytest.mark.asyncio
    async def test_close_position_loss_short(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test closing short position with loss."""
        # Open short position
        signal = create_signal(
            signal_type=SignalType.SHORT,
            entry_price=51000,
            take_profit=50000,
            stop_loss=51765,
        )
        await position_manager.open_position(signal)

        # Set exit price higher (loss for short)
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("52000"))

        # Close position
        record = await position_manager.close_position("止損")

        assert record is not None
        assert record.pnl < 0


# =============================================================================
# Test Cannot Open When Position Exists
# =============================================================================


class TestCannotOpenWhenPositionExists:
    """Test that cannot open position when one already exists."""

    @pytest.mark.asyncio
    async def test_cannot_open_when_position_exists(
        self,
        position_manager: PositionManager,
    ):
        """Test that opening position when one exists raises error."""
        # Open first position
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )
        await position_manager.open_position(signal)

        # Try to open another position
        signal2 = create_signal(
            signal_type=SignalType.SHORT,
            entry_price=51000,
            take_profit=50000,
            stop_loss=51765,
        )

        with pytest.raises(PositionExistsError):
            await position_manager.open_position(signal2)


# =============================================================================
# Test Cannot Close When No Position
# =============================================================================


class TestCannotCloseWhenNoPosition:
    """Test that cannot close position when none exists."""

    @pytest.mark.asyncio
    async def test_cannot_close_when_no_position(
        self,
        position_manager: PositionManager,
    ):
        """Test that closing position when none exists raises error."""
        with pytest.raises(NoPositionError):
            await position_manager.close_position("test")


# =============================================================================
# Test Leverage Setting
# =============================================================================


class TestLeverageSetting:
    """Test leverage settings."""

    @pytest.mark.asyncio
    async def test_leverage_setting(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test that leverage is set during initialization."""
        await position_manager.initialize()

        mock_exchange.futures_set_leverage.assert_called_once_with(
            symbol="BTCUSDT",
            leverage=3,
        )

    @pytest.mark.asyncio
    async def test_margin_type_setting(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test that margin type is set to ISOLATED."""
        await position_manager.initialize()

        mock_exchange.futures_set_margin_type.assert_called_once_with(
            symbol="BTCUSDT",
            margin_type="ISOLATED",
        )


# =============================================================================
# Test Has Position
# =============================================================================


class TestHasPosition:
    """Test has_position property."""

    def test_has_position_false_initially(
        self,
        position_manager: PositionManager,
    ):
        """Test has_position is False initially."""
        assert position_manager.has_position is False

    @pytest.mark.asyncio
    async def test_has_position_true_after_open(
        self,
        position_manager: PositionManager,
    ):
        """Test has_position is True after opening position."""
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )
        await position_manager.open_position(signal)

        assert position_manager.has_position is True

    @pytest.mark.asyncio
    async def test_has_position_false_after_close(
        self,
        position_manager: PositionManager,
    ):
        """Test has_position is False after closing position."""
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=49000,
            take_profit=50000,
            stop_loss=48265,
        )
        await position_manager.open_position(signal)
        await position_manager.close_position("test")

        assert position_manager.has_position is False


# =============================================================================
# Test Sync With Exchange
# =============================================================================


class TestSyncWithExchange:
    """Test synchronization with exchange."""

    @pytest.mark.asyncio
    async def test_sync_with_existing_position(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test sync finds existing position on exchange."""
        # Setup exchange to return existing position
        mock_exchange.futures_get_positions.return_value = [
            MockPosition(
                quantity=Decimal("0.01"),
                entry_price=Decimal("49000"),
            )
        ]

        await position_manager.sync_with_exchange()

        assert position_manager.has_position is True
        position = position_manager.get_position()
        assert position.quantity == Decimal("0.01")
        assert position.entry_price == Decimal("49000")

    @pytest.mark.asyncio
    async def test_sync_with_no_position(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test sync when no position on exchange."""
        mock_exchange.futures_get_positions.return_value = []

        await position_manager.sync_with_exchange()

        assert position_manager.has_position is False

    @pytest.mark.asyncio
    async def test_sync_detects_short_position(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test sync detects short position (negative quantity)."""
        mock_exchange.futures_get_positions.return_value = [
            MockPosition(
                quantity=Decimal("-0.01"),  # Negative = short
                entry_price=Decimal("51000"),
            )
        ]

        await position_manager.sync_with_exchange()

        position = position_manager.get_position()
        assert position.side == PositionSide.SHORT
        assert position.quantity == Decimal("0.01")  # Absolute value


# =============================================================================
# Test PnL Calculation
# =============================================================================


class TestPnLCalculation:
    """Test PnL calculation."""

    @pytest.mark.asyncio
    async def test_pnl_calculation_long_profit(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test PnL calculation for profitable long."""
        signal = create_signal(
            signal_type=SignalType.LONG,
            entry_price=50000,
            take_profit=51000,
            stop_loss=49250,
        )
        await position_manager.open_position(signal)

        # Exit at 51000
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("51000"))

        record = await position_manager.close_position("止盈")

        # PnL = (51000 - 50000) * 0.006 = 6 USDT
        assert record.pnl == Decimal("6.000")

    @pytest.mark.asyncio
    async def test_pnl_calculation_short_profit(
        self,
        position_manager: PositionManager,
        mock_exchange: Mock,
    ):
        """Test PnL calculation for profitable short."""
        signal = create_signal(
            signal_type=SignalType.SHORT,
            entry_price=50000,
            take_profit=49000,
            stop_loss=50750,
        )
        await position_manager.open_position(signal)

        # Exit at 49000
        mock_exchange.get_ticker.return_value = MockTicker(last_price=Decimal("49000"))

        record = await position_manager.close_position("止盈")

        # PnL = (50000 - 49000) * 0.006 = 6 USDT
        assert record.pnl == Decimal("6.000")
