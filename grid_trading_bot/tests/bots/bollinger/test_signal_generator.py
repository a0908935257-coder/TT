"""
Tests for Signal Generator.

Tests signal generation including:
- Long signals at lower band
- Short signals at upper band
- No signal in middle zone
- BBW squeeze filtering
- Exit conditions (middle band, stop loss, timeout)
"""

import pytest
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import Mock, MagicMock

from src.bots.bollinger.signal_generator import SignalGenerator
from src.bots.bollinger.indicators import BollingerCalculator
from src.bots.bollinger.models import (
    BollingerConfig,
    BollingerBands,
    BBWData,
    Signal,
    SignalType,
    Position,
    PositionSide,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockKline:
    """Mock kline for testing."""

    close: Decimal
    high: Decimal = Decimal("0")
    low: Decimal = Decimal("0")
    open: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    close_time: Optional[datetime] = None
    is_closed: bool = True

    def __post_init__(self):
        if self.high == Decimal("0"):
            self.high = self.close
        if self.low == Decimal("0"):
            self.low = self.close
        if self.open == Decimal("0"):
            self.open = self.close
        if self.close_time is None:
            self.close_time = datetime.now(timezone.utc)


def create_klines(prices: list[float]) -> list[MockKline]:
    """Create mock klines from price list."""
    return [MockKline(close=Decimal(str(p))) for p in prices]


@pytest.fixture
def config() -> BollingerConfig:
    """Create default config."""
    return BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        leverage=2,
        position_size_pct=Decimal("0.1"),
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=Decimal("0.25"),
        stop_loss_pct=Decimal("0.015"),
        timeout_bars=16,
    )


@pytest.fixture
def calculator() -> BollingerCalculator:
    """Create calculator."""
    return BollingerCalculator(
        period=20,
        std_multiplier=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=Decimal("0.25"),
    )


@pytest.fixture
def signal_generator(config: BollingerConfig, calculator: BollingerCalculator) -> SignalGenerator:
    """Create signal generator."""
    return SignalGenerator(config, calculator)


@pytest.fixture
def mock_calculator() -> Mock:
    """Create mock calculator for controlled testing."""
    mock = Mock(spec=BollingerCalculator)
    return mock


@pytest.fixture
def signal_generator_with_mock(config: BollingerConfig, mock_calculator: Mock) -> SignalGenerator:
    """Create signal generator with mock calculator."""
    return SignalGenerator(config, mock_calculator)


def create_bands(middle: float, width: float) -> BollingerBands:
    """Create Bollinger Bands with given middle and width."""
    m = Decimal(str(middle))
    w = Decimal(str(width))
    return BollingerBands(
        upper=m + w,
        middle=m,
        lower=m - w,
        std=w / 2,
    )


def create_bbw(is_squeeze: bool, percentile: float = 0.5) -> BBWData:
    """Create BBW data."""
    return BBWData(
        bbw=Decimal("0.04"),
        percentile=Decimal(str(percentile)),
        is_squeeze=is_squeeze,
    )


# =============================================================================
# Test Long Signal at Lower Band
# =============================================================================


class TestLongSignalAtLowerBand:
    """Test long signal generation at lower band."""

    def test_long_signal_at_lower_band(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that long signal is generated when price touches lower band."""
        # Setup: price at lower band, no squeeze
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price at lower band
        current_price = Decimal("49000")  # At lower band (50000 - 1000)
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.LONG
        assert signal.entry_price == current_price
        assert signal.take_profit is not None
        assert signal.stop_loss is not None

    def test_long_signal_below_lower_band(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that long signal is generated when price is below lower band."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price below lower band
        current_price = Decimal("48500")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.LONG


# =============================================================================
# Test Short Signal at Upper Band
# =============================================================================


class TestShortSignalAtUpperBand:
    """Test short signal generation at upper band."""

    def test_short_signal_at_upper_band(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that short signal is generated when price touches upper band."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price at upper band
        current_price = Decimal("51000")  # At upper band (50000 + 1000)
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.SHORT
        assert signal.entry_price == current_price
        assert signal.take_profit is not None
        assert signal.stop_loss is not None

    def test_short_signal_above_upper_band(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that short signal is generated when price is above upper band."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price above upper band
        current_price = Decimal("51500")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.SHORT


# =============================================================================
# Test No Signal in Middle Zone
# =============================================================================


class TestNoSignalInMiddle:
    """Test no signal in middle zone."""

    def test_no_signal_in_middle(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that no signal is generated when price is in middle zone."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price in middle
        current_price = Decimal("50000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.NONE

    def test_no_signal_near_middle(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that no signal is generated when price is near middle."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price slightly above middle but not at upper band
        current_price = Decimal("50500")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.NONE


# =============================================================================
# Test No Signal When Squeeze
# =============================================================================


class TestNoSignalWhenSqueeze:
    """Test no signal during BBW squeeze."""

    def test_no_signal_when_squeeze(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that no signal is generated during squeeze even at band."""
        bands = create_bands(middle=50000, width=1000)
        # Squeeze active (BBW at 10th percentile)
        bbw = create_bbw(is_squeeze=True, percentile=0.1)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price at lower band (would normally trigger long)
        current_price = Decimal("49000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        # No signal due to squeeze
        assert signal.signal_type == SignalType.NONE
        assert signal.bbw.is_squeeze is True

    def test_no_short_signal_when_squeeze(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test that no short signal during squeeze."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=True, percentile=0.1)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Price at upper band
        current_price = Decimal("51000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        assert signal.signal_type == SignalType.NONE


# =============================================================================
# Test Stop Loss Calculation
# =============================================================================


class TestStopLossCalculation:
    """Test stop loss calculation."""

    def test_stop_loss_calculation_long(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
        config: BollingerConfig,
    ):
        """Test stop loss calculation for long position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        current_price = Decimal("49000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        # Stop loss should be below entry by stop_loss_pct (1.5%)
        expected_sl = current_price * (1 - config.stop_loss_pct)
        assert signal.stop_loss is not None
        assert abs(float(signal.stop_loss - expected_sl)) < 1

    def test_stop_loss_calculation_short(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
        config: BollingerConfig,
    ):
        """Test stop loss calculation for short position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False, percentile=0.5)
        mock_calculator.get_all.return_value = (bands, bbw)

        current_price = Decimal("51000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        # Stop loss should be above entry by stop_loss_pct (1.5%)
        expected_sl = current_price * (1 + config.stop_loss_pct)
        assert signal.stop_loss is not None
        assert abs(float(signal.stop_loss - expected_sl)) < 1


# =============================================================================
# Test Exit Conditions
# =============================================================================


class TestExitAtMiddleBand:
    """Test exit at middle band (take profit)."""

    def test_exit_at_middle_band_long(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test exit triggered when price returns to middle band for long."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        # Long position
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.01"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )

        # Price at middle band
        current_price = Decimal("50000")
        klines = create_klines([50000] * 20)

        should_exit, reason = signal_generator_with_mock.check_exit(
            position=position,
            klines=klines,
            current_price=current_price,
            current_bar=5,
        )

        assert should_exit is True
        assert "中軌" in reason or "止盈" in reason


class TestExitAtStopLoss:
    """Test exit at stop loss."""

    def test_exit_at_stop_loss_long(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test exit triggered at stop loss for long position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.01"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )

        # Price at stop loss
        current_price = Decimal("48200")
        klines = create_klines([50000] * 20)

        should_exit, reason = signal_generator_with_mock.check_exit(
            position=position,
            klines=klines,
            current_price=current_price,
            current_bar=5,
        )

        assert should_exit is True
        assert "止損" in reason

    def test_exit_at_stop_loss_short(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test exit triggered at stop loss for short position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.SHORT,
            entry_price=Decimal("51000"),
            quantity=Decimal("0.01"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("51765"),
        )

        # Price at stop loss
        current_price = Decimal("51800")
        klines = create_klines([50000] * 20)

        should_exit, reason = signal_generator_with_mock.check_exit(
            position=position,
            klines=klines,
            current_price=current_price,
            current_bar=5,
        )

        assert should_exit is True
        assert "止損" in reason


class TestExitOnTimeout:
    """Test exit on timeout."""

    def test_exit_on_timeout(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
        config: BollingerConfig,
    ):
        """Test exit triggered on hold timeout."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.01"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )

        # Price still in position, neither TP nor SL
        current_price = Decimal("49500")
        klines = create_klines([50000] * 20)

        # Hold for max_hold_bars (16) + entry_bar (1) = 17
        current_bar = position.entry_bar + config.timeout_bars + 1

        should_exit, reason = signal_generator_with_mock.check_exit(
            position=position,
            klines=klines,
            current_price=current_price,
            current_bar=current_bar,
        )

        assert should_exit is True
        assert "超時" in reason

    def test_no_exit_before_timeout(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test no exit before timeout if price is in range."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.01"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )

        # Price between entry and TP
        current_price = Decimal("49500")
        klines = create_klines([50000] * 20)

        # Only 5 bars held
        current_bar = 6

        should_exit, reason = signal_generator_with_mock.check_exit(
            position=position,
            klines=klines,
            current_price=current_price,
            current_bar=current_bar,
        )

        assert should_exit is False


# =============================================================================
# Test Take Profit Calculation
# =============================================================================


class TestTakeProfitCalculation:
    """Test take profit calculation."""

    def test_take_profit_at_middle_band_long(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test take profit is at middle band for long position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        current_price = Decimal("49000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        # Take profit should be at middle band
        assert signal.take_profit == bands.middle

    def test_take_profit_at_middle_band_short(
        self,
        signal_generator_with_mock: SignalGenerator,
        mock_calculator: Mock,
    ):
        """Test take profit is at middle band for short position."""
        bands = create_bands(middle=50000, width=1000)
        bbw = create_bbw(is_squeeze=False)
        mock_calculator.get_all.return_value = (bands, bbw)

        current_price = Decimal("51000")
        klines = create_klines([50000] * 20)

        signal = signal_generator_with_mock.generate(klines, current_price)

        # Take profit should be at middle band
        assert signal.take_profit == bands.middle
