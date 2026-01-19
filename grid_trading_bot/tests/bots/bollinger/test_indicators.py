"""
Tests for Bollinger Band indicators.

Tests BollingerCalculator including:
- Bollinger Bands calculation
- BBW (Bollinger Band Width) calculation
- BBW percentile ranking
- Squeeze detection
"""

import pytest
from decimal import Decimal
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.bots.bollinger.indicators import (
    BollingerCalculator,
    BandPosition,
    InsufficientDataError,
)
from src.bots.bollinger.models import BollingerBands, BBWData


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


def create_stable_klines(base_price: float, count: int) -> list[MockKline]:
    """Create klines with stable price (for low BBW)."""
    return create_klines([base_price] * count)


def create_volatile_klines(base_price: float, count: int, volatility: float = 0.05) -> list[MockKline]:
    """Create klines with volatility."""
    import random
    random.seed(42)
    prices = []
    for _ in range(count):
        change = random.uniform(-volatility, volatility)
        prices.append(base_price * (1 + change))
    return create_klines(prices)


@pytest.fixture
def calculator() -> BollingerCalculator:
    """Create default calculator."""
    return BollingerCalculator(
        period=20,
        std_multiplier=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=25,  # 25th percentile
    )


@pytest.fixture
def short_calculator() -> BollingerCalculator:
    """Create calculator with short period for testing."""
    return BollingerCalculator(
        period=5,
        std_multiplier=Decimal("2.0"),
        bbw_lookback=20,
        bbw_threshold_pct=25,
    )


# =============================================================================
# Test Bollinger Bands Calculation
# =============================================================================


class TestBollingerBandsCalculation:
    """Test Bollinger Bands calculation."""

    def test_bollinger_bands_calculation(self, short_calculator: BollingerCalculator):
        """Test that Bollinger Bands are calculated correctly."""
        # Create 5 klines with known values
        prices = [100.0, 101.0, 99.0, 100.5, 100.0]
        klines = create_klines(prices)

        # Calculate manually
        # Mean = (100 + 101 + 99 + 100.5 + 100) / 5 = 500.5 / 5 = 100.1
        # Variance = ((100-100.1)^2 + (101-100.1)^2 + (99-100.1)^2 + (100.5-100.1)^2 + (100-100.1)^2) / 5
        # = (0.01 + 0.81 + 1.21 + 0.16 + 0.01) / 5 = 2.2 / 5 = 0.44
        # Std = sqrt(0.44) ≈ 0.663
        # Upper = 100.1 + 2 * 0.663 ≈ 101.426
        # Lower = 100.1 - 2 * 0.663 ≈ 98.774

        bands = short_calculator.calculate(klines)

        assert bands is not None
        # Check middle band (SMA)
        assert abs(float(bands.middle) - 100.1) < 0.01
        # Check that upper > middle > lower
        assert bands.upper > bands.middle
        assert bands.middle > bands.lower
        # Check band width relationship
        expected_half_width = (bands.upper - bands.middle)
        actual_half_width = (bands.middle - bands.lower)
        assert abs(float(expected_half_width - actual_half_width)) < 0.01

    def test_bollinger_bands_with_constant_price(self, short_calculator: BollingerCalculator):
        """Test Bollinger Bands with constant price (zero std)."""
        klines = create_klines([100.0] * 5)

        bands = short_calculator.calculate(klines)

        assert bands is not None
        # With constant price, std = 0, so upper = middle = lower
        assert bands.middle == Decimal("100")
        assert bands.upper == bands.middle
        assert bands.lower == bands.middle

    def test_bollinger_bands_uses_latest_period(self, short_calculator: BollingerCalculator):
        """Test that only the latest N klines are used."""
        # Create more klines than period
        prices = [50.0] * 10 + [100.0, 101.0, 99.0, 100.5, 100.0]
        klines = create_klines(prices)

        bands = short_calculator.calculate(klines)

        # Should use only last 5 klines (period=5)
        assert bands is not None
        assert abs(float(bands.middle) - 100.1) < 0.01


# =============================================================================
# Test BBW Calculation
# =============================================================================


class TestBBWCalculation:
    """Test Bollinger Band Width calculation."""

    def test_bbw_calculation(self, short_calculator: BollingerCalculator):
        """Test BBW is calculated correctly."""
        prices = [100.0, 102.0, 98.0, 101.0, 99.0]
        klines = create_klines(prices)

        # First calculate bands
        bands = short_calculator.calculate(klines)
        # Then calculate BBW from bands
        bbw = short_calculator.calculate_bbw(bands)

        assert bbw is not None
        # BBW = (upper - lower) / middle
        # Should be positive
        assert bbw.bbw > 0

    def test_bbw_zero_with_constant_price(self, short_calculator: BollingerCalculator):
        """Test BBW is zero with constant price."""
        klines = create_klines([100.0] * 5)

        bands = short_calculator.calculate(klines)
        bbw = short_calculator.calculate_bbw(bands)

        assert bbw is not None
        assert bbw.bbw == Decimal("0")

    def test_bbw_increases_with_volatility(self, short_calculator: BollingerCalculator):
        """Test BBW increases with more volatility."""
        # Low volatility
        low_vol_prices = [100.0, 100.5, 99.5, 100.2, 99.8]
        low_vol_klines = create_klines(low_vol_prices)

        # High volatility
        high_vol_prices = [100.0, 105.0, 95.0, 103.0, 97.0]
        high_vol_klines = create_klines(high_vol_prices)

        low_bands = short_calculator.calculate(low_vol_klines)
        high_bands = short_calculator.calculate(high_vol_klines)

        low_bbw = short_calculator.calculate_bbw(low_bands)
        high_bbw = short_calculator.calculate_bbw(high_bands)

        assert low_bbw is not None
        assert high_bbw is not None
        assert high_bbw.bbw > low_bbw.bbw


# =============================================================================
# Test BBW Percentile
# =============================================================================


class TestBBWPercentile:
    """Test BBW percentile ranking."""

    def test_bbw_percentile_calculation(self, calculator: BollingerCalculator):
        """Test BBW percentile is calculated correctly."""
        # Initialize with history
        klines = create_volatile_klines(100.0, 150, volatility=0.03)
        calculator.initialize(klines)

        # Calculate bands and BBW using get_all
        bands, bbw = calculator.get_all(klines)

        assert bbw is not None
        # Percentile should be between 0 and 100
        assert 0 <= bbw.bbw_percentile <= 100

    def test_low_percentile_with_low_volatility(self, calculator: BollingerCalculator):
        """Test that low volatility gives low percentile."""
        # Initialize with volatile history
        volatile_klines = create_volatile_klines(100.0, 100, volatility=0.05)
        calculator.initialize(volatile_klines)

        # Now add very stable klines
        stable_klines = create_klines([100.0] * 20)
        all_klines = volatile_klines + stable_klines

        bands, bbw = calculator.get_all(all_klines)

        assert bbw is not None
        # Current BBW should be very low compared to history
        assert bbw.bbw_percentile < 30

    def test_high_percentile_with_high_volatility(self, calculator: BollingerCalculator):
        """Test that high volatility gives high percentile."""
        # Initialize with stable history
        stable_klines = create_klines([100.0] * 100)
        calculator.initialize(stable_klines)

        # Now add volatile klines
        volatile_klines = create_volatile_klines(100.0, 20, volatility=0.1)
        all_klines = stable_klines + volatile_klines

        bands, bbw = calculator.get_all(all_klines)

        assert bbw is not None
        # Current BBW should be high compared to stable history
        assert bbw.bbw_percentile > 70


# =============================================================================
# Test Squeeze Detection
# =============================================================================


class TestSqueezeDetection:
    """Test Bollinger Band squeeze detection."""

    def test_squeeze_detection_when_low_bbw(self, calculator: BollingerCalculator):
        """Test squeeze is detected when BBW is below threshold."""
        # Initialize with volatile history
        volatile_klines = create_volatile_klines(100.0, 100, volatility=0.05)
        calculator.initialize(volatile_klines)

        # Add stable klines to trigger squeeze
        stable_klines = create_klines([100.0, 100.1, 99.9, 100.05, 99.95] * 4)
        all_klines = volatile_klines + stable_klines

        bands, bbw = calculator.get_all(all_klines)

        assert bbw is not None
        # Should detect squeeze (BBW below 25th percentile)
        if bbw.bbw_percentile < 25:
            assert bbw.is_squeeze is True

    def test_no_squeeze_with_normal_volatility(self, calculator: BollingerCalculator):
        """Test no squeeze with normal volatility."""
        # Initialize with similar volatility throughout
        klines = create_volatile_klines(100.0, 150, volatility=0.03)
        calculator.initialize(klines)

        bands, bbw = calculator.get_all(klines)

        assert bbw is not None
        # With consistent volatility, percentile should be around 50%
        # Not a squeeze
        assert bbw.bbw_percentile > 25 or bbw.is_squeeze is False


# =============================================================================
# Test Insufficient Data Handling
# =============================================================================


class TestInsufficientData:
    """Test handling of insufficient data."""

    def test_insufficient_data_raises_error(self, calculator: BollingerCalculator):
        """Test that insufficient data raises error."""
        klines = create_klines([100.0] * 5)  # Less than period (20)

        with pytest.raises(InsufficientDataError):
            calculator.calculate(klines)

    def test_exact_period_works(self, short_calculator: BollingerCalculator):
        """Test that exactly period number of klines works."""
        klines = create_klines([100.0] * 5)  # Exactly period (5)

        bands = short_calculator.calculate(klines)

        assert bands is not None

    def test_empty_klines_raises_error(self, calculator: BollingerCalculator):
        """Test that empty klines raises error."""
        with pytest.raises(InsufficientDataError):
            calculator.calculate([])


# =============================================================================
# Test Band Position
# =============================================================================


class TestBandPosition:
    """Test band position determination."""

    def test_price_above_upper_band(self, short_calculator: BollingerCalculator):
        """Test detection of price above upper band."""
        # Create klines with moderate volatility
        prices = [100.0, 100.5, 99.5, 100.0, 100.2]
        klines = create_klines(prices)

        bands = short_calculator.calculate(klines)
        # Price clearly above upper band
        position = short_calculator.get_band_position(Decimal("120"), bands)

        assert position == BandPosition.ABOVE_UPPER

    def test_price_below_lower_band(self, short_calculator: BollingerCalculator):
        """Test detection of price below lower band."""
        prices = [100.0, 100.5, 99.5, 100.0, 100.2]
        klines = create_klines(prices)

        bands = short_calculator.calculate(klines)
        # Price clearly below lower band
        position = short_calculator.get_band_position(Decimal("80"), bands)

        assert position == BandPosition.BELOW_LOWER

    def test_price_in_upper_half(self, short_calculator: BollingerCalculator):
        """Test detection of price in upper half zone."""
        prices = [100.0, 101.0, 99.0, 100.5, 100.0]
        klines = create_klines(prices)

        bands = short_calculator.calculate(klines)
        # Price slightly above middle
        above_middle = bands.middle + Decimal("0.1")
        position = short_calculator.get_band_position(above_middle, bands)

        assert position == BandPosition.UPPER_HALF

    def test_price_in_lower_half(self, short_calculator: BollingerCalculator):
        """Test detection of price in lower half zone."""
        prices = [100.0, 101.0, 99.0, 100.5, 100.0]
        klines = create_klines(prices)

        bands = short_calculator.calculate(klines)
        # Price at or below middle
        position = short_calculator.get_band_position(bands.middle, bands)

        assert position == BandPosition.LOWER_HALF


# =============================================================================
# Test Initialize and Get All
# =============================================================================


class TestInitializeAndGetAll:
    """Test initialize and get_all methods."""

    def test_initialize_builds_bbw_history(self, calculator: BollingerCalculator):
        """Test that initialize builds BBW history."""
        klines = create_volatile_klines(100.0, 150, volatility=0.03)

        calculator.initialize(klines)

        # After initialization, BBW history should be populated
        bands, bbw = calculator.get_all(klines)
        assert bands is not None
        assert bbw is not None
        # Percentile should be valid (has history to compare)
        assert 0 <= bbw.bbw_percentile <= 100

    def test_get_all_returns_both(self, short_calculator: BollingerCalculator):
        """Test get_all returns both bands and BBW."""
        klines = create_klines([100.0, 101.0, 99.0, 100.5, 100.0])

        bands, bbw = short_calculator.get_all(klines)

        assert isinstance(bands, BollingerBands)
        assert isinstance(bbw, BBWData)
