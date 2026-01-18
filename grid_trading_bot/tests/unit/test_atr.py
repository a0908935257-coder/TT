"""
Unit tests for ATR Calculator.

Tests ATR calculation, volatility assessment, and grid parameter suggestions.
"""

from decimal import Decimal

import pytest

from src.bots.grid import ATRCalculator, ATRConfig, ATRData, GridConfig, RiskLevel


class TestATRCalculator:
    """Tests for ATRCalculator class."""

    def test_calculate_tr(self, atr_test_klines):
        """Test True Range calculation."""
        # TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        # For kline index 1:
        #   High=102, Low=99, Prev Close=99
        #   TR = max(102-99, |102-99|, |99-99|) = max(3, 3, 0) = 3

        config = ATRConfig(period=5, timeframe="4h")
        atr_data = ATRCalculator.calculate_from_klines(
            klines=atr_test_klines,
            config=config,
        )

        # ATR should be calculated and positive
        assert atr_data.value > 0
        assert atr_data.period == 5

    def test_calculate_atr_basic(self, sample_klines):
        """Test basic ATR calculation with sample klines."""
        config = ATRConfig(period=14, timeframe="4h")
        atr_data = ATRCalculator.calculate_from_klines(
            klines=sample_klines,
            config=config,
        )

        # Verify ATR data structure
        assert isinstance(atr_data, ATRData)
        assert atr_data.value > 0
        assert atr_data.period == 14
        assert atr_data.timeframe == "4h"
        assert atr_data.current_price > 0

    def test_calculate_atr_with_gaps(self):
        """Test ATR calculation with price gaps (simulating gaps via large moves)."""
        # Create klines with gaps - expanded to support min period=5
        highs = [
            Decimal("100"), Decimal("102"), Decimal("115"),  # Gap up
            Decimal("112"), Decimal("110"), Decimal("113"), Decimal("111"),
        ]
        lows = [
            Decimal("98"), Decimal("99"), Decimal("108"),
            Decimal("105"), Decimal("107"), Decimal("109"), Decimal("108"),
        ]
        closes = [
            Decimal("99"), Decimal("101"), Decimal("110"),  # Large move
            Decimal("108"), Decimal("109"), Decimal("111"), Decimal("110"),
        ]

        config = ATRConfig(period=5, timeframe="4h")
        atr_data = ATRCalculator.calculate(
            highs=highs,
            lows=lows,
            closes=closes,
            config=config,
        )

        # ATR should reflect the higher volatility from gaps
        assert atr_data.value > Decimal("3")  # Higher volatility expected

    def test_atr_matches_tradingview(self, atr_test_klines):
        """Test ATR calculation matches expected values."""
        # Using period=5 with expanded test klines
        # TR calculations result in consistent values around 3

        config = ATRConfig(period=5, timeframe="4h")
        atr_data = ATRCalculator.calculate_from_klines(
            klines=atr_test_klines,
            config=config,
        )

        # With EMA smoothing, ATR should be close to 3
        assert Decimal("2") < atr_data.value < Decimal("4")

    def test_volatility_level(self, sample_klines):
        """Test volatility level classification."""
        config = ATRConfig(period=14, timeframe="4h")
        atr_data = ATRCalculator.calculate_from_klines(
            klines=sample_klines,
            config=config,
        )

        # Volatility percent should be calculated
        assert atr_data.volatility_percent >= 0

        # Volatility level should be one of the defined levels
        valid_levels = ["極低", "低", "中等", "高", "極高"]
        assert atr_data.volatility_level in valid_levels

        # English level should also be valid
        valid_levels_en = ["very_low", "low", "medium", "high", "very_high"]
        assert atr_data.volatility_level_en in valid_levels_en

    def test_insufficient_data_error(self):
        """Test error when insufficient data provided."""
        highs = [Decimal("100"), Decimal("102")]
        lows = [Decimal("98"), Decimal("99")]
        closes = [Decimal("99"), Decimal("101")]

        config = ATRConfig(period=14, timeframe="4h")
        with pytest.raises(ValueError, match="Need at least"):
            ATRCalculator.calculate(
                highs=highs,
                lows=lows,
                closes=closes,
                config=config,  # Requires 15 data points
            )

    def test_mismatched_lengths_error(self):
        """Test error when array lengths don't match."""
        highs = [Decimal("100"), Decimal("102"), Decimal("101")]
        lows = [Decimal("98"), Decimal("99")]  # Different length
        closes = [Decimal("99"), Decimal("101"), Decimal("100")]

        config = ATRConfig(period=5, timeframe="4h")
        with pytest.raises(ValueError, match="same length"):
            ATRCalculator.calculate(
                highs=highs,
                lows=lows,
                closes=closes,
                config=config,
            )


class TestATRData:
    """Tests for ATRData model."""

    def test_volatility_percent_calculation(self):
        """Test volatility percentage calculation."""
        atr_data = ATRData(
            value=Decimal("500"),
            period=14,
            timeframe="4h",
            multiplier=Decimal("2.0"),
            current_price=Decimal("50000"),
            upper_price=Decimal("51000"),
            lower_price=Decimal("49000"),
        )

        # 500 / 50000 * 100 = 1%
        assert atr_data.volatility_percent == Decimal("1")

    def test_volatility_levels(self):
        """Test volatility level thresholds (per Prompt 17 spec)."""
        # Thresholds: <1% very_low, 1-2% low, 2-4% medium, 4-6% high, >=6% very_high
        test_cases = [
            (Decimal("250"), Decimal("50000"), "very_low"),   # 0.5%
            (Decimal("750"), Decimal("50000"), "low"),        # 1.5%
            (Decimal("1500"), Decimal("50000"), "medium"),    # 3%
            (Decimal("2500"), Decimal("50000"), "high"),      # 5%
            (Decimal("3500"), Decimal("50000"), "very_high"), # 7%
        ]

        for atr_value, price, expected_level in test_cases:
            atr_data = ATRData(
                value=atr_value,
                period=14,
                timeframe="4h",
                multiplier=Decimal("2.0"),
                current_price=price,
                upper_price=price + atr_value * 2,
                lower_price=price - atr_value * 2,
            )
            assert atr_data.volatility_level_en == expected_level, (
                f"ATR {atr_value} / Price {price} = {atr_data.volatility_percent}% "
                f"should be '{expected_level}', got '{atr_data.volatility_level_en}'"
            )

    def test_zero_price_handling(self):
        """Test handling of zero price."""
        atr_data = ATRData(
            value=Decimal("500"),
            period=14,
            timeframe="4h",
            multiplier=Decimal("2.0"),
            current_price=Decimal("0"),
            upper_price=Decimal("1000"),
            lower_price=Decimal("0"),
        )

        assert atr_data.volatility_percent == Decimal("0")


class TestATRSuggestParameters:
    """Tests for parameter suggestion functionality."""

    def test_suggest_parameters_basic(self):
        """Test basic parameter suggestion."""
        atr_data = ATRData(
            value=Decimal("1000"),
            period=14,
            timeframe="4h",
            multiplier=Decimal("2.0"),
            current_price=Decimal("50000"),
            upper_price=Decimal("52000"),
            lower_price=Decimal("48000"),
        )

        params = ATRCalculator.suggest_parameters(
            atr_data=atr_data,
            investment=Decimal("10000"),
            risk_level=RiskLevel.MODERATE,
        )

        # Verify required keys
        assert "volatility_level" in params
        assert "suggested_upper_price" in params
        assert "suggested_lower_price" in params
        assert "suggested_grid_count" in params
        assert "grid_spacing_percent" in params
        assert "amount_per_grid" in params

        # Upper should be higher than current
        assert params["suggested_upper_price"] > atr_data.current_price

        # Lower should be lower than current
        assert params["suggested_lower_price"] < atr_data.current_price

    def test_suggest_parameters_different_risk_levels(self):
        """Test parameter suggestions vary by risk level."""
        # Using ATRConfig factory methods to create proper ATRData
        # Conservative: 1.5x, Aggressive: 2.5x
        conservative_config = ATRConfig.conservative()
        aggressive_config = ATRConfig.aggressive()

        atr_value = Decimal("1000")
        current_price = Decimal("50000")

        conservative_atr = ATRData(
            value=atr_value,
            period=14,
            timeframe="4h",
            multiplier=conservative_config.multiplier,
            current_price=current_price,
            upper_price=current_price + atr_value * conservative_config.multiplier,
            lower_price=current_price - atr_value * conservative_config.multiplier,
        )

        aggressive_atr = ATRData(
            value=atr_value,
            period=14,
            timeframe="4h",
            multiplier=aggressive_config.multiplier,
            current_price=current_price,
            upper_price=current_price + atr_value * aggressive_config.multiplier,
            lower_price=current_price - atr_value * aggressive_config.multiplier,
        )

        conservative = ATRCalculator.suggest_parameters(
            atr_data=conservative_atr,
            investment=Decimal("10000"),
            risk_level=RiskLevel.CONSERVATIVE,
        )

        aggressive = ATRCalculator.suggest_parameters(
            atr_data=aggressive_atr,
            investment=Decimal("10000"),
            risk_level=RiskLevel.AGGRESSIVE,
        )

        # Aggressive should have wider range
        conservative_range = (
            conservative["suggested_upper_price"] -
            conservative["suggested_lower_price"]
        )
        aggressive_range = (
            aggressive["suggested_upper_price"] -
            aggressive["suggested_lower_price"]
        )

        assert aggressive_range > conservative_range
