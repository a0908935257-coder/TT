"""
Unit tests for slippage models.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.backtest.slippage import (
    SlippageModel,
    SlippageModelType,
    SlippageContext,
    FixedSlippage,
    VolatilityBasedSlippage,
    MarketImpactSlippage,
    create_slippage_model,
)
from src.core.models import Kline, KlineInterval


def create_test_kline(
    open_price: Decimal = Decimal("100"),
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("95"),
    close: Decimal = Decimal("102"),
) -> Kline:
    """Create a test kline."""
    return Kline(
        symbol="BTCUSDT",
        interval=KlineInterval.h1,
        open_time=datetime(2024, 1, 1, 0, 0),
        close_time=datetime(2024, 1, 1, 0, 59),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=Decimal("1000"),
        quote_volume=Decimal("100000"),
        trades_count=100,
    )


class TestFixedSlippage:
    """Tests for FixedSlippage model."""

    def test_zero_slippage(self):
        """Test with zero slippage."""
        model = FixedSlippage(Decimal("0"))
        kline = create_test_kline()

        slippage = model.calculate_slippage(Decimal("100"), True, kline)
        assert slippage == Decimal("0")

    def test_buy_slippage(self):
        """Test slippage on buy orders."""
        model = FixedSlippage(Decimal("0.001"))  # 0.1%
        kline = create_test_kline()

        fill_price = model.apply_slippage(Decimal("100"), is_buy=True, kline=kline)
        # 100 + 0.1% = 100.1
        assert fill_price == Decimal("100.1")

    def test_sell_slippage(self):
        """Test slippage on sell orders."""
        model = FixedSlippage(Decimal("0.001"))  # 0.1%
        kline = create_test_kline()

        fill_price = model.apply_slippage(Decimal("100"), is_buy=False, kline=kline)
        # 100 - 0.1% = 99.9
        assert fill_price == Decimal("99.9")

    def test_buy_capped_at_high(self):
        """Test buy slippage capped at kline high."""
        model = FixedSlippage(Decimal("0.1"))  # 10% (large)
        kline = create_test_kline(high=Decimal("105"))

        fill_price = model.apply_slippage(Decimal("100"), is_buy=True, kline=kline)
        # Would be 110, but capped at high=105
        assert fill_price == Decimal("105")

    def test_sell_capped_at_low(self):
        """Test sell slippage capped at kline low."""
        model = FixedSlippage(Decimal("0.1"))  # 10% (large)
        kline = create_test_kline(low=Decimal("95"))

        fill_price = model.apply_slippage(Decimal("100"), is_buy=False, kline=kline)
        # Would be 90, but capped at low=95
        assert fill_price == Decimal("95")

    def test_negative_slippage_raises(self):
        """Test that negative slippage raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            FixedSlippage(Decimal("-0.001"))

    def test_property_access(self):
        """Test property accessors."""
        model = FixedSlippage(Decimal("0.0005"))
        assert model.slippage_pct == Decimal("0.0005")


class TestVolatilityBasedSlippage:
    """Tests for VolatilityBasedSlippage model."""

    def test_base_slippage_only(self):
        """Test with base slippage when no ATR provided."""
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0.001"),
            atr_multiplier=Decimal("0.1"),
        )
        kline = create_test_kline()

        slippage = model.calculate_slippage(Decimal("100"), True, kline)
        # Only base: 100 * 0.001 = 0.1
        assert slippage == Decimal("0.1")

    def test_with_atr_context(self):
        """Test with ATR provided in context."""
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0.001"),
            atr_multiplier=Decimal("0.1"),
        )
        kline = create_test_kline()
        context = SlippageContext(atr=Decimal("5"))  # ATR = 5

        slippage = model.calculate_slippage(Decimal("100"), True, kline, context)
        # base: 100 * 0.001 = 0.1
        # atr: 100 * (5/100) * 0.1 = 0.5
        # total: 0.6
        assert slippage == Decimal("0.6")

    def test_atr_calculation(self):
        """Test ATR calculation from klines."""
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0"),
            atr_multiplier=Decimal("1"),
            atr_period=3,
        )

        klines = [
            create_test_kline(
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("95"),
                close=Decimal("100"),
            ),
            create_test_kline(
                open_price=Decimal("100"),
                high=Decimal("108"),
                low=Decimal("97"),
                close=Decimal("103"),
            ),
            create_test_kline(
                open_price=Decimal("103"),
                high=Decimal("110"),
                low=Decimal("100"),
                close=Decimal("107"),
            ),
        ]

        atr = model.calculate_atr(klines)
        # TR1 = max(108-97, |108-100|, |97-100|) = max(11, 8, 3) = 11
        # TR2 = max(110-100, |110-103|, |100-103|) = max(10, 7, 3) = 10
        # ATR = (11 + 10) / 2 = 10.5
        assert atr == Decimal("10.5")

    def test_with_recent_klines_context(self):
        """Test with recent klines provided for ATR calculation."""
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0"),
            atr_multiplier=Decimal("1"),
            atr_period=2,
        )

        klines = [
            create_test_kline(
                open_price=Decimal("100"),
                high=Decimal("105"),
                low=Decimal("95"),
                close=Decimal("100"),
            ),
            create_test_kline(
                open_price=Decimal("100"),
                high=Decimal("110"),
                low=Decimal("90"),
                close=Decimal("105"),
            ),
        ]
        context = SlippageContext(recent_klines=klines)

        slippage = model.calculate_slippage(Decimal("100"), True, klines[-1], context)
        # ATR calculated from klines, then applied as ratio * multiplier
        assert slippage > 0

    def test_invalid_params_raise(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="base_pct cannot be negative"):
            VolatilityBasedSlippage(base_pct=Decimal("-0.001"))

        with pytest.raises(ValueError, match="atr_multiplier cannot be negative"):
            VolatilityBasedSlippage(atr_multiplier=Decimal("-0.1"))

        with pytest.raises(ValueError, match="atr_period must be at least 1"):
            VolatilityBasedSlippage(atr_period=0)

    def test_property_access(self):
        """Test property accessors."""
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0.001"),
            atr_multiplier=Decimal("0.2"),
            atr_period=20,
        )
        assert model.base_pct == Decimal("0.001")
        assert model.atr_multiplier == Decimal("0.2")
        assert model.atr_period == 20


class TestMarketImpactSlippage:
    """Tests for MarketImpactSlippage model."""

    def test_base_slippage_only(self):
        """Test with base slippage when no context provided."""
        model = MarketImpactSlippage(
            base_pct=Decimal("0.001"),
            impact_coefficient=Decimal("0.1"),
        )
        kline = create_test_kline()

        slippage = model.calculate_slippage(Decimal("100"), True, kline)
        # Only base: 100 * 0.001 = 0.1
        assert slippage == Decimal("0.1")

    def test_with_market_impact(self):
        """Test with order size and volume context."""
        model = MarketImpactSlippage(
            base_pct=Decimal("0"),
            impact_coefficient=Decimal("0.01"),
        )
        kline = create_test_kline()
        context = SlippageContext(
            order_size=Decimal("10000"),  # 10K order
            avg_volume=Decimal("1000000"),  # 1M volume
        )

        slippage = model.calculate_slippage(Decimal("100"), True, kline, context)
        # impact = 100 * sqrt(10000/1000000) * 0.01
        # impact = 100 * sqrt(0.01) * 0.01
        # impact = 100 * 0.1 * 0.01 = 0.1
        assert slippage == Decimal("0.1")

    def test_large_order_impact(self):
        """Test impact scales with order size."""
        model = MarketImpactSlippage(
            base_pct=Decimal("0"),
            impact_coefficient=Decimal("0.1"),
        )
        kline = create_test_kline()

        # Small order
        small_context = SlippageContext(
            order_size=Decimal("1000"),
            avg_volume=Decimal("100000"),
        )
        small_slippage = model.calculate_slippage(Decimal("100"), True, kline, small_context)

        # Large order (10x)
        large_context = SlippageContext(
            order_size=Decimal("10000"),
            avg_volume=Decimal("100000"),
        )
        large_slippage = model.calculate_slippage(Decimal("100"), True, kline, large_context)

        # sqrt(10) ~ 3.16, so large should be about 3.16x small
        assert large_slippage > small_slippage
        ratio = float(large_slippage / small_slippage)
        assert 3.1 < ratio < 3.2

    def test_invalid_params_raise(self):
        """Test that invalid parameters raise ValueError."""
        with pytest.raises(ValueError, match="base_pct cannot be negative"):
            MarketImpactSlippage(base_pct=Decimal("-0.001"))

        with pytest.raises(ValueError, match="impact_coefficient cannot be negative"):
            MarketImpactSlippage(impact_coefficient=Decimal("-0.1"))

    def test_property_access(self):
        """Test property accessors."""
        model = MarketImpactSlippage(
            base_pct=Decimal("0.0005"),
            impact_coefficient=Decimal("0.2"),
        )
        assert model.base_pct == Decimal("0.0005")
        assert model.impact_coefficient == Decimal("0.2")


class TestCreateSlippageModel:
    """Tests for create_slippage_model factory function."""

    def test_create_fixed(self):
        """Test creating fixed slippage model."""
        model = create_slippage_model(
            SlippageModelType.FIXED,
            slippage_pct=Decimal("0.001"),
        )
        assert isinstance(model, FixedSlippage)
        assert model.slippage_pct == Decimal("0.001")

    def test_create_volatility(self):
        """Test creating volatility-based slippage model."""
        model = create_slippage_model(
            SlippageModelType.VOLATILITY,
            slippage_pct=Decimal("0.0005"),
            atr_multiplier=Decimal("0.2"),
            atr_period=20,
        )
        assert isinstance(model, VolatilityBasedSlippage)
        assert model.base_pct == Decimal("0.0005")
        assert model.atr_multiplier == Decimal("0.2")
        assert model.atr_period == 20

    def test_create_market_impact(self):
        """Test creating market impact slippage model."""
        model = create_slippage_model(
            SlippageModelType.MARKET_IMPACT,
            slippage_pct=Decimal("0.0001"),
            impact_coefficient=Decimal("0.15"),
        )
        assert isinstance(model, MarketImpactSlippage)
        assert model.base_pct == Decimal("0.0001")
        assert model.impact_coefficient == Decimal("0.15")

    def test_invalid_type_raises(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown slippage model type"):
            create_slippage_model("invalid")
