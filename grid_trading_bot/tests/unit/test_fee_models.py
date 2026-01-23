"""
Unit tests for fee calculation models.
"""

import pytest
from decimal import Decimal

from src.backtest.fees import (
    FeeCalculator,
    FeeModelType,
    FeeContext,
    FeeTier,
    FixedFeeCalculator,
    MakerTakerFeeCalculator,
    TieredFeeCalculator,
    create_fee_calculator,
)


class TestFixedFeeCalculator:
    """Tests for FixedFeeCalculator."""

    def test_basic_fee_calculation(self):
        """Test basic fee calculation."""
        calc = FixedFeeCalculator(Decimal("0.0004"))  # 0.04%

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
        )
        # 100 * 10 * 0.0004 = 0.4
        assert fee == Decimal("0.4")

    def test_zero_fee(self):
        """Test with zero fee rate."""
        calc = FixedFeeCalculator(Decimal("0"))

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
        )
        assert fee == Decimal("0")

    def test_context_ignored(self):
        """Test that context is ignored for fixed calculator."""
        calc = FixedFeeCalculator(Decimal("0.0004"))
        context = FeeContext(is_maker=True)

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Same fee regardless of maker/taker
        assert fee == Decimal("0.4")

    def test_negative_fee_raises(self):
        """Test that negative fee rate raises ValueError."""
        with pytest.raises(ValueError, match="cannot be negative"):
            FixedFeeCalculator(Decimal("-0.0001"))

    def test_property_access(self):
        """Test property accessors."""
        calc = FixedFeeCalculator(Decimal("0.0005"))
        assert calc.fee_rate == Decimal("0.0005")
        assert calc.base_rate == Decimal("0.0005")


class TestMakerTakerFeeCalculator:
    """Tests for MakerTakerFeeCalculator."""

    def test_taker_fee_default(self):
        """Test taker fee when no context provided."""
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.0002"),
            taker_rate=Decimal("0.0004"),
        )

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
        )
        # Taker fee: 100 * 10 * 0.0004 = 0.4
        assert fee == Decimal("0.4")

    def test_maker_fee(self):
        """Test maker fee with context."""
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.0002"),
            taker_rate=Decimal("0.0004"),
        )
        context = FeeContext(is_maker=True)

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Maker fee: 100 * 10 * 0.0002 = 0.2
        assert fee == Decimal("0.2")

    def test_taker_fee_explicit(self):
        """Test taker fee with explicit context."""
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.0002"),
            taker_rate=Decimal("0.0004"),
        )
        context = FeeContext(is_maker=False)

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Taker fee: 100 * 10 * 0.0004 = 0.4
        assert fee == Decimal("0.4")

    def test_maker_saves_half(self):
        """Test that maker typically saves significant fees."""
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.0002"),
            taker_rate=Decimal("0.0004"),
        )

        maker_fee = calc.calculate_fee(
            Decimal("100"), Decimal("10"), FeeContext(is_maker=True)
        )
        taker_fee = calc.calculate_fee(
            Decimal("100"), Decimal("10"), FeeContext(is_maker=False)
        )

        # Maker saves 50% in this case
        assert maker_fee == taker_fee / 2

    def test_negative_rates_raise(self):
        """Test that negative rates raise ValueError."""
        with pytest.raises(ValueError, match="maker_rate cannot be negative"):
            MakerTakerFeeCalculator(maker_rate=Decimal("-0.0001"))

        with pytest.raises(ValueError, match="taker_rate cannot be negative"):
            MakerTakerFeeCalculator(taker_rate=Decimal("-0.0001"))

    def test_property_access(self):
        """Test property accessors."""
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.00015"),
            taker_rate=Decimal("0.00035"),
        )
        assert calc.maker_rate == Decimal("0.00015")
        assert calc.taker_rate == Decimal("0.00035")
        assert calc.base_rate == Decimal("0.00035")  # Taker as base


class TestTieredFeeCalculator:
    """Tests for TieredFeeCalculator."""

    def test_default_tiers_tier0(self):
        """Test default tier 0 (lowest volume)."""
        calc = TieredFeeCalculator()
        context = FeeContext(cumulative_volume=Decimal("0"))

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Tier 0 taker: 100 * 10 * 0.0004 = 0.4
        assert fee == Decimal("0.4")

    def test_default_tiers_tier1(self):
        """Test tier 1 (> $1M volume)."""
        calc = TieredFeeCalculator()
        context = FeeContext(cumulative_volume=Decimal("1500000"))

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Tier 1 taker: 100 * 10 * 0.00035 = 0.35
        assert fee == Decimal("0.35")

    def test_default_tiers_highest(self):
        """Test highest tier (> $50M volume)."""
        calc = TieredFeeCalculator()
        context = FeeContext(cumulative_volume=Decimal("100000000"))

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Tier 4 taker: 100 * 10 * 0.0002 = 0.2
        assert fee == Decimal("0.2")

    def test_maker_discount_applies(self):
        """Test that maker discount applies in tiers."""
        calc = TieredFeeCalculator()
        context = FeeContext(
            cumulative_volume=Decimal("0"),
            is_maker=True,
        )

        fee = calc.calculate_fee(
            price=Decimal("100"),
            quantity=Decimal("10"),
            context=context,
        )
        # Tier 0 maker: 100 * 10 * 0.0002 = 0.2
        assert fee == Decimal("0.2")

    def test_custom_tiers(self):
        """Test with custom tiers."""
        custom_tiers = [
            FeeTier(
                min_volume=Decimal("0"),
                maker_rate=Decimal("0.001"),
                taker_rate=Decimal("0.002"),
            ),
            FeeTier(
                min_volume=Decimal("10000"),
                maker_rate=Decimal("0.0005"),
                taker_rate=Decimal("0.001"),
            ),
        ]
        calc = TieredFeeCalculator(tiers=custom_tiers)

        # Low volume
        low_fee = calc.calculate_fee(
            Decimal("100"),
            Decimal("10"),
            FeeContext(cumulative_volume=Decimal("5000")),
        )
        # Tier 0: 100 * 10 * 0.002 = 2
        assert low_fee == Decimal("2")

        # High volume
        high_fee = calc.calculate_fee(
            Decimal("100"),
            Decimal("10"),
            FeeContext(cumulative_volume=Decimal("50000")),
        )
        # Tier 1: 100 * 10 * 0.001 = 1
        assert high_fee == Decimal("1")

    def test_get_tier(self):
        """Test get_tier method."""
        calc = TieredFeeCalculator()

        tier0 = calc.get_tier(Decimal("0"))
        assert tier0.min_volume == Decimal("0")

        tier1 = calc.get_tier(Decimal("2000000"))
        assert tier1.min_volume == Decimal("1000000")

        tier4 = calc.get_tier(Decimal("100000000"))
        assert tier4.min_volume == Decimal("50000000")

    def test_invalid_tier_order_raises(self):
        """Test that unsorted tiers raise ValueError."""
        invalid_tiers = [
            FeeTier(Decimal("1000"), Decimal("0.001"), Decimal("0.002")),
            FeeTier(Decimal("500"), Decimal("0.0005"), Decimal("0.001")),  # Out of order
        ]
        with pytest.raises(ValueError, match="must be sorted"):
            TieredFeeCalculator(tiers=invalid_tiers)

    def test_property_access(self):
        """Test property accessors."""
        calc = TieredFeeCalculator()
        assert calc.base_rate == Decimal("0.0004")  # Tier 0 taker
        assert len(calc.tiers) == 5


class TestCreateFeeCalculator:
    """Tests for create_fee_calculator factory function."""

    def test_create_fixed(self):
        """Test creating fixed fee calculator."""
        calc = create_fee_calculator(
            FeeModelType.FIXED,
            fee_rate=Decimal("0.0005"),
        )
        assert isinstance(calc, FixedFeeCalculator)
        assert calc.fee_rate == Decimal("0.0005")

    def test_create_maker_taker(self):
        """Test creating maker/taker fee calculator."""
        calc = create_fee_calculator(
            FeeModelType.MAKER_TAKER,
            maker_rate=Decimal("0.00015"),
            taker_rate=Decimal("0.00035"),
        )
        assert isinstance(calc, MakerTakerFeeCalculator)
        assert calc.maker_rate == Decimal("0.00015")
        assert calc.taker_rate == Decimal("0.00035")

    def test_create_tiered(self):
        """Test creating tiered fee calculator."""
        calc = create_fee_calculator(FeeModelType.TIERED)
        assert isinstance(calc, TieredFeeCalculator)

    def test_create_tiered_custom(self):
        """Test creating tiered fee calculator with custom tiers."""
        custom_tiers = [
            FeeTier(Decimal("0"), Decimal("0.001"), Decimal("0.002")),
        ]
        calc = create_fee_calculator(FeeModelType.TIERED, tiers=custom_tiers)
        assert isinstance(calc, TieredFeeCalculator)
        assert len(calc.tiers) == 1

    def test_invalid_type_raises(self):
        """Test that invalid model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown fee model type"):
            create_fee_calculator("invalid")


class TestFeeContext:
    """Tests for FeeContext dataclass."""

    def test_default_values(self):
        """Test default context values."""
        context = FeeContext()
        assert context.is_maker is False
        assert context.cumulative_volume == Decimal("0")
        assert context.vip_level == 0

    def test_custom_values(self):
        """Test custom context values."""
        context = FeeContext(
            is_maker=True,
            cumulative_volume=Decimal("5000000"),
            vip_level=3,
        )
        assert context.is_maker is True
        assert context.cumulative_volume == Decimal("5000000")
        assert context.vip_level == 3
