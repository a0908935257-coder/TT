"""
Unit tests for Smart Grid Calculator.

Tests grid calculation, spacing, allocation, and edge cases.
"""

from decimal import Decimal

import pytest

from src.strategy.grid import (
    GridConfig,
    GridSetup,
    GridType,
    InsufficientFundError,
    InvalidPriceRangeError,
    LevelSide,
    RebuildInfo,
    RiskLevel,
    SmartGridCalculator,
    create_grid,
    create_grid_with_manual_range,
    rebuild_grid,
)


class TestSmartGridCalculator:
    """Tests for SmartGridCalculator class."""

    def test_calculate_grid_basic(self, sample_klines):
        """Test basic grid calculation."""
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
            risk_level=RiskLevel.MODERATE,
            grid_type=GridType.GEOMETRIC,
        )

        calculator = SmartGridCalculator(config=config, klines=sample_klines)
        setup = calculator.calculate()

        # Verify GridSetup structure
        assert isinstance(setup, GridSetup)
        assert setup.grid_count >= config.min_grid_count
        assert setup.grid_count <= config.max_grid_count
        assert setup.upper_price > setup.lower_price
        assert setup.current_price >= setup.lower_price
        assert setup.current_price <= setup.upper_price
        assert len(setup.levels) == setup.grid_count + 1

    def test_geometric_spacing(self):
        """Test geometric spacing is correctly calculated."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
            grid_type=GridType.GEOMETRIC,
        )

        # Verify geometric spacing
        # ratio = (55000/45000)^(1/10) ≈ 1.0203
        expected_ratio = (Decimal("55000") / Decimal("45000")) ** (Decimal("1") / Decimal("10"))

        # Check spacing between consecutive levels
        for i in range(1, len(setup.levels)):
            actual_ratio = setup.levels[i].price / setup.levels[i - 1].price
            assert abs(actual_ratio - expected_ratio) < Decimal("0.001"), (
                f"Level {i}: ratio {actual_ratio} != expected {expected_ratio}"
            )

    def test_arithmetic_spacing(self):
        """Test arithmetic spacing is correctly calculated."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
            grid_type=GridType.ARITHMETIC,
        )

        # Verify arithmetic spacing
        # step = (55000 - 45000) / 10 = 1000
        expected_step = Decimal("1000")

        # Check spacing between consecutive levels
        for i in range(1, len(setup.levels)):
            actual_step = setup.levels[i].price - setup.levels[i - 1].price
            assert abs(actual_step - expected_step) < Decimal("0.01"), (
                f"Level {i}: step {actual_step} != expected {expected_step}"
            )

    def test_pyramid_allocation(self):
        """Test pyramid fund allocation (lower prices get more funds)."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
            grid_type=GridType.GEOMETRIC,
        )

        buy_levels = setup.buy_levels

        if len(buy_levels) >= 2:
            # Lowest price level should have highest allocation
            lowest_price_level = min(buy_levels, key=lambda l: l.price)
            highest_price_level = max(buy_levels, key=lambda l: l.price)

            assert lowest_price_level.allocated_amount >= highest_price_level.allocated_amount, (
                f"Lowest level {lowest_price_level.price} allocation "
                f"{lowest_price_level.allocated_amount} should be >= "
                f"highest level {highest_price_level.price} allocation "
                f"{highest_price_level.allocated_amount}"
            )

    def test_risk_levels(self, sample_klines):
        """Test different risk levels produce different ranges."""
        results = {}

        for risk_level in [RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE]:
            config = GridConfig(
                symbol="BTCUSDT",
                total_investment=Decimal("10000"),
                risk_level=risk_level,
                grid_type=GridType.GEOMETRIC,
            )

            calculator = SmartGridCalculator(config=config, klines=sample_klines)
            setup = calculator.calculate()
            results[risk_level] = setup

        # Conservative should have narrowest range
        conservative_range = (
            results[RiskLevel.CONSERVATIVE].upper_price -
            results[RiskLevel.CONSERVATIVE].lower_price
        )
        medium_range = (
            results[RiskLevel.MODERATE].upper_price -
            results[RiskLevel.MODERATE].lower_price
        )
        aggressive_range = (
            results[RiskLevel.AGGRESSIVE].upper_price -
            results[RiskLevel.AGGRESSIVE].lower_price
        )

        assert conservative_range < medium_range < aggressive_range

    def test_manual_override(self):
        """Test manual parameter overrides."""
        manual_upper = Decimal("60000")
        manual_lower = Decimal("40000")
        manual_count = 20

        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
            manual_upper_price=manual_upper,
            manual_lower_price=manual_lower,
            manual_grid_count=manual_count,
        )

        calculator = SmartGridCalculator(
            config=config,
            current_price=Decimal("50000"),
        )
        setup = calculator.calculate()

        assert setup.upper_price == manual_upper
        assert setup.lower_price == manual_lower
        assert setup.grid_count == manual_count

    def test_insufficient_fund(self):
        """Test error when investment is too low."""
        # This should raise during config validation
        with pytest.raises(ValueError, match="Total investment"):
            GridConfig(
                symbol="BTCUSDT",
                total_investment=Decimal("10"),  # 10 < 10 * 5 = 50
                min_order_value=Decimal("10"),
                min_grid_count=5,
            )

    def test_min_profit_check(self):
        """Test minimum profit validation."""
        # Create a very narrow range that wouldn't be profitable
        with pytest.raises(InvalidPriceRangeError):
            create_grid_with_manual_range(
                symbol="BTCUSDT",
                investment=10000,
                upper_price=50010,  # Only 0.02% range
                lower_price=50000,
                grid_count=10,
                current_price=50005,
            )


class TestGridSetup:
    """Tests for GridSetup model."""

    def test_buy_sell_levels_count(self, grid_setup):
        """Test buy and sell level counting."""
        # All levels should be either buy or sell
        assert grid_setup.total_buy_levels + grid_setup.total_sell_levels == len(grid_setup.levels)

    def test_price_range_percent(self, grid_setup):
        """Test price range percentage calculation."""
        expected = (
            (grid_setup.upper_price - grid_setup.lower_price) /
            grid_setup.lower_price * Decimal("100")
        )
        assert abs(grid_setup.price_range_percent - expected) < Decimal("0.01")

    def test_average_prices(self, grid_setup):
        """Test average buy/sell price calculations."""
        # Average buy price should be below current price
        if grid_setup.total_buy_levels > 0:
            assert grid_setup.average_buy_price < grid_setup.current_price

        # Average sell price should be above current price
        if grid_setup.total_sell_levels > 0:
            assert grid_setup.average_sell_price > grid_setup.current_price

    def test_get_level_at_price(self, grid_setup):
        """Test finding level by price."""
        # Find level at first level's price
        first_level = grid_setup.levels[0]
        found = grid_setup.get_level_at_price(first_level.price)

        assert found is not None
        assert found.index == first_level.index

    def test_summary(self, grid_setup):
        """Test summary generation."""
        summary = grid_setup.summary()

        assert "symbol" in summary
        assert "current_price" in summary
        assert "grid_count" in summary
        assert summary["grid_count"] == grid_setup.grid_count


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_create_grid(self, sample_klines):
        """Test create_grid function."""
        setup = create_grid(
            symbol="BTCUSDT",
            investment=10000,
            klines=sample_klines,
            risk_level=RiskLevel.MODERATE,
            grid_type=GridType.GEOMETRIC,
        )

        assert isinstance(setup, GridSetup)
        assert setup.config.symbol == "BTCUSDT"
        assert setup.config.total_investment == Decimal("10000")

    def test_create_grid_with_manual_range(self):
        """Test create_grid_with_manual_range function."""
        setup = create_grid_with_manual_range(
            symbol="ETHUSDT",
            investment=5000,
            upper_price=2500,
            lower_price=2000,
            grid_count=10,
            current_price=2250,
            grid_type=GridType.ARITHMETIC,
        )

        assert isinstance(setup, GridSetup)
        assert setup.config.symbol == "ETHUSDT"
        assert setup.upper_price == Decimal("2500")
        assert setup.lower_price == Decimal("2000")
        assert setup.grid_count == 10


class TestGridLevelSides:
    """Tests for level side assignment."""

    def test_sides_correct_based_on_current_price(self):
        """Test that sides are assigned correctly based on current price."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
        )

        for level in setup.levels:
            if level.price < setup.current_price:
                assert level.side == LevelSide.BUY, (
                    f"Level at {level.price} should be BUY (below {setup.current_price})"
                )
            else:
                assert level.side == LevelSide.SELL, (
                    f"Level at {level.price} should be SELL (above {setup.current_price})"
                )

    def test_levels_sorted_by_price(self):
        """Test that levels are sorted by price ascending."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
        )

        prices = [level.price for level in setup.levels]
        assert prices == sorted(prices), "Levels should be sorted by price ascending"

    def test_level_indices_sequential(self):
        """Test that level indices are sequential."""
        setup = create_grid_with_manual_range(
            symbol="BTCUSDT",
            investment=10000,
            upper_price=55000,
            lower_price=45000,
            grid_count=10,
            current_price=50000,
        )

        indices = [level.index for level in setup.levels]
        expected = list(range(len(setup.levels)))
        assert indices == expected, "Level indices should be sequential"


class TestCenterPriceCalculation:
    """Tests for calculate() with center_price parameter."""

    def test_calculate_with_center_price(self, sample_klines):
        """Test calculation with explicit center_price."""
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
            risk_level=RiskLevel.MODERATE,
            grid_type=GridType.GEOMETRIC,
        )

        calculator = SmartGridCalculator(config=config, klines=sample_klines)

        # Calculate with new center price
        new_center = Decimal("55000")
        setup = calculator.calculate(center_price=new_center)

        # Current price should be the center_price
        assert setup.current_price == new_center
        # Range should be centered around new_center
        assert setup.lower_price < new_center < setup.upper_price

    def test_center_price_recalculates_range(self, sample_klines):
        """Test that center_price triggers range recalculation."""
        config = GridConfig(
            symbol="BTCUSDT",
            total_investment=Decimal("10000"),
            risk_level=RiskLevel.MODERATE,
            grid_type=GridType.GEOMETRIC,
        )

        calculator = SmartGridCalculator(config=config, klines=sample_klines)

        # Calculate without center_price
        setup_normal = calculator.calculate()

        # Calculate with different center_price
        calculator2 = SmartGridCalculator(config=config, klines=sample_klines)
        setup_centered = calculator2.calculate(center_price=Decimal("60000"))

        # Ranges should be different
        assert setup_centered.current_price != setup_normal.current_price
        assert setup_centered.upper_price != setup_normal.upper_price
        assert setup_centered.lower_price != setup_normal.lower_price


class TestRebuildGrid:
    """Tests for rebuild_grid function."""

    def test_rebuild_grid_basic(self, sample_klines):
        """Test basic grid rebuild."""
        # Create initial setup
        initial_setup = create_grid(
            symbol="BTCUSDT",
            investment=10000,
            klines=sample_klines,
            risk_level=RiskLevel.MODERATE,
        )

        # Rebuild with new center price
        new_center = Decimal("57500")
        rebuilt_setup = rebuild_grid(
            old_setup=initial_setup,
            klines=sample_klines,
            new_center_price=new_center,
            reason="upper_breakout",
        )

        # Version should increment
        assert rebuilt_setup.version == initial_setup.version + 1

        # Should have rebuild_info
        assert rebuilt_setup.rebuild_info is not None
        assert isinstance(rebuilt_setup.rebuild_info, RebuildInfo)

    def test_rebuild_grid_info_populated(self, sample_klines):
        """Test that rebuild_info is correctly populated."""
        initial_setup = create_grid(
            symbol="BTCUSDT",
            investment=10000,
            klines=sample_klines,
        )

        new_center = Decimal("60000")
        rebuilt_setup = rebuild_grid(
            old_setup=initial_setup,
            klines=sample_klines,
            new_center_price=new_center,
            reason="manual_rebuild",
        )

        info = rebuilt_setup.rebuild_info
        assert info.old_upper == initial_setup.upper_price
        assert info.old_lower == initial_setup.lower_price
        assert info.new_center == new_center
        assert info.reason == "manual_rebuild"
        assert info.rebuilt_at is not None

    def test_rebuild_grid_preserves_config(self, sample_klines):
        """Test that rebuild preserves original config."""
        initial_setup = create_grid(
            symbol="ETHUSDT",
            investment=5000,
            klines=sample_klines,
            risk_level=RiskLevel.AGGRESSIVE,
            grid_type=GridType.ARITHMETIC,
        )

        rebuilt_setup = rebuild_grid(
            old_setup=initial_setup,
            klines=sample_klines,
            new_center_price=Decimal("55000"),
        )

        # Config should be preserved
        assert rebuilt_setup.config.symbol == initial_setup.config.symbol
        assert rebuilt_setup.config.total_investment == initial_setup.config.total_investment
        assert rebuilt_setup.config.grid_type == initial_setup.config.grid_type

    def test_rebuild_grid_multiple_times(self, sample_klines):
        """Test multiple sequential rebuilds."""
        setup_v1 = create_grid(
            symbol="BTCUSDT",
            investment=10000,
            klines=sample_klines,
        )
        assert setup_v1.version == 1

        setup_v2 = rebuild_grid(
            old_setup=setup_v1,
            klines=sample_klines,
            new_center_price=Decimal("55000"),
        )
        assert setup_v2.version == 2

        setup_v3 = rebuild_grid(
            old_setup=setup_v2,
            klines=sample_klines,
            new_center_price=Decimal("60000"),
        )
        assert setup_v3.version == 3

        # v3's rebuild_info should reference v2's boundaries
        assert setup_v3.rebuild_info.old_upper == setup_v2.upper_price
        assert setup_v3.rebuild_info.old_lower == setup_v2.lower_price


class TestRebuildInfo:
    """Tests for RebuildInfo dataclass."""

    def test_rebuild_info_creation(self):
        """Test RebuildInfo creation."""
        info = RebuildInfo(
            old_upper=Decimal("52000"),
            old_lower=Decimal("48000"),
            new_center=Decimal("55000"),
            reason="price_breakout",
        )

        assert info.old_upper == Decimal("52000")
        assert info.old_lower == Decimal("48000")
        assert info.new_center == Decimal("55000")
        assert info.reason == "price_breakout"
        assert info.rebuilt_at is not None

    def test_rebuild_info_type_conversion(self):
        """Test RebuildInfo converts types to Decimal."""
        info = RebuildInfo(
            old_upper=52000,  # int
            old_lower="48000",  # str
            new_center=55000.0,  # float
        )

        assert isinstance(info.old_upper, Decimal)
        assert isinstance(info.old_lower, Decimal)
        assert isinstance(info.new_center, Decimal)
