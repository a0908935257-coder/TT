"""
Tests for Drawdown Calculator.

Tests drawdown tracking, peak detection, and alert generation.
"""

from datetime import timedelta
from decimal import Decimal

import pytest

from src.risk.drawdown_calculator import DrawdownCalculator
from src.risk.models import (
    RiskAction,
    RiskConfig,
    RiskLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_config():
    """Create a risk configuration."""
    return RiskConfig(
        total_capital=Decimal("100000"),
        max_drawdown_warning=Decimal("0.15"),  # 15%
        max_drawdown_danger=Decimal("0.25"),  # 25%
    )


@pytest.fixture
def calculator(risk_config):
    """Create a drawdown calculator."""
    return DrawdownCalculator(risk_config)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestDrawdownCalculatorInit:
    """Tests for DrawdownCalculator initialization."""

    def test_init_with_config(self, risk_config):
        """Test initialization with config."""
        calculator = DrawdownCalculator(risk_config)

        assert calculator.config == risk_config
        assert calculator.peak_value == Decimal("0")
        assert calculator.peak_time is None
        assert calculator.current_drawdown is None
        assert calculator.max_drawdown is None

    def test_init_empty_history(self, calculator):
        """Test that history starts empty."""
        assert len(calculator.drawdown_history) == 0


# =============================================================================
# Peak Tracking Tests
# =============================================================================


class TestPeakTracking:
    """Tests for peak value tracking."""

    def test_first_update_sets_peak(self, calculator):
        """Test that first update sets peak."""
        calculator.update(Decimal("100000"))

        assert calculator.peak_value == Decimal("100000")
        assert calculator.peak_time is not None

    def test_higher_value_updates_peak(self, calculator):
        """Test that higher value updates peak."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("110000"))

        assert calculator.peak_value == Decimal("110000")

    def test_lower_value_does_not_update_peak(self, calculator):
        """Test that lower value doesn't update peak."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))

        assert calculator.peak_value == Decimal("100000")

    def test_set_initial_peak(self, calculator):
        """Test setting initial peak."""
        calculator.set_initial_peak(Decimal("100000"))

        assert calculator.peak_value == Decimal("100000")
        assert calculator.peak_time is not None


# =============================================================================
# Drawdown Calculation Tests
# =============================================================================


class TestDrawdownCalculation:
    """Tests for drawdown calculation."""

    def test_no_drawdown_at_peak(self, calculator):
        """Test no drawdown at peak."""
        dd = calculator.update(Decimal("100000"))

        assert dd.drawdown_amount == Decimal("0")
        assert dd.drawdown_pct == Decimal("0")

    def test_drawdown_calculation(self, calculator):
        """Test drawdown calculation."""
        calculator.update(Decimal("100000"))
        dd = calculator.update(Decimal("90000"))

        assert dd.drawdown_amount == Decimal("10000")
        assert dd.drawdown_pct == Decimal("0.1")  # 10%
        assert dd.peak_value == Decimal("100000")
        assert dd.current_value == Decimal("90000")

    def test_drawdown_deepens(self, calculator):
        """Test that drawdown deepens correctly."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))  # 10%
        dd = calculator.update(Decimal("80000"))  # 20%

        assert dd.drawdown_pct == Decimal("0.2")
        assert dd.drawdown_amount == Decimal("20000")

    def test_partial_recovery(self, calculator):
        """Test partial recovery reduces drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("80000"))  # 20% drawdown
        dd = calculator.update(Decimal("90000"))  # Partial recovery

        assert dd.drawdown_pct == Decimal("0.1")  # 10%
        assert dd.current_value == Decimal("90000")

    def test_full_recovery_creates_new_peak(self, calculator):
        """Test full recovery creates new peak."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("80000"))  # 20% drawdown
        dd = calculator.update(Decimal("105000"))  # New peak

        assert dd.drawdown_pct == Decimal("0")
        assert calculator.peak_value == Decimal("105000")

    def test_drawdown_duration(self, calculator):
        """Test that drawdown tracks duration."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))

        dd = calculator.get_current_drawdown()
        assert dd.duration >= timedelta(0)


# =============================================================================
# Max Drawdown Tests
# =============================================================================


class TestMaxDrawdown:
    """Tests for maximum drawdown tracking."""

    def test_max_drawdown_initially_none(self, calculator):
        """Test max drawdown is None initially."""
        assert calculator.max_drawdown is None

    def test_max_drawdown_set_on_first_drawdown(self, calculator):
        """Test max drawdown is set on first drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))

        assert calculator.max_drawdown is not None
        assert calculator.max_drawdown.drawdown_pct == Decimal("0.1")

    def test_max_drawdown_updates_when_deeper(self, calculator):
        """Test max drawdown updates when deeper."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))  # 10%
        calculator.update(Decimal("80000"))  # 20%

        assert calculator.max_drawdown.drawdown_pct == Decimal("0.2")

    def test_max_drawdown_preserved_after_recovery(self, calculator):
        """Test max drawdown is preserved after recovery."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("80000"))  # 20%
        calculator.update(Decimal("110000"))  # New peak
        calculator.update(Decimal("100000"))  # 9% from new peak

        # Max drawdown should still be 20%
        assert calculator.max_drawdown.drawdown_pct == Decimal("0.2")


# =============================================================================
# Drawdown History Tests
# =============================================================================


class TestDrawdownHistory:
    """Tests for drawdown history tracking."""

    def test_history_empty_initially(self, calculator):
        """Test history starts empty."""
        assert len(calculator.get_drawdown_history()) == 0

    def test_history_records_completed_drawdown(self, calculator):
        """Test history records completed drawdown period."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))  # In drawdown
        calculator.update(Decimal("110000"))  # New peak - ends drawdown

        history = calculator.get_drawdown_history()
        assert len(history) == 1
        assert history[0].drawdown_pct == Decimal("0.1")

    def test_multiple_drawdown_periods(self, calculator):
        """Test multiple drawdown periods in history."""
        # First drawdown period
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))  # 10%
        calculator.update(Decimal("110000"))  # Recovery

        # Second drawdown period
        calculator.update(Decimal("100000"))  # ~9%
        calculator.update(Decimal("120000"))  # Recovery

        history = calculator.get_drawdown_history()
        assert len(history) == 2

    def test_history_copy_is_returned(self, calculator):
        """Test that history copy is returned."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))
        calculator.update(Decimal("110000"))

        history = calculator.get_drawdown_history()
        history.clear()

        # Original should be unaffected
        assert len(calculator.get_drawdown_history()) == 1


# =============================================================================
# Alert Tests
# =============================================================================


class TestDrawdownAlerts:
    """Tests for alert generation."""

    def test_no_alerts_below_threshold(self, calculator):
        """Test no alerts when below threshold."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))  # 10% - below 15% warning

        alerts = calculator.check_alerts()
        assert len(alerts) == 0

    def test_warning_alert_at_threshold(self, calculator):
        """Test warning alert at threshold."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("85000"))  # 15% - at warning threshold

        alerts = calculator.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.WARNING
        assert alerts[0].metric == "max_drawdown"
        assert alerts[0].action_taken == RiskAction.NOTIFY

    def test_danger_alert_at_threshold(self, calculator):
        """Test danger alert at threshold."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("75000"))  # 25% - at danger threshold

        alerts = calculator.check_alerts()
        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.DANGER
        assert alerts[0].action_taken == RiskAction.PAUSE_ALL_BOTS

    def test_danger_overrides_warning(self, calculator):
        """Test danger alert overrides warning."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("70000"))  # 30% - above danger

        alerts = calculator.check_alerts()

        # Should only have danger, not warning
        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.DANGER

    def test_no_alerts_without_updates(self, calculator):
        """Test no alerts when no updates."""
        alerts = calculator.check_alerts()
        assert len(alerts) == 0


# =============================================================================
# Statistics Tests
# =============================================================================


class TestDrawdownStatistics:
    """Tests for drawdown statistics."""

    def test_statistics_empty_initially(self, calculator):
        """Test statistics with no data."""
        stats = calculator.get_statistics()

        assert stats["current_drawdown_pct"] == Decimal("0")
        assert stats["max_drawdown_pct"] == Decimal("0")
        assert stats["drawdown_count"] == 0
        assert stats["average_drawdown_pct"] == Decimal("0")

    def test_statistics_with_drawdown(self, calculator):
        """Test statistics with drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("85000"))

        stats = calculator.get_statistics()

        assert stats["current_drawdown_pct"] == Decimal("0.15")
        assert stats["max_drawdown_pct"] == Decimal("0.15")
        assert stats["peak_value"] == Decimal("100000")

    def test_average_drawdown(self, calculator):
        """Test average drawdown calculation."""
        # First drawdown: 10%
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))
        calculator.update(Decimal("110000"))  # End first

        # Second drawdown: 20%
        calculator.update(Decimal("88000"))  # 20% of 110000
        calculator.update(Decimal("120000"))  # End second

        stats = calculator.get_statistics()

        # Average of 10% and 20%
        assert stats["drawdown_count"] == 2
        assert stats["average_drawdown_pct"] == Decimal("0.15")


# =============================================================================
# Utility Tests
# =============================================================================


class TestDrawdownUtility:
    """Tests for utility methods."""

    def test_is_in_drawdown_false_at_peak(self, calculator):
        """Test is_in_drawdown false at peak."""
        calculator.update(Decimal("100000"))

        assert calculator.is_in_drawdown() is False

    def test_is_in_drawdown_true_below_peak(self, calculator):
        """Test is_in_drawdown true below peak."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))

        assert calculator.is_in_drawdown() is True

    def test_recovery_needed_no_drawdown(self, calculator):
        """Test recovery needed with no drawdown."""
        calculator.update(Decimal("100000"))

        assert calculator.get_recovery_needed() == Decimal("0")

    def test_recovery_needed_10_percent(self, calculator):
        """Test recovery needed for 10% drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("90000"))

        # 10% loss needs 11.1% gain to recover
        recovery = calculator.get_recovery_needed()
        expected = Decimal("0.1") / Decimal("0.9")  # ~0.1111

        # Compare with tolerance
        assert abs(recovery - expected) < Decimal("0.0001")

    def test_recovery_needed_50_percent(self, calculator):
        """Test recovery needed for 50% drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("50000"))

        # 50% loss needs 100% gain to recover
        recovery = calculator.get_recovery_needed()
        assert recovery == Decimal("1")


# =============================================================================
# Reset Tests
# =============================================================================


class TestDrawdownReset:
    """Tests for reset functionality."""

    def test_reset_clears_all_data(self, calculator):
        """Test reset clears all data."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("80000"))
        calculator.update(Decimal("110000"))

        calculator.reset()

        assert calculator.peak_value == Decimal("0")
        assert calculator.peak_time is None
        assert calculator.current_drawdown is None
        assert calculator.max_drawdown is None
        assert len(calculator.get_drawdown_history()) == 0

    def test_reset_allows_fresh_start(self, calculator):
        """Test reset allows fresh start."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("80000"))  # 20% drawdown

        calculator.reset()

        # Start fresh
        calculator.update(Decimal("50000"))
        calculator.update(Decimal("45000"))  # 10% drawdown

        assert calculator.peak_value == Decimal("50000")
        assert calculator.max_drawdown.drawdown_pct == Decimal("0.1")


# =============================================================================
# Edge Cases
# =============================================================================


class TestDrawdownEdgeCases:
    """Tests for edge cases."""

    def test_zero_peak_value(self, calculator):
        """Test handling of zero peak value."""
        calculator.update(Decimal("0"))
        calculator.update(Decimal("0"))

        # Should not crash
        dd = calculator.get_current_drawdown()
        assert dd is not None

    def test_same_value_updates(self, calculator):
        """Test multiple updates with same value."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("100000"))

        assert calculator.peak_value == Decimal("100000")
        assert calculator.is_in_drawdown() is False

    def test_very_small_drawdown(self, calculator):
        """Test very small drawdown."""
        calculator.update(Decimal("100000"))
        calculator.update(Decimal("99999.99"))

        dd = calculator.get_current_drawdown()
        assert dd.drawdown_pct > Decimal("0")
        assert dd.drawdown_pct < Decimal("0.0001")

    def test_large_values(self, calculator):
        """Test with large values."""
        calculator.update(Decimal("1000000000"))  # 1 billion
        calculator.update(Decimal("900000000"))  # 10% drawdown

        dd = calculator.get_current_drawdown()
        assert dd.drawdown_pct == Decimal("0.1")
