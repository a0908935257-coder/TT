"""Tests for business alert rules module."""

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.monitoring.alerts import AlertManager, AlertSeverity
from src.monitoring.business_alerts import (
    BusinessAlertMonitor,
    BusinessAlertRules,
    BusinessMetricsTracker,
    OrderStats,
    PnLStats,
)
from src.monitoring.system_metrics import SystemThresholds


class TestOrderStats:
    """Tests for OrderStats."""

    def test_success_rate_no_orders(self):
        """Test success rate with no orders."""
        stats = OrderStats()
        assert stats.success_rate == 1.0

    def test_success_rate_all_successful(self):
        """Test success rate with all successful orders."""
        stats = OrderStats(total_orders=10, successful_orders=10)
        assert stats.success_rate == 1.0

    def test_success_rate_partial(self):
        """Test success rate with partial success."""
        stats = OrderStats(total_orders=10, successful_orders=8, rejected_orders=2)
        assert stats.success_rate == 0.8

    def test_rejection_rate_no_orders(self):
        """Test rejection rate with no orders."""
        stats = OrderStats()
        assert stats.rejection_rate == 0.0

    def test_rejection_rate_with_rejections(self):
        """Test rejection rate with rejections."""
        stats = OrderStats(total_orders=100, rejected_orders=5)
        assert stats.rejection_rate == 0.05

    def test_avg_latency_no_orders(self):
        """Test average latency with no orders."""
        stats = OrderStats()
        assert stats.avg_latency_ms == 0.0

    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        stats = OrderStats(
            total_orders=10, successful_orders=5, total_latency_ms=500.0
        )
        assert stats.avg_latency_ms == 100.0


class TestPnLStats:
    """Tests for PnLStats."""

    def test_total_pnl(self):
        """Test total P&L calculation."""
        stats = PnLStats(
            realized_pnl=Decimal("100"),
            unrealized_pnl=Decimal("50"),
        )
        assert stats.total_pnl == Decimal("150")

    def test_drawdown_no_peak(self):
        """Test drawdown with no peak."""
        stats = PnLStats()
        assert stats.drawdown == Decimal("0")
        assert stats.drawdown_percent == 0.0

    def test_drawdown_calculation(self):
        """Test drawdown calculation."""
        stats = PnLStats(
            peak_value=Decimal("10000"),
            current_value=Decimal("9000"),
        )
        assert stats.drawdown == Decimal("1000")
        assert stats.drawdown_percent == 0.1

    def test_daily_loss_percent_no_limit(self):
        """Test daily loss percent with no limit."""
        stats = PnLStats(daily_pnl=Decimal("-100"))
        assert stats.daily_loss_percent == 0.0

    def test_daily_loss_percent_positive(self):
        """Test daily loss percent with positive P&L."""
        stats = PnLStats(daily_pnl=Decimal("100"), daily_limit=Decimal("1000"))
        assert stats.daily_loss_percent == 0.0

    def test_daily_loss_percent_calculation(self):
        """Test daily loss percent calculation."""
        stats = PnLStats(daily_pnl=Decimal("-500"), daily_limit=Decimal("1000"))
        assert stats.daily_loss_percent == 0.5


class TestBusinessMetricsTracker:
    """Tests for BusinessMetricsTracker."""

    def test_record_order_success(self):
        """Test recording successful orders."""
        tracker = BusinessMetricsTracker()

        tracker.record_order(success=True, latency_ms=100.0)
        tracker.record_order(success=True, latency_ms=150.0)

        stats = tracker.get_order_stats()
        assert stats.total_orders == 2
        assert stats.successful_orders == 2
        assert stats.total_latency_ms == 250.0

    def test_record_order_rejection(self):
        """Test recording rejected orders."""
        tracker = BusinessMetricsTracker()

        tracker.record_order(success=False, rejected=True)

        stats = tracker.get_order_stats()
        assert stats.total_orders == 1
        assert stats.rejected_orders == 1

    def test_record_order_cancelled(self):
        """Test recording cancelled orders."""
        tracker = BusinessMetricsTracker()

        tracker.record_order(success=False, cancelled=True)

        stats = tracker.get_order_stats()
        assert stats.cancelled_orders == 1

    def test_update_pnl(self):
        """Test updating P&L."""
        tracker = BusinessMetricsTracker()

        tracker.update_pnl(
            realized=Decimal("1000"),
            unrealized=Decimal("500"),
            current_value=Decimal("10500"),
            daily_pnl=Decimal("200"),
            daily_limit=Decimal("5000"),
        )

        stats = tracker.get_pnl_stats()
        assert stats.realized_pnl == Decimal("1000")
        assert stats.unrealized_pnl == Decimal("500")
        assert stats.current_value == Decimal("10500")
        assert stats.peak_value == Decimal("10500")

    def test_pnl_peak_tracking(self):
        """Test peak value tracking."""
        tracker = BusinessMetricsTracker()

        tracker.update_pnl(
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            current_value=Decimal("10000"),
            daily_pnl=Decimal("0"),
            daily_limit=Decimal("1000"),
        )

        tracker.update_pnl(
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            current_value=Decimal("11000"),
            daily_pnl=Decimal("0"),
            daily_limit=Decimal("1000"),
        )

        tracker.update_pnl(
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            current_value=Decimal("10500"),
            daily_pnl=Decimal("0"),
            daily_limit=Decimal("1000"),
        )

        stats = tracker.get_pnl_stats()
        assert stats.peak_value == Decimal("11000")
        assert stats.current_value == Decimal("10500")

    def test_record_risk_block(self):
        """Test recording risk blocks."""
        tracker = BusinessMetricsTracker()

        tracker.record_risk_block()
        tracker.record_risk_block()
        tracker.record_risk_block()

        assert tracker.get_risk_blocks_count(300) == 3

    def test_hourly_order_stats(self):
        """Test hourly order statistics aggregation."""
        tracker = BusinessMetricsTracker()

        tracker.record_order(success=True, latency_ms=100.0)
        tracker.record_order(success=True, latency_ms=100.0)
        tracker.record_order(success=False, rejected=True)

        stats = tracker.get_hourly_order_stats()
        assert stats.total_orders == 3
        assert stats.successful_orders == 2
        assert stats.rejected_orders == 1


class TestBusinessAlertRules:
    """Tests for BusinessAlertRules."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        alert_manager = AlertManager()
        tracker = BusinessMetricsTracker()
        thresholds = SystemThresholds(
            order_success_rate_warning=0.95,
            order_rejection_rate_warning=0.05,
            daily_loss_warning_percent=0.80,
            max_drawdown_warning_percent=0.10,
            max_drawdown_critical_percent=0.20,
        )
        rules = BusinessAlertRules(alert_manager, tracker, thresholds)
        return alert_manager, tracker, thresholds, rules

    def test_register_all_rules(self, setup):
        """Test registering all rules."""
        alert_manager, tracker, thresholds, rules = setup

        rules.register_all_rules()

        # Verify rules are registered
        assert alert_manager._rules
        # Should have multiple rules
        assert len(alert_manager._rules) >= 8

    def test_rules_registered_once(self, setup):
        """Test rules only registered once."""
        alert_manager, tracker, thresholds, rules = setup

        rules.register_all_rules()
        initial_count = len(alert_manager._rules)

        rules.register_all_rules()
        assert len(alert_manager._rules) == initial_count

    @pytest.mark.asyncio
    async def test_evaluate_order_success_rate(self, setup):
        """Test order success rate evaluation."""
        alert_manager, tracker, thresholds, rules = setup
        rules.register_all_rules()

        # Record orders with low success rate
        for _ in range(10):
            tracker.record_order(success=True)
        for _ in range(10):
            tracker.record_order(success=False, rejected=True)

        # 50% success rate should trigger alert
        with patch.object(alert_manager, "fire", new_callable=AsyncMock) as mock_fire:
            mock_fire.return_value = MagicMock(alert_id="test-123")
            fired = await rules.evaluate_all()

            # Should have fired at least one alert
            assert mock_fire.called

    @pytest.mark.asyncio
    async def test_evaluate_daily_loss(self, setup):
        """Test daily loss evaluation."""
        alert_manager, tracker, thresholds, rules = setup
        rules.register_all_rules()

        # Set daily loss at 90% of limit
        tracker.update_pnl(
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            current_value=Decimal("9100"),
            daily_pnl=Decimal("-900"),
            daily_limit=Decimal("1000"),
        )

        with patch.object(alert_manager, "fire", new_callable=AsyncMock) as mock_fire:
            mock_fire.return_value = MagicMock(alert_id="test-123")
            await rules.evaluate_all()

            # Should trigger daily loss warning
            assert mock_fire.called

    @pytest.mark.asyncio
    async def test_evaluate_drawdown(self, setup):
        """Test drawdown evaluation."""
        alert_manager, tracker, thresholds, rules = setup
        rules.register_all_rules()

        # Set 15% drawdown
        tracker.update_pnl(
            realized=Decimal("0"),
            unrealized=Decimal("0"),
            current_value=Decimal("8500"),
            daily_pnl=Decimal("-1500"),
            daily_limit=Decimal("10000"),
        )
        # Manually set peak value higher
        tracker._current_pnl.peak_value = Decimal("10000")

        with patch.object(alert_manager, "fire", new_callable=AsyncMock) as mock_fire:
            mock_fire.return_value = MagicMock(alert_id="test-123")
            await rules.evaluate_all()

            # Should trigger drawdown warning
            assert mock_fire.called

    @pytest.mark.asyncio
    async def test_evaluate_risk_blocks(self, setup):
        """Test risk blocks evaluation."""
        alert_manager, tracker, thresholds, rules = setup
        rules.register_all_rules()

        # Record multiple risk blocks
        for _ in range(5):
            tracker.record_risk_block()

        with patch.object(alert_manager, "fire", new_callable=AsyncMock) as mock_fire:
            mock_fire.return_value = MagicMock(alert_id="test-123")
            await rules.evaluate_all()

            # Should trigger risk blocks warning
            assert mock_fire.called


class TestBusinessAlertMonitor:
    """Tests for BusinessAlertMonitor."""

    @pytest.fixture
    def setup(self):
        """Set up test fixtures."""
        alert_manager = AlertManager()
        tracker = BusinessMetricsTracker()
        rules = BusinessAlertRules(alert_manager, tracker)
        monitor = BusinessAlertMonitor(rules, interval_seconds=0.1)
        return monitor, rules, tracker

    @pytest.mark.asyncio
    async def test_start_stop(self, setup):
        """Test starting and stopping the monitor."""
        monitor, rules, tracker = setup

        await monitor.start()
        assert monitor._running

        await asyncio.sleep(0.2)

        await monitor.stop()
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_monitor_evaluates_rules(self, setup):
        """Test that monitor evaluates rules periodically."""
        monitor, rules, tracker = setup

        with patch.object(
            rules, "evaluate_all", new_callable=AsyncMock
        ) as mock_evaluate:
            mock_evaluate.return_value = []

            await monitor.start()
            await asyncio.sleep(0.25)
            await monitor.stop()

            # Should have been called at least once
            assert mock_evaluate.called
