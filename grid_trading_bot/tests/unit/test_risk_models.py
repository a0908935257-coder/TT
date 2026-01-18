"""
Tests for Risk Management Models.

Verifies all risk models can be created correctly with proper types and defaults.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal

import pytest

from src.risk.models import (
    CapitalSnapshot,
    CircuitBreakerState,
    DailyPnL,
    DrawdownInfo,
    GlobalRiskStatus,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)


# =============================================================================
# RiskLevel Tests
# =============================================================================


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_risk_level_values(self):
        """Test risk level values are ordered correctly."""
        assert RiskLevel.NORMAL.value == 1
        assert RiskLevel.WARNING.value == 2
        assert RiskLevel.RISK.value == 3
        assert RiskLevel.DANGER.value == 4
        assert RiskLevel.CIRCUIT_BREAK.value == 5

    def test_risk_level_ordering(self):
        """Test risk levels can be compared."""
        assert RiskLevel.NORMAL.value < RiskLevel.WARNING.value
        assert RiskLevel.WARNING.value < RiskLevel.RISK.value
        assert RiskLevel.RISK.value < RiskLevel.DANGER.value
        assert RiskLevel.DANGER.value < RiskLevel.CIRCUIT_BREAK.value


# =============================================================================
# RiskAction Tests
# =============================================================================


class TestRiskAction:
    """Tests for RiskAction enum."""

    def test_risk_action_values(self):
        """Test risk action values."""
        assert RiskAction.NONE.value == "none"
        assert RiskAction.NOTIFY.value == "notify"
        assert RiskAction.PAUSE_NEW_ORDERS.value == "pause_new"
        assert RiskAction.PAUSE_ALL_BOTS.value == "pause_all"
        assert RiskAction.EMERGENCY_STOP.value == "emergency"


# =============================================================================
# RiskConfig Tests
# =============================================================================


class TestRiskConfig:
    """Tests for RiskConfig model."""

    def test_create_with_defaults(self):
        """Test creating config with default values."""
        config = RiskConfig(total_capital=Decimal("100000"))

        assert config.total_capital == Decimal("100000")
        assert config.warning_loss_pct == Decimal("0.10")
        assert config.danger_loss_pct == Decimal("0.20")
        assert config.daily_loss_warning == Decimal("0.05")
        assert config.daily_loss_danger == Decimal("0.10")
        assert config.max_drawdown_warning == Decimal("0.15")
        assert config.max_drawdown_danger == Decimal("0.25")
        assert config.consecutive_loss_warning == 5
        assert config.consecutive_loss_danger == 10
        assert config.circuit_breaker_cooldown == 3600
        assert config.auto_resume_enabled is False

    def test_create_with_custom_values(self):
        """Test creating config with custom values."""
        config = RiskConfig(
            total_capital=Decimal("50000"),
            warning_loss_pct=Decimal("0.05"),
            danger_loss_pct=Decimal("0.15"),
            daily_loss_warning=Decimal("0.03"),
            daily_loss_danger=Decimal("0.08"),
            max_drawdown_warning=Decimal("0.10"),
            max_drawdown_danger=Decimal("0.20"),
            consecutive_loss_warning=3,
            consecutive_loss_danger=7,
            circuit_breaker_cooldown=1800,
            auto_resume_enabled=True,
        )

        assert config.total_capital == Decimal("50000")
        assert config.warning_loss_pct == Decimal("0.05")
        assert config.danger_loss_pct == Decimal("0.15")
        assert config.circuit_breaker_cooldown == 1800
        assert config.auto_resume_enabled is True


# =============================================================================
# CapitalSnapshot Tests
# =============================================================================


class TestCapitalSnapshot:
    """Tests for CapitalSnapshot model."""

    def test_create_snapshot(self):
        """Test creating a capital snapshot."""
        now = datetime.now()
        snapshot = CapitalSnapshot(
            timestamp=now,
            total_capital=Decimal("100000"),
            available_balance=Decimal("80000"),
            position_value=Decimal("20000"),
            unrealized_pnl=Decimal("500"),
            realized_pnl=Decimal("1000"),
        )

        assert snapshot.timestamp == now
        assert snapshot.total_capital == Decimal("100000")
        assert snapshot.available_balance == Decimal("80000")
        assert snapshot.position_value == Decimal("20000")
        assert snapshot.unrealized_pnl == Decimal("500")
        assert snapshot.realized_pnl == Decimal("1000")


# =============================================================================
# DrawdownInfo Tests
# =============================================================================


class TestDrawdownInfo:
    """Tests for DrawdownInfo model."""

    def test_create_drawdown_info(self):
        """Test creating drawdown info."""
        peak_time = datetime.now() - timedelta(hours=2)
        info = DrawdownInfo(
            peak_value=Decimal("100000"),
            current_value=Decimal("90000"),
            drawdown_amount=Decimal("10000"),
            drawdown_pct=Decimal("0.10"),
            peak_time=peak_time,
            duration=timedelta(hours=2),
        )

        assert info.peak_value == Decimal("100000")
        assert info.current_value == Decimal("90000")
        assert info.drawdown_amount == Decimal("10000")
        assert info.drawdown_pct == Decimal("0.10")

    def test_calculate_drawdown(self):
        """Test calculating drawdown info from values."""
        peak_time = datetime.now() - timedelta(hours=1)
        info = DrawdownInfo.calculate(
            peak_value=Decimal("100000"),
            current_value=Decimal("85000"),
            peak_time=peak_time,
        )

        assert info.peak_value == Decimal("100000")
        assert info.current_value == Decimal("85000")
        assert info.drawdown_amount == Decimal("15000")
        assert info.drawdown_pct == Decimal("0.15")
        assert info.duration >= timedelta(hours=1)

    def test_calculate_zero_peak(self):
        """Test calculating drawdown with zero peak."""
        info = DrawdownInfo.calculate(
            peak_value=Decimal("0"),
            current_value=Decimal("0"),
            peak_time=datetime.now(),
        )

        assert info.drawdown_pct == Decimal("0")


# =============================================================================
# DailyPnL Tests
# =============================================================================


class TestDailyPnL:
    """Tests for DailyPnL model."""

    def test_create_daily_pnl(self):
        """Test creating daily P&L record."""
        today = date.today()
        pnl = DailyPnL(
            date=today,
            starting_capital=Decimal("100000"),
            ending_capital=Decimal("102000"),
            pnl=Decimal("2000"),
            pnl_pct=Decimal("0.02"),
            trade_count=10,
            win_count=7,
            loss_count=3,
        )

        assert pnl.date == today
        assert pnl.pnl == Decimal("2000")
        assert pnl.trade_count == 10

    def test_create_with_calculation(self):
        """Test creating daily P&L with calculated values."""
        today = date.today()
        pnl = DailyPnL.create(
            record_date=today,
            starting_capital=Decimal("100000"),
            ending_capital=Decimal("105000"),
            trade_count=20,
            win_count=12,
            loss_count=8,
        )

        assert pnl.pnl == Decimal("5000")
        assert pnl.pnl_pct == Decimal("0.05")
        assert pnl.win_count == 12
        assert pnl.loss_count == 8

    def test_win_rate(self):
        """Test win rate calculation."""
        pnl = DailyPnL.create(
            record_date=date.today(),
            starting_capital=Decimal("100000"),
            ending_capital=Decimal("100000"),
            trade_count=10,
            win_count=6,
            loss_count=4,
        )

        assert pnl.win_rate == Decimal("0.6")

    def test_win_rate_no_trades(self):
        """Test win rate with no trades."""
        pnl = DailyPnL.create(
            record_date=date.today(),
            starting_capital=Decimal("100000"),
            ending_capital=Decimal("100000"),
            trade_count=0,
            win_count=0,
            loss_count=0,
        )

        assert pnl.win_rate == Decimal("0")


# =============================================================================
# RiskAlert Tests
# =============================================================================


class TestRiskAlert:
    """Tests for RiskAlert model."""

    def test_create_alert(self):
        """Test creating a risk alert."""
        alert = RiskAlert.create(
            level=RiskLevel.WARNING,
            metric="daily_loss",
            current_value=Decimal("0.06"),
            threshold=Decimal("0.05"),
            message="Daily loss exceeded 5% threshold",
            action_taken=RiskAction.NOTIFY,
        )

        assert alert.level == RiskLevel.WARNING
        assert alert.metric == "daily_loss"
        assert alert.current_value == Decimal("0.06")
        assert alert.threshold == Decimal("0.05")
        assert alert.action_taken == RiskAction.NOTIFY
        assert alert.acknowledged is False
        assert alert.alert_id is not None

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        alert = RiskAlert.create(
            level=RiskLevel.DANGER,
            metric="drawdown",
            current_value=Decimal("0.20"),
            threshold=Decimal("0.15"),
            message="Drawdown exceeded danger threshold",
        )

        assert alert.acknowledged is False
        alert.acknowledge()
        assert alert.acknowledged is True


# =============================================================================
# CircuitBreakerState Tests
# =============================================================================


class TestCircuitBreakerState:
    """Tests for CircuitBreakerState model."""

    def test_initial_state(self):
        """Test initial circuit breaker state."""
        state = CircuitBreakerState()

        assert state.is_triggered is False
        assert state.triggered_at is None
        assert state.trigger_reason == ""
        assert state.cooldown_until is None
        assert state.trigger_count_today == 0

    def test_trigger(self):
        """Test triggering circuit breaker."""
        state = CircuitBreakerState()
        state.trigger(reason="Max drawdown exceeded", cooldown_seconds=3600)

        assert state.is_triggered is True
        assert state.triggered_at is not None
        assert state.trigger_reason == "Max drawdown exceeded"
        assert state.cooldown_until is not None
        assert state.trigger_count_today == 1

    def test_multiple_triggers(self):
        """Test multiple triggers increment count."""
        state = CircuitBreakerState()
        state.trigger(reason="First trigger", cooldown_seconds=3600)
        state.trigger(reason="Second trigger", cooldown_seconds=3600)

        assert state.trigger_count_today == 2
        assert state.trigger_reason == "Second trigger"

    def test_reset(self):
        """Test resetting circuit breaker."""
        state = CircuitBreakerState()
        state.trigger(reason="Test", cooldown_seconds=3600)
        state.reset()

        assert state.is_triggered is False
        assert state.triggered_at is None
        assert state.trigger_reason == ""
        assert state.cooldown_until is None
        # Note: trigger_count_today is preserved
        assert state.trigger_count_today == 1

    def test_reset_daily_count(self):
        """Test resetting daily count."""
        state = CircuitBreakerState()
        state.trigger(reason="Test", cooldown_seconds=3600)
        state.reset_daily_count()

        assert state.trigger_count_today == 0

    def test_is_in_cooldown(self):
        """Test cooldown check."""
        state = CircuitBreakerState()

        assert state.is_in_cooldown is False

        state.trigger(reason="Test", cooldown_seconds=3600)
        assert state.is_in_cooldown is True

    def test_cooldown_remaining(self):
        """Test cooldown remaining calculation."""
        state = CircuitBreakerState()

        assert state.cooldown_remaining == timedelta(0)

        state.trigger(reason="Test", cooldown_seconds=3600)
        remaining = state.cooldown_remaining

        assert remaining > timedelta(0)
        assert remaining <= timedelta(seconds=3600)


# =============================================================================
# GlobalRiskStatus Tests
# =============================================================================


class TestGlobalRiskStatus:
    """Tests for GlobalRiskStatus model."""

    @pytest.fixture
    def sample_status(self):
        """Create a sample global risk status."""
        now = datetime.now()
        today = date.today()

        return GlobalRiskStatus(
            level=RiskLevel.NORMAL,
            capital=CapitalSnapshot(
                timestamp=now,
                total_capital=Decimal("100000"),
                available_balance=Decimal("80000"),
                position_value=Decimal("20000"),
                unrealized_pnl=Decimal("500"),
                realized_pnl=Decimal("1000"),
            ),
            drawdown=DrawdownInfo(
                peak_value=Decimal("100000"),
                current_value=Decimal("100000"),
                drawdown_amount=Decimal("0"),
                drawdown_pct=Decimal("0"),
                peak_time=now,
                duration=timedelta(0),
            ),
            daily_pnl=DailyPnL(
                date=today,
                starting_capital=Decimal("100000"),
                ending_capital=Decimal("100500"),
                pnl=Decimal("500"),
                pnl_pct=Decimal("0.005"),
                trade_count=5,
                win_count=3,
                loss_count=2,
            ),
            circuit_breaker=CircuitBreakerState(),
        )

    def test_create_status(self, sample_status):
        """Test creating global risk status."""
        assert sample_status.level == RiskLevel.NORMAL
        assert sample_status.capital.total_capital == Decimal("100000")
        assert sample_status.drawdown.drawdown_pct == Decimal("0")
        assert sample_status.daily_pnl.pnl == Decimal("500")
        assert sample_status.circuit_breaker.is_triggered is False
        assert len(sample_status.active_alerts) == 0

    def test_add_alert(self, sample_status):
        """Test adding an alert."""
        alert = RiskAlert.create(
            level=RiskLevel.WARNING,
            metric="test",
            current_value=Decimal("0.06"),
            threshold=Decimal("0.05"),
            message="Test alert",
        )

        sample_status.add_alert(alert)

        assert len(sample_status.active_alerts) == 1
        assert sample_status.active_alerts[0] == alert

    def test_get_unacknowledged_alerts(self, sample_status):
        """Test getting unacknowledged alerts."""
        alert1 = RiskAlert.create(
            level=RiskLevel.WARNING,
            metric="test1",
            current_value=Decimal("1"),
            threshold=Decimal("1"),
            message="Alert 1",
        )
        alert2 = RiskAlert.create(
            level=RiskLevel.DANGER,
            metric="test2",
            current_value=Decimal("2"),
            threshold=Decimal("1"),
            message="Alert 2",
        )

        sample_status.add_alert(alert1)
        sample_status.add_alert(alert2)
        alert1.acknowledge()

        unacked = sample_status.get_unacknowledged_alerts()
        assert len(unacked) == 1
        assert unacked[0] == alert2

    def test_clear_acknowledged_alerts(self, sample_status):
        """Test clearing acknowledged alerts."""
        alert1 = RiskAlert.create(
            level=RiskLevel.WARNING,
            metric="test1",
            current_value=Decimal("1"),
            threshold=Decimal("1"),
            message="Alert 1",
        )
        alert2 = RiskAlert.create(
            level=RiskLevel.DANGER,
            metric="test2",
            current_value=Decimal("2"),
            threshold=Decimal("1"),
            message="Alert 2",
        )

        sample_status.add_alert(alert1)
        sample_status.add_alert(alert2)
        alert1.acknowledge()
        sample_status.clear_acknowledged_alerts()

        assert len(sample_status.active_alerts) == 1
        assert sample_status.active_alerts[0] == alert2

    def test_update_level(self, sample_status):
        """Test updating risk level."""
        assert sample_status.level == RiskLevel.NORMAL

        changed = sample_status.update_level(RiskLevel.WARNING)
        assert changed is True
        assert sample_status.level == RiskLevel.WARNING

        changed = sample_status.update_level(RiskLevel.WARNING)
        assert changed is False

    def test_update_level_updates_timestamp(self, sample_status):
        """Test that updating level updates timestamp."""
        old_time = sample_status.last_updated
        sample_status.update_level(RiskLevel.DANGER)
        assert sample_status.last_updated >= old_time
