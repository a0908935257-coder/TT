"""
Unit tests for TimeBasedStopLoss.

Tests the time-based stop loss functionality including:
- Max holding time stops
- No-profit timeout stops
- Session end stops
- Scheduled exit stops
- Gradual position reduction
"""

import pytest
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from unittest.mock import MagicMock

from src.risk.sltp.time_stop import (
    TimeBasedStopLoss,
    TimeStopAction,
    TimeStopCondition,
    TimeStopConfig,
    TimeStopResult,
    TimeStopState,
    TimeStopType,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_config():
    """Default time stop configuration."""
    return TimeStopConfig(
        stop_type=TimeStopType.MAX_HOLDING,
        max_holding_minutes=60,
        condition=TimeStopCondition.ALWAYS,
        action=TimeStopAction.CLOSE_FULL,
        warning_before_minutes=5,
        enabled=True,
    )


@pytest.fixture
def no_profit_config():
    """No-profit timeout configuration."""
    return TimeStopConfig(
        stop_type=TimeStopType.NO_PROFIT,
        no_profit_minutes=30,
        condition=TimeStopCondition.ALWAYS,
        action=TimeStopAction.CLOSE_FULL,
        warning_before_minutes=5,
        enabled=True,
    )


@pytest.fixture
def session_end_config():
    """Session end configuration."""
    return TimeStopConfig(
        stop_type=TimeStopType.SESSION_END,
        session_end_time=time(16, 0, 0),  # 4 PM UTC
        exit_before_close_minutes=5,
        condition=TimeStopCondition.ALWAYS,
        action=TimeStopAction.CLOSE_FULL,
        warning_before_minutes=5,
        enabled=True,
    )


@pytest.fixture
def scheduled_config():
    """Scheduled exit configuration."""
    return TimeStopConfig(
        stop_type=TimeStopType.SCHEDULED,
        scheduled_exit_time=time(14, 0, 0),  # 2 PM UTC
        condition=TimeStopCondition.ALWAYS,
        action=TimeStopAction.CLOSE_FULL,
        warning_before_minutes=5,
        enabled=True,
    )


@pytest.fixture
def gradual_config():
    """Gradual exit configuration."""
    return TimeStopConfig(
        stop_type=TimeStopType.GRADUAL,
        gradual_start_minutes=30,
        gradual_interval_minutes=10,
        gradual_reduction_pct=Decimal("0.25"),
        condition=TimeStopCondition.ALWAYS,
        enabled=True,
    )


@pytest.fixture
def time_stop(default_config):
    """Default TimeBasedStopLoss instance."""
    return TimeBasedStopLoss(default_config)


# =============================================================================
# TimeStopState Tests
# =============================================================================


class TestTimeStopState:
    """Tests for TimeStopState."""

    def test_state_initialization(self):
        """Test state initializes correctly."""
        entry_time = datetime.now(timezone.utc)
        state = TimeStopState(
            symbol="BTCUSDT",
            entry_time=entry_time,
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        assert state.symbol == "BTCUSDT"
        assert state.entry_price == Decimal("50000")
        assert state.is_long is True
        assert state.quantity == Decimal("1.0")
        assert state.remaining_quantity == Decimal("1.0")
        assert state.highest_pnl_pct == Decimal("0")
        assert state.gradual_exits_count == 0
        assert state.warning_sent is False
        assert state.triggered is False

    def test_holding_duration(self):
        """Test holding duration calculation."""
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        state = TimeStopState(
            symbol="BTCUSDT",
            entry_time=entry_time,
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        assert 29.9 <= state.holding_minutes <= 30.1

    def test_update_pnl(self):
        """Test P&L update tracking."""
        state = TimeStopState(
            symbol="BTCUSDT",
            entry_time=datetime.now(timezone.utc),
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # Update with profit
        state.update_pnl(Decimal("5.0"))  # 5% profit
        assert state.highest_pnl_pct == Decimal("5.0")
        assert state.last_profit_time is not None

        # Update with lower profit - highest should stay
        state.update_pnl(Decimal("3.0"))
        assert state.highest_pnl_pct == Decimal("5.0")

    def test_record_gradual_exit(self):
        """Test gradual exit recording."""
        state = TimeStopState(
            symbol="BTCUSDT",
            entry_time=datetime.now(timezone.utc),
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        state.record_gradual_exit(Decimal("0.25"))
        assert state.gradual_exits_count == 1
        assert state.remaining_quantity == Decimal("0.75")

    def test_mark_triggered(self):
        """Test marking state as triggered."""
        state = TimeStopState(
            symbol="BTCUSDT",
            entry_time=datetime.now(timezone.utc),
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        state.mark_triggered("Max holding time reached")
        assert state.triggered is True
        assert state.triggered_at is not None
        assert state.trigger_reason == "Max holding time reached"


# =============================================================================
# TimeBasedStopLoss Basic Tests
# =============================================================================


class TestTimeBasedStopLossBasic:
    """Basic tests for TimeBasedStopLoss."""

    def test_initialization(self, default_config):
        """Test manager initializes correctly."""
        manager = TimeBasedStopLoss(default_config)

        assert manager.config == default_config
        assert len(manager.active_positions) == 0

    def test_create_state(self, time_stop):
        """Test creating position state."""
        state = time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        assert state.symbol == "BTCUSDT"
        assert "BTCUSDT" in time_stop.active_positions

    def test_get_state(self, time_stop):
        """Test getting position state."""
        time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        state = time_stop.get_state("BTCUSDT")
        assert state is not None
        assert state.symbol == "BTCUSDT"

        # Non-existent
        assert time_stop.get_state("ETHUSDT") is None

    def test_remove_state(self, time_stop):
        """Test removing position state."""
        time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        assert time_stop.remove_state("BTCUSDT") is True
        assert "BTCUSDT" not in time_stop.active_positions
        assert time_stop.remove_state("BTCUSDT") is False

    def test_disabled_time_stop(self, default_config):
        """Test time stop when disabled."""
        default_config.enabled = False
        manager = TimeBasedStopLoss(default_config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False
        assert "disabled" in result.reason.lower()

    def test_no_state_check(self, time_stop):
        """Test check with no state."""
        result = time_stop.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False


# =============================================================================
# Max Holding Time Tests
# =============================================================================


class TestMaxHoldingTime:
    """Tests for max holding time stop."""

    def test_not_yet_triggered(self, default_config):
        """Test when holding time not reached."""
        manager = TimeBasedStopLoss(default_config)

        # Create state with recent entry
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False
        assert result.minutes_until_trigger is not None
        assert result.minutes_until_trigger > 20

    def test_warning_before_trigger(self, default_config):
        """Test warning before max holding time."""
        manager = TimeBasedStopLoss(default_config)

        # 57 minutes ago (3 minutes before 60-min trigger, within 5-min warning)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=57)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.is_warning is True
        assert result.should_exit is True

    def test_trigger_after_max_time(self, default_config):
        """Test trigger after max holding time."""
        manager = TimeBasedStopLoss(default_config)

        # 65 minutes ago
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is True
        assert result.is_warning is False
        assert "max holding time" in result.reason.lower()

    def test_condition_if_loss(self, default_config):
        """Test max holding with IF_LOSS condition."""
        default_config.condition = TimeStopCondition.IF_LOSS
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # In profit - should not trigger
        result = manager.check("BTCUSDT", Decimal("52000"))
        assert result.should_exit is False
        assert "condition not met" in result.reason.lower()

        # At loss - should trigger
        result = manager.check("BTCUSDT", Decimal("48000"))
        assert result.should_exit is True

    def test_condition_if_no_profit(self, default_config):
        """Test max holding with IF_NO_PROFIT condition."""
        default_config.condition = TimeStopCondition.IF_NO_PROFIT
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # In profit - should not trigger
        result = manager.check("BTCUSDT", Decimal("52000"))
        assert result.should_exit is False

        # Break-even - should trigger
        result = manager.check("BTCUSDT", Decimal("50000"))
        assert result.should_exit is True


# =============================================================================
# No-Profit Timeout Tests
# =============================================================================


class TestNoProfitTimeout:
    """Tests for no-profit timeout stop."""

    def test_not_triggered_when_profitable(self, no_profit_config):
        """Test no trigger when position is profitable."""
        manager = TimeBasedStopLoss(no_profit_config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        result = manager.check("BTCUSDT", Decimal("52000"))  # 4% profit
        assert result.should_exit is False
        assert "currently in profit" in result.reason.lower()

    def test_trigger_after_no_profit_timeout(self, no_profit_config):
        """Test trigger after no profit for too long."""
        manager = TimeBasedStopLoss(no_profit_config)

        # Entry 35 minutes ago
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=35)
        state = manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )
        # last_profit_time is set to entry_time by default

        result = manager.check("BTCUSDT", Decimal("49000"))  # At loss
        assert result.should_exit is True
        assert "no profit timeout" in result.reason.lower()

    def test_reset_timer_on_profit(self, no_profit_config):
        """Test timer resets when position becomes profitable."""
        manager = TimeBasedStopLoss(no_profit_config)

        # Use 20 minutes to avoid warning period (warning at 25 min for 30 min threshold)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=20)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # First check at loss - not yet at warning/trigger threshold
        result = manager.check("BTCUSDT", Decimal("49000"))
        assert result.should_exit is False

        # Check at profit - this resets the timer
        result = manager.check("BTCUSDT", Decimal("52000"))
        assert result.should_exit is False
        assert "currently in profit" in result.reason.lower()


# =============================================================================
# Session End Tests
# =============================================================================


class TestSessionEnd:
    """Tests for session end stop."""

    def test_no_session_end_configured(self):
        """Test when no session end time is set."""
        config = TimeStopConfig(
            stop_type=TimeStopType.SESSION_END,
            session_end_time=None,
        )
        manager = TimeBasedStopLoss(config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False
        assert "no session end" in result.reason.lower()

    def test_trigger_before_session_end(self, session_end_config):
        """Test trigger before session end."""
        manager = TimeBasedStopLoss(session_end_config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 3 minutes before 4 PM (within exit_before_close_minutes=5)
        test_time = datetime(2024, 1, 15, 15, 57, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.should_exit is True
        assert "session end" in result.reason.lower()

    def test_warning_before_session_end(self, session_end_config):
        """Test warning before session end."""
        manager = TimeBasedStopLoss(session_end_config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 8 minutes before 4 PM (within warning period)
        test_time = datetime(2024, 1, 15, 15, 52, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.is_warning is True

    def test_not_triggered_far_from_session_end(self, session_end_config):
        """Test no trigger when far from session end."""
        manager = TimeBasedStopLoss(session_end_config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 2 PM - 2 hours before session end
        test_time = datetime(2024, 1, 15, 14, 0, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.should_exit is False

    def test_custom_session_end_callback(self, session_end_config):
        """Test custom session end time callback."""

        def get_session_end(symbol):
            if symbol == "BTCUSDT":
                return time(18, 0, 0)  # 6 PM for BTC
            return None

        manager = TimeBasedStopLoss(session_end_config, get_session_end=get_session_end)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 3 minutes before 6 PM
        test_time = datetime(2024, 1, 15, 17, 57, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.should_exit is True


# =============================================================================
# Scheduled Exit Tests
# =============================================================================


class TestScheduledExit:
    """Tests for scheduled exit stop."""

    def test_no_scheduled_time(self):
        """Test when no scheduled time is set."""
        config = TimeStopConfig(
            stop_type=TimeStopType.SCHEDULED,
            scheduled_exit_time=None,
        )
        manager = TimeBasedStopLoss(config)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False

    def test_trigger_at_scheduled_time(self, scheduled_config):
        """Test trigger at scheduled exit time."""
        manager = TimeBasedStopLoss(scheduled_config)

        # Entry before scheduled time
        entry_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # Check after scheduled time (2 PM)
        test_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.should_exit is True
        assert "past scheduled" in result.reason.lower()

    def test_warning_before_scheduled_time(self, scheduled_config):
        """Test warning before scheduled exit."""
        manager = TimeBasedStopLoss(scheduled_config)

        entry_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # 3 minutes before 2 PM
        test_time = datetime(2024, 1, 15, 13, 57, 0, tzinfo=timezone.utc)
        result = manager.check("BTCUSDT", Decimal("51000"), current_time=test_time)
        assert result.is_warning is True


# =============================================================================
# Gradual Exit Tests
# =============================================================================


class TestGradualExit:
    """Tests for gradual exit stop."""

    def test_not_started_yet(self, gradual_config):
        """Test when gradual exit hasn't started."""
        manager = TimeBasedStopLoss(gradual_config)

        # Entry 10 minutes ago (before gradual_start_minutes=30)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is False
        assert "starts after" in result.reason.lower()

    def test_first_gradual_exit(self, gradual_config):
        """Test first gradual exit."""
        manager = TimeBasedStopLoss(gradual_config)

        # Entry 41 minutes ago (after gradual_start_minutes=30 + interval=10)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=41)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is True
        assert result.action == TimeStopAction.CLOSE_PARTIAL
        assert result.quantity_to_close == Decimal("0.25")  # 25% of 1.0

    def test_multiple_gradual_exits(self, gradual_config):
        """Test multiple gradual exits over time."""
        manager = TimeBasedStopLoss(gradual_config)

        # Entry 51 minutes ago (enough for 2 expected exits: at 40m and 50m)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=51)
        state = manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # First exit (expected_exits=2, done=0)
        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is True
        state.record_gradual_exit(result.quantity_to_close)

        # Second exit (expected_exits=2, done=1)
        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.should_exit is True
        assert result.quantity_to_close == Decimal("0.1875")  # 25% of 0.75

    def test_gradual_with_condition(self, gradual_config):
        """Test gradual exit with condition."""
        gradual_config.condition = TimeStopCondition.IF_LOSS
        manager = TimeBasedStopLoss(gradual_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=41)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        # In profit - should not exit
        result = manager.check("BTCUSDT", Decimal("52000"))
        assert result.should_exit is False

        # At loss - should exit
        result = manager.check("BTCUSDT", Decimal("48000"))
        assert result.should_exit is True


# =============================================================================
# Callback Tests
# =============================================================================


class TestCallbacks:
    """Tests for callbacks."""

    def test_on_warning_callback(self, default_config):
        """Test warning callback is called."""
        callback = MagicMock()
        manager = TimeBasedStopLoss(default_config, on_warning=callback)

        # 57 minutes ago (within warning range)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=57)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.check("BTCUSDT", Decimal("51000"))
        callback.assert_called_once()

    def test_on_trigger_callback(self, default_config):
        """Test trigger callback is called."""
        callback = MagicMock()
        manager = TimeBasedStopLoss(default_config, on_trigger=callback)

        # 65 minutes ago (past max time)
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.check("BTCUSDT", Decimal("51000"))
        callback.assert_called_once()

    def test_warning_only_once(self, default_config):
        """Test warning callback only called once."""
        callback = MagicMock()
        manager = TimeBasedStopLoss(default_config, on_warning=callback)

        # 57 minutes ago
        entry_time = datetime.now(timezone.utc) - timedelta(minutes=57)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.check("BTCUSDT", Decimal("51000"))
        manager.check("BTCUSDT", Decimal("51000"))
        manager.check("BTCUSDT", Decimal("51000"))

        callback.assert_called_once()


# =============================================================================
# Batch Operations Tests
# =============================================================================


class TestBatchOperations:
    """Tests for batch operations."""

    def test_check_all(self, default_config):
        """Test checking all positions."""
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.create_state(
            symbol="ETHUSDT",
            entry_price=Decimal("3000"),
            is_long=True,
            quantity=Decimal("10.0"),
            entry_time=entry_time,
        )

        prices = {
            "BTCUSDT": Decimal("51000"),
            "ETHUSDT": Decimal("3100"),
        }

        results = manager.check_all(prices)
        assert "BTCUSDT" in results
        assert "ETHUSDT" in results
        assert len(results) == 2

    def test_get_positions_near_trigger(self, default_config):
        """Test getting positions near trigger."""
        manager = TimeBasedStopLoss(default_config)

        # Position far from trigger
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=30),
        )

        # Position near trigger
        manager.create_state(
            symbol="ETHUSDT",
            entry_price=Decimal("3000"),
            is_long=True,
            quantity=Decimal("10.0"),
            entry_time=datetime.now(timezone.utc) - timedelta(minutes=57),
        )

        near = manager.get_positions_near_trigger(minutes_threshold=5)
        assert len(near) == 1
        assert near[0][0] == "ETHUSDT"


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    def test_statistics_tracking(self, default_config):
        """Test statistics are tracked correctly."""
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.check("BTCUSDT", Decimal("51000"))

        stats = manager.get_statistics()
        assert stats["total_checks"] == 1
        assert stats["total_triggers"] == 1
        assert stats["active_positions"] == 0  # Triggered position not active

    def test_reset_statistics(self, default_config):
        """Test resetting statistics."""
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        manager.check("BTCUSDT", Decimal("51000"))
        manager.reset_statistics()

        stats = manager.get_statistics()
        assert stats["total_checks"] == 0
        assert stats["total_triggers"] == 0


# =============================================================================
# Configuration Tests
# =============================================================================


class TestConfiguration:
    """Tests for configuration management."""

    def test_update_config(self, default_config):
        """Test updating configuration."""
        manager = TimeBasedStopLoss(default_config)

        manager.update_config(max_holding_minutes=120, warning_before_minutes=10)

        assert manager.config.max_holding_minutes == 120
        assert manager.config.warning_before_minutes == 10

    def test_enable_disable(self, default_config):
        """Test enable/disable."""
        manager = TimeBasedStopLoss(default_config)

        manager.disable()
        assert manager.config.enabled is False

        manager.enable()
        assert manager.config.enabled is True


# =============================================================================
# P&L Calculation Tests
# =============================================================================


class TestPnLCalculation:
    """Tests for P&L calculation."""

    def test_long_position_profit(self, time_stop):
        """Test P&L calculation for long position in profit."""
        state = time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 4% profit (returned as ratio 0.04)
        pnl = time_stop._calculate_pnl_pct(state, Decimal("52000"))
        assert pnl == Decimal("0.04")

    def test_long_position_loss(self, time_stop):
        """Test P&L calculation for long position at loss."""
        state = time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
        )

        # 2% loss (returned as ratio -0.02)
        pnl = time_stop._calculate_pnl_pct(state, Decimal("49000"))
        assert pnl == Decimal("-0.02")

    def test_short_position_profit(self, time_stop):
        """Test P&L calculation for short position in profit."""
        state = time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=False,
            quantity=Decimal("1.0"),
        )

        # 2% profit (returned as ratio 0.02)
        pnl = time_stop._calculate_pnl_pct(state, Decimal("49000"))
        assert pnl == Decimal("0.02")

    def test_short_position_loss(self, time_stop):
        """Test P&L calculation for short position at loss."""
        state = time_stop.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=False,
            quantity=Decimal("1.0"),
        )

        # 4% loss (returned as ratio -0.04)
        pnl = time_stop._calculate_pnl_pct(state, Decimal("52000"))
        assert pnl == Decimal("-0.04")


# =============================================================================
# Action Type Tests
# =============================================================================


class TestActionTypes:
    """Tests for different action types."""

    def test_close_partial_action(self, default_config):
        """Test partial close action."""
        default_config.action = TimeStopAction.CLOSE_PARTIAL
        default_config.partial_close_pct = Decimal("0.50")
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.quantity_to_close == Decimal("0.50")

    def test_close_full_action(self, default_config):
        """Test full close action."""
        default_config.action = TimeStopAction.CLOSE_FULL
        manager = TimeBasedStopLoss(default_config)

        entry_time = datetime.now(timezone.utc) - timedelta(minutes=65)
        manager.create_state(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("1.0"),
            entry_time=entry_time,
        )

        result = manager.check("BTCUSDT", Decimal("51000"))
        assert result.quantity_to_close == Decimal("1.0")
