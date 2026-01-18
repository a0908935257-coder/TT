"""
Tests for Circuit Breaker.

Tests circuit breaker triggering, protection actions, and reset functionality.
"""

from datetime import timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CooldownNotFinishedError,
)
from src.risk.models import (
    RiskAction,
    RiskAlert,
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
        circuit_breaker_cooldown=3600,  # 1 hour
        auto_resume_enabled=False,
    )


@pytest.fixture
def mock_commander():
    """Create a mock bot commander."""
    commander = MagicMock()
    commander.pause_all = AsyncMock(return_value=["bot_1", "bot_2", "bot_3"])
    commander.resume_all = AsyncMock(return_value=["bot_1", "bot_2", "bot_3"])
    commander.send_command = AsyncMock(return_value=True)
    commander.get_running_bots = MagicMock(return_value=["bot_1", "bot_2", "bot_3"])
    return commander


@pytest.fixture
def mock_notifier():
    """Create a mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def breaker(risk_config):
    """Create a circuit breaker without dependencies."""
    return CircuitBreaker(risk_config)


@pytest.fixture
def breaker_with_deps(risk_config, mock_commander, mock_notifier):
    """Create a circuit breaker with mock dependencies."""
    return CircuitBreaker(risk_config, mock_commander, mock_notifier)


@pytest.fixture
def danger_alert():
    """Create a danger level alert."""
    return RiskAlert.create(
        level=RiskLevel.DANGER,
        metric="daily_loss",
        current_value=Decimal("-0.12"),
        threshold=Decimal("-0.10"),
        message="Daily loss -12% exceeded danger threshold -10%",
        action_taken=RiskAction.PAUSE_ALL_BOTS,
    )


@pytest.fixture
def warning_alert():
    """Create a warning level alert."""
    return RiskAlert.create(
        level=RiskLevel.WARNING,
        metric="daily_loss",
        current_value=Decimal("-0.06"),
        threshold=Decimal("-0.05"),
        message="Daily loss -6% exceeded warning threshold -5%",
        action_taken=RiskAction.NOTIFY,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestCircuitBreakerInit:
    """Tests for CircuitBreaker initialization."""

    def test_init_with_config(self, risk_config):
        """Test initialization with config."""
        breaker = CircuitBreaker(risk_config)

        assert breaker.config == risk_config
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_triggered is False
        assert breaker.triggered_at is None
        assert breaker.trigger_reason is None

    def test_init_with_dependencies(self, risk_config, mock_commander, mock_notifier):
        """Test initialization with dependencies."""
        breaker = CircuitBreaker(risk_config, mock_commander, mock_notifier)

        assert breaker._commander == mock_commander
        assert breaker._notifier == mock_notifier

    def test_init_with_callbacks(self, risk_config):
        """Test initialization with callbacks."""
        on_trigger = MagicMock()
        on_reset = MagicMock()

        breaker = CircuitBreaker(
            risk_config,
            on_trigger=on_trigger,
            on_reset=on_reset,
        )

        assert breaker._on_trigger == on_trigger
        assert breaker._on_reset == on_reset


# =============================================================================
# Trigger Tests
# =============================================================================


class TestCircuitBreakerTrigger:
    """Tests for circuit breaker triggering."""

    @pytest.mark.asyncio
    async def test_trigger_changes_state(self, breaker):
        """Test that trigger changes state to OPEN."""
        await breaker.trigger("Test reason")

        assert breaker.state == CircuitState.OPEN
        assert breaker.is_triggered is True

    @pytest.mark.asyncio
    async def test_trigger_sets_reason(self, breaker):
        """Test that trigger sets reason."""
        reason = "Daily loss exceeded 10%"
        await breaker.trigger(reason)

        assert breaker.trigger_reason == reason

    @pytest.mark.asyncio
    async def test_trigger_sets_timestamp(self, breaker):
        """Test that trigger sets timestamp."""
        await breaker.trigger("Test")

        assert breaker.triggered_at is not None

    @pytest.mark.asyncio
    async def test_trigger_sets_cooldown(self, breaker):
        """Test that trigger sets cooldown time."""
        await breaker.trigger("Test")

        assert breaker._cooldown_until is not None
        remaining = breaker.get_cooldown_remaining()
        assert remaining > timedelta(0)

    @pytest.mark.asyncio
    async def test_trigger_increments_daily_count(self, breaker):
        """Test that trigger increments daily count."""
        assert breaker.trigger_count_today == 0

        await breaker.trigger("Test 1")
        breaker.force_close()  # Reset state but keep count
        await breaker.trigger("Test 2")

        assert breaker.trigger_count_today == 2

    @pytest.mark.asyncio
    async def test_trigger_already_triggered(self, breaker):
        """Test that second trigger is ignored."""
        await breaker.trigger("First reason")
        await breaker.trigger("Second reason")

        # Should still have first reason
        assert breaker.trigger_reason == "First reason"

    @pytest.mark.asyncio
    async def test_trigger_executes_protection(self, breaker_with_deps, mock_commander):
        """Test that trigger executes protection actions."""
        await breaker_with_deps.trigger("Test")

        mock_commander.pause_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_trigger_sends_notification(self, breaker_with_deps, mock_notifier):
        """Test that trigger sends notification."""
        await breaker_with_deps.trigger("Test reason")

        mock_notifier.send.assert_called_once()
        call_args = mock_notifier.send.call_args
        assert "EMERGENCY" in call_args.kwargs["title"]
        assert call_args.kwargs["level"] == "critical"

    @pytest.mark.asyncio
    async def test_trigger_calls_callback(self, risk_config):
        """Test that trigger calls on_trigger callback."""
        callback = MagicMock()
        breaker = CircuitBreaker(risk_config, on_trigger=callback)

        await breaker.trigger("Test reason")

        callback.assert_called_once_with("Test reason")


# =============================================================================
# Check and Trigger Tests
# =============================================================================


class TestCheckAndTrigger:
    """Tests for check_and_trigger method."""

    @pytest.mark.asyncio
    async def test_triggers_on_danger_alert(self, breaker, danger_alert):
        """Test triggers on DANGER level alert."""
        result = await breaker.check_and_trigger([danger_alert])

        assert result is True
        assert breaker.is_triggered is True

    @pytest.mark.asyncio
    async def test_no_trigger_on_warning_alert(self, breaker, warning_alert):
        """Test does not trigger on WARNING level alert."""
        result = await breaker.check_and_trigger([warning_alert])

        assert result is False
        assert breaker.is_triggered is False

    @pytest.mark.asyncio
    async def test_no_trigger_on_empty_alerts(self, breaker):
        """Test does not trigger on empty alert list."""
        result = await breaker.check_and_trigger([])

        assert result is False
        assert breaker.is_triggered is False

    @pytest.mark.asyncio
    async def test_no_trigger_when_already_triggered(self, breaker, danger_alert):
        """Test does not trigger again when already triggered."""
        await breaker.trigger("First trigger")

        result = await breaker.check_and_trigger([danger_alert])

        assert result is False
        assert breaker.trigger_reason == "First trigger"

    @pytest.mark.asyncio
    async def test_uses_first_danger_message(self, breaker):
        """Test uses first danger alert message as reason."""
        alerts = [
            RiskAlert.create(
                level=RiskLevel.DANGER,
                metric="first",
                current_value=Decimal("1"),
                threshold=Decimal("1"),
                message="First danger message",
            ),
            RiskAlert.create(
                level=RiskLevel.DANGER,
                metric="second",
                current_value=Decimal("2"),
                threshold=Decimal("1"),
                message="Second danger message",
            ),
        ]

        await breaker.check_and_trigger(alerts)

        assert breaker.trigger_reason == "First danger message"


# =============================================================================
# Reset Tests
# =============================================================================


class TestCircuitBreakerReset:
    """Tests for circuit breaker reset."""

    @pytest.mark.asyncio
    async def test_reset_not_triggered(self, breaker):
        """Test reset when not triggered does nothing."""
        await breaker.reset()

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_before_cooldown_raises(self, breaker):
        """Test reset before cooldown raises error."""
        await breaker.trigger("Test")

        with pytest.raises(CooldownNotFinishedError):
            await breaker.reset()

    @pytest.mark.asyncio
    async def test_reset_force_bypasses_cooldown(self, breaker):
        """Test force reset bypasses cooldown."""
        await breaker.trigger("Test")

        await breaker.reset(force=True)

        assert breaker.is_triggered is False
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, breaker):
        """Test reset clears trigger state."""
        await breaker.trigger("Test")
        await breaker.reset(force=True)

        assert breaker.triggered_at is None
        assert breaker.trigger_reason is None

    @pytest.mark.asyncio
    async def test_reset_sends_notification(self, breaker_with_deps, mock_notifier):
        """Test reset sends notification."""
        await breaker_with_deps.trigger("Test")
        await breaker_with_deps.reset(force=True)

        # Should have 2 calls: trigger and reset
        assert mock_notifier.send.call_count == 2

    @pytest.mark.asyncio
    async def test_reset_calls_callback(self, risk_config):
        """Test reset calls on_reset callback."""
        callback = MagicMock()
        breaker = CircuitBreaker(risk_config, on_reset=callback)

        await breaker.trigger("Test")
        await breaker.reset(force=True)

        callback.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_auto_resume_when_enabled(self, mock_commander, mock_notifier):
        """Test auto-resume bots when enabled."""
        config = RiskConfig(
            total_capital=Decimal("100000"),
            circuit_breaker_cooldown=3600,
            auto_resume_enabled=True,
        )
        breaker = CircuitBreaker(config, mock_commander, mock_notifier)

        await breaker.trigger("Test")
        await breaker.reset(force=True)

        mock_commander.resume_all.assert_called_once()


# =============================================================================
# State Tests
# =============================================================================


class TestCircuitBreakerState:
    """Tests for state retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_state_not_triggered(self, breaker):
        """Test get_state when not triggered."""
        state = breaker.get_state()

        assert state.is_triggered is False
        assert state.triggered_at is None
        assert state.trigger_reason == ""
        assert state.cooldown_until is None
        assert state.trigger_count_today == 0

    @pytest.mark.asyncio
    async def test_get_state_triggered(self, breaker):
        """Test get_state when triggered."""
        await breaker.trigger("Test reason")

        state = breaker.get_state()

        assert state.is_triggered is True
        assert state.triggered_at is not None
        assert state.trigger_reason == "Test reason"
        assert state.cooldown_until is not None
        assert state.trigger_count_today == 1

    def test_get_cooldown_remaining_not_triggered(self, breaker):
        """Test cooldown remaining when not triggered."""
        remaining = breaker.get_cooldown_remaining()

        assert remaining == timedelta(0)

    @pytest.mark.asyncio
    async def test_get_cooldown_remaining_triggered(self, breaker):
        """Test cooldown remaining when triggered."""
        await breaker.trigger("Test")

        remaining = breaker.get_cooldown_remaining()

        # Should be close to config value (1 hour = 3600 seconds)
        assert remaining.total_seconds() > 3500  # Allow some tolerance
        assert remaining.total_seconds() <= 3600

    @pytest.mark.asyncio
    async def test_is_cooldown_finished_not_triggered(self, breaker):
        """Test is_cooldown_finished when not triggered."""
        assert breaker.is_cooldown_finished() is True

    @pytest.mark.asyncio
    async def test_is_cooldown_finished_just_triggered(self, breaker):
        """Test is_cooldown_finished when just triggered."""
        await breaker.trigger("Test")

        assert breaker.is_cooldown_finished() is False


# =============================================================================
# Protection Tests
# =============================================================================


class TestProtectionActions:
    """Tests for protection action execution."""

    @pytest.mark.asyncio
    async def test_protection_pauses_all_bots(self, breaker_with_deps, mock_commander):
        """Test that protection pauses all bots."""
        await breaker_with_deps.trigger("Test")

        mock_commander.pause_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_protection_cancels_orders(self, breaker_with_deps, mock_commander):
        """Test that protection cancels orders for each bot."""
        await breaker_with_deps.trigger("Test")

        # Should have 3 cancel_all_orders calls (one per bot)
        assert mock_commander.send_command.call_count == 3

        for call in mock_commander.send_command.call_args_list:
            assert call.args[1] == "cancel_all_orders"

    @pytest.mark.asyncio
    async def test_protection_tracks_paused_bots(self, breaker_with_deps):
        """Test that protection tracks paused bots."""
        await breaker_with_deps.trigger("Test")

        assert len(breaker_with_deps._paused_bots) == 3


# =============================================================================
# Manual Control Tests
# =============================================================================


class TestManualControl:
    """Tests for manual control methods."""

    @pytest.mark.asyncio
    async def test_force_close(self, breaker):
        """Test force close bypasses everything."""
        await breaker.trigger("Test")

        breaker.force_close()

        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_triggered is False
        assert breaker._cooldown_until is None

    @pytest.mark.asyncio
    async def test_extend_cooldown(self, breaker):
        """Test extending cooldown period."""
        await breaker.trigger("Test")

        original_cooldown = breaker._cooldown_until

        breaker.extend_cooldown(1800)  # Add 30 minutes

        assert breaker._cooldown_until > original_cooldown
        # Should be ~30 minutes more
        diff = (breaker._cooldown_until - original_cooldown).total_seconds()
        assert diff == 1800

    def test_extend_cooldown_not_triggered(self, breaker):
        """Test extend cooldown when not triggered does nothing."""
        breaker.extend_cooldown(1800)

        # Should not crash, just log warning
        assert breaker._cooldown_until is None


# =============================================================================
# Daily Count Tests
# =============================================================================


class TestDailyCount:
    """Tests for daily trigger count tracking."""

    @pytest.mark.asyncio
    async def test_trigger_count_starts_at_zero(self, breaker):
        """Test trigger count starts at zero."""
        assert breaker.trigger_count_today == 0

    @pytest.mark.asyncio
    async def test_trigger_count_increments(self, breaker):
        """Test trigger count increments on each trigger."""
        await breaker.trigger("Test 1")
        assert breaker.trigger_count_today == 1

        breaker.force_close()
        await breaker.trigger("Test 2")
        assert breaker.trigger_count_today == 2

    @pytest.mark.asyncio
    async def test_trigger_count_in_state(self, breaker):
        """Test trigger count is included in state."""
        await breaker.trigger("Test")

        state = breaker.get_state()
        assert state.trigger_count_today == 1


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_protection_without_commander(self, risk_config, mock_notifier):
        """Test protection runs without commander."""
        breaker = CircuitBreaker(risk_config, notifier=mock_notifier)

        # Should not crash
        await breaker.trigger("Test")

        assert breaker.is_triggered is True

    @pytest.mark.asyncio
    async def test_notification_without_notifier(self, risk_config, mock_commander):
        """Test notification skipped without notifier."""
        breaker = CircuitBreaker(risk_config, commander=mock_commander)

        # Should not crash
        await breaker.trigger("Test")

        assert breaker.is_triggered is True

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self, risk_config):
        """Test callback error doesn't crash trigger."""

        def bad_callback(reason):
            raise Exception("Callback error")

        breaker = CircuitBreaker(risk_config, on_trigger=bad_callback)

        # Should not crash despite callback error
        await breaker.trigger("Test")

        assert breaker.is_triggered is True

    @pytest.mark.asyncio
    async def test_commander_error_does_not_crash(self, risk_config, mock_notifier):
        """Test commander error doesn't crash trigger."""
        commander = MagicMock()
        commander.pause_all = AsyncMock(side_effect=Exception("Commander error"))
        commander.get_running_bots = MagicMock(return_value=[])

        breaker = CircuitBreaker(risk_config, commander, mock_notifier)

        # Should not crash despite commander error
        await breaker.trigger("Test")

        assert breaker.is_triggered is True
