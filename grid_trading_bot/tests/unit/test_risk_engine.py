"""
Tests for Risk Engine.

Tests integrated risk management including checks, alerts, and actions.
"""

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.risk.models import (
    RiskConfig,
    RiskLevel,
)
from src.risk.risk_engine import RiskEngine


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_config():
    """Create a risk configuration."""
    return RiskConfig(
        total_capital=Decimal("100000"),
        warning_loss_pct=Decimal("0.10"),
        danger_loss_pct=Decimal("0.20"),
        daily_loss_warning=Decimal("0.05"),
        daily_loss_danger=Decimal("0.10"),
        max_drawdown_warning=Decimal("0.15"),
        max_drawdown_danger=Decimal("0.25"),
        consecutive_loss_warning=5,
        consecutive_loss_danger=10,
        circuit_breaker_cooldown=3600,
    )


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = MagicMock()
    exchange.get_account = AsyncMock(return_value=MagicMock(
        balances=[
            MagicMock(asset="USDT", free=Decimal("80000"), locked=Decimal("20000"), total=Decimal("100000"))
        ]
    ))
    exchange.get_positions = AsyncMock(return_value=[])
    return exchange


@pytest.fixture
def mock_commander():
    """Create a mock bot commander."""
    commander = MagicMock()
    commander.pause_all = AsyncMock(return_value=["bot_1", "bot_2"])
    commander.stop_all = AsyncMock(return_value=["bot_1", "bot_2"])
    commander.broadcast = AsyncMock(return_value={})
    commander.get_running_bots = MagicMock(return_value=["bot_1", "bot_2"])
    return commander


@pytest.fixture
def mock_notifier():
    """Create a mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def engine(risk_config):
    """Create a risk engine without dependencies."""
    return RiskEngine(risk_config)


@pytest.fixture
def engine_with_deps(risk_config, mock_commander, mock_notifier):
    """Create a risk engine with mock dependencies (no exchange)."""
    return RiskEngine(
        risk_config,
        commander=mock_commander,
        notifier=mock_notifier,
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestRiskEngineInit:
    """Tests for RiskEngine initialization."""

    def test_init_with_config(self, risk_config):
        """Test initialization with config."""
        engine = RiskEngine(risk_config)

        assert engine.config == risk_config
        assert engine.is_running is False
        assert engine.last_status is None

    def test_init_with_dependencies(
        self, risk_config, mock_exchange, mock_commander, mock_notifier
    ):
        """Test initialization with dependencies."""
        engine = RiskEngine(
            risk_config,
            exchange=mock_exchange,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        assert engine._exchange == mock_exchange
        assert engine._commander == mock_commander
        assert engine._notifier == mock_notifier

    def test_init_creates_submodules(self, engine):
        """Test that initialization creates all submodules."""
        assert engine.capital_monitor is not None
        assert engine.drawdown_calculator is not None
        assert engine.circuit_breaker is not None
        assert engine.emergency_stop is not None

    def test_init_with_callback(self, risk_config):
        """Test initialization with level change callback."""
        callback = MagicMock()
        engine = RiskEngine(risk_config, on_level_change=callback)

        assert engine._on_level_change == callback


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestRiskEngineLifecycle:
    """Tests for engine start/stop."""

    @pytest.mark.asyncio
    async def test_start(self, engine):
        """Test starting the engine."""
        await engine.start()

        assert engine.is_running is True
        assert engine._check_task is not None

        await engine.stop()

    @pytest.mark.asyncio
    async def test_start_already_running(self, engine):
        """Test starting when already running."""
        await engine.start()
        await engine.start()  # Should not fail

        assert engine.is_running is True

        await engine.stop()

    @pytest.mark.asyncio
    async def test_stop(self, engine):
        """Test stopping the engine."""
        await engine.start()
        await engine.stop()

        assert engine.is_running is False
        assert engine._check_task is None

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, engine):
        """Test stopping when not running."""
        await engine.stop()  # Should not fail

        assert engine.is_running is False


# =============================================================================
# Check Tests
# =============================================================================


class TestRiskEngineCheck:
    """Tests for risk checking."""

    @pytest.mark.asyncio
    async def test_check_returns_status(self, engine):
        """Test check returns GlobalRiskStatus."""
        # Update capital manually first
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        status = await engine.check()

        assert status is not None
        assert status.level == RiskLevel.NORMAL
        assert status.capital is not None
        assert status.drawdown is not None

    @pytest.mark.asyncio
    async def test_check_updates_last_status(self, engine):
        """Test check updates last_status."""
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        await engine.check()

        assert engine.last_status is not None

    @pytest.mark.asyncio
    async def test_check_with_loss_triggers_warning(self, engine_with_deps):
        """Test check with loss triggers warning alert."""
        # Set initial capital then update with loss
        # Use 6% loss to trigger WARNING but not DANGER (daily_loss_danger is 10%)
        engine_with_deps.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        engine_with_deps.update_capital(
            total_capital=Decimal("94000"),  # 6% loss - triggers warning only
            available_balance=Decimal("94000"),
        )

        status = await engine_with_deps.check()

        assert status.level == RiskLevel.WARNING
        assert len(status.active_alerts) > 0

    @pytest.mark.asyncio
    async def test_check_with_severe_loss_triggers_danger(self, engine_with_deps):
        """Test check with severe loss triggers danger alert."""
        engine_with_deps.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        engine_with_deps.update_capital(
            total_capital=Decimal("75000"),  # 25% loss
            available_balance=Decimal("75000"),
        )

        status = await engine_with_deps.check()

        # Should be at least DANGER level
        assert status.level.value >= RiskLevel.DANGER.value


# =============================================================================
# Risk Level Tests
# =============================================================================


class TestRiskLevelCalculation:
    """Tests for risk level calculation."""

    def test_calculate_risk_level_no_alerts(self, engine):
        """Test calculation with no alerts."""
        level = engine._calculate_risk_level([])

        assert level == RiskLevel.NORMAL

    def test_calculate_risk_level_warning(self, engine):
        """Test calculation with warning alert."""
        from src.risk.models import RiskAlert, RiskAction

        alerts = [
            RiskAlert.create(
                level=RiskLevel.WARNING,
                metric="test",
                current_value=Decimal("1"),
                threshold=Decimal("1"),
                message="Test warning",
            )
        ]

        level = engine._calculate_risk_level(alerts)

        assert level == RiskLevel.WARNING

    def test_calculate_risk_level_takes_max(self, engine):
        """Test that calculation takes maximum level."""
        from src.risk.models import RiskAlert

        alerts = [
            RiskAlert.create(
                level=RiskLevel.WARNING,
                metric="test1",
                current_value=Decimal("1"),
                threshold=Decimal("1"),
                message="Warning",
            ),
            RiskAlert.create(
                level=RiskLevel.DANGER,
                metric="test2",
                current_value=Decimal("2"),
                threshold=Decimal("1"),
                message="Danger",
            ),
        ]

        level = engine._calculate_risk_level(alerts)

        assert level == RiskLevel.DANGER


# =============================================================================
# Action Tests
# =============================================================================


class TestRiskActions:
    """Tests for risk action execution."""

    @pytest.mark.asyncio
    async def test_warning_sends_notification(self, engine_with_deps, mock_notifier):
        """Test warning level sends notification."""
        from src.risk.models import RiskAlert

        alerts = [
            RiskAlert.create(
                level=RiskLevel.WARNING,
                metric="test",
                current_value=Decimal("1"),
                threshold=Decimal("1"),
                message="Test warning",
            )
        ]

        await engine_with_deps._execute_risk_action(RiskLevel.WARNING, alerts)

        mock_notifier.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_risk_level_pauses_new_orders(self, engine_with_deps, mock_commander):
        """Test risk level pauses new orders."""
        from src.risk.models import RiskAlert

        alerts = [
            RiskAlert.create(
                level=RiskLevel.RISK,
                metric="test",
                current_value=Decimal("1"),
                threshold=Decimal("1"),
                message="Test risk",
            )
        ]

        await engine_with_deps._execute_risk_action(RiskLevel.RISK, alerts)

        mock_commander.broadcast.assert_called_with("pause_new_orders")


# =============================================================================
# Consecutive Loss Tests
# =============================================================================


class TestConsecutiveLosses:
    """Tests for consecutive loss tracking."""

    def test_record_winning_trade_resets_count(self, engine):
        """Test winning trade resets consecutive losses."""
        engine._consecutive_losses = 5
        engine.record_trade_result(is_win=True)

        assert engine._consecutive_losses == 0

    def test_record_losing_trade_increments_count(self, engine):
        """Test losing trade increments consecutive losses."""
        engine.record_trade_result(is_win=False)
        engine.record_trade_result(is_win=False)
        engine.record_trade_result(is_win=False)

        assert engine._consecutive_losses == 3

    @pytest.mark.asyncio
    async def test_consecutive_loss_warning_alert(self, engine):
        """Test consecutive loss triggers warning alert."""
        # Set up 5 consecutive losses (warning threshold)
        for _ in range(5):
            engine.record_trade_result(is_win=False)

        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        status = await engine.check()

        # Should have consecutive loss alert
        loss_alerts = [
            a for a in status.active_alerts if a.metric == "consecutive_losses"
        ]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].level == RiskLevel.WARNING

    @pytest.mark.asyncio
    async def test_consecutive_loss_danger_alert(self, engine):
        """Test consecutive loss triggers danger alert."""
        # Set up 10 consecutive losses (danger threshold)
        for _ in range(10):
            engine.record_trade_result(is_win=False)

        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        status = await engine.check()

        loss_alerts = [
            a for a in status.active_alerts if a.metric == "consecutive_losses"
        ]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].level == RiskLevel.DANGER


# =============================================================================
# Manual Control Tests
# =============================================================================


class TestManualControl:
    """Tests for manual control methods."""

    @pytest.mark.asyncio
    async def test_trigger_emergency(self, engine_with_deps):
        """Test manual emergency trigger."""
        await engine_with_deps.trigger_emergency("Manual test")

        assert engine_with_deps.emergency_stop.is_activated is True

    @pytest.mark.asyncio
    async def test_reset_circuit_breaker(self, engine_with_deps):
        """Test reset circuit breaker."""
        # First trigger it
        await engine_with_deps.circuit_breaker.trigger("Test")

        # Then reset (force)
        result = await engine_with_deps.reset_circuit_breaker(force=True)

        assert result is True
        assert engine_with_deps.circuit_breaker.is_triggered is False

    def test_update_capital_manually(self, engine):
        """Test manual capital update."""
        snapshot = engine.update_capital(
            total_capital=Decimal("105000"),
            available_balance=Decimal("85000"),
            position_value=Decimal("20000"),
        )

        assert snapshot.total_capital == Decimal("105000")
        assert engine.capital_monitor.last_snapshot == snapshot


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatistics:
    """Tests for statistics retrieval."""

    @pytest.mark.asyncio
    async def test_get_statistics(self, engine):
        """Test getting statistics."""
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        await engine.check()

        stats = engine.get_statistics()

        assert "current_level" in stats
        assert "total_capital_change" in stats
        assert "daily_pnl" in stats
        assert "current_drawdown_pct" in stats
        assert "consecutive_losses" in stats


# =============================================================================
# Level Change Callback Tests
# =============================================================================


class TestLevelChangeCallback:
    """Tests for level change callback."""

    @pytest.mark.asyncio
    async def test_callback_called_on_level_change(self, risk_config):
        """Test callback is called when level changes."""
        callback = MagicMock()
        engine = RiskEngine(risk_config, on_level_change=callback)

        # First check - sets NORMAL
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        await engine.check()

        # Second check - should trigger WARNING (use 6% loss to avoid DANGER threshold)
        engine.update_capital(
            total_capital=Decimal("94000"),  # 6% loss - triggers warning only
            available_balance=Decimal("94000"),
        )
        await engine.check()

        callback.assert_called_once()
        # Check that it was called with old and new level
        args = callback.call_args[0]
        assert args[0] == RiskLevel.NORMAL
        assert args[1] == RiskLevel.WARNING


# =============================================================================
# Get Status Tests
# =============================================================================


class TestGetStatus:
    """Tests for get_status method."""

    def test_get_status_none_initially(self, engine):
        """Test get_status returns None initially."""
        status = engine.get_status()

        assert status is None

    @pytest.mark.asyncio
    async def test_get_status_after_check(self, engine):
        """Test get_status after check."""
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        await engine.check()

        status = engine.get_status()

        assert status is not None
        assert status.level == RiskLevel.NORMAL


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_check_without_exchange(self, engine):
        """Test check works without exchange."""
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        # Should not crash
        status = await engine.check()

        assert status is not None

    @pytest.mark.asyncio
    async def test_check_handles_exchange_error(self, risk_config, mock_notifier):
        """Test check handles exchange errors gracefully."""
        exchange = MagicMock()
        exchange.get_account = AsyncMock(side_effect=Exception("API error"))

        engine = RiskEngine(risk_config, exchange=exchange, notifier=mock_notifier)

        # Update manually first so there's some data
        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        # Should not crash
        status = await engine.check()

        assert status is not None

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self, risk_config):
        """Test callback error doesn't crash check."""

        def bad_callback(old, new):
            raise Exception("Callback error")

        engine = RiskEngine(risk_config, on_level_change=bad_callback)

        engine.update_capital(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        await engine.check()

        engine.update_capital(
            total_capital=Decimal("88000"),
            available_balance=Decimal("88000"),
        )

        # Should not crash despite callback error
        status = await engine.check()

        assert status is not None
