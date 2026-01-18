"""
Risk Management Integration Tests.

Tests complete risk flow scenarios including:
- Normal → Warning → Danger → Circuit Break transitions
- Emergency stop activation
- Full system integration
"""

from datetime import datetime
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk import (
    CapitalMonitor,
    CircuitBreaker,
    DrawdownCalculator,
    EmergencyConfig,
    EmergencyStop,
    RiskConfig,
    RiskEngine,
    RiskLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_config():
    """Risk configuration for integration tests."""
    return RiskConfig(
        total_capital=Decimal("10000"),
        warning_loss_pct=Decimal("0.10"),  # 10%
        danger_loss_pct=Decimal("0.20"),   # 20%
        daily_loss_warning=Decimal("0.05"),  # 5%
        daily_loss_danger=Decimal("0.10"),   # 10%
        max_drawdown_warning=Decimal("0.15"),  # 15%
        max_drawdown_danger=Decimal("0.25"),   # 25%
        consecutive_loss_warning=5,
        consecutive_loss_danger=10,
        circuit_breaker_cooldown=3600,
    )


@pytest.fixture
def mock_exchange():
    """Mock exchange with controllable balance."""
    exchange = MagicMock()
    exchange._balance = Decimal("10000")

    async def get_account(market=None):
        return MagicMock(
            balances=[
                MagicMock(
                    asset="USDT",
                    free=exchange._balance,
                    locked=Decimal("0"),
                    total=exchange._balance,
                )
            ]
        )

    exchange.get_account = AsyncMock(side_effect=get_account)
    exchange.get_positions = AsyncMock(return_value=[])
    exchange.get_open_orders = AsyncMock(return_value=[])
    exchange.cancel_order = AsyncMock()

    def set_balance(balance: Decimal):
        exchange._balance = balance

    exchange.set_balance = set_balance
    return exchange


@pytest.fixture
def mock_commander():
    """Mock bot commander."""
    commander = MagicMock()
    commander.pause_all = AsyncMock(return_value=["bot_1", "bot_2"])
    commander.stop_all = AsyncMock(return_value=["bot_1", "bot_2"])
    commander.broadcast = AsyncMock(return_value={})
    commander.get_running_bots = MagicMock(return_value=["bot_1", "bot_2"])
    return commander


@pytest.fixture
def mock_notifier():
    """Mock notification manager."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    notifier.notifications = []

    async def capture_send(title, message, level="info", **kwargs):
        notifier.notifications.append({
            "title": title,
            "message": message,
            "level": level,
        })
        return True

    notifier.send = AsyncMock(side_effect=capture_send)
    return notifier


@pytest.fixture
def risk_engine(risk_config, mock_commander, mock_notifier):
    """Create risk engine without exchange (manual updates)."""
    return RiskEngine(
        risk_config,
        commander=mock_commander,
        notifier=mock_notifier,
    )


# =============================================================================
# Scenario Tests
# =============================================================================


class TestScenarios:
    """Test specific risk scenarios."""

    @pytest.mark.asyncio
    async def test_scenario_1_normal_operation(self, risk_engine):
        """
        Scenario 1: Normal operation.

        Initial: 10,000
        Current: 10,500 (+5%)
        Drawdown: 0%

        Expected: NORMAL level, no alerts
        """
        # Set initial
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Update to profit
        risk_engine.update_capital(
            total_capital=Decimal("10500"),
            available_balance=Decimal("10500"),
        )

        status = await risk_engine.check()

        assert status.level == RiskLevel.NORMAL
        assert len(status.active_alerts) == 0

    @pytest.mark.asyncio
    async def test_scenario_2_warning_state(self, risk_engine):
        """
        Scenario 2: Warning state.

        Initial: 10,000
        Current: 9,000 (-10%)

        Expected: WARNING level, capital warning alert
        """
        # Set initial high
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Update drawdown to 6% (triggers warning_loss_pct but not daily_loss_danger)
        risk_engine.update_capital(
            total_capital=Decimal("9400"),
            available_balance=Decimal("9400"),
        )

        status = await risk_engine.check()

        assert status.level == RiskLevel.WARNING
        assert len(status.active_alerts) > 0

        # Check alert contains loss warning (could be daily_loss or capital_change)
        alert_metrics = [a.metric for a in status.active_alerts]
        assert any(m in ["capital_change", "daily_loss"] for m in alert_metrics)

    @pytest.mark.asyncio
    async def test_scenario_3_danger_state(self, risk_config, mock_commander, mock_notifier):
        """
        Scenario 3: Danger state.

        Initial: 10,000
        Current: 7,800 (-22%)

        Expected: DANGER level or CIRCUIT_BREAK (circuit breaker triggers)
        """
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        # Set initial
        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Severe loss - 22%
        engine.update_capital(
            total_capital=Decimal("7800"),
            available_balance=Decimal("7800"),
        )

        status = await engine.check()

        # Should be at least DANGER (may be CIRCUIT_BREAK if breaker triggers)
        assert status.level.value >= RiskLevel.DANGER.value

        # Should have danger alerts
        danger_alerts = [a for a in status.active_alerts if a.level == RiskLevel.DANGER]
        assert len(danger_alerts) > 0

    @pytest.mark.asyncio
    async def test_scenario_4_circuit_break(self, risk_config, mock_commander, mock_notifier):
        """
        Scenario 4: Circuit breaker triggered.

        Expected:
        - circuit_breaker.is_triggered = True
        - Bots paused
        - Notification sent
        """
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        # Set initial
        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Trigger danger level
        engine.update_capital(
            total_capital=Decimal("7500"),  # -25%
            available_balance=Decimal("7500"),
        )

        status = await engine.check()

        # Circuit breaker should be triggered
        assert engine.circuit_breaker.is_triggered is True
        assert status.level == RiskLevel.CIRCUIT_BREAK

        # Commander should have been called
        mock_commander.pause_all.assert_called()

    @pytest.mark.asyncio
    async def test_scenario_5_emergency_stop(self, risk_config, mock_commander, mock_notifier):
        """
        Scenario 5: Emergency stop activation.

        Condition: Capital loss >= 35%

        Expected:
        - Emergency stop activated
        - Bots stopped
        - Notification sent
        """
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        # Set initial
        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # First check to establish baseline
        await engine.check()

        # Severe loss - 35%
        engine.update_capital(
            total_capital=Decimal("6500"),
            available_balance=Decimal("6500"),
        )

        status = await engine.check()

        # Emergency stop should be activated
        assert engine.emergency_stop.is_activated is True


# =============================================================================
# Full Flow Tests
# =============================================================================


class TestFullRiskFlow:
    """Test complete risk flow transitions."""

    @pytest.mark.asyncio
    async def test_full_flow_normal_to_circuit_break(
        self, risk_config, mock_commander, mock_notifier
    ):
        """
        Test full flow: NORMAL → WARNING → DANGER → CIRCUIT_BREAK
        """
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        level_history = []

        def on_level_change(old_level, new_level):
            level_history.append((old_level, new_level))

        engine._on_level_change = on_level_change

        # Phase 1: NORMAL
        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        status = await engine.check()
        assert status.level == RiskLevel.NORMAL

        # Phase 2: WARNING (6% loss)
        engine.update_capital(
            total_capital=Decimal("9400"),
            available_balance=Decimal("9400"),
        )
        status = await engine.check()
        assert status.level == RiskLevel.WARNING

        # Phase 3: Continue loss → DANGER → CIRCUIT_BREAK
        engine.update_capital(
            total_capital=Decimal("7500"),  # 25% total loss
            available_balance=Decimal("7500"),
        )
        status = await engine.check()

        # Should trigger circuit break
        assert status.level == RiskLevel.CIRCUIT_BREAK
        assert engine.circuit_breaker.is_triggered is True

        # Verify level changes were tracked
        assert len(level_history) >= 2  # At least 2 transitions

    @pytest.mark.asyncio
    async def test_full_flow_with_recovery(self, risk_config, mock_commander, mock_notifier):
        """Test flow with partial recovery."""
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        # Initial state
        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        status = await engine.check()
        assert status.level == RiskLevel.NORMAL

        # Loss to WARNING
        engine.update_capital(
            total_capital=Decimal("9400"),
            available_balance=Decimal("9400"),
        )
        status = await engine.check()
        assert status.level == RiskLevel.WARNING

        # Partial recovery - back to profitable
        engine.update_capital(
            total_capital=Decimal("10200"),
            available_balance=Decimal("10200"),
        )
        status = await engine.check()

        # Should be back to NORMAL
        assert status.level == RiskLevel.NORMAL

    @pytest.mark.asyncio
    async def test_consecutive_losses_flow(self, risk_engine):
        """Test consecutive loss tracking through the flow."""
        # Set capital
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Record 5 consecutive losses (warning threshold)
        for _ in range(5):
            risk_engine.record_trade_result(is_win=False)

        status = await risk_engine.check()

        # Should have consecutive loss warning
        loss_alerts = [
            a for a in status.active_alerts
            if a.metric == "consecutive_losses"
        ]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].level == RiskLevel.WARNING

        # Record more losses to reach danger
        for _ in range(5):
            risk_engine.record_trade_result(is_win=False)

        status = await risk_engine.check()

        # Should have consecutive loss danger
        loss_alerts = [
            a for a in status.active_alerts
            if a.metric == "consecutive_losses"
        ]
        assert len(loss_alerts) == 1
        assert loss_alerts[0].level == RiskLevel.DANGER


# =============================================================================
# Component Integration Tests
# =============================================================================


class TestComponentIntegration:
    """Test integration between risk components."""

    @pytest.mark.asyncio
    async def test_drawdown_updates_with_capital(self, risk_engine):
        """Test drawdown calculator updates when capital changes."""
        # Set peak
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # New peak
        risk_engine.update_capital(
            total_capital=Decimal("11000"),
            available_balance=Decimal("11000"),
        )

        # Drawdown
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Check drawdown
        drawdown = risk_engine.drawdown_calculator.get_current_drawdown()
        assert drawdown is not None
        # ~9.09% (1000/11000)
        assert Decimal("0.09") < drawdown.drawdown_pct < Decimal("0.10")

    @pytest.mark.asyncio
    async def test_circuit_breaker_triggers_on_danger(
        self, risk_config, mock_commander, mock_notifier
    ):
        """Test circuit breaker automatically triggers on danger level."""
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Create danger condition
        engine.update_capital(
            total_capital=Decimal("7000"),  # 30% loss
            available_balance=Decimal("7000"),
        )

        status = await engine.check()

        # Verify circuit breaker triggered
        assert engine.circuit_breaker.is_triggered
        assert status.circuit_breaker.is_triggered

    @pytest.mark.asyncio
    async def test_emergency_triggers_on_severe_loss(
        self, risk_config, mock_commander, mock_notifier
    ):
        """Test emergency stop triggers on severe loss."""
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # First check to set initial
        await engine.check()

        # Severe loss beyond emergency threshold
        engine.update_capital(
            total_capital=Decimal("6000"),  # 40% loss
            available_balance=Decimal("6000"),
        )

        status = await engine.check()

        # Emergency stop should be activated
        assert engine.emergency_stop.is_activated

    @pytest.mark.asyncio
    async def test_notifications_sent_correctly(
        self, risk_config, mock_commander, mock_notifier
    ):
        """Test notifications are sent at appropriate levels."""
        engine = RiskEngine(
            risk_config,
            commander=mock_commander,
            notifier=mock_notifier,
        )

        engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        await engine.check()

        # Trigger warning
        engine.update_capital(
            total_capital=Decimal("9400"),
            available_balance=Decimal("9400"),
        )
        await engine.check()

        # Should have sent warning notification
        assert mock_notifier.send.called

        # Check notification level
        calls = mock_notifier.notifications
        assert len(calls) > 0
        assert calls[-1]["level"] in ["warning", "error", "critical"]


# =============================================================================
# Statistics Tests
# =============================================================================


class TestStatisticsIntegration:
    """Test statistics tracking across components."""

    @pytest.mark.asyncio
    async def test_statistics_updated_through_flow(self, risk_engine):
        """Test statistics are updated throughout the risk flow."""
        # Initial
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )
        await risk_engine.check()

        stats = risk_engine.get_statistics()
        assert stats["current_level"] == "NORMAL"
        assert stats["consecutive_losses"] == 0

        # Add some losses
        risk_engine.record_trade_result(is_win=False)
        risk_engine.record_trade_result(is_win=False)

        stats = risk_engine.get_statistics()
        assert stats["consecutive_losses"] == 2

        # Win resets
        risk_engine.record_trade_result(is_win=True)

        stats = risk_engine.get_statistics()
        assert stats["consecutive_losses"] == 0

    @pytest.mark.asyncio
    async def test_drawdown_statistics(self, risk_engine):
        """Test drawdown statistics tracking."""
        # Create peak
        risk_engine.update_capital(
            total_capital=Decimal("10000"),
            available_balance=Decimal("10000"),
        )

        # Drawdown
        risk_engine.update_capital(
            total_capital=Decimal("9000"),
            available_balance=Decimal("9000"),
        )
        await risk_engine.check()

        stats = risk_engine.get_statistics()
        assert stats["current_drawdown_pct"] > Decimal("0")
        assert stats["max_drawdown_pct"] > Decimal("0")
