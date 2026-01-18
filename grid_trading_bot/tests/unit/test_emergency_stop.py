"""
Tests for Emergency Stop.

Tests emergency stop activation, order cancellation, position closure, and bot stopping.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.risk.emergency_stop import (
    EmergencyConfig,
    EmergencyStop,
    EmergencyStopStatus,
)
from src.risk.models import (
    CapitalSnapshot,
    CircuitBreakerState,
    DailyPnL,
    DrawdownInfo,
    GlobalRiskStatus,
    RiskLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def emergency_config():
    """Create an emergency configuration."""
    return EmergencyConfig(
        auto_trigger_loss_pct=Decimal("0.30"),
        auto_close_positions=True,
        max_circuit_triggers=3,
        api_error_threshold=600,
        large_loss_threshold=Decimal("0.05"),
        initial_capital=Decimal("100000"),
    )


@pytest.fixture
def mock_order():
    """Create a mock order."""
    order = MagicMock()
    order.order_id = "order_123"
    order.symbol = "BTCUSDT"
    return order


@pytest.fixture
def mock_position():
    """Create a mock position."""
    position = MagicMock()
    position.symbol = "BTCUSDT"
    position.quantity = Decimal("0.5")
    return position


@pytest.fixture
def mock_exchange(mock_order, mock_position):
    """Create a mock exchange client."""
    exchange = MagicMock()

    # Mock orders
    exchange.get_open_orders = AsyncMock(return_value=[mock_order])
    exchange.cancel_order = AsyncMock(return_value=True)

    # Mock positions
    exchange.get_positions = AsyncMock(return_value=[mock_position])

    # Mock market orders
    closed_order = MagicMock()
    closed_order.order_id = "close_order_123"
    exchange.market_sell = AsyncMock(return_value=closed_order)
    exchange.market_buy = AsyncMock(return_value=closed_order)

    return exchange


@pytest.fixture
def mock_commander():
    """Create a mock bot commander."""
    commander = MagicMock()
    commander.stop_all = AsyncMock(return_value=["bot_1", "bot_2"])
    commander.get_running_bots = MagicMock(return_value=["bot_1", "bot_2"])
    return commander


@pytest.fixture
def mock_notifier():
    """Create a mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def emergency(emergency_config):
    """Create an emergency stop without dependencies."""
    return EmergencyStop(emergency_config)


@pytest.fixture
def emergency_with_deps(emergency_config, mock_commander, mock_exchange, mock_notifier):
    """Create an emergency stop with mock dependencies."""
    return EmergencyStop(
        emergency_config,
        commander=mock_commander,
        exchange=mock_exchange,
        notifier=mock_notifier,
    )


@pytest.fixture
def sample_risk_status():
    """Create a sample global risk status."""
    now = datetime.now()
    today = now.date()

    return GlobalRiskStatus(
        level=RiskLevel.NORMAL,
        capital=CapitalSnapshot(
            timestamp=now,
            total_capital=Decimal("100000"),
            available_balance=Decimal("80000"),
            position_value=Decimal("20000"),
            unrealized_pnl=Decimal("0"),
            realized_pnl=Decimal("0"),
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
            ending_capital=Decimal("100000"),
            pnl=Decimal("0"),
            pnl_pct=Decimal("0"),
            trade_count=0,
            win_count=0,
            loss_count=0,
        ),
        circuit_breaker=CircuitBreakerState(),
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestEmergencyStopInit:
    """Tests for EmergencyStop initialization."""

    def test_init_with_config(self, emergency_config):
        """Test initialization with config."""
        emergency = EmergencyStop(emergency_config)

        assert emergency.config == emergency_config
        assert emergency.is_activated is False
        assert emergency.activated_at is None
        assert emergency.activation_reason is None

    def test_init_with_dependencies(
        self, emergency_config, mock_commander, mock_exchange, mock_notifier
    ):
        """Test initialization with dependencies."""
        emergency = EmergencyStop(
            emergency_config,
            commander=mock_commander,
            exchange=mock_exchange,
            notifier=mock_notifier,
        )

        assert emergency._commander == mock_commander
        assert emergency._exchange == mock_exchange
        assert emergency._notifier == mock_notifier

    def test_init_with_callback(self, emergency_config):
        """Test initialization with callback."""
        callback = MagicMock()
        emergency = EmergencyStop(emergency_config, on_activate=callback)

        assert emergency._on_activate == callback


# =============================================================================
# Activation Tests
# =============================================================================


class TestEmergencyActivation:
    """Tests for emergency stop activation."""

    @pytest.mark.asyncio
    async def test_activate_changes_state(self, emergency):
        """Test that activate changes state."""
        await emergency.activate("Test reason")

        assert emergency.is_activated is True
        assert emergency.activation_reason == "Test reason"
        assert emergency.activated_at is not None

    @pytest.mark.asyncio
    async def test_activate_already_activated(self, emergency):
        """Test activate when already activated."""
        await emergency.activate("First reason")
        await emergency.activate("Second reason")

        # Should keep first reason
        assert emergency.activation_reason == "First reason"

    @pytest.mark.asyncio
    async def test_activate_returns_status(self, emergency):
        """Test activate returns status."""
        status = await emergency.activate("Test")

        assert isinstance(status, EmergencyStopStatus)
        assert status.is_activated is True
        assert status.activation_reason == "Test"

    @pytest.mark.asyncio
    async def test_activate_cancels_orders(self, emergency_with_deps, mock_exchange):
        """Test activate cancels all orders."""
        await emergency_with_deps.activate("Test")

        mock_exchange.get_open_orders.assert_called_once()
        mock_exchange.cancel_order.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_closes_positions(self, emergency_with_deps, mock_exchange):
        """Test activate closes positions."""
        await emergency_with_deps.activate("Test")

        mock_exchange.get_positions.assert_called_once()
        mock_exchange.market_sell.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_stops_bots(self, emergency_with_deps, mock_commander):
        """Test activate stops all bots."""
        await emergency_with_deps.activate("Test")

        mock_commander.stop_all.assert_called_once()

    @pytest.mark.asyncio
    async def test_activate_sends_notification(self, emergency_with_deps, mock_notifier):
        """Test activate sends notification."""
        await emergency_with_deps.activate("Test reason")

        mock_notifier.send.assert_called_once()
        call_args = mock_notifier.send.call_args
        assert "EMERGENCY" in call_args.kwargs["title"]
        assert call_args.kwargs["level"] == "critical"

    @pytest.mark.asyncio
    async def test_activate_calls_callback(self, emergency_config):
        """Test activate calls on_activate callback."""
        callback = MagicMock()
        emergency = EmergencyStop(emergency_config, on_activate=callback)

        await emergency.activate("Test reason")

        callback.assert_called_once_with("Test reason")

    @pytest.mark.asyncio
    async def test_activate_without_auto_close(self, emergency_with_deps, mock_exchange):
        """Test activate without auto close."""
        await emergency_with_deps.activate("Test", auto_close=False)

        # Should not close positions
        mock_exchange.get_positions.assert_not_called()


# =============================================================================
# Cancel Orders Tests
# =============================================================================


class TestCancelOrders:
    """Tests for order cancellation."""

    @pytest.mark.asyncio
    async def test_cancel_all_orders(self, emergency_with_deps, mock_exchange, mock_order):
        """Test cancelling all orders."""
        results = await emergency_with_deps.cancel_all_orders()

        assert len(results) == 1
        assert results[0]["order_id"] == mock_order.order_id
        assert results[0]["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_orders_without_exchange(self, emergency):
        """Test cancel orders without exchange configured."""
        results = await emergency.cancel_all_orders()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_cancel_orders_handles_error(self, emergency_config, mock_notifier):
        """Test cancel orders handles errors gracefully."""
        exchange = MagicMock()
        mock_order = MagicMock()
        mock_order.order_id = "order_123"
        mock_order.symbol = "BTCUSDT"

        exchange.get_open_orders = AsyncMock(return_value=[mock_order])
        exchange.cancel_order = AsyncMock(side_effect=Exception("Cancel failed"))

        emergency = EmergencyStop(
            emergency_config,
            exchange=exchange,
            notifier=mock_notifier,
        )

        results = await emergency.cancel_all_orders()

        assert len(results) == 1
        assert results[0]["status"] == "failed"
        assert "error" in results[0]


# =============================================================================
# Close Positions Tests
# =============================================================================


class TestClosePositions:
    """Tests for position closure."""

    @pytest.mark.asyncio
    async def test_close_all_positions(self, emergency_with_deps, mock_exchange):
        """Test closing all positions."""
        results = await emergency_with_deps.close_all_positions()

        assert len(results) == 1
        assert results[0]["status"] == "closed"
        assert results[0]["side"] == "SELL"  # Long position

    @pytest.mark.asyncio
    async def test_close_positions_without_exchange(self, emergency):
        """Test close positions without exchange configured."""
        results = await emergency.close_all_positions()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_close_short_position(self, emergency_config, mock_notifier):
        """Test closing a short position."""
        exchange = MagicMock()

        # Short position (negative quantity)
        short_pos = MagicMock()
        short_pos.symbol = "BTCUSDT"
        short_pos.quantity = Decimal("-0.5")

        closed_order = MagicMock()
        closed_order.order_id = "close_123"

        exchange.get_positions = AsyncMock(return_value=[short_pos])
        exchange.market_buy = AsyncMock(return_value=closed_order)

        emergency = EmergencyStop(
            emergency_config,
            exchange=exchange,
            notifier=mock_notifier,
        )

        results = await emergency.close_all_positions()

        assert len(results) == 1
        assert results[0]["side"] == "BUY"  # Short position closed with buy

    @pytest.mark.asyncio
    async def test_skip_zero_position(self, emergency_config, mock_notifier):
        """Test skipping zero quantity positions."""
        exchange = MagicMock()

        zero_pos = MagicMock()
        zero_pos.symbol = "BTCUSDT"
        zero_pos.quantity = Decimal("0")

        exchange.get_positions = AsyncMock(return_value=[zero_pos])

        emergency = EmergencyStop(
            emergency_config,
            exchange=exchange,
            notifier=mock_notifier,
        )

        results = await emergency.close_all_positions()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_close_positions_handles_error(self, emergency_config, mock_notifier):
        """Test close positions handles errors gracefully."""
        exchange = MagicMock()

        pos = MagicMock()
        pos.symbol = "BTCUSDT"
        pos.quantity = Decimal("0.5")

        exchange.get_positions = AsyncMock(return_value=[pos])
        exchange.market_sell = AsyncMock(side_effect=Exception("Close failed"))

        emergency = EmergencyStop(
            emergency_config,
            exchange=exchange,
            notifier=mock_notifier,
        )

        results = await emergency.close_all_positions()

        assert len(results) == 1
        assert results[0]["status"] == "failed"


# =============================================================================
# Stop Bots Tests
# =============================================================================


class TestStopBots:
    """Tests for stopping bots."""

    @pytest.mark.asyncio
    async def test_stop_all_bots(self, emergency_with_deps, mock_commander):
        """Test stopping all bots."""
        results = await emergency_with_deps.stop_all_bots()

        mock_commander.stop_all.assert_called_once()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_stop_bots_without_commander(self, emergency):
        """Test stop bots without commander configured."""
        results = await emergency.stop_all_bots()

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_stop_bots_handles_error(self, emergency_config, mock_notifier):
        """Test stop bots handles errors gracefully."""
        commander = MagicMock()
        commander.stop_all = AsyncMock(side_effect=Exception("Stop failed"))

        emergency = EmergencyStop(
            emergency_config,
            commander=commander,
            notifier=mock_notifier,
        )

        results = await emergency.stop_all_bots()

        assert len(results) == 0
        assert len(emergency._errors) == 1


# =============================================================================
# Auto Trigger Tests
# =============================================================================


class TestAutoTrigger:
    """Tests for automatic triggering."""

    def test_check_auto_trigger_no_trigger(self, emergency, sample_risk_status):
        """Test check auto trigger when no conditions met."""
        reason = emergency.check_auto_trigger(sample_risk_status)

        assert reason is None

    def test_check_auto_trigger_loss(self, emergency, sample_risk_status):
        """Test check auto trigger on capital loss."""
        # Set capital to 65% of initial (35% loss)
        sample_risk_status.capital.total_capital = Decimal("65000")

        reason = emergency.check_auto_trigger(sample_risk_status)

        assert reason is not None
        assert "Capital loss" in reason

    def test_check_auto_trigger_circuit_count(self, emergency, sample_risk_status):
        """Test check auto trigger on circuit breaker count."""
        sample_risk_status.circuit_breaker.trigger_count_today = 3

        reason = emergency.check_auto_trigger(sample_risk_status)

        assert reason is not None
        assert "Circuit breaker" in reason

    def test_check_auto_trigger_api_error(self, emergency, sample_risk_status):
        """Test check auto trigger on API error duration."""
        reason = emergency.check_auto_trigger(
            sample_risk_status,
            api_error_duration=700,  # > 600 threshold
        )

        assert reason is not None
        assert "API errors" in reason

    def test_check_auto_trigger_already_activated(self, emergency, sample_risk_status):
        """Test check auto trigger when already activated."""
        emergency._is_activated = True
        sample_risk_status.capital.total_capital = Decimal("50000")  # 50% loss

        reason = emergency.check_auto_trigger(sample_risk_status)

        assert reason is None

    @pytest.mark.asyncio
    async def test_auto_trigger_if_needed(self, emergency, sample_risk_status):
        """Test auto_trigger_if_needed triggers when conditions met."""
        sample_risk_status.capital.total_capital = Decimal("65000")

        triggered = await emergency.auto_trigger_if_needed(sample_risk_status)

        assert triggered is True
        assert emergency.is_activated is True

    @pytest.mark.asyncio
    async def test_auto_trigger_if_needed_no_trigger(self, emergency, sample_risk_status):
        """Test auto_trigger_if_needed does not trigger when conditions not met."""
        triggered = await emergency.auto_trigger_if_needed(sample_risk_status)

        assert triggered is False
        assert emergency.is_activated is False


# =============================================================================
# Status Tests
# =============================================================================


class TestEmergencyStatus:
    """Tests for status retrieval."""

    @pytest.mark.asyncio
    async def test_get_status_not_activated(self, emergency):
        """Test get_status when not activated."""
        status = emergency.get_status()

        assert status.is_activated is False
        assert status.activated_at is None
        assert status.activation_reason == ""
        assert status.cancelled_orders == 0
        assert status.closed_positions == 0
        assert status.stopped_bots == 0

    @pytest.mark.asyncio
    async def test_get_status_activated(self, emergency_with_deps):
        """Test get_status when activated."""
        await emergency_with_deps.activate("Test reason")

        status = emergency_with_deps.get_status()

        assert status.is_activated is True
        assert status.activation_reason == "Test reason"
        assert status.cancelled_orders == 1
        assert status.closed_positions == 1
        assert status.stopped_bots == 2


# =============================================================================
# Reset Tests
# =============================================================================


class TestEmergencyReset:
    """Tests for reset functionality."""

    @pytest.mark.asyncio
    async def test_reset_clears_state(self, emergency):
        """Test reset clears state."""
        await emergency.activate("Test")

        emergency.reset()

        assert emergency.is_activated is False
        assert emergency.activated_at is None
        assert emergency.activation_reason is None

    @pytest.mark.asyncio
    async def test_reset_clears_results(self, emergency_with_deps):
        """Test reset clears action results."""
        await emergency_with_deps.activate("Test")

        emergency_with_deps.reset()

        assert len(emergency_with_deps._cancel_results) == 0
        assert len(emergency_with_deps._close_results) == 0
        assert len(emergency_with_deps._stop_results) == 0


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self, emergency_config):
        """Test callback error doesn't crash activation."""

        def bad_callback(reason):
            raise Exception("Callback error")

        emergency = EmergencyStop(emergency_config, on_activate=bad_callback)

        # Should not crash
        await emergency.activate("Test")

        assert emergency.is_activated is True

    @pytest.mark.asyncio
    async def test_notification_error_does_not_crash(
        self, emergency_config, mock_commander, mock_exchange
    ):
        """Test notification error doesn't crash activation."""
        notifier = MagicMock()
        notifier.send = AsyncMock(side_effect=Exception("Notify failed"))

        emergency = EmergencyStop(
            emergency_config,
            commander=mock_commander,
            exchange=mock_exchange,
            notifier=notifier,
        )

        # Should not crash
        await emergency.activate("Test")

        assert emergency.is_activated is True

    def test_set_initial_capital(self, emergency):
        """Test setting initial capital."""
        emergency.set_initial_capital(Decimal("200000"))

        assert emergency.config.initial_capital == Decimal("200000")


# =============================================================================
# EmergencyConfig Tests
# =============================================================================


class TestEmergencyConfig:
    """Tests for EmergencyConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = EmergencyConfig()

        assert config.auto_trigger_loss_pct == Decimal("0.30")
        assert config.auto_close_positions is True
        assert config.max_circuit_triggers == 3
        assert config.api_error_threshold == 600
        assert config.large_loss_threshold == Decimal("0.05")

    def test_custom_values(self):
        """Test custom configuration values."""
        config = EmergencyConfig(
            auto_trigger_loss_pct=Decimal("0.25"),
            auto_close_positions=False,
            max_circuit_triggers=5,
            api_error_threshold=300,
            large_loss_threshold=Decimal("0.10"),
            initial_capital=Decimal("50000"),
        )

        assert config.auto_trigger_loss_pct == Decimal("0.25")
        assert config.auto_close_positions is False
        assert config.max_circuit_triggers == 5
        assert config.initial_capital == Decimal("50000")
