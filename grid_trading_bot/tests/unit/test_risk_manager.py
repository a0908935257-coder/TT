"""
Unit tests for Grid Risk Manager.

Tests breakout detection, risk actions, and bot state management.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import Kline, MarketType, OrderSide
from src.bots.grid import (
    BotState,
    BreakoutAction,
    BreakoutDirection,
    FilledRecord,
    GridOrderManager,
    GridRiskManager,
    RiskConfig,
    RiskState,
    VALID_STATE_TRANSITIONS,
    create_grid_with_manual_range,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


@pytest.fixture
def risk_config():
    """Create a risk configuration for testing."""
    return RiskConfig(
        upper_breakout_action=BreakoutAction.PAUSE,
        lower_breakout_action=BreakoutAction.STOP_LOSS,
        stop_loss_percent=Decimal("20"),
        breakout_buffer=Decimal("0.5"),
        daily_loss_limit=Decimal("5"),
        max_consecutive_losses=5,
        volatility_threshold=Decimal("10"),
        order_failure_threshold=3,
    )


@pytest.fixture
def order_manager_with_setup(mock_exchange, mock_data_manager, mock_notifier):
    """Create an order manager with grid setup."""
    setup = create_grid_with_manual_range(
        symbol="BTCUSDT",
        investment=10000,
        upper_price=55000,
        lower_price=45000,
        grid_count=10,
        current_price=50000,
    )

    manager = GridOrderManager(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
        bot_id="test_bot",
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
    )
    manager.initialize(setup)

    return manager


@pytest.fixture
def risk_manager(order_manager_with_setup, mock_notifier, risk_config):
    """Create a risk manager for testing."""
    return GridRiskManager(
        order_manager=order_manager_with_setup,
        notifier=mock_notifier,
        config=risk_config,
    )


class TestBreakoutDetection:
    """Tests for breakout detection."""

    def test_detect_upper_breakout(self, risk_manager):
        """Test upper breakout detection."""
        # Price above upper threshold (55000 * 1.005 = 55275)
        price = Decimal("56000")
        direction = risk_manager.check_breakout(price)

        assert direction == BreakoutDirection.UPPER

    def test_detect_lower_breakout(self, risk_manager):
        """Test lower breakout detection."""
        # Price below lower threshold (45000 * 0.995 = 44775)
        price = Decimal("44000")
        direction = risk_manager.check_breakout(price)

        assert direction == BreakoutDirection.LOWER

    def test_no_breakout_within_range(self, risk_manager):
        """Test no breakout when price is within range."""
        price = Decimal("50000")
        direction = risk_manager.check_breakout(price)

        assert direction == BreakoutDirection.NONE

    def test_breakout_buffer(self, risk_manager):
        """Test breakout buffer threshold."""
        # Just at upper bound (no buffer)
        at_upper = Decimal("55000")
        assert risk_manager.check_breakout(at_upper) == BreakoutDirection.NONE

        # Just past buffer (0.5%)
        past_buffer = Decimal("55300")  # 55000 * 1.005 = 55275
        assert risk_manager.check_breakout(past_buffer) == BreakoutDirection.UPPER


class TestBreakoutActions:
    """Tests for breakout action handling."""

    @pytest.mark.asyncio
    async def test_pause_action(self, risk_manager):
        """Test pause action on breakout."""
        # Set upper breakout action to PAUSE
        risk_manager._config.upper_breakout_action = BreakoutAction.PAUSE

        await risk_manager.handle_breakout(
            direction=BreakoutDirection.UPPER,
            current_price=Decimal("56000"),
        )

        assert risk_manager.state == RiskState.PAUSED

    @pytest.mark.asyncio
    async def test_hold_action(self, risk_manager):
        """Test hold action on breakout."""
        risk_manager._config.upper_breakout_action = BreakoutAction.HOLD

        await risk_manager.handle_breakout(
            direction=BreakoutDirection.UPPER,
            current_price=Decimal("56000"),
        )

        # State should be BREAKOUT_UPPER, not PAUSED
        assert risk_manager.state == RiskState.BREAKOUT_UPPER

    @pytest.mark.asyncio
    async def test_stop_loss_action(self, risk_manager, order_manager_with_setup):
        """Test stop loss action on breakout."""
        risk_manager._config.lower_breakout_action = BreakoutAction.STOP_LOSS

        # Add a filled buy to simulate position
        buy_record = FilledRecord(
            level_index=3,
            side=OrderSide.BUY,
            price=Decimal("48000"),
            quantity=Decimal("0.1"),
            fee=Decimal("4.8"),
            timestamp=datetime.now(timezone.utc),
            order_id="test_buy",
        )
        order_manager_with_setup._filled_history.append(buy_record)

        await risk_manager.handle_breakout(
            direction=BreakoutDirection.LOWER,
            current_price=Decimal("44000"),
        )

        assert risk_manager.state == RiskState.STOPPED


class TestDailyLossLimit:
    """Tests for daily loss limit."""

    def test_daily_loss_under_limit(self, risk_manager):
        """Test daily loss check under limit."""
        risk_manager._daily_pnl = Decimal("-400")  # 4% loss on 10000

        assert risk_manager.check_daily_loss() is False

    def test_daily_loss_at_limit(self, risk_manager):
        """Test daily loss check at limit."""
        risk_manager._daily_pnl = Decimal("-500")  # Exactly 5% loss

        assert risk_manager.check_daily_loss() is True

    def test_daily_loss_exceed_limit(self, risk_manager):
        """Test daily loss check exceeding limit."""
        risk_manager._daily_pnl = Decimal("-600")  # 6% loss

        assert risk_manager.check_daily_loss() is True


class TestConsecutiveLosses:
    """Tests for consecutive loss tracking."""

    def test_consecutive_losses_under_threshold(self, risk_manager):
        """Test consecutive losses under threshold."""
        risk_manager._consecutive_losses = 3

        assert risk_manager.check_consecutive_losses() is False

    def test_consecutive_losses_at_threshold(self, risk_manager):
        """Test consecutive losses at threshold."""
        risk_manager._consecutive_losses = 5

        assert risk_manager.check_consecutive_losses() is True

    def test_on_trade_completed_profit(self, risk_manager):
        """Test trade completed with profit resets counter."""
        risk_manager._consecutive_losses = 3

        risk_manager.on_trade_completed(Decimal("50"))  # Profit

        assert risk_manager._consecutive_losses == 0
        assert risk_manager._last_trade_profitable is True

    def test_on_trade_completed_loss(self, risk_manager):
        """Test trade completed with loss increments counter."""
        risk_manager._consecutive_losses = 2
        risk_manager._last_trade_profitable = False

        risk_manager.on_trade_completed(Decimal("-50"))  # Loss

        assert risk_manager._consecutive_losses == 3


class TestVolatilityCheck:
    """Tests for volatility checking."""

    def test_volatility_under_threshold(self, risk_manager):
        """Test volatility under threshold."""
        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("52000"),  # 4% range
            low=Decimal("50000"),
            close=Decimal("51000"),
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )

        assert risk_manager.check_volatility(kline) is False

    def test_volatility_exceed_threshold(self, risk_manager):
        """Test volatility exceeding threshold."""
        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=datetime.now(timezone.utc),
            close_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("57000"),  # 14% range
            low=Decimal("50000"),
            close=Decimal("55000"),
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )

        assert risk_manager.check_volatility(kline) is True


class TestPriceReturn:
    """Tests for price return detection."""

    def test_price_return_from_upper_breakout(self, risk_manager):
        """Test price return detection from upper breakout."""
        # Set state to upper breakout
        risk_manager._state = RiskState.BREAKOUT_UPPER

        # Price returned to range
        returned = risk_manager.check_price_return(Decimal("54000"))
        assert returned is True

        # Price still outside
        still_out = risk_manager.check_price_return(Decimal("56000"))
        assert still_out is False

    def test_price_return_from_lower_breakout(self, risk_manager):
        """Test price return detection from lower breakout."""
        risk_manager._state = RiskState.BREAKOUT_LOWER

        # Price returned to range
        returned = risk_manager.check_price_return(Decimal("46000"))
        assert returned is True

        # Price still outside
        still_out = risk_manager.check_price_return(Decimal("43000"))
        assert still_out is False


class TestStateTransitions:
    """Tests for bot state transitions."""

    def test_valid_state_transitions_defined(self):
        """Test that operational states have defined transitions."""
        # REGISTERED is managed by Master, not by GridRiskManager
        operational_states = [s for s in BotState if s != BotState.REGISTERED]
        for state in operational_states:
            assert state in VALID_STATE_TRANSITIONS

    def test_initializing_transitions(self):
        """Test valid transitions from INITIALIZING."""
        valid = VALID_STATE_TRANSITIONS[BotState.INITIALIZING]

        assert BotState.RUNNING in valid
        assert BotState.ERROR in valid
        assert BotState.STOPPED in valid
        assert BotState.PAUSED not in valid

    def test_running_transitions(self):
        """Test valid transitions from RUNNING."""
        valid = VALID_STATE_TRANSITIONS[BotState.RUNNING]

        assert BotState.PAUSED in valid
        assert BotState.STOPPING in valid
        assert BotState.ERROR in valid
        assert BotState.INITIALIZING not in valid

    def test_stopped_is_terminal(self):
        """Test that STOPPED is a terminal state."""
        valid = VALID_STATE_TRANSITIONS[BotState.STOPPED]

        assert len(valid) == 0

    @pytest.mark.asyncio
    async def test_change_state_valid(self, risk_manager):
        """Test valid state change."""
        risk_manager._bot_state = BotState.INITIALIZING

        result = await risk_manager.change_state(BotState.RUNNING, "Started")

        assert result is True
        assert risk_manager.bot_state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_change_state_invalid(self, risk_manager):
        """Test invalid state change is rejected."""
        risk_manager._bot_state = BotState.STOPPED

        result = await risk_manager.change_state(BotState.RUNNING, "Restart")

        assert result is False
        assert risk_manager.bot_state == BotState.STOPPED


class TestManualControl:
    """Tests for manual bot control."""

    @pytest.mark.asyncio
    async def test_pause(self, risk_manager):
        """Test manual pause."""
        risk_manager._bot_state = BotState.RUNNING

        result = await risk_manager.pause("Test pause")

        assert result is True
        assert risk_manager.bot_state == BotState.PAUSED
        assert risk_manager.state == RiskState.PAUSED
        assert "Test pause" in risk_manager.pause_reason

    @pytest.mark.asyncio
    async def test_resume(self, risk_manager, mock_data_manager):
        """Test resume from pause."""
        risk_manager._bot_state = BotState.PAUSED
        risk_manager._state = RiskState.PAUSED

        # Set price within range
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))

        result = await risk_manager.resume()

        assert result is True
        assert risk_manager.bot_state == BotState.RUNNING
        assert risk_manager.state == RiskState.NORMAL

    @pytest.mark.asyncio
    async def test_resume_blocked_during_breakout(self, risk_manager, mock_data_manager):
        """Test resume is blocked if still in breakout."""
        risk_manager._bot_state = BotState.PAUSED

        # Price still in breakout
        mock_data_manager.set_price("BTCUSDT", Decimal("60000"))

        result = await risk_manager.resume()

        assert result is False

    @pytest.mark.asyncio
    async def test_stop(self, risk_manager):
        """Test graceful stop."""
        risk_manager._bot_state = BotState.RUNNING

        result = await risk_manager.stop()

        assert result is True
        assert risk_manager.bot_state == BotState.STOPPED
        assert risk_manager.state == RiskState.STOPPED

    @pytest.mark.asyncio
    async def test_force_stop(self, risk_manager):
        """Test force stop."""
        risk_manager._bot_state = BotState.RUNNING

        result = await risk_manager.force_stop()

        assert result is True
        assert risk_manager.bot_state == BotState.STOPPED


class TestStopLoss:
    """Tests for stop loss calculation."""

    def test_check_stop_loss_under_threshold(self, risk_manager, order_manager_with_setup):
        """Test stop loss not triggered under threshold."""
        # Add a position with small loss
        buy_record = FilledRecord(
            level_index=5,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.1"),
            fee=Decimal("5"),
            timestamp=datetime.now(timezone.utc),
            order_id="test_buy",
        )
        order_manager_with_setup._filled_history.append(buy_record)

        # 5% drop = 47500, loss = 250 = 2.5% of 10000 investment
        result = risk_manager.check_stop_loss(Decimal("47500"))

        assert result is False

    def test_check_stop_loss_exceed_threshold(self, risk_manager, order_manager_with_setup):
        """Test stop loss triggered when exceeding threshold."""
        # Add a large position
        buy_record = FilledRecord(
            level_index=5,
            side=OrderSide.BUY,
            price=Decimal("50000"),
            quantity=Decimal("0.5"),  # Half of investment at 50000
            fee=Decimal("25"),
            timestamp=datetime.now(timezone.utc),
            order_id="test_buy",
        )
        order_manager_with_setup._filled_history.append(buy_record)

        # 50% drop (extreme), loss = 12500 = 125% of 10000 investment
        result = risk_manager.check_stop_loss(Decimal("25000"))

        assert result is True


class TestDailyStatsReset:
    """Tests for daily statistics reset."""

    @pytest.mark.asyncio
    async def test_reset_daily_stats(self, risk_manager):
        """Test daily stats reset."""
        risk_manager._daily_pnl = Decimal("-500")
        risk_manager._consecutive_losses = 3
        risk_manager._order_failures = 2

        await risk_manager.reset_daily_stats()

        assert risk_manager._daily_pnl == Decimal("0")
        assert risk_manager._consecutive_losses == 0
        assert risk_manager._order_failures == 0
