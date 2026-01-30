"""
Tests for BaseBot abstract class.

Validates the template method pattern, state transitions,
heartbeat mechanism, and health check framework.
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.bots.base import BaseBot, BotStats, InvalidStateError
from src.master.models import BotState


# =============================================================================
# Test Implementation of BaseBot
# =============================================================================


class MockBot(BaseBot):
    """Concrete implementation of BaseBot for testing."""

    def __init__(
        self,
        bot_id: str = "test_bot",
        config: Optional[Any] = None,
        exchange: Optional[Any] = None,
        data_manager: Optional[Any] = None,
        notifier: Optional[Any] = None,
        heartbeat_callback: Optional[callable] = None,
    ):
        super().__init__(
            bot_id=bot_id,
            config=config or {"symbol": "BTCUSDT"},
            exchange=exchange or MagicMock(),
            data_manager=data_manager or MagicMock(),
            notifier=notifier or MagicMock(),
            heartbeat_callback=heartbeat_callback,
        )
        self.do_start_called = False
        self.do_stop_called = False
        self.do_pause_called = False
        self.do_resume_called = False
        self.clear_position_value = None

    @property
    def bot_type(self) -> str:
        return "mock"

    @property
    def symbol(self) -> str:
        return self._config.get("symbol", "UNKNOWN")

    async def _do_start(self) -> None:
        self.do_start_called = True

    async def _do_stop(self, clear_position: bool = False) -> None:
        self.do_stop_called = True
        self.clear_position_value = clear_position

    async def _do_pause(self) -> None:
        self.do_pause_called = True

    async def _do_resume(self) -> None:
        self.do_resume_called = True

    def _get_extra_status(self) -> Dict[str, Any]:
        return {"mock_field": "mock_value"}

    async def _extra_health_checks(self) -> Dict[str, bool]:
        return {"mock_check": True}


# =============================================================================
# BotStats Tests
# =============================================================================


class TestBotStats:
    """Tests for BotStats dataclass."""

    def test_default_values(self):
        """Test default values are initialized correctly."""
        stats = BotStats()
        assert stats.total_trades == 0
        assert stats.total_profit == Decimal("0")
        assert stats.total_fees == Decimal("0")
        assert stats.today_trades == 0
        assert stats.today_profit == Decimal("0")
        assert stats.start_time is None
        assert stats.last_trade_at is None

    def test_record_trade(self):
        """Test recording a trade updates statistics."""
        stats = BotStats()
        stats.record_trade(profit=Decimal("10.5"), fee=Decimal("0.1"))

        assert stats.total_trades == 1
        assert stats.total_profit == Decimal("10.5")
        assert stats.total_fees == Decimal("0.1")
        assert stats.today_trades == 1
        assert stats.today_profit == Decimal("10.5")
        assert stats.last_trade_at is not None

    def test_record_multiple_trades(self):
        """Test recording multiple trades accumulates correctly."""
        stats = BotStats()
        stats.record_trade(profit=Decimal("10"), fee=Decimal("0.1"))
        stats.record_trade(profit=Decimal("5"), fee=Decimal("0.05"))
        stats.record_trade(profit=Decimal("-3"), fee=Decimal("0.08"))

        assert stats.total_trades == 3
        assert stats.total_profit == Decimal("12")
        assert stats.total_fees == Decimal("0.23")
        assert stats.today_trades == 3
        assert stats.today_profit == Decimal("12")

    def test_reset_daily(self):
        """Test resetting daily statistics."""
        stats = BotStats()
        stats.record_trade(profit=Decimal("10"), fee=Decimal("0.1"))
        stats.record_trade(profit=Decimal("5"), fee=Decimal("0.05"))

        stats.reset_daily()

        assert stats.total_trades == 2  # Total unchanged
        assert stats.total_profit == Decimal("15")  # Total unchanged
        assert stats.today_trades == 0  # Reset
        assert stats.today_profit == Decimal("0")  # Reset

    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = BotStats()
        stats.total_trades = 5
        stats.total_profit = Decimal("100.50")
        stats.total_fees = Decimal("1.25")
        stats.start_time = datetime(2026, 1, 18, 12, 0, 0, tzinfo=timezone.utc)

        result = stats.to_dict()

        assert result["total_trades"] == 5
        assert result["total_profit"] == "100.50"
        assert result["total_fees"] == "1.25"
        assert "2026-01-18" in result["start_time"]


# =============================================================================
# BaseBot Initialization Tests
# =============================================================================


class TestBaseBotInitialization:
    """Tests for BaseBot initialization."""

    def test_initial_state_is_registered(self):
        """Test bot starts in REGISTERED state."""
        bot = MockBot()
        assert bot.state == BotState.REGISTERED

    def test_properties(self):
        """Test read-only properties."""
        bot = MockBot(bot_id="test_123")
        assert bot.bot_id == "test_123"
        assert bot.bot_type == "mock"
        assert bot.symbol == "BTCUSDT"
        assert bot.is_running is False
        assert bot.error_message is None

    def test_stats_initialized(self):
        """Test stats are initialized."""
        bot = MockBot()
        assert isinstance(bot.stats, BotStats)
        assert bot.stats.total_trades == 0


# =============================================================================
# BaseBot Lifecycle Tests
# =============================================================================


class TestBaseBotStart:
    """Tests for BaseBot.start()."""

    @pytest.mark.asyncio
    async def test_start_from_registered(self):
        """Test starting from REGISTERED state."""
        bot = MockBot()
        result = await bot.start()

        assert result is True
        assert bot.state == BotState.RUNNING
        assert bot.is_running is True
        assert bot.do_start_called is True
        assert bot.stats.start_time is not None

    @pytest.mark.asyncio
    async def test_start_from_stopped(self):
        """Test starting from STOPPED state."""
        bot = MockBot()
        bot._state = BotState.STOPPED

        result = await bot.start()

        assert result is True
        assert bot.state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_start_from_invalid_state(self):
        """Test starting from invalid state raises error."""
        bot = MockBot()
        bot._state = BotState.RUNNING

        with pytest.raises(InvalidStateError):
            await bot.start()

    @pytest.mark.asyncio
    async def test_start_failure_sets_error_state(self):
        """Test start failure sets ERROR state and returns False."""
        bot = MockBot()

        async def failing_start():
            raise RuntimeError("Start failed")

        bot._do_start = failing_start

        result = await bot.start()

        assert result is False
        assert bot.state == BotState.ERROR
        assert "Start failed" in bot.error_message


class TestBaseBotStop:
    """Tests for BaseBot.stop()."""

    @pytest.mark.asyncio
    async def test_stop_from_running(self):
        """Test stopping from RUNNING state."""
        bot = MockBot()
        await bot.start()

        result = await bot.stop()

        assert result is True
        assert bot.state == BotState.STOPPED
        assert bot.is_running is False
        assert bot.do_stop_called is True

    @pytest.mark.asyncio
    async def test_stop_from_paused(self):
        """Test stopping from PAUSED state."""
        bot = MockBot()
        await bot.start()
        await bot.pause()

        result = await bot.stop()

        assert result is True
        assert bot.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_from_error(self):
        """Test stopping from ERROR state."""
        bot = MockBot()
        bot._state = BotState.ERROR
        bot._running = True

        result = await bot.stop()

        assert result is True
        assert bot.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_with_clear_position(self):
        """Test stop passes clear_position to _do_stop."""
        bot = MockBot()
        await bot.start()

        await bot.stop(clear_position=True)

        assert bot.clear_position_value is True

    @pytest.mark.asyncio
    async def test_stop_from_invalid_state(self):
        """Test stopping from invalid state raises error."""
        bot = MockBot()

        with pytest.raises(InvalidStateError):
            await bot.stop()


class TestBaseBotPauseResume:
    """Tests for BaseBot.pause() and resume()."""

    @pytest.mark.asyncio
    async def test_pause_from_running(self):
        """Test pausing from RUNNING state."""
        bot = MockBot()
        await bot.start()

        result = await bot.pause()

        assert result is True
        assert bot.state == BotState.PAUSED
        assert bot.do_pause_called is True

    @pytest.mark.asyncio
    async def test_pause_from_invalid_state(self):
        """Test pausing from invalid state raises error."""
        bot = MockBot()

        with pytest.raises(InvalidStateError):
            await bot.pause()

    @pytest.mark.asyncio
    async def test_resume_from_paused(self):
        """Test resuming from PAUSED state."""
        bot = MockBot()
        await bot.start()
        await bot.pause()

        result = await bot.resume()

        assert result is True
        assert bot.state == BotState.RUNNING
        assert bot.do_resume_called is True

    @pytest.mark.asyncio
    async def test_resume_from_invalid_state(self):
        """Test resuming from invalid state raises error."""
        bot = MockBot()
        await bot.start()

        with pytest.raises(InvalidStateError):
            await bot.resume()


# =============================================================================
# BaseBot Status Tests
# =============================================================================


class TestBaseBotStatus:
    """Tests for BaseBot.get_status()."""

    @pytest.mark.asyncio
    async def test_get_status_includes_base_fields(self):
        """Test get_status includes all base fields."""
        bot = MockBot(bot_id="status_test")
        await bot.start()

        status = bot.get_status()

        assert status["bot_id"] == "status_test"
        assert status["bot_type"] == "mock"
        assert status["symbol"] == "BTCUSDT"
        assert status["state"] == "running"
        assert "total_trades" in status
        assert "total_profit" in status
        assert "uptime" in status

    @pytest.mark.asyncio
    async def test_get_status_includes_extra_fields(self):
        """Test get_status includes subclass extra fields."""
        bot = MockBot()
        await bot.start()

        status = bot.get_status()

        assert status["mock_field"] == "mock_value"


# =============================================================================
# BaseBot Heartbeat Tests
# =============================================================================


class TestBaseBotHeartbeat:
    """Tests for BaseBot heartbeat mechanism."""

    @pytest.mark.asyncio
    async def test_heartbeat_callback_called(self):
        """Test heartbeat callback is invoked."""
        received_heartbeats = []

        def callback(hb):
            received_heartbeats.append(hb)

        bot = MockBot(heartbeat_callback=callback)
        await bot.start()

        # Manually trigger heartbeat (now async)
        await bot._send_heartbeat()

        assert len(received_heartbeats) == 1
        assert received_heartbeats[0].bot_id == "test_bot"

    @pytest.mark.asyncio
    async def test_no_heartbeat_without_callback(self):
        """Test no heartbeat task without callback."""
        bot = MockBot(heartbeat_callback=None)
        await bot.start()

        assert bot._heartbeat_task is None

    @pytest.mark.asyncio
    async def test_heartbeat_stopped_on_stop(self):
        """Test heartbeat is stopped when bot stops."""
        bot = MockBot(heartbeat_callback=lambda x: None)
        await bot.start()

        assert bot._heartbeat_task is not None

        await bot.stop()

        assert bot._heartbeat_task is None


# =============================================================================
# BaseBot Health Check Tests
# =============================================================================


class TestBaseBotHealthCheck:
    """Tests for BaseBot health check framework."""

    @pytest.mark.asyncio
    async def test_health_check_when_running(self):
        """Test health check when bot is running."""
        bot = MockBot()
        bot._exchange.get_account = AsyncMock(return_value={})
        await bot.start()

        result = await bot.health_check()

        assert result["healthy"] is True
        assert result["checks"]["state"] is True
        assert result["checks"]["exchange"] is True
        assert result["checks"]["mock_check"] is True

    @pytest.mark.asyncio
    async def test_health_check_state_not_running(self):
        """Test health check fails state check when not running."""
        bot = MockBot()
        bot._exchange.get_account = AsyncMock(return_value={})

        result = await bot.health_check()

        assert result["healthy"] is False
        assert result["checks"]["state"] is False

    @pytest.mark.asyncio
    async def test_health_check_exchange_failure(self):
        """Test health check handles exchange failure."""
        bot = MockBot()
        bot._exchange.get_account = AsyncMock(side_effect=Exception("Connection failed"))
        await bot.start()

        result = await bot.health_check()

        assert result["checks"]["exchange"] is False


# =============================================================================
# BaseBot Helper Methods Tests
# =============================================================================


class TestBaseBotHelpers:
    """Tests for BaseBot helper methods."""

    def test_get_uptime_no_start(self):
        """Test uptime when not started."""
        bot = MockBot()
        assert bot._get_uptime() == "0s"

    @pytest.mark.asyncio
    async def test_get_uptime_seconds(self):
        """Test uptime in seconds."""
        bot = MockBot()
        await bot.start()

        # Should be at least 0
        assert bot._get_uptime_seconds() >= 0

    def test_get_uptime_format(self):
        """Test uptime formatting."""
        bot = MockBot()
        bot._stats.start_time = datetime.now(timezone.utc)

        uptime = bot._get_uptime()
        assert "s" in uptime or "m" in uptime


# =============================================================================
# InvalidStateError Tests
# =============================================================================


class TestInvalidStateError:
    """Tests for InvalidStateError exception."""

    def test_error_message(self):
        """Test error message is preserved."""
        error = InvalidStateError("Cannot start from running")
        assert str(error) == "Cannot start from running"
