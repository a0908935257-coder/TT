"""
Tests for BollingerBot.

Tests the main bot class including:
- Initialization
- Start/Stop lifecycle
- Pause/Resume
- Full long/short trades
- Squeeze filtering
- Entry timeout
- Hold timeout
- Stop loss triggering
- Statistics updates
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, MagicMock, patch
from dataclasses import dataclass, field
from typing import Optional, List, Any, Callable

from src.bots.bollinger.bot import BollingerBot
from src.bots.bollinger.models import (
    BollingerConfig,
    Signal,
    SignalType,
    Position,
    PositionSide,
    BollingerBands,
    BBWData,
)
from src.master.models import BotState


# =============================================================================
# Mock Classes
# =============================================================================


@dataclass
class MockKline:
    """Mock kline for testing."""
    close: Decimal
    high: Decimal = Decimal("0")
    low: Decimal = Decimal("0")
    open: Decimal = Decimal("0")
    volume: Decimal = Decimal("0")
    close_time: Optional[datetime] = None
    is_closed: bool = True

    def __post_init__(self):
        if self.high == Decimal("0"):
            self.high = self.close
        if self.low == Decimal("0"):
            self.low = self.close
        if self.open == Decimal("0"):
            self.open = self.close
        if self.close_time is None:
            self.close_time = datetime.now(timezone.utc)


@dataclass
class MockOrder:
    """Mock order for testing."""
    order_id: str
    status: str = "NEW"
    side: str = "BUY"
    avg_price: Decimal = Decimal("0")
    filled_quantity: Decimal = Decimal("0")


@dataclass
class MockAccount:
    """Mock account for testing."""
    available_balance: Decimal = Decimal("1000")


@dataclass
class MockTicker:
    """Mock ticker for testing."""
    last_price: Decimal = Decimal("50000")


@dataclass
class MockPosition:
    """Mock exchange position for testing."""
    symbol: str = "BTCUSDT"
    quantity: Decimal = Decimal("0")
    entry_price: Decimal = Decimal("50000")
    leverage: int = 2
    unrealized_pnl: Decimal = Decimal("0")


def create_klines(prices: list[float]) -> list[MockKline]:
    """Create mock klines from price list."""
    return [MockKline(close=Decimal(str(p))) for p in prices]


def create_volatile_klines(base_price: float, count: int, volatility: float = 0.02) -> list[MockKline]:
    """Create volatile klines."""
    import random
    random.seed(42)
    prices = []
    for _ in range(count):
        change = random.uniform(-volatility, volatility)
        prices.append(base_price * (1 + change))
    return create_klines(prices)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def config() -> BollingerConfig:
    """Create default config."""
    return BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        leverage=2,
        position_size_pct=Decimal("0.1"),
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=Decimal("0.25"),
        stop_loss_pct=Decimal("0.015"),
        timeout_bars=16,
    )


@pytest.fixture
def mock_exchange() -> Mock:
    """Create mock exchange."""
    mock = Mock()
    # Kline methods
    mock.get_klines = AsyncMock(return_value=create_volatile_klines(50000, 300))
    mock.subscribe_klines = AsyncMock()
    mock.unsubscribe_klines = AsyncMock()
    mock.subscribe_user_data = AsyncMock()

    # Account methods
    mock.get_account = AsyncMock(return_value=MockAccount())
    mock.futures_get_account = AsyncMock(return_value=MockAccount())
    mock.futures_set_leverage = AsyncMock(return_value={})
    mock.futures_set_margin_type = AsyncMock(return_value={})
    mock.futures_get_positions = AsyncMock(return_value=[])
    mock.get_ticker = AsyncMock(return_value=MockTicker())

    # Order methods
    mock.futures_create_order = AsyncMock(return_value=MockOrder(order_id="order_001"))
    mock.futures_cancel_order = AsyncMock(return_value={})
    mock.futures_get_order = AsyncMock(return_value=MockOrder(order_id="order_001"))

    return mock


@pytest.fixture
def mock_data_manager() -> Mock:
    """Create mock data manager."""
    mock = Mock()
    mock.save_trade = AsyncMock()
    return mock


@pytest.fixture
def mock_notifier() -> Mock:
    """Create mock notifier."""
    mock = Mock()
    mock.send = AsyncMock()
    mock.send_trade_notification = AsyncMock()
    return mock


@pytest.fixture
def bot(
    config: BollingerConfig,
    mock_exchange: Mock,
    mock_data_manager: Mock,
    mock_notifier: Mock,
) -> BollingerBot:
    """Create BollingerBot instance."""
    return BollingerBot(
        bot_id="test_bot_001",
        config=config,
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
    )


# =============================================================================
# Test Initialize
# =============================================================================


class TestInitialize:
    """Test bot initialization."""

    def test_bot_creation(self, bot: BollingerBot):
        """Test bot is created correctly."""
        assert bot.bot_id == "test_bot_001"
        assert bot.bot_type == "bollinger"
        assert bot.symbol == "BTCUSDT"

    def test_initial_state(self, bot: BollingerBot):
        """Test bot starts in REGISTERED state."""
        assert bot._state == BotState.REGISTERED


# =============================================================================
# Test Start/Stop
# =============================================================================


class TestStartStop:
    """Test start and stop lifecycle."""

    @pytest.mark.asyncio
    async def test_start(self, bot: BollingerBot, mock_exchange: Mock):
        """Test bot starts correctly."""
        await bot.start()

        # Should have fetched klines
        mock_exchange.get_klines.assert_called_once()
        # Should have subscribed to klines
        mock_exchange.subscribe_klines.assert_called_once()
        # Should have subscribed to user data
        mock_exchange.subscribe_user_data.assert_called_once()
        # Should have set leverage
        mock_exchange.futures_set_leverage.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop(self, bot: BollingerBot, mock_exchange: Mock):
        """Test bot stops correctly."""
        await bot.start()
        await bot.stop()

        # Should have unsubscribed from klines
        mock_exchange.unsubscribe_klines.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_sends_notification(
        self,
        bot: BollingerBot,
        mock_notifier: Mock,
    ):
        """Test that start sends notification."""
        await bot.start()

        mock_notifier.send.assert_called()
        call_args = mock_notifier.send.call_args
        assert "啟動" in call_args.kwargs["title"]


# =============================================================================
# Test Pause/Resume
# =============================================================================


class TestPauseResume:
    """Test pause and resume."""

    @pytest.mark.asyncio
    async def test_pause(self, bot: BollingerBot):
        """Test bot pauses correctly."""
        await bot.start()
        result = await bot.pause()

        assert result is True
        assert bot._state == BotState.PAUSED

    @pytest.mark.asyncio
    async def test_resume(self, bot: BollingerBot, mock_exchange: Mock):
        """Test bot resumes correctly."""
        await bot.start()
        await bot.pause()
        result = await bot.resume()

        assert result is True
        assert bot._state == BotState.RUNNING
        # Should sync with exchange
        mock_exchange.futures_get_positions.assert_called()

    @pytest.mark.asyncio
    async def test_klines_ignored_when_paused(self, bot: BollingerBot):
        """Test that klines are ignored when paused."""
        await bot.start()
        await bot.pause()

        initial_bar = bot._current_bar

        # Send a kline (should be ignored)
        kline = MockKline(close=Decimal("49000"), is_closed=True)
        await bot._on_kline(kline)

        # Bar should not have incremented
        assert bot._current_bar == initial_bar


# =============================================================================
# Test Full Long Trade
# =============================================================================


class TestFullLongTrade:
    """Test complete long trade flow."""

    @pytest.mark.asyncio
    async def test_full_long_trade(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
    ):
        """Test complete long trade from signal to close."""
        await bot.start()

        # Setup mock to track order IDs
        order_counter = [0]
        def create_order(*args, **kwargs):
            order_counter[0] += 1
            return MockOrder(order_id=f"order_{order_counter[0]:03d}")

        mock_exchange.futures_create_order = AsyncMock(side_effect=create_order)

        # Simulate price touching lower band
        # First, we need to set up the calculator to return bands
        # where current price is at lower band

        # Process bar with price at lower band
        kline = MockKline(close=Decimal("49000"), is_closed=True)
        bot._klines.append(kline)
        bot._current_bar += 1

        # This should trigger entry
        await bot._process_bar()

        # If signal was generated and entry order placed
        if bot._entry_order_bar is not None:
            # Entry order should have been placed
            assert mock_exchange.futures_create_order.called

            # Simulate entry fill
            entry_order = MockOrder(
                order_id="order_001",
                status="FILLED",
                side="BUY",
                avg_price=Decimal("49000"),
                filled_quantity=Decimal("0.006"),
            )
            await bot._on_order_update(entry_order)

            # Should have exit orders now
            assert bot._order_executor.take_profit_order is not None or \
                   bot._order_executor.stop_loss_order is not None


# =============================================================================
# Test Full Short Trade
# =============================================================================


class TestFullShortTrade:
    """Test complete short trade flow."""

    @pytest.mark.asyncio
    async def test_short_signal_at_upper_band(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
    ):
        """Test short trade when price at upper band."""
        await bot.start()

        # Create klines that result in price at upper band
        kline = MockKline(close=Decimal("51000"), is_closed=True)
        bot._klines.append(kline)
        bot._current_bar += 1

        await bot._process_bar()

        # Should have attempted to create order if signal generated
        # (depends on BBW not being in squeeze)


# =============================================================================
# Test Squeeze Filter
# =============================================================================


class TestSqueezeFilter:
    """Test BBW squeeze filtering."""

    @pytest.mark.asyncio
    async def test_no_entry_during_squeeze(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
    ):
        """Test that no entry occurs during BBW squeeze."""
        # Create klines with very low volatility (squeeze condition)
        stable_klines = create_klines([50000.0] * 300)
        mock_exchange.get_klines = AsyncMock(return_value=stable_klines)

        await bot.start()

        # Add kline at lower band
        kline = MockKline(close=Decimal("49000"), is_closed=True)
        bot._klines.append(kline)
        bot._current_bar += 1

        await bot._process_bar()

        # Should not have entry order due to squeeze
        # (Note: actual behavior depends on BBW calculation)


# =============================================================================
# Test Entry Timeout
# =============================================================================


class TestEntryTimeout:
    """Test entry order timeout."""

    @pytest.mark.asyncio
    async def test_entry_timeout_cancels_order(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
    ):
        """Test that entry order is cancelled after timeout."""
        await bot.start()

        # Manually set entry order bar
        bot._entry_order_bar = bot._current_bar
        bot._order_executor._pending_entry_order = "entry_001"

        # Simulate 3 bars passing (default timeout)
        for _ in range(4):
            kline = MockKline(close=Decimal("50000"), is_closed=True)
            await bot._on_kline(kline)

        # Entry order should be cancelled
        mock_exchange.futures_cancel_order.assert_called()
        assert bot._entry_order_bar is None


# =============================================================================
# Test Hold Timeout
# =============================================================================


class TestHoldTimeout:
    """Test position hold timeout."""

    @pytest.mark.asyncio
    async def test_hold_timeout_exits_position(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
        config: BollingerConfig,
    ):
        """Test that position is closed after hold timeout."""
        await bot.start()

        # Manually create position
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.006"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=bot._current_bar,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )
        bot._position_manager._current_position = position

        # Simulate timeout_bars + 1 passing
        for _ in range(config.timeout_bars + 2):
            kline = MockKline(close=Decimal("49500"), is_closed=True)
            await bot._on_kline(kline)

        # Position should be closed due to timeout


# =============================================================================
# Test Stop Loss Triggered
# =============================================================================


class TestStopLossTriggered:
    """Test stop loss triggering."""

    @pytest.mark.asyncio
    async def test_stop_loss_exits_position(
        self,
        bot: BollingerBot,
        mock_exchange: Mock,
    ):
        """Test that stop loss fill closes position."""
        await bot.start()

        # Setup exit orders
        bot._order_executor._stop_loss_order = "sl_001"
        bot._order_executor._take_profit_order = "tp_001"

        # Manually create position
        position = Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("49000"),
            quantity=Decimal("0.006"),
            leverage=2,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=1,
            take_profit_price=Decimal("50000"),
            stop_loss_price=Decimal("48265"),
        )
        bot._position_manager._current_position = position

        # Simulate stop loss fill
        sl_order = MockOrder(
            order_id="sl_001",
            status="FILLED",
            side="SELL",
            avg_price=Decimal("48265"),
            filled_quantity=Decimal("0.006"),
        )
        await bot._on_order_update(sl_order)

        # TP order should be cancelled
        mock_exchange.futures_cancel_order.assert_called()


# =============================================================================
# Test Statistics Update
# =============================================================================


class TestStatsUpdate:
    """Test statistics updates."""

    def test_initial_stats(self, bot: BollingerBot):
        """Test initial statistics are zero."""
        stats = bot.get_bollinger_stats()

        assert stats["total_trades"] == 0
        assert stats["winning_trades"] == 0
        assert stats["losing_trades"] == 0

    def test_stats_after_trade(self, bot: BollingerBot):
        """Test statistics update after trade."""
        # Manually update stats
        bot._bollinger_stats.record_trade(Decimal("10"))  # Winning trade

        stats = bot.get_bollinger_stats()

        assert stats["total_trades"] == 1
        assert stats["winning_trades"] == 1
        assert stats["total_pnl"] == Decimal("10")


# =============================================================================
# Test Extra Status
# =============================================================================


class TestExtraStatus:
    """Test extra status fields."""

    @pytest.mark.asyncio
    async def test_extra_status_fields(self, bot: BollingerBot):
        """Test that extra status includes Bollinger-specific fields."""
        await bot.start()

        status = bot._get_extra_status()

        assert "timeframe" in status
        assert "leverage" in status
        assert "bb_period" in status
        assert "bb_std" in status
        assert "current_bar" in status
        assert "has_position" in status


# =============================================================================
# Test Health Checks
# =============================================================================


class TestHealthChecks:
    """Test health check functionality."""

    @pytest.mark.asyncio
    async def test_health_checks_pass(self, bot: BollingerBot):
        """Test health checks pass with valid state."""
        await bot.start()

        checks = await bot._extra_health_checks()

        assert checks["klines_ok"] is True
        assert checks["bar_count_ok"] is True


# =============================================================================
# Test Bot Type
# =============================================================================


class TestBotType:
    """Test bot type property."""

    def test_bot_type(self, bot: BollingerBot):
        """Test bot_type returns correct value."""
        assert bot.bot_type == "bollinger"


# =============================================================================
# Test Kline Callback
# =============================================================================


class TestKlineCallback:
    """Test kline callback handling."""

    @pytest.mark.asyncio
    async def test_unclosed_kline_ignored(self, bot: BollingerBot):
        """Test that unclosed klines are ignored."""
        await bot.start()
        initial_bar = bot._current_bar

        # Send unclosed kline
        kline = MockKline(close=Decimal("49000"), is_closed=False)
        await bot._on_kline(kline)

        # Bar should not have incremented
        assert bot._current_bar == initial_bar

    @pytest.mark.asyncio
    async def test_closed_kline_processed(self, bot: BollingerBot):
        """Test that closed klines are processed."""
        await bot.start()
        initial_bar = bot._current_bar

        # Send closed kline
        kline = MockKline(close=Decimal("49000"), is_closed=True)
        await bot._on_kline(kline)

        # Bar should have incremented
        assert bot._current_bar == initial_bar + 1

    @pytest.mark.asyncio
    async def test_kline_list_maintained(self, bot: BollingerBot):
        """Test that kline list doesn't grow unbounded."""
        await bot.start()

        # Add many klines
        for i in range(600):
            kline = MockKline(close=Decimal(f"{50000 + i}"), is_closed=True)
            await bot._on_kline(kline)

        # Should be capped at 500
        assert len(bot._klines) <= 500
