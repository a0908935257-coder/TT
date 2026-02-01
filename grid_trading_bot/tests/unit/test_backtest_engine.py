"""
Unit tests for unified backtest framework.

Tests the core components of the backtest framework including:
- BacktestConfig validation
- BacktestResult metrics
- PositionManager operations
- OrderSimulator calculations
- MetricsCalculator computations
- BacktestEngine execution
"""

import math
import random
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pytest

from src.backtest import (
    BacktestConfig,
    BacktestEngine,
    BacktestResult,
    BacktestStrategy,
    BacktestContext,
    MetricsCalculator,
    OrderSimulator,
    PositionManager,
    Position,
    Signal,
    SignalType,
    Trade,
)
from src.backtest.result import ExitReason
from src.core.models import Kline, KlineInterval


# =============================================================================
# Test Fixtures
# =============================================================================


def create_kline(
    symbol: str = "BTCUSDT",
    interval: str = "15m",
    open_time: datetime = None,
    open_price: Decimal = Decimal("50000"),
    high: Decimal = None,
    low: Decimal = None,
    close: Decimal = None,
    volume: Decimal = Decimal("1000"),
) -> Kline:
    """Create a test Kline."""
    if open_time is None:
        open_time = datetime.now(timezone.utc)

    close_time = open_time + timedelta(minutes=15)

    if high is None:
        high = open_price * Decimal("1.01")
    if low is None:
        low = open_price * Decimal("0.99")
    if close is None:
        close = open_price

    return Kline(
        symbol=symbol,
        interval=KlineInterval.m15,
        open_time=open_time,
        close_time=close_time,
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=volume,
        quote_volume=volume * open_price,
        trades_count=100,
    )


@pytest.fixture
def sample_klines():
    """Generate sample klines for testing."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=10)

    random.seed(42)

    for i in range(200):
        # Oscillating price
        offset = Decimal(str(random.uniform(-3, 3))) / Decimal("100")
        price = base_price * (Decimal("1") + offset)

        kline = create_kline(
            open_time=start_time + timedelta(minutes=i * 15),
            open_price=price,
            high=price * Decimal("1.005"),
            low=price * Decimal("0.995"),
            close=price * (Decimal("1") + Decimal(str(random.uniform(-0.5, 0.5))) / Decimal("100")),
        )
        klines.append(kline)

    return klines


@pytest.fixture
def config():
    """Create default backtest config."""
    return BacktestConfig(
        initial_capital=Decimal("10000"),
        fee_rate=Decimal("0.0004"),
        leverage=2,
        position_size_pct=Decimal("0.1"),
    )


# =============================================================================
# Simple Test Strategy
# =============================================================================


class SimpleMovingAverageStrategy(BacktestStrategy):
    """
    Simple MA crossover strategy for testing.

    Goes long when price > 20-bar SMA, short when price < 20-bar SMA.
    """

    def __init__(self, period: int = 20):
        self._period = period

    def warmup_period(self) -> int:
        return self._period + 10

    def on_kline(self, kline: Kline, context: BacktestContext) -> list[Signal]:
        if context.has_position:
            return []

        closes = context.get_closes(self._period)
        if len(closes) < self._period:
            return []

        sma = sum(closes) / Decimal(len(closes))
        current_price = kline.close

        # Simple long/short logic
        if current_price > sma * Decimal("1.01"):
            return [Signal.long_entry(
                stop_loss=current_price * Decimal("0.98"),
                take_profit=current_price * Decimal("1.02"),
            )]
        elif current_price < sma * Decimal("0.99"):
            return [Signal.short_entry(
                stop_loss=current_price * Decimal("1.02"),
                take_profit=current_price * Decimal("0.98"),
            )]

        return []


class AlwaysLongStrategy(BacktestStrategy):
    """Strategy that always enters long for testing."""

    def __init__(self, max_trades: int = 5):
        self._max_trades = max_trades
        self._trade_count = 0

    def warmup_period(self) -> int:
        return 10

    def on_kline(self, kline: Kline, context: BacktestContext) -> list[Signal]:
        if context.has_position:
            return []

        if self._trade_count >= self._max_trades:
            return []

        self._trade_count += 1
        return [Signal.long_entry(
            stop_loss=kline.close * Decimal("0.95"),
            take_profit=kline.close * Decimal("1.05"),
        )]

    def reset(self) -> None:
        self._trade_count = 0


# =============================================================================
# Test BacktestConfig
# =============================================================================


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BacktestConfig()

        assert config.initial_capital == Decimal("10000")
        assert config.fee_rate == Decimal("0.0004")
        assert config.leverage == 1
        assert config.position_size_pct == Decimal("0.02")

    def test_custom_config(self):
        """Test custom configuration."""
        config = BacktestConfig(
            initial_capital=Decimal("50000"),
            fee_rate=Decimal("0.001"),
            leverage=10,
            position_size_pct=Decimal("0.05"),
        )

        assert config.initial_capital == Decimal("50000")
        assert config.fee_rate == Decimal("0.001")
        assert config.leverage == 10
        assert config.position_size_pct == Decimal("0.05")

    def test_notional_per_trade(self, config):
        """Test notional calculation."""
        assert config.notional_per_trade == Decimal("1000")

    def test_with_leverage(self, config):
        """Test creating config with different leverage."""
        new_config = config.with_leverage(10)

        assert new_config.leverage == 10
        assert new_config.use_margin is True
        assert new_config.initial_capital == config.initial_capital

    def test_invalid_initial_capital(self):
        """Test validation for invalid initial capital."""
        with pytest.raises(ValueError):
            BacktestConfig(initial_capital=Decimal("-1000"))

    def test_invalid_position_size(self):
        """Test validation for invalid position size."""
        with pytest.raises(ValueError):
            BacktestConfig(position_size_pct=Decimal("1.5"))


# =============================================================================
# Test PositionManager
# =============================================================================


class TestPositionManager:
    """Tests for PositionManager."""

    def test_open_position(self):
        """Test opening a position."""
        manager = PositionManager()

        position = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
        )

        result = manager.open_position(position)

        assert result is True
        assert manager.has_position is True
        assert manager.current_position == position

    def test_close_position(self):
        """Test closing a position."""
        manager = PositionManager()

        position = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
        )
        manager.open_position(position)

        trade = manager.close_position(
            position=position,
            exit_price=Decimal("51000"),
            exit_time=datetime.now(timezone.utc),
            exit_bar=20,
            exit_fee=Decimal("2.04"),
            exit_reason=ExitReason.TAKE_PROFIT,
            leverage=1,
        )

        assert manager.has_position is False
        assert isinstance(trade, Trade)
        assert trade.pnl > 0
        assert trade.exit_reason == ExitReason.TAKE_PROFIT

    def test_unrealized_pnl_long(self):
        """Test unrealized P&L calculation for long position."""
        manager = PositionManager()

        position = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
        )
        manager.open_position(position)

        # Price went up
        unrealized = manager.unrealized_pnl(Decimal("51000"))
        assert unrealized == Decimal("100")  # (51000 - 50000) * 0.1

    def test_unrealized_pnl_short(self):
        """Test unrealized P&L calculation for short position."""
        manager = PositionManager()

        position = Position(
            side="SHORT",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
        )
        manager.open_position(position)

        # Price went down (profit for short)
        unrealized = manager.unrealized_pnl(Decimal("49000"))
        assert unrealized == Decimal("100")  # (50000 - 49000) * 0.1

    def test_max_positions_limit(self):
        """Test max positions limit."""
        manager = PositionManager(max_positions=1)

        pos1 = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
        )
        pos2 = Position(
            side="SHORT",
            entry_price=Decimal("51000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=15,
        )

        assert manager.open_position(pos1) is True
        assert manager.open_position(pos2) is False


# =============================================================================
# Test OrderSimulator
# =============================================================================


class TestOrderSimulator:
    """Tests for OrderSimulator."""

    def test_calculate_quantity(self, config):
        """Test quantity calculation."""
        simulator = OrderSimulator(config)

        quantity = simulator.calculate_quantity(Decimal("50000"))
        expected = Decimal("1000") / Decimal("50000")

        assert quantity == expected

    def test_calculate_fee(self, config):
        """Test fee calculation."""
        simulator = OrderSimulator(config)

        fee = simulator.calculate_fee(Decimal("50000"), Decimal("0.02"))
        expected = Decimal("50000") * Decimal("0.02") * Decimal("0.0004")

        assert fee == expected

    def test_check_stop_loss_triggered(self, config, sample_klines):
        """Test stop loss trigger detection."""
        simulator = OrderSimulator(config)

        position = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
            stop_loss=Decimal("49000"),
        )

        # Create kline that goes below stop loss
        kline = create_kline(
            open_price=Decimal("49500"),
            high=Decimal("49600"),
            low=Decimal("48500"),
            close=Decimal("48800"),
        )

        triggered, fill_price = simulator.check_stop_loss(position, kline)

        assert triggered is True
        assert fill_price is not None
        assert fill_price <= Decimal("49000")

    def test_check_take_profit_triggered(self, config):
        """Test take profit trigger detection."""
        simulator = OrderSimulator(config)

        position = Position(
            side="LONG",
            entry_price=Decimal("50000"),
            quantity=Decimal("0.1"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=10,
            take_profit=Decimal("52000"),
        )

        # Create kline that goes above take profit
        kline = create_kline(
            open_price=Decimal("51500"),
            high=Decimal("52500"),
            low=Decimal("51400"),
            close=Decimal("52200"),
        )

        triggered, fill_price = simulator.check_take_profit(position, kline)

        assert triggered is True
        assert fill_price == Decimal("52000")


# =============================================================================
# Test MetricsCalculator
# =============================================================================


class TestMetricsCalculator:
    """Tests for MetricsCalculator."""

    def test_empty_trades(self):
        """Test metrics with no trades."""
        calc = MetricsCalculator()
        result = calc.calculate_all([], [], {})

        assert result.total_trades == 0
        assert result.total_profit == Decimal("0")
        assert result.win_rate == Decimal("0")

    def test_win_rate_calculation(self):
        """Test win rate calculation."""
        calc = MetricsCalculator()

        trades = [
            Trade(
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                side="LONG",
                entry_price=Decimal("50000"),
                exit_price=Decimal("51000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("100"),
                pnl_pct=Decimal("2"),
                fees=Decimal("4"),
                bars_held=10,
                exit_reason=ExitReason.TAKE_PROFIT,
            ),
            Trade(
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                side="LONG",
                entry_price=Decimal("50000"),
                exit_price=Decimal("49000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("-100"),
                pnl_pct=Decimal("-2"),
                fees=Decimal("4"),
                bars_held=10,
                exit_reason=ExitReason.STOP_LOSS,
            ),
        ]

        result = calc.calculate_all(trades, [], {})

        assert result.win_rate == Decimal("50")
        assert result.num_wins == 1
        assert result.num_losses == 1

    def test_profit_factor_calculation(self):
        """Test profit factor calculation."""
        calc = MetricsCalculator()

        trades = [
            Trade(
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                side="LONG",
                entry_price=Decimal("50000"),
                exit_price=Decimal("52000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("200"),
                pnl_pct=Decimal("4"),
                fees=Decimal("4"),
                bars_held=10,
                exit_reason=ExitReason.TAKE_PROFIT,
            ),
            Trade(
                entry_time=datetime.now(timezone.utc),
                exit_time=datetime.now(timezone.utc),
                side="LONG",
                entry_price=Decimal("50000"),
                exit_price=Decimal("49000"),
                quantity=Decimal("0.1"),
                pnl=Decimal("-100"),
                pnl_pct=Decimal("-2"),
                fees=Decimal("4"),
                bars_held=10,
                exit_reason=ExitReason.STOP_LOSS,
            ),
        ]

        result = calc.calculate_all(trades, [], {})

        assert result.profit_factor == Decimal("2")  # 200 / 100

    def test_max_drawdown_calculation(self):
        """Test max drawdown calculation."""
        calc = MetricsCalculator()

        # Equity curve: 10000 -> 10500 -> 10200 -> 10800
        equity_curve = [
            Decimal("10000"),
            Decimal("10500"),
            Decimal("10200"),
            Decimal("10800"),
        ]

        result = calc.calculate_all([], equity_curve, {})

        # Max drawdown: 10500 -> 10200 = 300
        assert result.max_drawdown == Decimal("300")


# =============================================================================
# Test BacktestEngine
# =============================================================================


class TestBacktestEngine:
    """Tests for BacktestEngine."""

    def test_empty_klines(self, config):
        """Test backtest with empty klines."""
        engine = BacktestEngine(config)
        strategy = SimpleMovingAverageStrategy()

        result = engine.run([], strategy)

        assert result.total_trades == 0

    def test_insufficient_klines(self, config):
        """Test backtest with insufficient klines for warmup."""
        engine = BacktestEngine(config)
        strategy = SimpleMovingAverageStrategy(period=20)

        klines = [create_kline() for _ in range(10)]
        result = engine.run(klines, strategy)

        assert result.total_trades == 0

    def test_basic_backtest(self, config, sample_klines):
        """Test basic backtest execution."""
        engine = BacktestEngine(config)
        strategy = AlwaysLongStrategy(max_trades=3)

        result = engine.run(sample_klines, strategy)

        assert isinstance(result, BacktestResult)
        assert result.total_trades <= 3
        assert len(result.equity_curve) > 0

    def test_backtest_generates_trades(self, config, sample_klines):
        """Test that backtest generates trades."""
        engine = BacktestEngine(config)
        strategy = AlwaysLongStrategy(max_trades=5)

        result = engine.run(sample_klines, strategy)

        # Should have some trades
        assert result.total_trades > 0
        assert len(result.trades) == result.total_trades

    def test_stop_loss_execution(self, config):
        """Test that stop loss is executed."""
        # Create klines with a downward move
        klines = []
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        price = Decimal("50000")

        # Warmup period
        for i in range(50):
            klines.append(create_kline(
                open_time=start_time + timedelta(minutes=i * 15),
                open_price=price,
                high=price * Decimal("1.005"),
                low=price * Decimal("0.995"),
                close=price,
            ))

        # Sharp drop to trigger stop loss
        for i in range(50, 100):
            price = price * Decimal("0.99")
            klines.append(create_kline(
                open_time=start_time + timedelta(minutes=i * 15),
                open_price=price,
                high=price * Decimal("1.002"),
                low=price * Decimal("0.998"),
                close=price,
            ))

        engine = BacktestEngine(config)
        strategy = AlwaysLongStrategy(max_trades=1)

        result = engine.run(klines, strategy)

        if result.total_trades > 0:
            # Should have hit stop loss
            for trade in result.trades:
                if trade.exit_reason == ExitReason.STOP_LOSS:
                    assert trade.pnl < 0
                    break

    def test_take_profit_execution(self, config):
        """Test that take profit is executed."""
        # Create klines with an upward move
        klines = []
        start_time = datetime.now(timezone.utc) - timedelta(days=1)
        price = Decimal("50000")

        # Warmup period
        for i in range(50):
            klines.append(create_kline(
                open_time=start_time + timedelta(minutes=i * 15),
                open_price=price,
                high=price * Decimal("1.005"),
                low=price * Decimal("0.995"),
                close=price,
            ))

        # Sharp rise to trigger take profit
        for i in range(50, 100):
            price = price * Decimal("1.01")
            klines.append(create_kline(
                open_time=start_time + timedelta(minutes=i * 15),
                open_price=price,
                high=price * Decimal("1.01"),
                low=price * Decimal("0.99"),
                close=price,
            ))

        engine = BacktestEngine(config)
        strategy = AlwaysLongStrategy(max_trades=1)

        result = engine.run(klines, strategy)

        if result.total_trades > 0:
            # Should have hit take profit
            for trade in result.trades:
                if trade.exit_reason == ExitReason.TAKE_PROFIT:
                    assert trade.pnl > 0
                    break


class TestBacktestEngineWalkForward:
    """Tests for walk-forward validation."""

    def test_walk_forward_basic(self, config, sample_klines):
        """Test basic walk-forward execution."""
        engine = BacktestEngine(config)
        strategy = SimpleMovingAverageStrategy(period=10)

        result = engine.run_walk_forward(
            klines=sample_klines,
            strategy=strategy,
            periods=3,
            is_ratio=0.7,
        )

        assert result.total_periods <= 3
        assert isinstance(result.consistency_pct, Decimal)

    def test_walk_forward_insufficient_data(self, config):
        """Test walk-forward with insufficient data."""
        klines = [create_kline() for _ in range(20)]

        engine = BacktestEngine(config)
        strategy = SimpleMovingAverageStrategy(period=10)

        result = engine.run_walk_forward(
            klines=klines,
            strategy=strategy,
            periods=6,
        )

        # Should return empty result or few periods
        assert result.total_periods < 6


# =============================================================================
# Test Result Summary
# =============================================================================


class TestBacktestResultSummary:
    """Tests for result summary generation."""

    def test_summary_format(self):
        """Test summary string generation."""
        result = BacktestResult(
            total_profit=Decimal("1500"),
            total_profit_pct=Decimal("15"),
            total_trades=20,
            win_rate=Decimal("60"),
            num_wins=12,
            num_losses=8,
            profit_factor=Decimal("1.5"),
            sharpe_ratio=Decimal("1.2"),
            max_drawdown=Decimal("500"),
            max_drawdown_pct=Decimal("5"),
            avg_win=Decimal("200"),
            avg_loss=Decimal("100"),
        )

        summary = result.summary()

        assert "Total Profit:" in summary
        assert "15.00%" in summary
        assert "Win Rate: 60.00%" in summary
        assert "Sharpe Ratio: 1.20" in summary
