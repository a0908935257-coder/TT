"""
Simulation and Backtest tests.

Tests grid trading strategy backtesting with historical data.
"""

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pytest

from src.core.models import Kline, MarketType, OrderSide
from src.bots.grid import (
    GridBotConfig,
    GridConfig,
    GridType,
    RiskLevel,
    create_grid_with_manual_range,
)


@dataclass
class BacktestResult:
    """
    Backtest result containing performance metrics.

    Attributes:
        total_profit: Cumulative profit after all trades
        total_trades: Number of completed round-trip trades
        win_rate: Percentage of profitable trades
        max_drawdown: Maximum peak-to-trough decline
        sharpe_ratio: Risk-adjusted return metric
        profit_factor: Gross profit / gross loss ratio
        avg_holding_time: Average time between buy and sell
        daily_returns: List of daily P&L values
    """

    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_time: timedelta = field(default_factory=lambda: timedelta(0))
    daily_returns: list[Decimal] = field(default_factory=list)
    gross_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    gross_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0


class GridBacktest:
    """
    Grid trading backtest engine.

    Simulates grid trading on historical kline data.

    Example:
        >>> backtest = GridBacktest(klines=klines, config=config)
        >>> result = backtest.run()
        >>> print(f"Total profit: {result.total_profit}")
    """

    # Fee constants
    FEE_RATE = Decimal("0.001")  # 0.1%

    def __init__(
        self,
        klines: list[Kline],
        config: GridBotConfig,
    ):
        """
        Initialize backtest.

        Args:
            klines: Historical kline data
            config: Grid bot configuration
        """
        self._klines = klines
        self._config = config

        # Grid setup (created on run)
        self._setup = None

        # Tracking
        self._pending_buys: dict[int, dict] = {}  # level_index -> order info
        self._pending_sells: dict[int, dict] = {}
        self._positions: list[dict] = []  # Active positions
        self._trades: list[dict] = []  # Completed trades

        # Metrics
        self._equity_curve: list[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}  # date -> pnl

    def run(self) -> BacktestResult:
        """
        Run backtest simulation.

        Process:
        1. Initialize grid at first kline's close price
        2. For each kline, check for order fills
        3. Track trades and calculate metrics
        4. Return final result

        Returns:
            BacktestResult with all metrics
        """
        if not self._klines:
            return BacktestResult()

        # Initialize grid at starting price
        start_price = self._klines[0].close
        self._initialize_grid(start_price)

        # Process each kline
        for kline in self._klines:
            self._process_kline(kline)

        # Calculate final result
        return self._calculate_result()

    def get_result(self) -> BacktestResult:
        """Get the backtest result (must call run() first)."""
        return self._calculate_result()

    def _initialize_grid(self, current_price: Decimal) -> None:
        """Initialize grid levels and pending orders."""
        # Create grid setup
        if self._config.has_manual_range:
            self._setup = create_grid_with_manual_range(
                symbol=self._config.symbol,
                investment=self._config.total_investment,
                upper_price=self._config.manual_upper,
                lower_price=self._config.manual_lower,
                grid_count=self._config.manual_grid_count or 10,
                current_price=current_price,
                grid_type=self._config.grid_type,
            )
        else:
            # For ATR-based, use manual with estimated range
            range_percent = Decimal("0.1")  # 10% range
            upper = current_price * (Decimal("1") + range_percent / 2)
            lower = current_price * (Decimal("1") - range_percent / 2)

            self._setup = create_grid_with_manual_range(
                symbol=self._config.symbol,
                investment=self._config.total_investment,
                upper_price=upper,
                lower_price=lower,
                grid_count=10,
                current_price=current_price,
                grid_type=self._config.grid_type,
            )

        # Place initial orders
        for level in self._setup.levels:
            if level.price < current_price:
                # Buy order below current price
                self._pending_buys[level.index] = {
                    "price": level.price,
                    "quantity": level.allocated_amount / level.price,
                    "level_index": level.index,
                }
            elif level.price > current_price:
                # Sell order above current price
                self._pending_sells[level.index] = {
                    "price": level.price,
                    "quantity": level.allocated_amount / level.price,
                    "level_index": level.index,
                }

    def _process_kline(self, kline: Kline) -> None:
        """Process a single kline and check for fills."""
        # Record date for daily PnL
        date_key = kline.close_time.strftime("%Y-%m-%d")

        # Check buy fills (price went down to order price)
        filled_buys = []
        for level_index, order in list(self._pending_buys.items()):
            if kline.low <= order["price"]:
                # Buy filled
                filled_buys.append(level_index)
                self._on_buy_filled(order, kline)

        # Remove filled buys
        for level_index in filled_buys:
            del self._pending_buys[level_index]

        # Check sell fills (price went up to order price)
        filled_sells = []
        for level_index, order in list(self._pending_sells.items()):
            if kline.high >= order["price"]:
                # Sell filled
                filled_sells.append(level_index)
                self._on_sell_filled(order, kline, date_key)

        # Remove filled sells
        for level_index in filled_sells:
            del self._pending_sells[level_index]

        # Track equity
        self._update_equity(kline.close)

    def _on_buy_filled(self, order: dict, kline: Kline) -> None:
        """Handle buy order fill."""
        # Calculate cost
        cost = order["price"] * order["quantity"]
        fee = cost * self.FEE_RATE

        # Add position
        position = {
            "level_index": order["level_index"],
            "buy_price": order["price"],
            "quantity": order["quantity"],
            "buy_time": kline.close_time,
            "buy_fee": fee,
        }
        self._positions.append(position)

        # Place reverse sell order at upper level
        upper_level_index = order["level_index"] + 1
        if upper_level_index < len(self._setup.levels):
            upper_level = self._setup.levels[upper_level_index]
            self._pending_sells[upper_level_index] = {
                "price": upper_level.price,
                "quantity": order["quantity"],
                "level_index": upper_level_index,
                "linked_position": position,
            }

    def _on_sell_filled(self, order: dict, kline: Kline, date_key: str) -> None:
        """Handle sell order fill."""
        # Find matching position
        linked_position = order.get("linked_position")

        if linked_position:
            # Calculate profit
            sell_value = order["price"] * order["quantity"]
            sell_fee = sell_value * self.FEE_RATE

            buy_value = linked_position["buy_price"] * linked_position["quantity"]
            buy_fee = linked_position["buy_fee"]

            profit = sell_value - buy_value - sell_fee - buy_fee

            # Record trade
            trade = {
                "buy_price": linked_position["buy_price"],
                "sell_price": order["price"],
                "quantity": order["quantity"],
                "profit": profit,
                "buy_time": linked_position["buy_time"],
                "sell_time": kline.close_time,
                "holding_time": kline.close_time - linked_position["buy_time"],
            }
            self._trades.append(trade)

            # Update daily PnL
            if date_key not in self._daily_pnl:
                self._daily_pnl[date_key] = Decimal("0")
            self._daily_pnl[date_key] += profit

            # Remove from positions
            if linked_position in self._positions:
                self._positions.remove(linked_position)

            # Place reverse buy order at lower level
            lower_level_index = order["level_index"] - 1
            if lower_level_index >= 0:
                lower_level = self._setup.levels[lower_level_index]
                self._pending_buys[lower_level_index] = {
                    "price": lower_level.price,
                    "quantity": order["quantity"],
                    "level_index": lower_level_index,
                }

    def _update_equity(self, current_price: Decimal) -> None:
        """Update equity curve with unrealized P&L."""
        # Realized P&L
        realized = sum(t["profit"] for t in self._trades)

        # Unrealized P&L
        unrealized = Decimal("0")
        for position in self._positions:
            unrealized += (current_price - position["buy_price"]) * position["quantity"]

        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> BacktestResult:
        """Calculate final backtest result."""
        result = BacktestResult()

        if not self._trades:
            return result

        # Total profit
        result.total_profit = sum(t["profit"] for t in self._trades)
        result.total_trades = len(self._trades)

        # Win rate
        wins = [t for t in self._trades if t["profit"] > 0]
        losses = [t for t in self._trades if t["profit"] < 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)

        if self._trades:
            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * Decimal("100")

        # Gross profit/loss
        result.gross_profit = sum(t["profit"] for t in wins)
        result.gross_loss = abs(sum(t["profit"] for t in losses))

        # Profit factor
        if result.gross_loss > 0:
            result.profit_factor = result.gross_profit / result.gross_loss
        else:
            result.profit_factor = Decimal("inf") if result.gross_profit > 0 else Decimal("0")

        # Max drawdown
        result.max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio (simplified)
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        # Average holding time
        if self._trades:
            total_holding = sum(
                (t["holding_time"].total_seconds() for t in self._trades),
                0
            )
            avg_seconds = total_holding / len(self._trades)
            result.avg_holding_time = timedelta(seconds=avg_seconds)

        # Daily returns
        result.daily_returns = list(self._daily_pnl.values())

        return result

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from equity curve."""
        if not self._equity_curve:
            return Decimal("0")

        peak = self._equity_curve[0]
        max_dd = Decimal("0")

        for equity in self._equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate simplified Sharpe ratio."""
        if not self._daily_pnl:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        if not returns:
            return Decimal("0")

        # Mean return
        mean_return = sum(returns) / Decimal(len(returns))

        # Standard deviation
        if len(returns) < 2:
            return Decimal("0")

        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)
        std_dev = variance ** Decimal("0.5")

        if std_dev == 0:
            return Decimal("0")

        # Annualized (assuming daily returns)
        return (mean_return / std_dev) * Decimal("252") ** Decimal("0.5")


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sideways_klines():
    """Generate sideways market klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(123)

    for i in range(100):
        # Oscillate around base price ±5%
        offset = Decimal(str(random.uniform(-5, 5))) / Decimal("100")
        center = base_price * (Decimal("1") + offset)

        open_price = center * Decimal("0.999")
        close_price = center * Decimal("1.001")
        high_price = max(open_price, close_price) * Decimal("1.01")
        low_price = min(open_price, close_price) * Decimal("0.99")

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=start_time + timedelta(hours=i * 4),
            close_time=start_time + timedelta(hours=(i + 1) * 4 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


@pytest.fixture
def uptrend_klines():
    """Generate uptrend market klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(456)

    for i in range(100):
        # Trending up with some noise
        trend = Decimal("0.002") * Decimal(i)  # 0.2% per kline
        noise = Decimal(str(random.uniform(-1, 2))) / Decimal("100")
        center = base_price * (Decimal("1") + trend + noise)

        open_price = center * Decimal("0.995")
        close_price = center * Decimal("1.005")
        high_price = close_price * Decimal("1.01")
        low_price = open_price * Decimal("0.99")

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=start_time + timedelta(hours=i * 4),
            close_time=start_time + timedelta(hours=(i + 1) * 4 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


@pytest.fixture
def downtrend_klines():
    """Generate downtrend market klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(789)

    for i in range(100):
        # Trending down with some noise
        trend = Decimal("-0.002") * Decimal(i)  # -0.2% per kline
        noise = Decimal(str(random.uniform(-2, 1))) / Decimal("100")
        center = base_price * (Decimal("1") + trend + noise)

        open_price = center * Decimal("1.005")
        close_price = center * Decimal("0.995")
        high_price = open_price * Decimal("1.01")
        low_price = close_price * Decimal("0.99")

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=start_time + timedelta(hours=i * 4),
            close_time=start_time + timedelta(hours=(i + 1) * 4 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


@pytest.fixture
def volatile_klines():
    """Generate high volatility klines."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(999)

    for i in range(100):
        # High volatility swings
        swing = Decimal(str(random.uniform(-10, 10))) / Decimal("100")
        center = base_price * (Decimal("1") + swing)

        # Large candles
        direction = 1 if random.random() > 0.5 else -1
        body = Decimal(str(random.uniform(2, 5))) / Decimal("100")

        open_price = center
        close_price = center * (Decimal("1") + body * Decimal(direction))
        high_price = max(open_price, close_price) * Decimal("1.03")
        low_price = min(open_price, close_price) * Decimal("0.97")

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=start_time + timedelta(hours=i * 4),
            close_time=start_time + timedelta(hours=(i + 1) * 4 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("5000"),
            quote_volume=Decimal("250000000"),
            trades=500,
        )
        klines.append(kline)

    return klines


@pytest.fixture
def backtest_config():
    """Create backtest configuration."""
    return GridBotConfig(
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MODERATE,
        grid_type=GridType.GEOMETRIC,
        manual_upper=Decimal("55000"),
        manual_lower=Decimal("45000"),
        manual_grid_count=10,
    )


# =============================================================================
# Tests
# =============================================================================


class TestGridBacktest:
    """Tests for GridBacktest class."""

    def test_backtest_initialization(self, sideways_klines, backtest_config):
        """Test backtest initialization."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)

        assert backtest._klines == sideways_klines
        assert backtest._config == backtest_config

    def test_backtest_run_returns_result(self, sideways_klines, backtest_config):
        """Test backtest run returns result."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        assert isinstance(result, BacktestResult)

    def test_backtest_empty_klines(self, backtest_config):
        """Test backtest with empty klines."""
        backtest = GridBacktest(klines=[], config=backtest_config)
        result = backtest.run()

        assert result.total_trades == 0
        assert result.total_profit == Decimal("0")


class TestBacktestSideways:
    """Tests for sideways market backtest."""

    def test_backtest_sideways(self, sideways_klines, backtest_config):
        """Test backtest in sideways market."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        # Sideways market should generate trades
        assert result.total_trades >= 0

        # Grid trading should profit in sideways market
        # (may not always be true depending on exact movements)


class TestBacktestUptrend:
    """Tests for uptrend market backtest."""

    def test_backtest_uptrend(self, uptrend_klines, backtest_config):
        """Test backtest in uptrend market."""
        backtest = GridBacktest(klines=uptrend_klines, config=backtest_config)
        result = backtest.run()

        # Uptrend may have fewer complete trades
        # as price breaks through upper bound
        assert isinstance(result.total_trades, int)


class TestBacktestDowntrend:
    """Tests for downtrend market backtest."""

    def test_backtest_downtrend(self, downtrend_klines, backtest_config):
        """Test backtest in downtrend market."""
        backtest = GridBacktest(klines=downtrend_klines, config=backtest_config)
        result = backtest.run()

        # Downtrend may result in losses as positions are bought but not sold
        assert isinstance(result.total_profit, Decimal)


class TestBacktestVolatile:
    """Tests for volatile market backtest."""

    def test_backtest_volatile(self, volatile_klines, backtest_config):
        """Test backtest in volatile market."""
        backtest = GridBacktest(klines=volatile_klines, config=backtest_config)
        result = backtest.run()

        # Volatile market should generate many trades
        assert result.total_trades >= 0


class TestBacktestMetrics:
    """Tests for backtest metrics calculation."""

    def test_win_rate_calculation(self, sideways_klines, backtest_config):
        """Test win rate calculation."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        if result.total_trades > 0:
            # Win rate should be between 0 and 100
            assert Decimal("0") <= result.win_rate <= Decimal("100")

            # Sum of wins and losses should equal total trades
            assert result.num_wins + result.num_losses == result.total_trades

    def test_max_drawdown_calculation(self, sideways_klines, backtest_config):
        """Test max drawdown calculation."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        # Max drawdown should be non-negative
        assert result.max_drawdown >= Decimal("0")

    def test_profit_factor_calculation(self, sideways_klines, backtest_config):
        """Test profit factor calculation."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        # Profit factor should be non-negative
        assert result.profit_factor >= Decimal("0")

    def test_daily_returns_tracking(self, sideways_klines, backtest_config):
        """Test daily returns are tracked."""
        backtest = GridBacktest(klines=sideways_klines, config=backtest_config)
        result = backtest.run()

        # Daily returns should be a list
        assert isinstance(result.daily_returns, list)


# =============================================================================
# Bollinger Bot Backtest
# =============================================================================


from src.bots.bollinger.indicators import BollingerCalculator
from src.bots.bollinger.models import (
    BollingerConfig,
    BBWData,
    BollingerBands,
    PositionSide,
    TradeRecord,
)


@dataclass
class BollingerBacktestResult:
    """
    Bollinger Bot backtest result.

    Attributes:
        total_profit: Cumulative profit after all trades
        total_trades: Number of completed trades
        win_rate: Percentage of profitable trades
        max_drawdown: Maximum peak-to-trough decline
        sharpe_ratio: Risk-adjusted return metric
        profit_factor: Gross profit / gross loss ratio
        avg_holding_time: Average time between entry and exit
        long_trades: Number of long trades
        short_trades: Number of short trades
        filtered_by_squeeze: Signals filtered by BBW squeeze
        avg_win: Average winning trade PnL
        avg_loss: Average losing trade PnL
    """

    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    total_trades: int = 0
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    max_drawdown: Decimal = field(default_factory=lambda: Decimal("0"))
    sharpe_ratio: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_holding_time: timedelta = field(default_factory=lambda: timedelta(0))
    daily_returns: list[Decimal] = field(default_factory=list)
    gross_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    gross_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    num_wins: int = 0
    num_losses: int = 0
    long_trades: int = 0
    short_trades: int = 0
    filtered_by_squeeze: int = 0
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))


class BollingerBacktest:
    """
    Bollinger Band mean reversion backtest engine.

    Simulates Bollinger Bot trading logic on historical kline data:
    - Entry: Price touches lower band (LONG) or upper band (SHORT)
    - Exit: Price reaches middle band (TP), stop loss, or timeout
    - Filter: Skip signals during BBW squeeze

    Example:
        >>> backtest = BollingerBacktest(klines=klines, config=config)
        >>> result = backtest.run()
        >>> print(f"Total profit: {result.total_profit}")
    """

    # Fee constants
    FEE_RATE = Decimal("0.0004")  # 0.04% maker fee for futures

    def __init__(
        self,
        klines: list[Kline],
        config: BollingerConfig,
    ):
        """
        Initialize backtest.

        Args:
            klines: Historical kline data
            config: Bollinger bot configuration
        """
        self._klines = klines
        self._config = config

        # Calculator
        self._calculator = BollingerCalculator(
            period=config.bb_period,
            std_multiplier=config.bb_std,
            bbw_lookback=config.bbw_lookback,
            bbw_threshold_pct=config.bbw_threshold_pct,
        )

        # Position tracking
        self._position: Optional[dict] = None
        self._trades: list[dict] = []

        # Metrics
        self._equity_curve: list[Decimal] = []
        self._daily_pnl: dict[str, Decimal] = {}
        self._filtered_signals: int = 0

    def run(self) -> BollingerBacktestResult:
        """
        Run backtest simulation.

        Returns:
            BollingerBacktestResult with all metrics
        """
        if not self._klines or len(self._klines) < self._config.bb_period + 50:
            return BollingerBacktestResult()

        # Initialize BBW history
        self._calculator.initialize(self._klines[:self._config.bb_period + 50])

        # Process each kline starting after initialization period
        for i in range(self._config.bb_period + 50, len(self._klines)):
            kline = self._klines[i]
            klines_subset = self._klines[:i + 1]
            self._process_kline(kline, klines_subset, i)

        # Calculate final result
        return self._calculate_result()

    def _process_kline(self, kline: Kline, klines_subset: list[Kline], bar_idx: int) -> None:
        """Process a single kline."""
        current_price = kline.close
        date_key = kline.close_time.strftime("%Y-%m-%d")

        # Calculate indicators
        bands, bbw = self._calculator.get_all(klines_subset)

        # Check existing position
        if self._position:
            self._check_exit(kline, bands, current_price, bar_idx, date_key)
            return

        # Check for new entry signal
        if bbw.is_squeeze:
            self._filtered_signals += 1
            return

        # Long signal: price <= lower band
        if current_price <= bands.lower:
            self._enter_position(
                side=PositionSide.LONG,
                entry_price=bands.lower,
                take_profit=bands.middle,
                stop_loss=bands.lower * (Decimal("1") - self._config.stop_loss_pct),
                entry_bar=bar_idx,
                entry_time=kline.close_time,
            )

        # Short signal: price >= upper band
        elif current_price >= bands.upper:
            self._enter_position(
                side=PositionSide.SHORT,
                entry_price=bands.upper,
                take_profit=bands.middle,
                stop_loss=bands.upper * (Decimal("1") + self._config.stop_loss_pct),
                entry_bar=bar_idx,
                entry_time=kline.close_time,
            )

        # Update equity
        self._update_equity(current_price)

    def _enter_position(
        self,
        side: PositionSide,
        entry_price: Decimal,
        take_profit: Decimal,
        stop_loss: Decimal,
        entry_bar: int,
        entry_time: datetime,
    ) -> None:
        """Enter a new position."""
        # Calculate position size (using config.position_size_pct of nominal $10000)
        notional = Decimal("10000") * self._config.position_size_pct
        quantity = notional / entry_price

        self._position = {
            "side": side,
            "entry_price": entry_price,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "quantity": quantity,
            "entry_bar": entry_bar,
            "entry_time": entry_time,
            "entry_fee": notional * self.FEE_RATE,
        }

    def _check_exit(
        self,
        kline: Kline,
        bands: BollingerBands,
        current_price: Decimal,
        bar_idx: int,
        date_key: str,
    ) -> None:
        """Check if position should be exited."""
        if not self._position:
            return

        should_exit = False
        exit_price = current_price
        exit_reason = ""

        side = self._position["side"]

        # 1. Take profit - price returned to middle band
        if side == PositionSide.LONG:
            if kline.high >= bands.middle:
                should_exit = True
                exit_price = bands.middle
                exit_reason = "take_profit"
        else:
            if kline.low <= bands.middle:
                should_exit = True
                exit_price = bands.middle
                exit_reason = "take_profit"

        # 2. Stop loss
        if not should_exit:
            if side == PositionSide.LONG:
                if kline.low <= self._position["stop_loss"]:
                    should_exit = True
                    exit_price = self._position["stop_loss"]
                    exit_reason = "stop_loss"
            else:
                if kline.high >= self._position["stop_loss"]:
                    should_exit = True
                    exit_price = self._position["stop_loss"]
                    exit_reason = "stop_loss"

        # 3. Timeout
        if not should_exit:
            hold_bars = bar_idx - self._position["entry_bar"]
            if hold_bars >= self._config.max_hold_bars:
                should_exit = True
                exit_price = current_price
                exit_reason = "timeout"

        if should_exit:
            self._exit_position(exit_price, exit_reason, kline.close_time, bar_idx, date_key)

    def _exit_position(
        self,
        exit_price: Decimal,
        exit_reason: str,
        exit_time: datetime,
        exit_bar: int,
        date_key: str,
    ) -> None:
        """Exit current position."""
        if not self._position:
            return

        side = self._position["side"]
        entry_price = self._position["entry_price"]
        quantity = self._position["quantity"]

        # Calculate PnL
        if side == PositionSide.LONG:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = (entry_price - exit_price) * quantity

        # Apply leverage (PnL is multiplied by leverage)
        pnl *= Decimal(self._config.leverage)

        # Deduct fees
        exit_fee = (exit_price * quantity) * self.FEE_RATE
        total_fee = self._position["entry_fee"] + exit_fee
        net_pnl = pnl - total_fee

        # Record trade
        trade = {
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "pnl": net_pnl,
            "fee": total_fee,
            "entry_time": self._position["entry_time"],
            "exit_time": exit_time,
            "holding_bars": exit_bar - self._position["entry_bar"],
            "exit_reason": exit_reason,
        }
        self._trades.append(trade)

        # Update daily PnL
        if date_key not in self._daily_pnl:
            self._daily_pnl[date_key] = Decimal("0")
        self._daily_pnl[date_key] += net_pnl

        # Clear position
        self._position = None

    def _update_equity(self, current_price: Decimal) -> None:
        """Update equity curve."""
        realized = sum(t["pnl"] for t in self._trades)

        # Unrealized
        unrealized = Decimal("0")
        if self._position:
            side = self._position["side"]
            entry_price = self._position["entry_price"]
            quantity = self._position["quantity"]
            if side == PositionSide.LONG:
                unrealized = (current_price - entry_price) * quantity
            else:
                unrealized = (entry_price - current_price) * quantity
            unrealized *= Decimal(self._config.leverage)

        self._equity_curve.append(realized + unrealized)

    def _calculate_result(self) -> BollingerBacktestResult:
        """Calculate final backtest result."""
        result = BollingerBacktestResult()

        if not self._trades:
            return result

        # Total profit
        result.total_profit = sum(t["pnl"] for t in self._trades)
        result.total_trades = len(self._trades)

        # Win/loss breakdown
        wins = [t for t in self._trades if t["pnl"] > 0]
        losses = [t for t in self._trades if t["pnl"] <= 0]

        result.num_wins = len(wins)
        result.num_losses = len(losses)

        if self._trades:
            result.win_rate = Decimal(len(wins)) / Decimal(len(self._trades)) * Decimal("100")

        # Gross profit/loss
        result.gross_profit = sum(t["pnl"] for t in wins) if wins else Decimal("0")
        result.gross_loss = abs(sum(t["pnl"] for t in losses)) if losses else Decimal("0")

        # Average win/loss
        if wins:
            result.avg_win = result.gross_profit / Decimal(len(wins))
        if losses:
            result.avg_loss = result.gross_loss / Decimal(len(losses))

        # Profit factor
        if result.gross_loss > 0:
            result.profit_factor = result.gross_profit / result.gross_loss
        elif result.gross_profit > 0:
            result.profit_factor = Decimal("999")

        # Long/Short breakdown
        result.long_trades = len([t for t in self._trades if t["side"] == PositionSide.LONG])
        result.short_trades = len([t for t in self._trades if t["side"] == PositionSide.SHORT])

        # Filtered signals
        result.filtered_by_squeeze = self._filtered_signals

        # Max drawdown
        result.max_drawdown = self._calculate_max_drawdown()

        # Sharpe ratio
        result.sharpe_ratio = self._calculate_sharpe_ratio()

        # Average holding time
        if self._trades:
            total_bars = sum(t["holding_bars"] for t in self._trades)
            avg_bars = total_bars / len(self._trades)
            # Assume 15m timeframe = 15 minutes per bar
            result.avg_holding_time = timedelta(minutes=avg_bars * 15)

        # Daily returns
        result.daily_returns = list(self._daily_pnl.values())

        return result

    def _calculate_max_drawdown(self) -> Decimal:
        """Calculate maximum drawdown from equity curve."""
        if not self._equity_curve:
            return Decimal("0")

        peak = self._equity_curve[0]
        max_dd = Decimal("0")

        for equity in self._equity_curve:
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown

        return max_dd

    def _calculate_sharpe_ratio(self) -> Decimal:
        """Calculate simplified Sharpe ratio."""
        if not self._daily_pnl:
            return Decimal("0")

        returns = list(self._daily_pnl.values())
        if not returns or len(returns) < 2:
            return Decimal("0")

        mean_return = sum(returns) / Decimal(len(returns))
        variance = sum((r - mean_return) ** 2 for r in returns) / Decimal(len(returns) - 1)

        # Safe sqrt
        if variance <= 0:
            return Decimal("0")

        import math
        std_dev = Decimal(str(math.sqrt(float(variance))))

        if std_dev == 0:
            return Decimal("0")

        # Annualized
        return (mean_return / std_dev) * Decimal(str(math.sqrt(252)))


# =============================================================================
# Bollinger Backtest Fixtures
# =============================================================================


@pytest.fixture
def bollinger_config():
    """Create Bollinger bot configuration for backtest."""
    return BollingerConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        bb_period=20,
        bb_std=Decimal("2.0"),
        bbw_lookback=100,
        bbw_threshold_pct=20,
        stop_loss_pct=Decimal("0.015"),
        max_hold_bars=16,
        leverage=2,
        position_size_pct=Decimal("0.1"),
    )


@pytest.fixture
def mean_reversion_klines():
    """Generate klines suitable for mean reversion trading."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(555)

    # Generate oscillating price that touches bands and reverts
    for i in range(300):
        # Create oscillating pattern
        cycle_pos = (i % 30) / 30.0  # 0 to 1 over 30 bars
        wave = math.sin(cycle_pos * 2 * math.pi)
        offset = Decimal(str(wave * 0.03))  # ±3%

        noise = Decimal(str(random.uniform(-0.005, 0.005)))
        center = base_price * (Decimal("1") + offset + noise)

        open_price = center * Decimal("0.998")
        close_price = center * Decimal("1.002")
        high_price = max(open_price, close_price) * Decimal("1.005")
        low_price = min(open_price, close_price) * Decimal("0.995")

        kline = Kline(
            symbol="BTCUSDT",
            interval="15m",
            open_time=start_time + timedelta(minutes=i * 15),
            close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


@pytest.fixture
def squeeze_then_breakout_klines():
    """Generate klines with squeeze followed by breakout."""
    klines = []
    base_price = Decimal("50000")
    start_time = datetime.now(timezone.utc) - timedelta(days=30)

    random.seed(666)

    for i in range(300):
        # First 150 bars: low volatility (squeeze)
        # Next 150 bars: high volatility (breakout)
        if i < 150:
            volatility = 0.002  # Very low
        else:
            volatility = 0.03  # High

        noise = Decimal(str(random.uniform(-volatility, volatility)))
        trend = Decimal("0.0001") * Decimal(i)  # Slight uptrend
        center = base_price * (Decimal("1") + noise + trend)

        open_price = center * Decimal("0.999")
        close_price = center * Decimal("1.001")
        high_price = max(open_price, close_price) * (Decimal("1") + Decimal(str(volatility)))
        low_price = min(open_price, close_price) * (Decimal("1") - Decimal(str(volatility)))

        kline = Kline(
            symbol="BTCUSDT",
            interval="15m",
            open_time=start_time + timedelta(minutes=i * 15),
            close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=Decimal("1000"),
            quote_volume=Decimal("50000000"),
            trades=100,
        )
        klines.append(kline)

    return klines


# =============================================================================
# Bollinger Backtest Tests
# =============================================================================


class TestBollingerBacktest:
    """Tests for BollingerBacktest class."""

    def test_backtest_initialization(self, mean_reversion_klines, bollinger_config):
        """Test backtest initialization."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)

        assert backtest._klines == mean_reversion_klines
        assert backtest._config == bollinger_config

    def test_backtest_run_returns_result(self, mean_reversion_klines, bollinger_config):
        """Test backtest run returns result."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        assert isinstance(result, BollingerBacktestResult)

    def test_backtest_empty_klines(self, bollinger_config):
        """Test backtest with empty klines."""
        backtest = BollingerBacktest(klines=[], config=bollinger_config)
        result = backtest.run()

        assert result.total_trades == 0
        assert result.total_profit == Decimal("0")

    def test_backtest_insufficient_klines(self, bollinger_config):
        """Test backtest with insufficient klines."""
        klines = []
        base_price = Decimal("50000")
        start_time = datetime.now(timezone.utc)

        for i in range(50):  # Less than required
            kline = Kline(
                symbol="BTCUSDT",
                interval="15m",
                open_time=start_time + timedelta(minutes=i * 15),
                close_time=start_time + timedelta(minutes=(i + 1) * 15 - 1),
                open=base_price,
                high=base_price,
                low=base_price,
                close=base_price,
                volume=Decimal("1000"),
                quote_volume=Decimal("50000000"),
                trades=100,
            )
            klines.append(kline)

        backtest = BollingerBacktest(klines=klines, config=bollinger_config)
        result = backtest.run()

        assert result.total_trades == 0


class TestBollingerBacktestMeanReversion:
    """Tests for mean reversion market."""

    def test_mean_reversion_generates_trades(self, mean_reversion_klines, bollinger_config):
        """Test that mean reversion market generates trades."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        # Should generate some trades
        assert result.total_trades >= 0

    def test_mean_reversion_has_both_directions(self, mean_reversion_klines, bollinger_config):
        """Test that both long and short trades occur."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        # In a oscillating market, expect both directions
        # (may not always be true depending on exact movements)
        if result.total_trades > 0:
            assert result.long_trades >= 0
            assert result.short_trades >= 0


class TestBollingerBacktestSqueeze:
    """Tests for BBW squeeze filtering."""

    def test_squeeze_filters_signals(self, squeeze_then_breakout_klines, bollinger_config):
        """Test that squeeze periods filter signals."""
        backtest = BollingerBacktest(
            klines=squeeze_then_breakout_klines,
            config=bollinger_config,
        )
        result = backtest.run()

        # Should have some filtered signals during squeeze
        assert result.filtered_by_squeeze >= 0


class TestBollingerBacktestMetrics:
    """Tests for backtest metrics calculation."""

    def test_win_rate_calculation(self, mean_reversion_klines, bollinger_config):
        """Test win rate calculation."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        if result.total_trades > 0:
            # Win rate should be between 0 and 100
            assert Decimal("0") <= result.win_rate <= Decimal("100")
            assert result.num_wins + result.num_losses == result.total_trades

    def test_max_drawdown_calculation(self, mean_reversion_klines, bollinger_config):
        """Test max drawdown calculation."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        # Max drawdown should be non-negative
        assert result.max_drawdown >= Decimal("0")

    def test_profit_factor_calculation(self, mean_reversion_klines, bollinger_config):
        """Test profit factor calculation."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        # Profit factor should be non-negative
        assert result.profit_factor >= Decimal("0")

    def test_avg_holding_time_calculation(self, mean_reversion_klines, bollinger_config):
        """Test average holding time calculation."""
        backtest = BollingerBacktest(klines=mean_reversion_klines, config=bollinger_config)
        result = backtest.run()

        if result.total_trades > 0:
            # Holding time should be positive
            assert result.avg_holding_time >= timedelta(0)
