"""
RSI-Grid Hybrid Bot Data Models.

Provides data models for RSI-Grid hybrid trading strategy.

Strategy: RSI Zone + Grid Entry + Trend Filter
- RSI determines allowed direction (oversold=long, overbought=short)
- Grid provides entry points at ATR-based price levels
- SMA trend filter adds direction bias

Design Goals:
- Target Sharpe > 3.0
- Walk-Forward Consistency > 90%
- Win Rate > 70%
- Max Drawdown < 5%
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


class RSIZone(str, Enum):
    """RSI zone classification."""
    OVERSOLD = "oversold"        # RSI < 30: Only LONG
    NEUTRAL = "neutral"          # RSI 30-70: Follow trend
    OVERBOUGHT = "overbought"    # RSI > 70: Only SHORT


class PositionSide(str, Enum):
    """Position side for futures."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class GridLevelState(str, Enum):
    """Grid level state."""
    EMPTY = "empty"
    LONG_FILLED = "long_filled"
    SHORT_FILLED = "short_filled"


class ExitReason(str, Enum):
    """Reason for exiting a position."""
    RSI_EXIT = "rsi_exit"          # RSI reversal signal
    GRID_PROFIT = "grid_profit"    # Grid take profit
    STOP_LOSS = "stop_loss"        # Stop loss triggered
    TAKE_PROFIT = "take_profit"    # Take profit reached
    GRID_REBUILD = "grid_rebuild"  # Grid rebuilt
    TREND_CHANGE = "trend_change"  # Trend direction changed
    MANUAL = "manual"
    BOT_STOP = "bot_stop"


@dataclass
class RSIGridConfig:
    """
    RSI-Grid Hybrid Bot configuration.

    Combines RSI mean reversion with Grid entry mechanism:
    - RSI determines allowed direction based on zone
    - Grid provides precise entry points
    - SMA trend filter adds direction bias

    Design Goals:
    - Target Sharpe > 3.0 (Grid Futures baseline: 4.50)
    - Walk-Forward Consistency > 90%
    - Win Rate > 70%
    - Max Drawdown < 5%

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "15m")
        leverage: Futures leverage (default 2)
        margin_type: ISOLATED or CROSSED (default ISOLATED)

        # RSI Parameters
        rsi_period: RSI calculation period (default 14)
        oversold_level: RSI oversold threshold (default 30)
        overbought_level: RSI overbought threshold (default 70)

        # Grid Parameters
        grid_count: Number of grid levels (default 10)
        atr_period: ATR calculation period (default 14)
        atr_multiplier: ATR multiplier for grid range (default 3.0)

        # Trend Filter
        trend_sma_period: SMA period for trend detection (default 20)
        use_trend_filter: Enable trend-based direction filter (default True)

        # Risk Management
        max_capital: Maximum capital to use
        position_size_pct: Size per trade as % of capital (default 10%)
        max_position_pct: Maximum total position as % (default 50%)
        stop_loss_atr_mult: Stop loss as ATR multiple (default 1.5)
        max_stop_loss_pct: Maximum stop loss percentage (default 3%)
        max_positions: Maximum concurrent positions (default 5)

    Example:
        >>> config = RSIGridConfig(symbol="BTCUSDT")  # Use defaults
    """

    symbol: str
    timeframe: str = "15m"
    leverage: int = 2
    margin_type: str = "ISOLATED"

    # RSI Parameters
    rsi_period: int = 14
    oversold_level: int = 30
    overbought_level: int = 70

    # Grid Parameters
    grid_count: int = 10
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.0"))

    # Trend Filter
    trend_sma_period: int = 20
    use_trend_filter: bool = True

    # Capital allocation
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Risk Management
    stop_loss_atr_mult: Decimal = field(default_factory=lambda: Decimal("1.5"))
    max_stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))
    take_profit_grids: int = 1
    max_positions: int = 5

    # Exchange-based stop loss
    use_exchange_stop_loss: bool = True

    # Risk control
    daily_loss_limit_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))
    max_consecutive_losses: int = 5

    # Fee rate
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    # Protective features (disabled by default based on backtest results showing
    # they hurt performance for low-frequency strategies like RSI Grid)
    use_hysteresis: bool = False  # Disable hysteresis (回測顯示對 RSI Grid 有負面影響)
    hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("0.002"))
    use_signal_cooldown: bool = False  # Disable signal cooldown (回測顯示對 RSI Grid 有負面影響)
    cooldown_bars: int = 2

    def __post_init__(self):
        """Validate and normalize configuration."""
        decimal_fields = [
            'atr_multiplier', 'position_size_pct', 'max_position_pct',
            'stop_loss_atr_mult', 'max_stop_loss_pct', 'daily_loss_limit_pct',
            'fee_rate', 'hysteresis_pct'
        ]
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.rsi_period < 5 or self.rsi_period > 50:
            raise ValueError(f"rsi_period must be 5-50, got {self.rsi_period}")

        if self.oversold_level < 10 or self.oversold_level > 40:
            raise ValueError(f"oversold_level must be 10-40, got {self.oversold_level}")

        if self.overbought_level < 60 or self.overbought_level > 90:
            raise ValueError(f"overbought_level must be 60-90, got {self.overbought_level}")

        if self.grid_count < 5 or self.grid_count > 50:
            raise ValueError(f"grid_count must be 5-50, got {self.grid_count}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.atr_multiplier < Decimal("0.5") or self.atr_multiplier > Decimal("5.0"):
            raise ValueError(f"atr_multiplier must be 0.5-5.0, got {self.atr_multiplier}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("0.5"):
            raise ValueError(f"position_size_pct must be 1%-50%, got {self.position_size_pct}")


@dataclass
class GridLevel:
    """Individual grid level."""
    index: int
    price: Decimal
    state: GridLevelState = GridLevelState.EMPTY
    order_id: Optional[str] = None
    filled_at: Optional[datetime] = None

    def __post_init__(self):
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))


@dataclass
class GridSetup:
    """Complete grid setup."""
    center_price: Decimal
    upper_price: Decimal
    lower_price: Decimal
    grid_spacing: Decimal
    levels: list[GridLevel]
    atr_value: Optional[Decimal] = None
    range_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1

    def __post_init__(self):
        for attr in ['center_price', 'upper_price', 'lower_price', 'grid_spacing', 'range_pct']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))
        if self.atr_value is not None and not isinstance(self.atr_value, Decimal):
            self.atr_value = Decimal(str(self.atr_value))

    @property
    def grid_count(self) -> int:
        return len(self.levels) - 1 if self.levels else 0


@dataclass
class RSIGridPosition:
    """Current position information."""
    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    leverage: int
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time: Optional[datetime] = None
    entry_rsi: Decimal = field(default_factory=lambda: Decimal("50"))
    entry_zone: RSIZone = RSIZone.NEUTRAL
    grid_level: int = 0
    stop_loss_order_id: Optional[str] = None
    stop_loss_price: Optional[Decimal] = None

    def __post_init__(self):
        for attr in ['entry_price', 'quantity', 'unrealized_pnl', 'entry_rsi']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))
        if self.stop_loss_price is not None and not isinstance(self.stop_loss_price, Decimal):
            self.stop_loss_price = Decimal(str(self.stop_loss_price))

    @property
    def notional_value(self) -> Decimal:
        return self.entry_price * self.quantity

    @property
    def margin_required(self) -> Decimal:
        if self.leverage <= 0:
            return self.notional_value
        return self.notional_value / Decimal(self.leverage)

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL at current price."""
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        if self.side == PositionSide.LONG:
            return self.quantity * (current_price - self.entry_price) * Decimal(self.leverage)
        elif self.side == PositionSide.SHORT:
            return self.quantity * (self.entry_price - current_price) * Decimal(self.leverage)
        return Decimal("0")

    def calculate_pnl_pct(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL percentage."""
        if self.entry_price <= 0:
            return Decimal("0")
        if self.side == PositionSide.LONG:
            return (current_price - self.entry_price) / self.entry_price
        elif self.side == PositionSide.SHORT:
            return (self.entry_price - current_price) / self.entry_price
        return Decimal("0")


@dataclass
class RSIGridTrade:
    """Completed trade record."""
    trade_id: str
    symbol: str
    side: PositionSide
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    fee: Decimal
    leverage: int
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    entry_rsi: Decimal
    exit_rsi: Decimal
    entry_zone: RSIZone
    grid_level: int

    def __post_init__(self):
        for attr in ['entry_price', 'exit_price', 'quantity', 'pnl', 'fee', 'entry_rsi', 'exit_rsi']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    @property
    def net_pnl(self) -> Decimal:
        return self.pnl - self.fee

    @property
    def pnl_pct(self) -> Decimal:
        if self.entry_price <= 0:
            return Decimal("0")
        base_pnl = (self.exit_price - self.entry_price) / self.entry_price
        if self.side == PositionSide.SHORT:
            base_pnl = -base_pnl
        return base_pnl * Decimal(self.leverage) * Decimal("100")


@dataclass
class RSIGridStats:
    """Bot trading statistics."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    long_trades: int = 0
    short_trades: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    total_fees: Decimal = field(default_factory=lambda: Decimal("0"))
    grid_rebuilds: int = 0
    max_drawdown_pct: Decimal = field(default_factory=lambda: Decimal("0"))

    # Zone-based stats
    oversold_entries: int = 0
    overbought_entries: int = 0
    neutral_entries: int = 0

    # Internal tracking
    _peak_equity: Decimal = field(default_factory=lambda: Decimal("0"), repr=False)
    _total_win_pnl: Decimal = field(default_factory=lambda: Decimal("0"), repr=False)
    _total_loss_pnl: Decimal = field(default_factory=lambda: Decimal("0"), repr=False)

    @property
    def win_rate(self) -> Decimal:
        if self.total_trades == 0:
            return Decimal("0")
        return Decimal(self.winning_trades) / Decimal(self.total_trades) * Decimal("100")

    @property
    def profit_factor(self) -> Decimal:
        if self._total_loss_pnl == 0:
            return Decimal("999") if self._total_win_pnl > 0 else Decimal("0")
        return self._total_win_pnl / self._total_loss_pnl

    @property
    def avg_win(self) -> Decimal:
        if self.winning_trades == 0:
            return Decimal("0")
        return self._total_win_pnl / Decimal(self.winning_trades)

    @property
    def avg_loss(self) -> Decimal:
        if self.losing_trades == 0:
            return Decimal("0")
        return self._total_loss_pnl / Decimal(self.losing_trades)

    @property
    def net_pnl(self) -> Decimal:
        return self.total_pnl - self.total_fees

    def record_trade(self, trade: RSIGridTrade) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.total_fees += trade.fee

        if trade.side == PositionSide.LONG:
            self.long_trades += 1
        else:
            self.short_trades += 1

        # Zone tracking
        if trade.entry_zone == RSIZone.OVERSOLD:
            self.oversold_entries += 1
        elif trade.entry_zone == RSIZone.OVERBOUGHT:
            self.overbought_entries += 1
        else:
            self.neutral_entries += 1

        if trade.pnl > 0:
            self.winning_trades += 1
            self._total_win_pnl += trade.pnl
        else:
            self.losing_trades += 1
            self._total_loss_pnl += abs(trade.pnl)

    def update_drawdown(self, current_equity: Decimal, initial_capital: Decimal) -> None:
        """Update max drawdown tracking."""
        if current_equity > self._peak_equity:
            self._peak_equity = current_equity

        if self._peak_equity > 0:
            drawdown = (self._peak_equity - current_equity) / self._peak_equity * Decimal("100")
            if drawdown > self.max_drawdown_pct:
                self.max_drawdown_pct = drawdown

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "long_trades": self.long_trades,
            "short_trades": self.short_trades,
            "win_rate": f"{self.win_rate:.1f}%",
            "total_pnl": str(self.total_pnl),
            "total_fees": str(self.total_fees),
            "net_pnl": str(self.net_pnl),
            "profit_factor": f"{self.profit_factor:.2f}",
            "grid_rebuilds": self.grid_rebuilds,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.1f}%",
            "zone_distribution": {
                "oversold": self.oversold_entries,
                "neutral": self.neutral_entries,
                "overbought": self.overbought_entries,
            },
        }
