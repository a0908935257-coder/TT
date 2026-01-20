"""
Grid Futures Bot Data Models.

Provides data models for futures-based grid trading with leverage,
trend filtering, and bidirectional trading support.

Optimized configuration based on backtesting:
- 年化 >30%, 回撤 <50%, Sharpe >1.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


# =============================================================================
# Enums
# =============================================================================


class GridDirection(str, Enum):
    """Grid trading direction mode."""

    LONG_ONLY = "long_only"        # Only long positions (buy low, sell high)
    SHORT_ONLY = "short_only"      # Only short positions (sell high, buy low)
    NEUTRAL = "neutral"            # Both directions simultaneously
    TREND_FOLLOW = "trend_follow"  # Follow trend: long in uptrend, short in downtrend


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
    """Reason for closing a position."""

    GRID_PROFIT = "grid_profit"      # Normal grid profit take
    TREND_CHANGE = "trend_change"    # Trend direction changed
    STOP_LOSS = "stop_loss"          # Stop loss triggered
    GRID_REBUILD = "grid_rebuild"    # Grid needs rebuilding
    MANUAL = "manual"                # Manual close
    BOT_STOP = "bot_stop"            # Bot stopped


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GridFuturesConfig:
    """
    Grid Futures Bot configuration.

    Optimized defaults based on backtesting (年化 113.6%, 回撤 24.7%, Sharpe 1.54):
    - leverage: 3x
    - grid_count: 15
    - direction: trend_follow
    - trend_period: 30
    - atr_multiplier: 2.0
    - position_size_pct: 10%

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe for indicators (default "1h")
        leverage: Futures leverage (default 3)
        margin_type: ISOLATED or CROSSED (default ISOLATED)

        # Grid settings
        grid_count: Number of grid levels (default 15)
        direction: Trading direction mode (default TREND_FOLLOW)

        # Trend filter
        use_trend_filter: Enable trend-based direction (default True)
        trend_period: SMA period for trend detection (default 30)

        # Dynamic range
        use_atr_range: Use ATR for dynamic grid range (default True)
        atr_period: ATR calculation period (default 14)
        atr_multiplier: ATR multiplier for range (default 2.0)
        fallback_range_pct: Range when ATR unavailable (default 8%)

        # Position sizing
        max_capital: Maximum capital to use (None = all balance)
        position_size_pct: Size per grid trade as % of capital (default 10%)
        max_position_pct: Maximum total position as % of capital (default 50%)

        # Risk management
        stop_loss_pct: Stop loss percentage per position (default 5%)
        rebuild_threshold_pct: Price deviation to trigger rebuild (default 2%)

    Example:
        >>> config = GridFuturesConfig(
        ...     symbol="BTCUSDT",
        ...     leverage=3,
        ...     grid_count=15,
        ...     direction=GridDirection.TREND_FOLLOW,
        ... )
    """

    symbol: str
    timeframe: str = "1h"
    leverage: int = 3
    margin_type: str = "ISOLATED"

    # Grid settings
    grid_count: int = 15
    direction: GridDirection = GridDirection.TREND_FOLLOW

    # Trend filter (optimized: period=30)
    use_trend_filter: bool = True
    trend_period: int = 30

    # Dynamic ATR range (optimized: multiplier=2.0)
    use_atr_range: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    fallback_range_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))

    # Position sizing (optimized: 10% per trade)
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Risk management
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))
    rebuild_threshold_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))

    # Fee rate (Binance Futures: 0.04% maker/taker)
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure Decimal types
        decimal_fields = [
            'atr_multiplier', 'fallback_range_pct', 'position_size_pct',
            'max_position_pct', 'stop_loss_pct', 'rebuild_threshold_pct', 'fee_rate'
        ]
        for field_name in decimal_fields:
            value = getattr(self, field_name)
            if not isinstance(value, Decimal):
                setattr(self, field_name, Decimal(str(value)))

        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))

        # Convert direction from string if needed
        if isinstance(self.direction, str):
            self.direction = GridDirection(self.direction)

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.grid_count < 5 or self.grid_count > 100:
            raise ValueError(f"grid_count must be 5-100, got {self.grid_count}")

        if self.trend_period < 10 or self.trend_period > 200:
            raise ValueError(f"trend_period must be 10-200, got {self.trend_period}")

        if self.atr_period < 5 or self.atr_period > 50:
            raise ValueError(f"atr_period must be 5-50, got {self.atr_period}")

        if self.atr_multiplier < Decimal("0.5") or self.atr_multiplier > Decimal("5.0"):
            raise ValueError(f"atr_multiplier must be 0.5-5.0, got {self.atr_multiplier}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("0.5"):
            raise ValueError(f"position_size_pct must be 1%-50%, got {self.position_size_pct}")

        if self.max_position_pct < Decimal("0.1") or self.max_position_pct > Decimal("1.0"):
            raise ValueError(f"max_position_pct must be 10%-100%, got {self.max_position_pct}")

        valid_timeframes = ["15m", "30m", "1h", "4h"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")

        if self.margin_type not in ["ISOLATED", "CROSSED"]:
            raise ValueError(f"margin_type must be ISOLATED or CROSSED")


# =============================================================================
# Grid Data
# =============================================================================


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


# =============================================================================
# Position
# =============================================================================


@dataclass
class FuturesPosition:
    """Current futures position."""

    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    leverage: int
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time: Optional[datetime] = None
    liquidation_price: Optional[Decimal] = None

    def __post_init__(self):
        for attr in ['entry_price', 'quantity', 'unrealized_pnl']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))
        if self.liquidation_price is not None and not isinstance(self.liquidation_price, Decimal):
            self.liquidation_price = Decimal(str(self.liquidation_price))

    @property
    def notional_value(self) -> Decimal:
        return self.entry_price * self.quantity

    @property
    def margin_required(self) -> Decimal:
        if self.leverage <= 0:
            return self.notional_value
        return self.notional_value / Decimal(self.leverage)

    @property
    def is_long(self) -> bool:
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT

    def calculate_pnl(self, current_price: Decimal) -> Decimal:
        """Calculate unrealized PnL at current price."""
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        if self.side == PositionSide.LONG:
            return self.quantity * (current_price - self.entry_price) * Decimal(self.leverage)
        elif self.side == PositionSide.SHORT:
            return self.quantity * (self.entry_price - current_price) * Decimal(self.leverage)
        return Decimal("0")


# =============================================================================
# Trade Record
# =============================================================================


@dataclass
class GridTrade:
    """Completed grid trade record."""

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
    grid_level: int

    def __post_init__(self):
        for attr in ['entry_price', 'exit_price', 'quantity', 'pnl', 'fee']:
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


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class GridFuturesStats:
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

    def record_trade(self, trade: GridTrade) -> None:
        """Record a completed trade."""
        self.total_trades += 1
        self.total_pnl += trade.pnl
        self.total_fees += trade.fee

        if trade.side == PositionSide.LONG:
            self.long_trades += 1
        else:
            self.short_trades += 1

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
        }
