"""
Bollinger Band Mean Reversion Bot Data Models.

Provides data models for Bollinger Band mean reversion strategy
running on futures market.

Conforms to Prompt 64 specification.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


# =============================================================================
# Enums
# =============================================================================


class SignalType(str, Enum):
    """Trading signal type."""

    LONG = "long"    # Long signal (buy at lower band)
    SHORT = "short"  # Short signal (sell at upper band)
    NONE = "none"    # No signal


class PositionSide(str, Enum):
    """Position side for futures."""

    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class ExitReason(str, Enum):
    """Reason for exiting a position."""

    TAKE_PROFIT = "take_profit"  # Price returned to middle band
    STOP_LOSS = "stop_loss"      # Stop loss triggered
    TIMEOUT = "timeout"          # Max hold bars exceeded
    MANUAL = "manual"            # Manual exit
    BOT_STOP = "bot_stop"        # Bot stopped


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BollingerConfig:
    """
    Bollinger Bot configuration.

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "1h")
        bb_period: Bollinger Band period (default 15)
        bb_std: Standard deviation multiplier (default 2.0)
        bbw_lookback: BBW history lookback period (default 200)
        bbw_threshold_pct: BBW squeeze threshold percentile (default 20)
        stop_loss_pct: Stop loss percentage (default 1.5%, fallback if ATR disabled)
        max_hold_bars: Maximum bars to hold position (default 24)
        leverage: Futures leverage (default 2)
        position_size_pct: Position size as percentage of balance (default 10%)

        # Trend Filter (optimized)
        use_trend_filter: Enable trend filter (default True)
        trend_period: SMA period for trend detection (default 50)

        # ATR Stop Loss (optimized)
        use_atr_stop: Use ATR-based dynamic stop loss (default True)
        atr_period: ATR calculation period (default 14)
        atr_multiplier: ATR multiplier for stop distance (default 2.5)

    Example:
        >>> config = BollingerConfig(
        ...     symbol="BTCUSDT",
        ...     leverage=3,
        ...     use_trend_filter=True,
        ... )
    """

    symbol: str
    timeframe: str = "1h"  # Optimized: 1h has better signal quality
    bb_period: int = 15  # Optimized: shorter period is more responsive
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.0"))
    bbw_lookback: int = 200
    bbw_threshold_pct: int = 20
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.015"))
    max_hold_bars: int = 24  # Optimized: longer hold time
    leverage: int = 2
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # Trend Filter
    use_trend_filter: bool = True
    trend_period: int = 50

    # ATR Stop Loss
    use_atr_stop: bool = True
    atr_period: int = 14
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.5"))

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure Decimal types
        if not isinstance(self.bb_std, Decimal):
            self.bb_std = Decimal(str(self.bb_std))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.bb_period < 5 or self.bb_period > 100:
            raise ValueError(f"bb_period must be 5-100, got {self.bb_period}")

        if self.bb_std < Decimal("0.5") or self.bb_std > Decimal("5.0"):
            raise ValueError(f"bb_std must be 0.5-5.0, got {self.bb_std}")

        if self.bbw_lookback < 50 or self.bbw_lookback > 500:
            raise ValueError(f"bbw_lookback must be 50-500, got {self.bbw_lookback}")

        if self.bbw_threshold_pct < 5 or self.bbw_threshold_pct > 50:
            raise ValueError(f"bbw_threshold_pct must be 5-50, got {self.bbw_threshold_pct}")

        if self.stop_loss_pct < Decimal("0.005") or self.stop_loss_pct > Decimal("0.10"):
            raise ValueError(f"stop_loss_pct must be 0.5%-10%, got {self.stop_loss_pct}")

        if self.max_hold_bars < 1 or self.max_hold_bars > 100:
            raise ValueError(f"max_hold_bars must be 1-100, got {self.max_hold_bars}")

        if self.leverage < 1 or self.leverage > 20:
            raise ValueError(f"leverage must be 1-20, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")

        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")

        # Trend filter validation
        if self.trend_period < 10 or self.trend_period > 200:
            raise ValueError(f"trend_period must be 10-200, got {self.trend_period}")

        # ATR validation
        if self.atr_period < 5 or self.atr_period > 50:
            raise ValueError(f"atr_period must be 5-50, got {self.atr_period}")

        if self.atr_multiplier < Decimal("0.5") or self.atr_multiplier > Decimal("5.0"):
            raise ValueError(f"atr_multiplier must be 0.5-5.0, got {self.atr_multiplier}")


# =============================================================================
# Indicator Data
# =============================================================================


@dataclass
class BollingerBands:
    """
    Bollinger Bands calculation result.

    Attributes:
        upper: Upper band (middle + std * multiplier)
        middle: Middle band (SMA)
        lower: Lower band (middle - std * multiplier)
        std: Standard deviation
        timestamp: Calculation timestamp
    """

    upper: Decimal
    middle: Decimal
    lower: Decimal
    std: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ["upper", "middle", "lower", "std"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def width(self) -> Decimal:
        """Band width (upper - lower)."""
        return self.upper - self.lower

    @property
    def width_percent(self) -> Decimal:
        """Band width as percentage of middle."""
        if self.middle <= 0:
            return Decimal("0")
        return (self.width / self.middle) * Decimal("100")


@dataclass
class BBWData:
    """
    Bollinger Band Width (BBW) filter data.

    BBW is used to detect squeeze conditions where mean reversion
    is less reliable due to potential breakout.

    Attributes:
        bbw: Current BBW value (upper - lower) / middle
        bbw_percentile: BBW percentile rank in history
        is_squeeze: Whether in squeeze state (percentile < threshold)
        threshold: Squeeze threshold value
    """

    bbw: Decimal
    bbw_percentile: int
    is_squeeze: bool
    threshold: Decimal

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.bbw, Decimal):
            self.bbw = Decimal(str(self.bbw))
        if not isinstance(self.threshold, Decimal):
            self.threshold = Decimal(str(self.threshold))


# =============================================================================
# Signal
# =============================================================================


@dataclass
class Signal:
    """
    Trading signal.

    Attributes:
        signal_type: Signal type (LONG, SHORT, NONE)
        entry_price: Suggested entry price (band price)
        take_profit: Take profit price (middle band)
        stop_loss: Stop loss price
        bands: Bollinger Bands at signal time
        bbw: BBW data at signal time
        timestamp: Signal timestamp
        reason: Signal reason description
        trend_sma: Trend SMA value (for trend filter)
        atr: ATR value (for dynamic stop loss)
    """

    signal_type: SignalType
    bands: BollingerBands
    bbw: BBWData
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    entry_price: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    trend_sma: Optional[Decimal] = None
    atr: Optional[Decimal] = None

    def __post_init__(self):
        """Ensure Decimal types."""
        if self.entry_price is not None and not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if self.take_profit is not None and not isinstance(self.take_profit, Decimal):
            self.take_profit = Decimal(str(self.take_profit))
        if self.stop_loss is not None and not isinstance(self.stop_loss, Decimal):
            self.stop_loss = Decimal(str(self.stop_loss))
        if self.trend_sma is not None and not isinstance(self.trend_sma, Decimal):
            self.trend_sma = Decimal(str(self.trend_sma))
        if self.atr is not None and not isinstance(self.atr, Decimal):
            self.atr = Decimal(str(self.atr))

    @property
    def is_valid(self) -> bool:
        """Check if this is a valid trading signal."""
        return self.signal_type != SignalType.NONE

    @property
    def is_long(self) -> bool:
        """Check if this is a long signal."""
        return self.signal_type == SignalType.LONG

    @property
    def is_short(self) -> bool:
        """Check if this is a short signal."""
        return self.signal_type == SignalType.SHORT


# =============================================================================
# Position
# =============================================================================


@dataclass
class Position:
    """
    Current futures position.

    Attributes:
        symbol: Trading pair
        side: Position side (LONG/SHORT)
        entry_price: Entry price
        quantity: Position quantity
        leverage: Leverage used
        unrealized_pnl: Unrealized profit/loss
        entry_time: Entry timestamp
        entry_bar: K-line bar number at entry
        take_profit_price: Take profit price
        stop_loss_price: Stop loss price
    """

    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    leverage: int
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time: Optional[datetime] = None
    entry_bar: int = 0
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ["entry_price", "quantity", "unrealized_pnl"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

        if self.take_profit_price is not None and not isinstance(self.take_profit_price, Decimal):
            self.take_profit_price = Decimal(str(self.take_profit_price))
        if self.stop_loss_price is not None and not isinstance(self.stop_loss_price, Decimal):
            self.stop_loss_price = Decimal(str(self.stop_loss_price))

    @property
    def notional_value(self) -> Decimal:
        """Position notional value."""
        return self.entry_price * self.quantity

    @property
    def margin_required(self) -> Decimal:
        """Required margin for this position."""
        if self.leverage <= 0:
            return self.notional_value
        return self.notional_value / Decimal(self.leverage)

    @property
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.side == PositionSide.SHORT


# =============================================================================
# Trade Record
# =============================================================================


@dataclass
class TradeRecord:
    """
    Completed trade record.

    Attributes:
        trade_id: Unique trade ID
        symbol: Trading pair
        side: Position side
        entry_price: Entry price
        exit_price: Exit price
        quantity: Trade quantity
        pnl: Realized profit/loss
        pnl_pct: PnL percentage
        fee: Total fees paid
        entry_time: Entry timestamp
        exit_time: Exit timestamp
        exit_reason: Reason for exit
        hold_bars: Number of K-line bars held
    """

    trade_id: str
    symbol: str
    side: PositionSide
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    pnl_pct: Decimal
    fee: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time: Optional[datetime] = None
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    hold_bars: int = 0

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ["entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "fee"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def is_profitable(self) -> bool:
        """Check if trade was profitable."""
        return self.pnl > 0

    @property
    def net_pnl(self) -> Decimal:
        """Net PnL after fees."""
        return self.pnl - self.fee


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class BollingerBotStats:
    """
    Bot trading statistics.

    Attributes:
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        total_pnl: Total profit/loss
        win_rate: Win rate percentage
        avg_win: Average winning trade PnL
        avg_loss: Average losing trade PnL
        profit_factor: Gross profit / gross loss
        max_consecutive_loss: Maximum consecutive losing trades
        signals_generated: Total signals generated
        signals_filtered: Signals filtered by BBW squeeze
    """

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    win_rate: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_win: Decimal = field(default_factory=lambda: Decimal("0"))
    avg_loss: Decimal = field(default_factory=lambda: Decimal("0"))
    profit_factor: Decimal = field(default_factory=lambda: Decimal("0"))
    max_consecutive_loss: int = 0
    signals_generated: int = 0
    signals_filtered: int = 0

    # Internal tracking
    _current_consecutive_loss: int = field(default=0, repr=False)
    _total_win_pnl: Decimal = field(default_factory=lambda: Decimal("0"), repr=False)
    _total_loss_pnl: Decimal = field(default_factory=lambda: Decimal("0"), repr=False)

    def record_trade(self, pnl: Decimal) -> None:
        """
        Record a completed trade.

        Args:
            pnl: Trade profit/loss
        """
        if not isinstance(pnl, Decimal):
            pnl = Decimal(str(pnl))

        self.total_trades += 1
        self.total_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
            self._total_win_pnl += pnl
            self._current_consecutive_loss = 0
        else:
            self.losing_trades += 1
            self._total_loss_pnl += abs(pnl)
            self._current_consecutive_loss += 1
            if self._current_consecutive_loss > self.max_consecutive_loss:
                self.max_consecutive_loss = self._current_consecutive_loss

        # Update derived metrics
        self._update_metrics()

    def _update_metrics(self) -> None:
        """Update derived metrics."""
        if self.total_trades > 0:
            self.win_rate = Decimal(self.winning_trades) / Decimal(self.total_trades)

        if self.winning_trades > 0:
            self.avg_win = self._total_win_pnl / Decimal(self.winning_trades)

        if self.losing_trades > 0:
            self.avg_loss = self._total_loss_pnl / Decimal(self.losing_trades)

        if self._total_loss_pnl > 0:
            self.profit_factor = self._total_win_pnl / self._total_loss_pnl
        elif self._total_win_pnl > 0:
            self.profit_factor = Decimal("999")  # All wins

    def record_signal(self, filtered: bool = False) -> None:
        """
        Record a signal generation.

        Args:
            filtered: Whether signal was filtered by BBW
        """
        self.signals_generated += 1
        if filtered:
            self.signals_filtered += 1

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "total_pnl": str(self.total_pnl),
            "win_rate": f"{self.win_rate * 100:.1f}%",
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "profit_factor": str(self.profit_factor),
            "max_consecutive_loss": self.max_consecutive_loss,
            "signals_generated": self.signals_generated,
            "signals_filtered": self.signals_filtered,
        }
