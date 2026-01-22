"""
Bollinger Trend Bot Data Models.

Provides data models for Bollinger Trend strategy (Supertrend + BB combination)
running on futures market.

Strategy: Enter on BB band touch when aligned with Supertrend direction.
Exit: Supertrend flip or ATR stop loss.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


# =============================================================================
# Enums
# =============================================================================


class StrategyMode(str, Enum):
    """Trading strategy mode."""

    BOLLINGER_TREND = "bollinger_trend"  # Supertrend + BB combination


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

    TREND_FLIP = "trend_flip"    # Supertrend flipped direction
    STOP_LOSS = "stop_loss"      # ATR stop loss triggered
    MANUAL = "manual"            # Manual exit
    BOT_STOP = "bot_stop"        # Bot stopped


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BollingerConfig:
    """
    Bollinger Trend Bot configuration.

    ⚠️ 警告：此策略未通過 Walk-Forward 驗證，不建議用於實盤交易！

    驗證結果 (2025-01 ~ 2026-01, 1 年數據):
        - 報酬: +9.0%, Sharpe: 0.58, 回撤: 14.1%
        - Walk-Forward 一致性: 50% (3/6 時段獲利) ❌ 需要 ≥67%
        - 各時段: P1:-5% | P2:+4% | P3:-5% | P4:+4% | P5:+16% | P6:-3%

    結論：一致性僅 50%，表示策略在不同時期表現不穩定，
          有一半時間虧損，不建議實盤使用。

    策略邏輯:
    - 進場: Supertrend 看多時在 BB 下軌買入，看空時在 BB 上軌賣出
    - 出場: Supertrend 翻轉（主要）或 ATR 止損（保護）

    參數 (6 個):
    - bb_period: 20, bb_std: 2.5
    - st_atr_period: 20, st_atr_multiplier: 3.5
    - atr_stop_multiplier: 2.0
    - leverage: 5

    Example:
        >>> config = BollingerConfig(symbol="BTCUSDT", leverage=5)
    """

    symbol: str
    timeframe: str = "15m"

    # Bollinger Bands parameters
    bb_period: int = 20
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.5"))

    # Supertrend parameters
    st_atr_period: int = 20
    st_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.5"))

    # ATR Stop Loss
    atr_stop_multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Position settings
    leverage: int = 5
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # BBW filter (retained for indicator compatibility)
    bbw_lookback: int = 200
    bbw_threshold_pct: int = 20

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.bb_std, Decimal):
            self.bb_std = Decimal(str(self.bb_std))
        if not isinstance(self.st_atr_multiplier, Decimal):
            self.st_atr_multiplier = Decimal(str(self.st_atr_multiplier))
        if not isinstance(self.atr_stop_multiplier, Decimal):
            self.atr_stop_multiplier = Decimal(str(self.atr_stop_multiplier))
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.bb_period < 5 or self.bb_period > 100:
            raise ValueError(f"bb_period must be 5-100, got {self.bb_period}")

        if self.bb_std < Decimal("0.5") or self.bb_std > Decimal("5.0"):
            raise ValueError(f"bb_std must be 0.5-5.0, got {self.bb_std}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")

        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")

        if self.st_atr_period < 5 or self.st_atr_period > 50:
            raise ValueError(f"st_atr_period must be 5-50, got {self.st_atr_period}")

        if self.st_atr_multiplier < Decimal("1.0") or self.st_atr_multiplier > Decimal("10.0"):
            raise ValueError(f"st_atr_multiplier must be 1.0-10.0, got {self.st_atr_multiplier}")

        if self.atr_stop_multiplier < Decimal("0.5") or self.atr_stop_multiplier > Decimal("5.0"):
            raise ValueError(f"atr_stop_multiplier must be 0.5-5.0, got {self.atr_stop_multiplier}")


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


@dataclass
class SupertrendData:
    """
    Supertrend indicator data for BOLLINGER_TREND mode.

    Attributes:
        upper_band: Upper Supertrend band
        lower_band: Lower Supertrend band
        supertrend: Current Supertrend value (support/resistance line)
        trend: Trend direction (1 = bullish, -1 = bearish)
        atr: Current ATR value
        timestamp: Calculation timestamp
    """

    upper_band: Decimal
    lower_band: Decimal
    supertrend: Decimal
    trend: int  # 1 = bullish, -1 = bearish
    atr: Decimal
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ["upper_band", "lower_band", "supertrend", "atr"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def is_bullish(self) -> bool:
        """Check if trend is bullish."""
        return self.trend == 1

    @property
    def is_bearish(self) -> bool:
        """Check if trend is bearish."""
        return self.trend == -1


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
        stop_loss: ATR-based stop loss price
        bands: Bollinger Bands at signal time
        bbw: BBW data at signal time
        supertrend: Supertrend data
        timestamp: Signal timestamp
        reason: Signal reason description
        atr: ATR value (for stop loss calculation)
    """

    signal_type: SignalType
    bands: BollingerBands
    bbw: BBWData
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: str = ""
    entry_price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    supertrend: Optional[SupertrendData] = None

    def __post_init__(self):
        """Ensure Decimal types."""
        if self.entry_price is not None and not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))
        if self.stop_loss is not None and not isinstance(self.stop_loss, Decimal):
            self.stop_loss = Decimal(str(self.stop_loss))
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
        max_price: Maximum price since entry (for trailing stop)
        min_price: Minimum price since entry (for trailing stop)
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
    max_price: Optional[Decimal] = None  # Track max price for trailing stop
    min_price: Optional[Decimal] = None  # Track min price for trailing stop

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
        if self.max_price is not None and not isinstance(self.max_price, Decimal):
            self.max_price = Decimal(str(self.max_price))
        if self.min_price is not None and not isinstance(self.min_price, Decimal):
            self.min_price = Decimal(str(self.min_price))

        # Initialize max/min price from entry price if not set
        if self.max_price is None:
            self.max_price = self.entry_price
        if self.min_price is None:
            self.min_price = self.entry_price

    def update_extremes(self, current_price: Decimal) -> None:
        """Update max/min price tracking for trailing stop."""
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))
        if self.max_price is None or current_price > self.max_price:
            self.max_price = current_price
        if self.min_price is None or current_price < self.min_price:
            self.min_price = current_price

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
