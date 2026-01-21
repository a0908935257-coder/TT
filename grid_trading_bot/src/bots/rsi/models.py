"""
RSI Momentum Bot Data Models.

Provides data models for RSI-based momentum trading strategy.

Strategy: RSI crossover (trend following, not mean reversion)
- Long when RSI crosses above entry_level + threshold
- Short when RSI crosses below entry_level - threshold

Optimized configuration based on backtesting (1 year):
- RSI Period: 21
- Entry Level: 50, Momentum Threshold: 5
- Leverage: 5x
- Walk-Forward 一致性: 67%
- Sharpe: 1.03
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


class PositionSide(str, Enum):
    """Position side for futures."""
    LONG = "LONG"
    SHORT = "SHORT"
    NONE = "NONE"


class SignalType(str, Enum):
    """Trading signal type."""
    LONG = "long"
    SHORT = "short"
    NONE = "none"


class ExitReason(str, Enum):
    """Reason for exiting a position."""
    RSI_EXIT = "rsi_exit"      # RSI reversal signal
    STOP_LOSS = "stop_loss"    # Stop loss triggered
    TAKE_PROFIT = "take_profit"  # Take profit reached
    MANUAL = "manual"
    BOT_STOP = "bot_stop"


@dataclass
class RSIConfig:
    """
    RSI Momentum Bot configuration.

    Uses momentum strategy (trend following) instead of mean reversion:
    - Long when RSI crosses above entry_level + momentum_threshold
    - Short when RSI crosses below entry_level - momentum_threshold
    - Exit on opposite RSI crossover or SL/TP

    Optimized defaults based on 1 year backtesting (67% walk-forward):
    - RSI Period: 21
    - Entry Level: 50
    - Momentum Threshold: 5
    - Leverage: 5x
    - Stop Loss: 2%
    - Take Profit: 4%
    - Sharpe: 1.03, Max DD: 6.9%

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "15m")
        rsi_period: RSI calculation period (default 21)
        entry_level: RSI center level for crossover detection (default 50)
        momentum_threshold: RSI must cross by this amount to trigger (default 5)
        leverage: Futures leverage (default 5)
        position_size_pct: Position size as percentage of balance (default 10%)
        stop_loss_pct: Stop loss percentage (default 2%)
        take_profit_pct: Take profit percentage (default 4%)
    """
    symbol: str
    timeframe: str = "15m"
    rsi_period: int = 21
    entry_level: int = 50  # Center level for RSI crossover
    momentum_threshold: int = 5  # RSI must cross entry_level by this amount
    leverage: int = 5
    margin_type: str = "ISOLATED"

    # Deprecated but kept for backwards compatibility
    oversold: int = 30  # Used as entry_level - momentum_threshold fallback
    overbought: int = 70  # Used as entry_level + momentum_threshold fallback

    # Capital allocation
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # Risk management
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))

    # Exchange-based stop loss
    use_exchange_stop_loss: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if not isinstance(self.take_profit_pct, Decimal):
            self.take_profit_pct = Decimal(str(self.take_profit_pct))
        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.rsi_period < 5 or self.rsi_period > 50:
            raise ValueError(f"rsi_period must be 5-50, got {self.rsi_period}")

        if self.entry_level < 30 or self.entry_level > 70:
            raise ValueError(f"entry_level must be 30-70, got {self.entry_level}")

        if self.momentum_threshold < 1 or self.momentum_threshold > 20:
            raise ValueError(f"momentum_threshold must be 1-20, got {self.momentum_threshold}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")


@dataclass
class RSIData:
    """RSI indicator data."""
    timestamp: datetime
    rsi: Decimal
    prev_rsi: Decimal = field(default_factory=lambda: Decimal("50"))
    entry_level: int = 50
    momentum_threshold: int = 5

    @property
    def is_bullish_crossover(self) -> bool:
        """RSI crossed above entry_level + threshold (bullish momentum)."""
        threshold = self.entry_level + self.momentum_threshold
        return float(self.prev_rsi) <= self.entry_level and float(self.rsi) > threshold

    @property
    def is_bearish_crossover(self) -> bool:
        """RSI crossed below entry_level - threshold (bearish momentum)."""
        threshold = self.entry_level - self.momentum_threshold
        return float(self.prev_rsi) >= self.entry_level and float(self.rsi) < threshold

    @property
    def signal(self) -> SignalType:
        if self.is_bullish_crossover:
            return SignalType.LONG
        elif self.is_bearish_crossover:
            return SignalType.SHORT
        return SignalType.NONE


@dataclass
class Position:
    """Current position information."""
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_rsi: Decimal = field(default_factory=lambda: Decimal("50"))
    stop_loss_order_id: Optional[str] = None
    stop_loss_price: Optional[Decimal] = None

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
class Trade:
    """Completed trade record."""
    side: PositionSide
    entry_price: Decimal
    exit_price: Decimal
    quantity: Decimal
    pnl: Decimal
    fee: Decimal
    entry_time: datetime
    exit_time: datetime
    exit_reason: ExitReason
    entry_rsi: Decimal
    exit_rsi: Decimal
