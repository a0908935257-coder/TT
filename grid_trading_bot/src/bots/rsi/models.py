"""
RSI Mean Reversion Bot Data Models.

Provides data models for RSI-based mean reversion trading strategy.

Optimized configuration based on backtesting:
- RSI Period: 14
- Oversold: 20, Overbought: 80
- Leverage: 7x
- Walk-Forward 一致性: 83%
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
    RSI_EXIT = "rsi_exit"      # RSI returned to neutral
    STOP_LOSS = "stop_loss"    # Stop loss triggered
    TAKE_PROFIT = "take_profit"  # Take profit reached
    MANUAL = "manual"
    BOT_STOP = "bot_stop"


@dataclass
class RSIConfig:
    """
    RSI Mean Reversion Bot configuration.

    Optimized defaults based on backtesting (83% walk-forward consistency):
    - RSI Period: 14
    - Oversold: 20
    - Overbought: 80
    - Leverage: 7x
    - Stop Loss: 2%
    - Take Profit: 3%

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "15m")
        rsi_period: RSI calculation period (default 14)
        oversold: RSI level to trigger long entry (default 20)
        overbought: RSI level to trigger short entry (default 80)
        exit_level: RSI level to exit positions (default 50)
        leverage: Futures leverage (default 7)
        position_size_pct: Position size as percentage of balance (default 10%)
        stop_loss_pct: Stop loss percentage (default 2%)
        take_profit_pct: Take profit percentage (default 3%)
    """
    symbol: str
    timeframe: str = "15m"
    rsi_period: int = 14
    oversold: int = 20
    overbought: int = 80
    exit_level: int = 50
    leverage: int = 7
    margin_type: str = "ISOLATED"

    # Capital allocation
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # Risk management
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))

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

        if self.oversold < 10 or self.oversold > 40:
            raise ValueError(f"oversold must be 10-40, got {self.oversold}")

        if self.overbought < 60 or self.overbought > 90:
            raise ValueError(f"overbought must be 60-90, got {self.overbought}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")


@dataclass
class RSIData:
    """RSI indicator data."""
    timestamp: datetime
    rsi: Decimal
    is_oversold: bool
    is_overbought: bool

    @property
    def signal(self) -> SignalType:
        if self.is_oversold:
            return SignalType.LONG
        elif self.is_overbought:
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
