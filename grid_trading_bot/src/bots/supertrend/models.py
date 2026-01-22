"""
Supertrend Bot Data Models.

Provides data models for Supertrend trend-following strategy.
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
    SIGNAL_FLIP = "signal_flip"  # Supertrend flipped direction
    STOP_LOSS = "stop_loss"
    MANUAL = "manual"
    BOT_STOP = "bot_stop"


@dataclass
class SupertrendConfig:
    """
    Supertrend Bot configuration.

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "15m")
        atr_period: ATR calculation period (default 10)
        atr_multiplier: ATR multiplier for bands (default 3.0)
        leverage: Futures leverage (default 5)
        position_size_pct: Position size as percentage of balance (default 10%)
        use_trailing_stop: Enable trailing stop loss
        trailing_stop_pct: Trailing stop percentage (fallback)

    樣本外驗證 (2 年數據, 70/30 分割, 2024-01 ~ 2026-01):
        - ATR=5, M=2.5, L=5x: 樣本外 +25.9%, 衰退 14.9%, DD 5.5% ✅ 預設
        - ATR=18, M=3.5, L=5x: 樣本外 +16.8%, 衰退 8.9%, DD 10.2%
        - ATR=12, M=3.5, L=5x: 樣本外 +16.1%, 衰退 18.8%, DD 4.5%

    過度擬合測試 (2024-01 ~ 2026-01):
        ATR=5, M=2.5 通過樣本外測試 ✅
        - 樣本內報酬: +30.4%
        - 樣本外報酬: +25.9%
        - 績效衰退: 14.9% (< 50% 門檻)
        - 樣本外最大回撤: 5.5%
    """
    symbol: str
    timeframe: str = "15m"
    atr_period: int = 5  # Out-of-sample validated: +25.9%
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("2.5"))  # Out-of-sample validated
    leverage: int = 5  # Risk management
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED

    # Capital allocation (資金分配)
    max_capital: Optional[Decimal] = None  # 最大可用資金，None = 使用全部餘額
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # Optional trailing stop (software-based)
    use_trailing_stop: bool = False
    trailing_stop_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))

    # Exchange-based stop loss (recommended for safety)
    use_exchange_stop_loss: bool = True  # Place STOP_MARKET order on exchange
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))  # 2% default

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if not isinstance(self.trailing_stop_pct, Decimal):
            self.trailing_stop_pct = Decimal(str(self.trailing_stop_pct))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.atr_period < 5 or self.atr_period > 50:
            raise ValueError(f"atr_period must be 5-50, got {self.atr_period}")

        if self.atr_multiplier < Decimal("1.0") or self.atr_multiplier > Decimal("10.0"):
            raise ValueError(f"atr_multiplier must be 1.0-10.0, got {self.atr_multiplier}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")


@dataclass
class SupertrendData:
    """Supertrend indicator data."""
    timestamp: datetime
    upper_band: Decimal
    lower_band: Decimal
    supertrend: Decimal  # Current supertrend value
    trend: int  # 1 = bullish, -1 = bearish
    atr: Decimal

    @property
    def is_bullish(self) -> bool:
        return self.trend == 1

    @property
    def is_bearish(self) -> bool:
        return self.trend == -1


@dataclass
class Position:
    """Current position information."""
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    highest_price: Decimal = field(default_factory=lambda: Decimal("0"))
    lowest_price: Decimal = field(default_factory=lambda: Decimal("0"))
    stop_loss_order_id: Optional[str] = None  # Exchange stop loss order ID
    stop_loss_price: Optional[Decimal] = None  # Stop loss trigger price

    @property
    def max_price(self) -> Optional[Decimal]:
        """Alias for highest_price."""
        return self.highest_price if self.highest_price != Decimal("0") else None

    @property
    def min_price(self) -> Optional[Decimal]:
        """Alias for lowest_price."""
        return self.lowest_price if self.lowest_price != Decimal("0") else None

    def update_extremes(self, current_price: Decimal) -> None:
        """Update highest/lowest prices for trailing stop."""
        if self.highest_price == Decimal("0") or current_price > self.highest_price:
            self.highest_price = current_price
        if self.lowest_price == Decimal("0") or current_price < self.lowest_price:
            self.lowest_price = current_price


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
    holding_duration: int  # in bars
