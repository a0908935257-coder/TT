"""
Supertrend Bot Data Models.

Provides data models for Supertrend trend-following strategy.

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割)

驗證結果 - 最佳配置 (2x, ATR=25, M=3.0, SL=3%):
    - Walk-Forward 一致性: 75% (6/8 時段獲利) ✓
    - 報酬: +6.7% (2 年), 年化 +3.3%
    - Sharpe: 0.39
    - 最大回撤: 11.5%

與原始配置比較:
    - 原始 (5x, ATR=10, SL=2%): +0.9%, 一致性 62%, DD 27.4%
    - 優化後 (2x, ATR=25, SL=3%): +6.7%, 一致性 75%, DD 11.5%
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

    ✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割)

    驗證結果 - 最佳配置:
    - Walk-Forward 一致性: 75% (6/8 時段獲利)
    - 報酬: +6.7% (2 年), 年化 +3.3%
    - Sharpe: 0.39, 最大回撤: 11.5%

    默認參數 (Walk-Forward 驗證通過):
    - ATR Period: 25 (更長週期減少雜訊)
    - ATR Multiplier: 3.0
    - Leverage: 2x (降低風險)
    - Stop Loss: 3% (給予更多空間)

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "15m")
        atr_period: ATR calculation period (default 25, validated)
        atr_multiplier: ATR multiplier for bands (default 3.0, validated)
        leverage: Futures leverage (default 2, validated)
        position_size_pct: Position size as percentage of balance (default 10%)
        use_trailing_stop: Enable trailing stop loss
        trailing_stop_pct: Trailing stop percentage (fallback)
    """
    symbol: str
    timeframe: str = "15m"
    atr_period: int = 25  # Walk-Forward validated: ATR=25 (更長週期)
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.0"))  # Validated: 3.0
    leverage: int = 2  # Walk-Forward validated: 2x (降低風險)
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED

    # Capital allocation (資金分配)
    max_capital: Optional[Decimal] = None  # 最大可用資金，None = 使用全部餘額
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))  # 最大持倉佔資金比例

    # Optional trailing stop (software-based)
    use_trailing_stop: bool = False
    trailing_stop_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))

    # Exchange-based stop loss (recommended for safety)
    use_exchange_stop_loss: bool = True  # Place STOP_MARKET order on exchange
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))  # Walk-Forward validated: 3%

    # Risk control (風險控制)
    daily_loss_limit_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))  # 每日虧損限制 5%
    max_consecutive_losses: int = 5  # 最大連續虧損次數

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
        if not isinstance(self.max_position_pct, Decimal):
            self.max_position_pct = Decimal(str(self.max_position_pct))
        if not isinstance(self.daily_loss_limit_pct, Decimal):
            self.daily_loss_limit_pct = Decimal(str(self.daily_loss_limit_pct))
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

        if self.max_position_pct < Decimal("0.1") or self.max_position_pct > Decimal("1.0"):
            raise ValueError(f"max_position_pct must be 10%-100%, got {self.max_position_pct}")


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
