"""
RSI Momentum Bot Data Models.

Provides data models for RSI-based momentum trading strategy.

Strategy: RSI crossover (trend following, not mean reversion)
- Long when RSI crosses above entry_level + threshold
- Short when RSI crosses below entry_level - threshold

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割)

驗證結果 - 最佳配置 RSI(21), 2x, SL 4%, TP 8%:
    - Walk-Forward 一致性: 88% (7/8 時段獲利) ✓
    - 報酬: +7.74% (2 年)
    - Sharpe: 0.80
    - 最大回撤: 6.5%
    - OOS 效率: 140% (OOS 表現優於樣本內)

OOS 測試結果 (2025-07 ~ 2026-01):
    - OOS 報酬: +2.1%, Sharpe 0.86, DD 2.1%
    - 月均報酬: OOS +0.32%/月 > IS +0.23%/月
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

    ✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割)

    驗證結果:
    - Walk-Forward 一致性: 88% (7/8 時段獲利)
    - OOS 效率: 140%
    - 報酬: +7.74%, Sharpe: 0.80, DD: 6.5%

    默認參數 (Walk-Forward + OOS 驗證通過):
    - RSI Period: 21
    - Entry Level: 50, Momentum Threshold: 5
    - Leverage: 2x (降低風險)
    - Stop Loss: 4%, Take Profit: 8%

    Example:
        >>> config = RSIConfig(symbol="BTCUSDT")  # 使用默認參數
    """
    symbol: str
    timeframe: str = "15m"
    rsi_period: int = 21  # Walk-Forward validated: RSI=21, 88% consistency
    entry_level: int = 50  # Center level for RSI crossover
    momentum_threshold: int = 5  # RSI must cross entry_level by this amount
    leverage: int = 2  # 降低槓桿提高穩定性
    margin_type: str = "ISOLATED"

    # Capital allocation
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))  # 最大持倉佔資金比例

    # Risk management - Walk-Forward 驗證通過的參數
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))  # 4%
    take_profit_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))  # 8%

    # Exchange-based stop loss
    use_exchange_stop_loss: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if not isinstance(self.max_position_pct, Decimal):
            self.max_position_pct = Decimal(str(self.max_position_pct))
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

        if self.max_position_pct < Decimal("0.1") or self.max_position_pct > Decimal("1.0"):
            raise ValueError(f"max_position_pct must be 10%-100%, got {self.max_position_pct}")


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
