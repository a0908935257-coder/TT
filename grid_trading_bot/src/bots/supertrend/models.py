"""
Supertrend Bot Data Models.

Provides data models for Supertrend HYBRID_GRID strategy with RSI filter.

✅ 嚴格成本約束優化 + Walk-Forward 驗證 (2026-01-30)
成本模型: 手續費 0.06% + 滑價 0.05%

HYBRID_GRID 模式:
    - 雙向交易 + Supertrend 趨勢偏移
    - RSI 過濾器避免極端進場
    - 超時出場 + 網格自動重置

驗證結果 (2026-01-30):
    - 年化報酬: 227.32%, Sharpe: 9.64
    - 最大回撤: 1.80%, 勝率: 63.2%
    - Walk-Forward 一致性: 100% (9/9)
    - Monte Carlo: ROBUST (100%)

優化後參數:
    - ATR Period: 25, ATR Multiplier: 3.5
    - Grid Count: 10, Grid ATR Multiplier: 9.5
    - RSI Period: 21, Overbought: 71, Oversold: 31
    - Stop Loss: 1%, Trailing Stop: 3%
    - Leverage: 10x, Max Hold: 12 bars
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
    CAPITAL_RECALLED = "capital_recalled"


@dataclass
class SupertrendConfig:
    """
    Supertrend Bot configuration (HYBRID_GRID mode).

    ✅ 嚴格成本約束優化 + Walk-Forward 驗證 (2026-01-30)
    成本模型: 手續費 0.06% (含資金費率) + 滑價 0.05%

    優化後驗證結果:
    - 年化報酬: 227.32%, Sharpe: 9.64
    - 最大回撤: 1.80%, 勝率: 63.2%
    - Walk-Forward 一致性: 100% (9/9)
    - Monte Carlo: ROBUST (100%)
    - OOS/IS Sharpe: 0.96

    優化後默認參數:
    - Timeframe: 1h, Leverage: 10x
    - ATR Period: 25, ATR Multiplier: 3.5
    - Grid Count: 10, Grid ATR Multiplier: 9.5
    - Stop Loss: 1%, Trailing Stop: 3%
    - RSI(21, 71/31)

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe (default "1h")
        atr_period: ATR calculation period (default 25, optimized)
        atr_multiplier: ATR multiplier for bands (default 3.5, optimized)
        leverage: Futures leverage (default 10)
        position_size_pct: Position size as percentage of balance (default 10%)
    """
    symbol: str
    timeframe: str = "1h"
    atr_period: int = 25  # 優化後: 25
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("3.5"))  # 優化後: 3.5
    leverage: int = 10  # 優化後: 10x
    margin_type: str = "ISOLATED"  # ISOLATED or CROSSED

    # Capital allocation (資金分配)
    max_capital: Optional[Decimal] = None  # 最大可用資金，None = 使用全部餘額
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))  # 最大持倉佔資金比例

    # Grid Settings (網格設定) - 優化後
    grid_count: int = 10  # 優化後: 10
    grid_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("9.5"))  # 優化後: 9.5
    take_profit_grids: int = 1  # 止盈網格數

    # Trailing stop (優化後啟用)
    use_trailing_stop: bool = True
    trailing_stop_pct: Decimal = field(default_factory=lambda: Decimal("0.03"))  # 優化後: 3%

    # Exchange-based stop loss (recommended for safety)
    use_exchange_stop_loss: bool = True  # Place STOP_MARKET order on exchange
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.01"))  # 優化後: 1%

    # Risk control (風險控制)
    daily_loss_limit_pct: Decimal = field(default_factory=lambda: Decimal("0.05"))  # 每日虧損限制 5%
    max_consecutive_losses: int = 5  # 最大連續虧損次數

    # RSI Filter (RSI 過濾器) - 優化後
    use_rsi_filter: bool = True
    rsi_period: int = 21  # 優化後: 21
    rsi_overbought: int = 71  # 優化後: 71
    rsi_oversold: int = 31  # 優化後: 31

    # Trend confirmation (優化後)
    min_trend_bars: int = 1  # 優化後: 1

    # Protective Features (優化後)
    use_hysteresis: bool = False  # 優化後: 關閉
    hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("0.008"))  # 優化後: 0.8%
    use_signal_cooldown: bool = False  # 優化後: 關閉
    cooldown_bars: int = 0  # 優化後: 0

    # Volatility Regime Filter (v2)
    use_volatility_filter: bool = False  # 優化後: 關閉
    vol_atr_baseline_period: int = 200
    vol_ratio_low: float = 0.6
    vol_ratio_high: float = 2.5

    # Timeout Exit (v2)
    max_hold_bars: int = 12  # 優化後: 12

    # HYBRID_GRID mode (v3)
    mode: str = "hybrid_grid"  # "trend_grid" or "hybrid_grid"
    hybrid_grid_bias_pct: Decimal = field(default_factory=lambda: Decimal("0.75"))  # 優化後: 0.75
    hybrid_tp_multiplier_trend: Decimal = field(default_factory=lambda: Decimal("1.75"))  # 優化後: 1.75
    hybrid_tp_multiplier_counter: Decimal = field(default_factory=lambda: Decimal("0.5"))  # 優化後: 0.5
    hybrid_sl_multiplier_counter: Decimal = field(default_factory=lambda: Decimal("0.5"))  # 優化後: 0.5
    hybrid_rsi_asymmetric: bool = False  # 優化後: 關閉

    @classmethod
    def from_yaml(cls, symbol: str, settings_path=None, **overrides):
        """從 settings.yaml 載入參數（單一來源）。"""
        from src.config.strategy_loader import load_strategy_config
        params = load_strategy_config("supertrend", settings_path)
        params.update(overrides)
        params["symbol"] = symbol
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))
        if not isinstance(self.grid_atr_multiplier, Decimal):
            self.grid_atr_multiplier = Decimal(str(self.grid_atr_multiplier))
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
        if not isinstance(self.hysteresis_pct, Decimal):
            self.hysteresis_pct = Decimal(str(self.hysteresis_pct))
        if not isinstance(self.hybrid_grid_bias_pct, Decimal):
            self.hybrid_grid_bias_pct = Decimal(str(self.hybrid_grid_bias_pct))
        if not isinstance(self.hybrid_tp_multiplier_trend, Decimal):
            self.hybrid_tp_multiplier_trend = Decimal(str(self.hybrid_tp_multiplier_trend))
        if not isinstance(self.hybrid_tp_multiplier_counter, Decimal):
            self.hybrid_tp_multiplier_counter = Decimal(str(self.hybrid_tp_multiplier_counter))
        if not isinstance(self.hybrid_sl_multiplier_counter, Decimal):
            self.hybrid_sl_multiplier_counter = Decimal(str(self.hybrid_sl_multiplier_counter))

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.atr_period < 2 or self.atr_period > 60:
            raise ValueError(f"atr_period must be 2-60, got {self.atr_period}")

        if self.atr_multiplier < Decimal("1.0") or self.atr_multiplier > Decimal("10.0"):
            raise ValueError(f"atr_multiplier must be 1.0-10.0, got {self.atr_multiplier}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")

        if self.max_position_pct < Decimal("0.1") or self.max_position_pct > Decimal("1.0"):
            raise ValueError(f"max_position_pct must be 10%-100%, got {self.max_position_pct}")

        # Grid validation
        if self.grid_count < 3 or self.grid_count > 50:
            raise ValueError(f"grid_count must be 3-50, got {self.grid_count}")

        if self.grid_atr_multiplier < Decimal("1.0") or self.grid_atr_multiplier > Decimal("10.0"):
            raise ValueError(f"grid_atr_multiplier must be 1.0-10.0, got {self.grid_atr_multiplier}")


@dataclass
class GridLevel:
    """Individual grid level for TREND_GRID mode."""
    index: int
    price: Decimal
    is_filled: bool = False


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
