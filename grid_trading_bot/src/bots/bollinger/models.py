"""
Bollinger Bot Data Models.

支援模式:
1. BB_TREND_GRID: 趨勢跟隨 + 網格交易 (已驗證)
2. BB_NEUTRAL_GRID: 雙向中性 + 網格交易 (類似 Grid Futures)

✅ BB_TREND_GRID Walk-Forward 驗證通過 (2024-01 ~ 2026-01):
- Walk-Forward 一致性: 100% (9/9 時段獲利)
- Monte Carlo 穩健性: 100% (15/15)
- 年化報酬: 17.39%
- 穩健性: ROBUST

策略邏輯:
BB_TREND_GRID:
- 趨勢判斷: BB 中軌 (SMA)
  - Price > SMA = 看多 (只做 LONG)
  - Price < SMA = 看空 (只做 SHORT)
- 進場: 網格交易 (K線觸及網格)
- 出場: 止盈 2 個網格 或 止損 2.5%

BB_NEUTRAL_GRID:
- 雙向交易: 隨時可做多/做空 (無趨勢過濾)
- 網格範圍: ATR 動態計算
- 止損: 0.5-1% (更緊)
- 預期交易量: 2-3x BB_TREND_GRID
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional


# =============================================================================
# Enums
# =============================================================================


class StrategyMode(str, Enum):
    """Trading strategy mode."""

    BB_TREND_GRID = "bb_trend_grid"      # BB trend + grid trading (validated)
    BB_NEUTRAL_GRID = "bb_neutral_grid"  # Neutral bi-directional grid (like Grid Futures)


class SignalType(str, Enum):
    """Trading signal type."""

    LONG = "long"
    SHORT = "short"
    NONE = "none"


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

    GRID_PROFIT = "grid_profit"      # Normal grid profit take
    TREND_CHANGE = "trend_change"    # Trend direction changed
    STOP_LOSS = "stop_loss"          # Stop loss triggered
    GRID_REBUILD = "grid_rebuild"    # Grid needs rebuilding
    TIMEOUT = "timeout"              # Max hold bars exceeded (losing position)
    MANUAL = "manual"                # Manual exit
    BOT_STOP = "bot_stop"            # Bot stopped


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class BollingerConfig:
    """
    Bollinger Bot configuration.

    支援模式:
    1. BB_TREND_GRID (預設): 趨勢跟隨 + 網格交易
    2. BB_NEUTRAL_GRID: 雙向中性 + 網格交易 (類似 Grid Futures)

    ✅ BB_TREND_GRID Walk-Forward 驗證通過 (2026-01-27):
        - Walk-Forward 一致性: 100% (9/9 時段獲利)
        - Monte Carlo 穩健性: 100% (15/15 測試獲利)
        - 年化報酬: 17.39%
        - 最大回撤: 6.60%
        - Sharpe: 4.70
        - 勝率: 63.3%
        - 穩健性: ROBUST

    ✅ BB_NEUTRAL_GRID WF v5 Optuna 300 trials (2026-02-01):
    - 9/9 折獲利, 平均 Sharpe: 10.37, 總交易: 5,000
    - 兩年報酬: +289.59%, 年化: 97.38%, 最大回撤: 1.09%
    - Monte Carlo 100%, 參數穩健性 100%

    BB_NEUTRAL_GRID 默認參數 (WF v5 2026-02-01):
    - bb_period: 21, bb_std: 2.0
    - grid_count: 10, take_profit_grids: 1
    - use_atr_range: true, atr_period: 13, atr_multiplier: 6.5
    - stop_loss_pct: 0.2%, leverage: 18x
    - use_hysteresis: true, hysteresis_pct: 0.3%, cooldown_bars: 1
    """

    symbol: str
    timeframe: str = "1h"

    # Strategy mode (Optuna 300 trials 2026-01-31: BB_NEUTRAL_GRID)
    mode: StrategyMode = StrategyMode.BB_NEUTRAL_GRID

    # Bollinger Bands parameters (用於參考，不作為趨勢過濾)
    bb_period: int = 21                                                        # WF v5 優化: 21
    bb_std: Decimal = field(default_factory=lambda: Decimal("2.0"))

    # Grid parameters (WF v5 2026-02-01, 年化97%, Sharpe 8.34)
    grid_count: int = 10                                                       # WF v5 優化: 10
    grid_range_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))   # 備用 (ATR 優先)
    take_profit_grids: int = 1                                                 # WF v5 優化: 1

    # Position settings (18x leverage)
    leverage: int = 18                                                         # 優化後: 18x
    margin_type: str = "ISOLATED"
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))  # 10% per trade
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))  # Max 50% exposure

    # Risk management (WF v5 2026-02-01)
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.002"))   # WF v5 優化: 0.2%
    rebuild_threshold_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))  # Rebuild when price moves 2% from grid

    # BBW filter (for squeeze detection)
    bbw_lookback: int = 200
    bbw_threshold_pct: int = 20

    # Protective features (WF v5 2026-02-01)
    use_hysteresis: bool = True                                                # WF v5 優化: 開啟
    hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("0.003"))  # WF v5 優化: 0.3%
    use_signal_cooldown: bool = False                                          # WF v5 優化: 關閉
    cooldown_bars: int = 1                                                     # WF v5 優化: 1

    # Max hold bars (0 = disabled, >0 = close losing position after N bars)
    max_hold_bars: int = 0

    # Fee rate (Binance Futures: 0.04% maker/taker)
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    # Exchange stop loss order (place STOP_MARKET on exchange instead of local-only monitoring)
    use_exchange_stop_loss: bool = True

    # ATR dynamic range (WF v5 2026-02-01)
    use_atr_range: bool = True                                                 # WF v5 優化: 啟用
    atr_period: int = 13                                                       # WF v5 優化: 13
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("6.5"))    # WF v5 優化: 6.5
    fallback_range_pct: Decimal = field(default_factory=lambda: Decimal("0.04"))  # ATR 不可用時的備用範圍

    @classmethod
    def from_yaml(cls, symbol: str, settings_path=None, **overrides):
        """從 settings.yaml 載入參數（單一來源）。"""
        from src.config.strategy_loader import load_strategy_config
        params = load_strategy_config("bollinger", settings_path)
        params.update(overrides)
        params["symbol"] = symbol
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        """Validate and normalize configuration."""
        if not isinstance(self.bb_std, Decimal):
            self.bb_std = Decimal(str(self.bb_std))
        if not isinstance(self.grid_range_pct, Decimal):
            self.grid_range_pct = Decimal(str(self.grid_range_pct))
        if not isinstance(self.position_size_pct, Decimal):
            self.position_size_pct = Decimal(str(self.position_size_pct))
        if not isinstance(self.max_position_pct, Decimal):
            self.max_position_pct = Decimal(str(self.max_position_pct))
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if not isinstance(self.rebuild_threshold_pct, Decimal):
            self.rebuild_threshold_pct = Decimal(str(self.rebuild_threshold_pct))
        if not isinstance(self.hysteresis_pct, Decimal):
            self.hysteresis_pct = Decimal(str(self.hysteresis_pct))
        if not isinstance(self.fee_rate, Decimal):
            self.fee_rate = Decimal(str(self.fee_rate))
        if self.max_capital is not None and not isinstance(self.max_capital, Decimal):
            self.max_capital = Decimal(str(self.max_capital))
        # ATR range parameters
        if not isinstance(self.atr_multiplier, Decimal):
            self.atr_multiplier = Decimal(str(self.atr_multiplier))
        if not isinstance(self.fallback_range_pct, Decimal):
            self.fallback_range_pct = Decimal(str(self.fallback_range_pct))
        # Convert mode string to enum if needed
        if isinstance(self.mode, str):
            self.mode = StrategyMode(self.mode)

        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        if self.bb_period < 5 or self.bb_period > 100:
            raise ValueError(f"bb_period must be 5-100, got {self.bb_period}")

        if self.bb_std < Decimal("0.5") or self.bb_std > Decimal("5.0"):
            raise ValueError(f"bb_std must be 0.5-5.0, got {self.bb_std}")

        if self.grid_count < 3 or self.grid_count > 50:
            raise ValueError(f"grid_count must be 3-50, got {self.grid_count}")

        if self.leverage < 1 or self.leverage > 125:
            raise ValueError(f"leverage must be 1-125, got {self.leverage}")

        if self.position_size_pct < Decimal("0.01") or self.position_size_pct > Decimal("1.0"):
            raise ValueError(f"position_size_pct must be 1%-100%, got {self.position_size_pct}")

        valid_timeframes = ["1m", "3m", "5m", "15m", "30m", "1h", "4h"]
        if self.timeframe not in valid_timeframes:
            raise ValueError(f"timeframe must be one of {valid_timeframes}")


# =============================================================================
# Grid Data
# =============================================================================


@dataclass
class GridLevel:
    """Individual grid level."""

    index: int
    price: Decimal
    state: GridLevelState = GridLevelState.EMPTY
    entry_price: Optional[Decimal] = None
    entry_time: Optional[datetime] = None

    def __post_init__(self):
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if self.entry_price is not None and not isinstance(self.entry_price, Decimal):
            self.entry_price = Decimal(str(self.entry_price))


@dataclass
class GridSetup:
    """Grid configuration and levels."""

    symbol: str
    center_price: Decimal
    upper_price: Decimal
    lower_price: Decimal
    grid_count: int
    levels: List[GridLevel]
    version: int = 1

    def __post_init__(self):
        for attr in ['center_price', 'upper_price', 'lower_price']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def grid_spacing(self) -> Decimal:
        """Grid spacing between levels."""
        if self.grid_count <= 0:
            return self.upper_price - self.lower_price
        return (self.upper_price - self.lower_price) / Decimal(self.grid_count)


# =============================================================================
# Indicator Data
# =============================================================================


@dataclass
class BollingerBands:
    """
    Bollinger Bands calculation result.

    Attributes:
        upper: Upper band (middle + std * multiplier)
        middle: Middle band (SMA) - used for trend detection
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
    """Bollinger Band Width (BBW) filter data."""

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
# Position
# =============================================================================


@dataclass
class Position:
    """Current futures position."""

    symbol: str
    side: PositionSide
    entry_price: Decimal
    quantity: Decimal
    leverage: int
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    entry_time: Optional[datetime] = None
    grid_level_index: Optional[int] = None
    take_profit_price: Optional[Decimal] = None
    stop_loss_price: Optional[Decimal] = None
    stop_loss_order_id: Optional[str] = None  # Exchange stop loss order ID
    highest_price: Decimal = field(default_factory=lambda: Decimal("0"))  # MFE/MAE tracking
    lowest_price: Decimal = field(default_factory=lambda: Decimal("0"))   # MFE/MAE tracking

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

    def update_extremes(self, current_price: Decimal) -> None:
        """Update highest/lowest prices for MFE/MAE tracking."""
        if self.highest_price == Decimal("0") or current_price > self.highest_price:
            self.highest_price = current_price
        if self.lowest_price == Decimal("0") or current_price < self.lowest_price:
            self.lowest_price = current_price

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
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        return self.side == PositionSide.SHORT


# =============================================================================
# Trade Record
# =============================================================================


@dataclass
class TradeRecord:
    """Completed trade record."""

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
    mfe: Optional[Decimal] = None  # Max Favorable Excursion (%)
    mae: Optional[Decimal] = None  # Max Adverse Excursion (%)

    def __post_init__(self):
        """Ensure Decimal types."""
        for attr in ["entry_price", "exit_price", "quantity", "pnl", "pnl_pct", "fee"]:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))

    @property
    def is_profitable(self) -> bool:
        return self.pnl > 0

    @property
    def net_pnl(self) -> Decimal:
        return self.pnl - self.fee


# =============================================================================
# Statistics
# =============================================================================


@dataclass
class BollingerBotStats:
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
        # Use abs() to handle potential negative loss values
        loss_pnl = abs(self._total_loss_pnl)
        if loss_pnl == 0:
            return Decimal("999") if self._total_win_pnl > 0 else Decimal("0")
        return abs(self._total_win_pnl) / loss_pnl

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

    @property
    def total_profit(self) -> Decimal:
        """Alias for total_pnl (required by base heartbeat)."""
        return self.total_pnl

    def record_trade(self, trade: TradeRecord) -> None:
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
            "total_pnl": str(self.total_pnl),
            "total_fees": str(self.total_fees),
            "net_pnl": str(self.net_pnl),
            "win_rate": f"{self.win_rate:.1f}%",
            "profit_factor": f"{self.profit_factor:.2f}",
            "avg_win": str(self.avg_win),
            "avg_loss": str(self.avg_loss),
            "grid_rebuilds": self.grid_rebuilds,
            "max_drawdown_pct": f"{self.max_drawdown_pct:.2f}%",
        }
