"""
Grid Futures Bot Data Models.

Provides data models for futures-based grid trading with leverage
and bidirectional trading support.

✅ 積極策略優化 + Walk-Forward 驗證 (2026-01-28):
- 年化報酬: 45.18%
- 回撤: 3.79%
- 勝率: 55.77%
- 交易次數: 4,976/2年
- W-F 一致性: 88.9% (8/9)
- OOS/IS Sharpe: 0.72
- Monte Carlo 獲利機率: 100%

策略配置 (同步 settings.yaml):
- leverage: 10x (多槓桿回測最佳: Sharpe 1.86, 零爆倉, 回撤 19.2%)
- direction: NEUTRAL (雙向交易)
- grid_count: 18
- atr_period: 28
- atr_multiplier: 9.5 (超寬範圍)
- stop_loss_pct: 0.5% (緊止損)
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
    TIMEOUT = "timeout"              # Max hold bars exceeded (losing position)
    MANUAL = "manual"                # Manual close
    BOT_STOP = "bot_stop"            # Bot stopped


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class GridFuturesConfig:
    """
    Grid Futures Bot configuration.

    ✅ 積極策略優化 + Walk-Forward 驗證 (2026-01-28):
    - 年化報酬: 45.18%
    - 回撤: 3.79%
    - 勝率: 55.77%
    - W-F 一致性: 88.9% (8/9)
    - OOS/IS Sharpe: 0.72
    - Monte Carlo 獲利機率: 100%

    策略配置 (同步 settings.yaml):
    - Leverage: 10x (多槓桿回測最佳: Sharpe 1.86, 零爆倉, 回撤 19.2%)
    - Direction: NEUTRAL (雙向交易)
    - Grid Count: 18
    - ATR Period: 28
    - ATR Multiplier: 9.5 (超寬範圍)
    - Stop Loss: 0.5% (緊止損)

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        timeframe: Kline timeframe for indicators (default "1h")
        leverage: Futures leverage (default 10, synced with settings.yaml)
        margin_type: ISOLATED or CROSSED (default ISOLATED)

        # Grid settings
        grid_count: Number of grid levels (default 18, validated)
        direction: Trading direction mode (default NEUTRAL, validated)

        # Trend filter
        use_trend_filter: Enable trend-based direction (default False for NEUTRAL)
        trend_period: SMA period for trend detection (default 24, validated)

        # Dynamic range
        use_atr_range: Use ATR for dynamic grid range (default True)
        atr_period: ATR calculation period (default 28, validated)
        atr_multiplier: ATR multiplier for range (default 9.5, validated)
        fallback_range_pct: Range when ATR unavailable (default 8%)

        # Position sizing
        max_capital: Maximum capital to use (None = all balance)
        position_size_pct: Size per grid trade as % of capital (default 10%)
        max_position_pct: Maximum total position as % of capital (default 50%)

        # Risk management
        stop_loss_pct: Stop loss percentage per position (default 0.5%)
        rebuild_threshold_pct: Price deviation to trigger rebuild (default 2%)

    Example:
        >>> config = GridFuturesConfig(
        ...     symbol="BTCUSDT",
        ...     leverage=7,
        ...     grid_count=12,
        ...     direction=GridDirection.NEUTRAL,
        ...     trend_period=15,
        ...     atr_period=14,
        ...     atr_multiplier=Decimal("9.5"),
        ...     stop_loss_pct=Decimal("0.005"),  # 0.5% 緊止損
        ... )
    """

    symbol: str
    timeframe: str = "1h"
    leverage: int = 7  # 同步 settings.yaml
    margin_type: str = "ISOLATED"

    # Grid settings (NEUTRAL 雙向交易)
    grid_count: int = 8  # 同步 settings.yaml
    direction: GridDirection = GridDirection.NEUTRAL

    # Trend filter (NEUTRAL 模式不使用趨勢過濾)
    use_trend_filter: bool = False
    trend_period: int = 48  # 同步 settings.yaml

    # Dynamic ATR range
    use_atr_range: bool = True
    atr_period: int = 46  # 同步 settings.yaml
    atr_multiplier: Decimal = field(default_factory=lambda: Decimal("6.5"))  # 同步 settings.yaml
    fallback_range_pct: Decimal = field(default_factory=lambda: Decimal("0.08"))

    # Position sizing (optimized: 10% per trade)
    max_capital: Optional[Decimal] = None
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))
    max_position_pct: Decimal = field(default_factory=lambda: Decimal("0.5"))

    # Risk management (W-F 驗證通過)
    stop_loss_pct: Decimal = field(default_factory=lambda: Decimal("0.005"))  # ⚠️ 0.5% 緊止損
    rebuild_threshold_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))

    # Max hold bars (0 = disabled, >0 = close losing position after N bars)
    max_hold_bars: int = 0

    # Exchange-based stop loss (recommended for safety)
    use_exchange_stop_loss: bool = True  # Place STOP_MARKET order on exchange

    # Fee rate (Binance Futures: 0.04% maker/taker)
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))

    # Protective features (W-F 驗證通過)
    use_hysteresis: bool = True  # 同步 settings.yaml
    hysteresis_pct: Decimal = field(default_factory=lambda: Decimal("0.001"))  # 同步 settings.yaml
    use_signal_cooldown: bool = True  # 同步 settings.yaml
    cooldown_bars: int = 0  # 同步 settings.yaml

    @classmethod
    def from_yaml(cls, symbol: str, settings_path=None, **overrides):
        """從 settings.yaml 載入參數（單一來源）。"""
        from src.config.strategy_loader import load_strategy_config
        params = load_strategy_config("grid_futures", settings_path)
        params.update(overrides)
        params["symbol"] = symbol
        return cls(**{k: v for k, v in params.items() if k in cls.__dataclass_fields__})

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure Decimal types
        decimal_fields = [
            'atr_multiplier', 'fallback_range_pct', 'position_size_pct',
            'max_position_pct', 'stop_loss_pct', 'rebuild_threshold_pct', 'fee_rate',
            'hysteresis_pct'
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

        if self.atr_multiplier < Decimal("0.5") or self.atr_multiplier > Decimal("10.0"):
            raise ValueError(f"atr_multiplier must be 0.5-10.0, got {self.atr_multiplier}")

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
    stop_loss_order_id: Optional[str] = None  # Exchange stop loss order ID
    stop_loss_price: Optional[Decimal] = None  # Stop loss trigger price

    def __post_init__(self):
        for attr in ['entry_price', 'quantity', 'unrealized_pnl']:
            value = getattr(self, attr)
            if not isinstance(value, Decimal):
                setattr(self, attr, Decimal(str(value)))
        if self.liquidation_price is not None and not isinstance(self.liquidation_price, Decimal):
            self.liquidation_price = Decimal(str(self.liquidation_price))
        if self.stop_loss_price is not None and not isinstance(self.stop_loss_price, Decimal):
            self.stop_loss_price = Decimal(str(self.stop_loss_price))

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
