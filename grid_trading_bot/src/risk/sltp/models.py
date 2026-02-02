"""
Stop Loss / Take Profit Models.

Defines data models for unified SLTP management including:
- Stop loss types and configurations
- Take profit types and configurations (multi-level support)
- Trailing stop types and configurations
- SLTP state tracking
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional


class StopLossType(Enum):
    """Stop loss calculation type."""

    FIXED = "fixed"  # Fixed price level
    PERCENTAGE = "percentage"  # Percentage from entry
    ATR_BASED = "atr_based"  # ATR multiplier based
    INDICATOR = "indicator"  # Indicator based (e.g., Supertrend)
    TRAILING = "trailing"  # Trailing stop (uses TrailingStopConfig)


class TakeProfitType(Enum):
    """Take profit calculation type."""

    FIXED = "fixed"  # Fixed price level
    PERCENTAGE = "percentage"  # Percentage from entry
    ATR_BASED = "atr_based"  # ATR multiplier based
    RISK_REWARD = "risk_reward"  # Based on risk/reward ratio
    INDICATOR = "indicator"  # Indicator based
    MULTI_LEVEL = "multi_level"  # Multiple take profit levels


class TrailingStopType(Enum):
    """Trailing stop calculation type."""

    FIXED_DISTANCE = "fixed_distance"  # Fixed price distance
    PERCENTAGE = "percentage"  # Percentage distance
    ATR_BASED = "atr_based"  # ATR multiplier distance
    BREAK_EVEN = "break_even"  # Move to break even after threshold


@dataclass
class StopLossConfig:
    """Configuration for stop loss."""

    stop_type: StopLossType = StopLossType.PERCENTAGE
    value: Decimal = Decimal("0.02")  # 2% default for percentage
    atr_multiplier: Decimal = Decimal("2.0")  # For ATR_BASED
    fixed_price: Optional[Decimal] = None  # For FIXED type
    indicator_key: Optional[str] = None  # For INDICATOR type (e.g., "supertrend")
    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.stop_type == StopLossType.FIXED and self.fixed_price is None:
            raise ValueError("fixed_price required for FIXED stop loss type")


@dataclass
class TakeProfitLevel:
    """Single take profit level."""

    price: Decimal  # Target price
    percentage: Decimal  # Percentage of position to close (0-1)
    triggered: bool = False  # Whether this level has been triggered
    triggered_at: Optional[datetime] = None  # When triggered
    order_id: Optional[str] = None  # Exchange order ID if placed


@dataclass
class TakeProfitConfig:
    """Configuration for take profit."""

    tp_type: TakeProfitType = TakeProfitType.PERCENTAGE
    value: Decimal = Decimal("0.04")  # 4% default for percentage
    atr_multiplier: Decimal = Decimal("3.0")  # For ATR_BASED
    risk_reward_ratio: Decimal = Decimal("2.0")  # For RISK_REWARD
    fixed_price: Optional[Decimal] = None  # For FIXED type
    indicator_key: Optional[str] = None  # For INDICATOR type

    # Multi-level take profit configuration
    levels: List[TakeProfitLevel] = field(default_factory=list)
    level_percentages: List[Decimal] = field(default_factory=list)  # e.g., [0.03, 0.05, 0.08]
    level_close_pcts: List[Decimal] = field(default_factory=list)  # e.g., [0.33, 0.33, 0.34]

    enabled: bool = True

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.tp_type == TakeProfitType.FIXED and self.fixed_price is None:
            raise ValueError("fixed_price required for FIXED take profit type")
        if self.tp_type == TakeProfitType.MULTI_LEVEL:
            if not self.level_percentages or not self.level_close_pcts:
                raise ValueError(
                    "level_percentages and level_close_pcts required for MULTI_LEVEL"
                )
            if len(self.level_percentages) != len(self.level_close_pcts):
                raise ValueError(
                    "level_percentages and level_close_pcts must have same length"
                )


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop."""

    trailing_type: TrailingStopType = TrailingStopType.PERCENTAGE
    distance: Decimal = Decimal("0.015")  # Distance for trailing
    atr_multiplier: Decimal = Decimal("1.5")  # For ATR_BASED
    activation_pct: Decimal = Decimal("0.01")  # Activate after X% profit
    break_even_trigger: Decimal = Decimal("0.02")  # Trigger for break even move
    callback_rate: Decimal = Decimal("0.01")  # Callback rate for exchange trailing
    enabled: bool = False


@dataclass
class SLTPConfig:
    """Integrated SLTP configuration."""

    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)
    trailing_stop: TrailingStopConfig = field(default_factory=TrailingStopConfig)

    # Exchange order placement settings
    use_exchange_orders: bool = True  # True for live trading, False for backtest
    place_sl_order: bool = True  # Whether to place SL order on exchange
    place_tp_order: bool = True  # Whether to place TP order on exchange

    @classmethod
    def from_dict(cls, data: dict) -> "SLTPConfig":
        """Create config from dictionary."""
        sl_data = data.get("stop_loss", {})
        tp_data = data.get("take_profit", {})
        ts_data = data.get("trailing_stop", {})

        stop_loss = StopLossConfig(
            stop_type=StopLossType(sl_data.get("type", "percentage")),
            value=Decimal(str(sl_data.get("value", "0.02"))),
            atr_multiplier=Decimal(str(sl_data.get("atr_multiplier", "2.0"))),
            fixed_price=(
                Decimal(str(sl_data["fixed_price"]))
                if sl_data.get("fixed_price")
                else None
            ),
            enabled=sl_data.get("enabled", True),
        )

        take_profit = TakeProfitConfig(
            tp_type=TakeProfitType(tp_data.get("type", "percentage")),
            value=Decimal(str(tp_data.get("value", "0.04"))),
            atr_multiplier=Decimal(str(tp_data.get("atr_multiplier", "3.0"))),
            risk_reward_ratio=Decimal(str(tp_data.get("risk_reward_ratio", "2.0"))),
            fixed_price=(
                Decimal(str(tp_data["fixed_price"]))
                if tp_data.get("fixed_price")
                else None
            ),
            level_percentages=[
                Decimal(str(x)) for x in tp_data.get("level_percentages", [])
            ],
            level_close_pcts=[
                Decimal(str(x)) for x in tp_data.get("level_close_pcts", [])
            ],
            enabled=tp_data.get("enabled", True),
        )

        trailing_stop = TrailingStopConfig(
            trailing_type=TrailingStopType(ts_data.get("type", "percentage")),
            distance=Decimal(str(ts_data.get("distance", "0.015"))),
            atr_multiplier=Decimal(str(ts_data.get("atr_multiplier", "1.5"))),
            activation_pct=Decimal(str(ts_data.get("activation_pct", "0.01"))),
            callback_rate=Decimal(str(ts_data.get("callback_rate", "0.01"))),
            enabled=ts_data.get("enabled", False),
        )

        return cls(
            stop_loss=stop_loss,
            take_profit=take_profit,
            trailing_stop=trailing_stop,
            use_exchange_orders=data.get("use_exchange_orders", True),
            place_sl_order=data.get("place_sl_order", True),
            place_tp_order=data.get("place_tp_order", True),
        )


@dataclass
class SLTPState:
    """State tracking for SLTP management."""

    symbol: str
    entry_price: Decimal
    is_long: bool
    quantity: Decimal

    # Calculated levels
    initial_stop_loss: Decimal
    current_stop_loss: Decimal
    take_profit_levels: List[TakeProfitLevel] = field(default_factory=list)

    # Trailing stop tracking
    trailing_activated: bool = False
    highest_price: Decimal = Decimal("0")  # Highest price since entry (for long)
    lowest_price: Decimal = Decimal("999999999")  # Lowest price since entry (for short)
    trailing_stop_price: Optional[Decimal] = None

    # Order IDs
    sl_order_id: Optional[str] = None
    tp_order_ids: List[str] = field(default_factory=list)
    trailing_order_id: Optional[str] = None

    # Status
    stop_loss_triggered: bool = False
    stop_loss_triggered_at: Optional[datetime] = None
    all_tp_triggered: bool = False
    closed_quantity: Decimal = Decimal("0")

    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def update_price_extremes(self, high: Decimal, low: Decimal) -> None:
        """Update highest/lowest price tracking."""
        if high > self.highest_price:
            self.highest_price = high
        if low < self.lowest_price:
            self.lowest_price = low
        self.updated_at = datetime.now(timezone.utc)

    def update_stop_loss(self, new_stop: Decimal) -> bool:
        """Update stop loss price. Returns True if changed."""
        if self.is_long and new_stop > self.current_stop_loss:
            self.current_stop_loss = new_stop
            self.updated_at = datetime.now(timezone.utc)
            return True
        elif not self.is_long and new_stop < self.current_stop_loss:
            self.current_stop_loss = new_stop
            self.updated_at = datetime.now(timezone.utc)
            return True
        return False

    def mark_tp_triggered(self, level_index: int) -> Decimal:
        """Mark a take profit level as triggered. Returns close percentage."""
        if 0 <= level_index < len(self.take_profit_levels):
            level = self.take_profit_levels[level_index]
            if not level.triggered:
                level.triggered = True
                level.triggered_at = datetime.now(timezone.utc)
                self.closed_quantity += self.quantity * level.percentage
                self.updated_at = datetime.now(timezone.utc)

                # Check if all levels triggered
                if all(tp.triggered for tp in self.take_profit_levels):
                    self.all_tp_triggered = True

                return level.percentage
        return Decimal("0")

    def mark_sl_triggered(self) -> None:
        """Mark stop loss as triggered."""
        self.stop_loss_triggered = True
        self.stop_loss_triggered_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining open quantity."""
        return self.quantity - self.closed_quantity

    @property
    def is_active(self) -> bool:
        """Check if SLTP is still active."""
        return not self.stop_loss_triggered and not self.all_tp_triggered
