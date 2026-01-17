"""
Grid Trading Data Models.

Provides data models for intelligent grid trading strategy.
User only needs to provide symbol and investment amount, rest is auto-calculated.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Optional


class GridType(str, Enum):
    """Grid spacing type."""

    ARITHMETIC = "arithmetic"  # Equal price spacing
    GEOMETRIC = "geometric"    # Equal percentage spacing


class GridMode(str, Enum):
    """Grid trading direction mode."""

    LONG = "long"       # Long-biased grid (buy low, sell high)
    SHORT = "short"     # Short-biased grid (sell high, buy low)
    NEUTRAL = "neutral" # Neutral grid (both directions)


class RiskLevel(str, Enum):
    """Risk level for grid range calculation."""

    CONSERVATIVE = "conservative"  # Narrow range, easy breakout
    MEDIUM = "medium"              # Balanced (default)
    AGGRESSIVE = "aggressive"      # Wide range, hard breakout

    @property
    def atr_multiplier(self) -> Decimal:
        """Get ATR multiplier for this risk level."""
        multipliers = {
            RiskLevel.CONSERVATIVE: Decimal("1.5"),
            RiskLevel.MEDIUM: Decimal("2.5"),
            RiskLevel.AGGRESSIVE: Decimal("3.5"),
        }
        return multipliers[self]


class LevelSide(str, Enum):
    """Grid level side."""

    BUY = "buy"
    SELL = "sell"


class LevelState(str, Enum):
    """Grid level state."""

    EMPTY = "empty"              # No order
    PENDING_BUY = "pending_buy"   # Buy order pending
    PENDING_SELL = "pending_sell" # Sell order pending
    FILLED = "filled"            # Order filled, waiting for opposite


@dataclass
class ATRData:
    """
    ATR (Average True Range) calculation result.

    Example:
        >>> atr = ATRData(
        ...     value=Decimal("500"),
        ...     period=14,
        ...     timeframe="4h",
        ...     current_price=Decimal("50000"),
        ... )
        >>> print(atr.volatility_percent)
        1.00
        >>> print(atr.volatility_level)
        低
    """

    value: Decimal
    period: int
    timeframe: str
    current_price: Decimal

    @property
    def volatility_percent(self) -> Decimal:
        """
        Calculate volatility as percentage of current price.

        Returns:
            Volatility percentage
        """
        if self.current_price <= 0:
            return Decimal("0")
        return (self.value / self.current_price) * Decimal("100")

    @property
    def volatility_level(self) -> str:
        """
        Determine volatility level description.

        Returns:
            Volatility level in Chinese
        """
        pct = self.volatility_percent

        if pct < Decimal("0.5"):
            return "極低"
        elif pct < Decimal("1.0"):
            return "低"
        elif pct < Decimal("2.0"):
            return "中等"
        elif pct < Decimal("4.0"):
            return "高"
        else:
            return "極高"

    @property
    def volatility_level_en(self) -> str:
        """
        Determine volatility level description in English.

        Returns:
            Volatility level in English
        """
        pct = self.volatility_percent

        if pct < Decimal("0.5"):
            return "very_low"
        elif pct < Decimal("1.0"):
            return "low"
        elif pct < Decimal("2.0"):
            return "medium"
        elif pct < Decimal("4.0"):
            return "high"
        else:
            return "very_high"


@dataclass
class GridConfig:
    """
    User configuration for grid trading.

    Minimal required input is symbol and total_investment.
    All other parameters have sensible defaults.

    Example:
        >>> # Minimal config
        >>> config = GridConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("1000"),
        ... )

        >>> # With custom parameters
        >>> config = GridConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("1000"),
        ...     risk_level=RiskLevel.AGGRESSIVE,
        ...     manual_grid_count=20,
        ... )
    """

    # Required fields
    symbol: str
    total_investment: Decimal

    # Fields with defaults
    risk_level: RiskLevel = RiskLevel.MEDIUM
    grid_type: GridType = GridType.GEOMETRIC
    grid_mode: GridMode = GridMode.NEUTRAL
    min_grid_count: int = 5
    max_grid_count: int = 50
    min_order_value: Decimal = field(default_factory=lambda: Decimal("10"))
    atr_period: int = 14
    atr_timeframe: str = "4h"
    stop_loss_percent: Decimal = field(default_factory=lambda: Decimal("20"))

    # Optional manual overrides
    manual_upper_price: Optional[Decimal] = None
    manual_lower_price: Optional[Decimal] = None
    manual_grid_count: Optional[int] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Ensure Decimal types
        if not isinstance(self.total_investment, Decimal):
            self.total_investment = Decimal(str(self.total_investment))
        if not isinstance(self.min_order_value, Decimal):
            self.min_order_value = Decimal(str(self.min_order_value))
        if not isinstance(self.stop_loss_percent, Decimal):
            self.stop_loss_percent = Decimal(str(self.stop_loss_percent))
        if self.manual_upper_price is not None and not isinstance(self.manual_upper_price, Decimal):
            self.manual_upper_price = Decimal(str(self.manual_upper_price))
        if self.manual_lower_price is not None and not isinstance(self.manual_lower_price, Decimal):
            self.manual_lower_price = Decimal(str(self.manual_lower_price))

        # Validate
        self._validate()

    def _validate(self) -> None:
        """Validate configuration."""
        min_required = self.min_order_value * Decimal(self.min_grid_count)
        if self.total_investment < min_required:
            raise ValueError(
                f"Total investment ({self.total_investment}) must be >= "
                f"min_order_value ({self.min_order_value}) × min_grid_count ({self.min_grid_count}) = {min_required}"
            )

        if self.min_grid_count < 2:
            raise ValueError("min_grid_count must be at least 2")

        if self.max_grid_count < self.min_grid_count:
            raise ValueError("max_grid_count must be >= min_grid_count")

        if self.manual_grid_count is not None:
            if self.manual_grid_count < self.min_grid_count:
                raise ValueError(f"manual_grid_count must be >= {self.min_grid_count}")
            if self.manual_grid_count > self.max_grid_count:
                raise ValueError(f"manual_grid_count must be <= {self.max_grid_count}")

        if self.manual_upper_price is not None and self.manual_lower_price is not None:
            if self.manual_upper_price <= self.manual_lower_price:
                raise ValueError("manual_upper_price must be > manual_lower_price")

    @property
    def has_manual_range(self) -> bool:
        """Check if manual price range is set."""
        return self.manual_upper_price is not None and self.manual_lower_price is not None

    @property
    def has_manual_grid_count(self) -> bool:
        """Check if manual grid count is set."""
        return self.manual_grid_count is not None


@dataclass
class GridLevel:
    """
    Individual grid level.

    Each level represents a price point where buy/sell orders are placed.

    Example:
        >>> level = GridLevel(
        ...     index=5,
        ...     price=Decimal("49000"),
        ...     side=LevelSide.BUY,
        ...     state=LevelState.EMPTY,
        ...     allocated_amount=Decimal("100"),
        ... )
    """

    index: int
    price: Decimal
    side: LevelSide
    state: LevelState = LevelState.EMPTY
    allocated_amount: Decimal = field(default_factory=lambda: Decimal("0"))
    order_id: Optional[str] = None
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    filled_price: Optional[Decimal] = None

    @property
    def is_empty(self) -> bool:
        """Check if level has no active order."""
        return self.state == LevelState.EMPTY

    @property
    def is_pending(self) -> bool:
        """Check if level has pending order."""
        return self.state in (LevelState.PENDING_BUY, LevelState.PENDING_SELL)

    @property
    def is_filled(self) -> bool:
        """Check if level has been filled."""
        return self.state == LevelState.FILLED


@dataclass
class GridSetup:
    """
    Complete grid setup with calculated parameters.

    This is the result of grid calculation containing all necessary
    information to execute the grid strategy.

    Example:
        >>> setup = GridSetup(
        ...     config=config,
        ...     atr_data=atr_data,
        ...     upper_price=Decimal("52000"),
        ...     lower_price=Decimal("48000"),
        ...     current_price=Decimal("50000"),
        ...     grid_count=10,
        ...     grid_spacing_percent=Decimal("0.8"),
        ...     amount_per_grid=Decimal("100"),
        ...     levels=levels,
        ...     expected_profit_per_trade=Decimal("0.80"),
        ... )
    """

    config: GridConfig
    atr_data: ATRData
    upper_price: Decimal
    lower_price: Decimal
    current_price: Decimal
    grid_count: int
    grid_spacing_percent: Decimal
    amount_per_grid: Decimal
    levels: list[GridLevel]
    expected_profit_per_trade: Decimal

    @property
    def buy_levels(self) -> list[GridLevel]:
        """Get all buy levels."""
        return [l for l in self.levels if l.side == LevelSide.BUY]

    @property
    def sell_levels(self) -> list[GridLevel]:
        """Get all sell levels."""
        return [l for l in self.levels if l.side == LevelSide.SELL]

    @property
    def price_range_percent(self) -> Decimal:
        """
        Calculate price range as percentage.

        Returns:
            Price range percentage
        """
        if self.lower_price <= 0:
            return Decimal("0")
        return ((self.upper_price - self.lower_price) / self.lower_price) * Decimal("100")

    @property
    def total_buy_levels(self) -> int:
        """Get number of buy levels."""
        return len(self.buy_levels)

    @property
    def total_sell_levels(self) -> int:
        """Get number of sell levels."""
        return len(self.sell_levels)

    @property
    def total_investment_required(self) -> Decimal:
        """Calculate total investment required for buy orders."""
        return sum(l.allocated_amount for l in self.buy_levels)

    @property
    def average_buy_price(self) -> Decimal:
        """Calculate average buy price."""
        buy_levels = self.buy_levels
        if not buy_levels:
            return Decimal("0")
        return sum(l.price for l in buy_levels) / Decimal(len(buy_levels))

    @property
    def average_sell_price(self) -> Decimal:
        """Calculate average sell price."""
        sell_levels = self.sell_levels
        if not sell_levels:
            return Decimal("0")
        return sum(l.price for l in sell_levels) / Decimal(len(sell_levels))

    def get_level_at_price(self, price: Decimal, tolerance_percent: Decimal = Decimal("0.1")) -> Optional[GridLevel]:
        """
        Find level closest to a given price within tolerance.

        Args:
            price: Target price
            tolerance_percent: Tolerance percentage

        Returns:
            GridLevel if found within tolerance, else None
        """
        for level in self.levels:
            diff_percent = abs((level.price - price) / price) * Decimal("100")
            if diff_percent <= tolerance_percent:
                return level
        return None

    def summary(self) -> dict:
        """
        Get summary of grid setup.

        Returns:
            Summary dictionary
        """
        return {
            "symbol": self.config.symbol,
            "current_price": str(self.current_price),
            "upper_price": str(self.upper_price),
            "lower_price": str(self.lower_price),
            "price_range_percent": str(self.price_range_percent),
            "grid_count": self.grid_count,
            "grid_spacing_percent": str(self.grid_spacing_percent),
            "amount_per_grid": str(self.amount_per_grid),
            "total_investment": str(self.config.total_investment),
            "buy_levels": self.total_buy_levels,
            "sell_levels": self.total_sell_levels,
            "atr_value": str(self.atr_data.value),
            "volatility_percent": str(self.atr_data.volatility_percent),
            "volatility_level": self.atr_data.volatility_level,
            "expected_profit_per_trade": str(self.expected_profit_per_trade),
            "risk_level": self.config.risk_level.value,
            "grid_type": self.config.grid_type.value,
        }
