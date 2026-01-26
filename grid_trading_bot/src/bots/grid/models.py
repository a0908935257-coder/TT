"""
Grid Trading Data Models.

Provides data models for intelligent grid trading strategy.
User only needs to provide symbol and investment amount, rest is auto-calculated.

Conforms to Prompt 17 specification with adjustable ATR multiplier support.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional


# =============================================================================
# Enums
# =============================================================================


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
    """
    Risk level for grid range calculation.

    Each level has a default ATR multiplier:
    - conservative: 1.5 (narrow range, suitable for low volatility)
    - moderate: 2.0 (standard, default)
    - aggressive: 2.5 (wide range, suitable for high volatility)
    - custom: user-defined multiplier
    """

    CONSERVATIVE = "conservative"  # 1.5x ATR
    MODERATE = "moderate"          # 2.0x ATR (default)
    AGGRESSIVE = "aggressive"      # 2.5x ATR
    CUSTOM = "custom"              # User-defined

    @property
    def default_multiplier(self) -> Decimal:
        """Get default ATR multiplier for this risk level."""
        multipliers = {
            RiskLevel.CONSERVATIVE: Decimal("1.5"),
            RiskLevel.MODERATE: Decimal("2.0"),
            RiskLevel.AGGRESSIVE: Decimal("2.5"),
            RiskLevel.CUSTOM: Decimal("2.0"),  # Fallback, should use ATRConfig
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
    PARTIAL = "partial"          # Order partially filled
    FILLED = "filled"            # Order filled, waiting for opposite


# =============================================================================
# ATR Configuration
# =============================================================================


# Valid timeframes for ATR calculation
VALID_TIMEFRAMES = ["1h", "4h", "1d", "1w"]


@dataclass
class ATRConfig:
    """
    ATR (Average True Range) configuration.

    User can fully customize ATR-related parameters.

    Attributes:
        period: ATR calculation period (5-50, default 14)
        timeframe: K-line timeframe (1h, 4h, 1d, 1w)
        multiplier: ATR multiplier for range calculation (0.5-5.0)
        use_ema: Use EMA for calculation (True) or SMA (False)

    Example:
        >>> # Default moderate config
        >>> config = ATRConfig()
        >>> print(config.multiplier)
        2.0

        >>> # Custom config
        >>> config = ATRConfig(multiplier=Decimal("1.8"), period=20)

        >>> # Using factory methods
        >>> config = ATRConfig.aggressive()
    """

    period: int = 14
    timeframe: str = "4h"
    multiplier: Decimal = field(default_factory=lambda: Decimal("2.0"))
    use_ema: bool = True

    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure Decimal type
        if not isinstance(self.multiplier, Decimal):
            self.multiplier = Decimal(str(self.multiplier))

        # Validate
        self._validate()

    def _validate(self) -> None:
        """Validate configuration parameters."""
        # Period validation: 5 <= period <= 50
        if not (5 <= self.period <= 50):
            raise ValueError(f"period must be between 5 and 50, got {self.period}")

        # Multiplier validation: 0.5 <= multiplier <= 5.0
        if not (Decimal("0.5") <= self.multiplier <= Decimal("5.0")):
            raise ValueError(
                f"multiplier must be between 0.5 and 5.0, got {self.multiplier}"
            )

        # Timeframe validation
        if self.timeframe not in VALID_TIMEFRAMES:
            raise ValueError(
                f"timeframe must be one of {VALID_TIMEFRAMES}, got {self.timeframe}"
            )

    @classmethod
    def conservative(cls) -> "ATRConfig":
        """
        Create conservative config (1.5x multiplier).

        Narrow range, suitable for low volatility markets.
        """
        return cls(multiplier=Decimal("1.5"))

    @classmethod
    def moderate(cls) -> "ATRConfig":
        """
        Create moderate config (2.0x multiplier).

        Standard range, balanced approach (default).
        """
        return cls(multiplier=Decimal("2.0"))

    @classmethod
    def aggressive(cls) -> "ATRConfig":
        """
        Create aggressive config (2.5x multiplier).

        Wide range, suitable for high volatility markets.
        """
        return cls(multiplier=Decimal("2.5"))

    @classmethod
    def custom(
        cls,
        multiplier: Decimal | float | str,
        period: int = 14,
        timeframe: str = "4h",
        use_ema: bool = True,
    ) -> "ATRConfig":
        """
        Create custom config with specified parameters.

        Args:
            multiplier: ATR multiplier (0.5-5.0)
            period: ATR period (5-50)
            timeframe: K-line timeframe
            use_ema: Use EMA (True) or SMA (False)

        Returns:
            Custom ATRConfig instance
        """
        if not isinstance(multiplier, Decimal):
            multiplier = Decimal(str(multiplier))
        return cls(
            period=period,
            timeframe=timeframe,
            multiplier=multiplier,
            use_ema=use_ema,
        )


# =============================================================================
# ATR Data
# =============================================================================


@dataclass
class ATRData:
    """
    ATR (Average True Range) calculation result.

    Contains the calculated ATR value along with derived price boundaries
    and volatility information.

    Attributes:
        value: Calculated ATR value
        period: ATR calculation period used
        timeframe: K-line timeframe used
        multiplier: ATR multiplier used for range calculation
        current_price: Price at calculation time
        upper_price: Calculated upper boundary (price + ATR × multiplier)
        lower_price: Calculated lower boundary (price - ATR × multiplier)
        calculated_at: Timestamp of calculation

    Example:
        >>> atr = ATRData(
        ...     value=Decimal("1000"),
        ...     period=14,
        ...     timeframe="4h",
        ...     multiplier=Decimal("2.0"),
        ...     current_price=Decimal("50000"),
        ...     upper_price=Decimal("52000"),
        ...     lower_price=Decimal("48000"),
        ... )
        >>> print(atr.volatility_percent)
        2.00
        >>> print(atr.volatility_level)
        中等
    """

    value: Decimal
    period: int
    timeframe: str
    multiplier: Decimal
    current_price: Decimal
    upper_price: Decimal
    lower_price: Decimal
    calculated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.value, Decimal):
            self.value = Decimal(str(self.value))
        if not isinstance(self.multiplier, Decimal):
            self.multiplier = Decimal(str(self.multiplier))
        if not isinstance(self.current_price, Decimal):
            self.current_price = Decimal(str(self.current_price))
        if not isinstance(self.upper_price, Decimal):
            self.upper_price = Decimal(str(self.upper_price))
        if not isinstance(self.lower_price, Decimal):
            self.lower_price = Decimal(str(self.lower_price))

    @property
    def volatility_percent(self) -> Decimal:
        """
        Calculate volatility as percentage of current price.

        Formula: (ATR / current_price) × 100

        Returns:
            Volatility percentage
        """
        if self.current_price <= 0:
            return Decimal("0")
        return (self.value / self.current_price) * Decimal("100")

    @property
    def range_width(self) -> Decimal:
        """
        Calculate price range width.

        Returns:
            upper_price - lower_price
        """
        return self.upper_price - self.lower_price

    @property
    def range_percent(self) -> Decimal:
        """
        Calculate price range as percentage of current price.

        Returns:
            Range width percentage
        """
        if self.current_price <= 0:
            return Decimal("0")
        return (self.range_width / self.current_price) * Decimal("100")

    @property
    def volatility_level(self) -> str:
        """
        Determine volatility level description (Chinese).

        Thresholds (per Prompt 17 spec):
            < 1%  → 極低
            1-2%  → 低
            2-4%  → 中等
            4-6%  → 高
            >= 6% → 極高

        Returns:
            Volatility level in Chinese
        """
        pct = self.volatility_percent

        if pct < Decimal("1"):
            return "極低"
        elif pct < Decimal("2"):
            return "低"
        elif pct < Decimal("4"):
            return "中等"
        elif pct < Decimal("6"):
            return "高"
        else:
            return "極高"

    @property
    def volatility_level_en(self) -> str:
        """
        Determine volatility level description (English).

        Returns:
            Volatility level in English
        """
        pct = self.volatility_percent

        if pct < Decimal("1"):
            return "very_low"
        elif pct < Decimal("2"):
            return "low"
        elif pct < Decimal("4"):
            return "medium"
        elif pct < Decimal("6"):
            return "high"
        else:
            return "very_high"


# =============================================================================
# Dynamic Adjustment Configuration
# =============================================================================


@dataclass
class DynamicAdjustConfig:
    """
    Dynamic grid adjustment configuration.

    Controls automatic grid rebuilding when price breaks out of range.

    Trigger Condition:
        - Upper breakout: current_price > upper_price × (1 + breakout_threshold)
        - Lower breakout: current_price < lower_price × (1 - breakout_threshold)

    Cooldown Mechanism:
        - Maximum rebuilds within cooldown period
        - If limit reached, wait until oldest rebuild expires

    Check Interval Options:
        - "kline_close": Check at each K-line close (recommended)
        - "realtime": Real-time price monitoring
        - "interval_30s": Check every 30 seconds

    Validation Rules:
        - breakout_threshold: 0.01 <= threshold <= 0.20 (1% - 20%)
        - cooldown_days: 1 <= days <= 30
        - max_rebuilds: 1 <= max <= 10

    Example:
        >>> config = DynamicAdjustConfig(
        ...     enabled=True,
        ...     breakout_threshold=Decimal("0.04"),  # 4%
        ...     cooldown_days=7,
        ...     max_rebuilds=3,
        ...     check_interval="kline_close",
        ... )
    """

    enabled: bool = True
    breakout_threshold: Decimal = field(default_factory=lambda: Decimal("0.04"))  # 4%
    cooldown_days: int = 7
    max_rebuilds: int = 3
    check_interval: str = "kline_close"  # "kline_close", "realtime", "interval_30s"

    def __post_init__(self):
        """Ensure Decimal types and validate parameters."""
        if not isinstance(self.breakout_threshold, Decimal):
            self.breakout_threshold = Decimal(str(self.breakout_threshold))

        # Validate breakout_threshold: 1% - 20%
        if not (Decimal("0.01") <= self.breakout_threshold <= Decimal("0.20")):
            raise ValueError(
                f"breakout_threshold must be between 0.01 (1%) and 0.20 (20%), "
                f"got {self.breakout_threshold}"
            )

        # Validate cooldown_days: 1 - 30
        if not (1 <= self.cooldown_days <= 30):
            raise ValueError(
                f"cooldown_days must be between 1 and 30, got {self.cooldown_days}"
            )

        # Validate max_rebuilds: 1 - 10
        if not (1 <= self.max_rebuilds <= 10):
            raise ValueError(
                f"max_rebuilds must be between 1 and 10, got {self.max_rebuilds}"
            )

        # Validate check_interval
        valid_intervals = {"kline_close", "realtime", "interval_30s"}
        if self.check_interval not in valid_intervals:
            raise ValueError(
                f"check_interval must be one of {valid_intervals}, "
                f"got '{self.check_interval}'"
            )


# =============================================================================
# Grid Configuration
# =============================================================================


@dataclass
class GridConfig:
    """
    User configuration for grid trading.

    Minimal required input is symbol and total_investment.
    All other parameters have sensible defaults.

    Attributes:
        symbol: Trading pair (e.g., "BTCUSDT")
        total_investment: Total investment amount
        atr_config: ATR configuration (period, multiplier, etc.)
        risk_level: Risk level (determines default multiplier if not custom)
        grid_type: Grid spacing type (arithmetic/geometric)
        grid_mode: Trading direction mode
        min_grid_count: Minimum number of grid levels
        max_grid_count: Maximum number of grid levels
        min_order_value: Minimum order value per level
        stop_loss_percent: Stop loss percentage
        dynamic_adjust: Dynamic adjustment configuration
        manual_upper_price: Manual override for upper price
        manual_lower_price: Manual override for lower price
        manual_grid_count: Manual override for grid count

    Example:
        >>> # Minimal config (uses moderate defaults)
        >>> config = GridConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("1000"),
        ... )

        >>> # With custom ATR multiplier
        >>> config = GridConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("1000"),
        ...     risk_level=RiskLevel.CUSTOM,
        ...     atr_config=ATRConfig(multiplier=Decimal("1.8")),
        ... )

        >>> # Using factory method
        >>> config = GridConfig(
        ...     symbol="BTCUSDT",
        ...     total_investment=Decimal("1000"),
        ...     atr_config=ATRConfig.aggressive(),
        ... )
    """

    # Required fields
    symbol: str
    total_investment: Decimal

    # ATR configuration (new unified config)
    atr_config: ATRConfig = field(default_factory=ATRConfig)

    # Fields with defaults
    risk_level: RiskLevel = RiskLevel.MODERATE
    grid_type: GridType = GridType.GEOMETRIC
    grid_mode: GridMode = GridMode.NEUTRAL
    min_grid_count: int = 5
    max_grid_count: int = 50
    min_order_value: Decimal = field(default_factory=lambda: Decimal("10"))
    stop_loss_percent: Decimal = field(default_factory=lambda: Decimal("20"))

    # Dynamic adjustment settings
    dynamic_adjust: DynamicAdjustConfig = field(default_factory=DynamicAdjustConfig)

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

        # Sync ATR config multiplier with risk level if not custom
        if self.risk_level != RiskLevel.CUSTOM:
            self.atr_config = ATRConfig(
                period=self.atr_config.period,
                timeframe=self.atr_config.timeframe,
                multiplier=self.risk_level.default_multiplier,
                use_ema=self.atr_config.use_ema,
            )

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

        # Validate CUSTOM risk level requires explicit multiplier
        if self.risk_level == RiskLevel.CUSTOM:
            if self.atr_config.multiplier == Decimal("2.0"):
                # Using default, which is fine but could warn
                pass

    @property
    def has_manual_range(self) -> bool:
        """Check if manual price range is set."""
        return self.manual_upper_price is not None and self.manual_lower_price is not None

    @property
    def has_manual_grid_count(self) -> bool:
        """Check if manual grid count is set."""
        return self.manual_grid_count is not None

    @property
    def effective_multiplier(self) -> Decimal:
        """Get the effective ATR multiplier being used."""
        return self.atr_config.multiplier


# =============================================================================
# Grid Level
# =============================================================================


@dataclass
class GridLevel:
    """
    Individual grid level.

    Each level represents a price point where buy/sell orders are placed.

    Attributes:
        index: Level index (0 = lowest price)
        price: Price at this level
        side: Buy or sell side
        state: Current state of this level
        allocated_amount: Amount of funds allocated to this level
        quantity: Calculated quantity (allocated_amount / price)
        order_id: Associated order ID (if any)
        filled_quantity: Quantity that has been filled
        filled_price: Actual fill price
        filled_at: Timestamp when order was filled

    Example:
        >>> level = GridLevel(
        ...     index=5,
        ...     price=Decimal("49000"),
        ...     side=LevelSide.BUY,
        ...     state=LevelState.EMPTY,
        ...     allocated_amount=Decimal("100"),
        ... )
        >>> print(level.quantity)
        0.00204081632653061224489795918
    """

    index: int
    price: Decimal
    side: LevelSide
    state: LevelState = LevelState.EMPTY
    allocated_amount: Decimal = field(default_factory=lambda: Decimal("0"))
    order_id: Optional[str] = None
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    filled_price: Optional[Decimal] = None
    filled_at: Optional[datetime] = None

    @property
    def quantity(self) -> Decimal:
        """
        Calculate quantity for this level.

        Returns:
            allocated_amount / price
        """
        if self.price <= 0:
            return Decimal("0")
        return self.allocated_amount / self.price

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


# =============================================================================
# Rebuild Info
# =============================================================================


@dataclass
class RebuildInfo:
    """
    Information about a grid rebuild event.

    Created when a grid is rebuilt due to price breakout or manual trigger.
    Contains the old boundaries, new center price, and rebuild timestamp.

    Attributes:
        old_upper: Previous upper boundary before rebuild
        old_lower: Previous lower boundary before rebuild
        new_center: New center price used for rebuild
        rebuilt_at: Timestamp when rebuild occurred
        reason: Optional reason for rebuild (e.g., "upper_breakout", "lower_breakout")

    Example:
        >>> info = RebuildInfo(
        ...     old_upper=Decimal("52000"),
        ...     old_lower=Decimal("48000"),
        ...     new_center=Decimal("55000"),
        ...     reason="upper_breakout",
        ... )
    """

    old_upper: Decimal
    old_lower: Decimal
    new_center: Decimal
    rebuilt_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reason: Optional[str] = None

    def __post_init__(self):
        """Ensure Decimal types."""
        if not isinstance(self.old_upper, Decimal):
            self.old_upper = Decimal(str(self.old_upper))
        if not isinstance(self.old_lower, Decimal):
            self.old_lower = Decimal(str(self.old_lower))
        if not isinstance(self.new_center, Decimal):
            self.new_center = Decimal(str(self.new_center))


# =============================================================================
# Grid Setup
# =============================================================================


@dataclass
class GridSetup:
    """
    Complete grid setup with calculated parameters.

    This is the result of grid calculation containing all necessary
    information to execute the grid strategy.

    Attributes:
        config: Original grid configuration
        atr_data: ATR calculation result
        upper_price: Final upper boundary
        lower_price: Final lower boundary
        current_price: Price when grid was created
        grid_count: Number of grid levels
        grid_spacing_percent: Spacing between levels as percentage
        amount_per_grid: Amount allocated per level
        levels: List of all grid levels
        expected_profit_per_trade: Expected profit per completed grid trade
        created_at: Timestamp when grid was created
        version: Version number (increments on rebuild)
        rebuild_info: Information about rebuild (if this is a rebuilt grid)

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
        ...     expected_profit_per_trade=Decimal("0.60"),
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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    rebuild_info: Optional[RebuildInfo] = None

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
    def total_allocated(self) -> Decimal:
        """Calculate total allocated funds across all levels."""
        return sum(l.allocated_amount for l in self.levels)

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

    @property
    def expected_profit_per_grid(self) -> Decimal:
        """Alias for expected_profit_per_trade."""
        return self.expected_profit_per_trade

    def get_level_at_price(self, price: Decimal, tolerance_percent: Decimal = Decimal("0.1")) -> Optional[GridLevel]:
        """
        Find level closest to a given price within tolerance.

        Args:
            price: Target price
            tolerance_percent: Tolerance percentage

        Returns:
            GridLevel if found within tolerance, else None
        """
        # Protect against division by zero
        if price <= 0:
            return None

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
            "total_allocated": str(self.total_allocated),
            "buy_levels": self.total_buy_levels,
            "sell_levels": self.total_sell_levels,
            "atr_value": str(self.atr_data.value),
            "atr_multiplier": str(self.atr_data.multiplier),
            "volatility_percent": str(self.atr_data.volatility_percent),
            "volatility_level": self.atr_data.volatility_level,
            "expected_profit_per_trade": str(self.expected_profit_per_trade),
            "risk_level": self.config.risk_level.value,
            "grid_type": self.config.grid_type.value,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }
