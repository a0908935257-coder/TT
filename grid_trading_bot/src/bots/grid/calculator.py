"""
Smart Grid Calculator.

Provides intelligent grid calculation with automatic range, grid count,
price levels, and pyramid fund allocation.
"""

from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from typing import Any, Optional, Sequence

from src.core import get_logger

from .atr import ATRCalculator
from .exceptions import (
    GridCalculationError,
    InsufficientDataError,
    InsufficientFundError,
    InvalidPriceRangeError,
)
from .models import (
    ATRConfig,
    ATRData,
    GridConfig,
    GridLevel,
    GridSetup,
    GridType,
    LevelSide,
    LevelState,
    RebuildInfo,
    RiskLevel,
)

logger = get_logger(__name__)


class SmartGridCalculator:
    """
    Intelligent grid calculator with automatic parameter optimization.

    Automatically calculates:
    - Price range based on ATR and risk level
    - Optimal grid count based on investment and profitability
    - Price levels (arithmetic or geometric spacing)
    - Pyramid fund allocation (buy more at lower prices)
    - Expected profit per trade

    Example:
        >>> config = GridConfig(symbol="BTCUSDT", total_investment=Decimal("10000"))
        >>> calculator = SmartGridCalculator(config, klines=klines)
        >>> setup = calculator.calculate()
        >>> print(setup.summary())
    """

    # Fee constants
    MAKER_FEE = Decimal("0.001")       # 0.1%
    TAKER_FEE = Decimal("0.001")       # 0.1%
    ROUND_TRIP_FEE = Decimal("0.002")  # 0.2% (buy + sell)
    MIN_PROFIT_RATE = Decimal("0.001") # Minimum 0.1% profit after fees

    def __init__(
        self,
        config: GridConfig,
        klines: Optional[Sequence[Any]] = None,
        current_price: Optional[Decimal] = None,
    ):
        """
        Initialize SmartGridCalculator.

        Args:
            config: Grid configuration
            klines: K-line data for ATR calculation (optional if manual range set)
            current_price: Current market price (optional, derived from klines)
        """
        self._config = config
        self._klines = klines
        self._current_price = Decimal(str(current_price)) if current_price else None
        self._atr_data: Optional[ATRData] = None

    def calculate(self, center_price: Optional[Decimal] = None) -> GridSetup:
        """
        Calculate complete grid setup.

        Calculation flow:
        1. Calculate ATR (if klines provided)
        2. Validate market conditions
        3. Calculate price range (ATR × multiplier)
        4. Calculate optimal grid count
        5. Calculate spacing percentage
        6. Generate price levels
        7. Create levels with pyramid fund allocation
        8. Calculate expected profit

        Args:
            center_price: Optional center price for range calculation.
                         If provided, uses this as center instead of klines[-1].close.
                         Useful for grid rebuilding with new center price.

        Returns:
            GridSetup with all calculated parameters

        Raises:
            InsufficientDataError: If klines insufficient
            InsufficientFundError: If investment too low
            GridCalculationError: If calculation fails

        Example:
            >>> # Normal calculation using klines
            >>> setup = calculator.calculate()

            >>> # Rebuild with new center price
            >>> setup = calculator.calculate(center_price=Decimal("57500"))
        """
        logger.info(f"Calculating grid for {self._config.symbol}")

        # If center_price provided, use it as current price
        if center_price is not None:
            self._current_price = Decimal(str(center_price))

        # Step 1: Calculate ATR or use manual range
        if self._config.has_manual_range:
            self._atr_data = self._create_placeholder_atr()
            upper_price = self._config.manual_upper_price
            lower_price = self._config.manual_lower_price
        else:
            self._calculate_atr()
            # If center_price provided, recalculate range around it
            if center_price is not None:
                upper_price, lower_price = self._calculate_price_range_from_center(center_price)
            else:
                upper_price, lower_price = self._calculate_price_range()

        # Validate price range
        self._validate_price_range(upper_price, lower_price)

        # Get current price
        current_price = self._get_current_price()

        # Step 4: Calculate optimal grid count
        grid_count = self._calculate_optimal_grid_count(upper_price, lower_price)

        # Step 5: Calculate spacing
        grid_spacing_percent = self._calculate_spacing_percent(
            upper_price, lower_price, grid_count
        )

        # Step 6 & 7: Generate levels with pyramid allocation
        levels = self._generate_levels(
            upper_price=upper_price,
            lower_price=lower_price,
            current_price=current_price,
            grid_count=grid_count,
        )

        # Step 8: Calculate expected profit
        expected_profit = self._calculate_expected_profit(grid_spacing_percent)

        # Calculate amount per grid (average)
        amount_per_grid = self._config.total_investment / Decimal(grid_count)

        setup = GridSetup(
            config=self._config,
            atr_data=self._atr_data,
            upper_price=upper_price,
            lower_price=lower_price,
            current_price=current_price,
            grid_count=grid_count,
            grid_spacing_percent=grid_spacing_percent,
            amount_per_grid=amount_per_grid,
            levels=levels,
            expected_profit_per_trade=expected_profit,
        )

        logger.info(
            f"Grid calculated: {len(levels)} levels ({grid_count} grids), "
            f"range {lower_price:.2f}-{upper_price:.2f}, "
            f"spacing {grid_spacing_percent:.2f}%"
        )

        return setup

    def _calculate_atr(self) -> None:
        """Calculate ATR from klines."""
        atr_config = self._config.atr_config

        if self._klines is None or len(self._klines) == 0:
            raise InsufficientDataError(
                required=atr_config.period + 1,
                actual=0,
            )

        min_required = atr_config.period + 1
        if len(self._klines) < min_required:
            raise InsufficientDataError(
                required=min_required,
                actual=len(self._klines),
            )

        self._atr_data = ATRCalculator.calculate_from_klines(
            klines=self._klines,
            config=atr_config,
        )

        # Update current price from klines if not set
        if self._current_price is None:
            self._current_price = self._atr_data.current_price

        logger.debug(
            f"ATR calculated: {self._atr_data.value:.2f} "
            f"({self._atr_data.volatility_percent:.2f}%)"
        )

    def _create_placeholder_atr(self) -> ATRData:
        """Create placeholder ATR data for manual range mode."""
        current_price = self._get_current_price()
        upper = self._config.manual_upper_price
        lower = self._config.manual_lower_price
        atr_config = self._config.atr_config

        # Estimate ATR from range (assuming 2x multiplier)
        estimated_atr = (upper - lower) / (atr_config.multiplier * Decimal("2"))

        return ATRData(
            value=estimated_atr,
            period=atr_config.period,
            timeframe=atr_config.timeframe,
            multiplier=atr_config.multiplier,
            current_price=current_price,
            upper_price=upper,
            lower_price=lower,
        )

    def _calculate_price_range(self) -> tuple[Decimal, Decimal]:
        """
        Calculate price range based on ATR and risk level.

        The ATRData already contains calculated upper/lower prices
        based on the configured multiplier, so we use those directly.

        Returns:
            Tuple of (upper_price, lower_price)
        """
        # ATRData already has calculated boundaries using the configured multiplier
        return self._atr_data.upper_price, self._atr_data.lower_price

    def _calculate_price_range_from_center(
        self,
        center_price: Decimal,
    ) -> tuple[Decimal, Decimal]:
        """
        Calculate price range centered around a specific price.

        Used for grid rebuilding when price has moved to a new range.

        Args:
            center_price: The price to center the range around

        Returns:
            Tuple of (upper_price, lower_price)

        Raises:
            GridCalculationError: If ATR data is not available
        """
        if self._atr_data is None:
            raise GridCalculationError("ATR data not calculated - call calculate() first")

        multiplier = self._config.atr_config.multiplier
        range_width = self._atr_data.value * multiplier

        upper_price = center_price + range_width
        lower_price = center_price - range_width

        # Ensure lower price is positive
        if lower_price <= 0:
            lower_price = center_price * Decimal("0.5")
            upper_price = center_price * Decimal("1.5")

        return upper_price, lower_price

    def _validate_price_range(self, upper: Decimal, lower: Decimal) -> None:
        """Validate price range."""
        if upper <= lower:
            raise InvalidPriceRangeError(
                str(upper), str(lower), "upper must be > lower"
            )

        if lower <= 0:
            raise InvalidPriceRangeError(
                str(upper), str(lower), "lower must be positive"
            )

        # Check range is reasonable (not too narrow)
        range_percent = ((upper - lower) / lower) * Decimal("100")
        min_range = self.ROUND_TRIP_FEE * Decimal("100") * Decimal("2")  # At least 2x fees

        if range_percent < min_range:
            raise InvalidPriceRangeError(
                str(upper), str(lower),
                f"range too narrow ({range_percent:.2f}%), need at least {min_range:.2f}%"
            )

    def _get_current_price(self) -> Decimal:
        """Get current price from various sources."""
        if self._current_price is not None:
            return self._current_price

        if self._atr_data is not None:
            return self._atr_data.current_price

        if self._config.has_manual_range:
            # Use midpoint of manual range
            return (self._config.manual_upper_price + self._config.manual_lower_price) / Decimal("2")

        raise GridCalculationError("Cannot determine current price")

    def _calculate_optimal_grid_count(
        self,
        upper: Decimal,
        lower: Decimal,
    ) -> int:
        """
        Calculate optimal grid count.

        Considers:
        - Maximum by available funds
        - Maximum for profitable trades (spacing > fees)
        - User-configured limits

        Returns:
            Optimal grid count
        """
        # Check for manual override
        if self._config.has_manual_grid_count:
            return self._config.manual_grid_count

        # Max grids by fund (each grid needs min_order_value, buy and sell sides)
        # We allocate to buy side primarily
        max_by_fund = int(
            self._config.total_investment / self._config.min_order_value
        )

        # Max grids for profitable trades
        # Each trade should profit at least MIN_PROFIT_RATE after fees
        range_percent = ((upper - lower) / lower) * Decimal("100")
        min_spacing = (self.ROUND_TRIP_FEE + self.MIN_PROFIT_RATE) * Decimal("100")
        max_by_profit = int(range_percent / min_spacing)

        # Calculate optimal
        optimal = min(
            max_by_fund,
            max_by_profit,
            self._config.max_grid_count,
        )
        optimal = max(optimal, self._config.min_grid_count)

        # Validate we have enough funds
        required_funds = self._config.min_order_value * Decimal(optimal)
        if self._config.total_investment < required_funds:
            raise InsufficientFundError(
                required=str(required_funds),
                actual=str(self._config.total_investment),
                min_per_grid=str(self._config.min_order_value),
                grid_count=optimal,
            )

        logger.debug(
            f"Grid count: max_by_fund={max_by_fund}, max_by_profit={max_by_profit}, "
            f"optimal={optimal}"
        )

        return optimal

    def _calculate_spacing_percent(
        self,
        upper: Decimal,
        lower: Decimal,
        grid_count: int,
    ) -> Decimal:
        """
        Calculate grid spacing percentage.

        For geometric grids:
            ratio = (upper / lower) ^ (1 / grid_count) - 1

        For arithmetic grids:
            spacing = (upper - lower) / grid_count / lower

        Returns:
            Spacing percentage
        """
        if self._config.grid_type == GridType.GEOMETRIC:
            # Geometric: ratio = (upper/lower)^(1/n) - 1
            ratio = (upper / lower) ** (Decimal("1") / Decimal(grid_count))
            spacing_percent = (ratio - Decimal("1")) * Decimal("100")
        else:
            # Arithmetic: spacing = (upper - lower) / n / lower
            step = (upper - lower) / Decimal(grid_count)
            spacing_percent = (step / lower) * Decimal("100")

        return spacing_percent

    def _generate_levels(
        self,
        upper_price: Decimal,
        lower_price: Decimal,
        current_price: Decimal,
        grid_count: int,
    ) -> list[GridLevel]:
        """
        Generate grid levels with pyramid fund allocation.

        Pyramid allocation: Lower price levels get more funds (buy more when cheaper).
        Weight formula: weight[i] = 1 + 0.5 × (n - 1 - i) / (n - 1)
        This gives the lowest price ~1.5x the allocation of the highest.

        Args:
            upper_price: Upper boundary
            lower_price: Lower boundary
            current_price: Current market price
            grid_count: Number of grid levels

        Returns:
            List of GridLevel objects
        """
        # Generate price levels
        prices = self._generate_prices(upper_price, lower_price, grid_count)

        # Separate into buy and sell levels based on current price
        buy_prices = [p for p in prices if p < current_price]
        sell_prices = [p for p in prices if p >= current_price]

        # Calculate pyramid weights for buy levels
        buy_allocations = self._calculate_pyramid_allocation(
            len(buy_prices),
            self._config.total_investment,
        )

        # Create levels
        levels = []

        # Buy levels (below current price)
        for i, (price, amount) in enumerate(zip(buy_prices, buy_allocations)):
            level = GridLevel(
                index=i,
                price=price,
                side=LevelSide.BUY,
                state=LevelState.EMPTY,
                allocated_amount=amount,
            )
            levels.append(level)

        # Sell levels (above current price) - no initial allocation
        for i, price in enumerate(sell_prices):
            level = GridLevel(
                index=len(buy_prices) + i,
                price=price,
                side=LevelSide.SELL,
                state=LevelState.EMPTY,
                allocated_amount=Decimal("0"),
            )
            levels.append(level)

        # Sort by price ascending
        levels.sort(key=lambda x: x.price)

        # Re-index after sorting
        for i, level in enumerate(levels):
            level.index = i

        return levels

    def _generate_prices(
        self,
        upper: Decimal,
        lower: Decimal,
        grid_count: int,
    ) -> list[Decimal]:
        """
        Generate price levels.

        For geometric grids:
            prices[i] = lower × ratio^i
            where ratio = (upper/lower)^(1/n)

        For arithmetic grids:
            prices[i] = lower + step × i
            where step = (upper - lower) / n

        Returns:
            List of prices from lower to upper
        """
        prices = []

        if self._config.grid_type == GridType.GEOMETRIC:
            ratio = (upper / lower) ** (Decimal("1") / Decimal(grid_count))
            for i in range(grid_count + 1):
                price = lower * (ratio ** Decimal(i))
                prices.append(price)
        else:
            step = (upper - lower) / Decimal(grid_count)
            for i in range(grid_count + 1):
                price = lower + step * Decimal(i)
                prices.append(price)

        return prices

    def _calculate_pyramid_allocation(
        self,
        buy_levels_count: int,
        total_investment: Decimal,
    ) -> list[Decimal]:
        """
        Calculate pyramid fund allocation for buy levels.

        Lower prices get higher allocation (buy more when cheaper).
        Weight formula: weight[i] = 1 + 0.5 × (n - 1 - i) / (n - 1)

        Args:
            buy_levels_count: Number of buy levels
            total_investment: Total investment amount

        Returns:
            List of allocations for each buy level (lowest price first)
        """
        if buy_levels_count == 0:
            return []

        if buy_levels_count == 1:
            return [total_investment]

        # Calculate weights (higher weight for lower prices)
        # Prices are sorted low to high, so index 0 is lowest price
        weights = []
        n = buy_levels_count

        # Defensive check: n must be >= 2 to avoid division by zero
        # (n <= 1 cases are already handled above, but guard anyway)
        if n < 2:
            return [total_investment] if n == 1 else []

        for i in range(n):
            # Index 0 (lowest price) gets highest weight
            # weight = 1 + 0.5 × (n - 1 - i) / (n - 1)
            weight = Decimal("1") + Decimal("0.5") * Decimal(n - 1 - i) / Decimal(n - 1)
            weights.append(weight)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            # Fallback: equal allocation if weights sum to zero (should not happen)
            equal_alloc = (total_investment / Decimal(n)).quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            )
            return [equal_alloc] * n

        allocations = [
            (w / total_weight * total_investment).quantize(
                Decimal("0.01"), rounding=ROUND_DOWN
            )
            for w in weights
        ]

        # Adjust for rounding errors - add remainder to lowest price level
        remainder = total_investment - sum(allocations)
        if remainder > 0 and allocations:
            allocations[0] += remainder

        return allocations

    def _calculate_expected_profit(self, spacing_percent: Decimal) -> Decimal:
        """
        Calculate expected profit per trade.

        profit = spacing% - round_trip_fee%

        Returns:
            Expected profit percentage per trade
        """
        fee_percent = self.ROUND_TRIP_FEE * Decimal("100")
        return spacing_percent - fee_percent


# =========================================================================
# Convenience Functions
# =========================================================================


def create_grid(
    symbol: str,
    investment: Decimal | float | str,
    klines: Sequence[Any],
    risk_level: RiskLevel = RiskLevel.MODERATE,
    grid_type: GridType = GridType.GEOMETRIC,
    atr_config: Optional[ATRConfig] = None,
) -> GridSetup:
    """
    Quick function to create grid setup.

    Args:
        symbol: Trading pair symbol
        investment: Total investment amount
        klines: K-line data for ATR calculation
        risk_level: Risk level (default: MODERATE)
        grid_type: Grid type (default: GEOMETRIC)
        atr_config: ATR configuration (optional, uses defaults if not provided)

    Returns:
        Calculated GridSetup

    Example:
        >>> setup = create_grid(
        ...     symbol="BTCUSDT",
        ...     investment=10000,
        ...     klines=klines,
        ... )

        >>> # With custom ATR config
        >>> setup = create_grid(
        ...     symbol="BTCUSDT",
        ...     investment=10000,
        ...     klines=klines,
        ...     atr_config=ATRConfig.aggressive(),
        ... )
    """
    if atr_config is None:
        atr_config = ATRConfig()

    config = GridConfig(
        symbol=symbol,
        total_investment=Decimal(str(investment)),
        risk_level=risk_level,
        grid_type=grid_type,
        atr_config=atr_config,
    )

    calculator = SmartGridCalculator(config=config, klines=klines)
    return calculator.calculate()


def create_grid_with_manual_range(
    symbol: str,
    investment: Decimal | float | str,
    upper_price: Decimal | float | str,
    lower_price: Decimal | float | str,
    grid_count: int,
    current_price: Decimal | float | str,
    grid_type: GridType = GridType.GEOMETRIC,
) -> GridSetup:
    """
    Create grid setup with manual price range.

    Args:
        symbol: Trading pair symbol
        investment: Total investment amount
        upper_price: Upper price boundary
        lower_price: Lower price boundary
        grid_count: Number of grid levels
        current_price: Current market price
        grid_type: Grid type (default: GEOMETRIC)

    Returns:
        Calculated GridSetup

    Example:
        >>> setup = create_grid_with_manual_range(
        ...     symbol="BTCUSDT",
        ...     investment=10000,
        ...     upper_price=55000,
        ...     lower_price=45000,
        ...     grid_count=10,
        ...     current_price=50000,
        ... )
    """
    config = GridConfig(
        symbol=symbol,
        total_investment=Decimal(str(investment)),
        grid_type=grid_type,
        manual_upper_price=Decimal(str(upper_price)),
        manual_lower_price=Decimal(str(lower_price)),
        manual_grid_count=grid_count,
    )

    calculator = SmartGridCalculator(
        config=config,
        current_price=Decimal(str(current_price)),
    )
    return calculator.calculate()


def rebuild_grid(
    old_setup: GridSetup,
    klines: Sequence[Any],
    new_center_price: Decimal | float | str,
    reason: Optional[str] = None,
) -> GridSetup:
    """
    Rebuild grid with new center price, preserving original configuration.

    This function creates a new grid setup centered around a new price while
    keeping the same grid parameters (investment, grid count, risk level, etc.).
    The new setup's version is incremented and rebuild_info is populated with
    the old boundaries and new center price.

    Use this when:
    - Price breaks out of the current grid range
    - Manual grid reconfiguration is needed
    - Dynamic adjustment triggers a grid rebuild

    Args:
        old_setup: The existing GridSetup to rebuild from
        klines: K-line data for ATR recalculation
        new_center_price: The new center price for the grid
        reason: Optional reason for rebuild (e.g., "upper_breakout", "lower_breakout")

    Returns:
        New GridSetup with:
        - version: old_setup.version + 1
        - rebuild_info populated with old boundaries and new center
        - New price range centered around new_center_price

    Example:
        >>> # Price broke out above grid, rebuild with new center
        >>> new_setup = rebuild_grid(
        ...     old_setup=current_setup,
        ...     klines=latest_klines,
        ...     new_center_price=Decimal("57500"),
        ...     reason="upper_breakout",
        ... )
        >>> print(f"Version: {new_setup.version}")  # old version + 1
        >>> print(f"Old range: {new_setup.rebuild_info.old_lower}-{new_setup.rebuild_info.old_upper}")
    """
    new_center = Decimal(str(new_center_price))

    # Create calculator with original config
    calculator = SmartGridCalculator(
        config=old_setup.config,
        klines=klines,
    )

    # Calculate new setup with the new center price
    new_setup = calculator.calculate(center_price=new_center)

    # Update version and rebuild info
    new_setup.version = old_setup.version + 1
    new_setup.rebuild_info = RebuildInfo(
        old_upper=old_setup.upper_price,
        old_lower=old_setup.lower_price,
        new_center=new_center,
        reason=reason,
    )

    logger.info(
        f"Grid rebuilt: v{old_setup.version} -> v{new_setup.version}, "
        f"center {old_setup.current_price:.2f} -> {new_center:.2f}, "
        f"range {new_setup.lower_price:.2f}-{new_setup.upper_price:.2f}"
    )

    return new_setup
