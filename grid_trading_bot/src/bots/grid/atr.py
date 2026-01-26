"""
ATR (Average True Range) Calculator.

Provides ATR calculation and grid parameter suggestions based on volatility.
Supports both EMA and SMA calculation methods.

Conforms to Prompt 17 specification.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, Sequence

from .models import (
    ATRConfig,
    ATRData,
    GridConfig,
    GridLevel,
    GridSetup,
    GridType,
    LevelSide,
    LevelState,
    RiskLevel,
)


class ATRCalculator:
    """
    ATR Calculator with EMA/SMA smoothing.

    ATR Formula:
        TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        ATR = EMA(TR, period) or SMA(TR, period)

    EMA Formula:
        multiplier = 2 / (period + 1)
        EMA = (value - prev_EMA) × multiplier + prev_EMA

    SMA Formula:
        SMA = sum(values) / period

    Example:
        >>> from core.models import Kline
        >>> config = ATRConfig(period=14, multiplier=Decimal("2.0"))
        >>> atr_data = ATRCalculator.calculate_from_klines(klines, config)
        >>> print(f"ATR: {atr_data.value}, Range: {atr_data.lower_price} - {atr_data.upper_price}")
    """

    # =========================================================================
    # Core Calculation Methods
    # =========================================================================

    @classmethod
    def calculate_from_klines(
        cls,
        klines: Sequence[Any],
        config: Optional[ATRConfig] = None,
    ) -> ATRData:
        """
        Calculate ATR from Kline objects.

        Args:
            klines: Sequence of Kline objects with high, low, close attributes
            config: ATR configuration (period, multiplier, timeframe, use_ema)

        Returns:
            ATRData with calculated ATR value and price boundaries

        Raises:
            ValueError: If insufficient data
        """
        if config is None:
            config = ATRConfig()

        if len(klines) < config.period + 1:
            raise ValueError(f"Need at least {config.period + 1} klines, got {len(klines)}")

        highs = [Decimal(str(k.high)) for k in klines]
        lows = [Decimal(str(k.low)) for k in klines]
        closes = [Decimal(str(k.close)) for k in klines]
        current_price = closes[-1]

        return cls.calculate(
            highs=highs,
            lows=lows,
            closes=closes,
            config=config,
            current_price=current_price,
        )

    @classmethod
    def calculate(
        cls,
        highs: Sequence[Decimal | float | str],
        lows: Sequence[Decimal | float | str],
        closes: Sequence[Decimal | float | str],
        config: Optional[ATRConfig] = None,
        current_price: Optional[Decimal | float | str] = None,
    ) -> ATRData:
        """
        Calculate ATR from price arrays.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            config: ATR configuration
            current_price: Current price (default: last close)

        Returns:
            ATRData with calculated ATR value and price boundaries

        Raises:
            ValueError: If insufficient data or mismatched lengths
        """
        if config is None:
            config = ATRConfig()

        # Convert to Decimal
        highs = [Decimal(str(h)) for h in highs]
        lows = [Decimal(str(l)) for l in lows]
        closes = [Decimal(str(c)) for c in closes]

        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("highs, lows, and closes must have same length")

        if len(highs) < config.period + 1:
            raise ValueError(f"Need at least {config.period + 1} data points, got {len(highs)}")

        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(highs)):
            tr = cls.calculate_tr(highs[i], lows[i], closes[i - 1])
            true_ranges.append(tr)

        # Calculate ATR using EMA or SMA
        if config.use_ema:
            atr = cls.calculate_ema(true_ranges, config.period)
        else:
            atr = cls.calculate_sma(true_ranges, config.period)

        # Current price
        if current_price is None:
            current_price = closes[-1]
        else:
            current_price = Decimal(str(current_price))

        # Calculate upper and lower boundaries
        upper_price = current_price + (atr * config.multiplier)
        lower_price = current_price - (atr * config.multiplier)

        # Ensure lower price is positive
        if lower_price <= 0:
            lower_price = current_price * Decimal("0.5")
            upper_price = current_price * Decimal("1.5")

        return ATRData(
            value=atr,
            period=config.period,
            timeframe=config.timeframe,
            multiplier=config.multiplier,
            current_price=current_price,
            upper_price=upper_price,
            lower_price=lower_price,
            calculated_at=datetime.now(timezone.utc),
        )

    # =========================================================================
    # Helper Calculation Methods
    # =========================================================================

    @staticmethod
    def calculate_tr(
        high: Decimal | float | str,
        low: Decimal | float | str,
        prev_close: Decimal | float | str,
    ) -> Decimal:
        """
        Calculate True Range for a single period.

        TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)

        Args:
            high: High price
            low: Low price
            prev_close: Previous close price

        Returns:
            True Range value
        """
        high = Decimal(str(high))
        low = Decimal(str(low))
        prev_close = Decimal(str(prev_close))

        return max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )

    @staticmethod
    def calculate_ema(values: list[Decimal], period: int) -> Decimal:
        """
        Calculate EMA (Exponential Moving Average).

        EMA Formula:
            multiplier = 2 / (period + 1)
            EMA[0] = SMA(values, period)
            EMA[i] = (value[i] - EMA[i-1]) × multiplier + EMA[i-1]

        Args:
            values: List of values
            period: EMA period

        Returns:
            EMA value

        Raises:
            ValueError: If insufficient data
        """
        if len(values) < period:
            raise ValueError(f"Need at least {period} values for EMA, got {len(values)}")

        # Initial SMA for first EMA value
        sma = sum(values[:period]) / Decimal(period)

        # EMA multiplier
        multiplier = Decimal("2") / (Decimal(period) + Decimal("1"))

        # Calculate EMA
        ema = sma
        for value in values[period:]:
            ema = (value - ema) * multiplier + ema

        return ema

    @staticmethod
    def calculate_sma(values: list[Decimal], period: int) -> Decimal:
        """
        Calculate SMA (Simple Moving Average).

        SMA = sum(last N values) / N

        Args:
            values: List of values
            period: SMA period

        Returns:
            SMA value

        Raises:
            ValueError: If insufficient data
        """
        if len(values) < period:
            raise ValueError(f"Need at least {period} values for SMA, got {len(values)}")

        # Use the last 'period' values
        return sum(values[-period:]) / Decimal(period)

    @staticmethod
    def get_volatility_level(volatility_percent: Decimal | float | str) -> str:
        """
        Get volatility level description based on percentage.

        Thresholds:
            < 1%  → 極低 (very_low)
            1-2%  → 低 (low)
            2-4%  → 中等 (medium)
            4-6%  → 高 (high)
            >= 6% → 極高 (very_high)

        Args:
            volatility_percent: Volatility as percentage

        Returns:
            Volatility level in Chinese
        """
        pct = Decimal(str(volatility_percent))

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

    # =========================================================================
    # Parameter Suggestion Methods
    # =========================================================================

    @classmethod
    def suggest_multiplier(cls, volatility_percent: Decimal | float | str) -> Decimal:
        """
        Suggest ATR multiplier based on current volatility.

        Logic:
            - volatility < 1%  → 2.5 - 3.0 (range too narrow, need to expand)
            - volatility 1-2%  → 2.0 (standard)
            - volatility 2-4%  → 1.5 - 2.0 (moderate volatility)
            - volatility > 4%  → 1.0 - 1.5 (already wide enough)

        Args:
            volatility_percent: Current volatility percentage

        Returns:
            Suggested ATR multiplier
        """
        pct = Decimal(str(volatility_percent))

        if pct < Decimal("1"):
            # Very low volatility - need wider range
            return Decimal("2.5")
        elif pct < Decimal("2"):
            # Low volatility - standard multiplier
            return Decimal("2.0")
        elif pct < Decimal("4"):
            # Medium volatility - slightly narrower
            return Decimal("1.75")
        else:
            # High volatility - narrow range sufficient
            return Decimal("1.5")

    @classmethod
    def suggest_parameters(
        cls,
        atr_data: ATRData,
        investment: Decimal | float | str,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        min_grid_count: int = 5,
        max_grid_count: int = 50,
        min_order_value: Decimal | float | str = 10,
    ) -> dict[str, Any]:
        """
        Suggest optimal grid parameters based on ATR and investment.

        Args:
            atr_data: ATR calculation result
            investment: Total investment amount
            risk_level: Risk level for range calculation
            min_grid_count: Minimum grid levels
            max_grid_count: Maximum grid levels
            min_order_value: Minimum order value

        Returns:
            Dictionary with suggested parameters
        """
        investment = Decimal(str(investment))
        min_order_value = Decimal(str(min_order_value))

        # Use the multiplier from ATR data (already calculated)
        upper_price = atr_data.upper_price
        lower_price = atr_data.lower_price
        price_range = upper_price - lower_price
        current_price = atr_data.current_price

        # Calculate optimal grid count based on investment
        max_grids_by_investment = int(investment / min_order_value)
        suggested_grid_count = min(max(max_grids_by_investment, min_grid_count), max_grid_count)

        # Calculate grid spacing (with division by zero protection)
        if suggested_grid_count <= 0:
            grid_spacing_percent = Decimal("0")
            amount_per_grid = investment
        else:
            grid_spacing_percent = (price_range / lower_price / Decimal(suggested_grid_count)) * Decimal("100")
            amount_per_grid = investment / Decimal(suggested_grid_count)

        # Expected profit per trade (approximately grid_spacing minus fees)
        # Assuming 0.1% fee per trade (maker/taker)
        fee_percent = Decimal("0.2")  # Round trip fee
        expected_profit_per_trade = grid_spacing_percent - fee_percent

        # Suggest multiplier based on volatility
        suggested_multiplier = cls.suggest_multiplier(atr_data.volatility_percent)

        return {
            "volatility_level": atr_data.volatility_level,
            "volatility_level_en": atr_data.volatility_level_en,
            "suggested_risk_level": risk_level.value,
            "suggested_multiplier": suggested_multiplier,
            "current_multiplier": atr_data.multiplier,
            "suggested_upper_price": upper_price,
            "suggested_lower_price": lower_price,
            "suggested_grid_count": suggested_grid_count,
            "grid_spacing_percent": grid_spacing_percent,
            "amount_per_grid": amount_per_grid,
            "expected_profit_per_trade": expected_profit_per_trade,
            "price_range_percent": (price_range / current_price) * Decimal("100"),
            "atr_value": atr_data.value,
            "atr_percent": atr_data.volatility_percent,
        }

    # =========================================================================
    # Grid Setup Creation
    # =========================================================================

    @classmethod
    def create_grid_setup(
        cls,
        config: GridConfig,
        atr_data: ATRData,
        current_price: Optional[Decimal] = None,
        version: int = 1,
    ) -> GridSetup:
        """
        Create complete grid setup from config and ATR data.

        Args:
            config: Grid configuration
            atr_data: ATR calculation result
            current_price: Current price (default: from ATR data)
            version: Grid version number (default: 1)

        Returns:
            GridSetup with all calculated parameters
        """
        if current_price is None:
            current_price = atr_data.current_price

        # Determine price range
        if config.has_manual_range:
            upper_price = config.manual_upper_price
            lower_price = config.manual_lower_price
        else:
            # Use boundaries from ATR data (already calculated with multiplier)
            upper_price = atr_data.upper_price
            lower_price = atr_data.lower_price

            # Recalculate if current price differs significantly from ATR data
            if abs(current_price - atr_data.current_price) / atr_data.current_price > Decimal("0.01"):
                # Price changed more than 1%, recalculate boundaries
                multiplier = config.effective_multiplier
                upper_price = current_price + (atr_data.value * multiplier)
                lower_price = current_price - (atr_data.value * multiplier)

                # Ensure lower price is positive
                if lower_price <= 0:
                    lower_price = current_price * Decimal("0.5")
                    upper_price = current_price * Decimal("1.5")

        # Determine grid count
        if config.has_manual_grid_count:
            grid_count = config.manual_grid_count
        else:
            # Calculate optimal grid count
            max_grids_by_investment = int(config.total_investment / config.min_order_value)
            grid_count = min(max(max_grids_by_investment, config.min_grid_count), config.max_grid_count)

        # Calculate grid spacing (with division by zero protection)
        price_range = upper_price - lower_price
        if grid_count <= 0:
            grid_spacing = price_range
            grid_spacing_percent = Decimal("0")
            amount_per_grid = config.total_investment
        else:
            grid_spacing = price_range / Decimal(grid_count)
            grid_spacing_percent = (grid_spacing / lower_price) * Decimal("100")
            amount_per_grid = config.total_investment / Decimal(grid_count)

        # Create grid levels
        levels = cls._create_levels(
            upper_price=upper_price,
            lower_price=lower_price,
            current_price=current_price,
            grid_count=grid_count,
            amount_per_grid=amount_per_grid,
            grid_type=config.grid_type,
        )

        # Expected profit per trade
        fee_percent = Decimal("0.2")  # Round trip fee assumption
        expected_profit_per_trade = grid_spacing_percent - fee_percent

        return GridSetup(
            config=config,
            atr_data=atr_data,
            upper_price=upper_price,
            lower_price=lower_price,
            current_price=current_price,
            grid_count=grid_count,
            grid_spacing_percent=grid_spacing_percent,
            amount_per_grid=amount_per_grid,
            levels=levels,
            expected_profit_per_trade=expected_profit_per_trade,
            created_at=datetime.now(timezone.utc),
            version=version,
        )

    @classmethod
    def _create_levels(
        cls,
        upper_price: Decimal,
        lower_price: Decimal,
        current_price: Decimal,
        grid_count: int,
        amount_per_grid: Decimal,
        grid_type: GridType | str,
    ) -> list[GridLevel]:
        """
        Create grid levels.

        Args:
            upper_price: Upper price boundary
            lower_price: Lower price boundary
            current_price: Current market price
            grid_count: Number of grid levels
            amount_per_grid: Amount allocated per level
            grid_type: ARITHMETIC or GEOMETRIC

        Returns:
            List of GridLevel objects
        """
        levels = []

        # Handle both enum and string types
        grid_type_str = grid_type.value if isinstance(grid_type, GridType) else grid_type

        # Guard against division by zero
        if grid_count <= 0:
            prices = [lower_price, upper_price]
        elif grid_type_str == "geometric":
            # Geometric: equal percentage spacing
            ratio = (upper_price / lower_price) ** (Decimal("1") / Decimal(grid_count))
            prices = [lower_price * (ratio ** Decimal(i)) for i in range(grid_count + 1)]
        else:
            # Arithmetic: equal price spacing
            step = (upper_price - lower_price) / Decimal(grid_count)
            prices = [lower_price + step * Decimal(i) for i in range(grid_count + 1)]

        for i, price in enumerate(prices):
            # Determine side based on current price
            # Levels below current price are BUY levels
            # Levels above current price are SELL levels
            if price < current_price:
                side = LevelSide.BUY
            else:
                side = LevelSide.SELL

            level = GridLevel(
                index=i,
                price=price,
                side=side,
                state=LevelState.EMPTY,
                allocated_amount=amount_per_grid,
            )
            levels.append(level)

        return levels


# =============================================================================
# Backward Compatibility
# =============================================================================

# For backward compatibility with code using old parameter style
def calculate_atr_legacy(
    klines: Sequence[Any],
    period: int = 14,
    timeframe: str = "4h",
    multiplier: Decimal | float | str = "2.0",
    use_ema: bool = True,
) -> ATRData:
    """
    Legacy function for backward compatibility.

    Use ATRCalculator.calculate_from_klines(klines, ATRConfig(...)) instead.
    """
    config = ATRConfig(
        period=period,
        timeframe=timeframe,
        multiplier=Decimal(str(multiplier)),
        use_ema=use_ema,
    )
    return ATRCalculator.calculate_from_klines(klines, config)
