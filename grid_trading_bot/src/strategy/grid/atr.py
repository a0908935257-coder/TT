"""
ATR (Average True Range) Calculator.

Provides ATR calculation and grid parameter suggestions based on volatility.
"""

from decimal import Decimal
from typing import Any, Optional, Sequence

from .models import ATRData, GridConfig, GridLevel, GridSetup, LevelSide, LevelState, RiskLevel


class ATRCalculator:
    """
    ATR Calculator with EMA smoothing.

    ATR Formula:
        TR = max(High - Low, |High - Prev Close|, |Low - Prev Close|)
        ATR = EMA(TR, period)

    EMA Formula:
        multiplier = 2 / (period + 1)
        EMA = (value - prev_EMA) × multiplier + prev_EMA

    Example:
        >>> from core.models import Kline
        >>> atr_data = ATRCalculator.calculate_from_klines(klines, period=14)
        >>> print(f"ATR: {atr_data.value}, Volatility: {atr_data.volatility_level}")
    """

    @classmethod
    def calculate_from_klines(
        cls,
        klines: Sequence[Any],
        period: int = 14,
        timeframe: str = "4h",
    ) -> ATRData:
        """
        Calculate ATR from Kline objects.

        Args:
            klines: Sequence of Kline objects with high, low, close attributes
            period: ATR period (default 14)
            timeframe: Kline timeframe string

        Returns:
            ATRData with calculated ATR value

        Raises:
            ValueError: If insufficient data
        """
        if len(klines) < period + 1:
            raise ValueError(f"Need at least {period + 1} klines, got {len(klines)}")

        highs = [Decimal(str(k.high)) for k in klines]
        lows = [Decimal(str(k.low)) for k in klines]
        closes = [Decimal(str(k.close)) for k in klines]
        current_price = closes[-1]

        return cls.calculate(
            highs=highs,
            lows=lows,
            closes=closes,
            period=period,
            current_price=current_price,
            timeframe=timeframe,
        )

    @classmethod
    def calculate(
        cls,
        highs: Sequence[Decimal | float | str],
        lows: Sequence[Decimal | float | str],
        closes: Sequence[Decimal | float | str],
        period: int = 14,
        current_price: Optional[Decimal | float | str] = None,
        timeframe: str = "4h",
    ) -> ATRData:
        """
        Calculate ATR from price arrays.

        Args:
            highs: High prices
            lows: Low prices
            closes: Close prices
            period: ATR period (default 14)
            current_price: Current price (default: last close)
            timeframe: Timeframe string

        Returns:
            ATRData with calculated ATR value

        Raises:
            ValueError: If insufficient data or mismatched lengths
        """
        # Convert to Decimal
        highs = [Decimal(str(h)) for h in highs]
        lows = [Decimal(str(l)) for l in lows]
        closes = [Decimal(str(c)) for c in closes]

        if len(highs) != len(lows) or len(highs) != len(closes):
            raise ValueError("highs, lows, and closes must have same length")

        if len(highs) < period + 1:
            raise ValueError(f"Need at least {period + 1} data points, got {len(highs)}")

        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(highs)):
            high = highs[i]
            low = lows[i]
            prev_close = closes[i - 1]

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        # Calculate ATR using EMA
        atr = cls._calculate_ema(true_ranges, period)

        # Current price
        if current_price is None:
            current_price = closes[-1]
        else:
            current_price = Decimal(str(current_price))

        return ATRData(
            value=atr,
            period=period,
            timeframe=timeframe,
            current_price=current_price,
        )

    @classmethod
    def _calculate_ema(cls, values: list[Decimal], period: int) -> Decimal:
        """
        Calculate EMA (Exponential Moving Average).

        Args:
            values: List of values
            period: EMA period

        Returns:
            EMA value
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

    @classmethod
    def suggest_parameters(
        cls,
        atr_data: ATRData,
        investment: Decimal | float | str,
        risk_level: RiskLevel = RiskLevel.MEDIUM,
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

        # Calculate price range based on ATR and risk level
        atr_multiplier = risk_level.atr_multiplier
        price_range = atr_data.value * atr_multiplier * Decimal("2")  # Total range (upper - lower)

        current_price = atr_data.current_price
        half_range = price_range / Decimal("2")

        upper_price = current_price + half_range
        lower_price = current_price - half_range

        # Ensure lower price is positive
        if lower_price <= 0:
            lower_price = current_price * Decimal("0.5")
            upper_price = current_price * Decimal("1.5")
            price_range = upper_price - lower_price

        # Calculate optimal grid count based on investment
        max_grids_by_investment = int(investment / min_order_value)
        suggested_grid_count = min(max(max_grids_by_investment, min_grid_count), max_grid_count)

        # Calculate grid spacing
        grid_spacing_percent = (price_range / lower_price / Decimal(suggested_grid_count)) * Decimal("100")

        # Amount per grid
        amount_per_grid = investment / Decimal(suggested_grid_count)

        # Expected profit per trade (approximately grid_spacing minus fees)
        # Assuming 0.1% fee per trade (maker/taker)
        fee_percent = Decimal("0.2")  # Round trip fee
        expected_profit_per_trade = grid_spacing_percent - fee_percent

        return {
            "volatility_level": atr_data.volatility_level,
            "volatility_level_en": atr_data.volatility_level_en,
            "suggested_risk_level": risk_level.value,
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

    @classmethod
    def create_grid_setup(
        cls,
        config: GridConfig,
        atr_data: ATRData,
        current_price: Optional[Decimal] = None,
    ) -> GridSetup:
        """
        Create complete grid setup from config and ATR data.

        Args:
            config: Grid configuration
            atr_data: ATR calculation result
            current_price: Current price (default: from ATR data)

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
            # Calculate from ATR
            atr_multiplier = config.risk_level.atr_multiplier
            half_range = atr_data.value * atr_multiplier

            upper_price = current_price + half_range
            lower_price = current_price - half_range

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

        # Calculate grid spacing
        price_range = upper_price - lower_price
        grid_spacing = price_range / Decimal(grid_count)
        grid_spacing_percent = (grid_spacing / lower_price) * Decimal("100")

        # Amount per grid
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
        )

    @classmethod
    def _create_levels(
        cls,
        upper_price: Decimal,
        lower_price: Decimal,
        current_price: Decimal,
        grid_count: int,
        amount_per_grid: Decimal,
        grid_type: str,
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

        if grid_type == "geometric":
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
