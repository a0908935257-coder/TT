"""
Slippage Models Module.

Provides various slippage calculation models for realistic order execution simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..core.models import Kline


class SlippageModelType(str, Enum):
    """Slippage model type enumeration."""

    FIXED = "fixed"
    VOLATILITY = "volatility"
    MARKET_IMPACT = "market_impact"


@dataclass
class SlippageContext:
    """
    Context for slippage calculation.

    Attributes:
        order_size: Size of the order (notional value)
        avg_volume: Average trading volume (for market impact)
        atr: Average True Range (for volatility-based)
        recent_klines: Recent klines for volatility calculation
    """

    order_size: Decimal = Decimal("0")
    avg_volume: Optional[Decimal] = None
    atr: Optional[Decimal] = None
    recent_klines: Optional[list[Kline]] = None


class SlippageModel(ABC):
    """
    Abstract base class for slippage models.

    Slippage models calculate the price impact of order execution,
    accounting for factors like volatility, order size, and liquidity.
    """

    @abstractmethod
    def calculate_slippage(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        context: Optional[SlippageContext] = None,
    ) -> Decimal:
        """
        Calculate slippage amount.

        Args:
            target_price: Target execution price
            is_buy: True if buy order, False if sell
            kline: Current kline data
            context: Additional context for calculation

        Returns:
            Slippage amount (positive value)
        """
        pass

    def apply_slippage(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        context: Optional[SlippageContext] = None,
    ) -> Decimal:
        """
        Apply slippage to target price.

        Args:
            target_price: Target execution price
            is_buy: True if buy order, False if sell
            kline: Current kline data
            context: Additional context for calculation

        Returns:
            Adjusted fill price
        """
        slippage = self.calculate_slippage(target_price, is_buy, kline, context)

        if is_buy:
            # Buy at higher price (worse for buyer)
            fill_price = target_price + slippage
            # Cannot fill above kline high
            fill_price = min(fill_price, kline.high)
        else:
            # Sell at lower price (worse for seller)
            fill_price = target_price - slippage
            # Cannot fill below kline low
            fill_price = max(fill_price, kline.low)

        return fill_price


class FixedSlippage(SlippageModel):
    """
    Fixed percentage slippage model.

    Simple model that applies a constant percentage slippage
    regardless of market conditions. This is the default behavior.

    Example:
        model = FixedSlippage(Decimal("0.0005"))  # 0.05%
        fill_price = model.apply_slippage(price, is_buy=True, kline)
    """

    def __init__(self, slippage_pct: Decimal = Decimal("0")) -> None:
        """
        Initialize fixed slippage model.

        Args:
            slippage_pct: Slippage percentage (0.001 = 0.1%)
        """
        if slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        self._slippage_pct = slippage_pct

    @property
    def slippage_pct(self) -> Decimal:
        """Get slippage percentage."""
        return self._slippage_pct

    def calculate_slippage(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        context: Optional[SlippageContext] = None,
    ) -> Decimal:
        """Calculate fixed percentage slippage."""
        return target_price * self._slippage_pct


class VolatilityBasedSlippage(SlippageModel):
    """
    Volatility-based slippage model using ATR.

    Calculates slippage based on market volatility measured by
    Average True Range (ATR). Higher volatility leads to higher slippage.

    Formula: slippage = base_pct + (ATR / price) * multiplier

    Example:
        model = VolatilityBasedSlippage(
            base_pct=Decimal("0.0001"),     # 0.01% base
            atr_multiplier=Decimal("0.1"),  # 10% of ATR ratio
            atr_period=14
        )
    """

    def __init__(
        self,
        base_pct: Decimal = Decimal("0.0001"),
        atr_multiplier: Decimal = Decimal("0.1"),
        atr_period: int = 14,
    ) -> None:
        """
        Initialize volatility-based slippage model.

        Args:
            base_pct: Base slippage percentage
            atr_multiplier: Multiplier for ATR-based component
            atr_period: Period for ATR calculation
        """
        if base_pct < 0:
            raise ValueError("base_pct cannot be negative")
        if atr_multiplier < 0:
            raise ValueError("atr_multiplier cannot be negative")
        if atr_period < 1:
            raise ValueError("atr_period must be at least 1")

        self._base_pct = base_pct
        self._atr_multiplier = atr_multiplier
        self._atr_period = atr_period

    @property
    def base_pct(self) -> Decimal:
        """Get base slippage percentage."""
        return self._base_pct

    @property
    def atr_multiplier(self) -> Decimal:
        """Get ATR multiplier."""
        return self._atr_multiplier

    @property
    def atr_period(self) -> int:
        """Get ATR calculation period."""
        return self._atr_period

    def calculate_atr(self, klines: list[Kline]) -> Decimal:
        """
        Calculate Average True Range from klines.

        Args:
            klines: List of klines for ATR calculation

        Returns:
            ATR value
        """
        if len(klines) < 2:
            return Decimal("0")

        true_ranges = []
        for i in range(1, len(klines)):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            true_ranges.append(tr)

        if not true_ranges:
            return Decimal("0")

        return sum(true_ranges) / Decimal(len(true_ranges))

    def calculate_slippage(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        context: Optional[SlippageContext] = None,
    ) -> Decimal:
        """Calculate volatility-based slippage."""
        # Base slippage
        base_slippage = target_price * self._base_pct

        # ATR component
        atr_slippage = Decimal("0")
        if context and context.atr is not None and context.atr > 0:
            atr_ratio = context.atr / target_price
            atr_slippage = target_price * atr_ratio * self._atr_multiplier
        elif context and context.recent_klines:
            # Calculate ATR from recent klines
            atr = self.calculate_atr(context.recent_klines[-self._atr_period - 1 :])
            if atr > 0:
                atr_ratio = atr / target_price
                atr_slippage = target_price * atr_ratio * self._atr_multiplier

        return base_slippage + atr_slippage


class MarketImpactSlippage(SlippageModel):
    """
    Market impact slippage model based on order size.

    Calculates slippage based on order size relative to average volume.
    Larger orders relative to volume cause more market impact.

    Formula: slippage = base_pct + coefficient * sqrt(order_size / avg_volume)

    Example:
        model = MarketImpactSlippage(
            base_pct=Decimal("0.0001"),        # 0.01% base
            impact_coefficient=Decimal("0.1"), # Impact scaling
        )
    """

    def __init__(
        self,
        base_pct: Decimal = Decimal("0.0001"),
        impact_coefficient: Decimal = Decimal("0.1"),
    ) -> None:
        """
        Initialize market impact slippage model.

        Args:
            base_pct: Base slippage percentage
            impact_coefficient: Coefficient for sqrt market impact
        """
        if base_pct < 0:
            raise ValueError("base_pct cannot be negative")
        if impact_coefficient < 0:
            raise ValueError("impact_coefficient cannot be negative")

        self._base_pct = base_pct
        self._impact_coefficient = impact_coefficient

    @property
    def base_pct(self) -> Decimal:
        """Get base slippage percentage."""
        return self._base_pct

    @property
    def impact_coefficient(self) -> Decimal:
        """Get impact coefficient."""
        return self._impact_coefficient

    def calculate_slippage(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        context: Optional[SlippageContext] = None,
    ) -> Decimal:
        """Calculate market impact slippage."""
        # Base slippage
        base_slippage = target_price * self._base_pct

        # Market impact component
        impact_slippage = Decimal("0")
        if context and context.order_size > 0 and context.avg_volume:
            if context.avg_volume > 0:
                # sqrt(order_size / avg_volume) * coefficient
                size_ratio = float(context.order_size / context.avg_volume)
                impact_factor = Decimal(str(size_ratio**0.5))
                impact_slippage = target_price * impact_factor * self._impact_coefficient

        return base_slippage + impact_slippage


def create_slippage_model(
    model_type: SlippageModelType,
    slippage_pct: Decimal = Decimal("0"),
    atr_multiplier: Decimal = Decimal("0.1"),
    atr_period: int = 14,
    impact_coefficient: Decimal = Decimal("0.1"),
) -> SlippageModel:
    """
    Factory function to create slippage models.

    Args:
        model_type: Type of slippage model
        slippage_pct: Base slippage percentage
        atr_multiplier: ATR multiplier for volatility model
        atr_period: ATR period for volatility model
        impact_coefficient: Impact coefficient for market impact model

    Returns:
        SlippageModel instance
    """
    if model_type == SlippageModelType.FIXED:
        return FixedSlippage(slippage_pct)
    elif model_type == SlippageModelType.VOLATILITY:
        return VolatilityBasedSlippage(
            base_pct=slippage_pct,
            atr_multiplier=atr_multiplier,
            atr_period=atr_period,
        )
    elif model_type == SlippageModelType.MARKET_IMPACT:
        return MarketImpactSlippage(
            base_pct=slippage_pct,
            impact_coefficient=impact_coefficient,
        )
    else:
        raise ValueError(f"Unknown slippage model type: {model_type}")
