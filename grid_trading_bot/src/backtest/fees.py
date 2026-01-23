"""
Fee Calculation Module.

Provides various fee calculation models for realistic trading cost simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional


class FeeModelType(str, Enum):
    """Fee model type enumeration."""

    FIXED = "fixed"
    MAKER_TAKER = "maker_taker"
    TIERED = "tiered"


@dataclass
class FeeContext:
    """
    Context for fee calculation.

    Attributes:
        is_maker: True if order is maker (limit order providing liquidity)
        cumulative_volume: Cumulative 30-day trading volume for tiered fees
        vip_level: VIP level for tiered fee structures
    """

    is_maker: bool = False
    cumulative_volume: Decimal = Decimal("0")
    vip_level: int = 0


class FeeCalculator(ABC):
    """
    Abstract base class for fee calculators.

    Fee calculators compute trading fees based on various models
    including fixed rates, maker/taker differentiation, and volume tiers.
    """

    @abstractmethod
    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        context: Optional[FeeContext] = None,
    ) -> Decimal:
        """
        Calculate trading fee.

        Args:
            price: Fill price
            quantity: Order quantity
            context: Additional context for calculation

        Returns:
            Fee amount
        """
        pass

    @property
    @abstractmethod
    def base_rate(self) -> Decimal:
        """Get the base fee rate for reference."""
        pass


class FixedFeeCalculator(FeeCalculator):
    """
    Fixed rate fee calculator.

    Applies a constant fee rate regardless of order type or volume.
    This is the default behavior.

    Example:
        calc = FixedFeeCalculator(Decimal("0.0004"))  # 0.04%
        fee = calc.calculate_fee(price, quantity)
    """

    def __init__(self, fee_rate: Decimal = Decimal("0.0004")) -> None:
        """
        Initialize fixed fee calculator.

        Args:
            fee_rate: Fee rate (0.0004 = 0.04%)
        """
        if fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        self._fee_rate = fee_rate

    @property
    def base_rate(self) -> Decimal:
        """Get fee rate."""
        return self._fee_rate

    @property
    def fee_rate(self) -> Decimal:
        """Get fee rate (alias for base_rate)."""
        return self._fee_rate

    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        context: Optional[FeeContext] = None,
    ) -> Decimal:
        """Calculate fixed rate fee."""
        return price * quantity * self._fee_rate


class MakerTakerFeeCalculator(FeeCalculator):
    """
    Maker/Taker differentiated fee calculator.

    Applies different fee rates for maker orders (providing liquidity)
    and taker orders (removing liquidity).

    Typical rates:
    - Maker: 0.02% (limit orders that rest on the book)
    - Taker: 0.04% (market orders or aggressive limit orders)

    Example:
        calc = MakerTakerFeeCalculator(
            maker_rate=Decimal("0.0002"),  # 0.02%
            taker_rate=Decimal("0.0004"),  # 0.04%
        )
        fee = calc.calculate_fee(price, quantity, FeeContext(is_maker=True))
    """

    def __init__(
        self,
        maker_rate: Decimal = Decimal("0.0002"),
        taker_rate: Decimal = Decimal("0.0004"),
    ) -> None:
        """
        Initialize maker/taker fee calculator.

        Args:
            maker_rate: Fee rate for maker orders
            taker_rate: Fee rate for taker orders
        """
        if maker_rate < 0:
            raise ValueError("maker_rate cannot be negative")
        if taker_rate < 0:
            raise ValueError("taker_rate cannot be negative")

        self._maker_rate = maker_rate
        self._taker_rate = taker_rate

    @property
    def base_rate(self) -> Decimal:
        """Get taker rate as base reference."""
        return self._taker_rate

    @property
    def maker_rate(self) -> Decimal:
        """Get maker fee rate."""
        return self._maker_rate

    @property
    def taker_rate(self) -> Decimal:
        """Get taker fee rate."""
        return self._taker_rate

    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        context: Optional[FeeContext] = None,
    ) -> Decimal:
        """Calculate maker/taker differentiated fee."""
        if context and context.is_maker:
            rate = self._maker_rate
        else:
            rate = self._taker_rate

        return price * quantity * rate


@dataclass
class FeeTier:
    """
    Fee tier definition.

    Attributes:
        min_volume: Minimum volume threshold for this tier
        maker_rate: Maker fee rate for this tier
        taker_rate: Taker fee rate for this tier
    """

    min_volume: Decimal
    maker_rate: Decimal
    taker_rate: Decimal


class TieredFeeCalculator(FeeCalculator):
    """
    Volume-based tiered fee calculator.

    Applies fee rates based on cumulative trading volume,
    with lower rates for higher volume traders.

    Default tiers (based on typical exchange structures):
    - Tier 0 (< $1M): 0.02% maker, 0.04% taker
    - Tier 1 (< $5M): 0.016% maker, 0.035% taker
    - Tier 2 (< $10M): 0.014% maker, 0.03% taker
    - Tier 3 (< $50M): 0.012% maker, 0.025% taker
    - Tier 4 (>= $50M): 0.01% maker, 0.02% taker

    Example:
        calc = TieredFeeCalculator()  # Use default tiers
        fee = calc.calculate_fee(
            price, quantity,
            FeeContext(cumulative_volume=Decimal("2000000"))
        )
    """

    DEFAULT_TIERS = [
        FeeTier(
            min_volume=Decimal("0"),
            maker_rate=Decimal("0.0002"),
            taker_rate=Decimal("0.0004"),
        ),
        FeeTier(
            min_volume=Decimal("1000000"),
            maker_rate=Decimal("0.00016"),
            taker_rate=Decimal("0.00035"),
        ),
        FeeTier(
            min_volume=Decimal("5000000"),
            maker_rate=Decimal("0.00014"),
            taker_rate=Decimal("0.0003"),
        ),
        FeeTier(
            min_volume=Decimal("10000000"),
            maker_rate=Decimal("0.00012"),
            taker_rate=Decimal("0.00025"),
        ),
        FeeTier(
            min_volume=Decimal("50000000"),
            maker_rate=Decimal("0.0001"),
            taker_rate=Decimal("0.0002"),
        ),
    ]

    def __init__(self, tiers: Optional[list[FeeTier]] = None) -> None:
        """
        Initialize tiered fee calculator.

        Args:
            tiers: List of fee tiers (sorted by min_volume ascending).
                   Uses DEFAULT_TIERS if not provided.
        """
        self._tiers = tiers if tiers is not None else self.DEFAULT_TIERS.copy()

        # Validate tiers are sorted
        for i in range(1, len(self._tiers)):
            if self._tiers[i].min_volume <= self._tiers[i - 1].min_volume:
                raise ValueError("Tiers must be sorted by min_volume ascending")

    @property
    def base_rate(self) -> Decimal:
        """Get base tier taker rate."""
        return self._tiers[0].taker_rate if self._tiers else Decimal("0.0004")

    @property
    def tiers(self) -> list[FeeTier]:
        """Get fee tiers."""
        return self._tiers.copy()

    def get_tier(self, cumulative_volume: Decimal) -> FeeTier:
        """
        Get the applicable fee tier for a given volume.

        Args:
            cumulative_volume: Cumulative 30-day trading volume

        Returns:
            Applicable FeeTier
        """
        applicable_tier = self._tiers[0]
        for tier in self._tiers:
            if cumulative_volume >= tier.min_volume:
                applicable_tier = tier
            else:
                break
        return applicable_tier

    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        context: Optional[FeeContext] = None,
    ) -> Decimal:
        """Calculate tiered fee based on volume."""
        volume = context.cumulative_volume if context else Decimal("0")
        tier = self.get_tier(volume)

        if context and context.is_maker:
            rate = tier.maker_rate
        else:
            rate = tier.taker_rate

        return price * quantity * rate


def create_fee_calculator(
    model_type: FeeModelType,
    fee_rate: Decimal = Decimal("0.0004"),
    maker_rate: Decimal = Decimal("0.0002"),
    taker_rate: Decimal = Decimal("0.0004"),
    tiers: Optional[list[FeeTier]] = None,
) -> FeeCalculator:
    """
    Factory function to create fee calculators.

    Args:
        model_type: Type of fee model
        fee_rate: Fixed fee rate (for FIXED model)
        maker_rate: Maker fee rate (for MAKER_TAKER model)
        taker_rate: Taker fee rate (for MAKER_TAKER model)
        tiers: Fee tiers (for TIERED model)

    Returns:
        FeeCalculator instance
    """
    if model_type == FeeModelType.FIXED:
        return FixedFeeCalculator(fee_rate)
    elif model_type == FeeModelType.MAKER_TAKER:
        return MakerTakerFeeCalculator(maker_rate, taker_rate)
    elif model_type == FeeModelType.TIERED:
        return TieredFeeCalculator(tiers)
    else:
        raise ValueError(f"Unknown fee model type: {model_type}")
