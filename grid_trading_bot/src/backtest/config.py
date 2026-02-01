"""
Backtest Configuration Module.

Provides a unified configuration model for backtesting parameters.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .fees import FeeCalculator
    from .slippage import SlippageModel


class SlippageModelType(str, Enum):
    """Slippage model type enumeration."""

    FIXED = "fixed"
    VOLATILITY = "volatility"
    MARKET_IMPACT = "market_impact"


class FeeModelType(str, Enum):
    """Fee model type enumeration."""

    FIXED = "fixed"
    MAKER_TAKER = "maker_taker"
    TIERED = "tiered"


class IntraBarSequence(str, Enum):
    """Intra-bar price sequence for order simulation."""

    OHLC = "ohlc"  # Open -> High -> Low -> Close
    OLHC = "olhc"  # Open -> Low -> High -> Close
    WORST_CASE = "worst_case"  # Always assume worst execution


@dataclass
class BacktestConfig:
    """
    Configuration for backtest execution.

    Attributes:
        initial_capital: Starting capital for the backtest
        fee_rate: Trading fee rate (default 0.04% for futures)
        slippage_pct: Slippage percentage for price impact
        leverage: Leverage multiplier (1 for spot)
        position_size_pct: Percentage of capital per trade
        max_positions: Maximum concurrent positions (1 for single-position strategies)
        allow_pyramiding: Whether to allow adding to existing positions
        use_margin: Whether to use margin accounting (True for futures)

        # Fee model settings (new)
        fee_model_type: Type of fee model (FIXED, MAKER_TAKER, TIERED)
        maker_fee_rate: Fee rate for maker orders (limit orders)
        taker_fee_rate: Fee rate for taker orders (market orders)

        # Slippage model settings (new)
        slippage_model_type: Type of slippage model (FIXED, VOLATILITY, MARKET_IMPACT)
        slippage_atr_multiplier: ATR multiplier for volatility-based slippage
        slippage_atr_period: ATR period for volatility calculation
        slippage_impact_coefficient: Coefficient for market impact slippage

        # Market microstructure settings (new, disabled by default)
        enable_spread_simulation: Whether to simulate bid/ask spread
        spread_pct: Bid/ask spread percentage
        intra_bar_sequence: Intra-bar price sequence assumption

        # Order execution settings (new, disabled by default)
        enable_limit_orders: Whether to allow limit order simulation
        enable_partial_fills: Whether to simulate partial fills
    """

    # Core settings
    initial_capital: Decimal = field(default_factory=lambda: Decimal("10000"))
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))  # 0.04%
    slippage_pct: Decimal = field(default_factory=lambda: Decimal("0.0003"))  # 0.03% realistic slippage
    leverage: int = 1
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.02"))  # 2% of equity
    max_positions: int = 1
    allow_pyramiding: bool = False
    use_margin: bool = False

    # Position sizing limits
    max_notional: Decimal = field(default_factory=lambda: Decimal("0"))  # 0 = unlimited
    compound_sizing: bool = False  # False = fixed sizing based on initial_capital

    # Margin / liquidation settings (futures)
    maintenance_margin_pct: Decimal = field(default_factory=lambda: Decimal("0.004"))  # 0.4%
    liquidation_fee_pct: Decimal = field(default_factory=lambda: Decimal("0.006"))  # 0.6%

    # Funding rate settings (futures)
    funding_rate: Decimal = field(default_factory=lambda: Decimal("0.0003"))  # 0.03% per 8h (bull market avg)
    funding_interval_hours: int = 8

    # Fee model settings
    fee_model_type: FeeModelType = FeeModelType.MAKER_TAKER
    maker_fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0002"))  # 0.02%
    taker_fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))  # 0.04%

    # Slippage model settings
    slippage_model_type: SlippageModelType = SlippageModelType.FIXED
    slippage_atr_multiplier: Decimal = field(default_factory=lambda: Decimal("0.1"))
    slippage_atr_period: int = 14
    slippage_impact_coefficient: Decimal = field(default_factory=lambda: Decimal("0.1"))

    # Market microstructure settings (disabled by default)
    enable_spread_simulation: bool = False
    spread_pct: Decimal = field(default_factory=lambda: Decimal("0.0001"))  # 0.01%
    intra_bar_sequence: IntraBarSequence = IntraBarSequence.OHLC

    # Order execution settings (disabled by default)
    enable_limit_orders: bool = False
    enable_partial_fills: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Core validations
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if self.fee_rate < 0:
            raise ValueError("fee_rate cannot be negative")
        if self.slippage_pct < 0:
            raise ValueError("slippage_pct cannot be negative")
        if self.leverage < 1:
            raise ValueError("leverage must be at least 1")
        if not (Decimal("0") < self.position_size_pct <= Decimal("1")):
            raise ValueError("position_size_pct must be between 0 and 1")
        if self.max_notional < 0:
            raise ValueError("max_notional cannot be negative")
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")

        # Fee model validations
        if self.maker_fee_rate < 0:
            raise ValueError("maker_fee_rate cannot be negative")
        if self.taker_fee_rate < 0:
            raise ValueError("taker_fee_rate cannot be negative")

        # Slippage model validations
        if self.slippage_atr_multiplier < 0:
            raise ValueError("slippage_atr_multiplier cannot be negative")
        if self.slippage_atr_period < 1:
            raise ValueError("slippage_atr_period must be at least 1")
        if self.slippage_impact_coefficient < 0:
            raise ValueError("slippage_impact_coefficient cannot be negative")

        # Microstructure validations
        if self.spread_pct < 0:
            raise ValueError("spread_pct cannot be negative")

    @property
    def notional_per_trade(self) -> Decimal:
        """Calculate notional value per trade."""
        return self.initial_capital * self.position_size_pct

    def with_leverage(self, leverage: int) -> "BacktestConfig":
        """Return a new config with updated leverage."""
        from dataclasses import replace

        return replace(
            self,
            leverage=leverage,
            use_margin=True if leverage > 1 else self.use_margin,
        )

    def with_max_positions(self, max_positions: int) -> "BacktestConfig":
        """Return a new config with updated max_positions."""
        from dataclasses import replace

        return replace(self, max_positions=max_positions)

    def with_fee_rate(self, fee_rate: Decimal) -> "BacktestConfig":
        """Return a new config with updated fee rate."""
        from dataclasses import replace

        return replace(self, fee_rate=fee_rate)

    def create_slippage_model(self) -> "SlippageModel":
        """
        Create a slippage model based on configuration.

        Returns:
            SlippageModel instance configured according to settings
        """
        from .slippage import (
            FixedSlippage,
            MarketImpactSlippage,
            VolatilityBasedSlippage,
        )

        if self.slippage_model_type == SlippageModelType.FIXED:
            return FixedSlippage(self.slippage_pct)
        elif self.slippage_model_type == SlippageModelType.VOLATILITY:
            return VolatilityBasedSlippage(
                base_pct=self.slippage_pct,
                atr_multiplier=self.slippage_atr_multiplier,
                atr_period=self.slippage_atr_period,
            )
        elif self.slippage_model_type == SlippageModelType.MARKET_IMPACT:
            return MarketImpactSlippage(
                base_pct=self.slippage_pct,
                impact_coefficient=self.slippage_impact_coefficient,
            )
        else:
            raise ValueError(f"Unknown slippage model type: {self.slippage_model_type}")

    def create_fee_calculator(self) -> "FeeCalculator":
        """
        Create a fee calculator based on configuration.

        Returns:
            FeeCalculator instance configured according to settings
        """
        from .fees import (
            FixedFeeCalculator,
            MakerTakerFeeCalculator,
            TieredFeeCalculator,
        )

        if self.fee_model_type == FeeModelType.FIXED:
            return FixedFeeCalculator(self.fee_rate)
        elif self.fee_model_type == FeeModelType.MAKER_TAKER:
            return MakerTakerFeeCalculator(
                maker_rate=self.maker_fee_rate,
                taker_rate=self.taker_fee_rate,
            )
        elif self.fee_model_type == FeeModelType.TIERED:
            return TieredFeeCalculator()
        else:
            raise ValueError(f"Unknown fee model type: {self.fee_model_type}")
