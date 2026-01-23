"""
Backtest Configuration Module.

Provides a unified configuration model for backtesting parameters.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional


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
    """

    initial_capital: Decimal = field(default_factory=lambda: Decimal("10000"))
    fee_rate: Decimal = field(default_factory=lambda: Decimal("0.0004"))  # 0.04%
    slippage_pct: Decimal = field(default_factory=lambda: Decimal("0"))
    leverage: int = 1
    position_size_pct: Decimal = field(default_factory=lambda: Decimal("0.1"))  # 10%
    max_positions: int = 1
    allow_pyramiding: bool = False
    use_margin: bool = False

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
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
        if self.max_positions < 1:
            raise ValueError("max_positions must be at least 1")

    @property
    def notional_per_trade(self) -> Decimal:
        """Calculate notional value per trade."""
        return self.initial_capital * self.position_size_pct

    def with_leverage(self, leverage: int) -> "BacktestConfig":
        """Return a new config with updated leverage."""
        return BacktestConfig(
            initial_capital=self.initial_capital,
            fee_rate=self.fee_rate,
            slippage_pct=self.slippage_pct,
            leverage=leverage,
            position_size_pct=self.position_size_pct,
            max_positions=self.max_positions,
            allow_pyramiding=self.allow_pyramiding,
            use_margin=True if leverage > 1 else self.use_margin,
        )

    def with_fee_rate(self, fee_rate: Decimal) -> "BacktestConfig":
        """Return a new config with updated fee rate."""
        return BacktestConfig(
            initial_capital=self.initial_capital,
            fee_rate=fee_rate,
            slippage_pct=self.slippage_pct,
            leverage=self.leverage,
            position_size_pct=self.position_size_pct,
            max_positions=self.max_positions,
            allow_pyramiding=self.allow_pyramiding,
            use_margin=self.use_margin,
        )
