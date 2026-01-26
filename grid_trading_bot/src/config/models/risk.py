"""
Risk Configuration Model.

Provides configuration for risk management parameters.
"""

from decimal import Decimal
from typing import Optional

from pydantic import Field, field_validator

from .base import BaseConfig


class RiskConfig(BaseConfig):
    """
    Risk management configuration.

    Example:
        >>> config = RiskConfig(
        ...     max_drawdown=Decimal("20.0"),
        ...     daily_loss_limit=Decimal("5.0"),
        ...     position_limit=5,
        ... )
    """

    # Drawdown and loss limits (percentages)
    max_drawdown: Decimal = Field(
        default=Decimal("20.0"),
        ge=Decimal("1.0"),
        le=Decimal("100.0"),
        description="Maximum drawdown percentage before stopping",
    )
    daily_loss_limit: Decimal = Field(
        default=Decimal("5.0"),
        ge=Decimal("0.1"),
        le=Decimal("50.0"),
        description="Daily loss limit percentage",
    )
    weekly_loss_limit: Optional[Decimal] = Field(
        default=Decimal("15.0"),
        ge=Decimal("1.0"),
        le=Decimal("100.0"),
        description="Weekly loss limit percentage",
    )

    # Position limits
    position_limit: int = Field(
        default=5,
        ge=1,
        le=100,
        description="Maximum number of concurrent positions",
    )
    max_position_size: Decimal = Field(
        default=Decimal("10.0"),
        ge=Decimal("0.1"),
        le=Decimal("100.0"),
        description="Max position size as percentage of balance",
    )

    # Leverage limits
    max_leverage: int = Field(
        default=10,
        ge=1,
        le=125,
        description="Maximum allowed leverage",
    )

    # Warning thresholds
    liquidation_warning: Decimal = Field(
        default=Decimal("80.0"),
        ge=Decimal("50.0"),
        le=Decimal("99.0"),
        description="Warn when position is this % close to liquidation",
    )
    margin_warning: Decimal = Field(
        default=Decimal("50.0"),
        ge=Decimal("10.0"),
        le=Decimal("90.0"),
        description="Warn when margin ratio drops below this %",
    )

    # Order limits
    max_order_value: Optional[Decimal] = Field(
        default=None,
        ge=Decimal("0"),
        description="Maximum single order value in quote currency",
    )
    min_order_interval: int = Field(
        default=1,
        ge=0,
        description="Minimum seconds between orders",
    )

    # Circuit breaker
    circuit_breaker_enabled: bool = Field(
        default=True,
        description="Enable circuit breaker on excessive losses",
    )
    circuit_breaker_cooldown: int = Field(
        default=3600,
        ge=60,
        description="Circuit breaker cooldown in seconds",
    )

    @field_validator("max_drawdown", "daily_loss_limit", "weekly_loss_limit", mode="before")
    @classmethod
    def coerce_to_decimal(cls, v):
        """Coerce numeric values to Decimal with validation."""
        if v is None:
            return v
        if isinstance(v, (int, float, str)):
            try:
                decimal_val = Decimal(str(v))
                # Reject NaN and Infinity
                if not decimal_val.is_finite():
                    raise ValueError(f"Invalid value: {v} (NaN or Infinity not allowed)")
                return decimal_val
            except Exception as e:
                raise ValueError(f"Cannot convert '{v}' to Decimal: {e}")
        return v

    @field_validator("max_position_size", "liquidation_warning", "margin_warning", mode="before")
    @classmethod
    def coerce_percentage_to_decimal(cls, v):
        """Coerce percentage values to Decimal with validation."""
        if v is None:
            return v
        if isinstance(v, (int, float, str)):
            try:
                decimal_val = Decimal(str(v))
                # Reject NaN and Infinity
                if not decimal_val.is_finite():
                    raise ValueError(f"Invalid value: {v} (NaN or Infinity not allowed)")
                return decimal_val
            except Exception as e:
                raise ValueError(f"Cannot convert '{v}' to Decimal: {e}")
        return v

    def is_drawdown_exceeded(self, current_drawdown: Decimal) -> bool:
        """
        Check if drawdown limit is exceeded.

        Args:
            current_drawdown: Current drawdown percentage

        Returns:
            True if exceeded
        """
        return current_drawdown >= self.max_drawdown

    def is_daily_loss_exceeded(self, daily_loss: Decimal) -> bool:
        """
        Check if daily loss limit is exceeded.

        Args:
            daily_loss: Current daily loss percentage

        Returns:
            True if exceeded
        """
        return daily_loss >= self.daily_loss_limit

    def is_position_limit_reached(self, current_positions: int) -> bool:
        """
        Check if position limit is reached.

        Args:
            current_positions: Number of current positions

        Returns:
            True if limit reached
        """
        return current_positions >= self.position_limit

    def should_warn_liquidation(self, distance_to_liquidation: Decimal) -> bool:
        """
        Check if liquidation warning should be triggered.

        Args:
            distance_to_liquidation: Percentage distance to liquidation price

        Returns:
            True if warning should be triggered
        """
        threshold = Decimal("100") - self.liquidation_warning
        return distance_to_liquidation <= threshold
