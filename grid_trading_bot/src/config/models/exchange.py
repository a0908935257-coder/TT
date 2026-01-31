"""
Exchange Configuration Model.

Provides configuration for exchange connections.
"""

from typing import Literal, Optional

from pydantic import Field, field_validator

from .base import BaseConfig


class ExchangeConfig(BaseConfig):
    """
    Exchange connection configuration.

    Example:
        >>> config = ExchangeConfig(
        ...     name="binance",
        ...     api_key="${BINANCE_API_KEY}",
        ...     api_secret="${BINANCE_API_SECRET}",
        ...     testnet=True,
        ... )
    """

    name: str = Field(
        default="binance",
        description="Exchange name",
    )
    testnet: bool = Field(
        default=False,
        description="Use testnet endpoints",
    )
    api_key: str = Field(
        default="",
        description="API key for authentication",
    )
    api_secret: str = Field(
        default="",
        description="API secret for authentication",
    )
    leverage: int = Field(
        default=1,
        ge=1,
        le=125,
        description="Default leverage for futures trading",
    )
    margin_type: Literal["ISOLATED", "CROSSED"] = Field(
        default="ISOLATED",
        description="Margin type for futures",
    )

    # Optional additional settings
    recv_window: int = Field(
        default=5000,
        ge=1000,
        le=60000,
        description="Receive window in milliseconds",
    )
    rate_limit: int = Field(
        default=1200,
        ge=100,
        description="Rate limit (requests per minute)",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate exchange name."""
        v = v.lower().strip()
        supported = {"binance", "binance_testnet"}
        if v not in supported:
            # Allow any name but warn about unsupported
            pass
        return v

    @field_validator("leverage")
    @classmethod
    def validate_leverage(cls, v: int) -> int:
        """Validate leverage value."""
        if v < 1:
            return 1
        if v > 125:
            return 125
        return v

    @property
    def is_testnet(self) -> bool:
        """Check if using testnet."""
        return self.testnet or "testnet" in self.name.lower()

    @property
    def has_credentials(self) -> bool:
        """Check if API credentials are configured."""
        return bool(self.api_key and self.api_secret)
