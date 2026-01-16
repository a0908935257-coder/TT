"""
Custom exceptions for Grid Trading Bot.

Exception hierarchy:
    TradingBotError (base)
    ├── ExchangeError
    │   ├── ConnectionError
    │   ├── AuthenticationError
    │   ├── RateLimitError
    │   ├── InsufficientBalanceError
    │   └── OrderError
    ├── DataError
    │   ├── ValidationError
    │   └── NotFoundError
    ├── ConfigError
    └── StrategyError
"""

from typing import Any


class TradingBotError(Exception):
    """Base exception for all trading bot errors."""

    default_message = "Trading bot error occurred"

    def __init__(
        self,
        message: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        self.message = message or self.default_message
        self.code = code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        parts = [self.message]
        if self.code:
            parts.append(f"[{self.code}]")
        if self.details:
            parts.append(f"Details: {self.details}")
        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"message={self.message!r}, "
            f"code={self.code!r}, "
            f"details={self.details!r})"
        )


# Exchange-related errors
class ExchangeError(TradingBotError):
    """Base exception for exchange-related errors."""

    default_message = "Exchange error occurred"


class ConnectionError(ExchangeError):
    """Connection to exchange failed."""

    default_message = "Failed to connect to exchange"


class AuthenticationError(ExchangeError):
    """Authentication with exchange failed."""

    default_message = "Authentication failed"


class RateLimitError(ExchangeError):
    """Rate limit exceeded on exchange API."""

    default_message = "Rate limit exceeded"

    def __init__(
        self,
        message: str | None = None,
        retry_after: int = 60,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code, details)
        self.retry_after = retry_after

    def __str__(self) -> str:
        base = super().__str__()
        return f"{base} (retry after {self.retry_after}s)"


class InsufficientBalanceError(ExchangeError):
    """Insufficient balance for operation."""

    default_message = "Insufficient balance"


class OrderError(ExchangeError):
    """Order-related error."""

    default_message = "Order error occurred"

    def __init__(
        self,
        message: str | None = None,
        order_id: str | None = None,
        symbol: str | None = None,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ):
        super().__init__(message, code, details)
        self.order_id = order_id
        self.symbol = symbol

    def __str__(self) -> str:
        base = super().__str__()
        parts = [base]
        if self.order_id:
            parts.append(f"order_id={self.order_id}")
        if self.symbol:
            parts.append(f"symbol={self.symbol}")
        return " ".join(parts)


# Data-related errors
class DataError(TradingBotError):
    """Base exception for data-related errors."""

    default_message = "Data error occurred"


class ValidationError(DataError):
    """Data validation failed."""

    default_message = "Validation failed"


class NotFoundError(DataError):
    """Requested data not found."""

    default_message = "Data not found"


# Configuration error
class ConfigError(TradingBotError):
    """Configuration error."""

    default_message = "Configuration error"


# Strategy error
class StrategyError(TradingBotError):
    """Strategy-related error."""

    default_message = "Strategy error occurred"
