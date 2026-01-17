"""
Grid Trading Exceptions.

Provides custom exceptions for grid trading strategy.
"""


class GridError(Exception):
    """Base exception for grid trading errors."""

    pass


class GridCalculationError(GridError):
    """Raised when grid calculation fails."""

    def __init__(self, message: str, details: dict | None = None):
        self.details = details or {}
        super().__init__(message)


class InsufficientDataError(GridError):
    """Raised when there is insufficient K-line data for calculation."""

    def __init__(self, required: int, actual: int):
        self.required = required
        self.actual = actual
        super().__init__(
            f"Insufficient data: need at least {required} klines, got {actual}"
        )


class InsufficientFundError(GridError):
    """Raised when investment is insufficient for the grid configuration."""

    def __init__(
        self,
        required: str,
        actual: str,
        min_per_grid: str | None = None,
        grid_count: int | None = None,
    ):
        self.required = required
        self.actual = actual
        self.min_per_grid = min_per_grid
        self.grid_count = grid_count

        message = f"Insufficient funds: need at least {required}, got {actual}"
        if min_per_grid and grid_count:
            message += f" (min {min_per_grid} × {grid_count} grids)"
        super().__init__(message)


class InvalidPriceRangeError(GridError):
    """Raised when price range is invalid."""

    def __init__(self, upper: str, lower: str, reason: str = ""):
        self.upper = upper
        self.lower = lower
        message = f"Invalid price range: upper={upper}, lower={lower}"
        if reason:
            message += f" ({reason})"
        super().__init__(message)


class GridConfigurationError(GridError):
    """Raised when grid configuration is invalid."""

    def __init__(self, field: str, value: str, reason: str):
        self.field = field
        self.value = value
        self.reason = reason
        super().__init__(f"Invalid configuration '{field}={value}': {reason}")
