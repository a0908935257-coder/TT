"""
Data Validators.

Provides validation logic for market data integrity and sanity checks.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class ValidationSeverity(str, Enum):
    """Validation issue severity levels."""
    ERROR = "error"      # Data is invalid, must be rejected
    WARNING = "warning"  # Data is suspicious, may need attention
    INFO = "info"        # Minor issue, informational only


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: List[str] = field(default_factory=list)
    corrected_data: Optional[Any] = None  # Auto-corrected data if applicable

    def add_issue(self, message: str, severity: ValidationSeverity) -> None:
        """Add a validation issue."""
        if severity == ValidationSeverity.ERROR:
            self.errors.append(message)
            self.is_valid = False
        elif severity == ValidationSeverity.WARNING:
            self.warnings.append(message)
        else:
            self.info.append(message)

    def merge(self, other: "ValidationResult") -> "ValidationResult":
        """Merge another validation result into this one."""
        self.is_valid = self.is_valid and other.is_valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        return self

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [f"[{status}]"]
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        if self.warnings:
            parts.append(f"Warnings: {len(self.warnings)}")
        return " | ".join(parts)


class KlineValidator:
    """
    Validates K-line (candlestick) data.

    Checks:
    - OHLC relationship: high >= low, high >= open/close, low <= open/close
    - Volume non-negative
    - Timestamp validity
    - Price range sanity
    - Gap detection
    """

    def __init__(
        self,
        max_price_change_pct: float = 50.0,  # Max % change between candles
        max_wick_ratio: float = 10.0,  # Max wick to body ratio
        min_price: Decimal = Decimal("0.00000001"),
        max_price: Decimal = Decimal("10000000"),
    ):
        """
        Initialize validator.

        Args:
            max_price_change_pct: Maximum allowed price change percentage
            max_wick_ratio: Maximum wick to body ratio (detects manipulation)
            min_price: Minimum valid price
            max_price: Maximum valid price
        """
        self.max_price_change_pct = max_price_change_pct
        self.max_wick_ratio = max_wick_ratio
        self.min_price = min_price
        self.max_price = max_price

    def validate(
        self,
        open_price: Decimal,
        high: Decimal,
        low: Decimal,
        close: Decimal,
        volume: Decimal,
        timestamp: datetime,
        prev_close: Optional[Decimal] = None,
    ) -> ValidationResult:
        """
        Validate a single K-line.

        Args:
            open_price: Open price
            high: High price
            low: Low price
            close: Close price
            volume: Trading volume
            timestamp: Candle timestamp
            prev_close: Previous candle close (for gap detection)

        Returns:
            ValidationResult with validation status
        """
        result = ValidationResult(is_valid=True)

        # Check for None/NaN values
        for name, value in [("open", open_price), ("high", high),
                           ("low", low), ("close", close), ("volume", volume)]:
            if value is None:
                result.add_issue(f"{name} is None", ValidationSeverity.ERROR)
                return result

        # OHLC relationship checks
        if high < low:
            result.add_issue(
                f"Invalid OHLC: high ({high}) < low ({low})",
                ValidationSeverity.ERROR
            )

        if high < open_price:
            result.add_issue(
                f"Invalid OHLC: high ({high}) < open ({open_price})",
                ValidationSeverity.ERROR
            )

        if high < close:
            result.add_issue(
                f"Invalid OHLC: high ({high}) < close ({close})",
                ValidationSeverity.ERROR
            )

        if low > open_price:
            result.add_issue(
                f"Invalid OHLC: low ({low}) > open ({open_price})",
                ValidationSeverity.ERROR
            )

        if low > close:
            result.add_issue(
                f"Invalid OHLC: low ({low}) > close ({close})",
                ValidationSeverity.ERROR
            )

        # Volume check
        if volume < Decimal("0"):
            result.add_issue(
                f"Negative volume: {volume}",
                ValidationSeverity.ERROR
            )

        # Price range check
        for name, price in [("open", open_price), ("high", high),
                           ("low", low), ("close", close)]:
            if price < self.min_price:
                result.add_issue(
                    f"{name} price ({price}) below minimum ({self.min_price})",
                    ValidationSeverity.ERROR
                )
            if price > self.max_price:
                result.add_issue(
                    f"{name} price ({price}) above maximum ({self.max_price})",
                    ValidationSeverity.ERROR
                )

        # Wick ratio check (detect potential manipulation)
        body = abs(close - open_price)
        if body > Decimal("0"):
            upper_wick = high - max(open_price, close)
            lower_wick = min(open_price, close) - low
            total_wick = upper_wick + lower_wick

            if total_wick / body > Decimal(str(self.max_wick_ratio)):
                result.add_issue(
                    f"Suspicious wick ratio: {float(total_wick / body):.2f}",
                    ValidationSeverity.WARNING
                )

        # Price change check (against previous close)
        if prev_close and prev_close > Decimal("0"):
            price_change_pct = abs(
                (close - prev_close) / prev_close * Decimal("100")
            )
            if price_change_pct > Decimal(str(self.max_price_change_pct)):
                result.add_issue(
                    f"Large price change: {float(price_change_pct):.2f}%",
                    ValidationSeverity.WARNING
                )

        # Timestamp check
        if timestamp > datetime.now(timezone.utc) + timedelta(minutes=5):
            result.add_issue(
                f"Future timestamp: {timestamp}",
                ValidationSeverity.WARNING
            )

        return result

    def validate_series(
        self,
        klines: List[Dict[str, Any]],
        interval_minutes: int = 15,
    ) -> Tuple[ValidationResult, List[int]]:
        """
        Validate a series of K-lines.

        Args:
            klines: List of kline dictionaries with OHLCV data
            interval_minutes: Expected interval between candles

        Returns:
            Tuple of (overall result, list of invalid indices)
        """
        result = ValidationResult(is_valid=True)
        invalid_indices: List[int] = []
        prev_close = None
        prev_time = None

        for i, kline in enumerate(klines):
            # Validate individual candle
            candle_result = self.validate(
                open_price=Decimal(str(kline.get("open", 0))),
                high=Decimal(str(kline.get("high", 0))),
                low=Decimal(str(kline.get("low", 0))),
                close=Decimal(str(kline.get("close", 0))),
                volume=Decimal(str(kline.get("volume", 0))),
                timestamp=kline.get("timestamp", datetime.now(timezone.utc)),
                prev_close=prev_close,
            )

            if not candle_result.is_valid:
                invalid_indices.append(i)
                result.merge(candle_result)

            # Check for gaps
            if prev_time:
                expected_time = prev_time + timedelta(minutes=interval_minutes)
                actual_time = kline.get("timestamp")
                if actual_time and actual_time > expected_time + timedelta(minutes=1):
                    gap_minutes = (actual_time - expected_time).total_seconds() / 60
                    result.add_issue(
                        f"Gap detected at index {i}: {gap_minutes:.0f} minutes",
                        ValidationSeverity.WARNING
                    )

            # Check for duplicates
            if prev_time and kline.get("timestamp") == prev_time:
                result.add_issue(
                    f"Duplicate timestamp at index {i}",
                    ValidationSeverity.WARNING
                )

            prev_close = Decimal(str(kline.get("close", 0)))
            prev_time = kline.get("timestamp")

        return result, invalid_indices


class PriceValidator:
    """
    Validates price data.

    Checks:
    - Price positivity
    - Price range
    - Sudden spikes/drops
    - Stale prices
    """

    def __init__(
        self,
        min_price: Decimal = Decimal("0.00000001"),
        max_price: Decimal = Decimal("10000000"),
        max_change_pct: float = 20.0,
        stale_threshold_seconds: int = 60,
    ):
        """
        Initialize validator.

        Args:
            min_price: Minimum valid price
            max_price: Maximum valid price
            max_change_pct: Maximum allowed instant change
            stale_threshold_seconds: Seconds before price considered stale
        """
        self.min_price = min_price
        self.max_price = max_price
        self.max_change_pct = max_change_pct
        self.stale_threshold_seconds = stale_threshold_seconds
        self._last_prices: Dict[str, Tuple[Decimal, datetime]] = {}

    def validate(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> ValidationResult:
        """
        Validate a price.

        Args:
            symbol: Trading symbol
            price: Price value
            timestamp: Price timestamp

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)
        now = timestamp or datetime.now(timezone.utc)

        # None check
        if price is None:
            result.add_issue("Price is None", ValidationSeverity.ERROR)
            return result

        # Positivity check
        if price <= Decimal("0"):
            result.add_issue(
                f"Non-positive price: {price}",
                ValidationSeverity.ERROR
            )
            return result

        # Range check
        if price < self.min_price:
            result.add_issue(
                f"Price {price} below minimum {self.min_price}",
                ValidationSeverity.ERROR
            )

        if price > self.max_price:
            result.add_issue(
                f"Price {price} above maximum {self.max_price}",
                ValidationSeverity.ERROR
            )

        # Sudden change check
        if symbol in self._last_prices:
            last_price, last_time = self._last_prices[symbol]
            if last_price > Decimal("0"):
                change_pct = abs(
                    (price - last_price) / last_price * Decimal("100")
                )
                if change_pct > Decimal(str(self.max_change_pct)):
                    result.add_issue(
                        f"Sudden price change: {float(change_pct):.2f}% "
                        f"({last_price} -> {price})",
                        ValidationSeverity.WARNING
                    )

        # Update last price
        self._last_prices[symbol] = (price, now)

        return result

    def is_stale(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
    ) -> bool:
        """
        Check if price data is stale.

        Args:
            symbol: Trading symbol
            current_time: Current time for comparison

        Returns:
            True if price is stale
        """
        if symbol not in self._last_prices:
            return True

        _, last_time = self._last_prices[symbol]
        now = current_time or datetime.now(timezone.utc)
        age = (now - last_time).total_seconds()

        return age > self.stale_threshold_seconds


class OrderBookValidator:
    """
    Validates order book (depth) data.

    Checks:
    - Bid/ask ordering (bids descending, asks ascending)
    - Spread sanity (bid < ask)
    - Price/quantity positivity
    - Depth consistency
    """

    def __init__(
        self,
        max_spread_pct: float = 5.0,
        min_levels: int = 1,
    ):
        """
        Initialize validator.

        Args:
            max_spread_pct: Maximum allowed spread percentage
            min_levels: Minimum required depth levels
        """
        self.max_spread_pct = max_spread_pct
        self.min_levels = min_levels

    def validate(
        self,
        bids: List[Tuple[Decimal, Decimal]],  # (price, quantity)
        asks: List[Tuple[Decimal, Decimal]],
    ) -> ValidationResult:
        """
        Validate order book.

        Args:
            bids: List of (price, quantity) tuples, descending by price
            asks: List of (price, quantity) tuples, ascending by price

        Returns:
            ValidationResult
        """
        result = ValidationResult(is_valid=True)

        # Empty check
        if not bids:
            result.add_issue("No bids in order book", ValidationSeverity.ERROR)
        if not asks:
            result.add_issue("No asks in order book", ValidationSeverity.ERROR)

        if not bids or not asks:
            return result

        # Minimum levels check
        if len(bids) < self.min_levels:
            result.add_issue(
                f"Insufficient bid levels: {len(bids)} < {self.min_levels}",
                ValidationSeverity.WARNING
            )
        if len(asks) < self.min_levels:
            result.add_issue(
                f"Insufficient ask levels: {len(asks)} < {self.min_levels}",
                ValidationSeverity.WARNING
            )

        # Best bid/ask
        best_bid = bids[0][0]
        best_ask = asks[0][0]

        # Spread check (bid should be less than ask)
        if best_bid >= best_ask:
            result.add_issue(
                f"Invalid spread: bid ({best_bid}) >= ask ({best_ask})",
                ValidationSeverity.ERROR
            )

        # Spread percentage check
        if best_bid > Decimal("0"):
            spread_pct = (best_ask - best_bid) / best_bid * Decimal("100")
            if spread_pct > Decimal(str(self.max_spread_pct)):
                result.add_issue(
                    f"Wide spread: {float(spread_pct):.2f}%",
                    ValidationSeverity.WARNING
                )

        # Bid ordering check (should be descending)
        for i in range(1, len(bids)):
            if bids[i][0] > bids[i-1][0]:
                result.add_issue(
                    f"Bids not descending at level {i}",
                    ValidationSeverity.ERROR
                )
                break

        # Ask ordering check (should be ascending)
        for i in range(1, len(asks)):
            if asks[i][0] < asks[i-1][0]:
                result.add_issue(
                    f"Asks not ascending at level {i}",
                    ValidationSeverity.ERROR
                )
                break

        # Price/quantity positivity
        for i, (price, qty) in enumerate(bids):
            if price <= Decimal("0") or qty <= Decimal("0"):
                result.add_issue(
                    f"Invalid bid at level {i}: price={price}, qty={qty}",
                    ValidationSeverity.ERROR
                )

        for i, (price, qty) in enumerate(asks):
            if price <= Decimal("0") or qty <= Decimal("0"):
                result.add_issue(
                    f"Invalid ask at level {i}: price={price}, qty={qty}",
                    ValidationSeverity.ERROR
                )

        return result


class DataQualityChecker:
    """
    Comprehensive data quality checker.

    Aggregates validation across multiple data types and provides
    quality metrics.
    """

    def __init__(self):
        """Initialize checker with default validators."""
        self.kline_validator = KlineValidator()
        self.price_validator = PriceValidator()
        self.orderbook_validator = OrderBookValidator()

        # Quality metrics
        self._total_checks = 0
        self._passed_checks = 0
        self._warning_count = 0
        self._error_count = 0

    def check_kline(
        self,
        kline: Dict[str, Any],
        prev_close: Optional[Decimal] = None,
    ) -> ValidationResult:
        """Check a single K-line."""
        self._total_checks += 1

        result = self.kline_validator.validate(
            open_price=Decimal(str(kline.get("open", 0))),
            high=Decimal(str(kline.get("high", 0))),
            low=Decimal(str(kline.get("low", 0))),
            close=Decimal(str(kline.get("close", 0))),
            volume=Decimal(str(kline.get("volume", 0))),
            timestamp=kline.get("timestamp", datetime.now(timezone.utc)),
            prev_close=prev_close,
        )

        self._update_metrics(result)
        return result

    def check_price(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> ValidationResult:
        """Check a price value."""
        self._total_checks += 1

        result = self.price_validator.validate(symbol, price, timestamp)

        self._update_metrics(result)
        return result

    def check_orderbook(
        self,
        bids: List[Tuple[Decimal, Decimal]],
        asks: List[Tuple[Decimal, Decimal]],
    ) -> ValidationResult:
        """Check order book data."""
        self._total_checks += 1

        result = self.orderbook_validator.validate(bids, asks)

        self._update_metrics(result)
        return result

    def _update_metrics(self, result: ValidationResult) -> None:
        """Update quality metrics."""
        if result.is_valid:
            self._passed_checks += 1
        self._warning_count += len(result.warnings)
        self._error_count += len(result.errors)

    def get_quality_report(self) -> Dict[str, Any]:
        """
        Get quality metrics report.

        Returns:
            Dictionary with quality statistics
        """
        pass_rate = (
            self._passed_checks / self._total_checks * 100
            if self._total_checks > 0 else 0
        )

        return {
            "total_checks": self._total_checks,
            "passed_checks": self._passed_checks,
            "failed_checks": self._total_checks - self._passed_checks,
            "pass_rate_pct": round(pass_rate, 2),
            "total_warnings": self._warning_count,
            "total_errors": self._error_count,
        }

    def reset_metrics(self) -> None:
        """Reset quality metrics."""
        self._total_checks = 0
        self._passed_checks = 0
        self._warning_count = 0
        self._error_count = 0
