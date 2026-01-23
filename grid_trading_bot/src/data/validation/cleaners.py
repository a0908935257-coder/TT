"""
Data Cleaners.

Provides data cleaning and correction utilities for market data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class CleaningStats:
    """Statistics from a cleaning operation."""
    total_records: int = 0
    cleaned_records: int = 0
    removed_records: int = 0
    filled_gaps: int = 0
    corrected_values: int = 0
    duplicates_removed: int = 0

    def __str__(self) -> str:
        return (
            f"Total: {self.total_records}, Cleaned: {self.cleaned_records}, "
            f"Removed: {self.removed_records}, Gaps filled: {self.filled_gaps}, "
            f"Corrected: {self.corrected_values}, Duplicates: {self.duplicates_removed}"
        )


class KlineCleaner:
    """
    Cleans K-line data.

    Features:
    - OHLC correction (swap high/low if inverted)
    - Negative value handling
    - Gap filling with interpolation
    - Duplicate removal
    - Outlier smoothing
    """

    def __init__(
        self,
        fill_gaps: bool = True,
        remove_outliers: bool = True,
        outlier_std_threshold: float = 3.0,
        max_gap_fill_count: int = 5,
    ):
        """
        Initialize cleaner.

        Args:
            fill_gaps: Whether to fill missing candles
            remove_outliers: Whether to smooth outliers
            outlier_std_threshold: Standard deviations for outlier detection
            max_gap_fill_count: Maximum consecutive gaps to fill
        """
        self.fill_gaps = fill_gaps
        self.remove_outliers = remove_outliers
        self.outlier_std_threshold = outlier_std_threshold
        self.max_gap_fill_count = max_gap_fill_count

    def clean(
        self,
        klines: List[Dict[str, Any]],
        interval_minutes: int = 15,
    ) -> Tuple[List[Dict[str, Any]], CleaningStats]:
        """
        Clean a series of K-lines.

        Args:
            klines: List of kline dictionaries
            interval_minutes: Expected interval between candles

        Returns:
            Tuple of (cleaned klines, cleaning stats)
        """
        if not klines:
            return [], CleaningStats()

        stats = CleaningStats(total_records=len(klines))

        # Step 1: Remove duplicates
        klines, dup_count = self._remove_duplicates(klines)
        stats.duplicates_removed = dup_count

        # Step 2: Correct OHLC values
        klines, corrected = self._correct_ohlc(klines)
        stats.corrected_values = corrected

        # Step 3: Handle negative/zero values
        klines, removed = self._handle_invalid_values(klines)
        stats.removed_records = removed

        # Step 4: Fill gaps
        if self.fill_gaps:
            klines, gaps_filled = self._fill_gaps(klines, interval_minutes)
            stats.filled_gaps = gaps_filled

        # Step 5: Smooth outliers
        if self.remove_outliers:
            klines, smoothed = self._smooth_outliers(klines)
            stats.corrected_values += smoothed

        stats.cleaned_records = len(klines)

        logger.info(f"Kline cleaning complete: {stats}")
        return klines, stats

    def _remove_duplicates(
        self,
        klines: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Remove duplicate timestamps, keeping the last occurrence."""
        seen_times: Dict[datetime, int] = {}
        duplicates = 0

        for i, kline in enumerate(klines):
            ts = kline.get("timestamp") or kline.get("open_time")
            if ts in seen_times:
                duplicates += 1
            seen_times[ts] = i

        # Keep only the last occurrence of each timestamp
        unique_klines = [klines[i] for i in sorted(seen_times.values())]
        return unique_klines, duplicates

    def _correct_ohlc(
        self,
        klines: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Correct OHLC relationships."""
        corrected = 0

        for kline in klines:
            o = Decimal(str(kline.get("open", 0)))
            h = Decimal(str(kline.get("high", 0)))
            l = Decimal(str(kline.get("low", 0)))
            c = Decimal(str(kline.get("close", 0)))

            modified = False

            # Swap high/low if inverted
            if h < l:
                kline["high"], kline["low"] = str(l), str(h)
                h, l = l, h
                modified = True

            # Ensure high >= open, close
            max_oc = max(o, c)
            if h < max_oc:
                kline["high"] = str(max_oc)
                modified = True

            # Ensure low <= open, close
            min_oc = min(o, c)
            if l > min_oc:
                kline["low"] = str(min_oc)
                modified = True

            if modified:
                corrected += 1

        return klines, corrected

    def _handle_invalid_values(
        self,
        klines: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Remove klines with invalid (negative/zero/None) values."""
        valid_klines = []
        removed = 0

        for kline in klines:
            try:
                o = Decimal(str(kline.get("open", 0)))
                h = Decimal(str(kline.get("high", 0)))
                l = Decimal(str(kline.get("low", 0)))
                c = Decimal(str(kline.get("close", 0)))
                v = Decimal(str(kline.get("volume", 0)))

                # Check for positive prices
                if o <= 0 or h <= 0 or l <= 0 or c <= 0:
                    removed += 1
                    continue

                # Volume can be zero but not negative
                if v < 0:
                    kline["volume"] = "0"

                valid_klines.append(kline)
            except Exception:
                removed += 1

        return valid_klines, removed

    def _fill_gaps(
        self,
        klines: List[Dict[str, Any]],
        interval_minutes: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Fill missing candles with interpolation."""
        if len(klines) < 2:
            return klines, 0

        filled_klines = []
        gaps_filled = 0
        interval = timedelta(minutes=interval_minutes)

        for i, kline in enumerate(klines):
            filled_klines.append(kline)

            if i == len(klines) - 1:
                break

            current_time = kline.get("timestamp") or kline.get("open_time")
            next_time = klines[i + 1].get("timestamp") or klines[i + 1].get("open_time")

            if not current_time or not next_time:
                continue

            # Calculate gap
            expected_next = current_time + interval
            gap_count = 0

            while expected_next < next_time and gap_count < self.max_gap_fill_count:
                # Create interpolated candle
                current_close = Decimal(str(kline.get("close", 0)))
                next_open = Decimal(str(klines[i + 1].get("open", 0)))

                # Linear interpolation
                progress = Decimal(str((gap_count + 1) / (gap_count + 2)))
                interp_price = current_close + (next_open - current_close) * progress

                filled_candle = {
                    "timestamp": expected_next,
                    "open_time": expected_next,
                    "open": str(interp_price),
                    "high": str(interp_price),
                    "low": str(interp_price),
                    "close": str(interp_price),
                    "volume": "0",
                    "is_interpolated": True,
                }

                filled_klines.append(filled_candle)
                gaps_filled += 1
                gap_count += 1
                expected_next += interval

        return filled_klines, gaps_filled

    def _smooth_outliers(
        self,
        klines: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Smooth outliers using rolling statistics."""
        if len(klines) < 10:
            return klines, 0

        smoothed = 0
        window_size = 10

        # Calculate rolling mean and std for close prices
        closes = [Decimal(str(k.get("close", 0))) for k in klines]

        for i in range(window_size, len(klines)):
            window = closes[i - window_size:i]
            mean = sum(window) / len(window)
            variance = sum((x - mean) ** 2 for x in window) / len(window)
            std = variance.sqrt() if variance > 0 else Decimal("0")

            current = closes[i]
            threshold = Decimal(str(self.outlier_std_threshold))

            if std > 0 and abs(current - mean) > threshold * std:
                # Replace with mean
                klines[i]["close"] = str(mean)
                klines[i]["high"] = str(max(mean, Decimal(str(klines[i]["high"]))))
                klines[i]["low"] = str(min(mean, Decimal(str(klines[i]["low"]))))
                klines[i]["is_smoothed"] = True
                closes[i] = mean
                smoothed += 1

        return klines, smoothed


class PriceCleaner:
    """
    Cleans price data.

    Features:
    - Spike removal
    - Missing value handling
    - Price normalization
    """

    def __init__(
        self,
        max_spike_pct: float = 20.0,
        use_median_filter: bool = True,
        median_window: int = 5,
    ):
        """
        Initialize cleaner.

        Args:
            max_spike_pct: Maximum allowed percentage change
            use_median_filter: Apply median filter for spikes
            median_window: Window size for median filter
        """
        self.max_spike_pct = max_spike_pct
        self.use_median_filter = use_median_filter
        self.median_window = median_window

    def clean_series(
        self,
        prices: List[Tuple[datetime, Decimal]],
    ) -> Tuple[List[Tuple[datetime, Decimal]], CleaningStats]:
        """
        Clean a price series.

        Args:
            prices: List of (timestamp, price) tuples

        Returns:
            Tuple of (cleaned prices, stats)
        """
        if not prices:
            return [], CleaningStats()

        stats = CleaningStats(total_records=len(prices))

        # Remove None/negative values
        valid_prices = [
            (ts, p) for ts, p in prices
            if p is not None and p > Decimal("0")
        ]
        stats.removed_records = len(prices) - len(valid_prices)

        # Remove spikes with median filter
        if self.use_median_filter and len(valid_prices) > self.median_window:
            valid_prices, spike_count = self._remove_spikes(valid_prices)
            stats.corrected_values = spike_count

        stats.cleaned_records = len(valid_prices)
        return valid_prices, stats

    def _remove_spikes(
        self,
        prices: List[Tuple[datetime, Decimal]],
    ) -> Tuple[List[Tuple[datetime, Decimal]], int]:
        """Remove price spikes using median filter."""
        cleaned = []
        spike_count = 0
        window = self.median_window

        for i, (ts, price) in enumerate(prices):
            if i < window:
                cleaned.append((ts, price))
                continue

            # Get window prices
            window_prices = sorted([p for _, p in prices[i - window:i]])
            median = window_prices[len(window_prices) // 2]

            # Check if current price is a spike
            if median > Decimal("0"):
                change_pct = abs((price - median) / median * Decimal("100"))
                if change_pct > Decimal(str(self.max_spike_pct)):
                    # Replace with median
                    cleaned.append((ts, median))
                    spike_count += 1
                    continue

            cleaned.append((ts, price))

        return cleaned, spike_count


class DataCleaner:
    """
    Unified data cleaner facade.

    Provides a single interface for cleaning different data types.
    """

    def __init__(self):
        """Initialize with default cleaners."""
        self.kline_cleaner = KlineCleaner()
        self.price_cleaner = PriceCleaner()

    def clean_klines(
        self,
        klines: List[Dict[str, Any]],
        interval_minutes: int = 15,
    ) -> Tuple[List[Dict[str, Any]], CleaningStats]:
        """Clean K-line data."""
        return self.kline_cleaner.clean(klines, interval_minutes)

    def clean_prices(
        self,
        prices: List[Tuple[datetime, Decimal]],
    ) -> Tuple[List[Tuple[datetime, Decimal]], CleaningStats]:
        """Clean price series."""
        return self.price_cleaner.clean_series(prices)

    def validate_and_clean_kline(
        self,
        kline: Dict[str, Any],
        prev_close: Optional[Decimal] = None,
    ) -> Tuple[Optional[Dict[str, Any]], bool]:
        """
        Validate and clean a single K-line.

        Args:
            kline: K-line data dictionary
            prev_close: Previous close price

        Returns:
            Tuple of (cleaned kline or None, is_valid)
        """
        from .validators import KlineValidator

        validator = KlineValidator()

        # First validate
        result = validator.validate(
            open_price=Decimal(str(kline.get("open", 0))),
            high=Decimal(str(kline.get("high", 0))),
            low=Decimal(str(kline.get("low", 0))),
            close=Decimal(str(kline.get("close", 0))),
            volume=Decimal(str(kline.get("volume", 0))),
            timestamp=kline.get("timestamp", datetime.now(timezone.utc)),
            prev_close=prev_close,
        )

        if result.is_valid:
            return kline, True

        # Try to fix correctable issues
        if result.errors:
            # Check if OHLC can be corrected
            cleaned, stats = self.kline_cleaner._correct_ohlc([kline.copy()])
            if stats > 0:
                # Re-validate
                fixed_kline = cleaned[0]
                result2 = validator.validate(
                    open_price=Decimal(str(fixed_kline.get("open", 0))),
                    high=Decimal(str(fixed_kline.get("high", 0))),
                    low=Decimal(str(fixed_kline.get("low", 0))),
                    close=Decimal(str(fixed_kline.get("close", 0))),
                    volume=Decimal(str(fixed_kline.get("volume", 0))),
                    timestamp=fixed_kline.get("timestamp", datetime.now(timezone.utc)),
                    prev_close=prev_close,
                )
                if result2.is_valid:
                    return fixed_kline, True

        # Could not fix
        return None, False
