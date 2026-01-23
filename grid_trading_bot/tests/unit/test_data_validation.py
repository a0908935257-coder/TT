"""
Data Validation Unit Tests.

Tests for data cleaning, validation, and anomaly detection.
"""

from datetime import datetime, timezone, timedelta
from decimal import Decimal

import pytest

from src.data.validation.validators import (
    KlineValidator,
    PriceValidator,
    OrderBookValidator,
    DataQualityChecker,
    ValidationResult,
    ValidationSeverity,
)
from src.data.validation.cleaners import (
    KlineCleaner,
    PriceCleaner,
    DataCleaner,
    CleaningStats,
)
from src.data.validation.anomaly import (
    AnomalyDetector,
    AnomalyType,
    AnomalySeverity,
    AnomalyRecord,
)


class TestKlineValidator:
    """Test KlineValidator."""

    def test_valid_kline(self):
        """Test validation of a valid K-line."""
        validator = KlineValidator()
        result = validator.validate(
            open_price=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
        )
        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_high_low(self):
        """Test detection of high < low."""
        validator = KlineValidator()
        result = validator.validate(
            open_price=Decimal("100"),
            high=Decimal("90"),  # Invalid: high < low
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
        )
        assert not result.is_valid
        assert any("high" in e.lower() and "low" in e.lower() for e in result.errors)

    def test_invalid_high_less_than_close(self):
        """Test detection of high < close."""
        validator = KlineValidator()
        result = validator.validate(
            open_price=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("95"),
            close=Decimal("105"),  # Invalid: close > high
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
        )
        assert not result.is_valid

    def test_negative_volume(self):
        """Test detection of negative volume."""
        validator = KlineValidator()
        result = validator.validate(
            open_price=Decimal("100"),
            high=Decimal("110"),
            low=Decimal("95"),
            close=Decimal("105"),
            volume=Decimal("-100"),  # Invalid
            timestamp=datetime.now(timezone.utc),
        )
        assert not result.is_valid
        assert any("volume" in e.lower() for e in result.errors)

    def test_price_change_warning(self):
        """Test warning for large price changes."""
        validator = KlineValidator(max_price_change_pct=10.0)
        result = validator.validate(
            open_price=Decimal("100"),
            high=Decimal("150"),
            low=Decimal("95"),
            close=Decimal("145"),
            volume=Decimal("1000"),
            timestamp=datetime.now(timezone.utc),
            prev_close=Decimal("100"),  # 45% change
        )
        assert result.is_valid  # Still valid, just warnings
        assert len(result.warnings) > 0

    def test_validate_series_with_gaps(self):
        """Test series validation with gap detection."""
        validator = KlineValidator()
        now = datetime.now(timezone.utc)

        klines = [
            {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
             "timestamp": now},
            {"open": 102, "high": 108, "low": 100, "close": 106, "volume": 1200,
             "timestamp": now + timedelta(minutes=15)},
            # Gap here - missing one candle
            {"open": 106, "high": 112, "low": 104, "close": 110, "volume": 1100,
             "timestamp": now + timedelta(minutes=45)},
        ]

        result, invalid_indices = validator.validate_series(klines, interval_minutes=15)
        assert len(result.warnings) > 0  # Should detect gap


class TestPriceValidator:
    """Test PriceValidator."""

    def test_valid_price(self):
        """Test validation of a valid price."""
        validator = PriceValidator()
        result = validator.validate("BTCUSDT", Decimal("50000"))
        assert result.is_valid

    def test_negative_price(self):
        """Test detection of negative price."""
        validator = PriceValidator()
        result = validator.validate("BTCUSDT", Decimal("-100"))
        assert not result.is_valid

    def test_zero_price(self):
        """Test detection of zero price."""
        validator = PriceValidator()
        result = validator.validate("BTCUSDT", Decimal("0"))
        assert not result.is_valid

    def test_price_spike_warning(self):
        """Test warning for sudden price spike."""
        validator = PriceValidator(max_change_pct=10.0)

        # First price
        validator.validate("BTCUSDT", Decimal("50000"))

        # Second price with 25% jump
        result = validator.validate("BTCUSDT", Decimal("62500"))
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_stale_price_detection(self):
        """Test stale price detection."""
        validator = PriceValidator(stale_threshold_seconds=5)

        validator.validate(
            "BTCUSDT",
            Decimal("50000"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=10)
        )

        assert validator.is_stale("BTCUSDT")


class TestOrderBookValidator:
    """Test OrderBookValidator."""

    def test_valid_orderbook(self):
        """Test validation of valid order book."""
        validator = OrderBookValidator()
        result = validator.validate(
            bids=[
                (Decimal("49900"), Decimal("1.5")),
                (Decimal("49800"), Decimal("2.0")),
            ],
            asks=[
                (Decimal("50000"), Decimal("1.0")),
                (Decimal("50100"), Decimal("1.5")),
            ],
        )
        assert result.is_valid

    def test_invalid_spread(self):
        """Test detection of bid >= ask."""
        validator = OrderBookValidator()
        result = validator.validate(
            bids=[(Decimal("50100"), Decimal("1.5"))],
            asks=[(Decimal("50000"), Decimal("1.0"))],
        )
        assert not result.is_valid
        assert any("spread" in e.lower() for e in result.errors)

    def test_wide_spread_warning(self):
        """Test warning for wide spread."""
        validator = OrderBookValidator(max_spread_pct=1.0)
        result = validator.validate(
            bids=[(Decimal("49000"), Decimal("1.5"))],
            asks=[(Decimal("50000"), Decimal("1.0"))],  # ~2% spread
        )
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_bid_ordering(self):
        """Test detection of incorrect bid ordering."""
        validator = OrderBookValidator()
        result = validator.validate(
            bids=[
                (Decimal("49800"), Decimal("1.5")),
                (Decimal("49900"), Decimal("2.0")),  # Wrong order
            ],
            asks=[(Decimal("50000"), Decimal("1.0"))],
        )
        assert not result.is_valid


class TestKlineCleaner:
    """Test KlineCleaner."""

    def test_remove_duplicates(self):
        """Test duplicate removal."""
        cleaner = KlineCleaner()
        now = datetime.now(timezone.utc)

        klines = [
            {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
             "timestamp": now},
            {"open": 101, "high": 106, "low": 96, "close": 103, "volume": 1100,
             "timestamp": now},  # Duplicate timestamp
            {"open": 103, "high": 108, "low": 100, "close": 106, "volume": 1200,
             "timestamp": now + timedelta(minutes=15)},
        ]

        cleaned, stats = cleaner.clean(klines)
        assert stats.duplicates_removed == 1
        assert len(cleaned) == 2

    def test_correct_ohlc(self):
        """Test OHLC correction."""
        cleaner = KlineCleaner()

        klines = [
            {"open": 100, "high": 90, "low": 95, "close": 102, "volume": 1000,
             "timestamp": datetime.now(timezone.utc)},  # high < low (inverted)
        ]

        cleaned, stats = cleaner.clean(klines)
        assert stats.corrected_values > 0
        assert Decimal(str(cleaned[0]["high"])) >= Decimal(str(cleaned[0]["low"]))

    def test_fill_gaps(self):
        """Test gap filling."""
        cleaner = KlineCleaner(fill_gaps=True)
        now = datetime.now(timezone.utc)

        klines = [
            {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
             "timestamp": now},
            # Missing one candle
            {"open": 106, "high": 112, "low": 104, "close": 110, "volume": 1100,
             "timestamp": now + timedelta(minutes=30)},
        ]

        cleaned, stats = cleaner.clean(klines, interval_minutes=15)
        assert stats.filled_gaps == 1
        assert len(cleaned) == 3

    def test_remove_invalid_values(self):
        """Test removal of invalid values."""
        cleaner = KlineCleaner()

        klines = [
            {"open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
             "timestamp": datetime.now(timezone.utc)},
            {"open": -100, "high": 105, "low": 95, "close": 102, "volume": 1000,
             "timestamp": datetime.now(timezone.utc) + timedelta(minutes=15)},  # Negative
        ]

        cleaned, stats = cleaner.clean(klines)
        assert stats.removed_records == 1
        assert len(cleaned) == 1


class TestPriceCleaner:
    """Test PriceCleaner."""

    def test_remove_spikes(self):
        """Test spike removal."""
        cleaner = PriceCleaner(max_spike_pct=10.0)

        prices = [
            (datetime.now(timezone.utc) + timedelta(seconds=i), Decimal(str(100 + i % 5)))
            for i in range(20)
        ]
        # Add a spike
        prices.append((
            datetime.now(timezone.utc) + timedelta(seconds=20),
            Decimal("200")  # 100% spike
        ))

        cleaned, stats = cleaner.clean_series(prices)
        assert stats.corrected_values > 0


class TestAnomalyDetector:
    """Test AnomalyDetector."""

    def test_price_spike_detection(self):
        """Test price spike detection."""
        detector = AnomalyDetector(price_spike_pct=5.0)

        # Build history
        for i in range(20):
            detector.check_price("BTCUSDT", Decimal("50000"))

        # Trigger spike
        anomalies = detector.check_price("BTCUSDT", Decimal("55000"))  # 10% spike
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.PRICE_SPIKE for a in anomalies)

    def test_price_drop_detection(self):
        """Test price drop detection."""
        detector = AnomalyDetector(price_drop_pct=5.0)

        # Build history
        for i in range(20):
            detector.check_price("BTCUSDT", Decimal("50000"))

        # Trigger drop
        anomalies = detector.check_price("BTCUSDT", Decimal("45000"))  # 10% drop
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.PRICE_DROP for a in anomalies)

    def test_volume_spike_detection(self):
        """Test volume spike detection."""
        detector = AnomalyDetector(volume_spike_std=2.0)

        # Build history with normal volume (need slight variation for std > 0)
        for i in range(30):
            detector.check_volume("BTCUSDT", Decimal(str(1000 + (i % 10))))

        # Trigger spike (much larger than normal range)
        anomalies = detector.check_volume("BTCUSDT", Decimal("50000"))  # 50x normal
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.VOLUME_SPIKE for a in anomalies)

    def test_spread_detection(self):
        """Test spread anomaly detection."""
        detector = AnomalyDetector(spread_warning_pct=1.0, spread_critical_pct=3.0)

        # Wide spread
        anomalies = detector.check_spread(
            "BTCUSDT",
            bid=Decimal("49000"),
            ask=Decimal("51000"),  # ~4% spread
        )
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.SPREAD_WIDE for a in anomalies)

    def test_data_gap_detection(self):
        """Test data gap detection."""
        detector = AnomalyDetector(gap_threshold_multiplier=2.0)
        now = datetime.now(timezone.utc)

        anomalies = detector.check_data_gap(
            "BTCUSDT",
            current_time=now,
            last_time=now - timedelta(minutes=60),  # 60 min gap
            expected_interval=timedelta(minutes=15),
        )
        assert len(anomalies) > 0
        assert any(a.anomaly_type == AnomalyType.DATA_GAP for a in anomalies)

    def test_stale_data_detection(self):
        """Test stale data detection."""
        detector = AnomalyDetector(stale_threshold_seconds=30)

        # First check
        detector.check_price(
            "BTCUSDT",
            Decimal("50000"),
            timestamp=datetime.now(timezone.utc) - timedelta(seconds=60)
        )

        # Second check - should detect stale
        anomalies = detector.check_price(
            "BTCUSDT",
            Decimal("50000"),
            timestamp=datetime.now(timezone.utc)
        )
        assert any(a.anomaly_type == AnomalyType.DATA_STALE for a in anomalies)

    def test_stats_tracking(self):
        """Test statistics tracking."""
        detector = AnomalyDetector(price_spike_pct=5.0)

        # Build history and trigger anomaly
        for i in range(20):
            detector.check_price("BTCUSDT", Decimal("50000"))
        detector.check_price("BTCUSDT", Decimal("60000"))

        stats = detector.get_stats()
        assert stats["total_detected"] > 0


class TestDataQualityChecker:
    """Test DataQualityChecker."""

    def test_quality_report(self):
        """Test quality report generation."""
        checker = DataQualityChecker()

        # Run some checks
        checker.check_kline({
            "open": 100, "high": 105, "low": 95, "close": 102, "volume": 1000,
            "timestamp": datetime.now(timezone.utc)
        })
        checker.check_kline({
            "open": 100, "high": 90, "low": 95, "close": 102, "volume": 1000,  # Invalid
            "timestamp": datetime.now(timezone.utc)
        })

        report = checker.get_quality_report()
        assert report["total_checks"] == 2
        assert report["passed_checks"] == 1
        assert report["failed_checks"] == 1
        assert report["pass_rate_pct"] == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
