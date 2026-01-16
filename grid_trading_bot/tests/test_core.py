"""
Tests for core module: logger, exceptions, and utils.
"""

import sys
import time
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.logger import setup_logger, get_logger, Colors
from core.exceptions import (
    TradingBotError,
    ExchangeError,
    ConnectionError,
    AuthenticationError,
    RateLimitError,
    InsufficientBalanceError,
    OrderError,
    DataError,
    ValidationError,
    NotFoundError,
    ConfigError,
    StrategyError,
)
from core.utils import (
    timestamp_to_datetime,
    datetime_to_timestamp,
    now_timestamp,
    round_decimal,
    round_to_tick,
    format_quantity,
    format_price,
    calculate_percentage,
    generate_client_order_id,
    mask_secret,
    retry,
)


# =============================================================================
# Logger Tests
# =============================================================================


class TestLogger:
    """Test cases for logger module."""

    def test_setup_logger_returns_logger(self):
        """Test that setup_logger returns a logger instance."""
        logger = setup_logger("test.logger1")
        assert logger is not None
        assert logger.name == "test.logger1"

    def test_logger_has_handlers(self):
        """Test that logger has both console and file handlers."""
        logger = setup_logger("test.logger2")
        # Should have 2 handlers: console and file
        assert len(logger.handlers) == 2

    def test_get_logger_returns_same_logger(self):
        """Test that get_logger returns existing logger."""
        logger1 = setup_logger("test.logger3")
        logger2 = get_logger("test.logger3")
        assert logger1 is logger2

    def test_logger_outputs_to_terminal(self, capsys):
        """Test that logger outputs colored messages to terminal."""
        logger = setup_logger("test.terminal")
        logger.info("Test info message")
        # Note: capsys may not capture colored output properly
        # This is a basic test to ensure no errors occur

    def test_logger_creates_log_file(self, tmp_path):
        """Test that logger creates log file."""
        log_file = tmp_path / "test.log"
        logger = setup_logger("test.file", log_file=log_file)
        logger.info("Test message")
        assert log_file.exists()

    def test_no_duplicate_handlers(self):
        """Test that calling setup_logger twice doesn't add duplicate handlers."""
        logger1 = setup_logger("test.duplicate")
        handler_count = len(logger1.handlers)
        logger2 = setup_logger("test.duplicate")
        assert len(logger2.handlers) == handler_count


# =============================================================================
# Exceptions Tests
# =============================================================================


class TestExceptions:
    """Test cases for custom exceptions."""

    def test_trading_bot_error_basic(self):
        """Test basic TradingBotError."""
        error = TradingBotError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"

    def test_trading_bot_error_with_code(self):
        """Test TradingBotError with error code."""
        error = TradingBotError("Test error", code="E001")
        assert "E001" in str(error)
        assert error.code == "E001"

    def test_trading_bot_error_with_details(self):
        """Test TradingBotError with details."""
        error = TradingBotError("Test error", details={"key": "value"})
        assert error.details == {"key": "value"}

    def test_rate_limit_error_retry_after(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Too many requests", retry_after=30)
        assert error.retry_after == 30
        assert "30" in str(error)

    def test_order_error_with_order_details(self):
        """Test OrderError with order_id and symbol."""
        error = OrderError("Order failed", order_id="12345", symbol="BTCUSDT")
        assert error.order_id == "12345"
        assert error.symbol == "BTCUSDT"
        assert "12345" in str(error)
        assert "BTCUSDT" in str(error)

    def test_exception_hierarchy(self):
        """Test exception inheritance hierarchy."""
        # ExchangeError inherits from TradingBotError
        assert issubclass(ExchangeError, TradingBotError)
        assert issubclass(ConnectionError, ExchangeError)
        assert issubclass(AuthenticationError, ExchangeError)
        assert issubclass(RateLimitError, ExchangeError)
        assert issubclass(InsufficientBalanceError, ExchangeError)
        assert issubclass(OrderError, ExchangeError)

        # DataError inherits from TradingBotError
        assert issubclass(DataError, TradingBotError)
        assert issubclass(ValidationError, DataError)
        assert issubclass(NotFoundError, DataError)

        # Others inherit from TradingBotError
        assert issubclass(ConfigError, TradingBotError)
        assert issubclass(StrategyError, TradingBotError)

    def test_raise_and_catch_specific(self):
        """Test raising and catching specific exceptions."""
        with pytest.raises(RateLimitError) as exc_info:
            raise RateLimitError("Rate limited", retry_after=60)
        assert exc_info.value.retry_after == 60


# =============================================================================
# Utils Tests
# =============================================================================


class TestTimeUtils:
    """Test cases for time utility functions."""

    def test_timestamp_to_datetime_ms(self):
        """Test millisecond timestamp conversion."""
        # 2024-01-01 00:00:00 UTC
        ts = 1704067200000
        dt = timestamp_to_datetime(ts, unit="ms")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1
        assert dt.tzinfo == timezone.utc

    def test_timestamp_to_datetime_s(self):
        """Test second timestamp conversion."""
        ts = 1704067200
        dt = timestamp_to_datetime(ts, unit="s")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.day == 1

    def test_datetime_to_timestamp_ms(self):
        """Test datetime to millisecond timestamp."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = datetime_to_timestamp(dt, unit="ms")
        assert ts == 1704067200000

    def test_datetime_to_timestamp_s(self):
        """Test datetime to second timestamp."""
        dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
        ts = datetime_to_timestamp(dt, unit="s")
        assert ts == 1704067200

    def test_now_timestamp_ms(self):
        """Test current timestamp in milliseconds."""
        ts = now_timestamp(unit="ms")
        # Should be around current time (13 digits)
        assert len(str(ts)) == 13

    def test_now_timestamp_s(self):
        """Test current timestamp in seconds."""
        ts = now_timestamp(unit="s")
        # Should be around current time (10 digits)
        assert len(str(ts)) == 10

    def test_roundtrip_conversion(self):
        """Test timestamp <-> datetime roundtrip."""
        original_ts = 1704067200000
        dt = timestamp_to_datetime(original_ts)
        back_ts = datetime_to_timestamp(dt)
        assert original_ts == back_ts


class TestNumericUtils:
    """Test cases for numeric utility functions."""

    def test_round_decimal_basic(self):
        """Test basic decimal rounding."""
        result = round_decimal(Decimal("123.456"), 2)
        assert result == Decimal("123.45")

    def test_round_decimal_precision_0(self):
        """Test rounding to 0 decimal places."""
        result = round_decimal(Decimal("123.456"), 0)
        assert result == Decimal("123")

    def test_round_to_tick_basic(self):
        """Test round_to_tick with 0.01 tick."""
        result = round_to_tick(Decimal("123.456"), Decimal("0.01"))
        assert result == Decimal("123.45")

    def test_round_to_tick_larger(self):
        """Test round_to_tick with 0.05 tick."""
        result = round_to_tick(Decimal("123.456"), Decimal("0.05"))
        assert result == Decimal("123.45")

    def test_round_to_tick_whole_number(self):
        """Test round_to_tick with tick size of 1."""
        result = round_to_tick(Decimal("123.456"), Decimal("1"))
        assert result == Decimal("123")

    def test_format_quantity_removes_trailing_zeros(self):
        """Test format_quantity removes trailing zeros."""
        assert format_quantity(Decimal("1.50000"), 5) == "1.5"
        assert format_quantity(Decimal("1.00000"), 5) == "1"
        assert format_quantity(Decimal("1.23456"), 5) == "1.23456"

    def test_format_price(self):
        """Test format_price."""
        assert format_price(Decimal("50000.00"), 2) == "50000"
        assert format_price(Decimal("50000.50"), 2) == "50000.5"

    def test_calculate_percentage(self):
        """Test percentage calculation."""
        result = calculate_percentage(Decimal("25"), Decimal("100"))
        assert result == Decimal("25")

    def test_calculate_percentage_zero_whole(self):
        """Test percentage with zero denominator."""
        result = calculate_percentage(Decimal("25"), Decimal("0"))
        assert result == Decimal("0")


class TestStringUtils:
    """Test cases for string utility functions."""

    def test_generate_client_order_id_format(self):
        """Test client order ID format."""
        order_id = generate_client_order_id("GRID")
        parts = order_id.split("_")
        assert len(parts) == 3
        assert parts[0] == "GRID"
        # Timestamp should be numeric
        assert parts[1].isdigit()
        # Random suffix should be 6 chars
        assert len(parts[2]) == 6

    def test_generate_client_order_id_unique(self):
        """Test that generated IDs are unique."""
        ids = [generate_client_order_id() for _ in range(100)]
        assert len(set(ids)) == 100

    def test_mask_secret_long(self):
        """Test masking long secret."""
        result = mask_secret("abcdefghijklmnop", 4)
        assert result == "abcd********mnop"
        assert len(result) == len("abcdefghijklmnop")

    def test_mask_secret_short(self):
        """Test masking short secret."""
        result = mask_secret("short", 4)
        assert result == "***"


class TestRetryDecorator:
    """Test cases for retry decorator."""

    def test_retry_success_first_try(self):
        """Test retry when function succeeds immediately."""
        call_count = 0

        @retry(max_retries=3)
        def success_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = success_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_success_after_failures(self):
        """Test retry succeeds after some failures."""
        call_count = 0

        @retry(max_retries=3, delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Fail")
            return "success"

        result = fail_then_succeed()
        assert result == "success"
        assert call_count == 3

    def test_retry_all_failures(self):
        """Test retry exhausts all attempts."""
        call_count = 0

        @retry(max_retries=2, delay=0.01)
        def always_fail():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fail")

        with pytest.raises(ValueError):
            always_fail()
        assert call_count == 3  # Initial + 2 retries

    def test_retry_specific_exceptions(self):
        """Test retry only catches specified exceptions."""
        @retry(max_retries=3, delay=0.01, exceptions=(ValueError,))
        def raise_type_error():
            raise TypeError("Wrong type")

        # TypeError should not be caught
        with pytest.raises(TypeError):
            raise_type_error()


# =============================================================================
# Integration Test
# =============================================================================


def test_logger_with_colored_output():
    """
    Manual test for colored output.
    Run with: pytest tests/test_core.py::test_logger_with_colored_output -s
    """
    print("\n--- Logger Color Test ---")
    logger = setup_logger("color.test")
    logger.debug("This is a DEBUG message (gray)")
    logger.info("This is an INFO message (green)")
    logger.warning("This is a WARNING message (yellow)")
    logger.error("This is an ERROR message (red)")
    print("--- End Color Test ---\n")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
