"""
Retry and Failover Unit Tests.

Tests for retry mechanisms, circuit breakers, and exchange failover.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.retry import (
    RetryConfig,
    RetryStrategy,
    RetryResult,
    retry_async,
    retry_sync,
    with_retry,
)
from src.core.retry.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreaker,
    CircuitOpenError,
    CircuitBreakerRegistry,
    with_circuit_breaker,
)
from src.exchange.base import (
    ExchangeType,
    ExchangeConfig,
    TickerData,
)
from src.exchange.failover import (
    FailoverManager,
    FailoverConfig,
    FailoverStrategy,
    ExchangeHealth,
)


class TestRetryConfig:
    """Test RetryConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = RetryConfig()
        assert config.max_attempts == 3
        assert config.base_delay == 1.0
        assert config.strategy == RetryStrategy.EXPONENTIAL_JITTER

    def test_fixed_delay(self):
        """Test fixed delay calculation."""
        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=2.0)
        assert config.calculate_delay(1) == 2.0
        assert config.calculate_delay(3) == 2.0
        assert config.calculate_delay(5) == 2.0

    def test_linear_delay(self):
        """Test linear delay calculation."""
        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=1.0)
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 3.0

    def test_exponential_delay(self):
        """Test exponential delay calculation."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=1.0,
            exponential_base=2.0,
        )
        assert config.calculate_delay(1) == 1.0
        assert config.calculate_delay(2) == 2.0
        assert config.calculate_delay(3) == 4.0

    def test_max_delay_cap(self):
        """Test max delay is capped."""
        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=10.0,
            max_delay=30.0,
        )
        # 10 * 2^4 = 160, but should be capped at 30
        assert config.calculate_delay(5) == 30.0

    def test_should_retry(self):
        """Test retry decision logic."""
        config = RetryConfig(
            retryable_exceptions=(ValueError, TypeError),
            non_retryable_exceptions=(KeyError,),
        )

        assert config.should_retry(ValueError("test")) is True
        assert config.should_retry(TypeError("test")) is True
        assert config.should_retry(KeyError("test")) is False
        assert config.should_retry(RuntimeError("test")) is False


class TestRetryAsync:
    """Test async retry functionality."""

    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self):
        """Test successful execution on first attempt."""
        async def success_func():
            return "success"

        result = await retry_async(success_func)

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1

    @pytest.mark.asyncio
    async def test_success_after_retry(self):
        """Test success after initial failures."""
        attempt_count = 0

        async def flaky_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary error")
            return "success"

        config = RetryConfig(max_attempts=5, base_delay=0.01)
        result = await retry_async(flaky_func, config=config)

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 3

    @pytest.mark.asyncio
    async def test_max_attempts_exceeded(self):
        """Test failure after max attempts."""
        async def always_fail():
            raise ValueError("Always fails")

        config = RetryConfig(max_attempts=3, base_delay=0.01)
        result = await retry_async(always_fail, config=config)

        assert result.success is False
        assert result.attempts == 3
        assert isinstance(result.exception, ValueError)

    @pytest.mark.asyncio
    async def test_non_retryable_exception(self):
        """Test immediate failure on non-retryable exception."""
        async def non_retryable():
            raise KeyError("Non-retryable")

        config = RetryConfig(
            max_attempts=5,
            non_retryable_exceptions=(KeyError,),
            base_delay=0.01,
        )
        result = await retry_async(non_retryable, config=config)

        assert result.success is False
        assert result.attempts == 1  # No retries

    @pytest.mark.asyncio
    async def test_callback_on_retry(self):
        """Test on_retry callback is called."""
        callbacks = []

        def on_retry(attempt, exc, delay):
            callbacks.append((attempt, str(exc), delay))

        attempt_count = 0

        async def flaky():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Error")
            return "ok"

        config = RetryConfig(
            max_attempts=5,
            base_delay=0.01,
            on_retry=on_retry,
        )
        await retry_async(flaky, config=config)

        assert len(callbacks) == 2  # Called twice before success


class TestWithRetryDecorator:
    """Test with_retry decorator."""

    @pytest.mark.asyncio
    async def test_async_decorator(self):
        """Test decorator with async function."""
        attempt_count = 0

        @with_retry(max_attempts=3, base_delay=0.01)
        async def decorated_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ValueError("Error")
            return "success"

        result = await decorated_func()

        assert result == "success"
        assert attempt_count == 2


class TestCircuitBreaker:
    """Test CircuitBreaker."""

    @pytest.mark.asyncio
    async def test_closed_state(self):
        """Test normal operation in closed state."""
        breaker = CircuitBreaker("test")

        async def success():
            return "ok"

        result = await breaker.call(success)
        assert result == "ok"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_opens_after_failures(self):
        """Test circuit opens after failure threshold."""
        config = CircuitBreakerConfig(failure_threshold=3)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("Error")

        # Cause failures
        for _ in range(3):
            try:
                await breaker.call(fail)
            except ValueError:
                pass

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        """Test requests are rejected when circuit is open."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=60.0)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("Error")

        # Open the circuit
        try:
            await breaker.call(fail)
        except ValueError:
            pass

        # Try another call - should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(fail)

    @pytest.mark.asyncio
    async def test_half_open_after_timeout(self):
        """Test transition to half-open after timeout."""
        config = CircuitBreakerConfig(failure_threshold=1, timeout=0.1)
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("Error")

        async def success():
            return "ok"

        # Open the circuit
        try:
            await breaker.call(fail)
        except ValueError:
            pass

        assert breaker.state == CircuitState.OPEN

        # Wait for timeout
        await asyncio.sleep(0.15)

        # Next call should transition to half-open and succeed
        result = await breaker.call(success)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_closes_after_success_in_half_open(self):
        """Test circuit closes after successes in half-open."""
        config = CircuitBreakerConfig(
            failure_threshold=1,
            success_threshold=2,
            timeout=0.1,
        )
        breaker = CircuitBreaker("test", config)

        async def fail():
            raise ValueError("Error")

        async def success():
            return "ok"

        # Open the circuit
        try:
            await breaker.call(fail)
        except ValueError:
            pass

        await asyncio.sleep(0.15)

        # Succeed twice to close
        await breaker.call(success)
        await breaker.call(success)

        assert breaker.state == CircuitState.CLOSED

    def test_manual_reset(self):
        """Test manual reset."""
        config = CircuitBreakerConfig(failure_threshold=1)
        breaker = CircuitBreaker("test", config)

        # Force open
        breaker.force_open()
        assert breaker.state == CircuitState.OPEN

        # Reset
        breaker.reset()
        assert breaker.state == CircuitState.CLOSED


class TestCircuitBreakerRegistry:
    """Test CircuitBreakerRegistry."""

    def test_get_or_create(self):
        """Test get_or_create functionality."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("test")
        breaker2 = registry.get_or_create("test")

        assert breaker1 is breaker2

    def test_reset_all(self):
        """Test resetting all circuit breakers."""
        registry = CircuitBreakerRegistry()

        breaker1 = registry.get_or_create("test1")
        breaker2 = registry.get_or_create("test2")

        breaker1.force_open()
        breaker2.force_open()

        registry.reset_all()

        assert breaker1.state == CircuitState.CLOSED
        assert breaker2.state == CircuitState.CLOSED


class TestExchangeHealth:
    """Test ExchangeHealth."""

    def test_record_success(self):
        """Test recording success."""
        health = ExchangeHealth(exchange_type=ExchangeType.BINANCE)

        health.record_success(100.0)

        assert health.total_requests == 1
        assert health.consecutive_failures == 0
        assert health.average_latency_ms == 100.0

    def test_record_failure(self):
        """Test recording failure."""
        health = ExchangeHealth(exchange_type=ExchangeType.BINANCE)

        health.record_failure()
        health.record_failure()
        health.record_failure()

        assert health.consecutive_failures == 3
        assert health.is_healthy is False

    def test_recovery(self):
        """Test health recovery after time."""
        health = ExchangeHealth(exchange_type=ExchangeType.BINANCE)

        # Mark unhealthy
        for _ in range(3):
            health.record_failure()

        assert health.is_healthy is False

        # Simulate time passing
        health.last_failure = datetime.now(timezone.utc) - timedelta(minutes=10)
        health.check_recovery(recovery_time=timedelta(minutes=5))

        assert health.is_healthy is True


class TestFailoverManager:
    """Test FailoverManager."""

    def test_add_exchange(self):
        """Test adding exchanges."""
        manager = FailoverManager()

        # Create mock exchange
        mock_api = MagicMock()
        mock_api.exchange_type = ExchangeType.BINANCE

        manager.add_exchange(mock_api, priority=1)

        status = manager.get_status()
        assert ExchangeType.BINANCE.value in status["exchanges"]

    def test_priority_strategy(self):
        """Test priority-based selection."""
        config = FailoverConfig(strategy=FailoverStrategy.PRIORITY)
        manager = FailoverManager(config)

        mock_binance = MagicMock()
        mock_binance.exchange_type = ExchangeType.BINANCE

        mock_okx = MagicMock()
        mock_okx.exchange_type = ExchangeType.OKX

        manager.add_exchange(mock_binance, priority=2)
        manager.add_exchange(mock_okx, priority=1)

        # OKX should be selected (lower priority number)
        next_exchange = manager._get_next_exchange()
        assert next_exchange == ExchangeType.OKX

    @pytest.mark.asyncio
    async def test_failover_on_error(self):
        """Test automatic failover when exchange fails."""
        manager = FailoverManager()

        # Mock exchanges
        mock_binance = AsyncMock()
        mock_binance.exchange_type = ExchangeType.BINANCE
        mock_binance.get_ticker = AsyncMock(side_effect=ValueError("Binance error"))

        mock_okx = AsyncMock()
        mock_okx.exchange_type = ExchangeType.OKX
        mock_okx.get_ticker = AsyncMock(return_value=TickerData(
            symbol="BTCUSDT",
            price=Decimal("50000"),
        ))

        manager.add_exchange(mock_binance, priority=1)
        manager.add_exchange(mock_okx, priority=2)

        # Should failover to OKX
        result = await manager.get_ticker("BTCUSDT")

        assert result.symbol == "BTCUSDT"
        assert result.price == Decimal("50000")
        assert mock_okx.get_ticker.called

    def test_force_exchange(self):
        """Test forcing specific exchange."""
        manager = FailoverManager()

        mock_binance = MagicMock()
        mock_binance.exchange_type = ExchangeType.BINANCE

        mock_okx = MagicMock()
        mock_okx.exchange_type = ExchangeType.OKX

        manager.add_exchange(mock_binance, priority=1)
        manager.add_exchange(mock_okx, priority=2)

        manager.force_exchange(ExchangeType.OKX)

        assert manager._current_exchange == ExchangeType.OKX


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
