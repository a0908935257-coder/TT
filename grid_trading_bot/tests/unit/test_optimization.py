"""
Unit tests for Optimization module.

Tests for:
- ConnectionPool with health checking
- BatchOptimizer for request batching
- CircuitBreaker with state transitions
"""

import asyncio
import pytest
from datetime import datetime, timezone

from src.optimization.pool import (
    ConnectionPool,
    ConnectionFactory,
    PoolConfig,
    PoolStats,
    PooledConnection,
    BatchOptimizer,
    BatchConfig,
    BatchResult,
)
from src.optimization.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitConfig,
    CircuitStats,
    CircuitBreakerError,
    circuit_protected,
    get_circuit_breaker,
    RateLimitingCircuitBreaker,
)


# =============================================================================
# Test Fixtures
# =============================================================================


class MockConnection:
    """Mock connection for testing."""

    def __init__(self, conn_id: int):
        self.conn_id = conn_id
        self.closed = False
        self.healthy = True

    async def close(self):
        self.closed = True


class MockConnectionFactory(ConnectionFactory[MockConnection]):
    """Mock connection factory for testing."""

    def __init__(self):
        self._counter = 0

    async def create(self) -> MockConnection:
        self._counter += 1
        return MockConnection(self._counter)

    async def close(self, connection: MockConnection) -> None:
        connection.closed = True

    async def is_healthy(self, connection: MockConnection) -> bool:
        return not connection.closed and connection.healthy


# =============================================================================
# Connection Pool Tests
# =============================================================================


class TestConnectionPool:
    """Tests for ConnectionPool."""

    @pytest.fixture
    def factory(self):
        """Create mock factory."""
        return MockConnectionFactory()

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PoolConfig(
            min_size=2,
            max_size=5,
            max_idle_seconds=60.0,
            acquire_timeout=5.0,
        )

    @pytest.mark.asyncio
    async def test_pool_start(self, factory, config):
        """Test pool start creates minimum connections."""
        pool = ConnectionPool(factory, config)

        await pool.start()

        assert pool.get_stats().idle_connections == 2
        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_acquire_release(self, factory, config):
        """Test acquire and release connection."""
        pool = ConnectionPool(factory, config)
        await pool.start()

        conn = await pool.acquire()

        assert conn is not None
        assert pool.get_stats().in_use_connections == 1

        await pool.release(conn)

        assert pool.get_stats().in_use_connections == 0
        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_connection_reuse(self, factory, config):
        """Test connection is reused (not creating new ones)."""
        pool = ConnectionPool(factory, config)
        await pool.start()

        # Pool starts with min_size connections
        initial_created = pool.get_stats().total_created

        # Acquire and release multiple times
        for _ in range(5):
            conn = await pool.acquire()
            await pool.release(conn)

        # Should not have created new connections
        final_created = pool.get_stats().total_created
        assert final_created == initial_created

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_max_size(self, factory, config):
        """Test pool respects max size."""
        pool = ConnectionPool(factory, config)
        await pool.start()

        # Acquire all connections
        connections = []
        for _ in range(5):
            conn = await pool.acquire()
            connections.append(conn)

        assert pool.get_stats().in_use_connections == 5
        assert pool.get_stats().idle_connections == 0

        # Release all
        for conn in connections:
            await pool.release(conn)

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_overflow(self, factory):
        """Test overflow connections."""
        config = PoolConfig(
            min_size=1,
            max_size=2,
            enable_overflow=True,
            overflow_max=2,
        )
        pool = ConnectionPool(factory, config)
        await pool.start()

        # Acquire beyond max_size with overflow
        connections = []
        for _ in range(4):
            conn = await pool.acquire()
            connections.append(conn)

        stats = pool.get_stats()
        assert stats.in_use_connections == 2  # Regular connections
        assert stats.overflow_connections == 2  # Overflow

        for conn in connections:
            await pool.release(conn)

        await pool.stop()

    @pytest.mark.asyncio
    async def test_pool_stats(self, factory, config):
        """Test pool statistics."""
        pool = ConnectionPool(factory, config)
        await pool.start()

        conn = await pool.acquire()
        await pool.release(conn)

        stats = pool.get_stats()

        assert stats.total_acquires == 1
        assert stats.total_releases == 1
        assert stats.total_created >= 2  # min_size

        await pool.stop()


# =============================================================================
# Pooled Connection Tests
# =============================================================================


class TestPooledConnection:
    """Tests for PooledConnection."""

    def test_connection_initialization(self):
        """Test pooled connection initialization."""
        conn = MockConnection(1)
        pooled = PooledConnection(connection=conn)

        assert pooled.connection == conn
        assert pooled.use_count == 0
        assert pooled.is_overflow is False

    def test_mark_used(self):
        """Test marking connection as used."""
        conn = MockConnection(1)
        pooled = PooledConnection(connection=conn)

        pooled.mark_used()

        assert pooled.use_count == 1
        from src.optimization.pool import ConnectionState
        assert pooled.state == ConnectionState.IN_USE

    def test_age_calculation(self):
        """Test age calculation."""
        conn = MockConnection(1)
        pooled = PooledConnection(connection=conn)

        # Age should be very small
        assert pooled.age_seconds < 1.0


# =============================================================================
# Batch Optimizer Tests
# =============================================================================


class TestBatchOptimizer:
    """Tests for BatchOptimizer."""

    @pytest.fixture
    def processor(self):
        """Create mock batch processor."""
        async def process(items):
            return [i * 2 for i in items]
        return process

    @pytest.mark.asyncio
    async def test_batch_submit(self, processor):
        """Test submitting items for batching."""
        optimizer = BatchOptimizer(processor)
        await optimizer.start()

        result = await optimizer.submit(5)

        assert result.success is True
        assert result.result == 10

        await optimizer.stop()

    @pytest.mark.asyncio
    async def test_batch_multiple_items(self, processor):
        """Test batching multiple items."""
        optimizer = BatchOptimizer(processor, BatchConfig(max_wait_ms=50.0))
        await optimizer.start()

        # Submit multiple items concurrently
        tasks = [
            optimizer.submit(i) for i in range(5)
        ]
        results = await asyncio.gather(*tasks)

        assert all(r.success for r in results)
        expected = [0, 2, 4, 6, 8]
        actual = [r.result for r in results]
        assert sorted(actual) == expected

        await optimizer.stop()

    @pytest.mark.asyncio
    async def test_batch_error_handling(self):
        """Test error handling in batch."""
        async def failing_processor(items):
            raise ValueError("Processing failed")

        optimizer = BatchOptimizer(failing_processor)
        await optimizer.start()

        result = await optimizer.submit(5)

        assert result.success is False
        assert result.error is not None
        assert isinstance(result.error, ValueError)

        await optimizer.stop()

    @pytest.mark.asyncio
    async def test_batch_stats(self, processor):
        """Test batch statistics."""
        optimizer = BatchOptimizer(processor, BatchConfig(max_wait_ms=10.0))
        await optimizer.start()

        for i in range(10):
            await optimizer.submit(i)

        stats = optimizer.get_stats()

        assert stats["total_items"] == 10
        assert stats["total_batches"] >= 1

        await optimizer.stop()


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return CircuitConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout_seconds=1.0,
            window_seconds=60.0,
        )

    @pytest.mark.asyncio
    async def test_circuit_starts_closed(self, config):
        """Test circuit starts in closed state."""
        circuit = CircuitBreaker("test", config)

        assert circuit.state == CircuitState.CLOSED
        assert circuit.is_closed

    @pytest.mark.asyncio
    async def test_success_keeps_circuit_closed(self, config):
        """Test success keeps circuit closed."""
        circuit = CircuitBreaker("test", config)

        async def success_func():
            return "ok"

        for _ in range(5):
            result = await circuit.call(success_func)
            assert result == "ok"

        assert circuit.is_closed
        stats = circuit.get_stats()
        assert stats.total_successes == 5

    @pytest.mark.asyncio
    async def test_failures_open_circuit(self, config):
        """Test failures open circuit."""
        circuit = CircuitBreaker("test", config)

        async def failing_func():
            raise ValueError("Error")

        # Cause failures
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        assert circuit.is_open
        stats = circuit.get_stats()
        assert stats.times_opened == 1

    @pytest.mark.asyncio
    async def test_open_circuit_rejects_calls(self, config):
        """Test open circuit rejects calls."""
        circuit = CircuitBreaker("test", config)

        async def failing_func():
            raise ValueError("Error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        # Should reject calls
        with pytest.raises(CircuitBreakerError) as exc_info:
            await circuit.call(failing_func)

        assert "is open" in str(exc_info.value)
        assert circuit.get_stats().rejection_count == 1

    @pytest.mark.asyncio
    async def test_circuit_recovery(self, config):
        """Test circuit recovers after timeout."""
        circuit = CircuitBreaker("recovery_test", config)

        async def failing_func():
            raise ValueError("Error")

        async def success_func():
            return "ok"

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        assert circuit.is_open

        # Wait for timeout
        await asyncio.sleep(1.1)

        # Should transition to half-open and allow calls
        result = await circuit.call(success_func)
        assert result == "ok"
        assert circuit.state == CircuitState.HALF_OPEN

        # More successes should close circuit
        await circuit.call(success_func)
        assert circuit.is_closed

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self, config):
        """Test failure in half-open reopens circuit."""
        circuit = CircuitBreaker("half_open_fail_test", config)

        async def failing_func():
            raise ValueError("Error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        assert circuit.is_open

        # Wait for timeout (with some margin)
        await asyncio.sleep(1.5)

        # Fail in half-open - should transition then fail with ValueError
        with pytest.raises(ValueError):
            await circuit.call(failing_func)

        # Should be back to open
        assert circuit.is_open
        assert circuit.get_stats().times_opened == 2

    @pytest.mark.asyncio
    async def test_manual_reset(self, config):
        """Test manual circuit reset."""
        circuit = CircuitBreaker("test", config)

        async def failing_func():
            raise ValueError("Error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        assert circuit.is_open

        # Manual reset
        await circuit.reset()

        assert circuit.is_closed

    @pytest.mark.asyncio
    async def test_excluded_exceptions(self):
        """Test excluded exceptions don't count as failures."""
        config = CircuitConfig(
            failure_threshold=3,
            exclude_exceptions=(KeyError,),
        )
        circuit = CircuitBreaker("test", config)

        async def key_error_func():
            raise KeyError("Not found")

        # KeyError should not count as failure
        for _ in range(5):
            with pytest.raises(KeyError):
                await circuit.call(key_error_func)

        # Circuit should still be closed
        assert circuit.is_closed

    @pytest.mark.asyncio
    async def test_state_change_callbacks(self, config):
        """Test state change callbacks."""
        opened = []
        closed = []

        circuit = CircuitBreaker("test", config)
        circuit.on_state_change(
            on_open=lambda: opened.append(True),
            on_close=lambda: closed.append(True),
        )

        async def failing_func():
            raise ValueError("Error")

        # Open the circuit
        for _ in range(3):
            with pytest.raises(ValueError):
                await circuit.call(failing_func)

        assert len(opened) == 1

        await circuit.reset()

        assert len(closed) == 1


# =============================================================================
# Circuit Breaker Decorator Tests
# =============================================================================


class TestCircuitProtectedDecorator:
    """Tests for circuit_protected decorator."""

    @pytest.mark.asyncio
    async def test_decorator_protects_function(self):
        """Test decorator protects function."""
        call_count = 0

        @circuit_protected("test_decorator")
        async def protected_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await protected_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_trips_on_failures(self):
        """Test decorator trips circuit on failures."""
        config = CircuitConfig(failure_threshold=2)

        @circuit_protected("test_fail_decorator", config=config)
        async def failing_func():
            raise RuntimeError("Always fails")

        # Cause failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await failing_func()

        # Should be tripped
        with pytest.raises(CircuitBreakerError):
            await failing_func()


# =============================================================================
# Rate Limiting Circuit Breaker Tests
# =============================================================================


class TestRateLimitingCircuitBreaker:
    """Tests for RateLimitingCircuitBreaker."""

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Test rate limiting functionality."""
        circuit = RateLimitingCircuitBreaker(
            name="rate_test",
            max_calls_per_second=10.0,
        )

        async def fast_func():
            return "ok"

        # Should allow initial calls
        for _ in range(5):
            result = await circuit.call(fast_func)
            assert result == "ok"

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        """Test rate limit exceeded error."""
        circuit = RateLimitingCircuitBreaker(
            name="rate_exceed_test",
            max_calls_per_second=2.0,
        )

        async def fast_func():
            return "ok"

        # Exhaust tokens quickly
        results = []
        for _ in range(5):
            try:
                result = await circuit.call(fast_func)
                results.append(result)
            except CircuitBreakerError:
                results.append("rate_limited")

        # Some should be rate limited
        assert "rate_limited" in results


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestCircuitBreakerRegistry:
    """Tests for circuit breaker registry."""

    def test_get_circuit_breaker(self):
        """Test getting circuit breaker from registry."""
        circuit1 = get_circuit_breaker("registry_test")
        circuit2 = get_circuit_breaker("registry_test")

        # Should return same instance
        assert circuit1 is circuit2

    def test_different_names_different_circuits(self):
        """Test different names create different circuits."""
        circuit1 = get_circuit_breaker("circuit_a")
        circuit2 = get_circuit_breaker("circuit_b")

        assert circuit1 is not circuit2
        assert circuit1.name == "circuit_a"
        assert circuit2.name == "circuit_b"
