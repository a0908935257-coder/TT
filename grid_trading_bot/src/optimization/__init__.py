# Optimization module - Performance optimization for high-frequency trading
from .pool import (
    ConnectionPool,
    PooledConnection,
    PoolConfig,
    PoolStats,
    BatchOptimizer,
    BatchConfig,
    BatchResult,
    get_connection_pool,
)
from .circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    CircuitConfig,
    CircuitStats,
    circuit_protected,
    get_circuit_breaker,
)

__all__ = [
    # Connection pool
    "ConnectionPool",
    "PooledConnection",
    "PoolConfig",
    "PoolStats",
    "get_connection_pool",
    # Batch optimizer
    "BatchOptimizer",
    "BatchConfig",
    "BatchResult",
    # Circuit breaker
    "CircuitBreaker",
    "CircuitState",
    "CircuitConfig",
    "CircuitStats",
    "circuit_protected",
    "get_circuit_breaker",
]
