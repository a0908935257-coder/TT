"""
Connection pooling and batch optimization module.

Provides connection reuse and request batching for optimal throughput
in high-frequency trading scenarios.
"""

import asyncio
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Awaitable,
    Callable,
    Deque,
    Dict,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from src.core import get_logger

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


class ConnectionState(Enum):
    """Connection state in the pool."""

    IDLE = "idle"
    IN_USE = "in_use"
    STALE = "stale"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    """Connection pool configuration."""

    min_size: int = 2
    max_size: int = 10
    max_idle_seconds: float = 300.0
    acquire_timeout: float = 30.0
    health_check_interval: float = 60.0
    max_lifetime_seconds: float = 3600.0
    enable_overflow: bool = True
    overflow_max: int = 5


@dataclass
class PoolStats:
    """Connection pool statistics."""

    total_connections: int = 0
    idle_connections: int = 0
    in_use_connections: int = 0
    overflow_connections: int = 0
    total_acquires: int = 0
    total_releases: int = 0
    total_timeouts: int = 0
    total_created: int = 0
    total_closed: int = 0
    avg_wait_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_connections": self.total_connections,
            "idle_connections": self.idle_connections,
            "in_use_connections": self.in_use_connections,
            "overflow_connections": self.overflow_connections,
            "total_acquires": self.total_acquires,
            "total_releases": self.total_releases,
            "total_timeouts": self.total_timeouts,
            "total_created": self.total_created,
            "total_closed": self.total_closed,
            "avg_wait_time_ms": round(self.avg_wait_time_ms, 3),
        }


@dataclass
class PooledConnection(Generic[T]):
    """Wrapper for a pooled connection."""

    connection: T
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    is_overflow: bool = False

    @property
    def age_seconds(self) -> float:
        """Get connection age in seconds."""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()

    @property
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return (datetime.now(timezone.utc) - self.last_used_at).total_seconds()

    def mark_used(self) -> None:
        """Mark connection as used."""
        self.state = ConnectionState.IN_USE
        self.last_used_at = datetime.now(timezone.utc)
        self.use_count += 1

    def mark_idle(self) -> None:
        """Mark connection as idle."""
        self.state = ConnectionState.IDLE
        self.last_used_at = datetime.now(timezone.utc)


class ConnectionFactory(ABC, Generic[T]):
    """Abstract factory for creating connections."""

    @abstractmethod
    async def create(self) -> T:
        """Create a new connection."""
        pass

    @abstractmethod
    async def close(self, connection: T) -> None:
        """Close a connection."""
        pass

    @abstractmethod
    async def is_healthy(self, connection: T) -> bool:
        """Check if connection is healthy."""
        pass


class HTTPConnectionFactory(ConnectionFactory[Any]):
    """Factory for aiohttp client sessions."""

    def __init__(
        self,
        base_url: str,
        headers: Optional[Dict[str, str]] = None,
        timeout: float = 30.0,
    ):
        self._base_url = base_url
        self._headers = headers or {}
        self._timeout = timeout

    async def create(self) -> Any:
        """Create a new aiohttp session."""
        import aiohttp

        timeout = aiohttp.ClientTimeout(total=self._timeout)
        session = aiohttp.ClientSession(
            base_url=self._base_url,
            headers=self._headers,
            timeout=timeout,
        )
        return session

    async def close(self, connection: Any) -> None:
        """Close the session."""
        await connection.close()

    async def is_healthy(self, connection: Any) -> bool:
        """Check if session is healthy."""
        return not connection.closed


class ConnectionPool(Generic[T]):
    """
    Generic connection pool with health checking.

    Manages a pool of reusable connections with automatic
    health checking and lifecycle management.
    """

    def __init__(
        self,
        factory: ConnectionFactory[T],
        config: Optional[PoolConfig] = None,
    ):
        """
        Initialize the connection pool.

        Args:
            factory: Connection factory
            config: Pool configuration
        """
        self._factory = factory
        self._config = config or PoolConfig()

        self._pool: Deque[PooledConnection[T]] = deque()
        self._in_use: Dict[int, PooledConnection[T]] = {}
        self._overflow: Dict[int, PooledConnection[T]] = {}

        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)

        self._running = False
        self._health_task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = PoolStats()
        self._wait_times: Deque[float] = deque(maxlen=1000)

    async def start(self) -> None:
        """Start the pool and create initial connections."""
        if self._running:
            return

        self._running = True

        # Create minimum connections
        for _ in range(self._config.min_size):
            try:
                conn = await self._create_connection()
                self._pool.append(conn)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")

        # Start health check task
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info(
            f"Connection pool started with {len(self._pool)} connections"
        )

    async def stop(self) -> None:
        """Stop the pool and close all connections."""
        self._running = False

        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        async with self._lock:
            for pooled in self._pool:
                await self._close_connection(pooled)
            self._pool.clear()

            for pooled in self._in_use.values():
                await self._close_connection(pooled)
            self._in_use.clear()

            for pooled in self._overflow.values():
                await self._close_connection(pooled)
            self._overflow.clear()

        logger.info("Connection pool stopped")

    async def acquire(self) -> T:
        """
        Acquire a connection from the pool.

        Returns:
            A connection from the pool

        Raises:
            asyncio.TimeoutError: If acquire times out
        """
        start_time = time.perf_counter()

        async with self._not_empty:
            while True:
                # Try to get an idle connection
                while self._pool:
                    pooled = self._pool.popleft()

                    # Check if connection is still valid
                    if pooled.state == ConnectionState.CLOSED:
                        continue

                    if pooled.age_seconds > self._config.max_lifetime_seconds:
                        await self._close_connection(pooled)
                        continue

                    if pooled.idle_seconds > self._config.max_idle_seconds:
                        await self._close_connection(pooled)
                        continue

                    # Valid connection found
                    pooled.mark_used()
                    self._in_use[id(pooled.connection)] = pooled
                    self._stats.total_acquires += 1
                    self._record_wait_time(start_time)
                    return pooled.connection

                # No idle connections, try to create new
                total = (
                    len(self._pool)
                    + len(self._in_use)
                    + len(self._overflow)
                )

                if total < self._config.max_size:
                    # Create new connection
                    pooled = await self._create_connection()
                    pooled.mark_used()
                    self._in_use[id(pooled.connection)] = pooled
                    self._stats.total_acquires += 1
                    self._record_wait_time(start_time)
                    return pooled.connection

                # Try overflow if enabled
                if (
                    self._config.enable_overflow
                    and len(self._overflow) < self._config.overflow_max
                ):
                    pooled = await self._create_connection()
                    pooled.is_overflow = True
                    pooled.mark_used()
                    self._overflow[id(pooled.connection)] = pooled
                    self._stats.overflow_connections += 1
                    self._stats.total_acquires += 1
                    self._record_wait_time(start_time)
                    return pooled.connection

                # Wait for a connection to be released
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(),
                        timeout=self._config.acquire_timeout,
                    )
                except asyncio.TimeoutError:
                    self._stats.total_timeouts += 1
                    raise

    async def release(self, connection: T) -> None:
        """
        Release a connection back to the pool.

        Args:
            connection: The connection to release
        """
        conn_id = id(connection)

        async with self._not_empty:
            # Check if it's a regular connection
            if conn_id in self._in_use:
                pooled = self._in_use.pop(conn_id)
                pooled.mark_idle()

                # Return to pool if healthy
                if await self._factory.is_healthy(connection):
                    self._pool.append(pooled)
                else:
                    await self._close_connection(pooled)

                self._stats.total_releases += 1
                self._not_empty.notify()
                return

            # Check if it's an overflow connection
            if conn_id in self._overflow:
                pooled = self._overflow.pop(conn_id)
                await self._close_connection(pooled)
                self._stats.overflow_connections -= 1
                self._stats.total_releases += 1
                self._not_empty.notify()
                return

    async def _create_connection(self) -> PooledConnection[T]:
        """Create a new pooled connection."""
        connection = await self._factory.create()
        self._stats.total_created += 1
        return PooledConnection(connection=connection)

    async def _close_connection(self, pooled: PooledConnection[T]) -> None:
        """Close a pooled connection."""
        pooled.state = ConnectionState.CLOSED
        try:
            await self._factory.close(pooled.connection)
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
        self._stats.total_closed += 1

    async def _health_check_loop(self) -> None:
        """Periodic health check for idle connections."""
        while self._running:
            try:
                await asyncio.sleep(self._config.health_check_interval)

                async with self._lock:
                    to_remove = []

                    for pooled in self._pool:
                        # Check lifetime
                        if pooled.age_seconds > self._config.max_lifetime_seconds:
                            to_remove.append(pooled)
                            continue

                        # Check idle time
                        if pooled.idle_seconds > self._config.max_idle_seconds:
                            to_remove.append(pooled)
                            continue

                        # Health check
                        if not await self._factory.is_healthy(pooled.connection):
                            to_remove.append(pooled)

                    # Remove unhealthy connections
                    for pooled in to_remove:
                        if pooled in self._pool:
                            self._pool.remove(pooled)
                            await self._close_connection(pooled)

                    # Ensure minimum pool size
                    while len(self._pool) < self._config.min_size:
                        try:
                            conn = await self._create_connection()
                            self._pool.append(conn)
                        except Exception as e:
                            logger.error(f"Failed to replenish pool: {e}")
                            break

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check: {e}")

    def _record_wait_time(self, start_time: float) -> None:
        """Record connection wait time."""
        wait_ms = (time.perf_counter() - start_time) * 1000
        self._wait_times.append(wait_ms)
        if self._wait_times:
            self._stats.avg_wait_time_ms = sum(self._wait_times) / len(
                self._wait_times
            )

    def get_stats(self) -> PoolStats:
        """Get pool statistics."""
        self._stats.total_connections = (
            len(self._pool) + len(self._in_use) + len(self._overflow)
        )
        self._stats.idle_connections = len(self._pool)
        self._stats.in_use_connections = len(self._in_use)
        return self._stats

    def __aenter__(self) -> "ConnectionPoolContext[T]":
        """Async context manager entry."""
        return ConnectionPoolContext(self)


class ConnectionPoolContext(Generic[T]):
    """Context manager for acquiring pool connections."""

    def __init__(self, pool: ConnectionPool[T]):
        self._pool = pool
        self._connection: Optional[T] = None

    async def __aenter__(self) -> T:
        self._connection = await self._pool.acquire()
        return self._connection

    async def __aexit__(
        self, exc_type: Any, exc_val: Any, exc_tb: Any
    ) -> None:
        if self._connection is not None:
            await self._pool.release(self._connection)


# Global pool instance
_global_pool: Optional[ConnectionPool[Any]] = None


def get_connection_pool() -> Optional[ConnectionPool[Any]]:
    """Get the global connection pool instance."""
    return _global_pool


def set_connection_pool(pool: ConnectionPool[Any]) -> None:
    """Set the global connection pool instance."""
    global _global_pool
    _global_pool = pool


@dataclass
class BatchConfig:
    """Batch optimizer configuration."""

    max_batch_size: int = 100
    max_wait_ms: float = 10.0
    enable_deduplication: bool = True


@dataclass
class BatchResult(Generic[R]):
    """Result of a batched operation."""

    success: bool
    result: Optional[R] = None
    error: Optional[Exception] = None
    batch_size: int = 1
    duration_ms: float = 0.0


class BatchOptimizer(Generic[T, R]):
    """
    Batches multiple requests for efficient processing.

    Collects requests over a time window and processes them
    together for reduced overhead.
    """

    def __init__(
        self,
        processor: Callable[[List[T]], Awaitable[List[R]]],
        config: Optional[BatchConfig] = None,
    ):
        """
        Initialize the batch optimizer.

        Args:
            processor: Function to process batched items
            config: Batch configuration
        """
        self._processor = processor
        self._config = config or BatchConfig()

        self._pending: List[Tuple[T, asyncio.Future[BatchResult[R]]]] = []
        self._lock = asyncio.Lock()
        self._batch_event = asyncio.Event()
        self._running = False
        self._process_task: Optional[asyncio.Task] = None

        # Stats
        self._total_batches = 0
        self._total_items = 0

    async def start(self) -> None:
        """Start the batch processor."""
        if self._running:
            return

        self._running = True
        self._process_task = asyncio.create_task(self._process_loop())
        logger.info("Batch optimizer started")

    async def stop(self) -> None:
        """Stop the batch processor."""
        self._running = False
        self._batch_event.set()

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass

        # Cancel pending requests
        async with self._lock:
            for _, future in self._pending:
                if not future.done():
                    future.cancel()
            self._pending.clear()

    async def submit(self, item: T) -> BatchResult[R]:
        """
        Submit an item for batched processing.

        Args:
            item: Item to process

        Returns:
            BatchResult with the processing result
        """
        future: asyncio.Future[BatchResult[R]] = asyncio.get_event_loop().create_future()

        async with self._lock:
            # Check deduplication
            if self._config.enable_deduplication:
                for existing, existing_future in self._pending:
                    if existing == item:
                        # Return the same future for duplicate
                        return await existing_future

            self._pending.append((item, future))

            # Trigger batch if full
            if len(self._pending) >= self._config.max_batch_size:
                self._batch_event.set()

        return await future

    async def _process_loop(self) -> None:
        """Main batch processing loop."""
        while self._running:
            try:
                # Wait for batch trigger or timeout
                try:
                    await asyncio.wait_for(
                        self._batch_event.wait(),
                        timeout=self._config.max_wait_ms / 1000,
                    )
                except asyncio.TimeoutError:
                    pass

                self._batch_event.clear()

                # Get pending items
                async with self._lock:
                    if not self._pending:
                        continue

                    batch = self._pending[: self._config.max_batch_size]
                    self._pending = self._pending[self._config.max_batch_size :]

                # Process batch
                items = [item for item, _ in batch]
                futures = [future for _, future in batch]

                start_time = time.perf_counter()
                try:
                    results = await self._processor(items)
                    duration_ms = (time.perf_counter() - start_time) * 1000

                    # Set results
                    for i, (result, future) in enumerate(zip(results, futures)):
                        if not future.done():
                            future.set_result(
                                BatchResult(
                                    success=True,
                                    result=result,
                                    batch_size=len(items),
                                    duration_ms=duration_ms,
                                )
                            )

                    self._total_batches += 1
                    self._total_items += len(items)

                except Exception as e:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    for future in futures:
                        if not future.done():
                            future.set_result(
                                BatchResult(
                                    success=False,
                                    error=e,
                                    batch_size=len(items),
                                    duration_ms=duration_ms,
                                )
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in batch loop: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get batch optimizer statistics."""
        return {
            "pending_items": len(self._pending),
            "total_batches": self._total_batches,
            "total_items": self._total_items,
            "avg_batch_size": (
                self._total_items / self._total_batches
                if self._total_batches > 0
                else 0
            ),
        }
