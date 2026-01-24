"""
Performance monitoring module.

Provides latency tracking, throughput metering, and performance profiling
for high-frequency trading scenarios.
"""

import asyncio
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Deque, Dict, List, Optional, TypeVar, Union

from src.core import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class OperationType(Enum):
    """Types of operations to track."""

    API_CALL = "api_call"
    ORDER_SUBMIT = "order_submit"
    ORDER_CANCEL = "order_cancel"
    MARKET_DATA = "market_data"
    WEBSOCKET_MSG = "websocket_msg"
    DATABASE_QUERY = "database_query"
    CACHE_READ = "cache_read"
    CACHE_WRITE = "cache_write"
    STRATEGY_CALC = "strategy_calc"
    RISK_CHECK = "risk_check"
    CUSTOM = "custom"


@dataclass
class LatencyRecord:
    """Single latency measurement."""

    operation: str
    latency_ms: float
    timestamp: datetime
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for an operation type."""

    operation: str
    count: int
    min_ms: float
    max_ms: float
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    std_dev_ms: float
    success_rate: float
    period_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "count": self.count,
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "avg_ms": round(self.avg_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "success_rate": round(self.success_rate, 4),
            "period_seconds": self.period_seconds,
        }


class LatencyTracker:
    """
    Tracks latency for various operations.

    Uses a sliding window to maintain recent measurements and calculate
    percentiles efficiently.
    """

    def __init__(
        self,
        window_size: int = 10000,
        window_seconds: float = 300.0,
    ):
        """
        Initialize the latency tracker.

        Args:
            window_size: Maximum number of records to keep per operation
            window_seconds: Time window for statistics in seconds
        """
        self._window_size = window_size
        self._window_seconds = window_seconds
        self._records: Dict[str, Deque[LatencyRecord]] = {}
        self._lock = asyncio.Lock()

    def record(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a latency measurement.

        Args:
            operation: Operation name/type
            latency_ms: Latency in milliseconds
            success: Whether the operation succeeded
            metadata: Optional additional data
        """
        if operation not in self._records:
            self._records[operation] = deque(maxlen=self._window_size)

        record = LatencyRecord(
            operation=operation,
            latency_ms=latency_ms,
            timestamp=datetime.now(timezone.utc),
            success=success,
            metadata=metadata or {},
        )
        self._records[operation].append(record)

    def record_sync(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
    ) -> None:
        """Synchronous version of record for non-async contexts."""
        self.record(operation, latency_ms, success)

    def get_stats(
        self,
        operation: str,
        window_seconds: Optional[float] = None,
    ) -> Optional[LatencyStats]:
        """
        Get latency statistics for an operation.

        Args:
            operation: Operation name
            window_seconds: Time window (defaults to configured window)

        Returns:
            LatencyStats or None if no data
        """
        if operation not in self._records:
            return None

        window = window_seconds or self._window_seconds
        cutoff = datetime.now(timezone.utc).timestamp() - window

        # Filter to window
        records = [
            r for r in self._records[operation] if r.timestamp.timestamp() > cutoff
        ]

        if not records:
            return None

        latencies = [r.latency_ms for r in records]
        successes = [r for r in records if r.success]

        # Calculate percentiles
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        def percentile(p: float) -> float:
            idx = int(n * p / 100)
            return sorted_latencies[min(idx, n - 1)]

        return LatencyStats(
            operation=operation,
            count=n,
            min_ms=min(latencies),
            max_ms=max(latencies),
            avg_ms=statistics.mean(latencies),
            p50_ms=percentile(50),
            p95_ms=percentile(95),
            p99_ms=percentile(99),
            std_dev_ms=statistics.stdev(latencies) if n > 1 else 0.0,
            success_rate=len(successes) / n if n > 0 else 0.0,
            period_seconds=window,
        )

    def get_all_stats(
        self,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, LatencyStats]:
        """Get statistics for all tracked operations."""
        result = {}
        for operation in self._records:
            stats = self.get_stats(operation, window_seconds)
            if stats:
                result[operation] = stats
        return result

    def get_slow_operations(
        self,
        threshold_ms: float,
        limit: int = 100,
    ) -> List[LatencyRecord]:
        """
        Get recent operations that exceeded threshold.

        Args:
            threshold_ms: Latency threshold in milliseconds
            limit: Maximum number of records to return

        Returns:
            List of slow operation records
        """
        slow = []
        for records in self._records.values():
            for r in records:
                if r.latency_ms > threshold_ms:
                    slow.append(r)

        # Sort by latency descending
        slow.sort(key=lambda r: r.latency_ms, reverse=True)
        return slow[:limit]

    def clear(self, operation: Optional[str] = None) -> None:
        """Clear records for operation or all operations."""
        if operation:
            if operation in self._records:
                self._records[operation].clear()
        else:
            self._records.clear()


@dataclass
class ThroughputRecord:
    """Throughput measurement for a time bucket."""

    bucket_start: datetime
    count: int
    bytes_processed: int = 0
    errors: int = 0


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    operation: str
    current_rps: float  # Requests per second
    avg_rps: float
    peak_rps: float
    total_requests: int
    total_errors: int
    error_rate: float
    bytes_per_second: float
    period_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "operation": self.operation,
            "current_rps": round(self.current_rps, 2),
            "avg_rps": round(self.avg_rps, 2),
            "peak_rps": round(self.peak_rps, 2),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": round(self.error_rate, 4),
            "bytes_per_second": round(self.bytes_per_second, 2),
            "period_seconds": self.period_seconds,
        }


class ThroughputMeter:
    """
    Measures throughput (requests/second, bytes/second).

    Uses time-bucketed counters for efficient real-time tracking.
    """

    def __init__(
        self,
        bucket_seconds: float = 1.0,
        window_buckets: int = 300,
    ):
        """
        Initialize the throughput meter.

        Args:
            bucket_seconds: Time bucket size in seconds
            window_buckets: Number of buckets to keep
        """
        self._bucket_seconds = bucket_seconds
        self._window_buckets = window_buckets
        self._buckets: Dict[str, Deque[ThroughputRecord]] = {}
        self._current_bucket: Dict[str, ThroughputRecord] = {}
        self._lock = asyncio.Lock()

    def _get_bucket_key(self) -> datetime:
        """Get the current bucket key."""
        now = time.time()
        bucket_start = now - (now % self._bucket_seconds)
        return datetime.fromtimestamp(bucket_start, tz=timezone.utc)

    def _rotate_bucket(self, operation: str) -> None:
        """Rotate to new bucket if needed."""
        current_key = self._get_bucket_key()

        if operation not in self._buckets:
            self._buckets[operation] = deque(maxlen=self._window_buckets)
            self._current_bucket[operation] = ThroughputRecord(
                bucket_start=current_key, count=0
            )
            return

        current = self._current_bucket.get(operation)
        if current and current.bucket_start != current_key:
            # Store completed bucket
            self._buckets[operation].append(current)
            # Start new bucket
            self._current_bucket[operation] = ThroughputRecord(
                bucket_start=current_key, count=0
            )

    def record(
        self,
        operation: str,
        count: int = 1,
        bytes_processed: int = 0,
        is_error: bool = False,
    ) -> None:
        """
        Record throughput.

        Args:
            operation: Operation name
            count: Number of requests
            bytes_processed: Bytes processed
            is_error: Whether this was an error
        """
        self._rotate_bucket(operation)

        bucket = self._current_bucket[operation]
        bucket.count += count
        bucket.bytes_processed += bytes_processed
        if is_error:
            bucket.errors += count

    def get_stats(
        self,
        operation: str,
        window_seconds: Optional[float] = None,
    ) -> Optional[ThroughputStats]:
        """
        Get throughput statistics.

        Args:
            operation: Operation name
            window_seconds: Time window (defaults to full window)

        Returns:
            ThroughputStats or None if no data
        """
        if operation not in self._buckets:
            return None

        window = window_seconds or (self._bucket_seconds * self._window_buckets)
        cutoff = datetime.now(timezone.utc).timestamp() - window

        # Get relevant buckets
        buckets = [
            b
            for b in self._buckets[operation]
            if b.bucket_start.timestamp() > cutoff
        ]

        # Add current bucket
        current = self._current_bucket.get(operation)
        if current and current.bucket_start.timestamp() > cutoff:
            buckets.append(current)

        if not buckets:
            return None

        total_requests = sum(b.count for b in buckets)
        total_errors = sum(b.errors for b in buckets)
        total_bytes = sum(b.bytes_processed for b in buckets)

        # Calculate rates
        actual_seconds = len(buckets) * self._bucket_seconds
        avg_rps = total_requests / actual_seconds if actual_seconds > 0 else 0

        # Current rate (last bucket)
        current_rps = (
            buckets[-1].count / self._bucket_seconds if buckets else 0
        )

        # Peak rate
        peak_rps = max(b.count / self._bucket_seconds for b in buckets)

        return ThroughputStats(
            operation=operation,
            current_rps=current_rps,
            avg_rps=avg_rps,
            peak_rps=peak_rps,
            total_requests=total_requests,
            total_errors=total_errors,
            error_rate=total_errors / total_requests if total_requests > 0 else 0,
            bytes_per_second=total_bytes / actual_seconds if actual_seconds > 0 else 0,
            period_seconds=window,
        )

    def get_all_stats(
        self,
        window_seconds: Optional[float] = None,
    ) -> Dict[str, ThroughputStats]:
        """Get statistics for all tracked operations."""
        result = {}
        for operation in self._buckets:
            stats = self.get_stats(operation, window_seconds)
            if stats:
                result[operation] = stats
        return result


@dataclass
class PerformanceMetrics:
    """Combined performance metrics snapshot."""

    timestamp: datetime
    latency: Dict[str, LatencyStats]
    throughput: Dict[str, ThroughputStats]
    memory_mb: float
    cpu_percent: float
    active_tasks: int
    open_connections: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": {k: v.to_dict() for k, v in self.latency.items()},
            "throughput": {k: v.to_dict() for k, v in self.throughput.items()},
            "memory_mb": round(self.memory_mb, 2),
            "cpu_percent": round(self.cpu_percent, 2),
            "active_tasks": self.active_tasks,
            "open_connections": self.open_connections,
        }


class PerformanceProfiler:
    """
    Unified performance profiler combining latency and throughput tracking.

    Provides comprehensive performance monitoring for trading systems.
    """

    def __init__(
        self,
        latency_window_size: int = 10000,
        latency_window_seconds: float = 300.0,
        throughput_bucket_seconds: float = 1.0,
        throughput_window_buckets: int = 300,
    ):
        """
        Initialize the profiler.

        Args:
            latency_window_size: Max latency records per operation
            latency_window_seconds: Latency stats window
            throughput_bucket_seconds: Throughput bucket size
            throughput_window_buckets: Number of throughput buckets
        """
        self._latency = LatencyTracker(
            window_size=latency_window_size,
            window_seconds=latency_window_seconds,
        )
        self._throughput = ThroughputMeter(
            bucket_seconds=throughput_bucket_seconds,
            window_buckets=throughput_window_buckets,
        )
        self._connection_count = 0
        self._start_time = datetime.now(timezone.utc)

    @property
    def latency(self) -> LatencyTracker:
        """Get the latency tracker."""
        return self._latency

    @property
    def throughput(self) -> ThroughputMeter:
        """Get the throughput meter."""
        return self._throughput

    def record_operation(
        self,
        operation: str,
        latency_ms: float,
        success: bool = True,
        bytes_processed: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Record a complete operation with latency and throughput.

        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
            success: Whether operation succeeded
            bytes_processed: Bytes processed
            metadata: Optional metadata
        """
        self._latency.record(operation, latency_ms, success, metadata)
        self._throughput.record(
            operation,
            count=1,
            bytes_processed=bytes_processed,
            is_error=not success,
        )

    def time_operation(self, operation: str) -> "OperationTimer":
        """
        Create a context manager for timing operations.

        Args:
            operation: Operation name

        Returns:
            OperationTimer context manager
        """
        return OperationTimer(self, operation)

    def set_connection_count(self, count: int) -> None:
        """Set the current connection count."""
        self._connection_count = count

    def get_metrics(self) -> PerformanceMetrics:
        """
        Get current performance metrics snapshot.

        Returns:
            PerformanceMetrics with all current data
        """
        import os

        try:
            import psutil

            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent()
        except ImportError:
            memory_mb = 0.0
            cpu_percent = 0.0

        # Count active asyncio tasks
        try:
            active_tasks = len(asyncio.all_tasks())
        except RuntimeError:
            active_tasks = 0

        return PerformanceMetrics(
            timestamp=datetime.now(timezone.utc),
            latency=self._latency.get_all_stats(),
            throughput=self._throughput.get_all_stats(),
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            active_tasks=active_tasks,
            open_connections=self._connection_count,
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        metrics = self.get_metrics()
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()

        # Find critical metrics
        slowest_op = None
        slowest_p99 = 0.0
        for op, stats in metrics.latency.items():
            if stats.p99_ms > slowest_p99:
                slowest_p99 = stats.p99_ms
                slowest_op = op

        highest_throughput_op = None
        highest_rps = 0.0
        for op, stats in metrics.throughput.items():
            if stats.current_rps > highest_rps:
                highest_rps = stats.current_rps
                highest_throughput_op = op

        return {
            "uptime_seconds": round(uptime, 2),
            "memory_mb": round(metrics.memory_mb, 2),
            "cpu_percent": round(metrics.cpu_percent, 2),
            "active_tasks": metrics.active_tasks,
            "open_connections": metrics.open_connections,
            "operations_tracked": len(metrics.latency),
            "slowest_operation": slowest_op,
            "slowest_p99_ms": round(slowest_p99, 3),
            "highest_throughput_op": highest_throughput_op,
            "highest_rps": round(highest_rps, 2),
        }

    def clear(self) -> None:
        """Clear all tracked data."""
        self._latency.clear()
        # Throughput doesn't need clearing - buckets expire naturally


class OperationTimer:
    """Context manager for timing operations."""

    def __init__(self, profiler: PerformanceProfiler, operation: str):
        self._profiler = profiler
        self._operation = operation
        self._start: Optional[float] = None
        self._success = True
        self._bytes = 0
        self._metadata: Dict[str, Any] = {}

    def set_success(self, success: bool) -> "OperationTimer":
        """Set the success status."""
        self._success = success
        return self

    def set_bytes(self, bytes_processed: int) -> "OperationTimer":
        """Set bytes processed."""
        self._bytes = bytes_processed
        return self

    def set_metadata(self, **kwargs: Any) -> "OperationTimer":
        """Set metadata."""
        self._metadata.update(kwargs)
        return self

    def __enter__(self) -> "OperationTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._success = False

        latency_ms = (time.perf_counter() - self._start) * 1000
        self._profiler.record_operation(
            self._operation,
            latency_ms,
            self._success,
            self._bytes,
            self._metadata,
        )

    async def __aenter__(self) -> "OperationTimer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_global_profiler() -> PerformanceProfiler:
    """Get the global profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def latency_tracked(
    operation: Optional[str] = None,
    profiler: Optional[PerformanceProfiler] = None,
) -> Callable[[F], F]:
    """
    Decorator to track latency of a function.

    Args:
        operation: Operation name (defaults to function name)
        profiler: Profiler to use (defaults to global)

    Returns:
        Decorated function
    """

    def decorator(func: F) -> F:
        op_name = operation or func.__name__
        prof = profiler or get_global_profiler()

        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                success = True
                try:
                    return await func(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                    prof.record_operation(op_name, latency_ms, success)

            return async_wrapper  # type: ignore
        else:

            @wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                success = True
                try:
                    return func(*args, **kwargs)
                except Exception:
                    success = False
                    raise
                finally:
                    latency_ms = (time.perf_counter() - start) * 1000
                    prof.record_operation(op_name, latency_ms, success)

            return sync_wrapper  # type: ignore

    return decorator
