"""
Real-time metrics streaming module.

Provides WebSocket-based streaming of performance metrics
to connected dashboard clients.
"""

import asyncio
import json
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol, Set

from src.core import get_logger

logger = get_logger(__name__)


class StreamEventType(Enum):
    """Types of streaming events."""

    METRICS_UPDATE = "metrics_update"
    ALERT_FIRED = "alert_fired"
    ALERT_RESOLVED = "alert_resolved"
    HEALTH_CHANGE = "health_change"
    TRADE_EXECUTED = "trade_executed"
    ORDER_UPDATE = "order_update"
    SYSTEM_STATUS = "system_status"
    HEARTBEAT = "heartbeat"


@dataclass
class StreamEvent:
    """Event for streaming to clients."""

    event_type: StreamEventType
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    source: str = "system"

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(
            {
                "type": self.event_type.value,
                "data": self.data,
                "timestamp": self.timestamp.isoformat(),
                "source": self.source,
            }
        )


@dataclass
class MetricsSnapshot:
    """Snapshot of current metrics for streaming."""

    timestamp: datetime
    latency: Dict[str, Dict[str, float]]  # operation -> {p50, p95, p99, avg}
    throughput: Dict[str, Dict[str, float]]  # operation -> {current_rps, avg_rps}
    system: Dict[str, float]  # memory_mb, cpu_percent, etc.
    alerts: Dict[str, int]  # severity -> count
    orders: Dict[str, int]  # status -> count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "latency": self.latency,
            "throughput": self.throughput,
            "system": self.system,
            "alerts": self.alerts,
            "orders": self.orders,
        }


class StreamSubscriber(Protocol):
    """Protocol for stream subscribers."""

    async def send(self, message: str) -> None:
        """Send a message to the subscriber."""
        ...

    @property
    def is_connected(self) -> bool:
        """Check if subscriber is still connected."""
        ...


class WebSocketSubscriber:
    """WebSocket-based subscriber implementation."""

    def __init__(self, websocket: Any):
        """
        Initialize with a WebSocket connection.

        Args:
            websocket: WebSocket connection (aiohttp or similar)
        """
        self._websocket = websocket
        self._connected = True

    async def send(self, message: str) -> None:
        """Send message through WebSocket."""
        if not self._connected:
            return
        try:
            await self._websocket.send_str(message)
        except Exception:
            self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def disconnect(self) -> None:
        """Mark as disconnected."""
        self._connected = False


class BufferedSubscriber:
    """Subscriber that buffers events for batch delivery."""

    def __init__(
        self,
        target: StreamSubscriber,
        buffer_size: int = 100,
        flush_interval: float = 0.1,
    ):
        """
        Initialize buffered subscriber.

        Args:
            target: Target subscriber to send to
            buffer_size: Max buffer size before force flush
            flush_interval: Flush interval in seconds
        """
        self._target = target
        self._buffer: List[str] = []
        self._buffer_size = buffer_size
        self._flush_interval = flush_interval
        self._last_flush = time.time()
        self._lock = asyncio.Lock()

    async def send(self, message: str) -> None:
        """Buffer and send message."""
        async with self._lock:
            self._buffer.append(message)

            # Force flush if buffer full or interval passed
            should_flush = (
                len(self._buffer) >= self._buffer_size
                or time.time() - self._last_flush >= self._flush_interval
            )

            if should_flush:
                await self._flush()

    async def _flush(self) -> None:
        """Flush buffered messages."""
        if not self._buffer:
            return

        # Send as batch
        batch = json.dumps({"batch": self._buffer})
        try:
            await self._target.send(batch)
        except Exception as e:
            logger.error(f"Failed to flush buffer: {e}")

        self._buffer.clear()
        self._last_flush = time.time()

    @property
    def is_connected(self) -> bool:
        """Check target connection status."""
        return self._target.is_connected


class MetricsStreamer:
    """
    Streams real-time metrics to connected clients.

    Manages subscriber connections and broadcasts metrics updates.
    """

    def __init__(
        self,
        update_interval: float = 1.0,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize the metrics streamer.

        Args:
            update_interval: Metrics update interval in seconds
            heartbeat_interval: Heartbeat interval in seconds
        """
        self._update_interval = update_interval
        self._heartbeat_interval = heartbeat_interval
        self._subscribers: Set[StreamSubscriber] = set()
        self._subscriptions: Dict[str, Set[StreamEventType]] = {}
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

        # Metric collectors
        self._latency_collector: Optional[Callable[[], Dict[str, Any]]] = None
        self._throughput_collector: Optional[Callable[[], Dict[str, Any]]] = None
        self._system_collector: Optional[Callable[[], Dict[str, Any]]] = None
        self._alert_collector: Optional[Callable[[], Dict[str, int]]] = None
        self._order_collector: Optional[Callable[[], Dict[str, int]]] = None

    def set_latency_collector(
        self, collector: Callable[[], Dict[str, Any]]
    ) -> None:
        """Set the latency metrics collector."""
        self._latency_collector = collector

    def set_throughput_collector(
        self, collector: Callable[[], Dict[str, Any]]
    ) -> None:
        """Set the throughput metrics collector."""
        self._throughput_collector = collector

    def set_system_collector(
        self, collector: Callable[[], Dict[str, Any]]
    ) -> None:
        """Set the system metrics collector."""
        self._system_collector = collector

    def set_alert_collector(
        self, collector: Callable[[], Dict[str, int]]
    ) -> None:
        """Set the alert count collector."""
        self._alert_collector = collector

    def set_order_collector(
        self, collector: Callable[[], Dict[str, int]]
    ) -> None:
        """Set the order status collector."""
        self._order_collector = collector

    async def subscribe(
        self,
        subscriber: StreamSubscriber,
        event_types: Optional[List[StreamEventType]] = None,
    ) -> str:
        """
        Subscribe to metrics stream.

        Args:
            subscriber: The subscriber connection
            event_types: Event types to subscribe to (all if None)

        Returns:
            Subscription ID
        """
        async with self._lock:
            self._subscribers.add(subscriber)

            # Generate subscription ID
            sub_id = f"sub_{id(subscriber)}_{int(time.time())}"

            # Store subscription preferences
            if event_types:
                self._subscriptions[sub_id] = set(event_types)
            else:
                self._subscriptions[sub_id] = set(StreamEventType)

            logger.info(f"New subscriber: {sub_id}")
            return sub_id

    async def unsubscribe(self, subscriber: StreamSubscriber) -> None:
        """
        Unsubscribe from metrics stream.

        Args:
            subscriber: The subscriber to remove
        """
        async with self._lock:
            self._subscribers.discard(subscriber)

            # Clean up subscriptions
            sub_id = f"sub_{id(subscriber)}"
            keys_to_remove = [
                k for k in self._subscriptions if k.startswith(sub_id)
            ]
            for key in keys_to_remove:
                del self._subscriptions[key]

    async def broadcast(self, event: StreamEvent) -> int:
        """
        Broadcast event to all subscribers.

        Args:
            event: Event to broadcast

        Returns:
            Number of successful deliveries
        """
        message = event.to_json()
        delivered = 0
        disconnected: List[StreamSubscriber] = []

        async with self._lock:
            for subscriber in self._subscribers:
                if not subscriber.is_connected:
                    disconnected.append(subscriber)
                    continue

                try:
                    await subscriber.send(message)
                    delivered += 1
                except Exception as e:
                    logger.debug(f"Failed to send to subscriber: {e}")
                    disconnected.append(subscriber)

            # Clean up disconnected
            for sub in disconnected:
                self._subscribers.discard(sub)

        return delivered

    async def push_metrics(self, snapshot: MetricsSnapshot) -> None:
        """
        Push metrics snapshot to subscribers.

        Args:
            snapshot: Metrics snapshot to push
        """
        event = StreamEvent(
            event_type=StreamEventType.METRICS_UPDATE,
            data=snapshot.to_dict(),
        )
        await self.broadcast(event)

    async def push_alert(
        self,
        alert_data: Dict[str, Any],
        resolved: bool = False,
    ) -> None:
        """
        Push alert event to subscribers.

        Args:
            alert_data: Alert data
            resolved: Whether this is a resolution
        """
        event_type = (
            StreamEventType.ALERT_RESOLVED if resolved
            else StreamEventType.ALERT_FIRED
        )
        event = StreamEvent(
            event_type=event_type,
            data=alert_data,
        )
        await self.broadcast(event)

    async def push_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Push trade event to subscribers.

        Args:
            trade_data: Trade execution data
        """
        event = StreamEvent(
            event_type=StreamEventType.TRADE_EXECUTED,
            data=trade_data,
        )
        await self.broadcast(event)

    async def push_order_update(self, order_data: Dict[str, Any]) -> None:
        """
        Push order update to subscribers.

        Args:
            order_data: Order update data
        """
        event = StreamEvent(
            event_type=StreamEventType.ORDER_UPDATE,
            data=order_data,
        )
        await self.broadcast(event)

    async def start(self) -> None:
        """Start the metrics streaming loop."""
        if self._running:
            return

        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("Metrics streamer started")

    async def stop(self) -> None:
        """Stop the metrics streaming loop."""
        self._running = False

        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics streamer stopped")

    async def _update_loop(self) -> None:
        """Main update loop."""
        while self._running:
            try:
                if self._subscribers:
                    snapshot = self._collect_snapshot()
                    await self.push_metrics(snapshot)

                await asyncio.sleep(self._update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1.0)

    async def _heartbeat_loop(self) -> None:
        """Heartbeat loop to keep connections alive."""
        while self._running:
            try:
                event = StreamEvent(
                    event_type=StreamEventType.HEARTBEAT,
                    data={"timestamp": datetime.now(timezone.utc).isoformat()},
                )
                await self.broadcast(event)
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(5.0)

    def _collect_snapshot(self) -> MetricsSnapshot:
        """Collect current metrics snapshot."""
        latency = {}
        if self._latency_collector:
            try:
                latency = self._latency_collector()
            except Exception as e:
                logger.error(f"Latency collector error: {e}")

        throughput = {}
        if self._throughput_collector:
            try:
                throughput = self._throughput_collector()
            except Exception as e:
                logger.error(f"Throughput collector error: {e}")

        system = {}
        if self._system_collector:
            try:
                system = self._system_collector()
            except Exception as e:
                logger.error(f"System collector error: {e}")

        alerts: Dict[str, int] = {}
        if self._alert_collector:
            try:
                alerts = self._alert_collector()
            except Exception as e:
                logger.error(f"Alert collector error: {e}")

        orders: Dict[str, int] = {}
        if self._order_collector:
            try:
                orders = self._order_collector()
            except Exception as e:
                logger.error(f"Order collector error: {e}")

        return MetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            latency=latency,
            throughput=throughput,
            system=system,
            alerts=alerts,
            orders=orders,
        )

    @property
    def subscriber_count(self) -> int:
        """Get current subscriber count."""
        return len(self._subscribers)

    def get_stats(self) -> Dict[str, Any]:
        """Get streamer statistics."""
        return {
            "running": self._running,
            "subscriber_count": self.subscriber_count,
            "update_interval": self._update_interval,
            "heartbeat_interval": self._heartbeat_interval,
        }


class StreamIntegrator:
    """
    Integrates all monitoring components for streaming.

    Connects profiler, alert manager, and other sources to the streamer.
    """

    def __init__(
        self,
        streamer: MetricsStreamer,
    ):
        """
        Initialize the integrator.

        Args:
            streamer: Metrics streamer instance
        """
        self._streamer = streamer

    def connect_profiler(self, profiler: Any) -> None:
        """
        Connect a PerformanceProfiler to the streamer.

        Args:
            profiler: PerformanceProfiler instance
        """

        def latency_collector() -> Dict[str, Any]:
            result = {}
            for op, stats in profiler.latency.get_all_stats().items():
                result[op] = {
                    "p50": stats.p50_ms,
                    "p95": stats.p95_ms,
                    "p99": stats.p99_ms,
                    "avg": stats.avg_ms,
                }
            return result

        def throughput_collector() -> Dict[str, Any]:
            result = {}
            for op, stats in profiler.throughput.get_all_stats().items():
                result[op] = {
                    "current_rps": stats.current_rps,
                    "avg_rps": stats.avg_rps,
                    "peak_rps": stats.peak_rps,
                }
            return result

        def system_collector() -> Dict[str, Any]:
            metrics = profiler.get_metrics()
            return {
                "memory_mb": metrics.memory_mb,
                "cpu_percent": metrics.cpu_percent,
                "active_tasks": metrics.active_tasks,
                "open_connections": metrics.open_connections,
            }

        self._streamer.set_latency_collector(latency_collector)
        self._streamer.set_throughput_collector(throughput_collector)
        self._streamer.set_system_collector(system_collector)

    def connect_alert_manager(self, alert_manager: Any) -> None:
        """
        Connect an AlertManager to the streamer.

        Args:
            alert_manager: AlertManager instance
        """
        from .alerts import AlertSeverity

        async def on_alert(alert: Any) -> None:
            await self._streamer.push_alert(alert.to_dict())

        alert_manager.on_alert(on_alert)

        def alert_collector() -> Dict[str, int]:
            # Would need async support or cached counts
            return {}

        self._streamer.set_alert_collector(alert_collector)
