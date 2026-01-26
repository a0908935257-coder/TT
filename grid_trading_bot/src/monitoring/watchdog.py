"""
Monitoring System Watchdog.

Provides self-monitoring capabilities to ensure the monitoring system
itself remains operational. Includes:
- Watchdog heartbeat to detect monitoring failures
- Dead letter queue for undelivered alerts
- Self-health checks for all monitoring components
- External health endpoint for external monitoring systems
"""

import asyncio
import os
import signal
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class ComponentStatus(Enum):
    """Health status for monitoring components."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ComponentHealth:
    """Health status of a single component."""

    name: str
    status: ComponentStatus
    last_check: datetime
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "message": self.message,
            "latency_ms": self.latency_ms,
            "details": self.details,
        }


@dataclass
class DeadLetterAlert:
    """Alert that failed to deliver after all retries."""

    alert_id: str
    title: str
    message: str
    severity: str
    source: str
    channels_failed: List[str]
    first_attempt: datetime
    last_attempt: datetime
    attempt_count: int
    error_messages: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity,
            "source": self.source,
            "channels_failed": self.channels_failed,
            "first_attempt": self.first_attempt.isoformat(),
            "last_attempt": self.last_attempt.isoformat(),
            "attempt_count": self.attempt_count,
            "error_messages": self.error_messages,
        }


class DeadLetterQueue:
    """
    Persistent queue for alerts that failed to deliver.

    Stores failed alerts for later analysis and manual retry.
    """

    def __init__(
        self,
        max_size: int = 1000,
        retention_hours: int = 72,
    ):
        """
        Initialize dead letter queue.

        Args:
            max_size: Maximum number of dead letters to retain
            retention_hours: Hours to retain dead letters
        """
        self._max_size = max_size
        self._retention_hours = retention_hours
        self._queue: Deque[DeadLetterAlert] = deque(maxlen=max_size)
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_added": 0,
            "total_replayed": 0,
            "total_expired": 0,
        }

    async def add(
        self,
        alert_id: str,
        title: str,
        message: str,
        severity: str,
        source: str,
        channels_failed: List[str],
        attempt_count: int,
        error_messages: List[str],
    ) -> None:
        """Add a dead letter alert."""
        async with self._lock:
            now = datetime.now(timezone.utc)

            # Check if this alert already exists
            for dl in self._queue:
                if dl.alert_id == alert_id:
                    # Update existing
                    dl.last_attempt = now
                    dl.attempt_count = attempt_count
                    dl.channels_failed = channels_failed
                    dl.error_messages.extend(error_messages)
                    return

            # Add new
            dead_letter = DeadLetterAlert(
                alert_id=alert_id,
                title=title,
                message=message,
                severity=severity,
                source=source,
                channels_failed=channels_failed,
                first_attempt=now,
                last_attempt=now,
                attempt_count=attempt_count,
                error_messages=error_messages,
            )
            self._queue.append(dead_letter)
            self._stats["total_added"] += 1

            logger.warning(
                f"Alert added to dead letter queue: {alert_id} "
                f"(failed channels: {channels_failed})"
            )

    async def get_all(self) -> List[DeadLetterAlert]:
        """Get all dead letters."""
        async with self._lock:
            self._cleanup_expired()
            return list(self._queue)

    async def remove(self, alert_id: str) -> bool:
        """Remove a dead letter by ID."""
        async with self._lock:
            for i, dl in enumerate(self._queue):
                if dl.alert_id == alert_id:
                    del self._queue[i]
                    self._stats["total_replayed"] += 1
                    return True
            return False

    def _cleanup_expired(self) -> int:
        """Remove expired dead letters."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._retention_hours)
        expired = 0

        while self._queue and self._queue[0].last_attempt < cutoff:
            self._queue.popleft()
            expired += 1

        self._stats["total_expired"] += expired
        return expired

    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            **self._stats,
            "current_size": len(self._queue),
            "max_size": self._max_size,
        }


class MonitoringWatchdog:
    """
    Watchdog for monitoring the monitoring system.

    Performs self-health checks and sends heartbeats to ensure
    the monitoring infrastructure is operational.
    """

    def __init__(
        self,
        heartbeat_interval: int = 30,
        health_check_interval: int = 60,
        stale_threshold: int = 120,
    ):
        """
        Initialize the watchdog.

        Args:
            heartbeat_interval: Seconds between heartbeats
            health_check_interval: Seconds between health checks
            stale_threshold: Seconds before component is considered stale
        """
        self._heartbeat_interval = heartbeat_interval
        self._health_check_interval = health_check_interval
        self._stale_threshold = stale_threshold

        # Component health trackers
        self._component_checks: Dict[
            str, Callable[[], Coroutine[Any, Any, ComponentHealth]]
        ] = {}
        self._last_health: Dict[str, ComponentHealth] = {}

        # Heartbeat tracking
        self._last_heartbeat: Optional[datetime] = None
        self._heartbeat_file: Optional[str] = None

        # Dead letter queue
        self._dlq = DeadLetterQueue()

        # Running state
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None

        # External notification callback for watchdog alerts
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

        # Statistics
        self._stats = {
            "heartbeats_sent": 0,
            "health_checks_performed": 0,
            "unhealthy_detections": 0,
            "self_recovery_attempts": 0,
        }

    def register_component(
        self,
        name: str,
        check_func: Callable[[], Coroutine[Any, Any, ComponentHealth]],
    ) -> None:
        """
        Register a component for health monitoring.

        Args:
            name: Component name
            check_func: Async function that returns ComponentHealth
        """
        self._component_checks[name] = check_func
        logger.info(f"Registered watchdog check for: {name}")

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """
        Set callback for watchdog alerts.

        Args:
            callback: Async function(severity, title, message)
        """
        self._alert_callback = callback

    def set_heartbeat_file(self, path: str) -> None:
        """
        Set path for heartbeat file.

        External systems can monitor this file's mtime.

        Args:
            path: Path to heartbeat file
        """
        self._heartbeat_file = path

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """Get the dead letter queue."""
        return self._dlq

    async def start(self) -> None:
        """Start the watchdog."""
        if self._running:
            return

        self._running = True

        # Start heartbeat task
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Start health check task
        self._health_check_task = asyncio.create_task(self._health_check_loop())

        logger.info(
            f"Monitoring watchdog started "
            f"(heartbeat={self._heartbeat_interval}s, "
            f"health_check={self._health_check_interval}s)"
        )

    async def stop(self) -> None:
        """Stop the watchdog."""
        self._running = False

        tasks = [self._heartbeat_task, self._health_check_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Monitoring watchdog stopped")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)

    async def _send_heartbeat(self) -> None:
        """Send a single heartbeat."""
        self._last_heartbeat = datetime.now(timezone.utc)
        self._stats["heartbeats_sent"] += 1

        # Update heartbeat file if configured
        if self._heartbeat_file:
            try:
                # Touch the file to update mtime
                with open(self._heartbeat_file, "w") as f:
                    f.write(self._last_heartbeat.isoformat())
            except Exception as e:
                logger.error(f"Failed to update heartbeat file: {e}")

        logger.debug(f"Watchdog heartbeat sent at {self._last_heartbeat}")

    async def _health_check_loop(self) -> None:
        """Perform periodic health checks."""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self._health_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)

    async def _perform_health_checks(self) -> None:
        """Run all registered health checks."""
        self._stats["health_checks_performed"] += 1
        unhealthy_components = []

        for name, check_func in self._component_checks.items():
            try:
                start = time.time()
                health = await asyncio.wait_for(check_func(), timeout=30)
                health.latency_ms = (time.time() - start) * 1000
                self._last_health[name] = health

                if health.status == ComponentStatus.UNHEALTHY:
                    unhealthy_components.append(name)

            except asyncio.TimeoutError:
                self._last_health[name] = ComponentHealth(
                    name=name,
                    status=ComponentStatus.UNHEALTHY,
                    last_check=datetime.now(timezone.utc),
                    message="Health check timed out (>30s)",
                )
                unhealthy_components.append(name)

            except Exception as e:
                self._last_health[name] = ComponentHealth(
                    name=name,
                    status=ComponentStatus.UNHEALTHY,
                    last_check=datetime.now(timezone.utc),
                    message=f"Health check failed: {e}",
                )
                unhealthy_components.append(name)

        # Send alerts for unhealthy components
        if unhealthy_components:
            self._stats["unhealthy_detections"] += 1
            await self._send_watchdog_alert(unhealthy_components)

    async def _send_watchdog_alert(self, unhealthy: List[str]) -> None:
        """Send alert about unhealthy components."""
        if not self._alert_callback:
            logger.error(
                f"WATCHDOG ALERT: Unhealthy components detected: {unhealthy} "
                "(no alert callback configured)"
            )
            return

        try:
            details = []
            for name in unhealthy:
                health = self._last_health.get(name)
                if health:
                    details.append(f"- {name}: {health.message}")

            message = (
                f"監控系統偵測到 {len(unhealthy)} 個組件異常:\n"
                + "\n".join(details)
            )

            await self._alert_callback(
                "CRITICAL",
                "監控系統組件異常",
                message,
            )
        except Exception as e:
            # If we can't send alerts, log to stderr as last resort
            import sys

            print(
                f"CRITICAL: Watchdog cannot send alert: {e}. "
                f"Unhealthy: {unhealthy}",
                file=sys.stderr,
            )

    async def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall health status.

        Returns:
            Health status dictionary suitable for HTTP endpoint
        """
        # Check if we're stale
        is_stale = False
        if self._last_heartbeat:
            age = (datetime.now(timezone.utc) - self._last_heartbeat).total_seconds()
            is_stale = age > self._stale_threshold

        # Determine overall status
        statuses = [h.status for h in self._last_health.values()]
        if not statuses or is_stale:
            overall = ComponentStatus.UNKNOWN
        elif any(s == ComponentStatus.UNHEALTHY for s in statuses):
            overall = ComponentStatus.UNHEALTHY
        elif any(s == ComponentStatus.DEGRADED for s in statuses):
            overall = ComponentStatus.DEGRADED
        else:
            overall = ComponentStatus.HEALTHY

        return {
            "status": overall.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": self._last_heartbeat.isoformat()
            if self._last_heartbeat
            else None,
            "is_stale": is_stale,
            "components": {
                name: health.to_dict() for name, health in self._last_health.items()
            },
            "dead_letter_queue": self._dlq.get_stats(),
            "stats": self._stats,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get watchdog statistics."""
        return {
            **self._stats,
            "components_monitored": len(self._component_checks),
            "dlq": self._dlq.get_stats(),
        }


# =============================================================================
# Standard Health Check Functions
# =============================================================================


async def check_alert_manager_health(manager: Any) -> ComponentHealth:
    """
    Health check for AlertManager.

    Args:
        manager: AlertManager instance

    Returns:
        ComponentHealth status
    """
    try:
        stats = manager.get_stats()

        # Check for concerning patterns
        issues = []

        # Check rate limiter stats
        rate_stats = stats.get("rate_limiter", {})
        if rate_stats.get("total_rate_limited", 0) > 100:
            issues.append("High rate limiting activity")

        # Check retry queue
        retry_stats = stats.get("retry_queue", {})
        if retry_stats.get("pending", 0) > 50:
            issues.append(f"Large retry queue: {retry_stats['pending']}")

        if retry_stats.get("total_exhausted", 0) > 10:
            issues.append(f"Many alerts exhausted retries: {retry_stats['total_exhausted']}")

        # Determine status
        if issues:
            return ComponentHealth(
                name="alert_manager",
                status=ComponentStatus.DEGRADED,
                last_check=datetime.now(timezone.utc),
                message="; ".join(issues),
                details=stats,
            )

        return ComponentHealth(
            name="alert_manager",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="Operating normally",
            details=stats,
        )

    except Exception as e:
        return ComponentHealth(
            name="alert_manager",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.now(timezone.utc),
            message=f"Health check failed: {e}",
        )


async def check_notification_channel_health(
    channel_name: str,
    test_func: Callable[[], Coroutine[Any, Any, bool]],
) -> ComponentHealth:
    """
    Health check for a notification channel.

    Args:
        channel_name: Name of the channel
        test_func: Async function that tests the channel

    Returns:
        ComponentHealth status
    """
    try:
        start = time.time()
        success = await test_func()
        latency = (time.time() - start) * 1000

        if success:
            return ComponentHealth(
                name=f"channel_{channel_name}",
                status=ComponentStatus.HEALTHY,
                last_check=datetime.now(timezone.utc),
                message="Channel responding",
                latency_ms=latency,
            )
        else:
            return ComponentHealth(
                name=f"channel_{channel_name}",
                status=ComponentStatus.UNHEALTHY,
                last_check=datetime.now(timezone.utc),
                message="Channel test failed",
                latency_ms=latency,
            )

    except Exception as e:
        return ComponentHealth(
            name=f"channel_{channel_name}",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.now(timezone.utc),
            message=f"Channel error: {e}",
        )


async def check_database_health(db_path: str) -> ComponentHealth:
    """
    Health check for SQLite database.

    Args:
        db_path: Path to database file

    Returns:
        ComponentHealth status
    """
    try:
        import aiosqlite

        start = time.time()
        async with aiosqlite.connect(db_path) as db:
            # Simple query to verify database is accessible
            async with db.execute("SELECT 1") as cursor:
                await cursor.fetchone()

        latency = (time.time() - start) * 1000

        # Check database file size
        file_size_mb = os.path.getsize(db_path) / (1024 * 1024) if os.path.exists(db_path) else 0

        return ComponentHealth(
            name="database",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="Database accessible",
            latency_ms=latency,
            details={"file_size_mb": round(file_size_mb, 2)},
        )

    except Exception as e:
        return ComponentHealth(
            name="database",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.now(timezone.utc),
            message=f"Database error: {e}",
        )


async def check_redis_health(redis_client: Any) -> ComponentHealth:
    """
    Health check for Redis connection.

    Args:
        redis_client: Redis async client

    Returns:
        ComponentHealth status
    """
    try:
        start = time.time()
        await redis_client.ping()
        latency = (time.time() - start) * 1000

        # Get Redis info
        info = await redis_client.info("memory")
        used_memory_mb = info.get("used_memory", 0) / (1024 * 1024)

        return ComponentHealth(
            name="redis",
            status=ComponentStatus.HEALTHY,
            last_check=datetime.now(timezone.utc),
            message="Redis responding",
            latency_ms=latency,
            details={"used_memory_mb": round(used_memory_mb, 2)},
        )

    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=ComponentStatus.UNHEALTHY,
            last_check=datetime.now(timezone.utc),
            message=f"Redis error: {e}",
        )


# =============================================================================
# Global Watchdog Instance
# =============================================================================

_watchdog: Optional[MonitoringWatchdog] = None


def get_watchdog() -> MonitoringWatchdog:
    """Get the global watchdog instance."""
    global _watchdog
    if _watchdog is None:
        _watchdog = MonitoringWatchdog()
    return _watchdog


async def init_watchdog(
    alert_manager: Any = None,
    redis_client: Any = None,
    db_path: Optional[str] = None,
    heartbeat_file: Optional[str] = None,
) -> MonitoringWatchdog:
    """
    Initialize and start the global watchdog.

    Args:
        alert_manager: AlertManager instance
        redis_client: Redis client
        db_path: Path to SQLite database
        heartbeat_file: Path for heartbeat file

    Returns:
        Initialized watchdog
    """
    watchdog = get_watchdog()

    # Set heartbeat file if provided
    if heartbeat_file:
        watchdog.set_heartbeat_file(heartbeat_file)

    # Register alert manager health check
    if alert_manager:
        watchdog.register_component(
            "alert_manager",
            lambda: check_alert_manager_health(alert_manager),
        )

    # Register Redis health check
    if redis_client:
        watchdog.register_component(
            "redis",
            lambda: check_redis_health(redis_client),
        )

    # Register database health check
    if db_path:
        watchdog.register_component(
            "database",
            lambda: check_database_health(db_path),
        )

    await watchdog.start()
    return watchdog


async def stop_watchdog() -> None:
    """Stop the global watchdog."""
    global _watchdog
    if _watchdog:
        await _watchdog.stop()
        _watchdog = None
