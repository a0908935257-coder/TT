"""
Alert management module.

Provides alert persistence, deduplication, routing, and escalation
for production trading systems.

Features:
- Global rate limiting with priority bypass for critical alerts
- Alert aggregation to prevent storm
- Parallel channel sending for reduced latency
- Automatic retry with exponential backoff
"""

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Rate Limiting and Throttling
# =============================================================================


class AlertRateLimiter:
    """
    Global rate limiter with priority bypass for critical alerts.

    Prevents alert storms while ensuring critical alerts are never blocked.
    """

    def __init__(
        self,
        max_per_minute: int = 25,
        critical_bypass: bool = True,
        burst_limit: int = 10,
        burst_window_seconds: int = 5,
    ):
        """
        Initialize rate limiter.

        Args:
            max_per_minute: Maximum alerts per minute for non-critical
            critical_bypass: Allow CRITICAL alerts to bypass rate limit
            burst_limit: Maximum alerts in burst window
            burst_window_seconds: Burst detection window
        """
        self._max_per_minute = max_per_minute
        self._critical_bypass = critical_bypass
        self._burst_limit = burst_limit
        self._burst_window = burst_window_seconds

        # Sliding window for rate limiting
        self._window: Deque[Tuple[datetime, "AlertSeverity"]] = deque()
        self._lock = asyncio.Lock()

        # Statistics
        self._stats = {
            "total_allowed": 0,
            "total_rate_limited": 0,
            "total_burst_limited": 0,
            "critical_bypassed": 0,
        }

    async def should_allow(self, severity: "AlertSeverity") -> bool:
        """
        Check if alert should be allowed based on rate limits.

        Args:
            severity: Alert severity level

        Returns:
            True if alert should be sent
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            minute_ago = now - timedelta(minutes=1)
            burst_ago = now - timedelta(seconds=self._burst_window)

            # Clean old entries
            while self._window and self._window[0][0] < minute_ago:
                self._window.popleft()

            # Critical alerts always bypass (if enabled)
            if self._critical_bypass and severity == AlertSeverity.CRITICAL:
                self._window.append((now, severity))
                self._stats["critical_bypassed"] += 1
                self._stats["total_allowed"] += 1
                return True

            # Check burst limit
            burst_count = sum(1 for t, _ in self._window if t >= burst_ago)
            if burst_count >= self._burst_limit:
                self._stats["total_burst_limited"] += 1
                logger.warning(
                    f"Alert burst limited: {burst_count} alerts in {self._burst_window}s"
                )
                return False

            # Check per-minute limit
            if len(self._window) >= self._max_per_minute:
                self._stats["total_rate_limited"] += 1
                logger.warning(
                    f"Alert rate limited: {len(self._window)} alerts in last minute"
                )
                return False

            self._window.append((now, severity))
            self._stats["total_allowed"] += 1
            return True

    def get_stats(self) -> Dict[str, int]:
        """Get rate limiter statistics."""
        return self._stats.copy()

    def reset(self) -> None:
        """Reset rate limiter state."""
        self._window.clear()


class AlertAggregator:
    """
    Aggregates similar alerts to reduce notification noise.

    Groups alerts by fingerprint and sends periodic summaries.
    """

    def __init__(
        self,
        aggregation_window_seconds: int = 60,
        min_count_for_aggregation: int = 3,
    ):
        """
        Initialize aggregator.

        Args:
            aggregation_window_seconds: Window for collecting similar alerts
            min_count_for_aggregation: Minimum count before aggregating
        """
        self._window_seconds = aggregation_window_seconds
        self._min_count = min_count_for_aggregation

        # fingerprint -> list of (timestamp, alert_data)
        self._pending: Dict[str, List[Tuple[datetime, Dict[str, Any]]]] = defaultdict(
            list
        )
        self._lock = asyncio.Lock()

    async def add_or_aggregate(
        self,
        fingerprint: str,
        alert_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        Add alert to aggregator.

        Returns:
            Aggregated alert if threshold reached, None if still collecting
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self._window_seconds)

            # Clean old entries for this fingerprint
            self._pending[fingerprint] = [
                (t, d) for t, d in self._pending[fingerprint] if t >= cutoff
            ]

            # Add new alert
            self._pending[fingerprint].append((now, alert_data))

            # Check if we should aggregate
            count = len(self._pending[fingerprint])
            if count >= self._min_count:
                # Create aggregated alert
                alerts = self._pending[fingerprint]
                aggregated = self._create_aggregated_alert(fingerprint, alerts)

                # Clear pending
                self._pending[fingerprint] = []

                return aggregated

            return None

    def _create_aggregated_alert(
        self,
        fingerprint: str,
        alerts: List[Tuple[datetime, Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Create an aggregated alert summary."""
        count = len(alerts)
        first = alerts[0][1]
        last = alerts[-1][1]

        return {
            "title": f"[聚合 x{count}] {first.get('title', 'Alert')}",
            "message": (
                f"過去 {self._window_seconds} 秒內發生 {count} 次相同告警。\n"
                f"首次: {alerts[0][0].strftime('%H:%M:%S')}\n"
                f"最近: {alerts[-1][0].strftime('%H:%M:%S')}\n"
                f"原始訊息: {first.get('message', '')[:200]}"
            ),
            "severity": last.get("severity"),
            "source": first.get("source"),
            "fingerprint": fingerprint,
            "aggregated_count": count,
            "first_at": alerts[0][0],
            "last_at": alerts[-1][0],
        }

    async def flush_all(self) -> List[Dict[str, Any]]:
        """Flush all pending aggregations."""
        async with self._lock:
            results = []
            for fingerprint, alerts in self._pending.items():
                if alerts:
                    results.append(
                        self._create_aggregated_alert(fingerprint, alerts)
                    )
            self._pending.clear()
            return results

    def get_pending_count(self) -> int:
        """Get count of pending alerts across all fingerprints."""
        return sum(len(v) for v in self._pending.values())


class AlertSeverity(Enum):
    """Alert severity levels."""

    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertState(Enum):
    """Alert lifecycle states."""

    FIRING = "firing"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


class AlertChannel(Enum):
    """Alert notification channels."""

    LOG = "log"
    DISCORD = "discord"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"


@dataclass
class AlertRule:
    """
    Rule for triggering alerts based on conditions.

    Defines when an alert should fire, its severity, and routing.
    """

    name: str
    condition: Callable[..., bool]
    severity: AlertSeverity
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    cooldown_seconds: int = 300
    escalation_after_minutes: int = 30
    auto_resolve: bool = False
    labels: Dict[str, str] = field(default_factory=dict)
    description: str = ""

    def __post_init__(self) -> None:
        self._last_fired: Optional[datetime] = None

    def can_fire(self) -> bool:
        """Check if rule can fire (respects cooldown)."""
        if self._last_fired is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._last_fired).total_seconds()
        return elapsed >= self.cooldown_seconds

    def mark_fired(self) -> None:
        """Mark rule as fired."""
        self._last_fired = datetime.now(timezone.utc)


@dataclass
class PersistedAlert:
    """
    Alert with persistence support.

    Tracks full lifecycle including firing, acknowledgment, and resolution.
    """

    alert_id: str
    rule_name: str
    severity: AlertSeverity
    state: AlertState
    title: str
    message: str
    source: str
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    last_notified_at: Optional[datetime] = None

    # Tracking
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    notification_count: int = 0
    fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.fingerprint:
            self.fingerprint = self._compute_fingerprint()

    def _compute_fingerprint(self) -> str:
        """Compute unique fingerprint for deduplication."""
        data = f"{self.rule_name}:{self.source}:{json.dumps(self.labels, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def acknowledge(self, by: Optional[str] = None) -> None:
        """Acknowledge the alert."""
        self.state = AlertState.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = by

    def resolve(self, by: Optional[str] = None) -> None:
        """Resolve the alert."""
        self.state = AlertState.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.resolved_by = by

    def suppress(self) -> None:
        """Suppress the alert."""
        self.state = AlertState.SUPPRESSED

    @property
    def duration_seconds(self) -> float:
        """Get alert duration in seconds."""
        end = self.resolved_at or datetime.now(timezone.utc)
        return (end - self.fired_at).total_seconds()

    @property
    def is_active(self) -> bool:
        """Check if alert is still active."""
        return self.state in (AlertState.FIRING, AlertState.ACKNOWLEDGED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "rule_name": self.rule_name,
            "severity": self.severity.name,
            "state": self.state.value,
            "title": self.title,
            "message": self.message,
            "source": self.source,
            "labels": self.labels,
            "annotations": self.annotations,
            "fired_at": self.fired_at.isoformat(),
            "acknowledged_at": self.acknowledged_at.isoformat()
            if self.acknowledged_at
            else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_by": self.acknowledged_by,
            "resolved_by": self.resolved_by,
            "notification_count": self.notification_count,
            "fingerprint": self.fingerprint,
            "duration_seconds": self.duration_seconds,
        }


class AlertDeduplicator:
    """
    Deduplicates alerts based on fingerprints.

    Prevents alert storms by grouping similar alerts.
    """

    def __init__(
        self,
        window_seconds: int = 3600,
        max_per_fingerprint: int = 1,
    ):
        """
        Initialize the deduplicator.

        Args:
            window_seconds: Deduplication window
            max_per_fingerprint: Max alerts per fingerprint in window
        """
        self._window_seconds = window_seconds
        self._max_per_fingerprint = max_per_fingerprint
        self._seen: Dict[str, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def should_dedupe(self, fingerprint: str) -> bool:
        """
        Check if alert should be deduplicated.

        Args:
            fingerprint: Alert fingerprint

        Returns:
            True if alert should be suppressed
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            cutoff = now - timedelta(seconds=self._window_seconds)

            # Clean old entries
            self._seen[fingerprint] = [
                t for t in self._seen[fingerprint] if t > cutoff
            ]

            # Check count
            if len(self._seen[fingerprint]) >= self._max_per_fingerprint:
                return True

            # Record this occurrence
            self._seen[fingerprint].append(now)
            return False

    def clear(self) -> None:
        """Clear deduplication state."""
        self._seen.clear()


class AlertChannelHandler(ABC):
    """Abstract base for alert channel handlers."""

    @abstractmethod
    async def send(self, alert: PersistedAlert) -> bool:
        """
        Send alert through this channel.

        Args:
            alert: The alert to send

        Returns:
            True if sent successfully
        """
        pass


class LogChannelHandler(AlertChannelHandler):
    """Handler for logging alerts."""

    async def send(self, alert: PersistedAlert) -> bool:
        """Log the alert."""
        level_map = {
            AlertSeverity.DEBUG: logger.debug,
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.ERROR: logger.error,
            AlertSeverity.CRITICAL: logger.critical,
        }
        log_func = level_map.get(alert.severity, logger.info)
        log_func(
            f"[ALERT:{alert.state.value}] {alert.title} - {alert.message} "
            f"(source={alert.source}, fingerprint={alert.fingerprint})"
        )
        return True


class WebhookChannelHandler(AlertChannelHandler):
    """Handler for webhook notifications."""

    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        self._url = url
        self._headers = headers or {}

    async def send(self, alert: PersistedAlert) -> bool:
        """Send alert to webhook."""
        try:
            import aiohttp

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url,
                    json=alert.to_dict(),
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    return response.status < 400
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False


class AlertRetryQueue:
    """
    Manages failed alerts for retry with exponential backoff.
    """

    def __init__(
        self,
        max_retries: int = 3,
        base_delay_seconds: float = 5.0,
        max_delay_seconds: float = 60.0,
    ):
        """
        Initialize retry queue.

        Args:
            max_retries: Maximum retry attempts
            base_delay_seconds: Initial retry delay
            max_delay_seconds: Maximum retry delay
        """
        self._max_retries = max_retries
        self._base_delay = base_delay_seconds
        self._max_delay = max_delay_seconds

        # Queue: (alert, channel, attempt, next_retry_time)
        self._queue: Deque[
            Tuple[PersistedAlert, AlertChannel, int, datetime]
        ] = deque()
        self._lock = asyncio.Lock()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Statistics
        self._stats = {
            "total_retried": 0,
            "total_success_after_retry": 0,
            "total_exhausted": 0,
        }

    async def add(
        self,
        alert: PersistedAlert,
        channel: AlertChannel,
    ) -> None:
        """Add failed alert to retry queue."""
        async with self._lock:
            next_retry = datetime.now(timezone.utc) + timedelta(
                seconds=self._base_delay
            )
            self._queue.append((alert, channel, 1, next_retry))
            logger.debug(f"Added alert {alert.alert_id} to retry queue for {channel}")

    async def process_due(
        self,
        handler_func: Callable[[PersistedAlert, AlertChannel], Coroutine[Any, Any, bool]],
    ) -> int:
        """
        Process due retries.

        Args:
            handler_func: Function to call for retry

        Returns:
            Number of successful retries
        """
        async with self._lock:
            now = datetime.now(timezone.utc)
            success_count = 0
            remaining: Deque[
                Tuple[PersistedAlert, AlertChannel, int, datetime]
            ] = deque()

            while self._queue:
                alert, channel, attempt, next_retry = self._queue.popleft()

                if next_retry > now:
                    # Not due yet, put back
                    remaining.append((alert, channel, attempt, next_retry))
                    continue

                self._stats["total_retried"] += 1

                try:
                    success = await handler_func(alert, channel)
                    if success:
                        self._stats["total_success_after_retry"] += 1
                        success_count += 1
                        logger.info(
                            f"Retry success for alert {alert.alert_id} on {channel}"
                        )
                    else:
                        # Retry failed, schedule next attempt
                        if attempt < self._max_retries:
                            delay = min(
                                self._base_delay * (2 ** attempt),
                                self._max_delay,
                            )
                            next_time = now + timedelta(seconds=delay)
                            remaining.append((alert, channel, attempt + 1, next_time))
                        else:
                            self._stats["total_exhausted"] += 1
                            logger.error(
                                f"Alert {alert.alert_id} exhausted retries for {channel}"
                            )

                except Exception as e:
                    logger.error(f"Retry error for {alert.alert_id}: {e}")
                    if attempt < self._max_retries:
                        delay = min(
                            self._base_delay * (2 ** attempt),
                            self._max_delay,
                        )
                        next_time = now + timedelta(seconds=delay)
                        remaining.append((alert, channel, attempt + 1, next_time))

            self._queue = remaining
            return success_count

    def get_stats(self) -> Dict[str, int]:
        """Get retry queue statistics."""
        return {**self._stats, "pending": len(self._queue)}


class AlertRouter:
    """
    Routes alerts to appropriate channels based on rules.

    Supports channel-specific handlers, escalation, parallel sending,
    and automatic retry for failed deliveries.
    """

    # Channel priority for critical alerts (lower = higher priority)
    CHANNEL_PRIORITY: Dict[AlertChannel, int] = {
        AlertChannel.PAGERDUTY: 1,
        AlertChannel.SMS: 2,
        AlertChannel.EMAIL: 3,
        AlertChannel.DISCORD: 4,
        AlertChannel.WEBHOOK: 5,
        AlertChannel.LOG: 10,
    }

    def __init__(
        self,
        enable_retry: bool = True,
        parallel_send: bool = True,
    ):
        """
        Initialize the router.

        Args:
            enable_retry: Enable automatic retry for failed sends
            parallel_send: Send to channels in parallel
        """
        self._handlers: Dict[AlertChannel, AlertChannelHandler] = {
            AlertChannel.LOG: LogChannelHandler(),
        }
        self._escalation_handlers: Dict[AlertSeverity, List[AlertChannel]] = {}
        self._enable_retry = enable_retry
        self._parallel_send = parallel_send
        self._retry_queue = AlertRetryQueue() if enable_retry else None

    def register_handler(
        self,
        channel: AlertChannel,
        handler: AlertChannelHandler,
    ) -> None:
        """Register a channel handler."""
        self._handlers[channel] = handler

    def set_escalation(
        self,
        severity: AlertSeverity,
        channels: List[AlertChannel],
    ) -> None:
        """Set escalation channels for severity level."""
        self._escalation_handlers[severity] = channels

    def _sort_channels_by_priority(
        self,
        channels: List[AlertChannel],
        is_critical: bool,
    ) -> List[AlertChannel]:
        """Sort channels by priority for critical alerts."""
        if not is_critical:
            return channels

        return sorted(
            channels,
            key=lambda c: self.CHANNEL_PRIORITY.get(c, 99),
        )

    async def _send_to_channel(
        self,
        alert: PersistedAlert,
        channel: AlertChannel,
    ) -> Tuple[AlertChannel, bool]:
        """Send alert to a single channel."""
        handler = self._handlers.get(channel)
        if not handler:
            logger.warning(f"No handler for channel: {channel}")
            return (channel, False)

        try:
            success = await handler.send(alert)
            return (channel, success)
        except Exception as e:
            logger.error(f"Error routing to {channel}: {e}")
            return (channel, False)

    async def route(
        self,
        alert: PersistedAlert,
        channels: List[AlertChannel],
    ) -> Dict[AlertChannel, bool]:
        """
        Route alert to specified channels.

        For critical alerts, sends in priority order.
        For non-critical, sends in parallel for speed.

        Args:
            alert: The alert to route
            channels: Channels to send to

        Returns:
            Dict of channel to success status
        """
        is_critical = alert.severity == AlertSeverity.CRITICAL
        sorted_channels = self._sort_channels_by_priority(channels, is_critical)

        results: Dict[AlertChannel, bool] = {}

        if self._parallel_send and not is_critical:
            # Parallel send for non-critical alerts
            tasks = [
                self._send_to_channel(alert, channel)
                for channel in sorted_channels
            ]
            channel_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in channel_results:
                if isinstance(result, Exception):
                    logger.error(f"Channel send exception: {result}")
                    continue
                channel, success = result
                results[channel] = success
        else:
            # Sequential send for critical alerts (priority order)
            for channel in sorted_channels:
                channel, success = await self._send_to_channel(alert, channel)
                results[channel] = success

                # For critical, if high-priority channel succeeds, continue but log
                if is_critical and success and channel in (
                    AlertChannel.PAGERDUTY,
                    AlertChannel.SMS,
                ):
                    logger.info(
                        f"Critical alert {alert.alert_id} delivered to {channel}"
                    )

        # Queue failed sends for retry
        if self._enable_retry and self._retry_queue:
            for channel, success in results.items():
                if not success and channel != AlertChannel.LOG:
                    await self._retry_queue.add(alert, channel)

        return results

    async def escalate(self, alert: PersistedAlert) -> Dict[AlertChannel, bool]:
        """
        Escalate alert based on severity.

        Args:
            alert: The alert to escalate

        Returns:
            Dict of channel to success status
        """
        channels = self._escalation_handlers.get(alert.severity, [])
        if not channels:
            return {}

        return await self.route(alert, channels)

    async def process_retries(self) -> int:
        """
        Process pending retries.

        Returns:
            Number of successful retries
        """
        if not self._retry_queue:
            return 0

        return await self._retry_queue.process_due(self._send_to_channel)

    def get_retry_stats(self) -> Dict[str, int]:
        """Get retry queue statistics."""
        if self._retry_queue:
            return self._retry_queue.get_stats()
        return {}


class AlertStore(ABC):
    """Abstract base for alert persistence."""

    @abstractmethod
    async def save(self, alert: PersistedAlert) -> bool:
        """Save an alert."""
        pass

    @abstractmethod
    async def get(self, alert_id: str) -> Optional[PersistedAlert]:
        """Get an alert by ID."""
        pass

    @abstractmethod
    async def get_active(self) -> List[PersistedAlert]:
        """Get all active alerts."""
        pass

    @abstractmethod
    async def update(self, alert: PersistedAlert) -> bool:
        """Update an alert."""
        pass

    @abstractmethod
    async def get_by_fingerprint(self, fingerprint: str) -> Optional[PersistedAlert]:
        """Get active alert by fingerprint."""
        pass


class InMemoryAlertStore(AlertStore):
    """In-memory alert store for testing/simple deployments."""

    def __init__(self, max_alerts: int = 10000):
        self._alerts: Dict[str, PersistedAlert] = {}
        self._max_alerts = max_alerts
        self._lock = asyncio.Lock()

    async def save(self, alert: PersistedAlert) -> bool:
        """Save an alert."""
        async with self._lock:
            # Prune old resolved alerts if at capacity
            if len(self._alerts) >= self._max_alerts:
                resolved = [
                    a
                    for a in self._alerts.values()
                    if a.state == AlertState.RESOLVED
                ]
                resolved.sort(key=lambda a: a.resolved_at or a.fired_at)
                for old in resolved[: len(resolved) // 2]:
                    del self._alerts[old.alert_id]

            self._alerts[alert.alert_id] = alert
            return True

    async def get(self, alert_id: str) -> Optional[PersistedAlert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)

    async def get_active(self) -> List[PersistedAlert]:
        """Get all active alerts."""
        return [a for a in self._alerts.values() if a.is_active]

    async def update(self, alert: PersistedAlert) -> bool:
        """Update an alert."""
        if alert.alert_id in self._alerts:
            self._alerts[alert.alert_id] = alert
            return True
        return False

    async def get_by_fingerprint(self, fingerprint: str) -> Optional[PersistedAlert]:
        """Get active alert by fingerprint."""
        for alert in self._alerts.values():
            if alert.fingerprint == fingerprint and alert.is_active:
                return alert
        return None

    async def get_history(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> List[PersistedAlert]:
        """Get alert history with filters."""
        alerts = list(self._alerts.values())

        if start:
            alerts = [a for a in alerts if a.fired_at >= start]
        if end:
            alerts = [a for a in alerts if a.fired_at <= end]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        alerts.sort(key=lambda a: a.fired_at, reverse=True)
        return alerts[:limit]


class AlertManager:
    """
    Central alert management system.

    Handles alert lifecycle, deduplication, routing, and persistence.
    Now with enhanced features:
    - Global rate limiting with critical bypass
    - Alert aggregation to prevent storm
    - Parallel channel sending
    - Automatic retry for failed notifications
    """

    def __init__(
        self,
        store: Optional[AlertStore] = None,
        router: Optional[AlertRouter] = None,
        deduplicator: Optional[AlertDeduplicator] = None,
        rate_limiter: Optional[AlertRateLimiter] = None,
        aggregator: Optional[AlertAggregator] = None,
        enable_aggregation: bool = True,
        enable_rate_limiting: bool = True,
    ):
        """
        Initialize the alert manager.

        Args:
            store: Alert persistence store
            router: Alert router
            deduplicator: Alert deduplicator
            rate_limiter: Global rate limiter
            aggregator: Alert aggregator
            enable_aggregation: Enable alert aggregation
            enable_rate_limiting: Enable rate limiting
        """
        self._store = store or InMemoryAlertStore()
        self._router = router or AlertRouter(enable_retry=True, parallel_send=True)
        self._deduplicator = deduplicator or AlertDeduplicator()
        self._rate_limiter = rate_limiter or AlertRateLimiter()
        self._aggregator = aggregator or AlertAggregator()
        self._enable_aggregation = enable_aggregation
        self._enable_rate_limiting = enable_rate_limiting

        self._rules: Dict[str, AlertRule] = {}
        self._callbacks: List[Callable[[PersistedAlert], Coroutine[Any, Any, None]]] = (
            []
        )
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._retry_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        self._suppressed_sources: Set[str] = set()

        # Statistics
        self._stats = {
            "total_fired": 0,
            "total_rate_limited": 0,
            "total_deduplicated": 0,
            "total_aggregated": 0,
            "total_suppressed": 0,
        }

    def register_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self._rules[rule.name] = rule
        logger.info(f"Registered alert rule: {rule.name}")

    def unregister_rule(self, name: str) -> None:
        """Unregister an alert rule."""
        if name in self._rules:
            del self._rules[name]

    def on_alert(
        self, callback: Callable[[PersistedAlert], Coroutine[Any, Any, None]]
    ) -> None:
        """Register alert callback."""
        self._callbacks.append(callback)

    def suppress_source(self, source: str) -> None:
        """Suppress alerts from a source."""
        self._suppressed_sources.add(source)

    def unsuppress_source(self, source: str) -> None:
        """Remove suppression for a source."""
        self._suppressed_sources.discard(source)

    async def fire(
        self,
        rule_name: str,
        title: str,
        message: str,
        source: str,
        labels: Optional[Dict[str, str]] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ) -> Optional[PersistedAlert]:
        """
        Fire an alert based on a rule.

        Enhanced with rate limiting and aggregation to prevent alert storms.

        Args:
            rule_name: Name of the rule
            title: Alert title
            message: Alert message
            source: Alert source identifier
            labels: Additional labels
            annotations: Additional annotations

        Returns:
            The created alert or None if suppressed/deduplicated/rate-limited
        """
        rule = self._rules.get(rule_name)
        if not rule:
            logger.warning(f"Unknown alert rule: {rule_name}")
            return None

        # Check suppression
        if source in self._suppressed_sources:
            self._stats["total_suppressed"] += 1
            return None

        # Check cooldown
        if not rule.can_fire():
            return None

        # Create alert
        alert = PersistedAlert(
            alert_id=str(uuid.uuid4())[:8],
            rule_name=rule_name,
            severity=rule.severity,
            state=AlertState.FIRING,
            title=title,
            message=message,
            source=source,
            labels={**rule.labels, **(labels or {})},
            annotations=annotations or {},
        )

        # Check deduplication
        if await self._deduplicator.should_dedupe(alert.fingerprint):
            # Update existing alert instead
            existing = await self._store.get_by_fingerprint(alert.fingerprint)
            if existing:
                existing.notification_count += 1
                await self._store.update(existing)
            self._stats["total_deduplicated"] += 1
            return None

        # Check rate limiting (CRITICAL alerts bypass)
        if self._enable_rate_limiting:
            if not await self._rate_limiter.should_allow(alert.severity):
                self._stats["total_rate_limited"] += 1
                logger.warning(
                    f"Alert rate limited: {title} (severity={alert.severity.name})"
                )
                # Still save to store for history, just don't notify
                await self._store.save(alert)
                return None

        # Try aggregation for non-critical alerts
        if (
            self._enable_aggregation
            and alert.severity not in (AlertSeverity.CRITICAL, AlertSeverity.ERROR)
        ):
            aggregated = await self._aggregator.add_or_aggregate(
                alert.fingerprint,
                {
                    "title": title,
                    "message": message,
                    "severity": alert.severity,
                    "source": source,
                },
            )
            if aggregated:
                # Send aggregated alert instead
                self._stats["total_aggregated"] += 1
                agg_alert = PersistedAlert(
                    alert_id=str(uuid.uuid4())[:8],
                    rule_name=f"{rule_name}_aggregated",
                    severity=alert.severity,
                    state=AlertState.FIRING,
                    title=aggregated["title"],
                    message=aggregated["message"],
                    source=source,
                    labels={
                        **rule.labels,
                        **(labels or {}),
                        "aggregated": "true",
                        "count": str(aggregated["aggregated_count"]),
                    },
                )
                await self._store.save(agg_alert)
                await self._router.route(agg_alert, rule.channels)
                return agg_alert
            else:
                # Still collecting, save but don't notify yet
                await self._store.save(alert)
                return None

        # Save alert
        await self._store.save(alert)
        rule.mark_fired()
        self._stats["total_fired"] += 1

        # Route to channels
        await self._router.route(alert, rule.channels)
        alert.notification_count = 1
        alert.last_notified_at = datetime.now(timezone.utc)
        await self._store.update(alert)

        # Trigger callbacks
        for callback in self._callbacks:
            try:
                await callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

        return alert

    async def fire_simple(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str,
        channels: Optional[List[AlertChannel]] = None,
    ) -> PersistedAlert:
        """
        Fire a simple alert without a rule.

        Args:
            severity: Alert severity
            title: Alert title
            message: Alert message
            source: Alert source
            channels: Notification channels

        Returns:
            The created alert
        """
        alert = PersistedAlert(
            alert_id=str(uuid.uuid4())[:8],
            rule_name="_simple",
            severity=severity,
            state=AlertState.FIRING,
            title=title,
            message=message,
            source=source,
        )

        await self._store.save(alert)
        await self._router.route(alert, channels or [AlertChannel.LOG])

        return alert

    async def acknowledge(
        self, alert_id: str, by: Optional[str] = None
    ) -> Optional[PersistedAlert]:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            by: Who acknowledged

        Returns:
            The updated alert or None
        """
        alert = await self._store.get(alert_id)
        if alert and alert.is_active:
            alert.acknowledge(by)
            await self._store.update(alert)
            logger.info(f"Alert {alert_id} acknowledged by {by or 'system'}")
            return alert
        return None

    async def resolve(
        self, alert_id: str, by: Optional[str] = None
    ) -> Optional[PersistedAlert]:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID
            by: Who resolved

        Returns:
            The updated alert or None
        """
        alert = await self._store.get(alert_id)
        if alert and alert.is_active:
            alert.resolve(by)
            await self._store.update(alert)
            logger.info(f"Alert {alert_id} resolved by {by or 'system'}")
            return alert
        return None

    async def resolve_by_fingerprint(
        self, fingerprint: str, by: Optional[str] = None
    ) -> Optional[PersistedAlert]:
        """Resolve alert by fingerprint."""
        alert = await self._store.get_by_fingerprint(fingerprint)
        if alert:
            alert.resolve(by)
            await self._store.update(alert)
            return alert
        return None

    async def get_active_alerts(self) -> List[PersistedAlert]:
        """Get all active alerts."""
        return await self._store.get_active()

    async def get_alert(self, alert_id: str) -> Optional[PersistedAlert]:
        """Get alert by ID."""
        return await self._store.get(alert_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive alert statistics."""
        return {
            "rules_registered": len(self._rules),
            "suppressed_sources": len(self._suppressed_sources),
            "alerts": self._stats.copy(),
            "rate_limiter": self._rate_limiter.get_stats() if self._rate_limiter else {},
            "retry_queue": self._router.get_retry_stats(),
            "pending_aggregation": self._aggregator.get_pending_count()
            if self._aggregator
            else 0,
        }

    async def start_background_tasks(
        self,
        escalation_interval: int = 60,
        retry_interval: int = 30,
        aggregation_flush_interval: int = 120,
    ) -> None:
        """
        Start all background tasks.

        Args:
            escalation_interval: Seconds between escalation checks
            retry_interval: Seconds between retry processing
            aggregation_flush_interval: Seconds between aggregation flushes
        """
        if self._running:
            return

        self._running = True

        # Start escalation checker
        self._check_task = asyncio.create_task(
            self._escalation_loop(escalation_interval)
        )

        # Start retry processor
        self._retry_task = asyncio.create_task(
            self._retry_loop(retry_interval)
        )

        # Start aggregation flusher
        if self._enable_aggregation:
            self._aggregation_task = asyncio.create_task(
                self._aggregation_flush_loop(aggregation_flush_interval)
            )

        logger.info(
            f"Alert background tasks started "
            f"(escalation={escalation_interval}s, retry={retry_interval}s, "
            f"aggregation_flush={aggregation_flush_interval}s)"
        )

    async def start_escalation_checker(
        self, check_interval: int = 60
    ) -> None:
        """Start the escalation checker loop (legacy method)."""
        await self.start_background_tasks(escalation_interval=check_interval)

    async def stop_background_tasks(self) -> None:
        """Stop all background tasks."""
        self._running = False

        tasks = [
            self._check_task,
            self._retry_task,
            self._aggregation_task,
        ]

        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Flush any pending aggregations
        if self._enable_aggregation and self._aggregator:
            pending = await self._aggregator.flush_all()
            for agg in pending:
                logger.info(f"Flushed pending aggregation: {agg['title']}")

        logger.info("Alert background tasks stopped")

    async def stop_escalation_checker(self) -> None:
        """Stop the escalation checker (legacy method)."""
        await self.stop_background_tasks()

    async def _escalation_loop(self, interval: int) -> None:
        """Check for alerts needing escalation."""
        while self._running:
            try:
                active = await self._store.get_active()
                now = datetime.now(timezone.utc)

                for alert in active:
                    rule = self._rules.get(alert.rule_name)
                    if not rule:
                        continue

                    # Check if escalation needed
                    minutes_active = (now - alert.fired_at).total_seconds() / 60
                    if (
                        alert.state == AlertState.FIRING
                        and minutes_active >= rule.escalation_after_minutes
                    ):
                        await self._router.escalate(alert)
                        alert.notification_count += 1
                        alert.last_notified_at = now
                        await self._store.update(alert)

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in escalation loop: {e}")
                await asyncio.sleep(interval)

    async def _retry_loop(self, interval: int) -> None:
        """Process failed alert retries."""
        while self._running:
            try:
                success_count = await self._router.process_retries()
                if success_count > 0:
                    logger.info(f"Successfully retried {success_count} alert(s)")

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in retry loop: {e}")
                await asyncio.sleep(interval)

    async def _aggregation_flush_loop(self, interval: int) -> None:
        """Periodically flush pending aggregations."""
        while self._running:
            try:
                if self._aggregator:
                    pending_count = self._aggregator.get_pending_count()
                    if pending_count > 0:
                        aggregated = await self._aggregator.flush_all()
                        for agg in aggregated:
                            # Send flushed aggregations
                            agg_alert = PersistedAlert(
                                alert_id=str(uuid.uuid4())[:8],
                                rule_name="_aggregated_flush",
                                severity=agg.get("severity", AlertSeverity.INFO),
                                state=AlertState.FIRING,
                                title=agg["title"],
                                message=agg["message"],
                                source=agg.get("source", "aggregator"),
                                labels={"aggregated": "true", "flushed": "true"},
                            )
                            await self._store.save(agg_alert)
                            await self._router.route(
                                agg_alert,
                                [AlertChannel.LOG, AlertChannel.DISCORD],
                            )
                            logger.info(
                                f"Flushed aggregation: {agg['title']} "
                                f"({agg.get('aggregated_count', 0)} alerts)"
                            )

                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in aggregation flush loop: {e}")
                await asyncio.sleep(interval)
