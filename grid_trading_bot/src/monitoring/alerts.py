"""
Alert management module.

Provides alert persistence, deduplication, routing, and escalation
for production trading systems.
"""

import asyncio
import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set

from src.core import get_logger

logger = get_logger(__name__)


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


class AlertRouter:
    """
    Routes alerts to appropriate channels based on rules.

    Supports channel-specific handlers and escalation.
    """

    def __init__(self):
        """Initialize the router."""
        self._handlers: Dict[AlertChannel, AlertChannelHandler] = {
            AlertChannel.LOG: LogChannelHandler(),
        }
        self._escalation_handlers: Dict[AlertSeverity, List[AlertChannel]] = {}

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

    async def route(
        self,
        alert: PersistedAlert,
        channels: List[AlertChannel],
    ) -> Dict[AlertChannel, bool]:
        """
        Route alert to specified channels.

        Args:
            alert: The alert to route
            channels: Channels to send to

        Returns:
            Dict of channel to success status
        """
        results: Dict[AlertChannel, bool] = {}

        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    results[channel] = await handler.send(alert)
                except Exception as e:
                    logger.error(f"Error routing to {channel}: {e}")
                    results[channel] = False
            else:
                logger.warning(f"No handler for channel: {channel}")
                results[channel] = False

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
    """

    def __init__(
        self,
        store: Optional[AlertStore] = None,
        router: Optional[AlertRouter] = None,
        deduplicator: Optional[AlertDeduplicator] = None,
    ):
        """
        Initialize the alert manager.

        Args:
            store: Alert persistence store
            router: Alert router
            deduplicator: Alert deduplicator
        """
        self._store = store or InMemoryAlertStore()
        self._router = router or AlertRouter()
        self._deduplicator = deduplicator or AlertDeduplicator()
        self._rules: Dict[str, AlertRule] = {}
        self._callbacks: List[Callable[[PersistedAlert], Coroutine[Any, Any, None]]] = (
            []
        )
        self._running = False
        self._check_task: Optional[asyncio.Task] = None
        self._suppressed_sources: Set[str] = set()

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

        Args:
            rule_name: Name of the rule
            title: Alert title
            message: Alert message
            source: Alert source identifier
            labels: Additional labels
            annotations: Additional annotations

        Returns:
            The created alert or None if suppressed/deduplicated
        """
        rule = self._rules.get(rule_name)
        if not rule:
            logger.warning(f"Unknown alert rule: {rule_name}")
            return None

        # Check suppression
        if source in self._suppressed_sources:
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
            return None

        # Save alert
        await self._store.save(alert)
        rule.mark_fired()

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
        """Get alert statistics."""
        # This would need to query the store for full stats
        return {
            "rules_registered": len(self._rules),
            "suppressed_sources": len(self._suppressed_sources),
        }

    async def start_escalation_checker(
        self, check_interval: int = 60
    ) -> None:
        """Start the escalation checker loop."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(
            self._escalation_loop(check_interval)
        )

    async def stop_escalation_checker(self) -> None:
        """Stop the escalation checker."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

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
