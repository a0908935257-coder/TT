"""
Alert persistence module.

Provides SQLite-based storage for alerts and alert history.
"""

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.core import get_logger
from src.monitoring.alerts import AlertSeverity, AlertState, PersistedAlert

logger = get_logger(__name__)


@dataclass
class AlertQuery:
    """Query parameters for alert retrieval."""

    severity: Optional[AlertSeverity] = None
    state: Optional[AlertState] = None
    rule_name: Optional[str] = None
    source: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = 100
    offset: int = 0
    labels: Optional[Dict[str, str]] = None


@dataclass
class AlertStats:
    """Alert statistics."""

    total_alerts: int = 0
    active_alerts: int = 0
    acknowledged_alerts: int = 0
    resolved_alerts: int = 0
    by_severity: Dict[str, int] = None
    by_rule: Dict[str, int] = None
    avg_resolution_time_seconds: Optional[float] = None

    def __post_init__(self):
        if self.by_severity is None:
            self.by_severity = {}
        if self.by_rule is None:
            self.by_rule = {}


class AlertStore:
    """
    SQLite-based alert persistence.

    Stores alerts with full history for auditing and analysis.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        retention_days: int = 90,
    ):
        """
        Initialize the alert store.

        Args:
            db_path: Path to SQLite database
            retention_days: Days to retain resolved alerts
        """
        self._db_path = db_path or Path("data/alerts.db")
        self._retention_days = retention_days

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with self._get_connection() as conn:
            # Main alerts table - matches PersistedAlert fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    rule_name TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    state TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    source TEXT,
                    fingerprint TEXT,
                    fired_at TEXT NOT NULL,
                    acknowledged_at TEXT,
                    acknowledged_by TEXT,
                    resolved_at TEXT,
                    resolved_by TEXT,
                    last_notified_at TEXT,
                    labels_json TEXT,
                    annotations_json TEXT,
                    notification_count INTEGER DEFAULT 0
                )
            """)

            # Alert history for state changes
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    old_state TEXT,
                    new_state TEXT,
                    actor TEXT,
                    notes TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)

            # Alert notifications tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alert_notifications (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    FOREIGN KEY (alert_id) REFERENCES alerts(alert_id)
                )
            """)

            # Indexes
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_state
                ON alerts(state)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_severity
                ON alerts(severity)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_rule
                ON alerts(rule_name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_fired_at
                ON alerts(fired_at)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_fingerprint
                ON alerts(fingerprint)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_history_alert
                ON alert_history(alert_id)
            """)

            conn.commit()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get a database connection."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def save_alert(self, alert: PersistedAlert) -> None:
        """
        Save or update an alert.

        Args:
            alert: Alert to save
        """
        with self._get_connection() as conn:
            # Check if alert exists
            cursor = conn.execute(
                "SELECT alert_id, state FROM alerts WHERE alert_id = ?",
                (alert.alert_id,),
            )
            existing = cursor.fetchone()

            if existing:
                # Update existing alert
                old_state = existing["state"]

                conn.execute(
                    """
                    UPDATE alerts SET
                        state = ?,
                        title = ?,
                        message = ?,
                        acknowledged_at = ?,
                        acknowledged_by = ?,
                        resolved_at = ?,
                        resolved_by = ?,
                        last_notified_at = ?,
                        labels_json = ?,
                        annotations_json = ?,
                        notification_count = ?
                    WHERE alert_id = ?
                """,
                    (
                        alert.state.value,
                        alert.title,
                        alert.message,
                        (
                            alert.acknowledged_at.isoformat()
                            if alert.acknowledged_at
                            else None
                        ),
                        alert.acknowledged_by,
                        alert.resolved_at.isoformat() if alert.resolved_at else None,
                        alert.resolved_by,
                        (
                            alert.last_notified_at.isoformat()
                            if alert.last_notified_at
                            else None
                        ),
                        json.dumps(alert.labels) if alert.labels else None,
                        json.dumps(alert.annotations) if alert.annotations else None,
                        alert.notification_count,
                        alert.alert_id,
                    ),
                )

                # Record state change in history
                if old_state != alert.state.value:
                    self._record_history(
                        conn,
                        alert.alert_id,
                        "state_change",
                        old_state,
                        alert.state.value,
                    )
            else:
                # Insert new alert
                conn.execute(
                    """
                    INSERT INTO alerts
                    (alert_id, rule_name, severity, state, title, message,
                     source, fingerprint, fired_at,
                     acknowledged_at, acknowledged_by, resolved_at, resolved_by,
                     last_notified_at, labels_json, annotations_json, notification_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        alert.alert_id,
                        alert.rule_name,
                        alert.severity.value,
                        alert.state.value,
                        alert.title,
                        alert.message,
                        alert.source,
                        alert.fingerprint,
                        alert.fired_at.isoformat(),
                        (
                            alert.acknowledged_at.isoformat()
                            if alert.acknowledged_at
                            else None
                        ),
                        alert.acknowledged_by,
                        alert.resolved_at.isoformat() if alert.resolved_at else None,
                        alert.resolved_by,
                        (
                            alert.last_notified_at.isoformat()
                            if alert.last_notified_at
                            else None
                        ),
                        json.dumps(alert.labels) if alert.labels else None,
                        json.dumps(alert.annotations) if alert.annotations else None,
                        alert.notification_count,
                    ),
                )

                # Record creation in history
                self._record_history(
                    conn,
                    alert.alert_id,
                    "created",
                    None,
                    alert.state.value,
                )

            conn.commit()

        logger.debug(f"Alert saved: {alert.alert_id}")

    def _record_history(
        self,
        conn: sqlite3.Connection,
        alert_id: str,
        event_type: str,
        old_state: Optional[str],
        new_state: Optional[str],
        actor: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Record an event in alert history."""
        conn.execute(
            """
            INSERT INTO alert_history
            (alert_id, timestamp, event_type, old_state, new_state, actor, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                alert_id,
                datetime.now(timezone.utc).isoformat(),
                event_type,
                old_state,
                new_state,
                actor,
                notes,
            ),
        )

    def get_alert(self, alert_id: str) -> Optional[PersistedAlert]:
        """
        Get an alert by ID.

        Args:
            alert_id: Alert ID

        Returns:
            Alert if found, None otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM alerts WHERE alert_id = ?",
                (alert_id,),
            )
            row = cursor.fetchone()

            if not row:
                return None

            return self._row_to_alert(row)

    def get_alert_by_fingerprint(
        self,
        fingerprint: str,
        active_only: bool = True,
    ) -> Optional[PersistedAlert]:
        """
        Get alert by fingerprint.

        Args:
            fingerprint: Alert fingerprint
            active_only: Only return active alerts

        Returns:
            Alert if found, None otherwise
        """
        with self._get_connection() as conn:
            if active_only:
                cursor = conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE fingerprint = ? AND state IN ('firing', 'acknowledged')
                    ORDER BY fired_at DESC LIMIT 1
                """,
                    (fingerprint,),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE fingerprint = ?
                    ORDER BY fired_at DESC LIMIT 1
                """,
                    (fingerprint,),
                )

            row = cursor.fetchone()
            return self._row_to_alert(row) if row else None

    def query_alerts(self, query: AlertQuery) -> List[PersistedAlert]:
        """
        Query alerts with filters.

        Args:
            query: Query parameters

        Returns:
            List of matching alerts
        """
        conditions = []
        params = []

        if query.severity:
            conditions.append("severity = ?")
            params.append(query.severity.value)

        if query.state:
            conditions.append("state = ?")
            params.append(query.state.value)

        if query.rule_name:
            conditions.append("rule_name = ?")
            params.append(query.rule_name)

        if query.source:
            conditions.append("source = ?")
            params.append(query.source)

        if query.start_time:
            conditions.append("fired_at >= ?")
            params.append(query.start_time.isoformat())

        if query.end_time:
            conditions.append("fired_at <= ?")
            params.append(query.end_time.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
            SELECT * FROM alerts
            WHERE {where_clause}
            ORDER BY fired_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            alerts = [self._row_to_alert(row) for row in cursor]

            # Filter by labels if specified
            if query.labels:
                alerts = [
                    a
                    for a in alerts
                    if a.labels
                    and all(a.labels.get(k) == v for k, v in query.labels.items())
                ]

            return alerts

    def get_active_alerts(self) -> List[PersistedAlert]:
        """Get all active (firing or acknowledged) alerts."""
        return self.query_alerts(
            AlertQuery(
                limit=1000,
            )
        )

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
        notes: Optional[str] = None,
    ) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert ID
            acknowledged_by: Who acknowledged
            notes: Optional notes

        Returns:
            True if successful
        """
        now = datetime.now(timezone.utc)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT state FROM alerts WHERE alert_id = ?",
                (alert_id,),
            )
            row = cursor.fetchone()

            if not row:
                return False

            old_state = row["state"]

            conn.execute(
                """
                UPDATE alerts SET
                    state = ?,
                    acknowledged_at = ?,
                    acknowledged_by = ?
                WHERE alert_id = ?
            """,
                (
                    AlertState.ACKNOWLEDGED.value,
                    now.isoformat(),
                    acknowledged_by,
                    alert_id,
                ),
            )

            self._record_history(
                conn,
                alert_id,
                "acknowledged",
                old_state,
                AlertState.ACKNOWLEDGED.value,
                acknowledged_by,
                notes,
            )

            conn.commit()

        logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
        return True

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str = "system",
        notes: Optional[str] = None,
    ) -> bool:
        """
        Resolve an alert.

        Args:
            alert_id: Alert ID
            resolved_by: Who resolved
            notes: Optional notes

        Returns:
            True if successful
        """
        now = datetime.now(timezone.utc)

        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT state FROM alerts WHERE alert_id = ?",
                (alert_id,),
            )
            row = cursor.fetchone()

            if not row:
                return False

            old_state = row["state"]

            conn.execute(
                """
                UPDATE alerts SET
                    state = ?,
                    resolved_at = ?,
                    resolved_by = ?
                WHERE alert_id = ?
            """,
                (
                    AlertState.RESOLVED.value,
                    now.isoformat(),
                    resolved_by,
                    alert_id,
                ),
            )

            self._record_history(
                conn,
                alert_id,
                "resolved",
                old_state,
                AlertState.RESOLVED.value,
                resolved_by,
                notes,
            )

            conn.commit()

        logger.info(f"Alert {alert_id} resolved by {resolved_by}")
        return True

    def record_notification(
        self,
        alert_id: str,
        channel: str,
        success: bool,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Record a notification attempt.

        Args:
            alert_id: Alert ID
            channel: Notification channel
            success: Whether notification succeeded
            error_message: Error message if failed
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO alert_notifications
                (alert_id, timestamp, channel, success, error_message)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    alert_id,
                    datetime.now(timezone.utc).isoformat(),
                    channel,
                    1 if success else 0,
                    error_message,
                ),
            )

            # Update notification count
            conn.execute(
                """
                UPDATE alerts SET notification_count = notification_count + 1
                WHERE alert_id = ?
            """,
                (alert_id,),
            )

            conn.commit()

    def get_alert_history(
        self,
        alert_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get history for an alert.

        Args:
            alert_id: Alert ID

        Returns:
            List of history events
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM alert_history
                WHERE alert_id = ?
                ORDER BY timestamp ASC
            """,
                (alert_id,),
            )

            return [
                {
                    "timestamp": row["timestamp"],
                    "event_type": row["event_type"],
                    "old_state": row["old_state"],
                    "new_state": row["new_state"],
                    "actor": row["actor"],
                    "notes": row["notes"],
                }
                for row in cursor
            ]

    def get_statistics(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> AlertStats:
        """
        Get alert statistics.

        Args:
            start_time: Start of period
            end_time: End of period

        Returns:
            Alert statistics
        """
        start = start_time or (datetime.now(timezone.utc) - timedelta(days=7))
        end = end_time or datetime.now(timezone.utc)

        stats = AlertStats()

        with self._get_connection() as conn:
            # Total and state counts
            cursor = conn.execute(
                """
                SELECT state, COUNT(*) as count FROM alerts
                WHERE fired_at >= ? AND fired_at <= ?
                GROUP BY state
            """,
                (start.isoformat(), end.isoformat()),
            )

            for row in cursor:
                count = row["count"]
                stats.total_alerts += count
                if row["state"] == "firing":
                    stats.active_alerts = count
                elif row["state"] == "acknowledged":
                    stats.acknowledged_alerts = count
                elif row["state"] == "resolved":
                    stats.resolved_alerts = count

            # By severity
            cursor = conn.execute(
                """
                SELECT severity, COUNT(*) as count FROM alerts
                WHERE fired_at >= ? AND fired_at <= ?
                GROUP BY severity
            """,
                (start.isoformat(), end.isoformat()),
            )
            stats.by_severity = {row["severity"]: row["count"] for row in cursor}

            # By rule
            cursor = conn.execute(
                """
                SELECT rule_name, COUNT(*) as count FROM alerts
                WHERE fired_at >= ? AND fired_at <= ?
                GROUP BY rule_name
                ORDER BY count DESC
                LIMIT 20
            """,
                (start.isoformat(), end.isoformat()),
            )
            stats.by_rule = {row["rule_name"]: row["count"] for row in cursor}

            # Average resolution time
            cursor = conn.execute(
                """
                SELECT AVG(
                    julianday(resolved_at) - julianday(fired_at)
                ) * 86400 as avg_seconds
                FROM alerts
                WHERE state = 'resolved'
                AND fired_at >= ? AND fired_at <= ?
                AND resolved_at IS NOT NULL
            """,
                (start.isoformat(), end.isoformat()),
            )
            row = cursor.fetchone()
            if row and row["avg_seconds"]:
                stats.avg_resolution_time_seconds = row["avg_seconds"]

        return stats

    def cleanup_old_alerts(self) -> int:
        """
        Remove old resolved alerts beyond retention period.

        Returns:
            Number of alerts deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=self._retention_days)

        with self._get_connection() as conn:
            # Delete notifications first (foreign key)
            conn.execute(
                """
                DELETE FROM alert_notifications
                WHERE alert_id IN (
                    SELECT alert_id FROM alerts
                    WHERE state = 'resolved'
                    AND resolved_at < ?
                )
            """,
                (cutoff.isoformat(),),
            )

            # Delete history
            conn.execute(
                """
                DELETE FROM alert_history
                WHERE alert_id IN (
                    SELECT alert_id FROM alerts
                    WHERE state = 'resolved'
                    AND resolved_at < ?
                )
            """,
                (cutoff.isoformat(),),
            )

            # Delete alerts
            cursor = conn.execute(
                """
                DELETE FROM alerts
                WHERE state = 'resolved'
                AND resolved_at < ?
            """,
                (cutoff.isoformat(),),
            )

            deleted = cursor.rowcount
            conn.commit()

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old resolved alerts")

        return deleted

    def _row_to_alert(self, row: sqlite3.Row) -> PersistedAlert:
        """Convert a database row to a PersistedAlert."""
        labels = json.loads(row["labels_json"]) if row["labels_json"] else {}
        annotations = (
            json.loads(row["annotations_json"]) if row["annotations_json"] else {}
        )

        # Handle severity - stored as integer value
        severity_val = row["severity"]
        if isinstance(severity_val, str):
            severity_val = int(severity_val)
        severity = AlertSeverity(severity_val)

        return PersistedAlert(
            alert_id=row["alert_id"],
            rule_name=row["rule_name"],
            severity=severity,
            state=AlertState(row["state"]),
            title=row["title"],
            message=row["message"] or "",
            source=row["source"] or "",
            labels=labels,
            annotations=annotations,
            fired_at=datetime.fromisoformat(row["fired_at"]),
            acknowledged_at=(
                datetime.fromisoformat(row["acknowledged_at"])
                if row["acknowledged_at"]
                else None
            ),
            acknowledged_by=row["acknowledged_by"],
            resolved_at=(
                datetime.fromisoformat(row["resolved_at"])
                if row["resolved_at"]
                else None
            ),
            resolved_by=row["resolved_by"],
            last_notified_at=(
                datetime.fromisoformat(row["last_notified_at"])
                if row["last_notified_at"]
                else None
            ),
            notification_count=row["notification_count"],
            fingerprint=row["fingerprint"] or "",
        )
