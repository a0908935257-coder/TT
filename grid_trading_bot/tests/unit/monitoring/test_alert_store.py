"""Tests for alert store module."""

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from src.monitoring.alert_store import AlertQuery, AlertStats, AlertStore
from src.monitoring.alerts import AlertSeverity, AlertState, PersistedAlert


class TestAlertStore:
    """Tests for AlertStore."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir) / "test_alerts.db"

    @pytest.fixture
    def store(self, temp_db):
        """Create a test store."""
        return AlertStore(db_path=temp_db)

    @pytest.fixture
    def sample_alert(self):
        """Create a sample alert."""
        return PersistedAlert(
            alert_id="alert-001",
            rule_name="cpu_high",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="CPU Usage High",
            message="CPU usage is above 80%",
            source="system_monitor",
            labels={"host": "server-1", "metric": "cpu"},
        )

    def test_init_creates_database(self, temp_db):
        """Test that initialization creates database."""
        store = AlertStore(db_path=temp_db)
        assert temp_db.exists()

    def test_save_new_alert(self, store, sample_alert):
        """Test saving a new alert."""
        store.save_alert(sample_alert)

        retrieved = store.get_alert(sample_alert.alert_id)
        assert retrieved is not None
        assert retrieved.alert_id == sample_alert.alert_id
        assert retrieved.rule_name == sample_alert.rule_name
        assert retrieved.severity == sample_alert.severity

    def test_update_existing_alert(self, store, sample_alert):
        """Test updating an existing alert."""
        store.save_alert(sample_alert)

        # Update the alert
        sample_alert.state = AlertState.ACKNOWLEDGED
        sample_alert.acknowledged_at = datetime.now(timezone.utc)
        sample_alert.acknowledged_by = "operator"
        store.save_alert(sample_alert)

        retrieved = store.get_alert(sample_alert.alert_id)
        assert retrieved.state == AlertState.ACKNOWLEDGED
        assert retrieved.acknowledged_by == "operator"

    def test_get_alert_by_fingerprint(self, store, sample_alert):
        """Test getting alert by fingerprint."""
        store.save_alert(sample_alert)

        retrieved = store.get_alert_by_fingerprint(sample_alert.fingerprint)
        assert retrieved is not None
        assert retrieved.alert_id == sample_alert.alert_id

    def test_get_alert_by_fingerprint_active_only(self, store, sample_alert):
        """Test getting only active alerts by fingerprint."""
        # Save and resolve alert
        store.save_alert(sample_alert)
        store.resolve_alert(sample_alert.alert_id, "system")

        # Should not find resolved alert
        retrieved = store.get_alert_by_fingerprint(
            sample_alert.fingerprint, active_only=True
        )
        assert retrieved is None

        # Should find if not filtering
        retrieved = store.get_alert_by_fingerprint(
            sample_alert.fingerprint, active_only=False
        )
        assert retrieved is not None

    def test_query_alerts_by_severity(self, store):
        """Test querying alerts by severity."""
        # Create alerts with different severities
        for i, severity in enumerate(
            [AlertSeverity.INFO, AlertSeverity.WARNING, AlertSeverity.ERROR]
        ):
            alert = PersistedAlert(
                alert_id=f"alert-{i:03d}",
                rule_name="test_rule",
                severity=severity,
                state=AlertState.FIRING,
                title=f"Test Alert {i}",
                message=f"Test message {i}",
                source="test",
            )
            store.save_alert(alert)

        # Query warnings only
        warnings = store.query_alerts(AlertQuery(severity=AlertSeverity.WARNING))
        assert len(warnings) == 1
        assert warnings[0].severity == AlertSeverity.WARNING

    def test_query_alerts_by_state(self, store, sample_alert):
        """Test querying alerts by state."""
        store.save_alert(sample_alert)

        # Create another alert and resolve it
        resolved_alert = PersistedAlert(
            alert_id="alert-002",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.RESOLVED,
            title="Resolved Alert",
            message="This alert was resolved",
            source="test",
            resolved_at=datetime.now(timezone.utc),
        )
        store.save_alert(resolved_alert)

        # Query firing only
        firing = store.query_alerts(AlertQuery(state=AlertState.FIRING))
        assert len(firing) == 1
        assert firing[0].alert_id == sample_alert.alert_id

    def test_query_alerts_by_rule(self, store):
        """Test querying alerts by rule name."""
        for rule in ["cpu_high", "memory_high", "cpu_high"]:
            alert = PersistedAlert(
                alert_id=f"alert-{rule}-{datetime.now().timestamp()}",
                rule_name=rule,
                severity=AlertSeverity.WARNING,
                state=AlertState.FIRING,
                title=f"{rule} alert",
                message=f"{rule} is high",
                source="test",
            )
            store.save_alert(alert)

        cpu_alerts = store.query_alerts(AlertQuery(rule_name="cpu_high"))
        assert len(cpu_alerts) == 2

    def test_query_alerts_by_time_range(self, store):
        """Test querying alerts by time range."""
        now = datetime.now(timezone.utc)

        # Create old alert
        old_alert = PersistedAlert(
            alert_id="alert-old",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="Old Alert",
            message="Old alert message",
            source="test",
            fired_at=now - timedelta(days=10),
        )
        store.save_alert(old_alert)

        # Create recent alert
        recent_alert = PersistedAlert(
            alert_id="alert-recent",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="Recent Alert",
            message="Recent alert message",
            source="test",
            fired_at=now - timedelta(hours=1),
        )
        store.save_alert(recent_alert)

        # Query last 24 hours
        recent = store.query_alerts(
            AlertQuery(start_time=now - timedelta(days=1), end_time=now)
        )
        assert len(recent) == 1
        assert recent[0].alert_id == "alert-recent"

    def test_query_alerts_with_labels(self, store):
        """Test querying alerts with label filters."""
        # Create alerts with different labels
        alert1 = PersistedAlert(
            alert_id="alert-001",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="Alert 1",
            message="Alert 1 message",
            source="test",
            labels={"host": "server-1", "env": "prod"},
        )
        alert2 = PersistedAlert(
            alert_id="alert-002",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="Alert 2",
            message="Alert 2 message",
            source="test",
            labels={"host": "server-2", "env": "staging"},
        )
        store.save_alert(alert1)
        store.save_alert(alert2)

        # Query by label
        prod_alerts = store.query_alerts(AlertQuery(labels={"env": "prod"}))
        assert len(prod_alerts) == 1
        assert prod_alerts[0].alert_id == "alert-001"

    def test_acknowledge_alert(self, store, sample_alert):
        """Test acknowledging an alert."""
        store.save_alert(sample_alert)

        result = store.acknowledge_alert(
            sample_alert.alert_id, acknowledged_by="operator", notes="Looking into it"
        )

        assert result is True

        retrieved = store.get_alert(sample_alert.alert_id)
        assert retrieved.state == AlertState.ACKNOWLEDGED
        assert retrieved.acknowledged_by == "operator"
        assert retrieved.acknowledged_at is not None

    def test_acknowledge_nonexistent_alert(self, store):
        """Test acknowledging nonexistent alert."""
        result = store.acknowledge_alert("nonexistent", "operator")
        assert result is False

    def test_resolve_alert(self, store, sample_alert):
        """Test resolving an alert."""
        store.save_alert(sample_alert)

        result = store.resolve_alert(
            sample_alert.alert_id, resolved_by="system", notes="Issue fixed"
        )

        assert result is True

        retrieved = store.get_alert(sample_alert.alert_id)
        assert retrieved.state == AlertState.RESOLVED
        assert retrieved.resolved_by == "system"
        assert retrieved.resolved_at is not None

    def test_resolve_nonexistent_alert(self, store):
        """Test resolving nonexistent alert."""
        result = store.resolve_alert("nonexistent", "system")
        assert result is False

    def test_record_notification(self, store, sample_alert):
        """Test recording notification attempts."""
        store.save_alert(sample_alert)

        store.record_notification(sample_alert.alert_id, "email", True)
        store.record_notification(
            sample_alert.alert_id, "sms", False, "Connection timeout"
        )

        retrieved = store.get_alert(sample_alert.alert_id)
        assert retrieved.notification_count == 2

    def test_get_alert_history(self, store, sample_alert):
        """Test getting alert history."""
        store.save_alert(sample_alert)
        store.acknowledge_alert(sample_alert.alert_id, "operator")
        store.resolve_alert(sample_alert.alert_id, "system")

        history = store.get_alert_history(sample_alert.alert_id)

        assert len(history) == 3  # created, acknowledged, resolved
        assert history[0]["event_type"] == "created"
        assert history[1]["event_type"] == "acknowledged"
        assert history[2]["event_type"] == "resolved"

    def test_get_statistics(self, store):
        """Test getting alert statistics."""
        # Create various alerts
        severities = [
            AlertSeverity.INFO,
            AlertSeverity.WARNING,
            AlertSeverity.WARNING,
            AlertSeverity.ERROR,
        ]
        for i, severity in enumerate(severities):
            alert = PersistedAlert(
                alert_id=f"alert-{i:03d}",
                rule_name=f"rule_{i % 2}",
                severity=severity,
                state=AlertState.FIRING if i < 2 else AlertState.RESOLVED,
                title=f"Test Alert {i}",
                message=f"Test message {i}",
                source="test",
                resolved_at=(datetime.now(timezone.utc) if i >= 2 else None),
            )
            store.save_alert(alert)

        stats = store.get_statistics()

        assert stats.total_alerts == 4
        assert stats.active_alerts == 2  # firing
        assert stats.resolved_alerts == 2
        # Severity is stored as integer value (WARNING=2)
        assert stats.by_severity.get("2", 0) == 2 or stats.by_severity.get(2, 0) == 2
        assert len(stats.by_rule) == 2

    def test_cleanup_old_alerts(self, store):
        """Test cleaning up old resolved alerts."""
        now = datetime.now(timezone.utc)

        # Create old resolved alert
        old_alert = PersistedAlert(
            alert_id="alert-old",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.RESOLVED,
            title="Old Alert",
            message="Old alert message",
            source="test",
            fired_at=now - timedelta(days=100),
            resolved_at=now - timedelta(days=100),
        )
        store.save_alert(old_alert)

        # Create recent alert
        recent_alert = PersistedAlert(
            alert_id="alert-recent",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            state=AlertState.FIRING,
            title="Recent Alert",
            message="Recent alert message",
            source="test",
        )
        store.save_alert(recent_alert)

        # Cleanup with 90-day retention
        deleted = store.cleanup_old_alerts()

        assert deleted == 1
        assert store.get_alert("alert-old") is None
        assert store.get_alert("alert-recent") is not None

    def test_pagination(self, store):
        """Test query pagination."""
        # Create many alerts
        for i in range(25):
            alert = PersistedAlert(
                alert_id=f"alert-{i:03d}",
                rule_name="test_rule",
                severity=AlertSeverity.WARNING,
                state=AlertState.FIRING,
                title=f"Test Alert {i}",
                message=f"Test message {i}",
                source="test",
            )
            store.save_alert(alert)

        # Get first page
        page1 = store.query_alerts(AlertQuery(limit=10, offset=0))
        assert len(page1) == 10

        # Get second page
        page2 = store.query_alerts(AlertQuery(limit=10, offset=10))
        assert len(page2) == 10

        # Verify no overlap
        page1_ids = {a.alert_id for a in page1}
        page2_ids = {a.alert_id for a in page2}
        assert page1_ids.isdisjoint(page2_ids)


class TestAlertQuery:
    """Tests for AlertQuery."""

    def test_default_values(self):
        """Test default query values."""
        query = AlertQuery()

        assert query.severity is None
        assert query.state is None
        assert query.limit == 100
        assert query.offset == 0


class TestAlertStats:
    """Tests for AlertStats."""

    def test_default_values(self):
        """Test default stats values."""
        stats = AlertStats()

        assert stats.total_alerts == 0
        assert stats.active_alerts == 0
        assert stats.by_severity == {}
        assert stats.by_rule == {}
        assert stats.avg_resolution_time_seconds is None
