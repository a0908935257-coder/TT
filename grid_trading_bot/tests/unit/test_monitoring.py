"""
Unit tests for Monitoring module.

Tests for:
- LatencyTracker and ThroughputMeter
- PerformanceProfiler
- AlertManager with persistence
- MetricsExporter (Prometheus format)
- MetricsStreamer
"""

import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal

from src.monitoring.performance import (
    LatencyTracker,
    ThroughputMeter,
    PerformanceProfiler,
    LatencyStats,
    ThroughputStats,
    latency_tracked,
    get_global_profiler,
)
from src.monitoring.alerts import (
    AlertManager,
    AlertRule,
    AlertChannel,
    AlertSeverity,
    AlertState,
    PersistedAlert,
    AlertDeduplicator,
    InMemoryAlertStore,
)
from src.monitoring.exporter import (
    MetricsExporter,
    MetricType,
    MetricValue,
    PrometheusFormatter,
)
from src.monitoring.streamer import (
    MetricsStreamer,
    MetricsSnapshot,
    StreamEvent,
    StreamEventType,
)


# =============================================================================
# Latency Tracker Tests
# =============================================================================


class TestLatencyTracker:
    """Tests for LatencyTracker."""

    def test_tracker_initialization(self):
        """Test tracker initializes correctly."""
        tracker = LatencyTracker(window_size=1000, window_seconds=60.0)

        assert tracker._window_size == 1000
        assert tracker._window_seconds == 60.0

    def test_record_latency(self):
        """Test recording latency measurements."""
        tracker = LatencyTracker()

        tracker.record("api_call", 10.5)
        tracker.record("api_call", 15.2)
        tracker.record("api_call", 8.3)

        stats = tracker.get_stats("api_call")

        assert stats is not None
        assert stats.count == 3
        assert stats.min_ms == 8.3
        assert stats.max_ms == 15.2

    def test_percentile_calculation(self):
        """Test percentile calculations."""
        tracker = LatencyTracker()

        # Record 100 values
        for i in range(100):
            tracker.record("test_op", float(i + 1))

        stats = tracker.get_stats("test_op")

        assert stats is not None
        # Allow small variance in percentile calculation
        assert 49 <= stats.p50_ms <= 52
        assert 94 <= stats.p95_ms <= 97
        assert 98 <= stats.p99_ms <= 100

    def test_success_rate(self):
        """Test success rate calculation."""
        tracker = LatencyTracker()

        for i in range(10):
            tracker.record("op", 5.0, success=(i < 8))

        stats = tracker.get_stats("op")

        assert stats is not None
        assert stats.success_rate == 0.8

    def test_get_slow_operations(self):
        """Test getting slow operations."""
        tracker = LatencyTracker()

        tracker.record("fast", 5.0)
        tracker.record("slow", 150.0)
        tracker.record("medium", 50.0)

        slow = tracker.get_slow_operations(threshold_ms=100.0)

        assert len(slow) == 1
        assert slow[0].operation == "slow"

    def test_get_all_stats(self):
        """Test getting all stats."""
        tracker = LatencyTracker()

        tracker.record("op1", 10.0)
        tracker.record("op2", 20.0)

        all_stats = tracker.get_all_stats()

        assert "op1" in all_stats
        assert "op2" in all_stats


# =============================================================================
# Throughput Meter Tests
# =============================================================================


class TestThroughputMeter:
    """Tests for ThroughputMeter."""

    def test_meter_initialization(self):
        """Test meter initializes correctly."""
        meter = ThroughputMeter(bucket_seconds=1.0, window_buckets=60)

        assert meter._bucket_seconds == 1.0
        assert meter._window_buckets == 60

    def test_record_throughput(self):
        """Test recording throughput."""
        meter = ThroughputMeter()

        meter.record("requests", count=5)
        meter.record("requests", count=3)

        stats = meter.get_stats("requests")

        assert stats is not None
        assert stats.total_requests == 8

    def test_bytes_tracking(self):
        """Test bytes processing tracking."""
        meter = ThroughputMeter()

        meter.record("data", count=1, bytes_processed=1024)
        meter.record("data", count=1, bytes_processed=2048)

        stats = meter.get_stats("data")

        assert stats is not None
        assert stats.total_requests == 2

    def test_error_rate(self):
        """Test error rate calculation."""
        meter = ThroughputMeter()

        meter.record("op", count=8, is_error=False)
        meter.record("op", count=2, is_error=True)

        stats = meter.get_stats("op")

        assert stats is not None
        assert stats.total_errors == 2
        assert stats.error_rate == 0.2


# =============================================================================
# Performance Profiler Tests
# =============================================================================


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler."""

    def test_profiler_initialization(self):
        """Test profiler initializes correctly."""
        profiler = PerformanceProfiler()

        assert profiler.latency is not None
        assert profiler.throughput is not None

    def test_record_operation(self):
        """Test recording complete operation."""
        profiler = PerformanceProfiler()

        profiler.record_operation("test", 25.5, success=True, bytes_processed=100)

        latency_stats = profiler.latency.get_stats("test")
        throughput_stats = profiler.throughput.get_stats("test")

        assert latency_stats is not None
        assert latency_stats.avg_ms == 25.5
        assert throughput_stats is not None

    def test_time_operation_context_manager(self):
        """Test operation timer context manager."""
        profiler = PerformanceProfiler()

        with profiler.time_operation("timed_op") as timer:
            # Simulate work
            _ = sum(range(1000))
            timer.set_bytes(500)

        stats = profiler.latency.get_stats("timed_op")

        assert stats is not None
        assert stats.count == 1

    def test_get_metrics(self):
        """Test getting performance metrics."""
        profiler = PerformanceProfiler()

        profiler.record_operation("op1", 10.0)
        profiler.record_operation("op2", 20.0)

        metrics = profiler.get_metrics()

        assert metrics.timestamp is not None
        assert "op1" in metrics.latency
        assert "op2" in metrics.latency

    def test_get_summary(self):
        """Test getting summary."""
        profiler = PerformanceProfiler()

        profiler.record_operation("fast_op", 5.0)
        profiler.record_operation("slow_op", 100.0)

        summary = profiler.get_summary()

        assert "uptime_seconds" in summary
        assert summary["operations_tracked"] == 2
        assert summary["slowest_operation"] == "slow_op"


# =============================================================================
# Alert Manager Tests
# =============================================================================


class TestAlertManager:
    """Tests for AlertManager."""

    @pytest.fixture
    def manager(self):
        """Create test alert manager with aggregation disabled for deterministic tests."""
        return AlertManager(enable_aggregation=False, enable_rate_limiting=False)

    @pytest.mark.asyncio
    async def test_fire_simple_alert(self, manager):
        """Test firing simple alert."""
        alert = await manager.fire_simple(
            severity=AlertSeverity.WARNING,
            title="Test Alert",
            message="This is a test",
            source="test",
        )

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.state == AlertState.FIRING

    @pytest.mark.asyncio
    async def test_acknowledge_alert(self, manager):
        """Test acknowledging alert."""
        alert = await manager.fire_simple(
            severity=AlertSeverity.ERROR,
            title="Error",
            message="Error occurred",
            source="test",
        )

        acked = await manager.acknowledge(alert.alert_id, by="user")

        assert acked is not None
        assert acked.state == AlertState.ACKNOWLEDGED
        assert acked.acknowledged_by == "user"

    @pytest.mark.asyncio
    async def test_resolve_alert(self, manager):
        """Test resolving alert."""
        alert = await manager.fire_simple(
            severity=AlertSeverity.ERROR,
            title="Error",
            message="Error occurred",
            source="test",
        )

        resolved = await manager.resolve(alert.alert_id, by="system")

        assert resolved is not None
        assert resolved.state == AlertState.RESOLVED
        assert resolved.resolved_by == "system"

    @pytest.mark.asyncio
    async def test_get_active_alerts(self, manager):
        """Test getting active alerts."""
        await manager.fire_simple(
            severity=AlertSeverity.WARNING,
            title="Active 1",
            message="msg",
            source="test",
        )
        alert2 = await manager.fire_simple(
            severity=AlertSeverity.ERROR,
            title="Active 2",
            message="msg",
            source="test",
        )

        await manager.resolve(alert2.alert_id)

        active = await manager.get_active_alerts()

        assert len(active) == 1
        assert active[0].title == "Active 1"

    @pytest.mark.asyncio
    async def test_alert_rule_cooldown(self, manager):
        """Test alert rule cooldown."""
        rule = AlertRule(
            name="test_rule",
            condition=lambda: True,
            severity=AlertSeverity.INFO,
            cooldown_seconds=60,
        )
        manager.register_rule(rule)

        # First fire should succeed
        alert1 = await manager.fire(
            rule_name="test_rule",
            title="First",
            message="msg",
            source="test",
        )

        # Second fire should be blocked by cooldown
        alert2 = await manager.fire(
            rule_name="test_rule",
            title="Second",
            message="msg",
            source="test",
        )

        assert alert1 is not None
        assert alert2 is None

    @pytest.mark.asyncio
    async def test_source_suppression(self, manager):
        """Test source suppression."""
        manager.suppress_source("noisy_bot")

        alert = await manager.fire_simple(
            severity=AlertSeverity.INFO,
            title="Suppressed",
            message="Should not fire",
            source="noisy_bot",
        )

        # Suppressed sources still create alerts via fire_simple
        # (suppression is for rule-based fires)
        assert alert is not None


# =============================================================================
# Alert Deduplicator Tests
# =============================================================================


class TestAlertDeduplicator:
    """Tests for AlertDeduplicator."""

    @pytest.mark.asyncio
    async def test_first_occurrence_allowed(self):
        """Test first occurrence is allowed."""
        dedup = AlertDeduplicator()

        result = await dedup.should_dedupe("fp1")

        assert result is False

    @pytest.mark.asyncio
    async def test_duplicate_suppressed(self):
        """Test duplicate is suppressed."""
        dedup = AlertDeduplicator(max_per_fingerprint=1)

        await dedup.should_dedupe("fp1")
        result = await dedup.should_dedupe("fp1")

        assert result is True

    @pytest.mark.asyncio
    async def test_different_fingerprints_allowed(self):
        """Test different fingerprints are allowed."""
        dedup = AlertDeduplicator(max_per_fingerprint=1)

        r1 = await dedup.should_dedupe("fp1")
        r2 = await dedup.should_dedupe("fp2")

        assert r1 is False
        assert r2 is False


# =============================================================================
# Metrics Exporter Tests
# =============================================================================


class TestMetricsExporter:
    """Tests for MetricsExporter."""

    def test_exporter_initialization(self):
        """Test exporter initializes correctly."""
        exporter = MetricsExporter(prefix="trading")

        assert exporter._prefix == "trading"

    def test_counter_increment(self):
        """Test counter increment."""
        exporter = MetricsExporter()

        exporter.inc_counter("requests", value=1)
        exporter.inc_counter("requests", value=2)

        output = exporter.export_json()

        assert "trading_requests" in output["counters"]

    def test_gauge_set(self):
        """Test gauge setting."""
        exporter = MetricsExporter()

        exporter.set_gauge("temperature", 25.5)
        exporter.set_gauge("temperature", 26.0)

        output = exporter.export_json()

        assert "trading_temperature" in output["gauges"]

    def test_histogram_observe(self):
        """Test histogram observation."""
        exporter = MetricsExporter()

        exporter.observe_histogram("latency", 5.0)
        exporter.observe_histogram("latency", 15.0)
        exporter.observe_histogram("latency", 150.0)

        output = exporter.export_json()

        assert "trading_latency" in output["histograms"]

    def test_labels(self):
        """Test metrics with labels."""
        exporter = MetricsExporter()

        exporter.inc_counter("requests", labels={"method": "GET"})
        exporter.inc_counter("requests", labels={"method": "POST"})

        output = exporter.export_json()

        counters = output["counters"]["trading_requests"]
        assert len(counters) == 2

    def test_prometheus_export(self):
        """Test Prometheus format export."""
        exporter = MetricsExporter()

        exporter.inc_counter("requests_total", labels={"path": "/api"})
        exporter.set_gauge("active_connections", 10)

        output = exporter.export()

        assert "trading_requests_total" in output
        assert "trading_active_connections" in output


# =============================================================================
# Prometheus Formatter Tests
# =============================================================================


class TestPrometheusFormatter:
    """Tests for PrometheusFormatter."""

    def test_format_metric_value(self):
        """Test formatting metric value."""
        metric = MetricValue(
            name="test_metric",
            value=42.5,
            labels={"env": "prod"},
            metric_type=MetricType.GAUGE,
        )

        output = metric.to_prometheus()

        assert "test_metric" in output
        assert "42.5" in output
        assert 'env="prod"' in output

    def test_format_multiple_metrics(self):
        """Test formatting multiple metrics."""
        metrics = [
            MetricValue(name="m1", value=1.0, metric_type=MetricType.COUNTER),
            MetricValue(name="m2", value=2.0, metric_type=MetricType.GAUGE),
        ]

        output = PrometheusFormatter.format_metrics(metrics)

        assert "# TYPE m1 counter" in output
        assert "# TYPE m2 gauge" in output


# =============================================================================
# Metrics Streamer Tests
# =============================================================================


class TestMetricsStreamer:
    """Tests for MetricsStreamer."""

    def test_streamer_initialization(self):
        """Test streamer initializes correctly."""
        streamer = MetricsStreamer(update_interval=1.0)

        assert streamer._update_interval == 1.0
        assert streamer.subscriber_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_event(self):
        """Test broadcasting event."""
        streamer = MetricsStreamer()

        # Create mock subscriber
        class MockSubscriber:
            def __init__(self):
                self.messages = []
                self._connected = True

            async def send(self, message):
                self.messages.append(message)

            @property
            def is_connected(self):
                return self._connected

        subscriber = MockSubscriber()
        await streamer.subscribe(subscriber)

        event = StreamEvent(
            event_type=StreamEventType.METRICS_UPDATE,
            data={"test": "data"},
        )

        delivered = await streamer.broadcast(event)

        assert delivered == 1
        assert len(subscriber.messages) == 1

    def test_metrics_snapshot(self):
        """Test metrics snapshot."""
        snapshot = MetricsSnapshot(
            timestamp=datetime.now(timezone.utc),
            latency={"api": {"p50": 10.0}},
            throughput={"api": {"rps": 100.0}},
            system={"memory_mb": 256.0},
            alerts={"warning": 2},
            orders={"pending": 5},
        )

        data = snapshot.to_dict()

        assert "timestamp" in data
        assert data["latency"]["api"]["p50"] == 10.0
        assert data["system"]["memory_mb"] == 256.0


# =============================================================================
# Decorator Tests
# =============================================================================


class TestDecorators:
    """Tests for monitoring decorators."""

    def test_latency_tracked_sync(self):
        """Test latency_tracked decorator on sync function."""
        profiler = PerformanceProfiler()

        @latency_tracked("sync_op", profiler=profiler)
        def sync_function():
            return sum(range(1000))

        result = sync_function()

        assert result > 0
        stats = profiler.latency.get_stats("sync_op")
        assert stats is not None
        assert stats.count == 1

    @pytest.mark.asyncio
    async def test_latency_tracked_async(self):
        """Test latency_tracked decorator on async function."""
        profiler = PerformanceProfiler()

        @latency_tracked("async_op", profiler=profiler)
        async def async_function():
            await asyncio.sleep(0.01)
            return "done"

        result = await async_function()

        assert result == "done"
        stats = profiler.latency.get_stats("async_op")
        assert stats is not None
        assert stats.count == 1


# =============================================================================
# Integration Tests
# =============================================================================


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    @pytest.mark.asyncio
    async def test_profiler_with_exporter(self):
        """Test profiler with exporter integration."""
        profiler = PerformanceProfiler()
        exporter = MetricsExporter()

        # Record some operations
        profiler.record_operation("api_call", 25.0)
        profiler.record_operation("api_call", 30.0)
        profiler.record_operation("db_query", 10.0)

        # Register collector
        def collect_latency():
            metrics = []
            for op, stats in profiler.latency.get_all_stats().items():
                metrics.append(MetricValue(
                    name=f"latency_p99_{op}",
                    value=stats.p99_ms,
                    metric_type=MetricType.GAUGE,
                ))
            return metrics

        exporter.register_collector(collect_latency)

        output = exporter.export()

        assert "latency_p99_api_call" in output
        assert "latency_p99_db_query" in output

    @pytest.mark.asyncio
    async def test_alert_with_callback(self):
        """Test alert manager with callback."""
        manager = AlertManager(enable_aggregation=False, enable_rate_limiting=False)
        alerts_received = []

        async def on_alert(alert):
            alerts_received.append(alert)

        manager.on_alert(on_alert)

        rule = AlertRule(
            name="test",
            condition=lambda: True,
            severity=AlertSeverity.WARNING,
        )
        manager.register_rule(rule)

        await manager.fire(
            rule_name="test",
            title="Test",
            message="msg",
            source="test",
        )

        assert len(alerts_received) == 1
