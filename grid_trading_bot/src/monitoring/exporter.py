"""
Metrics exporter module.

Provides Prometheus-compatible metrics export for external monitoring systems.
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class MetricType(Enum):
    """Prometheus metric types."""

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """A single metric value with labels."""

    name: str
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    metric_type: MetricType = MetricType.GAUGE

    def prometheus_name(self) -> str:
        """Get Prometheus-formatted name."""
        # Replace invalid characters
        name = self.name.replace(".", "_").replace("-", "_")
        return name

    def prometheus_labels(self) -> str:
        """Get Prometheus-formatted labels."""
        if not self.labels:
            return ""
        parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
        return "{" + ",".join(parts) + "}"

    def to_prometheus(self) -> str:
        """Convert to Prometheus text format."""
        name = self.prometheus_name()
        labels = self.prometheus_labels()
        ts = ""
        if self.timestamp:
            ts = f" {int(self.timestamp.timestamp() * 1000)}"
        return f"{name}{labels} {self.value}{ts}"


@dataclass
class HistogramBucket:
    """Histogram bucket for latency distribution."""

    le: float  # Less than or equal
    count: int


@dataclass
class HistogramValue:
    """Histogram metric value."""

    name: str
    buckets: List[HistogramBucket]
    sum_value: float
    count: int
    labels: Dict[str, str] = field(default_factory=dict)

    def to_prometheus_lines(self) -> List[str]:
        """Convert to Prometheus text format lines."""
        name = self.name.replace(".", "_").replace("-", "_")
        labels_str = ""
        if self.labels:
            parts = [f'{k}="{v}"' for k, v in sorted(self.labels.items())]
            labels_str = "," + ",".join(parts)

        lines = []
        for bucket in self.buckets:
            le_label = f'le="{bucket.le}"' if bucket.le != float("inf") else 'le="+Inf"'
            lines.append(f"{name}_bucket{{{le_label}{labels_str}}} {bucket.count}")

        # Add sum and count
        label_str = "{" + labels_str.lstrip(",") + "}" if labels_str else ""
        lines.append(f"{name}_sum{label_str} {self.sum_value}")
        lines.append(f"{name}_count{label_str} {self.count}")

        return lines


class PrometheusFormatter:
    """Formats metrics in Prometheus exposition format."""

    @staticmethod
    def format_metrics(
        metrics: List[MetricValue],
        histograms: Optional[List[HistogramValue]] = None,
    ) -> str:
        """
        Format metrics as Prometheus text.

        Args:
            metrics: List of metric values
            histograms: Optional histogram values

        Returns:
            Prometheus exposition format string
        """
        lines = []

        # Group by metric name for TYPE and HELP
        by_name: Dict[str, List[MetricValue]] = defaultdict(list)
        for m in metrics:
            by_name[m.prometheus_name()].append(m)

        for name, values in sorted(by_name.items()):
            # Add TYPE header
            metric_type = values[0].metric_type.value
            lines.append(f"# TYPE {name} {metric_type}")

            # Add values
            for v in values:
                lines.append(v.to_prometheus())

        # Add histograms
        if histograms:
            for hist in histograms:
                name = hist.name.replace(".", "_").replace("-", "_")
                lines.append(f"# TYPE {name} histogram")
                lines.extend(hist.to_prometheus_lines())

        return "\n".join(lines)


class MetricsExporter:
    """
    Exports metrics in Prometheus format.

    Collects metrics from various sources and exposes them
    for Prometheus scraping.
    """

    # Default histogram buckets for latency (in ms)
    DEFAULT_BUCKETS = [1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000]

    def __init__(
        self,
        prefix: str = "trading",
        default_labels: Optional[Dict[str, str]] = None,
        histogram_buckets: Optional[List[float]] = None,
    ):
        """
        Initialize the exporter.

        Args:
            prefix: Metric name prefix
            default_labels: Labels to add to all metrics
            histogram_buckets: Bucket boundaries for histograms
        """
        self._prefix = prefix
        self._default_labels = default_labels or {}
        self._histogram_buckets = histogram_buckets or self.DEFAULT_BUCKETS

        # Metric storage
        self._counters: Dict[str, Dict[Tuple[str, ...], float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self._gauges: Dict[str, Dict[Tuple[str, ...], float]] = defaultdict(dict)
        self._histograms: Dict[
            str, Dict[Tuple[str, ...], Tuple[List[int], float, int]]
        ] = defaultdict(dict)

        # Metric collectors
        self._collectors: List[Callable[[], List[MetricValue]]] = []

        self._lock = asyncio.Lock()

    def _prefixed_name(self, name: str) -> str:
        """Get prefixed metric name."""
        return f"{self._prefix}_{name}"

    def _labels_key(self, labels: Dict[str, str]) -> Tuple[str, ...]:
        """Convert labels to hashable key."""
        merged = {**self._default_labels, **labels}
        return tuple(sorted(merged.items()))

    def _labels_from_key(self, key: Tuple[str, ...]) -> Dict[str, str]:
        """Convert key back to labels dict."""
        return dict(key)

    def inc_counter(
        self,
        name: str,
        value: float = 1.0,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Increment a counter.

        Args:
            name: Counter name
            value: Value to add
            labels: Metric labels
        """
        full_name = self._prefixed_name(name)
        key = self._labels_key(labels or {})
        self._counters[full_name][key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Set a gauge value.

        Args:
            name: Gauge name
            value: Value to set
            labels: Metric labels
        """
        full_name = self._prefixed_name(name)
        key = self._labels_key(labels or {})
        self._gauges[full_name][key] = value

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Observe a histogram value.

        Args:
            name: Histogram name
            value: Value to observe
            labels: Metric labels
        """
        full_name = self._prefixed_name(name)
        key = self._labels_key(labels or {})

        if key not in self._histograms[full_name]:
            # Initialize: [bucket_counts], sum, count
            self._histograms[full_name][key] = (
                [0] * len(self._histogram_buckets),
                0.0,
                0,
            )

        buckets, total_sum, count = self._histograms[full_name][key]

        # Update buckets
        for i, boundary in enumerate(self._histogram_buckets):
            if value <= boundary:
                buckets[i] += 1

        # Update sum and count
        self._histograms[full_name][key] = (buckets, total_sum + value, count + 1)

    def register_collector(
        self,
        collector: Callable[[], List[MetricValue]],
    ) -> None:
        """
        Register a metric collector.

        Collectors are functions that return a list of MetricValue
        objects. They are called during metric export.

        Args:
            collector: Collector function
        """
        self._collectors.append(collector)

    def _collect_internal_metrics(self) -> List[MetricValue]:
        """Collect internal counter and gauge metrics."""
        metrics = []

        # Counters
        for name, values in self._counters.items():
            for key, value in values.items():
                metrics.append(
                    MetricValue(
                        name=name,
                        value=value,
                        labels=self._labels_from_key(key),
                        metric_type=MetricType.COUNTER,
                    )
                )

        # Gauges
        for name, values in self._gauges.items():
            for key, value in values.items():
                metrics.append(
                    MetricValue(
                        name=name,
                        value=value,
                        labels=self._labels_from_key(key),
                        metric_type=MetricType.GAUGE,
                    )
                )

        return metrics

    def _collect_histograms(self) -> List[HistogramValue]:
        """Collect histogram metrics."""
        histograms = []

        for name, values in self._histograms.items():
            for key, (buckets, total_sum, count) in values.items():
                bucket_values = []
                cumulative = 0
                for i, boundary in enumerate(self._histogram_buckets):
                    cumulative += buckets[i]
                    bucket_values.append(
                        HistogramBucket(le=boundary, count=cumulative)
                    )
                # Add +Inf bucket
                bucket_values.append(
                    HistogramBucket(le=float("inf"), count=count)
                )

                histograms.append(
                    HistogramValue(
                        name=name,
                        buckets=bucket_values,
                        sum_value=total_sum,
                        count=count,
                        labels=self._labels_from_key(key),
                    )
                )

        return histograms

    def export(self) -> str:
        """
        Export all metrics in Prometheus format.

        Returns:
            Prometheus exposition format string
        """
        metrics = self._collect_internal_metrics()

        # Run collectors
        for collector in self._collectors:
            try:
                metrics.extend(collector())
            except Exception as e:
                logger.error(f"Collector error: {e}")

        histograms = self._collect_histograms()

        return PrometheusFormatter.format_metrics(metrics, histograms)

    def export_json(self) -> Dict[str, Any]:
        """
        Export metrics as JSON.

        Returns:
            Dictionary of metrics
        """
        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        # Counters
        for name, values in self._counters.items():
            result["counters"][name] = [
                {"labels": self._labels_from_key(k), "value": v}
                for k, v in values.items()
            ]

        # Gauges
        for name, values in self._gauges.items():
            result["gauges"][name] = [
                {"labels": self._labels_from_key(k), "value": v}
                for k, v in values.items()
            ]

        # Histograms
        for name, values in self._histograms.items():
            result["histograms"][name] = []
            for key, (buckets, total_sum, count) in values.items():
                result["histograms"][name].append(
                    {
                        "labels": self._labels_from_key(key),
                        "buckets": dict(zip(self._histogram_buckets, buckets)),
                        "sum": total_sum,
                        "count": count,
                    }
                )

        return result

    def reset(self) -> None:
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# Global exporter instance
_global_exporter: Optional[MetricsExporter] = None


def get_metrics_exporter() -> MetricsExporter:
    """Get the global metrics exporter instance."""
    global _global_exporter
    if _global_exporter is None:
        _global_exporter = MetricsExporter()
    return _global_exporter


class MetricsMiddleware:
    """
    Middleware for automatic request metrics collection.

    Tracks request count, latency, and errors.
    """

    def __init__(
        self,
        exporter: Optional[MetricsExporter] = None,
        latency_histogram: str = "http_request_duration_ms",
        request_counter: str = "http_requests_total",
    ):
        """
        Initialize the middleware.

        Args:
            exporter: Metrics exporter
            latency_histogram: Name for latency histogram
            request_counter: Name for request counter
        """
        self._exporter = exporter or get_metrics_exporter()
        self._latency_histogram = latency_histogram
        self._request_counter = request_counter

    def record(
        self,
        method: str,
        path: str,
        status_code: int,
        latency_ms: float,
    ) -> None:
        """
        Record request metrics.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            latency_ms: Request latency in ms
        """
        labels = {
            "method": method,
            "path": path,
            "status": str(status_code),
        }

        self._exporter.inc_counter(self._request_counter, labels=labels)
        self._exporter.observe_histogram(
            self._latency_histogram, latency_ms, labels=labels
        )

    def __call__(self, method: str, path: str) -> "RequestTimer":
        """Create a request timer context manager."""
        return RequestTimer(self, method, path)


class RequestTimer:
    """Context manager for timing requests."""

    def __init__(self, middleware: MetricsMiddleware, method: str, path: str):
        self._middleware = middleware
        self._method = method
        self._path = path
        self._start: float = 0.0
        self._status_code: int = 200

    def set_status(self, status_code: int) -> "RequestTimer":
        """Set the response status code."""
        self._status_code = status_code
        return self

    def __enter__(self) -> "RequestTimer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if exc_type is not None:
            self._status_code = 500
        latency_ms = (time.perf_counter() - self._start) * 1000
        self._middleware.record(
            self._method, self._path, self._status_code, latency_ms
        )

    async def __aenter__(self) -> "RequestTimer":
        self._start = time.perf_counter()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)
