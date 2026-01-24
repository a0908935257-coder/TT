"""
System metrics collection module.

Provides comprehensive system health monitoring including
CPU, memory, disk, network, and process metrics.
"""

import asyncio
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Deque, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class SystemThresholds:
    """Thresholds for system health alerts."""

    # CPU
    cpu_warning_percent: float = 70.0
    cpu_critical_percent: float = 90.0

    # Memory
    memory_warning_percent: float = 75.0
    memory_critical_percent: float = 90.0

    # Disk
    disk_warning_percent: float = 80.0
    disk_critical_percent: float = 95.0

    # Network
    network_error_rate_warning: float = 0.01  # 1%
    network_error_rate_critical: float = 0.05  # 5%

    # Latency (ms)
    latency_warning_ms: float = 100.0
    latency_critical_ms: float = 500.0

    # Connection
    connection_warning_seconds: float = 30.0
    connection_critical_seconds: float = 60.0


@dataclass
class CPUMetrics:
    """CPU usage metrics."""

    percent: float
    user_percent: float = 0.0
    system_percent: float = 0.0
    idle_percent: float = 0.0
    iowait_percent: float = 0.0
    core_count: int = 1
    per_core: List[float] = field(default_factory=list)
    load_avg_1m: float = 0.0
    load_avg_5m: float = 0.0
    load_avg_15m: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "percent": round(self.percent, 2),
            "user_percent": round(self.user_percent, 2),
            "system_percent": round(self.system_percent, 2),
            "idle_percent": round(self.idle_percent, 2),
            "iowait_percent": round(self.iowait_percent, 2),
            "core_count": self.core_count,
            "per_core": [round(c, 2) for c in self.per_core],
            "load_avg_1m": round(self.load_avg_1m, 2),
            "load_avg_5m": round(self.load_avg_5m, 2),
            "load_avg_15m": round(self.load_avg_15m, 2),
        }


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent: float
    swap_total_mb: float = 0.0
    swap_used_mb: float = 0.0
    swap_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_mb": round(self.total_mb, 2),
            "available_mb": round(self.available_mb, 2),
            "used_mb": round(self.used_mb, 2),
            "percent": round(self.percent, 2),
            "swap_total_mb": round(self.swap_total_mb, 2),
            "swap_used_mb": round(self.swap_used_mb, 2),
            "swap_percent": round(self.swap_percent, 2),
        }


@dataclass
class DiskMetrics:
    """Disk usage metrics."""

    total_gb: float
    used_gb: float
    free_gb: float
    percent: float
    read_bytes_per_sec: float = 0.0
    write_bytes_per_sec: float = 0.0
    read_count_per_sec: float = 0.0
    write_count_per_sec: float = 0.0
    partitions: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_gb": round(self.total_gb, 2),
            "used_gb": round(self.used_gb, 2),
            "free_gb": round(self.free_gb, 2),
            "percent": round(self.percent, 2),
            "read_bytes_per_sec": round(self.read_bytes_per_sec, 2),
            "write_bytes_per_sec": round(self.write_bytes_per_sec, 2),
            "read_count_per_sec": round(self.read_count_per_sec, 2),
            "write_count_per_sec": round(self.write_count_per_sec, 2),
            "partitions": self.partitions,
        }


@dataclass
class NetworkMetrics:
    """Network usage metrics."""

    bytes_sent_per_sec: float = 0.0
    bytes_recv_per_sec: float = 0.0
    packets_sent_per_sec: float = 0.0
    packets_recv_per_sec: float = 0.0
    errors_in: int = 0
    errors_out: int = 0
    drops_in: int = 0
    drops_out: int = 0
    connections_count: int = 0
    connections_by_status: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bytes_sent_per_sec": round(self.bytes_sent_per_sec, 2),
            "bytes_recv_per_sec": round(self.bytes_recv_per_sec, 2),
            "packets_sent_per_sec": round(self.packets_sent_per_sec, 2),
            "packets_recv_per_sec": round(self.packets_recv_per_sec, 2),
            "errors_in": self.errors_in,
            "errors_out": self.errors_out,
            "drops_in": self.drops_in,
            "drops_out": self.drops_out,
            "connections_count": self.connections_count,
            "connections_by_status": self.connections_by_status,
        }


@dataclass
class ProcessMetrics:
    """Process-specific metrics."""

    pid: int
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    num_threads: int
    num_fds: int = 0  # File descriptors
    status: str = "running"
    uptime_seconds: float = 0.0
    create_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "pid": self.pid,
            "cpu_percent": round(self.cpu_percent, 2),
            "memory_mb": round(self.memory_mb, 2),
            "memory_percent": round(self.memory_percent, 2),
            "num_threads": self.num_threads,
            "num_fds": self.num_fds,
            "status": self.status,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "create_time": self.create_time.isoformat() if self.create_time else None,
        }


@dataclass
class SystemSnapshot:
    """Complete system metrics snapshot."""

    timestamp: datetime
    cpu: CPUMetrics
    memory: MemoryMetrics
    disk: DiskMetrics
    network: NetworkMetrics
    process: ProcessMetrics
    health_status: str = "healthy"  # healthy, degraded, unhealthy
    alerts: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu": self.cpu.to_dict(),
            "memory": self.memory.to_dict(),
            "disk": self.disk.to_dict(),
            "network": self.network.to_dict(),
            "process": self.process.to_dict(),
            "health_status": self.health_status,
            "alerts": self.alerts,
        }


class SystemMetricsCollector:
    """
    Collects comprehensive system metrics.

    Provides CPU, memory, disk, network, and process monitoring
    with configurable thresholds and alerts.
    """

    def __init__(
        self,
        thresholds: Optional[SystemThresholds] = None,
        history_size: int = 3600,  # 1 hour at 1 sample/sec
    ):
        """
        Initialize the collector.

        Args:
            thresholds: Health check thresholds
            history_size: Number of snapshots to keep
        """
        self._thresholds = thresholds or SystemThresholds()
        self._history_size = history_size
        self._history: Deque[SystemSnapshot] = deque(maxlen=history_size)

        # For rate calculations
        self._last_net_io: Optional[Tuple[float, Any]] = None
        self._last_disk_io: Optional[Tuple[float, Any]] = None

        # Process info
        self._pid = os.getpid()
        self._process_start_time: Optional[datetime] = None

    def collect(self) -> SystemSnapshot:
        """
        Collect current system metrics.

        Returns:
            SystemSnapshot with all metrics
        """
        try:
            import psutil

            now = datetime.now(timezone.utc)
            alerts = []

            # CPU metrics
            cpu = self._collect_cpu(psutil)
            if cpu.percent >= self._thresholds.cpu_critical_percent:
                alerts.append(f"CPU critical: {cpu.percent:.1f}%")
            elif cpu.percent >= self._thresholds.cpu_warning_percent:
                alerts.append(f"CPU warning: {cpu.percent:.1f}%")

            # Memory metrics
            memory = self._collect_memory(psutil)
            if memory.percent >= self._thresholds.memory_critical_percent:
                alerts.append(f"Memory critical: {memory.percent:.1f}%")
            elif memory.percent >= self._thresholds.memory_warning_percent:
                alerts.append(f"Memory warning: {memory.percent:.1f}%")

            # Disk metrics
            disk = self._collect_disk(psutil)
            if disk.percent >= self._thresholds.disk_critical_percent:
                alerts.append(f"Disk critical: {disk.percent:.1f}%")
            elif disk.percent >= self._thresholds.disk_warning_percent:
                alerts.append(f"Disk warning: {disk.percent:.1f}%")

            # Network metrics
            network = self._collect_network(psutil)

            # Process metrics
            process = self._collect_process(psutil)

            # Determine health status
            if any("critical" in a for a in alerts):
                health_status = "unhealthy"
            elif any("warning" in a for a in alerts):
                health_status = "degraded"
            else:
                health_status = "healthy"

            snapshot = SystemSnapshot(
                timestamp=now,
                cpu=cpu,
                memory=memory,
                disk=disk,
                network=network,
                process=process,
                health_status=health_status,
                alerts=alerts,
            )

            self._history.append(snapshot)
            return snapshot

        except ImportError:
            logger.warning("psutil not installed, returning minimal metrics")
            return self._minimal_snapshot()

    def _collect_cpu(self, psutil: Any) -> CPUMetrics:
        """Collect CPU metrics."""
        cpu_times = psutil.cpu_times_percent()
        cpu_percent = psutil.cpu_percent()
        per_core = psutil.cpu_percent(percpu=True)
        load_avg = psutil.getloadavg() if hasattr(psutil, "getloadavg") else (0, 0, 0)

        return CPUMetrics(
            percent=cpu_percent,
            user_percent=cpu_times.user if hasattr(cpu_times, "user") else 0,
            system_percent=cpu_times.system if hasattr(cpu_times, "system") else 0,
            idle_percent=cpu_times.idle if hasattr(cpu_times, "idle") else 0,
            iowait_percent=cpu_times.iowait if hasattr(cpu_times, "iowait") else 0,
            core_count=psutil.cpu_count(),
            per_core=per_core,
            load_avg_1m=load_avg[0],
            load_avg_5m=load_avg[1],
            load_avg_15m=load_avg[2],
        )

    def _collect_memory(self, psutil: Any) -> MemoryMetrics:
        """Collect memory metrics."""
        mem = psutil.virtual_memory()
        swap = psutil.swap_memory()

        return MemoryMetrics(
            total_mb=mem.total / 1024 / 1024,
            available_mb=mem.available / 1024 / 1024,
            used_mb=mem.used / 1024 / 1024,
            percent=mem.percent,
            swap_total_mb=swap.total / 1024 / 1024,
            swap_used_mb=swap.used / 1024 / 1024,
            swap_percent=swap.percent,
        )

    def _collect_disk(self, psutil: Any) -> DiskMetrics:
        """Collect disk metrics."""
        # Main disk
        disk = psutil.disk_usage("/")

        # Disk I/O rates
        disk_io = psutil.disk_io_counters()
        now = time.time()

        read_rate = 0.0
        write_rate = 0.0
        read_count_rate = 0.0
        write_count_rate = 0.0

        if self._last_disk_io is not None:
            last_time, last_io = self._last_disk_io
            elapsed = now - last_time
            if elapsed > 0:
                read_rate = (disk_io.read_bytes - last_io.read_bytes) / elapsed
                write_rate = (disk_io.write_bytes - last_io.write_bytes) / elapsed
                read_count_rate = (disk_io.read_count - last_io.read_count) / elapsed
                write_count_rate = (disk_io.write_count - last_io.write_count) / elapsed

        self._last_disk_io = (now, disk_io)

        # Partition info
        partitions = {}
        for part in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(part.mountpoint)
                partitions[part.mountpoint] = {
                    "total_gb": usage.total / 1024 / 1024 / 1024,
                    "used_gb": usage.used / 1024 / 1024 / 1024,
                    "percent": usage.percent,
                }
            except (PermissionError, OSError):
                continue

        return DiskMetrics(
            total_gb=disk.total / 1024 / 1024 / 1024,
            used_gb=disk.used / 1024 / 1024 / 1024,
            free_gb=disk.free / 1024 / 1024 / 1024,
            percent=disk.percent,
            read_bytes_per_sec=read_rate,
            write_bytes_per_sec=write_rate,
            read_count_per_sec=read_count_rate,
            write_count_per_sec=write_count_rate,
            partitions=partitions,
        )

    def _collect_network(self, psutil: Any) -> NetworkMetrics:
        """Collect network metrics."""
        net_io = psutil.net_io_counters()
        now = time.time()

        bytes_sent_rate = 0.0
        bytes_recv_rate = 0.0
        packets_sent_rate = 0.0
        packets_recv_rate = 0.0

        if self._last_net_io is not None:
            last_time, last_io = self._last_net_io
            elapsed = now - last_time
            if elapsed > 0:
                bytes_sent_rate = (net_io.bytes_sent - last_io.bytes_sent) / elapsed
                bytes_recv_rate = (net_io.bytes_recv - last_io.bytes_recv) / elapsed
                packets_sent_rate = (net_io.packets_sent - last_io.packets_sent) / elapsed
                packets_recv_rate = (net_io.packets_recv - last_io.packets_recv) / elapsed

        self._last_net_io = (now, net_io)

        # Connection counts
        connections_by_status: Dict[str, int] = {}
        try:
            for conn in psutil.net_connections():
                status = conn.status
                connections_by_status[status] = connections_by_status.get(status, 0) + 1
        except (psutil.AccessDenied, PermissionError):
            pass

        return NetworkMetrics(
            bytes_sent_per_sec=bytes_sent_rate,
            bytes_recv_per_sec=bytes_recv_rate,
            packets_sent_per_sec=packets_sent_rate,
            packets_recv_per_sec=packets_recv_rate,
            errors_in=net_io.errin,
            errors_out=net_io.errout,
            drops_in=net_io.dropin,
            drops_out=net_io.dropout,
            connections_count=sum(connections_by_status.values()),
            connections_by_status=connections_by_status,
        )

    def _collect_process(self, psutil: Any) -> ProcessMetrics:
        """Collect process metrics."""
        try:
            proc = psutil.Process(self._pid)

            if self._process_start_time is None:
                self._process_start_time = datetime.fromtimestamp(
                    proc.create_time(), tz=timezone.utc
                )

            uptime = (datetime.now(timezone.utc) - self._process_start_time).total_seconds()

            # Get file descriptor count (Unix only)
            try:
                num_fds = proc.num_fds()
            except (AttributeError, psutil.AccessDenied):
                num_fds = 0

            return ProcessMetrics(
                pid=self._pid,
                cpu_percent=proc.cpu_percent(),
                memory_mb=proc.memory_info().rss / 1024 / 1024,
                memory_percent=proc.memory_percent(),
                num_threads=proc.num_threads(),
                num_fds=num_fds,
                status=proc.status(),
                uptime_seconds=uptime,
                create_time=self._process_start_time,
            )
        except Exception as e:
            logger.error(f"Error collecting process metrics: {e}")
            return ProcessMetrics(
                pid=self._pid,
                cpu_percent=0.0,
                memory_mb=0.0,
                memory_percent=0.0,
                num_threads=1,
            )

    def _minimal_snapshot(self) -> SystemSnapshot:
        """Create minimal snapshot when psutil not available."""
        now = datetime.now(timezone.utc)
        return SystemSnapshot(
            timestamp=now,
            cpu=CPUMetrics(percent=0.0),
            memory=MemoryMetrics(total_mb=0, available_mb=0, used_mb=0, percent=0),
            disk=DiskMetrics(total_gb=0, used_gb=0, free_gb=0, percent=0),
            network=NetworkMetrics(),
            process=ProcessMetrics(
                pid=self._pid, cpu_percent=0, memory_mb=0, memory_percent=0, num_threads=1
            ),
            health_status="unknown",
            alerts=["psutil not installed"],
        )

    def get_history(
        self,
        seconds: Optional[int] = None,
    ) -> List[SystemSnapshot]:
        """
        Get historical snapshots.

        Args:
            seconds: Number of seconds of history (default: all)

        Returns:
            List of snapshots
        """
        if seconds is None:
            return list(self._history)

        cutoff = datetime.now(timezone.utc).timestamp() - seconds
        return [s for s in self._history if s.timestamp.timestamp() >= cutoff]

    def get_averages(
        self,
        seconds: int = 60,
    ) -> Dict[str, float]:
        """
        Get average metrics over time period.

        Args:
            seconds: Time period

        Returns:
            Dictionary of average values
        """
        history = self.get_history(seconds)
        if not history:
            return {}

        n = len(history)
        return {
            "avg_cpu_percent": sum(s.cpu.percent for s in history) / n,
            "max_cpu_percent": max(s.cpu.percent for s in history),
            "avg_memory_percent": sum(s.memory.percent for s in history) / n,
            "max_memory_percent": max(s.memory.percent for s in history),
            "avg_disk_percent": sum(s.disk.percent for s in history) / n,
            "avg_process_memory_mb": sum(s.process.memory_mb for s in history) / n,
            "avg_process_cpu_percent": sum(s.process.cpu_percent for s in history) / n,
            "samples": n,
        }

    def check_health(self) -> Tuple[str, List[str]]:
        """
        Check system health against thresholds.

        Returns:
            Tuple of (status, list of alerts)
        """
        snapshot = self.collect()
        return snapshot.health_status, snapshot.alerts


class SystemMetricsMonitor:
    """
    Continuous system metrics monitoring.

    Collects metrics at regular intervals and triggers alerts.
    """

    def __init__(
        self,
        collector: Optional[SystemMetricsCollector] = None,
        interval_seconds: float = 1.0,
    ):
        """
        Initialize the monitor.

        Args:
            collector: Metrics collector
            interval_seconds: Collection interval
        """
        self._collector = collector or SystemMetricsCollector()
        self._interval = interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._callbacks: List[Any] = []

    def on_snapshot(self, callback: Any) -> None:
        """Register callback for new snapshots."""
        self._callbacks.append(callback)

    async def start(self) -> None:
        """Start continuous monitoring."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"System metrics monitor started (interval: {self._interval}s)")

    async def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("System metrics monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                snapshot = self._collector.collect()

                # Trigger callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(snapshot)
                        else:
                            callback(snapshot)
                    except Exception as e:
                        logger.error(f"Callback error: {e}")

                await asyncio.sleep(self._interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(1.0)

    @property
    def collector(self) -> SystemMetricsCollector:
        """Get the metrics collector."""
        return self._collector

    def get_current(self) -> SystemSnapshot:
        """Get current snapshot."""
        return self._collector.collect()

    def get_averages(self, seconds: int = 60) -> Dict[str, float]:
        """Get average metrics."""
        return self._collector.get_averages(seconds)
