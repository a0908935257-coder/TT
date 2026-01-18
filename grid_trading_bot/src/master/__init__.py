"""
Master Module.

Bot registration center, lifecycle management, heartbeat monitoring,
and health checking.
"""

from .models import (
    BotAlreadyExistsError,
    BotInfo,
    BotNotFoundError,
    BotState,
    BotType,
    InvalidStateTransitionError,
    MarketType,
    RegistryEvent,
    VALID_STATE_TRANSITIONS,
)
from .registry import BotRegistry
from .heartbeat import (
    HeartbeatConfig,
    HeartbeatData,
    HeartbeatMonitor,
)
from .health import (
    CheckItem,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
)
from .aggregator import (
    BotMetrics,
    MetricsAggregator,
)
from .dashboard import (
    Alert,
    AlertLevel,
    BotDetail,
    Dashboard,
    DashboardData,
    DashboardSummary,
)

__all__ = [
    # Models
    "BotInfo",
    "BotState",
    "BotType",
    "MarketType",
    "RegistryEvent",
    "VALID_STATE_TRANSITIONS",
    # Exceptions
    "BotAlreadyExistsError",
    "BotNotFoundError",
    "InvalidStateTransitionError",
    # Registry
    "BotRegistry",
    # Heartbeat
    "HeartbeatConfig",
    "HeartbeatData",
    "HeartbeatMonitor",
    # Health
    "CheckItem",
    "HealthCheckResult",
    "HealthChecker",
    "HealthStatus",
    # Aggregator
    "BotMetrics",
    "MetricsAggregator",
    # Dashboard
    "Alert",
    "AlertLevel",
    "BotDetail",
    "Dashboard",
    "DashboardData",
    "DashboardSummary",
]
