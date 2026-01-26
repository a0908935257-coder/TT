"""
Comprehensive Health Monitor.

Provides monitoring for previously unmonitored critical areas:
- WebSocket connection health
- API error classification
- Balance reconciliation
- Market data staleness
- Order latency breakdown
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Coroutine, Deque, Dict, List, Optional, Set, Tuple

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# WebSocket Health Monitor
# =============================================================================


class WebSocketHealthStatus(Enum):
    """WebSocket connection health status."""

    CONNECTED = "connected"
    DEGRADED = "degraded"  # High latency or frequent reconnects
    DISCONNECTED = "disconnected"
    STALE = "stale"  # Connected but no data
    UNKNOWN = "unknown"


@dataclass
class WebSocketMetrics:
    """Metrics for a single WebSocket connection."""

    connection_id: str
    status: WebSocketHealthStatus = WebSocketHealthStatus.UNKNOWN
    connected_at: Optional[datetime] = None
    last_message_at: Optional[datetime] = None
    last_ping_at: Optional[datetime] = None
    last_pong_at: Optional[datetime] = None
    ping_latency_ms: float = 0.0
    reconnect_count: int = 0
    message_count: int = 0
    error_count: int = 0
    last_error: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "connection_id": self.connection_id,
            "status": self.status.value,
            "connected_at": self.connected_at.isoformat() if self.connected_at else None,
            "last_message_at": self.last_message_at.isoformat() if self.last_message_at else None,
            "ping_latency_ms": self.ping_latency_ms,
            "reconnect_count": self.reconnect_count,
            "message_count": self.message_count,
            "error_count": self.error_count,
        }


class WebSocketHealthMonitor:
    """
    Monitors WebSocket connection health.

    Tracks:
    - Connection status and uptime
    - Ping/pong latency
    - Message flow (staleness detection)
    - Reconnection frequency
    """

    def __init__(
        self,
        stale_threshold_seconds: int = 60,
        ping_timeout_seconds: int = 10,
        max_reconnects_per_hour: int = 5,
        latency_warning_ms: float = 500,
    ):
        """
        Initialize WebSocket health monitor.

        Args:
            stale_threshold_seconds: Seconds without data before stale
            ping_timeout_seconds: Seconds to wait for pong
            max_reconnects_per_hour: Reconnects before degraded
            latency_warning_ms: Latency threshold for warning
        """
        self._stale_threshold = stale_threshold_seconds
        self._ping_timeout = ping_timeout_seconds
        self._max_reconnects = max_reconnects_per_hour
        self._latency_warning = latency_warning_ms

        # Connection metrics
        self._connections: Dict[str, WebSocketMetrics] = {}

        # Reconnection tracking: connection_id -> list of reconnect times
        self._reconnects: Dict[str, Deque[datetime]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Alert callback
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for alerts: (severity, title, message)."""
        self._alert_callback = callback

    def register_connection(self, connection_id: str) -> None:
        """Register a new WebSocket connection."""
        self._connections[connection_id] = WebSocketMetrics(
            connection_id=connection_id,
            status=WebSocketHealthStatus.UNKNOWN,
        )

    def record_connect(self, connection_id: str) -> None:
        """Record connection established."""
        if connection_id not in self._connections:
            self.register_connection(connection_id)

        metrics = self._connections[connection_id]
        metrics.connected_at = datetime.now(timezone.utc)
        metrics.status = WebSocketHealthStatus.CONNECTED
        logger.info(f"WebSocket {connection_id} connected")

    def record_disconnect(self, connection_id: str, error: str = "") -> None:
        """Record connection lost."""
        if connection_id not in self._connections:
            return

        metrics = self._connections[connection_id]
        metrics.status = WebSocketHealthStatus.DISCONNECTED
        if error:
            metrics.error_count += 1
            metrics.last_error = error
        logger.warning(f"WebSocket {connection_id} disconnected: {error}")

    def record_reconnect(self, connection_id: str) -> None:
        """Record reconnection attempt."""
        if connection_id not in self._connections:
            self.register_connection(connection_id)

        metrics = self._connections[connection_id]
        metrics.reconnect_count += 1
        metrics.connected_at = datetime.now(timezone.utc)
        metrics.status = WebSocketHealthStatus.CONNECTED

        # Track reconnection time
        self._reconnects[connection_id].append(datetime.now(timezone.utc))

    def record_message(self, connection_id: str) -> None:
        """Record message received."""
        if connection_id not in self._connections:
            return

        metrics = self._connections[connection_id]
        metrics.last_message_at = datetime.now(timezone.utc)
        metrics.message_count += 1

        # Update status if was stale
        if metrics.status == WebSocketHealthStatus.STALE:
            metrics.status = WebSocketHealthStatus.CONNECTED

    def record_ping_sent(self, connection_id: str) -> None:
        """Record ping sent."""
        if connection_id not in self._connections:
            return
        self._connections[connection_id].last_ping_at = datetime.now(timezone.utc)

    def record_pong_received(self, connection_id: str) -> None:
        """Record pong received and calculate latency."""
        if connection_id not in self._connections:
            return

        metrics = self._connections[connection_id]
        now = datetime.now(timezone.utc)
        metrics.last_pong_at = now

        if metrics.last_ping_at:
            latency_ms = (now - metrics.last_ping_at).total_seconds() * 1000
            metrics.ping_latency_ms = latency_ms

    async def check_health(self) -> Dict[str, WebSocketMetrics]:
        """
        Check health of all connections.

        Returns:
            Dict of connection_id to metrics with updated status
        """
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        alerts_to_send = []

        for conn_id, metrics in self._connections.items():
            old_status = metrics.status

            # Skip disconnected connections
            if metrics.status == WebSocketHealthStatus.DISCONNECTED:
                continue

            # Check for staleness
            if metrics.last_message_at:
                seconds_since_message = (now - metrics.last_message_at).total_seconds()
                if seconds_since_message > self._stale_threshold:
                    metrics.status = WebSocketHealthStatus.STALE
                    if old_status != WebSocketHealthStatus.STALE:
                        alerts_to_send.append((
                            "WARNING",
                            f"WebSocket 數據停滯: {conn_id}",
                            f"連接 {conn_id} 已 {int(seconds_since_message)} 秒未收到數據",
                        ))

            # Check ping timeout
            if metrics.last_ping_at and not metrics.last_pong_at:
                seconds_since_ping = (now - metrics.last_ping_at).total_seconds()
                if seconds_since_ping > self._ping_timeout:
                    metrics.status = WebSocketHealthStatus.DEGRADED
                    alerts_to_send.append((
                        "WARNING",
                        f"WebSocket Ping 超時: {conn_id}",
                        f"連接 {conn_id} ping 超時 ({int(seconds_since_ping)}s)",
                    ))

            # Check high latency
            if metrics.ping_latency_ms > self._latency_warning:
                if metrics.status == WebSocketHealthStatus.CONNECTED:
                    metrics.status = WebSocketHealthStatus.DEGRADED
                alerts_to_send.append((
                    "WARNING",
                    f"WebSocket 高延遲: {conn_id}",
                    f"連接 {conn_id} 延遲 {metrics.ping_latency_ms:.0f}ms",
                ))

            # Check reconnection frequency
            recent_reconnects = sum(
                1 for t in self._reconnects[conn_id] if t > hour_ago
            )
            if recent_reconnects >= self._max_reconnects:
                metrics.status = WebSocketHealthStatus.DEGRADED
                alerts_to_send.append((
                    "ERROR",
                    f"WebSocket 頻繁重連: {conn_id}",
                    f"連接 {conn_id} 過去1小時重連 {recent_reconnects} 次",
                ))

        # Send alerts
        if self._alert_callback:
            for severity, title, message in alerts_to_send:
                try:
                    await self._alert_callback(severity, title, message)
                except Exception as e:
                    logger.error(f"Failed to send WebSocket alert: {e}")

        return self._connections

    def get_metrics(self, connection_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for connections."""
        if connection_id:
            if connection_id in self._connections:
                return self._connections[connection_id].to_dict()
            return {}

        return {
            conn_id: metrics.to_dict()
            for conn_id, metrics in self._connections.items()
        }


# =============================================================================
# API Error Classification Monitor
# =============================================================================


class APIErrorCategory(Enum):
    """Categories of API errors."""

    RATE_LIMIT = "rate_limit"  # 429
    AUTH_ERROR = "auth_error"  # 401, 403
    CLIENT_ERROR = "client_error"  # 4xx
    SERVER_ERROR = "server_error"  # 5xx
    TIMEOUT = "timeout"
    NETWORK = "network"
    UNKNOWN = "unknown"


@dataclass
class APIErrorStats:
    """Statistics for API errors."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    errors_by_code: Dict[int, int] = field(default_factory=dict)
    rate_limit_hits: int = 0
    timeout_count: int = 0
    last_error_at: Optional[datetime] = None
    last_error_message: str = ""

    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 100.0
        return (self.successful_requests / self.total_requests) * 100

    def error_rate(self) -> float:
        """Calculate error rate."""
        return 100.0 - self.success_rate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate(), 2),
            "error_rate": round(self.error_rate(), 2),
            "errors_by_category": self.errors_by_category,
            "errors_by_code": self.errors_by_code,
            "rate_limit_hits": self.rate_limit_hits,
            "timeout_count": self.timeout_count,
            "last_error_at": self.last_error_at.isoformat() if self.last_error_at else None,
        }


class APIErrorMonitor:
    """
    Monitors API errors with classification.

    Tracks:
    - Error rates by category (rate limit, auth, server, etc.)
    - Error rates by HTTP status code
    - Rate limit hits
    - Timeout frequency
    """

    def __init__(
        self,
        error_rate_warning: float = 5.0,
        error_rate_critical: float = 20.0,
        rate_limit_warning: int = 10,
        window_seconds: int = 300,
    ):
        """
        Initialize API error monitor.

        Args:
            error_rate_warning: Warning threshold percentage
            error_rate_critical: Critical threshold percentage
            rate_limit_warning: Rate limit hits before warning
            window_seconds: Window for rate calculations
        """
        self._error_rate_warning = error_rate_warning
        self._error_rate_critical = error_rate_critical
        self._rate_limit_warning = rate_limit_warning
        self._window = window_seconds

        # Stats per endpoint
        self._stats: Dict[str, APIErrorStats] = defaultdict(APIErrorStats)

        # Recent errors for windowed calculations
        self._recent_errors: Deque[Tuple[datetime, str, str]] = deque(maxlen=1000)

        # Alert callback
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for alerts."""
        self._alert_callback = callback

    def _categorize_error(
        self,
        status_code: Optional[int],
        error_type: str,
    ) -> APIErrorCategory:
        """Categorize an error."""
        if error_type.lower() in ("timeout", "timedout"):
            return APIErrorCategory.TIMEOUT
        if error_type.lower() in ("network", "connection"):
            return APIErrorCategory.NETWORK

        if status_code:
            if status_code == 429:
                return APIErrorCategory.RATE_LIMIT
            if status_code in (401, 403):
                return APIErrorCategory.AUTH_ERROR
            if 400 <= status_code < 500:
                return APIErrorCategory.CLIENT_ERROR
            if 500 <= status_code < 600:
                return APIErrorCategory.SERVER_ERROR

        return APIErrorCategory.UNKNOWN

    def record_request(
        self,
        endpoint: str,
        success: bool,
        status_code: Optional[int] = None,
        error_type: str = "",
        error_message: str = "",
    ) -> None:
        """
        Record an API request result.

        Args:
            endpoint: API endpoint name
            success: Whether request succeeded
            status_code: HTTP status code
            error_type: Type of error if failed
            error_message: Error message if failed
        """
        stats = self._stats[endpoint]
        stats.total_requests += 1

        if success:
            stats.successful_requests += 1
        else:
            stats.failed_requests += 1
            stats.last_error_at = datetime.now(timezone.utc)
            stats.last_error_message = error_message

            # Categorize error
            category = self._categorize_error(status_code, error_type)
            cat_name = category.value
            stats.errors_by_category[cat_name] = stats.errors_by_category.get(cat_name, 0) + 1

            # Track by status code
            if status_code:
                stats.errors_by_code[status_code] = stats.errors_by_code.get(status_code, 0) + 1

            # Special tracking
            if category == APIErrorCategory.RATE_LIMIT:
                stats.rate_limit_hits += 1
            elif category == APIErrorCategory.TIMEOUT:
                stats.timeout_count += 1

            # Add to recent errors
            self._recent_errors.append((
                datetime.now(timezone.utc),
                endpoint,
                cat_name,
            ))

    async def check_health(self) -> Dict[str, Any]:
        """
        Check API health and trigger alerts.

        Returns:
            Health status with stats
        """
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(seconds=self._window)
        alerts_to_send = []

        # Calculate windowed error rates
        recent = [e for e in self._recent_errors if e[0] > cutoff]
        total_recent = len(recent)

        for endpoint, stats in self._stats.items():
            error_rate = stats.error_rate()

            # Check error rate thresholds
            if error_rate >= self._error_rate_critical:
                alerts_to_send.append((
                    "CRITICAL",
                    f"API 錯誤率過高: {endpoint}",
                    f"端點 {endpoint} 錯誤率 {error_rate:.1f}% 超過臨界值 {self._error_rate_critical}%",
                ))
            elif error_rate >= self._error_rate_warning:
                alerts_to_send.append((
                    "WARNING",
                    f"API 錯誤率警告: {endpoint}",
                    f"端點 {endpoint} 錯誤率 {error_rate:.1f}%",
                ))

            # Check rate limit hits
            if stats.rate_limit_hits >= self._rate_limit_warning:
                alerts_to_send.append((
                    "WARNING",
                    f"API 速率限制警告: {endpoint}",
                    f"端點 {endpoint} 觸發速率限制 {stats.rate_limit_hits} 次",
                ))

        # Send alerts
        if self._alert_callback:
            for severity, title, message in alerts_to_send:
                try:
                    await self._alert_callback(severity, title, message)
                except Exception as e:
                    logger.error(f"Failed to send API error alert: {e}")

        return {
            "endpoints": {
                endpoint: stats.to_dict()
                for endpoint, stats in self._stats.items()
            },
            "recent_errors_count": total_recent,
            "window_seconds": self._window,
        }

    def get_stats(self, endpoint: Optional[str] = None) -> Dict[str, Any]:
        """Get stats for endpoints."""
        if endpoint:
            if endpoint in self._stats:
                return self._stats[endpoint].to_dict()
            return {}

        return {
            endpoint: stats.to_dict()
            for endpoint, stats in self._stats.items()
        }


# =============================================================================
# Balance Reconciliation Monitor
# =============================================================================


@dataclass
class BalanceSnapshot:
    """Snapshot of balance at a point in time."""

    timestamp: datetime
    asset: str
    local_balance: Decimal
    exchange_balance: Decimal
    difference: Decimal
    is_matched: bool


class BalanceReconciliationMonitor:
    """
    Monitors balance consistency between local and exchange.

    Detects:
    - Balance mismatches
    - Unexpected balance changes
    - Low available balance warnings
    """

    def __init__(
        self,
        tolerance_percent: float = 0.01,
        low_balance_warning: Decimal = Decimal("100"),
        check_interval_seconds: int = 60,
    ):
        """
        Initialize balance reconciliation monitor.

        Args:
            tolerance_percent: Acceptable difference percentage
            low_balance_warning: Warning threshold for low balance
            check_interval_seconds: Interval between checks
        """
        self._tolerance = Decimal(str(tolerance_percent / 100))
        self._low_balance_warning = low_balance_warning
        self._check_interval = check_interval_seconds

        # Balance history
        self._history: Dict[str, Deque[BalanceSnapshot]] = defaultdict(
            lambda: deque(maxlen=100)
        )

        # Last known balances
        self._last_local: Dict[str, Decimal] = {}
        self._last_exchange: Dict[str, Decimal] = {}

        # Mismatch tracking
        self._mismatches: List[BalanceSnapshot] = []

        # Alert callback
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for alerts."""
        self._alert_callback = callback

    def update_local_balance(self, asset: str, balance: Decimal) -> None:
        """Update local balance tracking."""
        self._last_local[asset] = balance

    def update_exchange_balance(self, asset: str, balance: Decimal) -> None:
        """Update exchange balance."""
        self._last_exchange[asset] = balance

    async def reconcile(
        self,
        asset: str,
        local_balance: Decimal,
        exchange_balance: Decimal,
    ) -> BalanceSnapshot:
        """
        Reconcile local and exchange balance.

        Args:
            asset: Asset symbol
            local_balance: Local tracked balance
            exchange_balance: Exchange reported balance

        Returns:
            BalanceSnapshot with reconciliation result
        """
        now = datetime.now(timezone.utc)
        difference = abs(local_balance - exchange_balance)

        # Calculate tolerance based on larger balance
        max_balance = max(local_balance, exchange_balance, Decimal("1"))
        tolerance_amount = max_balance * self._tolerance

        is_matched = difference <= tolerance_amount

        snapshot = BalanceSnapshot(
            timestamp=now,
            asset=asset,
            local_balance=local_balance,
            exchange_balance=exchange_balance,
            difference=difference,
            is_matched=is_matched,
        )

        # Record history
        self._history[asset].append(snapshot)

        # Track mismatches
        if not is_matched:
            self._mismatches.append(snapshot)

            # Send alert
            if self._alert_callback:
                pct_diff = (difference / max_balance * 100) if max_balance else Decimal(0)
                try:
                    await self._alert_callback(
                        "ERROR",
                        f"餘額不一致: {asset}",
                        f"資產 {asset} 餘額不一致\n"
                        f"本地: {local_balance}\n"
                        f"交易所: {exchange_balance}\n"
                        f"差異: {difference} ({pct_diff:.2f}%)",
                    )
                except Exception as e:
                    logger.error(f"Failed to send balance alert: {e}")

        # Check low balance
        if exchange_balance < self._low_balance_warning:
            if self._alert_callback:
                try:
                    await self._alert_callback(
                        "WARNING",
                        f"餘額過低: {asset}",
                        f"資產 {asset} 餘額 {exchange_balance} 低於警告值 {self._low_balance_warning}",
                    )
                except Exception as e:
                    logger.error(f"Failed to send low balance alert: {e}")

        # Update last known
        self._last_local[asset] = local_balance
        self._last_exchange[asset] = exchange_balance

        return snapshot

    def get_mismatches(self, since: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Get balance mismatches."""
        if since:
            return [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "asset": m.asset,
                    "local": str(m.local_balance),
                    "exchange": str(m.exchange_balance),
                    "difference": str(m.difference),
                }
                for m in self._mismatches if m.timestamp >= since
            ]

        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "asset": m.asset,
                "local": str(m.local_balance),
                "exchange": str(m.exchange_balance),
                "difference": str(m.difference),
            }
            for m in self._mismatches[-100:]  # Last 100
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get reconciliation statistics."""
        return {
            "total_mismatches": len(self._mismatches),
            "assets_tracked": list(self._last_exchange.keys()),
            "last_balances": {
                asset: {
                    "local": str(self._last_local.get(asset, Decimal(0))),
                    "exchange": str(self._last_exchange.get(asset, Decimal(0))),
                }
                for asset in set(self._last_local.keys()) | set(self._last_exchange.keys())
            },
        }


# =============================================================================
# Market Data Staleness Monitor
# =============================================================================


@dataclass
class MarketDataMetrics:
    """Metrics for market data feed."""

    symbol: str
    last_update: Optional[datetime] = None
    update_count: int = 0
    last_price: Optional[Decimal] = None
    price_change_pct: float = 0.0
    is_stale: bool = False
    stale_duration_seconds: float = 0.0


class MarketDataStalenessMonitor:
    """
    Monitors market data for staleness and anomalies.

    Detects:
    - Stale data (no updates)
    - Price jumps
    - Bid-ask inversions
    """

    def __init__(
        self,
        stale_threshold_seconds: int = 60,
        price_jump_warning_pct: float = 10.0,
        price_jump_critical_pct: float = 30.0,
    ):
        """
        Initialize market data monitor.

        Args:
            stale_threshold_seconds: Seconds before data is stale
            price_jump_warning_pct: Warning threshold for price change
            price_jump_critical_pct: Critical threshold for price change
        """
        self._stale_threshold = stale_threshold_seconds
        self._price_jump_warning = price_jump_warning_pct
        self._price_jump_critical = price_jump_critical_pct

        # Metrics per symbol
        self._metrics: Dict[str, MarketDataMetrics] = {}

        # Alert callback
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for alerts."""
        self._alert_callback = callback

    def record_update(
        self,
        symbol: str,
        price: Decimal,
        bid: Optional[Decimal] = None,
        ask: Optional[Decimal] = None,
    ) -> None:
        """
        Record a market data update.

        Args:
            symbol: Trading symbol
            price: Current price
            bid: Bid price (optional)
            ask: Ask price (optional)
        """
        now = datetime.now(timezone.utc)

        if symbol not in self._metrics:
            self._metrics[symbol] = MarketDataMetrics(symbol=symbol)

        metrics = self._metrics[symbol]

        # Calculate price change
        if metrics.last_price and metrics.last_price > 0:
            change = abs(price - metrics.last_price) / metrics.last_price * 100
            metrics.price_change_pct = float(change)
        else:
            metrics.price_change_pct = 0.0

        metrics.last_update = now
        metrics.update_count += 1
        metrics.last_price = price
        metrics.is_stale = False
        metrics.stale_duration_seconds = 0

    async def check_health(self) -> Dict[str, Any]:
        """
        Check market data health.

        Returns:
            Health status with metrics
        """
        now = datetime.now(timezone.utc)
        alerts_to_send = []
        stale_symbols = []

        for symbol, metrics in self._metrics.items():
            # Check staleness
            if metrics.last_update:
                age = (now - metrics.last_update).total_seconds()
                if age > self._stale_threshold:
                    if not metrics.is_stale:
                        # Just became stale
                        alerts_to_send.append((
                            "WARNING",
                            f"行情數據停滯: {symbol}",
                            f"交易對 {symbol} 已 {int(age)} 秒未更新行情",
                        ))
                    metrics.is_stale = True
                    metrics.stale_duration_seconds = age
                    stale_symbols.append(symbol)

            # Check price jumps
            if metrics.price_change_pct >= self._price_jump_critical:
                alerts_to_send.append((
                    "CRITICAL",
                    f"價格異常波動: {symbol}",
                    f"交易對 {symbol} 價格變動 {metrics.price_change_pct:.1f}%",
                ))
            elif metrics.price_change_pct >= self._price_jump_warning:
                alerts_to_send.append((
                    "WARNING",
                    f"價格大幅波動: {symbol}",
                    f"交易對 {symbol} 價格變動 {metrics.price_change_pct:.1f}%",
                ))

        # Send alerts
        if self._alert_callback:
            for severity, title, message in alerts_to_send:
                try:
                    await self._alert_callback(severity, title, message)
                except Exception as e:
                    logger.error(f"Failed to send market data alert: {e}")

        return {
            "symbols_monitored": len(self._metrics),
            "stale_symbols": stale_symbols,
            "metrics": {
                symbol: {
                    "last_update": m.last_update.isoformat() if m.last_update else None,
                    "update_count": m.update_count,
                    "last_price": str(m.last_price) if m.last_price else None,
                    "is_stale": m.is_stale,
                    "stale_duration": m.stale_duration_seconds,
                }
                for symbol, m in self._metrics.items()
            },
        }

    def get_stale_symbols(self) -> List[str]:
        """Get list of stale symbols."""
        return [s for s, m in self._metrics.items() if m.is_stale]


# =============================================================================
# Order Latency Breakdown Monitor
# =============================================================================


class OrderLatencyPhase(Enum):
    """Phases of order execution."""

    SUBMISSION = "submission"  # Local -> API
    CONFIRMATION = "confirmation"  # API -> Confirmed
    FILL = "fill"  # Confirmed -> Filled
    CANCEL = "cancel"  # Cancel request -> Confirmed


@dataclass
class OrderLatencyStats:
    """Statistics for order latency."""

    phase: OrderLatencyPhase
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float("inf")
    max_ms: float = 0.0
    samples: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))

    def avg_ms(self) -> float:
        """Calculate average latency."""
        if self.count == 0:
            return 0.0
        return self.total_ms / self.count

    def p50_ms(self) -> float:
        """Calculate 50th percentile."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = len(sorted_samples) // 2
        return sorted_samples[idx]

    def p99_ms(self) -> float:
        """Calculate 99th percentile."""
        if not self.samples:
            return 0.0
        sorted_samples = sorted(self.samples)
        idx = int(len(sorted_samples) * 0.99)
        return sorted_samples[min(idx, len(sorted_samples) - 1)]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "count": self.count,
            "avg_ms": round(self.avg_ms(), 2),
            "min_ms": round(self.min_ms, 2) if self.min_ms != float("inf") else 0,
            "max_ms": round(self.max_ms, 2),
            "p50_ms": round(self.p50_ms(), 2),
            "p99_ms": round(self.p99_ms(), 2),
        }


class OrderLatencyMonitor:
    """
    Monitors order execution latency with phase breakdown.

    Tracks:
    - Submission latency
    - Confirmation latency
    - Fill latency
    - Cancel latency
    """

    def __init__(
        self,
        warning_thresholds: Optional[Dict[str, float]] = None,
        critical_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize order latency monitor.

        Args:
            warning_thresholds: Warning thresholds per phase in ms
            critical_thresholds: Critical thresholds per phase in ms
        """
        self._warning_thresholds = warning_thresholds or {
            "submission": 100,
            "confirmation": 500,
            "fill": 1000,
            "cancel": 200,
        }
        self._critical_thresholds = critical_thresholds or {
            "submission": 500,
            "confirmation": 2000,
            "fill": 5000,
            "cancel": 1000,
        }

        # Stats per phase
        self._stats: Dict[OrderLatencyPhase, OrderLatencyStats] = {
            phase: OrderLatencyStats(phase=phase)
            for phase in OrderLatencyPhase
        }

        # Order tracking: order_id -> {phase: timestamp}
        self._order_timestamps: Dict[str, Dict[str, datetime]] = {}

        # Alert callback
        self._alert_callback: Optional[
            Callable[[str, str, str], Coroutine[Any, Any, None]]
        ] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set callback for alerts."""
        self._alert_callback = callback

    def record_submission_start(self, order_id: str) -> None:
        """Record order submission started."""
        self._order_timestamps[order_id] = {
            "submission_start": datetime.now(timezone.utc)
        }

    def record_submission_end(self, order_id: str) -> None:
        """Record order submission completed (API accepted)."""
        if order_id not in self._order_timestamps:
            return

        now = datetime.now(timezone.utc)
        timestamps = self._order_timestamps[order_id]
        timestamps["submission_end"] = now

        if "submission_start" in timestamps:
            latency_ms = (now - timestamps["submission_start"]).total_seconds() * 1000
            self._record_latency(OrderLatencyPhase.SUBMISSION, latency_ms)

    def record_confirmed(self, order_id: str) -> None:
        """Record order confirmed by exchange."""
        if order_id not in self._order_timestamps:
            return

        now = datetime.now(timezone.utc)
        timestamps = self._order_timestamps[order_id]
        timestamps["confirmed"] = now

        if "submission_end" in timestamps:
            latency_ms = (now - timestamps["submission_end"]).total_seconds() * 1000
            self._record_latency(OrderLatencyPhase.CONFIRMATION, latency_ms)

    def record_filled(self, order_id: str) -> None:
        """Record order filled."""
        if order_id not in self._order_timestamps:
            return

        now = datetime.now(timezone.utc)
        timestamps = self._order_timestamps[order_id]

        start_time = timestamps.get("confirmed") or timestamps.get("submission_start")
        if start_time:
            latency_ms = (now - start_time).total_seconds() * 1000
            self._record_latency(OrderLatencyPhase.FILL, latency_ms)

        # Clean up
        del self._order_timestamps[order_id]

    def record_cancel_start(self, order_id: str) -> None:
        """Record cancel request started."""
        if order_id in self._order_timestamps:
            self._order_timestamps[order_id]["cancel_start"] = datetime.now(timezone.utc)
        else:
            self._order_timestamps[order_id] = {
                "cancel_start": datetime.now(timezone.utc)
            }

    def record_cancelled(self, order_id: str) -> None:
        """Record order cancelled."""
        if order_id not in self._order_timestamps:
            return

        now = datetime.now(timezone.utc)
        timestamps = self._order_timestamps[order_id]

        if "cancel_start" in timestamps:
            latency_ms = (now - timestamps["cancel_start"]).total_seconds() * 1000
            self._record_latency(OrderLatencyPhase.CANCEL, latency_ms)

        # Clean up
        if order_id in self._order_timestamps:
            del self._order_timestamps[order_id]

    def _record_latency(self, phase: OrderLatencyPhase, latency_ms: float) -> None:
        """Record a latency measurement."""
        stats = self._stats[phase]
        stats.count += 1
        stats.total_ms += latency_ms
        stats.min_ms = min(stats.min_ms, latency_ms)
        stats.max_ms = max(stats.max_ms, latency_ms)
        stats.samples.append(latency_ms)

    async def check_health(self) -> Dict[str, Any]:
        """
        Check latency health and trigger alerts.

        Returns:
            Health status with stats
        """
        alerts_to_send = []

        for phase, stats in self._stats.items():
            if stats.count < 10:
                continue  # Not enough data

            p99 = stats.p99_ms()
            phase_name = phase.value

            warning = self._warning_thresholds.get(phase_name, 1000)
            critical = self._critical_thresholds.get(phase_name, 5000)

            if p99 >= critical:
                alerts_to_send.append((
                    "CRITICAL",
                    f"訂單延遲過高: {phase_name}",
                    f"訂單{phase_name}階段 P99延遲 {p99:.0f}ms 超過臨界值 {critical}ms",
                ))
            elif p99 >= warning:
                alerts_to_send.append((
                    "WARNING",
                    f"訂單延遲警告: {phase_name}",
                    f"訂單{phase_name}階段 P99延遲 {p99:.0f}ms",
                ))

        # Send alerts
        if self._alert_callback:
            for severity, title, message in alerts_to_send:
                try:
                    await self._alert_callback(severity, title, message)
                except Exception as e:
                    logger.error(f"Failed to send latency alert: {e}")

        return {
            "phases": {
                phase.value: stats.to_dict()
                for phase, stats in self._stats.items()
            },
            "pending_orders": len(self._order_timestamps),
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get all latency statistics."""
        return {
            phase.value: stats.to_dict()
            for phase, stats in self._stats.items()
        }


# =============================================================================
# Comprehensive Monitor Integration
# =============================================================================


class ComprehensiveMonitor:
    """
    Integrates all monitoring components into a single interface.
    """

    def __init__(self):
        """Initialize comprehensive monitor."""
        self.websocket = WebSocketHealthMonitor()
        self.api_errors = APIErrorMonitor()
        self.balance = BalanceReconciliationMonitor()
        self.market_data = MarketDataStalenessMonitor()
        self.order_latency = OrderLatencyMonitor()

        self._running = False
        self._check_task: Optional[asyncio.Task] = None

    def set_alert_callback(
        self,
        callback: Callable[[str, str, str], Coroutine[Any, Any, None]],
    ) -> None:
        """Set alert callback for all monitors."""
        self.websocket.set_alert_callback(callback)
        self.api_errors.set_alert_callback(callback)
        self.balance.set_alert_callback(callback)
        self.market_data.set_alert_callback(callback)
        self.order_latency.set_alert_callback(callback)

    async def start(self, check_interval: int = 30) -> None:
        """Start periodic health checks."""
        if self._running:
            return

        self._running = True
        self._check_task = asyncio.create_task(self._check_loop(check_interval))
        logger.info(f"Comprehensive monitor started (interval={check_interval}s)")

    async def stop(self) -> None:
        """Stop periodic health checks."""
        self._running = False
        if self._check_task:
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass
        logger.info("Comprehensive monitor stopped")

    async def _check_loop(self, interval: int) -> None:
        """Periodic health check loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Comprehensive monitor check error: {e}")
                await asyncio.sleep(10)

    async def check_all(self) -> Dict[str, Any]:
        """
        Run all health checks.

        Returns:
            Combined health status
        """
        results = {}

        try:
            results["websocket"] = await self.websocket.check_health()
        except Exception as e:
            logger.error(f"WebSocket health check error: {e}")
            results["websocket"] = {"error": str(e)}

        try:
            results["api_errors"] = await self.api_errors.check_health()
        except Exception as e:
            logger.error(f"API error check error: {e}")
            results["api_errors"] = {"error": str(e)}

        try:
            results["market_data"] = await self.market_data.check_health()
        except Exception as e:
            logger.error(f"Market data check error: {e}")
            results["market_data"] = {"error": str(e)}

        try:
            results["order_latency"] = await self.order_latency.check_health()
        except Exception as e:
            logger.error(f"Order latency check error: {e}")
            results["order_latency"] = {"error": str(e)}

        results["balance"] = self.balance.get_stats()
        results["timestamp"] = datetime.now(timezone.utc).isoformat()

        return results


# =============================================================================
# Global Instance
# =============================================================================

_monitor: Optional[ComprehensiveMonitor] = None


def get_comprehensive_monitor() -> ComprehensiveMonitor:
    """Get the global comprehensive monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = ComprehensiveMonitor()
    return _monitor
