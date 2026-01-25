"""
Anomaly Detection.

Provides anomaly detection for market data to identify unusual patterns.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
from collections import deque
import statistics

from src.core import get_logger

logger = get_logger(__name__)


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    PRICE_SPIKE = "price_spike"           # Sudden large price movement
    PRICE_DROP = "price_drop"             # Sudden large price drop
    VOLUME_SPIKE = "volume_spike"         # Abnormal volume increase
    VOLUME_DRY = "volume_dry"             # Abnormally low volume
    SPREAD_WIDE = "spread_wide"           # Abnormally wide bid-ask spread
    DATA_GAP = "data_gap"                 # Missing data
    DATA_STALE = "data_stale"             # No updates for extended period
    FLASH_CRASH = "flash_crash"           # Rapid price drop and recovery
    MANIPULATION = "manipulation"          # Suspected manipulation pattern
    OUTLIER = "outlier"                   # Statistical outlier
    GHOST_LIQUIDITY = "ghost_liquidity"   # Large order appeared and disappeared quickly
    SPOOFING = "spoofing"                 # Repeated ghost orders (suspected spoofing)


class AnomalySeverity(str, Enum):
    """Severity of detected anomalies."""
    LOW = "low"           # Informational, no action needed
    MEDIUM = "medium"     # Worth monitoring
    HIGH = "high"         # Requires attention
    CRITICAL = "critical" # Immediate action required


@dataclass
class AnomalyRecord:
    """Record of a detected anomaly."""
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    symbol: str
    description: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "symbol": self.symbol,
            "description": self.description,
            "value": self.value,
            "threshold": self.threshold,
            "metadata": self.metadata,
        }


@dataclass
class AnomalyStats:
    """Statistics for anomaly detection."""
    total_detected: int = 0
    by_type: Dict[AnomalyType, int] = field(default_factory=dict)
    by_severity: Dict[AnomalySeverity, int] = field(default_factory=dict)
    last_anomaly: Optional[AnomalyRecord] = None

    def record(self, anomaly: AnomalyRecord) -> None:
        """Record an anomaly."""
        self.total_detected += 1
        self.by_type[anomaly.anomaly_type] = (
            self.by_type.get(anomaly.anomaly_type, 0) + 1
        )
        self.by_severity[anomaly.severity] = (
            self.by_severity.get(anomaly.severity, 0) + 1
        )
        self.last_anomaly = anomaly


class AnomalyDetector:
    """
    Detects anomalies in market data.

    Uses statistical methods and rule-based detection to identify
    unusual patterns in price, volume, and order book data.
    """

    def __init__(
        self,
        # Price thresholds
        price_spike_pct: float = 5.0,
        price_drop_pct: float = 5.0,
        flash_crash_pct: float = 10.0,
        flash_crash_window_seconds: int = 60,

        # Volume thresholds
        volume_spike_std: float = 3.0,
        volume_dry_threshold_pct: float = 10.0,

        # Spread thresholds
        spread_warning_pct: float = 1.0,
        spread_critical_pct: float = 3.0,

        # Data freshness
        stale_threshold_seconds: int = 60,
        gap_threshold_multiplier: float = 2.0,

        # Statistical settings
        lookback_window: int = 100,
        z_score_threshold: float = 3.0,
        min_history_for_zscore: int = 20,

        # Cold-start fallback settings
        cold_start_price_change_pct: float = 10.0,  # Fallback when history < min

        # Callbacks
        on_anomaly: Optional[Callable[[AnomalyRecord], None]] = None,
    ):
        """
        Initialize anomaly detector.

        Args:
            price_spike_pct: % change to trigger price spike alert
            price_drop_pct: % change to trigger price drop alert
            flash_crash_pct: % change for flash crash detection
            flash_crash_window_seconds: Time window for flash crash
            volume_spike_std: Std deviations for volume spike
            volume_dry_threshold_pct: % of avg volume for dry detection
            spread_warning_pct: Spread % for warning
            spread_critical_pct: Spread % for critical
            stale_threshold_seconds: Seconds for stale data
            gap_threshold_multiplier: Multiplier for gap detection
            lookback_window: Window for statistical calculations
            z_score_threshold: Z-score for outlier detection
            min_history_for_zscore: Minimum data points required for z-score
            cold_start_price_change_pct: Fallback threshold when history insufficient
            on_anomaly: Callback function when anomaly detected
        """
        self.price_spike_pct = price_spike_pct
        self.price_drop_pct = price_drop_pct
        self.flash_crash_pct = flash_crash_pct
        self.flash_crash_window = timedelta(seconds=flash_crash_window_seconds)
        self.volume_spike_std = volume_spike_std
        self.volume_dry_threshold_pct = volume_dry_threshold_pct
        self.spread_warning_pct = spread_warning_pct
        self.spread_critical_pct = spread_critical_pct
        self.stale_threshold = timedelta(seconds=stale_threshold_seconds)
        self.gap_threshold_multiplier = gap_threshold_multiplier
        self.lookback_window = lookback_window
        self.z_score_threshold = z_score_threshold
        self.min_history_for_zscore = min_history_for_zscore
        self.cold_start_price_change_pct = cold_start_price_change_pct
        self.on_anomaly = on_anomaly

        # State tracking per symbol
        self._price_history: Dict[str, deque] = {}
        self._volume_history: Dict[str, deque] = {}
        self._last_update: Dict[str, datetime] = {}
        self._recent_prices: Dict[str, deque] = {}  # For flash crash detection

        # Statistics
        self.stats = AnomalyStats()

    def check_price(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyRecord]:
        """
        Check price for anomalies.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyRecord] = []
        now = timestamp or datetime.now(timezone.utc)
        price_float = float(price)

        # Initialize history if needed
        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self.lookback_window)
            self._recent_prices[symbol] = deque(maxlen=60)  # Last 60 prices

        history = self._price_history[symbol]
        recent = self._recent_prices[symbol]

        # Check for stale data
        if symbol in self._last_update:
            time_since_update = now - self._last_update[symbol]
            if time_since_update > self.stale_threshold:
                anomalies.append(AnomalyRecord(
                    timestamp=now,
                    anomaly_type=AnomalyType.DATA_STALE,
                    severity=AnomalySeverity.MEDIUM,
                    symbol=symbol,
                    description=f"No updates for {time_since_update.total_seconds():.0f}s",
                    value=time_since_update.total_seconds(),
                    threshold=self.stale_threshold.total_seconds(),
                ))

        # Check for price spike/drop
        if history:
            last_price = history[-1]
            if last_price > 0:
                change_pct = (price_float - last_price) / last_price * 100

                if change_pct > self.price_spike_pct:
                    severity = (
                        AnomalySeverity.HIGH if change_pct > self.price_spike_pct * 2
                        else AnomalySeverity.MEDIUM
                    )
                    anomalies.append(AnomalyRecord(
                        timestamp=now,
                        anomaly_type=AnomalyType.PRICE_SPIKE,
                        severity=severity,
                        symbol=symbol,
                        description=f"Price spiked {change_pct:.2f}%",
                        value=change_pct,
                        threshold=self.price_spike_pct,
                        metadata={"from": last_price, "to": price_float},
                    ))

                elif change_pct < -self.price_drop_pct:
                    severity = (
                        AnomalySeverity.HIGH if change_pct < -self.price_drop_pct * 2
                        else AnomalySeverity.MEDIUM
                    )
                    anomalies.append(AnomalyRecord(
                        timestamp=now,
                        anomaly_type=AnomalyType.PRICE_DROP,
                        severity=severity,
                        symbol=symbol,
                        description=f"Price dropped {abs(change_pct):.2f}%",
                        value=change_pct,
                        threshold=-self.price_drop_pct,
                        metadata={"from": last_price, "to": price_float},
                    ))

        # Check for flash crash (rapid drop and recovery)
        if len(recent) >= 10:
            recent_list = list(recent)
            min_price = min(recent_list)
            max_price = max(recent_list)

            if max_price > 0:
                range_pct = (max_price - min_price) / max_price * 100
                if range_pct > self.flash_crash_pct:
                    # Check if price recovered (current near max)
                    recovery_pct = (price_float - min_price) / (max_price - min_price) * 100
                    if recovery_pct > 50:  # More than 50% recovery
                        anomalies.append(AnomalyRecord(
                            timestamp=now,
                            anomaly_type=AnomalyType.FLASH_CRASH,
                            severity=AnomalySeverity.CRITICAL,
                            symbol=symbol,
                            description=f"Flash crash detected: {range_pct:.2f}% swing",
                            value=range_pct,
                            threshold=self.flash_crash_pct,
                            metadata={
                                "min": min_price,
                                "max": max_price,
                                "recovery_pct": recovery_pct,
                            },
                        ))

        # Statistical outlier detection with cold-start handling
        if len(history) >= self.min_history_for_zscore:
            # Normal z-score detection with sufficient history
            try:
                mean = statistics.mean(history)
                std = statistics.stdev(history)
                if std > 0:
                    z_score = (price_float - mean) / std
                    if abs(z_score) > self.z_score_threshold:
                        anomalies.append(AnomalyRecord(
                            timestamp=now,
                            anomaly_type=AnomalyType.OUTLIER,
                            severity=AnomalySeverity.MEDIUM,
                            symbol=symbol,
                            description=f"Statistical outlier: z-score={z_score:.2f}",
                            value=z_score,
                            threshold=self.z_score_threshold,
                            metadata={"mean": mean, "std": std, "history_size": len(history)},
                        ))
            except statistics.StatisticsError:
                pass
        elif len(history) >= 2:
            # Cold-start fallback: use fixed percentage threshold when history < min
            # This ensures we don't miss extreme anomalies during startup
            recent_prices = list(history)[-5:] if len(history) >= 5 else list(history)
            avg_recent = sum(recent_prices) / len(recent_prices)
            if avg_recent > 0:
                change_from_avg = abs((price_float - avg_recent) / avg_recent * 100)
                if change_from_avg > self.cold_start_price_change_pct:
                    anomalies.append(AnomalyRecord(
                        timestamp=now,
                        anomaly_type=AnomalyType.OUTLIER,
                        severity=AnomalySeverity.MEDIUM,
                        symbol=symbol,
                        description=(
                            f"Cold-start outlier: {change_from_avg:.2f}% deviation "
                            f"(history: {len(history)} points)"
                        ),
                        value=change_from_avg,
                        threshold=self.cold_start_price_change_pct,
                        metadata={
                            "avg_recent": avg_recent,
                            "history_size": len(history),
                            "cold_start": True,
                        },
                    ))

        # Update history
        history.append(price_float)
        recent.append(price_float)
        self._last_update[symbol] = now

        # Record anomalies
        for anomaly in anomalies:
            self.stats.record(anomaly)
            if self.on_anomaly:
                self.on_anomaly(anomaly)
            logger.warning(f"Anomaly detected: {anomaly.description}")

        return anomalies

    def check_volume(
        self,
        symbol: str,
        volume: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyRecord]:
        """
        Check volume for anomalies.

        Args:
            symbol: Trading symbol
            volume: Current volume
            timestamp: Volume timestamp

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyRecord] = []
        now = timestamp or datetime.now(timezone.utc)
        volume_float = float(volume)

        # Initialize history if needed
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self.lookback_window)

        history = self._volume_history[symbol]

        if len(history) >= 20:
            try:
                mean = statistics.mean(history)
                std = statistics.stdev(history)

                # Volume spike detection
                if std > 0:
                    z_score = (volume_float - mean) / std
                    if z_score > self.volume_spike_std:
                        anomalies.append(AnomalyRecord(
                            timestamp=now,
                            anomaly_type=AnomalyType.VOLUME_SPIKE,
                            severity=AnomalySeverity.MEDIUM,
                            symbol=symbol,
                            description=f"Volume spike: {z_score:.2f} std above mean",
                            value=z_score,
                            threshold=self.volume_spike_std,
                            metadata={"volume": volume_float, "mean": mean},
                        ))

                # Volume dry detection
                if mean > 0:
                    volume_pct = volume_float / mean * 100
                    if volume_pct < self.volume_dry_threshold_pct:
                        anomalies.append(AnomalyRecord(
                            timestamp=now,
                            anomaly_type=AnomalyType.VOLUME_DRY,
                            severity=AnomalySeverity.LOW,
                            symbol=symbol,
                            description=f"Low volume: {volume_pct:.1f}% of average",
                            value=volume_pct,
                            threshold=self.volume_dry_threshold_pct,
                            metadata={"volume": volume_float, "mean": mean},
                        ))
            except statistics.StatisticsError:
                pass

        # Update history
        history.append(volume_float)

        # Record anomalies
        for anomaly in anomalies:
            self.stats.record(anomaly)
            if self.on_anomaly:
                self.on_anomaly(anomaly)

        return anomalies

    def check_spread(
        self,
        symbol: str,
        bid: Decimal,
        ask: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyRecord]:
        """
        Check bid-ask spread for anomalies.

        Args:
            symbol: Trading symbol
            bid: Best bid price
            ask: Best ask price
            timestamp: Timestamp

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyRecord] = []
        now = timestamp or datetime.now(timezone.utc)

        if bid <= Decimal("0") or ask <= Decimal("0"):
            return anomalies

        spread_pct = float((ask - bid) / bid * Decimal("100"))

        if spread_pct > self.spread_critical_pct:
            anomalies.append(AnomalyRecord(
                timestamp=now,
                anomaly_type=AnomalyType.SPREAD_WIDE,
                severity=AnomalySeverity.HIGH,
                symbol=symbol,
                description=f"Critical spread: {spread_pct:.2f}%",
                value=spread_pct,
                threshold=self.spread_critical_pct,
                metadata={"bid": float(bid), "ask": float(ask)},
            ))
        elif spread_pct > self.spread_warning_pct:
            anomalies.append(AnomalyRecord(
                timestamp=now,
                anomaly_type=AnomalyType.SPREAD_WIDE,
                severity=AnomalySeverity.MEDIUM,
                symbol=symbol,
                description=f"Wide spread: {spread_pct:.2f}%",
                value=spread_pct,
                threshold=self.spread_warning_pct,
                metadata={"bid": float(bid), "ask": float(ask)},
            ))

        # Record anomalies
        for anomaly in anomalies:
            self.stats.record(anomaly)
            if self.on_anomaly:
                self.on_anomaly(anomaly)

        return anomalies

    def check_data_gap(
        self,
        symbol: str,
        current_time: datetime,
        last_time: datetime,
        expected_interval: timedelta,
    ) -> List[AnomalyRecord]:
        """
        Check for data gaps.

        Args:
            symbol: Trading symbol
            current_time: Current data timestamp
            last_time: Previous data timestamp
            expected_interval: Expected time between data points

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyRecord] = []

        actual_interval = current_time - last_time
        threshold = expected_interval * self.gap_threshold_multiplier

        if actual_interval > threshold:
            gap_seconds = actual_interval.total_seconds()
            expected_seconds = expected_interval.total_seconds()

            severity = (
                AnomalySeverity.HIGH if gap_seconds > expected_seconds * 5
                else AnomalySeverity.MEDIUM
            )

            anomalies.append(AnomalyRecord(
                timestamp=current_time,
                anomaly_type=AnomalyType.DATA_GAP,
                severity=severity,
                symbol=symbol,
                description=f"Data gap: {gap_seconds:.0f}s (expected {expected_seconds:.0f}s)",
                value=gap_seconds,
                threshold=threshold.total_seconds(),
                metadata={
                    "last_time": last_time.isoformat(),
                    "current_time": current_time.isoformat(),
                },
            ))

        # Record anomalies
        for anomaly in anomalies:
            self.stats.record(anomaly)
            if self.on_anomaly:
                self.on_anomaly(anomaly)

        return anomalies

    def get_stats(self) -> Dict[str, Any]:
        """Get anomaly detection statistics."""
        return {
            "total_detected": self.stats.total_detected,
            "by_type": {k.value: v for k, v in self.stats.by_type.items()},
            "by_severity": {k.value: v for k, v in self.stats.by_severity.items()},
            "last_anomaly": (
                self.stats.last_anomaly.to_dict()
                if self.stats.last_anomaly else None
            ),
        }

    def reset_stats(self) -> None:
        """Reset statistics."""
        self.stats = AnomalyStats()

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """
        Clear price/volume history.

        Args:
            symbol: Symbol to clear, or None for all
        """
        if symbol:
            self._price_history.pop(symbol, None)
            self._volume_history.pop(symbol, None)
            self._last_update.pop(symbol, None)
            self._recent_prices.pop(symbol, None)
        else:
            self._price_history.clear()
            self._volume_history.clear()
            self._last_update.clear()
            self._recent_prices.clear()


@dataclass
class OrderBookSnapshot:
    """Snapshot of an order book level."""
    price: float
    quantity: float
    timestamp: datetime
    side: str  # "bid" or "ask"


@dataclass
class GhostOrderRecord:
    """Record of a potential ghost order."""
    price: float
    quantity: float
    first_seen: datetime
    last_seen: datetime
    side: str
    times_seen: int = 1


class OrderBookAnomalyDetector:
    """
    Detects order book anomalies including ghost liquidity and spoofing.

    Ghost liquidity: Large orders that appear briefly and disappear,
    potentially indicating market manipulation or spoofing.

    Key features:
    - Tracks large orders across order book updates
    - Detects orders that appear and vanish quickly
    - Identifies patterns of repeated ghost orders (spoofing)
    """

    def __init__(
        self,
        # Ghost order detection thresholds
        large_order_pct: float = 5.0,       # % of total book to be "large"
        min_ghost_quantity_usd: float = 10000,  # Minimum USD value for ghost detection
        ghost_lifetime_seconds: float = 5.0,    # Max seconds for order to be "ghost"
        ghost_lookback_seconds: float = 60.0,   # Window for tracking orders

        # Spoofing detection
        spoofing_count_threshold: int = 3,  # Ghost orders in window to flag spoofing
        spoofing_window_seconds: float = 300.0,  # Window for spoofing detection

        # Callbacks
        on_anomaly: Optional[Callable[[AnomalyRecord], None]] = None,
    ):
        """
        Initialize order book anomaly detector.

        Args:
            large_order_pct: Percentage of book to consider order "large"
            min_ghost_quantity_usd: Minimum USD value for ghost detection
            ghost_lifetime_seconds: Maximum lifetime for ghost order classification
            ghost_lookback_seconds: Window for tracking order appearances
            spoofing_count_threshold: Count for spoofing detection
            spoofing_window_seconds: Time window for spoofing pattern
            on_anomaly: Callback for detected anomalies
        """
        self.large_order_pct = large_order_pct
        self.min_ghost_quantity_usd = min_ghost_quantity_usd
        self.ghost_lifetime = timedelta(seconds=ghost_lifetime_seconds)
        self.ghost_lookback = timedelta(seconds=ghost_lookback_seconds)
        self.spoofing_count_threshold = spoofing_count_threshold
        self.spoofing_window = timedelta(seconds=spoofing_window_seconds)
        self.on_anomaly = on_anomaly

        # State tracking per symbol
        # {symbol: {price: GhostOrderRecord}}
        self._tracked_orders: Dict[str, Dict[float, GhostOrderRecord]] = {}
        # {symbol: [GhostOrderRecord]} - detected ghost orders
        self._ghost_history: Dict[str, deque] = {}
        # Previous order book snapshot
        self._prev_books: Dict[str, Dict[str, List[Tuple[float, float]]]] = {}

        # Statistics
        self.stats = AnomalyStats()

    def check_orderbook(
        self,
        symbol: str,
        bids: List[Tuple[float, float]],  # (price, quantity)
        asks: List[Tuple[float, float]],
        mid_price: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> List[AnomalyRecord]:
        """
        Check order book for ghost liquidity and spoofing patterns.

        Args:
            symbol: Trading symbol
            bids: List of (price, quantity) tuples, descending by price
            asks: List of (price, quantity) tuples, ascending by price
            mid_price: Current mid price (for USD value calculation)
            timestamp: Current timestamp

        Returns:
            List of detected anomalies
        """
        anomalies: List[AnomalyRecord] = []
        now = timestamp or datetime.now(timezone.utc)

        # Initialize tracking for symbol
        if symbol not in self._tracked_orders:
            self._tracked_orders[symbol] = {}
        if symbol not in self._ghost_history:
            self._ghost_history[symbol] = deque(maxlen=100)

        # Calculate total book volume
        total_bid_qty = sum(qty for _, qty in bids) if bids else 0
        total_ask_qty = sum(qty for _, qty in asks) if asks else 0
        total_qty = total_bid_qty + total_ask_qty

        # Calculate threshold for "large" order
        large_threshold_qty = total_qty * (self.large_order_pct / 100) if total_qty > 0 else 0

        # Use mid price for USD calculation
        if mid_price is None and bids and asks:
            mid_price = (bids[0][0] + asks[0][0]) / 2

        # Build current large orders set
        current_large_orders: Dict[float, Tuple[float, str]] = {}  # price -> (qty, side)

        for price, qty in bids:
            usd_value = qty * mid_price if mid_price else 0
            if qty >= large_threshold_qty or usd_value >= self.min_ghost_quantity_usd:
                current_large_orders[price] = (qty, "bid")

        for price, qty in asks:
            usd_value = qty * mid_price if mid_price else 0
            if qty >= large_threshold_qty or usd_value >= self.min_ghost_quantity_usd:
                current_large_orders[price] = (qty, "ask")

        tracked = self._tracked_orders[symbol]

        # Check for disappeared orders (potential ghosts)
        prices_to_remove = []
        for price, record in tracked.items():
            if price not in current_large_orders:
                # Order disappeared
                lifetime = (now - record.first_seen).total_seconds()

                if lifetime <= self.ghost_lifetime.total_seconds():
                    # This is a ghost order!
                    usd_value = record.quantity * mid_price if mid_price else 0

                    ghost_record = GhostOrderRecord(
                        price=price,
                        quantity=record.quantity,
                        first_seen=record.first_seen,
                        last_seen=record.last_seen,
                        side=record.side,
                        times_seen=record.times_seen,
                    )
                    self._ghost_history[symbol].append(ghost_record)

                    anomalies.append(AnomalyRecord(
                        timestamp=now,
                        anomaly_type=AnomalyType.GHOST_LIQUIDITY,
                        severity=AnomalySeverity.HIGH,
                        symbol=symbol,
                        description=(
                            f"Ghost {record.side}: {record.quantity:.4f} @ {price:.2f} "
                            f"(${usd_value:,.0f}, lived {lifetime:.1f}s)"
                        ),
                        value=lifetime,
                        threshold=self.ghost_lifetime.total_seconds(),
                        metadata={
                            "price": price,
                            "quantity": record.quantity,
                            "usd_value": usd_value,
                            "side": record.side,
                            "lifetime_seconds": lifetime,
                            "times_seen": record.times_seen,
                        },
                    ))

                prices_to_remove.append(price)

        # Remove tracked orders that disappeared
        for price in prices_to_remove:
            del tracked[price]

        # Update tracked orders with current large orders
        for price, (qty, side) in current_large_orders.items():
            if price in tracked:
                # Update existing
                tracked[price].last_seen = now
                tracked[price].times_seen += 1
            else:
                # New large order
                tracked[price] = GhostOrderRecord(
                    price=price,
                    quantity=qty,
                    first_seen=now,
                    last_seen=now,
                    side=side,
                )

        # Clean up old tracked orders
        cutoff = now - self.ghost_lookback
        stale_prices = [
            p for p, r in tracked.items()
            if r.last_seen < cutoff
        ]
        for price in stale_prices:
            del tracked[price]

        # Check for spoofing pattern (multiple ghost orders in window)
        ghost_history = self._ghost_history[symbol]
        recent_ghosts = [
            g for g in ghost_history
            if (now - g.last_seen) <= self.spoofing_window
        ]

        if len(recent_ghosts) >= self.spoofing_count_threshold:
            # Potential spoofing detected
            total_ghost_value = sum(
                g.quantity * mid_price if mid_price else 0
                for g in recent_ghosts
            )
            bid_ghosts = sum(1 for g in recent_ghosts if g.side == "bid")
            ask_ghosts = sum(1 for g in recent_ghosts if g.side == "ask")

            anomalies.append(AnomalyRecord(
                timestamp=now,
                anomaly_type=AnomalyType.SPOOFING,
                severity=AnomalySeverity.CRITICAL,
                symbol=symbol,
                description=(
                    f"Spoofing pattern: {len(recent_ghosts)} ghost orders "
                    f"(${total_ghost_value:,.0f}) in {self.spoofing_window.total_seconds():.0f}s"
                ),
                value=len(recent_ghosts),
                threshold=self.spoofing_count_threshold,
                metadata={
                    "ghost_count": len(recent_ghosts),
                    "bid_ghosts": bid_ghosts,
                    "ask_ghosts": ask_ghosts,
                    "total_ghost_value_usd": total_ghost_value,
                    "window_seconds": self.spoofing_window.total_seconds(),
                },
            ))

        # Store current book for next comparison
        self._prev_books[symbol] = {"bids": bids, "asks": asks}

        # Record anomalies
        for anomaly in anomalies:
            self.stats.record(anomaly)
            if self.on_anomaly:
                self.on_anomaly(anomaly)
            logger.warning(f"OrderBook anomaly: {anomaly.description}")

        return anomalies

    def get_tracked_orders(self, symbol: str) -> Dict[float, GhostOrderRecord]:
        """Get currently tracked large orders for a symbol."""
        return self._tracked_orders.get(symbol, {})

    def get_ghost_history(self, symbol: str) -> List[GhostOrderRecord]:
        """Get recent ghost order history for a symbol."""
        return list(self._ghost_history.get(symbol, []))

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "total_detected": self.stats.total_detected,
            "by_type": {k.value: v for k, v in self.stats.by_type.items()},
            "by_severity": {k.value: v for k, v in self.stats.by_severity.items()},
            "symbols_tracked": list(self._tracked_orders.keys()),
        }

    def reset(self, symbol: Optional[str] = None) -> None:
        """
        Reset detection state.

        Args:
            symbol: Symbol to reset, or None for all
        """
        if symbol:
            self._tracked_orders.pop(symbol, None)
            self._ghost_history.pop(symbol, None)
            self._prev_books.pop(symbol, None)
        else:
            self._tracked_orders.clear()
            self._ghost_history.clear()
            self._prev_books.clear()
            self.stats = AnomalyStats()
