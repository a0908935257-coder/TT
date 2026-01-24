"""
Market Anomaly Detector for Risk Management.

Provides real-time market anomaly detection for risk management:
- Extreme volatility detection
- Liquidity monitoring (spread, depth)
- Flash crash detection
- Market manipulation patterns
- Circuit breaker integration

Part of the second line of defense (real-time risk monitoring).
Builds upon src/data/validation/anomaly.py for base detection.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Callable, Deque, Dict, List, Optional, Protocol, Set
import statistics

from src.core import get_logger

logger = get_logger(__name__)


class MarketCondition(Enum):
    """Overall market condition assessment."""

    NORMAL = "normal"  # Normal trading conditions
    VOLATILE = "volatile"  # Elevated volatility
    ILLIQUID = "illiquid"  # Reduced liquidity
    EXTREME = "extreme"  # Extreme conditions, consider halting
    CIRCUIT_BREAK = "circuit_break"  # Trading should stop


class AnomalyCategory(Enum):
    """Category of market anomaly."""

    VOLATILITY = "volatility"  # Price volatility anomalies
    LIQUIDITY = "liquidity"  # Liquidity/spread anomalies
    VOLUME = "volume"  # Volume anomalies
    DATA = "data"  # Data quality anomalies
    MANIPULATION = "manipulation"  # Potential manipulation


class RiskAction(Enum):
    """Recommended action based on anomaly."""

    NONE = "none"  # No action needed
    MONITOR = "monitor"  # Increased monitoring
    REDUCE_SIZE = "reduce_size"  # Reduce position/order sizes
    WIDEN_LIMITS = "widen_limits"  # Widen price deviation limits
    PAUSE_SYMBOL = "pause_symbol"  # Pause trading for symbol
    PAUSE_ALL = "pause_all"  # Pause all trading
    CIRCUIT_BREAK = "circuit_break"  # Trigger circuit breaker


@dataclass
class MarketAnomalyAlert:
    """Alert for market anomaly detection."""

    timestamp: datetime
    symbol: str
    category: AnomalyCategory
    condition: MarketCondition
    description: str
    value: Decimal
    threshold: Decimal
    recommended_action: RiskAction
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False
    triggered_circuit_breaker: bool = False

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True


@dataclass
class LiquidityMetrics:
    """Metrics for liquidity assessment."""

    symbol: str
    timestamp: datetime
    bid_price: Decimal
    ask_price: Decimal
    spread_pct: Decimal
    bid_depth: Decimal  # Total bid volume at best levels
    ask_depth: Decimal  # Total ask volume at best levels
    depth_ratio: Decimal  # bid_depth / ask_depth
    mid_price: Decimal

    @property
    def is_liquid(self) -> bool:
        """Check if market is liquid."""
        return self.spread_pct < Decimal("0.5") and self.bid_depth > 0 and self.ask_depth > 0


@dataclass
class VolatilityMetrics:
    """Metrics for volatility assessment."""

    symbol: str
    timestamp: datetime
    current_price: Decimal
    price_change_1m: Decimal  # 1-minute change %
    price_change_5m: Decimal  # 5-minute change %
    price_change_15m: Decimal  # 15-minute change %
    realized_volatility: Decimal  # Historical volatility
    atr_pct: Decimal  # ATR as percentage of price
    z_score: Decimal  # Current price z-score

    @property
    def is_extreme(self) -> bool:
        """Check if volatility is extreme."""
        return (
            abs(self.price_change_1m) > Decimal("3")
            or abs(self.price_change_5m) > Decimal("5")
            or abs(self.z_score) > Decimal("3")
        )


@dataclass
class MarketSnapshot:
    """Snapshot of market conditions."""

    timestamp: datetime
    symbols_monitored: int
    overall_condition: MarketCondition
    liquidity_metrics: Dict[str, LiquidityMetrics]
    volatility_metrics: Dict[str, VolatilityMetrics]
    active_alerts: List[MarketAnomalyAlert]
    symbols_paused: Set[str]


@dataclass
class MarketAnomalyConfig:
    """Configuration for market anomaly detection."""

    # Spread thresholds
    spread_warning_pct: Decimal = Decimal("0.5")  # 0.5% spread warning
    spread_danger_pct: Decimal = Decimal("1.0")  # 1.0% spread danger
    spread_critical_pct: Decimal = Decimal("2.0")  # 2.0% spread critical

    # Volatility thresholds (price change %)
    volatility_1m_warning: Decimal = Decimal("2.0")  # 2% in 1 min
    volatility_1m_danger: Decimal = Decimal("3.0")  # 3% in 1 min
    volatility_5m_warning: Decimal = Decimal("3.0")  # 3% in 5 min
    volatility_5m_danger: Decimal = Decimal("5.0")  # 5% in 5 min
    volatility_15m_warning: Decimal = Decimal("5.0")  # 5% in 15 min
    volatility_15m_danger: Decimal = Decimal("8.0")  # 8% in 15 min

    # Flash crash detection
    flash_crash_pct: Decimal = Decimal("10.0")  # 10% rapid move
    flash_crash_window_seconds: int = 60  # Within 60 seconds

    # Volume thresholds
    volume_spike_multiplier: Decimal = Decimal("5.0")  # 5x average
    volume_dry_pct: Decimal = Decimal("10.0")  # Below 10% of average

    # Depth thresholds
    depth_imbalance_warning: Decimal = Decimal("3.0")  # 3:1 ratio
    depth_imbalance_danger: Decimal = Decimal("5.0")  # 5:1 ratio
    min_depth_warning: Decimal = Decimal("1000")  # Minimum depth in quote
    min_depth_danger: Decimal = Decimal("500")  # Critical minimum depth

    # Data quality
    stale_data_warning_seconds: int = 30
    stale_data_danger_seconds: int = 60

    # Z-score thresholds
    z_score_warning: Decimal = Decimal("2.5")
    z_score_danger: Decimal = Decimal("3.5")

    # Circuit breaker triggers
    circuit_break_on_flash_crash: bool = True
    circuit_break_on_extreme_spread: bool = True
    circuit_break_on_data_loss: bool = True

    # Auto-pause thresholds
    auto_pause_after_alerts: int = 3  # Pause after N alerts for symbol
    alert_window_seconds: int = 300  # Within 5 minutes

    # Lookback for calculations
    lookback_periods: int = 100
    min_periods_for_stats: int = 20

    enabled: bool = True


class OrderBookProvider(Protocol):
    """Protocol for order book data access."""

    def get_best_bid(self, symbol: str) -> Optional[Decimal]:
        """Get best bid price."""
        ...

    def get_best_ask(self, symbol: str) -> Optional[Decimal]:
        """Get best ask price."""
        ...

    def get_bid_depth(self, symbol: str, levels: int = 5) -> Decimal:
        """Get total bid depth for N levels."""
        ...

    def get_ask_depth(self, symbol: str, levels: int = 5) -> Decimal:
        """Get total ask depth for N levels."""
        ...


class MarketDataProvider(Protocol):
    """Protocol for market data access."""

    def get_price(self, symbol: str) -> Optional[Decimal]:
        """Get current price."""
        ...

    def get_volume(self, symbol: str) -> Optional[Decimal]:
        """Get current volume."""
        ...


class MarketAnomalyDetector:
    """
    Market anomaly detector for real-time risk management.

    Monitors market conditions and detects anomalies that may require
    risk management actions such as pausing trading or triggering
    circuit breakers.

    Example:
        >>> config = MarketAnomalyConfig(
        ...     spread_warning_pct=Decimal("0.5"),
        ...     volatility_1m_danger=Decimal("3.0"),
        ... )
        >>> detector = MarketAnomalyDetector(config)
        >>> detector.update_price("BTCUSDT", Decimal("50000"))
        >>> alerts = detector.check_all("BTCUSDT")
        >>> condition = detector.get_market_condition("BTCUSDT")
    """

    def __init__(
        self,
        config: MarketAnomalyConfig,
        orderbook_provider: Optional[OrderBookProvider] = None,
        market_provider: Optional[MarketDataProvider] = None,
        on_alert: Optional[Callable[[MarketAnomalyAlert], None]] = None,
        on_circuit_break: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Initialize MarketAnomalyDetector.

        Args:
            config: Detection configuration
            orderbook_provider: Provider for order book data
            market_provider: Provider for market data
            on_alert: Callback when alert is generated
            on_circuit_break: Callback when circuit breaker triggered (symbol, reason)
        """
        self._config = config
        self._orderbook = orderbook_provider
        self._market = market_provider
        self._on_alert = on_alert
        self._on_circuit_break = on_circuit_break

        # Price history for volatility calculation
        self._price_history: Dict[str, Deque[tuple]] = {}  # (timestamp, price)
        self._volume_history: Dict[str, Deque[Decimal]] = {}

        # Alert tracking
        self._active_alerts: List[MarketAnomalyAlert] = []
        self._alert_history: List[MarketAnomalyAlert] = []
        self._symbol_alert_counts: Dict[str, Deque[datetime]] = {}

        # Paused symbols
        self._paused_symbols: Set[str] = set()

        # Market condition cache
        self._market_conditions: Dict[str, MarketCondition] = {}

        # Statistics
        self._total_checks: int = 0
        self._total_alerts: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> MarketAnomalyConfig:
        """Get configuration."""
        return self._config

    @property
    def active_alerts(self) -> List[MarketAnomalyAlert]:
        """Get active (unacknowledged) alerts."""
        return [a for a in self._active_alerts if not a.acknowledged]

    @property
    def paused_symbols(self) -> Set[str]:
        """Get set of paused symbols."""
        return self._paused_symbols.copy()

    # =========================================================================
    # Data Update Methods
    # =========================================================================

    def update_price(
        self,
        symbol: str,
        price: Decimal,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Update price data for a symbol.

        Args:
            symbol: Trading symbol
            price: Current price
            timestamp: Price timestamp
        """
        now = timestamp or datetime.now(timezone.utc)

        if symbol not in self._price_history:
            self._price_history[symbol] = deque(maxlen=self._config.lookback_periods)

        self._price_history[symbol].append((now, price))

    def update_volume(self, symbol: str, volume: Decimal) -> None:
        """
        Update volume data for a symbol.

        Args:
            symbol: Trading symbol
            volume: Current volume
        """
        if symbol not in self._volume_history:
            self._volume_history[symbol] = deque(maxlen=self._config.lookback_periods)

        self._volume_history[symbol].append(volume)

    # =========================================================================
    # Main Check Methods
    # =========================================================================

    def check_all(self, symbol: str) -> List[MarketAnomalyAlert]:
        """
        Perform all anomaly checks for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            List of generated alerts
        """
        if not self._config.enabled:
            return []

        self._total_checks += 1
        alerts: List[MarketAnomalyAlert] = []

        # Check volatility
        alerts.extend(self.check_volatility(symbol))

        # Check liquidity
        alerts.extend(self.check_liquidity(symbol))

        # Check volume
        alerts.extend(self.check_volume(symbol))

        # Check data freshness
        alerts.extend(self.check_data_freshness(symbol))

        # Update market condition
        self._update_market_condition(symbol, alerts)

        # Process alerts
        for alert in alerts:
            self._process_alert(alert)

        return alerts

    def check_volatility(self, symbol: str) -> List[MarketAnomalyAlert]:
        """
        Check for volatility anomalies.

        Args:
            symbol: Trading symbol

        Returns:
            List of volatility alerts
        """
        alerts: List[MarketAnomalyAlert] = []
        now = datetime.now(timezone.utc)

        history = self._price_history.get(symbol, deque())
        if len(history) < 2:
            return alerts

        current_price = history[-1][1]

        # Calculate price changes over different windows
        changes = self._calculate_price_changes(symbol)

        # Check 1-minute change
        if changes.get("1m") is not None:
            change_1m = abs(changes["1m"])
            if change_1m >= self._config.volatility_1m_danger:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.VOLATILITY,
                    condition=MarketCondition.EXTREME,
                    description=f"Extreme 1m volatility: {change_1m:.2f}%",
                    value=change_1m,
                    threshold=self._config.volatility_1m_danger,
                    action=RiskAction.PAUSE_SYMBOL,
                    metadata={"timeframe": "1m", "direction": "up" if changes["1m"] > 0 else "down"},
                ))
            elif change_1m >= self._config.volatility_1m_warning:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.VOLATILITY,
                    condition=MarketCondition.VOLATILE,
                    description=f"High 1m volatility: {change_1m:.2f}%",
                    value=change_1m,
                    threshold=self._config.volatility_1m_warning,
                    action=RiskAction.REDUCE_SIZE,
                ))

        # Check 5-minute change
        if changes.get("5m") is not None:
            change_5m = abs(changes["5m"])
            if change_5m >= self._config.volatility_5m_danger:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.VOLATILITY,
                    condition=MarketCondition.EXTREME,
                    description=f"Extreme 5m volatility: {change_5m:.2f}%",
                    value=change_5m,
                    threshold=self._config.volatility_5m_danger,
                    action=RiskAction.PAUSE_SYMBOL,
                ))
            elif change_5m >= self._config.volatility_5m_warning:
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.VOLATILITY,
                    condition=MarketCondition.VOLATILE,
                    description=f"High 5m volatility: {change_5m:.2f}%",
                    value=change_5m,
                    threshold=self._config.volatility_5m_warning,
                    action=RiskAction.MONITOR,
                ))

        # Check for flash crash
        flash_crash = self._detect_flash_crash(symbol)
        if flash_crash:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.VOLATILITY,
                condition=MarketCondition.CIRCUIT_BREAK,
                description=f"Flash crash detected: {flash_crash['range_pct']:.2f}% swing",
                value=Decimal(str(flash_crash["range_pct"])),
                threshold=self._config.flash_crash_pct,
                action=RiskAction.CIRCUIT_BREAK if self._config.circuit_break_on_flash_crash else RiskAction.PAUSE_ALL,
                metadata=flash_crash,
            ))

        # Check z-score
        z_score = self._calculate_z_score(symbol)
        if z_score is not None and abs(z_score) >= self._config.z_score_danger:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.VOLATILITY,
                condition=MarketCondition.EXTREME,
                description=f"Statistical outlier: z-score={z_score:.2f}",
                value=Decimal(str(abs(z_score))),
                threshold=self._config.z_score_danger,
                action=RiskAction.REDUCE_SIZE,
            ))

        return alerts

    def check_liquidity(self, symbol: str) -> List[MarketAnomalyAlert]:
        """
        Check for liquidity anomalies.

        Args:
            symbol: Trading symbol

        Returns:
            List of liquidity alerts
        """
        alerts: List[MarketAnomalyAlert] = []

        if self._orderbook is None:
            return alerts

        bid = self._orderbook.get_best_bid(symbol)
        ask = self._orderbook.get_best_ask(symbol)

        if bid is None or ask is None or bid <= 0:
            return alerts

        # Check spread
        spread_pct = (ask - bid) / bid * Decimal("100")

        if spread_pct >= self._config.spread_critical_pct:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.LIQUIDITY,
                condition=MarketCondition.CIRCUIT_BREAK if self._config.circuit_break_on_extreme_spread else MarketCondition.EXTREME,
                description=f"Critical spread: {spread_pct:.3f}%",
                value=spread_pct,
                threshold=self._config.spread_critical_pct,
                action=RiskAction.CIRCUIT_BREAK if self._config.circuit_break_on_extreme_spread else RiskAction.PAUSE_SYMBOL,
                metadata={"bid": str(bid), "ask": str(ask)},
            ))
        elif spread_pct >= self._config.spread_danger_pct:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.LIQUIDITY,
                condition=MarketCondition.ILLIQUID,
                description=f"Wide spread: {spread_pct:.3f}%",
                value=spread_pct,
                threshold=self._config.spread_danger_pct,
                action=RiskAction.WIDEN_LIMITS,
            ))
        elif spread_pct >= self._config.spread_warning_pct:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.LIQUIDITY,
                condition=MarketCondition.VOLATILE,
                description=f"Elevated spread: {spread_pct:.3f}%",
                value=spread_pct,
                threshold=self._config.spread_warning_pct,
                action=RiskAction.MONITOR,
            ))

        # Check depth
        bid_depth = self._orderbook.get_bid_depth(symbol)
        ask_depth = self._orderbook.get_ask_depth(symbol)

        if bid_depth > 0 and ask_depth > 0:
            depth_ratio = bid_depth / ask_depth if ask_depth > 0 else Decimal("0")
            inverse_ratio = ask_depth / bid_depth if bid_depth > 0 else Decimal("0")
            max_ratio = max(depth_ratio, inverse_ratio)

            if max_ratio >= self._config.depth_imbalance_danger:
                side = "bid" if depth_ratio > inverse_ratio else "ask"
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.LIQUIDITY,
                    condition=MarketCondition.ILLIQUID,
                    description=f"Severe depth imbalance: {max_ratio:.1f}:1 ({side} heavy)",
                    value=max_ratio,
                    threshold=self._config.depth_imbalance_danger,
                    action=RiskAction.REDUCE_SIZE,
                    metadata={"bid_depth": str(bid_depth), "ask_depth": str(ask_depth)},
                ))
            elif max_ratio >= self._config.depth_imbalance_warning:
                side = "bid" if depth_ratio > inverse_ratio else "ask"
                alerts.append(self._create_alert(
                    symbol=symbol,
                    category=AnomalyCategory.LIQUIDITY,
                    condition=MarketCondition.VOLATILE,
                    description=f"Depth imbalance: {max_ratio:.1f}:1 ({side} heavy)",
                    value=max_ratio,
                    threshold=self._config.depth_imbalance_warning,
                    action=RiskAction.MONITOR,
                ))

        # Check minimum depth
        min_depth = min(bid_depth, ask_depth)
        if min_depth < self._config.min_depth_danger:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.LIQUIDITY,
                condition=MarketCondition.ILLIQUID,
                description=f"Critical low depth: {min_depth:.2f}",
                value=min_depth,
                threshold=self._config.min_depth_danger,
                action=RiskAction.PAUSE_SYMBOL,
            ))
        elif min_depth < self._config.min_depth_warning:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.LIQUIDITY,
                condition=MarketCondition.ILLIQUID,
                description=f"Low depth: {min_depth:.2f}",
                value=min_depth,
                threshold=self._config.min_depth_warning,
                action=RiskAction.REDUCE_SIZE,
            ))

        return alerts

    def check_volume(self, symbol: str) -> List[MarketAnomalyAlert]:
        """
        Check for volume anomalies.

        Args:
            symbol: Trading symbol

        Returns:
            List of volume alerts
        """
        alerts: List[MarketAnomalyAlert] = []

        history = self._volume_history.get(symbol, deque())
        if len(history) < self._config.min_periods_for_stats:
            return alerts

        volumes = list(history)
        current = volumes[-1]
        mean_vol = Decimal(str(statistics.mean([float(v) for v in volumes[:-1]])))

        if mean_vol <= 0:
            return alerts

        ratio = current / mean_vol

        # Volume spike
        if ratio >= self._config.volume_spike_multiplier:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.VOLUME,
                condition=MarketCondition.VOLATILE,
                description=f"Volume spike: {ratio:.1f}x average",
                value=ratio,
                threshold=self._config.volume_spike_multiplier,
                action=RiskAction.MONITOR,
                metadata={"current": str(current), "average": str(mean_vol)},
            ))

        # Volume dry
        volume_pct = ratio * Decimal("100")
        if volume_pct < self._config.volume_dry_pct:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.VOLUME,
                condition=MarketCondition.ILLIQUID,
                description=f"Low volume: {volume_pct:.1f}% of average",
                value=volume_pct,
                threshold=self._config.volume_dry_pct,
                action=RiskAction.REDUCE_SIZE,
            ))

        return alerts

    def check_data_freshness(self, symbol: str) -> List[MarketAnomalyAlert]:
        """
        Check for stale data.

        Args:
            symbol: Trading symbol

        Returns:
            List of data quality alerts
        """
        alerts: List[MarketAnomalyAlert] = []
        now = datetime.now(timezone.utc)

        history = self._price_history.get(symbol, deque())
        if not history:
            return alerts

        last_update = history[-1][0]
        age_seconds = (now - last_update).total_seconds()

        if age_seconds >= self._config.stale_data_danger_seconds:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.DATA,
                condition=MarketCondition.CIRCUIT_BREAK if self._config.circuit_break_on_data_loss else MarketCondition.EXTREME,
                description=f"Stale data: {age_seconds:.0f}s since last update",
                value=Decimal(str(age_seconds)),
                threshold=Decimal(str(self._config.stale_data_danger_seconds)),
                action=RiskAction.CIRCUIT_BREAK if self._config.circuit_break_on_data_loss else RiskAction.PAUSE_SYMBOL,
            ))
        elif age_seconds >= self._config.stale_data_warning_seconds:
            alerts.append(self._create_alert(
                symbol=symbol,
                category=AnomalyCategory.DATA,
                condition=MarketCondition.VOLATILE,
                description=f"Data delay: {age_seconds:.0f}s since last update",
                value=Decimal(str(age_seconds)),
                threshold=Decimal(str(self._config.stale_data_warning_seconds)),
                action=RiskAction.MONITOR,
            ))

        return alerts

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_price_changes(self, symbol: str) -> Dict[str, Optional[Decimal]]:
        """Calculate price changes over various windows."""
        history = self._price_history.get(symbol, deque())
        if len(history) < 2:
            return {}

        now = history[-1][0]
        current_price = history[-1][1]
        changes = {}

        windows = [
            ("1m", timedelta(minutes=1)),
            ("5m", timedelta(minutes=5)),
            ("15m", timedelta(minutes=15)),
        ]

        for name, window in windows:
            cutoff = now - window
            # Find price at or before cutoff
            for ts, price in reversed(history):
                if ts <= cutoff:
                    if price > 0:
                        change_pct = (current_price - price) / price * Decimal("100")
                        changes[name] = change_pct
                    break
            else:
                changes[name] = None

        return changes

    def _detect_flash_crash(self, symbol: str) -> Optional[Dict]:
        """Detect flash crash pattern."""
        history = self._price_history.get(symbol, deque())
        if len(history) < 10:
            return None

        now = history[-1][0]
        window_start = now - timedelta(seconds=self._config.flash_crash_window_seconds)

        # Get prices within window
        window_prices = [
            float(price) for ts, price in history
            if ts >= window_start
        ]

        if len(window_prices) < 5:
            return None

        min_price = min(window_prices)
        max_price = max(window_prices)
        current = window_prices[-1]

        if max_price <= 0:
            return None

        range_pct = (max_price - min_price) / max_price * 100

        if range_pct >= float(self._config.flash_crash_pct):
            # Check if there was recovery
            recovery_pct = (current - min_price) / (max_price - min_price) * 100 if max_price > min_price else 0

            return {
                "range_pct": range_pct,
                "min_price": min_price,
                "max_price": max_price,
                "current_price": current,
                "recovery_pct": recovery_pct,
            }

        return None

    def _calculate_z_score(self, symbol: str) -> Optional[Decimal]:
        """Calculate z-score of current price."""
        history = self._price_history.get(symbol, deque())
        if len(history) < self._config.min_periods_for_stats:
            return None

        prices = [float(price) for _, price in history]
        current = prices[-1]

        try:
            mean = statistics.mean(prices[:-1])
            std = statistics.stdev(prices[:-1])
            if std > 0:
                return Decimal(str((current - mean) / std))
        except statistics.StatisticsError:
            pass

        return None

    def _create_alert(
        self,
        symbol: str,
        category: AnomalyCategory,
        condition: MarketCondition,
        description: str,
        value: Decimal,
        threshold: Decimal,
        action: RiskAction,
        metadata: Dict = None,
    ) -> MarketAnomalyAlert:
        """Create a market anomaly alert."""
        return MarketAnomalyAlert(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            category=category,
            condition=condition,
            description=description,
            value=value,
            threshold=threshold,
            recommended_action=action,
            metadata=metadata or {},
        )

    def _process_alert(self, alert: MarketAnomalyAlert) -> None:
        """Process a generated alert."""
        self._active_alerts.append(alert)
        self._alert_history.append(alert)
        self._total_alerts += 1

        # Track alerts per symbol
        if alert.symbol not in self._symbol_alert_counts:
            self._symbol_alert_counts[alert.symbol] = deque(maxlen=100)
        self._symbol_alert_counts[alert.symbol].append(alert.timestamp)

        # Check for auto-pause
        self._check_auto_pause(alert.symbol)

        # Trigger circuit breaker if needed
        if alert.recommended_action == RiskAction.CIRCUIT_BREAK:
            alert.triggered_circuit_breaker = True
            if self._on_circuit_break:
                self._on_circuit_break(alert.symbol, alert.description)
            logger.critical(f"CIRCUIT BREAKER: {alert.symbol} - {alert.description}")

        # Callback
        if self._on_alert:
            self._on_alert(alert)

        logger.warning(f"Market anomaly: {alert.symbol} - {alert.description}")

    def _check_auto_pause(self, symbol: str) -> None:
        """Check if symbol should be auto-paused."""
        if symbol in self._paused_symbols:
            return

        alert_times = self._symbol_alert_counts.get(symbol, deque())
        if not alert_times:
            return

        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self._config.alert_window_seconds)
        recent_alerts = sum(1 for t in alert_times if t > cutoff)

        if recent_alerts >= self._config.auto_pause_after_alerts:
            self.pause_symbol(symbol, f"Auto-paused after {recent_alerts} alerts")

    def _update_market_condition(
        self, symbol: str, alerts: List[MarketAnomalyAlert]
    ) -> None:
        """Update market condition based on alerts."""
        if not alerts:
            self._market_conditions[symbol] = MarketCondition.NORMAL
            return

        # Use worst condition from alerts
        conditions = [a.condition for a in alerts]
        condition_priority = {
            MarketCondition.CIRCUIT_BREAK: 5,
            MarketCondition.EXTREME: 4,
            MarketCondition.ILLIQUID: 3,
            MarketCondition.VOLATILE: 2,
            MarketCondition.NORMAL: 1,
        }

        worst = max(conditions, key=lambda c: condition_priority.get(c, 0))
        self._market_conditions[symbol] = worst

    # =========================================================================
    # Symbol Management
    # =========================================================================

    def pause_symbol(self, symbol: str, reason: str = "") -> None:
        """Pause trading for a symbol."""
        self._paused_symbols.add(symbol)
        logger.warning(f"Paused trading for {symbol}: {reason}")

    def resume_symbol(self, symbol: str) -> bool:
        """Resume trading for a symbol."""
        if symbol in self._paused_symbols:
            self._paused_symbols.discard(symbol)
            logger.info(f"Resumed trading for {symbol}")
            return True
        return False

    def is_symbol_paused(self, symbol: str) -> bool:
        """Check if symbol is paused."""
        return symbol in self._paused_symbols

    def get_market_condition(self, symbol: str) -> MarketCondition:
        """Get current market condition for symbol."""
        return self._market_conditions.get(symbol, MarketCondition.NORMAL)

    # =========================================================================
    # Metrics and Reporting
    # =========================================================================

    def get_liquidity_metrics(self, symbol: str) -> Optional[LiquidityMetrics]:
        """Get liquidity metrics for a symbol."""
        if self._orderbook is None:
            return None

        bid = self._orderbook.get_best_bid(symbol)
        ask = self._orderbook.get_best_ask(symbol)

        if bid is None or ask is None or bid <= 0:
            return None

        bid_depth = self._orderbook.get_bid_depth(symbol)
        ask_depth = self._orderbook.get_ask_depth(symbol)

        return LiquidityMetrics(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            bid_price=bid,
            ask_price=ask,
            spread_pct=(ask - bid) / bid * Decimal("100"),
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            depth_ratio=bid_depth / ask_depth if ask_depth > 0 else Decimal("0"),
            mid_price=(bid + ask) / Decimal("2"),
        )

    def get_volatility_metrics(self, symbol: str) -> Optional[VolatilityMetrics]:
        """Get volatility metrics for a symbol."""
        history = self._price_history.get(symbol, deque())
        if len(history) < 2:
            return None

        current_price = history[-1][1]
        changes = self._calculate_price_changes(symbol)
        z_score = self._calculate_z_score(symbol)

        # Calculate realized volatility
        prices = [float(price) for _, price in history]
        try:
            realized_vol = Decimal(str(statistics.stdev(prices) / statistics.mean(prices) * 100))
        except (statistics.StatisticsError, ZeroDivisionError):
            realized_vol = Decimal("0")

        return VolatilityMetrics(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            current_price=current_price,
            price_change_1m=changes.get("1m", Decimal("0")) or Decimal("0"),
            price_change_5m=changes.get("5m", Decimal("0")) or Decimal("0"),
            price_change_15m=changes.get("15m", Decimal("0")) or Decimal("0"),
            realized_volatility=realized_vol,
            atr_pct=Decimal("0"),  # Would need OHLC data
            z_score=z_score or Decimal("0"),
        )

    def get_snapshot(self) -> MarketSnapshot:
        """Get current market snapshot."""
        liquidity = {}
        volatility = {}

        for symbol in self._price_history.keys():
            liq = self.get_liquidity_metrics(symbol)
            if liq:
                liquidity[symbol] = liq

            vol = self.get_volatility_metrics(symbol)
            if vol:
                volatility[symbol] = vol

        # Determine overall condition
        conditions = list(self._market_conditions.values())
        if not conditions:
            overall = MarketCondition.NORMAL
        else:
            condition_priority = {
                MarketCondition.CIRCUIT_BREAK: 5,
                MarketCondition.EXTREME: 4,
                MarketCondition.ILLIQUID: 3,
                MarketCondition.VOLATILE: 2,
                MarketCondition.NORMAL: 1,
            }
            overall = max(conditions, key=lambda c: condition_priority.get(c, 0))

        return MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbols_monitored=len(self._price_history),
            overall_condition=overall,
            liquidity_metrics=liquidity,
            volatility_metrics=volatility,
            active_alerts=self.active_alerts,
            symbols_paused=self.paused_symbols,
        )

    def get_statistics(self) -> Dict:
        """Get detector statistics."""
        return {
            "total_checks": self._total_checks,
            "total_alerts": self._total_alerts,
            "active_alerts": len(self.active_alerts),
            "symbols_monitored": len(self._price_history),
            "symbols_paused": len(self._paused_symbols),
            "paused_list": list(self._paused_symbols),
            "enabled": self._config.enabled,
        }

    # =========================================================================
    # Alert Management
    # =========================================================================

    def acknowledge_alert(self, alert: MarketAnomalyAlert) -> None:
        """Acknowledge an alert."""
        alert.acknowledge()

    def acknowledge_all_alerts(self, symbol: Optional[str] = None) -> int:
        """Acknowledge all alerts (optionally for specific symbol)."""
        count = 0
        for alert in self._active_alerts:
            if not alert.acknowledged:
                if symbol is None or alert.symbol == symbol:
                    alert.acknowledge()
                    count += 1
        return count

    def clear_history(self, symbol: Optional[str] = None) -> None:
        """Clear price/volume history."""
        if symbol:
            self._price_history.pop(symbol, None)
            self._volume_history.pop(symbol, None)
            self._market_conditions.pop(symbol, None)
        else:
            self._price_history.clear()
            self._volume_history.clear()
            self._market_conditions.clear()

    # =========================================================================
    # Configuration
    # =========================================================================

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated market anomaly config: {key} = {value}")

    def enable(self) -> None:
        """Enable anomaly detection."""
        self._config.enabled = True
        logger.info("Market anomaly detection enabled")

    def disable(self) -> None:
        """Disable anomaly detection."""
        self._config.enabled = False
        logger.warning("Market anomaly detection disabled")
