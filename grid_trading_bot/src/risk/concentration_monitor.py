"""
Concentration Monitor.

Monitors portfolio concentration risk in real-time:
- Single position concentration
- Sector/category concentration
- Correlated assets exposure
- Leverage concentration
- Geographic/exchange concentration

Part of the second line of defense (real-time risk monitoring).
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Callable, Dict, List, Optional, Protocol, Set

from src.core import get_logger

logger = get_logger(__name__)


class ConcentrationLevel(Enum):
    """Concentration risk level."""

    NORMAL = 1  # Within acceptable limits
    ELEVATED = 2  # Approaching limits
    HIGH = 3  # At or exceeding soft limits
    CRITICAL = 4  # Exceeding hard limits


class ConcentrationAlertType(Enum):
    """Type of concentration alert."""

    SINGLE_POSITION = "single_position"  # Single asset too large
    SECTOR = "sector"  # Sector exposure too high
    CORRELATION = "correlation"  # Correlated assets too high
    LEVERAGE = "leverage"  # Leverage concentration
    EXCHANGE = "exchange"  # Exchange concentration
    DIRECTION = "direction"  # Long/short imbalance


@dataclass
class AssetCategory:
    """Asset category/sector definition."""

    name: str  # Category name (e.g., "DeFi", "Layer1", "Meme")
    symbols: Set[str] = field(default_factory=set)  # Symbols in this category
    max_exposure_pct: Decimal = Decimal("0.40")  # Max 40% in one category
    correlation_group: Optional[str] = None  # Group for correlation calc


@dataclass
class PositionInfo:
    """Position information for concentration analysis."""

    symbol: str
    quantity: Decimal
    market_value: Decimal  # Position value in quote currency
    unrealized_pnl: Decimal = Decimal("0")
    leverage: Decimal = Decimal("1")
    side: str = "LONG"  # LONG or SHORT
    category: Optional[str] = None
    exchange: str = "binance"


@dataclass
class ConcentrationAlert:
    """Alert for concentration risk."""

    alert_type: ConcentrationAlertType
    level: ConcentrationLevel
    message: str
    current_value: Decimal  # Current concentration %
    threshold: Decimal  # Threshold that was breached
    affected_symbols: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True


@dataclass
class ConcentrationSnapshot:
    """Snapshot of concentration metrics."""

    timestamp: datetime
    total_portfolio_value: Decimal

    # Position concentrations
    position_concentrations: Dict[str, Decimal]  # symbol -> % of portfolio
    largest_position_pct: Decimal
    largest_position_symbol: str

    # Category concentrations
    category_concentrations: Dict[str, Decimal]  # category -> % of portfolio

    # Direction concentrations
    long_exposure_pct: Decimal
    short_exposure_pct: Decimal
    net_exposure_pct: Decimal

    # Leverage metrics
    average_leverage: Decimal
    max_leverage: Decimal
    weighted_leverage: Decimal

    # Diversity metrics
    position_count: int
    herfindahl_index: Decimal  # Concentration index (0=diversified, 1=concentrated)
    effective_positions: Decimal  # 1/HHI - effective number of positions


@dataclass
class ConcentrationConfig:
    """Configuration for concentration monitoring."""

    # Single position limits
    single_position_warning_pct: Decimal = Decimal("0.20")  # 20% warning
    single_position_danger_pct: Decimal = Decimal("0.30")  # 30% danger
    single_position_critical_pct: Decimal = Decimal("0.40")  # 40% critical

    # Category/sector limits
    category_warning_pct: Decimal = Decimal("0.30")  # 30% warning
    category_danger_pct: Decimal = Decimal("0.50")  # 50% danger

    # Correlation group limits
    correlation_group_warning_pct: Decimal = Decimal("0.40")
    correlation_group_danger_pct: Decimal = Decimal("0.60")

    # Direction limits
    direction_imbalance_warning_pct: Decimal = Decimal("0.70")  # 70% one direction
    direction_imbalance_danger_pct: Decimal = Decimal("0.85")  # 85% one direction

    # Leverage limits
    max_average_leverage: Decimal = Decimal("5.0")
    max_single_leverage: Decimal = Decimal("20.0")
    max_weighted_leverage: Decimal = Decimal("10.0")

    # Diversity requirements
    min_positions_for_diversity: int = 3  # Min positions for diversification
    max_herfindahl_index: Decimal = Decimal("0.35")  # Max concentration index

    # Monitoring settings
    check_interval_seconds: int = 30  # How often to check
    enabled: bool = True

    # Asset categories
    categories: List[AssetCategory] = field(default_factory=list)

    # Correlation groups (symbols that tend to move together)
    correlation_groups: Dict[str, Set[str]] = field(default_factory=dict)


class PositionDataProvider(Protocol):
    """Protocol for position data access."""

    def get_all_positions(self) -> List[PositionInfo]:
        """Get all current positions."""
        ...

    def get_total_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        ...


class ConcentrationMonitor:
    """
    Real-time concentration risk monitor.

    Monitors portfolio concentration across multiple dimensions:
    - Single position size
    - Sector/category exposure
    - Correlated asset groups
    - Leverage distribution
    - Long/short balance

    Example:
        >>> config = ConcentrationConfig(
        ...     single_position_warning_pct=Decimal("0.20"),
        ...     categories=[
        ...         AssetCategory("DeFi", {"UNIUSDT", "AAVEUSDT", "COMPUSDT"}),
        ...         AssetCategory("Layer1", {"BTCUSDT", "ETHUSDT", "SOLUSDT"}),
        ...     ],
        ... )
        >>> monitor = ConcentrationMonitor(config, position_provider)
        >>> snapshot = monitor.analyze()
        >>> alerts = monitor.check_alerts()
    """

    def __init__(
        self,
        config: ConcentrationConfig,
        position_provider: Optional[PositionDataProvider] = None,
        on_alert: Optional[Callable[[ConcentrationAlert], None]] = None,
    ):
        """
        Initialize ConcentrationMonitor.

        Args:
            config: Concentration monitoring configuration
            position_provider: Provider for position data
            on_alert: Callback when alert is generated
        """
        self._config = config
        self._position_provider = position_provider
        self._on_alert = on_alert

        # State
        self._last_snapshot: Optional[ConcentrationSnapshot] = None
        self._active_alerts: List[ConcentrationAlert] = []
        self._alert_history: List[ConcentrationAlert] = []

        # Build category lookup
        self._symbol_to_category: Dict[str, str] = {}
        for category in config.categories:
            for symbol in category.symbols:
                self._symbol_to_category[symbol] = category.name

        # Statistics
        self._total_checks: int = 0
        self._total_alerts: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> ConcentrationConfig:
        """Get configuration."""
        return self._config

    @property
    def last_snapshot(self) -> Optional[ConcentrationSnapshot]:
        """Get last concentration snapshot."""
        return self._last_snapshot

    @property
    def active_alerts(self) -> List[ConcentrationAlert]:
        """Get active (unacknowledged) alerts."""
        return [a for a in self._active_alerts if not a.acknowledged]

    @property
    def current_level(self) -> ConcentrationLevel:
        """Get current overall concentration level."""
        if not self._active_alerts:
            return ConcentrationLevel.NORMAL

        max_level = max(a.level.value for a in self._active_alerts if not a.acknowledged)
        for level in ConcentrationLevel:
            if level.value == max_level:
                return level
        return ConcentrationLevel.NORMAL

    # =========================================================================
    # Core Analysis
    # =========================================================================

    def analyze(
        self, positions: Optional[List[PositionInfo]] = None
    ) -> ConcentrationSnapshot:
        """
        Analyze current portfolio concentration.

        Args:
            positions: Optional positions to analyze (uses provider if None)

        Returns:
            ConcentrationSnapshot with all metrics
        """
        # Get positions
        if positions is None:
            if self._position_provider is None:
                raise RuntimeError("No position data available")
            positions = self._position_provider.get_all_positions()
            total_value = self._position_provider.get_total_portfolio_value()
        else:
            total_value = sum(p.market_value for p in positions)

        if total_value <= 0:
            total_value = Decimal("1")  # Avoid division by zero

        # Calculate position concentrations
        position_concentrations: Dict[str, Decimal] = {}
        for pos in positions:
            pct = pos.market_value / total_value
            position_concentrations[pos.symbol] = pct

        # Find largest position
        if position_concentrations:
            largest_symbol = max(position_concentrations, key=position_concentrations.get)
            largest_pct = position_concentrations[largest_symbol]
        else:
            largest_symbol = ""
            largest_pct = Decimal("0")

        # Calculate category concentrations
        category_values: Dict[str, Decimal] = {}
        for pos in positions:
            category = self._symbol_to_category.get(pos.symbol, "Other")
            category_values[category] = category_values.get(category, Decimal("0")) + pos.market_value

        category_concentrations = {
            cat: val / total_value for cat, val in category_values.items()
        }

        # Calculate direction exposure
        long_value = sum(p.market_value for p in positions if p.side == "LONG")
        short_value = sum(p.market_value for p in positions if p.side == "SHORT")
        total_exposure = long_value + short_value

        if total_exposure > 0:
            long_pct = long_value / total_exposure
            short_pct = short_value / total_exposure
        else:
            long_pct = Decimal("0")
            short_pct = Decimal("0")

        net_exposure_pct = (long_value - short_value) / total_value if total_value > 0 else Decimal("0")

        # Calculate leverage metrics
        leverages = [p.leverage for p in positions if p.leverage > 0]
        if leverages:
            avg_leverage = sum(leverages) / len(leverages)
            max_leverage = max(leverages)
            # Weighted by position size
            weighted_lev = sum(
                p.leverage * (p.market_value / total_value)
                for p in positions
            )
        else:
            avg_leverage = Decimal("1")
            max_leverage = Decimal("1")
            weighted_lev = Decimal("1")

        # Calculate Herfindahl Index (concentration measure)
        hhi = sum(pct ** 2 for pct in position_concentrations.values())
        effective_positions = Decimal("1") / hhi if hhi > 0 else Decimal(str(len(positions)))

        snapshot = ConcentrationSnapshot(
            timestamp=datetime.now(),
            total_portfolio_value=total_value,
            position_concentrations=position_concentrations,
            largest_position_pct=largest_pct,
            largest_position_symbol=largest_symbol,
            category_concentrations=category_concentrations,
            long_exposure_pct=long_pct,
            short_exposure_pct=short_pct,
            net_exposure_pct=net_exposure_pct,
            average_leverage=avg_leverage,
            max_leverage=max_leverage,
            weighted_leverage=weighted_lev,
            position_count=len(positions),
            herfindahl_index=hhi,
            effective_positions=effective_positions,
        )

        self._last_snapshot = snapshot
        self._total_checks += 1

        return snapshot

    def check_alerts(
        self, snapshot: Optional[ConcentrationSnapshot] = None
    ) -> List[ConcentrationAlert]:
        """
        Check for concentration alerts based on snapshot.

        Args:
            snapshot: Snapshot to check (uses last if None)

        Returns:
            List of new alerts generated
        """
        if snapshot is None:
            snapshot = self._last_snapshot
        if snapshot is None:
            return []

        new_alerts: List[ConcentrationAlert] = []

        # Check single position concentration
        new_alerts.extend(self._check_single_position(snapshot))

        # Check category concentration
        new_alerts.extend(self._check_category_concentration(snapshot))

        # Check correlation groups
        new_alerts.extend(self._check_correlation_groups(snapshot))

        # Check direction imbalance
        new_alerts.extend(self._check_direction_balance(snapshot))

        # Check leverage
        new_alerts.extend(self._check_leverage(snapshot))

        # Check diversity
        new_alerts.extend(self._check_diversity(snapshot))

        # Update active alerts
        for alert in new_alerts:
            self._active_alerts.append(alert)
            self._alert_history.append(alert)
            self._total_alerts += 1

            # Trigger callback
            if self._on_alert:
                try:
                    self._on_alert(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

            logger.warning(
                f"Concentration alert: {alert.alert_type.value} - {alert.message}"
            )

        return new_alerts

    # =========================================================================
    # Individual Checks
    # =========================================================================

    def _check_single_position(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check single position concentration."""
        alerts = []

        for symbol, pct in snapshot.position_concentrations.items():
            level = None
            threshold = None

            if pct >= self._config.single_position_critical_pct:
                level = ConcentrationLevel.CRITICAL
                threshold = self._config.single_position_critical_pct
            elif pct >= self._config.single_position_danger_pct:
                level = ConcentrationLevel.HIGH
                threshold = self._config.single_position_danger_pct
            elif pct >= self._config.single_position_warning_pct:
                level = ConcentrationLevel.ELEVATED
                threshold = self._config.single_position_warning_pct

            if level:
                # Check if we already have an active alert for this
                existing = self._find_existing_alert(
                    ConcentrationAlertType.SINGLE_POSITION, [symbol]
                )
                if existing and existing.level.value >= level.value:
                    continue

                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.SINGLE_POSITION,
                        level=level,
                        message=f"Position {symbol} is {pct:.1%} of portfolio",
                        current_value=pct,
                        threshold=threshold,
                        affected_symbols=[symbol],
                    )
                )

        return alerts

    def _check_category_concentration(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check category/sector concentration."""
        alerts = []

        for category, pct in snapshot.category_concentrations.items():
            if category == "Other":
                continue

            # Get category-specific limit if defined
            cat_config = next(
                (c for c in self._config.categories if c.name == category), None
            )
            if cat_config:
                warning_pct = cat_config.max_exposure_pct * Decimal("0.8")
                danger_pct = cat_config.max_exposure_pct
            else:
                warning_pct = self._config.category_warning_pct
                danger_pct = self._config.category_danger_pct

            level = None
            threshold = None

            if pct >= danger_pct:
                level = ConcentrationLevel.HIGH
                threshold = danger_pct
            elif pct >= warning_pct:
                level = ConcentrationLevel.ELEVATED
                threshold = warning_pct

            if level:
                # Get affected symbols
                affected = [
                    sym for sym, cat in self._symbol_to_category.items()
                    if cat == category and sym in snapshot.position_concentrations
                ]

                existing = self._find_existing_alert(
                    ConcentrationAlertType.SECTOR, affected
                )
                if existing and existing.level.value >= level.value:
                    continue

                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.SECTOR,
                        level=level,
                        message=f"Sector {category} exposure is {pct:.1%}",
                        current_value=pct,
                        threshold=threshold,
                        affected_symbols=affected,
                    )
                )

        return alerts

    def _check_correlation_groups(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check correlation group concentration."""
        alerts = []

        for group_name, symbols in self._config.correlation_groups.items():
            # Calculate total exposure in this group
            group_pct = sum(
                snapshot.position_concentrations.get(sym, Decimal("0"))
                for sym in symbols
            )

            level = None
            threshold = None

            if group_pct >= self._config.correlation_group_danger_pct:
                level = ConcentrationLevel.HIGH
                threshold = self._config.correlation_group_danger_pct
            elif group_pct >= self._config.correlation_group_warning_pct:
                level = ConcentrationLevel.ELEVATED
                threshold = self._config.correlation_group_warning_pct

            if level:
                affected = [
                    sym for sym in symbols
                    if sym in snapshot.position_concentrations
                ]

                existing = self._find_existing_alert(
                    ConcentrationAlertType.CORRELATION, affected
                )
                if existing and existing.level.value >= level.value:
                    continue

                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.CORRELATION,
                        level=level,
                        message=f"Correlated group '{group_name}' exposure is {group_pct:.1%}",
                        current_value=group_pct,
                        threshold=threshold,
                        affected_symbols=affected,
                    )
                )

        return alerts

    def _check_direction_balance(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check long/short balance."""
        alerts = []

        # Check for excessive long or short exposure
        dominant_pct = max(snapshot.long_exposure_pct, snapshot.short_exposure_pct)
        direction = "LONG" if snapshot.long_exposure_pct > snapshot.short_exposure_pct else "SHORT"

        level = None
        threshold = None

        if dominant_pct >= self._config.direction_imbalance_danger_pct:
            level = ConcentrationLevel.HIGH
            threshold = self._config.direction_imbalance_danger_pct
        elif dominant_pct >= self._config.direction_imbalance_warning_pct:
            level = ConcentrationLevel.ELEVATED
            threshold = self._config.direction_imbalance_warning_pct

        if level:
            existing = self._find_existing_alert(ConcentrationAlertType.DIRECTION, [])
            if not existing or existing.level.value < level.value:
                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.DIRECTION,
                        level=level,
                        message=f"Portfolio is {dominant_pct:.1%} {direction}",
                        current_value=dominant_pct,
                        threshold=threshold,
                        affected_symbols=[],
                    )
                )

        return alerts

    def _check_leverage(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check leverage concentration."""
        alerts = []

        # Check average leverage
        if snapshot.average_leverage > self._config.max_average_leverage:
            existing = self._find_existing_alert(ConcentrationAlertType.LEVERAGE, [])
            if not existing:
                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.LEVERAGE,
                        level=ConcentrationLevel.HIGH,
                        message=f"Average leverage {snapshot.average_leverage:.1f}x exceeds limit",
                        current_value=snapshot.average_leverage,
                        threshold=self._config.max_average_leverage,
                        affected_symbols=[],
                    )
                )

        # Check max leverage
        if snapshot.max_leverage > self._config.max_single_leverage:
            alerts.append(
                ConcentrationAlert(
                    alert_type=ConcentrationAlertType.LEVERAGE,
                    level=ConcentrationLevel.CRITICAL,
                    message=f"Max leverage {snapshot.max_leverage:.1f}x exceeds limit",
                    current_value=snapshot.max_leverage,
                    threshold=self._config.max_single_leverage,
                    affected_symbols=[],
                )
            )

        # Check weighted leverage
        if snapshot.weighted_leverage > self._config.max_weighted_leverage:
            alerts.append(
                ConcentrationAlert(
                    alert_type=ConcentrationAlertType.LEVERAGE,
                    level=ConcentrationLevel.HIGH,
                    message=f"Weighted leverage {snapshot.weighted_leverage:.1f}x exceeds limit",
                    current_value=snapshot.weighted_leverage,
                    threshold=self._config.max_weighted_leverage,
                    affected_symbols=[],
                )
            )

        return alerts

    def _check_diversity(
        self, snapshot: ConcentrationSnapshot
    ) -> List[ConcentrationAlert]:
        """Check portfolio diversity."""
        alerts = []

        # Check minimum positions
        if snapshot.position_count < self._config.min_positions_for_diversity:
            # This is informational, not necessarily a problem
            pass

        # Check Herfindahl Index (concentration measure)
        if snapshot.herfindahl_index > self._config.max_herfindahl_index:
            level = ConcentrationLevel.ELEVATED
            if snapshot.herfindahl_index > self._config.max_herfindahl_index * Decimal("1.5"):
                level = ConcentrationLevel.HIGH

            existing = self._find_existing_alert(ConcentrationAlertType.SINGLE_POSITION, [])
            if not existing or "diversification" not in existing.message.lower():
                alerts.append(
                    ConcentrationAlert(
                        alert_type=ConcentrationAlertType.SINGLE_POSITION,
                        level=level,
                        message=f"Portfolio too concentrated (HHI: {snapshot.herfindahl_index:.3f}, "
                        f"effective positions: {snapshot.effective_positions:.1f})",
                        current_value=snapshot.herfindahl_index,
                        threshold=self._config.max_herfindahl_index,
                        affected_symbols=list(snapshot.position_concentrations.keys()),
                    )
                )

        return alerts

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _find_existing_alert(
        self, alert_type: ConcentrationAlertType, symbols: List[str]
    ) -> Optional[ConcentrationAlert]:
        """Find existing active alert of same type."""
        for alert in self._active_alerts:
            if alert.acknowledged:
                continue
            if alert.alert_type != alert_type:
                continue
            if symbols and set(alert.affected_symbols) != set(symbols):
                continue
            return alert
        return None

    # =========================================================================
    # Category Management
    # =========================================================================

    def add_category(self, category: AssetCategory) -> None:
        """Add or update asset category."""
        # Remove existing if present
        self._config.categories = [
            c for c in self._config.categories if c.name != category.name
        ]
        self._config.categories.append(category)

        # Update lookup
        for symbol in category.symbols:
            self._symbol_to_category[symbol] = category.name

        logger.info(f"Added category: {category.name} with {len(category.symbols)} symbols")

    def add_symbol_to_category(self, symbol: str, category_name: str) -> bool:
        """Add symbol to existing category."""
        for cat in self._config.categories:
            if cat.name == category_name:
                cat.symbols.add(symbol)
                self._symbol_to_category[symbol] = category_name
                return True
        return False

    def add_correlation_group(self, group_name: str, symbols: Set[str]) -> None:
        """Add or update correlation group."""
        self._config.correlation_groups[group_name] = symbols
        logger.info(f"Added correlation group: {group_name} with {len(symbols)} symbols")

    # =========================================================================
    # Alert Management
    # =========================================================================

    def acknowledge_alert(self, alert: ConcentrationAlert) -> None:
        """Acknowledge an alert."""
        alert.acknowledge()

    def acknowledge_all_alerts(self) -> int:
        """Acknowledge all active alerts. Returns count."""
        count = 0
        for alert in self._active_alerts:
            if not alert.acknowledged:
                alert.acknowledge()
                count += 1
        return count

    def clear_old_alerts(self, max_age_hours: int = 24) -> int:
        """Clear alerts older than max_age_hours. Returns count removed."""
        cutoff = datetime.now()
        from datetime import timedelta
        cutoff = cutoff - timedelta(hours=max_age_hours)

        original_count = len(self._active_alerts)
        self._active_alerts = [
            a for a in self._active_alerts
            if a.timestamp > cutoff or not a.acknowledged
        ]
        return original_count - len(self._active_alerts)

    # =========================================================================
    # Statistics and Reporting
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get monitoring statistics."""
        return {
            "total_checks": self._total_checks,
            "total_alerts": self._total_alerts,
            "active_alerts": len(self.active_alerts),
            "current_level": self.current_level.name,
            "categories_count": len(self._config.categories),
            "correlation_groups_count": len(self._config.correlation_groups),
            "enabled": self._config.enabled,
        }

    def get_concentration_summary(self) -> Optional[Dict]:
        """Get summary of current concentration."""
        if not self._last_snapshot:
            return None

        snapshot = self._last_snapshot

        return {
            "timestamp": snapshot.timestamp.isoformat(),
            "total_value": float(snapshot.total_portfolio_value),
            "position_count": snapshot.position_count,
            "largest_position": {
                "symbol": snapshot.largest_position_symbol,
                "percentage": float(snapshot.largest_position_pct),
            },
            "direction": {
                "long_pct": float(snapshot.long_exposure_pct),
                "short_pct": float(snapshot.short_exposure_pct),
                "net_exposure_pct": float(snapshot.net_exposure_pct),
            },
            "leverage": {
                "average": float(snapshot.average_leverage),
                "max": float(snapshot.max_leverage),
                "weighted": float(snapshot.weighted_leverage),
            },
            "diversity": {
                "herfindahl_index": float(snapshot.herfindahl_index),
                "effective_positions": float(snapshot.effective_positions),
            },
            "categories": {
                cat: float(pct)
                for cat, pct in snapshot.category_concentrations.items()
            },
        }

    def get_rebalancing_suggestions(self) -> List[Dict]:
        """
        Get suggestions for rebalancing to reduce concentration.

        Returns:
            List of suggestions with symbol and recommended action
        """
        if not self._last_snapshot:
            return []

        suggestions = []
        snapshot = self._last_snapshot

        # Suggest reducing oversized positions
        for symbol, pct in snapshot.position_concentrations.items():
            if pct > self._config.single_position_warning_pct:
                target_pct = self._config.single_position_warning_pct * Decimal("0.8")
                reduce_pct = pct - target_pct

                suggestions.append({
                    "symbol": symbol,
                    "action": "REDUCE",
                    "reason": f"Position is {pct:.1%}, reduce to ~{target_pct:.1%}",
                    "reduce_percentage": float(reduce_pct),
                    "priority": "HIGH" if pct > self._config.single_position_danger_pct else "MEDIUM",
                })

        # Suggest balancing direction
        if snapshot.long_exposure_pct > self._config.direction_imbalance_warning_pct:
            suggestions.append({
                "symbol": "PORTFOLIO",
                "action": "ADD_SHORTS",
                "reason": f"Long exposure at {snapshot.long_exposure_pct:.1%}",
                "priority": "MEDIUM",
            })
        elif snapshot.short_exposure_pct > self._config.direction_imbalance_warning_pct:
            suggestions.append({
                "symbol": "PORTFOLIO",
                "action": "ADD_LONGS",
                "reason": f"Short exposure at {snapshot.short_exposure_pct:.1%}",
                "priority": "MEDIUM",
            })

        return suggestions

    # =========================================================================
    # Configuration
    # =========================================================================

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated concentration config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def enable(self) -> None:
        """Enable concentration monitoring."""
        self._config.enabled = True
        logger.info("Concentration monitoring enabled")

    def disable(self) -> None:
        """Disable concentration monitoring."""
        self._config.enabled = False
        logger.warning("Concentration monitoring disabled")
