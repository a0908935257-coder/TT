"""
Liquidation Price Monitor.

Monitors futures positions for proximity to liquidation prices and generates
alerts when positions approach dangerous levels.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import TYPE_CHECKING, Dict, List, Optional, Protocol

from src.core import get_logger

from .models import (
    LiquidationSnapshot,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)

if TYPE_CHECKING:
    from src.core.models import Position

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_positions(self, symbol: Optional[str] = None) -> list:
        """Get positions."""
        ...


class LiquidationMonitor:
    """
    Liquidation price monitor for futures positions.

    Monitors all futures positions and generates alerts when positions
    approach their liquidation prices.

    Liquidation distance calculation:
    - LONG: distance = (mark_price - liquidation_price) / mark_price
      (Price drops toward liquidation_price)
    - SHORT: distance = (liquidation_price - mark_price) / mark_price
      (Price rises toward liquidation_price)

    Risk levels:
    - NORMAL: distance >= 10% (configurable)
    - WARNING: distance < 10%
    - DANGER: distance < 5%
    - CIRCUIT_BREAK: distance < 2% or already past liquidation

    Example:
        >>> monitor = LiquidationMonitor(config, exchange)
        >>> snapshots = await monitor.update()
        >>> alerts = monitor.check_alerts()
    """

    def __init__(
        self,
        config: RiskConfig,
        exchange: Optional[ExchangeProtocol] = None,
    ):
        """
        Initialize LiquidationMonitor.

        Args:
            config: Risk configuration with liquidation thresholds
            exchange: Exchange client for fetching positions
        """
        self._config = config
        self._exchange = exchange
        self._snapshots: Dict[str, LiquidationSnapshot] = {}
        self._last_update: Optional[datetime] = None

    @property
    def snapshots(self) -> Dict[str, LiquidationSnapshot]:
        """Get current liquidation snapshots."""
        return self._snapshots

    @property
    def last_update(self) -> Optional[datetime]:
        """Get last update timestamp."""
        return self._last_update

    async def update(self) -> Dict[str, LiquidationSnapshot]:
        """
        Update liquidation risk snapshots for all positions.

        Fetches current positions from exchange and calculates
        liquidation distance for each.

        Returns:
            Dict[symbol, LiquidationSnapshot] for each position
        """
        if not self._exchange:
            return {}

        try:
            positions = await self._exchange.get_positions()
        except Exception as e:
            logger.error(f"Failed to get positions for liquidation check: {e}")
            return self._snapshots  # Return last known snapshots

        new_snapshots: Dict[str, LiquidationSnapshot] = {}
        for pos in positions:
            snapshot = self._calculate_snapshot(pos)
            if snapshot:
                # Use symbol + side as key for hedge mode support
                key = f"{pos.symbol}_{pos.side.value if hasattr(pos.side, 'value') else pos.side}"
                new_snapshots[key] = snapshot

        self._snapshots = new_snapshots
        self._last_update = datetime.now(timezone.utc)

        # Log warnings for high-risk positions
        for key, snapshot in new_snapshots.items():
            if snapshot.risk_level.value >= RiskLevel.WARNING.value:
                logger.warning(
                    f"Liquidation risk {snapshot.risk_level.name}: {snapshot.symbol} "
                    f"{snapshot.side} is {snapshot.distance_pct * 100:.2f}% from liquidation"
                )

        return self._snapshots

    def _calculate_snapshot(self, pos: "Position") -> Optional[LiquidationSnapshot]:
        """
        Calculate liquidation risk snapshot for a single position.

        Handles edge cases:
        - liquidation_price is None or 0: skip (no liquidation info)
        - mark_price is 0: skip (avoid division by zero)
        - side is BOTH: skip (no actual position)
        - quantity is 0: skip (no position)

        Args:
            pos: Position to analyze

        Returns:
            LiquidationSnapshot or None if position should be skipped
        """
        # Edge case: no quantity
        if pos.quantity <= 0:
            return None

        # Edge case: BOTH side means no actual directional position
        side_value = pos.side.value if hasattr(pos.side, "value") else str(pos.side)
        if side_value == "BOTH":
            return None

        # Edge case: no liquidation price
        if pos.liquidation_price is None:
            return None
        if pos.liquidation_price <= 0:
            return None

        # Edge case: invalid mark price (avoid division by zero)
        if pos.mark_price <= 0:
            return None

        # Calculate distance to liquidation
        if pos.is_long:
            # LONG: liquidation when price drops
            # Normal state: mark_price > liquidation_price
            # Distance = (mark - liq) / mark
            distance_pct = (pos.mark_price - pos.liquidation_price) / pos.mark_price
        else:
            # SHORT: liquidation when price rises
            # Normal state: mark_price < liquidation_price
            # Distance = (liq - mark) / mark
            distance_pct = (pos.liquidation_price - pos.mark_price) / pos.mark_price

        # Calculate risk level
        risk_level = self._calculate_risk_level(distance_pct)

        return LiquidationSnapshot(
            symbol=pos.symbol,
            side=side_value,
            mark_price=pos.mark_price,
            liquidation_price=pos.liquidation_price,
            entry_price=pos.entry_price,
            distance_pct=distance_pct,
            leverage=pos.leverage,
            quantity=pos.quantity,
            unrealized_pnl=pos.unrealized_pnl,
            risk_level=risk_level,
        )

    def _calculate_risk_level(self, distance_pct: Decimal) -> RiskLevel:
        """
        Calculate risk level based on distance to liquidation.

        Args:
            distance_pct: Distance to liquidation as decimal

        Returns:
            RiskLevel based on thresholds
        """
        # Already past liquidation price (negative distance)
        if distance_pct <= 0:
            return RiskLevel.CIRCUIT_BREAK

        # Critical - very close to liquidation
        if distance_pct < self._config.liquidation_critical_pct:
            return RiskLevel.CIRCUIT_BREAK

        # Danger - close to liquidation
        if distance_pct < self._config.liquidation_danger_pct:
            return RiskLevel.DANGER

        # Warning - approaching liquidation
        if distance_pct < self._config.liquidation_warning_pct:
            return RiskLevel.WARNING

        return RiskLevel.NORMAL

    def check_alerts(self) -> List[RiskAlert]:
        """
        Check all positions and generate alerts for risky ones.

        Returns:
            List of RiskAlert for positions with elevated risk
        """
        alerts: List[RiskAlert] = []

        for key, snapshot in self._snapshots.items():
            if snapshot.risk_level == RiskLevel.NORMAL:
                continue

            message = self._format_alert_message(snapshot)
            action = self._determine_action(snapshot.risk_level)
            threshold = self._get_threshold_for_level(snapshot.risk_level)

            alert = RiskAlert.create(
                level=snapshot.risk_level,
                metric="liquidation_distance",
                current_value=snapshot.distance_pct * 100,  # Convert to percentage
                threshold=threshold * 100,
                message=message,
                action_taken=action,
            )
            alerts.append(alert)

        return alerts

    def _format_alert_message(self, snapshot: LiquidationSnapshot) -> str:
        """
        Format alert message for a position.

        Args:
            snapshot: Liquidation snapshot

        Returns:
            Human-readable alert message
        """
        distance_str = f"{snapshot.distance_pct * 100:.2f}%"

        if snapshot.distance_pct <= 0:
            return (
                f"CRITICAL: {snapshot.symbol} {snapshot.side} position has EXCEEDED "
                f"liquidation price! Mark: {snapshot.mark_price}, "
                f"Liq: {snapshot.liquidation_price}, Leverage: {snapshot.leverage}x"
            )

        return (
            f"{snapshot.symbol} {snapshot.side} position is {distance_str} from liquidation. "
            f"Mark: {snapshot.mark_price}, Liq: {snapshot.liquidation_price}, "
            f"Leverage: {snapshot.leverage}x, PnL: {snapshot.unrealized_pnl}"
        )

    def _determine_action(self, level: RiskLevel) -> RiskAction:
        """
        Determine action based on risk level.

        Args:
            level: Risk level

        Returns:
            Appropriate RiskAction
        """
        if level == RiskLevel.CIRCUIT_BREAK:
            return RiskAction.EMERGENCY_STOP
        if level == RiskLevel.DANGER:
            return RiskAction.PAUSE_ALL_BOTS
        if level == RiskLevel.WARNING:
            return RiskAction.NOTIFY
        return RiskAction.NONE

    def _get_threshold_for_level(self, level: RiskLevel) -> Decimal:
        """
        Get the threshold that was breached for a risk level.

        Args:
            level: Risk level

        Returns:
            Threshold value as decimal
        """
        if level == RiskLevel.CIRCUIT_BREAK:
            return self._config.liquidation_critical_pct
        if level == RiskLevel.DANGER:
            return self._config.liquidation_danger_pct
        if level == RiskLevel.WARNING:
            return self._config.liquidation_warning_pct
        return Decimal("1.0")  # 100% for normal

    def get_highest_risk(self) -> Optional[LiquidationSnapshot]:
        """
        Get the position with highest liquidation risk.

        Returns:
            LiquidationSnapshot with lowest distance_pct, or None if no positions
        """
        if not self._snapshots:
            return None

        return min(self._snapshots.values(), key=lambda s: s.distance_pct)

    def get_risky_positions(self, min_level: RiskLevel = RiskLevel.WARNING) -> List[LiquidationSnapshot]:
        """
        Get all positions with risk level at or above min_level.

        Args:
            min_level: Minimum risk level to include

        Returns:
            List of risky LiquidationSnapshots
        """
        return [
            snapshot
            for snapshot in self._snapshots.values()
            if snapshot.risk_level.value >= min_level.value
        ]
