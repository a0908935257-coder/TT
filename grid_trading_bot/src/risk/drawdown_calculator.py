"""
Drawdown Calculator.

Tracks maximum drawdown and historical drawdown periods.
"""

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional

from src.core import get_logger

from .models import (
    DrawdownInfo,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)

logger = get_logger(__name__)


class DrawdownCalculator:
    """
    Calculates and tracks drawdown metrics.

    Monitors capital changes to identify drawdown periods and track
    the maximum drawdown for risk management.

    Example:
        >>> config = RiskConfig(total_capital=Decimal("100000"))
        >>> calculator = DrawdownCalculator(config)
        >>> calculator.update(Decimal("100000"))  # Peak
        >>> calculator.update(Decimal("90000"))   # 10% drawdown
        >>> dd = calculator.get_current_drawdown()
        >>> print(f"Current drawdown: {dd.drawdown_pct:.1%}")
    """

    def __init__(self, config: RiskConfig):
        """
        Initialize DrawdownCalculator.

        Args:
            config: Risk configuration with thresholds
        """
        self._config = config

        # Peak tracking
        self._peak_value: Decimal = Decimal("0")
        self._peak_time: Optional[datetime] = None

        # Drawdown tracking
        self._current_drawdown: Optional[DrawdownInfo] = None
        self._max_drawdown: Optional[DrawdownInfo] = None

        # History
        self._drawdown_history: List[DrawdownInfo] = []

        # Time tracking for statistics
        self._first_update_time: Optional[datetime] = None
        self._time_in_drawdown: timedelta = timedelta(0)
        self._last_update_time: Optional[datetime] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> RiskConfig:
        """Get risk configuration."""
        return self._config

    @property
    def peak_value(self) -> Decimal:
        """Get current peak value."""
        return self._peak_value

    @property
    def peak_time(self) -> Optional[datetime]:
        """Get time when peak was reached."""
        return self._peak_time

    @property
    def current_drawdown(self) -> Optional[DrawdownInfo]:
        """Get current drawdown info."""
        return self._current_drawdown

    @property
    def max_drawdown(self) -> Optional[DrawdownInfo]:
        """Get maximum drawdown info."""
        return self._max_drawdown

    @property
    def drawdown_history(self) -> List[DrawdownInfo]:
        """Get drawdown history (completed drawdown periods)."""
        return self._drawdown_history.copy()

    # =========================================================================
    # Core Methods
    # =========================================================================

    def update(self, current_value: Decimal) -> DrawdownInfo:
        """
        Update drawdown calculation with new value.

        Args:
            current_value: Current capital value

        Returns:
            Current DrawdownInfo
        """
        now = datetime.now()

        # Track first update time
        if self._first_update_time is None:
            self._first_update_time = now

        # Track time in drawdown
        if (
            self._last_update_time
            and self._current_drawdown
            and self._current_drawdown.drawdown_pct > 0
        ):
            self._time_in_drawdown += now - self._last_update_time

        self._last_update_time = now

        # Check if new peak (also treat first update as new peak)
        if current_value > self._peak_value or self._peak_time is None:
            self._handle_new_peak(current_value, now)
        else:
            self._handle_drawdown(current_value, now)

        return self._current_drawdown

    def _handle_new_peak(self, current_value: Decimal, now: datetime) -> None:
        """
        Handle new peak value reached.

        Args:
            current_value: New peak value
            now: Current timestamp
        """
        # End current drawdown period if exists and was in drawdown
        if (
            self._current_drawdown
            and self._current_drawdown.drawdown_pct > Decimal("0")
        ):
            self._drawdown_history.append(self._current_drawdown)
            logger.debug(
                f"Drawdown period ended: {self._current_drawdown.drawdown_pct:.2%} "
                f"over {self._current_drawdown.duration}"
            )

        # Update peak
        self._peak_value = current_value
        self._peak_time = now

        # Reset current drawdown (no drawdown at peak)
        self._current_drawdown = DrawdownInfo(
            peak_value=current_value,
            current_value=current_value,
            drawdown_amount=Decimal("0"),
            drawdown_pct=Decimal("0"),
            peak_time=now,
            duration=timedelta(0),
        )

        logger.debug(f"New peak: {current_value}")

    def _handle_drawdown(self, current_value: Decimal, now: datetime) -> None:
        """
        Handle drawdown (value below peak).

        Args:
            current_value: Current value
            now: Current timestamp
        """
        if self._peak_value <= 0:
            return

        # Calculate drawdown
        drawdown_amount = self._peak_value - current_value
        drawdown_pct = drawdown_amount / self._peak_value
        duration = now - self._peak_time if self._peak_time else timedelta(0)

        self._current_drawdown = DrawdownInfo(
            peak_value=self._peak_value,
            current_value=current_value,
            drawdown_amount=drawdown_amount,
            drawdown_pct=drawdown_pct,
            peak_time=self._peak_time,
            duration=duration,
        )

        # Update max drawdown if this is worse
        if self._max_drawdown is None or drawdown_pct > self._max_drawdown.drawdown_pct:
            self._max_drawdown = self._current_drawdown
            logger.debug(f"New max drawdown: {drawdown_pct:.2%}")

    def get_current_drawdown(self) -> Optional[DrawdownInfo]:
        """
        Get current drawdown information.

        Returns:
            Current DrawdownInfo or None if no updates yet
        """
        return self._current_drawdown

    def get_max_drawdown(self) -> Optional[DrawdownInfo]:
        """
        Get maximum drawdown information.

        Returns:
            Maximum DrawdownInfo or None if never in drawdown
        """
        return self._max_drawdown

    def get_drawdown_history(self) -> List[DrawdownInfo]:
        """
        Get history of completed drawdown periods.

        Returns:
            List of DrawdownInfo for completed periods
        """
        return self._drawdown_history.copy()

    def check_alerts(self) -> List[RiskAlert]:
        """
        Check if current drawdown triggers alerts.

        Returns:
            List of RiskAlerts for breached thresholds
        """
        alerts: List[RiskAlert] = []

        if self._current_drawdown is None:
            return alerts

        dd_pct = self._current_drawdown.drawdown_pct

        # Check danger threshold first
        if dd_pct >= self._config.max_drawdown_danger:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.DANGER,
                    metric="max_drawdown",
                    current_value=dd_pct,
                    threshold=self._config.max_drawdown_danger,
                    message=f"Drawdown at {dd_pct:.1%}, reached danger threshold "
                    f"({self._config.max_drawdown_danger:.0%})",
                    action_taken=RiskAction.PAUSE_ALL_BOTS,
                )
            )
        # Check warning threshold
        elif dd_pct >= self._config.max_drawdown_warning:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.WARNING,
                    metric="max_drawdown",
                    current_value=dd_pct,
                    threshold=self._config.max_drawdown_warning,
                    message=f"Drawdown at {dd_pct:.1%}, reached warning threshold "
                    f"({self._config.max_drawdown_warning:.0%})",
                    action_taken=RiskAction.NOTIFY,
                )
            )

        return alerts

    def reset(self) -> None:
        """
        Reset the calculator (use with caution).

        Clears all tracking data including peak, history, and max drawdown.
        """
        logger.warning("Drawdown calculator reset - all data cleared")

        self._peak_value = Decimal("0")
        self._peak_time = None
        self._current_drawdown = None
        self._max_drawdown = None
        self._drawdown_history.clear()
        self._first_update_time = None
        self._time_in_drawdown = timedelta(0)
        self._last_update_time = None

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict:
        """
        Get comprehensive drawdown statistics.

        Returns:
            Dictionary with drawdown statistics
        """
        return {
            "current_drawdown_pct": (
                self._current_drawdown.drawdown_pct
                if self._current_drawdown
                else Decimal("0")
            ),
            "max_drawdown_pct": (
                self._max_drawdown.drawdown_pct
                if self._max_drawdown
                else Decimal("0")
            ),
            "max_drawdown_duration": (
                self._max_drawdown.duration if self._max_drawdown else timedelta(0)
            ),
            "drawdown_count": len(self._drawdown_history),
            "average_drawdown_pct": self._calc_avg_drawdown(),
            "time_in_drawdown": self._time_in_drawdown,
            "time_in_drawdown_pct": self._calc_time_in_drawdown_pct(),
            "peak_value": self._peak_value,
        }

    def _calc_avg_drawdown(self) -> Decimal:
        """
        Calculate average drawdown percentage from history.

        Returns:
            Average drawdown percentage
        """
        if not self._drawdown_history:
            return Decimal("0")

        total = sum(dd.drawdown_pct for dd in self._drawdown_history)
        return total / len(self._drawdown_history)

    def _calc_time_in_drawdown_pct(self) -> Decimal:
        """
        Calculate percentage of time spent in drawdown.

        Returns:
            Percentage as Decimal (0.0 to 1.0)
        """
        if not self._first_update_time or not self._last_update_time:
            return Decimal("0")

        total_time = self._last_update_time - self._first_update_time
        if total_time.total_seconds() <= 0:
            return Decimal("0")

        pct = Decimal(str(self._time_in_drawdown.total_seconds())) / Decimal(
            str(total_time.total_seconds())
        )
        return min(pct, Decimal("1"))  # Cap at 100%

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def is_in_drawdown(self) -> bool:
        """
        Check if currently in drawdown.

        Returns:
            True if current drawdown > 0
        """
        return (
            self._current_drawdown is not None
            and self._current_drawdown.drawdown_pct > Decimal("0")
        )

    def get_recovery_needed(self) -> Decimal:
        """
        Calculate percentage gain needed to recover from current drawdown.

        The formula: recovery_pct = drawdown / (1 - drawdown)
        For 10% drawdown, need 11.1% gain to recover.
        For 50% drawdown, need 100% gain to recover.

        Returns:
            Percentage gain needed to recover
        """
        if not self._current_drawdown or self._current_drawdown.drawdown_pct <= 0:
            return Decimal("0")

        dd_pct = self._current_drawdown.drawdown_pct
        if dd_pct >= Decimal("1"):
            return Decimal("inf")

        return dd_pct / (Decimal("1") - dd_pct)

    def set_initial_peak(self, value: Decimal) -> None:
        """
        Set initial peak value (for initialization without update).

        Args:
            value: Initial peak value
        """
        if self._peak_value == Decimal("0"):
            self._peak_value = value
            self._peak_time = datetime.now()
            logger.debug(f"Initial peak set to: {value}")
