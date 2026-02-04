"""
Risk Management Models.

Defines data models for global risk management including:
- Risk levels and actions
- Capital snapshots
- Drawdown tracking
- Daily P&L records
- Circuit breaker state
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Optional
from uuid import uuid4


class RiskLevel(Enum):
    """Risk level classification."""

    NORMAL = 1  # Normal operation
    WARNING = 2  # Warning - approaching thresholds
    RISK = 3  # Risk - exceeded thresholds
    DANGER = 4  # Danger - severe loss
    CIRCUIT_BREAK = 5  # Circuit break - emergency stop triggered


class RiskAction(Enum):
    """Actions to take based on risk level."""

    NONE = "none"  # No action needed
    NOTIFY = "notify"  # Send notification
    PAUSE_NEW_ORDERS = "pause_new"  # Pause new order creation
    PAUSE_ALL_BOTS = "pause_all"  # Pause all bots
    EMERGENCY_STOP = "emergency"  # Emergency stop with position closure


@dataclass
class RiskConfig:
    """Configuration for risk management thresholds."""

    # Total initial capital
    total_capital: Decimal

    # Loss thresholds (percentage as decimal, e.g., 0.10 = 10%)
    warning_loss_pct: Decimal = Decimal("0.10")  # 10% loss triggers warning
    danger_loss_pct: Decimal = Decimal("0.20")  # 20% loss triggers danger

    # Daily loss thresholds
    daily_loss_warning: Decimal = Decimal("0.05")  # 5% daily loss warning
    daily_loss_danger: Decimal = Decimal("0.10")  # 10% daily loss danger

    # Maximum drawdown thresholds
    max_drawdown_warning: Decimal = Decimal("0.15")  # 15% drawdown warning
    max_drawdown_danger: Decimal = Decimal("0.25")  # 25% drawdown danger

    # Consecutive loss thresholds
    consecutive_loss_warning: int = 5  # 5 consecutive losses warning
    consecutive_loss_danger: int = 10  # 10 consecutive losses danger

    # Circuit breaker settings
    circuit_breaker_cooldown: int = 3600  # Cooldown period in seconds (1 hour)

    # Auto resume settings
    auto_resume_enabled: bool = False  # Whether to auto-resume after cooldown

    # Liquidation risk thresholds (percentage as decimal)
    liquidation_warning_pct: Decimal = Decimal("0.10")  # 10% from liquidation = WARNING
    liquidation_danger_pct: Decimal = Decimal("0.05")  # 5% from liquidation = DANGER
    liquidation_critical_pct: Decimal = Decimal("0.02")  # 2% from liquidation = CIRCUIT_BREAK


@dataclass
class CapitalSnapshot:
    """Snapshot of capital state at a point in time."""

    timestamp: datetime
    total_capital: Decimal  # Total capital value
    available_balance: Decimal  # Available balance for trading
    position_value: Decimal  # Value of open positions
    unrealized_pnl: Decimal  # Unrealized profit/loss
    realized_pnl: Decimal  # Realized profit/loss


@dataclass
class DrawdownInfo:
    """Information about current drawdown state."""

    peak_value: Decimal  # Peak capital value
    current_value: Decimal  # Current capital value
    drawdown_amount: Decimal  # Drawdown amount (peak - current)
    drawdown_pct: Decimal  # Drawdown percentage
    peak_time: datetime  # Time when peak was reached
    duration: timedelta  # Duration of drawdown

    @classmethod
    def calculate(
        cls, peak_value: Decimal, current_value: Decimal, peak_time: datetime
    ) -> "DrawdownInfo":
        """Calculate drawdown info from peak and current values."""
        drawdown_amount = peak_value - current_value
        drawdown_pct = (
            (drawdown_amount / peak_value) if peak_value > 0 else Decimal("0")
        )
        duration = datetime.now(timezone.utc) - peak_time

        return cls(
            peak_value=peak_value,
            current_value=current_value,
            drawdown_amount=drawdown_amount,
            drawdown_pct=drawdown_pct,
            peak_time=peak_time,
            duration=duration,
        )


@dataclass
class DailyPnL:
    """Daily profit and loss record."""

    date: date  # Date of record
    starting_capital: Decimal  # Capital at start of day
    ending_capital: Decimal  # Capital at end of day
    pnl: Decimal  # Profit/loss amount
    pnl_pct: Decimal  # Profit/loss percentage
    trade_count: int  # Total number of trades
    win_count: int  # Number of winning trades
    loss_count: int  # Number of losing trades

    @classmethod
    def create(
        cls,
        record_date: date,
        starting_capital: Decimal,
        ending_capital: Decimal,
        trade_count: int = 0,
        win_count: int = 0,
        loss_count: int = 0,
    ) -> "DailyPnL":
        """Create a DailyPnL record with calculated P&L."""
        pnl = ending_capital - starting_capital
        pnl_pct = (pnl / starting_capital) if starting_capital > 0 else Decimal("0")

        return cls(
            date=record_date,
            starting_capital=starting_capital,
            ending_capital=ending_capital,
            pnl=pnl,
            pnl_pct=pnl_pct,
            trade_count=trade_count,
            win_count=win_count,
            loss_count=loss_count,
        )

    @property
    def win_rate(self) -> Decimal:
        """Calculate win rate as percentage."""
        if self.trade_count == 0:
            return Decimal("0")
        return Decimal(self.win_count) / Decimal(self.trade_count)


@dataclass
class RiskAlert:
    """Alert generated when risk threshold is breached."""

    alert_id: str  # Unique alert ID
    level: RiskLevel  # Risk level of alert
    metric: str  # Which metric triggered the alert
    current_value: Decimal  # Current value of the metric
    threshold: Decimal  # Threshold that was breached
    message: str  # Human-readable message
    action_taken: RiskAction  # Action taken in response
    timestamp: datetime  # When alert was generated
    acknowledged: bool = False  # Whether alert has been acknowledged

    @classmethod
    def create(
        cls,
        level: RiskLevel,
        metric: str,
        current_value: Decimal,
        threshold: Decimal,
        message: str,
        action_taken: RiskAction = RiskAction.NONE,
    ) -> "RiskAlert":
        """Create a new risk alert."""
        return cls(
            alert_id=str(uuid4()),
            level=level,
            metric=metric,
            current_value=current_value,
            threshold=threshold,
            message=message,
            action_taken=action_taken,
            timestamp=datetime.now(timezone.utc),
            acknowledged=False,
        )

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True


@dataclass
class CircuitBreakerState:
    """State of the circuit breaker."""

    is_triggered: bool = False  # Whether circuit breaker is active
    triggered_at: Optional[datetime] = None  # When it was triggered
    trigger_reason: str = ""  # Reason for trigger
    cooldown_until: Optional[datetime] = None  # When cooldown ends
    trigger_count_today: int = 0  # Number of triggers today

    def trigger(self, reason: str, cooldown_seconds: int) -> None:
        """Trigger the circuit breaker."""
        now = datetime.now(timezone.utc)
        self.is_triggered = True
        self.triggered_at = now
        self.trigger_reason = reason
        self.cooldown_until = now + timedelta(seconds=cooldown_seconds)
        self.trigger_count_today += 1

    def reset(self) -> None:
        """Reset the circuit breaker."""
        self.is_triggered = False
        self.triggered_at = None
        self.trigger_reason = ""
        self.cooldown_until = None

    def reset_daily_count(self) -> None:
        """Reset the daily trigger count."""
        self.trigger_count_today = 0

    @property
    def is_in_cooldown(self) -> bool:
        """Check if still in cooldown period."""
        if not self.cooldown_until:
            return False
        return datetime.now(timezone.utc) < self.cooldown_until

    @property
    def cooldown_remaining(self) -> timedelta:
        """Get remaining cooldown time."""
        if not self.cooldown_until:
            return timedelta(0)
        remaining = self.cooldown_until - datetime.now(timezone.utc)
        return remaining if remaining > timedelta(0) else timedelta(0)


@dataclass
class LiquidationSnapshot:
    """
    Snapshot of liquidation risk for a single position.

    Tracks how close a futures position is to its liquidation price.
    """

    symbol: str  # Trading pair symbol
    side: str  # Position side: "LONG" or "SHORT"
    mark_price: Decimal  # Current mark price
    liquidation_price: Decimal  # Liquidation price
    entry_price: Decimal  # Entry price
    distance_pct: Decimal  # Distance to liquidation as decimal (0.10 = 10%)
    leverage: int  # Position leverage
    quantity: Decimal  # Position quantity
    unrealized_pnl: Decimal  # Unrealized P&L
    risk_level: RiskLevel  # Calculated risk level
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GlobalRiskStatus:
    """Overall global risk status."""

    level: RiskLevel  # Current risk level
    capital: CapitalSnapshot  # Latest capital snapshot
    drawdown: DrawdownInfo  # Current drawdown info
    daily_pnl: DailyPnL  # Today's P&L
    circuit_breaker: CircuitBreakerState  # Circuit breaker state
    active_alerts: List[RiskAlert] = field(default_factory=list)  # Active alerts
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # Last update time

    def add_alert(self, alert: RiskAlert) -> None:
        """Add an alert to the active alerts list."""
        self.active_alerts.append(alert)
        self.last_updated = datetime.now(timezone.utc)

    def get_unacknowledged_alerts(self) -> List[RiskAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self.active_alerts if not a.acknowledged]

    def clear_acknowledged_alerts(self) -> None:
        """Remove all acknowledged alerts."""
        self.active_alerts = [a for a in self.active_alerts if not a.acknowledged]
        self.last_updated = datetime.now(timezone.utc)

    def update_level(self, new_level: RiskLevel) -> bool:
        """Update risk level if it has changed. Returns True if changed."""
        if self.level != new_level:
            self.level = new_level
            self.last_updated = datetime.now(timezone.utc)
            return True
        return False
