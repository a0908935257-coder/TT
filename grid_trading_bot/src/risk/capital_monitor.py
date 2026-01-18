"""
Capital Monitor.

Monitors total capital changes and detects anomalies.
Tracks capital snapshots, calculates daily P&L, and triggers alerts.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple

from src.core import get_logger
from src.core.models import AccountInfo, MarketType, Position

from .models import (
    CapitalSnapshot,
    DailyPnL,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_account(self, market: MarketType = MarketType.SPOT) -> AccountInfo:
        """Get account information."""
        ...

    async def get_positions(self, symbol: Optional[str] = None) -> List[Position]:
        """Get positions (futures only)."""
        ...


class CapitalMonitor:
    """
    Capital monitoring for risk management.

    Tracks total capital, calculates changes, and triggers alerts
    when thresholds are breached.

    Example:
        >>> config = RiskConfig(total_capital=Decimal("100000"))
        >>> monitor = CapitalMonitor(config, exchange)
        >>> await monitor.update()
        >>> change, change_pct = monitor.get_capital_change()
        >>> alerts = monitor.check_alerts()
    """

    def __init__(
        self,
        config: RiskConfig,
        exchange: Optional[ExchangeProtocol] = None,
        market_type: MarketType = MarketType.SPOT,
    ):
        """
        Initialize CapitalMonitor.

        Args:
            config: Risk configuration with thresholds
            exchange: Exchange client for fetching account data
            market_type: Market type to monitor (SPOT or FUTURES)
        """
        self._config = config
        self._exchange = exchange
        self._market_type = market_type

        # Capital tracking
        self._initial_capital: Decimal = config.total_capital
        self._peak_capital: Decimal = config.total_capital
        self._peak_time: datetime = datetime.now()

        # Daily tracking
        self._daily_start_capital: Optional[Decimal] = None
        self._daily_start_time: Optional[datetime] = None

        # Trade statistics for daily P&L
        self._daily_trade_count: int = 0
        self._daily_win_count: int = 0
        self._daily_loss_count: int = 0

        # Snapshot history
        self._snapshots: List[CapitalSnapshot] = []
        self._last_snapshot: Optional[CapitalSnapshot] = None

        # Maximum snapshots to keep in memory
        self._max_snapshots: int = 1000

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> RiskConfig:
        """Get risk configuration."""
        return self._config

    @property
    def initial_capital(self) -> Decimal:
        """Get initial capital."""
        return self._initial_capital

    @property
    def peak_capital(self) -> Decimal:
        """Get peak capital value."""
        return self._peak_capital

    @property
    def peak_time(self) -> datetime:
        """Get time when peak was reached."""
        return self._peak_time

    @property
    def last_snapshot(self) -> Optional[CapitalSnapshot]:
        """Get last capital snapshot."""
        return self._last_snapshot

    @property
    def snapshots(self) -> List[CapitalSnapshot]:
        """Get snapshot history."""
        return self._snapshots.copy()

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def update(self) -> CapitalSnapshot:
        """
        Update capital snapshot from exchange.

        Fetches account data, calculates totals, and creates a new snapshot.

        Returns:
            New CapitalSnapshot

        Raises:
            RuntimeError: If no exchange client configured
        """
        if self._exchange is None:
            raise RuntimeError("No exchange client configured")

        # Get account info
        account = await self._exchange.get_account(self._market_type)

        # Calculate totals from balances
        total_balance = Decimal("0")
        available_balance = Decimal("0")

        for balance in account.balances:
            # For spot, just sum USDT-like stablecoin balances
            # In real usage, would need price conversion for all assets
            if balance.asset in ("USDT", "BUSD", "USDC"):
                total_balance += balance.total
                available_balance += balance.free

        # For futures, get positions
        position_value = Decimal("0")
        unrealized_pnl = Decimal("0")

        if self._market_type == MarketType.FUTURES:
            positions = await self._exchange.get_positions()
            for pos in positions:
                position_value += pos.quantity * pos.mark_price
                unrealized_pnl += pos.unrealized_pnl

        # Calculate total capital
        total_capital = total_balance + position_value

        # Create snapshot
        snapshot = CapitalSnapshot(
            timestamp=datetime.now(),
            total_capital=total_capital,
            available_balance=available_balance,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=Decimal("0"),  # Would need to track this separately
        )

        # Update internal state
        self._update_snapshot(snapshot)

        return snapshot

    def update_from_values(
        self,
        total_capital: Decimal,
        available_balance: Decimal,
        position_value: Decimal = Decimal("0"),
        unrealized_pnl: Decimal = Decimal("0"),
        realized_pnl: Decimal = Decimal("0"),
    ) -> CapitalSnapshot:
        """
        Update capital snapshot from provided values.

        Useful for testing or when not using exchange client directly.

        Args:
            total_capital: Total capital value
            available_balance: Available balance
            position_value: Value of open positions
            unrealized_pnl: Unrealized P&L
            realized_pnl: Realized P&L

        Returns:
            New CapitalSnapshot
        """
        snapshot = CapitalSnapshot(
            timestamp=datetime.now(),
            total_capital=total_capital,
            available_balance=available_balance,
            position_value=position_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=realized_pnl,
        )

        self._update_snapshot(snapshot)
        return snapshot

    def _update_snapshot(self, snapshot: CapitalSnapshot) -> None:
        """
        Update internal state with new snapshot.

        Args:
            snapshot: New capital snapshot
        """
        # Initialize daily start capital if not set
        if self._daily_start_capital is None:
            self._daily_start_capital = snapshot.total_capital
            self._daily_start_time = snapshot.timestamp

        # Update peak if new high
        if snapshot.total_capital > self._peak_capital:
            self._peak_capital = snapshot.total_capital
            self._peak_time = snapshot.timestamp
            logger.debug(f"New peak capital: {self._peak_capital}")

        # Store snapshot
        self._snapshots.append(snapshot)
        self._last_snapshot = snapshot

        # Trim old snapshots
        if len(self._snapshots) > self._max_snapshots:
            self._snapshots = self._snapshots[-self._max_snapshots :]

        logger.debug(
            f"Capital updated: {snapshot.total_capital} "
            f"(peak: {self._peak_capital})"
        )

    def get_current(self) -> Optional[CapitalSnapshot]:
        """
        Get current capital snapshot.

        Returns:
            Latest CapitalSnapshot or None if no updates yet
        """
        return self._last_snapshot

    def get_capital_change(self) -> Tuple[Decimal, Decimal]:
        """
        Calculate capital change from initial.

        Returns:
            Tuple of (change_amount, change_percentage)
            Returns (0, 0) if no snapshot available
        """
        if self._last_snapshot is None:
            return Decimal("0"), Decimal("0")

        current = self._last_snapshot.total_capital
        change = current - self._initial_capital

        if self._initial_capital > 0:
            change_pct = change / self._initial_capital
        else:
            change_pct = Decimal("0")

        return change, change_pct

    def get_daily_pnl(self) -> DailyPnL:
        """
        Calculate today's P&L.

        Returns:
            DailyPnL record for today
        """
        today = date.today()

        # If no snapshot, return empty P&L
        if self._last_snapshot is None:
            return DailyPnL(
                date=today,
                starting_capital=self._initial_capital,
                ending_capital=self._initial_capital,
                pnl=Decimal("0"),
                pnl_pct=Decimal("0"),
                trade_count=0,
                win_count=0,
                loss_count=0,
            )

        # Use daily start capital or initial capital
        start_capital = self._daily_start_capital or self._initial_capital
        current = self._last_snapshot.total_capital
        pnl = current - start_capital

        if start_capital > 0:
            pnl_pct = pnl / start_capital
        else:
            pnl_pct = Decimal("0")

        return DailyPnL(
            date=today,
            starting_capital=start_capital,
            ending_capital=current,
            pnl=pnl,
            pnl_pct=pnl_pct,
            trade_count=self._daily_trade_count,
            win_count=self._daily_win_count,
            loss_count=self._daily_loss_count,
        )

    def check_alerts(self) -> List[RiskAlert]:
        """
        Check if any alert thresholds are breached.

        Returns:
            List of triggered RiskAlerts
        """
        alerts: List[RiskAlert] = []

        if self._last_snapshot is None:
            return alerts

        change, change_pct = self.get_capital_change()
        daily = self.get_daily_pnl()

        # Check total capital loss (danger threshold)
        if change_pct <= -self._config.danger_loss_pct:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.DANGER,
                    metric="total_capital_loss",
                    current_value=change_pct,
                    threshold=-self._config.danger_loss_pct,
                    message=f"Total capital loss {change_pct:.1%} reached danger threshold "
                    f"({-self._config.danger_loss_pct:.1%})",
                    action_taken=RiskAction.PAUSE_ALL_BOTS,
                )
            )
        # Check total capital loss (warning threshold)
        elif change_pct <= -self._config.warning_loss_pct:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.WARNING,
                    metric="total_capital_loss",
                    current_value=change_pct,
                    threshold=-self._config.warning_loss_pct,
                    message=f"Total capital loss {change_pct:.1%} reached warning threshold "
                    f"({-self._config.warning_loss_pct:.1%})",
                    action_taken=RiskAction.NOTIFY,
                )
            )

        # Check daily loss (danger threshold)
        if daily.pnl_pct <= -self._config.daily_loss_danger:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.DANGER,
                    metric="daily_loss",
                    current_value=daily.pnl_pct,
                    threshold=-self._config.daily_loss_danger,
                    message=f"Daily loss {daily.pnl_pct:.1%} reached danger threshold "
                    f"({-self._config.daily_loss_danger:.1%})",
                    action_taken=RiskAction.PAUSE_ALL_BOTS,
                )
            )
        # Check daily loss (warning threshold)
        elif daily.pnl_pct <= -self._config.daily_loss_warning:
            alerts.append(
                RiskAlert.create(
                    level=RiskLevel.WARNING,
                    metric="daily_loss",
                    current_value=daily.pnl_pct,
                    threshold=-self._config.daily_loss_warning,
                    message=f"Daily loss {daily.pnl_pct:.1%} reached warning threshold "
                    f"({-self._config.daily_loss_warning:.1%})",
                    action_taken=RiskAction.NOTIFY,
                )
            )

        return alerts

    def reset_daily(self) -> DailyPnL:
        """
        Reset daily statistics (call at start of each day).

        Returns:
            Yesterday's DailyPnL record
        """
        # Get yesterday's P&L before resetting
        yesterday_pnl = self.get_daily_pnl()

        # Reset daily tracking
        if self._last_snapshot:
            self._daily_start_capital = self._last_snapshot.total_capital
        else:
            self._daily_start_capital = self._initial_capital

        self._daily_start_time = datetime.now()
        self._daily_trade_count = 0
        self._daily_win_count = 0
        self._daily_loss_count = 0

        logger.info(
            f"Daily reset - Yesterday P&L: {yesterday_pnl.pnl} "
            f"({yesterday_pnl.pnl_pct:.2%}), "
            f"New start capital: {self._daily_start_capital}"
        )

        return yesterday_pnl

    # =========================================================================
    # Trade Recording
    # =========================================================================

    def record_trade(self, is_win: bool) -> None:
        """
        Record a completed trade for daily statistics.

        Args:
            is_win: True if the trade was profitable
        """
        self._daily_trade_count += 1
        if is_win:
            self._daily_win_count += 1
        else:
            self._daily_loss_count += 1

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def set_initial_capital(self, capital: Decimal) -> None:
        """
        Set initial capital (for recalibration).

        Args:
            capital: New initial capital value
        """
        self._initial_capital = capital
        self._config.total_capital = capital
        logger.info(f"Initial capital set to: {capital}")

    def get_snapshots_since(
        self, since: datetime
    ) -> List[CapitalSnapshot]:
        """
        Get snapshots since a specific time.

        Args:
            since: Start time

        Returns:
            List of snapshots since the given time
        """
        return [s for s in self._snapshots if s.timestamp >= since]

    def get_snapshots_for_period(
        self, start: datetime, end: datetime
    ) -> List[CapitalSnapshot]:
        """
        Get snapshots within a time period.

        Args:
            start: Start time
            end: End time

        Returns:
            List of snapshots within the period
        """
        return [s for s in self._snapshots if start <= s.timestamp <= end]

    def clear_snapshots(self) -> None:
        """Clear all snapshot history."""
        self._snapshots.clear()
        logger.debug("Snapshot history cleared")
