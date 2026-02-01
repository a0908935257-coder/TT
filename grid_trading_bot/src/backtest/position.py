"""
Position Management Module.

Handles position tracking, P&L calculation, and trade recording.
"""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional

from .result import ExitReason, Trade


@dataclass
class Position:
    """
    Represents an open position.

    Attributes:
        side: Position direction ('LONG' or 'SHORT')
        entry_price: Entry fill price
        quantity: Position size
        entry_time: Entry timestamp
        entry_bar: Bar index at entry
        entry_fee: Fee paid on entry
        stop_loss: Stop loss price (optional)
        take_profit: Take profit price (optional)
        max_favorable_price: Most favorable price seen (for trailing stop)
        max_adverse_price: Most adverse price seen (for tracking)
    """

    side: str
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    entry_bar: int
    entry_fee: Decimal = field(default_factory=lambda: Decimal("0"))
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    max_favorable_price: Optional[Decimal] = None
    max_adverse_price: Optional[Decimal] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize tracking prices."""
        if self.max_favorable_price is None:
            self.max_favorable_price = self.entry_price
        if self.max_adverse_price is None:
            self.max_adverse_price = self.entry_price

    @property
    def notional(self) -> Decimal:
        """Notional value at entry."""
        return self.entry_price * self.quantity

    def update_tracking_prices(self, current_price: Decimal) -> None:
        """Update max favorable/adverse prices based on current price."""
        if self.side == "LONG":
            if current_price > self.max_favorable_price:
                self.max_favorable_price = current_price
            if current_price < self.max_adverse_price:
                self.max_adverse_price = current_price
        else:  # SHORT
            if current_price < self.max_favorable_price:
                self.max_favorable_price = current_price
            if current_price > self.max_adverse_price:
                self.max_adverse_price = current_price

    def liquidation_price(self, leverage: int, maintenance_margin_pct: Decimal) -> Optional[Decimal]:
        """Calculate liquidation price for leveraged position.

        Args:
            leverage: Leverage multiplier
            maintenance_margin_pct: Maintenance margin percentage (e.g. 0.004 for 0.4%)

        Returns:
            Liquidation price, or None if leverage <= 1
        """
        if leverage <= 1:
            return None
        lev = Decimal(leverage)
        if self.side == "LONG":
            return self.entry_price * (Decimal("1") - (Decimal("1") - maintenance_margin_pct) / lev)
        else:
            return self.entry_price * (Decimal("1") + (Decimal("1") - maintenance_margin_pct) / lev)

    def unrealized_pnl(self, current_price: Decimal, leverage: int = 1) -> Decimal:
        """Calculate unrealized P&L at given price."""
        if self.side == "LONG":
            pnl = (current_price - self.entry_price) * self.quantity
        else:
            pnl = (self.entry_price - current_price) * self.quantity
        return pnl

    def unrealized_pnl_pct(self, current_price: Decimal, leverage: int = 1) -> Decimal:
        """Calculate unrealized P&L percentage."""
        if self.entry_price == 0:
            return Decimal("0")
        if self.side == "LONG":
            pct = (current_price - self.entry_price) / self.entry_price
        else:
            pct = (self.entry_price - current_price) / self.entry_price
        return pct * Decimal("100")


class PositionManager:
    """
    Manages positions and records completed trades.

    Handles single or multiple position tracking, P&L calculation,
    and trade history.
    """

    def __init__(self, max_positions: int = 1) -> None:
        """
        Initialize position manager.

        Args:
            max_positions: Maximum concurrent positions allowed
        """
        self._max_positions = max_positions
        self._positions: list[Position] = []
        self._trades: list[Trade] = []
        self._realized_pnl: Decimal = Decimal("0")
        self._total_funding_paid: Decimal = Decimal("0")

    @property
    def positions(self) -> list[Position]:
        """Get all open positions."""
        return self._positions.copy()

    @property
    def trades(self) -> list[Trade]:
        """Get all completed trades."""
        return self._trades.copy()

    @property
    def realized_pnl(self) -> Decimal:
        """Get total realized P&L."""
        return self._realized_pnl

    @property
    def has_position(self) -> bool:
        """Check if any position is open."""
        return len(self._positions) > 0

    @property
    def can_open_position(self) -> bool:
        """Check if new position can be opened."""
        return len(self._positions) < self._max_positions

    @property
    def current_position(self) -> Optional[Position]:
        """Get current position (for single-position mode)."""
        if self._positions:
            return self._positions[0]
        return None

    def open_position(self, position: Position) -> bool:
        """
        Open a new position.

        Args:
            position: Position to open

        Returns:
            True if position was opened, False if at max positions
        """
        if not self.can_open_position:
            return False
        self._positions.append(position)
        return True

    def close_position(
        self,
        position: Position,
        exit_price: Decimal,
        exit_time: datetime,
        exit_bar: int,
        exit_fee: Decimal,
        exit_reason: ExitReason,
        leverage: int = 1,
    ) -> Trade:
        """
        Close a position and record the trade.

        Args:
            position: Position to close
            exit_price: Exit fill price
            exit_time: Exit timestamp
            exit_bar: Bar index at exit
            exit_fee: Fee paid on exit
            exit_reason: Reason for closing
            leverage: Leverage multiplier

        Returns:
            Completed Trade record
        """
        # Calculate P&L with leverage
        # In futures trading, leverage multiplies the position's effective exposure
        if position.side == "LONG":
            gross_pnl = (exit_price - position.entry_price) * position.quantity
        else:
            gross_pnl = (position.entry_price - exit_price) * position.quantity

        # quantity already includes leverage, so gross_pnl is correct as-is

        # Deduct fees
        total_fees = position.entry_fee + exit_fee
        net_pnl = gross_pnl - total_fees

        # Calculate ROI percentage based on notional
        if position.notional > 0:
            pnl_pct = (net_pnl / position.notional) * Decimal("100")
        else:
            pnl_pct = Decimal("0")

        # Create trade record
        trade = Trade(
            entry_time=position.entry_time,
            exit_time=exit_time,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=total_fees,
            bars_held=exit_bar - position.entry_bar,
            exit_reason=exit_reason,
            metadata=position.metadata,
        )

        # Update state
        self._trades.append(trade)
        self._realized_pnl += net_pnl
        self._positions.remove(position)

        return trade

    def close_all_positions(
        self,
        exit_price: Decimal,
        exit_time: datetime,
        exit_bar: int,
        fee_rate: Decimal,
        exit_reason: ExitReason,
        leverage: int = 1,
    ) -> list[Trade]:
        """
        Close all open positions.

        Args:
            exit_price: Exit fill price
            exit_time: Exit timestamp
            exit_bar: Bar index at exit
            fee_rate: Fee rate for exit calculation
            exit_reason: Reason for closing
            leverage: Leverage multiplier

        Returns:
            List of completed Trade records
        """
        trades = []
        for position in self._positions.copy():
            exit_fee = exit_price * position.quantity * fee_rate
            trade = self.close_position(
                position=position,
                exit_price=exit_price,
                exit_time=exit_time,
                exit_bar=exit_bar,
                exit_fee=exit_fee,
                exit_reason=exit_reason,
                leverage=leverage,
            )
            trades.append(trade)
        return trades

    @property
    def total_funding_paid(self) -> Decimal:
        """Get total funding fees paid."""
        return self._total_funding_paid

    def add_funding_payment(self, amount: Decimal) -> None:
        """Record a funding rate payment."""
        self._total_funding_paid += amount

    def used_margin(self, leverage: int = 1) -> Decimal:
        """Calculate total margin used by open positions (notional / leverage)."""
        lev = Decimal(leverage) if leverage > 1 else Decimal("1")
        return sum((pos.notional / lev for pos in self._positions), Decimal("0"))

    def available_margin(self, current_price: Decimal, initial_capital: Decimal, leverage: int = 1) -> Decimal:
        """Calculate available margin for new positions."""
        equity = self.total_equity(current_price, initial_capital, leverage)
        return equity - self.used_margin(leverage)

    def unrealized_pnl(self, current_price: Decimal, leverage: int = 1) -> Decimal:
        """Calculate total unrealized P&L across all positions."""
        return sum(
            pos.unrealized_pnl(current_price, leverage) for pos in self._positions
        )

    def total_equity(
        self, current_price: Decimal, initial_capital: Decimal, leverage: int = 1
    ) -> Decimal:
        """Calculate total equity including unrealized P&L and funding costs."""
        return initial_capital + self._realized_pnl + self.unrealized_pnl(
            current_price, leverage
        ) - self._total_funding_paid

    def reset(self) -> None:
        """Reset all positions and trades."""
        self._positions.clear()
        self._trades.clear()
        self._realized_pnl = Decimal("0")
        self._total_funding_paid = Decimal("0")
