"""
Order Simulation Module.

Handles order matching simulation using kline high/low prices.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..core.models import Kline
from .config import BacktestConfig
from .position import Position


class SignalType(str, Enum):
    """Trading signal type."""

    LONG_ENTRY = "long_entry"
    SHORT_ENTRY = "short_entry"
    LONG_EXIT = "long_exit"
    SHORT_EXIT = "short_exit"
    CLOSE_ALL = "close_all"


@dataclass
class Signal:
    """
    Trading signal from strategy.

    Attributes:
        signal_type: Type of signal
        price: Target price (None for market orders)
        stop_loss: Stop loss price (for entries)
        take_profit: Take profit price (for entries)
        reason: Optional reason/tag for the signal
    """

    signal_type: SignalType
    price: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    reason: Optional[str] = None

    @classmethod
    def long_entry(
        cls,
        price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> "Signal":
        """Create a long entry signal."""
        return cls(
            signal_type=SignalType.LONG_ENTRY,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    @classmethod
    def short_entry(
        cls,
        price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
        reason: Optional[str] = None,
    ) -> "Signal":
        """Create a short entry signal."""
        return cls(
            signal_type=SignalType.SHORT_ENTRY,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    @classmethod
    def close_all(cls, reason: Optional[str] = None) -> "Signal":
        """Create a close all positions signal."""
        return cls(signal_type=SignalType.CLOSE_ALL, reason=reason)


class OrderSimulator:
    """
    Simulates order matching using kline data.

    Uses kline high/low prices to simulate order fills,
    applying configurable slippage and fees.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize order simulator.

        Args:
            config: Backtest configuration
        """
        self._config = config

    @property
    def fee_rate(self) -> Decimal:
        """Get fee rate."""
        return self._config.fee_rate

    @property
    def slippage_pct(self) -> Decimal:
        """Get slippage percentage."""
        return self._config.slippage_pct

    def calculate_fill_price(
        self, target_price: Decimal, is_buy: bool, kline: Kline
    ) -> Decimal:
        """
        Calculate the actual fill price with slippage.

        Args:
            target_price: Target/signal price
            is_buy: Whether this is a buy order
            kline: Current kline for price bounds

        Returns:
            Adjusted fill price
        """
        # Apply slippage
        if is_buy:
            # Buy at slightly higher price
            slippage_amount = target_price * self.slippage_pct
            fill_price = target_price + slippage_amount
            # Can't fill above kline high
            fill_price = min(fill_price, kline.high)
        else:
            # Sell at slightly lower price
            slippage_amount = target_price * self.slippage_pct
            fill_price = target_price - slippage_amount
            # Can't fill below kline low
            fill_price = max(fill_price, kline.low)

        return fill_price

    def calculate_quantity(self, price: Decimal) -> Decimal:
        """
        Calculate position quantity based on config.

        Args:
            price: Entry price

        Returns:
            Position quantity
        """
        notional = self._config.notional_per_trade
        return notional / price

    def calculate_fee(self, price: Decimal, quantity: Decimal) -> Decimal:
        """
        Calculate trading fee.

        Args:
            price: Fill price
            quantity: Order quantity

        Returns:
            Fee amount
        """
        return price * quantity * self.fee_rate

    def create_position(
        self,
        side: str,
        kline: Kline,
        bar_index: int,
        target_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> Position:
        """
        Create a new position from a fill.

        Args:
            side: 'LONG' or 'SHORT'
            kline: Current kline
            bar_index: Current bar index
            target_price: Target entry price (or kline.close for market)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            New Position object
        """
        # Determine entry price
        if target_price is None:
            entry_price = kline.close
        else:
            is_buy = side == "LONG"
            entry_price = self.calculate_fill_price(target_price, is_buy, kline)

        # Calculate quantity and fee
        quantity = self.calculate_quantity(entry_price)
        entry_fee = self.calculate_fee(entry_price, quantity)

        return Position(
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=kline.close_time,
            entry_bar=bar_index,
            entry_fee=entry_fee,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

    def check_stop_loss(
        self, position: Position, kline: Kline
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if stop loss was hit.

        Args:
            position: Current position
            kline: Current kline

        Returns:
            Tuple of (triggered, fill_price)
        """
        if position.stop_loss is None:
            return False, None

        if position.side == "LONG":
            if kline.low <= position.stop_loss:
                # Fill at stop or worse (kline low if gapped through)
                fill_price = min(position.stop_loss, kline.open)
                fill_price = max(fill_price, kline.low)
                return True, fill_price
        else:  # SHORT
            if kline.high >= position.stop_loss:
                # Fill at stop or worse (kline high if gapped through)
                fill_price = max(position.stop_loss, kline.open)
                fill_price = min(fill_price, kline.high)
                return True, fill_price

        return False, None

    def check_take_profit(
        self, position: Position, kline: Kline
    ) -> tuple[bool, Optional[Decimal]]:
        """
        Check if take profit was hit.

        Args:
            position: Current position
            kline: Current kline

        Returns:
            Tuple of (triggered, fill_price)
        """
        if position.take_profit is None:
            return False, None

        if position.side == "LONG":
            if kline.high >= position.take_profit:
                return True, position.take_profit
        else:  # SHORT
            if kline.low <= position.take_profit:
                return True, position.take_profit

        return False, None

    def simulate_exit_order(
        self, position: Position, kline: Kline, exit_price: Optional[Decimal] = None
    ) -> tuple[Decimal, Decimal]:
        """
        Simulate an exit order fill.

        Args:
            position: Position to exit
            kline: Current kline
            exit_price: Target exit price (or kline.close for market)

        Returns:
            Tuple of (fill_price, fee)
        """
        if exit_price is None:
            fill_price = kline.close
        else:
            is_buy = position.side == "SHORT"  # Exit short = buy
            fill_price = self.calculate_fill_price(exit_price, is_buy, kline)

        fee = self.calculate_fee(fill_price, position.quantity)
        return fill_price, fee
