"""
Order Simulation Module.

Handles order matching simulation using kline high/low prices.
"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING, Optional

from ..core.models import Kline
from .config import BacktestConfig
from .fees import FeeCalculator, FeeContext
from .position import Position
from .slippage import SlippageContext, SlippageModel

if TYPE_CHECKING:
    pass


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
    applying configurable slippage and fees via pluggable models.
    """

    def __init__(self, config: BacktestConfig) -> None:
        """
        Initialize order simulator.

        Args:
            config: Backtest configuration
        """
        self._config = config
        # Create models from config factory methods
        self._slippage_model: SlippageModel = config.create_slippage_model()
        self._fee_calculator: FeeCalculator = config.create_fee_calculator()
        # Context for advanced slippage/fee calculation
        self._recent_klines: list[Kline] = []
        self._cumulative_volume: Decimal = Decimal("0")

    @property
    def fee_rate(self) -> Decimal:
        """Get base fee rate (for backward compatibility)."""
        return self._fee_calculator.base_rate

    @property
    def slippage_pct(self) -> Decimal:
        """Get slippage percentage (for backward compatibility)."""
        return self._config.slippage_pct

    @property
    def slippage_model(self) -> SlippageModel:
        """Get the slippage model."""
        return self._slippage_model

    @property
    def fee_calculator(self) -> FeeCalculator:
        """Get the fee calculator."""
        return self._fee_calculator

    def update_context(
        self,
        kline: Kline,
        max_klines: int = 20,
    ) -> None:
        """
        Update context with recent kline data for advanced models.

        Args:
            kline: Current kline to add
            max_klines: Maximum klines to keep for ATR calculation
        """
        self._recent_klines.append(kline)
        if len(self._recent_klines) > max_klines:
            self._recent_klines.pop(0)

    def update_cumulative_volume(self, volume: Decimal) -> None:
        """
        Update cumulative trading volume for tiered fee calculation.

        Args:
            volume: Volume to add
        """
        self._cumulative_volume += volume

    def calculate_fill_price(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        order_size: Optional[Decimal] = None,
        avg_volume: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate the actual fill price with slippage.

        Args:
            target_price: Target/signal price
            is_buy: Whether this is a buy order
            kline: Current kline for price bounds
            order_size: Order size for market impact calculation
            avg_volume: Average volume for market impact calculation

        Returns:
            Adjusted fill price
        """
        # Build slippage context
        context = SlippageContext(
            order_size=order_size if order_size else Decimal("0"),
            avg_volume=avg_volume,
            recent_klines=self._recent_klines if self._recent_klines else None,
        )

        # Use slippage model to calculate fill price
        return self._slippage_model.apply_slippage(target_price, is_buy, kline, context)

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

    def calculate_fee(
        self,
        price: Decimal,
        quantity: Decimal,
        is_maker: bool = False,
    ) -> Decimal:
        """
        Calculate trading fee.

        Args:
            price: Fill price
            quantity: Order quantity
            is_maker: Whether this is a maker order (limit order)

        Returns:
            Fee amount
        """
        context = FeeContext(
            is_maker=is_maker,
            cumulative_volume=self._cumulative_volume,
        )
        return self._fee_calculator.calculate_fee(price, quantity, context)

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
