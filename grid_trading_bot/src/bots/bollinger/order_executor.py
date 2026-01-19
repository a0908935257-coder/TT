"""
Order Executor for Bollinger Bot.

Handles order placement, cancellation, and fill callbacks.
Optimizes for lower fees using limit orders where possible.

Conforms to Prompt 68 specification.

Order Strategy:
    - Entry: Limit order (Maker) - 0.02% fee
    - Take Profit: Limit order (Maker) - 0.02% fee
    - Stop Loss: Stop Market order (Taker) - 0.05% fee (guarantees execution)

Normal trade cost: 0.04%
Stop loss trade cost: 0.07%
"""

from decimal import Decimal
from typing import Any, Dict, Optional, Protocol

from src.core import get_logger

from .models import (
    BollingerConfig,
    Position,
    PositionSide,
    Signal,
    SignalType,
)

logger = get_logger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class OrderNotFoundError(Exception):
    """Raised when order is not found on exchange."""
    pass


# =============================================================================
# Protocols
# =============================================================================


class OrderProtocol(Protocol):
    """Protocol for order data."""

    @property
    def order_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def side(self) -> str: ...

    @property
    def avg_price(self) -> Decimal: ...

    @property
    def filled_quantity(self) -> Decimal: ...


class ExchangeProtocol(Protocol):
    """Protocol for exchange client."""

    async def futures_create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Any: ...

    async def futures_cancel_order(self, symbol: str, order_id: str) -> Any: ...

    async def futures_get_order(self, symbol: str, order_id: str) -> Any: ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def send_trade_notification(
        self,
        title: str,
        side: str,
        price: Decimal,
        quantity: Decimal,
    ) -> None: ...


# =============================================================================
# Order Executor
# =============================================================================


class OrderExecutor:
    """
    Order executor for Bollinger Bot.

    Handles order placement and management with fee optimization.
    Uses limit orders for entry and take profit (Maker fee),
    and stop market orders for stop loss (guaranteed execution).

    Example:
        >>> executor = OrderExecutor(config, exchange, notifier)
        >>> order_id = await executor.place_entry_order(signal, quantity)
        >>> await executor.place_exit_orders(position)
    """

    def __init__(
        self,
        config: BollingerConfig,
        exchange: ExchangeProtocol,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize OrderExecutor.

        Args:
            config: BollingerConfig with trading parameters
            exchange: Exchange client for API calls
            notifier: Optional notification manager
        """
        self._config = config
        self._exchange = exchange
        self._notifier = notifier

        # Order tracking
        self._pending_entry_order: Optional[str] = None
        self._take_profit_order: Optional[str] = None
        self._stop_loss_order: Optional[str] = None

        # Entry timeout
        self._entry_timeout_bars: int = 3

    @property
    def pending_entry_order(self) -> Optional[str]:
        """Get pending entry order ID."""
        return self._pending_entry_order

    @property
    def take_profit_order(self) -> Optional[str]:
        """Get take profit order ID."""
        return self._take_profit_order

    @property
    def stop_loss_order(self) -> Optional[str]:
        """Get stop loss order ID."""
        return self._stop_loss_order

    @property
    def has_pending_entry(self) -> bool:
        """Check if there is a pending entry order."""
        return self._pending_entry_order is not None

    # =========================================================================
    # Entry Order
    # =========================================================================

    async def place_entry_order(
        self,
        signal: Signal,
        quantity: Decimal,
    ) -> str:
        """
        Place entry limit order (Maker).

        Args:
            signal: Trading signal with entry price
            quantity: Position quantity

        Returns:
            Order ID
        """
        symbol = self._config.symbol

        # Determine side
        if signal.signal_type == SignalType.LONG:
            side = "BUY"
        else:
            side = "SELL"

        # Place limit order
        order = await self._exchange.futures_create_order(
            symbol=symbol,
            side=side,
            order_type="LIMIT",
            quantity=quantity,
            price=signal.entry_price,
            time_in_force="GTC",
        )

        self._pending_entry_order = order.order_id

        logger.info(
            f"Entry order placed: {side} {quantity} @ {signal.entry_price}, "
            f"order_id={order.order_id}"
        )

        return order.order_id

    async def cancel_entry_order(self) -> bool:
        """
        Cancel pending entry order.

        Returns:
            True if cancelled, False if not found
        """
        if self._pending_entry_order is None:
            return False

        try:
            await self._exchange.futures_cancel_order(
                symbol=self._config.symbol,
                order_id=self._pending_entry_order,
            )
            logger.info(f"Entry order cancelled: {self._pending_entry_order}")
            self._pending_entry_order = None
            return True

        except Exception as e:
            # Order may already be filled or cancelled
            logger.debug(f"Cancel entry order: {e}")
            self._pending_entry_order = None
            return False

    # =========================================================================
    # Exit Orders
    # =========================================================================

    async def place_exit_orders(self, position: Position) -> None:
        """
        Place take profit and stop loss orders.

        Args:
            position: Current position with TP/SL prices
        """
        symbol = self._config.symbol
        quantity = position.quantity

        # Determine close side (opposite of position)
        if position.side == PositionSide.LONG:
            close_side = "SELL"
        else:
            close_side = "BUY"

        # 1. Place take profit limit order (Maker)
        if position.take_profit_price is not None:
            tp_order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="LIMIT",
                quantity=quantity,
                price=position.take_profit_price,
                time_in_force="GTC",
                reduce_only=True,
            )
            self._take_profit_order = tp_order.order_id
            logger.info(
                f"Take profit order placed: {close_side} {quantity} @ {position.take_profit_price}"
            )

        # 2. Place stop loss order (Stop Market - Taker)
        if position.stop_loss_price is not None:
            sl_order = await self._exchange.futures_create_order(
                symbol=symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=quantity,
                stop_price=position.stop_loss_price,
                reduce_only=True,
            )
            self._stop_loss_order = sl_order.order_id
            logger.info(
                f"Stop loss order placed: {close_side} {quantity}, trigger @ {position.stop_loss_price}"
            )

    async def cancel_exit_orders(self) -> None:
        """Cancel all exit orders (TP and SL)."""
        # Cancel take profit
        if self._take_profit_order is not None:
            try:
                await self._exchange.futures_cancel_order(
                    symbol=self._config.symbol,
                    order_id=self._take_profit_order,
                )
                logger.info(f"Take profit order cancelled: {self._take_profit_order}")
            except Exception as e:
                logger.debug(f"Cancel TP order: {e}")
            self._take_profit_order = None

        # Cancel stop loss
        if self._stop_loss_order is not None:
            try:
                await self._exchange.futures_cancel_order(
                    symbol=self._config.symbol,
                    order_id=self._stop_loss_order,
                )
                logger.info(f"Stop loss order cancelled: {self._stop_loss_order}")
            except Exception as e:
                logger.debug(f"Cancel SL order: {e}")
            self._stop_loss_order = None

    # =========================================================================
    # Market Close
    # =========================================================================

    async def close_position_market(self, position: Position) -> Any:
        """
        Close position with market order (for timeout or emergency).

        Args:
            position: Position to close

        Returns:
            Order object
        """
        # First cancel existing exit orders
        await self.cancel_exit_orders()

        # Determine close side
        if position.side == PositionSide.LONG:
            close_side = "SELL"
        else:
            close_side = "BUY"

        # Market close
        order = await self._exchange.futures_create_order(
            symbol=self._config.symbol,
            side=close_side,
            order_type="MARKET",
            quantity=position.quantity,
            reduce_only=True,
        )

        logger.info(f"Market close: {close_side} {position.quantity}")

        return order

    # =========================================================================
    # Order Status
    # =========================================================================

    async def check_order_status(self) -> Dict[str, Any]:
        """
        Check status of all tracked orders.

        Returns:
            Dictionary with order statuses
        """
        result = {
            "entry_order": None,
            "take_profit_order": None,
            "stop_loss_order": None,
        }

        # Check entry order
        if self._pending_entry_order:
            try:
                order = await self._exchange.futures_get_order(
                    symbol=self._config.symbol,
                    order_id=self._pending_entry_order,
                )
                result["entry_order"] = {
                    "id": order.order_id,
                    "status": order.status,
                    "filled": str(order.filled_quantity),
                }
            except Exception as e:
                logger.debug(f"Check entry order: {e}")

        # Check take profit order
        if self._take_profit_order:
            try:
                order = await self._exchange.futures_get_order(
                    symbol=self._config.symbol,
                    order_id=self._take_profit_order,
                )
                result["take_profit_order"] = {
                    "id": order.order_id,
                    "status": order.status,
                    "filled": str(order.filled_quantity),
                }
            except Exception as e:
                logger.debug(f"Check TP order: {e}")

        # Check stop loss order
        if self._stop_loss_order:
            try:
                order = await self._exchange.futures_get_order(
                    symbol=self._config.symbol,
                    order_id=self._stop_loss_order,
                )
                result["stop_loss_order"] = {
                    "id": order.order_id,
                    "status": order.status,
                }
            except Exception as e:
                logger.debug(f"Check SL order: {e}")

        return result

    # =========================================================================
    # Order Fill Callback
    # =========================================================================

    async def on_order_filled(self, order: OrderProtocol) -> Optional[str]:
        """
        Handle order fill callback.

        Args:
            order: Filled order data

        Returns:
            Fill type: "entry_filled", "take_profit_filled", "stop_loss_filled", or None
        """
        order_id = order.order_id

        if order_id == self._pending_entry_order:
            # Entry order filled
            self._pending_entry_order = None
            logger.info(f"Entry filled: {order.filled_quantity} @ {order.avg_price}")

            # Send notification
            if self._notifier:
                await self._notifier.send_trade_notification(
                    title="📈 開倉成交",
                    side=order.side,
                    price=order.avg_price,
                    quantity=order.filled_quantity,
                )

            return "entry_filled"

        elif order_id == self._take_profit_order:
            # Take profit filled
            self._take_profit_order = None

            # Cancel stop loss
            await self._cancel_order_safe(self._stop_loss_order)
            self._stop_loss_order = None

            logger.info(f"Take profit filled: {order.filled_quantity} @ {order.avg_price}")

            return "take_profit_filled"

        elif order_id == self._stop_loss_order:
            # Stop loss filled
            self._stop_loss_order = None

            # Cancel take profit
            await self._cancel_order_safe(self._take_profit_order)
            self._take_profit_order = None

            logger.info(f"Stop loss filled: {order.filled_quantity} @ {order.avg_price}")

            return "stop_loss_filled"

        return None

    async def _cancel_order_safe(self, order_id: Optional[str]) -> None:
        """Cancel order without raising exception."""
        if order_id is None:
            return

        try:
            await self._exchange.futures_cancel_order(
                symbol=self._config.symbol,
                order_id=order_id,
            )
        except Exception as e:
            logger.debug(f"Cancel order {order_id}: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def clear_all_orders(self) -> None:
        """Clear all tracked order IDs."""
        self._pending_entry_order = None
        self._take_profit_order = None
        self._stop_loss_order = None

    def get_entry_timeout_bars(self) -> int:
        """Get entry order timeout in bars."""
        return self._entry_timeout_bars

    def set_entry_timeout_bars(self, bars: int) -> None:
        """Set entry order timeout in bars."""
        self._entry_timeout_bars = bars
