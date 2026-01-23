"""
SLTP Exchange Adapter.

Provides exchange-specific implementations for SLTP order operations.
Uses Protocol for interface definition to allow easy mocking and testing.
"""

import logging
from decimal import Decimal
from typing import Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class ExchangeClient(Protocol):
    """Protocol defining required exchange client methods."""

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        reduce_only: bool = False,
        **kwargs,
    ) -> dict:
        """Place an order on the exchange."""
        ...

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an order."""
        ...

    async def modify_order(
        self,
        symbol: str,
        order_id: str,
        quantity: Optional[Decimal] = None,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
    ) -> dict:
        """Modify an existing order."""
        ...


class SLTPExchangeAdapter:
    """
    Adapter for exchange SLTP operations.

    Provides a unified interface for placing and managing SLTP orders
    across different exchange implementations.
    """

    def __init__(self, client: ExchangeClient) -> None:
        """
        Initialize adapter.

        Args:
            client: Exchange client implementing ExchangeClient protocol
        """
        self._client = client

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
    ) -> str:
        """
        Place a stop loss order.

        Args:
            symbol: Trading symbol
            side: Order side ("BUY" or "SELL")
            quantity: Order quantity
            stop_price: Stop trigger price

        Returns:
            Order ID
        """
        result = await self._client.place_order(
            symbol=symbol,
            side=side,
            order_type="STOP_MARKET",
            quantity=quantity,
            stop_price=stop_price,
            reduce_only=True,
            close_position=True,
        )
        order_id = result.get("orderId", result.get("order_id", ""))
        logger.info(
            f"Placed stop loss order: {symbol} {side} {quantity} @ {stop_price}, "
            f"order_id={order_id}"
        )
        return str(order_id)

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> str:
        """
        Place a take profit order.

        Args:
            symbol: Trading symbol
            side: Order side ("BUY" or "SELL")
            quantity: Order quantity
            price: Take profit price

        Returns:
            Order ID
        """
        result = await self._client.place_order(
            symbol=symbol,
            side=side,
            order_type="TAKE_PROFIT_MARKET",
            quantity=quantity,
            stop_price=price,
            reduce_only=True,
        )
        order_id = result.get("orderId", result.get("order_id", ""))
        logger.info(
            f"Placed take profit order: {symbol} {side} {quantity} @ {price}, "
            f"order_id={order_id}"
        )
        return str(order_id)

    async def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        callback_rate: Decimal,
    ) -> str:
        """
        Place a trailing stop order.

        Args:
            symbol: Trading symbol
            side: Order side ("BUY" or "SELL")
            quantity: Order quantity
            callback_rate: Callback rate (e.g., 0.01 for 1%)

        Returns:
            Order ID
        """
        # Convert callback rate to percentage (Binance uses 1-5 for 1%-5%)
        callback_pct = callback_rate * Decimal("100")
        if callback_pct < Decimal("0.1"):
            callback_pct = Decimal("0.1")  # Minimum 0.1%
        if callback_pct > Decimal("5"):
            callback_pct = Decimal("5")  # Maximum 5%

        result = await self._client.place_order(
            symbol=symbol,
            side=side,
            order_type="TRAILING_STOP_MARKET",
            quantity=quantity,
            callback_rate=callback_pct,
            reduce_only=True,
        )
        order_id = result.get("orderId", result.get("order_id", ""))
        logger.info(
            f"Placed trailing stop order: {symbol} {side} {quantity} "
            f"callback={callback_pct}%, order_id={order_id}"
        )
        return str(order_id)

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        try:
            await self._client.cancel_order(symbol, order_id)
            logger.info(f"Cancelled order: {symbol} {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {e}")
            return False

    async def modify_stop_loss(
        self,
        symbol: str,
        order_id: str,
        new_stop_price: Decimal,
    ) -> bool:
        """
        Modify stop loss price.

        Note: Some exchanges don't support order modification.
        In that case, cancel and replace.

        Args:
            symbol: Trading symbol
            order_id: Order ID to modify
            new_stop_price: New stop price

        Returns:
            True if modified successfully
        """
        try:
            await self._client.modify_order(
                symbol=symbol,
                order_id=order_id,
                stop_price=new_stop_price,
            )
            logger.info(f"Modified stop loss: {symbol} {order_id} -> {new_stop_price}")
            return True
        except NotImplementedError:
            # Exchange doesn't support modification, need cancel and replace
            logger.warning(
                f"Exchange doesn't support order modification for {order_id}"
            )
            return False
        except Exception as e:
            logger.error(f"Failed to modify stop loss {order_id}: {e}")
            return False


class MockExchangeAdapter:
    """
    Mock exchange adapter for testing and backtesting.

    Simulates exchange order operations without making real API calls.
    """

    def __init__(self) -> None:
        """Initialize mock adapter."""
        self._order_counter = 0
        self._orders: dict = {}  # order_id -> order info

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
    ) -> str:
        """Place a mock stop loss order."""
        self._order_counter += 1
        order_id = f"MOCK_SL_{self._order_counter}"
        self._orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "type": "STOP_MARKET",
            "quantity": quantity,
            "stop_price": stop_price,
            "status": "NEW",
        }
        return order_id

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> str:
        """Place a mock take profit order."""
        self._order_counter += 1
        order_id = f"MOCK_TP_{self._order_counter}"
        self._orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "type": "TAKE_PROFIT_MARKET",
            "quantity": quantity,
            "price": price,
            "status": "NEW",
        }
        return order_id

    async def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        callback_rate: Decimal,
    ) -> str:
        """Place a mock trailing stop order."""
        self._order_counter += 1
        order_id = f"MOCK_TS_{self._order_counter}"
        self._orders[order_id] = {
            "symbol": symbol,
            "side": side,
            "type": "TRAILING_STOP_MARKET",
            "quantity": quantity,
            "callback_rate": callback_rate,
            "status": "NEW",
        }
        return order_id

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel a mock order."""
        if order_id in self._orders:
            self._orders[order_id]["status"] = "CANCELED"
            return True
        return False

    async def modify_stop_loss(
        self,
        symbol: str,
        order_id: str,
        new_stop_price: Decimal,
    ) -> bool:
        """Modify a mock stop loss order."""
        if order_id in self._orders:
            self._orders[order_id]["stop_price"] = new_stop_price
            return True
        return False

    def get_order(self, order_id: str) -> Optional[dict]:
        """Get order info by ID."""
        return self._orders.get(order_id)

    def get_all_orders(self) -> dict:
        """Get all orders."""
        return self._orders.copy()

    def reset(self) -> None:
        """Reset all orders."""
        self._order_counter = 0
        self._orders = {}
