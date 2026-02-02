"""
Stop Loss / Take Profit Manager.

Provides unified SLTP management including:
- State management for positions
- Stop loss and take profit checking
- Trailing stop updates
- Exchange order management
"""

import logging
from decimal import Decimal
from typing import Callable, Dict, List, Optional, Protocol

from src.risk.sltp.calculator import SLTPCalculator
from src.risk.sltp.models import (
    SLTPConfig,
    SLTPState,
    TakeProfitLevel,
    TrailingStopConfig,
)

logger = logging.getLogger(__name__)


class ExchangeAdapter(Protocol):
    """Protocol for exchange operations."""

    async def place_stop_loss(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        stop_price: Decimal,
    ) -> str:
        """Place a stop loss order. Returns order ID."""
        ...

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        price: Decimal,
    ) -> str:
        """Place a take profit order. Returns order ID."""
        ...

    async def place_trailing_stop(
        self,
        symbol: str,
        side: str,
        quantity: Decimal,
        callback_rate: Decimal,
    ) -> str:
        """Place a trailing stop order. Returns order ID."""
        ...

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> bool:
        """Cancel an order. Returns True if successful."""
        ...

    async def modify_stop_loss(
        self,
        symbol: str,
        order_id: str,
        new_stop_price: Decimal,
    ) -> bool:
        """Modify stop loss price. Returns True if successful."""
        ...


class SLTPManager:
    """
    Unified Stop Loss / Take Profit Manager.

    Supports two modes:
    - Live trading: Uses exchange orders for SL/TP
    - Backtest: Uses local price checking (no exchange orders)
    """

    def __init__(
        self,
        exchange_adapter: Optional[ExchangeAdapter] = None,
        indicator_providers: Optional[Dict[str, Callable[[str], Decimal]]] = None,
    ) -> None:
        """
        Initialize SLTP Manager.

        Args:
            exchange_adapter: Exchange adapter for placing orders (None for backtest)
            indicator_providers: Dict mapping indicator keys to provider functions
        """
        self._exchange = exchange_adapter
        self._calculator = SLTPCalculator(indicator_providers)
        self._states: Dict[str, SLTPState] = {}  # symbol -> state

    @property
    def is_live_mode(self) -> bool:
        """Check if running in live mode (with exchange adapter)."""
        return self._exchange is not None

    async def initialize_sltp(
        self,
        symbol: str,
        entry_price: Decimal,
        is_long: bool,
        quantity: Decimal,
        config: SLTPConfig,
        atr: Optional[Decimal] = None,
    ) -> SLTPState:
        """
        Initialize SLTP for a new position.

        Args:
            symbol: Trading symbol
            entry_price: Entry price
            is_long: True if long position
            quantity: Position quantity
            config: SLTP configuration
            atr: Current ATR value

        Returns:
            SLTPState object tracking this position
        """
        # Calculate initial stop loss
        stop_loss = self._calculator.calculate_stop_loss(
            config.stop_loss, entry_price, is_long, atr, symbol
        )

        # Calculate take profit levels
        tp_levels = self._calculator.calculate_take_profit(
            config.take_profit, entry_price, is_long, stop_loss, atr, symbol
        )

        # Create state object
        state = SLTPState(
            symbol=symbol,
            entry_price=entry_price,
            is_long=is_long,
            quantity=quantity,
            initial_stop_loss=stop_loss,
            current_stop_loss=stop_loss,
            take_profit_levels=tp_levels,
            highest_price=entry_price,
            lowest_price=entry_price,
        )

        # Place exchange orders if in live mode
        if self._exchange is not None and config.use_exchange_orders:
            close_side = "SELL" if is_long else "BUY"

            if config.place_sl_order and config.stop_loss.enabled:
                try:
                    order_id = await self._exchange.place_stop_loss(
                        symbol, close_side, quantity, stop_loss
                    )
                    state.sl_order_id = order_id
                    logger.info(f"Placed SL order {order_id} at {stop_loss}")
                except Exception as e:
                    logger.error(f"Failed to place SL order: {e}")

            if config.place_tp_order and config.take_profit.enabled:
                for i, level in enumerate(tp_levels):
                    try:
                        tp_quantity = quantity * level.percentage
                        order_id = await self._exchange.place_take_profit(
                            symbol, close_side, tp_quantity, level.price
                        )
                        level.order_id = order_id
                        state.tp_order_ids.append(order_id)
                        logger.info(f"Placed TP order {order_id} at {level.price}")
                    except Exception as e:
                        logger.error(f"Failed to place TP order {i}: {e}")

        self._states[symbol] = state
        logger.info(
            f"Initialized SLTP for {symbol}: SL={stop_loss}, "
            f"TP levels={[l.price for l in tp_levels]}"
        )

        return state

    def check_stop_loss(
        self,
        symbol: str,
        current_price: Decimal,
        high: Decimal,
        low: Decimal,
    ) -> bool:
        """
        Check if stop loss was hit.

        Args:
            symbol: Trading symbol
            current_price: Current price
            high: High price of the period
            low: Low price of the period

        Returns:
            True if stop loss was hit
        """
        state = self._states.get(symbol)
        if state is None or not state.is_active:
            return False

        # Update price extremes
        state.update_price_extremes(high, low)

        # Check stop loss
        hit = self._calculator.check_stop_loss_hit(
            state.current_stop_loss,
            current_price,
            high,
            low,
            state.is_long,
        )

        if hit:
            state.mark_sl_triggered()
            logger.info(
                f"Stop loss triggered for {symbol} at {state.current_stop_loss}"
            )

        return hit

    def check_take_profit(
        self,
        symbol: str,
        current_price: Decimal,
        high: Decimal,
        low: Decimal,
    ) -> List[int]:
        """
        Check which take profit levels were hit.

        Args:
            symbol: Trading symbol
            current_price: Current price
            high: High price of the period
            low: Low price of the period

        Returns:
            List of triggered level indices
        """
        state = self._states.get(symbol)
        if state is None or not state.is_active:
            return []

        # Update price extremes
        state.update_price_extremes(high, low)

        # Check take profit levels
        hit_indices = self._calculator.check_take_profit_hit(
            state.take_profit_levels,
            current_price,
            high,
            low,
            state.is_long,
        )

        # Mark triggered levels
        for idx in hit_indices:
            close_pct = state.mark_tp_triggered(idx)
            logger.info(
                f"Take profit level {idx} triggered for {symbol}, "
                f"closing {close_pct*100}% of position"
            )

        return hit_indices

    async def update_trailing_stop(
        self,
        symbol: str,
        current_price: Decimal,
        config: TrailingStopConfig,
        atr: Optional[Decimal] = None,
    ) -> Optional[Decimal]:
        """
        Update trailing stop if conditions are met.

        Args:
            symbol: Trading symbol
            current_price: Current price
            config: Trailing stop configuration
            atr: Current ATR value

        Returns:
            New stop loss price if updated, None otherwise
        """
        state = self._states.get(symbol)
        if state is None or not state.is_active:
            return None

        # Calculate new trailing stop
        new_stop = self._calculator.calculate_trailing_stop(
            config,
            current_price,
            state.highest_price,
            state.lowest_price,
            state.is_long,
            state.entry_price,
            state.current_stop_loss,
            atr,
        )

        if new_stop is not None:
            # Update state
            old_stop = state.current_stop_loss
            old_activated = state.trailing_activated
            if state.update_stop_loss(new_stop):
                state.trailing_activated = True
                state.trailing_stop_price = new_stop
                logger.info(f"Trailing stop updated for {symbol}: {new_stop}")

                # Update exchange order if in live mode
                if self._exchange is not None and state.sl_order_id:
                    try:
                        await self._exchange.modify_stop_loss(
                            symbol, state.sl_order_id, new_stop
                        )
                    except Exception as e:
                        logger.error(f"Failed to modify SL order: {e}, rolling back local state")
                        # Rollback local state to prevent desync with exchange
                        state.current_stop_loss = old_stop
                        state.trailing_stop_price = old_stop
                        state.trailing_activated = old_activated

        return new_stop

    async def cancel_all_orders(self, symbol: str) -> int:
        """
        Cancel all SLTP orders for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            Number of orders cancelled
        """
        state = self._states.get(symbol)
        if state is None or self._exchange is None:
            return 0

        cancelled = 0

        # Cancel SL order
        if state.sl_order_id:
            try:
                if await self._exchange.cancel_order(symbol, state.sl_order_id):
                    cancelled += 1
                    state.sl_order_id = None
            except Exception as e:
                logger.error(f"Failed to cancel SL order: {e}")

        # Cancel TP orders
        for order_id in state.tp_order_ids:
            try:
                if await self._exchange.cancel_order(symbol, order_id):
                    cancelled += 1
            except Exception as e:
                logger.error(f"Failed to cancel TP order {order_id}: {e}")
        state.tp_order_ids = []

        # Cancel trailing order
        if state.trailing_order_id:
            try:
                if await self._exchange.cancel_order(symbol, state.trailing_order_id):
                    cancelled += 1
                    state.trailing_order_id = None
            except Exception as e:
                logger.error(f"Failed to cancel trailing order: {e}")

        logger.info(f"Cancelled {cancelled} orders for {symbol}")
        return cancelled

    def get_state(self, symbol: str) -> Optional[SLTPState]:
        """Get SLTP state for a symbol."""
        return self._states.get(symbol)

    def get_effective_stop_loss(self, symbol: str) -> Optional[Decimal]:
        """Get effective (current) stop loss price for a symbol."""
        state = self._states.get(symbol)
        if state is None:
            return None
        return state.current_stop_loss

    def get_untriggered_tp_levels(self, symbol: str) -> List[TakeProfitLevel]:
        """Get untriggered take profit levels for a symbol."""
        state = self._states.get(symbol)
        if state is None:
            return []
        return [tp for tp in state.take_profit_levels if not tp.triggered]

    def remove_state(self, symbol: str) -> bool:
        """Remove SLTP state for a symbol (when position is fully closed)."""
        if symbol in self._states:
            del self._states[symbol]
            logger.info(f"Removed SLTP state for {symbol}")
            return True
        return False

    def get_all_active_symbols(self) -> List[str]:
        """Get all symbols with active SLTP states."""
        return [sym for sym, state in self._states.items() if state.is_active]

    def process_price_update(
        self,
        symbol: str,
        current_price: Decimal,
        high: Decimal,
        low: Decimal,
        config: SLTPConfig,
        atr: Optional[Decimal] = None,
    ) -> Dict[str, any]:
        """
        Process a price update for a symbol (for backtest mode).

        This is a convenience method that checks SL, TP, and updates trailing stop.

        Args:
            symbol: Trading symbol
            current_price: Current price
            high: High price of the period
            low: Low price of the period
            config: SLTP configuration
            atr: Current ATR value

        Returns:
            Dict with check results:
            - sl_hit: bool
            - tp_levels_hit: List[int]
            - new_trailing_stop: Optional[Decimal]
        """
        state = self._states.get(symbol)
        if state is None:
            return {"sl_hit": False, "tp_levels_hit": [], "new_trailing_stop": None}

        # Update price extremes first
        state.update_price_extremes(high, low)

        # Check stop loss
        sl_hit = self.check_stop_loss(symbol, current_price, high, low)

        # Check take profit (only if SL not hit)
        tp_hit = []
        if not sl_hit:
            tp_hit = self.check_take_profit(symbol, current_price, high, low)

        # Update trailing stop (only if still active)
        new_trailing = None
        if state.is_active and config.trailing_stop.enabled:
            new_trailing = self._calculator.calculate_trailing_stop(
                config.trailing_stop,
                current_price,
                state.highest_price,
                state.lowest_price,
                state.is_long,
                state.entry_price,
                state.current_stop_loss,
                atr,
            )
            if new_trailing is not None:
                state.update_stop_loss(new_trailing)
                state.trailing_activated = True
                state.trailing_stop_price = new_trailing

        return {
            "sl_hit": sl_hit,
            "tp_levels_hit": tp_hit,
            "new_trailing_stop": new_trailing,
        }
