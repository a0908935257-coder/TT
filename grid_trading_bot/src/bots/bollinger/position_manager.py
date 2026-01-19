"""
Position Manager for Bollinger Bot.

Manages futures positions including opening, closing, and margin monitoring.

Conforms to Prompt 67 specification.

Features:
    - Leverage configuration
    - Isolated margin mode (recommended)
    - Position size calculation based on available balance
    - Margin monitoring
    - Exchange synchronization
"""

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional, Protocol

from src.core import get_logger

from .models import (
    BollingerConfig,
    Position,
    PositionSide,
    Signal,
    SignalType,
    TradeRecord,
)

logger = get_logger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class PositionExistsError(Exception):
    """Raised when trying to open position but one already exists."""
    pass


class NoPositionError(Exception):
    """Raised when trying to close position but none exists."""
    pass


# =============================================================================
# Protocols
# =============================================================================


class FuturesAPIProtocol(Protocol):
    """Protocol for futures API."""

    async def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]: ...

    async def set_margin_type(self, symbol: str, margin_type: str) -> Dict[str, Any]: ...

    async def get_account(self) -> Any: ...

    async def get_positions(self, symbol: str) -> list: ...


class ExchangeProtocol(Protocol):
    """Protocol for exchange client."""

    @property
    def futures(self) -> FuturesAPIProtocol: ...

    async def get_ticker(self, symbol: str) -> Any: ...


class DataManagerProtocol(Protocol):
    """Protocol for data manager."""

    async def save_trade(self, trade: TradeRecord) -> None: ...


# =============================================================================
# Position Manager
# =============================================================================


class PositionManager:
    """
    Futures position manager for Bollinger Bot.

    Handles position lifecycle including opening, closing, and monitoring.
    Uses isolated margin mode for better risk control.

    Example:
        >>> manager = PositionManager(config, exchange, data_manager)
        >>> await manager.initialize()
        >>> position = await manager.open_position(signal)
        >>> record = await manager.close_position("止盈")
    """

    def __init__(
        self,
        config: BollingerConfig,
        exchange: ExchangeProtocol,
        data_manager: Optional[DataManagerProtocol] = None,
    ):
        """
        Initialize PositionManager.

        Args:
            config: BollingerConfig with trading parameters
            exchange: Exchange client for API calls
            data_manager: Optional data manager for persistence
        """
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._current_position: Optional[Position] = None
        self._leverage = config.leverage
        self._initialized = False

    @property
    def config(self) -> BollingerConfig:
        """Get configuration."""
        return self._config

    @property
    def leverage(self) -> int:
        """Get leverage."""
        return self._leverage

    @property
    def has_position(self) -> bool:
        """Check if there is an open position."""
        return self._current_position is not None

    def get_position(self) -> Optional[Position]:
        """Get current position."""
        return self._current_position

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> None:
        """
        Initialize position manager.

        Sets leverage and margin type on the exchange.
        """
        symbol = self._config.symbol

        # 1. Set leverage
        try:
            await self._exchange.futures.set_leverage(
                symbol=symbol,
                leverage=self._leverage,
            )
            logger.info(f"Leverage set to {self._leverage}x for {symbol}")
        except Exception as e:
            logger.warning(f"Failed to set leverage: {e}")

        # 2. Set margin type to ISOLATED
        try:
            await self._exchange.futures.set_margin_type(
                symbol=symbol,
                margin_type="ISOLATED",
            )
            logger.info(f"Margin type set to ISOLATED for {symbol}")
        except Exception as e:
            # May fail if already set to ISOLATED
            logger.debug(f"Set margin type: {e}")

        # 3. Sync with exchange to get existing positions
        await self.sync_with_exchange()

        self._initialized = True
        logger.info(f"PositionManager initialized for {symbol}")

    # =========================================================================
    # Position Size Calculation
    # =========================================================================

    async def calculate_position_size(self, entry_price: Decimal) -> Decimal:
        """
        Calculate position size based on available balance.

        Formula:
            position_value = available_balance × position_size_pct
            notional_value = position_value × leverage
            quantity = notional_value / entry_price

        Args:
            entry_price: Expected entry price

        Returns:
            Position quantity
        """
        if not isinstance(entry_price, Decimal):
            entry_price = Decimal(str(entry_price))

        # Get account balance
        account = await self._exchange.futures.get_account()
        available_balance = Decimal(str(account.available_balance))

        # Calculate position value
        position_value = available_balance * self._config.position_size_pct

        # Calculate notional value with leverage
        notional_value = position_value * Decimal(self._leverage)

        # Calculate quantity
        quantity = notional_value / entry_price

        # Adjust precision (simplified - should use exchange info)
        quantity = self._adjust_quantity_precision(quantity)

        logger.info(
            f"Position size: balance={available_balance}, "
            f"use={position_value} ({self._config.position_size_pct:.0%}), "
            f"leverage={self._leverage}x, quantity={quantity}"
        )

        return quantity

    def _adjust_quantity_precision(self, quantity: Decimal) -> Decimal:
        """
        Adjust quantity to exchange precision.

        Args:
            quantity: Raw quantity

        Returns:
            Adjusted quantity
        """
        # Simplified: round to 3 decimal places
        # In production, should use exchange's lot size rules
        return quantity.quantize(Decimal("0.001"))

    # =========================================================================
    # Open Position
    # =========================================================================

    async def open_position(self, signal: Signal) -> Position:
        """
        Open a new position based on signal.

        Args:
            signal: Trading signal with entry, TP, SL

        Returns:
            New Position object

        Raises:
            PositionExistsError: If position already exists
        """
        if self._current_position is not None:
            raise PositionExistsError("Position already exists, cannot open new one")

        # Calculate position size
        quantity = await self.calculate_position_size(signal.entry_price)

        # Determine position side
        if signal.signal_type == SignalType.LONG:
            position_side = PositionSide.LONG
        else:
            position_side = PositionSide.SHORT

        # Create position object
        position = Position(
            symbol=self._config.symbol,
            side=position_side,
            entry_price=signal.entry_price,
            quantity=quantity,
            leverage=self._leverage,
            unrealized_pnl=Decimal("0"),
            entry_time=datetime.now(timezone.utc),
            entry_bar=0,  # Set by bot
            take_profit_price=signal.take_profit,
            stop_loss_price=signal.stop_loss,
        )

        self._current_position = position

        logger.info(
            f"Position opened: {position_side.value} {quantity} {self._config.symbol} "
            f"@ {signal.entry_price}, TP={signal.take_profit}, SL={signal.stop_loss}"
        )

        return position

    # =========================================================================
    # Close Position
    # =========================================================================

    async def close_position(self, reason: str) -> TradeRecord:
        """
        Close current position.

        Args:
            reason: Reason for closing (take_profit, stop_loss, timeout, etc.)

        Returns:
            TradeRecord with trade details

        Raises:
            NoPositionError: If no position to close
        """
        if self._current_position is None:
            raise NoPositionError("No position to close")

        pos = self._current_position

        # Get current price for PnL calculation
        ticker = await self._exchange.get_ticker(pos.symbol)
        exit_price = Decimal(str(ticker.last_price))

        # Calculate PnL
        if pos.side == PositionSide.LONG:
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        # Calculate PnL percentage (based on margin)
        margin = pos.entry_price * pos.quantity / Decimal(self._leverage)
        pnl_pct = pnl / margin if margin > 0 else Decimal("0")

        # Create trade record
        record = TradeRecord(
            trade_id=str(uuid.uuid4()),
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=pos.quantity,
            pnl=pnl,
            pnl_pct=pnl_pct,
            fee=Decimal("0"),  # Updated by OrderExecutor
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            exit_reason=reason,
            hold_bars=0,  # Set by bot
        )

        # Clear position
        self._current_position = None

        # Save to database
        await self._save_trade_record(record)

        logger.info(
            f"Position closed: {pos.side.value} {pos.quantity} {pos.symbol}, "
            f"PnL={pnl:+.4f} ({pnl_pct:+.2%}), reason={reason}"
        )

        return record

    async def _save_trade_record(self, record: TradeRecord) -> None:
        """Save trade record to database."""
        if self._data_manager is not None:
            try:
                await self._data_manager.save_trade(record)
            except Exception as e:
                logger.warning(f"Failed to save trade record: {e}")

    # =========================================================================
    # Margin Monitoring
    # =========================================================================

    async def check_margin(self) -> Dict[str, Any]:
        """
        Check margin status for current position.

        Returns:
            Dictionary with margin information
        """
        if self._current_position is None:
            return {"status": "no_position"}

        try:
            positions = await self._exchange.futures.get_positions(
                symbol=self._config.symbol
            )

            for pos in positions:
                if abs(Decimal(str(pos.quantity))) > 0:
                    margin = Decimal(str(pos.margin))
                    notional = Decimal(str(pos.notional_value))
                    margin_ratio = margin / notional if notional > 0 else Decimal("0")

                    status = "ok" if margin_ratio > Decimal("0.3") else "warning"

                    return {
                        "status": status,
                        "margin": str(margin),
                        "margin_ratio": str(margin_ratio),
                        "liquidation_price": str(pos.liquidation_price),
                        "unrealized_pnl": str(pos.unrealized_pnl),
                    }

            return {"status": "no_position"}

        except Exception as e:
            logger.warning(f"Failed to check margin: {e}")
            return {"status": "error", "error": str(e)}

    # =========================================================================
    # Exchange Sync
    # =========================================================================

    async def sync_with_exchange(self) -> None:
        """
        Synchronize position state with exchange.

        Useful for recovery after restart.
        """
        try:
            positions = await self._exchange.futures.get_positions(
                symbol=self._config.symbol
            )

            for pos in positions:
                qty = Decimal(str(pos.quantity))
                if qty != 0:
                    # Found existing position
                    self._current_position = Position(
                        symbol=pos.symbol,
                        side=PositionSide.LONG if qty > 0 else PositionSide.SHORT,
                        entry_price=Decimal(str(pos.entry_price)),
                        quantity=abs(qty),
                        leverage=int(pos.leverage),
                        unrealized_pnl=Decimal(str(pos.unrealized_pnl)),
                        entry_time=None,  # Unknown from exchange
                        entry_bar=0,
                        take_profit_price=None,
                        stop_loss_price=None,
                    )
                    logger.info(f"Synced position: {self._current_position}")
                    return

            # No position found
            self._current_position = None
            logger.info("No existing position found")

        except Exception as e:
            logger.warning(f"Failed to sync with exchange: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def update_position_bar(self, bar: int) -> None:
        """Update position entry bar number."""
        if self._current_position is not None:
            self._current_position.entry_bar = bar

    def update_position_prices(
        self,
        entry_price: Optional[Decimal] = None,
        stop_loss: Optional[Decimal] = None,
        take_profit: Optional[Decimal] = None,
    ) -> None:
        """Update position prices (e.g., after fill)."""
        if self._current_position is None:
            return

        if entry_price is not None:
            self._current_position.entry_price = entry_price
        if stop_loss is not None:
            self._current_position.stop_loss_price = stop_loss
        if take_profit is not None:
            self._current_position.take_profit_price = take_profit

    def clear_position(self) -> None:
        """Clear position without creating trade record."""
        self._current_position = None
