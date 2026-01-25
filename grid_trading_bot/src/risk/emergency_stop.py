"""
Emergency Stop.

Provides emergency stop mechanism for extreme situations.
Can be triggered manually or automatically, immediately stops all trading and closes positions.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.core import get_logger

from .models import GlobalRiskStatus

logger = get_logger(__name__)


@dataclass
class EmergencyConfig:
    """Configuration for emergency stop."""

    # Auto trigger threshold (30% loss)
    auto_trigger_loss_pct: Decimal = Decimal("0.30")

    # Whether to automatically close positions
    auto_close_positions: bool = True

    # Maximum circuit breaker triggers per day before emergency stop
    max_circuit_triggers: int = 3

    # API error duration threshold in seconds (10 minutes)
    api_error_threshold: int = 600

    # Large loss threshold for single trade (5%)
    large_loss_threshold: Decimal = Decimal("0.05")

    # Initial capital for loss calculation
    initial_capital: Decimal = Decimal("0")


@dataclass
class EmergencyStopStatus:
    """Status of emergency stop."""

    is_activated: bool = False
    activated_at: Optional[datetime] = None
    activation_reason: str = ""
    cancelled_orders: int = 0
    closed_positions: int = 0
    stopped_bots: int = 0
    errors: List[str] = field(default_factory=list)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client interface."""

    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Any]:
        """Get all open orders."""
        ...

    async def cancel_order(self, symbol: str, order_id: str, **kwargs) -> Any:
        """Cancel an order."""
        ...

    async def get_positions(self, symbol: Optional[str] = None) -> List[Any]:
        """Get all positions."""
        ...

    async def market_sell(self, symbol: str, quantity: Any, **kwargs) -> Any:
        """Place market sell order."""
        ...

    async def market_buy(self, symbol: str, quantity: Any, **kwargs) -> Any:
        """Place market buy order."""
        ...


class BotCommanderProtocol(Protocol):
    """Protocol for bot commander interface."""

    async def stop_all(self) -> List[str]:
        """Stop all running bots."""
        ...

    def get_running_bots(self) -> List[str]:
        """Get list of running bot IDs."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager interface."""

    async def send(
        self,
        title: str,
        message: str,
        level: str = "info",
        **kwargs,
    ) -> bool:
        """Send a notification."""
        ...


class EmergencyStop:
    """
    Emergency stop mechanism for extreme situations.

    Provides immediate protection by:
    - Cancelling all open orders
    - Closing all positions (optional)
    - Stopping all bots

    Example:
        >>> emergency = EmergencyStop(config, commander, exchange, notifier)
        >>> await emergency.activate("Manual emergency stop")
    """

    def __init__(
        self,
        config: EmergencyConfig,
        commander: Optional[BotCommanderProtocol] = None,
        exchange: Optional[ExchangeProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        on_activate: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize EmergencyStop.

        Args:
            config: Emergency stop configuration
            commander: Bot commander for stopping bots
            exchange: Exchange client for orders/positions
            notifier: Notification manager for alerts
            on_activate: Optional callback when activated
        """
        self._config = config
        self._commander = commander
        self._exchange = exchange
        self._notifier = notifier
        self._on_activate = on_activate

        # State tracking
        self._is_activated = False
        self._activated_at: Optional[datetime] = None
        self._activation_reason: Optional[str] = None

        # Action tracking
        self._cancel_results: List[Dict] = []
        self._close_results: List[Dict] = []
        self._stop_results: List[str] = []
        self._errors: List[str] = []

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> EmergencyConfig:
        """Get emergency configuration."""
        return self._config

    @property
    def is_activated(self) -> bool:
        """Check if emergency stop is activated."""
        return self._is_activated

    @property
    def activated_at(self) -> Optional[datetime]:
        """Get activation time."""
        return self._activated_at

    @property
    def activation_reason(self) -> Optional[str]:
        """Get activation reason."""
        return self._activation_reason

    # =========================================================================
    # Core Methods
    # =========================================================================

    async def activate(
        self,
        reason: str,
        auto_close: Optional[bool] = None,
    ) -> EmergencyStopStatus:
        """
        Activate emergency stop.

        Args:
            reason: Reason for activation
            auto_close: Whether to close positions (defaults to config)

        Returns:
            EmergencyStopStatus with results
        """
        if self._is_activated:
            logger.warning("Emergency stop already activated")
            return self.get_status()

        now = datetime.now(timezone.utc)
        self._is_activated = True
        self._activated_at = now
        self._activation_reason = reason

        # Use config default if not specified
        if auto_close is None:
            auto_close = self._config.auto_close_positions

        logger.critical(f"EMERGENCY STOP ACTIVATED: {reason}")

        # Reset tracking
        self._cancel_results = []
        self._close_results = []
        self._stop_results = []
        self._errors = []

        # ===== Execution order is important =====

        # 1. Cancel all orders first (prevent new fills)
        self._cancel_results = await self.cancel_all_orders()
        logger.info(f"Cancelled {len(self._cancel_results)} orders")

        # 2. Close all positions (if enabled)
        if auto_close:
            self._close_results = await self.close_all_positions()
            logger.info(f"Closed {len(self._close_results)} positions")

        # 3. Stop all bots
        self._stop_results = await self.stop_all_bots()
        logger.info(f"Stopped {len(self._stop_results)} bots")

        # 4. Send emergency notification
        await self._send_emergency_notification(reason)

        # 5. Call callback if set
        if self._on_activate:
            try:
                self._on_activate(reason)
            except Exception as e:
                logger.error(f"Error in on_activate callback: {e}")

        return self.get_status()

    async def cancel_all_orders(self) -> List[Dict]:
        """
        Cancel all open orders.

        Returns:
            List of cancel results
        """
        results: List[Dict] = []

        if not self._exchange:
            logger.warning("No exchange configured, skipping order cancellation")
            return results

        try:
            # Get all open orders
            open_orders = await self._exchange.get_open_orders()

            for order in open_orders:
                try:
                    await self._exchange.cancel_order(
                        symbol=order.symbol,
                        order_id=order.order_id,
                    )
                    results.append({
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "status": "cancelled",
                    })
                except Exception as e:
                    error_msg = f"Failed to cancel order {order.order_id}: {e}"
                    self._errors.append(error_msg)
                    logger.error(error_msg)
                    results.append({
                        "order_id": order.order_id,
                        "symbol": order.symbol,
                        "status": "failed",
                        "error": str(e),
                    })

        except Exception as e:
            error_msg = f"Failed to get open orders: {e}"
            self._errors.append(error_msg)
            logger.error(error_msg)

        return results

    async def close_all_positions(self) -> List[Dict]:
        """
        Close all open positions with market orders.

        Returns:
            List of close results
        """
        results: List[Dict] = []

        if not self._exchange:
            logger.warning("No exchange configured, skipping position closure")
            return results

        try:
            # Get all positions
            positions = await self._exchange.get_positions()

            for pos in positions:
                # Skip zero positions
                if pos.quantity == 0:
                    continue

                try:
                    # Determine side and quantity
                    if pos.quantity > 0:
                        # Long position - sell to close
                        order = await self._exchange.market_sell(
                            symbol=pos.symbol,
                            quantity=abs(pos.quantity),
                        )
                        side = "SELL"
                    else:
                        # Short position - buy to close
                        order = await self._exchange.market_buy(
                            symbol=pos.symbol,
                            quantity=abs(pos.quantity),
                        )
                        side = "BUY"

                    results.append({
                        "symbol": pos.symbol,
                        "side": side,
                        "quantity": str(abs(pos.quantity)),
                        "status": "closed",
                        "order_id": order.order_id,
                    })

                except Exception as e:
                    error_msg = f"Failed to close position {pos.symbol}: {e}"
                    self._errors.append(error_msg)
                    logger.error(error_msg)
                    results.append({
                        "symbol": pos.symbol,
                        "status": "failed",
                        "error": str(e),
                    })

        except Exception as e:
            error_msg = f"Failed to get positions: {e}"
            self._errors.append(error_msg)
            logger.error(error_msg)

        return results

    async def stop_all_bots(self) -> List[str]:
        """
        Stop all running bots.

        Returns:
            List of stopped bot IDs
        """
        results: List[str] = []

        if not self._commander:
            logger.warning("No commander configured, skipping bot stop")
            return results

        try:
            results = await self._commander.stop_all()
        except Exception as e:
            error_msg = f"Failed to stop bots: {e}"
            self._errors.append(error_msg)
            logger.error(error_msg)

        return results

    def check_auto_trigger(
        self,
        status: GlobalRiskStatus,
        api_error_duration: int = 0,
    ) -> Optional[str]:
        """
        Check if automatic emergency stop should be triggered.

        Args:
            status: Current global risk status
            api_error_duration: Duration of API errors in seconds

        Returns:
            Trigger reason if should trigger, None otherwise
        """
        if self._is_activated:
            return None

        reasons: List[str] = []

        # 1. Check capital loss
        if status.capital and self._config.initial_capital > 0:
            loss_pct = (
                status.capital.total_capital - self._config.initial_capital
            ) / self._config.initial_capital

            if loss_pct <= -self._config.auto_trigger_loss_pct:
                reasons.append(f"Capital loss {loss_pct:.1%}")

        # 2. Check circuit breaker trigger count
        if status.circuit_breaker:
            if status.circuit_breaker.trigger_count_today >= self._config.max_circuit_triggers:
                reasons.append(
                    f"Circuit breaker triggered {status.circuit_breaker.trigger_count_today} times today"
                )

        # 3. Check API error duration
        if api_error_duration >= self._config.api_error_threshold:
            reasons.append(f"API errors for {api_error_duration} seconds")

        if reasons:
            return "; ".join(reasons)

        return None

    async def auto_trigger_if_needed(
        self,
        status: GlobalRiskStatus,
        api_error_duration: int = 0,
    ) -> bool:
        """
        Check conditions and trigger if needed.

        Args:
            status: Current global risk status
            api_error_duration: Duration of API errors in seconds

        Returns:
            True if emergency stop was triggered
        """
        reason = self.check_auto_trigger(status, api_error_duration)

        if reason:
            await self.activate(reason, auto_close=self._config.auto_close_positions)
            return True

        return False

    def get_status(self) -> EmergencyStopStatus:
        """
        Get current emergency stop status.

        Returns:
            EmergencyStopStatus with all state information
        """
        cancelled = len([r for r in self._cancel_results if r.get("status") == "cancelled"])
        closed = len([r for r in self._close_results if r.get("status") == "closed"])

        return EmergencyStopStatus(
            is_activated=self._is_activated,
            activated_at=self._activated_at,
            activation_reason=self._activation_reason or "",
            cancelled_orders=cancelled,
            closed_positions=closed,
            stopped_bots=len(self._stop_results),
            errors=self._errors.copy(),
        )

    # =========================================================================
    # Notification Methods
    # =========================================================================

    async def _send_emergency_notification(self, reason: str) -> None:
        """Send emergency notification."""
        if not self._notifier:
            logger.debug("No notifier configured, skipping notification")
            return

        # Count results
        orders_cancelled = len([r for r in self._cancel_results if r.get("status") == "cancelled"])
        orders_failed = len([r for r in self._cancel_results if r.get("status") == "failed"])
        positions_closed = len([r for r in self._close_results if r.get("status") == "closed"])
        positions_failed = len([r for r in self._close_results if r.get("status") == "failed"])
        bots_stopped = len(self._stop_results)

        # Build message
        message = f"Reason: {reason}\n\n"
        message += "Actions taken:\n"
        message += f"- Orders cancelled: {orders_cancelled}"
        if orders_failed:
            message += f" (failed: {orders_failed})"
        message += "\n"

        message += f"- Positions closed: {positions_closed}"
        if positions_failed:
            message += f" (failed: {positions_failed})"
        message += "\n"

        message += f"- Bots stopped: {bots_stopped}\n\n"

        # Add position details
        if self._close_results:
            message += "Position details:\n"
            for result in self._close_results:
                if result.get("status") == "closed":
                    message += f"- {result['symbol']}: {result['side']} {result['quantity']} @ market\n"
                else:
                    message += f"- {result['symbol']}: FAILED - {result.get('error', 'Unknown error')}\n"
            message += "\n"

        # Add errors if any
        if self._errors:
            message += f"Errors ({len(self._errors)}):\n"
            for error in self._errors[:5]:  # Limit to first 5
                message += f"- {error}\n"
            if len(self._errors) > 5:
                message += f"... and {len(self._errors) - 5} more\n"
            message += "\n"

        message += "System is completely stopped. Manual intervention required.\n"
        message += f"Time: {self._activated_at.strftime('%Y-%m-%d %H:%M:%S UTC') if self._activated_at else 'Unknown'}"

        try:
            await self._notifier.send(
                title="SOS: EMERGENCY STOP ACTIVATED",
                message=message,
                level="critical",
            )
        except Exception as e:
            logger.error(f"Failed to send emergency notification: {e}")

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def reset(self) -> None:
        """
        Reset emergency stop state (for testing or manual recovery).

        WARNING: Use with extreme caution!
        """
        logger.warning("Emergency stop reset - manual recovery initiated")

        self._is_activated = False
        self._activated_at = None
        self._activation_reason = None
        self._cancel_results = []
        self._close_results = []
        self._stop_results = []
        self._errors = []

    def set_initial_capital(self, capital: Decimal) -> None:
        """
        Set initial capital for loss calculations.

        Args:
            capital: Initial capital value
        """
        self._config.initial_capital = capital
