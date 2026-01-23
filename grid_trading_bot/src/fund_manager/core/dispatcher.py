"""
Fund Dispatcher.

Handles dispatching fund allocation updates to trading bots.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol

from src.core import get_logger
from src.ipc.messages import Command, CommandType

from ..models.records import AllocationRecord

logger = get_logger(__name__)


class BotProtocol(Protocol):
    """Protocol for bot interface."""

    @property
    def bot_id(self) -> str:
        """Get bot ID."""
        ...

    async def update_capital(self, new_max_capital: Decimal) -> bool:
        """Update maximum capital allocation."""
        ...


class BotRegistryProtocol(Protocol):
    """Protocol for bot registry interface."""

    def get(self, bot_id: str) -> Optional[Any]:
        """Get bot info by ID."""
        ...

    def get_instance(self, bot_id: str) -> Optional[BotProtocol]:
        """Get bot instance by ID."""
        ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager interface."""

    async def notify_fund_allocated(
        self,
        bot_id: str,
        amount: Decimal,
        total_allocated: Decimal,
    ) -> None:
        """Notify about fund allocation."""
        ...


class Dispatcher:
    """
    Fund dispatcher for sending allocation updates to bots.

    Handles the actual dispatch of fund allocations to running bots,
    either through direct method calls or IPC messages.

    Example:
        >>> dispatcher = Dispatcher(registry, notifier)
        >>> result = await dispatcher.dispatch(allocations)
    """

    def __init__(
        self,
        registry: Optional[BotRegistryProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize Dispatcher.

        Args:
            registry: Bot registry for getting bot instances
            notifier: Notification manager for alerts
        """
        self._registry = registry
        self._notifier = notifier

    async def dispatch(
        self,
        allocations: Dict[str, Decimal],
        trigger: str = "manual",
    ) -> List[AllocationRecord]:
        """
        Dispatch fund allocations to bots.

        Args:
            allocations: Dictionary mapping bot_id to allocation amount
            trigger: What triggered the dispatch

        Returns:
            List of AllocationRecord for each dispatch attempt
        """
        records: List[AllocationRecord] = []

        for bot_id, amount in allocations.items():
            record = await self.notify_bot(bot_id, amount, trigger)
            records.append(record)

        # Log summary
        successful = sum(1 for r in records if r.success)
        failed = sum(1 for r in records if not r.success)
        logger.info(
            f"Dispatch complete: {successful} successful, {failed} failed"
        )

        return records

    async def notify_bot(
        self,
        bot_id: str,
        amount: Decimal,
        trigger: str = "manual",
    ) -> AllocationRecord:
        """
        Notify a single bot of its new capital allocation.

        Args:
            bot_id: Bot identifier
            amount: New allocation amount
            trigger: What triggered the allocation

        Returns:
            AllocationRecord with dispatch result
        """
        record = AllocationRecord(
            bot_id=bot_id,
            amount=amount,
            trigger=trigger,
            new_allocation=amount,
        )

        try:
            if not self._registry:
                raise RuntimeError("No registry configured")

            # Try to get bot instance directly
            instance = self._get_bot_instance(bot_id)

            if instance is not None:
                # Direct method call
                success = await self._notify_direct(instance, amount)
                record.success = success
                if not success:
                    record.error_message = "Bot rejected capital update"
            else:
                # Bot not running or not accessible
                # In future, could use IPC for remote bots
                logger.warning(f"Bot {bot_id} not found or not running")
                record.success = False
                record.error_message = "Bot not found or not running"

            # Send notification if successful
            if record.success and self._notifier:
                await self._send_notification(bot_id, amount)

        except Exception as e:
            logger.error(f"Failed to notify bot {bot_id}: {e}")
            record.success = False
            record.error_message = str(e)

        return record

    def _get_bot_instance(self, bot_id: str) -> Optional[BotProtocol]:
        """
        Get bot instance from registry.

        Args:
            bot_id: Bot identifier

        Returns:
            Bot instance if found and running
        """
        if not self._registry:
            return None

        # Try to get instance directly
        if hasattr(self._registry, "get_instance"):
            return self._registry.get_instance(bot_id)

        # Fallback: try to get from bot info
        bot_info = self._registry.get(bot_id)
        if bot_info and hasattr(bot_info, "instance"):
            return bot_info.instance

        return None

    async def _notify_direct(
        self,
        instance: BotProtocol,
        amount: Decimal,
    ) -> bool:
        """
        Notify bot directly via method call.

        Args:
            instance: Bot instance
            amount: New allocation amount

        Returns:
            True if successful
        """
        try:
            return await instance.update_capital(amount)
        except Exception as e:
            logger.error(f"Direct notification failed: {e}")
            return False

    async def _send_notification(
        self,
        bot_id: str,
        amount: Decimal,
    ) -> None:
        """
        Send notification about allocation.

        Args:
            bot_id: Bot identifier
            amount: Allocated amount
        """
        if not self._notifier:
            return

        try:
            if hasattr(self._notifier, "notify_fund_allocated"):
                await self._notifier.notify_fund_allocated(
                    bot_id=bot_id,
                    amount=amount,
                    total_allocated=amount,
                )
        except Exception as e:
            logger.warning(f"Failed to send allocation notification: {e}")

    def create_fund_update_command(
        self,
        bot_id: str,
        new_capital: Decimal,
    ) -> Command:
        """
        Create IPC command for fund update.

        Args:
            bot_id: Target bot ID
            new_capital: New capital amount

        Returns:
            Command message for IPC
        """
        return Command(
            type=CommandType.FUND_UPDATE,
            params={
                "bot_id": bot_id,
                "new_capital": str(new_capital),
            },
        )
