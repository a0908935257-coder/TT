"""
Fund Dispatcher.

Handles dispatching fund allocation updates to trading bots.
Supports multiple notification methods: direct API call, IPC, and file-based.
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

from src.core import get_logger, get_timeout_config, with_timeout
from src.core.timeout import TimeoutError as OperationTimeout
from src.ipc.messages import Command, CommandType

from ..models.records import AllocationRecord

if TYPE_CHECKING:
    from ..notifier.bot_notifier import BotNotifier

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
    supporting multiple notification methods:
    - Direct method calls (for in-process bots)
    - IPC messages (for remote bots)
    - File-based notifications (for loosely coupled bots)

    Example:
        >>> dispatcher = Dispatcher(registry, notifier)
        >>> result = await dispatcher.dispatch(allocations)
    """

    def __init__(
        self,
        registry: Optional[BotRegistryProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        bot_notifier: Optional["BotNotifier"] = None,
    ):
        """
        Initialize Dispatcher.

        Args:
            registry: Bot registry for getting bot instances
            notifier: Notification manager for alerts
            bot_notifier: File-based bot notifier for file notifications
        """
        self._registry = registry
        self._notifier = notifier
        self._bot_notifier = bot_notifier

        # Bot notification method mapping (bot_id -> method)
        self._notification_methods: Dict[str, str] = {}

    def set_notification_method(
        self,
        bot_id: str,
        method: str,
        target: Optional[str] = None,
    ) -> None:
        """
        Set notification method for a bot.

        Args:
            bot_id: Bot identifier
            method: Notification method (direct, file, api, none)
            target: Target path or endpoint for file/api methods
        """
        self._notification_methods[bot_id] = method

        # Register with file notifier if using file method
        if method == "file" and self._bot_notifier:
            self._bot_notifier.register_bot(bot_id, method, target)

        logger.debug(f"Set notification method for {bot_id}: {method}")

    def get_notification_method(self, bot_id: str) -> str:
        """
        Get notification method for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Notification method (defaults to 'direct')
        """
        return self._notification_methods.get(bot_id, "direct")

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
        previous_allocation: Decimal = Decimal("0"),
    ) -> AllocationRecord:
        """
        Notify a single bot of its new capital allocation.

        Supports multiple notification methods:
        - direct: Call bot's update_capital method directly
        - file: Write JSON notification file for bot to read
        - none: Skip notification (for manual tracking)

        Args:
            bot_id: Bot identifier
            amount: New allocation amount (total, not increment)
            trigger: What triggered the allocation
            previous_allocation: Previous allocation amount

        Returns:
            AllocationRecord with dispatch result
        """
        record = AllocationRecord(
            bot_id=bot_id,
            amount=amount - previous_allocation,  # Store the increment
            trigger=trigger,
            previous_allocation=previous_allocation,
            new_allocation=amount,
        )

        method = self.get_notification_method(bot_id)

        try:
            if method == "none":
                # No notification, just record the allocation
                record.success = True
                logger.debug(f"Notification skipped for {bot_id} (method=none)")

            elif method == "file":
                # File-based notification
                record.success = await self._notify_via_file(
                    bot_id, amount, previous_allocation, trigger
                )
                if not record.success:
                    record.error_message = "Failed to write notification file"

            else:
                # Direct method call (default)
                if not self._registry:
                    # No registry, try file notification as fallback
                    if self._bot_notifier:
                        record.success = await self._notify_via_file(
                            bot_id, amount, previous_allocation, trigger
                        )
                        if not record.success:
                            record.error_message = "Failed to write notification file"
                    else:
                        record.success = False
                        record.error_message = "No registry or notifier configured"
                else:
                    instance = self._get_bot_instance(bot_id)

                    if instance is not None:
                        success = await self._notify_direct(instance, amount)
                        record.success = success
                        if not success:
                            record.error_message = "Bot rejected capital update"
                    else:
                        # Bot not running, try file notification as fallback
                        if self._bot_notifier:
                            record.success = await self._notify_via_file(
                                bot_id, amount, previous_allocation, trigger
                            )
                            if not record.success:
                                record.error_message = "Failed to write notification file"
                            else:
                                logger.info(
                                    f"Bot {bot_id} not running, used file notification"
                                )
                        else:
                            logger.warning(f"Bot {bot_id} not found or not running")
                            record.success = False
                            record.error_message = "Bot not found or not running"

            # Send alert notification if successful
            if record.success and self._notifier:
                await self._send_notification(bot_id, amount)

        except Exception as e:
            logger.error(f"Failed to notify bot {bot_id}: {e}")
            record.success = False
            record.error_message = str(e)

        return record

    async def _notify_via_file(
        self,
        bot_id: str,
        total_allocation: Decimal,
        previous_allocation: Decimal,
        trigger: str,
    ) -> bool:
        """
        Notify bot via file-based notification.

        Args:
            bot_id: Bot identifier
            total_allocation: Total allocation after update
            previous_allocation: Previous allocation amount
            trigger: Allocation trigger

        Returns:
            True if notification file was written successfully
        """
        if not self._bot_notifier:
            logger.warning("No bot notifier configured for file notifications")
            return False

        try:
            new_allocation = total_allocation - previous_allocation
            return await self._bot_notifier.notify_allocation(
                bot_id=bot_id,
                new_allocation=new_allocation,
                total_allocation=total_allocation,
                trigger=trigger,
            )
        except Exception as e:
            logger.error(f"File notification failed for {bot_id}: {e}")
            return False

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

        # Try to get instance using get_bot_instance method
        if hasattr(self._registry, "get_bot_instance"):
            return self._registry.get_bot_instance(bot_id)

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
        Notify bot directly via method call with timeout protection.

        Args:
            instance: Bot instance
            amount: New allocation amount

        Returns:
            True if successful
        """
        timeout_config = get_timeout_config()
        try:
            return await with_timeout(
                instance.update_capital(amount),
                timeout=timeout_config.bot_notification,
                operation_name=f"notify_bot_{instance.bot_id}",
                raise_on_timeout=False,
                default_result=False,
            )
        except OperationTimeout:
            logger.error(f"Bot notification timed out for {instance.bot_id}")
            return False
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
