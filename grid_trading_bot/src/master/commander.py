"""
Bot Commander for unified control commands.

Provides centralized command interface for bot lifecycle management.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Protocol, Tuple

from src.core import get_logger
from src.master.factory import BotFactory, InvalidBotConfigError, UnsupportedBotTypeError
from src.master.models import BotNotFoundError, BotState, BotType

if TYPE_CHECKING:
    from src.master.registry import BotRegistry

logger = get_logger(__name__)


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def send(self, message: str, **kwargs: Any) -> bool: ...


@dataclass
class CommandResult:
    """
    Result of a command execution.

    Attributes:
        success: Whether the command succeeded
        message: Human-readable message
        bot_id: Bot ID (if applicable)
        data: Additional data
    """

    success: bool
    message: str
    bot_id: Optional[str] = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "message": self.message,
            "bot_id": self.bot_id,
            "data": self.data,
        }


class BotCommander:
    """
    Unified command interface for bot lifecycle management.

    Handles all bot commands: create, start, stop, pause, resume, restart, delete.
    Provides thread-safe operations with proper locking.

    Example:
        >>> commander = BotCommander(registry, factory, notifier)
        >>> result = await commander.create(BotType.GRID, config)
        >>> if result.success:
        ...     await commander.start(result.bot_id)
    """

    def __init__(
        self,
        registry: "BotRegistry",
        factory: BotFactory,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize BotCommander.

        Args:
            registry: Bot registry instance
            factory: Bot factory instance
            notifier: Optional notification manager
        """
        self._registry = registry
        self._factory = factory
        self._notifier = notifier
        self._lock = asyncio.Lock()

    # =========================================================================
    # Core Commands
    # =========================================================================

    async def create(
        self,
        bot_type: BotType,
        config: dict[str, Any],
        bot_id: Optional[str] = None,
    ) -> CommandResult:
        """
        Create a new bot.

        Args:
            bot_type: Type of bot to create
            config: Bot configuration
            bot_id: Optional custom bot ID (auto-generated if not provided)

        Returns:
            CommandResult with success status and bot_id
        """
        async with self._lock:
            try:
                # Generate bot_id if not provided
                if bot_id is None:
                    bot_id = self._generate_bot_id(bot_type)

                # Check if bot already exists
                if self._registry.get(bot_id):
                    return CommandResult(
                        success=False,
                        message=f"Bot {bot_id} already exists",
                        bot_id=bot_id,
                    )

                # Register in registry
                await self._registry.register(
                    bot_id=bot_id,
                    bot_type=bot_type,
                    config=config,
                )

                # Create instance using factory
                try:
                    instance = self._factory.create(bot_type, bot_id, config)
                    self._registry.bind_instance(bot_id, instance)
                except (UnsupportedBotTypeError, InvalidBotConfigError) as e:
                    # Rollback registration
                    await self._registry.unregister(bot_id)
                    return CommandResult(
                        success=False,
                        message=str(e),
                        bot_id=bot_id,
                    )

                # Send notification
                await self._notify(
                    f"Bot created: {bot_id} ({bot_type.value})\n"
                    f"Symbol: {config.get('symbol')}"
                )

                logger.info(f"Bot created: {bot_id}")
                return CommandResult(
                    success=True,
                    message=f"Bot {bot_id} created successfully",
                    bot_id=bot_id,
                    data={"bot_type": bot_type.value, "config": config},
                )

            except Exception as e:
                logger.error(f"Failed to create bot: {e}")
                return CommandResult(
                    success=False,
                    message=f"Failed to create bot: {e}",
                    bot_id=bot_id,
                )

    async def start(self, bot_id: str) -> CommandResult:
        """
        Start a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        async with self._lock:
            # Validate can start
            valid, error_msg = self._validate_can_start(bot_id)
            if not valid:
                return CommandResult(
                    success=False,
                    message=error_msg,
                    bot_id=bot_id,
                )

            try:
                # Get instance
                instance = self._registry.get_bot_instance(bot_id)
                if not instance:
                    return CommandResult(
                        success=False,
                        message=f"Bot instance not found: {bot_id}",
                        bot_id=bot_id,
                    )

                # Update state to INITIALIZING
                await self._registry.update_state(bot_id, BotState.INITIALIZING)

                # Start the bot
                success = await instance.start()

                if success:
                    # Update state to RUNNING
                    await self._registry.update_state(bot_id, BotState.RUNNING)

                    # Get bot info for notification
                    bot_info = self._registry.get(bot_id)
                    await self._notify(
                        f"Bot started: {bot_id}\n"
                        f"Type: {bot_info.bot_type.value}\n"
                        f"Symbol: {bot_info.symbol}"
                    )

                    logger.info(f"Bot started: {bot_id}")
                    return CommandResult(
                        success=True,
                        message=f"Bot {bot_id} started successfully",
                        bot_id=bot_id,
                    )
                else:
                    # Update state to ERROR
                    await self._registry.update_state(
                        bot_id, BotState.ERROR, "Failed to start"
                    )
                    return CommandResult(
                        success=False,
                        message=f"Bot {bot_id} failed to start",
                        bot_id=bot_id,
                    )

            except Exception as e:
                logger.error(f"Error starting bot {bot_id}: {e}")
                try:
                    await self._registry.update_state(
                        bot_id, BotState.ERROR, str(e)
                    )
                except Exception:
                    pass
                return CommandResult(
                    success=False,
                    message=f"Error starting bot: {e}",
                    bot_id=bot_id,
                )

    async def stop(
        self,
        bot_id: str,
        clear_position: bool = False,
    ) -> CommandResult:
        """
        Stop a bot.

        Args:
            bot_id: Bot identifier
            clear_position: Whether to clear positions when stopping

        Returns:
            CommandResult with success status
        """
        async with self._lock:
            # Validate can stop
            valid, error_msg = self._validate_can_stop(bot_id)
            if not valid:
                return CommandResult(
                    success=False,
                    message=error_msg,
                    bot_id=bot_id,
                )

            try:
                # Get instance
                instance = self._registry.get_bot_instance(bot_id)
                if not instance:
                    return CommandResult(
                        success=False,
                        message=f"Bot instance not found: {bot_id}",
                        bot_id=bot_id,
                    )

                # Update state to STOPPING
                await self._registry.update_state(bot_id, BotState.STOPPING)

                # Stop the bot
                reason = "Manual stop"
                if clear_position:
                    reason += " (clearing positions)"

                success = await instance.stop(reason=reason)

                if success:
                    # Update state to STOPPED
                    await self._registry.update_state(bot_id, BotState.STOPPED)

                    # Get statistics if available
                    stats = {}
                    if hasattr(instance, "get_statistics"):
                        stats = instance.get_statistics()

                    await self._notify(
                        f"Bot stopped: {bot_id}\n"
                        f"Trades: {stats.get('trade_count', 'N/A')}\n"
                        f"Profit: {stats.get('total_profit', 'N/A')}"
                    )

                    logger.info(f"Bot stopped: {bot_id}")
                    return CommandResult(
                        success=True,
                        message=f"Bot {bot_id} stopped successfully",
                        bot_id=bot_id,
                        data={"statistics": stats},
                    )
                else:
                    return CommandResult(
                        success=False,
                        message=f"Bot {bot_id} failed to stop",
                        bot_id=bot_id,
                    )

            except Exception as e:
                logger.error(f"Error stopping bot {bot_id}: {e}")
                return CommandResult(
                    success=False,
                    message=f"Error stopping bot: {e}",
                    bot_id=bot_id,
                )

    async def pause(self, bot_id: str) -> CommandResult:
        """
        Pause a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        async with self._lock:
            # Validate can pause
            valid, error_msg = self._validate_can_pause(bot_id)
            if not valid:
                return CommandResult(
                    success=False,
                    message=error_msg,
                    bot_id=bot_id,
                )

            try:
                # Get instance
                instance = self._registry.get_bot_instance(bot_id)
                if not instance:
                    return CommandResult(
                        success=False,
                        message=f"Bot instance not found: {bot_id}",
                        bot_id=bot_id,
                    )

                # Pause the bot
                success = await instance.pause(reason="Manual pause")

                if success:
                    # Update state to PAUSED
                    await self._registry.update_state(bot_id, BotState.PAUSED)

                    await self._notify(f"Bot paused: {bot_id}")

                    logger.info(f"Bot paused: {bot_id}")
                    return CommandResult(
                        success=True,
                        message=f"Bot {bot_id} paused successfully",
                        bot_id=bot_id,
                    )
                else:
                    return CommandResult(
                        success=False,
                        message=f"Bot {bot_id} failed to pause",
                        bot_id=bot_id,
                    )

            except Exception as e:
                logger.error(f"Error pausing bot {bot_id}: {e}")
                return CommandResult(
                    success=False,
                    message=f"Error pausing bot: {e}",
                    bot_id=bot_id,
                )

    async def resume(self, bot_id: str) -> CommandResult:
        """
        Resume a paused bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        async with self._lock:
            # Validate can resume
            valid, error_msg = self._validate_can_resume(bot_id)
            if not valid:
                return CommandResult(
                    success=False,
                    message=error_msg,
                    bot_id=bot_id,
                )

            try:
                # Get instance
                instance = self._registry.get_bot_instance(bot_id)
                if not instance:
                    return CommandResult(
                        success=False,
                        message=f"Bot instance not found: {bot_id}",
                        bot_id=bot_id,
                    )

                # Resume the bot
                success = await instance.resume()

                if success:
                    # Update state to RUNNING
                    await self._registry.update_state(bot_id, BotState.RUNNING)

                    await self._notify(f"Bot resumed: {bot_id}")

                    logger.info(f"Bot resumed: {bot_id}")
                    return CommandResult(
                        success=True,
                        message=f"Bot {bot_id} resumed successfully",
                        bot_id=bot_id,
                    )
                else:
                    return CommandResult(
                        success=False,
                        message=f"Bot {bot_id} failed to resume",
                        bot_id=bot_id,
                    )

            except Exception as e:
                logger.error(f"Error resuming bot {bot_id}: {e}")
                return CommandResult(
                    success=False,
                    message=f"Error resuming bot: {e}",
                    bot_id=bot_id,
                )

    async def restart(self, bot_id: str) -> CommandResult:
        """
        Restart a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        # Stop first (without lock since stop/start acquire it)
        stop_result = await self.stop(bot_id)
        if not stop_result.success:
            return CommandResult(
                success=False,
                message=f"Failed to stop bot for restart: {stop_result.message}",
                bot_id=bot_id,
            )

        # Wait before restarting
        await asyncio.sleep(2)

        # Start again
        start_result = await self.start(bot_id)
        if not start_result.success:
            return CommandResult(
                success=False,
                message=f"Failed to start bot after restart: {start_result.message}",
                bot_id=bot_id,
            )

        await self._notify(f"Bot restarted: {bot_id}")

        logger.info(f"Bot restarted: {bot_id}")
        return CommandResult(
            success=True,
            message=f"Bot {bot_id} restarted successfully",
            bot_id=bot_id,
        )

    async def delete(self, bot_id: str) -> CommandResult:
        """
        Delete a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        # Check if bot exists
        bot_info = self._registry.get(bot_id)
        if not bot_info:
            return CommandResult(
                success=False,
                message=f"Bot not found: {bot_id}",
                bot_id=bot_id,
            )

        # Stop if running
        if bot_info.state in (BotState.RUNNING, BotState.PAUSED):
            stop_result = await self.stop(bot_id)
            if not stop_result.success:
                return CommandResult(
                    success=False,
                    message=f"Failed to stop bot before delete: {stop_result.message}",
                    bot_id=bot_id,
                )

        async with self._lock:
            try:
                # Unregister from registry
                await self._registry.unregister(bot_id)

                await self._notify(f"Bot deleted: {bot_id}")

                logger.info(f"Bot deleted: {bot_id}")
                return CommandResult(
                    success=True,
                    message=f"Bot {bot_id} deleted successfully",
                    bot_id=bot_id,
                )

            except Exception as e:
                logger.error(f"Error deleting bot {bot_id}: {e}")
                return CommandResult(
                    success=False,
                    message=f"Error deleting bot: {e}",
                    bot_id=bot_id,
                )

    # =========================================================================
    # Batch Commands
    # =========================================================================

    async def start_all(self) -> List[CommandResult]:
        """
        Start all startable bots.

        Returns:
            List of CommandResult for each bot
        """
        results = []

        # Get bots that can be started
        registered = self._registry.get_by_state(BotState.REGISTERED)
        stopped = self._registry.get_by_state(BotState.STOPPED)
        bots_to_start = registered + stopped

        for bot_info in bots_to_start:
            result = await self.start(bot_info.bot_id)
            results.append(result)
            # Small delay between starts to avoid overwhelming
            await asyncio.sleep(1)

        started_count = sum(1 for r in results if r.success)
        logger.info(f"Started {started_count}/{len(results)} bots")

        return results

    async def stop_all(self) -> List[CommandResult]:
        """
        Stop all running bots.

        Returns:
            List of CommandResult for each bot
        """
        results = []

        # Get bots that can be stopped
        running = self._registry.get_by_state(BotState.RUNNING)
        paused = self._registry.get_by_state(BotState.PAUSED)
        bots_to_stop = running + paused

        for bot_info in bots_to_stop:
            result = await self.stop(bot_info.bot_id)
            results.append(result)

        stopped_count = sum(1 for r in results if r.success)
        logger.info(f"Stopped {stopped_count}/{len(results)} bots")

        return results

    async def pause_all(self) -> List[CommandResult]:
        """
        Pause all running bots.

        Returns:
            List of CommandResult for each bot
        """
        results = []

        # Get running bots
        running = self._registry.get_by_state(BotState.RUNNING)

        for bot_info in running:
            result = await self.pause(bot_info.bot_id)
            results.append(result)

        paused_count = sum(1 for r in results if r.success)
        logger.info(f"Paused {paused_count}/{len(results)} bots")

        return results

    async def resume_all(self) -> List[CommandResult]:
        """
        Resume all paused bots.

        Returns:
            List of CommandResult for each bot
        """
        results = []

        # Get paused bots
        paused = self._registry.get_by_state(BotState.PAUSED)

        for bot_info in paused:
            result = await self.resume(bot_info.bot_id)
            results.append(result)

        resumed_count = sum(1 for r in results if r.success)
        logger.info(f"Resumed {resumed_count}/{len(results)} bots")

        return results

    # =========================================================================
    # Validation Helpers
    # =========================================================================

    def _validate_can_start(self, bot_id: str) -> Tuple[bool, str]:
        """
        Validate if bot can be started.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        bot_info = self._registry.get(bot_id)
        if bot_info is None:
            return False, f"Bot not found: {bot_id}"

        valid_states = [BotState.REGISTERED, BotState.STOPPED]
        if bot_info.state not in valid_states:
            return False, f"Cannot start bot in state: {bot_info.state.value}"

        return True, ""

    def _validate_can_stop(self, bot_id: str) -> Tuple[bool, str]:
        """
        Validate if bot can be stopped.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        bot_info = self._registry.get(bot_id)
        if bot_info is None:
            return False, f"Bot not found: {bot_id}"

        valid_states = [BotState.RUNNING, BotState.PAUSED]
        if bot_info.state not in valid_states:
            return False, f"Cannot stop bot in state: {bot_info.state.value}"

        return True, ""

    def _validate_can_pause(self, bot_id: str) -> Tuple[bool, str]:
        """
        Validate if bot can be paused.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        bot_info = self._registry.get(bot_id)
        if bot_info is None:
            return False, f"Bot not found: {bot_id}"

        if bot_info.state != BotState.RUNNING:
            return False, f"Cannot pause bot in state: {bot_info.state.value}"

        return True, ""

    def _validate_can_resume(self, bot_id: str) -> Tuple[bool, str]:
        """
        Validate if bot can be resumed.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (is_valid, error_message)
        """
        bot_info = self._registry.get(bot_id)
        if bot_info is None:
            return False, f"Bot not found: {bot_id}"

        if bot_info.state != BotState.PAUSED:
            return False, f"Cannot resume bot in state: {bot_info.state.value}"

        return True, ""

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _generate_bot_id(self, bot_type: BotType) -> str:
        """
        Generate a unique bot ID.

        Args:
            bot_type: Type of bot

        Returns:
            Generated bot ID
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        short_uuid = str(uuid.uuid4())[:8]
        return f"{bot_type.value}_{timestamp}_{short_uuid}"

    async def _notify(self, message: str) -> None:
        """
        Send notification.

        Args:
            message: Notification message
        """
        if self._notifier:
            try:
                await self._notifier.send(message)
            except Exception as e:
                logger.warning(f"Failed to send notification: {e}")
