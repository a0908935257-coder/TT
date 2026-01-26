"""
Master IPC Handler.

Handles inter-process communication between Master and Bot processes.
"""

import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

from src.core import get_logger
from src.ipc import (
    Channel,
    Command,
    CommandType,
    Event,
    Heartbeat,
    IPCPublisher,
    IPCSubscriber,
    Response,
)

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from src.master.heartbeat import HeartbeatMonitor
    from src.master.registry import BotRegistry

logger = get_logger(__name__)


class MasterIPCHandler:
    """
    Master IPC Handler.

    Manages IPC communication between Master and Bot processes:
    - Sends commands to bots and waits for responses
    - Receives heartbeats from bots
    - Receives events from bots

    Example:
        handler = MasterIPCHandler(redis, registry, heartbeat_monitor)
        await handler.start()
        response = await handler.send_command("bot-001", CommandType.START)
        await handler.stop()
    """

    def __init__(
        self,
        redis: "Redis",
        registry: "BotRegistry",
        heartbeat_monitor: "HeartbeatMonitor",
        event_callback: Optional[Callable[[Event], Any]] = None,
    ):
        """
        Initialize MasterIPCHandler.

        Args:
            redis: Async Redis client instance
            registry: Bot registry for tracking bot states
            heartbeat_monitor: Heartbeat monitor for processing heartbeats
            event_callback: Optional callback for bot events
        """
        self._redis = redis
        self._registry = registry
        self._heartbeat_monitor = heartbeat_monitor
        self._event_callback = event_callback

        self._publisher = IPCPublisher(redis)
        self._subscriber = IPCSubscriber(redis)

        # Pending command responses: command_id -> Future
        self._pending: Dict[str, asyncio.Future] = {}

        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if handler is running."""
        return self._running

    async def start(self) -> None:
        """
        Start the IPC handler.

        Subscribes to response, heartbeat, and event channels for all
        registered bots, then starts the listener.
        """
        if self._running:
            logger.warning("IPC handler already running")
            return

        logger.info("Starting Master IPC handler")

        # Subscribe to event channel (broadcast from all bots)
        await self._subscriber.subscribe(Channel.event(), self._on_event)

        # Subscribe to response and heartbeat channels for registered bots
        for bot_info in self._registry.get_all():
            await self._subscribe_bot(bot_info.bot_id)

        # Start listening
        self._running = True
        self._listen_task = asyncio.create_task(self._subscriber.start())

        logger.info("Master IPC handler started")

    async def stop(self) -> None:
        """Stop the IPC handler."""
        logger.info("Stopping Master IPC handler")
        self._running = False

        # Cancel pending commands safely
        for cmd_id, future in list(self._pending.items()):
            try:
                if not future.done():
                    future.set_exception(asyncio.CancelledError("IPC handler stopped"))
            except asyncio.InvalidStateError:
                # Future was already completed or cancelled
                pass
        self._pending.clear()

        # Stop subscriber
        await self._subscriber.stop()

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        logger.info("Master IPC handler stopped")

    async def subscribe_bot(self, bot_id: str) -> None:
        """
        Subscribe to a new bot's channels.

        Call this when a new bot is registered.

        Args:
            bot_id: Bot identifier to subscribe to
        """
        await self._subscribe_bot(bot_id)

    async def unsubscribe_bot(self, bot_id: str) -> None:
        """
        Unsubscribe from a bot's channels.

        Call this when a bot is removed.

        Args:
            bot_id: Bot identifier to unsubscribe from
        """
        await self._subscriber.unsubscribe(Channel.response(bot_id))
        await self._subscriber.unsubscribe(Channel.heartbeat(bot_id))
        logger.debug(f"Unsubscribed from bot {bot_id}")

    async def _subscribe_bot(self, bot_id: str) -> None:
        """Subscribe to a bot's response and heartbeat channels."""
        await self._subscriber.subscribe(
            Channel.response(bot_id),
            self._on_response,
        )
        await self._subscriber.subscribe(
            Channel.heartbeat(bot_id),
            self._on_heartbeat,
        )
        logger.debug(f"Subscribed to bot {bot_id}")

    async def send_command(
        self,
        bot_id: str,
        cmd_type: CommandType,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 30.0,
    ) -> Response:
        """
        Send a command to a bot and wait for response.

        Args:
            bot_id: Target bot identifier
            cmd_type: Type of command to send
            params: Optional command parameters
            timeout: Timeout in seconds

        Returns:
            Response from bot (success or error)
        """
        # Create command
        cmd = Command(
            id=str(uuid.uuid4()),
            type=cmd_type,
            params=params or {},
        )

        logger.info(f"Sending command {cmd_type.value} to {bot_id} (id={cmd.id})")

        # Create Future for response
        future: asyncio.Future = asyncio.Future()
        self._pending[cmd.id] = future

        try:
            # Send command
            await self._publisher.send_command(bot_id, cmd)

            # Wait for response with timeout
            response = await asyncio.wait_for(future, timeout=timeout)
            logger.info(f"Received response for command {cmd.id}: success={response.success}")
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Command {cmd.id} timed out after {timeout}s")
            self._pending.pop(cmd.id, None)
            return Response(
                command_id=cmd.id,
                success=False,
                error=f"Command timed out after {timeout}s",
            )
        except asyncio.CancelledError:
            self._pending.pop(cmd.id, None)
            return Response(
                command_id=cmd.id,
                success=False,
                error="Command cancelled",
            )
        except Exception as e:
            logger.error(f"Error sending command {cmd.id}: {e}")
            self._pending.pop(cmd.id, None)
            return Response(
                command_id=cmd.id,
                success=False,
                error=str(e),
            )

    async def _on_response(self, data: str) -> None:
        """
        Handle incoming response from a bot.

        Args:
            data: JSON string containing Response
        """
        try:
            response = Response.from_json(data)
            logger.debug(f"Received response for command {response.command_id}")

            # Find and complete the pending Future
            future = self._pending.pop(response.command_id, None)
            if future is None:
                logger.warning(f"No pending Future for command {response.command_id}")
            else:
                try:
                    if not future.done():
                        future.set_result(response)
                except asyncio.InvalidStateError:
                    # Future was cancelled or already completed between check and set
                    logger.debug(f"Future for command {response.command_id} already completed or cancelled")

        except Exception as e:
            logger.error(f"Error processing response: {e}")

    async def _on_heartbeat(self, data: str) -> None:
        """
        Handle incoming heartbeat from a bot.

        Args:
            data: JSON string containing Heartbeat
        """
        try:
            heartbeat = Heartbeat.from_json(data)
            logger.debug(f"Received heartbeat from {heartbeat.bot_id}: state={heartbeat.state}")

            # Update registry with heartbeat data
            self._registry.update_heartbeat(
                heartbeat.bot_id,
                heartbeat.state,
                heartbeat.metrics,
            )

            # Notify heartbeat monitor
            # Convert IPC Heartbeat to master heartbeat format
            # Use the heartbeat's state directly (not from registry) to avoid stale state
            from src.master.heartbeat import HeartbeatData
            from src.master.models import BotState
            try:
                bot_state = BotState(heartbeat.state)
            except ValueError:
                # Unknown state string, default to UNKNOWN or use registry state as fallback
                bot_info = self._registry.get(heartbeat.bot_id)
                bot_state = bot_info.state if bot_info else BotState.UNKNOWN
                logger.warning(f"Unknown state '{heartbeat.state}' from bot {heartbeat.bot_id}")

            hb_data = HeartbeatData(
                bot_id=heartbeat.bot_id,
                state=bot_state,
                metrics=heartbeat.metrics,
            )
            await self._heartbeat_monitor.receive(hb_data)

        except Exception as e:
            logger.error(f"Error processing heartbeat: {e}")

    async def _on_event(self, data: str) -> None:
        """
        Handle incoming event from a bot.

        Args:
            data: JSON string containing Event
        """
        try:
            event = Event.from_json(data)
            logger.info(f"Received event {event.type.value} from {event.bot_id}")

            # Update registry based on event type
            from src.ipc import EventType
            from src.master.models import BotState

            if event.type == EventType.STARTED:
                self._registry.update_state(event.bot_id, BotState.RUNNING)
            elif event.type == EventType.STOPPED:
                self._registry.update_state(event.bot_id, BotState.STOPPED)
            elif event.type == EventType.ERROR:
                self._registry.update_state(event.bot_id, BotState.ERROR)
                self._registry.set_error(event.bot_id, event.data.get("error", "Unknown error"))

            # Invoke callback if provided
            if self._event_callback:
                try:
                    result = self._event_callback(event)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Event callback error: {e}")

        except Exception as e:
            logger.error(f"Error processing event: {e}")
