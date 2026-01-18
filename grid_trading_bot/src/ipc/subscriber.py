"""
IPC Subscriber.

Handles subscribing to Redis Pub/Sub channels and dispatching messages.
"""

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable, Dict, Optional, Union

from src.core import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from redis.asyncio.client import PubSub

logger = get_logger(__name__)

# Type alias for message handlers
MessageHandler = Callable[[str], Awaitable[None]]


class IPCSubscriber:
    """
    Subscriber for IPC messages via Redis Pub/Sub.

    Manages subscriptions to multiple channels and dispatches
    received messages to registered handlers.

    Example:
        subscriber = IPCSubscriber(redis)

        async def handle_command(data: str):
            command = Command.from_json(data)
            print(f"Received: {command.type}")

        await subscriber.subscribe("trading:cmd:bot-001", handle_command)
        await subscriber.start()

        # Later...
        await subscriber.stop()
    """

    def __init__(self, redis: "Redis"):
        """
        Initialize subscriber.

        Args:
            redis: Async Redis client instance
        """
        self._redis = redis
        self._pubsub: Optional["PubSub"] = None
        self._handlers: Dict[str, MessageHandler] = {}
        self._pattern_handlers: Dict[str, MessageHandler] = {}
        self._running = False
        self._listen_task: Optional[asyncio.Task] = None

    @property
    def is_running(self) -> bool:
        """Check if subscriber is running."""
        return self._running

    @property
    def subscribed_channels(self) -> list:
        """Get list of subscribed channels."""
        return list(self._handlers.keys())

    @property
    def subscribed_patterns(self) -> list:
        """Get list of subscribed patterns."""
        return list(self._pattern_handlers.keys())

    async def _ensure_pubsub(self) -> "PubSub":
        """Ensure pubsub is initialized."""
        if self._pubsub is None:
            self._pubsub = self._redis.pubsub()
        return self._pubsub

    async def subscribe(self, channel: str, handler: MessageHandler) -> None:
        """
        Subscribe to a channel with a handler.

        Args:
            channel: Channel name to subscribe to
            handler: Async function to handle received messages
        """
        pubsub = await self._ensure_pubsub()
        await pubsub.subscribe(channel)
        self._handlers[channel] = handler
        logger.debug(f"Subscribed to channel: {channel}")

    async def psubscribe(self, pattern: str, handler: MessageHandler) -> None:
        """
        Subscribe to channels matching a pattern.

        Args:
            pattern: Channel pattern (e.g., "trading:hb:*")
            handler: Async function to handle received messages
        """
        pubsub = await self._ensure_pubsub()
        await pubsub.psubscribe(pattern)
        self._pattern_handlers[pattern] = handler
        logger.debug(f"Subscribed to pattern: {pattern}")

    async def unsubscribe(self, channel: str) -> None:
        """
        Unsubscribe from a channel.

        Args:
            channel: Channel name to unsubscribe from
        """
        if self._pubsub:
            await self._pubsub.unsubscribe(channel)
        self._handlers.pop(channel, None)
        logger.debug(f"Unsubscribed from channel: {channel}")

    async def punsubscribe(self, pattern: str) -> None:
        """
        Unsubscribe from a pattern.

        Args:
            pattern: Channel pattern to unsubscribe from
        """
        if self._pubsub:
            await self._pubsub.punsubscribe(pattern)
        self._pattern_handlers.pop(pattern, None)
        logger.debug(f"Unsubscribed from pattern: {pattern}")

    async def start(self) -> None:
        """
        Start the message listening loop.

        Creates a background task that continuously listens for
        messages and dispatches them to handlers.
        """
        if self._running:
            logger.warning("Subscriber already running")
            return

        if not self._handlers and not self._pattern_handlers:
            logger.warning("No subscriptions, not starting listener")
            return

        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        logger.info("IPC subscriber started")

    async def stop(self) -> None:
        """
        Stop the message listening loop.

        Cancels the listening task and closes the pubsub connection.
        """
        self._running = False

        if self._listen_task:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            self._listen_task = None

        if self._pubsub:
            await self._pubsub.close()
            self._pubsub = None

        self._handlers.clear()
        self._pattern_handlers.clear()
        logger.info("IPC subscriber stopped")

    async def _listen_loop(self) -> None:
        """
        Main message listening loop.

        Continuously polls for messages and dispatches them to
        appropriate handlers.
        """
        pubsub = await self._ensure_pubsub()

        while self._running:
            try:
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=0.1,
                )

                if message is None:
                    await asyncio.sleep(0.01)
                    continue

                await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in listen loop: {e}")
                await asyncio.sleep(0.1)

    async def _handle_message(self, message: dict) -> None:
        """
        Handle a received message.

        Args:
            message: Redis pubsub message dict
        """
        msg_type = message.get("type")
        if msg_type not in ("message", "pmessage"):
            return

        # Decode channel and data
        channel = message.get("channel")
        if isinstance(channel, bytes):
            channel = channel.decode("utf-8")

        data = message.get("data")
        if isinstance(data, bytes):
            data = data.decode("utf-8")

        if not channel or not data:
            return

        # Find and call handler
        handler: Optional[MessageHandler] = None

        if msg_type == "message":
            handler = self._handlers.get(channel)
        elif msg_type == "pmessage":
            pattern = message.get("pattern")
            if isinstance(pattern, bytes):
                pattern = pattern.decode("utf-8")
            handler = self._pattern_handlers.get(pattern)

        if handler:
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in message handler for {channel}: {e}")
        else:
            logger.warning(f"No handler for channel: {channel}")
