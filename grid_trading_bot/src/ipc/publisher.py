"""
IPC Publisher.

Handles publishing messages to Redis Pub/Sub channels.
"""

from typing import TYPE_CHECKING, Union

from src.core import get_logger

from .channel import Channel
from .messages import BaseMessage, Command, Event, Heartbeat, Response

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger(__name__)


class IPCPublisher:
    """
    Publisher for IPC messages via Redis Pub/Sub.

    Provides typed methods for publishing different message types
    to their appropriate channels.

    Example:
        publisher = IPCPublisher(redis)
        await publisher.send_command("bot-001", command)
        await publisher.send_heartbeat("bot-001", heartbeat)
    """

    def __init__(self, redis: "Redis"):
        """
        Initialize publisher.

        Args:
            redis: Async Redis client instance
        """
        self._redis = redis

    async def publish(self, channel: str, message: Union[BaseMessage, str]) -> int:
        """
        Publish a message to a channel.

        Args:
            channel: Redis channel name
            message: Message to publish (BaseMessage or JSON string)

        Returns:
            Number of subscribers that received the message
        """
        if isinstance(message, BaseMessage):
            json_str = message.to_json()
        else:
            json_str = message

        result = await self._redis.publish(channel, json_str)
        logger.debug(f"Published to {channel}: {json_str[:100]}...")
        return result

    async def send_command(self, bot_id: str, command: Command) -> int:
        """
        Send a command to a specific bot.

        Args:
            bot_id: Target bot identifier
            command: Command message to send

        Returns:
            Number of subscribers that received the message
        """
        channel = Channel.command(bot_id)
        logger.debug(f"Sending command {command.type.value} to {bot_id}")
        return await self.publish(channel, command)

    async def send_response(self, bot_id: str, response: Response) -> int:
        """
        Send a response from a bot to Master.

        Args:
            bot_id: Source bot identifier
            response: Response message to send

        Returns:
            Number of subscribers that received the message
        """
        channel = Channel.response(bot_id)
        logger.debug(f"Sending response for command {response.command_id} from {bot_id}")
        return await self.publish(channel, response)

    async def send_heartbeat(self, bot_id: str, heartbeat: Heartbeat) -> int:
        """
        Send a heartbeat from a bot to Master.

        Args:
            bot_id: Source bot identifier
            heartbeat: Heartbeat message to send

        Returns:
            Number of subscribers that received the message
        """
        channel = Channel.heartbeat(bot_id)
        logger.debug(f"Sending heartbeat from {bot_id}, state={heartbeat.state}")
        return await self.publish(channel, heartbeat)

    async def send_event(self, event: Event) -> int:
        """
        Send an event to the event channel (broadcast).

        Args:
            event: Event message to send

        Returns:
            Number of subscribers that received the message
        """
        channel = Channel.event()
        logger.debug(f"Sending event {event.type.value} from {event.bot_id}")
        return await self.publish(channel, event)
