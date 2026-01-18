"""
IPC Channel Names.

Defines channel naming conventions for Redis Pub/Sub communication.
"""


class Channel:
    """
    Channel name generator for IPC communication.

    Channel naming convention:
    - trading:cmd:{bot_id}  - Commands from Master to specific Bot
    - trading:resp:{bot_id} - Responses from Bot to Master
    - trading:hb:{bot_id}   - Heartbeats from Bot to Master
    - trading:event         - Events from any Bot to Master (broadcast)

    Example:
        channel = Channel.command("grid-bot-001")
        # Returns: "trading:cmd:grid-bot-001"
    """

    PREFIX = "trading"

    @staticmethod
    def command(bot_id: str) -> str:
        """
        Get command channel for a specific bot.

        Commands flow: Master -> Bot

        Args:
            bot_id: Unique bot identifier

        Returns:
            Channel name for sending commands to this bot
        """
        return f"{Channel.PREFIX}:cmd:{bot_id}"

    @staticmethod
    def response(bot_id: str) -> str:
        """
        Get response channel for a specific bot.

        Responses flow: Bot -> Master

        Args:
            bot_id: Unique bot identifier

        Returns:
            Channel name for receiving responses from this bot
        """
        return f"{Channel.PREFIX}:resp:{bot_id}"

    @staticmethod
    def heartbeat(bot_id: str) -> str:
        """
        Get heartbeat channel for a specific bot.

        Heartbeats flow: Bot -> Master

        Args:
            bot_id: Unique bot identifier

        Returns:
            Channel name for receiving heartbeats from this bot
        """
        return f"{Channel.PREFIX}:hb:{bot_id}"

    @staticmethod
    def event() -> str:
        """
        Get event channel (broadcast).

        Events flow: Any Bot -> Master

        Returns:
            Channel name for receiving events from all bots
        """
        return f"{Channel.PREFIX}:event"

    @staticmethod
    def pattern_all_heartbeats() -> str:
        """
        Get pattern to subscribe to all heartbeat channels.

        Returns:
            Pattern for psubscribe to match all heartbeat channels
        """
        return f"{Channel.PREFIX}:hb:*"

    @staticmethod
    def pattern_all_responses() -> str:
        """
        Get pattern to subscribe to all response channels.

        Returns:
            Pattern for psubscribe to match all response channels
        """
        return f"{Channel.PREFIX}:resp:*"
