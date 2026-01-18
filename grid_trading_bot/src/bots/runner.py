"""
Bot Runner - Independent Process Entry Point.

Provides the entry point for running bots as independent processes.
Handles IPC communication with Master, heartbeat reporting, and command execution.

Usage:
    python -m src.bots.runner --bot-id bot_001 --config config/bot_001.yaml
"""

import argparse
import asyncio
import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Type

import yaml

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

from .base import BaseBot

logger = get_logger(__name__)

# Bot type registry - maps bot_type string to class
BOT_REGISTRY: Dict[str, Type[BaseBot]] = {}


def register_bot(bot_type: str, bot_class: Type[BaseBot]) -> None:
    """Register a bot type."""
    BOT_REGISTRY[bot_type] = bot_class


def get_bot_class(bot_type: str) -> Optional[Type[BaseBot]]:
    """Get bot class by type."""
    return BOT_REGISTRY.get(bot_type)


# Register built-in bot types
def _register_builtin_bots() -> None:
    """Register built-in bot types."""
    try:
        from .grid import GridBot
        register_bot("grid", GridBot)
    except ImportError:
        logger.warning("GridBot not available")


@dataclass
class RunnerConfig:
    """
    Runner configuration.

    Attributes:
        bot_id: Unique bot identifier
        bot_type: Type of bot (grid, dca, etc.)
        redis_url: Redis connection URL
        bot_config: Bot-specific configuration dict
    """

    bot_id: str
    bot_type: str
    redis_url: str = "redis://localhost:6379"
    bot_config: Dict[str, Any] = None

    def __post_init__(self):
        if self.bot_config is None:
            self.bot_config = {}


class BotRunner:
    """
    Bot process runner.

    Manages a single bot instance as an independent process.
    Handles IPC communication with Master, heartbeat reporting,
    and command execution.

    Example:
        runner = BotRunner("bot_001", "config/bot_001.yaml")
        await runner.run()
    """

    def __init__(self, bot_id: str, config_path: str):
        """
        Initialize BotRunner.

        Args:
            bot_id: Unique bot identifier
            config_path: Path to configuration file (YAML or JSON)
        """
        self._bot_id = bot_id
        self._config_path = config_path
        self._config: Optional[RunnerConfig] = None
        self._bot: Optional[BaseBot] = None
        self._redis = None
        self._publisher: Optional[IPCPublisher] = None
        self._subscriber: Optional[IPCSubscriber] = None
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()

    @property
    def bot_id(self) -> str:
        """Get bot ID."""
        return self._bot_id

    @property
    def is_running(self) -> bool:
        """Check if runner is running."""
        return self._running

    async def run(self) -> None:
        """
        Main entry point - run the bot process.

        1. Load configuration
        2. Initialize bot
        3. Initialize IPC
        4. Subscribe to commands
        5. Start heartbeat loop
        6. Start listening
        """
        try:
            logger.info(f"Starting BotRunner for {self._bot_id}")

            # Setup signal handlers
            self._setup_signals()

            # Load configuration
            self._config = self._load_config()
            logger.info(f"Configuration loaded: bot_type={self._config.bot_type}")

            # Initialize bot
            await self._init_bot()
            logger.info(f"Bot initialized: {self._bot.bot_type}")

            # Initialize IPC
            await self._init_ipc()
            logger.info("IPC initialized")

            # Subscribe to command channel
            await self._subscriber.subscribe(
                Channel.command(self._bot_id),
                self._handle_command,
            )
            logger.info(f"Subscribed to {Channel.command(self._bot_id)}")

            # Start heartbeat loop
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Mark as running and send started event
            self._running = True
            await self._send_event(Event.started(self._bot_id, {"pid": os.getpid()}))

            # Start listening
            await self._subscriber.start()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

        except Exception as e:
            logger.error(f"BotRunner error: {e}")
            await self._send_event(Event.error(self._bot_id, str(e)))
            raise
        finally:
            await self._shutdown()

    def _load_config(self) -> RunnerConfig:
        """
        Load configuration from file.

        Returns:
            RunnerConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(self._config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            if config_path.suffix in (".yaml", ".yml"):
                data = yaml.safe_load(f)
            else:
                import json
                data = json.load(f)

        # Validate required fields
        if "bot_type" not in data:
            raise ValueError("Config missing required field: bot_type")

        return RunnerConfig(
            bot_id=self._bot_id,
            bot_type=data["bot_type"],
            redis_url=data.get("redis_url", "redis://localhost:6379"),
            bot_config=data.get("config", {}),
        )

    async def _init_bot(self) -> None:
        """
        Initialize the bot instance.

        Raises:
            ValueError: If bot type is not registered
        """
        # Register builtin bots
        _register_builtin_bots()

        bot_class = get_bot_class(self._config.bot_type)
        if bot_class is None:
            raise ValueError(f"Unknown bot type: {self._config.bot_type}")

        # Create mock dependencies for now
        # In production, these would be properly initialized
        from unittest.mock import MagicMock

        exchange = MagicMock()
        data_manager = MagicMock()
        notifier = MagicMock()

        # Try to create bot with the config
        try:
            self._bot = bot_class(
                bot_id=self._bot_id,
                config=self._config.bot_config,
                exchange=exchange,
                data_manager=data_manager,
                notifier=notifier,
                heartbeat_callback=None,  # We handle heartbeat ourselves
            )
        except Exception as e:
            logger.error(f"Failed to create bot: {e}")
            raise

    async def _init_ipc(self) -> None:
        """Initialize IPC publisher and subscriber."""
        try:
            import redis.asyncio as redis
        except ImportError:
            logger.error("redis package not installed. Install with: pip install redis")
            raise

        self._redis = redis.from_url(self._config.redis_url)
        self._publisher = IPCPublisher(self._redis)
        self._subscriber = IPCSubscriber(self._redis)

    async def _handle_command(self, data: str) -> None:
        """
        Handle incoming command from Master.

        Args:
            data: JSON string containing Command
        """
        cmd = Command.from_json(data)
        logger.info(f"Received command: {cmd.type.value} (id={cmd.id})")

        try:
            result = await self._execute_command(cmd)
            response = Response.success_response(cmd.id, result)
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            response = Response.error_response(cmd.id, str(e))

        await self._publisher.send_response(self._bot_id, response)

    async def _execute_command(self, cmd: Command) -> Dict[str, Any]:
        """
        Execute a command and return result.

        Args:
            cmd: Command to execute

        Returns:
            Result dictionary

        Raises:
            Exception: If command execution fails
        """
        if cmd.type == CommandType.START:
            await self._bot.start()
            return {"status": "started", "state": self._bot.state.value}

        elif cmd.type == CommandType.STOP:
            clear_position = cmd.params.get("clear_position", False)
            await self._bot.stop(clear_position=clear_position)
            return {"status": "stopped", "state": self._bot.state.value}

        elif cmd.type == CommandType.PAUSE:
            await self._bot.pause()
            return {"status": "paused", "state": self._bot.state.value}

        elif cmd.type == CommandType.RESUME:
            await self._bot.resume()
            return {"status": "resumed", "state": self._bot.state.value}

        elif cmd.type == CommandType.STATUS:
            return self._bot.get_status()

        elif cmd.type == CommandType.SHUTDOWN:
            # Signal shutdown
            self._shutdown_event.set()
            return {"status": "shutdown_initiated"}

        else:
            raise ValueError(f"Unknown command type: {cmd.type}")

    async def _heartbeat_loop(self) -> None:
        """
        Send periodic heartbeats to Master.

        Runs every 10 seconds while running.
        """
        while self._running:
            try:
                heartbeat = Heartbeat(
                    bot_id=self._bot_id,
                    state=self._bot.state.value if self._bot else "unknown",
                    pid=os.getpid(),
                    metrics=self._get_metrics(),
                )
                await self._publisher.send_heartbeat(self._bot_id, heartbeat)
                logger.debug(f"Heartbeat sent: state={heartbeat.state}")
            except Exception as e:
                logger.warning(f"Failed to send heartbeat: {e}")

            await asyncio.sleep(10)

    def _get_metrics(self) -> Dict[str, Any]:
        """Get current metrics for heartbeat."""
        if not self._bot:
            return {}

        return {
            "uptime_seconds": self._bot._get_uptime_seconds(),
            "total_trades": self._bot.stats.total_trades,
            "total_profit": float(self._bot.stats.total_profit),
            "today_trades": self._bot.stats.today_trades,
            "today_profit": float(self._bot.stats.today_profit),
        }

    async def _send_event(self, event: Event) -> None:
        """Send an event to Master."""
        if self._publisher:
            try:
                await self._publisher.send_event(event)
            except Exception as e:
                logger.warning(f"Failed to send event: {e}")

    async def _shutdown(self) -> None:
        """Shutdown the runner and cleanup resources."""
        logger.info(f"Shutting down BotRunner for {self._bot_id}")
        self._running = False

        # Cancel heartbeat task
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Stop bot if running
        if self._bot and self._bot.is_running:
            try:
                await self._bot.stop(reason="Runner shutdown")
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")

        # Stop subscriber
        if self._subscriber:
            await self._subscriber.stop()

        # Send stopped event
        await self._send_event(Event.stopped(self._bot_id, "Runner shutdown"))

        # Close Redis connection
        if self._redis:
            await self._redis.close()

        logger.info(f"BotRunner shutdown complete for {self._bot_id}")

    def _setup_signals(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        loop = asyncio.get_event_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda: asyncio.create_task(self._signal_handler(sig)),
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                signal.signal(sig, lambda s, f: asyncio.create_task(self._signal_handler(s)))

    async def _signal_handler(self, sig) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {sig}, initiating shutdown")
        self._shutdown_event.set()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run a trading bot as an independent process"
    )
    parser.add_argument(
        "--bot-id",
        required=True,
        help="Unique bot identifier",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration file (YAML or JSON)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    import logging
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Run the bot
    runner = BotRunner(args.bot_id, args.config)
    try:
        asyncio.run(runner.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Runner failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
