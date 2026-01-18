"""
Base Bot Abstract Class.

Provides common interface for all trading bots with multi-process support.
"""

import asyncio
import signal
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from src.core import get_logger

logger = get_logger(__name__)


class BotStatus(str, Enum):
    """Bot operational status."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class BotMetrics:
    """Bot performance metrics for heartbeat reporting."""

    uptime_seconds: int = 0
    total_trades: int = 0
    total_profit: Decimal = field(default_factory=lambda: Decimal("0"))
    pending_orders: int = 0
    last_trade_at: Optional[datetime] = None
    memory_mb: float = 0.0
    cpu_percent: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uptime_seconds": self.uptime_seconds,
            "total_trades": self.total_trades,
            "total_profit": str(self.total_profit),
            "pending_orders": self.pending_orders,
            "last_trade_at": self.last_trade_at.isoformat() if self.last_trade_at else None,
            "memory_mb": self.memory_mb,
            "cpu_percent": self.cpu_percent,
        }


class BaseBot(ABC):
    """
    Abstract base class for all trading bots.

    Provides:
    - Common lifecycle management (start, stop, pause, resume)
    - Heartbeat reporting for multi-process monitoring
    - Signal handling for graceful shutdown
    - Standard interface for Master to manage

    Example:
        class GridBot(BaseBot):
            async def _on_start(self):
                # Initialize grid, place orders
                pass

            async def _on_stop(self):
                # Cancel orders, save state
                pass

            async def _run_iteration(self):
                # Main trading logic
                pass
    """

    def __init__(
        self,
        bot_id: str,
        config: dict[str, Any],
        heartbeat_interval: int = 10,
    ):
        """
        Initialize BaseBot.

        Args:
            bot_id: Unique bot identifier
            config: Bot configuration
            heartbeat_interval: Heartbeat interval in seconds
        """
        self._bot_id = bot_id
        self._config = config
        self._heartbeat_interval = heartbeat_interval

        self._status = BotStatus.INITIALIZING
        self._start_time: Optional[datetime] = None
        self._metrics = BotMetrics()
        self._running = False

        # Tasks
        self._main_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Redis connection for IPC (optional)
        self._redis: Optional[Any] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def bot_id(self) -> str:
        """Get bot ID."""
        return self._bot_id

    @property
    def status(self) -> BotStatus:
        """Get current status."""
        return self._status

    @property
    def config(self) -> dict[str, Any]:
        """Get configuration."""
        return self._config

    @property
    def start_time(self) -> Optional[datetime]:
        """Get start time."""
        return self._start_time

    @property
    def is_running(self) -> bool:
        """Check if bot is running."""
        return self._running and self._status == BotStatus.RUNNING

    @property
    def metrics(self) -> BotMetrics:
        """Get current metrics."""
        return self._metrics

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> bool:
        """
        Start the bot.

        Returns:
            True if started successfully
        """
        if self._running:
            logger.warning(f"Bot {self._bot_id} already running")
            return False

        try:
            logger.info(f"Starting bot: {self._bot_id}")
            self._status = BotStatus.INITIALIZING

            # Setup signal handlers
            self._setup_signal_handlers()

            # Initialize Redis for IPC (if configured)
            await self._setup_redis()

            # Call subclass initialization
            await self._on_start()

            # Update state
            self._status = BotStatus.RUNNING
            self._running = True
            self._start_time = datetime.now(timezone.utc)

            # Start main loop and heartbeat
            self._main_task = asyncio.create_task(self._main_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info(f"Bot {self._bot_id} started successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot {self._bot_id}: {e}")
            self._status = BotStatus.ERROR
            return False

    async def stop(self, reason: str = "Manual stop") -> bool:
        """
        Stop the bot.

        Args:
            reason: Stop reason for logging

        Returns:
            True if stopped successfully
        """
        if not self._running:
            return True

        try:
            logger.info(f"Stopping bot {self._bot_id}: {reason}")
            self._status = BotStatus.STOPPING
            self._running = False

            # Cancel tasks
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

            if self._main_task:
                self._main_task.cancel()
                try:
                    await self._main_task
                except asyncio.CancelledError:
                    pass

            # Call subclass cleanup
            await self._on_stop()

            # Close Redis
            await self._close_redis()

            self._status = BotStatus.STOPPED
            logger.info(f"Bot {self._bot_id} stopped")
            return True

        except Exception as e:
            logger.error(f"Error stopping bot {self._bot_id}: {e}")
            self._status = BotStatus.ERROR
            return False

    async def pause(self, reason: str = "Manual pause") -> bool:
        """
        Pause the bot.

        Args:
            reason: Pause reason

        Returns:
            True if paused successfully
        """
        if self._status != BotStatus.RUNNING:
            return False

        logger.info(f"Pausing bot {self._bot_id}: {reason}")
        self._status = BotStatus.PAUSED
        await self._on_pause()
        return True

    async def resume(self) -> bool:
        """
        Resume the bot from paused state.

        Returns:
            True if resumed successfully
        """
        if self._status != BotStatus.PAUSED:
            return False

        logger.info(f"Resuming bot {self._bot_id}")
        self._status = BotStatus.RUNNING
        await self._on_resume()
        return True

    # =========================================================================
    # Abstract Methods (Subclass Must Implement)
    # =========================================================================

    @abstractmethod
    async def _on_start(self) -> None:
        """Called when bot starts. Initialize resources here."""
        pass

    @abstractmethod
    async def _on_stop(self) -> None:
        """Called when bot stops. Cleanup resources here."""
        pass

    @abstractmethod
    async def _run_iteration(self) -> None:
        """Main trading logic. Called in each iteration of the main loop."""
        pass

    # =========================================================================
    # Optional Hooks
    # =========================================================================

    async def _on_pause(self) -> None:
        """Called when bot is paused. Override if needed."""
        pass

    async def _on_resume(self) -> None:
        """Called when bot resumes. Override if needed."""
        pass

    async def _on_command(self, command: str, data: dict[str, Any]) -> Any:
        """
        Handle command from Master.

        Args:
            command: Command name
            data: Command data

        Returns:
            Response data
        """
        if command == "status":
            return self._get_status_dict()
        elif command == "pause":
            return await self.pause(data.get("reason", "Master command"))
        elif command == "resume":
            return await self.resume()
        elif command == "stop":
            return await self.stop(data.get("reason", "Master command"))
        else:
            logger.warning(f"Unknown command: {command}")
            return None

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _main_loop(self) -> None:
        """Main bot loop."""
        while self._running:
            try:
                if self._status == BotStatus.RUNNING:
                    await self._run_iteration()

                # Check for commands from Master
                await self._check_commands()

                # Small delay to prevent busy loop
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1)

    # =========================================================================
    # Heartbeat
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Send heartbeat to Master periodically."""
        while self._running:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self._heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
                await asyncio.sleep(self._heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to Master via Redis."""
        if not self._redis:
            return

        # Update metrics
        self._update_metrics()

        heartbeat = {
            "bot_id": self._bot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": self._status.value,
            "metrics": self._metrics.to_dict(),
        }

        try:
            import json
            await self._redis.publish(
                f"bot:{self._bot_id}:heartbeat",
                json.dumps(heartbeat),
            )
        except Exception as e:
            logger.debug(f"Failed to send heartbeat: {e}")

    def _update_metrics(self) -> None:
        """Update bot metrics. Override for custom metrics."""
        if self._start_time:
            elapsed = datetime.now(timezone.utc) - self._start_time
            self._metrics.uptime_seconds = int(elapsed.total_seconds())

    # =========================================================================
    # Redis IPC
    # =========================================================================

    async def _setup_redis(self) -> None:
        """Setup Redis connection for IPC."""
        redis_url = self._config.get("redis_url")
        if not redis_url:
            return

        try:
            import redis.asyncio as redis
            self._redis = redis.from_url(redis_url)
            logger.info(f"Connected to Redis: {redis_url}")
        except ImportError:
            logger.warning("redis package not installed, IPC disabled")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")

    async def _close_redis(self) -> None:
        """Close Redis connection."""
        if self._redis:
            await self._redis.close()
            self._redis = None

    async def _check_commands(self) -> None:
        """Check for commands from Master via Redis."""
        if not self._redis:
            return

        try:
            import json
            message = await self._redis.lpop(f"bot:{self._bot_id}:commands")
            if message:
                data = json.loads(message)
                command = data.get("command")
                response = await self._on_command(command, data)

                # Send response
                if response is not None:
                    await self._redis.publish(
                        f"bot:{self._bot_id}:response",
                        json.dumps({"command": command, "response": response}),
                    )
        except Exception as e:
            logger.debug(f"Command check error: {e}")

    # =========================================================================
    # Signal Handling
    # =========================================================================

    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown."""
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self._handle_signal(s)),
                )

    async def _handle_signal(self, sig: signal.Signals) -> None:
        """Handle shutdown signal."""
        logger.info(f"Received signal {sig.name}, shutting down...")
        await self.stop(f"Signal {sig.name}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_status_dict(self) -> dict[str, Any]:
        """Get status as dictionary."""
        return {
            "bot_id": self._bot_id,
            "status": self._status.value,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "metrics": self._metrics.to_dict(),
        }

    # =========================================================================
    # Entry Point for Multi-Process
    # =========================================================================

    @classmethod
    def run_standalone(cls, bot_id: str, config: dict[str, Any]) -> None:
        """
        Run bot as standalone process.

        This is the entry point when bot is started as a subprocess.

        Args:
            bot_id: Bot identifier
            config: Bot configuration

        Example:
            # Start bot as subprocess
            python -m bots.grid --bot-id grid_001 --config config.json
        """
        async def main():
            bot = cls(bot_id, config)
            await bot.start()

            # Wait until stopped
            while bot.is_running:
                await asyncio.sleep(1)

        asyncio.run(main())
