"""
Heartbeat Monitor.

Monitors bot heartbeats and detects timeouts for automated health tracking.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional, Protocol

from src.core import get_logger

from .models import BotState
from .registry import BotRegistry

logger = get_logger(__name__)


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def notify_bot_timeout(
        self, bot_id: str, missed_count: int, last_heartbeat: Optional[datetime]
    ) -> None: ...

    async def notify_bot_recovered(self, bot_id: str) -> None: ...


@dataclass
class HeartbeatConfig:
    """
    Heartbeat monitoring configuration.

    Attributes:
        interval: Heartbeat check interval in seconds
        timeout: Timeout threshold in seconds
        max_missed: Maximum missed heartbeats before action
        auto_restart: Whether to auto-restart timed out bots
    """

    interval: int = 10
    timeout: int = 60
    max_missed: int = 3
    auto_restart: bool = False


@dataclass
class HeartbeatData:
    """
    Heartbeat data from a bot.

    Attributes:
        bot_id: Bot identifier
        timestamp: Heartbeat timestamp
        state: Current bot state
        metrics: Performance metrics
    """

    bot_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    state: BotState = BotState.RUNNING
    metrics: dict[str, Any] = field(default_factory=dict)

    @property
    def uptime_seconds(self) -> int:
        """Get uptime from metrics."""
        return self.metrics.get("uptime_seconds", 0)

    @property
    def total_trades(self) -> int:
        """Get total trades from metrics."""
        return self.metrics.get("total_trades", 0)

    @property
    def total_profit(self) -> Decimal:
        """Get total profit from metrics."""
        profit = self.metrics.get("total_profit", 0)
        return Decimal(str(profit)) if not isinstance(profit, Decimal) else profit

    @property
    def pending_orders(self) -> int:
        """Get pending orders from metrics."""
        return self.metrics.get("pending_orders", 0)

    @property
    def memory_mb(self) -> float:
        """Get memory usage from metrics."""
        return self.metrics.get("memory_mb", 0.0)

    @property
    def cpu_percent(self) -> float:
        """Get CPU usage from metrics."""
        return self.metrics.get("cpu_percent", 0.0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "timestamp": self.timestamp.isoformat(),
            "state": self.state.value,
            "metrics": self.metrics,
        }


class HeartbeatMonitor:
    """
    Heartbeat Monitor.

    Monitors bot heartbeats and detects timeouts.

    Example:
        >>> monitor = HeartbeatMonitor(registry, config, notifier)
        >>> await monitor.start()
        >>> monitor.receive(HeartbeatData(bot_id="bot_001", ...))
        >>> await monitor.stop()
    """

    def __init__(
        self,
        registry: BotRegistry,
        config: Optional[HeartbeatConfig] = None,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize HeartbeatMonitor.

        Args:
            registry: BotRegistry instance
            config: HeartbeatConfig (uses defaults if not provided)
            notifier: Optional notification manager
        """
        self._registry = registry
        self._config = config or HeartbeatConfig()
        self._notifier = notifier

        self._heartbeats: dict[str, HeartbeatData] = {}
        self._missed_counts: dict[str, int] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

    @property
    def config(self) -> HeartbeatConfig:
        """Get configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """Start the heartbeat monitoring loop."""
        if self._running:
            logger.warning("HeartbeatMonitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(
            f"HeartbeatMonitor started (interval={self._config.interval}s, "
            f"timeout={self._config.timeout}s)"
        )

    async def stop(self) -> None:
        """Stop the heartbeat monitoring loop."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("HeartbeatMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self._running:
            try:
                await self.check_all()
                await asyncio.sleep(self._config.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in heartbeat monitor loop: {e}")
                await asyncio.sleep(self._config.interval)

    # =========================================================================
    # Core Methods
    # =========================================================================

    def receive(self, heartbeat: HeartbeatData) -> None:
        """
        Receive a heartbeat from a bot.

        Args:
            heartbeat: HeartbeatData from the bot
        """
        bot_id = heartbeat.bot_id

        # Check if this was a previously timed out bot
        was_timed_out = self._missed_counts.get(bot_id, 0) >= self._config.max_missed

        # Update heartbeat record
        self._heartbeats[bot_id] = heartbeat

        # Reset missed count
        self._missed_counts[bot_id] = 0

        # Update registry heartbeat timestamp
        asyncio.create_task(self._registry.update_heartbeat(bot_id))

        # Notify recovery if bot was timed out
        if was_timed_out and self._notifier:
            asyncio.create_task(self._notify_recovered(bot_id))

        logger.debug(
            f"Heartbeat received: {bot_id} "
            f"(state={heartbeat.state.value}, trades={heartbeat.total_trades})"
        )

    async def check_all(self) -> dict[str, bool]:
        """
        Check all running bots for heartbeat timeout.

        Returns:
            Dict mapping bot_id to alive status
        """
        results: dict[str, bool] = {}
        now = datetime.now(timezone.utc)

        # Get all running bots
        running_bots = self._registry.get_by_state(BotState.RUNNING)

        for bot_info in running_bots:
            bot_id = bot_info.bot_id
            last_heartbeat = self._heartbeats.get(bot_id)

            if last_heartbeat is None:
                # Never received heartbeat
                self._missed_counts[bot_id] = self._missed_counts.get(bot_id, 0) + 1
                results[bot_id] = False
                logger.warning(f"Bot {bot_id}: no heartbeat received")

            elif (now - last_heartbeat.timestamp).total_seconds() > self._config.timeout:
                # Heartbeat timed out
                self._missed_counts[bot_id] = self._missed_counts.get(bot_id, 0) + 1
                results[bot_id] = False
                elapsed = (now - last_heartbeat.timestamp).total_seconds()
                logger.warning(f"Bot {bot_id}: heartbeat timeout ({elapsed:.1f}s)")

            else:
                # Normal
                results[bot_id] = True
                continue

            # Check if max missed reached
            if self._missed_counts[bot_id] >= self._config.max_missed:
                await self._handle_timeout(bot_id)

        return results

    def is_alive(self, bot_id: str) -> bool:
        """
        Check if a bot is alive (has recent heartbeat).

        Args:
            bot_id: Bot identifier

        Returns:
            True if bot has recent heartbeat
        """
        last_heartbeat = self._heartbeats.get(bot_id)
        if last_heartbeat is None:
            return False

        elapsed = (datetime.now(timezone.utc) - last_heartbeat.timestamp).total_seconds()
        return elapsed <= self._config.timeout

    def get_status(self, bot_id: str) -> dict[str, Any]:
        """
        Get heartbeat status for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Dict with heartbeat status information
        """
        last_heartbeat = self._heartbeats.get(bot_id)
        missed_count = self._missed_counts.get(bot_id, 0)

        if last_heartbeat is None:
            return {
                "bot_id": bot_id,
                "alive": False,
                "last_heartbeat": None,
                "elapsed_seconds": None,
                "missed_count": missed_count,
                "metrics": {},
            }

        now = datetime.now(timezone.utc)
        elapsed = (now - last_heartbeat.timestamp).total_seconds()

        return {
            "bot_id": bot_id,
            "alive": elapsed <= self._config.timeout,
            "last_heartbeat": last_heartbeat.timestamp,
            "elapsed_seconds": elapsed,
            "missed_count": missed_count,
            "state": last_heartbeat.state.value,
            "metrics": last_heartbeat.metrics,
        }

    def get_all_status(self) -> list[dict[str, Any]]:
        """
        Get heartbeat status for all monitored bots.

        Returns:
            List of status dicts for all bots
        """
        return [
            self.get_status(bot_id) for bot_id in self._heartbeats.keys()
        ]

    def get_missed_count(self, bot_id: str) -> int:
        """
        Get missed heartbeat count for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Number of missed heartbeats
        """
        return self._missed_counts.get(bot_id, 0)

    def clear(self, bot_id: str) -> None:
        """
        Clear heartbeat data for a bot.

        Args:
            bot_id: Bot identifier
        """
        self._heartbeats.pop(bot_id, None)
        self._missed_counts.pop(bot_id, None)

    # =========================================================================
    # Timeout Handling
    # =========================================================================

    async def _handle_timeout(self, bot_id: str) -> None:
        """
        Handle bot heartbeat timeout.

        Args:
            bot_id: Bot identifier
        """
        logger.error(
            f"Bot {bot_id} heartbeat timeout "
            f"(missed={self._missed_counts.get(bot_id, 0)})"
        )

        # Send alert notification
        if self._notifier:
            last_heartbeat = self._heartbeats.get(bot_id)
            last_time = last_heartbeat.timestamp if last_heartbeat else None
            try:
                await self._notifier.notify_bot_timeout(
                    bot_id=bot_id,
                    missed_count=self._missed_counts.get(bot_id, 0),
                    last_heartbeat=last_time,
                )
            except Exception as e:
                logger.warning(f"Failed to send timeout notification: {e}")

        # Mark bot state as ERROR
        try:
            await self._registry.update_state(
                bot_id=bot_id,
                new_state=BotState.ERROR,
                message=f"Heartbeat timeout (missed {self._missed_counts.get(bot_id, 0)} heartbeats)",
            )
        except Exception as e:
            logger.warning(f"Failed to update bot state to ERROR: {e}")

        # Auto-restart if enabled
        if self._config.auto_restart:
            await self._attempt_restart(bot_id)

    async def _attempt_restart(self, bot_id: str) -> bool:
        """
        Attempt to restart a timed out bot.

        Args:
            bot_id: Bot identifier

        Returns:
            True if restart successful
        """
        logger.info(f"Attempting to restart bot: {bot_id}")

        try:
            # Get bot instance
            instance = self._registry.get_bot_instance(bot_id)
            if instance is None:
                logger.warning(f"Cannot restart bot {bot_id}: no instance bound")
                return False

            # Stop the bot first
            await instance.stop()

            # Reset state to STOPPED
            await self._registry.update_state(bot_id, BotState.STOPPED)

            # Clear heartbeat data
            self.clear(bot_id)

            # Start the bot again
            # Note: This assumes the bot instance has a start() method
            if hasattr(instance, "start"):
                await instance.start()
                logger.info(f"Bot {bot_id} restarted successfully")
                return True

            return False

        except Exception as e:
            logger.error(f"Failed to restart bot {bot_id}: {e}")
            return False

    async def _notify_recovered(self, bot_id: str) -> None:
        """
        Send recovery notification.

        Args:
            bot_id: Bot identifier
        """
        if self._notifier:
            try:
                await self._notifier.notify_bot_recovered(bot_id)
            except Exception as e:
                logger.warning(f"Failed to send recovery notification: {e}")

    # =========================================================================
    # Container Methods
    # =========================================================================

    def __contains__(self, bot_id: str) -> bool:
        """Check if bot has heartbeat data."""
        return bot_id in self._heartbeats

    def __len__(self) -> int:
        """Get number of monitored bots."""
        return len(self._heartbeats)
