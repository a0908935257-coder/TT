"""
Heartbeat Monitor.

Monitors bot heartbeats and detects timeouts for automated health tracking.
Includes RestartManager for controlled automatic restarts with circuit breaker.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

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

    async def notify_circuit_breaker_open(
        self, bot_id: str, consecutive_failures: int, duration: int
    ) -> None: ...

    async def notify_circuit_breaker_closed(self, bot_id: str) -> None: ...


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
class RestartConfig:
    """
    Restart limiting configuration.

    Prevents restart storms by limiting restart frequency and
    implementing circuit breaker pattern.

    Attributes:
        max_restarts_per_window: Maximum restarts within time window
        window_seconds: Time window for counting restarts (default: 5 minutes)
        initial_backoff: Initial backoff delay after failed restart
        max_backoff: Maximum backoff delay
        backoff_multiplier: Multiplier for exponential backoff
        circuit_break_threshold: Consecutive failures to trigger circuit breaker
        circuit_break_duration: How long circuit stays open (seconds)
    """

    max_restarts_per_window: int = 3
    window_seconds: int = 300  # 5 minutes
    initial_backoff: float = 5.0
    max_backoff: float = 300.0  # 5 minutes
    backoff_multiplier: float = 2.0
    circuit_break_threshold: int = 3
    circuit_break_duration: int = 600  # 10 minutes


@dataclass
class RestartTracker:
    """
    Tracks restart attempts for a single bot.

    Attributes:
        restart_timestamps: List of restart attempt timestamps
        restart_count: Total restarts in current window
        consecutive_failures: Number of consecutive failed restarts
        last_restart_attempt: Timestamp of last restart attempt
        circuit_open_until: When circuit breaker will close (None = closed)
        backoff_until: When backoff period ends (None = no backoff)
        current_backoff: Current backoff duration in seconds
    """

    restart_timestamps: List[datetime] = field(default_factory=list)
    restart_count: int = 0
    consecutive_failures: int = 0
    last_restart_attempt: Optional[datetime] = None
    circuit_open_until: Optional[datetime] = None
    backoff_until: Optional[datetime] = None
    current_backoff: float = 0.0

    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open."""
        if self.circuit_open_until is None:
            return False
        return datetime.now(timezone.utc) < self.circuit_open_until

    def is_in_backoff(self) -> bool:
        """Check if currently in backoff period."""
        if self.backoff_until is None:
            return False
        return datetime.now(timezone.utc) < self.backoff_until


class RestartManager:
    """
    Manages automatic restart attempts with rate limiting and circuit breaker.

    Prevents restart storms by:
    1. Limiting restarts per time window
    2. Exponential backoff between retries
    3. Circuit breaker for repeated failures

    Example:
        >>> manager = RestartManager(config, notifier)
        >>> can_restart, reason = await manager.can_restart("bot_001")
        >>> if can_restart:
        ...     success = await manager.execute_restart("bot_001", restart_func)
    """

    def __init__(
        self,
        config: Optional[RestartConfig] = None,
        notifier: Optional[NotifierProtocol] = None,
    ):
        """
        Initialize RestartManager.

        Args:
            config: Restart configuration
            notifier: Notification manager for alerts
        """
        self._config = config or RestartConfig()
        self._notifier = notifier
        self._trackers: Dict[str, RestartTracker] = {}

    def _get_tracker(self, bot_id: str) -> RestartTracker:
        """Get or create tracker for a bot."""
        if bot_id not in self._trackers:
            self._trackers[bot_id] = RestartTracker()
        return self._trackers[bot_id]

    def _clean_old_timestamps(self, tracker: RestartTracker) -> None:
        """Remove timestamps outside the window."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - self._config.window_seconds
        tracker.restart_timestamps = [
            ts for ts in tracker.restart_timestamps
            if ts.timestamp() > cutoff
        ]
        tracker.restart_count = len(tracker.restart_timestamps)

    async def can_restart(self, bot_id: str) -> Tuple[bool, str]:
        """
        Check if a bot can be restarted.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (can_restart, reason_if_not)
        """
        tracker = self._get_tracker(bot_id)

        # Check circuit breaker
        if tracker.is_circuit_open():
            remaining = (tracker.circuit_open_until - datetime.now(timezone.utc)).seconds
            return False, f"Circuit breaker open, {remaining}s remaining"

        # Check backoff
        if tracker.is_in_backoff():
            remaining = (tracker.backoff_until - datetime.now(timezone.utc)).seconds
            return False, f"In backoff period, {remaining}s remaining"

        # Clean old timestamps and check rate limit
        self._clean_old_timestamps(tracker)
        if tracker.restart_count >= self._config.max_restarts_per_window:
            return False, (
                f"Rate limit: {tracker.restart_count} restarts in "
                f"{self._config.window_seconds}s window"
            )

        return True, ""

    async def execute_restart(
        self,
        bot_id: str,
        restart_func: Callable[[], Any],
    ) -> bool:
        """
        Execute restart with tracking.

        Args:
            bot_id: Bot identifier
            restart_func: Async function to restart the bot

        Returns:
            True if restart succeeded
        """
        can_restart, reason = await self.can_restart(bot_id)
        if not can_restart:
            logger.warning(f"Cannot restart {bot_id}: {reason}")
            return False

        tracker = self._get_tracker(bot_id)
        now = datetime.now(timezone.utc)

        # Record attempt
        tracker.restart_timestamps.append(now)
        tracker.restart_count += 1
        tracker.last_restart_attempt = now

        logger.info(
            f"Executing restart for {bot_id} "
            f"(attempt #{tracker.restart_count} in window)"
        )

        try:
            # Execute restart
            result = restart_func()
            if asyncio.iscoroutine(result):
                success = await result
            else:
                success = result

            if success:
                # Reset failure tracking on success
                tracker.consecutive_failures = 0
                tracker.current_backoff = 0.0
                tracker.backoff_until = None
                logger.info(f"Restart successful for {bot_id}")
                return True
            else:
                # Handle failure
                await self._handle_restart_failure(bot_id, tracker)
                return False

        except Exception as e:
            logger.error(f"Restart failed for {bot_id}: {e}")
            await self._handle_restart_failure(bot_id, tracker)
            return False

    async def _handle_restart_failure(
        self,
        bot_id: str,
        tracker: RestartTracker,
    ) -> None:
        """Handle a failed restart attempt."""
        tracker.consecutive_failures += 1

        # Calculate backoff
        if tracker.current_backoff == 0:
            tracker.current_backoff = self._config.initial_backoff
        else:
            tracker.current_backoff = min(
                tracker.current_backoff * self._config.backoff_multiplier,
                self._config.max_backoff,
            )

        tracker.backoff_until = datetime.now(timezone.utc) + \
            asyncio.get_event_loop().time().__class__(
                seconds=tracker.current_backoff
            ).__class__.__bases__[0].__call__(
                seconds=tracker.current_backoff
            ) if False else None

        # Use timedelta for backoff calculation
        from datetime import timedelta
        tracker.backoff_until = datetime.now(timezone.utc) + timedelta(
            seconds=tracker.current_backoff
        )

        logger.warning(
            f"Restart failed for {bot_id}, "
            f"consecutive failures: {tracker.consecutive_failures}, "
            f"backoff: {tracker.current_backoff}s"
        )

        # Check circuit breaker threshold
        if tracker.consecutive_failures >= self._config.circuit_break_threshold:
            await self._open_circuit(bot_id, tracker)

    async def _open_circuit(
        self,
        bot_id: str,
        tracker: RestartTracker,
    ) -> None:
        """Open circuit breaker for a bot."""
        from datetime import timedelta

        tracker.circuit_open_until = datetime.now(timezone.utc) + timedelta(
            seconds=self._config.circuit_break_duration
        )

        logger.error(
            f"Circuit breaker OPEN for {bot_id}: "
            f"{tracker.consecutive_failures} consecutive failures, "
            f"will close in {self._config.circuit_break_duration}s"
        )

        # Send notification
        if self._notifier:
            try:
                await self._notifier.notify_circuit_breaker_open(
                    bot_id=bot_id,
                    consecutive_failures=tracker.consecutive_failures,
                    duration=self._config.circuit_break_duration,
                )
            except Exception as e:
                logger.warning(f"Failed to send circuit breaker notification: {e}")

    def reset_tracker(self, bot_id: str) -> None:
        """
        Reset restart tracker for a bot.

        Call this when a bot successfully recovers or is manually restarted.

        Args:
            bot_id: Bot identifier
        """
        if bot_id in self._trackers:
            self._trackers[bot_id] = RestartTracker()
            logger.info(f"Restart tracker reset for {bot_id}")

    def get_status(self, bot_id: str) -> Dict[str, Any]:
        """
        Get restart status for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Status dictionary
        """
        tracker = self._get_tracker(bot_id)
        self._clean_old_timestamps(tracker)

        return {
            "bot_id": bot_id,
            "restarts_in_window": tracker.restart_count,
            "consecutive_failures": tracker.consecutive_failures,
            "circuit_open": tracker.is_circuit_open(),
            "circuit_open_until": (
                tracker.circuit_open_until.isoformat()
                if tracker.circuit_open_until else None
            ),
            "in_backoff": tracker.is_in_backoff(),
            "backoff_until": (
                tracker.backoff_until.isoformat()
                if tracker.backoff_until else None
            ),
            "current_backoff": tracker.current_backoff,
            "last_restart_attempt": (
                tracker.last_restart_attempt.isoformat()
                if tracker.last_restart_attempt else None
            ),
        }

    def get_all_status(self) -> List[Dict[str, Any]]:
        """Get restart status for all tracked bots."""
        return [self.get_status(bot_id) for bot_id in self._trackers]


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
        restart_config: Optional[RestartConfig] = None,
    ):
        """
        Initialize HeartbeatMonitor.

        Args:
            registry: BotRegistry instance
            config: HeartbeatConfig (uses defaults if not provided)
            notifier: Optional notification manager
            restart_config: RestartConfig for restart limiting (uses defaults if not provided)
        """
        self._registry = registry
        self._config = config or HeartbeatConfig()
        self._notifier = notifier

        self._heartbeats: dict[str, HeartbeatData] = {}
        self._missed_counts: dict[str, int] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Timeout callback registration
        self._timeout_callbacks: List[Callable[[str], Any]] = []

        # Initialize RestartManager for controlled restarts
        self._restart_manager = RestartManager(
            config=restart_config or RestartConfig(),
            notifier=notifier,
        )

    @property
    def config(self) -> HeartbeatConfig:
        """Get configuration."""
        return self._config

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def restart_manager(self) -> RestartManager:
        """Get restart manager."""
        return self._restart_manager

    def on_timeout(self, callback: Callable[[str], Any]) -> None:
        """
        Register a callback for timeout events.

        Args:
            callback: Function to call with bot_id when timeout occurs
        """
        self._timeout_callbacks.append(callback)

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
        Attempt to restart a timed out bot with rate limiting and circuit breaker.

        Uses RestartManager to:
        - Limit restarts per time window
        - Implement exponential backoff on failures
        - Circuit breaker for repeated failures

        Args:
            bot_id: Bot identifier

        Returns:
            True if restart successful
        """
        # Check if restart is allowed
        can_restart, reason = await self._restart_manager.can_restart(bot_id)
        if not can_restart:
            logger.warning(f"Restart blocked for {bot_id}: {reason}")
            return False

        logger.info(f"Attempting to restart bot: {bot_id}")

        async def do_restart() -> bool:
            """Internal restart function."""
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
                if hasattr(instance, "start"):
                    await instance.start()
                    logger.info(f"Bot {bot_id} restarted successfully")
                    return True

                return False

            except Exception as e:
                logger.error(f"Failed to restart bot {bot_id}: {e}")
                return False

        # Execute restart through RestartManager
        success = await self._restart_manager.execute_restart(bot_id, do_restart)

        # If restart succeeded, also reset the tracker on first heartbeat
        if success:
            # Reset missed count
            self._missed_counts[bot_id] = 0

        return success

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
