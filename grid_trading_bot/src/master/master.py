"""
Master Control Console.

Main class integrating all bot management modules into a unified control interface.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, List, Optional, Protocol

from src.core import get_logger
from src.fund_manager import FundManager
from src.fund_manager.models.config import FundManagerConfig
from src.master.aggregator import MetricsAggregator
from src.master.commander import BotCommander, CommandResult
from src.master.dashboard import Dashboard, DashboardData
from src.master.factory import BotFactory
from src.master.health import HealthCheckResult, HealthChecker
from src.master.heartbeat import HeartbeatConfig, HeartbeatMonitor
from src.master.models import BotInfo, BotState, BotType
from src.master.registry import BotRegistry

if TYPE_CHECKING:
    from src.fund_manager.models.records import DispatchResult

logger = get_logger(__name__)


class ExchangeProtocol(Protocol):
    """Protocol for exchange client."""

    pass


class DataManagerProtocol(Protocol):
    """Protocol for data manager."""

    pass


class DatabaseManagerProtocol(Protocol):
    """Protocol for database manager."""

    async def get_all_bots(self) -> List[dict[str, Any]]: ...
    async def upsert_bot(self, data: dict[str, Any]) -> None: ...
    async def delete_bot(self, bot_id: str) -> bool: ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def send(self, message: str, **kwargs: Any) -> bool: ...


@dataclass
class MasterConfig:
    """
    Master configuration.

    Attributes:
        heartbeat: Heartbeat monitoring configuration
        auto_restart: Whether to automatically restart failed bots
        max_bots: Maximum number of bots allowed
        snapshot_interval: Dashboard snapshot interval in seconds
        restore_on_start: Whether to restore bots from database on start
        fund_manager: Fund manager configuration
    """

    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    auto_restart: bool = False
    max_bots: int = 100
    snapshot_interval: int = 3600
    restore_on_start: bool = True
    fund_manager: Optional[FundManagerConfig] = None


class Master:
    """
    Master Control Console (Singleton).

    Integrates all bot management modules:
    - BotRegistry: Bot registration and state tracking
    - BotFactory: Bot instance creation
    - BotCommander: Command execution
    - HeartbeatMonitor: Heartbeat monitoring
    - HealthChecker: Health checking
    - MetricsAggregator: Metrics collection
    - Dashboard: Monitoring dashboard

    Example:
        >>> master = Master(exchange, data_manager, db_manager, notifier)
        >>> await master.start()
        >>> result = await master.create_bot(BotType.GRID, config)
        >>> await master.start_bot(result.bot_id)
        >>> dashboard = master.get_dashboard()
        >>> await master.stop()
    """

    _instance: Optional["Master"] = None

    def __new__(cls, *args: Any, **kwargs: Any) -> "Master":
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        exchange: Optional[ExchangeProtocol] = None,
        data_manager: Optional[DataManagerProtocol] = None,
        db_manager: Optional[DatabaseManagerProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        config: Optional[MasterConfig] = None,
    ):
        """
        Initialize Master Control Console.

        Args:
            exchange: Exchange client instance
            data_manager: Market data manager instance
            db_manager: Database manager instance
            notifier: Notification manager instance
            config: Master configuration
        """
        # Only initialize once
        if hasattr(self, "_initialized") and self._initialized:
            return

        self._exchange = exchange
        self._data_manager = data_manager
        self._db_manager = db_manager
        self._notifier = notifier
        self._config = config or MasterConfig()

        # Initialize sub-modules
        self._registry = BotRegistry(db_manager, notifier)
        self._heartbeat = HeartbeatMonitor(
            self._registry,
            self._config.heartbeat,
            notifier,
        )
        self._factory = BotFactory(
            exchange, data_manager, notifier,
            heartbeat_callback=self._heartbeat.receive,
        )
        self._commander = BotCommander(self._registry, self._factory, notifier)
        self._health = HealthChecker(self._registry)
        self._aggregator = MetricsAggregator(self._registry)
        self._dashboard = Dashboard(self._registry, self._aggregator, self._health)

        # Initialize fund manager if config provided
        self._fund_manager: Optional[FundManager] = None
        if self._config.fund_manager:
            self._fund_manager = FundManager(
                exchange=exchange,
                registry=self._registry,
                notifier=notifier,
                config=self._config.fund_manager,
            )

        # State
        self._running = False
        self._snapshot_task: Optional[asyncio.Task] = None
        self._initialized = True

        logger.info("Master Control Console initialized")

    @classmethod
    def get_instance(cls) -> Optional["Master"]:
        """Get singleton instance if exists."""
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        if cls._instance is not None:
            cls._instance._initialized = False
        cls._instance = None
        BotRegistry.reset_instance()

    # =========================================================================
    # Properties (Read-only access to sub-modules)
    # =========================================================================

    @property
    def registry(self) -> BotRegistry:
        """Get bot registry."""
        return self._registry

    @property
    def commander(self) -> BotCommander:
        """Get bot commander."""
        return self._commander

    @property
    def dashboard(self) -> Dashboard:
        """Get dashboard."""
        return self._dashboard

    @property
    def factory(self) -> BotFactory:
        """Get bot factory."""
        return self._factory

    @property
    def health_checker(self) -> HealthChecker:
        """Get health checker."""
        return self._health

    @property
    def heartbeat_monitor(self) -> HeartbeatMonitor:
        """Get heartbeat monitor."""
        return self._heartbeat

    @property
    def is_running(self) -> bool:
        """Check if master is running."""
        return self._running

    @property
    def config(self) -> MasterConfig:
        """Get master configuration."""
        return self._config

    @property
    def fund_manager(self) -> Optional[FundManager]:
        """Get fund manager (if configured)."""
        return self._fund_manager

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def start(self) -> None:
        """
        Start the Master Control Console.

        Loads saved bots, restores running bots, and starts monitoring.
        """
        if self._running:
            logger.warning("Master is already running")
            return

        logger.info("Starting Master Control Console...")

        try:
            # Load saved bots from database
            if self._config.restore_on_start and self._db_manager:
                loaded = await self._registry.load_from_db()
                logger.info(f"Loaded {loaded} bots from database")

                # Restore previously running bots
                running_bots = self._registry.get_by_state(BotState.RUNNING)
                for bot_info in running_bots:
                    await self._restore_bot(bot_info.bot_id)

            # Start heartbeat monitoring
            await self._heartbeat.start()

            # Start dashboard snapshot loop
            if self._config.snapshot_interval > 0:
                await self._dashboard.start_snapshot_loop(self._config.snapshot_interval)

            # Register heartbeat timeout callback for auto-restart
            if self._config.auto_restart:
                self._heartbeat.on_timeout(self._handle_bot_timeout)

            # Start fund manager if configured
            if self._fund_manager:
                await self._fund_manager.start()

            self._running = True

            # Send startup notification
            await self._notify("Master Control Console started")

            logger.info("Master Control Console started successfully")

        except Exception as e:
            logger.error(f"Failed to start Master: {e}")
            raise

    async def stop(self) -> None:
        """
        Stop the Master Control Console.

        Stops all bots, monitoring, and saves state.
        """
        if not self._running:
            logger.warning("Master is not running")
            return

        logger.info("Stopping Master Control Console...")

        try:
            # Stop fund manager first
            if self._fund_manager:
                await self._fund_manager.stop()

            # Stop all running bots
            await self._commander.stop_all()

            # Stop heartbeat monitoring
            await self._heartbeat.stop()

            # Stop dashboard snapshot loop
            await self._dashboard.stop_snapshot_loop()

            # Save current state
            await self._save_state()

            self._running = False

            # Send shutdown notification
            await self._notify("Master Control Console stopped")

            logger.info("Master Control Console stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping Master: {e}")
            raise

    async def _restore_bot(self, bot_id: str) -> bool:
        """
        Restore a bot from saved state.

        Args:
            bot_id: Bot identifier

        Returns:
            True if restored successfully
        """
        try:
            bot_info = self._registry.get(bot_id)
            if not bot_info:
                logger.warning(f"Cannot restore bot {bot_id}: not found")
                return False

            # Create instance using factory
            instance = self._factory.create(
                bot_info.bot_type,
                bot_id,
                bot_info.config,
            )

            # Bind instance to registry
            self._registry.bind_instance(bot_id, instance)

            # Restore state from database if bot supports it
            if hasattr(instance, "restore_from_db"):
                await instance.restore_from_db()

            # Restart if was running
            if bot_info.state == BotState.RUNNING:
                await instance.start()
                logger.info(f"Restored and started bot: {bot_id}")
            else:
                logger.info(f"Restored bot: {bot_id} (state: {bot_info.state.value})")

            return True

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return False

    async def _save_state(self) -> None:
        """Save current state to database."""
        if not self._db_manager:
            return

        try:
            for bot_info in self._registry.get_all():
                await self._db_manager.upsert_bot(bot_info.to_dict())
            logger.info("State saved to database")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    async def _handle_bot_timeout(self, bot_id: str) -> None:
        """
        Handle bot timeout (for auto-restart).

        Args:
            bot_id: Bot identifier
        """
        if not self._config.auto_restart:
            return

        logger.warning(f"Bot {bot_id} timed out, attempting auto-restart")

        try:
            # Stop the bot first
            await self._commander.stop(bot_id)

            # Wait a moment
            await asyncio.sleep(2)

            # Restart
            result = await self._commander.start(bot_id)
            if result.success:
                logger.info(f"Auto-restarted bot: {bot_id}")
            else:
                logger.error(f"Failed to auto-restart bot {bot_id}: {result.message}")

        except Exception as e:
            logger.error(f"Error during auto-restart of {bot_id}: {e}")

    # =========================================================================
    # Bot Management (Convenience Methods)
    # =========================================================================

    async def create_bot(
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
            bot_id: Optional custom bot ID

        Returns:
            CommandResult with success status and bot_id
        """
        # Check max bots limit
        if len(self._registry.get_all()) >= self._config.max_bots:
            return CommandResult(
                success=False,
                message=f"Maximum bot limit ({self._config.max_bots}) reached",
            )

        return await self._commander.create(bot_type, config, bot_id)

    async def start_bot(self, bot_id: str) -> CommandResult:
        """
        Start a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        return await self._commander.start(bot_id)

    async def stop_bot(
        self,
        bot_id: str,
        clear_position: bool = False,
    ) -> CommandResult:
        """
        Stop a bot.

        Args:
            bot_id: Bot identifier
            clear_position: Whether to clear positions

        Returns:
            CommandResult with success status
        """
        return await self._commander.stop(bot_id, clear_position)

    async def pause_bot(self, bot_id: str) -> CommandResult:
        """
        Pause a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        return await self._commander.pause(bot_id)

    async def resume_bot(self, bot_id: str) -> CommandResult:
        """
        Resume a paused bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        return await self._commander.resume(bot_id)

    async def restart_bot(self, bot_id: str) -> CommandResult:
        """
        Restart a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        return await self._commander.restart(bot_id)

    async def delete_bot(self, bot_id: str) -> CommandResult:
        """
        Delete a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            CommandResult with success status
        """
        return await self._commander.delete(bot_id)

    # =========================================================================
    # Batch Operations
    # =========================================================================

    async def start_all(self) -> List[CommandResult]:
        """
        Start all startable bots.

        Returns:
            List of CommandResult for each bot
        """
        return await self._commander.start_all()

    async def stop_all(self) -> List[CommandResult]:
        """
        Stop all running bots.

        Returns:
            List of CommandResult for each bot
        """
        return await self._commander.stop_all()

    async def pause_all(self) -> List[CommandResult]:
        """
        Pause all running bots.

        Returns:
            List of CommandResult for each bot
        """
        return await self._commander.pause_all()

    async def resume_all(self) -> List[CommandResult]:
        """
        Resume all paused bots.

        Returns:
            List of CommandResult for each bot
        """
        return await self._commander.resume_all()

    # =========================================================================
    # Query Methods
    # =========================================================================

    def get_bot(self, bot_id: str) -> Optional[BotInfo]:
        """
        Get bot information.

        Args:
            bot_id: Bot identifier

        Returns:
            BotInfo if found, None otherwise
        """
        return self._registry.get(bot_id)

    def get_all_bots(self) -> List[BotInfo]:
        """
        Get all registered bots.

        Returns:
            List of BotInfo
        """
        return self._registry.get_all()

    def get_bots_by_state(self, state: BotState) -> List[BotInfo]:
        """
        Get bots filtered by state.

        Args:
            state: BotState to filter by

        Returns:
            List of BotInfo matching the state
        """
        return self._registry.get_by_state(state)

    def get_bots_by_type(self, bot_type: BotType) -> List[BotInfo]:
        """
        Get bots filtered by type.

        Args:
            bot_type: BotType to filter by

        Returns:
            List of BotInfo matching the type
        """
        return self._registry.get_by_type(bot_type)

    def get_dashboard_data(self) -> DashboardData:
        """
        Get dashboard data.

        Returns:
            DashboardData with summary, bots, and alerts
        """
        return self._dashboard.get_data()

    def get_summary(self) -> dict[str, Any]:
        """
        Get system summary.

        Returns:
            Dictionary with system statistics
        """
        summary = self._registry.get_summary()
        summary["is_running"] = self._running
        summary["config"] = {
            "max_bots": self._config.max_bots,
            "auto_restart": self._config.auto_restart,
            "snapshot_interval": self._config.snapshot_interval,
        }
        return summary

    # =========================================================================
    # Health Checking
    # =========================================================================

    async def health_check(self, bot_id: str) -> Optional[HealthCheckResult]:
        """
        Run health check for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            HealthCheckResult if bot exists
        """
        return await self._health.check(bot_id)

    async def health_check_all(self) -> List[HealthCheckResult]:
        """
        Run health check for all bots.

        Returns:
            List of HealthCheckResult
        """
        return await self._health.check_all()

    # =========================================================================
    # Alert Management
    # =========================================================================

    def get_alerts(
        self,
        bot_id: Optional[str] = None,
        unacknowledged_only: bool = False,
    ) -> List[Any]:
        """
        Get alerts.

        Args:
            bot_id: Filter by bot ID
            unacknowledged_only: Only return unacknowledged alerts

        Returns:
            List of alerts
        """
        return self._dashboard.get_alerts(
            bot_id=bot_id,
            unacknowledged_only=unacknowledged_only,
        )

    def acknowledge_alert(self, alert_id: str, by: Optional[str] = None) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_id: Alert identifier
            by: Who acknowledged

        Returns:
            True if acknowledged
        """
        return self._dashboard.acknowledge_alert(alert_id, by)

    # =========================================================================
    # Fund Management
    # =========================================================================

    async def dispatch_funds(self, trigger: str = "manual") -> Optional["DispatchResult"]:
        """
        Dispatch funds to running bots.

        Args:
            trigger: What triggered the dispatch

        Returns:
            DispatchResult if fund manager is configured
        """
        if self._fund_manager:
            return await self._fund_manager.dispatch_funds(trigger)
        return None

    def get_fund_status(self) -> Optional[dict[str, Any]]:
        """
        Get fund manager status.

        Returns:
            Status dictionary if fund manager is configured
        """
        if self._fund_manager:
            return self._fund_manager.get_status()
        return None

    # =========================================================================
    # Helper Methods
    # =========================================================================

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
