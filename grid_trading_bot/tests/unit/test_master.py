"""
Unit tests for Master Control Console.

Tests the main Master class integration and lifecycle management.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master import (
    BotRegistry,
    BotState,
    BotType,
    CommandResult,
    Master,
    MasterConfig,
)


@pytest.fixture(autouse=True)
def reset_master():
    """Reset singleton before each test."""
    Master.reset_instance()
    yield
    Master.reset_instance()


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()
    exchange.get_ticker = AsyncMock(return_value={"price": "50000"})
    return exchange


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    return MagicMock()


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    db = MagicMock()
    db.get_all_bots = AsyncMock(return_value=[])
    db.upsert_bot = AsyncMock(return_value=None)
    db.delete_bot = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def sample_config():
    """Create sample bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "total_investment": "10000",
    }


@pytest.fixture
def master_config():
    """Create master configuration."""
    return MasterConfig(
        auto_restart=False,
        max_bots=10,
        snapshot_interval=0,  # Disable for tests
        restore_on_start=False,
    )


@pytest.fixture
def master(mock_exchange, mock_data_manager, mock_db_manager, mock_notifier, master_config):
    """Create master instance for testing."""
    return Master(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        db_manager=mock_db_manager,
        notifier=mock_notifier,
        config=master_config,
    )


# =============================================================================
# Test: MasterConfig
# =============================================================================


class TestMasterConfig:
    """Tests for MasterConfig model."""

    def test_default_config(self):
        """Test default configuration values."""
        config = MasterConfig()

        assert config.auto_restart is False
        assert config.max_bots == 100
        assert config.snapshot_interval == 3600
        assert config.restore_on_start is True

    def test_custom_config(self):
        """Test custom configuration values."""
        config = MasterConfig(
            auto_restart=True,
            max_bots=50,
            snapshot_interval=1800,
        )

        assert config.auto_restart is True
        assert config.max_bots == 50
        assert config.snapshot_interval == 1800


# =============================================================================
# Test: Master Singleton
# =============================================================================


class TestMasterSingleton:
    """Tests for Master singleton pattern."""

    def test_singleton(self, mock_exchange, mock_data_manager, mock_db_manager, mock_notifier):
        """Test singleton pattern returns same instance."""
        master1 = Master(mock_exchange, mock_data_manager, mock_db_manager, mock_notifier)
        master2 = Master()

        assert master1 is master2

    def test_get_instance(self, master):
        """Test getting singleton instance."""
        instance = Master.get_instance()

        assert instance is master

    def test_get_instance_none(self):
        """Test getting instance when none exists."""
        instance = Master.get_instance()

        assert instance is None

    def test_reset_instance(self, master):
        """Test resetting singleton instance."""
        assert Master.get_instance() is not None

        Master.reset_instance()

        assert Master.get_instance() is None


# =============================================================================
# Test: Master Properties
# =============================================================================


class TestMasterProperties:
    """Tests for Master properties."""

    def test_registry_property(self, master):
        """Test registry property."""
        assert master.registry is not None
        assert isinstance(master.registry, BotRegistry)

    def test_commander_property(self, master):
        """Test commander property."""
        assert master.commander is not None

    def test_dashboard_property(self, master):
        """Test dashboard property."""
        assert master.dashboard is not None

    def test_factory_property(self, master):
        """Test factory property."""
        assert master.factory is not None

    def test_health_checker_property(self, master):
        """Test health checker property."""
        assert master.health_checker is not None

    def test_heartbeat_monitor_property(self, master):
        """Test heartbeat monitor property."""
        assert master.heartbeat_monitor is not None

    def test_config_property(self, master, master_config):
        """Test config property."""
        assert master.config == master_config

    def test_is_running_false(self, master):
        """Test is_running returns False initially."""
        assert master.is_running is False


# =============================================================================
# Test: Master Lifecycle
# =============================================================================


class TestMasterLifecycle:
    """Tests for Master lifecycle methods."""

    @pytest.mark.asyncio
    async def test_start(self, master):
        """Test starting master."""
        await master.start()

        assert master.is_running is True

    @pytest.mark.asyncio
    async def test_start_already_running(self, master):
        """Test starting master when already running."""
        await master.start()
        await master.start()  # Should not raise

        assert master.is_running is True

    @pytest.mark.asyncio
    async def test_stop(self, master):
        """Test stopping master."""
        await master.start()
        await master.stop()

        assert master.is_running is False

    @pytest.mark.asyncio
    async def test_stop_not_running(self, master):
        """Test stopping master when not running."""
        await master.stop()  # Should not raise

        assert master.is_running is False

    @pytest.mark.asyncio
    async def test_start_with_restore(
        self, mock_exchange, mock_data_manager, mock_db_manager, mock_notifier
    ):
        """Test starting master with restore enabled."""
        config = MasterConfig(restore_on_start=True, snapshot_interval=0)
        master = Master(
            mock_exchange,
            mock_data_manager,
            mock_db_manager,
            mock_notifier,
            config,
        )

        await master.start()

        mock_db_manager.get_all_bots.assert_called_once()
        assert master.is_running is True


# =============================================================================
# Test: Bot Management
# =============================================================================


class TestMasterBotManagement:
    """Tests for Master bot management methods."""

    @pytest.mark.asyncio
    async def test_create_bot(self, master, sample_config):
        """Test creating a bot."""
        # Patch factory.create to return a mock bot
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        master.factory.create = MagicMock(return_value=mock_bot)

        result = await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        assert result.success is True
        assert result.bot_id == "test_bot"

    @pytest.mark.asyncio
    async def test_create_bot_max_limit(self, master, sample_config):
        """Test creating bot when max limit reached."""
        # Set max_bots to 0 for testing
        master._config.max_bots = 0

        result = await master.create_bot(BotType.GRID, sample_config)

        assert result.success is False
        assert "Maximum bot limit" in result.message

    @pytest.mark.asyncio
    async def test_start_bot(self, master, sample_config):
        """Test starting a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        result = await master.start_bot("test_bot")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_stop_bot(self, master, sample_config):
        """Test stopping a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={})
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")
        result = await master.stop_bot("test_bot")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_pause_bot(self, master, sample_config):
        """Test pausing a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")
        result = await master.pause_bot("test_bot")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_resume_bot(self, master, sample_config):
        """Test resuming a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        mock_bot.resume = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")
        await master.pause_bot("test_bot")
        result = await master.resume_bot("test_bot")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_restart_bot(self, master, sample_config):
        """Test restarting a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={})
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")

        with patch("asyncio.sleep", return_value=None):
            result = await master.restart_bot("test_bot")

        assert result.success is True

    @pytest.mark.asyncio
    async def test_delete_bot(self, master, sample_config):
        """Test deleting a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.stop = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        result = await master.delete_bot("test_bot")

        assert result.success is True
        assert master.get_bot("test_bot") is None


# =============================================================================
# Test: Batch Operations
# =============================================================================


class TestMasterBatchOperations:
    """Tests for Master batch operations."""

    @pytest.mark.asyncio
    async def test_start_all(self, master, sample_config):
        """Test starting all bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        with patch("asyncio.sleep", return_value=None):
            results = await master.start_all()

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_stop_all(self, master, sample_config):
        """Test stopping all bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={})
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        with patch("asyncio.sleep", return_value=None):
            await master.start_all()
            results = await master.stop_all()

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_pause_all(self, master, sample_config):
        """Test pausing all bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        with patch("asyncio.sleep", return_value=None):
            await master.start_bot("test_bot")
            results = await master.pause_all()

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_resume_all(self, master, sample_config):
        """Test resuming all bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        mock_bot.resume = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")
        await master.pause_bot("test_bot")

        results = await master.resume_all()

        assert len(results) == 1
        assert results[0].success is True


# =============================================================================
# Test: Query Methods
# =============================================================================


class TestMasterQueryMethods:
    """Tests for Master query methods."""

    @pytest.mark.asyncio
    async def test_get_bot(self, master, sample_config):
        """Test getting bot information."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        bot_info = master.get_bot("test_bot")

        assert bot_info is not None
        assert bot_info.bot_id == "test_bot"

    def test_get_bot_not_found(self, master):
        """Test getting nonexistent bot."""
        bot_info = master.get_bot("nonexistent")

        assert bot_info is None

    @pytest.mark.asyncio
    async def test_get_all_bots(self, master, sample_config):
        """Test getting all bots."""
        mock_bot1 = MagicMock()
        mock_bot1.bot_id = "bot_001"
        mock_bot2 = MagicMock()
        mock_bot2.bot_id = "bot_002"
        master.factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        await master.create_bot(BotType.GRID, sample_config, bot_id="bot_001")
        await master.create_bot(BotType.GRID, sample_config, bot_id="bot_002")

        bots = master.get_all_bots()

        assert len(bots) == 2

    @pytest.mark.asyncio
    async def test_get_bots_by_state(self, master, sample_config):
        """Test getting bots by state."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")

        running = master.get_bots_by_state(BotState.RUNNING)
        registered = master.get_bots_by_state(BotState.REGISTERED)

        assert len(running) == 1
        assert len(registered) == 0

    @pytest.mark.asyncio
    async def test_get_bots_by_type(self, master, sample_config):
        """Test getting bots by type."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        grid_bots = master.get_bots_by_type(BotType.GRID)
        dca_bots = master.get_bots_by_type(BotType.DCA)

        assert len(grid_bots) == 1
        assert len(dca_bots) == 0

    @pytest.mark.asyncio
    async def test_get_dashboard_data(self, master, sample_config):
        """Test getting dashboard data."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        data = master.get_dashboard_data()

        assert data is not None
        assert data.summary.total_bots == 1

    def test_get_summary(self, master):
        """Test getting system summary."""
        summary = master.get_summary()

        assert "total_bots" in summary
        assert "is_running" in summary
        assert "config" in summary


# =============================================================================
# Test: Health Checking
# =============================================================================


class TestMasterHealthChecking:
    """Tests for Master health checking."""

    @pytest.mark.asyncio
    async def test_health_check(self, master, sample_config):
        """Test running health check for a bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")

        result = await master.health_check("test_bot")

        assert result is not None

    @pytest.mark.asyncio
    async def test_health_check_not_found(self, master):
        """Test health check for nonexistent bot returns unhealthy status."""
        from src.master import HealthStatus

        result = await master.health_check("nonexistent")

        # Health checker returns a result with UNHEALTHY/UNKNOWN status for missing bots
        assert result is not None
        assert result.status in (HealthStatus.UNHEALTHY, HealthStatus.UNKNOWN)

    @pytest.mark.asyncio
    async def test_health_check_all(self, master, sample_config):
        """Test running health check for all bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")
        await master.start_bot("test_bot")

        results = await master.health_check_all()

        assert len(results) >= 1


# =============================================================================
# Test: Alert Management
# =============================================================================


class TestMasterAlertManagement:
    """Tests for Master alert management."""

    def test_get_alerts_empty(self, master):
        """Test getting alerts when none exist."""
        alerts = master.get_alerts()

        assert alerts == []

    def test_add_and_get_alerts(self, master):
        """Test adding and getting alerts."""
        from src.master import Alert, AlertLevel

        alert = Alert(
            bot_id="test_bot",
            level=AlertLevel.WARNING,
            message="Test alert",
        )
        master.dashboard.add_alert(alert)

        alerts = master.get_alerts()

        assert len(alerts) == 1
        assert alerts[0].message == "Test alert"

    def test_acknowledge_alert(self, master):
        """Test acknowledging an alert."""
        from src.master import Alert, AlertLevel

        alert = Alert(
            bot_id="test_bot",
            level=AlertLevel.WARNING,
            message="Test alert",
        )
        master.dashboard.add_alert(alert)

        result = master.acknowledge_alert(alert.alert_id, "admin")

        assert result is True
        assert master.dashboard.unacknowledged_alert_count == 0

    def test_acknowledge_nonexistent_alert(self, master):
        """Test acknowledging nonexistent alert."""
        result = master.acknowledge_alert("nonexistent")

        assert result is False


# =============================================================================
# Test: Restore Bot
# =============================================================================


class TestMasterRestoreBot:
    """Tests for Master bot restoration."""

    @pytest.mark.asyncio
    async def test_restore_bot_not_found(self, master):
        """Test restoring nonexistent bot."""
        result = await master._restore_bot("nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_restore_bot_success(self, master, sample_config):
        """Test restoring a bot."""
        # Create mock bot with proper async methods
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.restore_from_db = AsyncMock(return_value=None)

        # Factory will return this mock for both create calls
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.create_bot(BotType.GRID, sample_config, bot_id="test_bot")

        # Unbind instance to simulate restart
        master.registry.unbind_instance("test_bot")

        # Restore (bot is in REGISTERED state, so it won't start automatically)
        result = await master._restore_bot("test_bot")

        assert result is True
        # Instance should be bound again
        assert master.registry.get_bot_instance("test_bot") is not None


# =============================================================================
# Test: Notification
# =============================================================================


class TestMasterNotification:
    """Tests for Master notification."""

    @pytest.mark.asyncio
    async def test_notify_with_notifier(self, master, mock_notifier):
        """Test notification is sent when notifier exists."""
        await master._notify("Test message")

        mock_notifier.send.assert_called_with("Test message")

    @pytest.mark.asyncio
    async def test_notify_without_notifier(
        self, mock_exchange, mock_data_manager, mock_db_manager, master_config
    ):
        """Test notification is skipped when no notifier."""
        Master.reset_instance()
        master = Master(
            mock_exchange,
            mock_data_manager,
            mock_db_manager,
            notifier=None,
            config=master_config,
        )

        # Should not raise
        await master._notify("Test message")
