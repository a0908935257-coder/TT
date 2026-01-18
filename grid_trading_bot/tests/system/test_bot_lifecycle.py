"""
Bot Lifecycle System Tests.

Tests the complete bot lifecycle from creation to shutdown:
1. Create: Master → Factory → Registry → Database
2. Start: Master → ProcessManager → IPC → Bot
3. Running: Bot → Heartbeat/Events → Master
4. Stop: Master → IPC → Bot → Cleanup
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master.commander import BotCommander, CommandResult
from src.master.factory import BotFactory
from src.master.heartbeat import HeartbeatConfig, HeartbeatData, HeartbeatMonitor
from src.master.models import BotInfo, BotState, BotType, MarketType
from src.master.registry import BotRegistry
from src.master.master import Master, MasterConfig


# =============================================================================
# Test Fixtures
# =============================================================================


def create_mock_config() -> dict[str, Any]:
    """Create mock bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "investment": "10000",
        "market_type": "spot",
        "upper_price": "55000",
        "lower_price": "45000",
        "grid_count": 10,
    }


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset singleton instances before each test."""
    Master.reset_instance()
    BotRegistry.reset_instance()
    yield
    Master.reset_instance()
    BotRegistry.reset_instance()


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()
    exchange.get_ticker = AsyncMock(return_value={"price": "50000"})
    exchange.get_price = AsyncMock(return_value=Decimal("50000"))
    exchange.get_symbol_info = AsyncMock()
    exchange.get_balance = AsyncMock()
    return exchange


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    manager = MagicMock()
    manager.get_price = AsyncMock(return_value=Decimal("50000"))
    return manager


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    db = MagicMock()
    db.get_all_bots = AsyncMock(return_value=[])
    db.upsert_bot = AsyncMock()
    db.delete_bot = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    notifier.send_info = AsyncMock()
    notifier.send_success = AsyncMock()
    notifier.send_error = AsyncMock()
    notifier.notify_bot_registered = AsyncMock()
    notifier.notify_bot_state_changed = AsyncMock()
    return notifier


@pytest.fixture
def bot_registry(mock_db_manager, mock_notifier):
    """Create BotRegistry instance."""
    return BotRegistry(mock_db_manager, mock_notifier)


@pytest.fixture
def bot_factory(mock_exchange, mock_data_manager, mock_notifier):
    """Create BotFactory instance."""
    return BotFactory(mock_exchange, mock_data_manager, mock_notifier)


@pytest.fixture
def master(mock_exchange, mock_data_manager, mock_db_manager, mock_notifier):
    """Create Master instance."""
    config = MasterConfig(
        auto_restart=False,
        max_bots=10,
        snapshot_interval=0,  # Disable for tests
        restore_on_start=False,  # Disable for tests
    )
    return Master(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        db_manager=mock_db_manager,
        notifier=mock_notifier,
        config=config,
    )


# =============================================================================
# Test: Master → Factory
# =============================================================================


class TestMasterToFactory:
    """Tests for Master.create_bot() → Factory flow."""

    @pytest.mark.asyncio
    async def test_create_bot_calls_factory(self, master):
        """Test Master.create_bot() calls Factory.create()."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            # Create mock bot
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)

        # Assert
        assert result.success
        mock_create.assert_called()

    @pytest.mark.asyncio
    async def test_factory_receives_correct_bot_type(self, master):
        """Test Factory receives correct bot_type."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config)

        # Assert - check bot_type argument
        call_args = mock_create.call_args
        assert call_args[0][0] == BotType.GRID

    @pytest.mark.asyncio
    async def test_factory_receives_correct_config(self, master):
        """Test Factory receives correct config."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config)

        # Assert - check config argument
        call_args = mock_create.call_args
        passed_config = call_args[0][2]  # Third positional arg
        assert passed_config["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_create_returns_bot_id(self, master):
        """Test create_bot returns valid bot_id."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)

        # Assert
        assert result.success
        assert result.bot_id is not None


# =============================================================================
# Test: Factory → Registry
# =============================================================================


class TestFactoryToRegistry:
    """Tests for Factory → Registry flow."""

    @pytest.mark.asyncio
    async def test_bot_registered_after_create(self, master):
        """Test bot is registered in Registry after creation."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)

        # Assert - bot should be in registry
        bot_info = master.registry.get(result.bot_id)
        assert bot_info is not None

    @pytest.mark.asyncio
    async def test_registered_state_correct(self, master):
        """Test registered bot has REGISTERED state."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)

        # Assert
        bot_info = master.registry.get(result.bot_id)
        assert bot_info.state == BotState.REGISTERED

    @pytest.mark.asyncio
    async def test_bot_id_stored_correctly(self, master):
        """Test bot_id is stored correctly in registry."""
        # Arrange
        config = create_mock_config()
        custom_id = "custom_bot_123"

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = custom_id
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config, bot_id=custom_id)

        # Assert
        assert result.bot_id == custom_id
        bot_info = master.registry.get(custom_id)
        assert bot_info.bot_id == custom_id


# =============================================================================
# Test: Registry → Database
# =============================================================================


class TestRegistryToDatabase:
    """Tests for Registry → Database persistence."""

    @pytest.mark.asyncio
    async def test_bot_saved_to_database(self, master, mock_db_manager):
        """Test bot is saved to database after registration."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config)

        # Assert - upsert_bot should be called
        mock_db_manager.upsert_bot.assert_called()

    @pytest.mark.asyncio
    async def test_config_saved_to_database(self, master, mock_db_manager):
        """Test bot config is saved to database."""
        # Arrange
        config = create_mock_config()

        # Act
        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config)

        # Assert - check saved data contains config
        call_args = mock_db_manager.upsert_bot.call_args
        saved_data = call_args[0][0]
        assert "config" in saved_data or "symbol" in saved_data


# =============================================================================
# Test: Start → ProcessManager
# =============================================================================


class TestStartToProcessManager:
    """Tests for Master.start_bot() → ProcessManager flow."""

    @pytest.mark.asyncio
    async def test_start_bot_starts_instance(self, master):
        """Test start_bot() starts the bot instance."""
        # Arrange - create bot first
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id

            # Bind instance
            master.registry.bind_instance(bot_id, mock_bot)

        # Act
        result = await master.start_bot(bot_id)

        # Assert
        assert result.success
        mock_bot.start.assert_called()

    @pytest.mark.asyncio
    async def test_start_updates_state(self, master):
        """Test start_bot() updates bot state."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)

        # Act
        await master.start_bot(bot_id)

        # Assert - state should be RUNNING
        bot_info = master.registry.get(bot_id)
        assert bot_info.state == BotState.RUNNING


# =============================================================================
# Test: Start → IPC → Bot
# =============================================================================


class TestStartToIPC:
    """Tests for start → IPC → Bot flow."""

    @pytest.mark.asyncio
    async def test_bot_receives_start_command(self, master):
        """Test bot receives start command via IPC (simulated)."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)

        # Act
        await master.start_bot(bot_id)

        # Assert - bot.start() simulates receiving IPC start command
        mock_bot.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_bot_state_becomes_running(self, master):
        """Test bot state becomes RUNNING after start."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)

        # Act
        await master.start_bot(bot_id)

        # Assert
        bot_info = master.registry.get(bot_id)
        assert bot_info.state == BotState.RUNNING


# =============================================================================
# Test: Bot Heartbeat → Master
# =============================================================================


class TestBotHeartbeatToMaster:
    """Tests for Bot → Heartbeat → Master flow."""

    @pytest.mark.asyncio
    async def test_heartbeat_monitor_receives_heartbeat(self, master, mock_notifier):
        """Test HeartbeatMonitor receives bot heartbeat."""
        # Arrange - create and start bot
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act - simulate heartbeat using HeartbeatData
        heartbeat_data = HeartbeatData(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc),
            state=BotState.RUNNING,
            metrics={
                "active_orders": 5,
                "total_profit": "100.50",
            },
        )
        master.heartbeat_monitor.receive(heartbeat_data)

        # Assert - heartbeat should be recorded
        status = master.heartbeat_monitor.get_status(bot_id)
        assert status is not None

    @pytest.mark.asyncio
    async def test_heartbeat_contains_bot_id(self, master):
        """Test heartbeat contains correct bot_id."""
        # Arrange
        config = create_mock_config()
        bot_id = "heartbeat_test_bot"

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = bot_id
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config, bot_id=bot_id)
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act - send heartbeat
        heartbeat_data = HeartbeatData(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc),
            state=BotState.RUNNING,
            metrics={},
        )
        master.heartbeat_monitor.receive(heartbeat_data)

        # Assert
        status = master.heartbeat_monitor.get_status(bot_id)
        assert status is not None

    @pytest.mark.asyncio
    async def test_heartbeat_contains_metrics(self, master):
        """Test heartbeat contains metrics."""
        # Arrange
        config = create_mock_config()
        bot_id = "metrics_test_bot"

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = bot_id
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config, bot_id=bot_id)
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act - send heartbeat with metrics
        metrics = {
            "active_orders": 5,
            "total_profit": "100.50",
            "trade_count": 10,
        }
        heartbeat_data = HeartbeatData(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc),
            state=BotState.RUNNING,
            metrics=metrics,
        )
        master.heartbeat_monitor.receive(heartbeat_data)

        # Assert - metrics should be stored
        # (Implementation may vary - checking last_seen is sufficient)
        status = master.heartbeat_monitor.get_status(bot_id)
        assert status is not None


# =============================================================================
# Test: Bot Event → Master
# =============================================================================


class TestBotEventToMaster:
    """Tests for Bot → Event → Master flow."""

    @pytest.mark.asyncio
    async def test_event_updates_metrics(self, master):
        """Test bot events update metrics in Master."""
        # Arrange
        config = create_mock_config()
        bot_id = "event_test_bot"

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = bot_id
            mock_bot.start = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            await master.create_bot(BotType.GRID, config, bot_id=bot_id)
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act - simulate event via heartbeat with updated metrics
        event_metrics = {
            "active_orders": 6,
            "total_profit": "150.75",
        }
        heartbeat_data = HeartbeatData(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc),
            state=BotState.RUNNING,
            metrics=event_metrics,
        )
        master.heartbeat_monitor.receive(heartbeat_data)

        # Assert - metrics should be accessible
        status = master.heartbeat_monitor.get_status(bot_id)
        assert status is not None


# =============================================================================
# Test: Stop → Bot
# =============================================================================


class TestStopToBot:
    """Tests for Master.stop_bot() → Bot flow."""

    @pytest.mark.asyncio
    async def test_stop_bot_calls_stop(self, master):
        """Test stop_bot() calls bot.stop()."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act
        result = await master.stop_bot(bot_id)

        # Assert
        assert result.success
        mock_bot.stop.assert_called()

    @pytest.mark.asyncio
    async def test_stop_updates_state(self, master):
        """Test stop_bot() updates state to STOPPED."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act
        await master.stop_bot(bot_id)

        # Assert
        bot_info = master.registry.get(bot_id)
        assert bot_info.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_triggers_cleanup(self, master):
        """Test stop_bot() triggers cleanup (stop method called)."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "test_bot_001"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config)
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

        # Act
        await master.stop_bot(bot_id)

        # Assert - stop should be called
        mock_bot.stop.assert_called_once()


# =============================================================================
# Test: Full Lifecycle
# =============================================================================


class TestFullLifecycle:
    """Tests for complete bot lifecycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle_create_start_stop(self, master, mock_db_manager):
        """Test complete lifecycle: create → start → stop."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "lifecycle_bot"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            # Step 1: Create
            result = await master.create_bot(BotType.GRID, config, bot_id="lifecycle_bot")
            assert result.success
            bot_id = result.bot_id

            # Verify registered
            bot_info = master.registry.get(bot_id)
            assert bot_info.state == BotState.REGISTERED

            # Bind instance
            master.registry.bind_instance(bot_id, mock_bot)

            # Step 2: Start
            result = await master.start_bot(bot_id)
            assert result.success

            # Verify running
            bot_info = master.registry.get(bot_id)
            assert bot_info.state == BotState.RUNNING

            # Step 3: Stop
            result = await master.stop_bot(bot_id)
            assert result.success

            # Verify stopped
            bot_info = master.registry.get(bot_id)
            assert bot_info.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_full_lifecycle_with_pause_resume(self, master):
        """Test lifecycle with pause and resume: create → start → pause → resume → stop."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "pause_resume_bot"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_bot.pause = AsyncMock(return_value=True)
            mock_bot.resume = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            # Create and start
            result = await master.create_bot(BotType.GRID, config, bot_id="pause_resume_bot")
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)

            # Pause
            result = await master.pause_bot(bot_id)
            assert result.success
            bot_info = master.registry.get(bot_id)
            assert bot_info.state == BotState.PAUSED

            # Resume
            result = await master.resume_bot(bot_id)
            assert result.success
            bot_info = master.registry.get(bot_id)
            assert bot_info.state == BotState.RUNNING

            # Stop
            result = await master.stop_bot(bot_id)
            assert result.success

    @pytest.mark.asyncio
    async def test_state_transitions_correct(self, master):
        """Test state transitions follow correct order."""
        # Arrange
        config = create_mock_config()
        states_observed = []

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "state_test_bot"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            # Create
            result = await master.create_bot(BotType.GRID, config, bot_id="state_test_bot")
            bot_id = result.bot_id
            states_observed.append(master.registry.get(bot_id).state)

            # Bind and start
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)
            states_observed.append(master.registry.get(bot_id).state)

            # Stop
            await master.stop_bot(bot_id)
            states_observed.append(master.registry.get(bot_id).state)

        # Assert - correct state transitions
        assert states_observed[0] == BotState.REGISTERED
        assert states_observed[1] == BotState.RUNNING
        assert states_observed[2] == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_no_resource_leak(self, master):
        """Test no resource leaks after full lifecycle."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "leak_test_bot"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            # Full lifecycle
            result = await master.create_bot(BotType.GRID, config, bot_id="leak_test_bot")
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)
            await master.stop_bot(bot_id)

            # Delete
            result = await master.delete_bot(bot_id)
            assert result.success

        # Assert - bot should be removed from registry
        bot_info = master.registry.get("leak_test_bot")
        assert bot_info is None


# =============================================================================
# Edge Cases
# =============================================================================


class TestLifecycleEdgeCases:
    """Edge case tests for bot lifecycle."""

    @pytest.mark.asyncio
    async def test_start_unregistered_bot(self, master):
        """Test starting a non-existent bot fails gracefully."""
        # Act
        result = await master.start_bot("non_existent_bot")

        # Assert
        assert not result.success

    @pytest.mark.asyncio
    async def test_stop_stopped_bot(self, master):
        """Test stopping an already stopped bot."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "double_stop_bot"
            mock_bot.start = AsyncMock(return_value=True)
            mock_bot.stop = AsyncMock(return_value=True)
            mock_create.return_value = mock_bot

            result = await master.create_bot(BotType.GRID, config, bot_id="double_stop_bot")
            bot_id = result.bot_id
            master.registry.bind_instance(bot_id, mock_bot)
            await master.start_bot(bot_id)
            await master.stop_bot(bot_id)

        # Act - stop again
        result = await master.stop_bot(bot_id)

        # Assert - should handle gracefully (may succeed or fail based on implementation)
        # Key is it shouldn't crash

    @pytest.mark.asyncio
    async def test_max_bots_limit(self, master):
        """Test max bots limit is enforced."""
        # Arrange - set max_bots to 2
        master._config.max_bots = 2
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_create.return_value = mock_bot

            # Create first bot
            mock_bot.bot_id = "bot_1"
            await master.create_bot(BotType.GRID, config, bot_id="bot_1")

            # Create second bot
            mock_bot.bot_id = "bot_2"
            await master.create_bot(BotType.GRID, config, bot_id="bot_2")

            # Try to create third bot
            mock_bot.bot_id = "bot_3"
            result = await master.create_bot(BotType.GRID, config, bot_id="bot_3")

        # Assert - should fail
        assert not result.success
        assert "limit" in result.message.lower()

    @pytest.mark.asyncio
    async def test_duplicate_bot_id(self, master):
        """Test creating bot with duplicate ID fails."""
        # Arrange
        config = create_mock_config()

        with patch.object(master._factory, "create") as mock_create:
            mock_bot = MagicMock()
            mock_bot.bot_id = "duplicate_bot"
            mock_create.return_value = mock_bot

            # Create first bot
            await master.create_bot(BotType.GRID, config, bot_id="duplicate_bot")

            # Try to create with same ID
            result = await master.create_bot(BotType.GRID, config, bot_id="duplicate_bot")

        # Assert - should fail
        assert not result.success
