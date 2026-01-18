"""
Unit tests for BotCommander and BotFactory.

Tests command execution, bot lifecycle management, and factory creation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master import (
    BotCommander,
    BotFactory,
    BotRegistry,
    BotState,
    BotType,
    CommandResult,
    InvalidBotConfigError,
    UnsupportedBotTypeError,
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset singleton before each test."""
    BotRegistry.reset_instance()
    yield
    BotRegistry.reset_instance()


@pytest.fixture
def registry():
    """Create a fresh registry instance."""
    return BotRegistry()


@pytest.fixture
def sample_config():
    """Create sample bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "total_investment": "10000",
    }


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
def mock_notifier():
    """Create mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def factory(mock_exchange, mock_data_manager, mock_notifier):
    """Create bot factory for testing."""
    return BotFactory(mock_exchange, mock_data_manager, mock_notifier)


@pytest.fixture
def commander(registry, factory, mock_notifier):
    """Create bot commander for testing."""
    return BotCommander(registry, factory, mock_notifier)


# =============================================================================
# Test: CommandResult Model
# =============================================================================


class TestCommandResult:
    """Tests for CommandResult model."""

    def test_create_success_result(self):
        """Test creating a success result."""
        result = CommandResult(
            success=True,
            message="Operation successful",
            bot_id="bot_001",
        )

        assert result.success is True
        assert result.message == "Operation successful"
        assert result.bot_id == "bot_001"
        assert result.data == {}

    def test_create_failure_result(self):
        """Test creating a failure result."""
        result = CommandResult(
            success=False,
            message="Operation failed",
            bot_id="bot_001",
            data={"error_code": "E001"},
        )

        assert result.success is False
        assert result.message == "Operation failed"
        assert result.data["error_code"] == "E001"

    def test_to_dict(self):
        """Test CommandResult to_dict."""
        result = CommandResult(
            success=True,
            message="Test",
            bot_id="bot_001",
            data={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["message"] == "Test"
        assert result_dict["bot_id"] == "bot_001"
        assert result_dict["data"]["key"] == "value"


# =============================================================================
# Test: BotFactory
# =============================================================================


class TestBotFactory:
    """Tests for BotFactory."""

    def test_create_factory(self, mock_exchange, mock_data_manager, mock_notifier):
        """Test creating a factory."""
        factory = BotFactory(mock_exchange, mock_data_manager, mock_notifier)

        assert factory._exchange == mock_exchange
        assert factory._data_manager == mock_data_manager
        assert factory._notifier == mock_notifier

    def test_supported_types(self, factory):
        """Test getting supported bot types."""
        supported = factory.supported_types

        assert BotType.GRID in supported
        assert BotType.DCA in supported
        assert BotType.TRAILING_STOP in supported
        assert BotType.SIGNAL in supported

    def test_validate_base_config_missing_symbol(self, factory):
        """Test validation fails without symbol."""
        config = {"total_investment": "1000"}

        with pytest.raises(InvalidBotConfigError, match="Missing required field: symbol"):
            factory.create(BotType.GRID, "bot_001", config)

    def test_create_unsupported_bot_type(self, factory, sample_config):
        """Test creating unsupported bot type raises error."""
        # DCA is registered but not implemented
        with pytest.raises(InvalidBotConfigError):
            factory.create(BotType.DCA, "bot_001", sample_config)

    def test_create_grid_bot(self, factory, sample_config):
        """Test creating a grid bot."""
        # Patch the GridBot class to avoid actual initialization
        with patch("src.bots.grid.bot.GridBot") as MockGridBot:
            mock_bot = MagicMock()
            mock_bot.bot_id = "bot_001"
            MockGridBot.return_value = mock_bot

            bot = factory.create(BotType.GRID, "bot_001", sample_config)

            assert bot.bot_id == "bot_001"
            MockGridBot.assert_called_once()

    def test_register_custom_creator(self, factory):
        """Test registering a custom creator."""
        custom_creator = MagicMock(return_value=MagicMock(bot_id="custom_001"))

        factory.register_creator(BotType.DCA, custom_creator)

        config = {"symbol": "BTCUSDT"}
        bot = factory.create(BotType.DCA, "custom_001", config)

        custom_creator.assert_called_once_with("custom_001", config)
        assert bot.bot_id == "custom_001"


# =============================================================================
# Test: BotCommander - Create
# =============================================================================


class TestBotCommanderCreate:
    """Tests for BotCommander.create."""

    @pytest.mark.asyncio
    async def test_create_bot_success(self, commander, sample_config):
        """Test creating a bot successfully."""
        # Patch factory.create to return a mock bot
        mock_bot = MagicMock()
        mock_bot.bot_id = "test_bot"
        commander._factory.create = MagicMock(return_value=mock_bot)

        result = await commander.create(BotType.GRID, sample_config, bot_id="test_bot")

        assert result.success is True
        assert result.bot_id == "test_bot"
        assert "created successfully" in result.message

    @pytest.mark.asyncio
    async def test_create_bot_auto_id(self, commander, sample_config):
        """Test creating a bot with auto-generated ID."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "grid_test"
        commander._factory.create = MagicMock(return_value=mock_bot)

        result = await commander.create(BotType.GRID, sample_config)

        assert result.success is True
        assert result.bot_id is not None
        assert result.bot_id.startswith("grid_")

    @pytest.mark.asyncio
    async def test_create_bot_already_exists(self, commander, registry, sample_config):
        """Test creating a bot that already exists."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        commander._factory.create = MagicMock(return_value=mock_bot)

        # Create first bot
        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        # Try to create again
        result = await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        assert result.success is False
        assert "already exists" in result.message

    @pytest.mark.asyncio
    async def test_create_bot_factory_error(self, commander, sample_config):
        """Test creating a bot when factory fails."""
        commander._factory.create = MagicMock(
            side_effect=InvalidBotConfigError("Invalid config")
        )

        result = await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        assert result.success is False
        assert "Invalid config" in result.message


# =============================================================================
# Test: BotCommander - Start
# =============================================================================


class TestBotCommanderStart:
    """Tests for BotCommander.start."""

    @pytest.mark.asyncio
    async def test_start_bot_success(self, commander, registry, sample_config):
        """Test starting a bot successfully."""
        # Setup bot
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        # Start the bot
        result = await commander.start("bot_001")

        assert result.success is True
        assert "started successfully" in result.message
        mock_bot.start.assert_called_once()

        # Verify state
        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_start_bot_not_found(self, commander):
        """Test starting a nonexistent bot."""
        result = await commander.start("nonexistent")

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_start_bot_invalid_state(self, commander, registry, sample_config):
        """Test starting a bot in invalid state."""
        # Create and start bot
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        # Try to start again (already running)
        result = await commander.start("bot_001")

        assert result.success is False
        assert "Cannot start" in result.message

    @pytest.mark.asyncio
    async def test_start_bot_failure(self, commander, registry, sample_config):
        """Test starting a bot that fails to start."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=False)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        result = await commander.start("bot_001")

        assert result.success is False
        assert "failed to start" in result.message.lower()

        # Verify state is ERROR
        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.ERROR


# =============================================================================
# Test: BotCommander - Stop
# =============================================================================


class TestBotCommanderStop:
    """Tests for BotCommander.stop."""

    @pytest.mark.asyncio
    async def test_stop_bot_success(self, commander, registry, sample_config):
        """Test stopping a bot successfully."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={"trade_count": 10})
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        result = await commander.stop("bot_001")

        assert result.success is True
        assert "stopped successfully" in result.message
        mock_bot.stop.assert_called_once()

        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_stop_bot_not_found(self, commander):
        """Test stopping a nonexistent bot."""
        result = await commander.stop("nonexistent")

        assert result.success is False
        assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_stop_bot_invalid_state(self, commander, registry, sample_config):
        """Test stopping a bot in invalid state."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        # Try to stop (not running)
        result = await commander.stop("bot_001")

        assert result.success is False
        assert "Cannot stop" in result.message


# =============================================================================
# Test: BotCommander - Pause/Resume
# =============================================================================


class TestBotCommanderPauseResume:
    """Tests for BotCommander.pause and resume."""

    @pytest.mark.asyncio
    async def test_pause_bot_success(self, commander, registry, sample_config):
        """Test pausing a bot successfully."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        result = await commander.pause("bot_001")

        assert result.success is True
        assert "paused successfully" in result.message

        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.PAUSED

    @pytest.mark.asyncio
    async def test_pause_bot_not_running(self, commander, registry, sample_config):
        """Test pausing a bot that's not running."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        result = await commander.pause("bot_001")

        assert result.success is False
        assert "Cannot pause" in result.message

    @pytest.mark.asyncio
    async def test_resume_bot_success(self, commander, registry, sample_config):
        """Test resuming a paused bot."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        mock_bot.resume = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")
        await commander.pause("bot_001")

        result = await commander.resume("bot_001")

        assert result.success is True
        assert "resumed successfully" in result.message

        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_resume_bot_not_paused(self, commander, registry, sample_config):
        """Test resuming a bot that's not paused."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        result = await commander.resume("bot_001")

        assert result.success is False
        assert "Cannot resume" in result.message


# =============================================================================
# Test: BotCommander - Restart
# =============================================================================


class TestBotCommanderRestart:
    """Tests for BotCommander.restart."""

    @pytest.mark.asyncio
    async def test_restart_bot_success(self, commander, registry, sample_config):
        """Test restarting a bot successfully."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={})
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        # Patch sleep to speed up test
        with patch("asyncio.sleep", return_value=None):
            result = await commander.restart("bot_001")

        assert result.success is True
        assert "restarted successfully" in result.message

        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.RUNNING


# =============================================================================
# Test: BotCommander - Delete
# =============================================================================


class TestBotCommanderDelete:
    """Tests for BotCommander.delete."""

    @pytest.mark.asyncio
    async def test_delete_bot_success(self, commander, registry, sample_config):
        """Test deleting a bot successfully."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.stop = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        result = await commander.delete("bot_001")

        assert result.success is True
        assert "deleted successfully" in result.message

        # Bot should no longer exist
        assert registry.get("bot_001") is None

    @pytest.mark.asyncio
    async def test_delete_running_bot(self, commander, registry, sample_config):
        """Test deleting a running bot (should stop first)."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.stop = AsyncMock(return_value=True)
        mock_bot.get_statistics = MagicMock(return_value={})
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        result = await commander.delete("bot_001")

        assert result.success is True
        mock_bot.stop.assert_called()
        assert registry.get("bot_001") is None

    @pytest.mark.asyncio
    async def test_delete_bot_not_found(self, commander):
        """Test deleting a nonexistent bot."""
        result = await commander.delete("nonexistent")

        assert result.success is False
        assert "not found" in result.message.lower()


# =============================================================================
# Test: BotCommander - Batch Operations
# =============================================================================


class TestBotCommanderBatch:
    """Tests for batch operations."""

    @pytest.mark.asyncio
    async def test_start_all(self, commander, registry, sample_config):
        """Test starting all bots."""
        mock_bot1 = MagicMock()
        mock_bot1.bot_id = "bot_001"
        mock_bot1.start = AsyncMock(return_value=True)

        mock_bot2 = MagicMock()
        mock_bot2.bot_id = "bot_002"
        mock_bot2.start = AsyncMock(return_value=True)

        commander._factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.create(BotType.GRID, sample_config, bot_id="bot_002")

        # Patch sleep to speed up test
        with patch("asyncio.sleep", return_value=None):
            results = await commander.start_all()

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_stop_all(self, commander, registry, sample_config):
        """Test stopping all bots."""
        mock_bot1 = MagicMock()
        mock_bot1.bot_id = "bot_001"
        mock_bot1.start = AsyncMock(return_value=True)
        mock_bot1.stop = AsyncMock(return_value=True)
        mock_bot1.get_statistics = MagicMock(return_value={})

        mock_bot2 = MagicMock()
        mock_bot2.bot_id = "bot_002"
        mock_bot2.start = AsyncMock(return_value=True)
        mock_bot2.stop = AsyncMock(return_value=True)
        mock_bot2.get_statistics = MagicMock(return_value={})

        commander._factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.create(BotType.GRID, sample_config, bot_id="bot_002")

        with patch("asyncio.sleep", return_value=None):
            await commander.start_all()
            results = await commander.stop_all()

        assert len(results) == 2
        assert all(r.success for r in results)

    @pytest.mark.asyncio
    async def test_pause_all(self, commander, registry, sample_config):
        """Test pausing all running bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")

        with patch("asyncio.sleep", return_value=None):
            await commander.start("bot_001")
            results = await commander.pause_all()

        assert len(results) == 1
        assert results[0].success is True

    @pytest.mark.asyncio
    async def test_resume_all(self, commander, registry, sample_config):
        """Test resuming all paused bots."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        mock_bot.pause = AsyncMock(return_value=True)
        mock_bot.resume = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")
        await commander.pause("bot_001")

        results = await commander.resume_all()

        assert len(results) == 1
        assert results[0].success is True


# =============================================================================
# Test: BotCommander - Validation
# =============================================================================


class TestBotCommanderValidation:
    """Tests for validation helpers."""

    def test_validate_can_start_not_found(self, commander):
        """Test validation for nonexistent bot."""
        valid, msg = commander._validate_can_start("nonexistent")

        assert valid is False
        assert "not found" in msg.lower()

    @pytest.mark.asyncio
    async def test_validate_can_start_invalid_state(self, commander, registry, sample_config):
        """Test validation for invalid start state."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        valid, msg = commander._validate_can_start("bot_001")

        assert valid is False
        assert "Cannot start" in msg

    @pytest.mark.asyncio
    async def test_validate_can_stop_valid(self, commander, registry, sample_config):
        """Test validation for valid stop state."""
        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"
        mock_bot.start = AsyncMock(return_value=True)
        commander._factory.create = MagicMock(return_value=mock_bot)

        await commander.create(BotType.GRID, sample_config, bot_id="bot_001")
        await commander.start("bot_001")

        valid, msg = commander._validate_can_stop("bot_001")

        assert valid is True
        assert msg == ""


# =============================================================================
# Test: BotCommander - Helper Methods
# =============================================================================


class TestBotCommanderHelpers:
    """Tests for helper methods."""

    def test_generate_bot_id(self, commander):
        """Test bot ID generation."""
        bot_id = commander._generate_bot_id(BotType.GRID)

        assert bot_id.startswith("grid_")
        assert len(bot_id) > 20  # timestamp + uuid

    @pytest.mark.asyncio
    async def test_notify_with_notifier(self, commander, mock_notifier):
        """Test notification is sent when notifier exists."""
        await commander._notify("Test message")

        mock_notifier.send.assert_called_once_with("Test message")

    @pytest.mark.asyncio
    async def test_notify_without_notifier(self, registry, factory):
        """Test notification is skipped when no notifier."""
        commander = BotCommander(registry, factory, notifier=None)

        # Should not raise
        await commander._notify("Test message")
