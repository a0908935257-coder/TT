"""
Integration Tests for Bot Runner.

Tests Bot Runner command handling and IPC response flow.
These tests don't require actual Redis - they test the runner logic directly.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from src.bots.base import BaseBot
from src.bots.runner import BotRunner, register_bot
from src.ipc import Channel, Command, CommandType, Response
from src.master.models import BotState


# =============================================================================
# Mock Bot for Testing
# =============================================================================


class IntegrationMockBot(BaseBot):
    """Mock bot for integration testing."""

    def __init__(self, bot_id, config, exchange, data_manager, notifier, heartbeat_callback=None):
        super().__init__(bot_id, config, exchange, data_manager, notifier, heartbeat_callback)

    @property
    def bot_type(self) -> str:
        return "integration_mock"

    @property
    def symbol(self) -> str:
        return self._config.get("symbol", "BTCUSDT")

    async def _do_start(self) -> None:
        pass

    async def _do_stop(self, clear_position: bool = False) -> None:
        pass

    async def _do_pause(self) -> None:
        pass

    async def _do_resume(self) -> None:
        pass

    def _get_extra_status(self):
        return {"integration_test": True}

    async def _extra_health_checks(self):
        return {"mock_check": True}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_config():
    """Create temporary config file."""
    config = {
        "bot_type": "integration_mock",
        "redis_url": "redis://localhost:6379",
        "config": {
            "symbol": "BTCUSDT",
            "total_investment": 1000,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f)
        f.flush()
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_publisher():
    """Create mock IPC publisher that tracks messages."""
    messages = []

    publisher = AsyncMock()

    async def send_response(bot_id, response):
        messages.append(("response", bot_id, response))

    async def send_event(event):
        messages.append(("event", event.bot_id, event))

    async def send_heartbeat(bot_id, heartbeat):
        messages.append(("heartbeat", bot_id, heartbeat))

    publisher.send_response = send_response
    publisher.send_event = send_event
    publisher.send_heartbeat = send_heartbeat
    publisher.messages = messages

    return publisher


# =============================================================================
# Integration Tests
# =============================================================================


class TestBotRunnerIntegration:
    """Integration tests for Bot Runner."""

    @pytest.fixture(autouse=True)
    def register_mock_bot(self):
        """Register mock bot for testing."""
        register_bot("integration_mock", IntegrationMockBot)

    @pytest.mark.asyncio
    async def test_runner_initialization(self, temp_config):
        """Test runner initializes correctly with loaded config."""
        runner = BotRunner("test-bot", temp_config)
        config = runner._load_config()

        assert config.bot_id == "test-bot"
        assert config.bot_type == "integration_mock"
        assert config.bot_config["symbol"] == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_runner_init_bot(self, temp_config):
        """Test runner creates bot instance correctly."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()

        assert runner._bot is not None
        assert runner._bot.bot_type == "integration_mock"
        assert runner._bot.state == BotState.REGISTERED

    @pytest.mark.asyncio
    async def test_runner_handles_start_command(self, temp_config, mock_publisher):
        """Test runner handles START command correctly."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Send START command
        cmd = Command(id="cmd-001", type=CommandType.START)
        await runner._handle_command(cmd.to_json())

        # Verify bot state
        assert runner._bot.state == BotState.RUNNING

        # Verify response was sent
        assert len(mock_publisher.messages) == 1
        msg_type, bot_id, response = mock_publisher.messages[0]
        assert msg_type == "response"
        assert bot_id == "test-bot"
        assert response.command_id == "cmd-001"
        assert response.success is True

    @pytest.mark.asyncio
    async def test_runner_handles_status_command(self, temp_config, mock_publisher):
        """Test runner handles STATUS command correctly."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Start bot first
        await runner._bot.start()

        # Send STATUS command
        cmd = Command(id="cmd-002", type=CommandType.STATUS)
        await runner._handle_command(cmd.to_json())

        # Verify response
        msg_type, bot_id, response = mock_publisher.messages[0]
        assert response.command_id == "cmd-002"
        assert response.success is True
        assert response.data["state"] == "running"
        assert response.data["bot_type"] == "integration_mock"

    @pytest.mark.asyncio
    async def test_runner_handles_pause_resume(self, temp_config, mock_publisher):
        """Test runner handles PAUSE and RESUME commands."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Start bot
        await runner._bot.start()
        assert runner._bot.state == BotState.RUNNING

        # Pause
        cmd = Command(id="cmd-003", type=CommandType.PAUSE)
        await runner._handle_command(cmd.to_json())
        assert runner._bot.state == BotState.PAUSED

        # Resume
        cmd = Command(id="cmd-004", type=CommandType.RESUME)
        await runner._handle_command(cmd.to_json())
        assert runner._bot.state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_runner_handles_stop_command(self, temp_config, mock_publisher):
        """Test runner handles STOP command correctly."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Start bot
        await runner._bot.start()

        # Stop bot
        cmd = Command(id="cmd-005", type=CommandType.STOP)
        await runner._handle_command(cmd.to_json())

        assert runner._bot.state == BotState.STOPPED

        # Verify response
        response = mock_publisher.messages[0][2]
        assert response.success is True
        assert response.data["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_runner_handles_shutdown_command(self, temp_config, mock_publisher):
        """Test runner handles SHUTDOWN command correctly."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Send SHUTDOWN command
        cmd = Command(id="cmd-006", type=CommandType.SHUTDOWN)
        await runner._handle_command(cmd.to_json())

        # Verify shutdown event was set
        assert runner._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_runner_handles_invalid_command(self, temp_config, mock_publisher):
        """Test runner handles invalid state transition gracefully."""
        runner = BotRunner("test-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # Try to pause without starting (should fail)
        cmd = Command(id="cmd-007", type=CommandType.PAUSE)
        await runner._handle_command(cmd.to_json())

        # Verify error response
        response = mock_publisher.messages[0][2]
        assert response.success is False
        assert response.error is not None

    @pytest.mark.asyncio
    async def test_runner_command_flow(self, temp_config, mock_publisher):
        """Test complete command flow: START -> STATUS -> PAUSE -> RESUME -> STOP."""
        runner = BotRunner("flow-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        runner._publisher = mock_publisher

        # START
        await runner._handle_command(Command(id="1", type=CommandType.START).to_json())
        assert runner._bot.state == BotState.RUNNING

        # STATUS
        await runner._handle_command(Command(id="2", type=CommandType.STATUS).to_json())
        status_response = mock_publisher.messages[1][2]
        assert status_response.data["state"] == "running"

        # PAUSE
        await runner._handle_command(Command(id="3", type=CommandType.PAUSE).to_json())
        assert runner._bot.state == BotState.PAUSED

        # RESUME
        await runner._handle_command(Command(id="4", type=CommandType.RESUME).to_json())
        assert runner._bot.state == BotState.RUNNING

        # STOP
        await runner._handle_command(Command(id="5", type=CommandType.STOP).to_json())
        assert runner._bot.state == BotState.STOPPED

        # Verify all responses were successful
        assert len(mock_publisher.messages) == 5
        assert all(msg[2].success for msg in mock_publisher.messages)


class TestBotRunnerHeartbeat:
    """Tests for Bot Runner heartbeat functionality."""

    @pytest.fixture(autouse=True)
    def register_mock_bot(self):
        """Register mock bot for testing."""
        register_bot("integration_mock", IntegrationMockBot)

    @pytest.mark.asyncio
    async def test_heartbeat_metrics(self, temp_config):
        """Test heartbeat contains correct metrics."""
        runner = BotRunner("hb-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()
        await runner._bot.start()

        runner._running = True
        metrics = runner._get_metrics()

        assert "uptime_seconds" in metrics
        assert "total_trades" in metrics
        assert "total_profit" in metrics
        assert metrics["total_trades"] == 0
        assert metrics["total_profit"] == 0.0

    @pytest.mark.asyncio
    async def test_get_metrics_without_bot(self, temp_config):
        """Test metrics returns empty dict without bot."""
        runner = BotRunner("no-bot", temp_config)
        runner._config = runner._load_config()

        metrics = runner._get_metrics()
        assert metrics == {}
