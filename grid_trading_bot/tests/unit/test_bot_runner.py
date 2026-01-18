"""
Tests for BotRunner.

Validates bot process runner functionality.
"""

import asyncio
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.bots.runner import (
    BOT_REGISTRY,
    BotRunner,
    RunnerConfig,
    get_bot_class,
    register_bot,
)
from src.bots.base import BaseBot
from src.ipc import Command, CommandType, Response
from src.master.models import BotState


# =============================================================================
# Mock Bot for Testing
# =============================================================================


class MockBot(BaseBot):
    """Mock bot for testing."""

    def __init__(self, bot_id, config, exchange, data_manager, notifier, heartbeat_callback=None):
        super().__init__(bot_id, config, exchange, data_manager, notifier, heartbeat_callback)
        self.start_called = False
        self.stop_called = False
        self.pause_called = False
        self.resume_called = False

    @property
    def bot_type(self) -> str:
        return "mock"

    @property
    def symbol(self) -> str:
        return self._config.get("symbol", "BTCUSDT")

    async def _do_start(self) -> None:
        self.start_called = True

    async def _do_stop(self, clear_position: bool = False) -> None:
        self.stop_called = True

    async def _do_pause(self) -> None:
        self.pause_called = True

    async def _do_resume(self) -> None:
        self.resume_called = True

    def _get_extra_status(self):
        return {"mock": True}

    async def _extra_health_checks(self):
        return {"mock_check": True}


# =============================================================================
# RunnerConfig Tests
# =============================================================================


class TestRunnerConfig:
    """Tests for RunnerConfig dataclass."""

    def test_default_values(self):
        """Test default values."""
        config = RunnerConfig(
            bot_id="test-bot",
            bot_type="grid",
        )
        assert config.bot_id == "test-bot"
        assert config.bot_type == "grid"
        assert config.redis_url == "redis://localhost:6379"
        assert config.bot_config == {}

    def test_custom_values(self):
        """Test custom values."""
        config = RunnerConfig(
            bot_id="test-bot",
            bot_type="grid",
            redis_url="redis://custom:6380",
            bot_config={"symbol": "ETHUSDT"},
        )
        assert config.redis_url == "redis://custom:6380"
        assert config.bot_config["symbol"] == "ETHUSDT"


# =============================================================================
# Bot Registry Tests
# =============================================================================


class TestBotRegistry:
    """Tests for bot registration."""

    def test_register_bot(self):
        """Test registering a bot type."""
        register_bot("mock", MockBot)
        assert "mock" in BOT_REGISTRY
        assert BOT_REGISTRY["mock"] == MockBot

    def test_get_bot_class(self):
        """Test getting a registered bot class."""
        register_bot("mock", MockBot)
        bot_class = get_bot_class("mock")
        assert bot_class == MockBot

    def test_get_unknown_bot_class(self):
        """Test getting an unknown bot class returns None."""
        bot_class = get_bot_class("unknown_type")
        assert bot_class is None


# =============================================================================
# BotRunner Tests
# =============================================================================


class TestBotRunner:
    """Tests for BotRunner."""

    @pytest.fixture
    def temp_config_yaml(self):
        """Create a temporary YAML config file."""
        config = {
            "bot_type": "mock",
            "redis_url": "redis://localhost:6379",
            "config": {
                "symbol": "BTCUSDT",
                "investment": 1000,
            },
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def temp_config_json(self):
        """Create a temporary JSON config file."""
        config = {
            "bot_type": "mock",
            "config": {"symbol": "ETHUSDT"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_initialization(self, temp_config_yaml):
        """Test runner initialization."""
        runner = BotRunner("test-bot", temp_config_yaml)
        assert runner.bot_id == "test-bot"
        assert runner.is_running is False

    def test_load_yaml_config(self, temp_config_yaml):
        """Test loading YAML configuration."""
        runner = BotRunner("test-bot", temp_config_yaml)
        config = runner._load_config()

        assert config.bot_id == "test-bot"
        assert config.bot_type == "mock"
        assert config.bot_config["symbol"] == "BTCUSDT"

    def test_load_json_config(self, temp_config_json):
        """Test loading JSON configuration."""
        runner = BotRunner("test-bot", temp_config_json)
        config = runner._load_config()

        assert config.bot_type == "mock"
        assert config.bot_config["symbol"] == "ETHUSDT"

    def test_load_missing_config(self):
        """Test loading non-existent config raises error."""
        runner = BotRunner("test-bot", "/nonexistent/config.yaml")

        with pytest.raises(FileNotFoundError):
            runner._load_config()

    def test_load_invalid_config(self):
        """Test loading config without bot_type raises error."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"invalid": "config"}, f)
            temp_path = f.name

        try:
            runner = BotRunner("test-bot", temp_path)
            with pytest.raises(ValueError, match="bot_type"):
                runner._load_config()
        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_init_bot(self, temp_config_yaml):
        """Test bot initialization."""
        # Register mock bot
        register_bot("mock", MockBot)

        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()

        await runner._init_bot()

        assert runner._bot is not None
        assert runner._bot.bot_type == "mock"

    @pytest.mark.asyncio
    async def test_init_bot_unknown_type(self, temp_config_yaml):
        """Test init bot with unknown type raises error."""
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = RunnerConfig(
            bot_id="test-bot",
            bot_type="unknown_type_xyz",
        )

        with pytest.raises(ValueError, match="Unknown bot type"):
            await runner._init_bot()

    @pytest.mark.asyncio
    async def test_execute_start_command(self, temp_config_yaml):
        """Test executing START command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        cmd = Command(type=CommandType.START)
        result = await runner._execute_command(cmd)

        assert result["status"] == "started"
        assert runner._bot.start_called is True

    @pytest.mark.asyncio
    async def test_execute_stop_command(self, temp_config_yaml):
        """Test executing STOP command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()
        await runner._bot.start()

        cmd = Command(type=CommandType.STOP, params={"clear_position": True})
        result = await runner._execute_command(cmd)

        assert result["status"] == "stopped"
        assert runner._bot.stop_called is True

    @pytest.mark.asyncio
    async def test_execute_pause_command(self, temp_config_yaml):
        """Test executing PAUSE command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()
        await runner._bot.start()

        cmd = Command(type=CommandType.PAUSE)
        result = await runner._execute_command(cmd)

        assert result["status"] == "paused"
        assert runner._bot.pause_called is True

    @pytest.mark.asyncio
    async def test_execute_resume_command(self, temp_config_yaml):
        """Test executing RESUME command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()
        await runner._bot.start()
        await runner._bot.pause()

        cmd = Command(type=CommandType.RESUME)
        result = await runner._execute_command(cmd)

        assert result["status"] == "resumed"
        assert runner._bot.resume_called is True

    @pytest.mark.asyncio
    async def test_execute_status_command(self, temp_config_yaml):
        """Test executing STATUS command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        cmd = Command(type=CommandType.STATUS)
        result = await runner._execute_command(cmd)

        assert "bot_id" in result
        assert "state" in result
        assert result["bot_type"] == "mock"

    @pytest.mark.asyncio
    async def test_execute_shutdown_command(self, temp_config_yaml):
        """Test executing SHUTDOWN command."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        cmd = Command(type=CommandType.SHUTDOWN)
        result = await runner._execute_command(cmd)

        assert result["status"] == "shutdown_initiated"
        assert runner._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_handle_command_success(self, temp_config_yaml):
        """Test handling command with success response."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Mock publisher
        runner._publisher = AsyncMock()
        runner._publisher.send_response = AsyncMock()

        cmd = Command(id="cmd-123", type=CommandType.STATUS)
        await runner._handle_command(cmd.to_json())

        # Verify response was sent
        runner._publisher.send_response.assert_called_once()
        call_args = runner._publisher.send_response.call_args
        assert call_args[0][0] == "test-bot"  # bot_id
        response = call_args[0][1]
        assert response.command_id == "cmd-123"
        assert response.success is True

    @pytest.mark.asyncio
    async def test_handle_command_failure(self, temp_config_yaml):
        """Test handling command with error response."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Mock publisher
        runner._publisher = AsyncMock()
        runner._publisher.send_response = AsyncMock()

        # Try to pause when not running (should fail)
        cmd = Command(id="cmd-456", type=CommandType.PAUSE)
        await runner._handle_command(cmd.to_json())

        # Verify error response was sent
        runner._publisher.send_response.assert_called_once()
        call_args = runner._publisher.send_response.call_args
        response = call_args[0][1]
        assert response.command_id == "cmd-456"
        assert response.success is False
        assert response.error is not None

    def test_get_metrics(self, temp_config_yaml):
        """Test getting metrics."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()

        # Without bot
        metrics = runner._get_metrics()
        assert metrics == {}

    @pytest.mark.asyncio
    async def test_get_metrics_with_bot(self, temp_config_yaml):
        """Test getting metrics with initialized bot."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        metrics = runner._get_metrics()

        assert "uptime_seconds" in metrics
        assert "total_trades" in metrics
        assert "total_profit" in metrics

    @pytest.mark.asyncio
    async def test_shutdown(self, temp_config_yaml):
        """Test shutdown cleanup."""
        register_bot("mock", MockBot)
        runner = BotRunner("test-bot", temp_config_yaml)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Mock IPC
        runner._publisher = AsyncMock()
        runner._publisher.send_event = AsyncMock()
        runner._subscriber = AsyncMock()
        runner._subscriber.stop = AsyncMock()
        runner._redis = AsyncMock()
        runner._redis.close = AsyncMock()

        runner._running = True
        await runner._shutdown()

        assert runner._running is False
        runner._subscriber.stop.assert_called_once()
        runner._redis.close.assert_called_once()


# =============================================================================
# Integration Tests
# =============================================================================


class TestBotRunnerIntegration:
    """Integration tests for BotRunner."""

    @pytest.fixture
    def temp_config(self):
        """Create a temporary config file."""
        config = {
            "bot_type": "mock",
            "redis_url": "redis://localhost:6379",
            "config": {"symbol": "BTCUSDT"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            yield f.name
        os.unlink(f.name)

    @pytest.mark.asyncio
    async def test_full_command_flow(self, temp_config):
        """Test full command flow from JSON to response."""
        register_bot("mock", MockBot)
        runner = BotRunner("integration-bot", temp_config)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Mock publisher
        sent_responses = []

        async def capture_response(bot_id, response):
            sent_responses.append(response)

        runner._publisher = AsyncMock()
        runner._publisher.send_response = capture_response

        # Send START command
        start_cmd = Command(id="start-001", type=CommandType.START)
        await runner._handle_command(start_cmd.to_json())

        assert len(sent_responses) == 1
        assert sent_responses[0].command_id == "start-001"
        assert sent_responses[0].success is True
        assert runner._bot.state == BotState.RUNNING

        # Send STATUS command
        status_cmd = Command(id="status-001", type=CommandType.STATUS)
        await runner._handle_command(status_cmd.to_json())

        assert len(sent_responses) == 2
        assert sent_responses[1].data["state"] == "running"

        # Send STOP command
        stop_cmd = Command(id="stop-001", type=CommandType.STOP)
        await runner._handle_command(stop_cmd.to_json())

        assert len(sent_responses) == 3
        assert sent_responses[2].success is True
        assert runner._bot.state == BotState.STOPPED
