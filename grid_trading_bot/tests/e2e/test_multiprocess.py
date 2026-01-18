"""
E2E Tests for Multiprocess Architecture.

Tests complete multiprocess flow with Master, IPC, and Bot processes.
Uses mocked Redis and subprocess for isolation.
"""

import asyncio
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from src.bots.base import BaseBot
from src.bots.runner import BotRunner, register_bot
from src.ipc import Channel, Command, CommandType, Event, EventType, Heartbeat, Response
from src.master import Master, MasterConfig, BotType, MarketType
from src.master.ipc_handler import MasterIPCHandler
from src.master.models import BotState
from src.master.process_manager import ProcessManager


# =============================================================================
# Mock Bot for E2E Testing
# =============================================================================


class E2EMockBot(BaseBot):
    """Mock bot for E2E testing."""

    def __init__(self, bot_id, config, exchange, data_manager, notifier, heartbeat_callback=None):
        super().__init__(bot_id, config, exchange, data_manager, notifier, heartbeat_callback)

    @property
    def bot_type(self) -> str:
        return "e2e_mock"

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
        return {"e2e_test": True}

    async def _extra_health_checks(self):
        return {"mock_check": True}


# =============================================================================
# Mock IPC Infrastructure
# =============================================================================


class MockIPCBus:
    """
    Simulated IPC bus for testing multiprocess communication.

    Provides in-memory publish/subscribe functionality that mimics Redis.
    """

    def __init__(self):
        self.channels = {}  # channel -> list of handlers
        self.messages = []  # all published messages

    async def publish(self, channel: str, data: str) -> int:
        """Publish message to channel."""
        self.messages.append((channel, data))

        # Deliver to subscribers
        handlers = self.channels.get(channel, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception as e:
                print(f"Handler error: {e}")

        return len(handlers)

    async def subscribe(self, channel: str, handler):
        """Subscribe to channel."""
        if channel not in self.channels:
            self.channels[channel] = []
        self.channels[channel].append(handler)

    async def unsubscribe(self, channel: str, handler=None):
        """Unsubscribe from channel."""
        if handler and channel in self.channels:
            self.channels[channel] = [h for h in self.channels[channel] if h != handler]
        elif channel in self.channels:
            del self.channels[channel]

    def get_messages(self, channel: str = None):
        """Get messages, optionally filtered by channel."""
        if channel:
            return [(c, d) for c, d in self.messages if c == channel]
        return self.messages


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def ipc_bus():
    """Create mock IPC bus."""
    return MockIPCBus()


@pytest.fixture
def temp_bot_config():
    """Create temporary bot config file."""
    config = {
        "bot_type": "e2e_mock",
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
def mock_redis(ipc_bus):
    """Create mock Redis client using IPC bus."""
    redis = AsyncMock()

    async def mock_publish(channel, data):
        return await ipc_bus.publish(channel, data)

    redis.publish = mock_publish

    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.get_message = AsyncMock(return_value=None)
    mock_pubsub.close = AsyncMock()

    redis.pubsub = MagicMock(return_value=mock_pubsub)
    redis.close = AsyncMock()

    return redis


# =============================================================================
# E2E Multiprocess Tests
# =============================================================================


class TestMultiprocessE2E:
    """E2E tests for multiprocess architecture."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment."""
        register_bot("e2e_mock", E2EMockBot)
        Master.reset_instance()
        yield
        Master.reset_instance()

    @pytest.mark.asyncio
    async def test_ipc_message_flow(self, ipc_bus):
        """Test IPC message serialization and flow."""
        received_commands = []
        received_responses = []

        # Subscribe to channels
        async def command_handler(data):
            received_commands.append(Command.from_json(data))

        async def response_handler(data):
            received_responses.append(Response.from_json(data))

        await ipc_bus.subscribe(Channel.command("bot-001"), command_handler)
        await ipc_bus.subscribe(Channel.response("bot-001"), response_handler)

        # Publish command
        cmd = Command(id="test-cmd", type=CommandType.START)
        await ipc_bus.publish(Channel.command("bot-001"), cmd.to_json())

        # Verify command received
        assert len(received_commands) == 1
        assert received_commands[0].id == "test-cmd"
        assert received_commands[0].type == CommandType.START

        # Publish response
        resp = Response.success_response("test-cmd", {"status": "started"})
        await ipc_bus.publish(Channel.response("bot-001"), resp.to_json())

        # Verify response received
        assert len(received_responses) == 1
        assert received_responses[0].command_id == "test-cmd"
        assert received_responses[0].success is True

    @pytest.mark.asyncio
    async def test_heartbeat_flow(self, ipc_bus):
        """Test heartbeat message flow."""
        received_heartbeats = []

        async def heartbeat_handler(data):
            received_heartbeats.append(Heartbeat.from_json(data))

        await ipc_bus.subscribe(Channel.heartbeat("bot-001"), heartbeat_handler)

        # Publish heartbeat
        hb = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10, "profit": 100.0},
        )
        await ipc_bus.publish(Channel.heartbeat("bot-001"), hb.to_json())

        # Verify heartbeat received
        assert len(received_heartbeats) == 1
        assert received_heartbeats[0].bot_id == "bot-001"
        assert received_heartbeats[0].state == "running"
        assert received_heartbeats[0].metrics["trades"] == 10

    @pytest.mark.asyncio
    async def test_event_broadcast(self, ipc_bus):
        """Test event broadcast to Master."""
        received_events = []

        async def event_handler(data):
            received_events.append(Event.from_json(data))

        await ipc_bus.subscribe(Channel.event(), event_handler)

        # Publish events from multiple bots
        event1 = Event.started("bot-001", {"pid": 1001})
        event2 = Event.trade("bot-002", {"side": "BUY", "profit": "10.5"})
        event3 = Event.error("bot-003", "Connection failed")

        await ipc_bus.publish(Channel.event(), event1.to_json())
        await ipc_bus.publish(Channel.event(), event2.to_json())
        await ipc_bus.publish(Channel.event(), event3.to_json())

        # Verify all events received
        assert len(received_events) == 3
        assert received_events[0].type == EventType.STARTED
        assert received_events[1].type == EventType.TRADE
        assert received_events[2].type == EventType.ERROR

    @pytest.mark.asyncio
    async def test_bot_runner_command_response(self, temp_bot_config, ipc_bus):
        """Test Bot Runner receiving commands and sending responses."""
        # Setup runner without actual Redis
        runner = BotRunner("e2e-bot", temp_bot_config)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Create mock publisher that uses IPC bus
        mock_publisher = AsyncMock()

        async def send_response(bot_id, response):
            await ipc_bus.publish(Channel.response(bot_id), response.to_json())

        mock_publisher.send_response = send_response
        runner._publisher = mock_publisher

        # Subscribe to responses
        received_responses = []

        async def response_handler(data):
            received_responses.append(Response.from_json(data))

        await ipc_bus.subscribe(Channel.response("e2e-bot"), response_handler)

        # Send START command
        start_cmd = Command(id="start-001", type=CommandType.START)
        await runner._handle_command(start_cmd.to_json())

        # Verify response
        assert len(received_responses) == 1
        assert received_responses[0].success is True
        assert received_responses[0].data["status"] == "started"
        assert runner._bot.state == BotState.RUNNING

        # Send STATUS command
        status_cmd = Command(id="status-001", type=CommandType.STATUS)
        await runner._handle_command(status_cmd.to_json())

        # Verify status response
        assert len(received_responses) == 2
        assert received_responses[1].data["state"] == "running"

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, temp_bot_config, ipc_bus):
        """Test full bot lifecycle: register -> start -> pause -> resume -> stop."""
        # Setup runner without actual Redis
        runner = BotRunner("lifecycle-bot", temp_bot_config)
        runner._config = runner._load_config()
        await runner._init_bot()

        # Create mock publisher
        mock_publisher = AsyncMock()
        mock_publisher.send_response = AsyncMock()
        runner._publisher = mock_publisher

        # Track state transitions
        states = []

        def track_state():
            states.append(runner._bot.state.value)

        # 1. START
        await runner._handle_command(Command(id="1", type=CommandType.START).to_json())
        track_state()
        assert states[-1] == "running"

        # 2. PAUSE
        await runner._handle_command(Command(id="2", type=CommandType.PAUSE).to_json())
        track_state()
        assert states[-1] == "paused"

        # 3. RESUME
        await runner._handle_command(Command(id="3", type=CommandType.RESUME).to_json())
        track_state()
        assert states[-1] == "running"

        # 4. STOP
        await runner._handle_command(Command(id="4", type=CommandType.STOP).to_json())
        track_state()
        assert states[-1] == "stopped"

        # Verify state transitions
        assert states == ["running", "paused", "running", "stopped"]

    @pytest.mark.asyncio
    async def test_multiple_bots(self, temp_bot_config, ipc_bus):
        """Test managing multiple bots simultaneously."""
        runners = []
        bot_ids = ["bot-a", "bot-b", "bot-c"]

        # Create multiple runners
        for bot_id in bot_ids:
            runner = BotRunner(bot_id, temp_bot_config)
            runner._config = runner._load_config()
            await runner._init_bot()

            mock_publisher = AsyncMock()
            mock_publisher.send_response = AsyncMock()
            runner._publisher = mock_publisher
            runners.append(runner)

        # Start all bots
        for i, runner in enumerate(runners):
            await runner._handle_command(
                Command(id=f"start-{i}", type=CommandType.START).to_json()
            )

        # Verify all running
        for runner in runners:
            assert runner._bot.state == BotState.RUNNING

        # Stop all bots
        for i, runner in enumerate(runners):
            await runner._handle_command(
                Command(id=f"stop-{i}", type=CommandType.STOP).to_json()
            )

        # Verify all stopped
        for runner in runners:
            assert runner._bot.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_error_handling(self, temp_bot_config, ipc_bus):
        """Test error handling in IPC communication."""
        runner = BotRunner("error-bot", temp_bot_config)
        runner._config = runner._load_config()
        await runner._init_bot()

        received_responses = []

        async def send_response(bot_id, response):
            received_responses.append(response)

        mock_publisher = AsyncMock()
        mock_publisher.send_response = send_response
        runner._publisher = mock_publisher

        # Try invalid operation (pause without start)
        await runner._handle_command(
            Command(id="err-001", type=CommandType.PAUSE).to_json()
        )

        # Verify error response
        assert len(received_responses) == 1
        assert received_responses[0].success is False
        assert "Cannot pause" in received_responses[0].error


class TestProcessManagerE2E:
    """E2E tests for Process Manager."""

    def test_process_info_tracking(self):
        """Test process info is tracked correctly."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 99999
            mock_process.poll.return_value = None
            mock_popen.return_value = mock_process

            manager = ProcessManager()

            # Create temp config
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"bot_type": "test"}, f)
                f.flush()
                config_path = f.name

            try:
                manager.spawn("test-bot", config_path)

                info = manager.get_info("test-bot")
                assert info is not None
                assert info.bot_id == "test-bot"
                assert info.pid == 99999
                assert info.is_alive is True

                manager.kill("test-bot")
            finally:
                os.unlink(config_path)

    def test_multiple_process_management(self):
        """Test managing multiple processes."""
        with patch("subprocess.Popen") as mock_popen:
            mock_process = MagicMock()
            mock_process.pid = 10000
            mock_process.poll.return_value = None
            mock_process.wait.return_value = 0
            mock_popen.return_value = mock_process

            manager = ProcessManager()

            # Create temp config
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                yaml.dump({"bot_type": "test"}, f)
                f.flush()
                config_path = f.name

            try:
                # Spawn multiple
                for i in range(3):
                    manager.spawn(f"bot-{i}", config_path)

                assert manager.active_count == 3
                assert len(manager.all_bot_ids) == 3

                # Kill all
                killed = manager.kill_all()
                assert killed == 3
                assert manager.active_count == 0

            finally:
                os.unlink(config_path)


class TestChannelNaming:
    """Tests for IPC channel naming conventions."""

    def test_command_channel(self):
        """Test command channel naming."""
        assert Channel.command("bot-001") == "trading:cmd:bot-001"
        assert Channel.command("my-grid-bot") == "trading:cmd:my-grid-bot"

    def test_response_channel(self):
        """Test response channel naming."""
        assert Channel.response("bot-001") == "trading:resp:bot-001"

    def test_heartbeat_channel(self):
        """Test heartbeat channel naming."""
        assert Channel.heartbeat("bot-001") == "trading:hb:bot-001"

    def test_event_channel(self):
        """Test event channel naming."""
        assert Channel.event() == "trading:event"

    def test_patterns(self):
        """Test channel patterns for subscription."""
        assert Channel.pattern_all_heartbeats() == "trading:hb:*"
        assert Channel.pattern_all_responses() == "trading:resp:*"
