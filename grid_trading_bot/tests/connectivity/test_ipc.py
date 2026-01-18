"""
IPC Connectivity Tests.

Tests Master-Bot IPC communication via Redis Pub/Sub.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ipc import (
    Channel,
    Command,
    CommandType,
    Event,
    EventType,
    Heartbeat,
    IPCPublisher,
    IPCSubscriber,
    Response,
)
from src.master import BotInfo, BotState, BotType, MarketType
from src.master.ipc_handler import MasterIPCHandler


# =============================================================================
# Mock IPC Bus
# =============================================================================


class MockIPCBus:
    """In-memory IPC bus for testing."""

    def __init__(self):
        self.channels = {}
        self.messages = []

    async def publish(self, channel: str, data: str) -> int:
        """Publish message to channel."""
        self.messages.append((channel, data))

        handlers = self.channels.get(channel, [])
        for handler in handlers:
            try:
                await handler(data)
            except Exception:
                pass

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


# =============================================================================
# IPC Message Tests
# =============================================================================


class TestIPCMessages:
    """Tests for IPC message serialization."""

    def test_command_serialize(self):
        """Test Command serialization."""
        cmd = Command(id="test-001", type=CommandType.START)
        json_str = cmd.to_json()

        restored = Command.from_json(json_str)

        assert restored.id == "test-001"
        assert restored.type == CommandType.START

    def test_response_serialize(self):
        """Test Response serialization."""
        resp = Response.success_response("cmd-001", {"status": "ok"})
        json_str = resp.to_json()

        restored = Response.from_json(json_str)

        assert restored.command_id == "cmd-001"
        assert restored.success is True
        assert restored.data["status"] == "ok"

    def test_heartbeat_serialize(self):
        """Test Heartbeat serialization."""
        hb = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10},
        )
        json_str = hb.to_json()

        restored = Heartbeat.from_json(json_str)

        assert restored.bot_id == "bot-001"
        assert restored.state == "running"
        assert restored.pid == 12345
        assert restored.metrics["trades"] == 10

    def test_event_serialize(self):
        """Test Event serialization."""
        event = Event.trade("bot-001", {"side": "BUY", "profit": 100.0})
        json_str = event.to_json()

        restored = Event.from_json(json_str)

        assert restored.bot_id == "bot-001"
        assert restored.type == EventType.TRADE
        assert restored.data["side"] == "BUY"


# =============================================================================
# Channel Tests
# =============================================================================


class TestChannels:
    """Tests for IPC channel naming."""

    def test_command_channel(self):
        """Test command channel naming."""
        assert Channel.command("bot-001") == "trading:cmd:bot-001"

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
        """Test channel patterns."""
        assert Channel.pattern_all_heartbeats() == "trading:hb:*"
        assert Channel.pattern_all_responses() == "trading:resp:*"


# =============================================================================
# IPC Publisher Tests
# =============================================================================


class TestIPCPublisher:
    """Tests for IPC Publisher."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.publish = AsyncMock(return_value=1)
        return redis

    @pytest.mark.asyncio
    async def test_publish_command(self, mock_redis):
        """Test publishing a command."""
        publisher = IPCPublisher(mock_redis)

        cmd = Command(id="test-001", type=CommandType.START)
        result = await publisher.send_command("bot-001", cmd)

        assert result == 1
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_response(self, mock_redis):
        """Test publishing a response."""
        publisher = IPCPublisher(mock_redis)

        resp = Response.success_response("cmd-001", {"status": "ok"})
        result = await publisher.send_response("bot-001", resp)

        assert result == 1
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_heartbeat(self, mock_redis):
        """Test publishing a heartbeat."""
        publisher = IPCPublisher(mock_redis)

        hb = Heartbeat(bot_id="bot-001", state="running", pid=12345)
        result = await publisher.send_heartbeat("bot-001", hb)

        assert result == 1
        mock_redis.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_event(self, mock_redis):
        """Test publishing an event."""
        publisher = IPCPublisher(mock_redis)

        event = Event.started("bot-001", {"pid": 12345})
        result = await publisher.send_event(event)

        assert result == 1
        mock_redis.publish.assert_called_once()


# =============================================================================
# IPC Subscriber Tests
# =============================================================================


class TestIPCSubscriber:
    """Tests for IPC Subscriber."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.psubscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.punsubscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_pubsub.close = AsyncMock()

        redis.pubsub = MagicMock(return_value=mock_pubsub)

        return redis

    @pytest.mark.asyncio
    async def test_subscribe(self, mock_redis):
        """Test subscribing to a channel."""
        subscriber = IPCSubscriber(mock_redis)

        callback = AsyncMock()
        await subscriber.subscribe("trading:cmd:bot-001", callback)

        mock_redis.pubsub().subscribe.assert_called()

    @pytest.mark.asyncio
    async def test_unsubscribe(self, mock_redis):
        """Test unsubscribing from a channel."""
        subscriber = IPCSubscriber(mock_redis)

        await subscriber.subscribe("trading:cmd:bot-001", AsyncMock())
        await subscriber.unsubscribe("trading:cmd:bot-001")

        mock_redis.pubsub().unsubscribe.assert_called()


# =============================================================================
# IPC Flow Tests (Using Mock Bus)
# =============================================================================


class TestIPCFlow:
    """Tests for complete IPC flows."""

    @pytest.fixture
    def ipc_bus(self):
        """Create mock IPC bus."""
        return MockIPCBus()

    @pytest.mark.asyncio
    async def test_command_response_flow(self, ipc_bus):
        """Test command-response flow."""
        received_commands = []
        received_responses = []

        async def command_handler(data):
            cmd = Command.from_json(data)
            received_commands.append(cmd)

            # Send response
            resp = Response.success_response(cmd.id, {"status": "ok"})
            await ipc_bus.publish(Channel.response("bot-001"), resp.to_json())

        async def response_handler(data):
            received_responses.append(Response.from_json(data))

        # Subscribe
        await ipc_bus.subscribe(Channel.command("bot-001"), command_handler)
        await ipc_bus.subscribe(Channel.response("bot-001"), response_handler)

        # Send command
        cmd = Command(id="cmd-001", type=CommandType.STATUS)
        await ipc_bus.publish(Channel.command("bot-001"), cmd.to_json())

        # Verify
        assert len(received_commands) == 1
        assert received_commands[0].id == "cmd-001"
        assert len(received_responses) == 1
        assert received_responses[0].success is True

    @pytest.mark.asyncio
    async def test_heartbeat_flow(self, ipc_bus):
        """Test heartbeat flow."""
        received_heartbeats = []

        async def heartbeat_handler(data):
            received_heartbeats.append(Heartbeat.from_json(data))

        await ipc_bus.subscribe(Channel.heartbeat("bot-001"), heartbeat_handler)

        # Send heartbeat
        hb = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10},
        )
        await ipc_bus.publish(Channel.heartbeat("bot-001"), hb.to_json())

        # Verify
        assert len(received_heartbeats) == 1
        assert received_heartbeats[0].bot_id == "bot-001"
        assert received_heartbeats[0].metrics["trades"] == 10

    @pytest.mark.asyncio
    async def test_event_broadcast(self, ipc_bus):
        """Test event broadcast."""
        received_events = []

        async def event_handler(data):
            received_events.append(Event.from_json(data))

        await ipc_bus.subscribe(Channel.event(), event_handler)

        # Send events from different bots
        event1 = Event.started("bot-001", {"pid": 1001})
        event2 = Event.trade("bot-002", {"side": "BUY"})
        event3 = Event.error("bot-003", "Connection failed")

        await ipc_bus.publish(Channel.event(), event1.to_json())
        await ipc_bus.publish(Channel.event(), event2.to_json())
        await ipc_bus.publish(Channel.event(), event3.to_json())

        # Verify
        assert len(received_events) == 3
        assert received_events[0].type == EventType.STARTED
        assert received_events[1].type == EventType.TRADE
        assert received_events[2].type == EventType.ERROR

    @pytest.mark.asyncio
    async def test_multiple_bots(self, ipc_bus):
        """Test IPC with multiple bots."""
        bot_commands = {"bot-a": [], "bot-b": [], "bot-c": []}

        async def make_handler(bot_id):
            async def handler(data):
                bot_commands[bot_id].append(Command.from_json(data))
            return handler

        # Subscribe each bot
        for bot_id in bot_commands:
            handler = await make_handler(bot_id)
            await ipc_bus.subscribe(Channel.command(bot_id), handler)

        # Send command to specific bot
        cmd = Command(id="cmd-001", type=CommandType.START)
        await ipc_bus.publish(Channel.command("bot-b"), cmd.to_json())

        # Verify only bot-b received it
        assert len(bot_commands["bot-a"]) == 0
        assert len(bot_commands["bot-b"]) == 1
        assert len(bot_commands["bot-c"]) == 0


# =============================================================================
# Master IPC Handler Tests
# =============================================================================


class TestMasterIPCHandler:
    """Tests for Master IPC Handler."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.publish = AsyncMock(return_value=1)

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.psubscribe = AsyncMock()
        mock_pubsub.unsubscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_pubsub.close = AsyncMock()

        redis.pubsub = MagicMock(return_value=mock_pubsub)

        return redis

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        registry = MagicMock()
        registry.get_all.return_value = []
        registry.get.return_value = None
        registry.update_heartbeat = MagicMock()
        registry.update_state = MagicMock()
        registry.set_error = MagicMock()
        return registry

    @pytest.fixture
    def mock_heartbeat_monitor(self):
        """Create mock heartbeat monitor."""
        monitor = MagicMock()
        monitor.receive = MagicMock()
        return monitor

    @pytest.mark.asyncio
    async def test_handler_start(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handler start."""
        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        await handler.start()

        assert handler.is_running is True

        await handler.stop()

    @pytest.mark.asyncio
    async def test_handler_stop(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handler stop."""
        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        await handler.start()
        await handler.stop()

        assert handler.is_running is False

    @pytest.mark.asyncio
    async def test_on_heartbeat(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handling heartbeat."""
        mock_registry.get.return_value = BotInfo(
            bot_id="bot-001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            market_type=MarketType.SPOT,
            state=BotState.RUNNING,
        )

        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        hb = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10},
        )
        await handler._on_heartbeat(hb.to_json())

        mock_registry.update_heartbeat.assert_called_once()
        mock_heartbeat_monitor.receive.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_event_started(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handling STARTED event."""
        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        event = Event.started("bot-001", {"pid": 12345})
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with("bot-001", BotState.RUNNING)

    @pytest.mark.asyncio
    async def test_on_event_stopped(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handling STOPPED event."""
        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        event = Event.stopped("bot-001", "Manual stop")
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with("bot-001", BotState.STOPPED)

    @pytest.mark.asyncio
    async def test_on_event_error(self, mock_redis, mock_registry, mock_heartbeat_monitor):
        """Test handling ERROR event."""
        handler = MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)

        event = Event.error("bot-001", "Connection failed")
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with("bot-001", BotState.ERROR)
        mock_registry.set_error.assert_called_once()
