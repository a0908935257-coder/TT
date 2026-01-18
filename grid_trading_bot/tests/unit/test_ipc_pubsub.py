"""
Tests for IPC Pub/Sub (Channel, Publisher, Subscriber).

Uses mocked Redis client for unit testing.
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


# =============================================================================
# Channel Tests
# =============================================================================


class TestChannel:
    """Tests for Channel name generation."""

    def test_prefix(self):
        """Test channel prefix is correct."""
        assert Channel.PREFIX == "trading"

    def test_command_channel(self):
        """Test command channel generation."""
        channel = Channel.command("bot-001")
        assert channel == "trading:cmd:bot-001"

    def test_response_channel(self):
        """Test response channel generation."""
        channel = Channel.response("bot-002")
        assert channel == "trading:resp:bot-002"

    def test_heartbeat_channel(self):
        """Test heartbeat channel generation."""
        channel = Channel.heartbeat("bot-003")
        assert channel == "trading:hb:bot-003"

    def test_event_channel(self):
        """Test event channel generation."""
        channel = Channel.event()
        assert channel == "trading:event"

    def test_pattern_all_heartbeats(self):
        """Test heartbeat pattern."""
        pattern = Channel.pattern_all_heartbeats()
        assert pattern == "trading:hb:*"

    def test_pattern_all_responses(self):
        """Test response pattern."""
        pattern = Channel.pattern_all_responses()
        assert pattern == "trading:resp:*"

    def test_unique_channels_per_bot(self):
        """Test different bots have different channels."""
        cmd1 = Channel.command("bot-a")
        cmd2 = Channel.command("bot-b")
        assert cmd1 != cmd2
        assert "bot-a" in cmd1
        assert "bot-b" in cmd2


# =============================================================================
# IPCPublisher Tests
# =============================================================================


class TestIPCPublisher:
    """Tests for IPCPublisher."""

    @pytest.fixture
    def mock_redis(self):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.publish = AsyncMock(return_value=1)
        return redis

    @pytest.fixture
    def publisher(self, mock_redis):
        """Create publisher with mock Redis."""
        return IPCPublisher(mock_redis)

    @pytest.mark.asyncio
    async def test_publish_string(self, publisher, mock_redis):
        """Test publishing a string message."""
        result = await publisher.publish("test:channel", '{"test": "data"}')

        assert result == 1
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "test:channel"
        assert call_args[0][1] == '{"test": "data"}'

    @pytest.mark.asyncio
    async def test_publish_message(self, publisher, mock_redis):
        """Test publishing a BaseMessage."""
        command = Command(id="test-123", type=CommandType.STATUS)
        result = await publisher.publish("test:channel", command)

        assert result == 1
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert "test-123" in call_args[0][1]
        assert "status" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_send_command(self, publisher, mock_redis):
        """Test sending a command."""
        command = Command(type=CommandType.START, params={"symbol": "BTCUSDT"})
        result = await publisher.send_command("bot-001", command)

        assert result == 1
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "trading:cmd:bot-001"
        assert "start" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_send_response(self, publisher, mock_redis):
        """Test sending a response."""
        response = Response.success_response("cmd-123", {"state": "running"})
        result = await publisher.send_response("bot-002", response)

        assert result == 1
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "trading:resp:bot-002"
        assert "cmd-123" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_send_heartbeat(self, publisher, mock_redis):
        """Test sending a heartbeat."""
        heartbeat = Heartbeat(bot_id="bot-003", state="running", pid=12345)
        result = await publisher.send_heartbeat("bot-003", heartbeat)

        assert result == 1
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "trading:hb:bot-003"
        assert "running" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_send_event(self, publisher, mock_redis):
        """Test sending an event."""
        event = Event.trade("bot-004", {"side": "BUY", "price": "50000"})
        result = await publisher.send_event(event)

        assert result == 1
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "trading:event"
        assert "trade" in call_args[0][1]


# =============================================================================
# IPCSubscriber Tests
# =============================================================================


class TestIPCSubscriber:
    """Tests for IPCSubscriber."""

    @pytest.fixture
    def mock_pubsub(self):
        """Create mock PubSub."""
        pubsub = AsyncMock()
        pubsub.subscribe = AsyncMock()
        pubsub.psubscribe = AsyncMock()
        pubsub.unsubscribe = AsyncMock()
        pubsub.punsubscribe = AsyncMock()
        pubsub.close = AsyncMock()
        pubsub.get_message = AsyncMock(return_value=None)
        return pubsub

    @pytest.fixture
    def mock_redis(self, mock_pubsub):
        """Create mock Redis client."""
        redis = AsyncMock()
        redis.pubsub = MagicMock(return_value=mock_pubsub)
        return redis

    @pytest.fixture
    def subscriber(self, mock_redis):
        """Create subscriber with mock Redis."""
        return IPCSubscriber(mock_redis)

    def test_initial_state(self, subscriber):
        """Test initial state is correct."""
        assert subscriber.is_running is False
        assert subscriber.subscribed_channels == []
        assert subscriber.subscribed_patterns == []

    @pytest.mark.asyncio
    async def test_subscribe(self, subscriber, mock_pubsub):
        """Test subscribing to a channel."""
        handler = AsyncMock()
        await subscriber.subscribe("test:channel", handler)

        mock_pubsub.subscribe.assert_called_once_with("test:channel")
        assert "test:channel" in subscriber.subscribed_channels

    @pytest.mark.asyncio
    async def test_psubscribe(self, subscriber, mock_pubsub):
        """Test subscribing to a pattern."""
        handler = AsyncMock()
        await subscriber.psubscribe("test:*", handler)

        mock_pubsub.psubscribe.assert_called_once_with("test:*")
        assert "test:*" in subscriber.subscribed_patterns

    @pytest.mark.asyncio
    async def test_unsubscribe(self, subscriber, mock_pubsub):
        """Test unsubscribing from a channel."""
        handler = AsyncMock()
        await subscriber.subscribe("test:channel", handler)
        await subscriber.unsubscribe("test:channel")

        mock_pubsub.unsubscribe.assert_called_once_with("test:channel")
        assert "test:channel" not in subscriber.subscribed_channels

    @pytest.mark.asyncio
    async def test_punsubscribe(self, subscriber, mock_pubsub):
        """Test unsubscribing from a pattern."""
        handler = AsyncMock()
        await subscriber.psubscribe("test:*", handler)
        await subscriber.punsubscribe("test:*")

        mock_pubsub.punsubscribe.assert_called_once_with("test:*")
        assert "test:*" not in subscriber.subscribed_patterns

    @pytest.mark.asyncio
    async def test_start_creates_task(self, subscriber, mock_pubsub):
        """Test starting creates a listen task."""
        handler = AsyncMock()
        await subscriber.subscribe("test:channel", handler)
        await subscriber.start()

        assert subscriber.is_running is True
        assert subscriber._listen_task is not None

        await subscriber.stop()

    @pytest.mark.asyncio
    async def test_start_without_subscriptions(self, subscriber):
        """Test starting without subscriptions doesn't start."""
        await subscriber.start()
        assert subscriber.is_running is False

    @pytest.mark.asyncio
    async def test_stop(self, subscriber, mock_pubsub):
        """Test stopping the subscriber."""
        handler = AsyncMock()
        await subscriber.subscribe("test:channel", handler)
        await subscriber.start()
        await subscriber.stop()

        assert subscriber.is_running is False
        assert subscriber._listen_task is None
        mock_pubsub.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_message(self, subscriber, mock_pubsub):
        """Test message handling."""
        received_messages = []

        async def handler(data):
            received_messages.append(data)

        await subscriber.subscribe("test:channel", handler)

        # Simulate a message
        message = {
            "type": "message",
            "channel": b"test:channel",
            "data": b'{"test": "data"}',
        }
        await subscriber._handle_message(message)

        assert len(received_messages) == 1
        assert received_messages[0] == '{"test": "data"}'

    @pytest.mark.asyncio
    async def test_handle_pmessage(self, subscriber, mock_pubsub):
        """Test pattern message handling."""
        received_messages = []

        async def handler(data):
            received_messages.append(data)

        await subscriber.psubscribe("test:*", handler)

        # Simulate a pattern message
        message = {
            "type": "pmessage",
            "pattern": b"test:*",
            "channel": b"test:channel",
            "data": b'{"test": "pdata"}',
        }
        await subscriber._handle_message(message)

        assert len(received_messages) == 1
        assert received_messages[0] == '{"test": "pdata"}'

    @pytest.mark.asyncio
    async def test_ignore_subscribe_messages(self, subscriber, mock_pubsub):
        """Test subscribe confirmation messages are ignored."""
        received_messages = []

        async def handler(data):
            received_messages.append(data)

        await subscriber.subscribe("test:channel", handler)

        # Simulate a subscribe confirmation
        message = {
            "type": "subscribe",
            "channel": b"test:channel",
            "data": 1,
        }
        await subscriber._handle_message(message)

        assert len(received_messages) == 0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIPCPubSubIntegration:
    """Integration tests for IPC Pub/Sub (mocked)."""

    @pytest.mark.asyncio
    async def test_command_response_flow(self):
        """Test full command-response flow with mocked Redis."""
        # Create mocked Redis
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)

        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_pubsub.close = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        # Create publisher and subscriber
        publisher = IPCPublisher(mock_redis)
        subscriber = IPCSubscriber(mock_redis)

        # Track received commands
        received_commands = []

        async def command_handler(data):
            cmd = Command.from_json(data)
            received_commands.append(cmd)

        # Subscribe to commands
        bot_id = "test-bot"
        await subscriber.subscribe(Channel.command(bot_id), command_handler)

        # Publish a command
        command = Command(type=CommandType.STATUS)
        await publisher.send_command(bot_id, command)

        # Verify publish was called
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == f"trading:cmd:{bot_id}"

        # Simulate receiving the message
        message = {
            "type": "message",
            "channel": f"trading:cmd:{bot_id}".encode(),
            "data": command.to_json().encode(),
        }
        await subscriber._handle_message(message)

        # Verify handler was called
        assert len(received_commands) == 1
        assert received_commands[0].type == CommandType.STATUS

    @pytest.mark.asyncio
    async def test_heartbeat_flow(self):
        """Test heartbeat flow with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)

        publisher = IPCPublisher(mock_redis)

        # Send heartbeat
        heartbeat = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10},
        )
        await publisher.send_heartbeat("bot-001", heartbeat)

        # Verify
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "trading:hb:bot-001"
        assert "running" in call_args[0][1]
        assert "12345" in call_args[0][1]

    @pytest.mark.asyncio
    async def test_event_broadcast(self):
        """Test event broadcast with mocked Redis."""
        mock_redis = AsyncMock()
        mock_redis.publish = AsyncMock(return_value=1)

        publisher = IPCPublisher(mock_redis)

        # Send events from multiple bots
        event1 = Event.trade("bot-001", {"side": "BUY"})
        event2 = Event.trade("bot-002", {"side": "SELL"})

        await publisher.send_event(event1)
        await publisher.send_event(event2)

        # Verify both went to the same channel
        assert mock_redis.publish.call_count == 2
        for call in mock_redis.publish.call_args_list:
            assert call[0][0] == "trading:event"

    @pytest.mark.asyncio
    async def test_multiple_subscriptions(self):
        """Test subscribing to multiple channels."""
        mock_redis = AsyncMock()
        mock_pubsub = AsyncMock()
        mock_pubsub.subscribe = AsyncMock()
        mock_pubsub.get_message = AsyncMock(return_value=None)
        mock_pubsub.close = AsyncMock()
        mock_redis.pubsub = MagicMock(return_value=mock_pubsub)

        subscriber = IPCSubscriber(mock_redis)

        handlers_called = {"cmd": 0, "hb": 0}

        async def cmd_handler(data):
            handlers_called["cmd"] += 1

        async def hb_handler(data):
            handlers_called["hb"] += 1

        # Subscribe to multiple channels
        await subscriber.subscribe(Channel.command("bot-001"), cmd_handler)
        await subscriber.subscribe(Channel.heartbeat("bot-001"), hb_handler)

        assert len(subscriber.subscribed_channels) == 2

        # Simulate messages
        cmd_msg = {
            "type": "message",
            "channel": b"trading:cmd:bot-001",
            "data": b'{"type": "status"}',
        }
        hb_msg = {
            "type": "message",
            "channel": b"trading:hb:bot-001",
            "data": b'{"state": "running"}',
        }

        await subscriber._handle_message(cmd_msg)
        await subscriber._handle_message(hb_msg)

        assert handlers_called["cmd"] == 1
        assert handlers_called["hb"] == 1
