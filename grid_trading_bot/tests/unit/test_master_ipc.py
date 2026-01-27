"""
Tests for Master IPC Handler.

Validates command sending, response handling, and heartbeat processing.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ipc import Command, CommandType, Event, EventType, Heartbeat, Response
from src.master.ipc_handler import MasterIPCHandler
from src.master.models import BotInfo, BotState, BotType, MarketType


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = AsyncMock()
    redis.publish = AsyncMock(return_value=1)

    mock_pubsub = AsyncMock()
    mock_pubsub.subscribe = AsyncMock()
    mock_pubsub.unsubscribe = AsyncMock()
    mock_pubsub.get_message = AsyncMock(return_value=None)
    mock_pubsub.close = AsyncMock()
    redis.pubsub = MagicMock(return_value=mock_pubsub)

    return redis


@pytest.fixture
def mock_registry():
    """Create mock registry."""
    registry = MagicMock()
    registry.get_all.return_value = []
    registry.get.return_value = None
    registry.update_heartbeat = AsyncMock()  # async method
    registry.update_state = AsyncMock()  # async method
    registry.set_error = MagicMock()
    return registry


@pytest.fixture
def mock_heartbeat_monitor():
    """Create mock heartbeat monitor."""
    monitor = MagicMock()
    monitor.receive = AsyncMock()  # receive is async
    return monitor


@pytest.fixture
def handler(mock_redis, mock_registry, mock_heartbeat_monitor):
    """Create handler with mocks."""
    return MasterIPCHandler(mock_redis, mock_registry, mock_heartbeat_monitor)


# =============================================================================
# MasterIPCHandler Tests
# =============================================================================


class TestMasterIPCHandler:
    """Tests for MasterIPCHandler."""

    def test_initialization(self, handler):
        """Test handler initialization."""
        assert handler.is_running is False
        assert len(handler._pending) == 0

    @pytest.mark.asyncio
    async def test_start(self, handler, mock_registry):
        """Test starting the handler."""
        mock_registry.get_all.return_value = [
            BotInfo(bot_id="bot-001", bot_type=BotType.GRID, symbol="BTCUSDT", market_type=MarketType.SPOT),
        ]

        await handler.start()

        assert handler.is_running is True
        # Verify subscriptions were made
        assert handler._subscriber._pubsub.subscribe.called

    @pytest.mark.asyncio
    async def test_stop(self, handler):
        """Test stopping the handler."""
        await handler.start()
        await handler.stop()

        assert handler.is_running is False

    @pytest.mark.asyncio
    async def test_subscribe_bot(self, handler):
        """Test subscribing to a new bot."""
        await handler.subscribe_bot("new-bot")

        # Verify subscriptions were made
        assert handler._subscriber._pubsub.subscribe.call_count >= 2

    @pytest.mark.asyncio
    async def test_unsubscribe_bot(self, handler):
        """Test unsubscribing from a bot."""
        await handler.subscribe_bot("bot-001")
        await handler.unsubscribe_bot("bot-001")

        assert handler._subscriber._pubsub.unsubscribe.called

    @pytest.mark.asyncio
    async def test_send_command_success(self, handler, mock_redis):
        """Test sending a command with successful response."""
        # Setup
        await handler.start()

        # Simulate response arriving
        async def mock_publish(channel, data):
            # Parse command to get ID
            import json
            cmd_data = json.loads(data)
            cmd_id = cmd_data["id"]

            # Simulate response arriving after a delay
            await asyncio.sleep(0.1)
            if cmd_id in handler._pending:
                response = Response.success_response(cmd_id, {"status": "started"})
                handler._pending[cmd_id].set_result(response)
            return 1

        mock_redis.publish = mock_publish

        # Send command
        response = await handler.send_command("bot-001", CommandType.START, timeout=5.0)

        assert response.success is True
        assert response.data["status"] == "started"

        await handler.stop()

    @pytest.mark.asyncio
    async def test_send_command_timeout(self, handler):
        """Test command timeout."""
        await handler.start()

        # Send command that will timeout (no response)
        response = await handler.send_command("bot-001", CommandType.STATUS, timeout=0.1)

        assert response.success is False
        assert "timed out" in response.error

        await handler.stop()

    @pytest.mark.asyncio
    async def test_on_response(self, handler):
        """Test processing incoming response."""
        # Create pending command
        future = asyncio.Future()
        handler._pending["cmd-123"] = future

        # Process response
        response = Response.success_response("cmd-123", {"test": "data"})
        await handler._on_response(response.to_json())

        # Verify future was completed
        assert future.done()
        result = future.result()
        assert result.success is True
        assert result.data["test"] == "data"

    @pytest.mark.asyncio
    async def test_on_response_no_pending(self, handler):
        """Test response for unknown command is ignored."""
        response = Response.success_response("unknown-cmd", {})
        # Should not raise
        await handler._on_response(response.to_json())

    @pytest.mark.asyncio
    async def test_on_heartbeat(self, handler, mock_registry, mock_heartbeat_monitor):
        """Test processing incoming heartbeat."""
        mock_registry.get.return_value = BotInfo(
            bot_id="bot-001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            market_type=MarketType.SPOT,
            state=BotState.RUNNING,
        )

        heartbeat = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 10},
        )
        await handler._on_heartbeat(heartbeat.to_json())

        # Verify registry was updated
        mock_registry.update_heartbeat.assert_called_once()

        # Verify heartbeat monitor was notified
        mock_heartbeat_monitor.receive.assert_called_once()

    @pytest.mark.asyncio
    async def test_on_event_started(self, handler, mock_registry):
        """Test processing STARTED event."""
        event = Event.started("bot-001", {"pid": 12345})
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with("bot-001", BotState.RUNNING)

    @pytest.mark.asyncio
    async def test_on_event_stopped(self, handler, mock_registry):
        """Test processing STOPPED event."""
        event = Event.stopped("bot-001", "Manual stop")
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with("bot-001", BotState.STOPPED)

    @pytest.mark.asyncio
    async def test_on_event_error(self, handler, mock_registry):
        """Test processing ERROR event."""
        event = Event.error("bot-001", "Connection failed")
        await handler._on_event(event.to_json())

        mock_registry.update_state.assert_called_once_with(
            "bot-001", BotState.ERROR, message="Connection failed"
        )

    @pytest.mark.asyncio
    async def test_event_callback(self, handler):
        """Test event callback is invoked."""
        callback_events = []

        async def callback(event):
            callback_events.append(event)

        handler._event_callback = callback

        event = Event.trade("bot-001", {"side": "BUY"})
        await handler._on_event(event.to_json())

        assert len(callback_events) == 1
        assert callback_events[0].type == EventType.TRADE


# =============================================================================
# Integration Tests
# =============================================================================


class TestMasterIPCIntegration:
    """Integration tests for Master IPC."""

    @pytest.mark.asyncio
    async def test_command_response_flow(self, handler, mock_redis):
        """Test full command-response flow."""
        await handler.start()

        # Create a task to send response
        async def respond_to_command():
            await asyncio.sleep(0.05)
            # Find the pending command
            if handler._pending:
                cmd_id = list(handler._pending.keys())[0]
                response = Response.success_response(cmd_id, {"state": "running"})
                await handler._on_response(response.to_json())

        # Start responder task
        responder = asyncio.create_task(respond_to_command())

        # Send command
        response = await handler.send_command("bot-001", CommandType.STATUS, timeout=1.0)

        await responder

        assert response.success is True
        assert response.data["state"] == "running"

        await handler.stop()
