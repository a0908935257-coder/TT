"""
Tests for IPC message models.

Validates serialization, deserialization, and factory methods.
"""

import json
import os
from datetime import datetime, timezone

import pytest

from src.ipc import (
    Command,
    CommandType,
    Event,
    EventType,
    Heartbeat,
    Response,
)


# =============================================================================
# CommandType Tests
# =============================================================================


class TestCommandType:
    """Tests for CommandType enum."""

    def test_all_values_exist(self):
        """Test all command types are defined."""
        assert CommandType.START.value == "start"
        assert CommandType.STOP.value == "stop"
        assert CommandType.PAUSE.value == "pause"
        assert CommandType.RESUME.value == "resume"
        assert CommandType.STATUS.value == "status"
        assert CommandType.SHUTDOWN.value == "shutdown"

    def test_from_string(self):
        """Test creating from string value."""
        assert CommandType("start") == CommandType.START
        assert CommandType("stop") == CommandType.STOP


# =============================================================================
# EventType Tests
# =============================================================================


class TestEventType:
    """Tests for EventType enum."""

    def test_all_values_exist(self):
        """Test all event types are defined."""
        assert EventType.STARTED.value == "started"
        assert EventType.STOPPED.value == "stopped"
        assert EventType.ERROR.value == "error"
        assert EventType.TRADE.value == "trade"
        assert EventType.ALERT.value == "alert"

    def test_from_string(self):
        """Test creating from string value."""
        assert EventType("started") == EventType.STARTED
        assert EventType("error") == EventType.ERROR


# =============================================================================
# Command Tests
# =============================================================================


class TestCommand:
    """Tests for Command message."""

    def test_default_values(self):
        """Test default values are set correctly."""
        cmd = Command()
        assert cmd.id is not None
        assert len(cmd.id) == 36  # UUID format
        assert cmd.type == CommandType.STATUS
        assert cmd.params == {}
        assert cmd.timestamp is not None

    def test_custom_values(self):
        """Test custom values."""
        cmd = Command(
            id="test-123",
            type=CommandType.START,
            params={"symbol": "BTCUSDT"},
        )
        assert cmd.id == "test-123"
        assert cmd.type == CommandType.START
        assert cmd.params == {"symbol": "BTCUSDT"}

    def test_to_dict(self):
        """Test conversion to dictionary."""
        cmd = Command(
            id="test-123",
            type=CommandType.STOP,
            params={"clear_position": True},
        )
        result = cmd.to_dict()

        assert result["id"] == "test-123"
        assert result["type"] == "stop"
        assert result["params"] == {"clear_position": True}
        assert "timestamp" in result

    def test_to_json(self):
        """Test JSON serialization."""
        cmd = Command(id="test-123", type=CommandType.PAUSE)
        json_str = cmd.to_json()

        data = json.loads(json_str)
        assert data["id"] == "test-123"
        assert data["type"] == "pause"

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "id": "test-456",
            "type": "resume",
            "params": {"force": True},
            "timestamp": "2026-01-18T12:00:00+00:00",
        }
        cmd = Command.from_dict(data)

        assert cmd.id == "test-456"
        assert cmd.type == CommandType.RESUME
        assert cmd.params == {"force": True}
        assert cmd.timestamp.year == 2026

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"id": "test-789", "type": "status", "params": {}, "timestamp": "2026-01-18T12:00:00+00:00"}'
        cmd = Command.from_json(json_str)

        assert cmd.id == "test-789"
        assert cmd.type == CommandType.STATUS

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Command(
            id="roundtrip-test",
            type=CommandType.SHUTDOWN,
            params={"timeout": 30},
        )
        json_str = original.to_json()
        restored = Command.from_json(json_str)

        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.params == original.params


# =============================================================================
# Response Tests
# =============================================================================


class TestResponse:
    """Tests for Response message."""

    def test_default_values(self):
        """Test default values."""
        resp = Response()
        assert resp.command_id == ""
        assert resp.success is True
        assert resp.data == {}
        assert resp.error is None
        assert resp.timestamp is not None

    def test_success_response(self):
        """Test success response factory."""
        resp = Response.success_response(
            command_id="cmd-123",
            data={"status": "running"},
        )
        assert resp.command_id == "cmd-123"
        assert resp.success is True
        assert resp.data == {"status": "running"}
        assert resp.error is None

    def test_error_response(self):
        """Test error response factory."""
        resp = Response.error_response(
            command_id="cmd-456",
            error="Bot not found",
        )
        assert resp.command_id == "cmd-456"
        assert resp.success is False
        assert resp.error == "Bot not found"

    def test_to_json(self):
        """Test JSON serialization."""
        resp = Response(
            command_id="cmd-789",
            success=True,
            data={"trades": 10},
        )
        json_str = resp.to_json()
        data = json.loads(json_str)

        assert data["command_id"] == "cmd-789"
        assert data["success"] is True
        assert data["data"] == {"trades": 10}

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"command_id": "cmd-abc", "success": false, "data": {}, "error": "Timeout", "timestamp": "2026-01-18T12:00:00+00:00"}'
        resp = Response.from_json(json_str)

        assert resp.command_id == "cmd-abc"
        assert resp.success is False
        assert resp.error == "Timeout"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Response(
            command_id="roundtrip",
            success=True,
            data={"profit": "100.50"},
        )
        json_str = original.to_json()
        restored = Response.from_json(json_str)

        assert restored.command_id == original.command_id
        assert restored.success == original.success
        assert restored.data == original.data


# =============================================================================
# Heartbeat Tests
# =============================================================================


class TestHeartbeat:
    """Tests for Heartbeat message."""

    def test_default_values(self):
        """Test default values."""
        hb = Heartbeat()
        assert hb.bot_id == ""
        assert hb.state == "unknown"
        assert hb.pid == os.getpid()
        assert hb.metrics == {}
        assert hb.timestamp is not None

    def test_custom_values(self):
        """Test custom values."""
        hb = Heartbeat(
            bot_id="bot-001",
            state="running",
            pid=12345,
            metrics={"trades": 5, "profit": 100.0},
        )
        assert hb.bot_id == "bot-001"
        assert hb.state == "running"
        assert hb.pid == 12345
        assert hb.metrics["trades"] == 5

    def test_to_json(self):
        """Test JSON serialization."""
        hb = Heartbeat(
            bot_id="bot-002",
            state="paused",
            pid=99999,
            metrics={"uptime": 3600},
        )
        json_str = hb.to_json()
        data = json.loads(json_str)

        assert data["bot_id"] == "bot-002"
        assert data["state"] == "paused"
        assert data["pid"] == 99999
        assert data["metrics"]["uptime"] == 3600

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"bot_id": "bot-003", "state": "stopped", "pid": 11111, "metrics": {}, "timestamp": "2026-01-18T12:00:00+00:00"}'
        hb = Heartbeat.from_json(json_str)

        assert hb.bot_id == "bot-003"
        assert hb.state == "stopped"
        assert hb.pid == 11111

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Heartbeat(
            bot_id="roundtrip-bot",
            state="running",
            pid=54321,
            metrics={"total_trades": 100},
        )
        json_str = original.to_json()
        restored = Heartbeat.from_json(json_str)

        assert restored.bot_id == original.bot_id
        assert restored.state == original.state
        assert restored.pid == original.pid
        assert restored.metrics == original.metrics


# =============================================================================
# Event Tests
# =============================================================================


class TestEvent:
    """Tests for Event message."""

    def test_default_values(self):
        """Test default values."""
        event = Event()
        assert event.bot_id == ""
        assert event.type == EventType.ALERT
        assert event.data == {}
        assert event.timestamp is not None

    def test_started_factory(self):
        """Test STARTED event factory."""
        event = Event.started("bot-001", {"symbol": "BTCUSDT"})
        assert event.bot_id == "bot-001"
        assert event.type == EventType.STARTED
        assert event.data == {"symbol": "BTCUSDT"}

    def test_stopped_factory(self):
        """Test STOPPED event factory."""
        event = Event.stopped("bot-002", "Manual stop")
        assert event.bot_id == "bot-002"
        assert event.type == EventType.STOPPED
        assert event.data["reason"] == "Manual stop"

    def test_error_factory(self):
        """Test ERROR event factory."""
        event = Event.error(
            "bot-003",
            "Connection failed",
            {"retry_count": 3},
        )
        assert event.bot_id == "bot-003"
        assert event.type == EventType.ERROR
        assert event.data["error"] == "Connection failed"
        assert event.data["retry_count"] == 3

    def test_trade_factory(self):
        """Test TRADE event factory."""
        trade_data = {
            "side": "BUY",
            "price": "50000.00",
            "quantity": "0.001",
            "profit": "10.50",
        }
        event = Event.trade("bot-004", trade_data)
        assert event.bot_id == "bot-004"
        assert event.type == EventType.TRADE
        assert event.data == trade_data

    def test_alert_factory(self):
        """Test ALERT event factory."""
        event = Event.alert("bot-005", "Price reached target", "info")
        assert event.bot_id == "bot-005"
        assert event.type == EventType.ALERT
        assert event.data["message"] == "Price reached target"
        assert event.data["level"] == "info"

    def test_to_json(self):
        """Test JSON serialization."""
        event = Event(
            bot_id="bot-006",
            type=EventType.TRADE,
            data={"side": "SELL"},
        )
        json_str = event.to_json()
        data = json.loads(json_str)

        assert data["bot_id"] == "bot-006"
        assert data["type"] == "trade"
        assert data["data"]["side"] == "SELL"

    def test_from_json(self):
        """Test JSON deserialization."""
        json_str = '{"bot_id": "bot-007", "type": "error", "data": {"error": "Test"}, "timestamp": "2026-01-18T12:00:00+00:00"}'
        event = Event.from_json(json_str)

        assert event.bot_id == "bot-007"
        assert event.type == EventType.ERROR
        assert event.data["error"] == "Test"

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = Event(
            bot_id="roundtrip-bot",
            type=EventType.ALERT,
            data={"message": "Test alert", "level": "warning"},
        )
        json_str = original.to_json()
        restored = Event.from_json(json_str)

        assert restored.bot_id == original.bot_id
        assert restored.type == original.type
        assert restored.data == original.data


# =============================================================================
# Integration Tests
# =============================================================================


class TestIPCIntegration:
    """Integration tests for IPC messages."""

    def test_command_response_flow(self):
        """Test command and response flow."""
        # Master sends command
        cmd = Command(
            type=CommandType.STATUS,
            params={"include_metrics": True},
        )

        # Simulate sending over network
        cmd_json = cmd.to_json()

        # Bot receives and parses
        received_cmd = Command.from_json(cmd_json)

        # Bot creates response
        resp = Response.success_response(
            command_id=received_cmd.id,
            data={"state": "running", "trades": 10},
        )

        # Simulate sending over network
        resp_json = resp.to_json()

        # Master receives and parses
        received_resp = Response.from_json(resp_json)

        assert received_resp.command_id == cmd.id
        assert received_resp.success is True
        assert received_resp.data["state"] == "running"

    def test_heartbeat_flow(self):
        """Test heartbeat flow."""
        # Bot creates heartbeat
        hb = Heartbeat(
            bot_id="grid-bot-001",
            state="running",
            metrics={
                "uptime_seconds": 3600,
                "total_trades": 50,
                "total_profit": 250.0,
            },
        )

        # Simulate sending over network
        hb_json = hb.to_json()

        # Master receives and parses
        received_hb = Heartbeat.from_json(hb_json)

        assert received_hb.bot_id == "grid-bot-001"
        assert received_hb.state == "running"
        assert received_hb.metrics["total_trades"] == 50

    def test_event_notification_flow(self):
        """Test event notification flow."""
        # Bot creates trade event
        event = Event.trade(
            bot_id="grid-bot-002",
            trade_data={
                "order_id": "12345",
                "side": "BUY",
                "price": "50000.00",
                "quantity": "0.001",
                "profit": "5.25",
            },
        )

        # Simulate sending over network
        event_json = event.to_json()

        # Master receives and parses
        received_event = Event.from_json(event_json)

        assert received_event.bot_id == "grid-bot-002"
        assert received_event.type == EventType.TRADE
        assert received_event.data["order_id"] == "12345"
        assert received_event.data["profit"] == "5.25"
