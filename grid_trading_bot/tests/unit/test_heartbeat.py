"""
Unit tests for Heartbeat Monitor.

Tests heartbeat receiving, timeout detection, and monitoring functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master import (
    BotRegistry,
    BotState,
    BotType,
    HeartbeatConfig,
    HeartbeatData,
    HeartbeatMonitor,
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
def config():
    """Create heartbeat config for testing."""
    return HeartbeatConfig(
        interval=1,  # Short interval for testing
        timeout=5,   # Short timeout for testing
        max_missed=2,
        auto_restart=False,
    )


@pytest.fixture
def mock_notifier():
    """Create a mock notifier."""
    notifier = MagicMock()
    notifier.notify_bot_timeout = AsyncMock()
    notifier.notify_bot_recovered = AsyncMock()
    return notifier


@pytest.fixture
def monitor(registry, config, mock_notifier):
    """Create heartbeat monitor for testing."""
    return HeartbeatMonitor(registry, config, mock_notifier)


@pytest.fixture
def sample_config():
    """Create sample bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "total_investment": "10000",
    }


class TestHeartbeatData:
    """Tests for HeartbeatData model."""

    def test_create_heartbeat_data(self):
        """Test creating HeartbeatData."""
        data = HeartbeatData(
            bot_id="bot_001",
            state=BotState.RUNNING,
            metrics={
                "uptime_seconds": 3600,
                "total_trades": 10,
                "total_profit": "150.5",
                "pending_orders": 5,
                "memory_mb": 256.5,
                "cpu_percent": 15.2,
            },
        )

        assert data.bot_id == "bot_001"
        assert data.state == BotState.RUNNING
        assert data.uptime_seconds == 3600
        assert data.total_trades == 10
        assert data.total_profit == Decimal("150.5")
        assert data.pending_orders == 5
        assert data.memory_mb == 256.5
        assert data.cpu_percent == 15.2

    def test_to_dict(self):
        """Test converting HeartbeatData to dict."""
        data = HeartbeatData(
            bot_id="bot_001",
            state=BotState.RUNNING,
            metrics={"total_trades": 10},
        )

        result = data.to_dict()

        assert result["bot_id"] == "bot_001"
        assert result["state"] == "running"
        assert "timestamp" in result


class TestHeartbeatConfig:
    """Tests for HeartbeatConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = HeartbeatConfig()

        assert config.interval == 10
        assert config.timeout == 60
        assert config.max_missed == 3
        assert config.auto_restart is False

    def test_custom_config(self):
        """Test custom configuration values."""
        config = HeartbeatConfig(
            interval=5,
            timeout=30,
            max_missed=5,
            auto_restart=True,
        )

        assert config.interval == 5
        assert config.timeout == 30
        assert config.max_missed == 5
        assert config.auto_restart is True


class TestHeartbeatMonitor:
    """Tests for HeartbeatMonitor."""

    @pytest.mark.asyncio
    async def test_receive_heartbeat(self, monitor, registry, sample_config):
        """Test receiving heartbeat updates bot status."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        heartbeat = HeartbeatData(
            bot_id="bot_001",
            state=BotState.RUNNING,
            metrics={"total_trades": 5},
        )

        await monitor.receive(heartbeat)

        assert monitor.is_alive("bot_001")
        assert "bot_001" in monitor

    @pytest.mark.asyncio
    async def test_is_alive_no_heartbeat(self, monitor):
        """Test is_alive returns False for unknown bot."""
        assert not monitor.is_alive("nonexistent")

    @pytest.mark.asyncio
    async def test_get_status(self, monitor, registry, sample_config):
        """Test getting heartbeat status."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        heartbeat = HeartbeatData(
            bot_id="bot_001",
            state=BotState.RUNNING,
            metrics={"total_trades": 5},
        )
        await monitor.receive(heartbeat)

        status = monitor.get_status("bot_001")

        assert status["bot_id"] == "bot_001"
        assert status["alive"] is True
        assert status["missed_count"] == 0
        assert "last_heartbeat" in status

    @pytest.mark.asyncio
    async def test_get_status_unknown_bot(self, monitor):
        """Test getting status for unknown bot."""
        status = monitor.get_status("unknown")

        assert status["bot_id"] == "unknown"
        assert status["alive"] is False
        assert status["last_heartbeat"] is None

    @pytest.mark.asyncio
    async def test_receive_resets_missed_count(self, monitor, registry, sample_config):
        """Test receiving heartbeat resets missed count."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Simulate some missed heartbeats
        monitor._missed_counts["bot_001"] = 2

        # Receive heartbeat
        heartbeat = HeartbeatData(bot_id="bot_001", state=BotState.RUNNING)
        await monitor.receive(heartbeat)

        assert monitor.get_missed_count("bot_001") == 0

    @pytest.mark.asyncio
    async def test_check_all_detects_timeout(self, monitor, registry, sample_config):
        """Test check_all detects timed out bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Add old heartbeat
        old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001",
            timestamp=old_time,
            state=BotState.RUNNING,
        )

        results = await monitor.check_all()

        assert results["bot_001"] is False
        assert monitor.get_missed_count("bot_001") == 1

    @pytest.mark.asyncio
    async def test_check_all_healthy_bot(self, monitor, registry, sample_config):
        """Test check_all for healthy bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Add recent heartbeat
        await monitor.receive(HeartbeatData(bot_id="bot_001", state=BotState.RUNNING))

        results = await monitor.check_all()

        assert results["bot_001"] is True

    @pytest.mark.asyncio
    async def test_timeout_triggers_notification(
        self, monitor, registry, sample_config, mock_notifier
    ):
        """Test timeout triggers notification."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Set missed count to trigger timeout
        monitor._missed_counts["bot_001"] = 1

        # Add old heartbeat
        old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001",
            timestamp=old_time,
            state=BotState.RUNNING,
        )

        await monitor.check_all()

        # Should have notified and updated state
        mock_notifier.notify_bot_timeout.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_sets_error_state(
        self, monitor, registry, sample_config
    ):
        """Test timeout sets bot to ERROR state."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Set missed count to trigger timeout
        monitor._missed_counts["bot_001"] = 1

        # Add old heartbeat
        old_time = datetime.now(timezone.utc) - timedelta(seconds=10)
        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001",
            timestamp=old_time,
            state=BotState.RUNNING,
        )

        await monitor.check_all()

        bot_info = registry.get("bot_001")
        assert bot_info.state == BotState.ERROR

    @pytest.mark.asyncio
    async def test_clear_bot_data(self, monitor):
        """Test clearing bot heartbeat data."""
        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001", state=BotState.RUNNING
        )
        monitor._missed_counts["bot_001"] = 2

        monitor.clear("bot_001")

        assert "bot_001" not in monitor._heartbeats
        assert "bot_001" not in monitor._missed_counts

    @pytest.mark.asyncio
    async def test_get_all_status(self, monitor, registry, sample_config):
        """Test getting status for all bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)

        await monitor.receive(HeartbeatData(bot_id="bot_001", state=BotState.RUNNING))
        await monitor.receive(HeartbeatData(bot_id="bot_002", state=BotState.RUNNING))

        all_status = monitor.get_all_status()

        assert len(all_status) == 2
        bot_ids = [s["bot_id"] for s in all_status]
        assert "bot_001" in bot_ids
        assert "bot_002" in bot_ids

    def test_len(self, monitor):
        """Test __len__ method."""
        assert len(monitor) == 0

        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001", state=BotState.RUNNING
        )
        assert len(monitor) == 1

    def test_contains(self, monitor):
        """Test __contains__ method."""
        assert "bot_001" not in monitor

        monitor._heartbeats["bot_001"] = HeartbeatData(
            bot_id="bot_001", state=BotState.RUNNING
        )
        assert "bot_001" in monitor


class TestHeartbeatMonitorLifecycle:
    """Tests for monitor start/stop."""

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test starting and stopping monitor."""
        assert not monitor.is_running

        await monitor.start()
        assert monitor.is_running

        await monitor.stop()
        assert not monitor.is_running

    @pytest.mark.asyncio
    async def test_start_twice_warns(self, monitor):
        """Test starting twice shows warning."""
        await monitor.start()
        await monitor.start()  # Should warn but not fail

        assert monitor.is_running

        await monitor.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running(self, monitor):
        """Test stopping when not running."""
        await monitor.stop()  # Should not fail
        assert not monitor.is_running


class TestHeartbeatRecovery:
    """Tests for bot recovery detection."""

    @pytest.mark.asyncio
    async def test_receive_after_timeout_sends_recovery(
        self, monitor, registry, sample_config, mock_notifier
    ):
        """Test receiving heartbeat after timeout sends recovery notification."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        # Set bot as timed out
        monitor._missed_counts["bot_001"] = 3  # >= max_missed (2)

        # Receive new heartbeat
        heartbeat = HeartbeatData(bot_id="bot_001", state=BotState.RUNNING)
        await monitor.receive(heartbeat)

        # Allow async task to run
        await asyncio.sleep(0.1)

        mock_notifier.notify_bot_recovered.assert_called_once_with("bot_001")
