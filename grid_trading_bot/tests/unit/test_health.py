"""
Unit tests for Health Checker.

Tests health check functionality including individual checks and overall status.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master import (
    BotRegistry,
    BotState,
    BotType,
    CheckItem,
    HealthCheckResult,
    HealthChecker,
    HealthStatus,
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
def checker(registry):
    """Create health checker for testing."""
    return HealthChecker(registry)


@pytest.fixture
def sample_config():
    """Create sample bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "total_investment": "10000",
    }


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        """Test health status values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestCheckItem:
    """Tests for CheckItem model."""

    def test_create_check_item(self):
        """Test creating CheckItem."""
        item = CheckItem(
            name="heartbeat",
            status=HealthStatus.HEALTHY,
            message="正常",
            value=5.5,
        )

        assert item.name == "heartbeat"
        assert item.status == HealthStatus.HEALTHY
        assert item.message == "正常"
        assert item.value == 5.5

    def test_to_dict(self):
        """Test CheckItem to_dict."""
        item = CheckItem(
            name="memory",
            status=HealthStatus.DEGRADED,
            message="記憶體使用量高",
            value=75.5,
        )

        result = item.to_dict()

        assert result["name"] == "memory"
        assert result["status"] == "degraded"
        assert result["message"] == "記憶體使用量高"
        assert result["value"] == 75.5


class TestHealthCheckResult:
    """Tests for HealthCheckResult model."""

    def test_create_result(self):
        """Test creating HealthCheckResult."""
        checks = {
            "heartbeat": CheckItem("heartbeat", HealthStatus.HEALTHY, "OK"),
            "memory": CheckItem("memory", HealthStatus.DEGRADED, "High"),
        }

        result = HealthCheckResult(
            bot_id="bot_001",
            status=HealthStatus.DEGRADED,
            checks=checks,
        )

        assert result.bot_id == "bot_001"
        assert result.status == HealthStatus.DEGRADED
        assert len(result.checks) == 2
        assert result.checked_at is not None

    def test_is_healthy(self):
        """Test is_healthy property."""
        healthy = HealthCheckResult("bot_001", HealthStatus.HEALTHY)
        degraded = HealthCheckResult("bot_001", HealthStatus.DEGRADED)
        unhealthy = HealthCheckResult("bot_001", HealthStatus.UNHEALTHY)

        assert healthy.is_healthy is True
        assert degraded.is_healthy is False
        assert unhealthy.is_healthy is False

    def test_has_issues(self):
        """Test has_issues property."""
        healthy = HealthCheckResult("bot_001", HealthStatus.HEALTHY)
        degraded = HealthCheckResult("bot_001", HealthStatus.DEGRADED)
        unhealthy = HealthCheckResult("bot_001", HealthStatus.UNHEALTHY)

        assert healthy.has_issues is False
        assert degraded.has_issues is True
        assert unhealthy.has_issues is True

    def test_to_dict(self):
        """Test HealthCheckResult to_dict."""
        result = HealthCheckResult(
            bot_id="bot_001",
            status=HealthStatus.HEALTHY,
            checks={
                "test": CheckItem("test", HealthStatus.HEALTHY, "OK"),
            },
        )

        data = result.to_dict()

        assert data["bot_id"] == "bot_001"
        assert data["status"] == "healthy"
        assert "test" in data["checks"]
        assert "checked_at" in data


class TestHealthChecker:
    """Tests for HealthChecker."""

    @pytest.mark.asyncio
    async def test_check_unregistered_bot(self, checker):
        """Test checking unregistered bot returns unknown."""
        result = await checker.check("nonexistent")

        assert result.status == HealthStatus.UNKNOWN
        assert "registration" in result.checks
        assert result.checks["registration"].status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_check_registered_bot(self, checker, registry, sample_config):
        """Test checking registered bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Update heartbeat
        bot_info = registry.get("bot_001")
        bot_info.last_heartbeat = datetime.now(timezone.utc)

        result = await checker.check("bot_001")

        assert result.bot_id == "bot_001"
        assert "heartbeat" in result.checks
        assert "state" in result.checks
        assert "memory" in result.checks

    @pytest.mark.asyncio
    async def test_check_all(self, checker, registry, sample_config):
        """Test checking all bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        results = await checker.check_all()

        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_state_check_running(self, checker, registry, sample_config):
        """Test state check for running bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        bot_info = registry.get("bot_001")
        check = checker._check_state(bot_info)

        assert check.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_state_check_paused(self, checker, registry, sample_config):
        """Test state check for paused bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)
        await registry.update_state("bot_001", BotState.PAUSED)

        bot_info = registry.get("bot_001")
        check = checker._check_state(bot_info)

        assert check.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_state_check_error(self, checker, registry, sample_config):
        """Test state check for error bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.ERROR, "Test error")

        bot_info = registry.get("bot_001")
        check = checker._check_state(bot_info)

        assert check.status == HealthStatus.UNHEALTHY


class TestHeartbeatCheck:
    """Tests for heartbeat health check."""

    @pytest.mark.asyncio
    async def test_heartbeat_healthy(self, checker, registry, sample_config):
        """Test heartbeat check when healthy."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        bot_info = registry.get("bot_001")
        bot_info.last_heartbeat = datetime.now(timezone.utc)

        check = await checker._check_heartbeat("bot_001", bot_info)

        assert check.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_degraded(self, checker, registry, sample_config):
        """Test heartbeat check when degraded."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        bot_info = registry.get("bot_001")
        bot_info.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=45)

        check = await checker._check_heartbeat("bot_001", bot_info)

        assert check.status == HealthStatus.DEGRADED

    @pytest.mark.asyncio
    async def test_heartbeat_unhealthy(self, checker, registry, sample_config):
        """Test heartbeat check when unhealthy."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        bot_info = registry.get("bot_001")
        bot_info.last_heartbeat = datetime.now(timezone.utc) - timedelta(seconds=90)

        check = await checker._check_heartbeat("bot_001", bot_info)

        assert check.status == HealthStatus.UNHEALTHY

    @pytest.mark.asyncio
    async def test_heartbeat_no_data(self, checker, registry, sample_config):
        """Test heartbeat check with no heartbeat data."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        bot_info = registry.get("bot_001")
        # No last_heartbeat set

        check = await checker._check_heartbeat("bot_001", bot_info)

        assert check.status == HealthStatus.UNKNOWN


class TestMemoryCheck:
    """Tests for memory health check."""

    def test_memory_healthy(self, checker):
        """Test memory check when healthy."""
        with patch("src.master.health.HAS_PSUTIL", True), \
             patch("src.master.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(percent=50.0)
            check = checker._check_memory()

        assert check.status == HealthStatus.HEALTHY
        assert check.value == 50.0

    def test_memory_degraded(self, checker):
        """Test memory check when degraded."""
        with patch("src.master.health.HAS_PSUTIL", True), \
             patch("src.master.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(percent=75.0)
            check = checker._check_memory()

        assert check.status == HealthStatus.DEGRADED

    def test_memory_unhealthy(self, checker):
        """Test memory check when unhealthy."""
        with patch("src.master.health.HAS_PSUTIL", True), \
             patch("src.master.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.return_value = MagicMock(percent=90.0)
            check = checker._check_memory()

        assert check.status == HealthStatus.UNHEALTHY

    def test_memory_check_error(self, checker):
        """Test memory check when psutil fails."""
        with patch("src.master.health.HAS_PSUTIL", True), \
             patch("src.master.health.psutil") as mock_psutil:
            mock_psutil.virtual_memory.side_effect = Exception("psutil error")
            check = checker._check_memory()

        assert check.status == HealthStatus.UNKNOWN

    def test_memory_check_no_psutil(self, checker):
        """Test memory check when psutil is not available."""
        with patch("src.master.health.HAS_PSUTIL", False):
            check = checker._check_memory()

        assert check.status == HealthStatus.UNKNOWN
        assert "psutil not available" in check.message


class TestOverallStatusCalculation:
    """Tests for overall status calculation."""

    def test_all_healthy(self, checker):
        """Test overall status when all checks are healthy."""
        checks = {
            "check1": CheckItem("check1", HealthStatus.HEALTHY, "OK"),
            "check2": CheckItem("check2", HealthStatus.HEALTHY, "OK"),
            "check3": CheckItem("check3", HealthStatus.HEALTHY, "OK"),
        }

        status = checker._calculate_overall_status(checks)

        assert status == HealthStatus.HEALTHY

    def test_one_degraded(self, checker):
        """Test overall status when one check is degraded."""
        checks = {
            "check1": CheckItem("check1", HealthStatus.HEALTHY, "OK"),
            "check2": CheckItem("check2", HealthStatus.DEGRADED, "Warning"),
            "check3": CheckItem("check3", HealthStatus.HEALTHY, "OK"),
        }

        status = checker._calculate_overall_status(checks)

        assert status == HealthStatus.DEGRADED

    def test_one_unhealthy(self, checker):
        """Test overall status when one check is unhealthy."""
        checks = {
            "check1": CheckItem("check1", HealthStatus.HEALTHY, "OK"),
            "check2": CheckItem("check2", HealthStatus.UNHEALTHY, "Error"),
            "check3": CheckItem("check3", HealthStatus.DEGRADED, "Warning"),
        }

        status = checker._calculate_overall_status(checks)

        assert status == HealthStatus.UNHEALTHY

    def test_empty_checks(self, checker):
        """Test overall status with no checks."""
        status = checker._calculate_overall_status({})

        assert status == HealthStatus.UNKNOWN

    def test_all_unknown(self, checker):
        """Test overall status when all checks are unknown."""
        checks = {
            "check1": CheckItem("check1", HealthStatus.UNKNOWN, "Unknown"),
        }

        status = checker._calculate_overall_status(checks)

        assert status == HealthStatus.UNKNOWN


class TestExchangeCheck:
    """Tests for exchange health check."""

    @pytest.mark.asyncio
    async def test_exchange_check_no_instance(self, checker, registry, sample_config):
        """Test exchange check with no bot instance."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        result = await checker.check("bot_001")

        # No exchange check should be present without instance
        assert "exchange" not in result.checks

    @pytest.mark.asyncio
    async def test_exchange_check_healthy(self, checker, registry, sample_config):
        """Test exchange check when healthy."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        # Mock bot instance with exchange
        mock_exchange = MagicMock()
        mock_exchange.ping = AsyncMock()

        mock_instance = MagicMock()
        mock_instance._exchange = mock_exchange

        registry.bind_instance("bot_001", mock_instance)

        check = await checker._check_exchange(mock_instance)

        assert check.status == HealthStatus.HEALTHY

    @pytest.mark.asyncio
    async def test_exchange_check_error(self, checker, registry, sample_config):
        """Test exchange check when error occurs."""
        mock_exchange = MagicMock()
        mock_exchange.ping = AsyncMock(side_effect=Exception("Connection failed"))

        mock_instance = MagicMock()
        mock_instance._exchange = mock_exchange

        check = await checker._check_exchange(mock_instance)

        assert check.status == HealthStatus.UNHEALTHY


class TestWithHeartbeatMonitor:
    """Tests for HealthChecker with HeartbeatMonitor integration."""

    @pytest.mark.asyncio
    async def test_check_uses_heartbeat_monitor(self, registry, sample_config):
        """Test that checker uses heartbeat monitor when available."""
        monitor = HeartbeatMonitor(registry)

        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        # Add heartbeat
        monitor.receive(HeartbeatData(bot_id="bot_001", state=BotState.RUNNING))

        checker = HealthChecker(registry, heartbeat_monitor=monitor)
        result = await checker.check("bot_001")

        assert "heartbeat" in result.checks
        assert result.checks["heartbeat"].status == HealthStatus.HEALTHY


class TestGetSummary:
    """Tests for get_summary method."""

    @pytest.mark.asyncio
    async def test_get_summary(self, checker, registry, sample_config):
        """Test getting health summary."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        summary = checker.get_summary()

        assert summary["total"] == 2
        assert "by_state" in summary
        assert summary["by_state"]["running"] == 1
        assert summary["by_state"]["registered"] == 1
