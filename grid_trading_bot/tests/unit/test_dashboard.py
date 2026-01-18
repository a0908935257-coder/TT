"""
Unit tests for Dashboard and MetricsAggregator.

Tests metrics collection, dashboard data, and alert management.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.master import (
    Alert,
    AlertLevel,
    BotMetrics,
    BotRegistry,
    BotState,
    BotType,
    Dashboard,
    DashboardData,
    DashboardSummary,
    MetricsAggregator,
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
def aggregator(registry):
    """Create metrics aggregator for testing."""
    return MetricsAggregator(registry)


@pytest.fixture
def dashboard(registry, aggregator):
    """Create dashboard for testing."""
    return Dashboard(registry, aggregator)


# =============================================================================
# Test: BotMetrics Model
# =============================================================================


class TestBotMetrics:
    """Tests for BotMetrics model."""

    def test_create_bot_metrics(self):
        """Test creating BotMetrics."""
        metrics = BotMetrics(
            bot_id="bot_001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            state=BotState.RUNNING,
            total_investment=Decimal("10000"),
            total_profit=Decimal("500"),
        )

        assert metrics.bot_id == "bot_001"
        assert metrics.bot_type == BotType.GRID
        assert metrics.symbol == "BTCUSDT"
        assert metrics.state == BotState.RUNNING
        assert metrics.total_investment == Decimal("10000")
        assert metrics.total_profit == Decimal("500")

    def test_to_dict(self):
        """Test BotMetrics to_dict."""
        metrics = BotMetrics(
            bot_id="bot_001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            state=BotState.RUNNING,
            total_investment=Decimal("10000"),
            total_profit=Decimal("500"),
            profit_rate=Decimal("5.0"),
        )

        result = metrics.to_dict()

        assert result["bot_id"] == "bot_001"
        assert result["bot_type"] == "grid"
        assert result["state"] == "running"
        assert result["total_investment"] == "10000"
        assert result["total_profit"] == "500"


# =============================================================================
# Test: MetricsAggregator
# =============================================================================


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    @pytest.mark.asyncio
    async def test_collect_bot_metrics(self, aggregator, registry, sample_config):
        """Test collecting metrics for a single bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        metrics = aggregator.collect("bot_001")

        assert metrics is not None
        assert metrics.bot_id == "bot_001"
        assert metrics.bot_type == BotType.GRID
        assert metrics.symbol == "BTCUSDT"
        assert metrics.state == BotState.RUNNING

    @pytest.mark.asyncio
    async def test_collect_nonexistent_bot(self, aggregator):
        """Test collecting metrics for nonexistent bot returns None."""
        metrics = aggregator.collect("nonexistent")

        assert metrics is None

    @pytest.mark.asyncio
    async def test_collect_all(self, aggregator, registry, sample_config):
        """Test collecting all bot metrics."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        all_metrics = aggregator.collect_all()

        assert len(all_metrics) == 2
        bot_ids = [m.bot_id for m in all_metrics]
        assert "bot_001" in bot_ids
        assert "bot_002" in bot_ids

    @pytest.mark.asyncio
    async def test_get_cached(self, aggregator, registry, sample_config):
        """Test getting cached metrics."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        # First collect to populate cache
        aggregator.collect("bot_001")

        # Get from cache
        cached = aggregator.get_cached("bot_001")

        assert cached is not None
        assert cached.bot_id == "bot_001"

    @pytest.mark.asyncio
    async def test_refresh(self, aggregator, registry, sample_config):
        """Test refreshing all cached metrics."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        aggregator.refresh()

        assert aggregator.cache_size == 2
        assert aggregator.last_update is not None

    @pytest.mark.asyncio
    async def test_clear_cache(self, aggregator, registry, sample_config):
        """Test clearing cache."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        aggregator.collect("bot_001")

        assert aggregator.cache_size == 1

        aggregator.clear_cache()

        assert aggregator.cache_size == 0
        assert aggregator.last_update is None

    @pytest.mark.asyncio
    async def test_get_by_state(self, aggregator, registry, sample_config):
        """Test filtering metrics by state."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        aggregator.refresh()

        running = aggregator.get_by_state(BotState.RUNNING)
        registered = aggregator.get_by_state(BotState.REGISTERED)

        assert len(running) == 1
        assert running[0].bot_id == "bot_001"
        assert len(registered) == 1

    @pytest.mark.asyncio
    async def test_get_totals(self, aggregator, registry, sample_config):
        """Test getting aggregated totals."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)

        aggregator.refresh()

        totals = aggregator.get_totals()

        assert "total_investment" in totals
        assert "total_profit" in totals
        assert "total_trades" in totals


# =============================================================================
# Test: Alert
# =============================================================================


class TestAlert:
    """Tests for Alert model."""

    def test_create_alert(self):
        """Test creating an alert."""
        alert = Alert(
            bot_id="bot_001",
            level=AlertLevel.WARNING,
            message="Test alert",
        )

        assert alert.bot_id == "bot_001"
        assert alert.level == AlertLevel.WARNING
        assert alert.message == "Test alert"
        assert alert.acknowledged is False
        assert alert.alert_id is not None

    def test_alert_to_dict(self):
        """Test Alert to_dict."""
        alert = Alert(
            bot_id="bot_001",
            level=AlertLevel.ERROR,
            message="Error occurred",
        )

        result = alert.to_dict()

        assert result["bot_id"] == "bot_001"
        assert result["level"] == "error"
        assert result["message"] == "Error occurred"
        assert result["acknowledged"] is False

    def test_acknowledge_alert(self):
        """Test acknowledging an alert."""
        alert = Alert(
            bot_id="bot_001",
            level=AlertLevel.WARNING,
            message="Test",
        )

        alert.acknowledge("admin")

        assert alert.acknowledged is True
        assert alert.acknowledged_at is not None
        assert alert.acknowledged_by == "admin"


class TestAlertLevel:
    """Tests for AlertLevel enum."""

    def test_alert_levels(self):
        """Test alert level values."""
        assert AlertLevel.INFO.value == "info"
        assert AlertLevel.WARNING.value == "warning"
        assert AlertLevel.ERROR.value == "error"
        assert AlertLevel.CRITICAL.value == "critical"


# =============================================================================
# Test: DashboardSummary
# =============================================================================


class TestDashboardSummary:
    """Tests for DashboardSummary model."""

    def test_create_summary(self):
        """Test creating dashboard summary."""
        summary = DashboardSummary(
            total_bots=5,
            running_bots=3,
            paused_bots=1,
            stopped_bots=1,
            error_bots=0,
            total_investment=Decimal("50000"),
            total_profit=Decimal("2500"),
            total_profit_rate=Decimal("5.0"),
        )

        assert summary.total_bots == 5
        assert summary.running_bots == 3
        assert summary.total_investment == Decimal("50000")
        assert summary.total_profit_rate == Decimal("5.0")

    def test_summary_to_dict(self):
        """Test DashboardSummary to_dict."""
        summary = DashboardSummary(
            total_bots=3,
            running_bots=2,
            total_investment=Decimal("30000"),
            total_profit=Decimal("1500"),
        )

        result = summary.to_dict()

        assert result["total_bots"] == 3
        assert result["running_bots"] == 2
        assert result["total_investment"] == "30000"


# =============================================================================
# Test: Dashboard
# =============================================================================


class TestDashboard:
    """Tests for Dashboard class."""

    @pytest.mark.asyncio
    async def test_get_data(self, dashboard, registry, sample_config):
        """Test getting complete dashboard data."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        data = dashboard.get_data()

        assert isinstance(data, DashboardData)
        assert data.summary.total_bots == 2
        assert len(data.bots) == 2
        assert data.updated_at is not None

    @pytest.mark.asyncio
    async def test_get_summary(self, dashboard, registry, sample_config):
        """Test getting dashboard summary."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        summary = dashboard.get_summary()

        assert isinstance(summary, DashboardSummary)
        assert summary.total_bots == 1
        assert summary.running_bots == 1

    @pytest.mark.asyncio
    async def test_get_bot_detail(self, dashboard, registry, sample_config):
        """Test getting bot detail."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        detail = await dashboard.get_bot_detail("bot_001")

        assert detail is not None
        assert detail.bot_info["bot_id"] == "bot_001"
        assert detail.metrics is not None

    @pytest.mark.asyncio
    async def test_get_bot_detail_nonexistent(self, dashboard):
        """Test getting detail for nonexistent bot."""
        detail = await dashboard.get_bot_detail("nonexistent")

        assert detail is None


class TestDashboardAlerts:
    """Tests for Dashboard alert management."""

    def test_add_alert(self, dashboard):
        """Test adding an alert."""
        alert = Alert(
            bot_id="bot_001",
            level=AlertLevel.WARNING,
            message="Test alert",
        )

        dashboard.add_alert(alert)

        assert dashboard.alert_count == 1

    def test_create_alert(self, dashboard):
        """Test creating and adding an alert."""
        alert = dashboard.create_alert(
            bot_id="bot_001",
            level=AlertLevel.ERROR,
            message="Error message",
        )

        assert alert.bot_id == "bot_001"
        assert dashboard.alert_count == 1

    def test_get_alerts(self, dashboard):
        """Test getting alerts."""
        dashboard.create_alert("bot_001", AlertLevel.WARNING, "Warning 1")
        dashboard.create_alert("bot_001", AlertLevel.ERROR, "Error 1")
        dashboard.create_alert("bot_002", AlertLevel.WARNING, "Warning 2")

        all_alerts = dashboard.get_alerts()
        warnings = dashboard.get_alerts(level=AlertLevel.WARNING)
        bot_001_alerts = dashboard.get_alerts(bot_id="bot_001")

        assert len(all_alerts) == 3
        assert len(warnings) == 2
        assert len(bot_001_alerts) == 2

    def test_acknowledge_alert(self, dashboard):
        """Test acknowledging an alert."""
        alert = dashboard.create_alert("bot_001", AlertLevel.ERROR, "Error")

        result = dashboard.acknowledge_alert(alert.alert_id, "admin")

        assert result is True
        assert dashboard.unacknowledged_alert_count == 0

    def test_acknowledge_nonexistent_alert(self, dashboard):
        """Test acknowledging nonexistent alert."""
        result = dashboard.acknowledge_alert("nonexistent")

        assert result is False

    def test_acknowledge_all(self, dashboard):
        """Test acknowledging all alerts."""
        dashboard.create_alert("bot_001", AlertLevel.WARNING, "Warning 1")
        dashboard.create_alert("bot_002", AlertLevel.ERROR, "Error 1")

        assert dashboard.unacknowledged_alert_count == 2

        count = dashboard.acknowledge_all("admin")

        assert count == 2
        assert dashboard.unacknowledged_alert_count == 0

    def test_clear_alerts(self, dashboard):
        """Test clearing alerts."""
        alert1 = dashboard.create_alert("bot_001", AlertLevel.WARNING, "Warning")
        alert2 = dashboard.create_alert("bot_002", AlertLevel.ERROR, "Error")

        dashboard.acknowledge_alert(alert1.alert_id)

        # Clear only acknowledged
        cleared = dashboard.clear_alerts(acknowledged_only=True)

        assert cleared == 1
        assert dashboard.alert_count == 1

        # Clear all
        cleared = dashboard.clear_alerts(acknowledged_only=False)

        assert cleared == 1
        assert dashboard.alert_count == 0

    def test_unacknowledged_alerts_in_data(self, dashboard, registry, sample_config):
        """Test that get_data only returns unacknowledged alerts."""
        dashboard.create_alert("bot_001", AlertLevel.WARNING, "Warning")
        ack_alert = dashboard.create_alert("bot_002", AlertLevel.ERROR, "Error")
        dashboard.acknowledge_alert(ack_alert.alert_id)

        data = dashboard.get_data()

        assert len(data.alerts) == 1
        assert data.alerts[0].acknowledged is False


class TestDashboardRankings:
    """Tests for Dashboard rankings."""

    @pytest.mark.asyncio
    async def test_get_rankings(self, dashboard, registry, sample_config):
        """Test getting bot rankings."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)

        dashboard._aggregator.refresh()

        rankings = dashboard.get_rankings()

        assert "by_profit" in rankings
        assert "by_profit_rate" in rankings
        assert "by_trades" in rankings

    def test_get_rankings_empty(self, dashboard):
        """Test rankings with no bots."""
        rankings = dashboard.get_rankings()

        assert rankings["by_profit"] == []
        assert rankings["by_profit_rate"] == []
        assert rankings["by_trades"] == []


class TestDashboardSummaryCalculation:
    """Tests for dashboard summary calculation."""

    @pytest.mark.asyncio
    async def test_calculate_summary_with_running_bots(
        self, dashboard, registry, sample_config
    ):
        """Test summary calculation with running bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)
        await registry.update_state("bot_002", BotState.INITIALIZING)
        await registry.update_state("bot_002", BotState.RUNNING)
        await registry.update_state("bot_002", BotState.PAUSED)

        summary = dashboard.get_summary()

        assert summary.total_bots == 2
        assert summary.running_bots == 1
        assert summary.paused_bots == 1

    @pytest.mark.asyncio
    async def test_calculate_summary_with_errors(
        self, dashboard, registry, sample_config
    ):
        """Test summary calculation with error bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.ERROR, "Test error")

        summary = dashboard.get_summary()

        assert summary.error_bots == 1


class TestDashboardData:
    """Tests for DashboardData model."""

    def test_dashboard_data_to_dict(self):
        """Test DashboardData to_dict."""
        summary = DashboardSummary(total_bots=1, running_bots=1)
        metrics = BotMetrics(
            bot_id="bot_001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            state=BotState.RUNNING,
        )
        alert = Alert(bot_id="bot_001", level=AlertLevel.INFO, message="Test")

        data = DashboardData(
            summary=summary,
            bots=[metrics],
            alerts=[alert],
        )

        result = data.to_dict()

        assert "summary" in result
        assert "bots" in result
        assert "alerts" in result
        assert len(result["bots"]) == 1
        assert len(result["alerts"]) == 1
