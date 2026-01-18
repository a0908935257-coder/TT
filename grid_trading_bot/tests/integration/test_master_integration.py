"""
Integration tests for Master Control Console.

Tests the complete flow of Master with all modules integrated.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.master import (
    BotRegistry,
    BotState,
    BotType,
    Master,
    MasterConfig,
)


@pytest.fixture(autouse=True)
def reset_master():
    """Reset singleton before each test."""
    Master.reset_instance()
    yield
    Master.reset_instance()


@pytest.fixture
def mock_exchange():
    """Create mock exchange client."""
    exchange = MagicMock()
    exchange.get_ticker = AsyncMock(return_value={"price": "50000"})
    return exchange


@pytest.fixture
def mock_data_manager():
    """Create mock data manager."""
    return MagicMock()


@pytest.fixture
def mock_db_manager():
    """Create mock database manager."""
    db = MagicMock()
    db.get_all_bots = AsyncMock(return_value=[])
    db.upsert_bot = AsyncMock(return_value=None)
    db.delete_bot = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_notifier():
    """Create mock notifier."""
    notifier = MagicMock()
    notifier.send = AsyncMock(return_value=True)
    return notifier


@pytest.fixture
def config1():
    """First bot configuration."""
    return {
        "symbol": "BTCUSDT",
        "market_type": "spot",
        "total_investment": "10000",
    }


@pytest.fixture
def config2():
    """Second bot configuration."""
    return {
        "symbol": "ETHUSDT",
        "market_type": "spot",
        "total_investment": "5000",
    }


@pytest.fixture
def master_config():
    """Create master configuration for integration tests."""
    return MasterConfig(
        auto_restart=False,
        max_bots=10,
        snapshot_interval=0,  # Disable for tests
        restore_on_start=False,
    )


@pytest.fixture
def master(mock_exchange, mock_data_manager, mock_db_manager, mock_notifier, master_config):
    """Create master instance for testing."""
    return Master(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        db_manager=mock_db_manager,
        notifier=mock_notifier,
        config=master_config,
    )


def create_mock_bot(bot_id: str) -> MagicMock:
    """Create a mock bot with all required async methods."""
    from decimal import Decimal

    mock_bot = MagicMock()
    mock_bot.bot_id = bot_id
    mock_bot.start = AsyncMock(return_value=True)
    mock_bot.stop = AsyncMock(return_value=True)
    mock_bot.pause = AsyncMock(return_value=True)
    mock_bot.resume = AsyncMock(return_value=True)
    mock_bot.get_statistics = MagicMock(return_value={
        "trade_count": 10,
        "total_profit": "100.50",
    })

    # Mock config for aggregator
    mock_config = MagicMock()
    mock_config.total_investment = Decimal("10000")
    mock_bot.config = mock_config

    # Mock order_manager for aggregator
    mock_order_manager = MagicMock()
    mock_order_manager.active_order_count = 5
    mock_bot.order_manager = mock_order_manager

    # Mock get_today_stats for aggregator
    mock_bot.get_today_stats = MagicMock(return_value={
        "profit": "50.25",
        "trades": 3,
    })

    return mock_bot


class TestMasterFullFlow:
    """Integration tests for Master full workflow."""

    @pytest.mark.asyncio
    async def test_master_full_flow(self, master, config1, config2):
        """
        Test complete Master workflow:
        1. Start Master
        2. Create 2 bots
        3. Start all bots
        4. Check dashboard
        5. Pause one bot
        6. Check dashboard again
        7. Stop all bots
        8. Stop Master
        """
        # Create mock bots
        mock_bot1 = create_mock_bot("bot_001")
        mock_bot2 = create_mock_bot("bot_002")

        # Setup factory to return mocks
        master.factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        # 1. Start Master
        await master.start()
        assert master.is_running is True

        # 2. Create 2 bots
        result1 = await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        result2 = await master.create_bot(BotType.GRID, config2, bot_id="bot_002")
        assert result1.success is True
        assert result2.success is True

        # 3. Start all bots
        with patch("asyncio.sleep", return_value=None):
            results = await master.start_all()
        assert len(results) == 2
        assert all(r.success for r in results)

        # 4. Check dashboard - both running
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.total_bots == 2
        assert dashboard.summary.running_bots == 2

        # 5. Pause one bot
        pause_result = await master.pause_bot("bot_001")
        assert pause_result.success is True

        # 6. Check dashboard again
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 1
        assert dashboard.summary.paused_bots == 1

        # 7. Stop all bots
        stop_results = await master.stop_all()
        # Only bot_002 was running (bot_001 was paused)
        assert len(stop_results) == 2  # Both paused and running get stopped

        # Verify all stopped
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 0
        assert dashboard.summary.stopped_bots == 2

        # 8. Stop Master
        await master.stop()
        assert master.is_running is False

    @pytest.mark.asyncio
    async def test_create_start_stop_delete_flow(self, master, config1):
        """Test bot lifecycle: create -> start -> stop -> delete."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        # Start master
        await master.start()

        # Create bot
        result = await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        assert result.success is True
        assert master.get_bot("bot_001") is not None
        assert master.get_bot("bot_001").state == BotState.REGISTERED

        # Start bot
        result = await master.start_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001").state == BotState.RUNNING

        # Stop bot
        result = await master.stop_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001").state == BotState.STOPPED

        # Delete bot
        result = await master.delete_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001") is None

        # Stop master
        await master.stop()

    @pytest.mark.asyncio
    async def test_pause_resume_flow(self, master, config1):
        """Test pause and resume flow."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()

        # Create and start
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.start_bot("bot_001")
        assert master.get_bot("bot_001").state == BotState.RUNNING

        # Pause
        result = await master.pause_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001").state == BotState.PAUSED

        # Resume
        result = await master.resume_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001").state == BotState.RUNNING

        await master.stop()

    @pytest.mark.asyncio
    async def test_restart_flow(self, master, config1):
        """Test restart flow."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()

        # Create and start
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.start_bot("bot_001")

        # Restart
        with patch("asyncio.sleep", return_value=None):
            result = await master.restart_bot("bot_001")
        assert result.success is True
        assert master.get_bot("bot_001").state == BotState.RUNNING

        await master.stop()

    @pytest.mark.asyncio
    async def test_dashboard_reflects_state_changes(self, master, config1, config2):
        """Test that dashboard accurately reflects state changes."""
        mock_bot1 = create_mock_bot("bot_001")
        mock_bot2 = create_mock_bot("bot_002")
        master.factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        await master.start()

        # Initially empty
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.total_bots == 0

        # After creating bots
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.create_bot(BotType.GRID, config2, bot_id="bot_002")
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.total_bots == 2

        # After starting one
        await master.start_bot("bot_001")
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 1

        # After starting second
        await master.start_bot("bot_002")
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 2

        # After pausing one
        await master.pause_bot("bot_001")
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 1
        assert dashboard.summary.paused_bots == 1

        # After stopping one
        await master.stop_bot("bot_002")
        dashboard = master.get_dashboard_data()
        assert dashboard.summary.running_bots == 0
        assert dashboard.summary.paused_bots == 1
        assert dashboard.summary.stopped_bots == 1

        await master.stop()

    @pytest.mark.asyncio
    async def test_health_check_integration(self, master, config1):
        """Test health checking integration."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()

        # Create and start bot
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.start_bot("bot_001")

        # Run health check
        result = await master.health_check("bot_001")
        assert result is not None
        assert result.bot_id == "bot_001"

        # Run health check all
        results = await master.health_check_all()
        assert len(results) >= 1

        await master.stop()

    @pytest.mark.asyncio
    async def test_alert_management_integration(self, master, config1):
        """Test alert management integration."""
        from src.master import Alert, AlertLevel

        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")

        # Add alert
        alert = Alert(
            bot_id="bot_001",
            level=AlertLevel.WARNING,
            message="Test warning",
        )
        master.dashboard.add_alert(alert)

        # Get alerts
        alerts = master.get_alerts()
        assert len(alerts) == 1
        assert alerts[0].bot_id == "bot_001"

        # Get alerts by bot_id
        alerts = master.get_alerts(bot_id="bot_001")
        assert len(alerts) == 1

        # Acknowledge alert
        result = master.acknowledge_alert(alert.alert_id, "admin")
        assert result is True

        # Unacknowledged alerts should be empty
        alerts = master.get_alerts(unacknowledged_only=True)
        assert len(alerts) == 0

        await master.stop()

    @pytest.mark.asyncio
    async def test_max_bots_limit(self, master, config1):
        """Test that max bots limit is enforced."""
        # Set max_bots to 2
        master._config.max_bots = 2

        mock_bot1 = create_mock_bot("bot_001")
        mock_bot2 = create_mock_bot("bot_002")
        mock_bot3 = create_mock_bot("bot_003")
        master.factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2, mock_bot3])

        await master.start()

        # Create 2 bots (should succeed)
        result1 = await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        result2 = await master.create_bot(BotType.GRID, config1, bot_id="bot_002")
        assert result1.success is True
        assert result2.success is True

        # Try to create 3rd bot (should fail)
        result3 = await master.create_bot(BotType.GRID, config1, bot_id="bot_003")
        assert result3.success is False
        assert "Maximum bot limit" in result3.message

        await master.stop()

    @pytest.mark.asyncio
    async def test_query_methods(self, master, config1, config2):
        """Test various query methods."""
        mock_bot1 = create_mock_bot("bot_001")
        mock_bot2 = create_mock_bot("bot_002")
        master.factory.create = MagicMock(side_effect=[mock_bot1, mock_bot2])

        await master.start()

        # Create bots with different symbols
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.create_bot(BotType.GRID, config2, bot_id="bot_002")

        # Start one
        await master.start_bot("bot_001")

        # Query by state
        running = master.get_bots_by_state(BotState.RUNNING)
        registered = master.get_bots_by_state(BotState.REGISTERED)
        assert len(running) == 1
        assert len(registered) == 1

        # Query by type
        grid_bots = master.get_bots_by_type(BotType.GRID)
        dca_bots = master.get_bots_by_type(BotType.DCA)
        assert len(grid_bots) == 2
        assert len(dca_bots) == 0

        # Get all bots
        all_bots = master.get_all_bots()
        assert len(all_bots) == 2

        # Get specific bot
        bot = master.get_bot("bot_001")
        assert bot is not None
        assert bot.symbol == "BTCUSDT"

        # Get summary
        summary = master.get_summary()
        assert summary["total_bots"] == 2
        assert summary["is_running"] is True

        await master.stop()


class TestMasterErrorHandling:
    """Test error handling in Master integration."""

    @pytest.mark.asyncio
    async def test_start_nonexistent_bot(self, master):
        """Test starting a nonexistent bot."""
        await master.start()

        result = await master.start_bot("nonexistent")
        assert result.success is False
        assert "not found" in result.message.lower()

        await master.stop()

    @pytest.mark.asyncio
    async def test_stop_nonexistent_bot(self, master):
        """Test stopping a nonexistent bot."""
        await master.start()

        result = await master.stop_bot("nonexistent")
        assert result.success is False
        assert "not found" in result.message.lower()

        await master.stop()

    @pytest.mark.asyncio
    async def test_start_already_running_bot(self, master, config1):
        """Test starting an already running bot."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.start_bot("bot_001")

        # Try to start again
        result = await master.start_bot("bot_001")
        assert result.success is False
        assert "Cannot start" in result.message

        await master.stop()

    @pytest.mark.asyncio
    async def test_pause_non_running_bot(self, master, config1):
        """Test pausing a non-running bot."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")

        # Try to pause without starting
        result = await master.pause_bot("bot_001")
        assert result.success is False
        assert "Cannot pause" in result.message

        await master.stop()

    @pytest.mark.asyncio
    async def test_resume_non_paused_bot(self, master, config1):
        """Test resuming a non-paused bot."""
        mock_bot = create_mock_bot("bot_001")
        master.factory.create = MagicMock(return_value=mock_bot)

        await master.start()
        await master.create_bot(BotType.GRID, config1, bot_id="bot_001")
        await master.start_bot("bot_001")

        # Try to resume without pausing
        result = await master.resume_bot("bot_001")
        assert result.success is False
        assert "Cannot resume" in result.message

        await master.stop()
