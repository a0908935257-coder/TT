"""
Unit tests for Bot Registry.

Tests bot registration, state transitions, and lifecycle management.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.master import (
    BotAlreadyExistsError,
    BotInfo,
    BotNotFoundError,
    BotRegistry,
    BotState,
    BotType,
    InvalidStateTransitionError,
    MarketType,
    RegistryEvent,
    VALID_STATE_TRANSITIONS,
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
        "risk_level": "moderate",
    }


@pytest.fixture
def mock_db_manager():
    """Create a mock database manager."""
    db = MagicMock()
    db.get_all_bots = AsyncMock(return_value=[])
    db.upsert_bot = AsyncMock()
    db.delete_bot = AsyncMock(return_value=True)
    return db


@pytest.fixture
def mock_notifier():
    """Create a mock notifier."""
    notifier = MagicMock()
    notifier.notify_bot_registered = AsyncMock()
    notifier.notify_bot_state_changed = AsyncMock()
    return notifier


class TestBotRegistration:
    """Tests for bot registration."""

    @pytest.mark.asyncio
    async def test_register_bot_success(self, registry, sample_config):
        """Test successful bot registration."""
        info = await registry.register(
            bot_id="bot_001",
            bot_type=BotType.GRID,
            config=sample_config,
        )

        assert info.bot_id == "bot_001"
        assert info.bot_type == BotType.GRID
        assert info.state == BotState.REGISTERED
        assert info.symbol == "BTCUSDT"
        assert info.created_at is not None

    @pytest.mark.asyncio
    async def test_register_bot_duplicate_fails(self, registry, sample_config):
        """Test duplicate registration is rejected."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        with pytest.raises(BotAlreadyExistsError):
            await registry.register("bot_001", BotType.GRID, sample_config)

    @pytest.mark.asyncio
    async def test_register_bot_empty_config_fails(self, registry):
        """Test empty config is rejected."""
        with pytest.raises(ValueError, match="Config cannot be empty"):
            await registry.register("bot_001", BotType.GRID, {})

    @pytest.mark.asyncio
    async def test_register_bot_missing_symbol_fails(self, registry):
        """Test missing symbol is rejected."""
        config = {"total_investment": "10000"}

        with pytest.raises(ValueError, match="Symbol is required"):
            await registry.register("bot_001", BotType.GRID, config)

    @pytest.mark.asyncio
    async def test_register_with_explicit_symbol(self, registry):
        """Test registration with explicit symbol parameter."""
        config = {"total_investment": "10000"}

        info = await registry.register(
            bot_id="bot_001",
            bot_type=BotType.DCA,
            config=config,
            symbol="ETHUSDT",
            market_type=MarketType.FUTURES,
        )

        assert info.symbol == "ETHUSDT"
        assert info.market_type == MarketType.FUTURES


class TestBotUnregistration:
    """Tests for bot unregistration."""

    @pytest.mark.asyncio
    async def test_unregister_registered_bot(self, registry, sample_config):
        """Test unregistering a registered bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        result = await registry.unregister("bot_001")

        assert result is True
        assert registry.get("bot_001") is None

    @pytest.mark.asyncio
    async def test_unregister_stopped_bot(self, registry, sample_config):
        """Test unregistering a stopped bot."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)
        await registry.update_state("bot_001", BotState.STOPPING)
        await registry.update_state("bot_001", BotState.STOPPED)

        result = await registry.unregister("bot_001")

        assert result is True

    @pytest.mark.asyncio
    async def test_unregister_running_bot_fails(self, registry, sample_config):
        """Test unregistering a running bot is rejected."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        with pytest.raises(ValueError, match="Cannot unregister bot in state"):
            await registry.unregister("bot_001")

    @pytest.mark.asyncio
    async def test_unregister_nonexistent_bot_fails(self, registry):
        """Test unregistering non-existent bot fails."""
        with pytest.raises(BotNotFoundError):
            await registry.unregister("nonexistent")


class TestStateTransitions:
    """Tests for state transitions."""

    def test_valid_transitions_mapping(self):
        """Test valid state transitions are defined correctly."""
        assert BotState.INITIALIZING in VALID_STATE_TRANSITIONS[BotState.REGISTERED]
        assert BotState.RUNNING in VALID_STATE_TRANSITIONS[BotState.INITIALIZING]
        assert BotState.ERROR in VALID_STATE_TRANSITIONS[BotState.INITIALIZING]
        assert BotState.PAUSED in VALID_STATE_TRANSITIONS[BotState.RUNNING]
        assert BotState.STOPPING in VALID_STATE_TRANSITIONS[BotState.RUNNING]
        assert BotState.STOPPED in VALID_STATE_TRANSITIONS[BotState.STOPPING]

    @pytest.mark.asyncio
    async def test_valid_state_transition(self, registry, sample_config):
        """Test valid state transition is accepted."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        await registry.update_state("bot_001", BotState.INITIALIZING)

        info = registry.get("bot_001")
        assert info.state == BotState.INITIALIZING

    @pytest.mark.asyncio
    async def test_invalid_state_transition_fails(self, registry, sample_config):
        """Test invalid state transition is rejected."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        # REGISTERED -> STOPPED is invalid
        with pytest.raises(InvalidStateTransitionError):
            await registry.update_state("bot_001", BotState.STOPPED)

    @pytest.mark.asyncio
    async def test_full_lifecycle_transitions(self, registry, sample_config):
        """Test full lifecycle: REGISTERED -> RUNNING -> STOPPED."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        # Start
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        info = registry.get("bot_001")
        assert info.state == BotState.RUNNING
        assert info.started_at is not None

        # Stop
        await registry.update_state("bot_001", BotState.STOPPING)
        await registry.update_state("bot_001", BotState.STOPPED)

        info = registry.get("bot_001")
        assert info.state == BotState.STOPPED
        assert info.stopped_at is not None

    @pytest.mark.asyncio
    async def test_error_state_sets_message(self, registry, sample_config):
        """Test error state records error message."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)

        await registry.update_state(
            "bot_001",
            BotState.ERROR,
            message="Connection failed",
        )

        info = registry.get("bot_001")
        assert info.state == BotState.ERROR
        assert info.error_message == "Connection failed"

    def test_validate_transition_method(self, registry):
        """Test validate_transition method."""
        assert registry.validate_transition(BotState.REGISTERED, BotState.INITIALIZING)
        assert registry.validate_transition(BotState.RUNNING, BotState.PAUSED)
        assert not registry.validate_transition(BotState.REGISTERED, BotState.RUNNING)
        assert not registry.validate_transition(BotState.STOPPED, BotState.RUNNING)


class TestQueryMethods:
    """Tests for query methods."""

    @pytest.mark.asyncio
    async def test_get_bot(self, registry, sample_config):
        """Test getting bot by ID."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        info = registry.get("bot_001")

        assert info is not None
        assert info.bot_id == "bot_001"

    @pytest.mark.asyncio
    async def test_get_nonexistent_bot(self, registry):
        """Test getting non-existent bot returns None."""
        info = registry.get("nonexistent")

        assert info is None

    @pytest.mark.asyncio
    async def test_get_all_bots(self, registry, sample_config):
        """Test getting all bots."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        bots = registry.get_all()

        assert len(bots) == 2
        bot_ids = [b.bot_id for b in bots]
        assert "bot_001" in bot_ids
        assert "bot_002" in bot_ids

    @pytest.mark.asyncio
    async def test_get_by_state(self, registry, sample_config):
        """Test filtering bots by state."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        running = registry.get_by_state(BotState.RUNNING)
        registered = registry.get_by_state(BotState.REGISTERED)

        assert len(running) == 1
        assert running[0].bot_id == "bot_001"
        assert len(registered) == 1
        assert registered[0].bot_id == "bot_002"

    @pytest.mark.asyncio
    async def test_get_by_type(self, registry, sample_config):
        """Test filtering bots by type."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)

        grid_bots = registry.get_by_type(BotType.GRID)
        dca_bots = registry.get_by_type(BotType.DCA)

        assert len(grid_bots) == 1
        assert grid_bots[0].bot_id == "bot_001"
        assert len(dca_bots) == 1
        assert dca_bots[0].bot_id == "bot_002"

    @pytest.mark.asyncio
    async def test_get_by_symbol(self, registry):
        """Test filtering bots by symbol."""
        config1 = {"symbol": "BTCUSDT", "total_investment": "10000"}
        config2 = {"symbol": "ETHUSDT", "total_investment": "5000"}

        await registry.register("bot_001", BotType.GRID, config1)
        await registry.register("bot_002", BotType.GRID, config2)

        btc_bots = registry.get_by_symbol("BTCUSDT")
        eth_bots = registry.get_by_symbol("ETHUSDT")

        assert len(btc_bots) == 1
        assert len(eth_bots) == 1


class TestInstanceManagement:
    """Tests for bot instance management."""

    @pytest.mark.asyncio
    async def test_bind_instance(self, registry, sample_config):
        """Test binding bot instance."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        mock_bot = MagicMock()
        mock_bot.bot_id = "bot_001"

        registry.bind_instance("bot_001", mock_bot)

        assert registry.get_bot_instance("bot_001") == mock_bot

    def test_bind_instance_unregistered_bot_fails(self, registry):
        """Test binding instance to unregistered bot fails."""
        mock_bot = MagicMock()

        with pytest.raises(BotNotFoundError):
            registry.bind_instance("nonexistent", mock_bot)

    @pytest.mark.asyncio
    async def test_unbind_instance(self, registry, sample_config):
        """Test unbinding bot instance."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        mock_bot = MagicMock()
        registry.bind_instance("bot_001", mock_bot)

        registry.unbind_instance("bot_001")

        assert registry.get_bot_instance("bot_001") is None

    @pytest.mark.asyncio
    async def test_get_bot_instance_returns_none_when_not_bound(
        self, registry, sample_config
    ):
        """Test getting instance when not bound returns None."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        assert registry.get_bot_instance("bot_001") is None


class TestStatistics:
    """Tests for statistics and summary."""

    @pytest.mark.asyncio
    async def test_get_summary(self, registry, sample_config):
        """Test get summary statistics."""
        config_eth = {**sample_config, "symbol": "ETHUSDT"}

        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.DCA, sample_config)
        await registry.register("bot_003", BotType.GRID, config_eth)

        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        summary = registry.get_summary()

        assert summary["total_bots"] == 3
        assert summary["by_state"]["running"] == 1
        assert summary["by_state"]["registered"] == 2
        assert summary["by_type"]["grid"] == 2
        assert summary["by_type"]["dca"] == 1
        assert summary["by_symbol"]["BTCUSDT"] == 2
        assert summary["by_symbol"]["ETHUSDT"] == 1

    @pytest.mark.asyncio
    async def test_bot_count_property(self, registry, sample_config):
        """Test bot_count property."""
        assert registry.bot_count == 0

        await registry.register("bot_001", BotType.GRID, sample_config)

        assert registry.bot_count == 1

    @pytest.mark.asyncio
    async def test_running_count_property(self, registry, sample_config):
        """Test running_count property."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.register("bot_002", BotType.GRID, sample_config)

        assert registry.running_count == 0

        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)

        assert registry.running_count == 1


class TestEventRecording:
    """Tests for event recording."""

    @pytest.mark.asyncio
    async def test_events_recorded_on_register(self, registry, sample_config):
        """Test events are recorded on registration."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        events = registry.get_events("bot_001")

        assert len(events) == 1
        assert events[0].event_type == "registered"
        assert events[0].new_state == BotState.REGISTERED

    @pytest.mark.asyncio
    async def test_events_recorded_on_state_change(self, registry, sample_config):
        """Test events are recorded on state change."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)

        events = registry.get_events("bot_001")

        assert len(events) == 2
        assert events[0].event_type == "state_changed"
        assert events[0].old_state == BotState.REGISTERED
        assert events[0].new_state == BotState.INITIALIZING

    @pytest.mark.asyncio
    async def test_get_events_with_limit(self, registry, sample_config):
        """Test getting events with limit."""
        await registry.register("bot_001", BotType.GRID, sample_config)
        await registry.update_state("bot_001", BotState.INITIALIZING)
        await registry.update_state("bot_001", BotState.RUNNING)
        await registry.update_state("bot_001", BotState.PAUSED)

        events = registry.get_events("bot_001", limit=2)

        assert len(events) == 2
        # Should be newest first
        assert events[0].new_state == BotState.PAUSED


class TestSingletonPattern:
    """Tests for singleton pattern."""

    def test_singleton_returns_same_instance(self):
        """Test singleton returns same instance."""
        registry1 = BotRegistry.get_instance()
        registry2 = BotRegistry.get_instance()

        assert registry1 is registry2

    def test_reset_instance_creates_new(self):
        """Test reset creates new instance."""
        registry1 = BotRegistry.get_instance()
        BotRegistry.reset_instance()
        registry2 = BotRegistry.get_instance()

        assert registry1 is not registry2


class TestContainerMethods:
    """Tests for container-like methods."""

    @pytest.mark.asyncio
    async def test_contains(self, registry, sample_config):
        """Test __contains__ method."""
        await registry.register("bot_001", BotType.GRID, sample_config)

        assert "bot_001" in registry
        assert "nonexistent" not in registry

    @pytest.mark.asyncio
    async def test_len(self, registry, sample_config):
        """Test __len__ method."""
        assert len(registry) == 0

        await registry.register("bot_001", BotType.GRID, sample_config)

        assert len(registry) == 1


class TestPersistence:
    """Tests for database persistence."""

    @pytest.mark.asyncio
    async def test_register_saves_to_db(self, mock_db_manager, sample_config):
        """Test registration saves to database."""
        registry = BotRegistry(db_manager=mock_db_manager)

        await registry.register("bot_001", BotType.GRID, sample_config)

        mock_db_manager.upsert_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_state_change_saves_to_db(self, mock_db_manager, sample_config):
        """Test state change saves to database."""
        registry = BotRegistry(db_manager=mock_db_manager)
        await registry.register("bot_001", BotType.GRID, sample_config)
        mock_db_manager.upsert_bot.reset_mock()

        await registry.update_state("bot_001", BotState.INITIALIZING)

        mock_db_manager.upsert_bot.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_from_db(self, mock_db_manager):
        """Test loading bots from database."""
        mock_db_manager.get_all_bots = AsyncMock(
            return_value=[
                {
                    "bot_id": "bot_001",
                    "bot_type": "grid",
                    "symbol": "BTCUSDT",
                    "market_type": "spot",
                    "state": "stopped",
                    "config": {},
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
            ]
        )
        registry = BotRegistry(db_manager=mock_db_manager)

        loaded = await registry.load_from_db()

        assert loaded == 1
        assert "bot_001" in registry


class TestNotifications:
    """Tests for notifications."""

    @pytest.mark.asyncio
    async def test_register_sends_notification(self, mock_notifier, sample_config):
        """Test registration sends notification."""
        registry = BotRegistry(notifier=mock_notifier)

        await registry.register("bot_001", BotType.GRID, sample_config)

        mock_notifier.notify_bot_registered.assert_called_once_with(
            bot_id="bot_001",
            bot_type="grid",
            symbol="BTCUSDT",
        )

    @pytest.mark.asyncio
    async def test_state_change_sends_notification(self, mock_notifier, sample_config):
        """Test state change sends notification."""
        registry = BotRegistry(notifier=mock_notifier)
        await registry.register("bot_001", BotType.GRID, sample_config)
        mock_notifier.notify_bot_state_changed.reset_mock()

        await registry.update_state("bot_001", BotState.INITIALIZING)

        mock_notifier.notify_bot_state_changed.assert_called_once_with(
            bot_id="bot_001",
            old_state="registered",
            new_state="initializing",
            message="",
        )


class TestBotInfoModel:
    """Tests for BotInfo model."""

    def test_to_dict(self):
        """Test BotInfo to_dict conversion."""
        info = BotInfo(
            bot_id="bot_001",
            bot_type=BotType.GRID,
            symbol="BTCUSDT",
            market_type=MarketType.SPOT,
            config={"key": "value"},
        )

        data = info.to_dict()

        assert data["bot_id"] == "bot_001"
        assert data["bot_type"] == "grid"
        assert data["symbol"] == "BTCUSDT"
        assert data["market_type"] == "spot"
        assert data["state"] == "registered"

    def test_from_dict(self):
        """Test BotInfo from_dict conversion."""
        data = {
            "bot_id": "bot_001",
            "bot_type": "dca",
            "symbol": "ETHUSDT",
            "market_type": "futures",
            "state": "running",
            "config": {},
            "created_at": "2024-01-01T00:00:00+00:00",
        }

        info = BotInfo.from_dict(data)

        assert info.bot_id == "bot_001"
        assert info.bot_type == BotType.DCA
        assert info.symbol == "ETHUSDT"
        assert info.market_type == MarketType.FUTURES
        assert info.state == BotState.RUNNING


class TestRegistryEvent:
    """Tests for RegistryEvent model."""

    def test_create_event(self):
        """Test creating event with factory method."""
        event = RegistryEvent.create(
            bot_id="bot_001",
            event_type="test",
            old_state=BotState.REGISTERED,
            new_state=BotState.INITIALIZING,
            message="Test message",
        )

        assert event.bot_id == "bot_001"
        assert event.event_type == "test"
        assert event.old_state == BotState.REGISTERED
        assert event.new_state == BotState.INITIALIZING
        assert event.event_id is not None
        assert event.timestamp is not None

    def test_event_to_dict(self):
        """Test event to_dict conversion."""
        event = RegistryEvent.create(
            bot_id="bot_001",
            event_type="test",
            old_state=BotState.REGISTERED,
            new_state=BotState.RUNNING,
        )

        data = event.to_dict()

        assert data["bot_id"] == "bot_001"
        assert data["old_state"] == "registered"
        assert data["new_state"] == "running"
