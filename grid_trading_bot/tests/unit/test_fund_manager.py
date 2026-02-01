"""
Fund Manager Unit Tests.

Tests for the fund allocation and distribution system.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.fund_manager.core.allocator import (
    FixedRatioAllocator,
    FixedAmountAllocator,
    DynamicWeightAllocator,
    create_allocator,
)
from src.fund_manager.core.fund_pool import FundPool
from src.fund_manager.core.dispatcher import Dispatcher
from src.fund_manager.manager import FundManager
from src.fund_manager.models.config import (
    AllocationStrategy,
    BotAllocation,
    FundManagerConfig,
)
from src.fund_manager.models.records import (
    AllocationRecord,
    BalanceSnapshot,
    DispatchResult,
)
from src.fund_manager.notifier.bot_notifier import (
    BotNotifier,
    FileNotifier,
    NotificationMessage,
)


class TestFundManagerConfig:
    """Test FundManagerConfig."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = FundManagerConfig()

        assert config.enabled is True
        assert config.poll_interval == 60
        assert config.deposit_threshold == Decimal("10")
        assert config.reserve_ratio == Decimal("0.1")
        assert config.auto_dispatch is True
        assert config.strategy == AllocationStrategy.FIXED_RATIO

    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "enabled": True,
            "poll_interval": 30,
            "deposit_threshold": "50",
            "reserve_ratio": "0.2",
            "auto_dispatch": False,
            "strategy": "fixed_amount",
            "allocations": [
                {
                    "bot_pattern": "grid_*",
                    "ratio": "0.5",
                    "min_capital": "100",
                    "max_capital": "10000",
                    "priority": 10,
                    "enabled": True,
                }
            ],
        }

        config = FundManagerConfig.from_dict(data)

        assert config.poll_interval == 30
        assert config.deposit_threshold == Decimal("50")
        assert config.reserve_ratio == Decimal("0.2")
        assert config.auto_dispatch is False
        assert config.strategy == AllocationStrategy.FIXED_AMOUNT
        assert len(config.allocations) == 1
        assert config.allocations[0].bot_pattern == "grid_*"

    def test_bot_allocation_matching(self):
        """Test bot pattern matching."""
        alloc = BotAllocation(
            bot_pattern="grid_*",
            ratio=Decimal("0.5"),
        )

        assert alloc.matches("grid_btc") is True
        assert alloc.matches("grid_eth_usdt") is True
        assert alloc.matches("bollinger_btc") is False


class TestBotAllocation:
    """Test BotAllocation."""

    def test_allocation_defaults(self):
        """Test default allocation values."""
        alloc = BotAllocation(bot_pattern="test_*")

        assert alloc.ratio == Decimal("0.0")
        assert alloc.min_capital == Decimal("0")
        assert alloc.max_capital == Decimal("1000000")
        assert alloc.priority == 0
        assert alloc.enabled is True

    def test_allocation_to_dict(self):
        """Test converting allocation to dictionary."""
        alloc = BotAllocation(
            bot_pattern="grid_*",
            ratio=Decimal("0.3"),
            min_capital=Decimal("100"),
            max_capital=Decimal("5000"),
            priority=5,
        )

        data = alloc.to_dict()

        assert data["bot_pattern"] == "grid_*"
        assert data["ratio"] == "0.3"
        assert data["min_capital"] == "100"
        assert data["max_capital"] == "5000"
        assert data["priority"] == 5


class TestBalanceSnapshot:
    """Test BalanceSnapshot."""

    def test_snapshot_creation(self):
        """Test creating balance snapshot."""
        snapshot = BalanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=Decimal("10000"),
            available_balance=Decimal("8000"),
            allocated_balance=Decimal("5000"),
            reserved_balance=Decimal("1000"),
        )

        assert snapshot.total_balance == Decimal("10000")
        assert snapshot.available_balance == Decimal("8000")
        assert snapshot.unallocated_balance == Decimal("2000")  # 8000 - 5000 - 1000

    def test_snapshot_to_dict(self):
        """Test converting snapshot to dictionary."""
        snapshot = BalanceSnapshot(
            timestamp=datetime.now(timezone.utc),
            total_balance=Decimal("10000"),
            available_balance=Decimal("8000"),
        )

        data = snapshot.to_dict()

        assert "timestamp" in data
        assert data["total_balance"] == "10000"
        assert data["available_balance"] == "8000"


class TestAllocationRecord:
    """Test AllocationRecord."""

    def test_record_creation(self):
        """Test creating allocation record."""
        record = AllocationRecord(
            bot_id="grid_btc",
            amount=Decimal("500"),
            trigger="deposit",
            previous_allocation=Decimal("1000"),
            new_allocation=Decimal("1500"),
        )

        assert record.bot_id == "grid_btc"
        assert record.amount == Decimal("500")
        assert record.change == Decimal("500")  # 1500 - 1000
        assert record.success is True

    def test_record_to_dict(self):
        """Test converting record to dictionary."""
        record = AllocationRecord(
            bot_id="grid_btc",
            amount=Decimal("500"),
            trigger="manual",
        )

        data = record.to_dict()

        assert data["bot_id"] == "grid_btc"
        assert data["amount"] == "500"
        assert data["trigger"] == "manual"
        assert "id" in data
        assert "timestamp" in data


class TestFixedRatioAllocator:
    """Test FixedRatioAllocator."""

    def test_calculate_allocations(self):
        """Test calculating allocations with fixed ratios."""
        config = FundManagerConfig(
            allocations=[
                BotAllocation(bot_pattern="grid_*", ratio=Decimal("0.4")),
                BotAllocation(bot_pattern="bollinger_*", ratio=Decimal("0.3")),
            ]
        )

        allocator = FixedRatioAllocator(config)

        allocations = allocator.calculate(
            available_funds=Decimal("1000"),
            bot_allocations=config.allocations,
            current_allocations={},
            bot_ids=["grid_btc", "bollinger_eth"],
        )

        assert "grid_btc" in allocations
        assert "bollinger_eth" in allocations
        assert allocations["grid_btc"] == Decimal("400")
        assert allocations["bollinger_eth"] == Decimal("300")

    def test_empty_bot_ids(self):
        """Test with no bots."""
        config = FundManagerConfig()
        allocator = FixedRatioAllocator(config)

        allocations = allocator.calculate(
            available_funds=Decimal("1000"),
            bot_allocations=[],
            current_allocations={},
            bot_ids=[],
        )

        assert allocations == {}

    def test_zero_available_funds(self):
        """Test with zero available funds."""
        config = FundManagerConfig(
            allocations=[
                BotAllocation(bot_pattern="grid_*", ratio=Decimal("0.5")),
            ]
        )
        allocator = FixedRatioAllocator(config)

        allocations = allocator.calculate(
            available_funds=Decimal("0"),
            bot_allocations=config.allocations,
            current_allocations={},
            bot_ids=["grid_btc"],
        )

        assert allocations == {}


class TestFixedAmountAllocator:
    """Test FixedAmountAllocator."""

    def test_calculate_fixed_amounts(self):
        """Test calculating allocations with fixed amounts."""
        config = FundManagerConfig(
            strategy=AllocationStrategy.FIXED_AMOUNT,
            allocations=[
                BotAllocation(
                    bot_pattern="grid_btc",
                    fixed_amount=Decimal("500"),
                    min_capital=Decimal("100"),
                    priority=10,
                ),
                BotAllocation(
                    bot_pattern="grid_eth",
                    fixed_amount=Decimal("300"),
                    min_capital=Decimal("100"),
                    priority=5,
                ),
            ]
        )

        allocator = FixedAmountAllocator(config)

        allocations = allocator.calculate(
            available_funds=Decimal("1000"),
            bot_allocations=config.allocations,
            current_allocations={},
            bot_ids=["grid_btc", "grid_eth"],
        )

        assert "grid_btc" in allocations
        assert "grid_eth" in allocations
        assert allocations["grid_btc"] == Decimal("500")
        assert allocations["grid_eth"] == Decimal("300")


class TestDynamicWeightAllocator:
    """Test DynamicWeightAllocator."""

    def test_calculate_with_weights(self):
        """Test calculating allocations with dynamic weights."""
        config = FundManagerConfig(
            strategy=AllocationStrategy.DYNAMIC_WEIGHT,
            allocations=[
                BotAllocation(bot_pattern="grid_btc", ratio=Decimal("0.5")),
                BotAllocation(bot_pattern="grid_eth", ratio=Decimal("0.5")),
            ]
        )

        # Performance-based weights
        weights = {
            "grid_btc": Decimal("0.7"),  # Better performer
            "grid_eth": Decimal("0.3"),
        }

        allocator = DynamicWeightAllocator(config, weights)

        allocations = allocator.calculate(
            available_funds=Decimal("1000"),
            bot_allocations=config.allocations,
            current_allocations={},
            bot_ids=["grid_btc", "grid_eth"],
        )

        # Should allocate based on weights (70/30 split)
        assert allocations["grid_btc"] == Decimal("700")
        assert allocations["grid_eth"] == Decimal("300")


class TestCreateAllocator:
    """Test allocator factory."""

    def test_create_fixed_ratio(self):
        """Test creating fixed ratio allocator."""
        config = FundManagerConfig(strategy=AllocationStrategy.FIXED_RATIO)
        allocator = create_allocator(config)

        assert isinstance(allocator, FixedRatioAllocator)

    def test_create_fixed_amount(self):
        """Test creating fixed amount allocator."""
        config = FundManagerConfig(strategy=AllocationStrategy.FIXED_AMOUNT)
        allocator = create_allocator(config)

        assert isinstance(allocator, FixedAmountAllocator)

    def test_create_dynamic_weight(self):
        """Test creating dynamic weight allocator."""
        config = FundManagerConfig(strategy=AllocationStrategy.DYNAMIC_WEIGHT)
        allocator = create_allocator(config)

        assert isinstance(allocator, DynamicWeightAllocator)


class TestFundPool:
    """Test FundPool."""

    def test_pool_creation(self):
        """Test creating fund pool."""
        config = FundManagerConfig(reserve_ratio=Decimal("0.1"))
        pool = FundPool(config=config)

        assert pool.total_balance == Decimal("0")
        assert pool.available_balance == Decimal("0")
        assert pool.allocated_balance == Decimal("0")

    def test_update_from_values(self):
        """Test updating pool from values."""
        pool = FundPool()

        snapshot = pool.update_from_values(
            total_balance=Decimal("10000"),
            available_balance=Decimal("8000"),
        )

        assert pool.total_balance == Decimal("10000")
        assert pool.available_balance == Decimal("8000")
        assert snapshot.total_balance == Decimal("10000")

    def test_deposit_detection(self):
        """Test detecting deposits."""
        config = FundManagerConfig(deposit_threshold=Decimal("100"))
        pool = FundPool(config=config)

        # Initial balance
        pool.update_from_values(
            total_balance=Decimal("1000"),
            available_balance=Decimal("1000"),
        )

        assert pool.detect_deposit() is False

        # Simulate deposit
        pool.update_from_values(
            total_balance=Decimal("1500"),
            available_balance=Decimal("1500"),
        )

        assert pool.detect_deposit() is True
        assert pool.get_deposit_amount() == Decimal("500")

    def test_allocation_management(self):
        """Test managing allocations."""
        pool = FundPool()

        pool.set_allocation("grid_btc", Decimal("1000"))
        pool.set_allocation("grid_eth", Decimal("500"))

        assert pool.get_allocation("grid_btc") == Decimal("1000")
        assert pool.get_allocation("grid_eth") == Decimal("500")
        assert pool.allocated_balance == Decimal("1500")

        pool.add_allocation("grid_btc", Decimal("200"))
        assert pool.get_allocation("grid_btc") == Decimal("1200")

        removed = pool.remove_allocation("grid_eth")
        assert removed == Decimal("500")
        assert pool.get_allocation("grid_eth") == Decimal("0")

    def test_unallocated_balance(self):
        """Test calculating unallocated balance."""
        config = FundManagerConfig(reserve_ratio=Decimal("0.1"))
        pool = FundPool(config=config)

        pool.update_from_values(
            total_balance=Decimal("10000"),
            available_balance=Decimal("8000"),
        )
        pool.set_allocation("grid_btc", Decimal("3000"))

        # Unallocated = available - allocated - reserved
        # = 8000 - 3000 - (10000 * 0.1) = 8000 - 3000 - 1000 = 4000
        assert pool.get_unallocated() == Decimal("4000")


class TestNotificationMessage:
    """Test NotificationMessage."""

    def test_message_creation(self):
        """Test creating notification message."""
        msg = NotificationMessage(
            bot_id="grid_btc",
            event_type="allocation_update",
            new_allocation=Decimal("500"),
            total_allocation=Decimal("1500"),
            trigger="deposit",
            message="Test message",
        )

        assert msg.bot_id == "grid_btc"
        assert msg.new_allocation == Decimal("500")
        assert msg.total_allocation == Decimal("1500")

    def test_message_to_dict(self):
        """Test converting message to dictionary."""
        msg = NotificationMessage(
            bot_id="grid_btc",
            new_allocation=Decimal("500"),
            total_allocation=Decimal("1500"),
        )

        data = msg.to_dict()

        assert data["bot_id"] == "grid_btc"
        assert data["event_type"] == "allocation_update"
        assert data["data"]["new_allocation"] == 500.0
        assert data["data"]["total_allocation"] == 1500.0

    def test_message_to_json(self):
        """Test converting message to JSON."""
        msg = NotificationMessage(
            bot_id="grid_btc",
            new_allocation=Decimal("500"),
            total_allocation=Decimal("1500"),
        )

        json_str = msg.to_json()
        data = json.loads(json_str)

        assert data["bot_id"] == "grid_btc"


class TestFileNotifier:
    """Test FileNotifier."""

    @pytest.mark.asyncio
    async def test_notify_writes_file(self):
        """Test that notify writes a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = FileNotifier(default_path=tmpdir, retry_count=1)

            msg = NotificationMessage(
                bot_id="test_bot",
                new_allocation=Decimal("100"),
                total_allocation=Decimal("500"),
            )

            result = await notifier.notify(msg)

            assert result is True

            # Check file was created
            file_path = Path(tmpdir) / "test_bot.json"
            assert file_path.exists()

            # Check content
            with open(file_path) as f:
                data = json.load(f)

            assert data["bot_id"] == "test_bot"
            assert data["data"]["new_allocation"] == 100.0

    def test_read_notification(self):
        """Test reading notification file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = FileNotifier(default_path=tmpdir)

            # Write test file
            file_path = Path(tmpdir) / "test_bot.json"
            test_data = {
                "bot_id": "test_bot",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "event_type": "allocation_update",
                "data": {
                    "new_allocation": 100,
                    "total_allocation": 500,
                    "trigger": "test",
                    "message": "Test",
                },
            }
            with open(file_path, "w") as f:
                json.dump(test_data, f)

            # Read it back
            msg = notifier.read_notification("test_bot")

            assert msg is not None
            assert msg.bot_id == "test_bot"
            assert msg.new_allocation == Decimal("100")

    def test_clear_notification(self):
        """Test clearing notification file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = FileNotifier(default_path=tmpdir)

            # Create test file
            file_path = Path(tmpdir) / "test_bot.json"
            file_path.write_text("{}")

            assert file_path.exists()

            result = notifier.clear_notification("test_bot")

            assert result is True
            assert not file_path.exists()


class TestBotNotifier:
    """Test BotNotifier."""

    def test_register_bot(self):
        """Test registering bots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = BotNotifier(default_notification_path=tmpdir)

            notifier.register_bot("bot1", "file", f"{tmpdir}/bot1.json")
            notifier.register_bot("bot2", "api", "http://localhost:8080/notify")
            notifier.register_bot("bot3", "none")

            status = notifier.get_status()

            assert status["registered_bots"] == 3
            assert status["bot_methods"]["bot1"] == "file"
            assert status["bot_methods"]["bot2"] == "api"
            assert status["bot_methods"]["bot3"] == "none"

    @pytest.mark.asyncio
    async def test_notify_file_method(self):
        """Test notification with file method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            notifier = BotNotifier(default_notification_path=tmpdir)
            notifier.register_bot("test_bot", "file")

            result = await notifier.notify_allocation(
                bot_id="test_bot",
                new_allocation=Decimal("100"),
                total_allocation=Decimal("500"),
                trigger="test",
            )

            assert result is True

    @pytest.mark.asyncio
    async def test_notify_none_method(self):
        """Test notification with none method (disabled)."""
        notifier = BotNotifier()
        notifier.register_bot("test_bot", "none")

        result = await notifier.notify_allocation(
            bot_id="test_bot",
            new_allocation=Decimal("100"),
            total_allocation=Decimal("500"),
        )

        # Should return True even though no notification sent
        assert result is True


class TestDispatcher:
    """Test Dispatcher."""

    @pytest.mark.asyncio
    async def test_dispatch_with_file_notifier(self):
        """Test dispatching with file-based notification."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot_notifier = BotNotifier(default_notification_path=tmpdir)
            bot_notifier.register_bot("grid_btc", "file")

            dispatcher = Dispatcher(bot_notifier=bot_notifier)
            dispatcher.set_notification_method("grid_btc", "file")

            records = await dispatcher.dispatch(
                allocations={"grid_btc": Decimal("1000")},
                trigger="test",
            )

            assert len(records) == 1
            assert records[0].bot_id == "grid_btc"


class TestFundManager:
    """Test FundManager integration."""

    def setup_method(self):
        """Reset singleton before each test."""
        FundManager.reset_instance()

    def teardown_method(self):
        """Reset singleton after each test."""
        FundManager.reset_instance()

    def test_singleton_pattern(self):
        """Test that FundManager is a singleton."""
        config = FundManagerConfig()

        manager1 = FundManager(config=config)
        manager2 = FundManager(config=config)

        assert manager1 is manager2

    def test_manager_status(self):
        """Test getting manager status."""
        config = FundManagerConfig(
            enabled=True,
            poll_interval=30,
            auto_dispatch=True,
        )

        manager = FundManager(config=config)
        status = manager.get_status()

        assert status["enabled"] is True
        assert status["poll_interval"] == 30
        assert status["auto_dispatch"] is True
        assert status["running"] is False

    @pytest.mark.asyncio
    async def test_set_allocation(self):
        """Test setting allocation manually."""
        manager = FundManager()

        await manager.set_allocation("grid_btc", Decimal("1000"))

        assert manager.fund_pool.get_allocation("grid_btc") == Decimal("1000")

    def test_adjust_ratio(self):
        """Test adjusting allocation ratio."""
        config = FundManagerConfig(
            allocations=[
                BotAllocation(bot_pattern="grid_*", ratio=Decimal("0.3")),
            ]
        )

        manager = FundManager(config=config)

        result = manager.adjust_ratio("grid_*", Decimal("0.5"))

        assert result is True
        assert manager.config.allocations[0].ratio == Decimal("0.5")

    @pytest.mark.asyncio
    async def test_recall_funds(self):
        """Test recalling funds from a bot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot_notifier = BotNotifier(default_notification_path=tmpdir)
            bot_notifier.register_bot("grid_btc", "file")

            config = FundManagerConfig()
            manager = FundManager(config=config)
            manager._dispatcher._bot_notifier = bot_notifier
            manager._dispatcher.set_notification_method("grid_btc", "file")

            # Set initial allocation
            await manager.set_allocation("grid_btc", Decimal("1000"))

            # Recall funds
            record = await manager.recall_funds("grid_btc", Decimal("500"))

            assert record.success is True
            assert record.amount == Decimal("-500")  # Negative for recall
            assert manager.fund_pool.get_allocation("grid_btc") == Decimal("500")

    @pytest.mark.asyncio
    async def test_recall_all_funds(self):
        """Test recalling all funds from all bots."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bot_notifier = BotNotifier(default_notification_path=tmpdir)
            bot_notifier.register_bot("grid_btc", "file")
            bot_notifier.register_bot("grid_eth", "file")

            config = FundManagerConfig()
            manager = FundManager(config=config)
            manager._dispatcher._bot_notifier = bot_notifier
            manager._dispatcher.set_notification_method("grid_btc", "file")
            manager._dispatcher.set_notification_method("grid_eth", "file")

            # Set initial allocations
            await manager.set_allocation("grid_btc", Decimal("1000"))
            await manager.set_allocation("grid_eth", Decimal("500"))

            # Recall all
            records = await manager.recall_all_funds()

            assert len(records) == 2
            assert all(r.success for r in records)
            assert manager.fund_pool.allocated_balance == Decimal("0")


class TestDispatchResult:
    """Test DispatchResult."""

    def test_result_aggregation(self):
        """Test aggregating dispatch results."""
        result = DispatchResult(trigger="test")

        record1 = AllocationRecord(
            bot_id="bot1",
            amount=Decimal("100"),
            trigger="test",
            success=True,
        )
        record2 = AllocationRecord(
            bot_id="bot2",
            amount=Decimal("200"),
            trigger="test",
            success=True,
        )
        record3 = AllocationRecord(
            bot_id="bot3",
            amount=Decimal("300"),
            trigger="test",
            success=False,
            error_message="Failed",
        )

        result.add_allocation(record1)
        result.add_allocation(record2)
        result.add_allocation(record3)

        assert result.successful_count == 2
        assert result.failed_count == 1
        assert result.total_dispatched == Decimal("300")  # Only successful
        assert len(result.errors) == 1


class TestShouldRebalance:
    """Test _should_rebalance logic."""

    def setup_method(self):
        FundManager.reset_instance()

    def teardown_method(self):
        FundManager.reset_instance()

    def _make_manager(self, frequency="weekly", day=0, hour=0):
        config = FundManagerConfig(
            rebalance_frequency=frequency,
            rebalance_day=day,
            rebalance_hour=hour,
        )
        manager = FundManager(config=config)
        manager._last_rebalance_time = None
        return manager

    def test_never_frequency_returns_false(self):
        """rebalance_frequency='never' should always return False."""
        manager = self._make_manager(frequency="never")
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_weekly_correct_day_and_hour(self, mock_dt):
        """Weekly rebalance triggers on correct weekday + hour."""
        # Monday (weekday=0), hour=3
        fake_now = datetime(2026, 2, 2, 3, 15, 0, tzinfo=timezone.utc)  # Monday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="weekly", day=0, hour=3)
        assert manager._should_rebalance() is True

    @patch("src.fund_manager.manager.datetime")
    def test_weekly_wrong_day(self, mock_dt):
        """Weekly rebalance does NOT trigger on wrong weekday."""
        # Tuesday (weekday=1), hour=3
        fake_now = datetime(2026, 2, 3, 3, 15, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="weekly", day=0, hour=3)
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_weekly_wrong_hour(self, mock_dt):
        """Weekly rebalance does NOT trigger on wrong hour."""
        fake_now = datetime(2026, 2, 2, 10, 0, 0, tzinfo=timezone.utc)  # Monday hour=10
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="weekly", day=0, hour=3)
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_daily_correct_hour(self, mock_dt):
        """Daily rebalance triggers at correct hour regardless of weekday."""
        fake_now = datetime(2026, 2, 5, 8, 30, 0, tzinfo=timezone.utc)  # Thursday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="daily", day=0, hour=8)
        assert manager._should_rebalance() is True

    @patch("src.fund_manager.manager.datetime")
    def test_daily_wrong_hour(self, mock_dt):
        """Daily rebalance does NOT trigger at wrong hour."""
        fake_now = datetime(2026, 2, 5, 14, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="daily", day=0, hour=8)
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_monthly_correct_day(self, mock_dt):
        """Monthly rebalance triggers on correct day of month."""
        # rebalance_day=0 -> day of month 1
        fake_now = datetime(2026, 3, 1, 0, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="monthly", day=0, hour=0)
        assert manager._should_rebalance() is True

    @patch("src.fund_manager.manager.datetime")
    def test_monthly_wrong_day(self, mock_dt):
        """Monthly rebalance does NOT trigger on wrong day of month."""
        fake_now = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="monthly", day=0, hour=0)
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_duplicate_trigger_prevention(self, mock_dt):
        """Should not trigger again within the same hour."""
        fake_now = datetime(2026, 2, 2, 3, 15, 0, tzinfo=timezone.utc)
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="weekly", day=0, hour=3)

        # First call triggers
        assert manager._should_rebalance() is True

        # Simulate last rebalance was 30 minutes ago
        manager._last_rebalance_time = datetime(
            2026, 2, 2, 2, 45, 0, tzinfo=timezone.utc
        )
        assert manager._should_rebalance() is False

    @patch("src.fund_manager.manager.datetime")
    def test_trigger_after_cooldown(self, mock_dt):
        """Should trigger again after 1 hour cooldown."""
        fake_now = datetime(2026, 2, 9, 3, 15, 0, tzinfo=timezone.utc)  # Next Monday
        mock_dt.now.return_value = fake_now
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)

        manager = self._make_manager(frequency="weekly", day=0, hour=3)
        # Last rebalance was over 1 hour ago
        manager._last_rebalance_time = datetime(
            2026, 2, 2, 3, 0, 0, tzinfo=timezone.utc
        )
        assert manager._should_rebalance() is True


class TestExposureBlocking:
    """Test that dispatch_funds blocks when exposure limit is exceeded."""

    def setup_method(self):
        FundManager.reset_instance()

    def teardown_method(self):
        FundManager.reset_instance()

    @pytest.mark.asyncio
    async def test_dispatch_blocked_when_exposure_exceeded(self):
        """dispatch_funds should fail when exposure exceeds 3x limit."""
        config = FundManagerConfig(
            allocations=[
                BotAllocation(bot_pattern="grid_*", ratio=Decimal("0.5")),
            ]
        )
        manager = FundManager(config=config)

        # Set up balance and high-leverage allocation to exceed 3x
        pool = manager.fund_pool
        pool.update_from_values(
            total_balance=Decimal("1000"),
            available_balance=Decimal("800"),
        )
        pool.set_allocation("grid_btc", Decimal("500"))
        pool.set_leverage("grid_btc", 10)  # 500 * 10 = 5000 notional, 5x > 3x

        result = await manager.dispatch_funds(trigger="manual")

        assert result.success is False
        assert any("exposure" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_dispatch_allowed_when_exposure_within_limit(self):
        """dispatch_funds should proceed when exposure is within 3x limit."""
        config = FundManagerConfig(
            allocations=[
                BotAllocation(bot_pattern="grid_*", ratio=Decimal("0.5")),
            ]
        )
        manager = FundManager(config=config)

        pool = manager.fund_pool
        pool.update_from_values(
            total_balance=Decimal("1000"),
            available_balance=Decimal("800"),
        )
        pool.set_allocation("grid_btc", Decimal("200"))
        pool.set_leverage("grid_btc", 2)  # 200 * 2 = 400 notional, 0.4x < 3x

        # No bots registered so it will return "no active bots" — but NOT exposure error
        result = await manager.dispatch_funds(trigger="manual")

        # Should not contain exposure error
        assert not any("exposure" in e.lower() for e in result.errors)

    @pytest.mark.asyncio
    async def test_dispatch_ok_with_zero_balance(self):
        """dispatch_funds should not block on zero balance (no exposure issue)."""
        manager = FundManager(config=FundManagerConfig())

        # No balance set → total_balance=0, check_exposure_limit returns False
        result = await manager.dispatch_funds(trigger="manual")

        assert not any("exposure" in e.lower() for e in result.errors)


class TestRebalanceConfig:
    """Test rebalance config fields parsing."""

    def test_config_defaults_rebalance(self):
        """Default rebalance config should be 'never'."""
        config = FundManagerConfig()
        assert config.rebalance_frequency == "never"
        assert config.rebalance_day == 0
        assert config.rebalance_hour == 0

    def test_from_dict_with_rebalance(self):
        """from_dict should parse rebalance fields."""
        data = {
            "rebalance_frequency": "weekly",
            "rebalance_day": 2,
            "rebalance_hour": 14,
        }
        config = FundManagerConfig.from_dict(data)
        assert config.rebalance_frequency == "weekly"
        assert config.rebalance_day == 2
        assert config.rebalance_hour == 14

    def test_from_yaml_with_rebalance(self):
        """from_yaml should parse rebalance from system section."""
        yaml_dict = {
            "system": {
                "poll_interval": 60,
                "rebalance_frequency": "monthly",
                "rebalance_day": 14,
                "rebalance_hour": 8,
            },
            "strategy": {"type": "fixed_ratio"},
            "bots": [],
        }
        config = FundManagerConfig.from_yaml(yaml_dict)
        assert config.rebalance_frequency == "monthly"
        assert config.rebalance_day == 14
        assert config.rebalance_hour == 8


class TestFundPoolPersistence:
    """Test SQLite persistence for allocations."""

    def test_persist_and_restore(self, tmp_path):
        """Allocations should be restored from DB on new FundPool init."""
        db_file = str(tmp_path / "test_fund.db")

        # First pool: set allocations
        pool1 = FundPool(db_path=db_file)
        pool1.set_allocation("bot_a", Decimal("1000"))
        pool1.set_allocation("bot_b", Decimal("500"))

        # Second pool: should restore
        pool2 = FundPool(db_path=db_file)
        assert pool2.get_allocation("bot_a") == Decimal("1000")
        assert pool2.get_allocation("bot_b") == Decimal("500")

    def test_remove_allocation_persisted(self, tmp_path):
        """Removed allocations should not appear after restore."""
        db_file = str(tmp_path / "test_fund.db")

        pool1 = FundPool(db_path=db_file)
        pool1.set_allocation("bot_a", Decimal("1000"))
        pool1.remove_allocation("bot_a")

        pool2 = FundPool(db_path=db_file)
        assert pool2.get_allocation("bot_a") == Decimal("0")

    def test_clear_allocations_persisted(self, tmp_path):
        """clear_allocations should wipe DB entries."""
        db_file = str(tmp_path / "test_fund.db")

        pool1 = FundPool(db_path=db_file)
        pool1.set_allocation("bot_a", Decimal("1000"))
        pool1.set_allocation("bot_b", Decimal("500"))
        pool1.clear_allocations()

        pool2 = FundPool(db_path=db_file)
        assert pool2.allocated_balance == Decimal("0")

    def test_no_db_path_no_error(self):
        """FundPool without db_path should work without persistence."""
        pool = FundPool()
        pool.set_allocation("bot_a", Decimal("100"))
        assert pool.get_allocation("bot_a") == Decimal("100")

    def test_add_allocation_persisted(self, tmp_path):
        """add_allocation increments should persist correctly."""
        db_file = str(tmp_path / "test_fund.db")

        pool1 = FundPool(db_path=db_file)
        pool1.set_allocation("bot_a", Decimal("100"))
        pool1.add_allocation("bot_a", Decimal("50"))

        pool2 = FundPool(db_path=db_file)
        assert pool2.get_allocation("bot_a") == Decimal("150")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
