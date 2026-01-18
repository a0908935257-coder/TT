"""
Unit tests for dynamic grid adjustment functionality.

Tests the dynamic adjustment features in GridRiskManager:
- Breakout detection (4% threshold)
- Cooldown mechanism (7 days, max 3 rebuilds)
- Rebuild execution
- Record cleanup

Per Prompt 24 specification.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import MarketType

from src.strategy.grid import (
    ATRConfig,
    ATRData,
    DynamicAdjustConfig,
    GridConfig,
    GridLevel,
    GridSetup,
    GridType,
    LevelSide,
    LevelState,
    RiskLevel,
)
from src.strategy.grid.risk_manager import (
    BreakoutAction,
    GridRiskManager,
    RebuildRecord,
    RiskConfig,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_order_manager(mock_exchange: MockExchangeClient, mock_data_manager: MockDataManager, mock_notifier: MockNotifier):
    """Create a mock order manager with a grid setup."""
    from src.strategy.grid import GridOrderManager

    order_manager = GridOrderManager(
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
        bot_id="test_bot",
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
    )

    # Create a mock grid setup
    atr_data = ATRData(
        value=Decimal("1000"),
        period=14,
        timeframe="4h",
        multiplier=Decimal("2.0"),
        current_price=Decimal("50000"),
        upper_price=Decimal("55000"),
        lower_price=Decimal("45000"),
    )

    grid_config = GridConfig(
        symbol="BTCUSDT",
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MODERATE,
        grid_type=GridType.GEOMETRIC,
        manual_upper_price=Decimal("55000"),
        manual_lower_price=Decimal("45000"),
        manual_grid_count=10,
    )

    levels = []
    for i in range(11):
        price = Decimal("45000") + Decimal(i) * Decimal("1000")
        side = LevelSide.BUY if price < Decimal("50000") else LevelSide.SELL
        level = GridLevel(
            index=i,
            price=price,
            side=side,
            state=LevelState.EMPTY,
            allocated_amount=Decimal("1000"),
        )
        levels.append(level)

    setup = GridSetup(
        config=grid_config,
        atr_data=atr_data,
        upper_price=Decimal("55000"),
        lower_price=Decimal("45000"),
        current_price=Decimal("50000"),
        grid_count=10,
        grid_spacing_percent=Decimal("2.0"),
        amount_per_grid=Decimal("1000"),
        levels=levels,
        expected_profit_per_trade=Decimal("1.8"),
    )

    order_manager.initialize(setup)
    return order_manager


@pytest.fixture
def risk_manager_with_dynamic(mock_order_manager, mock_notifier: MockNotifier, adjustment_config: DynamicAdjustConfig):
    """Create a risk manager with dynamic adjustment enabled."""
    config = RiskConfig(
        auto_rebuild_enabled=True,
        breakout_threshold=Decimal("4.0"),  # 4%
        cooldown_days=7,
        max_rebuilds_in_period=3,
    )

    manager = GridRiskManager(
        order_manager=mock_order_manager,
        notifier=mock_notifier,
        config=config,
    )

    manager.set_dynamic_adjust_config(adjustment_config)

    return manager


# =============================================================================
# Test: Breakout Detection
# =============================================================================


class TestBreakoutDetection:
    """Tests for dynamic adjustment breakout detection."""

    def test_upper_breakout_detection(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test upper breakout detection with 4% threshold.

        Initial: upper = 55000
        Trigger: price > 55000 × 1.04 = 57200
        """
        manager = risk_manager_with_dynamic

        # Price above trigger threshold (55000 * 1.04 = 57200)
        # Using 57201 which is > 57200
        trigger_price = Decimal("57201")

        result = manager.check_dynamic_adjust_trigger(trigger_price)
        assert result == "upper"

    def test_upper_breakout_at_threshold(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test that price at exact threshold doesn't trigger (strict greater-than).

        Price 57200 = 55000 × 1.04, should NOT trigger (need > not >=).
        """
        manager = risk_manager_with_dynamic

        # At threshold (not above)
        price = Decimal("57200")

        result = manager.check_dynamic_adjust_trigger(price)
        assert result is None

    def test_lower_breakout_detection(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test lower breakout detection with 4% threshold.

        Initial: lower = 45000
        Trigger: price < 45000 × 0.96 = 43200
        """
        manager = risk_manager_with_dynamic

        # Price below trigger threshold (45000 * 0.96 = 43200)
        # Using 43199 which is < 43200
        trigger_price = Decimal("43199")

        result = manager.check_dynamic_adjust_trigger(trigger_price)
        assert result == "lower"

    def test_lower_breakout_at_threshold(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test that price at exact threshold doesn't trigger (strict less-than).

        Price 43200 = 45000 × 0.96, should NOT trigger (need < not <=).
        """
        manager = risk_manager_with_dynamic

        # At threshold (not below)
        price = Decimal("43200")

        result = manager.check_dynamic_adjust_trigger(price)
        assert result is None

    def test_no_breakout_within_range(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test that prices within grid range don't trigger.
        """
        manager = risk_manager_with_dynamic

        # Test various prices within range
        test_prices = [
            Decimal("45000"),  # At lower bound
            Decimal("50000"),  # Center
            Decimal("55000"),  # At upper bound
            Decimal("47500"),  # Random middle
            Decimal("52500"),  # Random middle
        ]

        for price in test_prices:
            result = manager.check_dynamic_adjust_trigger(price)
            assert result is None, f"Price {price} should not trigger breakout"

    def test_breakout_disabled_returns_none(self, mock_order_manager, mock_notifier: MockNotifier):
        """Test that disabled dynamic adjustment returns None."""
        config = RiskConfig(
            auto_rebuild_enabled=False,  # Disabled
        )

        manager = GridRiskManager(
            order_manager=mock_order_manager,
            notifier=mock_notifier,
            config=config,
        )

        # Even with extreme breakout price, should return None
        result = manager.check_dynamic_adjust_trigger(Decimal("100000"))
        assert result is None


# =============================================================================
# Test: Rebuild Execution
# =============================================================================


class TestRebuildExecution:
    """Tests for grid rebuild execution."""

    @pytest.mark.asyncio
    async def test_rebuild_on_breakout(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that rebuild is executed on breakout."""
        manager = risk_manager_with_dynamic

        # Set up a mock callback
        rebuild_called = []

        async def mock_rebuild(new_center: Decimal):
            rebuild_called.append(new_center)

        manager.set_rebuild_callback(mock_rebuild)

        # Execute rebuild
        result = await manager.execute_dynamic_adjust(
            current_price=Decimal("57500"),
            trigger_direction="upper",
        )

        assert result is True
        assert len(rebuild_called) == 1
        assert rebuild_called[0] == Decimal("57500")

    @pytest.mark.asyncio
    async def test_new_center_price_after_rebuild(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that rebuild uses the new trigger price as center."""
        manager = risk_manager_with_dynamic

        # Track the new center price
        captured_center = []

        async def mock_rebuild(new_center: Decimal):
            captured_center.append(new_center)

        manager.set_rebuild_callback(mock_rebuild)

        # Trigger upper breakout at 57500
        await manager.execute_dynamic_adjust(
            current_price=Decimal("57500"),
            trigger_direction="upper",
        )

        assert captured_center[0] == Decimal("57500")

        # Trigger lower breakout at 42000
        await manager.execute_dynamic_adjust(
            current_price=Decimal("42000"),
            trigger_direction="lower",
        )

        assert captured_center[1] == Decimal("42000")

    @pytest.mark.asyncio
    async def test_rebuild_records_event(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that rebuild creates a RebuildRecord."""
        manager = risk_manager_with_dynamic

        async def mock_rebuild(new_center: Decimal):
            pass

        manager.set_rebuild_callback(mock_rebuild)

        # Execute rebuild
        await manager.execute_dynamic_adjust(
            current_price=Decimal("57500"),
            trigger_direction="upper",
        )

        # Check rebuild history
        history = manager.rebuild_history
        assert len(history) == 1

        record = history[0]
        assert record.trigger_price == Decimal("57500")
        assert record.reason == "upper_breakout"
        assert record.old_upper == Decimal("55000")
        assert record.old_lower == Decimal("45000")


# =============================================================================
# Test: Cooldown Mechanism
# =============================================================================


class TestCooldownMechanism:
    """Tests for cooldown mechanism (7 days, max 3 rebuilds)."""

    @pytest.mark.asyncio
    async def test_cooldown_after_rebuild(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that rebuild is counted in cooldown period."""
        manager = risk_manager_with_dynamic

        async def mock_rebuild(new_center: Decimal):
            pass

        manager.set_rebuild_callback(mock_rebuild)

        # Initial count
        assert manager.rebuilds_in_cooldown_period == 0

        # Execute rebuild
        await manager.execute_dynamic_adjust(
            current_price=Decimal("57500"),
            trigger_direction="upper",
        )

        # Count should increase
        assert manager.rebuilds_in_cooldown_period == 1

    @pytest.mark.asyncio
    async def test_max_rebuilds_in_period(self, risk_manager_with_dynamic: GridRiskManager):
        """Test max rebuilds within cooldown period (3 rebuilds max)."""
        manager = risk_manager_with_dynamic

        async def mock_rebuild(new_center: Decimal):
            pass

        manager.set_rebuild_callback(mock_rebuild)

        # Execute 3 rebuilds
        for i in range(3):
            result = await manager.execute_dynamic_adjust(
                current_price=Decimal("57500") + Decimal(i * 1000),
                trigger_direction="upper",
            )
            assert result is True, f"Rebuild {i + 1} should succeed"

        assert manager.rebuilds_in_cooldown_period == 3
        assert manager.can_rebuild is False

    @pytest.mark.asyncio
    async def test_cooldown_blocks_rebuild(self, risk_manager_with_dynamic: GridRiskManager, mock_notifier: MockNotifier):
        """Test that 4th rebuild is blocked when at max."""
        manager = risk_manager_with_dynamic

        async def mock_rebuild(new_center: Decimal):
            pass

        manager.set_rebuild_callback(mock_rebuild)

        # Execute 3 rebuilds (max allowed)
        for i in range(3):
            await manager.execute_dynamic_adjust(
                current_price=Decimal("57500") + Decimal(i * 1000),
                trigger_direction="upper",
            )

        # 4th rebuild should be blocked
        result = await manager.execute_dynamic_adjust(
            current_price=Decimal("70000"),
            trigger_direction="upper",
        )

        assert result is False
        assert manager.rebuilds_in_cooldown_period == 3

        # Should have notification about blocked rebuild
        assert mock_notifier.has_message_containing("Rebuild Blocked") or \
               mock_notifier.has_message_containing("Cooldown")

    def test_can_rebuild_property(self, risk_manager_with_dynamic: GridRiskManager):
        """Test can_rebuild property reflects current state."""
        manager = risk_manager_with_dynamic

        # Initially can rebuild
        assert manager.can_rebuild is True

        # Manually add rebuild records to simulate max reached
        for i in range(3):
            record = RebuildRecord(
                timestamp=datetime.now(timezone.utc),
                reason="test",
                old_upper=Decimal("55000"),
                old_lower=Decimal("45000"),
                new_upper=Decimal("60000"),
                new_lower=Decimal("50000"),
                trigger_price=Decimal("57500"),
            )
            manager._rebuild_history.append(record)

        # Now cannot rebuild
        assert manager.can_rebuild is False

    def test_next_rebuild_available(self, risk_manager_with_dynamic: GridRiskManager):
        """Test next_rebuild_available returns correct datetime."""
        manager = risk_manager_with_dynamic

        # Initially available now (None means available)
        assert manager.next_rebuild_available is None

        # Add 3 rebuilds to max out
        now = datetime.now(timezone.utc)
        for i in range(3):
            record = RebuildRecord(
                timestamp=now - timedelta(days=i),  # D0, D1, D2 ago
                reason="test",
                old_upper=Decimal("55000"),
                old_lower=Decimal("45000"),
                new_upper=Decimal("60000"),
                new_lower=Decimal("50000"),
                trigger_price=Decimal("57500"),
            )
            manager._rebuild_history.append(record)

        # Oldest record is 2 days ago, so next available is 5 days from now
        # (7 day cooldown - 2 days elapsed = 5 days remaining)
        next_available = manager.next_rebuild_available
        assert next_available is not None

        # The oldest rebuild expires in 5 days
        expected = now - timedelta(days=2) + timedelta(days=7)
        assert abs((next_available - expected).total_seconds()) < 60  # Allow 1 minute tolerance


# =============================================================================
# Test: Cooldown Expiration
# =============================================================================


class TestCooldownExpiration:
    """Tests for cooldown expiration and record cleanup."""

    def test_expired_records_cleanup(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that expired rebuild records are cleaned up."""
        manager = risk_manager_with_dynamic

        now = datetime.now(timezone.utc)

        # Add an old record (8 days ago, expired with 7-day cooldown)
        old_record = RebuildRecord(
            timestamp=now - timedelta(days=8),
            reason="old",
            old_upper=Decimal("55000"),
            old_lower=Decimal("45000"),
            new_upper=Decimal("60000"),
            new_lower=Decimal("50000"),
            trigger_price=Decimal("57500"),
        )
        manager._rebuild_history.append(old_record)

        # Add a recent record (2 days ago, still valid)
        recent_record = RebuildRecord(
            timestamp=now - timedelta(days=2),
            reason="recent",
            old_upper=Decimal("55000"),
            old_lower=Decimal("45000"),
            new_upper=Decimal("60000"),
            new_lower=Decimal("50000"),
            trigger_price=Decimal("57500"),
        )
        manager._rebuild_history.append(recent_record)

        # Clean expired records
        removed = manager._clean_expired_rebuilds()

        assert removed == 1
        assert len(manager._rebuild_history) == 1
        assert manager._rebuild_history[0].reason == "recent"

    def test_cooldown_release_after_expiry(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Test scenario:
        Day 1: Rebuild #1
        Day 2: Rebuild #2
        Day 3: Rebuild #3
        Day 4: Blocked (3 in 7 days)
        Day 8: D1 expires -> Can rebuild
        """
        manager = risk_manager_with_dynamic

        now = datetime.now(timezone.utc)

        # Simulate 3 rebuilds on days 1, 2, 3 (6, 5, 4 days ago)
        for days_ago in [6, 5, 4]:
            record = RebuildRecord(
                timestamp=now - timedelta(days=days_ago),
                reason=f"day_{7 - days_ago}",
                old_upper=Decimal("55000"),
                old_lower=Decimal("45000"),
                new_upper=Decimal("60000"),
                new_lower=Decimal("50000"),
                trigger_price=Decimal("57500"),
            )
            manager._rebuild_history.append(record)

        # All 3 are within 7 days, cannot rebuild
        assert manager.can_rebuild is False
        assert manager.rebuilds_in_cooldown_period == 3

        # Simulate day 8 - oldest record (6 days ago) is still within 7 days
        # Let's move oldest to 8 days ago (expired)
        manager._rebuild_history[0].timestamp = now - timedelta(days=8)

        # Clean expired
        manager._clean_expired_rebuilds()

        # Now only 2 records, can rebuild
        assert manager.rebuilds_in_cooldown_period == 2
        assert manager.can_rebuild is True

    def test_get_recent_rebuilds_filters_correctly(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that _get_recent_rebuilds returns only non-expired records."""
        manager = risk_manager_with_dynamic

        now = datetime.now(timezone.utc)

        # Add records at various ages
        ages = [1, 3, 6, 8, 10]  # days ago
        for days_ago in ages:
            record = RebuildRecord(
                timestamp=now - timedelta(days=days_ago),
                reason=f"days_ago_{days_ago}",
                old_upper=Decimal("55000"),
                old_lower=Decimal("45000"),
                new_upper=Decimal("60000"),
                new_lower=Decimal("50000"),
                trigger_price=Decimal("57500"),
            )
            manager._rebuild_history.append(record)

        # With 7-day cooldown, records 1, 3, 6 days ago are recent
        # Records 8, 10 days ago are expired
        recent = manager._get_recent_rebuilds()

        assert len(recent) == 3
        reasons = [r.reason for r in recent]
        assert "days_ago_1" in reasons
        assert "days_ago_3" in reasons
        assert "days_ago_6" in reasons
        assert "days_ago_8" not in reasons
        assert "days_ago_10" not in reasons


# =============================================================================
# Test: Combined Check and Execute
# =============================================================================


class TestCheckAndExecute:
    """Tests for combined check_and_execute_dynamic_adjust."""

    @pytest.mark.asyncio
    async def test_check_and_execute_triggers_on_breakout(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that check_and_execute triggers rebuild on breakout."""
        manager = risk_manager_with_dynamic

        rebuild_called = []

        async def mock_rebuild(new_center: Decimal):
            rebuild_called.append(new_center)

        manager.set_rebuild_callback(mock_rebuild)

        # Price that triggers upper breakout (> 55000 * 1.04 = 57200)
        result = await manager.check_and_execute_dynamic_adjust(Decimal("58000"))

        assert result is True
        assert len(rebuild_called) == 1

    @pytest.mark.asyncio
    async def test_check_and_execute_no_trigger_in_range(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that check_and_execute returns False when in range."""
        manager = risk_manager_with_dynamic

        rebuild_called = []

        async def mock_rebuild(new_center: Decimal):
            rebuild_called.append(new_center)

        manager.set_rebuild_callback(mock_rebuild)

        # Price within range
        result = await manager.check_and_execute_dynamic_adjust(Decimal("50000"))

        assert result is False
        assert len(rebuild_called) == 0


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases in dynamic adjustment."""

    def test_no_setup_returns_none(self, mock_notifier: MockNotifier):
        """Test that check returns None when no setup exists."""
        from src.strategy.grid import GridOrderManager

        # Create order manager without setup
        order_manager = MagicMock()
        order_manager.setup = None

        config = RiskConfig(auto_rebuild_enabled=True)
        manager = GridRiskManager(
            order_manager=order_manager,
            notifier=mock_notifier,
            config=config,
        )

        result = manager.check_dynamic_adjust_trigger(Decimal("50000"))
        assert result is None

    @pytest.mark.asyncio
    async def test_no_callback_logs_warning(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that execute without callback logs warning but succeeds."""
        manager = risk_manager_with_dynamic

        # Don't set callback

        result = await manager.execute_dynamic_adjust(
            current_price=Decimal("57500"),
            trigger_direction="upper",
        )

        # Should still succeed but log warning
        assert result is True

    def test_rebuild_history_is_copy(self, risk_manager_with_dynamic: GridRiskManager):
        """Test that rebuild_history property returns a copy."""
        manager = risk_manager_with_dynamic

        # Add a record
        record = RebuildRecord(
            timestamp=datetime.now(timezone.utc),
            reason="test",
            old_upper=Decimal("55000"),
            old_lower=Decimal("45000"),
            new_upper=Decimal("60000"),
            new_lower=Decimal("50000"),
            trigger_price=Decimal("57500"),
        )
        manager._rebuild_history.append(record)

        # Get history (should be copy)
        history1 = manager.rebuild_history
        history2 = manager.rebuild_history

        # Modify one shouldn't affect the other or original
        history1.clear()

        assert len(history2) == 1
        assert len(manager._rebuild_history) == 1


# =============================================================================
# Test: Threshold Validation
# =============================================================================


class TestThresholdValidation:
    """Tests for breakout threshold calculation."""

    def test_exact_upper_threshold_calculation(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Verify exact upper threshold: upper × (1 + 0.04) = upper × 1.04
        55000 × 1.04 = 57200

        Uses strict greater-than comparison (>).
        """
        manager = risk_manager_with_dynamic

        # At threshold - should NOT trigger (need > not >=)
        assert manager.check_dynamic_adjust_trigger(Decimal("57200")) is None

        # Slightly above threshold - should trigger
        assert manager.check_dynamic_adjust_trigger(Decimal("57200.01")) == "upper"

        # Slightly below threshold - should NOT trigger
        assert manager.check_dynamic_adjust_trigger(Decimal("57199.99")) is None

    def test_exact_lower_threshold_calculation(self, risk_manager_with_dynamic: GridRiskManager):
        """
        Verify exact lower threshold: lower × (1 - 0.04) = lower × 0.96
        45000 × 0.96 = 43200

        Uses strict less-than comparison (<).
        """
        manager = risk_manager_with_dynamic

        # At threshold - should NOT trigger (need < not <=)
        assert manager.check_dynamic_adjust_trigger(Decimal("43200")) is None

        # Slightly below threshold - should trigger
        assert manager.check_dynamic_adjust_trigger(Decimal("43199.99")) == "lower"

        # Slightly above threshold - should NOT trigger
        assert manager.check_dynamic_adjust_trigger(Decimal("43200.01")) is None
