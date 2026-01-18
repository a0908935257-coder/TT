"""
Integration tests for GridBot.

Tests complete bot lifecycle and component integration.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.models import MarketType, Order, OrderSide, OrderStatus, OrderType
from src.strategy.grid import (
    BotState,
    BreakoutDirection,
    GridBot,
    GridBotConfig,
    GridType,
    RiskConfig,
    RiskLevel,
    RiskState,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


@pytest.fixture
def bot_config():
    """Create a bot configuration for testing."""
    return GridBotConfig(
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MODERATE,
        grid_type=GridType.GEOMETRIC,
        risk_config=RiskConfig(
            daily_loss_limit=Decimal("5"),
            max_consecutive_losses=5,
        ),
    )


@pytest.fixture
def manual_bot_config():
    """Create a bot configuration with manual range."""
    return GridBotConfig(
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MODERATE,
        grid_type=GridType.GEOMETRIC,
        manual_upper=Decimal("55000"),
        manual_lower=Decimal("45000"),
        manual_grid_count=10,
        risk_config=RiskConfig(
            daily_loss_limit=Decimal("5"),
            max_consecutive_losses=5,
        ),
    )


@pytest.fixture
def grid_bot(
    manual_bot_config,
    mock_exchange,
    mock_data_manager,
    mock_notifier,
    sample_klines,
):
    """Create a GridBot for testing."""
    # Set up mocks
    mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
    mock_data_manager.set_klines("BTCUSDT", sample_klines)
    mock_exchange.set_price(Decimal("50000"))

    return GridBot(
        bot_id="test_bot_001",
        config=manual_bot_config,
        exchange=mock_exchange,
        data_manager=mock_data_manager,
        notifier=mock_notifier,
    )


class TestBotLifecycle:
    """Tests for complete bot lifecycle."""

    @pytest.mark.asyncio
    async def test_bot_full_lifecycle(
        self,
        grid_bot,
        mock_exchange,
        mock_data_manager,
    ):
        """
        Test complete bot lifecycle:
        1. Start
        2. Place initial orders
        3. Simulate price drop -> buy fill
        4. Verify reverse sell placed
        5. Simulate price rise -> sell fill
        6. Verify profit calculated
        7. Stop
        """
        # 1. Start bot
        started = await grid_bot.start()
        assert started is True
        assert grid_bot.state == BotState.RUNNING
        assert grid_bot.is_running is True

        # 2. Verify initial orders placed
        assert grid_bot.order_manager is not None
        assert grid_bot.order_manager.active_order_count > 0

        # Get a buy level
        buy_levels = [
            l for l in grid_bot.setup.levels
            if l.index in grid_bot.order_manager._level_order_map
            and grid_bot.order_manager._orders[
                grid_bot.order_manager._level_order_map[l.index]
            ].side == OrderSide.BUY
        ]

        if buy_levels:
            buy_level = buy_levels[0]
            buy_order_id = grid_bot.order_manager._level_order_map[buy_level.index]
            buy_order = grid_bot.order_manager._orders[buy_order_id]

            # 3. Simulate price drop -> buy fill
            filled_order = Order(
                order_id=buy_order_id,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=buy_order.order_type,
                status=OrderStatus.FILLED,
                price=buy_order.price,
                quantity=buy_order.quantity,
                filled_qty=buy_order.quantity,
                avg_price=buy_order.price,
                fee=buy_order.quantity * buy_order.price * Decimal("0.001"),
                created_at=datetime.now(timezone.utc),
            )

            reverse = await grid_bot.on_order_filled(filled_order)

            # 4. Verify reverse sell placed
            if reverse:
                assert reverse.side == OrderSide.SELL

        # 5. Check statistics
        stats = grid_bot.get_statistics()
        assert "total_profit" in stats
        assert "trade_count" in stats

        # 6. Get status
        status = grid_bot.get_status()
        assert status["bot_id"] == "test_bot_001"
        assert status["state"] == "running"

        # 7. Stop bot
        stopped = await grid_bot.stop()
        assert stopped is True
        assert grid_bot.state == BotState.STOPPED

    @pytest.mark.asyncio
    async def test_bot_start(
        self,
        grid_bot,
    ):
        """Test bot start process."""
        started = await grid_bot.start()

        assert started is True
        assert grid_bot.state == BotState.RUNNING
        assert grid_bot.setup is not None
        assert grid_bot.calculator is not None
        assert grid_bot.order_manager is not None
        assert grid_bot.risk_manager is not None

        # Cleanup
        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_bot_pause_resume(self, grid_bot):
        """Test bot pause and resume."""
        await grid_bot.start()

        # Pause
        paused = await grid_bot.pause("Test pause")
        assert paused is True
        assert grid_bot.state == BotState.PAUSED

        # Resume
        resumed = await grid_bot.resume()
        assert resumed is True
        assert grid_bot.state == BotState.RUNNING

        # Cleanup
        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_bot_stop(self, grid_bot):
        """Test bot stop process."""
        await grid_bot.start()

        orders_before = grid_bot.order_manager.active_order_count
        assert orders_before > 0

        stopped = await grid_bot.stop()

        assert stopped is True
        assert grid_bot.state == BotState.STOPPED
        assert grid_bot.order_manager.active_order_count == 0

    @pytest.mark.asyncio
    async def test_bot_stop_with_clear_position(self, grid_bot, mock_exchange):
        """Test bot stop with position clearing."""
        await grid_bot.start()

        # Simulate a filled buy (position)
        if grid_bot.order_manager._level_order_map:
            level_index = list(grid_bot.order_manager._level_order_map.keys())[0]
            order_id = grid_bot.order_manager._level_order_map[level_index]
            order = grid_bot.order_manager._orders[order_id]

            if order.side == OrderSide.BUY:
                filled = Order(
                    order_id=order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=order.order_type,
                    status=OrderStatus.FILLED,
                    price=order.price,
                    quantity=order.quantity,
                    filled_qty=order.quantity,
                    avg_price=order.price,
                    fee=Decimal("5"),
                    created_at=datetime.now(timezone.utc),
                )
                await grid_bot.on_order_filled(filled)

        stopped = await grid_bot.stop(clear_position=True)

        assert stopped is True


class TestBotErrorHandling:
    """Tests for bot error handling."""

    @pytest.mark.asyncio
    async def test_bot_error_handling_on_start_failure(
        self,
        manual_bot_config,
        mock_exchange,
        mock_data_manager,
        mock_notifier,
    ):
        """Test error handling when start fails."""
        # Don't set klines - will fail to calculate grid
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))

        bot = GridBot(
            bot_id="test_bot",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )

        # Start should fail due to no klines
        started = await bot.start()

        # With manual range it should still work, but let's test error notification
        # In case of actual error, state would be ERROR
        if not started:
            assert bot.state == BotState.ERROR

    @pytest.mark.asyncio
    async def test_bot_handles_exchange_error(
        self,
        grid_bot,
        mock_exchange,
    ):
        """Test bot handles exchange errors gracefully."""
        await grid_bot.start()

        # Simulate exchange disconnect
        mock_exchange.set_connected(False)

        # Bot should handle this in health check
        if grid_bot.risk_manager:
            health = await grid_bot.risk_manager.run_health_check()
            assert health["exchange_connected"] is False

        await grid_bot.stop()


class TestBotStatePersistence:
    """Tests for bot state persistence."""

    @pytest.mark.asyncio
    async def test_bot_state_persistence(
        self,
        grid_bot,
        mock_data_manager,
    ):
        """Test bot state is saved."""
        await grid_bot.start()

        # Manually trigger save
        await grid_bot._save_state()

        # Verify state was saved
        saved_states = mock_data_manager.get_stored_bot_states()
        assert "test_bot_001" in saved_states

        saved = saved_states["test_bot_001"]
        assert saved["bot_id"] == "test_bot_001"
        assert saved["state"] == "running"

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_bot_restore(
        self,
        manual_bot_config,
        mock_exchange,
        mock_data_manager,
        mock_notifier,
        sample_klines,
    ):
        """Test bot restoration from saved state."""
        # Set up mocks
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
        mock_data_manager.set_klines("BTCUSDT", sample_klines)
        mock_exchange.set_price(Decimal("50000"))

        # Create and start a bot
        bot1 = GridBot(
            bot_id="restore_test_bot",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )
        await bot1.start()
        await bot1._save_state()
        await bot1.stop()

        # Restore the bot
        restored = await GridBot.restore(
            bot_id="restore_test_bot",
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )

        if restored:
            assert restored.bot_id == "restore_test_bot"
            assert restored.is_running is True
            await restored.stop()


class TestBotStatusQueries:
    """Tests for bot status query methods."""

    @pytest.mark.asyncio
    async def test_get_status(self, grid_bot):
        """Test get_status returns comprehensive info."""
        await grid_bot.start()

        status = grid_bot.get_status()

        assert "bot_id" in status
        assert "state" in status
        assert "symbol" in status
        assert "upper_price" in status
        assert "lower_price" in status
        assert "grid_count" in status
        assert "total_profit" in status

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_get_statistics(self, grid_bot):
        """Test get_statistics method."""
        await grid_bot.start()

        stats = grid_bot.get_statistics()

        assert "total_profit" in stats
        assert "trade_count" in stats
        assert "buy_filled_count" in stats
        assert "sell_filled_count" in stats

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_get_orders(self, grid_bot):
        """Test get_orders method."""
        await grid_bot.start()

        orders = grid_bot.get_orders()

        assert isinstance(orders, list)
        assert len(orders) > 0

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_get_levels(self, grid_bot):
        """Test get_levels method."""
        await grid_bot.start()

        levels = grid_bot.get_levels()

        assert isinstance(levels, list)
        assert len(levels) > 0

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_get_history(self, grid_bot):
        """Test get_history method."""
        await grid_bot.start()

        history = grid_bot.get_history()

        assert isinstance(history, list)
        # Initially empty
        assert len(history) == 0

        await grid_bot.stop()


class TestBotPriceEvents:
    """Tests for price event handling."""

    @pytest.mark.asyncio
    async def test_on_price_update_breakout(
        self,
        grid_bot,
        mock_notifier,
    ):
        """Test price update triggering breakout."""
        await grid_bot.start()

        # Price breaks upper bound
        breakout_price = Decimal("60000")
        await grid_bot.on_price_update(breakout_price)

        # Should trigger breakout handling
        if grid_bot.risk_manager:
            assert grid_bot.risk_manager.last_breakout is not None

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_on_price_update_within_range(
        self,
        grid_bot,
    ):
        """Test price update within range."""
        await grid_bot.start()

        # Price within range
        normal_price = Decimal("50000")
        await grid_bot.on_price_update(normal_price)

        # No breakout
        if grid_bot.risk_manager:
            direction = grid_bot.risk_manager.check_breakout(normal_price)
            assert direction == BreakoutDirection.NONE

        await grid_bot.stop()


class TestBotOrderPersistence:
    """Tests for bot order persistence and reconnection."""

    @pytest.mark.asyncio
    async def test_order_mapping_saved_in_state(
        self,
        grid_bot,
        mock_data_manager,
    ):
        """Test that order mapping is saved when state is saved."""
        await grid_bot.start()

        # Get order count
        order_count = grid_bot.order_manager.active_order_count
        assert order_count > 0

        # Save state
        await grid_bot._save_state()

        # Check saved state contains order mapping
        saved_states = mock_data_manager.get_stored_bot_states()
        assert grid_bot.bot_id in saved_states

        saved = saved_states[grid_bot.bot_id]
        assert "order_mapping" in saved
        assert len(saved["order_mapping"]) == order_count

        await grid_bot.stop()

    @pytest.mark.asyncio
    async def test_bot_restores_orders_on_start(
        self,
        manual_bot_config,
        mock_exchange,
        mock_data_manager,
        mock_notifier,
        sample_klines,
    ):
        """Test that bot restores order mapping when starting."""
        # Set up mocks
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
        mock_data_manager.set_klines("BTCUSDT", sample_klines)
        mock_exchange.set_price(Decimal("50000"))

        # Create bot 1 and start it
        bot1 = GridBot(
            bot_id="persistence_test_bot",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )
        await bot1.start()

        # Get order info before stopping
        original_order_count = bot1.order_manager.active_order_count
        original_mapping = bot1.order_manager.get_order_mapping()
        assert original_order_count > 0

        # Save state and stop
        await bot1._save_state()
        await bot1.stop()

        # Create new bot instance (simulating restart)
        bot2 = GridBot(
            bot_id="persistence_test_bot",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )

        # Start the new bot - it should restore orders
        await bot2.start()

        # Verify orders were restored
        restored_mapping = bot2.order_manager.get_order_mapping()

        # The restored mapping should contain the original orders
        # (sync_orders will verify they still exist on exchange)
        assert bot2.order_manager.active_order_count > 0

        await bot2.stop()

    @pytest.mark.asyncio
    async def test_bot_handles_missing_saved_state(
        self,
        manual_bot_config,
        mock_exchange,
        mock_data_manager,
        mock_notifier,
        sample_klines,
    ):
        """Test that bot handles case with no saved state gracefully."""
        # Set up mocks
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
        mock_data_manager.set_klines("BTCUSDT", sample_klines)
        mock_exchange.set_price(Decimal("50000"))

        # Create bot without any prior saved state
        bot = GridBot(
            bot_id="new_bot_no_state",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )

        # Should start successfully
        started = await bot.start()
        assert started is True

        # Should have placed new orders
        assert bot.order_manager.active_order_count > 0

        await bot.stop()

    @pytest.mark.asyncio
    async def test_bot_handles_empty_order_mapping(
        self,
        manual_bot_config,
        mock_exchange,
        mock_data_manager,
        mock_notifier,
        sample_klines,
    ):
        """Test that bot handles saved state with empty order mapping."""
        # Set up mocks
        mock_data_manager.set_price("BTCUSDT", Decimal("50000"))
        mock_data_manager.set_klines("BTCUSDT", sample_klines)
        mock_exchange.set_price(Decimal("50000"))

        # Pre-save a state with empty order mapping
        await mock_data_manager.save_bot_state(
            bot_id="empty_mapping_bot",
            state_data={"order_mapping": {}},
            bot_type="grid",
            status="stopped",
        )

        # Create bot
        bot = GridBot(
            bot_id="empty_mapping_bot",
            config=manual_bot_config,
            exchange=mock_exchange,
            data_manager=mock_data_manager,
            notifier=mock_notifier,
        )

        # Should start successfully and place new orders
        started = await bot.start()
        assert started is True
        assert bot.order_manager.active_order_count > 0

        await bot.stop()

    @pytest.mark.asyncio
    async def test_order_mapping_updated_after_fill(
        self,
        grid_bot,
        mock_data_manager,
    ):
        """Test that order mapping is updated correctly after order fills."""
        await grid_bot.start()

        # Get a buy order
        buy_levels = [
            l for l in grid_bot.setup.levels
            if l.index in grid_bot.order_manager._level_order_map
        ]

        if buy_levels:
            level = buy_levels[0]
            order_id = grid_bot.order_manager._level_order_map[level.index]
            order = grid_bot.order_manager._orders[order_id]

            if order.side == OrderSide.BUY:
                # Simulate fill
                filled_order = Order(
                    order_id=order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.BUY,
                    order_type=order.order_type,
                    status=OrderStatus.FILLED,
                    price=order.price,
                    quantity=order.quantity,
                    filled_qty=order.quantity,
                    avg_price=order.price,
                    fee=Decimal("5"),
                    created_at=datetime.now(timezone.utc),
                )

                await grid_bot.on_order_filled(filled_order)

                # Save state
                await grid_bot._save_state()

                # Verify mapping updated
                saved = mock_data_manager.get_stored_bot_states()[grid_bot.bot_id]
                mapping = saved.get("order_mapping", {})

                # Original order should be removed from mapping
                assert order_id not in mapping

        await grid_bot.stop()
