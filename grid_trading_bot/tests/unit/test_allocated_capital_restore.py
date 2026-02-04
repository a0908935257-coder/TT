"""
Test allocated_capital persistence in _save_state() and restore() methods.

Verifies that allocated_capital is correctly saved and restored for all bot types.
"""

import pytest
from decimal import Decimal
from datetime import datetime, timezone


class TestAllocatedCapitalRestore:
    """Test allocated_capital save/restore for all bot types."""

    @pytest.mark.asyncio
    async def test_bollinger_save_state_includes_allocated_capital(self):
        """Test Bollinger bot _save_state includes allocated_capital in config."""
        from src.bots.bollinger.bot import BollingerBot, BollingerConfig
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        config = BollingerConfig(
            symbol="BTCUSDT",
            timeframe="15m",
            bb_period=20,
            bb_std=Decimal("2.0"),
            grid_count=10,
            leverage=2,
            allocated_capital=Decimal("5000"),
            max_capital=Decimal("10000"),
        )

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        bot = BollingerBot(
            bot_id="test_bollinger",
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        # Manually trigger save
        await bot._save_state()

        # Check saved state
        saved = await data_manager.load_bot_state("test_bollinger")
        assert saved is not None
        assert "config" in saved
        config_data = saved["config"]
        assert config_data.get("allocated_capital") == "5000"
        assert config_data.get("max_capital") == "10000"

    @pytest.mark.asyncio
    async def test_bollinger_restore_allocated_capital(self):
        """Test Bollinger bot restore() correctly restores allocated_capital."""
        from src.bots.bollinger.bot import BollingerBot
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        # Pre-populate mock with saved state including allocated_capital
        data_manager._bot_states["test_bollinger_restore"] = {
            "bot_id": "test_bollinger_restore",
            "bot_type": "bollinger",
            "state": "stopped",
            "config": {
                "symbol": "ETHUSDT",
                "timeframe": "15m",
                "bb_period": 20,
                "bb_std": "2.0",
                "grid_count": 10,
                "leverage": 2,
                "allocated_capital": "7500",
                "max_capital": "15000",
            },
            "capital": "7500",
            "initial_capital": "7500",
            "current_trend": 0,
            "signal_cooldown": 0,
            "current_bar": 0,
            "entry_bar": 0,
            "stats": {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        # Set required mock data
        data_manager.set_price("ETHUSDT", Decimal("3000"))
        exchange.set_price(Decimal("3000"))

        # Restore
        restored = await BollingerBot.restore(
            bot_id="test_bollinger_restore",
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        assert restored is not None
        assert restored._config.allocated_capital == Decimal("7500")
        assert restored._config.max_capital == Decimal("15000")

    @pytest.mark.asyncio
    async def test_grid_futures_restore_allocated_capital(self):
        """Test Grid Futures bot restore() correctly restores allocated_capital."""
        from src.bots.grid_futures.bot import GridFuturesBot
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        # Pre-populate mock with saved state
        data_manager._bot_states["test_grid_futures_restore"] = {
            "bot_id": "test_grid_futures_restore",
            "bot_type": "grid_futures",
            "state": "stopped",
            "config": {
                "symbol": "BTCUSDT",
                "grid_count": 10,
                "leverage": 5,
                "direction": "neutral",
                "trend_period": 20,
                "atr_multiplier": "3.0",
                "allocated_capital": "8000",
                "max_capital": "20000",
            },
            "capital": "8000",
            "initial_capital": "8000",
            "current_trend": 0,
            "signal_cooldown": 0,
            "current_bar": 0,
            "entry_bar": 0,
            "stats": {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        data_manager.set_price("BTCUSDT", Decimal("50000"))
        exchange.set_price(Decimal("50000"))

        restored = await GridFuturesBot.restore(
            bot_id="test_grid_futures_restore",
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        assert restored is not None
        assert restored._config.allocated_capital == Decimal("8000")
        assert restored._config.max_capital == Decimal("20000")

    @pytest.mark.asyncio
    async def test_rsi_grid_restore_allocated_capital(self):
        """Test RSI Grid bot restore() correctly restores allocated_capital."""
        from src.bots.rsi_grid.bot import RSIGridBot
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        # Pre-populate mock with saved state
        data_manager._bot_states["test_rsi_grid_restore"] = {
            "bot_id": "test_rsi_grid_restore",
            "bot_type": "rsi_grid",
            "state": "stopped",
            "config": {
                "symbol": "ETHUSDT",
                "timeframe": "1h",
                "rsi_period": 7,
                "oversold_level": 33,
                "overbought_level": 66,
                "grid_count": 8,
                "atr_multiplier": "3.0",
                "leverage": 10,
                "allocated_capital": "6000",
                "max_capital": "12000",
            },
            "capital": "6000",
            "initial_capital": "6000",
            "current_trend": 0,
            "daily_pnl": "0",
            "consecutive_losses": 0,
            "risk_paused": False,
            "signal_cooldown": 0,
            "current_bar": 0,
            "stats": {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        data_manager.set_price("ETHUSDT", Decimal("3000"))
        exchange.set_price(Decimal("3000"))

        restored = await RSIGridBot.restore(
            bot_id="test_rsi_grid_restore",
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        assert restored is not None
        assert restored._config.allocated_capital == Decimal("6000")
        assert restored._config.max_capital == Decimal("12000")

    @pytest.mark.asyncio
    async def test_supertrend_restore_allocated_capital(self):
        """Test Supertrend bot restore() correctly restores allocated_capital."""
        from src.bots.supertrend.bot import SupertrendBot
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        # Pre-populate mock with saved state
        data_manager._bot_states["test_supertrend_restore"] = {
            "bot_id": "test_supertrend_restore",
            "bot_type": "supertrend",
            "state": "stopped",
            "config": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "atr_period": 25,
                "atr_multiplier": "3.0",
                "leverage": 2,
                "allocated_capital": "9000",
                "max_capital": "18000",
            },
            "total_pnl": "0",
            "current_bar": 0,
            "entry_bar": 0,
            "prev_trend": 0,
            "current_trend": 0,
            "trend_bars": 0,
            "signal_cooldown": 0,
            "daily_pnl": "0",
            "consecutive_losses": 0,
            "risk_paused": False,
            "stats": {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        data_manager.set_price("BTCUSDT", Decimal("50000"))
        exchange.set_price(Decimal("50000"))

        restored = await SupertrendBot.restore(
            bot_id="test_supertrend_restore",
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        assert restored is not None
        assert restored._config.allocated_capital == Decimal("9000")
        assert restored._config.max_capital == Decimal("18000")

    @pytest.mark.asyncio
    async def test_restore_without_allocated_capital_backward_compatible(self):
        """Test restore works when allocated_capital is not in saved state (backward compatibility)."""
        from src.bots.bollinger.bot import BollingerBot
        from tests.mocks.data_mock import MockDataManager, MockNotifier
        from tests.mocks.exchange_mock import MockExchangeClient as MockExchange

        data_manager = MockDataManager()
        exchange = MockExchange()
        notifier = MockNotifier()

        # Old saved state without allocated_capital
        data_manager._bot_states["test_backward_compat"] = {
            "bot_id": "test_backward_compat",
            "bot_type": "bollinger",
            "state": "stopped",
            "config": {
                "symbol": "BTCUSDT",
                "timeframe": "15m",
                "bb_period": 20,
                "bb_std": "2.0",
                "grid_count": 10,
                "leverage": 2,
                "max_capital": "10000",
                # No allocated_capital - simulating old saved state
            },
            "capital": "5000",
            "initial_capital": "5000",
            "current_trend": 0,
            "signal_cooldown": 0,
            "current_bar": 0,
            "entry_bar": 0,
            "stats": {},
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }

        data_manager.set_price("BTCUSDT", Decimal("50000"))
        exchange.set_price(Decimal("50000"))

        # Should not raise error
        restored = await BollingerBot.restore(
            bot_id="test_backward_compat",
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        assert restored is not None
        assert restored._config.allocated_capital is None  # Should be None for old states
        assert restored._config.max_capital == Decimal("10000")
