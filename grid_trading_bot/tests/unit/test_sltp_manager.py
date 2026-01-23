"""
Unit tests for SLTP Manager.

Tests stop loss, take profit, and trailing stop functionality.
"""

from decimal import Decimal

import pytest

from src.risk.sltp import (
    MockExchangeAdapter,
    SLTPCalculator,
    SLTPConfig,
    SLTPManager,
    SLTPState,
    StopLossConfig,
    StopLossType,
    TakeProfitConfig,
    TakeProfitLevel,
    TakeProfitType,
    TrailingStopConfig,
    TrailingStopType,
)


class TestSLTPCalculator:
    """Tests for SLTPCalculator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = SLTPCalculator()
        self.entry_price = Decimal("50000")
        self.atr = Decimal("500")

    def test_calculate_stop_loss_percentage_long(self):
        """Test percentage-based stop loss for long position."""
        config = StopLossConfig(
            stop_type=StopLossType.PERCENTAGE,
            value=Decimal("0.02"),  # 2%
        )
        sl = self.calculator.calculate_stop_loss(
            config, self.entry_price, is_long=True
        )
        expected = Decimal("50000") * (1 - Decimal("0.02"))
        assert sl == expected

    def test_calculate_stop_loss_percentage_short(self):
        """Test percentage-based stop loss for short position."""
        config = StopLossConfig(
            stop_type=StopLossType.PERCENTAGE,
            value=Decimal("0.02"),
        )
        sl = self.calculator.calculate_stop_loss(
            config, self.entry_price, is_long=False
        )
        expected = Decimal("50000") * (1 + Decimal("0.02"))
        assert sl == expected

    def test_calculate_stop_loss_atr_based(self):
        """Test ATR-based stop loss."""
        config = StopLossConfig(
            stop_type=StopLossType.ATR_BASED,
            atr_multiplier=Decimal("2.0"),
        )
        sl = self.calculator.calculate_stop_loss(
            config, self.entry_price, is_long=True, atr=self.atr
        )
        expected = self.entry_price - (self.atr * Decimal("2.0"))
        assert sl == expected

    def test_calculate_stop_loss_fixed(self):
        """Test fixed price stop loss."""
        config = StopLossConfig(
            stop_type=StopLossType.FIXED,
            fixed_price=Decimal("48000"),
        )
        sl = self.calculator.calculate_stop_loss(config, self.entry_price, is_long=True)
        assert sl == Decimal("48000")

    def test_calculate_stop_loss_disabled(self):
        """Test disabled stop loss returns extreme value."""
        config = StopLossConfig(enabled=False)
        sl = self.calculator.calculate_stop_loss(config, self.entry_price, is_long=True)
        assert sl == Decimal("0")

    def test_calculate_take_profit_percentage(self):
        """Test percentage-based take profit."""
        config = TakeProfitConfig(
            tp_type=TakeProfitType.PERCENTAGE,
            value=Decimal("0.04"),  # 4%
        )
        levels = self.calculator.calculate_take_profit(
            config, self.entry_price, is_long=True
        )
        assert len(levels) == 1
        expected = self.entry_price * (1 + Decimal("0.04"))
        assert levels[0].price == expected
        assert levels[0].percentage == Decimal("1.0")

    def test_calculate_take_profit_risk_reward(self):
        """Test risk/reward-based take profit."""
        config = TakeProfitConfig(
            tp_type=TakeProfitType.RISK_REWARD,
            risk_reward_ratio=Decimal("2.0"),
        )
        stop_loss = Decimal("49000")
        levels = self.calculator.calculate_take_profit(
            config, self.entry_price, is_long=True, stop_loss=stop_loss
        )
        risk = self.entry_price - stop_loss  # 1000
        expected = self.entry_price + (risk * Decimal("2.0"))  # 52000
        assert len(levels) == 1
        assert levels[0].price == expected

    def test_calculate_take_profit_multi_level(self):
        """Test multi-level take profit."""
        config = TakeProfitConfig(
            tp_type=TakeProfitType.MULTI_LEVEL,
            level_percentages=[Decimal("0.02"), Decimal("0.04"), Decimal("0.06")],
            level_close_pcts=[Decimal("0.33"), Decimal("0.33"), Decimal("0.34")],
        )
        levels = self.calculator.calculate_take_profit(
            config, self.entry_price, is_long=True
        )
        assert len(levels) == 3
        assert levels[0].price == self.entry_price * Decimal("1.02")
        assert levels[1].price == self.entry_price * Decimal("1.04")
        assert levels[2].price == self.entry_price * Decimal("1.06")
        assert levels[0].percentage == Decimal("0.33")

    def test_calculate_trailing_stop_percentage(self):
        """Test percentage-based trailing stop."""
        config = TrailingStopConfig(
            trailing_type=TrailingStopType.PERCENTAGE,
            distance=Decimal("0.01"),  # 1%
            activation_pct=Decimal("0.01"),  # Activate at 1% profit
            enabled=True,
        )
        current_price = Decimal("51000")  # 2% up from entry
        highest_price = Decimal("51500")  # Highest seen
        current_stop = Decimal("49000")

        new_stop = self.calculator.calculate_trailing_stop(
            config,
            current_price,
            highest_price,
            Decimal("49000"),  # lowest (not used for long)
            is_long=True,
            entry_price=self.entry_price,
            current_stop=current_stop,
        )

        expected = highest_price * (1 - Decimal("0.01"))  # 51500 * 0.99 = 50985
        assert new_stop == expected

    def test_calculate_trailing_stop_not_activated(self):
        """Test trailing stop not activated when profit is below threshold."""
        config = TrailingStopConfig(
            trailing_type=TrailingStopType.PERCENTAGE,
            distance=Decimal("0.01"),
            activation_pct=Decimal("0.02"),  # Need 2% profit to activate
            enabled=True,
        )
        current_price = Decimal("50250")  # Only 0.5% up
        highest_price = Decimal("50300")
        current_stop = Decimal("49000")

        new_stop = self.calculator.calculate_trailing_stop(
            config,
            current_price,
            highest_price,
            Decimal("49000"),
            is_long=True,
            entry_price=self.entry_price,
            current_stop=current_stop,
        )

        assert new_stop is None  # Not activated

    def test_check_stop_loss_hit_long(self):
        """Test stop loss hit detection for long position."""
        stop_loss = Decimal("49000")

        # Price dropped below stop loss
        assert self.calculator.check_stop_loss_hit(
            stop_loss,
            current_price=Decimal("48500"),
            high=Decimal("50000"),
            low=Decimal("48500"),
            is_long=True,
        )

        # Price above stop loss
        assert not self.calculator.check_stop_loss_hit(
            stop_loss,
            current_price=Decimal("49500"),
            high=Decimal("50000"),
            low=Decimal("49500"),
            is_long=True,
        )

    def test_check_stop_loss_hit_short(self):
        """Test stop loss hit detection for short position."""
        stop_loss = Decimal("51000")

        # Price rose above stop loss
        assert self.calculator.check_stop_loss_hit(
            stop_loss,
            current_price=Decimal("51500"),
            high=Decimal("51500"),
            low=Decimal("50000"),
            is_long=False,
        )

    def test_check_take_profit_hit(self):
        """Test take profit hit detection."""
        levels = [
            TakeProfitLevel(price=Decimal("51000"), percentage=Decimal("0.5")),
            TakeProfitLevel(price=Decimal("52000"), percentage=Decimal("0.5")),
        ]

        # First level hit
        hit_indices = self.calculator.check_take_profit_hit(
            levels,
            current_price=Decimal("51500"),
            high=Decimal("51500"),
            low=Decimal("50500"),
            is_long=True,
        )
        assert hit_indices == [0]

        # Both levels hit
        hit_indices = self.calculator.check_take_profit_hit(
            levels,
            current_price=Decimal("52500"),
            high=Decimal("52500"),
            low=Decimal("50500"),
            is_long=True,
        )
        assert hit_indices == [0, 1]


class TestSLTPState:
    """Tests for SLTPState."""

    def setup_method(self):
        """Set up test fixtures."""
        self.state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
            take_profit_levels=[
                TakeProfitLevel(price=Decimal("51000"), percentage=Decimal("0.5")),
                TakeProfitLevel(price=Decimal("52000"), percentage=Decimal("0.5")),
            ],
        )

    def test_update_price_extremes(self):
        """Test price extreme tracking."""
        self.state.update_price_extremes(Decimal("51000"), Decimal("49500"))
        assert self.state.highest_price == Decimal("51000")
        assert self.state.lowest_price == Decimal("49500")

        # Should only update if new extreme
        self.state.update_price_extremes(Decimal("50500"), Decimal("50000"))
        assert self.state.highest_price == Decimal("51000")  # Unchanged
        assert self.state.lowest_price == Decimal("49500")  # Unchanged

    def test_update_stop_loss_long(self):
        """Test stop loss update for long position."""
        # Should update - new stop is higher
        assert self.state.update_stop_loss(Decimal("49500"))
        assert self.state.current_stop_loss == Decimal("49500")

        # Should not update - new stop is lower
        assert not self.state.update_stop_loss(Decimal("49000"))
        assert self.state.current_stop_loss == Decimal("49500")

    def test_mark_tp_triggered(self):
        """Test marking take profit as triggered."""
        close_pct = self.state.mark_tp_triggered(0)
        assert close_pct == Decimal("0.5")
        assert self.state.take_profit_levels[0].triggered
        assert self.state.closed_quantity == Decimal("0.05")
        assert not self.state.all_tp_triggered

        # Trigger second level
        close_pct = self.state.mark_tp_triggered(1)
        assert close_pct == Decimal("0.5")
        assert self.state.all_tp_triggered

    def test_remaining_quantity(self):
        """Test remaining quantity calculation."""
        assert self.state.remaining_quantity == Decimal("0.1")
        self.state.mark_tp_triggered(0)
        assert self.state.remaining_quantity == Decimal("0.05")

    def test_is_active(self):
        """Test active state checking."""
        assert self.state.is_active

        # Mark SL triggered
        self.state.mark_sl_triggered()
        assert not self.state.is_active


class TestSLTPManager:
    """Tests for SLTPManager."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = SLTPManager()  # No exchange adapter = backtest mode
        self.config = SLTPConfig(
            stop_loss=StopLossConfig(
                stop_type=StopLossType.PERCENTAGE,
                value=Decimal("0.02"),
            ),
            take_profit=TakeProfitConfig(
                tp_type=TakeProfitType.PERCENTAGE,
                value=Decimal("0.04"),
            ),
            trailing_stop=TrailingStopConfig(
                trailing_type=TrailingStopType.PERCENTAGE,
                distance=Decimal("0.01"),
                activation_pct=Decimal("0.02"),
                enabled=True,
            ),
        )

    @pytest.mark.asyncio
    async def test_initialize_sltp(self):
        """Test SLTP initialization."""
        state = await self.manager.initialize_sltp(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            config=self.config,
        )

        assert state.symbol == "BTCUSDT"
        assert state.entry_price == Decimal("50000")
        assert state.is_long
        assert state.quantity == Decimal("0.1")
        assert state.initial_stop_loss == Decimal("49000")  # 50000 * 0.98
        assert len(state.take_profit_levels) == 1
        assert state.take_profit_levels[0].price == Decimal("52000")  # 50000 * 1.04

    def test_check_stop_loss(self):
        """Test stop loss checking."""
        # Setup state directly
        state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
        )
        self.manager._states["BTCUSDT"] = state

        # Price above SL
        hit = self.manager.check_stop_loss(
            "BTCUSDT",
            current_price=Decimal("50500"),
            high=Decimal("51000"),
            low=Decimal("50000"),
        )
        assert not hit

        # Price hits SL
        hit = self.manager.check_stop_loss(
            "BTCUSDT",
            current_price=Decimal("48500"),
            high=Decimal("50000"),
            low=Decimal("48500"),
        )
        assert hit
        assert state.stop_loss_triggered

    def test_check_take_profit(self):
        """Test take profit checking."""
        state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
            take_profit_levels=[
                TakeProfitLevel(price=Decimal("51000"), percentage=Decimal("0.5")),
                TakeProfitLevel(price=Decimal("52000"), percentage=Decimal("0.5")),
            ],
        )
        self.manager._states["BTCUSDT"] = state

        # First TP hit
        hit_indices = self.manager.check_take_profit(
            "BTCUSDT",
            current_price=Decimal("51500"),
            high=Decimal("51500"),
            low=Decimal("50500"),
        )
        assert hit_indices == [0]
        assert state.take_profit_levels[0].triggered

    def test_process_price_update(self):
        """Test comprehensive price update processing."""
        state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
            take_profit_levels=[
                TakeProfitLevel(price=Decimal("51000"), percentage=Decimal("1.0")),
            ],
        )
        self.manager._states["BTCUSDT"] = state

        # Normal price movement - nothing triggers
        result = self.manager.process_price_update(
            "BTCUSDT",
            current_price=Decimal("50500"),
            high=Decimal("50500"),
            low=Decimal("50000"),
            config=self.config,
        )
        assert not result["sl_hit"]
        assert result["tp_levels_hit"] == []

        # Price hits TP
        result = self.manager.process_price_update(
            "BTCUSDT",
            current_price=Decimal("51500"),
            high=Decimal("51500"),
            low=Decimal("50000"),
            config=self.config,
        )
        assert not result["sl_hit"]
        assert result["tp_levels_hit"] == [0]

    def test_get_state(self):
        """Test getting SLTP state."""
        assert self.manager.get_state("BTCUSDT") is None

        state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
        )
        self.manager._states["BTCUSDT"] = state

        assert self.manager.get_state("BTCUSDT") is state

    def test_remove_state(self):
        """Test removing SLTP state."""
        state = SLTPState(
            symbol="BTCUSDT",
            entry_price=Decimal("50000"),
            is_long=True,
            quantity=Decimal("0.1"),
            initial_stop_loss=Decimal("49000"),
            current_stop_loss=Decimal("49000"),
        )
        self.manager._states["BTCUSDT"] = state

        assert self.manager.remove_state("BTCUSDT")
        assert self.manager.get_state("BTCUSDT") is None
        assert not self.manager.remove_state("BTCUSDT")  # Already removed


class TestMockExchangeAdapter:
    """Tests for MockExchangeAdapter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.adapter = MockExchangeAdapter()

    @pytest.mark.asyncio
    async def test_place_stop_loss(self):
        """Test placing stop loss order."""
        order_id = await self.adapter.place_stop_loss(
            symbol="BTCUSDT",
            side="SELL",
            quantity=Decimal("0.1"),
            stop_price=Decimal("49000"),
        )
        assert order_id.startswith("MOCK_SL_")
        order = self.adapter.get_order(order_id)
        assert order is not None
        assert order["symbol"] == "BTCUSDT"
        assert order["stop_price"] == Decimal("49000")

    @pytest.mark.asyncio
    async def test_place_take_profit(self):
        """Test placing take profit order."""
        order_id = await self.adapter.place_take_profit(
            symbol="BTCUSDT",
            side="SELL",
            quantity=Decimal("0.1"),
            price=Decimal("52000"),
        )
        assert order_id.startswith("MOCK_TP_")

    @pytest.mark.asyncio
    async def test_cancel_order(self):
        """Test cancelling order."""
        order_id = await self.adapter.place_stop_loss(
            symbol="BTCUSDT",
            side="SELL",
            quantity=Decimal("0.1"),
            stop_price=Decimal("49000"),
        )
        assert await self.adapter.cancel_order("BTCUSDT", order_id)
        order = self.adapter.get_order(order_id)
        assert order["status"] == "CANCELED"

    @pytest.mark.asyncio
    async def test_modify_stop_loss(self):
        """Test modifying stop loss."""
        order_id = await self.adapter.place_stop_loss(
            symbol="BTCUSDT",
            side="SELL",
            quantity=Decimal("0.1"),
            stop_price=Decimal("49000"),
        )
        assert await self.adapter.modify_stop_loss(
            "BTCUSDT", order_id, Decimal("49500")
        )
        order = self.adapter.get_order(order_id)
        assert order["stop_price"] == Decimal("49500")


class TestSLTPConfig:
    """Tests for SLTPConfig."""

    def test_from_dict(self):
        """Test creating config from dictionary."""
        data = {
            "stop_loss": {
                "type": "percentage",
                "value": "0.02",
            },
            "take_profit": {
                "type": "risk_reward",
                "risk_reward_ratio": "2.5",
            },
            "trailing_stop": {
                "type": "atr_based",
                "atr_multiplier": "1.5",
                "enabled": True,
            },
            "use_exchange_orders": False,
        }

        config = SLTPConfig.from_dict(data)
        assert config.stop_loss.stop_type == StopLossType.PERCENTAGE
        assert config.stop_loss.value == Decimal("0.02")
        assert config.take_profit.tp_type == TakeProfitType.RISK_REWARD
        assert config.take_profit.risk_reward_ratio == Decimal("2.5")
        assert config.trailing_stop.trailing_type == TrailingStopType.ATR_BASED
        assert config.trailing_stop.enabled
        assert not config.use_exchange_orders
