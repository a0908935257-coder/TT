"""
End-to-end test for complete grid trading flow.

Simulates a complete trading cycle:
1. Bot startup and initial order placement
2. Price drop -> Buy fill -> Reverse sell placed
3. Price rise -> Sell fill -> Profit calculated
4. Price breakout -> Grid rebuild
5. Multiple breakouts -> Cooldown triggered
6. Bot stop -> Final statistics

Per Prompt 26f specification.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src.core.models import Kline, MarketType, Order, OrderSide, OrderStatus
from src.bots.grid import (
    ATRConfig,
    BotState,
    BreakoutDirection,
    DynamicAdjustConfig,
    GridBot,
    GridBotConfig,
    GridType,
    RiskConfig,
    RiskLevel,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


# =============================================================================
# Helper Functions
# =============================================================================


def generate_klines(count: int, base_price: Decimal = Decimal("50000")) -> list[Kline]:
    """
    Generate simulated K-line data for testing.

    Args:
        count: Number of klines to generate
        base_price: Starting price

    Returns:
        List of Kline objects
    """
    klines = []
    price = base_price
    base_time = datetime.now(timezone.utc) - timedelta(hours=count * 4)

    for i in range(count):
        # Small random-like variation
        variance = Decimal(str((i % 7 - 3) / 100))  # -0.03 to +0.03
        close = price * (Decimal("1") + variance)
        high = max(price, close) * Decimal("1.01")
        low = min(price, close) * Decimal("0.99")

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=base_time + timedelta(hours=i * 4),
            close_time=base_time + timedelta(hours=(i + 1) * 4 - 1),
            open=price,
            high=high,
            low=low,
            close=close,
            volume=Decimal("100"),
            quote_volume=close * Decimal("100"),
            trades=100,
        )
        klines.append(kline)
        price = close

    return klines


def create_breakout_kline(close_price: Decimal) -> Kline:
    """Create a K-line for breakout simulation."""
    return Kline(
        symbol="BTCUSDT",
        interval="4h",
        open_time=datetime.now(timezone.utc),
        close_time=datetime.now(timezone.utc),
        open=close_price * Decimal("0.99"),
        high=close_price * Decimal("1.01"),
        low=close_price * Decimal("0.98"),
        close=close_price,
        volume=Decimal("100"),
        quote_volume=close_price * Decimal("100"),
        trades=100,
    )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def e2e_exchange():
    """Create mock exchange for e2e tests."""
    exchange = MockExchangeClient()
    exchange.set_price(Decimal("50000"))
    exchange.set_balance("USDT", Decimal("100000"))
    exchange.set_balance("BTC", Decimal("2"))
    exchange.set_precision("BTCUSDT", price_precision=2, quantity_precision=6)
    return exchange


@pytest.fixture
def e2e_data_manager():
    """Create mock data manager for e2e tests."""
    data_manager = MockDataManager()
    data_manager.set_price("BTCUSDT", Decimal("50000"))
    data_manager.set_klines("BTCUSDT", generate_klines(50))
    return data_manager


@pytest.fixture
def e2e_notifier():
    """Create mock notifier for e2e tests."""
    return MockNotifier()


@pytest.fixture
def e2e_config():
    """Create bot config with dynamic adjustment enabled."""
    return GridBotConfig(
        symbol="BTCUSDT",
        market_type=MarketType.SPOT,
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MODERATE,
        grid_type=GridType.GEOMETRIC,
        manual_upper=Decimal("55000"),
        manual_lower=Decimal("45000"),
        manual_grid_count=10,
        atr_config=ATRConfig(
            period=14,
            timeframe="4h",
            multiplier=Decimal("2.0"),
        ),
        dynamic_adjust=DynamicAdjustConfig(
            enabled=True,
            breakout_threshold=Decimal("0.04"),  # 4%
            cooldown_days=7,
            max_rebuilds=3,
        ),
        risk_config=RiskConfig(
            auto_rebuild_enabled=True,
            breakout_threshold=Decimal("4.0"),
            cooldown_days=7,
            max_rebuilds_in_period=3,
        ),
    )


@pytest.fixture
def e2e_bot(e2e_config, e2e_exchange, e2e_data_manager, e2e_notifier):
    """Create GridBot for e2e testing."""
    return GridBot(
        bot_id="e2e_test_001",
        config=e2e_config,
        exchange=e2e_exchange,
        data_manager=e2e_data_manager,
        notifier=e2e_notifier,
    )


# =============================================================================
# E2E Tests
# =============================================================================


class TestFullTradingFlow:
    """Complete end-to-end trading flow tests."""

    @pytest.mark.asyncio
    async def test_complete_trading_cycle(
        self,
        e2e_bot,
        e2e_exchange,
        e2e_data_manager,
        e2e_notifier,
    ):
        """
        Test complete trading cycle:

        1. Bot startup and initial order placement
        2. Price drop -> Buy fill -> Reverse sell placed
        3. Price rise -> Sell fill -> Profit calculated
        4. Price breakout -> Grid rebuild
        5. Multiple breakouts -> Cooldown triggered
        6. Bot stop -> Final statistics
        """
        bot = e2e_bot
        exchange = e2e_exchange
        data_manager = e2e_data_manager

        # ========== Step 1: Bot Startup ==========
        print("\n========== Step 1: Bot Startup ==========")

        started = await bot.start()
        assert started is True, "Bot should start successfully"
        assert bot.state == BotState.RUNNING, "Bot state should be RUNNING"

        status = bot.get_status()
        assert status["grid_version"] == 1, "Initial grid version should be 1"
        assert bot.order_manager.active_order_count > 0, "Should have placed orders"

        initial_buy_count = len([
            o for o in bot.order_manager._orders.values()
            if o.side == OrderSide.BUY
        ])
        initial_sell_count = len([
            o for o in bot.order_manager._orders.values()
            if o.side == OrderSide.SELL
        ])

        print(f"  Bot started with {initial_buy_count} buy and {initial_sell_count} sell orders")
        print(f"  Grid range: {status['lower_price']} - {status['upper_price']}")

        # ========== Step 2: Buy Order Fill ==========
        print("\n========== Step 2: Simulate Buy Fill ==========")

        # Find a buy order
        buy_level_index = None
        buy_order = None
        for level_idx, order_id in bot.order_manager._level_order_map.items():
            order = bot.order_manager._orders.get(order_id)
            if order and order.side == OrderSide.BUY:
                buy_level_index = level_idx
                buy_order = order
                break

        if buy_order:
            # Simulate fill - use order_manager directly to get reverse order
            filled_buy = Order(
                order_id=buy_order.order_id,
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

            reverse_sell = await bot.order_manager.on_order_filled(filled_buy)

            stats = bot.order_manager.get_statistics()
            assert stats["buy_filled_count"] >= 1, "Should have recorded buy fill"

            if reverse_sell:
                assert reverse_sell.side == OrderSide.SELL, "Reverse should be SELL"
                print(f"  Buy filled at {buy_order.price}, reverse sell at {reverse_sell.price}")

                # ========== Step 3: Sell Order Fill ==========
                print("\n========== Step 3: Simulate Sell Fill ==========")

                filled_sell = Order(
                    order_id=reverse_sell.order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.SELL,
                    order_type=reverse_sell.order_type,
                    status=OrderStatus.FILLED,
                    price=reverse_sell.price,
                    quantity=reverse_sell.quantity,
                    filled_qty=reverse_sell.quantity,
                    avg_price=reverse_sell.price,
                    fee=reverse_sell.quantity * reverse_sell.price * Decimal("0.001"),
                    created_at=datetime.now(timezone.utc),
                )

                await bot.order_manager.on_order_filled(filled_sell)

                stats = bot.order_manager.get_statistics()
                assert stats["sell_filled_count"] >= 1, "Should have recorded sell fill"
                assert stats["total_profit"] > 0, "Should have profit after round trip"

                print(f"  Sell filled, profit: {stats['total_profit']}")

        # ========== Step 4: Breakout and Rebuild ==========
        print("\n========== Step 4: Simulate Breakout ==========")

        # Trigger upper breakout (> 55000 * 1.04 = 57200)
        breakout_price = Decimal("57500")
        exchange.set_price(breakout_price)
        data_manager.set_price("BTCUSDT", breakout_price)

        # Update klines with new price
        new_klines = generate_klines(50, base_price=breakout_price)
        data_manager.set_klines("BTCUSDT", new_klines)

        # Trigger via kline close
        breakout_kline = create_breakout_kline(breakout_price)
        await bot.on_kline_close(breakout_kline)

        status = bot.get_status()
        print(f"  Breakout triggered, rebuilds used: {status['rebuilds_used']}")

        # Verify rebuild happened
        assert status["rebuilds_used"] >= 1, "Should have triggered rebuild"

        # ========== Step 5: Cooldown Mechanism ==========
        print("\n========== Step 5: Test Cooldown Mechanism ==========")

        # Simulate additional breakouts to reach cooldown limit
        from src.bots.grid.risk_manager import RebuildRecord

        # Manually add rebuild records to test cooldown
        for i in range(2):  # Add 2 more to reach 3 total
            record = RebuildRecord(
                timestamp=datetime.now(timezone.utc),
                reason=f"test_breakout_{i + 2}",
                old_upper=Decimal("55000"),
                old_lower=Decimal("45000"),
                new_upper=Decimal("60000"),
                new_lower=Decimal("50000"),
                trigger_price=Decimal("57500"),
            )
            bot.risk_manager._rebuild_history.append(record)

        status = bot.get_status()
        assert status["rebuilds_used"] == 3, "Should have 3 rebuilds recorded"
        assert status["is_in_cooldown"] is True, "Should be in cooldown"
        assert status["rebuilds_remaining"] == 0, "No rebuilds remaining"

        print(f"  Cooldown active: {status['is_in_cooldown']}")
        print(f"  Rebuilds used: {status['rebuilds_used']}/{status['rebuilds_used'] + status['rebuilds_remaining']}")

        # ========== Step 6: Bot Stop ==========
        print("\n========== Step 6: Stop Bot ==========")

        stopped = await bot.stop()
        assert stopped is True, "Bot should stop successfully"
        assert bot.state == BotState.STOPPED, "Bot state should be STOPPED"
        assert bot.order_manager.active_order_count == 0, "All orders should be cancelled"

        final_status = bot.get_status()
        final_stats = bot.get_statistics()

        print(f"""
  Final Statistics:
  - State: {final_status['state']}
  - Total Trades: {final_stats['trade_count']}
  - Total Profit: {final_stats['total_profit']}
  - Grid Version: {final_status['grid_version']}
  - Rebuilds Used: {final_status['rebuilds_used']}
        """)

        print("\n========== E2E Test Complete ==========")

    @pytest.mark.asyncio
    async def test_multiple_round_trips(
        self,
        e2e_bot,
        e2e_exchange,
        e2e_data_manager,
    ):
        """Test multiple buy-sell round trips."""
        bot = e2e_bot

        await bot.start()

        completed_trades = 0

        # Simulate round trips - need to complete buy->sell pairs
        # Call order_manager.on_order_filled directly to get reverse orders
        for round_num in range(3):
            # Find a buy order that is still active
            buy_order = None
            buy_level_idx = None
            for level_idx, order_id in list(bot.order_manager._level_order_map.items()):
                order = bot.order_manager._orders.get(order_id)
                if order and order.side == OrderSide.BUY:
                    buy_order = order
                    buy_level_idx = level_idx
                    break

            if not buy_order:
                break

            # Fill buy - call order_manager directly to get reverse order
            filled_buy = Order(
                order_id=buy_order.order_id,
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

            # Call order_manager.on_order_filled to get the reverse order
            reverse_sell = await bot.order_manager.on_order_filled(filled_buy)

            if reverse_sell:
                # Fill the sell to complete the round trip
                filled_sell = Order(
                    order_id=reverse_sell.order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.SELL,
                    order_type=reverse_sell.order_type,
                    status=OrderStatus.FILLED,
                    price=reverse_sell.price,
                    quantity=reverse_sell.quantity,
                    filled_qty=reverse_sell.quantity,
                    avg_price=reverse_sell.price,
                    fee=reverse_sell.quantity * reverse_sell.price * Decimal("0.001"),
                    created_at=datetime.now(timezone.utc),
                )

                await bot.order_manager.on_order_filled(filled_sell)
                completed_trades += 1

        stats = bot.get_statistics()
        # A "trade" requires complete buy-sell pair
        assert stats["buy_filled_count"] >= 1, "Should have buy fills"
        assert stats["sell_filled_count"] >= 1, "Should have sell fills"
        # Profit should be positive from completed round trips
        assert stats["total_profit"] > 0, "Should have positive profit from round trips"

        await bot.stop()

    @pytest.mark.asyncio
    async def test_status_includes_all_required_fields(
        self,
        e2e_bot,
    ):
        """Verify status includes all required fields per spec."""
        bot = e2e_bot

        await bot.start()

        status = bot.get_status()

        # Required fields from spec
        required_fields = [
            "bot_id",
            "state",
            "symbol",
            "upper_price",
            "lower_price",
            "grid_count",
            "total_profit",
            "grid_version",
            "auto_rebuild_enabled",
            "rebuilds_used",
            "rebuilds_remaining",
            "is_in_cooldown",
        ]

        for field in required_fields:
            assert field in status, f"Status should include '{field}'"

        await bot.stop()

    @pytest.mark.asyncio
    async def test_pause_resume_flow(
        self,
        e2e_bot,
    ):
        """Test pause and resume functionality."""
        bot = e2e_bot

        await bot.start()
        assert bot.state == BotState.RUNNING

        # Pause
        paused = await bot.pause("E2E test pause")
        assert paused is True
        assert bot.state == BotState.PAUSED

        # Resume
        resumed = await bot.resume()
        assert resumed is True
        assert bot.state == BotState.RUNNING

        await bot.stop()

    @pytest.mark.asyncio
    async def test_profit_tracking_accuracy(
        self,
        e2e_bot,
    ):
        """Test accurate profit tracking through trades."""
        bot = e2e_bot

        await bot.start()

        # Find a buy order at a known level
        buy_order = None
        buy_level = None
        for level_idx, order_id in bot.order_manager._level_order_map.items():
            order = bot.order_manager._orders.get(order_id)
            if order and order.side == OrderSide.BUY:
                buy_order = order
                buy_level = bot.setup.levels[level_idx]
                break

        if buy_order and buy_level:
            buy_price = buy_order.price
            quantity = buy_order.quantity
            buy_fee = buy_price * quantity * Decimal("0.001")

            # Fill buy - use order_manager directly to get reverse order
            filled_buy = Order(
                order_id=buy_order.order_id,
                symbol="BTCUSDT",
                side=OrderSide.BUY,
                order_type=buy_order.order_type,
                status=OrderStatus.FILLED,
                price=buy_price,
                quantity=quantity,
                filled_qty=quantity,
                avg_price=buy_price,
                fee=buy_fee,
                created_at=datetime.now(timezone.utc),
            )

            reverse_sell = await bot.order_manager.on_order_filled(filled_buy)

            if reverse_sell:
                sell_price = reverse_sell.price
                sell_fee = sell_price * quantity * Decimal("0.001")

                # Fill sell
                filled_sell = Order(
                    order_id=reverse_sell.order_id,
                    symbol="BTCUSDT",
                    side=OrderSide.SELL,
                    order_type=reverse_sell.order_type,
                    status=OrderStatus.FILLED,
                    price=sell_price,
                    quantity=quantity,
                    filled_qty=quantity,
                    avg_price=sell_price,
                    fee=sell_fee,
                    created_at=datetime.now(timezone.utc),
                )

                await bot.order_manager.on_order_filled(filled_sell)

                # Calculate expected profit
                gross_profit = (sell_price - buy_price) * quantity
                total_fees = buy_fee + sell_fee
                expected_profit = gross_profit - total_fees

                stats = bot.get_statistics()
                # Allow small rounding difference
                assert abs(stats["total_profit"] - expected_profit) < Decimal("0.01"), \
                    f"Profit mismatch: {stats['total_profit']} vs expected {expected_profit}"

        await bot.stop()


class TestModuleIntegration:
    """Tests verifying correct module integration."""

    @pytest.mark.asyncio
    async def test_calculator_to_order_manager_integration(
        self,
        e2e_bot,
    ):
        """Verify Calculator -> OrderManager integration."""
        bot = e2e_bot

        await bot.start()

        # Verify setup was passed to order manager
        assert bot.order_manager.setup is not None
        assert bot.order_manager.setup == bot.setup

        # Verify levels match
        assert len(bot.order_manager._setup.levels) == bot.setup.grid_count + 1

        await bot.stop()

    @pytest.mark.asyncio
    async def test_risk_manager_to_dynamic_adjuster_integration(
        self,
        e2e_bot,
        e2e_exchange,
        e2e_data_manager,
    ):
        """Verify RiskManager -> DynamicAdjuster integration."""
        bot = e2e_bot
        exchange = e2e_exchange
        data_manager = e2e_data_manager

        await bot.start()

        # Verify risk manager has dynamic adjustment configured
        assert bot.risk_manager is not None
        assert bot.risk_manager.can_rebuild is True

        # Simulate breakout
        breakout_price = Decimal("57500")
        exchange.set_price(breakout_price)
        data_manager.set_price("BTCUSDT", breakout_price)

        # Check breakout detection
        direction = bot.risk_manager.check_dynamic_adjust_trigger(breakout_price)
        assert direction == "upper", "Should detect upper breakout"

        await bot.stop()

    @pytest.mark.asyncio
    async def test_order_manager_to_exchange_integration(
        self,
        e2e_bot,
        e2e_exchange,
    ):
        """Verify OrderManager -> Exchange integration."""
        bot = e2e_bot
        exchange = e2e_exchange

        await bot.start()

        # Verify orders were placed on exchange
        initial_order_count = bot.order_manager.active_order_count
        exchange_orders = await exchange.get_open_orders("BTCUSDT")

        assert len(exchange_orders) == initial_order_count, \
            "Exchange should have same number of orders as manager"

        await bot.stop()

    @pytest.mark.asyncio
    async def test_notifier_receives_events(
        self,
        e2e_bot,
        e2e_notifier,
    ):
        """Verify Notifier receives events."""
        bot = e2e_bot
        notifier = e2e_notifier

        await bot.start()

        # Should have notification about bot start (use count() method)
        assert notifier.count() > 0 or len(notifier.messages) > 0, "Notifier should receive start message"

        await bot.stop()

        # Note: The bot may or may not send notifications via the notifier
        # depending on implementation. This test verifies the integration path works.
        # If messages are sent, they should be recorded
        assert isinstance(notifier.messages, list), "Notifier should have messages list"
