"""
Tests for Capital Monitor.

Tests capital tracking, change calculations, and alert generation.
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import AccountInfo, Balance, MarketType, Position, PositionSide
from src.risk.capital_monitor import CapitalMonitor
from src.risk.models import (
    RiskAction,
    RiskConfig,
    RiskLevel,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_config():
    """Create a risk configuration."""
    return RiskConfig(
        total_capital=Decimal("100000"),
        warning_loss_pct=Decimal("0.10"),  # 10%
        danger_loss_pct=Decimal("0.20"),  # 20%
        daily_loss_warning=Decimal("0.05"),  # 5%
        daily_loss_danger=Decimal("0.10"),  # 10%
    )


@pytest.fixture
def mock_exchange():
    """Create a mock exchange client."""
    exchange = MagicMock()

    # Mock get_account to return account with USDT balance
    async def get_account_impl(market=MarketType.SPOT):
        return AccountInfo(
            market_type=market,
            balances=[
                Balance(asset="USDT", free=Decimal("80000"), locked=Decimal("20000")),
                Balance(asset="BTC", free=Decimal("0.5"), locked=Decimal("0")),
            ],
            positions=[],
            updated_at=datetime.now(),
        )

    exchange.get_account = AsyncMock(side_effect=get_account_impl)
    exchange.get_positions = AsyncMock(return_value=[])

    return exchange


@pytest.fixture
def monitor(risk_config):
    """Create a capital monitor without exchange."""
    return CapitalMonitor(risk_config)


@pytest.fixture
def monitor_with_exchange(risk_config, mock_exchange):
    """Create a capital monitor with mock exchange."""
    return CapitalMonitor(risk_config, mock_exchange)


# =============================================================================
# Initialization Tests
# =============================================================================


class TestCapitalMonitorInit:
    """Tests for CapitalMonitor initialization."""

    def test_init_with_config(self, risk_config):
        """Test initialization with config."""
        monitor = CapitalMonitor(risk_config)

        assert monitor.config == risk_config
        assert monitor.initial_capital == Decimal("100000")
        assert monitor.peak_capital == Decimal("100000")
        assert monitor.last_snapshot is None

    def test_init_with_exchange(self, risk_config, mock_exchange):
        """Test initialization with exchange."""
        monitor = CapitalMonitor(risk_config, mock_exchange)

        assert monitor._exchange == mock_exchange


# =============================================================================
# Update Tests
# =============================================================================


class TestCapitalMonitorUpdate:
    """Tests for capital update functionality."""

    @pytest.mark.asyncio
    async def test_update_from_exchange(self, monitor_with_exchange):
        """Test updating capital from exchange."""
        snapshot = await monitor_with_exchange.update()

        assert snapshot is not None
        assert snapshot.total_capital == Decimal("100000")  # 80000 + 20000
        assert snapshot.available_balance == Decimal("80000")
        assert snapshot.position_value == Decimal("0")

    @pytest.mark.asyncio
    async def test_update_without_exchange_raises(self, monitor):
        """Test that update without exchange raises error."""
        with pytest.raises(RuntimeError, match="No exchange client configured"):
            await monitor.update()

    def test_update_from_values(self, monitor):
        """Test updating from explicit values."""
        snapshot = monitor.update_from_values(
            total_capital=Decimal("105000"),
            available_balance=Decimal("85000"),
            position_value=Decimal("20000"),
            unrealized_pnl=Decimal("500"),
        )

        assert snapshot.total_capital == Decimal("105000")
        assert snapshot.available_balance == Decimal("85000")
        assert monitor.last_snapshot == snapshot

    def test_update_tracks_peak(self, monitor):
        """Test that updates track peak capital."""
        # Initial state
        assert monitor.peak_capital == Decimal("100000")

        # Update with higher value
        monitor.update_from_values(
            total_capital=Decimal("110000"),
            available_balance=Decimal("110000"),
        )
        assert monitor.peak_capital == Decimal("110000")

        # Update with lower value - peak should stay
        monitor.update_from_values(
            total_capital=Decimal("105000"),
            available_balance=Decimal("105000"),
        )
        assert monitor.peak_capital == Decimal("110000")

    def test_update_initializes_daily_start(self, monitor):
        """Test that first update sets daily start capital."""
        assert monitor._daily_start_capital is None

        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        assert monitor._daily_start_capital == Decimal("100000")


# =============================================================================
# Capital Change Tests
# =============================================================================


class TestCapitalChange:
    """Tests for capital change calculation."""

    def test_get_capital_change_no_snapshot(self, monitor):
        """Test capital change with no snapshot."""
        change, change_pct = monitor.get_capital_change()

        assert change == Decimal("0")
        assert change_pct == Decimal("0")

    def test_get_capital_change_profit(self, monitor):
        """Test capital change with profit."""
        monitor.update_from_values(
            total_capital=Decimal("110000"),
            available_balance=Decimal("110000"),
        )

        change, change_pct = monitor.get_capital_change()

        assert change == Decimal("10000")
        assert change_pct == Decimal("0.1")

    def test_get_capital_change_loss(self, monitor):
        """Test capital change with loss."""
        monitor.update_from_values(
            total_capital=Decimal("90000"),
            available_balance=Decimal("90000"),
        )

        change, change_pct = monitor.get_capital_change()

        assert change == Decimal("-10000")
        assert change_pct == Decimal("-0.1")


# =============================================================================
# Daily P&L Tests
# =============================================================================


class TestDailyPnL:
    """Tests for daily P&L calculation."""

    def test_get_daily_pnl_no_snapshot(self, monitor):
        """Test daily P&L with no snapshot."""
        pnl = monitor.get_daily_pnl()

        assert pnl.date == date.today()
        assert pnl.pnl == Decimal("0")
        assert pnl.pnl_pct == Decimal("0")

    def test_get_daily_pnl_profit(self, monitor):
        """Test daily P&L with profit."""
        # First update sets daily start
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        # Second update shows profit
        monitor.update_from_values(
            total_capital=Decimal("103000"),
            available_balance=Decimal("103000"),
        )

        pnl = monitor.get_daily_pnl()

        assert pnl.pnl == Decimal("3000")
        assert pnl.pnl_pct == Decimal("0.03")

    def test_get_daily_pnl_loss(self, monitor):
        """Test daily P&L with loss."""
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        monitor.update_from_values(
            total_capital=Decimal("95000"),
            available_balance=Decimal("95000"),
        )

        pnl = monitor.get_daily_pnl()

        assert pnl.pnl == Decimal("-5000")
        assert pnl.pnl_pct == Decimal("-0.05")

    def test_get_daily_pnl_with_trades(self, monitor):
        """Test daily P&L with recorded trades."""
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        # Record some trades
        monitor.record_trade(is_win=True)
        monitor.record_trade(is_win=True)
        monitor.record_trade(is_win=False)

        pnl = monitor.get_daily_pnl()

        assert pnl.trade_count == 3
        assert pnl.win_count == 2
        assert pnl.loss_count == 1


# =============================================================================
# Alert Tests
# =============================================================================


class TestCapitalAlerts:
    """Tests for alert generation."""

    def test_check_alerts_no_snapshot(self, monitor):
        """Test alerts with no snapshot."""
        alerts = monitor.check_alerts()

        assert len(alerts) == 0

    def test_check_alerts_no_breach(self, monitor):
        """Test alerts with no threshold breach."""
        monitor.update_from_values(
            total_capital=Decimal("95000"),  # 5% loss - below warning
            available_balance=Decimal("95000"),
        )

        alerts = monitor.check_alerts()

        assert len(alerts) == 0

    def test_check_alerts_warning_total_loss(self, monitor):
        """Test warning alert for total capital loss."""
        monitor.update_from_values(
            total_capital=Decimal("88000"),  # 12% loss
            available_balance=Decimal("88000"),
        )

        alerts = monitor.check_alerts()

        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.WARNING
        assert alerts[0].metric == "total_capital_loss"
        assert alerts[0].action_taken == RiskAction.NOTIFY

    def test_check_alerts_danger_total_loss(self, monitor):
        """Test danger alert for total capital loss."""
        monitor.update_from_values(
            total_capital=Decimal("75000"),  # 25% loss
            available_balance=Decimal("75000"),
        )

        alerts = monitor.check_alerts()

        # Should have danger alert (not warning, since danger takes precedence)
        total_loss_alerts = [a for a in alerts if a.metric == "total_capital_loss"]
        assert len(total_loss_alerts) == 1
        assert total_loss_alerts[0].level == RiskLevel.DANGER
        assert total_loss_alerts[0].action_taken == RiskAction.PAUSE_ALL_BOTS

    def test_check_alerts_warning_daily_loss(self, monitor):
        """Test warning alert for daily loss."""
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        monitor.update_from_values(
            total_capital=Decimal("93000"),  # 7% daily loss
            available_balance=Decimal("93000"),
        )

        alerts = monitor.check_alerts()

        daily_alerts = [a for a in alerts if a.metric == "daily_loss"]
        assert len(daily_alerts) == 1
        assert daily_alerts[0].level == RiskLevel.WARNING

    def test_check_alerts_danger_daily_loss(self, monitor):
        """Test danger alert for daily loss."""
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        monitor.update_from_values(
            total_capital=Decimal("88000"),  # 12% daily loss
            available_balance=Decimal("88000"),
        )

        alerts = monitor.check_alerts()

        daily_alerts = [a for a in alerts if a.metric == "daily_loss"]
        assert len(daily_alerts) == 1
        assert daily_alerts[0].level == RiskLevel.DANGER

    def test_check_alerts_multiple(self, monitor):
        """Test multiple alerts triggered simultaneously."""
        # Severe loss triggers both total and daily alerts
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )
        monitor.update_from_values(
            total_capital=Decimal("70000"),  # 30% loss
            available_balance=Decimal("70000"),
        )

        alerts = monitor.check_alerts()

        # Should have both total loss and daily loss danger alerts
        assert len(alerts) == 2
        metrics = {a.metric for a in alerts}
        assert "total_capital_loss" in metrics
        assert "daily_loss" in metrics


# =============================================================================
# Daily Reset Tests
# =============================================================================


class TestDailyReset:
    """Tests for daily reset functionality."""

    def test_reset_daily(self, monitor):
        """Test daily reset."""
        # First update sets daily start capital
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        # Second update shows profit
        monitor.update_from_values(
            total_capital=Decimal("105000"),
            available_balance=Decimal("105000"),
        )
        monitor.record_trade(is_win=True)
        monitor.record_trade(is_win=False)

        # Reset
        yesterday_pnl = monitor.reset_daily()

        # Check yesterday's P&L was returned
        assert yesterday_pnl.pnl == Decimal("5000")
        assert yesterday_pnl.trade_count == 2

        # Check state was reset
        assert monitor._daily_start_capital == Decimal("105000")
        assert monitor._daily_trade_count == 0
        assert monitor._daily_win_count == 0
        assert monitor._daily_loss_count == 0

    def test_reset_daily_without_snapshot(self, monitor):
        """Test daily reset without any snapshots."""
        yesterday_pnl = monitor.reset_daily()

        assert yesterday_pnl.pnl == Decimal("0")
        assert monitor._daily_start_capital == Decimal("100000")


# =============================================================================
# Trade Recording Tests
# =============================================================================


class TestTradeRecording:
    """Tests for trade recording."""

    def test_record_winning_trade(self, monitor):
        """Test recording a winning trade."""
        monitor.record_trade(is_win=True)

        assert monitor._daily_trade_count == 1
        assert monitor._daily_win_count == 1
        assert monitor._daily_loss_count == 0

    def test_record_losing_trade(self, monitor):
        """Test recording a losing trade."""
        monitor.record_trade(is_win=False)

        assert monitor._daily_trade_count == 1
        assert monitor._daily_win_count == 0
        assert monitor._daily_loss_count == 1

    def test_record_multiple_trades(self, monitor):
        """Test recording multiple trades."""
        for _ in range(5):
            monitor.record_trade(is_win=True)
        for _ in range(3):
            monitor.record_trade(is_win=False)

        assert monitor._daily_trade_count == 8
        assert monitor._daily_win_count == 5
        assert monitor._daily_loss_count == 3


# =============================================================================
# Utility Tests
# =============================================================================


class TestCapitalMonitorUtility:
    """Tests for utility methods."""

    def test_set_initial_capital(self, monitor):
        """Test setting initial capital."""
        monitor.set_initial_capital(Decimal("200000"))

        assert monitor.initial_capital == Decimal("200000")
        assert monitor.config.total_capital == Decimal("200000")

    def test_get_snapshots_since(self, monitor):
        """Test getting snapshots since a time."""
        # Add some snapshots
        for i in range(5):
            monitor.update_from_values(
                total_capital=Decimal(f"{100000 + i * 1000}"),
                available_balance=Decimal(f"{100000 + i * 1000}"),
            )

        # Get snapshots since 2 ago
        since = datetime.now() - timedelta(seconds=1)
        recent = monitor.get_snapshots_since(since)

        # Should include all snapshots (they were just created)
        assert len(recent) >= 1

    def test_get_snapshots_for_period(self, monitor):
        """Test getting snapshots within a period."""
        # Add some snapshots
        for i in range(3):
            monitor.update_from_values(
                total_capital=Decimal(f"{100000 + i * 1000}"),
                available_balance=Decimal(f"{100000 + i * 1000}"),
            )

        start = datetime.now() - timedelta(hours=1)
        end = datetime.now() + timedelta(hours=1)
        snapshots = monitor.get_snapshots_for_period(start, end)

        assert len(snapshots) == 3

    def test_clear_snapshots(self, monitor):
        """Test clearing snapshot history."""
        monitor.update_from_values(
            total_capital=Decimal("100000"),
            available_balance=Decimal("100000"),
        )

        assert len(monitor.snapshots) == 1

        monitor.clear_snapshots()

        assert len(monitor.snapshots) == 0

    def test_max_snapshots_limit(self, monitor):
        """Test that snapshots are trimmed when exceeding max."""
        monitor._max_snapshots = 5

        # Add more than max snapshots
        for i in range(10):
            monitor.update_from_values(
                total_capital=Decimal(f"{100000 + i * 100}"),
                available_balance=Decimal(f"{100000 + i * 100}"),
            )

        assert len(monitor.snapshots) == 5


# =============================================================================
# Exchange Integration Tests
# =============================================================================


class TestExchangeIntegration:
    """Tests for exchange integration."""

    @pytest.mark.asyncio
    async def test_update_with_positions(self, risk_config):
        """Test update with futures positions."""
        exchange = MagicMock()

        async def get_account_impl(market=MarketType.SPOT):
            return AccountInfo(
                market_type=market,
                balances=[
                    Balance(asset="USDT", free=Decimal("50000"), locked=Decimal("0")),
                ],
                positions=[],
                updated_at=datetime.now(),
            )

        async def get_positions_impl(symbol=None):
            return [
                Position(
                    symbol="BTCUSDT",
                    side=PositionSide.LONG,
                    quantity=Decimal("0.5"),
                    entry_price=Decimal("40000"),
                    mark_price=Decimal("42000"),
                    leverage=10,
                    margin=Decimal("2000"),
                    unrealized_pnl=Decimal("1000"),
                    updated_at=datetime.now(),
                )
            ]

        exchange.get_account = AsyncMock(side_effect=get_account_impl)
        exchange.get_positions = AsyncMock(side_effect=get_positions_impl)

        monitor = CapitalMonitor(risk_config, exchange, MarketType.FUTURES)
        snapshot = await monitor.update()

        # Total = 50000 USDT + (0.5 * 42000) position value
        assert snapshot.total_capital == Decimal("50000") + Decimal("21000")
        assert snapshot.unrealized_pnl == Decimal("1000")
