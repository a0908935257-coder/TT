"""
Tests for Liquidation Monitor.

Tests liquidation distance calculation, risk level assessment, and alert generation.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.models import Position, PositionSide
from src.risk.liquidation_monitor import LiquidationMonitor
from src.risk.models import RiskAction, RiskConfig, RiskLevel


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def risk_config():
    """Create a risk configuration with liquidation thresholds."""
    return RiskConfig(
        total_capital=Decimal("100000"),
        liquidation_warning_pct=Decimal("0.10"),   # 10%
        liquidation_danger_pct=Decimal("0.05"),    # 5%
        liquidation_critical_pct=Decimal("0.02"),  # 2%
    )


@pytest.fixture
def monitor(risk_config):
    """Create a liquidation monitor without exchange."""
    return LiquidationMonitor(risk_config)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange."""
    exchange = MagicMock()
    exchange.get_positions = AsyncMock(return_value=[])
    return exchange


@pytest.fixture
def monitor_with_exchange(risk_config, mock_exchange):
    """Create a liquidation monitor with mock exchange."""
    return LiquidationMonitor(risk_config, mock_exchange)


def create_position(
    symbol: str = "BTCUSDT",
    side: PositionSide = PositionSide.LONG,
    quantity: Decimal = Decimal("1.0"),
    entry_price: Decimal = Decimal("100"),
    mark_price: Decimal = Decimal("100"),
    liquidation_price: Decimal = Decimal("80"),
    leverage: int = 10,
    unrealized_pnl: Decimal = Decimal("0"),
) -> Position:
    """Helper to create a Position for testing."""
    return Position(
        symbol=symbol,
        side=side,
        quantity=quantity,
        entry_price=entry_price,
        mark_price=mark_price,
        liquidation_price=liquidation_price,
        leverage=leverage,
        margin=Decimal("10"),
        unrealized_pnl=unrealized_pnl,
        updated_at=datetime.now(timezone.utc),
    )


# =============================================================================
# Long Position Distance Calculation Tests
# =============================================================================


class TestLongLiquidationDistance:
    """Tests for LONG position liquidation distance calculation."""

    def test_long_normal_distance(self, monitor):
        """Test LONG position with safe distance (20%)."""
        # mark=100, liq=80 -> distance = (100-80)/100 = 20%
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("80"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.20")
        assert snapshot.risk_level == RiskLevel.NORMAL

    def test_long_warning_distance(self, monitor):
        """Test LONG position approaching warning (8%)."""
        # mark=100, liq=92 -> distance = (100-92)/100 = 8%
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("92"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.08")
        assert snapshot.risk_level == RiskLevel.WARNING

    def test_long_danger_distance(self, monitor):
        """Test LONG position in danger zone (3%)."""
        # mark=100, liq=97 -> distance = (100-97)/100 = 3%
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("97"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.03")
        assert snapshot.risk_level == RiskLevel.DANGER

    def test_long_critical_distance(self, monitor):
        """Test LONG position in critical zone (1%)."""
        # mark=100, liq=99 -> distance = (100-99)/100 = 1%
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("99"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.01")
        assert snapshot.risk_level == RiskLevel.CIRCUIT_BREAK

    def test_long_exceeded_liquidation(self, monitor):
        """Test LONG position that has exceeded liquidation price."""
        # mark=79, liq=80 -> distance = (79-80)/79 = -0.0126... (negative)
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("79"),
            liquidation_price=Decimal("80"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct < Decimal("0")  # Negative = exceeded
        assert snapshot.risk_level == RiskLevel.CIRCUIT_BREAK


# =============================================================================
# Short Position Distance Calculation Tests
# =============================================================================


class TestShortLiquidationDistance:
    """Tests for SHORT position liquidation distance calculation."""

    def test_short_normal_distance(self, monitor):
        """Test SHORT position with safe distance (20%)."""
        # mark=100, liq=120 -> distance = (120-100)/100 = 20%
        pos = create_position(
            side=PositionSide.SHORT,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("120"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.20")
        assert snapshot.risk_level == RiskLevel.NORMAL

    def test_short_warning_distance(self, monitor):
        """Test SHORT position approaching warning (8%)."""
        # mark=100, liq=108 -> distance = (108-100)/100 = 8%
        pos = create_position(
            side=PositionSide.SHORT,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("108"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.08")
        assert snapshot.risk_level == RiskLevel.WARNING

    def test_short_danger_distance(self, monitor):
        """Test SHORT position in danger zone (3%)."""
        # mark=100, liq=103 -> distance = (103-100)/100 = 3%
        pos = create_position(
            side=PositionSide.SHORT,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("103"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.03")
        assert snapshot.risk_level == RiskLevel.DANGER

    def test_short_critical_distance(self, monitor):
        """Test SHORT position in critical zone (1%)."""
        # mark=100, liq=101 -> distance = (101-100)/100 = 1%
        pos = create_position(
            side=PositionSide.SHORT,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("101"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.01")
        assert snapshot.risk_level == RiskLevel.CIRCUIT_BREAK

    def test_short_exceeded_liquidation(self, monitor):
        """Test SHORT position that has exceeded liquidation price."""
        # mark=121, liq=120 -> distance = (120-121)/121 = negative
        pos = create_position(
            side=PositionSide.SHORT,
            mark_price=Decimal("121"),
            liquidation_price=Decimal("120"),
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct < Decimal("0")  # Negative = exceeded
        assert snapshot.risk_level == RiskLevel.CIRCUIT_BREAK


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_liquidation_price_none(self, monitor):
        """Test position with liquidation_price = None."""
        pos = create_position(liquidation_price=None)
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_liquidation_price_zero(self, monitor):
        """Test position with liquidation_price = 0."""
        pos = create_position(liquidation_price=Decimal("0"))
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_mark_price_zero(self, monitor):
        """Test position with mark_price = 0 (avoid division by zero)."""
        pos = create_position(mark_price=Decimal("0"))
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_quantity_zero(self, monitor):
        """Test position with quantity = 0 (no actual position)."""
        pos = create_position(quantity=Decimal("0"))
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_quantity_negative(self, monitor):
        """Test position with negative quantity."""
        pos = create_position(quantity=Decimal("-1"))
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_side_both(self, monitor):
        """Test position with side = BOTH (no directional position)."""
        pos = create_position(side=PositionSide.BOTH)
        snapshot = monitor._calculate_snapshot(pos)
        assert snapshot is None

    def test_exact_threshold_warning(self, monitor):
        """Test position exactly at warning threshold (10%)."""
        # At exactly 10%, should be WARNING (< not <=)
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("90"),  # 10% distance
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.10")
        # At exactly threshold, should be NORMAL (not breached yet)
        assert snapshot.risk_level == RiskLevel.NORMAL

    def test_exact_threshold_danger(self, monitor):
        """Test position exactly at danger threshold (5%)."""
        pos = create_position(
            side=PositionSide.LONG,
            mark_price=Decimal("100"),
            liquidation_price=Decimal("95"),  # 5% distance
        )
        snapshot = monitor._calculate_snapshot(pos)

        assert snapshot is not None
        assert snapshot.distance_pct == Decimal("0.05")
        # At exactly 5%, should be WARNING (danger is <5%)
        assert snapshot.risk_level == RiskLevel.WARNING


# =============================================================================
# Alert Generation Tests
# =============================================================================


class TestAlertGeneration:
    """Tests for alert generation."""

    @pytest.mark.asyncio
    async def test_no_alerts_for_normal_positions(self, monitor_with_exchange, mock_exchange):
        """Test no alerts generated for normal positions."""
        mock_exchange.get_positions.return_value = [
            create_position(
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("70"),  # 30% distance = NORMAL
            )
        ]

        await monitor_with_exchange.update()
        alerts = monitor_with_exchange.check_alerts()

        assert len(alerts) == 0

    @pytest.mark.asyncio
    async def test_alert_for_warning_position(self, monitor_with_exchange, mock_exchange):
        """Test alert generated for warning position."""
        mock_exchange.get_positions.return_value = [
            create_position(
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("92"),  # 8% distance = WARNING
            )
        ]

        await monitor_with_exchange.update()
        alerts = monitor_with_exchange.check_alerts()

        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.WARNING
        assert alerts[0].metric == "liquidation_distance"
        assert alerts[0].action_taken == RiskAction.NOTIFY

    @pytest.mark.asyncio
    async def test_alert_for_danger_position(self, monitor_with_exchange, mock_exchange):
        """Test alert generated for danger position."""
        mock_exchange.get_positions.return_value = [
            create_position(
                side=PositionSide.SHORT,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("103"),  # 3% distance = DANGER
            )
        ]

        await monitor_with_exchange.update()
        alerts = monitor_with_exchange.check_alerts()

        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.DANGER
        assert alerts[0].action_taken == RiskAction.PAUSE_ALL_BOTS

    @pytest.mark.asyncio
    async def test_alert_for_critical_position(self, monitor_with_exchange, mock_exchange):
        """Test alert generated for critical position."""
        mock_exchange.get_positions.return_value = [
            create_position(
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("99"),  # 1% distance = CIRCUIT_BREAK
            )
        ]

        await monitor_with_exchange.update()
        alerts = monitor_with_exchange.check_alerts()

        assert len(alerts) == 1
        assert alerts[0].level == RiskLevel.CIRCUIT_BREAK
        assert alerts[0].action_taken == RiskAction.EMERGENCY_STOP

    @pytest.mark.asyncio
    async def test_multiple_position_alerts(self, monitor_with_exchange, mock_exchange):
        """Test alerts for multiple positions with different risk levels."""
        mock_exchange.get_positions.return_value = [
            create_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("92"),  # WARNING
            ),
            create_position(
                symbol="ETHUSDT",
                side=PositionSide.SHORT,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("103"),  # DANGER
            ),
        ]

        await monitor_with_exchange.update()
        alerts = monitor_with_exchange.check_alerts()

        assert len(alerts) == 2
        levels = {a.level for a in alerts}
        assert RiskLevel.WARNING in levels
        assert RiskLevel.DANGER in levels


# =============================================================================
# Update Tests
# =============================================================================


class TestUpdate:
    """Tests for update functionality."""

    @pytest.mark.asyncio
    async def test_update_without_exchange(self, monitor):
        """Test update without exchange returns empty dict."""
        result = await monitor.update()
        assert result == {}

    @pytest.mark.asyncio
    async def test_update_with_positions(self, monitor_with_exchange, mock_exchange):
        """Test update stores snapshots correctly."""
        mock_exchange.get_positions.return_value = [
            create_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("80"),
            )
        ]

        snapshots = await monitor_with_exchange.update()

        assert len(snapshots) == 1
        assert "BTCUSDT_LONG" in snapshots
        assert snapshots["BTCUSDT_LONG"].symbol == "BTCUSDT"
        assert snapshots["BTCUSDT_LONG"].side == "LONG"

    @pytest.mark.asyncio
    async def test_update_on_exchange_error(self, monitor_with_exchange, mock_exchange):
        """Test update returns last snapshots on exchange error."""
        # First successful update
        mock_exchange.get_positions.return_value = [
            create_position(symbol="BTCUSDT")
        ]
        await monitor_with_exchange.update()

        # Exchange error on second update
        mock_exchange.get_positions.side_effect = Exception("Connection error")
        snapshots = await monitor_with_exchange.update()

        # Should return last known snapshots
        assert len(snapshots) == 1
        assert "BTCUSDT_LONG" in snapshots


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.mark.asyncio
    async def test_get_highest_risk_empty(self, monitor):
        """Test get_highest_risk with no positions."""
        result = monitor.get_highest_risk()
        assert result is None

    @pytest.mark.asyncio
    async def test_get_highest_risk(self, monitor_with_exchange, mock_exchange):
        """Test get_highest_risk returns position closest to liquidation."""
        mock_exchange.get_positions.return_value = [
            create_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("80"),  # 20% distance
            ),
            create_position(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("95"),  # 5% distance (highest risk)
            ),
        ]

        await monitor_with_exchange.update()
        highest_risk = monitor_with_exchange.get_highest_risk()

        assert highest_risk is not None
        assert highest_risk.symbol == "ETHUSDT"
        assert highest_risk.distance_pct == Decimal("0.05")

    @pytest.mark.asyncio
    async def test_get_risky_positions(self, monitor_with_exchange, mock_exchange):
        """Test get_risky_positions filters by risk level."""
        mock_exchange.get_positions.return_value = [
            create_position(
                symbol="BTCUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("70"),  # NORMAL
            ),
            create_position(
                symbol="ETHUSDT",
                side=PositionSide.LONG,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("92"),  # WARNING
            ),
            create_position(
                symbol="SOLUSDT",
                side=PositionSide.SHORT,
                mark_price=Decimal("100"),
                liquidation_price=Decimal("103"),  # DANGER
            ),
        ]

        await monitor_with_exchange.update()

        # Get WARNING and above
        risky = monitor_with_exchange.get_risky_positions(RiskLevel.WARNING)
        assert len(risky) == 2

        # Get DANGER only
        danger = monitor_with_exchange.get_risky_positions(RiskLevel.DANGER)
        assert len(danger) == 1
        assert danger[0].symbol == "SOLUSDT"
