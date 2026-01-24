"""
Tests for ConcentrationMonitor.

Tests concentration risk monitoring including:
- Single position concentration
- Sector/category concentration
- Correlation group monitoring
- Direction balance
- Leverage monitoring
- Diversity metrics
"""

import pytest
from datetime import datetime
from decimal import Decimal
from typing import List

from src.risk.concentration_monitor import (
    AssetCategory,
    ConcentrationAlert,
    ConcentrationAlertType,
    ConcentrationConfig,
    ConcentrationLevel,
    ConcentrationMonitor,
    ConcentrationSnapshot,
    PositionInfo,
)


class MockPositionProvider:
    """Mock position data provider for testing."""

    def __init__(
        self,
        positions: List[PositionInfo] = None,
        total_value: Decimal = None,
    ):
        self._positions = positions or []
        self._total_value = total_value

    def get_all_positions(self) -> List[PositionInfo]:
        return self._positions

    def get_total_portfolio_value(self) -> Decimal:
        if self._total_value is not None:
            return self._total_value
        return sum(p.market_value for p in self._positions)


@pytest.fixture
def config() -> ConcentrationConfig:
    """Default concentration config for testing."""
    return ConcentrationConfig(
        single_position_warning_pct=Decimal("0.20"),
        single_position_danger_pct=Decimal("0.30"),
        single_position_critical_pct=Decimal("0.40"),
        category_warning_pct=Decimal("0.30"),
        category_danger_pct=Decimal("0.50"),
        correlation_group_warning_pct=Decimal("0.40"),
        correlation_group_danger_pct=Decimal("0.60"),
        direction_imbalance_warning_pct=Decimal("0.70"),
        direction_imbalance_danger_pct=Decimal("0.85"),
        max_average_leverage=Decimal("5.0"),
        max_single_leverage=Decimal("20.0"),
        max_herfindahl_index=Decimal("0.35"),
        categories=[
            AssetCategory("Layer1", {"BTCUSDT", "ETHUSDT", "SOLUSDT"}),
            AssetCategory("DeFi", {"UNIUSDT", "AAVEUSDT", "COMPUSDT"}),
            AssetCategory("Meme", {"DOGEUSDT", "SHIBUSDT"}),
        ],
        correlation_groups={
            "BTC_correlated": {"BTCUSDT", "ETHUSDT"},
        },
    )


@pytest.fixture
def balanced_positions() -> List[PositionInfo]:
    """Well-balanced portfolio positions."""
    return [
        PositionInfo(
            symbol="BTCUSDT",
            quantity=Decimal("0.1"),
            market_value=Decimal("5000"),
            side="LONG",
        ),
        PositionInfo(
            symbol="ETHUSDT",
            quantity=Decimal("1"),
            market_value=Decimal("3000"),
            side="LONG",
        ),
        PositionInfo(
            symbol="SOLUSDT",
            quantity=Decimal("10"),
            market_value=Decimal("1000"),
            side="LONG",
        ),
        PositionInfo(
            symbol="UNIUSDT",
            quantity=Decimal("100"),
            market_value=Decimal("1000"),
            side="SHORT",
        ),
    ]


@pytest.fixture
def monitor(config: ConcentrationConfig) -> ConcentrationMonitor:
    """Concentration monitor for testing."""
    return ConcentrationMonitor(config=config)


class TestBasicAnalysis:
    """Test basic concentration analysis."""

    def test_analyze_balanced_portfolio(
        self, monitor: ConcentrationMonitor, balanced_positions: List[PositionInfo]
    ):
        """Test analysis of a balanced portfolio."""
        snapshot = monitor.analyze(balanced_positions)

        assert snapshot is not None
        assert snapshot.position_count == 4
        assert snapshot.total_portfolio_value == Decimal("10000")

        # Check position concentrations
        assert snapshot.position_concentrations["BTCUSDT"] == Decimal("0.5")  # 50%
        assert snapshot.position_concentrations["ETHUSDT"] == Decimal("0.3")  # 30%

        # Check largest position
        assert snapshot.largest_position_symbol == "BTCUSDT"
        assert snapshot.largest_position_pct == Decimal("0.5")

    def test_analyze_empty_portfolio(self, monitor: ConcentrationMonitor):
        """Test analysis of empty portfolio."""
        snapshot = monitor.analyze([])

        assert snapshot is not None
        assert snapshot.position_count == 0
        assert len(snapshot.position_concentrations) == 0

    def test_category_concentration(
        self, monitor: ConcentrationMonitor, balanced_positions: List[PositionInfo]
    ):
        """Test category concentration calculation."""
        snapshot = monitor.analyze(balanced_positions)

        # Layer1 = BTC (50%) + ETH (30%) + SOL (10%) = 90%
        assert "Layer1" in snapshot.category_concentrations
        assert snapshot.category_concentrations["Layer1"] == Decimal("0.9")

        # DeFi = UNI (10%) = 10%
        # Note: UNI is in DeFi category
        assert "DeFi" in snapshot.category_concentrations


class TestSinglePositionAlerts:
    """Test single position concentration alerts."""

    def test_warning_alert_triggered(self, config: ConcentrationConfig):
        """Test warning alert when position exceeds warning threshold."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),  # 25% of 10000
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("7500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        # BTC at 25% should trigger warning (threshold 20%)
        btc_alerts = [
            a for a in alerts
            if a.alert_type == ConcentrationAlertType.SINGLE_POSITION
            and "BTCUSDT" in a.affected_symbols
        ]
        assert len(btc_alerts) >= 1
        assert btc_alerts[0].level == ConcentrationLevel.ELEVATED

    def test_danger_alert_triggered(self, config: ConcentrationConfig):
        """Test danger alert when position exceeds danger threshold."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("3500"),  # 35% of 10000
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("6500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        btc_alerts = [
            a for a in alerts
            if a.alert_type == ConcentrationAlertType.SINGLE_POSITION
            and "BTCUSDT" in a.affected_symbols
        ]
        assert len(btc_alerts) >= 1
        assert btc_alerts[0].level == ConcentrationLevel.HIGH

    def test_critical_alert_triggered(self, config: ConcentrationConfig):
        """Test critical alert when position exceeds critical threshold."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("4500"),  # 45% of 10000
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        btc_alerts = [
            a for a in alerts
            if a.alert_type == ConcentrationAlertType.SINGLE_POSITION
            and "BTCUSDT" in a.affected_symbols
        ]
        assert len(btc_alerts) >= 1
        assert btc_alerts[0].level == ConcentrationLevel.CRITICAL


class TestCategoryAlerts:
    """Test category/sector concentration alerts."""

    def test_category_warning_alert(self, config: ConcentrationConfig):
        """Test category warning when sector exposure too high."""
        # All positions in Layer1 category (35%)
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("1500"),
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("1500"),
                side="LONG",
            ),
            PositionInfo(
                symbol="SOLUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("500"),  # Total Layer1 = 35%
                side="LONG",
            ),
            PositionInfo(
                symbol="UNIUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("6500"),  # DeFi
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        sector_alerts = [
            a for a in alerts if a.alert_type == ConcentrationAlertType.SECTOR
        ]
        # Layer1 at 35% should trigger warning (threshold 30%)
        layer1_alerts = [a for a in sector_alerts if "Layer1" in a.message]
        assert len(layer1_alerts) >= 1


class TestCorrelationGroups:
    """Test correlation group monitoring."""

    def test_correlation_group_warning(self, config: ConcentrationConfig):
        """Test warning when correlated assets exceed threshold."""
        # BTC + ETH are in BTC_correlated group
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),  # 25%
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2000"),  # 20% -> total 45%
                side="LONG",
            ),
            PositionInfo(
                symbol="UNIUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        corr_alerts = [
            a for a in alerts if a.alert_type == ConcentrationAlertType.CORRELATION
        ]
        # BTC_correlated at 45% should trigger warning (threshold 40%)
        assert len(corr_alerts) >= 1
        assert corr_alerts[0].level == ConcentrationLevel.ELEVATED


class TestDirectionBalance:
    """Test direction (long/short) balance monitoring."""

    def test_direction_imbalance_warning(self, config: ConcentrationConfig):
        """Test warning when direction is too imbalanced."""
        # 80% long, 20% short
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("4000"),
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("4000"),
                side="LONG",
            ),
            PositionInfo(
                symbol="SOLUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2000"),
                side="SHORT",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)

        # Check direction metrics
        assert snapshot.long_exposure_pct == Decimal("0.8")
        assert snapshot.short_exposure_pct == Decimal("0.2")

        alerts = monitor.check_alerts(snapshot)

        direction_alerts = [
            a for a in alerts if a.alert_type == ConcentrationAlertType.DIRECTION
        ]
        # 80% long should trigger warning (threshold 70%)
        assert len(direction_alerts) >= 1
        assert direction_alerts[0].level == ConcentrationLevel.ELEVATED


class TestLeverageMonitoring:
    """Test leverage concentration monitoring."""

    def test_high_average_leverage_alert(self, config: ConcentrationConfig):
        """Test alert when average leverage is too high."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),
                leverage=Decimal("10"),
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),
                leverage=Decimal("8"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)

        # Average leverage = 9x
        assert snapshot.average_leverage == Decimal("9")

        alerts = monitor.check_alerts(snapshot)

        leverage_alerts = [
            a for a in alerts if a.alert_type == ConcentrationAlertType.LEVERAGE
        ]
        # 9x average exceeds 5x limit
        assert len(leverage_alerts) >= 1

    def test_max_leverage_alert(self, config: ConcentrationConfig):
        """Test alert when single position leverage is too high."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),
                leverage=Decimal("25"),  # Exceeds 20x limit
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),
                leverage=Decimal("2"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        snapshot = monitor.analyze(positions)
        alerts = monitor.check_alerts(snapshot)

        leverage_alerts = [
            a for a in alerts
            if a.alert_type == ConcentrationAlertType.LEVERAGE
            and a.level == ConcentrationLevel.CRITICAL
        ]
        assert len(leverage_alerts) >= 1


class TestDiversityMetrics:
    """Test portfolio diversity metrics."""

    def test_herfindahl_index_calculation(self, monitor: ConcentrationMonitor):
        """Test Herfindahl index calculation."""
        # Equal weights: HHI = n * (1/n)^2 = 1/n
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),
                side="LONG",
            ),
            PositionInfo(
                symbol="SOLUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),
                side="LONG",
            ),
            PositionInfo(
                symbol="UNIUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("2500"),
                side="LONG",
            ),
        ]

        snapshot = monitor.analyze(positions)

        # HHI for 4 equal positions = 4 * (0.25)^2 = 0.25
        assert snapshot.herfindahl_index == Decimal("0.25")
        # Effective positions = 1/0.25 = 4
        assert snapshot.effective_positions == Decimal("4")

    def test_concentrated_portfolio_hhi(self, monitor: ConcentrationMonitor):
        """Test HHI for concentrated portfolio."""
        # One position dominates
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("9000"),  # 90%
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("1000"),  # 10%
                side="LONG",
            ),
        ]

        snapshot = monitor.analyze(positions)

        # HHI = 0.9^2 + 0.1^2 = 0.81 + 0.01 = 0.82
        assert snapshot.herfindahl_index == Decimal("0.82")
        # Effective positions ≈ 1.22
        assert float(snapshot.effective_positions) == pytest.approx(1.22, rel=0.01)


class TestCategoryManagement:
    """Test category management functions."""

    def test_add_category(self, monitor: ConcentrationMonitor):
        """Test adding new category."""
        new_category = AssetCategory(
            name="Gaming",
            symbols={"AXSUSDT", "SANDUSDT"},
            max_exposure_pct=Decimal("0.20"),
        )

        monitor.add_category(new_category)

        # Verify category added
        assert any(c.name == "Gaming" for c in monitor.config.categories)

    def test_add_symbol_to_category(self, monitor: ConcentrationMonitor):
        """Test adding symbol to existing category."""
        result = monitor.add_symbol_to_category("AVAXUSDT", "Layer1")

        assert result is True
        layer1 = next(c for c in monitor.config.categories if c.name == "Layer1")
        assert "AVAXUSDT" in layer1.symbols

    def test_add_correlation_group(self, monitor: ConcentrationMonitor):
        """Test adding correlation group."""
        monitor.add_correlation_group("ETH_killers", {"SOLUSDT", "AVAXUSDT", "NEARUSDT"})

        assert "ETH_killers" in monitor.config.correlation_groups
        assert monitor.config.correlation_groups["ETH_killers"] == {
            "SOLUSDT", "AVAXUSDT", "NEARUSDT"
        }


class TestAlertManagement:
    """Test alert management functions."""

    def test_acknowledge_alert(self, config: ConcentrationConfig):
        """Test acknowledging alerts."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),  # 50%
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5000"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        monitor.analyze(positions)
        alerts = monitor.check_alerts()

        assert len(alerts) > 0
        assert len(monitor.active_alerts) > 0

        # Acknowledge all
        count = monitor.acknowledge_all_alerts()

        assert count > 0
        assert len(monitor.active_alerts) == 0  # No unacknowledged alerts

    def test_current_level(self, config: ConcentrationConfig):
        """Test current concentration level property."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("4500"),  # 45% - CRITICAL
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("5500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        monitor.analyze(positions)
        monitor.check_alerts()

        assert monitor.current_level == ConcentrationLevel.CRITICAL


class TestRebalancingSuggestions:
    """Test rebalancing suggestions."""

    def test_reduce_suggestions(self, config: ConcentrationConfig):
        """Test suggestions to reduce oversized positions."""
        positions = [
            PositionInfo(
                symbol="BTCUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("3500"),  # 35%
                side="LONG",
            ),
            PositionInfo(
                symbol="ETHUSDT",
                quantity=Decimal("1"),
                market_value=Decimal("6500"),
                side="LONG",
            ),
        ]

        monitor = ConcentrationMonitor(config=config)
        monitor.analyze(positions)
        suggestions = monitor.get_rebalancing_suggestions()

        # Should suggest reducing BTC
        btc_suggestions = [s for s in suggestions if s["symbol"] == "BTCUSDT"]
        assert len(btc_suggestions) >= 1
        assert btc_suggestions[0]["action"] == "REDUCE"


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_tracking(
        self, monitor: ConcentrationMonitor, balanced_positions: List[PositionInfo]
    ):
        """Test that statistics are tracked."""
        monitor.analyze(balanced_positions)
        monitor.check_alerts()

        stats = monitor.get_statistics()

        assert stats["total_checks"] == 1
        assert "current_level" in stats
        assert "active_alerts" in stats

    def test_concentration_summary(
        self, monitor: ConcentrationMonitor, balanced_positions: List[PositionInfo]
    ):
        """Test concentration summary output."""
        monitor.analyze(balanced_positions)
        summary = monitor.get_concentration_summary()

        assert summary is not None
        assert "timestamp" in summary
        assert "largest_position" in summary
        assert "direction" in summary
        assert "leverage" in summary
        assert "diversity" in summary
        assert "categories" in summary


class TestConfigUpdates:
    """Test configuration updates."""

    def test_update_config(self, monitor: ConcentrationMonitor):
        """Test updating configuration."""
        monitor.update_config(single_position_warning_pct=Decimal("0.25"))

        assert monitor.config.single_position_warning_pct == Decimal("0.25")

    def test_enable_disable(self, monitor: ConcentrationMonitor):
        """Test enabling and disabling monitor."""
        monitor.disable()
        assert monitor.config.enabled is False

        monitor.enable()
        assert monitor.config.enabled is True
