"""
Tests for Discord Bot Embeds.
"""

from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import MagicMock

import pytest

from src.discord_bot.embeds import (
    build_alert_embed,
    build_bot_detail_embed,
    build_bot_list_embed,
    build_circuit_break_embed,
    build_dashboard_embed,
    build_risk_status_embed,
    build_trade_embed,
    format_number,
    format_percentage,
    format_profit,
    get_risk_color,
    get_risk_emoji,
    get_state_color,
    get_status_emoji,
)
from tests.discord_bot import MockBotInfo, MockBotState, MockBotType, create_test_bots


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestFormatNumber:
    """Tests for format_number function."""

    def test_format_number_basic(self):
        """Test basic number formatting."""
        assert format_number(1234.56) == "1,234.56"
        assert format_number(1000000) == "1,000,000.00"

    def test_format_number_decimals(self):
        """Test decimal places."""
        assert format_number(1234.5678, 4) == "1,234.5678"
        assert format_number(1234.5678, 0) == "1,235"

    def test_format_number_none(self):
        """Test None value."""
        assert format_number(None) == "N/A"

    def test_format_number_decimal_type(self):
        """Test Decimal type."""
        assert format_number(Decimal("1234.56")) == "1,234.56"


class TestFormatProfit:
    """Tests for format_profit function."""

    def test_format_profit_positive(self):
        """Test positive profit."""
        assert format_profit(123.4567) == "+123.4567"

    def test_format_profit_negative(self):
        """Test negative profit."""
        assert format_profit(-123.4567) == "-123.4567"

    def test_format_profit_zero(self):
        """Test zero profit."""
        assert format_profit(0) == "+0.0000"

    def test_format_profit_none(self):
        """Test None value."""
        assert format_profit(None) == "N/A"


class TestFormatPercentage:
    """Tests for format_percentage function."""

    def test_format_percentage_positive(self):
        """Test positive percentage."""
        assert format_percentage(0.05) == "+5.00%"

    def test_format_percentage_negative(self):
        """Test negative percentage."""
        assert format_percentage(-0.03) == "-3.00%"

    def test_format_percentage_none(self):
        """Test None value."""
        assert format_percentage(None) == "N/A"


class TestStatusEmoji:
    """Tests for get_status_emoji function."""

    def test_running_emoji(self):
        """Test running state emoji."""
        state = MockBotState("running")
        assert get_status_emoji(state) == "🟢"

    def test_paused_emoji(self):
        """Test paused state emoji."""
        state = MockBotState("paused")
        assert get_status_emoji(state) == "🟡"

    def test_stopped_emoji(self):
        """Test stopped state emoji."""
        state = MockBotState("stopped")
        assert get_status_emoji(state) == "🔴"

    def test_error_emoji(self):
        """Test error state emoji."""
        state = MockBotState("error")
        assert get_status_emoji(state) == "❌"

    def test_unknown_emoji(self):
        """Test unknown state emoji."""
        state = MockBotState("unknown")
        assert get_status_emoji(state) == "❓"


class TestRiskEmoji:
    """Tests for get_risk_emoji function."""

    def test_normal_emoji(self):
        """Test normal level emoji."""
        level = MagicMock()
        level.name = "NORMAL"
        assert get_risk_emoji(level) == "🟢"

    def test_warning_emoji(self):
        """Test warning level emoji."""
        level = MagicMock()
        level.name = "WARNING"
        assert get_risk_emoji(level) == "🟡"

    def test_danger_emoji(self):
        """Test danger level emoji."""
        level = MagicMock()
        level.name = "DANGER"
        assert get_risk_emoji(level) == "🔴"

    def test_circuit_break_emoji(self):
        """Test circuit break emoji."""
        level = MagicMock()
        level.name = "CIRCUIT_BREAK"
        assert get_risk_emoji(level) == "⛔"


# =============================================================================
# Bot Embed Tests
# =============================================================================


class TestBotDetailEmbed:
    """Tests for build_bot_detail_embed function."""

    def test_basic_embed(self):
        """Test basic bot detail embed."""
        bot_info = MockBotInfo(
            bot_id="test_bot_1",
            symbol="BTCUSDT",
            state=MockBotState("running"),
            bot_type=MockBotType("grid"),
        )

        embed = build_bot_detail_embed(bot_info)

        assert "test_bot_1" in embed.title
        assert embed.color.value == 0x2ECC71  # Green for running

    def test_embed_with_stats(self):
        """Test embed with statistics."""
        bot_info = MockBotInfo(
            bot_id="test_bot_1",
            symbol="BTCUSDT",
            state=MockBotState("running"),
        )

        stats = {
            "total_profit": 123.45,
            "trade_count": 50,
            "profit_rate": 0.05,
            "win_rate": 0.65,
        }

        embed = build_bot_detail_embed(bot_info, stats=stats)

        # Check stats are in embed fields
        field_values = [f.value for f in embed.fields]
        assert any("123.45" in str(v) for v in field_values)

    def test_embed_with_instance_status(self):
        """Test embed with instance status."""
        bot_info = MockBotInfo(
            bot_id="test_bot_1",
            symbol="BTCUSDT",
            state=MockBotState("running"),
        )

        instance_status = {
            "lower_price": 40000,
            "upper_price": 50000,
            "grid_count": 20,
            "pending_buy_orders": 5,
            "pending_sell_orders": 5,
        }

        embed = build_bot_detail_embed(bot_info, instance_status=instance_status)

        field_names = [f.name for f in embed.fields]
        assert any("Grid" in name for name in field_names)


class TestBotListEmbed:
    """Tests for build_bot_list_embed function."""

    def test_empty_list(self):
        """Test embed with empty bot list."""
        embed = build_bot_list_embed([])

        assert "No bots" in embed.description

    def test_bot_list(self):
        """Test embed with bots."""
        bots = create_test_bots(3)
        embed = build_bot_list_embed(bots)

        assert len(embed.fields) == 3
        assert "Running" in embed.description
        assert "Paused" in embed.description
        assert "Stopped" in embed.description

    def test_max_bots_limit(self):
        """Test max bots limit."""
        bots = create_test_bots(20)
        embed = build_bot_list_embed(bots, max_bots=5)

        assert len(embed.fields) == 5
        assert "Showing 5 of 20" in embed.footer.text


# =============================================================================
# Dashboard Embed Tests
# =============================================================================


class TestDashboardEmbed:
    """Tests for build_dashboard_embed function."""

    def test_dashboard_embed(self):
        """Test dashboard embed creation."""
        dashboard_data = MagicMock(
            summary=MagicMock(
                total_bots=5,
                running_bots=3,
                paused_bots=1,
                error_bots=1,
                total_investment=Decimal("10000"),
                total_value=Decimal("10500"),
                total_profit=Decimal("500"),
                total_profit_rate=Decimal("0.05"),
                today_profit=Decimal("50"),
                today_trades=10,
            )
        )

        embed = build_dashboard_embed(dashboard_data)

        assert "Dashboard" in embed.title
        field_names = [f.name for f in embed.fields]
        assert any("Bot" in name for name in field_names)
        assert any("Fund" in name for name in field_names)
        assert any("Today" in name for name in field_names)

    def test_dashboard_color_positive_profit(self):
        """Test dashboard color for positive profit."""
        dashboard_data = MagicMock(
            summary=MagicMock(
                total_bots=1,
                running_bots=1,
                paused_bots=0,
                error_bots=0,
                total_investment=Decimal("10000"),
                total_profit=Decimal("500"),
                today_profit=Decimal("50"),
                today_trades=10,
            )
        )

        embed = build_dashboard_embed(dashboard_data)
        assert embed.color.value == 0x2ECC71  # Green

    def test_dashboard_color_negative_profit(self):
        """Test dashboard color for negative profit."""
        dashboard_data = MagicMock(
            summary=MagicMock(
                total_bots=1,
                running_bots=1,
                paused_bots=0,
                error_bots=0,
                total_investment=Decimal("10000"),
                total_profit=Decimal("-500"),
                today_profit=Decimal("-50"),
                today_trades=10,
            )
        )

        embed = build_dashboard_embed(dashboard_data)
        assert embed.color.value == 0xE74C3C  # Red


# =============================================================================
# Trade Embed Tests
# =============================================================================


class TestTradeEmbed:
    """Tests for build_trade_embed function."""

    def test_buy_trade_embed(self):
        """Test buy trade embed."""
        trade = MagicMock(
            bot_id="bot_1",
            symbol="BTCUSDT",
            side="BUY",
            price=45000,
            quantity=0.01,
            amount=450,
            fee=0.45,
        )

        embed = build_trade_embed(trade)

        assert "Buy" in embed.title
        assert embed.color.value == 0x2ECC71  # Green

    def test_sell_trade_embed(self):
        """Test sell trade embed."""
        trade = MagicMock(
            bot_id="bot_1",
            symbol="BTCUSDT",
            side="SELL",
            price=46000,
            quantity=0.01,
            amount=460,
            profit=10,
            fee=0.46,
        )

        embed = build_trade_embed(trade)

        assert "Sell" in embed.title
        assert embed.color.value == 0xE74C3C  # Red

        # Check profit field exists
        field_names = [f.name for f in embed.fields]
        assert any("Profit" in name for name in field_names)


# =============================================================================
# Risk Embed Tests
# =============================================================================


class TestAlertEmbed:
    """Tests for build_alert_embed function."""

    def test_warning_alert(self):
        """Test warning alert embed."""
        level = MagicMock()
        level.name = "WARNING"
        alert = MagicMock(
            level=level,
            message="Drawdown exceeded 5%",
            metric="drawdown",
            current_value=0.06,
            threshold=0.05,
        )

        embed = build_alert_embed(alert)

        assert "Warning" in embed.title
        # discord.Color.yellow() = 0xFEE75C
        assert embed.color.value == 0xFEE75C

    def test_danger_alert(self):
        """Test danger alert embed."""
        level = MagicMock()
        level.name = "DANGER"
        alert = MagicMock(
            level=level,
            message="Daily loss exceeded limit",
            metric="daily_loss",
            current_value=0.12,
            threshold=0.10,
        )

        embed = build_alert_embed(alert)

        assert "Danger" in embed.title
        assert embed.color.value == 0xE74C3C  # Red


class TestCircuitBreakEmbed:
    """Tests for build_circuit_break_embed function."""

    def test_triggered_embed(self):
        """Test circuit breaker triggered embed."""
        embed = build_circuit_break_embed(
            triggered=True,
            reason="Daily loss exceeded 10%",
            triggered_at=datetime.now(timezone.utc),
            cooldown_until=datetime.now(timezone.utc),
        )

        assert "TRIGGERED" in embed.title
        assert embed.color.value == 0x992D22  # Dark red

    def test_reset_embed(self):
        """Test circuit breaker reset embed."""
        embed = build_circuit_break_embed(triggered=False)

        assert "Reset" in embed.title
        assert embed.color.value == 0x2ECC71  # Green


class TestRiskStatusEmbed:
    """Tests for build_risk_status_embed function."""

    def test_normal_status(self):
        """Test normal risk status embed."""
        level = MagicMock()
        level.name = "NORMAL"
        status = MagicMock(
            level=level,
            capital=MagicMock(
                total_capital=Decimal("100000"),
                initial_capital=Decimal("100000"),
                available_balance=Decimal("50000"),
            ),
            drawdown=MagicMock(
                drawdown_pct=Decimal("0.02"),
                max_drawdown_pct=Decimal("0.05"),
                peak_value=Decimal("102000"),
            ),
            circuit_breaker=MagicMock(is_triggered=False),
            daily_pnl=MagicMock(
                pnl=Decimal("500"),
                pnl_pct=Decimal("0.005"),
            ),
            active_alerts=[],
            statistics=MagicMock(
                total_checks=100,
                violations=0,
                circuit_breaker_triggers=0,
            ),
        )

        embed = build_risk_status_embed(status)

        assert "Risk Status" in embed.title
        assert embed.color.value == 0x2ECC71  # Green

    def test_circuit_breaker_triggered_status(self):
        """Test risk status with circuit breaker triggered."""
        level = MagicMock()
        level.name = "CIRCUIT_BREAK"
        status = MagicMock(
            level=level,
            capital=MagicMock(
                total_capital=Decimal("90000"),
                initial_capital=Decimal("100000"),
                available_balance=Decimal("45000"),
            ),
            drawdown=MagicMock(
                drawdown_pct=Decimal("0.10"),
                max_drawdown_pct=Decimal("0.10"),
                peak_value=Decimal("100000"),
            ),
            circuit_breaker=MagicMock(
                is_triggered=True,
                trigger_reason="Daily loss exceeded",
                cooldown_until=datetime.now(timezone.utc),
            ),
            active_alerts=[],
        )

        embed = build_risk_status_embed(status)

        # Check circuit breaker field shows triggered
        field_values = [f.value for f in embed.fields]
        assert any("TRIGGERED" in str(v) for v in field_values)
