"""
Unit tests for RSI-Grid v2 bot logic.

Tests:
- ExitReason.TIMEOUT_EXIT exists
- RSIGridPosition: entry_bar, highest_price, lowest_price, update_extremes()
- RSIGridConfig: trailing_stop_pct decimal validation
- Bot helpers: _update_volatility_baseline, _check_volatility_regime,
               _check_timeout_exit, _check_trailing_stop
"""

import pytest
from decimal import Decimal
from unittest.mock import MagicMock, AsyncMock, patch

from src.bots.rsi_grid.models import (
    ExitReason,
    RSIGridConfig,
    RSIGridPosition,
    PositionSide,
    RSIZone,
)


# =============================================================================
# Models Tests
# =============================================================================


class TestExitReason:
    """Test ExitReason enum."""

    def test_timeout_exit_exists(self):
        assert ExitReason.TIMEOUT_EXIT == "timeout_exit"

    def test_all_exit_reasons(self):
        expected = {
            "rsi_exit", "grid_profit", "stop_loss", "take_profit",
            "grid_rebuild", "trend_change", "timeout_exit", "manual", "bot_stop",
        }
        actual = {e.value for e in ExitReason}
        assert expected == actual


class TestRSIGridPosition:
    """Test RSIGridPosition v2 fields."""

    def _make_position(self, **kwargs) -> RSIGridPosition:
        defaults = dict(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            entry_price=Decimal("50000"),
            quantity=Decimal("0.01"),
            leverage=7,
        )
        defaults.update(kwargs)
        return RSIGridPosition(**defaults)

    def test_entry_bar_default(self):
        pos = self._make_position()
        assert pos.entry_bar == 0

    def test_entry_bar_custom(self):
        pos = self._make_position(entry_bar=42)
        assert pos.entry_bar == 42

    def test_highest_lowest_default(self):
        pos = self._make_position()
        assert pos.highest_price == Decimal("0")
        assert pos.lowest_price == Decimal("0")

    def test_max_min_price_none_when_zero(self):
        pos = self._make_position()
        assert pos.max_price is None
        assert pos.min_price is None

    def test_update_extremes_first_call(self):
        pos = self._make_position()
        pos.update_extremes(Decimal("51000"))
        assert pos.highest_price == Decimal("51000")
        assert pos.lowest_price == Decimal("51000")

    def test_update_extremes_tracks_both(self):
        pos = self._make_position()
        pos.update_extremes(Decimal("51000"))
        pos.update_extremes(Decimal("49000"))
        pos.update_extremes(Decimal("50500"))
        assert pos.highest_price == Decimal("51000")
        assert pos.lowest_price == Decimal("49000")

    def test_max_min_price_properties(self):
        pos = self._make_position()
        pos.update_extremes(Decimal("52000"))
        pos.update_extremes(Decimal("48000"))
        assert pos.max_price == Decimal("52000")
        assert pos.min_price == Decimal("48000")


class TestRSIGridConfig:
    """Test RSIGridConfig v2 fields."""

    def test_trailing_stop_pct_default(self):
        config = RSIGridConfig(symbol="BTCUSDT")
        assert config.trailing_stop_pct == Decimal("0.03")

    def test_trailing_stop_pct_decimal_conversion(self):
        config = RSIGridConfig(symbol="BTCUSDT", trailing_stop_pct=0.05)
        assert isinstance(config.trailing_stop_pct, Decimal)
        assert config.trailing_stop_pct == Decimal("0.05")

    def test_volatility_filter_defaults(self):
        config = RSIGridConfig(symbol="BTCUSDT")
        assert config.use_volatility_filter is True
        assert config.vol_atr_baseline_period == 100
        assert config.vol_ratio_low == 0.4
        assert config.vol_ratio_high == 2.0

    def test_max_hold_bars_default(self):
        config = RSIGridConfig(symbol="BTCUSDT")
        assert config.max_hold_bars == 16


# =============================================================================
# Bot Helper Tests (unit-test via direct method calls)
# =============================================================================


def _make_bot():
    """Create a minimal RSIGridBot for unit testing helpers."""
    from src.bots.rsi_grid.bot import RSIGridBot

    config = RSIGridConfig(symbol="BTCUSDT")

    with patch.object(RSIGridBot, '__init__', lambda self, *a, **kw: None):
        bot = RSIGridBot.__new__(RSIGridBot)

    # Manually set required attributes
    bot._config = config
    bot._current_bar = 0
    bot._atr_history = []
    bot._atr_baseline = None
    bot._position = None

    # Mock ATR calculator
    bot._atr_calc = MagicMock()
    bot._atr_calc.atr = Decimal("500")

    return bot


class TestUpdateVolatilityBaseline:
    """Test _update_volatility_baseline."""

    def test_accumulates_history(self):
        bot = _make_bot()
        for i in range(50):
            bot._update_volatility_baseline(Decimal("100"))
        assert len(bot._atr_history) == 50
        assert bot._atr_baseline is None  # Not enough (need 100)

    def test_calculates_baseline_at_threshold(self):
        bot = _make_bot()
        for i in range(100):
            bot._update_volatility_baseline(Decimal("200"))
        assert bot._atr_baseline == Decimal("200")

    def test_trims_history(self):
        bot = _make_bot()
        for i in range(150):
            bot._update_volatility_baseline(Decimal("100"))
        assert len(bot._atr_history) == 100


class TestCheckVolatilityRegime:
    """Test _check_volatility_regime."""

    def test_passes_when_disabled(self):
        bot = _make_bot()
        bot._config.use_volatility_filter = False
        assert bot._check_volatility_regime() is True

    def test_passes_when_no_baseline(self):
        bot = _make_bot()
        bot._atr_baseline = None
        assert bot._check_volatility_regime() is True

    def test_passes_normal_ratio(self):
        bot = _make_bot()
        bot._atr_baseline = Decimal("500")
        bot._atr_calc.atr = Decimal("500")  # ratio = 1.0
        assert bot._check_volatility_regime() is True

    def test_blocks_high_volatility(self):
        bot = _make_bot()
        bot._atr_baseline = Decimal("500")
        bot._atr_calc.atr = Decimal("1500")  # ratio = 3.0 > 2.0
        assert bot._check_volatility_regime() is False

    def test_blocks_low_volatility(self):
        bot = _make_bot()
        bot._atr_baseline = Decimal("500")
        bot._atr_calc.atr = Decimal("100")  # ratio = 0.2 < 0.4
        assert bot._check_volatility_regime() is False

    def test_boundary_low(self):
        bot = _make_bot()
        bot._atr_baseline = Decimal("1000")
        bot._atr_calc.atr = Decimal("400")  # ratio = 0.4 exactly
        assert bot._check_volatility_regime() is True

    def test_boundary_high(self):
        bot = _make_bot()
        bot._atr_baseline = Decimal("1000")
        bot._atr_calc.atr = Decimal("2000")  # ratio = 2.0 exactly
        assert bot._check_volatility_regime() is True


class TestCheckTimeoutExit:
    """Test _check_timeout_exit."""

    def test_no_position(self):
        bot = _make_bot()
        assert bot._check_timeout_exit(Decimal("50000")) is False

    def test_not_exceeded_bars(self):
        bot = _make_bot()
        bot._current_bar = 10
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=5,
        )
        # held 5 bars < 16
        assert bot._check_timeout_exit(Decimal("49000")) is False

    def test_exceeded_but_profitable_long(self):
        bot = _make_bot()
        bot._current_bar = 100
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=10,
        )
        # price > entry → profitable, should NOT timeout
        assert bot._check_timeout_exit(Decimal("51000")) is False

    def test_exceeded_and_losing_long(self):
        bot = _make_bot()
        bot._current_bar = 100
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=10,
        )
        # price < entry → losing
        assert bot._check_timeout_exit(Decimal("49000")) is True

    def test_exceeded_and_losing_short(self):
        bot = _make_bot()
        bot._current_bar = 100
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=10,
        )
        # price > entry → losing for short
        assert bot._check_timeout_exit(Decimal("51000")) is True

    def test_exceeded_but_profitable_short(self):
        bot = _make_bot()
        bot._current_bar = 100
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=10,
        )
        assert bot._check_timeout_exit(Decimal("49000")) is False

    def test_max_hold_zero_disables(self):
        bot = _make_bot()
        bot._config.max_hold_bars = 0
        bot._current_bar = 1000
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7, entry_bar=0,
        )
        assert bot._check_timeout_exit(Decimal("40000")) is False


class TestCheckTrailingStop:
    """Test _check_trailing_stop."""

    def test_no_position(self):
        bot = _make_bot()
        assert bot._check_trailing_stop(Decimal("50000")) is False

    def test_long_no_trigger(self):
        bot = _make_bot()
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7,
        )
        bot._position.update_extremes(Decimal("52000"))
        # stop_price = 52000 * 0.97 = 50440
        # current 51000 > 50440 → no trigger
        assert bot._check_trailing_stop(Decimal("51000")) is False

    def test_long_triggers(self):
        bot = _make_bot()
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7,
        )
        bot._position.update_extremes(Decimal("52000"))
        # stop_price = 52000 * 0.97 = 50440
        # current 50000 <= 50440 → trigger
        assert bot._check_trailing_stop(Decimal("50000")) is True

    def test_short_no_trigger(self):
        bot = _make_bot()
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7,
        )
        bot._position.update_extremes(Decimal("48000"))
        # stop_price = 48000 * 1.03 = 49440
        # current 49000 < 49440 → no trigger
        assert bot._check_trailing_stop(Decimal("49000")) is False

    def test_short_triggers(self):
        bot = _make_bot()
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.SHORT,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7,
        )
        bot._position.update_extremes(Decimal("48000"))
        # stop_price = 48000 * 1.03 = 49440
        # current 50000 >= 49440 → trigger
        assert bot._check_trailing_stop(Decimal("50000")) is True

    def test_no_extreme_tracked_yet(self):
        bot = _make_bot()
        bot._position = RSIGridPosition(
            symbol="BTCUSDT", side=PositionSide.LONG,
            entry_price=Decimal("50000"), quantity=Decimal("0.01"),
            leverage=7,
        )
        # max_price is None (no update_extremes called)
        assert bot._check_trailing_stop(Decimal("40000")) is False
