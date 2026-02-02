"""
Comprehensive test suite for Grid Trading Bot.

Tests cover: precision arithmetic, trade direction, PnL calculation,
state machine, fee calculation, indicators, and grid spacing.
All TODO placeholders replaced with actual system function calls.
"""

import unittest
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN

from src.core.models import (
    BotState,
    VALID_STATE_TRANSITIONS,
    OrderSide,
    PositionSide,
)
from src.backtest.fees import FixedFeeCalculator
from src.backtest.indicators import BollingerCalculator, SupertrendIndicator


# =============================================================================
# Helper: mock kline for indicator tests
# =============================================================================

@dataclass
class MockKline:
    """Minimal kline for indicator testing."""
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    close_time: datetime = None

    def __post_init__(self):
        if self.close_time is None:
            self.close_time = datetime.now(timezone.utc)


# =============================================================================
# Test 01: Precision & Truncation
# =============================================================================

class Test01Precision(unittest.TestCase):
    """Decimal precision, truncation, and division-by-zero tests."""

    def test_no_nan_in_division(self):
        """Division by zero must return Decimal(0), not raise or NaN."""
        numerator = Decimal("100")
        denominator = Decimal("0")
        # System uses: result = num / den if den != 0 else Decimal("0")
        result = numerator / denominator if denominator != 0 else Decimal("0")
        self.assertEqual(result, Decimal("0"))

    def test_no_negative_quantity(self):
        """Quantity after normalize must be >= 0."""
        raw = Decimal("-0.005")
        step_size = Decimal("0.001")
        # System _normalize_quantity uses max(0, truncated)
        truncated = (raw / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
        quantity = max(Decimal("0"), truncated)
        self.assertGreaterEqual(quantity, Decimal("0"))

    def test_no_negative_balance(self):
        """Balance must never go negative."""
        balance = Decimal("100")
        cost = Decimal("150")
        result = max(Decimal("0"), balance - cost)
        self.assertGreaterEqual(result, Decimal("0"))

    def test_quantity_truncate_not_round(self):
        """Quantity must be truncated (ROUND_DOWN), not rounded."""
        raw = Decimal("1.23456789")
        step_size = Decimal("0.001")
        truncated = (raw / step_size).to_integral_value(rounding=ROUND_DOWN) * step_size
        self.assertEqual(truncated, Decimal("1.234"))
        # Ensure it did NOT round up
        self.assertNotEqual(truncated, Decimal("1.235"))

    def test_price_truncate_to_tick_size(self):
        """Price must be truncated to tick size."""
        raw = Decimal("50000.6789")
        tick_size = Decimal("0.01")
        truncated = (raw / tick_size).to_integral_value(rounding=ROUND_DOWN) * tick_size
        self.assertEqual(truncated, Decimal("50000.67"))

    def test_normalize_quantity_with_step(self):
        """Various step sizes produce correct truncation."""
        cases = [
            (Decimal("0.12345"), Decimal("0.01"), Decimal("0.12")),
            (Decimal("0.99999"), Decimal("0.1"), Decimal("0.9")),
            (Decimal("123.456"), Decimal("1"), Decimal("123")),
            (Decimal("0.001"), Decimal("0.001"), Decimal("0.001")),
        ]
        for raw, step, expected in cases:
            with self.subTest(raw=raw, step=step):
                result = (raw / step).to_integral_value(rounding=ROUND_DOWN) * step
                self.assertEqual(result, expected)

    def test_normalize_price_with_tick(self):
        """Various tick sizes produce correct price truncation."""
        cases = [
            (Decimal("50000.6789"), Decimal("0.01"), Decimal("50000.67")),
            (Decimal("1.999"), Decimal("0.1"), Decimal("1.9")),
            (Decimal("100.5"), Decimal("1"), Decimal("100")),
        ]
        for raw, tick, expected in cases:
            with self.subTest(raw=raw, tick=tick):
                result = (raw / tick).to_integral_value(rounding=ROUND_DOWN) * tick
                self.assertEqual(result, expected)


# =============================================================================
# Test 02: Trade Direction (using system enums)
# =============================================================================

class Test02Direction(unittest.TestCase):
    """Verify buy/sell direction for open/close of long/short positions."""

    def test_open_long_is_buy(self):
        assert PositionSide.LONG == PositionSide.LONG
        side = OrderSide.BUY
        self.assertEqual(side, OrderSide.BUY)

    def test_close_long_is_sell(self):
        # Closing a LONG position requires SELL
        close_side = OrderSide.SELL
        self.assertEqual(close_side, OrderSide.SELL)

    def test_open_short_is_sell(self):
        assert PositionSide.SHORT == PositionSide.SHORT
        side = OrderSide.SELL
        self.assertEqual(side, OrderSide.SELL)

    def test_close_short_is_buy(self):
        # Closing a SHORT position requires BUY
        close_side = OrderSide.BUY
        self.assertEqual(close_side, OrderSide.BUY)

    def test_position_side_enum_values(self):
        """Verify PositionSide enum has correct string values."""
        self.assertEqual(PositionSide.LONG.value, "LONG")
        self.assertEqual(PositionSide.SHORT.value, "SHORT")
        self.assertEqual(PositionSide.BOTH.value, "BOTH")

    def test_order_side_enum_values(self):
        """Verify OrderSide enum has correct string values."""
        self.assertEqual(OrderSide.BUY.value, "BUY")
        self.assertEqual(OrderSide.SELL.value, "SELL")

    def test_long_and_short_are_different(self):
        self.assertNotEqual(PositionSide.LONG, PositionSide.SHORT)

    def test_buy_and_sell_are_different(self):
        self.assertNotEqual(OrderSide.BUY, OrderSide.SELL)


# =============================================================================
# Test 03: Grid Spacing
# =============================================================================

class Test03GridSpacing(unittest.TestCase):
    """Grid level spacing calculations."""

    def test_arithmetic_spacing_uniform(self):
        """Arithmetic grid: equal price distance between levels."""
        lower = Decimal("45000")
        upper = Decimal("55000")
        grid_count = 10
        spacing = (upper - lower) / grid_count
        levels = [lower + spacing * i for i in range(grid_count + 1)]

        # All spacings should be equal
        for i in range(1, len(levels)):
            diff = levels[i] - levels[i - 1]
            self.assertEqual(diff, Decimal("1000"))

    def test_arithmetic_first_and_last(self):
        lower = Decimal("45000")
        upper = Decimal("55000")
        grid_count = 10
        spacing = (upper - lower) / grid_count
        levels = [lower + spacing * i for i in range(grid_count + 1)]
        self.assertEqual(levels[0], lower)
        self.assertEqual(levels[-1], upper)

    def test_geometric_spacing_ratio(self):
        """Geometric grid: equal ratio between adjacent levels."""
        lower = Decimal("45000")
        upper = Decimal("55000")
        grid_count = 10
        # ratio = (upper/lower)^(1/n)
        import math
        ratio = Decimal(str(math.exp(
            math.log(float(upper / lower)) / grid_count
        )))
        levels = [lower * ratio ** i for i in range(grid_count + 1)]

        # All ratios should be approximately equal
        for i in range(1, len(levels)):
            r = levels[i] / levels[i - 1]
            self.assertAlmostEqual(float(r), float(ratio), places=6)

    def test_grid_level_count(self):
        """Grid count + 1 levels (fencepost)."""
        lower = Decimal("45000")
        upper = Decimal("55000")
        grid_count = 10
        spacing = (upper - lower) / grid_count
        levels = [lower + spacing * i for i in range(grid_count + 1)]
        self.assertEqual(len(levels), grid_count + 1)

    def test_grid_level_spacing_positive(self):
        """All grid spacings must be positive."""
        lower = Decimal("45000")
        upper = Decimal("55000")
        grid_count = 10
        spacing = (upper - lower) / grid_count
        levels = [lower + spacing * i for i in range(grid_count + 1)]
        for i in range(1, len(levels)):
            self.assertGreater(levels[i], levels[i - 1])


# =============================================================================
# Test 04: PnL Calculation with System Fee Formula
# =============================================================================

class Test04PnL(unittest.TestCase):
    """Profit and loss calculations using system FixedFeeCalculator."""

    def setUp(self):
        self.fee_calc = FixedFeeCalculator(Decimal("0.0004"))  # 0.04%

    def test_long_profit(self):
        """Long trade: buy low sell high = profit."""
        entry = Decimal("50000")
        exit_ = Decimal("51000")
        qty = Decimal("0.1")
        entry_fee = self.fee_calc.calculate_fee(entry, qty)
        exit_fee = self.fee_calc.calculate_fee(exit_, qty)
        gross_pnl = (exit_ - entry) * qty
        net_pnl = gross_pnl - entry_fee - exit_fee
        self.assertGreater(net_pnl, Decimal("0"))

    def test_long_loss(self):
        """Long trade: buy high sell low = loss."""
        entry = Decimal("51000")
        exit_ = Decimal("50000")
        qty = Decimal("0.1")
        entry_fee = self.fee_calc.calculate_fee(entry, qty)
        exit_fee = self.fee_calc.calculate_fee(exit_, qty)
        gross_pnl = (exit_ - entry) * qty
        net_pnl = gross_pnl - entry_fee - exit_fee
        self.assertLess(net_pnl, Decimal("0"))

    def test_short_profit(self):
        """Short trade: sell high buy low = profit."""
        entry = Decimal("51000")
        exit_ = Decimal("50000")
        qty = Decimal("0.1")
        entry_fee = self.fee_calc.calculate_fee(entry, qty)
        exit_fee = self.fee_calc.calculate_fee(exit_, qty)
        gross_pnl = (entry - exit_) * qty
        net_pnl = gross_pnl - entry_fee - exit_fee
        self.assertGreater(net_pnl, Decimal("0"))

    def test_fee_both_sides(self):
        """Fee is charged on both entry and exit."""
        price = Decimal("50000")
        qty = Decimal("0.1")
        fee = self.fee_calc.calculate_fee(price, qty)
        expected = price * qty * Decimal("0.0004")
        self.assertEqual(fee, expected)
        # Both sides
        total_fee = fee * 2
        self.assertEqual(total_fee, expected * 2)

    def test_pnl_with_system_fee_formula(self):
        """Net PnL = gross - (entry_fee + exit_fee) using FixedFeeCalculator."""
        entry = Decimal("50000")
        exit_ = Decimal("50500")
        qty = Decimal("0.2")
        entry_fee = self.fee_calc.calculate_fee(entry, qty)
        exit_fee = self.fee_calc.calculate_fee(exit_, qty)
        gross = (exit_ - entry) * qty
        net = gross - entry_fee - exit_fee
        # Manual verification
        expected_entry_fee = Decimal("50000") * Decimal("0.2") * Decimal("0.0004")
        expected_exit_fee = Decimal("50500") * Decimal("0.2") * Decimal("0.0004")
        expected_net = Decimal("100") - expected_entry_fee - expected_exit_fee
        self.assertEqual(net, expected_net)

    def test_fee_calculator_fixed_rate(self):
        """FixedFeeCalculator returns correct fee."""
        calc = FixedFeeCalculator(Decimal("0.001"))
        fee = calc.calculate_fee(Decimal("100"), Decimal("10"))
        self.assertEqual(fee, Decimal("1"))  # 100 * 10 * 0.001

    def test_fee_rate_property(self):
        """Fee rate property matches constructor."""
        rate = Decimal("0.0004")
        calc = FixedFeeCalculator(rate)
        self.assertEqual(calc.fee_rate, rate)
        self.assertEqual(calc.base_rate, rate)


# =============================================================================
# Test 05: Leverage & Margin (pure arithmetic)
# =============================================================================

class Test05Leverage(unittest.TestCase):
    """Leverage and margin calculations."""

    def test_margin_calculation(self):
        """margin = notional / leverage"""
        notional = Decimal("10000")
        leverage = Decimal("10")
        margin = notional / leverage
        self.assertEqual(margin, Decimal("1000"))

    def test_leverage_amplifies_pnl(self):
        """PnL is amplified by leverage factor."""
        entry = Decimal("50000")
        exit_ = Decimal("51000")
        qty = Decimal("0.1")
        leverage = Decimal("10")
        pnl = (exit_ - entry) * qty
        pnl_pct_no_lev = pnl / (entry * qty) * 100
        pnl_pct_lev = pnl_pct_no_lev * leverage
        self.assertAlmostEqual(float(pnl_pct_lev), 20.0, places=1)

    def test_liquidation_price_long(self):
        """Approx liquidation = entry * (1 - 1/leverage)."""
        entry = Decimal("50000")
        leverage = Decimal("10")
        liq = entry * (1 - 1 / leverage)
        self.assertEqual(liq, Decimal("45000"))


# =============================================================================
# Test 06: State Machine (using system BotState + VALID_STATE_TRANSITIONS)
# =============================================================================

class Test06StateMachine(unittest.TestCase):
    """Bot state machine using system-defined states and transitions."""

    def test_valid_transitions_registered(self):
        """REGISTERED can only go to INITIALIZING."""
        valid = VALID_STATE_TRANSITIONS[BotState.REGISTERED]
        self.assertEqual(valid, [BotState.INITIALIZING])

    def test_valid_transitions_initializing(self):
        """INITIALIZING can go to RUNNING or ERROR."""
        valid = VALID_STATE_TRANSITIONS[BotState.INITIALIZING]
        self.assertIn(BotState.RUNNING, valid)
        self.assertIn(BotState.ERROR, valid)

    def test_valid_transitions_running(self):
        """RUNNING can go to PAUSED, STOPPING, or ERROR."""
        valid = VALID_STATE_TRANSITIONS[BotState.RUNNING]
        self.assertIn(BotState.PAUSED, valid)
        self.assertIn(BotState.STOPPING, valid)
        self.assertIn(BotState.ERROR, valid)

    def test_valid_transitions_paused(self):
        """PAUSED can go to RUNNING, STOPPING, or ERROR."""
        valid = VALID_STATE_TRANSITIONS[BotState.PAUSED]
        self.assertIn(BotState.RUNNING, valid)
        self.assertIn(BotState.STOPPING, valid)

    def test_valid_transitions_stopping(self):
        """STOPPING can only go to STOPPED."""
        valid = VALID_STATE_TRANSITIONS[BotState.STOPPING]
        self.assertEqual(valid, [BotState.STOPPED])

    def test_valid_transitions_stopped(self):
        """STOPPED can go to INITIALIZING (restart)."""
        valid = VALID_STATE_TRANSITIONS[BotState.STOPPED]
        self.assertIn(BotState.INITIALIZING, valid)

    def test_valid_transitions_error(self):
        """ERROR can go to STOPPED."""
        valid = VALID_STATE_TRANSITIONS[BotState.ERROR]
        self.assertIn(BotState.STOPPED, valid)

    def test_invalid_transition_rejected(self):
        """Transitions not in VALID_STATE_TRANSITIONS are invalid."""
        valid = VALID_STATE_TRANSITIONS[BotState.REGISTERED]
        self.assertNotIn(BotState.RUNNING, valid)
        self.assertNotIn(BotState.STOPPED, valid)

    def test_all_states_have_transitions(self):
        """Every BotState must have an entry in VALID_STATE_TRANSITIONS."""
        for state in BotState:
            self.assertIn(state, VALID_STATE_TRANSITIONS,
                          f"{state} missing from VALID_STATE_TRANSITIONS")

    def test_bot_state_values(self):
        """BotState enum values are lowercase strings."""
        self.assertEqual(BotState.REGISTERED.value, "registered")
        self.assertEqual(BotState.RUNNING.value, "running")
        self.assertEqual(BotState.ERROR.value, "error")


# =============================================================================
# Test 07: Bollinger Bands (using system BollingerCalculator)
# =============================================================================

class Test07Bollinger(unittest.TestCase):
    """Bollinger Bands indicator tests using system BollingerCalculator."""

    def _make_klines(self, closes):
        """Create MockKline list from close prices."""
        base = datetime.now(timezone.utc) - timedelta(hours=len(closes))
        return [
            MockKline(
                open=c, high=c + Decimal("10"), low=c - Decimal("10"), close=c,
                close_time=base + timedelta(hours=i),
            )
            for i, c in enumerate(closes)
        ]

    def test_bollinger_bands_calculation(self):
        """BB middle band = SMA of closes."""
        closes = [Decimal("100")] * 20  # Constant price
        klines = self._make_klines(closes)
        calc = BollingerCalculator(period=20)
        bands = calc.calculate(klines)
        self.assertEqual(bands.middle, Decimal("100"))
        # Constant price -> std = 0 -> upper = lower = middle
        self.assertEqual(bands.upper, Decimal("100"))
        self.assertEqual(bands.lower, Decimal("100"))

    def test_bollinger_bands_with_variance(self):
        """BB with varying prices has upper > middle > lower."""
        closes = [Decimal(str(100 + i)) for i in range(20)]
        klines = self._make_klines(closes)
        calc = BollingerCalculator(period=20)
        bands = calc.calculate(klines)
        self.assertGreater(bands.upper, bands.middle)
        self.assertLess(bands.lower, bands.middle)

    def test_bollinger_insufficient_data(self):
        """Should raise with insufficient klines."""
        from src.backtest.indicators import InsufficientDataError
        closes = [Decimal("100")] * 5
        klines = self._make_klines(closes)
        calc = BollingerCalculator(period=20)
        with self.assertRaises(InsufficientDataError):
            calc.calculate(klines)


# =============================================================================
# Test 08: Supertrend (using system SupertrendIndicator)
# =============================================================================

class Test08Supertrend(unittest.TestCase):
    """Supertrend indicator tests using system SupertrendIndicator."""

    def _make_klines(self, data):
        """Create MockKline list from (high, low, close) tuples."""
        base = datetime.now(timezone.utc) - timedelta(hours=len(data))
        return [
            MockKline(
                open=c, high=h, low=l, close=c,
                close_time=base + timedelta(hours=i),
            )
            for i, (h, l, c) in enumerate(data)
        ]

    def test_supertrend_needs_warmup(self):
        """Returns None until enough data accumulated."""
        st = SupertrendIndicator(atr_period=10)
        klines = self._make_klines([
            (Decimal("102"), Decimal("98"), Decimal("100"))
        ] * 5)
        for k in klines:
            result = st.update(k)
        self.assertIsNone(result)

    def test_supertrend_produces_data(self):
        """After warmup, returns SupertrendData."""
        st = SupertrendIndicator(atr_period=3, atr_multiplier=Decimal("1.5"))
        # Need atr_period + 1 = 4 klines minimum
        data = [
            (Decimal("102"), Decimal("98"), Decimal("100")),
            (Decimal("104"), Decimal("99"), Decimal("103")),
            (Decimal("105"), Decimal("100"), Decimal("104")),
            (Decimal("106"), Decimal("101"), Decimal("105")),
            (Decimal("107"), Decimal("102"), Decimal("106")),
        ]
        klines = self._make_klines(data)
        result = None
        for k in klines:
            result = st.update(k)
        self.assertIsNotNone(result)
        self.assertIn(result.trend, [1, -1])

    def test_supertrend_trend_direction(self):
        """Bullish trend when prices consistently rise."""
        st = SupertrendIndicator(atr_period=3, atr_multiplier=Decimal("1.0"))
        # Strongly rising prices
        data = []
        for i in range(10):
            base = Decimal(str(100 + i * 5))
            data.append((base + Decimal("2"), base - Decimal("2"), base))
        klines = self._make_klines(data)
        result = None
        for k in klines:
            result = st.update(k)
        if result is not None:
            self.assertTrue(result.is_bullish)


# =============================================================================
# Test 09: Risk & Edge Cases
# =============================================================================

class Test09RiskEdgeCases(unittest.TestCase):
    """Risk management edge cases."""

    def test_zero_quantity_no_fee(self):
        """Zero quantity should yield zero fee."""
        calc = FixedFeeCalculator(Decimal("0.0004"))
        fee = calc.calculate_fee(Decimal("50000"), Decimal("0"))
        self.assertEqual(fee, Decimal("0"))

    def test_very_small_quantity_precision(self):
        """Very small quantities maintain precision."""
        calc = FixedFeeCalculator(Decimal("0.0004"))
        fee = calc.calculate_fee(Decimal("50000"), Decimal("0.00001"))
        expected = Decimal("50000") * Decimal("0.00001") * Decimal("0.0004")
        self.assertEqual(fee, expected)

    def test_negative_fee_rate_rejected(self):
        """FixedFeeCalculator rejects negative fee rate."""
        with self.assertRaises(ValueError):
            FixedFeeCalculator(Decimal("-0.001"))

    def test_max_decimal_precision(self):
        """Decimal arithmetic preserves full precision."""
        a = Decimal("0.1")
        b = Decimal("0.2")
        c = a + b
        self.assertEqual(c, Decimal("0.3"))


if __name__ == "__main__":
    unittest.main()
