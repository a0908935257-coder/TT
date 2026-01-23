"""
Unit tests for market microstructure simulation.
"""

import pytest
from datetime import datetime
from decimal import Decimal

from src.backtest.config import IntraBarSequence
from src.backtest.microstructure import (
    MarketMicrostructure,
    SpreadContext,
    PriceSequence,
    create_microstructure_from_config,
)
from src.core.models import Kline, KlineInterval


def create_test_kline(
    open_price: Decimal = Decimal("100"),
    high: Decimal = Decimal("105"),
    low: Decimal = Decimal("95"),
    close: Decimal = Decimal("102"),
) -> Kline:
    """Create a test kline."""
    return Kline(
        symbol="BTCUSDT",
        interval=KlineInterval.h1,
        open_time=datetime(2024, 1, 1, 0, 0),
        close_time=datetime(2024, 1, 1, 0, 59),
        open=open_price,
        high=high,
        low=low,
        close=close,
        volume=Decimal("1000"),
        quote_volume=Decimal("100000"),
        trades_count=100,
    )


class TestMarketMicrostructure:
    """Tests for MarketMicrostructure class."""

    def test_default_initialization(self):
        """Test default initialization."""
        ms = MarketMicrostructure()
        assert ms.base_spread_pct == Decimal("0.0001")
        assert ms.spread_volatility_factor == Decimal("0.5")
        assert ms.intra_bar_sequence == IntraBarSequence.OHLC

    def test_custom_initialization(self):
        """Test custom initialization."""
        ms = MarketMicrostructure(
            base_spread_pct=Decimal("0.0005"),
            spread_volatility_factor=Decimal("0.3"),
            intra_bar_sequence=IntraBarSequence.OLHC,
        )
        assert ms.base_spread_pct == Decimal("0.0005")
        assert ms.spread_volatility_factor == Decimal("0.3")
        assert ms.intra_bar_sequence == IntraBarSequence.OLHC

    def test_invalid_spread_raises(self):
        """Test that negative spread raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            MarketMicrostructure(base_spread_pct=Decimal("-0.001"))

    def test_invalid_volatility_factor_raises(self):
        """Test that negative volatility factor raises error."""
        with pytest.raises(ValueError, match="cannot be negative"):
            MarketMicrostructure(spread_volatility_factor=Decimal("-0.1"))


class TestSpreadCalculation:
    """Tests for spread calculation."""

    def test_base_spread(self):
        """Test basic spread calculation."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.001"))  # 0.1%
        kline = create_test_kline()

        spread = ms.calculate_spread(Decimal("100"), kline)
        assert spread == Decimal("0.1")  # 100 * 0.001

    def test_volatility_adjusted_spread(self):
        """Test spread widens with high volatility."""
        ms = MarketMicrostructure(
            base_spread_pct=Decimal("0.001"),
            spread_volatility_factor=Decimal("0.5"),
        )
        kline = create_test_kline()

        # Normal volatility
        normal_context = SpreadContext(volatility_ratio=Decimal("1.0"))
        normal_spread = ms.calculate_spread(Decimal("100"), kline, normal_context)

        # High volatility (2x normal)
        high_vol_context = SpreadContext(volatility_ratio=Decimal("2.0"))
        high_vol_spread = ms.calculate_spread(Decimal("100"), kline, high_vol_context)

        # High volatility should widen spread
        assert high_vol_spread > normal_spread

    def test_low_liquidity_widens_spread(self):
        """Test spread widens with low liquidity."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.001"))
        kline = create_test_kline()

        # Normal liquidity
        normal_context = SpreadContext(liquidity_factor=Decimal("1.0"))
        normal_spread = ms.calculate_spread(Decimal("100"), kline, normal_context)

        # Low liquidity (50% of normal)
        low_liq_context = SpreadContext(liquidity_factor=Decimal("0.5"))
        low_liq_spread = ms.calculate_spread(Decimal("100"), kline, low_liq_context)

        # Low liquidity should widen spread (2x)
        assert low_liq_spread == normal_spread * 2


class TestExecutionPrice:
    """Tests for execution price calculation."""

    def test_aggressive_buy(self):
        """Test aggressive buy pays the spread."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.01"))  # 1%
        kline = create_test_kline()

        exec_price = ms.get_execution_price(
            target_price=Decimal("100"),
            is_buy=True,
            kline=kline,
            is_aggressive=True,
        )

        # Buy at ask (higher than target by half spread)
        # Half spread = 100 * 0.01 / 2 = 0.5
        assert exec_price == Decimal("100.5")

    def test_aggressive_sell(self):
        """Test aggressive sell pays the spread."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.01"))
        kline = create_test_kline()

        exec_price = ms.get_execution_price(
            target_price=Decimal("100"),
            is_buy=False,
            kline=kline,
            is_aggressive=True,
        )

        # Sell at bid (lower than target by half spread)
        assert exec_price == Decimal("99.5")

    def test_passive_order(self):
        """Test passive order gets target price."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.01"))
        kline = create_test_kline()

        exec_price = ms.get_execution_price(
            target_price=Decimal("100"),
            is_buy=True,
            kline=kline,
            is_aggressive=False,  # Passive
        )

        # Passive gets target price
        assert exec_price == Decimal("100")

    def test_buy_capped_at_high(self):
        """Test aggressive buy capped at kline high."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.2"))  # 20% spread
        kline = create_test_kline(high=Decimal("105"))

        exec_price = ms.get_execution_price(
            target_price=Decimal("100"),
            is_buy=True,
            kline=kline,
            is_aggressive=True,
        )

        # Should be capped at high=105, not 100 + 10 = 110
        assert exec_price == Decimal("105")

    def test_sell_capped_at_low(self):
        """Test aggressive sell capped at kline low."""
        ms = MarketMicrostructure(base_spread_pct=Decimal("0.2"))  # 20% spread
        kline = create_test_kline(low=Decimal("95"))

        exec_price = ms.get_execution_price(
            target_price=Decimal("100"),
            is_buy=False,
            kline=kline,
            is_aggressive=True,
        )

        # Should be capped at low=95, not 100 - 10 = 90
        assert exec_price == Decimal("95")


class TestPriceSequence:
    """Tests for price sequence determination."""

    def test_ohlc_returns_high_first(self):
        """Test OHLC sequence returns high first."""
        ms = MarketMicrostructure(intra_bar_sequence=IntraBarSequence.OHLC)
        kline = create_test_kline()

        sequence = ms.determine_price_sequence(kline)
        assert sequence == PriceSequence.HIGH_FIRST

    def test_olhc_returns_low_first(self):
        """Test OLHC sequence returns low first."""
        ms = MarketMicrostructure(intra_bar_sequence=IntraBarSequence.OLHC)
        kline = create_test_kline()

        sequence = ms.determine_price_sequence(kline)
        assert sequence == PriceSequence.LOW_FIRST

    def test_worst_case_bullish(self):
        """Test worst case with bullish bar returns low first."""
        ms = MarketMicrostructure(intra_bar_sequence=IntraBarSequence.WORST_CASE)
        bullish_kline = create_test_kline(
            open_price=Decimal("100"),
            close=Decimal("105"),  # Bullish
        )

        sequence = ms.determine_price_sequence(bullish_kline)
        assert sequence == PriceSequence.LOW_FIRST

    def test_worst_case_bearish(self):
        """Test worst case with bearish bar returns high first."""
        ms = MarketMicrostructure(intra_bar_sequence=IntraBarSequence.WORST_CASE)
        bearish_kline = create_test_kline(
            open_price=Decimal("105"),
            close=Decimal("100"),  # Bearish
        )

        sequence = ms.determine_price_sequence(bearish_kline)
        assert sequence == PriceSequence.HIGH_FIRST


class TestSLTPExecution:
    """Tests for stop loss and take profit execution."""

    @pytest.fixture
    def ms_ohlc(self):
        """Create microstructure with OHLC sequence."""
        return MarketMicrostructure(intra_bar_sequence=IntraBarSequence.OHLC)

    @pytest.fixture
    def ms_olhc(self):
        """Create microstructure with OLHC sequence."""
        return MarketMicrostructure(intra_bar_sequence=IntraBarSequence.OLHC)

    def test_long_sl_triggered(self, ms_ohlc):
        """Test long position stop loss triggered."""
        kline = create_test_kline(low=Decimal("95"))

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="LONG",
            kline=kline,
            stop_loss=Decimal("98"),
            take_profit=None,
        )

        assert sl_triggered is True
        assert tp_triggered is False
        assert sl_fill == Decimal("98")
        assert tp_fill is None

    def test_long_tp_triggered(self, ms_ohlc):
        """Test long position take profit triggered."""
        kline = create_test_kline(high=Decimal("105"))

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="LONG",
            kline=kline,
            stop_loss=None,
            take_profit=Decimal("103"),
        )

        assert sl_triggered is False
        assert tp_triggered is True
        assert sl_fill is None
        assert tp_fill == Decimal("103")

    def test_short_sl_triggered(self, ms_ohlc):
        """Test short position stop loss triggered."""
        kline = create_test_kline(high=Decimal("105"))

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="SHORT",
            kline=kline,
            stop_loss=Decimal("103"),
            take_profit=None,
        )

        assert sl_triggered is True
        assert tp_triggered is False
        assert sl_fill == Decimal("103")
        assert tp_fill is None

    def test_short_tp_triggered(self, ms_ohlc):
        """Test short position take profit triggered."""
        kline = create_test_kline(low=Decimal("95"))

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="SHORT",
            kline=kline,
            stop_loss=None,
            take_profit=Decimal("97"),
        )

        assert sl_triggered is False
        assert tp_triggered is True
        assert sl_fill is None
        assert tp_fill == Decimal("97")

    def test_long_both_hit_ohlc(self, ms_ohlc):
        """Test long with both SL and TP hit - OHLC means TP first."""
        # High first (105), then low (95)
        kline = create_test_kline(
            open_price=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
        )

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="LONG",
            kline=kline,
            stop_loss=Decimal("97"),
            take_profit=Decimal("103"),
        )

        # OHLC = high first, so TP should be triggered for long
        assert sl_triggered is False
        assert tp_triggered is True
        assert tp_fill == Decimal("103")

    def test_long_both_hit_olhc(self, ms_olhc):
        """Test long with both SL and TP hit - OLHC means SL first."""
        # Low first (95), then high (105)
        kline = create_test_kline(
            open_price=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
        )

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_olhc.determine_sl_tp_execution(
            position_side="LONG",
            kline=kline,
            stop_loss=Decimal("97"),
            take_profit=Decimal("103"),
        )

        # OLHC = low first, so SL should be triggered for long
        assert sl_triggered is True
        assert tp_triggered is False
        assert sl_fill == Decimal("97")

    def test_neither_triggered(self, ms_ohlc):
        """Test neither SL nor TP triggered."""
        kline = create_test_kline(
            high=Decimal("102"),
            low=Decimal("99"),
        )

        sl_triggered, tp_triggered, sl_fill, tp_fill = ms_ohlc.determine_sl_tp_execution(
            position_side="LONG",
            kline=kline,
            stop_loss=Decimal("95"),
            take_profit=Decimal("105"),
        )

        assert sl_triggered is False
        assert tp_triggered is False


class TestGapFillPrice:
    """Tests for gap fill price calculation."""

    @pytest.fixture
    def ms(self):
        """Create microstructure."""
        return MarketMicrostructure()

    def test_long_sl_no_gap(self, ms):
        """Test long SL without gap fills at target."""
        kline = create_test_kline(
            open_price=Decimal("100"),
            low=Decimal("97"),
        )

        fill = ms.calculate_gap_fill_price(
            target_price=Decimal("98"),
            kline=kline,
            is_stop=True,
            is_long=True,
        )

        assert fill == Decimal("98")

    def test_long_sl_gap_down(self, ms):
        """Test long SL with gap down fills at open."""
        kline = create_test_kline(
            open_price=Decimal("95"),  # Gapped below SL
            low=Decimal("94"),
        )

        fill = ms.calculate_gap_fill_price(
            target_price=Decimal("98"),
            kline=kline,
            is_stop=True,
            is_long=True,
        )

        # Fills at open when gapped through
        assert fill == Decimal("95")

    def test_short_sl_gap_up(self, ms):
        """Test short SL with gap up fills at open."""
        kline = create_test_kline(
            open_price=Decimal("105"),  # Gapped above SL
            high=Decimal("106"),
        )

        fill = ms.calculate_gap_fill_price(
            target_price=Decimal("103"),
            kline=kline,
            is_stop=True,
            is_long=False,
        )

        # Fills at open when gapped through
        assert fill == Decimal("105")

    def test_tp_fills_at_target(self, ms):
        """Test take profit always fills at target."""
        kline = create_test_kline(
            open_price=Decimal("102"),
            high=Decimal("110"),  # Gapped well above TP
        )

        fill = ms.calculate_gap_fill_price(
            target_price=Decimal("105"),
            kline=kline,
            is_stop=False,  # Take profit
            is_long=True,
        )

        # TP fills at target
        assert fill == Decimal("105")


class TestHelperMethods:
    """Tests for helper methods."""

    @pytest.fixture
    def ms(self):
        """Create microstructure."""
        return MarketMicrostructure()

    def test_mid_price(self, ms):
        """Test mid price calculation."""
        kline = create_test_kline(
            high=Decimal("110"),
            low=Decimal("90"),
        )

        mid = ms.get_mid_price(kline)
        assert mid == Decimal("100")

    def test_vwap_estimate(self, ms):
        """Test VWAP estimation (typical price)."""
        kline = create_test_kline(
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("100"),
        )

        vwap = ms.get_vwap_estimate(kline)
        # (110 + 90 + 100) / 3 = 100
        assert vwap == Decimal("100")


class TestFactoryFunction:
    """Tests for factory function."""

    def test_disabled_returns_none(self):
        """Test that disabled spread returns None."""
        result = create_microstructure_from_config(enable_spread=False)
        assert result is None

    def test_enabled_returns_instance(self):
        """Test that enabled spread returns instance."""
        result = create_microstructure_from_config(
            enable_spread=True,
            spread_pct=Decimal("0.0005"),
            intra_bar_sequence=IntraBarSequence.OLHC,
        )

        assert result is not None
        assert isinstance(result, MarketMicrostructure)
        assert result.base_spread_pct == Decimal("0.0005")
        assert result.intra_bar_sequence == IntraBarSequence.OLHC
