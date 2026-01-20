"""
Signal Generator for Bollinger Bot.

Generates entry and exit signals based on Bollinger Bands, trend filter, and BBW filter.
Supports two strategy modes: MEAN_REVERSION and BREAKOUT.

MEAN_REVERSION Mode (original):
    1. Check BBW squeeze - if squeeze, no signal (avoid breakout)
    2. Check trend filter - only trade in trend direction (50 SMA)
    3. Price <= lower band -> LONG signal (if uptrend or no filter)
    4. Price >= upper band -> SHORT signal (if downtrend or no filter)
    5. Take profit = middle band
    6. Stop loss = ATR-based (dynamic) or fixed percentage (fallback)

BREAKOUT Mode (trend-following):
    1. Check BBW expansion - only trade when volatility is increasing
    2. Price > upper band -> LONG signal (breakout above)
    3. Price < lower band -> SHORT signal (breakout below)
    4. Use trailing stop to let profits run
    5. Exit on reverse signal or stop hit

Exit Conditions:
    1. Price returns to middle band (take profit) - mean reversion only
    2. Price hits stop loss / trailing stop
    3. Reverse signal (breakout mode)
    4. Hold time exceeds max_hold_bars (timeout)
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Protocol, Tuple

from src.core import get_logger

from .indicators import BollingerCalculator
from .models import (
    BBWData,
    BollingerBands,
    BollingerConfig,
    Position,
    PositionSide,
    Signal,
    SignalType,
    StrategyMode,
)

logger = get_logger(__name__)


class KlineProtocol(Protocol):
    """Protocol for Kline data."""

    @property
    def close(self) -> Decimal: ...

    @property
    def close_time(self) -> datetime: ...


class SignalGenerator:
    """
    Trading signal generator for Bollinger Band strategies.

    Supports two modes:
    - MEAN_REVERSION: Entry when price touches bands, exit at middle band
    - BREAKOUT: Entry when price breaks bands, exit with trailing stop

    Example:
        >>> generator = SignalGenerator(config, calculator)
        >>> signal = generator.generate(klines, current_price)
        >>> if signal.is_valid:
        ...     print(f"Entry: {signal.entry_price}, TP: {signal.take_profit}")
    """

    def __init__(
        self,
        config: BollingerConfig,
        calculator: BollingerCalculator,
    ):
        """
        Initialize SignalGenerator.

        Args:
            config: BollingerConfig with strategy parameters
            calculator: BollingerCalculator for indicator calculation
        """
        self._config = config
        self._calculator = calculator
        self._last_signal: Optional[Signal] = None
        self._signal_cooldown: int = 0

    @property
    def config(self) -> BollingerConfig:
        """Get configuration."""
        return self._config

    @property
    def last_signal(self) -> Optional[Signal]:
        """Get last generated signal."""
        return self._last_signal

    def generate(
        self,
        klines: List[KlineProtocol],
        current_price: Decimal,
    ) -> Signal:
        """
        Generate entry signal based on current market conditions.

        Args:
            klines: List of Kline data for indicator calculation
            current_price: Current market price

        Returns:
            Signal with entry/exit prices or NONE if no signal
        """
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        # Calculate indicators
        bands, bbw = self._calculator.get_all(klines)

        # Calculate trend SMA (for trend filter)
        trend_sma = None
        if self._config.use_trend_filter:
            trend_sma = self._calculator.calculate_sma(klines, self._config.trend_period)

        # Calculate ATR (for dynamic stop loss)
        atr = None
        if self._config.use_atr_stop:
            atr = self._calculator.calculate_atr(klines, self._config.atr_period)

        # Check cooldown
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1
            return Signal(
                signal_type=SignalType.NONE,
                bands=bands,
                bbw=bbw,
                timestamp=datetime.now(timezone.utc),
                reason="訊號冷卻中",
                trend_sma=trend_sma,
                atr=atr,
            )

        # Strategy mode determines signal logic
        is_breakout = self._config.strategy_mode == StrategyMode.BREAKOUT

        if is_breakout:
            # BREAKOUT MODE: Trade when BBW is expanding (high volatility)
            if bbw.is_squeeze:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    timestamp=datetime.now(timezone.utc),
                    reason=f"BBW 低波動（百分位 {bbw.bbw_percentile}%），等待擴張",
                    trend_sma=trend_sma,
                    atr=atr,
                )

            # Breakout signals: price breaks through bands
            signal_type = SignalType.NONE
            entry_price = current_price
            reason = ""

            if current_price > bands.upper:
                signal_type = SignalType.LONG
                reason = f"突破上軌 {bands.upper:.2f}，做多追勢"
            elif current_price < bands.lower:
                signal_type = SignalType.SHORT
                reason = f"突破下軌 {bands.lower:.2f}，做空追勢"
            else:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    timestamp=datetime.now(timezone.utc),
                    reason="價格在通道內，無突破訊號",
                    trend_sma=trend_sma,
                    atr=atr,
                )

            # For breakout, no take profit (use trailing stop instead)
            take_profit = None
            stop_loss = self._calculate_stop_loss(entry_price, signal_type, atr)

        else:
            # MEAN REVERSION MODE: Trade when BBW is not in squeeze
            if bbw.is_squeeze:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    timestamp=datetime.now(timezone.utc),
                    reason=f"BBW 壓縮（百分位 {bbw.bbw_percentile}%），不交易",
                    trend_sma=trend_sma,
                    atr=atr,
                )

            # Mean reversion signals: price touches bands
            signal_type = SignalType.NONE
            entry_price = current_price
            reason = ""

            if current_price <= bands.lower:
                signal_type = SignalType.LONG
                entry_price = bands.lower
                reason = f"價格 {current_price} 觸碰下軌 {bands.lower}，做多"
            elif current_price >= bands.upper:
                signal_type = SignalType.SHORT
                entry_price = bands.upper
                reason = f"價格 {current_price} 觸碰上軌 {bands.upper}，做空"
            else:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    timestamp=datetime.now(timezone.utc),
                    reason="價格在通道內，無訊號",
                    trend_sma=trend_sma,
                    atr=atr,
                )

            # Trend filter check (mean reversion only)
            if self._config.use_trend_filter and trend_sma is not None:
                if signal_type == SignalType.LONG and current_price < trend_sma:
                    return Signal(
                        signal_type=SignalType.NONE,
                        bands=bands,
                        bbw=bbw,
                        timestamp=datetime.now(timezone.utc),
                        reason=f"趨勢過濾：價格 {current_price:.2f} < SMA {trend_sma:.2f}，不做多",
                        trend_sma=trend_sma,
                        atr=atr,
                    )
                elif signal_type == SignalType.SHORT and current_price > trend_sma:
                    return Signal(
                        signal_type=SignalType.NONE,
                        bands=bands,
                        bbw=bbw,
                        timestamp=datetime.now(timezone.utc),
                        reason=f"趨勢過濾：價格 {current_price:.2f} > SMA {trend_sma:.2f}，不做空",
                        trend_sma=trend_sma,
                        atr=atr,
                    )

            # Mean reversion TP = middle band
            take_profit = bands.middle
            stop_loss = self._calculate_stop_loss(entry_price, signal_type, atr)

        # Set cooldown
        self._signal_cooldown = 1

        signal = Signal(
            signal_type=signal_type,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            bands=bands,
            bbw=bbw,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            trend_sma=trend_sma,
            atr=atr,
        )

        self._last_signal = signal
        logger.info(
            f"Signal: {signal_type.value}, entry={entry_price}, tp={take_profit}, "
            f"sl={stop_loss}, sma={trend_sma}, atr={atr}"
        )

        return signal

    def check_exit(
        self,
        position: Position,
        klines: List[KlineProtocol],
        current_price: Decimal,
        current_bar: int,
        max_price: Optional[Decimal] = None,
        min_price: Optional[Decimal] = None,
    ) -> Tuple[bool, str]:
        """
        Check if position should be exited.

        Args:
            position: Current position
            klines: Historical kline data
            current_price: Current market price
            current_bar: Current bar number
            max_price: Maximum price since entry (for trailing stop)
            min_price: Minimum price since entry (for trailing stop)

        Returns:
            Tuple of (should_exit, reason)
        """
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        bands, _ = self._calculator.get_all(klines)
        is_breakout = self._config.strategy_mode == StrategyMode.BREAKOUT

        if is_breakout:
            # BREAKOUT MODE EXIT LOGIC

            # 1. Trailing stop (if enabled)
            if self._config.use_trailing_stop and self._config.use_atr_stop:
                atr = self._calculator.calculate_atr(klines, self._config.atr_period)
                if atr is not None:
                    trailing_distance = atr * self._config.trailing_atr_mult

                    if position.side == PositionSide.LONG:
                        if max_price is not None:
                            trailing_stop = max_price - trailing_distance
                            if current_price <= trailing_stop:
                                return True, f"追蹤止損：價格 {current_price:.2f} <= {trailing_stop:.2f}"
                    else:
                        if min_price is not None:
                            trailing_stop = min_price + trailing_distance
                            if current_price >= trailing_stop:
                                return True, f"追蹤止損：價格 {current_price:.2f} >= {trailing_stop:.2f}"

            # 2. Reverse signal (exit when price breaks opposite band)
            if position.side == PositionSide.LONG:
                if current_price < bands.lower:
                    return True, f"反向訊號：價格 {current_price:.2f} 跌破下軌 {bands.lower:.2f}"
            else:
                if current_price > bands.upper:
                    return True, f"反向訊號：價格 {current_price:.2f} 突破上軌 {bands.upper:.2f}"

            # 3. Fixed stop loss (fallback)
            if position.stop_loss_price is not None:
                if position.side == PositionSide.LONG:
                    if current_price <= position.stop_loss_price:
                        return True, f"止損：價格 {current_price} <= {position.stop_loss_price}"
                else:
                    if current_price >= position.stop_loss_price:
                        return True, f"止損：價格 {current_price} >= {position.stop_loss_price}"

        else:
            # MEAN REVERSION MODE EXIT LOGIC

            # 1. Take profit - price returned to middle band
            if position.side == PositionSide.LONG:
                if current_price >= bands.middle:
                    return True, "止盈：價格回到中軌"
            else:
                if current_price <= bands.middle:
                    return True, "止盈：價格回到中軌"

            # 2. Stop loss
            if position.stop_loss_price is not None:
                if position.side == PositionSide.LONG:
                    if current_price <= position.stop_loss_price:
                        return True, f"止損：價格 {current_price} <= {position.stop_loss_price}"
                else:
                    if current_price >= position.stop_loss_price:
                        return True, f"止損：價格 {current_price} >= {position.stop_loss_price}"

        # 3. Timeout (both modes)
        hold_bars = current_bar - position.entry_bar
        if hold_bars >= self._config.max_hold_bars:
            return True, f"超時：持倉 {hold_bars} 根 K 線，超過上限 {self._config.max_hold_bars}"

        return False, ""

    def _calculate_stop_loss(
        self,
        entry_price: Decimal,
        side: SignalType,
        atr: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate stop loss price.

        Uses ATR-based dynamic stop loss when available and enabled,
        otherwise falls back to fixed percentage.

        Args:
            entry_price: Entry price
            side: Signal type (LONG/SHORT)
            atr: ATR value (optional)

        Returns:
            Stop loss price
        """
        if not isinstance(entry_price, Decimal):
            entry_price = Decimal(str(entry_price))

        # Use ATR-based stop loss if enabled and ATR is available
        if self._config.use_atr_stop and atr is not None:
            atr_distance = atr * self._config.atr_multiplier
            if side == SignalType.LONG:
                return entry_price - atr_distance
            else:
                return entry_price + atr_distance

        # Fallback to fixed percentage stop loss
        stop_pct = self._config.stop_loss_pct
        if side == SignalType.LONG:
            return entry_price * (Decimal("1") - stop_pct)
        else:
            return entry_price * (Decimal("1") + stop_pct)

    def get_signal_summary(self, signal: Signal) -> str:
        """Generate signal summary for notification."""
        if signal.signal_type == SignalType.NONE:
            return f"無訊號 - {signal.reason}"

        direction = "做多" if signal.signal_type == SignalType.LONG else "做空"

        return (
            f"{direction}訊號\n"
            f"進場價: {signal.entry_price:.2f}\n"
            f"止盈價: {signal.take_profit:.2f}\n"
            f"止損價: {signal.stop_loss:.2f}\n"
            f"BBW: {signal.bbw.bbw:.4f} (百分位 {signal.bbw.bbw_percentile}%)"
        )

    def reset_cooldown(self) -> None:
        """Reset signal cooldown."""
        self._signal_cooldown = 0
