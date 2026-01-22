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

from .indicators import BollingerCalculator, SupertrendCalculator
from .models import (
    BBWData,
    BollingerBands,
    BollingerConfig,
    Position,
    PositionSide,
    Signal,
    SignalType,
    StrategyMode,
    SupertrendData,
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
        supertrend_calculator: Optional[SupertrendCalculator] = None,
    ):
        """
        Initialize SignalGenerator.

        Args:
            config: BollingerConfig with strategy parameters
            calculator: BollingerCalculator for indicator calculation
            supertrend_calculator: SupertrendCalculator for BOLLINGER_TREND mode
        """
        self._config = config
        self._calculator = calculator
        self._last_signal: Optional[Signal] = None
        self._signal_cooldown: int = 0

        # Initialize Supertrend calculator for BOLLINGER_TREND mode
        if config.strategy_mode == StrategyMode.BOLLINGER_TREND:
            self._supertrend = supertrend_calculator or SupertrendCalculator(
                atr_period=config.st_atr_period,
                atr_multiplier=config.st_atr_multiplier,
            )
        else:
            self._supertrend = None

        # Track previous Supertrend trend for flip detection
        self._prev_supertrend_trend: int = 0

    @property
    def config(self) -> BollingerConfig:
        """Get configuration."""
        return self._config

    @property
    def last_signal(self) -> Optional[Signal]:
        """Get last generated signal."""
        return self._last_signal

    @property
    def supertrend(self) -> Optional[SupertrendCalculator]:
        """Get Supertrend calculator (BOLLINGER_TREND mode only)."""
        return self._supertrend

    @property
    def supertrend_trend(self) -> int:
        """Get current Supertrend trend (1=bullish, -1=bearish, 0=unknown)."""
        return self._supertrend.trend if self._supertrend else 0

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

        # Calculate Bollinger Bands
        bands, bbw = self._calculator.get_all(klines)

        # Calculate Supertrend (for BOLLINGER_TREND mode)
        supertrend_data: Optional[SupertrendData] = None
        if self._supertrend and klines:
            # Update Supertrend with latest kline
            supertrend_data = self._supertrend.update(klines[-1])

        # Calculate ATR for stop loss
        atr = self._calculator.calculate_atr(klines, self._config.atr_period)

        # Legacy: trend SMA (for old modes)
        trend_sma = None
        if self._config.use_trend_filter:
            trend_sma = self._calculator.calculate_sma(klines, self._config.trend_period)

        # Check cooldown
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1
            return Signal(
                signal_type=SignalType.NONE,
                bands=bands,
                bbw=bbw,
                supertrend=supertrend_data,
                timestamp=datetime.now(timezone.utc),
                reason="訊號冷卻中",
                trend_sma=trend_sma,
                atr=atr,
            )

        # Strategy mode determines signal logic
        if self._config.strategy_mode == StrategyMode.BOLLINGER_TREND:
            return self._generate_bollinger_trend(
                klines, current_price, bands, bbw, supertrend_data, atr
            )
        elif self._config.strategy_mode == StrategyMode.BREAKOUT:
            return self._generate_breakout(
                current_price, bands, bbw, trend_sma, atr
            )
        else:
            return self._generate_mean_reversion(
                current_price, bands, bbw, trend_sma, atr
            )

    def _generate_bollinger_trend(
        self,
        klines: List[KlineProtocol],
        current_price: Decimal,
        bands: BollingerBands,
        bbw: BBWData,
        supertrend_data: Optional[SupertrendData],
        atr: Optional[Decimal],
    ) -> Signal:
        """
        Generate signal for BOLLINGER_TREND mode.

        Entry Logic:
        - LONG: Supertrend bullish (trend=1) AND price <= BB lower band
        - SHORT: Supertrend bearish (trend=-1) AND price >= BB upper band

        Exit: Supertrend flip or ATR stop loss (handled in check_exit)
        """
        # Wait for Supertrend to initialize
        if supertrend_data is None:
            return Signal(
                signal_type=SignalType.NONE,
                bands=bands,
                bbw=bbw,
                supertrend=supertrend_data,
                timestamp=datetime.now(timezone.utc),
                reason="等待 Supertrend 初始化",
                atr=atr,
            )

        trend = supertrend_data.trend
        signal_type = SignalType.NONE
        entry_price = current_price
        reason = ""

        # Entry conditions: Supertrend direction + BB band touch
        if trend == 1:  # Bullish Supertrend
            if current_price <= bands.lower:
                signal_type = SignalType.LONG
                entry_price = bands.lower
                reason = f"Supertrend 看多 + 價格觸及下軌 {bands.lower:.2f}，做多"
            else:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    supertrend=supertrend_data,
                    timestamp=datetime.now(timezone.utc),
                    reason=f"Supertrend 看多，等待價格觸及下軌 {bands.lower:.2f}",
                    atr=atr,
                )
        elif trend == -1:  # Bearish Supertrend
            if current_price >= bands.upper:
                signal_type = SignalType.SHORT
                entry_price = bands.upper
                reason = f"Supertrend 看空 + 價格觸及上軌 {bands.upper:.2f}，做空"
            else:
                return Signal(
                    signal_type=SignalType.NONE,
                    bands=bands,
                    bbw=bbw,
                    supertrend=supertrend_data,
                    timestamp=datetime.now(timezone.utc),
                    reason=f"Supertrend 看空，等待價格觸及上軌 {bands.upper:.2f}",
                    atr=atr,
                )

        # Calculate ATR-based stop loss
        stop_loss = self._calculate_atr_stop_loss(entry_price, signal_type, supertrend_data.atr)

        # No take profit - exit on Supertrend flip
        take_profit = None

        # Set cooldown
        self._signal_cooldown = 3

        signal = Signal(
            signal_type=signal_type,
            entry_price=entry_price,
            take_profit=take_profit,
            stop_loss=stop_loss,
            bands=bands,
            bbw=bbw,
            supertrend=supertrend_data,
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            atr=supertrend_data.atr,
        )

        self._last_signal = signal
        logger.info(
            f"Signal (BOLLINGER_TREND): {signal_type.value}, entry={entry_price}, "
            f"sl={stop_loss}, supertrend={supertrend_data.supertrend:.2f}, trend={trend}"
        )

        return signal

    def _generate_breakout(
        self,
        current_price: Decimal,
        bands: BollingerBands,
        bbw: BBWData,
        trend_sma: Optional[Decimal],
        atr: Optional[Decimal],
    ) -> Signal:
        """Generate signal for legacy BREAKOUT mode."""
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

        take_profit = None
        stop_loss = self._calculate_stop_loss(entry_price, signal_type, atr)
        self._signal_cooldown = 3

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
        logger.info(f"Signal (BREAKOUT): {signal_type.value}, entry={entry_price}, sl={stop_loss}")
        return signal

    def _generate_mean_reversion(
        self,
        current_price: Decimal,
        bands: BollingerBands,
        bbw: BBWData,
        trend_sma: Optional[Decimal],
        atr: Optional[Decimal],
    ) -> Signal:
        """Generate signal for legacy MEAN_REVERSION mode."""
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

        # Trend filter check
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

        take_profit = bands.middle
        stop_loss = self._calculate_stop_loss(entry_price, signal_type, atr)
        self._signal_cooldown = 3

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
        logger.info(f"Signal (MEAN_REVERSION): {signal_type.value}, entry={entry_price}, tp={take_profit}, sl={stop_loss}")
        return signal

    def _calculate_atr_stop_loss(
        self,
        entry_price: Decimal,
        side: SignalType,
        atr: Decimal,
    ) -> Decimal:
        """Calculate ATR-based stop loss for BOLLINGER_TREND mode."""
        atr_distance = atr * self._config.atr_stop_multiplier
        if side == SignalType.LONG:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

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

        # BOLLINGER_TREND MODE EXIT LOGIC
        if self._config.strategy_mode == StrategyMode.BOLLINGER_TREND:
            return self._check_exit_bollinger_trend(position, klines, current_price)

        # BREAKOUT MODE EXIT LOGIC
        if self._config.strategy_mode == StrategyMode.BREAKOUT:
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

        # Timeout (legacy modes only)
        hold_bars = current_bar - position.entry_bar
        if hold_bars >= self._config.max_hold_bars:
            return True, f"超時：持倉 {hold_bars} 根 K 線，超過上限 {self._config.max_hold_bars}"

        return False, ""

    def _check_exit_bollinger_trend(
        self,
        position: Position,
        klines: List[KlineProtocol],
        current_price: Decimal,
    ) -> Tuple[bool, str]:
        """
        Check exit conditions for BOLLINGER_TREND mode.

        Exit conditions:
        1. Supertrend flip (primary) - trend changes direction
        2. ATR stop loss (protection) - price hits stop loss

        No timeout in this mode - let profits run with Supertrend.
        """
        if not self._supertrend or not klines:
            return False, ""

        # Update Supertrend with latest kline
        supertrend_data = self._supertrend.update(klines[-1])
        if supertrend_data is None:
            return False, ""

        current_trend = supertrend_data.trend

        # 1. Check Supertrend flip (primary exit)
        if position.side == PositionSide.LONG:
            # Long position: exit when Supertrend turns bearish
            if current_trend == -1:
                return True, f"Supertrend 翻轉做空，平多倉 (ST={supertrend_data.supertrend:.2f})"
        else:
            # Short position: exit when Supertrend turns bullish
            if current_trend == 1:
                return True, f"Supertrend 翻轉做多，平空倉 (ST={supertrend_data.supertrend:.2f})"

        # 2. Check ATR stop loss (protection)
        if position.stop_loss_price is not None:
            if position.side == PositionSide.LONG:
                if current_price <= position.stop_loss_price:
                    return True, f"ATR 止損：價格 {current_price:.2f} <= {position.stop_loss_price:.2f}"
            else:
                if current_price >= position.stop_loss_price:
                    return True, f"ATR 止損：價格 {current_price:.2f} >= {position.stop_loss_price:.2f}"

        return False, ""

    def initialize_supertrend(self, klines: List[KlineProtocol]) -> None:
        """
        Initialize Supertrend calculator with historical data.

        Should be called at bot startup.

        Args:
            klines: Historical klines for initialization
        """
        if self._supertrend and klines:
            self._supertrend.initialize(klines)
            self._prev_supertrend_trend = self._supertrend.trend

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
