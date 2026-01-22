"""
Signal Generator for Bollinger Bot.

BOLLINGER_TREND Mode (Supertrend + Bollinger Bands):
    Entry Logic:
    - LONG: Supertrend bullish (trend=1) AND price <= BB lower band
    - SHORT: Supertrend bearish (trend=-1) AND price >= BB upper band

    Exit Logic:
    - Supertrend flip (primary) - trend changes direction
    - ATR stop loss (protection) - price hits stop loss

Walk-Forward Validated Parameters (2024-01-01 to 2025-01-01):
    - bb_period: 20, bb_std: 2.5
    - st_atr_period: 20, st_atr_multiplier: 3.5
    - atr_stop_multiplier: 2.0
    - Consistency: 50% (below 67% target, use with caution)
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
    Trading signal generator for Bollinger Band + Supertrend strategy.

    BOLLINGER_TREND mode combines Supertrend direction with BB band touch for entries,
    and uses Supertrend flip or ATR stop loss for exits.

    Example:
        >>> generator = SignalGenerator(config, calculator, supertrend_calculator)
        >>> signal = generator.generate(klines, current_price)
        >>> if signal.signal_type != SignalType.NONE:
        ...     print(f"Entry: {signal.entry_price}, SL: {signal.stop_loss}")
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

        # Initialize Supertrend calculator
        self._supertrend = supertrend_calculator or SupertrendCalculator(
            atr_period=config.st_atr_period,
            atr_multiplier=config.st_atr_multiplier,
        )

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
    def supertrend(self) -> SupertrendCalculator:
        """Get Supertrend calculator."""
        return self._supertrend

    @property
    def supertrend_trend(self) -> int:
        """Get current Supertrend trend (1=bullish, -1=bearish, 0=unknown)."""
        return self._supertrend.trend

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

        # Update Supertrend with latest kline
        supertrend_data: Optional[SupertrendData] = None
        if klines:
            supertrend_data = self._supertrend.update(klines[-1])

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
                atr=supertrend_data.atr if supertrend_data else None,
            )

        # Generate BOLLINGER_TREND signal
        return self._generate_bollinger_trend(
            klines, current_price, bands, bbw, supertrend_data
        )

    def _generate_bollinger_trend(
        self,
        klines: List[KlineProtocol],
        current_price: Decimal,
        bands: BollingerBands,
        bbw: BBWData,
        supertrend_data: Optional[SupertrendData],
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

        Exit conditions for BOLLINGER_TREND mode:
        1. Supertrend flip (primary) - trend changes direction
        2. ATR stop loss (protection) - price hits stop loss

        Args:
            position: Current position
            klines: Historical kline data
            current_price: Current market price
            current_bar: Current bar number (unused)
            max_price: Maximum price since entry (unused)
            min_price: Minimum price since entry (unused)

        Returns:
            Tuple of (should_exit, reason)
        """
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        if not klines:
            return False, ""

        # Update Supertrend with latest kline
        supertrend_data = self._supertrend.update(klines[-1])
        if supertrend_data is None:
            return False, ""

        current_trend = supertrend_data.trend

        # 1. Check Supertrend flip (primary exit)
        if position.side == PositionSide.LONG:
            if current_trend == -1:
                return True, f"Supertrend 翻轉做空，平多倉 (ST={supertrend_data.supertrend:.2f})"
        else:
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
        if klines:
            self._supertrend.initialize(klines)
            self._prev_supertrend_trend = self._supertrend.trend

    def get_signal_summary(self, signal: Signal) -> str:
        """Generate signal summary for notification."""
        if signal.signal_type == SignalType.NONE:
            return f"無訊號 - {signal.reason}"

        direction = "做多" if signal.signal_type == SignalType.LONG else "做空"

        # BOLLINGER_TREND mode: no take profit (exit on Supertrend flip)
        summary = (
            f"{direction}訊號\n"
            f"進場價: {signal.entry_price:.2f}\n"
            f"止損價: {signal.stop_loss:.2f}\n"
            f"BBW: {signal.bbw.bbw:.4f} (百分位 {signal.bbw.bbw_percentile}%)"
        )
        if signal.supertrend:
            trend_str = "看多" if signal.supertrend.trend == 1 else "看空"
            summary += f"\nSupertrend: {trend_str} ({signal.supertrend.supertrend:.2f})"
        return summary

    def reset_cooldown(self) -> None:
        """Reset signal cooldown."""
        self._signal_cooldown = 0
