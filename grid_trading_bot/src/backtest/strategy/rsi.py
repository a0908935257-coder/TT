"""
RSI Momentum Strategy Adapter.

Adapts the RSI momentum strategy for the unified backtest framework.

Strategy Logic (Momentum/Trend Following):
- Entry Long: RSI crosses above entry_level + momentum_threshold
- Entry Short: RSI crosses below entry_level - momentum_threshold
- Exit: Opposite RSI crossover, stop loss, or take profit

Walk-Forward Validated Parameters:
- RSI Period: 21
- Entry Level: 50, Momentum Threshold: 5
- Stop Loss: 4%, Take Profit: 8%
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from ...core.models import Kline
from ..order import Signal
from ..position import Position
from .base import BacktestContext, BacktestStrategy


@dataclass
class RSIStrategyConfig:
    """
    Configuration for RSI backtest strategy.

    Attributes:
        rsi_period: RSI calculation period (default 21, validated)
        entry_level: Center level for RSI crossover (default 50)
        momentum_threshold: RSI must cross entry_level by this amount (default 5)
        stop_loss_pct: Stop loss percentage (default 4%)
        take_profit_pct: Take profit percentage (default 8%)
    """

    rsi_period: int = 21
    entry_level: int = 50
    momentum_threshold: int = 5
    stop_loss_pct: Decimal = Decimal("0.04")
    take_profit_pct: Decimal = Decimal("0.08")

    def __post_init__(self):
        if not isinstance(self.stop_loss_pct, Decimal):
            self.stop_loss_pct = Decimal(str(self.stop_loss_pct))
        if not isinstance(self.take_profit_pct, Decimal):
            self.take_profit_pct = Decimal(str(self.take_profit_pct))


class RSICalculatorSimple:
    """
    Simple RSI calculator for backtest.

    Uses Wilder's smoothing method.
    """

    def __init__(self, period: int = 14):
        self._period = period
        self._avg_gain: Optional[Decimal] = None
        self._avg_loss: Optional[Decimal] = None
        self._prev_close: Optional[Decimal] = None
        self._rsi: Optional[Decimal] = None
        self._initialized = False

    def calculate(self, closes: list[Decimal]) -> Optional[Decimal]:
        """
        Calculate RSI from close prices.

        Args:
            closes: List of close prices (at least period + 1)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(closes) < self._period + 1:
            return None

        # Calculate price changes
        changes = []
        for i in range(1, len(closes)):
            changes.append(closes[i] - closes[i - 1])

        # Take last 'period' changes
        recent_changes = changes[-self._period:]

        gains = [c for c in recent_changes if c > 0]
        losses = [abs(c) for c in recent_changes if c < 0]

        avg_gain = sum(gains) / Decimal(self._period) if gains else Decimal("0")
        avg_loss = sum(losses) / Decimal(self._period) if losses else Decimal("0")

        if avg_loss == 0:
            return Decimal("100") if avg_gain > 0 else Decimal("50")

        rs = avg_gain / avg_loss
        rsi = Decimal("100") - (Decimal("100") / (Decimal("1") + rs))

        return rsi

    def reset(self):
        """Reset calculator state."""
        self._avg_gain = None
        self._avg_loss = None
        self._prev_close = None
        self._rsi = None
        self._initialized = False


class RSIBacktestStrategy(BacktestStrategy):
    """
    RSI Momentum Trading Strategy Adapter.

    Implements RSI momentum crossover strategy:
    - Long when RSI crosses above entry_level + momentum_threshold
    - Short when RSI crosses below entry_level - momentum_threshold
    - Exit on opposite RSI crossover, stop loss, or take profit

    Example:
        config = RSIStrategyConfig(
            rsi_period=21,
            entry_level=50,
            momentum_threshold=5,
        )
        strategy = RSIBacktestStrategy(config)
        result = engine.run(klines, strategy)
    """

    def __init__(self, config: Optional[RSIStrategyConfig] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration (uses defaults if None)
        """
        self._config = config or RSIStrategyConfig()
        self._rsi_calc = RSICalculatorSimple(period=self._config.rsi_period)
        self._prev_rsi: Optional[Decimal] = None
        self._current_rsi: Optional[Decimal] = None

    def warmup_period(self) -> int:
        """Return warmup period for RSI initialization."""
        return self._config.rsi_period + 10

    def reset(self) -> None:
        """Reset strategy state."""
        self._rsi_calc.reset()
        self._prev_rsi = None
        self._current_rsi = None

    def on_kline(self, kline: Kline, context: BacktestContext) -> Optional[Signal]:
        """
        Process kline and generate signal.

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            Entry signal or None
        """
        # Get close prices for RSI calculation
        closes = context.get_closes(self._config.rsi_period + 5)

        # Calculate current RSI
        self._prev_rsi = self._current_rsi
        self._current_rsi = self._rsi_calc.calculate(closes)

        if self._current_rsi is None:
            return None

        # First bar after warmup - just record RSI
        if self._prev_rsi is None:
            return None

        # Already have position - don't generate new entry signals
        if context.has_position:
            return None

        entry_level = self._config.entry_level
        threshold = self._config.momentum_threshold
        current_price = kline.close

        # Check for bullish momentum crossover
        # prev_rsi was at/below entry_level, now above entry_level + threshold
        if float(self._prev_rsi) <= entry_level and float(self._current_rsi) > entry_level + threshold:
            stop_loss = current_price * (Decimal("1") - self._config.stop_loss_pct)
            take_profit = current_price * (Decimal("1") + self._config.take_profit_pct)
            return Signal.long_entry(
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="rsi_bullish_crossover",
            )

        # Check for bearish momentum crossover
        # prev_rsi was at/above entry_level, now below entry_level - threshold
        if float(self._prev_rsi) >= entry_level and float(self._current_rsi) < entry_level - threshold:
            stop_loss = current_price * (Decimal("1") + self._config.stop_loss_pct)
            take_profit = current_price * (Decimal("1") - self._config.take_profit_pct)
            return Signal.short_entry(
                price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason="rsi_bearish_crossover",
            )

        return None

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check for exit conditions.

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal or None
        """
        if self._current_rsi is None:
            return None

        entry_level = self._config.entry_level
        threshold = self._config.momentum_threshold
        current_rsi = float(self._current_rsi)

        # Check RSI reversal
        if position.side == "LONG":
            # Exit long when RSI shows bearish momentum
            if current_rsi < entry_level - threshold:
                return Signal.close_all(reason="rsi_reversal_bearish")
        else:  # SHORT
            # Exit short when RSI shows bullish momentum
            if current_rsi > entry_level + threshold:
                return Signal.close_all(reason="rsi_reversal_bullish")

        return None
