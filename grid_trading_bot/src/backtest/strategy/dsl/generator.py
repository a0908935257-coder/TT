"""
DSL Strategy Generator.

Generates BacktestStrategy subclasses from DSL definitions.
"""

from decimal import Decimal
from typing import Any, Optional, Type

from ....core.models import Kline
from ...order import Signal
from ...position import Position
from ...result import Trade
from ..base import BacktestContext, BacktestStrategy
from .conditions import ConditionEvaluator
from .indicators import BaseIndicator, IndicatorFactory
from .models import (
    EntryDefinition,
    ExitDefinition,
    StopLossDefinition,
    StopLossType,
    StrategyDefinition,
    TakeProfitDefinition,
    TakeProfitType,
)
from .validator import DSLValidator


class DSLStrategyGenerationError(Exception):
    """Exception raised when strategy generation fails."""
    pass


class DSLGeneratedStrategy(BacktestStrategy):
    """
    Base class for DSL-generated strategies.

    Provides the runtime execution logic for DSL-defined strategies.
    """

    def __init__(
        self,
        definition: StrategyDefinition,
        param_overrides: Optional[dict[str, Any]] = None,
    ):
        """
        Initialize DSL strategy.

        Args:
            definition: The strategy definition
            param_overrides: Optional parameter overrides
        """
        self._definition = definition
        self._param_values = definition.get_parameter_values(param_overrides)

        # Create indicator instances
        self._factory = IndicatorFactory()
        self._indicators: dict[str, BaseIndicator] = {}
        for name, ind_def in definition.indicators.items():
            self._indicators[name] = self._factory.create(ind_def, self._param_values)

        # Condition evaluator
        self._evaluator = ConditionEvaluator()

        # State for cross detection
        self._prev_indicator_values: dict[str, dict[str, Any]] = {}

        # Calculate warmup
        self._warmup = max(
            (ind.warmup_period for ind in self._indicators.values()),
            default=50
        ) + 10

        # Track current ATR for stop loss/take profit
        self._current_atr: Optional[Decimal] = None

    @property
    def name(self) -> str:
        """Get strategy name."""
        return self._definition.metadata.name

    @property
    def param_values(self) -> dict[str, Any]:
        """Get current parameter values."""
        return self._param_values.copy()

    def warmup_period(self) -> int:
        """Return warmup period."""
        return self._warmup

    def on_kline(self, kline: Kline, context: BacktestContext) -> list[Signal]:
        """
        Process kline and generate signal.

        Args:
            kline: Current kline
            context: Backtest context

        Returns:
            List of signals
        """
        # Skip if in position
        if context.has_position:
            return []

        # Calculate all indicators
        indicator_values = self._calculate_indicators(context.klines)
        if not indicator_values:
            return []

        # Get price data for condition evaluation
        price_data = {
            "close": float(kline.close),
            "open": float(kline.open),
            "high": float(kline.high),
            "low": float(kline.low),
            "volume": float(kline.volume),
        }

        # Calculate ATR if needed
        self._current_atr = self._calculate_atr(context.klines, 14)

        # Check long entry
        if self._definition.entry_long:
            if self._check_entry_conditions(
                self._definition.entry_long, indicator_values, price_data
            ):
                signal = self._create_entry_signal(
                    "LONG", self._definition.entry_long, kline, indicator_values
                )
                self._update_prev_values(indicator_values)
                return [signal] if signal else []

        # Check short entry
        if self._definition.entry_short:
            if self._check_entry_conditions(
                self._definition.entry_short, indicator_values, price_data
            ):
                signal = self._create_entry_signal(
                    "SHORT", self._definition.entry_short, kline, indicator_values
                )
                self._update_prev_values(indicator_values)
                return [signal] if signal else []

        self._update_prev_values(indicator_values)
        return []

    def check_exit(
        self, position: Position, kline: Kline, context: BacktestContext
    ) -> Optional[Signal]:
        """
        Check if position should be exited.

        Args:
            position: Current position
            kline: Current kline
            context: Backtest context

        Returns:
            Exit signal if conditions met
        """
        # Calculate indicators
        indicator_values = self._calculate_indicators(context.klines)
        if not indicator_values:
            return None

        price_data = {
            "close": float(kline.close),
            "open": float(kline.open),
            "high": float(kline.high),
            "low": float(kline.low),
            "volume": float(kline.volume),
        }

        # Check exit conditions based on position side
        if position.side == "LONG" and self._definition.exit_long:
            if self._check_exit_conditions(
                self._definition.exit_long, indicator_values, price_data
            ):
                return Signal.close_all(reason="dsl_exit_condition")

        elif position.side == "SHORT" and self._definition.exit_short:
            if self._check_exit_conditions(
                self._definition.exit_short, indicator_values, price_data
            ):
                return Signal.close_all(reason="dsl_exit_condition")

        return None

    def _calculate_indicators(self, klines: list[Kline]) -> dict[str, dict[str, Any]]:
        """Calculate all indicator values."""
        values = {}

        for name, indicator in self._indicators.items():
            result = indicator.calculate(klines)
            if not result.is_ready:
                return {}  # Not all indicators ready
            values[name] = result.values

        return values

    def _check_entry_conditions(
        self,
        entry: EntryDefinition,
        indicator_values: dict[str, dict[str, Any]],
        price_data: dict[str, Any],
    ) -> bool:
        """Check if entry conditions are met."""
        if not entry.conditions:
            return False

        return self._evaluator.evaluate(
            entry.conditions,
            indicator_values,
            self._param_values,
            price_data,
            self._prev_indicator_values,
        )

    def _check_exit_conditions(
        self,
        exit_def: ExitDefinition,
        indicator_values: dict[str, dict[str, Any]],
        price_data: dict[str, Any],
    ) -> bool:
        """Check if exit conditions are met."""
        if not exit_def.conditions:
            return False

        return self._evaluator.evaluate(
            exit_def.conditions,
            indicator_values,
            self._param_values,
            price_data,
            self._prev_indicator_values,
        )

    def _create_entry_signal(
        self,
        side: str,
        entry: EntryDefinition,
        kline: Kline,
        indicator_values: dict[str, dict[str, Any]],
    ) -> Signal:
        """Create entry signal with stop loss and take profit."""
        price = kline.close

        # Calculate stop loss
        stop_loss = None
        if entry.stop_loss:
            stop_loss = self._calculate_stop_loss(
                entry.stop_loss, side, price, indicator_values
            )

        # Calculate take profit
        take_profit = None
        if entry.take_profit:
            take_profit = self._calculate_take_profit(
                entry.take_profit, side, price, stop_loss, indicator_values
            )

        reason = f"dsl_{self._definition.metadata.name}_{side.lower()}"

        if side == "LONG":
            return Signal.long_entry(
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
            )
        else:
            return Signal.short_entry(
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=reason,
            )

    def _calculate_stop_loss(
        self,
        sl_def: StopLossDefinition,
        side: str,
        price: Decimal,
        indicator_values: dict[str, dict[str, Any]],
    ) -> Optional[Decimal]:
        """Calculate stop loss price."""
        if sl_def.sl_type == StopLossType.FIXED:
            if sl_def.value is None:
                return None
            if side == "LONG":
                return price - Decimal(str(sl_def.value))
            else:
                return price + Decimal(str(sl_def.value))

        elif sl_def.sl_type == StopLossType.PERCENTAGE:
            if sl_def.value is None:
                return None
            pct = Decimal(str(sl_def.value)) / Decimal("100")
            if side == "LONG":
                return price * (Decimal("1") - pct)
            else:
                return price * (Decimal("1") + pct)

        elif sl_def.sl_type == StopLossType.ATR_BASED:
            if self._current_atr is None:
                return None
            distance = self._current_atr * Decimal(str(sl_def.multiplier))
            if side == "LONG":
                return price - distance
            else:
                return price + distance

        elif sl_def.sl_type == StopLossType.INDICATOR:
            if sl_def.indicator and sl_def.field:
                if sl_def.indicator in indicator_values:
                    value = indicator_values[sl_def.indicator].get(sl_def.field)
                    if value is not None:
                        return Decimal(str(value))

        return None

    def _calculate_take_profit(
        self,
        tp_def: TakeProfitDefinition,
        side: str,
        price: Decimal,
        stop_loss: Optional[Decimal],
        indicator_values: dict[str, dict[str, Any]],
    ) -> Optional[Decimal]:
        """Calculate take profit price."""
        if tp_def.tp_type == TakeProfitType.FIXED:
            if tp_def.value is None:
                return None
            if side == "LONG":
                return price + Decimal(str(tp_def.value))
            else:
                return price - Decimal(str(tp_def.value))

        elif tp_def.tp_type == TakeProfitType.PERCENTAGE:
            if tp_def.value is None:
                return None
            pct = Decimal(str(tp_def.value)) / Decimal("100")
            if side == "LONG":
                return price * (Decimal("1") + pct)
            else:
                return price * (Decimal("1") - pct)

        elif tp_def.tp_type == TakeProfitType.ATR_BASED:
            if self._current_atr is None:
                return None
            distance = self._current_atr * Decimal(str(tp_def.multiplier))
            if side == "LONG":
                return price + distance
            else:
                return price - distance

        elif tp_def.tp_type == TakeProfitType.RISK_REWARD:
            if stop_loss is None:
                return None
            risk = abs(price - stop_loss)
            reward = risk * Decimal(str(tp_def.multiplier))
            if side == "LONG":
                return price + reward
            else:
                return price - reward

        elif tp_def.tp_type == TakeProfitType.INDICATOR:
            if tp_def.indicator and tp_def.field:
                if tp_def.indicator in indicator_values:
                    value = indicator_values[tp_def.indicator].get(tp_def.field)
                    if value is not None:
                        return Decimal(str(value))

        return None

    def _calculate_atr(self, klines: list[Kline], period: int) -> Optional[Decimal]:
        """Calculate ATR for stop loss/take profit."""
        if len(klines) < period + 1:
            return None

        true_ranges = []
        for i in range(-period, 0):
            kline = klines[i]
            prev_close = klines[i - 1].close
            tr = max(
                kline.high - kline.low,
                abs(kline.high - prev_close),
                abs(kline.low - prev_close),
            )
            true_ranges.append(tr)

        return sum(true_ranges) / Decimal(len(true_ranges))

    def _update_prev_values(self, indicator_values: dict[str, dict[str, Any]]) -> None:
        """Store current values for next bar's cross detection."""
        self._prev_indicator_values = {
            name: values.copy() for name, values in indicator_values.items()
        }

    def reset(self) -> None:
        """Reset strategy state."""
        for indicator in self._indicators.values():
            indicator.reset()
        self._prev_indicator_values = {}
        self._current_atr = None


class DSLStrategyGenerator:
    """
    Generates BacktestStrategy classes from DSL definitions.

    Example:
        generator = DSLStrategyGenerator()
        StrategyClass = generator.generate(definition)
        strategy = StrategyClass()
        # or with param overrides
        strategy = StrategyClass(param_overrides={"bb_period": 30})
    """

    def __init__(self, validate: bool = True):
        """
        Initialize generator.

        Args:
            validate: Whether to validate definitions before generation
        """
        self._validate = validate
        self._validator = DSLValidator() if validate else None

    def generate(self, definition: StrategyDefinition) -> Type[DSLGeneratedStrategy]:
        """
        Generate a strategy class from definition.

        Args:
            definition: The strategy definition

        Returns:
            A class that can be instantiated as a strategy

        Raises:
            DSLStrategyGenerationError: If generation fails
        """
        # Validate if enabled
        if self._validator:
            result = self._validator.validate(definition)
            if not result.is_valid:
                raise DSLStrategyGenerationError(
                    f"Invalid strategy definition: {result.errors}"
                )

        # Create a custom class that captures the definition
        class GeneratedStrategy(DSLGeneratedStrategy):
            """DSL-generated strategy class."""

            def __init__(self, param_overrides: Optional[dict[str, Any]] = None):
                super().__init__(definition, param_overrides)

        # Set class name
        GeneratedStrategy.__name__ = definition.metadata.name.replace(" ", "")
        GeneratedStrategy.__qualname__ = GeneratedStrategy.__name__

        return GeneratedStrategy

    def generate_instance(
        self,
        definition: StrategyDefinition,
        param_overrides: Optional[dict[str, Any]] = None,
    ) -> DSLGeneratedStrategy:
        """
        Generate a strategy instance directly.

        Args:
            definition: The strategy definition
            param_overrides: Optional parameter overrides

        Returns:
            A strategy instance ready for backtesting
        """
        strategy_class = self.generate(definition)
        return strategy_class(param_overrides)
