"""
DSL Strategy Models.

Data models for the declarative strategy definition language.
Supports YAML-based strategy definitions with parameters, indicators,
entry/exit conditions, and risk management rules.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Optional, Union


class ParameterType(str, Enum):
    """Parameter type enumeration."""
    INT = "int"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOL = "bool"
    STRING = "string"


class ComparisonOperator(str, Enum):
    """Comparison operators for conditions."""
    EQ = "=="
    NE = "!="
    GT = ">"
    GE = ">="
    LT = "<"
    LE = "<="
    CROSS_ABOVE = "cross_above"
    CROSS_BELOW = "cross_below"


class LogicalOperator(str, Enum):
    """Logical operators for combining conditions."""
    AND = "AND"
    OR = "OR"


class StopLossType(str, Enum):
    """Stop loss calculation methods."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    ATR_BASED = "atr_based"
    INDICATOR = "indicator"


class TakeProfitType(str, Enum):
    """Take profit calculation methods."""
    FIXED = "fixed"
    PERCENTAGE = "percentage"
    ATR_BASED = "atr_based"
    RISK_REWARD = "risk_reward"
    INDICATOR = "indicator"


@dataclass
class ParameterDefinition:
    """
    Definition of a strategy parameter.

    Attributes:
        name: Parameter name (used as variable reference)
        param_type: Type of the parameter
        default: Default value
        range_min: Minimum value for optimization
        range_max: Maximum value for optimization
        step: Step size for grid search
        choices: List of valid choices (for categorical)
        description: Human-readable description
    """
    name: str
    param_type: ParameterType = ParameterType.FLOAT
    default: Any = None
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[list[Any]] = None
    description: str = ""

    def get_value(self, overrides: Optional[dict[str, Any]] = None) -> Any:
        """Get parameter value, with optional override."""
        if overrides and self.name in overrides:
            return self._cast_value(overrides[self.name])
        return self._cast_value(self.default)

    def _cast_value(self, value: Any) -> Any:
        """Cast value to the correct type."""
        if value is None:
            return None
        if self.param_type == ParameterType.INT:
            return int(value)
        elif self.param_type == ParameterType.FLOAT:
            return float(value)
        elif self.param_type == ParameterType.DECIMAL:
            return Decimal(str(value))
        elif self.param_type == ParameterType.BOOL:
            return bool(value)
        return value


@dataclass
class IndicatorParams:
    """
    Parameters for an indicator instance.

    Values can be literal or parameter references (prefixed with $).
    """
    params: dict[str, Any] = field(default_factory=dict)

    def resolve(self, param_values: dict[str, Any]) -> dict[str, Any]:
        """Resolve parameter references to actual values."""
        resolved = {}
        for key, value in self.params.items():
            if isinstance(value, str) and value.startswith("$"):
                param_name = value[1:]  # Remove $ prefix
                if param_name in param_values:
                    resolved[key] = param_values[param_name]
                else:
                    raise ValueError(f"Unknown parameter reference: {value}")
            else:
                resolved[key] = value
        return resolved


@dataclass
class IndicatorDefinition:
    """
    Definition of an indicator instance.

    Attributes:
        name: Instance name (used for referencing in conditions)
        indicator_type: Type of indicator (e.g., "BollingerBands", "Supertrend")
        params: Indicator parameters
    """
    name: str
    indicator_type: str
    params: IndicatorParams = field(default_factory=IndicatorParams)


@dataclass
class ConditionRule:
    """
    A single condition rule.

    Compares an indicator field to a value or another indicator field.

    Attributes:
        indicator: Name of the indicator to check
        field: Field on the indicator (e.g., "trend", "lower", "upper")
        compare: Comparison operator
        value: Value to compare against (literal or reference)
        indicator2: Optional second indicator for indicator-to-indicator comparison
        field2: Optional field on second indicator
    """
    indicator: str
    field: str
    compare: ComparisonOperator
    value: Any = None
    indicator2: Optional[str] = None
    field2: Optional[str] = None

    def is_indicator_comparison(self) -> bool:
        """Check if this is an indicator-to-indicator comparison."""
        return self.indicator2 is not None


@dataclass
class ConditionGroup:
    """
    A group of conditions with a logical operator.

    Supports nested condition groups for complex logic.

    Attributes:
        operator: Logical operator (AND/OR)
        rules: List of condition rules
        groups: Nested condition groups
    """
    operator: LogicalOperator = LogicalOperator.AND
    rules: list[ConditionRule] = field(default_factory=list)
    groups: list["ConditionGroup"] = field(default_factory=list)


@dataclass
class StopLossDefinition:
    """
    Stop loss configuration.

    Attributes:
        sl_type: Method for calculating stop loss
        value: Fixed value or percentage
        multiplier: ATR multiplier (for atr_based)
        indicator: Indicator reference (for indicator type)
        field: Field on indicator
    """
    sl_type: StopLossType = StopLossType.ATR_BASED
    value: Optional[float] = None
    multiplier: float = 2.0
    indicator: Optional[str] = None
    field: Optional[str] = None


@dataclass
class TakeProfitDefinition:
    """
    Take profit configuration.

    Attributes:
        tp_type: Method for calculating take profit
        value: Fixed value or percentage
        multiplier: ATR multiplier or risk-reward ratio
        indicator: Indicator reference (for indicator type)
        field: Field on indicator
    """
    tp_type: TakeProfitType = TakeProfitType.RISK_REWARD
    value: Optional[float] = None
    multiplier: float = 2.0
    indicator: Optional[str] = None
    field: Optional[str] = None


@dataclass
class EntryDefinition:
    """
    Entry conditions for a direction.

    Attributes:
        conditions: Condition group for entry signals
        stop_loss: Stop loss configuration
        take_profit: Take profit configuration
    """
    conditions: ConditionGroup = field(default_factory=ConditionGroup)
    stop_loss: Optional[StopLossDefinition] = None
    take_profit: Optional[TakeProfitDefinition] = None


@dataclass
class ExitDefinition:
    """
    Exit conditions for a direction.

    Attributes:
        conditions: Condition group for exit signals
    """
    conditions: ConditionGroup = field(default_factory=ConditionGroup)


@dataclass
class StrategyMetadata:
    """
    Strategy metadata.

    Attributes:
        name: Strategy name
        version: Strategy version
        description: Human-readable description
        author: Strategy author
        tags: Categorization tags
    """
    name: str = "Unnamed Strategy"
    version: str = "1.0.0"
    description: str = ""
    author: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class StrategyDefinition:
    """
    Complete strategy definition.

    Represents a fully parsed DSL strategy that can be
    validated and converted to a BacktestStrategy.

    Attributes:
        metadata: Strategy metadata
        parameters: Parameter definitions
        indicators: Indicator definitions
        entry_long: Long entry conditions
        entry_short: Short entry conditions
        exit_long: Long exit conditions
        exit_short: Short exit conditions
    """
    metadata: StrategyMetadata = field(default_factory=StrategyMetadata)
    parameters: dict[str, ParameterDefinition] = field(default_factory=dict)
    indicators: dict[str, IndicatorDefinition] = field(default_factory=dict)
    entry_long: Optional[EntryDefinition] = None
    entry_short: Optional[EntryDefinition] = None
    exit_long: Optional[ExitDefinition] = None
    exit_short: Optional[ExitDefinition] = None

    def get_warmup_period(self) -> int:
        """
        Calculate the required warmup period.

        Based on indicator parameters that affect lookback.
        """
        max_period = 0
        for indicator in self.indicators.values():
            # Check common period parameters
            period = indicator.params.params.get("period", 0)
            atr_period = indicator.params.params.get("atr_period", 0)
            lookback = indicator.params.params.get("lookback", 0)

            # Handle parameter references
            for p in [period, atr_period, lookback]:
                if isinstance(p, str) and p.startswith("$"):
                    param_name = p[1:]
                    if param_name in self.parameters:
                        p = self.parameters[param_name].default or 0
                if isinstance(p, (int, float)):
                    max_period = max(max_period, int(p))

        # Add safety margin
        return max_period + 50

    def get_parameter_values(self, overrides: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Get all parameter values with optional overrides."""
        values = {}
        for name, param_def in self.parameters.items():
            values[name] = param_def.get_value(overrides)
        return values

    def validate_basic(self) -> list[str]:
        """
        Perform basic validation.

        Returns list of error messages (empty if valid).
        """
        errors = []

        # Check for at least one entry condition
        if self.entry_long is None and self.entry_short is None:
            errors.append("Strategy must define at least one entry condition (long or short)")

        # Check indicator references in conditions
        all_conditions = []
        for entry in [self.entry_long, self.entry_short]:
            if entry and entry.conditions:
                all_conditions.extend(entry.conditions.rules)
        for exit_def in [self.exit_long, self.exit_short]:
            if exit_def and exit_def.conditions:
                all_conditions.extend(exit_def.conditions.rules)

        for rule in all_conditions:
            if rule.indicator not in self.indicators:
                errors.append(f"Unknown indicator reference: {rule.indicator}")
            if rule.indicator2 and rule.indicator2 not in self.indicators:
                errors.append(f"Unknown indicator reference: {rule.indicator2}")

        return errors
