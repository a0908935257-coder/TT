"""
DSL Strategy Validator.

Validates DSL strategy definitions for semantic correctness.
"""

from dataclasses import dataclass, field
from typing import Optional

from .models import (
    ComparisonOperator,
    ConditionGroup,
    ConditionRule,
    EntryDefinition,
    ExitDefinition,
    IndicatorDefinition,
    ParameterDefinition,
    StopLossDefinition,
    StopLossType,
    StrategyDefinition,
    TakeProfitDefinition,
    TakeProfitType,
)


class DSLValidationError(Exception):
    """Exception raised when validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


@dataclass
class ValidationResult:
    """
    Result of validation.

    Attributes:
        is_valid: Whether the definition is valid
        errors: List of error messages
        warnings: List of warning messages
    """
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


# Supported indicator types and their required/optional fields
SUPPORTED_INDICATORS = {
    "BollingerBands": {
        "required_params": ["period"],
        "optional_params": ["std_multiplier", "lookback"],
        "output_fields": ["upper", "middle", "lower", "bandwidth", "percent_b"],
    },
    "Supertrend": {
        "required_params": ["atr_period", "atr_multiplier"],
        "optional_params": [],
        "output_fields": ["trend", "value", "upper_band", "lower_band"],
    },
    "SMA": {
        "required_params": ["period"],
        "optional_params": [],
        "output_fields": ["value"],
    },
    "EMA": {
        "required_params": ["period"],
        "optional_params": [],
        "output_fields": ["value"],
    },
    "RSI": {
        "required_params": ["period"],
        "optional_params": [],
        "output_fields": ["value"],
    },
    "MACD": {
        "required_params": ["fast_period", "slow_period", "signal_period"],
        "optional_params": [],
        "output_fields": ["macd", "signal", "histogram"],
    },
    "ATR": {
        "required_params": ["period"],
        "optional_params": [],
        "output_fields": ["value"],
    },
    "Stochastic": {
        "required_params": ["k_period", "d_period"],
        "optional_params": ["smooth_k"],
        "output_fields": ["k", "d"],
    },
    "ADX": {
        "required_params": ["period"],
        "optional_params": [],
        "output_fields": ["adx", "plus_di", "minus_di"],
    },
    "VWAP": {
        "required_params": [],
        "optional_params": [],
        "output_fields": ["value"],
    },
}

# Special value references
SPECIAL_VALUES = {"$close", "$open", "$high", "$low", "$volume"}


class DSLValidator:
    """
    DSL Strategy Validator.

    Validates semantic correctness of strategy definitions including:
    - Indicator type validity
    - Parameter reference validity
    - Condition rule validity
    - Stop loss / take profit configuration

    Example:
        validator = DSLValidator()
        result = validator.validate(definition)
        if not result.is_valid:
            print("Errors:", result.errors)
    """

    def __init__(self, strict: bool = False):
        """
        Initialize validator.

        Args:
            strict: If True, unknown indicators cause errors.
                   If False, they generate warnings.
        """
        self._strict = strict

    def validate(self, definition: StrategyDefinition) -> ValidationResult:
        """
        Validate a strategy definition.

        Args:
            definition: The strategy definition to validate

        Returns:
            ValidationResult with errors and warnings
        """
        result = ValidationResult()

        # Validate metadata
        result.merge(self._validate_metadata(definition))

        # Validate parameters
        result.merge(self._validate_parameters(definition))

        # Validate indicators
        result.merge(self._validate_indicators(definition))

        # Validate entry conditions
        result.merge(self._validate_entries(definition))

        # Validate exit conditions
        result.merge(self._validate_exits(definition))

        # Validate overall structure
        result.merge(self._validate_structure(definition))

        return result

    def validate_and_raise(self, definition: StrategyDefinition) -> None:
        """
        Validate and raise exception if invalid.

        Args:
            definition: The strategy definition to validate

        Raises:
            DSLValidationError: If validation fails
        """
        result = self.validate(definition)
        if not result.is_valid:
            raise DSLValidationError(result.errors)

    def _validate_metadata(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate strategy metadata."""
        result = ValidationResult()
        metadata = definition.metadata

        if not metadata.name or metadata.name == "Unnamed Strategy":
            result.add_warning("Strategy should have a descriptive name")

        if not metadata.version:
            result.add_warning("Strategy version not specified")

        return result

    def _validate_parameters(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate parameter definitions."""
        result = ValidationResult()

        for name, param in definition.parameters.items():
            # Check for default value
            if param.default is None:
                result.add_error(f"Parameter '{name}' has no default value")

            # Check range for numeric types
            if param.param_type.value in ("int", "float", "decimal"):
                if param.range_min is not None and param.range_max is not None:
                    if param.range_min > param.range_max:
                        result.add_error(
                            f"Parameter '{name}': range_min ({param.range_min}) > "
                            f"range_max ({param.range_max})"
                        )

                    # Check default is within range
                    if param.default is not None:
                        default_val = float(param.default)
                        if default_val < param.range_min or default_val > param.range_max:
                            result.add_warning(
                                f"Parameter '{name}': default ({param.default}) "
                                f"is outside range [{param.range_min}, {param.range_max}]"
                            )

        return result

    def _validate_indicators(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate indicator definitions."""
        result = ValidationResult()
        param_names = set(definition.parameters.keys())

        for name, indicator in definition.indicators.items():
            # Check if indicator type is known
            if indicator.indicator_type not in SUPPORTED_INDICATORS:
                if self._strict:
                    result.add_error(f"Unknown indicator type: {indicator.indicator_type}")
                else:
                    result.add_warning(
                        f"Unknown indicator type '{indicator.indicator_type}' "
                        f"for indicator '{name}'"
                    )
                continue

            # Validate indicator parameters
            ind_spec = SUPPORTED_INDICATORS[indicator.indicator_type]
            provided_params = set(indicator.params.params.keys())

            # Check required parameters
            for req_param in ind_spec["required_params"]:
                if req_param not in provided_params:
                    result.add_error(
                        f"Indicator '{name}' ({indicator.indicator_type}): "
                        f"missing required parameter '{req_param}'"
                    )

            # Validate parameter references
            for param_key, param_value in indicator.params.params.items():
                if isinstance(param_value, str) and param_value.startswith("$"):
                    ref_name = param_value[1:]
                    if ref_name not in param_names:
                        result.add_error(
                            f"Indicator '{name}': unknown parameter reference '{param_value}'"
                        )

        return result

    def _validate_entries(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate entry definitions."""
        result = ValidationResult()

        if definition.entry_long:
            result.merge(
                self._validate_entry(definition.entry_long, "long", definition)
            )

        if definition.entry_short:
            result.merge(
                self._validate_entry(definition.entry_short, "short", definition)
            )

        return result

    def _validate_entry(
        self,
        entry: EntryDefinition,
        direction: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate a single entry definition."""
        result = ValidationResult()

        # Validate conditions
        if entry.conditions:
            result.merge(
                self._validate_condition_group(
                    entry.conditions, f"entry.{direction}", definition
                )
            )

        # Validate stop loss
        if entry.stop_loss:
            result.merge(
                self._validate_stop_loss(entry.stop_loss, f"entry.{direction}", definition)
            )

        # Validate take profit
        if entry.take_profit:
            result.merge(
                self._validate_take_profit(
                    entry.take_profit, f"entry.{direction}", definition
                )
            )

        return result

    def _validate_exits(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate exit definitions."""
        result = ValidationResult()

        if definition.exit_long:
            result.merge(
                self._validate_exit(definition.exit_long, "long", definition)
            )

        if definition.exit_short:
            result.merge(
                self._validate_exit(definition.exit_short, "short", definition)
            )

        return result

    def _validate_exit(
        self,
        exit_def: ExitDefinition,
        direction: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate a single exit definition."""
        result = ValidationResult()

        if exit_def.conditions:
            result.merge(
                self._validate_condition_group(
                    exit_def.conditions, f"exit.{direction}", definition
                )
            )

        return result

    def _validate_condition_group(
        self,
        group: ConditionGroup,
        context: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate a condition group."""
        result = ValidationResult()

        # Validate rules
        for i, rule in enumerate(group.rules):
            result.merge(
                self._validate_condition_rule(rule, f"{context}.rule[{i}]", definition)
            )

        # Validate nested groups
        for i, nested_group in enumerate(group.groups):
            result.merge(
                self._validate_condition_group(
                    nested_group, f"{context}.group[{i}]", definition
                )
            )

        # Check for empty group
        if not group.rules and not group.groups:
            result.add_warning(f"{context}: empty condition group")

        return result

    def _validate_condition_rule(
        self,
        rule: ConditionRule,
        context: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate a single condition rule."""
        result = ValidationResult()
        indicator_names = set(definition.indicators.keys())
        param_names = set(definition.parameters.keys())

        # Validate indicator reference
        if rule.indicator not in indicator_names:
            result.add_error(f"{context}: unknown indicator '{rule.indicator}'")
        else:
            # Validate field reference
            indicator = definition.indicators[rule.indicator]
            if indicator.indicator_type in SUPPORTED_INDICATORS:
                valid_fields = SUPPORTED_INDICATORS[indicator.indicator_type]["output_fields"]
                if rule.field not in valid_fields:
                    result.add_error(
                        f"{context}: indicator '{rule.indicator}' ({indicator.indicator_type}) "
                        f"has no field '{rule.field}'. Valid fields: {valid_fields}"
                    )

        # Validate comparison value
        if rule.value is not None:
            if isinstance(rule.value, str):
                if rule.value.startswith("$"):
                    # Parameter or special value reference
                    ref = rule.value[1:]
                    if rule.value not in SPECIAL_VALUES and ref not in param_names:
                        result.add_error(f"{context}: unknown reference '{rule.value}'")

        # Validate second indicator for indicator-to-indicator comparison
        if rule.indicator2:
            if rule.indicator2 not in indicator_names:
                result.add_error(f"{context}: unknown indicator '{rule.indicator2}'")
            elif rule.field2:
                indicator2 = definition.indicators[rule.indicator2]
                if indicator2.indicator_type in SUPPORTED_INDICATORS:
                    valid_fields = SUPPORTED_INDICATORS[indicator2.indicator_type]["output_fields"]
                    if rule.field2 not in valid_fields:
                        result.add_error(
                            f"{context}: indicator '{rule.indicator2}' has no field '{rule.field2}'"
                        )

        return result

    def _validate_stop_loss(
        self,
        stop_loss: StopLossDefinition,
        context: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate stop loss configuration."""
        result = ValidationResult()

        if stop_loss.sl_type == StopLossType.FIXED:
            if stop_loss.value is None:
                result.add_error(f"{context}.stop_loss: fixed type requires 'value'")

        elif stop_loss.sl_type == StopLossType.PERCENTAGE:
            if stop_loss.value is None:
                result.add_error(f"{context}.stop_loss: percentage type requires 'value'")
            elif stop_loss.value <= 0 or stop_loss.value >= 100:
                result.add_warning(
                    f"{context}.stop_loss: percentage {stop_loss.value} may be incorrect"
                )

        elif stop_loss.sl_type == StopLossType.ATR_BASED:
            if stop_loss.multiplier <= 0:
                result.add_error(f"{context}.stop_loss: ATR multiplier must be positive")

        elif stop_loss.sl_type == StopLossType.INDICATOR:
            if stop_loss.indicator is None:
                result.add_error(f"{context}.stop_loss: indicator type requires 'indicator'")
            elif stop_loss.indicator not in definition.indicators:
                result.add_error(
                    f"{context}.stop_loss: unknown indicator '{stop_loss.indicator}'"
                )

        return result

    def _validate_take_profit(
        self,
        take_profit: TakeProfitDefinition,
        context: str,
        definition: StrategyDefinition,
    ) -> ValidationResult:
        """Validate take profit configuration."""
        result = ValidationResult()

        if take_profit.tp_type == TakeProfitType.FIXED:
            if take_profit.value is None:
                result.add_error(f"{context}.take_profit: fixed type requires 'value'")

        elif take_profit.tp_type == TakeProfitType.PERCENTAGE:
            if take_profit.value is None:
                result.add_error(f"{context}.take_profit: percentage type requires 'value'")

        elif take_profit.tp_type == TakeProfitType.ATR_BASED:
            if take_profit.multiplier <= 0:
                result.add_error(f"{context}.take_profit: ATR multiplier must be positive")

        elif take_profit.tp_type == TakeProfitType.RISK_REWARD:
            if take_profit.multiplier <= 0:
                result.add_error(f"{context}.take_profit: risk-reward ratio must be positive")

        elif take_profit.tp_type == TakeProfitType.INDICATOR:
            if take_profit.indicator is None:
                result.add_error(f"{context}.take_profit: indicator type requires 'indicator'")
            elif take_profit.indicator not in definition.indicators:
                result.add_error(
                    f"{context}.take_profit: unknown indicator '{take_profit.indicator}'"
                )

        return result

    def _validate_structure(self, definition: StrategyDefinition) -> ValidationResult:
        """Validate overall strategy structure."""
        result = ValidationResult()

        # Must have at least one entry
        if definition.entry_long is None and definition.entry_short is None:
            result.add_error("Strategy must define at least one entry (long or short)")

        # If there's an entry, there should be an exit (warning)
        if definition.entry_long and not definition.exit_long:
            result.add_warning("Long entry defined but no explicit long exit conditions")

        if definition.entry_short and not definition.exit_short:
            result.add_warning("Short entry defined but no explicit short exit conditions")

        # Must have at least one indicator
        if not definition.indicators:
            result.add_error("Strategy must define at least one indicator")

        return result
