"""
DSL Condition Evaluator.

Evaluates condition groups and rules against indicator values.
"""

from decimal import Decimal
from typing import Any, Optional

from .models import (
    ComparisonOperator,
    ConditionGroup,
    ConditionRule,
    LogicalOperator,
)


class ConditionEvaluationError(Exception):
    """Exception raised when condition evaluation fails."""
    pass


class ConditionEvaluator:
    """
    Evaluates DSL conditions against indicator values.

    Supports:
    - Comparison operators: ==, !=, >, >=, <, <=, cross_above, cross_below
    - Logical operators: AND, OR
    - Nested condition groups
    - Indicator-to-indicator comparisons
    - Parameter and special value references

    Example:
        evaluator = ConditionEvaluator()
        result = evaluator.evaluate(
            condition_group,
            indicator_values={"bollinger": {"lower": 100, "upper": 110}},
            param_values={"threshold": 5},
            price_data={"close": 105, "open": 103},
            prev_indicator_values={"bollinger": {"lower": 99, "upper": 109}},
        )
    """

    def evaluate(
        self,
        group: ConditionGroup,
        indicator_values: dict[str, dict[str, Any]],
        param_values: Optional[dict[str, Any]] = None,
        price_data: Optional[dict[str, Any]] = None,
        prev_indicator_values: Optional[dict[str, dict[str, Any]]] = None,
    ) -> bool:
        """
        Evaluate a condition group.

        Args:
            group: The condition group to evaluate
            indicator_values: Current indicator values {name: {field: value}}
            param_values: Parameter values for reference resolution
            price_data: Price data for special references ($close, etc.)
            prev_indicator_values: Previous bar's indicator values (for cross detection)

        Returns:
            True if conditions are met, False otherwise
        """
        param_values = param_values or {}
        price_data = price_data or {}
        prev_indicator_values = prev_indicator_values or {}

        # Evaluate rules
        rule_results = [
            self._evaluate_rule(
                rule, indicator_values, param_values, price_data, prev_indicator_values
            )
            for rule in group.rules
        ]

        # Evaluate nested groups
        group_results = [
            self.evaluate(
                nested_group, indicator_values, param_values, price_data, prev_indicator_values
            )
            for nested_group in group.groups
        ]

        # Combine all results
        all_results = rule_results + group_results

        if not all_results:
            # Empty group evaluates to True (no conditions = always true)
            return True

        # Apply logical operator
        if group.operator == LogicalOperator.AND:
            return all(all_results)
        else:  # OR
            return any(all_results)

    def _evaluate_rule(
        self,
        rule: ConditionRule,
        indicator_values: dict[str, dict[str, Any]],
        param_values: dict[str, Any],
        price_data: dict[str, Any],
        prev_indicator_values: dict[str, dict[str, Any]],
    ) -> bool:
        """Evaluate a single condition rule."""
        # Get left-hand value (indicator.field)
        left_value = self._get_indicator_value(
            rule.indicator, rule.field, indicator_values
        )

        # Get right-hand value
        if rule.is_indicator_comparison():
            # Indicator-to-indicator comparison
            right_value = self._get_indicator_value(
                rule.indicator2, rule.field2, indicator_values
            )
        else:
            # Value comparison
            right_value = self._resolve_value(rule.value, param_values, price_data)

        # Handle cross detection
        if rule.compare in (ComparisonOperator.CROSS_ABOVE, ComparisonOperator.CROSS_BELOW):
            return self._evaluate_cross(
                rule, left_value, right_value, indicator_values, prev_indicator_values,
                param_values, price_data
            )

        # Standard comparison
        return self._compare(left_value, rule.compare, right_value)

    def _get_indicator_value(
        self,
        indicator_name: str,
        field_name: str,
        indicator_values: dict[str, dict[str, Any]],
    ) -> Any:
        """Get an indicator field value."""
        if indicator_name not in indicator_values:
            raise ConditionEvaluationError(
                f"Indicator '{indicator_name}' not found in values"
            )

        indicator = indicator_values[indicator_name]
        if field_name not in indicator:
            raise ConditionEvaluationError(
                f"Field '{field_name}' not found in indicator '{indicator_name}'"
            )

        return indicator[field_name]

    def _resolve_value(
        self,
        value: Any,
        param_values: dict[str, Any],
        price_data: dict[str, Any],
    ) -> Any:
        """Resolve a value reference."""
        if not isinstance(value, str):
            return value

        if value.startswith("$"):
            ref = value[1:]

            # Check special values
            special_mapping = {
                "close": "close",
                "open": "open",
                "high": "high",
                "low": "low",
                "volume": "volume",
            }

            if ref in special_mapping:
                return price_data.get(special_mapping[ref])

            # Check parameters
            if ref in param_values:
                return param_values[ref]

            raise ConditionEvaluationError(f"Unknown reference: {value}")

        return value

    def _evaluate_cross(
        self,
        rule: ConditionRule,
        current_left: Any,
        current_right: Any,
        indicator_values: dict[str, dict[str, Any]],
        prev_indicator_values: dict[str, dict[str, Any]],
        param_values: dict[str, Any],
        price_data: dict[str, Any],
    ) -> bool:
        """Evaluate cross_above or cross_below conditions."""
        # Get previous values
        try:
            prev_left = self._get_indicator_value(
                rule.indicator, rule.field, prev_indicator_values
            )
        except ConditionEvaluationError:
            # No previous value available, can't detect cross
            return False

        if rule.is_indicator_comparison():
            try:
                prev_right = self._get_indicator_value(
                    rule.indicator2, rule.field2, prev_indicator_values
                )
            except ConditionEvaluationError:
                return False
        else:
            prev_right = self._resolve_value(rule.value, param_values, price_data)

        # Convert to comparable types
        current_left = self._to_decimal(current_left)
        current_right = self._to_decimal(current_right)
        prev_left = self._to_decimal(prev_left)
        prev_right = self._to_decimal(prev_right)

        if any(v is None for v in [current_left, current_right, prev_left, prev_right]):
            return False

        if rule.compare == ComparisonOperator.CROSS_ABOVE:
            # Cross above: was below or equal, now above
            return prev_left <= prev_right and current_left > current_right

        else:  # CROSS_BELOW
            # Cross below: was above or equal, now below
            return prev_left >= prev_right and current_left < current_right

    def _compare(self, left: Any, operator: ComparisonOperator, right: Any) -> bool:
        """Perform comparison operation."""
        # Convert to Decimal for precise comparison
        left_dec = self._to_decimal(left)
        right_dec = self._to_decimal(right)

        if left_dec is None or right_dec is None:
            # Fall back to direct comparison for non-numeric types
            if operator == ComparisonOperator.EQ:
                return left == right
            elif operator == ComparisonOperator.NE:
                return left != right
            else:
                # Can't do numeric comparison
                return False

        if operator == ComparisonOperator.EQ:
            return left_dec == right_dec
        elif operator == ComparisonOperator.NE:
            return left_dec != right_dec
        elif operator == ComparisonOperator.GT:
            return left_dec > right_dec
        elif operator == ComparisonOperator.GE:
            return left_dec >= right_dec
        elif operator == ComparisonOperator.LT:
            return left_dec < right_dec
        elif operator == ComparisonOperator.LE:
            return left_dec <= right_dec
        else:
            raise ConditionEvaluationError(f"Unexpected operator: {operator}")

    def _to_decimal(self, value: Any) -> Optional[Decimal]:
        """Convert value to Decimal if possible."""
        if value is None:
            return None
        if isinstance(value, Decimal):
            return value
        if isinstance(value, (int, float)):
            return Decimal(str(value))
        if isinstance(value, str):
            try:
                return Decimal(value)
            except Exception:
                return None
        return None
