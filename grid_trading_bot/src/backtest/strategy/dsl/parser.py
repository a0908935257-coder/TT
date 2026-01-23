"""
DSL Strategy Parser.

Parses YAML strategy definitions into StrategyDefinition objects.
"""

from pathlib import Path
from typing import Any, Optional, Union

import yaml

from .models import (
    ComparisonOperator,
    ConditionGroup,
    ConditionRule,
    EntryDefinition,
    ExitDefinition,
    IndicatorDefinition,
    IndicatorParams,
    LogicalOperator,
    ParameterDefinition,
    ParameterType,
    StopLossDefinition,
    StopLossType,
    StrategyDefinition,
    StrategyMetadata,
    TakeProfitDefinition,
    TakeProfitType,
)


class DSLParseError(Exception):
    """Exception raised when parsing fails."""
    pass


class DSLParser:
    """
    YAML DSL Strategy Parser.

    Parses YAML files or strings into StrategyDefinition objects.

    Example:
        parser = DSLParser()
        definition = parser.parse_file("strategies/my_strategy.yaml")
        # or
        definition = parser.parse_string(yaml_content)
    """

    def parse_file(self, file_path: Union[str, Path]) -> StrategyDefinition:
        """
        Parse a YAML file into a StrategyDefinition.

        Args:
            file_path: Path to the YAML file

        Returns:
            Parsed StrategyDefinition

        Raises:
            DSLParseError: If parsing fails
            FileNotFoundError: If file doesn't exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Strategy file not found: {file_path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        return self.parse_string(content)

    def parse_string(self, yaml_content: str) -> StrategyDefinition:
        """
        Parse a YAML string into a StrategyDefinition.

        Args:
            yaml_content: YAML content as string

        Returns:
            Parsed StrategyDefinition

        Raises:
            DSLParseError: If parsing fails
        """
        try:
            data = yaml.safe_load(yaml_content)
        except yaml.YAMLError as e:
            raise DSLParseError(f"Invalid YAML syntax: {e}")

        if not isinstance(data, dict):
            raise DSLParseError("Root element must be a dictionary")

        return self._parse_strategy(data)

    def _parse_strategy(self, data: dict) -> StrategyDefinition:
        """Parse the complete strategy definition."""
        definition = StrategyDefinition()

        # Parse metadata
        if "strategy" in data:
            definition.metadata = self._parse_metadata(data["strategy"])

        # Parse parameters
        if "parameters" in data:
            definition.parameters = self._parse_parameters(data["parameters"])

        # Parse indicators
        if "indicators" in data:
            definition.indicators = self._parse_indicators(data["indicators"])

        # Parse entry conditions
        if "entry" in data:
            entry_data = data["entry"]
            if "long" in entry_data:
                definition.entry_long = self._parse_entry(entry_data["long"])
            if "short" in entry_data:
                definition.entry_short = self._parse_entry(entry_data["short"])

        # Parse exit conditions
        if "exit" in data:
            exit_data = data["exit"]
            if "long" in exit_data:
                definition.exit_long = self._parse_exit(exit_data["long"])
            if "short" in exit_data:
                definition.exit_short = self._parse_exit(exit_data["short"])

        return definition

    def _parse_metadata(self, data: dict) -> StrategyMetadata:
        """Parse strategy metadata."""
        return StrategyMetadata(
            name=data.get("name", "Unnamed Strategy"),
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            author=data.get("author", ""),
            tags=data.get("tags", []),
        )

    def _parse_parameters(self, data: dict) -> dict[str, ParameterDefinition]:
        """Parse parameter definitions."""
        parameters = {}

        for name, param_data in data.items():
            param_type = ParameterType(param_data.get("type", "float"))

            # Parse range
            range_min = None
            range_max = None
            if "range" in param_data:
                range_val = param_data["range"]
                if isinstance(range_val, list) and len(range_val) == 2:
                    range_min, range_max = range_val

            parameters[name] = ParameterDefinition(
                name=name,
                param_type=param_type,
                default=param_data.get("default"),
                range_min=range_min,
                range_max=range_max,
                step=param_data.get("step"),
                choices=param_data.get("choices"),
                description=param_data.get("description", ""),
            )

        return parameters

    def _parse_indicators(self, data: dict) -> dict[str, IndicatorDefinition]:
        """Parse indicator definitions."""
        indicators = {}

        for name, ind_data in data.items():
            indicator_type = ind_data.get("type", "")
            if not indicator_type:
                raise DSLParseError(f"Indicator '{name}' missing 'type'")

            params = IndicatorParams(params=ind_data.get("params", {}))

            indicators[name] = IndicatorDefinition(
                name=name,
                indicator_type=indicator_type,
                params=params,
            )

        return indicators

    def _parse_entry(self, data: dict) -> EntryDefinition:
        """Parse entry definition."""
        entry = EntryDefinition()

        # Parse conditions
        if "conditions" in data:
            conditions = data["conditions"]
            if isinstance(conditions, list):
                # List of conditions treated as AND
                entry.conditions = self._parse_condition_group(
                    {"operator": "AND", "rules": conditions}
                )
            elif isinstance(conditions, dict):
                entry.conditions = self._parse_condition_group(conditions)

        # Parse stop loss
        if "stop_loss" in data:
            entry.stop_loss = self._parse_stop_loss(data["stop_loss"])

        # Parse take profit
        if "take_profit" in data:
            entry.take_profit = self._parse_take_profit(data["take_profit"])

        return entry

    def _parse_exit(self, data: dict) -> ExitDefinition:
        """Parse exit definition."""
        exit_def = ExitDefinition()

        # Parse conditions
        if "conditions" in data:
            conditions = data["conditions"]
            if isinstance(conditions, list):
                # List of conditions treated as OR for exits
                exit_def.conditions = self._parse_condition_group(
                    {"operator": "OR", "rules": conditions}
                )
            elif isinstance(conditions, dict):
                exit_def.conditions = self._parse_condition_group(conditions)

        return exit_def

    def _parse_condition_group(self, data: Union[dict, list]) -> ConditionGroup:
        """Parse a condition group."""
        if isinstance(data, list):
            # List of rules with implicit AND
            return ConditionGroup(
                operator=LogicalOperator.AND,
                rules=[self._parse_condition_rule(r) for r in data],
            )

        operator = LogicalOperator(data.get("operator", "AND"))
        rules = []
        groups = []

        # Parse rules
        if "rules" in data:
            for rule_data in data["rules"]:
                if "operator" in rule_data:
                    # Nested group
                    groups.append(self._parse_condition_group(rule_data))
                else:
                    rules.append(self._parse_condition_rule(rule_data))

        return ConditionGroup(
            operator=operator,
            rules=rules,
            groups=groups,
        )

    def _parse_condition_rule(self, data: dict) -> ConditionRule:
        """Parse a single condition rule."""
        indicator = data.get("indicator", "")
        field = data.get("field", "")
        compare_str = data.get("compare", "==")

        # Map string to ComparisonOperator
        compare = self._map_comparison_operator(compare_str)

        # Check for indicator-to-indicator comparison
        value = data.get("value")
        indicator2 = data.get("indicator2")
        field2 = data.get("field2")

        # Handle special value references
        if isinstance(value, str):
            if value.startswith("$"):
                # Parameter or special value reference
                if value in ("$close", "$open", "$high", "$low", "$volume"):
                    # Keep as special reference
                    pass
            elif value.startswith("indicator:"):
                # Indicator field reference: "indicator:name.field"
                ref = value[10:]  # Remove "indicator:" prefix
                parts = ref.split(".")
                if len(parts) == 2:
                    indicator2, field2 = parts
                    value = None

        return ConditionRule(
            indicator=indicator,
            field=field,
            compare=compare,
            value=value,
            indicator2=indicator2,
            field2=field2,
        )

    def _map_comparison_operator(self, op_str: str) -> ComparisonOperator:
        """Map string to ComparisonOperator enum."""
        mapping = {
            "==": ComparisonOperator.EQ,
            "!=": ComparisonOperator.NE,
            ">": ComparisonOperator.GT,
            ">=": ComparisonOperator.GE,
            "<": ComparisonOperator.LT,
            "<=": ComparisonOperator.LE,
            "cross_above": ComparisonOperator.CROSS_ABOVE,
            "cross_below": ComparisonOperator.CROSS_BELOW,
        }
        if op_str in mapping:
            return mapping[op_str]
        raise DSLParseError(f"Unknown comparison operator: {op_str}")

    def _parse_stop_loss(self, data: dict) -> StopLossDefinition:
        """Parse stop loss definition."""
        sl_type_str = data.get("type", "atr_based")
        sl_type = StopLossType(sl_type_str)

        return StopLossDefinition(
            sl_type=sl_type,
            value=data.get("value"),
            multiplier=data.get("multiplier", 2.0),
            indicator=data.get("indicator"),
            field=data.get("field"),
        )

    def _parse_take_profit(self, data: dict) -> TakeProfitDefinition:
        """Parse take profit definition."""
        tp_type_str = data.get("type", "risk_reward")
        tp_type = TakeProfitType(tp_type_str)

        return TakeProfitDefinition(
            tp_type=tp_type,
            value=data.get("value"),
            multiplier=data.get("multiplier", 2.0),
            indicator=data.get("indicator"),
            field=data.get("field"),
        )
