"""
DSL Strategy Module.

Provides a declarative YAML-based language for defining trading strategies.

Example:
    from src.backtest.strategy.dsl import DSLParser, DSLStrategyGenerator

    # Parse YAML strategy
    parser = DSLParser()
    definition = parser.parse_file("my_strategy.yaml")

    # Validate
    validator = DSLValidator()
    result = validator.validate(definition)
    if not result.is_valid:
        print("Errors:", result.errors)

    # Generate strategy class
    generator = DSLStrategyGenerator()
    StrategyClass = generator.generate(definition)

    # Create instance with optional parameter overrides
    strategy = StrategyClass(param_overrides={"bb_period": 30})
"""

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
from .parser import DSLParser, DSLParseError
from .validator import DSLValidator, DSLValidationError, ValidationResult
from .conditions import ConditionEvaluator, ConditionEvaluationError
from .indicators import (
    BaseIndicator,
    IndicatorFactory,
    IndicatorResult,
    IndicatorError,
    BollingerBandsIndicator,
    SupertrendIndicator,
    SMAIndicator,
    EMAIndicator,
    RSIIndicator,
    ATRIndicator,
    MACDIndicator,
)
from .generator import (
    DSLStrategyGenerator,
    DSLGeneratedStrategy,
    DSLStrategyGenerationError,
)

__all__ = [
    # Models
    "ComparisonOperator",
    "ConditionGroup",
    "ConditionRule",
    "EntryDefinition",
    "ExitDefinition",
    "IndicatorDefinition",
    "IndicatorParams",
    "LogicalOperator",
    "ParameterDefinition",
    "ParameterType",
    "StopLossDefinition",
    "StopLossType",
    "StrategyDefinition",
    "StrategyMetadata",
    "TakeProfitDefinition",
    "TakeProfitType",
    # Parser
    "DSLParser",
    "DSLParseError",
    # Validator
    "DSLValidator",
    "DSLValidationError",
    "ValidationResult",
    # Conditions
    "ConditionEvaluator",
    "ConditionEvaluationError",
    # Indicators
    "BaseIndicator",
    "IndicatorFactory",
    "IndicatorResult",
    "IndicatorError",
    "BollingerBandsIndicator",
    "SupertrendIndicator",
    "SMAIndicator",
    "EMAIndicator",
    "RSIIndicator",
    "ATRIndicator",
    "MACDIndicator",
    # Generator
    "DSLStrategyGenerator",
    "DSLGeneratedStrategy",
    "DSLStrategyGenerationError",
]
