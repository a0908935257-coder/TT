"""
Configuration Validator.

Provides comprehensive configuration validation including:
- Parameter range validation with clear units
- Type coercion and normalization
- Configuration checksum for sync verification
- Hot reload support
- Version tracking
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Generic

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Units Definition
# =============================================================================


class Unit(Enum):
    """Parameter unit types for clarity."""

    # Percentage units
    PERCENT_DECIMAL = "decimal"  # 0.05 = 5%
    PERCENT_WHOLE = "whole"      # 5 = 5%

    # Time units
    SECONDS = "seconds"
    MILLISECONDS = "milliseconds"
    MINUTES = "minutes"
    HOURS = "hours"

    # Count units
    COUNT = "count"

    # Price/Value units
    PRICE = "price"
    QUANTITY = "quantity"

    # Multiplier
    MULTIPLIER = "multiplier"

    # Ratio (0-1)
    RATIO = "ratio"


# =============================================================================
# Validation Rules
# =============================================================================


@dataclass
class ValidationRule:
    """Defines validation rules for a parameter."""

    name: str
    unit: Unit
    min_value: Optional[Decimal] = None
    max_value: Optional[Decimal] = None
    allowed_values: Optional[List[Any]] = None
    required: bool = True
    default: Any = None
    description: str = ""

    def validate(self, value: Any) -> Tuple[bool, str, Any]:
        """
        Validate a value against this rule.

        Returns:
            Tuple of (is_valid, error_message, normalized_value)
        """
        # Handle None
        if value is None:
            if self.required and self.default is None:
                return False, f"{self.name} is required", None
            return True, "", self.default

        # Convert to Decimal for numeric validation
        try:
            if isinstance(value, (int, float, str)):
                decimal_value = Decimal(str(value))
            elif isinstance(value, Decimal):
                decimal_value = value
            else:
                decimal_value = None
        except Exception:
            decimal_value = None

        # Check allowed values
        if self.allowed_values is not None:
            if value not in self.allowed_values:
                return False, f"{self.name} must be one of {self.allowed_values}, got {value}", None
            return True, "", value

        # Range validation for numeric types
        if decimal_value is not None:
            if self.min_value is not None and decimal_value < self.min_value:
                return False, f"{self.name} must be >= {self.min_value} ({self._unit_hint()}), got {value}", None
            if self.max_value is not None and decimal_value > self.max_value:
                return False, f"{self.name} must be <= {self.max_value} ({self._unit_hint()}), got {value}", None
            return True, "", decimal_value

        return True, "", value

    def _unit_hint(self) -> str:
        """Get human-readable unit hint."""
        hints = {
            Unit.PERCENT_DECIMAL: "as decimal, e.g., 0.05 = 5%",
            Unit.PERCENT_WHOLE: "as whole number, e.g., 5 = 5%",
            Unit.SECONDS: "in seconds",
            Unit.MILLISECONDS: "in milliseconds",
            Unit.MINUTES: "in minutes",
            Unit.HOURS: "in hours",
            Unit.COUNT: "count",
            Unit.MULTIPLIER: "multiplier",
            Unit.RATIO: "ratio 0-1",
        }
        return hints.get(self.unit, str(self.unit.value))


# =============================================================================
# Common Validation Rules
# =============================================================================


# Percentage rules (decimal format: 0.05 = 5%)
PERCENTAGE_RULES = {
    "stop_loss_pct": ValidationRule(
        name="stop_loss_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.001"),   # 0.1%
        max_value=Decimal("0.50"),    # 50%
        description="Stop loss percentage (0.05 = 5%)",
    ),
    "take_profit_pct": ValidationRule(
        name="take_profit_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.001"),
        max_value=Decimal("1.0"),
        description="Take profit percentage (0.10 = 10%)",
    ),
    "position_size_pct": ValidationRule(
        name="position_size_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.01"),    # 1%
        max_value=Decimal("1.0"),     # 100%
        description="Position size as % of capital (0.10 = 10%)",
    ),
    "max_position_pct": ValidationRule(
        name="max_position_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.1"),     # 10%
        max_value=Decimal("1.0"),     # 100%
        description="Maximum position as % of capital (0.50 = 50%)",
    ),
    "grid_range_pct": ValidationRule(
        name="grid_range_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.01"),    # 1%
        max_value=Decimal("0.50"),    # 50%
        description="Grid range percentage (0.04 = 4%)",
    ),
    "daily_loss_limit_pct": ValidationRule(
        name="daily_loss_limit_pct",
        unit=Unit.PERCENT_DECIMAL,
        min_value=Decimal("0.01"),
        max_value=Decimal("0.50"),
        description="Daily loss limit (0.05 = 5%)",
    ),
}

# Count rules
COUNT_RULES = {
    "leverage": ValidationRule(
        name="leverage",
        unit=Unit.MULTIPLIER,
        min_value=Decimal("1"),
        max_value=Decimal("125"),
        description="Futures leverage multiplier (1-125x)",
    ),
    "grid_count": ValidationRule(
        name="grid_count",
        unit=Unit.COUNT,
        min_value=Decimal("3"),
        max_value=Decimal("100"),
        description="Number of grid levels (3-100)",
    ),
    "atr_period": ValidationRule(
        name="atr_period",
        unit=Unit.COUNT,
        min_value=Decimal("5"),
        max_value=Decimal("50"),
        description="ATR calculation period (5-50)",
    ),
    "bb_period": ValidationRule(
        name="bb_period",
        unit=Unit.COUNT,
        min_value=Decimal("5"),
        max_value=Decimal("100"),
        description="Bollinger Bands period (5-100)",
    ),
    "rsi_period": ValidationRule(
        name="rsi_period",
        unit=Unit.COUNT,
        min_value=Decimal("5"),
        max_value=Decimal("50"),
        description="RSI calculation period (5-50)",
    ),
}

# Multiplier rules
MULTIPLIER_RULES = {
    "atr_multiplier": ValidationRule(
        name="atr_multiplier",
        unit=Unit.MULTIPLIER,
        min_value=Decimal("0.5"),
        max_value=Decimal("10.0"),
        description="ATR multiplier for bands/range (0.5-10.0)",
    ),
    "bb_std": ValidationRule(
        name="bb_std",
        unit=Unit.MULTIPLIER,
        min_value=Decimal("0.5"),
        max_value=Decimal("5.0"),
        description="Bollinger Bands standard deviation multiplier (0.5-5.0)",
    ),
}

# Timeframe rules
TIMEFRAME_RULE = ValidationRule(
    name="timeframe",
    unit=Unit.COUNT,
    allowed_values=["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d"],
    description="Kline timeframe",
)

# Margin type rules
MARGIN_TYPE_RULE = ValidationRule(
    name="margin_type",
    unit=Unit.COUNT,
    allowed_values=["ISOLATED", "CROSSED"],
    description="Futures margin type",
)


# =============================================================================
# Configuration Validator
# =============================================================================


class ConfigValidator:
    """
    Validates configuration parameters against defined rules.

    Features:
    - Parameter range validation with clear units
    - Type coercion and normalization
    - Detailed error messages with unit hints
    - Configuration checksum for sync verification
    """

    def __init__(self, rules: Optional[Dict[str, ValidationRule]] = None):
        """
        Initialize validator with rules.

        Args:
            rules: Dictionary of parameter name to ValidationRule
        """
        # Combine all default rules
        self._rules: Dict[str, ValidationRule] = {}
        self._rules.update(PERCENTAGE_RULES)
        self._rules.update(COUNT_RULES)
        self._rules.update(MULTIPLIER_RULES)
        self._rules["timeframe"] = TIMEFRAME_RULE
        self._rules["margin_type"] = MARGIN_TYPE_RULE

        # Add custom rules
        if rules:
            self._rules.update(rules)

    def add_rule(self, rule: ValidationRule) -> None:
        """Add a validation rule."""
        self._rules[rule.name] = rule

    def validate(
        self,
        config: Dict[str, Any],
        strict: bool = False,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate a configuration dictionary.

        Args:
            config: Configuration to validate
            strict: If True, fail on unknown parameters

        Returns:
            Tuple of (is_valid, errors, normalized_config)
        """
        errors: List[str] = []
        normalized: Dict[str, Any] = {}

        for key, value in config.items():
            if key in self._rules:
                rule = self._rules[key]
                is_valid, error, norm_value = rule.validate(value)

                if not is_valid:
                    errors.append(error)
                else:
                    normalized[key] = norm_value
            else:
                if strict:
                    errors.append(f"Unknown parameter: {key}")
                else:
                    normalized[key] = value

        # Check required rules not in config
        for name, rule in self._rules.items():
            if name not in config and rule.required and rule.default is None:
                errors.append(f"Missing required parameter: {name}")

        return len(errors) == 0, errors, normalized

    def validate_single(
        self,
        name: str,
        value: Any,
    ) -> Tuple[bool, str, Any]:
        """
        Validate a single parameter.

        Args:
            name: Parameter name
            value: Parameter value

        Returns:
            Tuple of (is_valid, error, normalized_value)
        """
        if name not in self._rules:
            return True, "", value

        return self._rules[name].validate(value)

    def get_rule(self, name: str) -> Optional[ValidationRule]:
        """Get validation rule for a parameter."""
        return self._rules.get(name)

    def get_all_rules(self) -> Dict[str, ValidationRule]:
        """Get all validation rules."""
        return self._rules.copy()


# =============================================================================
# Configuration Checksum
# =============================================================================


def compute_config_checksum(config: Dict[str, Any]) -> str:
    """
    Compute checksum of configuration for sync verification.

    Args:
        config: Configuration dictionary

    Returns:
        SHA256 checksum (first 16 characters)
    """
    # Sort keys for consistent hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def verify_config_checksum(config: Dict[str, Any], expected: str) -> bool:
    """
    Verify configuration matches expected checksum.

    Args:
        config: Configuration to verify
        expected: Expected checksum

    Returns:
        True if checksum matches
    """
    actual = compute_config_checksum(config)
    return actual == expected


# =============================================================================
# Configuration Sync Checker
# =============================================================================


@dataclass
class ConfigSyncStatus:
    """Status of configuration synchronization."""

    is_synced: bool
    local_checksum: str
    remote_checksum: Optional[str]
    differences: List[str]
    checked_at: datetime


class ConfigSyncChecker:
    """
    Checks configuration synchronization across nodes.

    Used to detect configuration drift in multi-node deployments.
    """

    def __init__(
        self,
        get_remote_config: Optional[Callable[[], Dict[str, Any]]] = None,
    ):
        """
        Initialize sync checker.

        Args:
            get_remote_config: Function to get remote/master configuration
        """
        self._get_remote = get_remote_config
        self._last_check: Optional[ConfigSyncStatus] = None

    def check_sync(
        self,
        local_config: Dict[str, Any],
    ) -> ConfigSyncStatus:
        """
        Check if local config is synchronized with remote.

        Args:
            local_config: Local configuration

        Returns:
            Sync status
        """
        local_checksum = compute_config_checksum(local_config)

        if self._get_remote is None:
            # No remote to check against
            status = ConfigSyncStatus(
                is_synced=True,
                local_checksum=local_checksum,
                remote_checksum=None,
                differences=[],
                checked_at=datetime.now(timezone.utc),
            )
            self._last_check = status
            return status

        try:
            remote_config = self._get_remote()
            remote_checksum = compute_config_checksum(remote_config)

            # Find differences
            differences = self._find_differences(local_config, remote_config)

            status = ConfigSyncStatus(
                is_synced=local_checksum == remote_checksum,
                local_checksum=local_checksum,
                remote_checksum=remote_checksum,
                differences=differences,
                checked_at=datetime.now(timezone.utc),
            )

        except Exception as e:
            logger.error(f"Failed to check config sync: {e}")
            status = ConfigSyncStatus(
                is_synced=False,
                local_checksum=local_checksum,
                remote_checksum=None,
                differences=[f"Failed to fetch remote config: {e}"],
                checked_at=datetime.now(timezone.utc),
            )

        self._last_check = status
        return status

    def _find_differences(
        self,
        local: Dict[str, Any],
        remote: Dict[str, Any],
        prefix: str = "",
    ) -> List[str]:
        """Find differences between two configs."""
        differences = []

        all_keys = set(local.keys()) | set(remote.keys())

        for key in all_keys:
            full_key = f"{prefix}.{key}" if prefix else key

            local_val = local.get(key)
            remote_val = remote.get(key)

            if key not in local:
                differences.append(f"Missing locally: {full_key}")
            elif key not in remote:
                differences.append(f"Extra locally: {full_key}")
            elif isinstance(local_val, dict) and isinstance(remote_val, dict):
                differences.extend(
                    self._find_differences(local_val, remote_val, full_key)
                )
            elif local_val != remote_val:
                differences.append(
                    f"Different: {full_key} (local={local_val}, remote={remote_val})"
                )

        return differences

    @property
    def last_check(self) -> Optional[ConfigSyncStatus]:
        """Get last sync check result."""
        return self._last_check


# =============================================================================
# Version Tracker
# =============================================================================


@dataclass
class DeploymentInfo:
    """Information about current deployment."""

    version: str
    git_commit: Optional[str]
    git_branch: Optional[str]
    deployed_at: datetime
    deployed_by: str
    environment: str
    config_checksum: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "git_commit": self.git_commit,
            "git_branch": self.git_branch,
            "deployed_at": self.deployed_at.isoformat(),
            "deployed_by": self.deployed_by,
            "environment": self.environment,
            "config_checksum": self.config_checksum,
        }


def get_deployment_info(
    config: Dict[str, Any],
    version: str = "1.0.0",
    environment: str = "production",
) -> DeploymentInfo:
    """
    Get current deployment information.

    Args:
        config: Current configuration
        version: Application version
        environment: Deployment environment

    Returns:
        Deployment information
    """
    import subprocess

    # Try to get git info
    git_commit = None
    git_branch = None

    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()[:8]
    except Exception:
        pass

    try:
        git_branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    return DeploymentInfo(
        version=version,
        git_commit=git_commit,
        git_branch=git_branch,
        deployed_at=datetime.now(timezone.utc),
        deployed_by="system",
        environment=environment,
        config_checksum=compute_config_checksum(config),
    )


# =============================================================================
# Hot Reload Support
# =============================================================================


class ConfigHotReloader:
    """
    Supports hot-reloading of configuration.

    Features:
    - Watch for config file changes
    - Validate new config before applying
    - Notify listeners of config changes
    - Rollback on validation failure
    """

    def __init__(
        self,
        validator: ConfigValidator,
        on_reload: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ):
        """
        Initialize hot reloader.

        Args:
            validator: Configuration validator
            on_reload: Callback when config is reloaded (return True to accept)
        """
        self._validator = validator
        self._on_reload = on_reload
        self._current_config: Dict[str, Any] = {}
        self._reload_listeners: List[Callable[[Dict[str, Any], Dict[str, Any]], Any]] = []
        self._last_reload: Optional[datetime] = None
        self._reload_count: int = 0

    def set_config(self, config: Dict[str, Any]) -> None:
        """Set current configuration."""
        self._current_config = config.copy()

    def reload(
        self,
        new_config: Dict[str, Any],
        source: str = "manual",
    ) -> Tuple[bool, List[str]]:
        """
        Attempt to reload configuration.

        Args:
            new_config: New configuration to apply
            source: Source of the reload (e.g., "file", "api", "manual")

        Returns:
            Tuple of (success, errors)
        """
        # Validate new config
        is_valid, errors, normalized = self._validator.validate(new_config)

        if not is_valid:
            logger.error(f"Config reload failed validation: {errors}")
            return False, errors

        # Check if config actually changed
        old_checksum = compute_config_checksum(self._current_config)
        new_checksum = compute_config_checksum(normalized)

        if old_checksum == new_checksum:
            logger.info("Config unchanged, skipping reload")
            return True, []

        # Call on_reload callback
        if self._on_reload:
            try:
                accepted = self._on_reload(normalized)
                if not accepted:
                    return False, ["Reload rejected by callback"]
            except Exception as e:
                logger.error(f"Reload callback error: {e}")
                return False, [f"Callback error: {e}"]

        # Apply new config
        old_config = self._current_config.copy()
        self._current_config = normalized.copy()
        self._last_reload = datetime.now(timezone.utc)
        self._reload_count += 1

        logger.info(
            f"Config reloaded from {source} "
            f"(checksum: {old_checksum[:8]} -> {new_checksum[:8]})"
        )

        # Notify listeners
        for listener in self._reload_listeners:
            try:
                listener(old_config, normalized)
            except Exception as e:
                logger.error(f"Reload listener error: {e}")

        return True, []

    def add_reload_listener(
        self,
        listener: Callable[[Dict[str, Any], Dict[str, Any]], Any],
    ) -> Callable[[], None]:
        """
        Add a listener for config reloads.

        Args:
            listener: Callback(old_config, new_config)

        Returns:
            Unsubscribe function
        """
        self._reload_listeners.append(listener)

        def unsubscribe():
            if listener in self._reload_listeners:
                self._reload_listeners.remove(listener)

        return unsubscribe

    @property
    def current_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self._current_config.copy()

    @property
    def last_reload(self) -> Optional[datetime]:
        """Get last reload time."""
        return self._last_reload

    @property
    def reload_count(self) -> int:
        """Get reload count."""
        return self._reload_count


# =============================================================================
# Unit Conversion Helpers
# =============================================================================


def pct_to_decimal(value: float) -> Decimal:
    """
    Convert percentage (5) to decimal (0.05).

    Args:
        value: Percentage as whole number (e.g., 5 for 5%)

    Returns:
        Decimal representation (e.g., 0.05)
    """
    return Decimal(str(value)) / Decimal("100")


def decimal_to_pct(value: Decimal) -> float:
    """
    Convert decimal (0.05) to percentage (5).

    Args:
        value: Decimal representation (e.g., 0.05)

    Returns:
        Percentage as whole number (e.g., 5.0)
    """
    return float(value * Decimal("100"))


def normalize_percentage(
    value: Any,
    expected_unit: Unit = Unit.PERCENT_DECIMAL,
) -> Decimal:
    """
    Normalize a percentage value to expected unit.

    Detects if value is likely in wrong unit and converts.

    Args:
        value: Input value
        expected_unit: Expected unit type

    Returns:
        Normalized Decimal value
    """
    decimal_val = Decimal(str(value))

    if expected_unit == Unit.PERCENT_DECIMAL:
        # If value > 1, assume it's in whole percentage format
        if decimal_val > Decimal("1"):
            logger.warning(
                f"Auto-converting percentage {value} to decimal format "
                f"(assuming {value}% -> {decimal_val / 100})"
            )
            return decimal_val / Decimal("100")
        return decimal_val

    elif expected_unit == Unit.PERCENT_WHOLE:
        # If value < 1, assume it's in decimal format
        if decimal_val < Decimal("1"):
            logger.warning(
                f"Auto-converting percentage {value} to whole format "
                f"(assuming {value} -> {decimal_val * 100}%)"
            )
            return decimal_val * Decimal("100")
        return decimal_val

    return decimal_val
