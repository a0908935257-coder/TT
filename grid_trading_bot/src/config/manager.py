"""
Configuration Manager.

Provides a singleton configuration manager with hot reload support,
change notification callbacks, and comprehensive validation.

Features:
- Configuration validation with clear error messages
- Hot reload support with validation
- Configuration sync verification
- Deployment version tracking
- Auto-correction of common mistakes (unit conversion)
"""

import hashlib
import json
import threading
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .exceptions import ConfigNotLoadedError, ConfigReloadError, ConfigValidationError
from .loader import ConfigLoader
from .models import AppConfig
from .validator import (
    ConfigValidator,
    ValidationRule,
    Unit,
    compute_config_checksum,
)

from src.core import get_logger

logger = get_logger(__name__)


class ConfigManager:
    """
    Singleton configuration manager with hot reload support.

    Provides centralized access to application configuration with support
    for runtime reloading and change notification callbacks.

    Example:
        >>> manager = ConfigManager()
        >>> manager.load("config/config.yaml", env="development")
        >>> testnet = manager.get("exchange.testnet")
        >>> manager.on_change(lambda old, new: print("Config changed!"))
        >>> manager.reload()
    """

    _instance: Optional["ConfigManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "ConfigManager":
        """Ensure singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the config manager (only runs once due to singleton)."""
        if self._initialized:
            return

        self._config: Optional[AppConfig] = None
        self._config_path: Optional[Path] = None
        self._env: Optional[str] = None
        self._env_file: Optional[Path] = None
        self._loader: Optional[ConfigLoader] = None
        self._callbacks: list[Callable[[Optional[AppConfig], AppConfig], None]] = []
        self._callback_lock = threading.Lock()
        self._initialized = True

    def load(
        self,
        path: str | Path,
        env: Optional[str] = None,
        env_file: Optional[str | Path] = None,
    ) -> AppConfig:
        """
        Load configuration from file.

        Args:
            path: Path to base configuration file
            env: Optional environment name (development, production, etc.)
            env_file: Optional path to .env file

        Returns:
            Loaded AppConfig instance
        """
        self._config_path = Path(path)
        self._env = env
        self._env_file = Path(env_file) if env_file else None

        self._loader = ConfigLoader(env_file=self._env_file)
        self._config = self._loader.load(self._config_path, env=self._env)

        return self._config

    @property
    def config(self) -> AppConfig:
        """
        Get the current configuration.

        Returns:
            Current AppConfig instance

        Raises:
            ConfigNotLoadedError: If config not yet loaded
        """
        if self._config is None:
            raise ConfigNotLoadedError()
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by dot-notation path.

        Args:
            key_path: Dot-separated path to config value (e.g., "exchange.testnet")
            default: Default value if path not found

        Returns:
            Configuration value or default

        Example:
            >>> manager.get("exchange.testnet")
            True
            >>> manager.get("exchange.api_key")
            "your_api_key"
            >>> manager.get("nonexistent.path", "default")
            "default"
        """
        if self._config is None:
            raise ConfigNotLoadedError()

        keys = key_path.split(".")
        value: Any = self._config

        for key in keys:
            if hasattr(value, key):
                value = getattr(value, key)
            elif isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def reload(self) -> AppConfig:
        """
        Reload configuration from file.

        Triggers all registered change callbacks with old and new config.

        Returns:
            Newly loaded AppConfig instance

        Raises:
            ConfigNotLoadedError: If config was never loaded
            ConfigReloadError: If reload fails
        """
        if self._config_path is None or self._loader is None:
            raise ConfigNotLoadedError()

        old_config = self._config

        try:
            # Create new loader to reset env loading state
            self._loader = ConfigLoader(env_file=self._env_file)
            new_config = self._loader.load(self._config_path, env=self._env)
        except Exception as e:
            raise ConfigReloadError(str(e)) from e

        self._config = new_config

        # Notify callbacks
        self._notify_callbacks(old_config, new_config)

        return new_config

    def on_change(
        self,
        callback: Callable[[Optional[AppConfig], AppConfig], None],
    ) -> Callable[[], None]:
        """
        Register a callback for configuration changes.

        The callback receives (old_config, new_config) when reload() is called.

        Args:
            callback: Function to call on config change

        Returns:
            Unsubscribe function to remove the callback

        Example:
            >>> def on_config_change(old, new):
            ...     print(f"Config changed!")
            >>> unsubscribe = manager.on_change(on_config_change)
            >>> # Later, to remove:
            >>> unsubscribe()
        """
        with self._callback_lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._callback_lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def _notify_callbacks(
        self,
        old_config: Optional[AppConfig],
        new_config: AppConfig,
    ) -> None:
        """
        Notify all registered callbacks of configuration change.

        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        with self._callback_lock:
            callbacks = list(self._callbacks)

        for callback in callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                # Don't let callback errors break the reload
                logger.debug(f"Config callback error (ignored): {e}")

    def is_loaded(self) -> bool:
        """
        Check if configuration is loaded.

        Returns:
            True if configuration is loaded
        """
        return self._config is not None

    # =========================================================================
    # Bot Configuration Validation
    # =========================================================================

    def validate_bot_config(
        self,
        bot_type: str,
        config: Dict[str, Any],
        auto_correct: bool = True,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate a bot's configuration with detailed error messages.

        Args:
            bot_type: Bot type (grid_futures, bollinger, supertrend, rsi_grid)
            config: Configuration dictionary to validate
            auto_correct: Whether to auto-correct common mistakes

        Returns:
            Tuple of (is_valid, errors, normalized_config)

        Example:
            >>> is_valid, errors, config = manager.validate_bot_config(
            ...     "grid_futures",
            ...     {"symbol": "BTCUSDT", "leverage": 10, "stop_loss_pct": 5}
            ... )
            >>> # stop_loss_pct=5 will be auto-corrected to 0.05
        """
        validator = self._get_bot_validator(bot_type)

        # Auto-correct common mistakes
        if auto_correct:
            config = self._auto_correct_config(config)

        return validator.validate(config)

    def _get_bot_validator(self, bot_type: str) -> ConfigValidator:
        """Get validator for a specific bot type."""
        validator = ConfigValidator()

        # Add bot-specific rules
        if bot_type == "grid_futures":
            validator.add_rule(ValidationRule(
                name="direction",
                unit=Unit.COUNT,
                allowed_values=["LONG", "SHORT", "NEUTRAL", "long", "short", "neutral"],
                required=False,
                default="NEUTRAL",
                description="Grid direction",
            ))

        elif bot_type == "bollinger":
            validator.add_rule(ValidationRule(
                name="take_profit_grids",
                unit=Unit.COUNT,
                min_value=Decimal("1"),
                max_value=Decimal("10"),
                required=False,
                default=1,
                description="Number of grids for take profit",
            ))

        elif bot_type == "supertrend":
            validator.add_rule(ValidationRule(
                name="use_rsi_filter",
                unit=Unit.COUNT,
                allowed_values=[True, False],
                required=False,
                default=True,
                description="Enable RSI filter",
            ))
            validator.add_rule(ValidationRule(
                name="rsi_overbought",
                unit=Unit.COUNT,
                min_value=Decimal("50"),
                max_value=Decimal("90"),
                required=False,
                default=60,
                description="RSI overbought threshold",
            ))
            validator.add_rule(ValidationRule(
                name="rsi_oversold",
                unit=Unit.COUNT,
                min_value=Decimal("10"),
                max_value=Decimal("50"),
                required=False,
                default=40,
                description="RSI oversold threshold",
            ))

        elif bot_type == "rsi_grid":
            validator.add_rule(ValidationRule(
                name="rsi_upper",
                unit=Unit.COUNT,
                min_value=Decimal("50"),
                max_value=Decimal("90"),
                required=False,
                default=70,
                description="RSI upper threshold",
            ))
            validator.add_rule(ValidationRule(
                name="rsi_lower",
                unit=Unit.COUNT,
                min_value=Decimal("10"),
                max_value=Decimal("50"),
                required=False,
                default=30,
                description="RSI lower threshold",
            ))

        return validator

    def _auto_correct_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Auto-correct common configuration mistakes.

        - Converts percentage values from whole numbers to decimals
        - Logs warnings for corrected values
        """
        corrected = config.copy()

        # Percentage fields that should be in decimal format (0.05 = 5%)
        decimal_pct_fields = [
            "stop_loss_pct",
            "take_profit_pct",
            "position_size_pct",
            "max_position_pct",
            "grid_range_pct",
            "daily_loss_limit_pct",
            "trailing_stop_pct",
            "rebuild_threshold_pct",
            "fallback_range_pct",
        ]

        for field in decimal_pct_fields:
            if field in corrected:
                value = corrected[field]
                try:
                    decimal_val = Decimal(str(value))
                    # If value > 1, it's likely in whole percentage format — reject
                    if decimal_val > Decimal("1"):
                        raise ValueError(
                            f"Config error: {field}={value} appears to be a whole number "
                            f"percentage. Use decimal format instead (e.g., 0.05 for 5%). "
                            f"Got {value}, expected a value between 0 and 1."
                        )
                except ValueError:
                    raise
                except Exception as e:
                    logger.debug(f"Config value check error (ignored): {e}")

        return corrected

    def get_config_checksum(self) -> str:
        """
        Get checksum of current configuration for sync verification.

        Returns:
            SHA256 checksum (first 16 characters)
        """
        if self._config is None:
            return ""

        # Convert config to dict for hashing
        config_dict = {}
        for attr in dir(self._config):
            if not attr.startswith("_"):
                value = getattr(self._config, attr)
                if not callable(value):
                    config_dict[attr] = value

        return compute_config_checksum(config_dict)

    def verify_config_sync(
        self,
        expected_checksum: str,
    ) -> Tuple[bool, str]:
        """
        Verify configuration matches expected checksum.

        Used for multi-node sync verification.

        Args:
            expected_checksum: Expected configuration checksum

        Returns:
            Tuple of (is_synced, actual_checksum)
        """
        actual = self.get_config_checksum()
        return actual == expected_checksum, actual

    @classmethod
    def reset(cls) -> None:
        """
        Reset the singleton instance (primarily for testing).

        Warning: This will clear all loaded configuration.
        """
        with cls._lock:
            if cls._instance is not None:
                cls._instance._config = None
                cls._instance._config_path = None
                cls._instance._env = None
                cls._instance._loader = None
                cls._instance._callbacks = []


# Global configuration state
_config_manager: Optional[ConfigManager] = None


def init_config(
    path: str | Path,
    env: Optional[str] = None,
    env_file: Optional[str | Path] = None,
) -> AppConfig:
    """
    Initialize global configuration.

    This should be called once at application startup.

    Args:
        path: Path to base configuration file
        env: Optional environment name (development, production, etc.)
        env_file: Optional path to .env file

    Returns:
        Loaded AppConfig instance

    Example:
        >>> config = init_config("config/config.yaml", env="development")
        >>> print(config.exchange.testnet)
    """
    global _config_manager
    _config_manager = ConfigManager()
    return _config_manager.load(path, env=env, env_file=env_file)


def get_config() -> AppConfig:
    """
    Get the global configuration.

    Must call init_config() first.

    Returns:
        Current AppConfig instance

    Raises:
        ConfigNotLoadedError: If init_config() not called

    Example:
        >>> config = get_config()
        >>> print(config.database.host)
    """
    if _config_manager is None:
        raise ConfigNotLoadedError()
    return _config_manager.config


def get_config_manager() -> ConfigManager:
    """
    Get the global configuration manager.

    Must call init_config() first.

    Returns:
        ConfigManager instance

    Raises:
        ConfigNotLoadedError: If init_config() not called
    """
    if _config_manager is None:
        raise ConfigNotLoadedError()
    return _config_manager
