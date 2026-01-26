"""
Configuration Loader.

Provides functionality to load and merge YAML configuration files
with environment variable substitution and validation.
"""

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv

from .exceptions import (
    ConfigFileNotFoundError,
    ConfigParseError,
    ConfigValidationError,
)
from .models import AppConfig

# Pattern to match environment variables: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


class ConfigLoader:
    """
    Configuration loader with YAML support and environment variable substitution.

    Supports loading configuration from multiple YAML files with environment-specific
    overrides and environment variable substitution.

    Example:
        >>> loader = ConfigLoader()
        >>> config = loader.load("config/config.yaml", env="development")
        >>> print(config.exchange.testnet)
    """

    def __init__(self, env_file: Optional[str | Path] = None):
        """
        Initialize ConfigLoader.

        Args:
            env_file: Optional path to .env file. If not provided,
                     will look for .env in the config directory.
        """
        self._env_file = Path(env_file) if env_file else None
        self._loaded_env = False

    def load(
        self,
        path: str | Path,
        env: Optional[str] = None,
    ) -> AppConfig:
        """
        Load configuration from YAML file with optional environment overlay.

        Loading flow:
        1. Load .env file (if exists)
        2. Load base config.yaml
        3. Load config.{env}.yaml (if env specified and file exists)
        4. Deep merge configurations
        5. Substitute environment variables
        6. Validate with Pydantic
        7. Return AppConfig

        Args:
            path: Path to base configuration file
            env: Optional environment name (development, production, etc.)

        Returns:
            Validated AppConfig instance

        Raises:
            ConfigFileNotFoundError: If base config file not found
            ConfigParseError: If YAML parsing fails
            ConfigValidationError: If Pydantic validation fails
        """
        path = Path(path)

        # Step 1: Load .env file
        self._load_env_file(path.parent)

        # Step 2: Load base configuration
        base_config = self.load_yaml(path)

        # Step 3: Load environment-specific configuration
        if env:
            env_config_path = path.parent / f"{path.stem}.{env}{path.suffix}"
            if env_config_path.exists():
                env_config = self.load_yaml(env_config_path)
                # Step 4: Deep merge
                base_config = self.merge_configs(base_config, env_config)

        # Step 5: Substitute environment variables
        final_config = self.substitute_env_vars(base_config)

        # Step 6 & 7: Pydantic validation and return
        try:
            return AppConfig(**final_config)
        except Exception as e:
            raise ConfigValidationError([str(e)]) from e

    def load_yaml(self, path: str | Path) -> dict[str, Any]:
        """
        Load a YAML configuration file.

        Args:
            path: Path to YAML file

        Returns:
            Parsed YAML as dictionary

        Raises:
            ConfigFileNotFoundError: If file not found
            ConfigParseError: If YAML parsing fails
        """
        path = Path(path)

        if not path.exists():
            raise ConfigFileNotFoundError(str(path))

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                return data if data is not None else {}
        except yaml.YAMLError as e:
            raise ConfigParseError(str(path), str(e)) from e

    def merge_configs(
        self,
        base: dict[str, Any],
        override: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Deep merge two configuration dictionaries.

        Override values take precedence. Nested dictionaries are merged recursively.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary

        Example:
            >>> base = {"a": {"b": 1, "c": 2}, "d": 3}
            >>> override = {"a": {"b": 10}, "e": 4}
            >>> merged = loader.merge_configs(base, override)
            >>> # Result: {"a": {"b": 10, "c": 2}, "d": 3, "e": 4}
        """
        result = deepcopy(base)

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self.merge_configs(result[key], value)
            else:
                # Override value
                result[key] = deepcopy(value)

        return result

    def substitute_env_vars(self, data: Any) -> Any:
        """
        Substitute environment variables in configuration data.

        Supports ${VAR} and ${VAR:default} syntax.

        Args:
            data: Configuration data (dict, list, or scalar)

        Returns:
            Data with environment variables substituted

        Example:
            >>> data = {"key": "${MY_VAR:default_value}"}
            >>> result = loader.substitute_env_vars(data)
        """
        if isinstance(data, dict):
            return {k: self.substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self.substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            return self._substitute_string(data)
        else:
            return data

    def _substitute_string(self, value: str) -> Any:
        """
        Substitute environment variables in a string value.

        Args:
            value: String potentially containing ${VAR} patterns

        Returns:
            String with substitutions, or converted type if full match
        """
        # Check if the entire string is a single env var reference
        full_match = ENV_VAR_PATTERN.fullmatch(value)
        if full_match:
            var_name, default = full_match.groups()
            env_value = os.environ.get(var_name, default)

            if env_value is None:
                return value  # Keep original if no env var and no default

            # Try to convert to appropriate type
            return self._convert_value(env_value)

        # Replace all occurrences in the string
        def replace_match(match: re.Match) -> str:
            var_name, default = match.groups()
            return os.environ.get(var_name, default if default is not None else match.group(0))

        return ENV_VAR_PATTERN.sub(replace_match, value)

    def _convert_value(self, value: str) -> Any:
        """
        Convert string value to appropriate Python type.

        Args:
            value: String value to convert

        Returns:
            Converted value (bool, int, float, or original string)
        """
        # Handle boolean
        if value.lower() in ("true", "yes", "1", "on"):
            return True
        if value.lower() in ("false", "no", "0", "off"):
            return False

        # Handle numeric values - try float first to preserve decimals
        try:
            float_val = float(value)
            # Check if it's actually an integer (no decimal part)
            if float_val.is_integer() and "." not in value and "e" not in value.lower():
                return int(float_val)
            return float_val
        except ValueError:
            pass

        # Return as string
        return value

    def _load_env_file(self, config_dir: Path) -> None:
        """
        Load .env file if not already loaded.

        Args:
            config_dir: Directory to look for .env file
        """
        if self._loaded_env:
            return

        # Check for explicit env file
        if self._env_file and self._env_file.exists():
            load_dotenv(self._env_file)
            self._loaded_env = True
            return

        # Look for .env in config directory
        env_path = config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            self._loaded_env = True
            return

        # Look for .env in parent directory
        parent_env = config_dir.parent / ".env"
        if parent_env.exists():
            load_dotenv(parent_env)
            self._loaded_env = True
            return

        # Look for .env in current working directory
        cwd_env = Path.cwd() / ".env"
        if cwd_env.exists():
            load_dotenv(cwd_env)
            self._loaded_env = True


# Convenience function
def load_config(
    path: str | Path,
    env: Optional[str] = None,
    env_file: Optional[str | Path] = None,
) -> AppConfig:
    """
    Load configuration from YAML file.

    Convenience function that creates a ConfigLoader and loads configuration.

    Args:
        path: Path to base configuration file
        env: Optional environment name
        env_file: Optional path to .env file

    Returns:
        Validated AppConfig instance
    """
    loader = ConfigLoader(env_file=env_file)
    return loader.load(path, env=env)
