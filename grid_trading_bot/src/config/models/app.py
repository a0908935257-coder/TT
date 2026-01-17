"""
Application Configuration Model.

Provides the main application configuration that integrates all sub-configurations.
"""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import Field, field_validator

from .base import BaseConfig
from .database import DatabaseConfig, RedisConfig
from .exchange import ExchangeConfig
from .notification import NotificationConfig
from .risk import RiskConfig


class AppConfig(BaseConfig):
    """
    Main application configuration.

    Integrates all sub-configurations into a single configuration object.

    Example:
        >>> config = AppConfig(
        ...     exchange=ExchangeConfig(api_key="${API_KEY}"),
        ...     database=DatabaseConfig(password="${DB_PASSWORD}"),
        ...     redis=RedisConfig(),
        ...     notification=NotificationConfig(),
        ...     risk=RiskConfig(),
        ... )

        >>> # Load from file
        >>> config = AppConfig.from_yaml("config.yaml")
    """

    # Application metadata
    app_name: str = Field(
        default="Grid Trading Bot",
        description="Application name",
    )
    environment: str = Field(
        default="development",
        description="Environment (development, staging, production)",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    # Sub-configurations
    exchange: ExchangeConfig = Field(
        default_factory=ExchangeConfig,
        description="Exchange configuration",
    )
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig,
        description="Database configuration",
    )
    redis: RedisConfig = Field(
        default_factory=RedisConfig,
        description="Redis configuration",
    )
    notification: NotificationConfig = Field(
        default_factory=NotificationConfig,
        description="Notification configuration",
    )
    risk: RiskConfig = Field(
        default_factory=RiskConfig,
        description="Risk management configuration",
    )

    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate and normalize environment."""
        v = v.lower().strip()
        valid_envs = {"development", "staging", "production", "dev", "prod", "test"}
        if v not in valid_envs:
            v = "development"
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate and normalize log level."""
        v = v.upper().strip()
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v not in valid_levels:
            v = "INFO"
        return v

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment in ("production", "prod")

    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.environment in ("development", "dev")

    @property
    def is_testnet(self) -> bool:
        """Check if using testnet."""
        return self.exchange.is_testnet

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            AppConfig instance

        Example:
            >>> config = AppConfig.from_yaml("config.yaml")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls(**data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            AppConfig instance
        """
        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Note: Sensitive fields will be masked.

        Args:
            path: Path to save YAML file
        """
        path = Path(path)

        # Use masked dict to avoid writing secrets
        data = self.masked_dict()

        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

    def get_db_config_dict(self) -> dict:
        """Get database config as dict for DatabaseManager."""
        return self.database.to_connection_dict()

    def get_redis_config_dict(self) -> dict:
        """Get redis config as dict for RedisManager."""
        return self.redis.to_connection_dict()

    def validate_for_trading(self) -> list[str]:
        """
        Validate configuration for live trading.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check exchange credentials
        if not self.exchange.has_credentials:
            errors.append("Exchange API credentials not configured")

        # Check database
        if not self.database.host:
            errors.append("Database host not configured")

        # Check risk parameters
        if self.risk.max_leverage > 20 and self.is_production:
            errors.append("High leverage (>20x) not recommended for production")

        if self.risk.max_drawdown > 50:
            errors.append("Max drawdown >50% is very risky")

        # Check notification
        if self.is_production and not self.notification.is_enabled:
            errors.append("Notifications disabled in production - recommended to enable")

        return errors
