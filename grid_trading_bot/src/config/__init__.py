# Config module - Application configuration system
from .exceptions import (
    ConfigError,
    ConfigFileNotFoundError,
    ConfigNotLoadedError,
    ConfigParseError,
    ConfigReloadError,
    ConfigValidationError,
)
from .loader import ConfigLoader, load_config
from .manager import (
    ConfigManager,
    get_config,
    get_config_manager,
    init_config,
)
from .models import (
    AppConfig,
    BaseConfig,
    DatabaseConfig,
    ExchangeConfig,
    NotificationConfig,
    RedisConfig,
    RiskConfig,
)
from .strategy_loader import (
    get_all_bot_configs,
    get_bot_allocation,
    load_strategy_config,
    validate_config_consistency,
)

__all__ = [
    # Exceptions
    "ConfigError",
    "ConfigFileNotFoundError",
    "ConfigParseError",
    "ConfigValidationError",
    "ConfigNotLoadedError",
    "ConfigReloadError",
    # Loader
    "ConfigLoader",
    "load_config",
    # Manager
    "ConfigManager",
    "init_config",
    "get_config",
    "get_config_manager",
    # Strategy Loader (單一來源配置)
    "load_strategy_config",
    "get_all_bot_configs",
    "get_bot_allocation",
    "validate_config_consistency",
    # Models - Base
    "BaseConfig",
    # Models - Individual configs
    "ExchangeConfig",
    "DatabaseConfig",
    "RedisConfig",
    "NotificationConfig",
    "RiskConfig",
    # Models - Main config
    "AppConfig",
]
