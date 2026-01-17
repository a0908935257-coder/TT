"""
Configuration Exceptions.

Provides custom exceptions for configuration loading and management.
"""


class ConfigError(Exception):
    """Base exception for configuration errors."""

    pass


class ConfigFileNotFoundError(ConfigError):
    """Raised when a configuration file is not found."""

    def __init__(self, path: str):
        self.path = path
        super().__init__(f"Configuration file not found: {path}")


class ConfigParseError(ConfigError):
    """Raised when a configuration file cannot be parsed."""

    def __init__(self, path: str, reason: str):
        self.path = path
        self.reason = reason
        super().__init__(f"Failed to parse configuration file '{path}': {reason}")


class ConfigValidationError(ConfigError):
    """Raised when configuration validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


class ConfigNotLoadedError(ConfigError):
    """Raised when trying to access config before initialization."""

    def __init__(self):
        super().__init__(
            "Configuration not loaded. Call init_config() first."
        )


class ConfigReloadError(ConfigError):
    """Raised when configuration reload fails."""

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Failed to reload configuration: {reason}")
