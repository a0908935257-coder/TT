"""
Base Configuration Model.

Provides base configuration class with environment variable substitution
and sensitive field masking capabilities.
"""

import os
import re
from typing import Any, ClassVar, Optional, Set

from pydantic import BaseModel, ConfigDict, model_validator


# Pattern for environment variable substitution: ${VAR} or ${VAR:default}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}:]+)(?::([^}]*))?\}")


def substitute_env_vars(value: str) -> str:
    """
    Substitute environment variables in a string.

    Supports formats:
    - ${VAR} - substitutes with VAR value, empty if not set
    - ${VAR:default} - substitutes with VAR value, or 'default' if not set

    Args:
        value: String potentially containing env var references

    Returns:
        String with env vars substituted
    """

    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        default_value = match.group(2)  # May be None

        env_value = os.environ.get(var_name)

        if env_value is not None:
            return env_value
        elif default_value is not None:
            return default_value
        else:
            return ""

    return ENV_VAR_PATTERN.sub(replace_match, value)


def process_value(value: Any) -> Any:
    """
    Process a value, substituting env vars if it's a string.

    Args:
        value: Value to process

    Returns:
        Processed value
    """
    if isinstance(value, str):
        return substitute_env_vars(value)
    elif isinstance(value, dict):
        return {k: process_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [process_value(item) for item in value]
    return value


class BaseConfig(BaseModel):
    """
    Base configuration model with common functionality.

    Features:
    - Environment variable substitution: ${VAR} or ${VAR:default}
    - Sensitive field masking for display
    - Immutable by default (frozen)

    Example:
        >>> class MyConfig(BaseConfig):
        ...     api_key: str
        ...     host: str = "localhost"
        ...
        >>> config = MyConfig(api_key="${API_KEY:default_key}")
    """

    model_config = ConfigDict(
        frozen=True,  # Immutable
        extra="ignore",  # Ignore extra fields (e.g., fund_manager loaded separately)
        validate_default=True,
        str_strip_whitespace=True,
    )

    # Fields to mask when displaying (subclasses should override)
    _sensitive_fields: ClassVar[Set[str]] = {
        "api_key",
        "api_secret",
        "password",
        "secret",
        "token",
        "webhook_url",
    }

    @model_validator(mode="before")
    @classmethod
    def substitute_environment_variables(cls, data: Any) -> Any:
        """Substitute environment variables in all string values."""
        if isinstance(data, dict):
            return process_value(data)
        return data

    def masked_dict(self) -> dict[str, Any]:
        """
        Get dictionary with sensitive fields masked.

        Returns:
            Dict with sensitive values replaced by '***'
        """
        data = self.model_dump()
        return self._mask_sensitive(data)

    def _mask_sensitive(self, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively mask sensitive fields in a dictionary.

        Args:
            data: Dictionary to mask

        Returns:
            Masked dictionary
        """
        result = {}
        for key, value in data.items():
            if self._is_sensitive_field(key) and value:
                result[key] = "***"
            elif isinstance(value, dict):
                result[key] = self._mask_sensitive(value)
            elif isinstance(value, list):
                result[key] = [
                    self._mask_sensitive(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def _is_sensitive_field(self, field_name: str) -> bool:
        """
        Check if a field name indicates sensitive data.

        Args:
            field_name: Name of the field

        Returns:
            True if field is sensitive
        """
        field_lower = field_name.lower()
        return any(
            sensitive in field_lower for sensitive in self._sensitive_fields
        )

    def __repr__(self) -> str:
        """Return repr with masked sensitive fields."""
        masked = self.masked_dict()
        fields = ", ".join(f"{k}={v!r}" for k, v in masked.items())
        return f"{self.__class__.__name__}({fields})"

    def __str__(self) -> str:
        """Return string representation with masked sensitive fields."""
        return self.__repr__()
