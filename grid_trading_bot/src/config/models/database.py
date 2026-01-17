"""
Database Configuration Models.

Provides configuration for PostgreSQL and Redis connections.
"""

from typing import Optional

from pydantic import Field, computed_field

from .base import BaseConfig


class DatabaseConfig(BaseConfig):
    """
    PostgreSQL database configuration.

    Example:
        >>> config = DatabaseConfig(
        ...     host="localhost",
        ...     password="${DB_PASSWORD}",
        ... )
        >>> print(config.url)
        postgresql+asyncpg://postgres:***@localhost:5432/trading_bot
    """

    host: str = Field(
        default="localhost",
        description="Database host",
    )
    port: int = Field(
        default=5432,
        ge=1,
        le=65535,
        description="Database port",
    )
    name: str = Field(
        default="trading_bot",
        description="Database name",
    )
    user: str = Field(
        default="postgres",
        description="Database user",
    )
    password: str = Field(
        default="",
        description="Database password",
    )
    pool_size: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Connection pool size",
    )
    max_overflow: int = Field(
        default=20,
        ge=0,
        le=100,
        description="Max overflow connections",
    )
    pool_timeout: int = Field(
        default=30,
        ge=1,
        description="Pool timeout in seconds",
    )
    echo: bool = Field(
        default=False,
        description="Echo SQL queries (debug)",
    )

    @computed_field
    @property
    def url(self) -> str:
        """
        Build async database connection URL.

        Returns:
            PostgreSQL async connection string
        """
        password_part = f":{self.password}" if self.password else ""
        return (
            f"postgresql+asyncpg://{self.user}{password_part}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @computed_field
    @property
    def sync_url(self) -> str:
        """
        Build sync database connection URL.

        Returns:
            PostgreSQL sync connection string
        """
        password_part = f":{self.password}" if self.password else ""
        return (
            f"postgresql://{self.user}{password_part}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    def to_connection_dict(self) -> dict:
        """
        Convert to dictionary for DatabaseManager.

        Returns:
            Connection parameters dict
        """
        return {
            "host": self.host,
            "port": self.port,
            "database": self.name,
            "user": self.user,
            "password": self.password,
            "pool_size": self.pool_size,
            "max_overflow": self.max_overflow,
            "pool_timeout": self.pool_timeout,
            "echo": self.echo,
        }


class RedisConfig(BaseConfig):
    """
    Redis configuration.

    Example:
        >>> config = RedisConfig(
        ...     host="localhost",
        ...     password="${REDIS_PASSWORD:}",
        ... )
        >>> print(config.url)
        redis://localhost:6379/0
    """

    host: str = Field(
        default="localhost",
        description="Redis host",
    )
    port: int = Field(
        default=6379,
        ge=1,
        le=65535,
        description="Redis port",
    )
    db: int = Field(
        default=0,
        ge=0,
        le=15,
        description="Redis database number",
    )
    password: Optional[str] = Field(
        default=None,
        description="Redis password",
    )
    key_prefix: str = Field(
        default="trading:",
        description="Key prefix for all keys",
    )
    socket_timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Socket timeout in seconds",
    )
    socket_connect_timeout: float = Field(
        default=5.0,
        ge=0.1,
        description="Connection timeout in seconds",
    )
    max_connections: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max connections in pool",
    )

    @computed_field
    @property
    def url(self) -> str:
        """
        Build Redis connection URL.

        Returns:
            Redis connection string
        """
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    def to_connection_dict(self) -> dict:
        """
        Convert to dictionary for RedisManager.

        Returns:
            Connection parameters dict
        """
        return {
            "host": self.host,
            "port": self.port,
            "db": self.db,
            "password": self.password,
            "key_prefix": self.key_prefix,
            "socket_timeout": self.socket_timeout,
            "socket_connect_timeout": self.socket_connect_timeout,
            "max_connections": self.max_connections,
        }
