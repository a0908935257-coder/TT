"""
PostgreSQL Database Connection Manager.

Provides async connection pool management using SQLAlchemy 2.0 and asyncpg.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core import get_logger

logger = get_logger(__name__)


class DatabaseManager:
    """
    PostgreSQL database connection pool manager.

    Uses SQLAlchemy 2.0 async engine with asyncpg driver for high-performance
    async database operations.

    Example:
        >>> db = DatabaseManager(
        ...     host="localhost",
        ...     database="trading_bot",
        ...     user="postgres",
        ...     password="secret",
        ... )
        >>> await db.connect()
        >>>
        >>> async with db.get_session() as session:
        ...     result = await session.execute(text("SELECT 1"))
        ...     print(result.scalar())
        >>>
        >>> await db.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 5432,
        database: str = "trading_bot",
        user: str = "postgres",
        password: str = "",
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        echo: bool = False,
    ):
        """
        Initialize DatabaseManager.

        Args:
            host: Database host
            port: Database port
            database: Database name
            user: Database user
            password: Database password
            pool_size: Connection pool size
            max_overflow: Maximum overflow connections
            pool_timeout: Pool connection timeout in seconds
            echo: Echo SQL queries (for debugging)
        """
        self._host = host
        self._port = port
        self._database = database
        self._user = user
        self._password = password
        self._pool_size = pool_size
        self._max_overflow = max_overflow
        self._pool_timeout = pool_timeout
        self._echo = echo

        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._connected = False

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self._connected and self._engine is not None

    @property
    def engine(self) -> Optional[AsyncEngine]:
        """Get the SQLAlchemy async engine."""
        return self._engine

    @property
    def connection_url(self) -> str:
        """Build the database connection URL."""
        # Use asyncpg driver for async PostgreSQL
        password_part = f":{self._password}" if self._password else ""
        return (
            f"postgresql+asyncpg://{self._user}{password_part}"
            f"@{self._host}:{self._port}/{self._database}"
        )

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish database connection and create connection pool.

        Returns:
            True if connection successful
        """
        if self._connected:
            logger.debug("Already connected to database")
            return True

        try:
            logger.info(f"Connecting to PostgreSQL at {self._host}:{self._port}/{self._database}")

            # Create async engine with connection pool
            self._engine = create_async_engine(
                self.connection_url,
                pool_size=self._pool_size,
                max_overflow=self._max_overflow,
                pool_timeout=self._pool_timeout,
                pool_pre_ping=True,  # Enable connection health check
                echo=self._echo,
            )

            # Create session factory
            self._session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autoflush=False,
            )

            # Test connection
            if await self.health_check():
                self._connected = True
                logger.info("Database connection established")
                return True
            else:
                logger.error("Database health check failed")
                return False

        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close database connection and dispose connection pool."""
        if self._engine:
            logger.info("Disconnecting from database")
            await self._engine.dispose()
            self._engine = None
            self._session_factory = None
            self._connected = False
            logger.info("Database disconnected")

    async def __aenter__(self) -> "DatabaseManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    # =========================================================================
    # Session Management
    # =========================================================================

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session as async context manager.

        Automatically handles commit/rollback on success/failure.

        Yields:
            AsyncSession instance

        Example:
            >>> async with db.get_session() as session:
            ...     result = await session.execute(query)
        """
        if not self._session_factory:
            raise RuntimeError("Database not connected. Call connect() first.")

        session = self._session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a session with explicit transaction control.

        Same as get_session() but makes the transaction semantics explicit.

        Yields:
            AsyncSession instance within a transaction
        """
        async with self.get_session() as session:
            yield session

    # =========================================================================
    # Query Execution
    # =========================================================================

    async def execute(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Execute a raw SQL query.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Query result

        Example:
            >>> result = await db.execute(
            ...     "SELECT * FROM orders WHERE symbol = :symbol",
            ...     {"symbol": "BTCUSDT"}
            ... )
        """
        async with self.get_session() as session:
            result = await session.execute(text(query), params or {})
            return result

    async def execute_many(
        self,
        query: str,
        params_list: list[dict[str, Any]],
    ) -> None:
        """
        Execute a query multiple times with different parameters.

        Args:
            query: SQL query string
            params_list: List of parameter dictionaries
        """
        async with self.get_session() as session:
            for params in params_list:
                await session.execute(text(query), params)

    async def fetch_one(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        """
        Fetch a single row from query result.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Single row or None
        """
        result = await self.execute(query, params)
        return result.fetchone()

    async def fetch_all(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> list[Any]:
        """
        Fetch all rows from query result.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            List of rows
        """
        result = await self.execute(query, params)
        return result.fetchall()

    async def fetch_scalar(
        self,
        query: str,
        params: Optional[dict[str, Any]] = None,
    ) -> Any:
        """
        Fetch a scalar value from query result.

        Args:
            query: SQL query string
            params: Query parameters

        Returns:
            Scalar value
        """
        result = await self.execute(query, params)
        return result.scalar()

    # =========================================================================
    # Database Operations
    # =========================================================================

    async def health_check(self) -> bool:
        """
        Check database connection health.

        Returns:
            True if database is healthy
        """
        try:
            if not self._engine:
                return False

            async with self._engine.connect() as conn:
                result = await conn.execute(text("SELECT 1"))
                return result.scalar() == 1

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    async def create_tables(self, base) -> None:
        """
        Create all tables defined in SQLAlchemy Base.

        Args:
            base: SQLAlchemy declarative base with table definitions
        """
        if not self._engine:
            raise RuntimeError("Database not connected")

        async with self._engine.begin() as conn:
            await conn.run_sync(base.metadata.create_all)
            logger.info("Database tables created")

    async def drop_tables(self, base) -> None:
        """
        Drop all tables defined in SQLAlchemy Base.

        WARNING: This will delete all data!

        Args:
            base: SQLAlchemy declarative base with table definitions
        """
        if not self._engine:
            raise RuntimeError("Database not connected")

        async with self._engine.begin() as conn:
            await conn.run_sync(base.metadata.drop_all)
            logger.warning("Database tables dropped")

    async def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists
        """
        query = """
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public'
                AND table_name = :table_name
            )
        """
        result = await self.fetch_scalar(query, {"table_name": table_name})
        return bool(result)
