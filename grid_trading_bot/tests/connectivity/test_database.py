"""
Database Connectivity Tests.

Tests PostgreSQL database connectivity using SQLAlchemy async.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.data import DatabaseManager


# =============================================================================
# Mock-based Tests (Always Run)
# =============================================================================


class TestDatabaseConnectivityMock:
    """Mock-based database connectivity tests."""

    @pytest.fixture
    def mock_engine(self):
        """Create mock async engine."""
        engine = AsyncMock()
        engine.dispose = AsyncMock()

        # Mock connection context
        mock_conn = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalar.return_value = 1
        mock_conn.execute = AsyncMock(return_value=mock_result)

        engine.connect = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_conn),
            __aexit__=AsyncMock(return_value=None)
        ))

        return engine

    @pytest.mark.asyncio
    async def test_connect(self):
        """Test database connection."""
        db = DatabaseManager(
            host="localhost",
            database="test_db",
            user="postgres",
            password="test",
        )

        with patch("src.data.database.connection.create_async_engine") as mock_create:
            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            # Mock health check
            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            result = await db.connect()

            assert result is True
            assert db.is_connected is True

            await db.disconnect()

    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check."""
        db = DatabaseManager(
            host="localhost",
            database="test_db",
            user="postgres",
        )

        with patch("src.data.database.connection.create_async_engine") as mock_create:
            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            await db.connect()

            # Health check should return True
            health = await db.health_check()
            assert health is True

            await db.disconnect()

    @pytest.mark.asyncio
    async def test_execute_query(self):
        """Test executing a query."""
        db = DatabaseManager(
            host="localhost",
            database="test_db",
            user="postgres",
        )

        with patch("src.data.database.connection.create_async_engine") as mock_create, \
             patch("src.data.database.connection.async_sessionmaker") as mock_session_maker:

            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            # Mock health check
            mock_conn = AsyncMock()
            mock_health_result = MagicMock()
            mock_health_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_health_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            # Mock session
            mock_session = AsyncMock()
            mock_query_result = MagicMock()
            mock_query_result.scalar.return_value = 1
            mock_session.execute = AsyncMock(return_value=mock_query_result)
            mock_session.commit = AsyncMock()
            mock_session.close = AsyncMock()

            mock_factory = MagicMock(return_value=mock_session)
            mock_session_maker.return_value = mock_factory

            await db.connect()

            result = await db.fetch_scalar("SELECT 1 as test")
            assert result == 1

            await db.disconnect()

    @pytest.mark.asyncio
    async def test_transaction(self):
        """Test transaction context manager."""
        db = DatabaseManager(
            host="localhost",
            database="test_db",
            user="postgres",
        )

        with patch("src.data.database.connection.create_async_engine") as mock_create, \
             patch("src.data.database.connection.async_sessionmaker") as mock_session_maker:

            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            # Mock health check
            mock_conn = AsyncMock()
            mock_health_result = MagicMock()
            mock_health_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_health_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            # Mock session
            mock_session = AsyncMock()
            mock_session.execute = AsyncMock()
            mock_session.commit = AsyncMock()
            mock_session.rollback = AsyncMock()
            mock_session.close = AsyncMock()

            mock_factory = MagicMock(return_value=mock_session)
            mock_session_maker.return_value = mock_factory

            await db.connect()

            async with db.transaction() as session:
                await session.execute("SELECT 1")

            # Verify commit was called
            mock_session.commit.assert_called()

            await db.disconnect()

    @pytest.mark.asyncio
    async def test_disconnect(self):
        """Test database disconnection."""
        db = DatabaseManager(
            host="localhost",
            database="test_db",
            user="postgres",
        )

        with patch("src.data.database.connection.create_async_engine") as mock_create:
            mock_engine = AsyncMock()
            mock_create.return_value = mock_engine

            mock_conn = AsyncMock()
            mock_result = MagicMock()
            mock_result.scalar.return_value = 1
            mock_conn.execute = AsyncMock(return_value=mock_result)

            mock_engine.connect = MagicMock(return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_conn),
                __aexit__=AsyncMock(return_value=None)
            ))

            await db.connect()
            await db.disconnect()

            assert db.is_connected is False
            mock_engine.dispose.assert_called_once()


# =============================================================================
# Live Tests (Skip if no database)
# =============================================================================


@pytest.mark.skipif(
    not os.getenv("POSTGRES_HOST"),
    reason="No PostgreSQL connection configured"
)
class TestDatabaseConnectivityLive:
    """Live database connectivity tests."""

    @pytest.fixture
    def db_config(self):
        """Get database config from environment."""
        return {
            "host": os.getenv("POSTGRES_HOST", "localhost"),
            "port": int(os.getenv("POSTGRES_PORT", "5432")),
            "database": os.getenv("POSTGRES_DB", "trading_bot"),
            "user": os.getenv("POSTGRES_USER", "postgres"),
            "password": os.getenv("POSTGRES_PASSWORD", ""),
        }

    @pytest.mark.asyncio
    async def test_connect(self, db_config):
        """Test live database connection."""
        async with DatabaseManager(**db_config) as db:
            assert db.is_connected is True

    @pytest.mark.asyncio
    async def test_execute_query(self, db_config):
        """Test executing a query."""
        async with DatabaseManager(**db_config) as db:
            result = await db.fetch_scalar("SELECT 1 as test")
            assert result == 1

    @pytest.mark.asyncio
    async def test_health_check(self, db_config):
        """Test health check."""
        async with DatabaseManager(**db_config) as db:
            health = await db.health_check()
            assert health is True

    @pytest.mark.asyncio
    async def test_insert_select(self, db_config):
        """Test insert and select."""
        async with DatabaseManager(**db_config) as db:
            # Create test table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS test_conn (
                    id SERIAL PRIMARY KEY,
                    value TEXT
                )
            """)

            try:
                # Insert
                await db.execute(
                    "INSERT INTO test_conn (value) VALUES (:value)",
                    {"value": "test_value"}
                )

                # Select
                result = await db.fetch_one(
                    "SELECT value FROM test_conn WHERE value = :value",
                    {"value": "test_value"}
                )
                assert result is not None

            finally:
                # Cleanup
                await db.execute("DROP TABLE IF EXISTS test_conn")
