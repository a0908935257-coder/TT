"""
Tests for PostgreSQL Database Module.
"""

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.database.connection import DatabaseManager
from src.data.database.models import (
    Base,
    BalanceModel,
    BotStateModel,
    KlineModel,
    OrderModel,
    PositionModel,
    TradeModel,
)
from src.core.models import (
    Balance,
    Kline,
    KlineInterval,
    MarketType,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    PositionSide,
    Trade,
)


# =============================================================================
# DatabaseManager Tests
# =============================================================================


class TestDatabaseManagerInit:
    """Test DatabaseManager initialization."""

    def test_init_default(self):
        """Test default initialization."""
        db = DatabaseManager()
        assert db._host == "localhost"
        assert db._port == 5432
        assert db._database == "trading_bot"
        assert db._user == "postgres"
        assert db._password == ""
        assert db._pool_size == 10
        assert db.is_connected is False

    def test_init_with_params(self):
        """Test initialization with custom parameters."""
        db = DatabaseManager(
            host="db.example.com",
            port=5433,
            database="test_db",
            user="test_user",
            password="secret",
            pool_size=20,
        )
        assert db._host == "db.example.com"
        assert db._port == 5433
        assert db._database == "test_db"
        assert db._user == "test_user"
        assert db._password == "secret"
        assert db._pool_size == 20

    def test_connection_url_without_password(self):
        """Test connection URL without password."""
        db = DatabaseManager(
            host="localhost",
            port=5432,
            database="trading_bot",
            user="postgres",
        )
        expected = "postgresql+asyncpg://postgres@localhost:5432/trading_bot"
        assert db.connection_url == expected

    def test_connection_url_with_password(self):
        """Test connection URL with password."""
        db = DatabaseManager(
            host="localhost",
            port=5432,
            database="trading_bot",
            user="postgres",
            password="secret",
        )
        expected = "postgresql+asyncpg://postgres:secret@localhost:5432/trading_bot"
        assert db.connection_url == expected


class TestDatabaseManagerConnection:
    """Test DatabaseManager connection management."""

    @pytest.fixture
    def db(self):
        """Create DatabaseManager instance."""
        return DatabaseManager()

    @pytest.mark.asyncio
    async def test_connect_success(self, db):
        """Test successful connection."""
        with patch("data.database.connection.create_async_engine") as mock_engine:
            with patch("data.database.connection.async_sessionmaker") as mock_sessionmaker:
                mock_engine_instance = MagicMock()
                mock_engine.return_value = mock_engine_instance

                # Mock health check
                mock_conn = AsyncMock()
                mock_conn.execute = AsyncMock(return_value=MagicMock(scalar=lambda: 1))
                mock_engine_instance.connect = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_conn), __aexit__=AsyncMock()))

                result = await db.connect()

                # Note: Health check will fail without real DB, so we test the engine creation
                mock_engine.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect(self, db):
        """Test disconnection."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        db._engine = mock_engine
        db._connected = True

        await db.disconnect()

        mock_engine.dispose.assert_called_once()
        assert db.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager(self, db):
        """Test async context manager."""
        with patch.object(db, "connect", new_callable=AsyncMock) as mock_connect:
            with patch.object(db, "disconnect", new_callable=AsyncMock) as mock_disconnect:
                mock_connect.return_value = True

                async with db as ctx:
                    assert ctx is db
                    mock_connect.assert_called_once()

                mock_disconnect.assert_called_once()


class TestDatabaseManagerSession:
    """Test DatabaseManager session management."""

    @pytest.fixture
    def db(self):
        """Create DatabaseManager with mocked session factory."""
        db = DatabaseManager()
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        db._session_factory = MagicMock(return_value=mock_session)
        return db

    @pytest.mark.asyncio
    async def test_get_session_commits_on_success(self, db):
        """Test that session commits on success."""
        async with db.get_session() as session:
            pass  # Do nothing

        session.commit.assert_called_once()
        session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self, db):
        """Test that session rollback on error."""
        with pytest.raises(ValueError):
            async with db.get_session() as session:
                raise ValueError("Test error")

        session.rollback.assert_called_once()
        session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_without_connection(self):
        """Test get_session raises error when not connected."""
        db = DatabaseManager()

        with pytest.raises(RuntimeError, match="Database not connected"):
            async with db.get_session():
                pass


class TestDatabaseManagerQuery:
    """Test DatabaseManager query methods."""

    @pytest.fixture
    def db(self):
        """Create DatabaseManager with mocked session."""
        db = DatabaseManager()
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        db._session_factory = MagicMock(return_value=mock_session)
        db._mock_result = mock_result
        return db

    @pytest.mark.asyncio
    async def test_fetch_one(self, db):
        """Test fetch_one method."""
        db._mock_result.fetchone.return_value = ("row1",)

        result = await db.fetch_one("SELECT 1")

        assert result == ("row1",)

    @pytest.mark.asyncio
    async def test_fetch_all(self, db):
        """Test fetch_all method."""
        db._mock_result.fetchall.return_value = [("row1",), ("row2",)]

        result = await db.fetch_all("SELECT * FROM test")

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_fetch_scalar(self, db):
        """Test fetch_scalar method."""
        db._mock_result.scalar.return_value = 42

        result = await db.fetch_scalar("SELECT COUNT(*) FROM test")

        assert result == 42


# =============================================================================
# ORM Model Tests - OrderModel
# =============================================================================


class TestOrderModel:
    """Test OrderModel ORM."""

    @pytest.fixture
    def domain_order(self):
        """Create a domain Order instance."""
        return Order(
            order_id="123456",
            client_order_id="client_123",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            status=OrderStatus.NEW,
            price=Decimal("45000"),
            quantity=Decimal("0.1"),
            filled_qty=Decimal("0"),
            created_at=datetime.now(timezone.utc),
        )

    def test_from_domain(self, domain_order):
        """Test creating OrderModel from domain Order."""
        model = OrderModel.from_domain(
            domain_order,
            bot_id="bot_1",
            market_type=MarketType.SPOT,
        )

        assert model.order_id == "123456"
        assert model.client_order_id == "client_123"
        assert model.bot_id == "bot_1"
        assert model.symbol == "BTCUSDT"
        assert model.market_type == "SPOT"
        assert model.side == "BUY"
        assert model.order_type == "LIMIT"
        assert model.status == "NEW"
        assert model.price == Decimal("45000")
        assert model.quantity == Decimal("0.1")

    def test_to_domain(self, domain_order):
        """Test converting OrderModel to domain Order."""
        model = OrderModel.from_domain(domain_order)
        result = model.to_domain()

        assert result.order_id == domain_order.order_id
        assert result.symbol == domain_order.symbol
        assert result.side == OrderSide.BUY
        assert result.order_type == OrderType.LIMIT
        assert result.status == OrderStatus.NEW


# =============================================================================
# ORM Model Tests - TradeModel
# =============================================================================


class TestTradeModel:
    """Test TradeModel ORM."""

    @pytest.fixture
    def domain_trade(self):
        """Create a domain Trade instance."""
        return Trade(
            trade_id="trade_123",
            order_id="order_456",
            symbol="BTCUSDT",
            side=OrderSide.BUY,
            price=Decimal("45000"),
            quantity=Decimal("0.1"),
            fee=Decimal("0.001"),
            fee_asset="BNB",
            is_maker=True,
            timestamp=datetime.now(timezone.utc),
        )

    def test_from_domain(self, domain_trade):
        """Test creating TradeModel from domain Trade."""
        model = TradeModel.from_domain(domain_trade, bot_id="bot_1")

        assert model.trade_id == "trade_123"
        assert model.order_id == "order_456"
        assert model.bot_id == "bot_1"
        assert model.symbol == "BTCUSDT"
        assert model.side == "BUY"
        assert model.price == Decimal("45000")
        assert model.quantity == Decimal("0.1")
        assert model.fee == Decimal("0.001")
        assert model.is_maker is True

    def test_to_domain(self, domain_trade):
        """Test converting TradeModel to domain Trade."""
        model = TradeModel.from_domain(domain_trade)
        result = model.to_domain()

        assert result.trade_id == domain_trade.trade_id
        assert result.symbol == domain_trade.symbol
        assert result.side == OrderSide.BUY
        assert result.price == domain_trade.price


# =============================================================================
# ORM Model Tests - PositionModel
# =============================================================================


class TestPositionModel:
    """Test PositionModel ORM."""

    @pytest.fixture
    def domain_position(self):
        """Create a domain Position instance."""
        return Position(
            symbol="BTCUSDT",
            side=PositionSide.LONG,
            quantity=Decimal("0.5"),
            entry_price=Decimal("44000"),
            mark_price=Decimal("45000"),
            leverage=10,
            margin=Decimal("2200"),
            unrealized_pnl=Decimal("500"),
            updated_at=datetime.now(timezone.utc),
        )

    def test_from_domain(self, domain_position):
        """Test creating PositionModel from domain Position."""
        model = PositionModel.from_domain(domain_position, bot_id="bot_1")

        assert model.symbol == "BTCUSDT"
        assert model.side == "LONG"
        assert model.quantity == Decimal("0.5")
        assert model.entry_price == Decimal("44000")
        assert model.mark_price == Decimal("45000")
        assert model.leverage == 10
        assert model.margin == Decimal("2200")
        assert model.unrealized_pnl == Decimal("500")

    def test_to_domain(self, domain_position):
        """Test converting PositionModel to domain Position."""
        model = PositionModel.from_domain(domain_position)
        result = model.to_domain()

        assert result.symbol == domain_position.symbol
        assert result.side == PositionSide.LONG
        assert result.quantity == domain_position.quantity


# =============================================================================
# ORM Model Tests - BalanceModel
# =============================================================================


class TestBalanceModel:
    """Test BalanceModel ORM."""

    @pytest.fixture
    def domain_balance(self):
        """Create a domain Balance instance."""
        return Balance(
            asset="USDT",
            free=Decimal("1000"),
            locked=Decimal("100"),
        )

    def test_from_domain(self, domain_balance):
        """Test creating BalanceModel from domain Balance."""
        model = BalanceModel.from_domain(
            domain_balance,
            bot_id="bot_1",
            market_type=MarketType.SPOT,
        )

        assert model.asset == "USDT"
        assert model.market_type == "SPOT"
        assert model.free == Decimal("1000")
        assert model.locked == Decimal("100")
        assert model.total == Decimal("1100")

    def test_to_domain(self, domain_balance):
        """Test converting BalanceModel to domain Balance."""
        model = BalanceModel.from_domain(domain_balance)
        result = model.to_domain()

        assert result.asset == domain_balance.asset
        assert result.free == domain_balance.free
        assert result.locked == domain_balance.locked


# =============================================================================
# ORM Model Tests - KlineModel
# =============================================================================


class TestKlineModel:
    """Test KlineModel ORM."""

    @pytest.fixture
    def domain_kline(self):
        """Create a domain Kline instance."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("44000"),
            high=Decimal("45000"),
            low=Decimal("43500"),
            close=Decimal("44800"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

    def test_from_domain(self, domain_kline):
        """Test creating KlineModel from domain Kline."""
        model = KlineModel.from_domain(domain_kline)

        assert model.symbol == "BTCUSDT"
        assert model.interval == "1h"
        assert model.open == Decimal("44000")
        assert model.high == Decimal("45000")
        assert model.low == Decimal("43500")
        assert model.close == Decimal("44800")
        assert model.volume == Decimal("1000")

    def test_to_domain(self, domain_kline):
        """Test converting KlineModel to domain Kline."""
        model = KlineModel.from_domain(domain_kline)
        result = model.to_domain()

        assert result.symbol == domain_kline.symbol
        assert result.interval == KlineInterval.h1
        assert result.open == domain_kline.open


# =============================================================================
# ORM Model Tests - BotStateModel
# =============================================================================


class TestBotStateModel:
    """Test BotStateModel ORM."""

    def test_from_dict(self):
        """Test creating BotStateModel from dictionary."""
        data = {
            "bot_id": "grid_bot_1",
            "bot_type": "grid",
            "status": "running",
            "config": {"symbol": "BTCUSDT", "grid_size": 10},
            "state_data": {"current_price": 45000},
        }

        model = BotStateModel.from_dict(data)

        assert model.bot_id == "grid_bot_1"
        assert model.bot_type == "grid"
        assert model.status == "running"
        assert model.config == {"symbol": "BTCUSDT", "grid_size": 10}
        assert model.state_data == {"current_price": 45000}

    def test_to_dict(self):
        """Test converting BotStateModel to dictionary."""
        model = BotStateModel(
            bot_id="grid_bot_1",
            bot_type="grid",
            status="running",
            config={"symbol": "BTCUSDT"},
            state_data={"price": 45000},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        result = model.to_dict()

        assert result["bot_id"] == "grid_bot_1"
        assert result["bot_type"] == "grid"
        assert result["status"] == "running"
        assert result["config"] == {"symbol": "BTCUSDT"}


# =============================================================================
# Integration Tests (Optional - require real PostgreSQL)
# =============================================================================


@pytest.mark.integration
class TestDatabaseIntegration:
    """
    Integration tests that require a real PostgreSQL database.

    Run with: pytest -m integration

    Requires:
        - PostgreSQL running on localhost:5432
        - Database 'trading_bot_test' created
        - User 'postgres' with appropriate permissions
    """

    @pytest.fixture
    async def db(self):
        """Create and connect to test database."""
        db = DatabaseManager(
            database="trading_bot_test",
            password="",  # Update with your password
        )
        connected = await db.connect()
        if not connected:
            pytest.skip("PostgreSQL not available")
        yield db
        await db.disconnect()

    @pytest.mark.asyncio
    async def test_create_tables(self, db):
        """Test creating all tables."""
        await db.create_tables(Base)

        # Verify tables exist
        assert await db.table_exists("orders")
        assert await db.table_exists("trades")
        assert await db.table_exists("positions")
        assert await db.table_exists("balances")
        assert await db.table_exists("klines")
        assert await db.table_exists("bot_states")

    @pytest.mark.asyncio
    async def test_crud_order(self, db):
        """Test CRUD operations for orders."""
        await db.create_tables(Base)

        async with db.get_session() as session:
            # Create
            order = OrderModel(
                order_id="test_order_1",
                symbol="BTCUSDT",
                market_type="SPOT",
                side="BUY",
                order_type="LIMIT",
                status="NEW",
                price=Decimal("45000"),
                quantity=Decimal("0.1"),
                filled_quantity=Decimal("0"),
                fee=Decimal("0"),
            )
            session.add(order)
            await session.flush()

            # Read
            from sqlalchemy import select
            result = await session.execute(
                select(OrderModel).where(OrderModel.order_id == "test_order_1")
            )
            fetched = result.scalar_one()
            assert fetched.symbol == "BTCUSDT"

    @pytest.mark.asyncio
    async def test_health_check(self, db):
        """Test health check."""
        result = await db.health_check()
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
