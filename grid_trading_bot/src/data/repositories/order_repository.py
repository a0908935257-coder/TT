"""
Order Repository.

Provides data access for order records.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import and_, select

from src.core import get_logger
from src.core.models import MarketType, Order, OrderStatus
from src.data.database import DatabaseManager, OrderModel

from .base import BaseRepository

logger = get_logger(__name__)


class OrderRepository(BaseRepository[Order, OrderModel]):
    """
    Repository for Order data access.

    Provides CRUD operations and queries for order records.

    Example:
        >>> repo = OrderRepository(db)
        >>> order = await repo.create(order, bot_id="bot_1")
        >>> open_orders = await repo.get_open_orders("bot_1", "BTCUSDT")
    """

    def __init__(
        self,
        db: DatabaseManager,
        market_type: MarketType = MarketType.SPOT,
    ):
        """
        Initialize OrderRepository.

        Args:
            db: DatabaseManager instance
            market_type: Default market type for orders
        """
        super().__init__(db)
        self._market_type = market_type

    def _to_domain(self, orm_model: OrderModel) -> Order:
        """Convert ORM model to domain model."""
        return orm_model.to_domain()

    def _to_orm(self, domain_model: Order, **kwargs) -> OrderModel:
        """Convert domain model to ORM model."""
        bot_id = kwargs.get("bot_id")
        market_type = kwargs.get("market_type", self._market_type)
        return OrderModel.from_domain(domain_model, bot_id, market_type)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        order: Order,
        bot_id: Optional[str] = None,
        market_type: Optional[MarketType] = None,
    ) -> Order:
        """
        Create new order record.

        Args:
            order: Order domain model
            bot_id: Bot identifier
            market_type: Market type (defaults to repository default)

        Returns:
            Created Order with generated fields
        """
        orm_model = self._to_orm(
            order,
            bot_id=bot_id,
            market_type=market_type or self._market_type,
        )
        saved = await self._add(orm_model)
        logger.debug(f"Created order: {order.order_id}")
        return self._to_domain(saved)

    async def get_by_order_id(self, order_id: str) -> Optional[Order]:
        """
        Get order by order ID.

        Args:
            order_id: Exchange order ID

        Returns:
            Order or None if not found
        """
        query = select(OrderModel).where(OrderModel.order_id == order_id)
        orm_model = await self._fetch_one(query)
        return self._to_domain(orm_model) if orm_model else None

    async def update(self, order: Order, bot_id: Optional[str] = None) -> Order:
        """
        Update existing order.

        Args:
            order: Order with updated fields
            bot_id: Bot identifier

        Returns:
            Updated Order
        """
        # Get existing record
        query = select(OrderModel).where(OrderModel.order_id == order.order_id)
        existing = await self._fetch_one(query)

        if not existing:
            raise ValueError(f"Order not found: {order.order_id}")

        # Update fields
        existing.status = order.status.value if hasattr(order.status, "value") else order.status
        existing.filled_quantity = order.filled_qty
        existing.average_price = order.avg_price
        existing.fee = order.fee
        existing.fee_asset = order.fee_asset
        existing.updated_at = order.updated_at

        updated = await self._update(existing)
        logger.debug(f"Updated order: {order.order_id}")
        return self._to_domain(updated)

    async def update_status(
        self,
        order_id: str,
        status: OrderStatus,
    ) -> Optional[Order]:
        """
        Update order status.

        Args:
            order_id: Order ID
            status: New status

        Returns:
            Updated Order or None if not found
        """
        query = select(OrderModel).where(OrderModel.order_id == order_id)
        existing = await self._fetch_one(query)

        if not existing:
            return None

        existing.status = status.value if hasattr(status, "value") else status

        updated = await self._update(existing)
        logger.debug(f"Updated order status: {order_id} -> {status}")
        return self._to_domain(updated)

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_open_orders(
        self,
        bot_id: str,
        symbol: Optional[str] = None,
    ) -> list[Order]:
        """
        Get open (active) orders for a bot.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair filter (optional)

        Returns:
            List of open orders
        """
        conditions = [
            OrderModel.bot_id == bot_id,
            OrderModel.status.in_([OrderStatus.NEW.value, OrderStatus.PARTIALLY_FILLED.value]),
        ]

        if symbol:
            conditions.append(OrderModel.symbol == symbol.upper())

        query = (
            select(OrderModel)
            .where(and_(*conditions))
            .order_by(OrderModel.created_at.desc())
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_orders_by_bot(
        self,
        bot_id: str,
        limit: int = 100,
        symbol: Optional[str] = None,
    ) -> list[Order]:
        """
        Get orders for a specific bot.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of orders
            symbol: Trading pair filter (optional)

        Returns:
            List of orders (most recent first)
        """
        conditions = [OrderModel.bot_id == bot_id]

        if symbol:
            conditions.append(OrderModel.symbol == symbol.upper())

        query = (
            select(OrderModel)
            .where(and_(*conditions))
            .order_by(OrderModel.created_at.desc())
            .limit(limit)
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_orders_in_range(
        self,
        start: datetime,
        end: datetime,
        bot_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[Order]:
        """
        Get orders within a time range.

        Args:
            start: Start time (inclusive)
            end: End time (inclusive)
            bot_id: Bot identifier filter (optional)
            symbol: Trading pair filter (optional)

        Returns:
            List of orders in range
        """
        conditions = [
            OrderModel.created_at >= start,
            OrderModel.created_at <= end,
        ]

        if bot_id:
            conditions.append(OrderModel.bot_id == bot_id)
        if symbol:
            conditions.append(OrderModel.symbol == symbol.upper())

        query = (
            select(OrderModel)
            .where(and_(*conditions))
            .order_by(OrderModel.created_at.asc())
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_filled_orders(
        self,
        bot_id: str,
        limit: int = 100,
    ) -> list[Order]:
        """
        Get filled orders for a bot.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of orders

        Returns:
            List of filled orders
        """
        query = (
            select(OrderModel)
            .where(
                and_(
                    OrderModel.bot_id == bot_id,
                    OrderModel.status == OrderStatus.FILLED.value,
                )
            )
            .order_by(OrderModel.created_at.desc())
            .limit(limit)
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def count_orders(
        self,
        bot_id: str,
        status: Optional[OrderStatus] = None,
    ) -> int:
        """
        Count orders for a bot.

        Args:
            bot_id: Bot identifier
            status: Status filter (optional)

        Returns:
            Number of orders
        """
        from sqlalchemy import func

        conditions = [OrderModel.bot_id == bot_id]

        if status:
            conditions.append(OrderModel.status == status.value)

        query = select(func.count()).select_from(OrderModel).where(and_(*conditions))

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or 0
