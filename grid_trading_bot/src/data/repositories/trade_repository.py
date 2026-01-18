"""
Trade Repository.

Provides data access for trade (fill) records.
"""

from datetime import datetime
from decimal import Decimal
from typing import Optional

from sqlalchemy import and_, func, select

from src.core import get_logger
from src.core.models import Trade
from src.data.database import DatabaseManager, TradeModel

from .base import BaseRepository

logger = get_logger(__name__)


class TradeRepository(BaseRepository[Trade, TradeModel]):
    """
    Repository for Trade data access.

    Provides CRUD operations and statistical queries for trade records.

    Example:
        >>> repo = TradeRepository(db)
        >>> trade = await repo.create(trade, bot_id="bot_1")
        >>> total_fees = await repo.get_total_fees("bot_1", start, end)
    """

    def _to_domain(self, orm_model: TradeModel) -> Trade:
        """Convert ORM model to domain model."""
        return orm_model.to_domain()

    def _to_orm(self, domain_model: Trade, **kwargs) -> TradeModel:
        """Convert domain model to ORM model."""
        bot_id = kwargs.get("bot_id")
        return TradeModel.from_domain(domain_model, bot_id)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def create(
        self,
        trade: Trade,
        bot_id: Optional[str] = None,
    ) -> Trade:
        """
        Create new trade record.

        Args:
            trade: Trade domain model
            bot_id: Bot identifier

        Returns:
            Created Trade
        """
        orm_model = self._to_orm(trade, bot_id=bot_id)
        saved = await self._add(orm_model)
        logger.debug(f"Created trade: {trade.trade_id}")
        return self._to_domain(saved)

    async def create_batch(
        self,
        trades: list[Trade],
        bot_id: Optional[str] = None,
    ) -> list[Trade]:
        """
        Create multiple trade records.

        Args:
            trades: List of Trade domain models
            bot_id: Bot identifier

        Returns:
            List of created Trades
        """
        orm_models = [self._to_orm(t, bot_id=bot_id) for t in trades]
        saved = await self._add_all(orm_models)
        logger.debug(f"Created {len(saved)} trades")
        return [self._to_domain(m) for m in saved]

    async def get_by_trade_id(self, trade_id: str) -> Optional[Trade]:
        """
        Get trade by trade ID.

        Args:
            trade_id: Exchange trade ID

        Returns:
            Trade or None if not found
        """
        query = select(TradeModel).where(TradeModel.trade_id == trade_id)
        orm_model = await self._fetch_one(query)
        return self._to_domain(orm_model) if orm_model else None

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_trades_by_order(self, order_id: str) -> list[Trade]:
        """
        Get all trades for an order.

        Args:
            order_id: Order ID

        Returns:
            List of trades for the order
        """
        query = (
            select(TradeModel)
            .where(TradeModel.order_id == order_id)
            .order_by(TradeModel.timestamp.asc())
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_trades_by_bot(
        self,
        bot_id: str,
        limit: int = 100,
        symbol: Optional[str] = None,
    ) -> list[Trade]:
        """
        Get trades for a specific bot.

        Args:
            bot_id: Bot identifier
            limit: Maximum number of trades
            symbol: Trading pair filter (optional)

        Returns:
            List of trades (most recent first)
        """
        conditions = [TradeModel.bot_id == bot_id]

        if symbol:
            conditions.append(TradeModel.symbol == symbol.upper())

        query = (
            select(TradeModel)
            .where(and_(*conditions))
            .order_by(TradeModel.timestamp.desc())
            .limit(limit)
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_trades_in_range(
        self,
        start: datetime,
        end: datetime,
        bot_id: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[Trade]:
        """
        Get trades within a time range.

        Args:
            start: Start time (inclusive)
            end: End time (inclusive)
            bot_id: Bot identifier filter (optional)
            symbol: Trading pair filter (optional)

        Returns:
            List of trades in range
        """
        conditions = [
            TradeModel.timestamp >= start,
            TradeModel.timestamp <= end,
        ]

        if bot_id:
            conditions.append(TradeModel.bot_id == bot_id)
        if symbol:
            conditions.append(TradeModel.symbol == symbol.upper())

        query = (
            select(TradeModel)
            .where(and_(*conditions))
            .order_by(TradeModel.timestamp.asc())
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    # =========================================================================
    # Statistical Operations
    # =========================================================================

    async def get_total_fees(
        self,
        bot_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        fee_asset: Optional[str] = None,
    ) -> Decimal:
        """
        Get total fees for a bot.

        Args:
            bot_id: Bot identifier
            start: Start time filter (optional)
            end: End time filter (optional)
            fee_asset: Fee asset filter (optional)

        Returns:
            Total fees
        """
        conditions = [TradeModel.bot_id == bot_id]

        if start:
            conditions.append(TradeModel.timestamp >= start)
        if end:
            conditions.append(TradeModel.timestamp <= end)
        if fee_asset:
            conditions.append(TradeModel.fee_asset == fee_asset.upper())

        query = (
            select(func.coalesce(func.sum(TradeModel.fee), Decimal("0")))
            .where(and_(*conditions))
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or Decimal("0")

    async def get_realized_pnl(
        self,
        bot_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Decimal:
        """
        Get total realized PnL for a bot.

        Args:
            bot_id: Bot identifier
            start: Start time filter (optional)
            end: End time filter (optional)

        Returns:
            Total realized PnL
        """
        conditions = [
            TradeModel.bot_id == bot_id,
            TradeModel.realized_pnl.isnot(None),
        ]

        if start:
            conditions.append(TradeModel.timestamp >= start)
        if end:
            conditions.append(TradeModel.timestamp <= end)

        query = (
            select(func.coalesce(func.sum(TradeModel.realized_pnl), Decimal("0")))
            .where(and_(*conditions))
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or Decimal("0")

    async def get_trade_volume(
        self,
        bot_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Decimal:
        """
        Get total trade volume for a bot.

        Args:
            bot_id: Bot identifier
            start: Start time filter (optional)
            end: End time filter (optional)

        Returns:
            Total trade volume (price * quantity)
        """
        conditions = [TradeModel.bot_id == bot_id]

        if start:
            conditions.append(TradeModel.timestamp >= start)
        if end:
            conditions.append(TradeModel.timestamp <= end)

        query = (
            select(
                func.coalesce(
                    func.sum(TradeModel.price * TradeModel.quantity),
                    Decimal("0"),
                )
            )
            .where(and_(*conditions))
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or Decimal("0")

    async def count_trades(
        self,
        bot_id: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        """
        Count trades for a bot.

        Args:
            bot_id: Bot identifier
            start: Start time filter (optional)
            end: End time filter (optional)

        Returns:
            Number of trades
        """
        conditions = [TradeModel.bot_id == bot_id]

        if start:
            conditions.append(TradeModel.timestamp >= start)
        if end:
            conditions.append(TradeModel.timestamp <= end)

        query = select(func.count()).select_from(TradeModel).where(and_(*conditions))

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or 0
