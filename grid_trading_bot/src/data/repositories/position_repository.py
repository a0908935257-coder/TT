"""
Position Repository.

Provides data access for futures position records.
"""

from typing import Optional

from sqlalchemy import and_, delete, select

from src.core import get_logger
from src.core.models import Position, PositionSide
from src.data.database import DatabaseManager, PositionModel

from .base import BaseRepository

logger = get_logger(__name__)


class PositionRepository(BaseRepository[Position, PositionModel]):
    """
    Repository for Position data access.

    Provides CRUD operations for futures position records.

    Example:
        >>> repo = PositionRepository(db)
        >>> await repo.upsert(position, bot_id="bot_1")
        >>> pos = await repo.get_position("bot_1", "BTCUSDT")
    """

    def _to_domain(self, orm_model: PositionModel) -> Position:
        """Convert ORM model to domain model."""
        return orm_model.to_domain()

    def _to_orm(self, domain_model: Position, **kwargs) -> PositionModel:
        """Convert domain model to ORM model."""
        bot_id = kwargs.get("bot_id")
        return PositionModel.from_domain(domain_model, bot_id)

    # =========================================================================
    # CRUD Operations
    # =========================================================================

    async def upsert(
        self,
        position: Position,
        bot_id: str,
    ) -> Position:
        """
        Insert or update position.

        If position exists for bot/symbol/side, update it.
        Otherwise, create new position record.

        Args:
            position: Position domain model
            bot_id: Bot identifier

        Returns:
            Upserted Position
        """
        side_value = position.side.value if hasattr(position.side, "value") else position.side

        # Check for existing position
        query = select(PositionModel).where(
            and_(
                PositionModel.bot_id == bot_id,
                PositionModel.symbol == position.symbol,
                PositionModel.side == side_value,
            )
        )
        existing = await self._fetch_one(query)

        if existing:
            # Update existing position
            existing.quantity = position.quantity
            existing.entry_price = position.entry_price
            existing.mark_price = position.mark_price
            existing.liquidation_price = position.liquidation_price
            existing.leverage = position.leverage
            existing.margin = position.margin
            existing.unrealized_pnl = position.unrealized_pnl
            existing.margin_type = position.margin_type
            existing.updated_at = position.updated_at

            updated = await self._update(existing)
            logger.debug(f"Updated position: {bot_id}/{position.symbol}")
            return self._to_domain(updated)
        else:
            # Create new position
            orm_model = self._to_orm(position, bot_id=bot_id)
            saved = await self._add(orm_model)
            logger.debug(f"Created position: {bot_id}/{position.symbol}")
            return self._to_domain(saved)

    async def get_position(
        self,
        bot_id: str,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> Optional[Position]:
        """
        Get position for a bot and symbol.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair
            side: Position side filter (optional)

        Returns:
            Position or None if not found
        """
        conditions = [
            PositionModel.bot_id == bot_id,
            PositionModel.symbol == symbol.upper(),
        ]

        if side:
            side_value = side.value if hasattr(side, "value") else side
            conditions.append(PositionModel.side == side_value)

        query = select(PositionModel).where(and_(*conditions))
        orm_model = await self._fetch_one(query)
        return self._to_domain(orm_model) if orm_model else None

    async def get_all_positions(
        self,
        bot_id: str,
        with_quantity: bool = True,
    ) -> list[Position]:
        """
        Get all positions for a bot.

        Args:
            bot_id: Bot identifier
            with_quantity: Only return positions with quantity > 0

        Returns:
            List of positions
        """
        conditions = [PositionModel.bot_id == bot_id]

        if with_quantity:
            conditions.append(PositionModel.quantity > 0)

        query = (
            select(PositionModel)
            .where(and_(*conditions))
            .order_by(PositionModel.symbol)
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def delete_position(
        self,
        bot_id: str,
        symbol: str,
        side: Optional[PositionSide] = None,
    ) -> bool:
        """
        Delete position record.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair
            side: Position side (optional, deletes all sides if not specified)

        Returns:
            True if deleted
        """
        conditions = [
            PositionModel.bot_id == bot_id,
            PositionModel.symbol == symbol.upper(),
        ]

        if side:
            side_value = side.value if hasattr(side, "value") else side
            conditions.append(PositionModel.side == side_value)

        query = delete(PositionModel).where(and_(*conditions))

        async with self._db.get_session() as session:
            result = await session.execute(query)
            deleted = result.rowcount > 0

        if deleted:
            logger.debug(f"Deleted position: {bot_id}/{symbol}")

        return deleted

    async def delete_all_positions(self, bot_id: str) -> int:
        """
        Delete all positions for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Number of positions deleted
        """
        query = delete(PositionModel).where(PositionModel.bot_id == bot_id)

        async with self._db.get_session() as session:
            result = await session.execute(query)
            count = result.rowcount

        logger.debug(f"Deleted {count} positions for bot: {bot_id}")
        return count

    async def get_total_unrealized_pnl(self, bot_id: str) -> "Decimal":
        """
        Get total unrealized PnL for all positions.

        Args:
            bot_id: Bot identifier

        Returns:
            Total unrealized PnL
        """
        from decimal import Decimal

        from sqlalchemy import func

        query = (
            select(func.coalesce(func.sum(PositionModel.unrealized_pnl), Decimal("0")))
            .where(
                and_(
                    PositionModel.bot_id == bot_id,
                    PositionModel.quantity > 0,
                )
            )
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar() or Decimal("0")
