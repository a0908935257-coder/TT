"""
Balance Repository.

Provides data access for balance snapshot records.
"""

from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

from sqlalchemy import and_, func, select

from core import get_logger
from core.models import Balance, MarketType
from data.database import BalanceModel, DatabaseManager

from .base import BaseRepository

logger = get_logger(__name__)


class BalanceRepository(BaseRepository[Balance, BalanceModel]):
    """
    Repository for Balance data access.

    Provides operations for balance snapshots and history.

    Example:
        >>> repo = BalanceRepository(db)
        >>> await repo.save_snapshot("bot_1", balances)
        >>> latest = await repo.get_latest_snapshot("bot_1")
    """

    def __init__(
        self,
        db: DatabaseManager,
        market_type: MarketType = MarketType.SPOT,
    ):
        """
        Initialize BalanceRepository.

        Args:
            db: DatabaseManager instance
            market_type: Default market type for balances
        """
        super().__init__(db)
        self._market_type = market_type

    def _to_domain(self, orm_model: BalanceModel) -> Balance:
        """Convert ORM model to domain model."""
        return orm_model.to_domain()

    def _to_orm(self, domain_model: Balance, **kwargs) -> BalanceModel:
        """Convert domain model to ORM model."""
        bot_id = kwargs.get("bot_id")
        market_type = kwargs.get("market_type", self._market_type)
        return BalanceModel.from_domain(domain_model, bot_id, market_type)

    # =========================================================================
    # Snapshot Operations
    # =========================================================================

    async def save_snapshot(
        self,
        bot_id: str,
        balances: list[Balance],
        market_type: Optional[MarketType] = None,
    ) -> list[Balance]:
        """
        Save balance snapshot for a bot.

        Creates new balance records with current timestamp.

        Args:
            bot_id: Bot identifier
            balances: List of Balance domain models
            market_type: Market type (defaults to repository default)

        Returns:
            List of saved Balances
        """
        mt = market_type or self._market_type
        timestamp = datetime.now(timezone.utc)

        orm_models = []
        for balance in balances:
            model = self._to_orm(balance, bot_id=bot_id, market_type=mt)
            model.timestamp = timestamp
            orm_models.append(model)

        saved = await self._add_all(orm_models)
        logger.debug(f"Saved {len(saved)} balance snapshots for bot: {bot_id}")
        return [self._to_domain(m) for m in saved]

    async def get_latest_snapshot(
        self,
        bot_id: str,
        market_type: Optional[MarketType] = None,
    ) -> list[Balance]:
        """
        Get latest balance snapshot for a bot.

        Args:
            bot_id: Bot identifier
            market_type: Market type filter (optional)

        Returns:
            List of latest balances
        """
        conditions = [BalanceModel.bot_id == bot_id]

        if market_type:
            mt_value = market_type.value if hasattr(market_type, "value") else market_type
            conditions.append(BalanceModel.market_type == mt_value)

        # Subquery to get the latest timestamp
        subquery = (
            select(func.max(BalanceModel.timestamp))
            .where(and_(*conditions))
            .scalar_subquery()
        )

        # Get all balances at the latest timestamp
        query = select(BalanceModel).where(
            and_(
                *conditions,
                BalanceModel.timestamp == subquery,
            )
        )

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_balance_history(
        self,
        bot_id: str,
        asset: str,
        days: int = 30,
        market_type: Optional[MarketType] = None,
    ) -> list[dict]:
        """
        Get balance history for an asset.

        Args:
            bot_id: Bot identifier
            asset: Asset name (e.g., "USDT")
            days: Number of days of history
            market_type: Market type filter (optional)

        Returns:
            List of dicts with timestamp and balance info
        """
        start_time = datetime.now(timezone.utc) - timedelta(days=days)

        conditions = [
            BalanceModel.bot_id == bot_id,
            BalanceModel.asset == asset.upper(),
            BalanceModel.timestamp >= start_time,
        ]

        if market_type:
            mt_value = market_type.value if hasattr(market_type, "value") else market_type
            conditions.append(BalanceModel.market_type == mt_value)

        query = (
            select(BalanceModel)
            .where(and_(*conditions))
            .order_by(BalanceModel.timestamp.asc())
        )

        orm_models = await self._fetch_all(query)

        return [
            {
                "timestamp": m.timestamp,
                "free": m.free,
                "locked": m.locked,
                "total": m.total,
            }
            for m in orm_models
        ]

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_balance_at_time(
        self,
        bot_id: str,
        asset: str,
        timestamp: datetime,
    ) -> Optional[Balance]:
        """
        Get balance at a specific point in time.

        Returns the most recent balance before or at the given timestamp.

        Args:
            bot_id: Bot identifier
            asset: Asset name
            timestamp: Point in time

        Returns:
            Balance or None if not found
        """
        query = (
            select(BalanceModel)
            .where(
                and_(
                    BalanceModel.bot_id == bot_id,
                    BalanceModel.asset == asset.upper(),
                    BalanceModel.timestamp <= timestamp,
                )
            )
            .order_by(BalanceModel.timestamp.desc())
            .limit(1)
        )

        orm_model = await self._fetch_one(query)
        return self._to_domain(orm_model) if orm_model else None

    async def get_total_balance(
        self,
        bot_id: str,
        quote_asset: str = "USDT",
        market_type: Optional[MarketType] = None,
    ) -> Decimal:
        """
        Get total balance in quote asset.

        Note: This only returns balance of the quote asset itself.
        For full portfolio value, conversion rates would be needed.

        Args:
            bot_id: Bot identifier
            quote_asset: Quote asset (e.g., "USDT")
            market_type: Market type filter (optional)

        Returns:
            Total balance of quote asset
        """
        latest = await self.get_latest_snapshot(bot_id, market_type)

        for balance in latest:
            if balance.asset == quote_asset.upper():
                return balance.total

        return Decimal("0")

    async def delete_old_snapshots(
        self,
        bot_id: str,
        days_to_keep: int = 90,
    ) -> int:
        """
        Delete old balance snapshots.

        Args:
            bot_id: Bot identifier
            days_to_keep: Number of days to retain

        Returns:
            Number of records deleted
        """
        from sqlalchemy import delete

        cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

        query = delete(BalanceModel).where(
            and_(
                BalanceModel.bot_id == bot_id,
                BalanceModel.timestamp < cutoff,
            )
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            count = result.rowcount

        logger.debug(f"Deleted {count} old balance snapshots for bot: {bot_id}")
        return count
