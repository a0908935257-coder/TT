"""
Bot State Repository.

Provides data access for bot state records.
"""

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, update

from src.core import get_logger
from src.data.database import BotStateBackupModel, BotStateModel, DatabaseManager

from .base import BaseRepository

logger = get_logger(__name__)


class BotStateRepository(BaseRepository[dict, BotStateModel]):
    """
    Repository for Bot State data access.

    Provides operations for bot state persistence.

    Example:
        >>> repo = BotStateRepository(db)
        >>> await repo.save_state("bot_1", "grid", "running", config, state_data)
        >>> state = await repo.get_state("bot_1")
    """

    def _to_domain(self, orm_model: BotStateModel) -> dict:
        """Convert ORM model to domain dict."""
        return orm_model.to_dict()

    def _to_orm(self, domain_model: dict, **kwargs) -> BotStateModel:
        """Convert domain dict to ORM model."""
        return BotStateModel.from_dict(domain_model)

    # =========================================================================
    # State Operations
    # =========================================================================

    async def backup_state(self, bot_id: str) -> Optional[dict]:
        """
        Backup current bot state before saving new state.

        Args:
            bot_id: Bot identifier

        Returns:
            Backup dict or None if no state to backup
        """
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        existing = await self._fetch_one(query)

        if not existing:
            return None

        try:
            backup = BotStateBackupModel(
                bot_id=existing.bot_id,
                bot_type=existing.bot_type,
                status=existing.status,
                config=existing.config,
                state_data=existing.state_data,
            )
            async with self._db.get_session() as session:
                session.add(backup)
            logger.debug(f"Backed up bot state: {bot_id}")
            return backup.to_dict()
        except Exception as e:
            logger.error(f"Failed to backup bot state {bot_id}: {e}")
            return None

    async def restore_from_backup(self, bot_id: str) -> Optional[dict]:
        """
        Restore bot state from most recent backup.

        Args:
            bot_id: Bot identifier

        Returns:
            Restored state dict or None if no backup found
        """
        from sqlalchemy import desc

        query = (
            select(BotStateBackupModel)
            .where(BotStateBackupModel.bot_id == bot_id)
            .order_by(desc(BotStateBackupModel.backed_up_at))
            .limit(1)
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            backup = result.scalar_one_or_none()

        if not backup:
            logger.warning(f"No backup found for bot: {bot_id}")
            return None

        # Restore by saving the backup data as current state
        restored = await self.save_state(
            bot_id=backup.bot_id,
            bot_type=backup.bot_type,
            status=backup.status,
            config=backup.config,
            state_data=backup.state_data,
        )

        logger.info(f"Restored bot state from backup: {bot_id}")
        return restored

    async def save_state(
        self,
        bot_id: str,
        bot_type: str,
        status: str,
        config: dict[str, Any],
        state_data: dict[str, Any],
    ) -> dict:
        """
        Save or update bot state (auto-backup before update).

        Args:
            bot_id: Bot identifier
            bot_type: Type of bot (e.g., "grid", "dca")
            status: Bot status (e.g., "running", "stopped")
            config: Bot configuration
            state_data: Bot runtime state data

        Returns:
            Saved state as dict
        """
        # Check if exists
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        existing = await self._fetch_one(query)

        if existing:
            # Auto-backup before update
            await self.backup_state(bot_id)
            # Update existing state
            existing.bot_type = bot_type
            existing.status = status
            existing.config = config
            existing.state_data = state_data
            existing.updated_at = datetime.now(timezone.utc)

            updated = await self._update(existing)
            logger.debug(f"Updated bot state: {bot_id}")
            return self._to_domain(updated)
        else:
            # Create new state
            model = BotStateModel(
                bot_id=bot_id,
                bot_type=bot_type,
                status=status,
                config=config,
                state_data=state_data,
            )
            saved = await self._add(model)
            logger.debug(f"Created bot state: {bot_id}")
            return self._to_domain(saved)

    async def get_state(self, bot_id: str) -> Optional[dict]:
        """
        Get bot state.

        Args:
            bot_id: Bot identifier

        Returns:
            State dict or None if not found
        """
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        orm_model = await self._fetch_one(query)
        return self._to_domain(orm_model) if orm_model else None

    async def update_status(
        self,
        bot_id: str,
        status: str,
    ) -> Optional[dict]:
        """
        Update bot status only.

        Args:
            bot_id: Bot identifier
            status: New status

        Returns:
            Updated state dict or None if not found
        """
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        existing = await self._fetch_one(query)

        if not existing:
            return None

        existing.status = status
        existing.updated_at = datetime.now(timezone.utc)

        updated = await self._update(existing)
        logger.debug(f"Updated bot status: {bot_id} -> {status}")
        return self._to_domain(updated)

    async def update_state_data(
        self,
        bot_id: str,
        state_data: dict[str, Any],
    ) -> Optional[dict]:
        """
        Update bot state data only.

        Args:
            bot_id: Bot identifier
            state_data: New state data

        Returns:
            Updated state dict or None if not found
        """
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        existing = await self._fetch_one(query)

        if not existing:
            return None

        existing.state_data = state_data
        existing.updated_at = datetime.now(timezone.utc)

        updated = await self._update(existing)
        logger.debug(f"Updated bot state data: {bot_id}")
        return self._to_domain(updated)

    async def update_config(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> Optional[dict]:
        """
        Update bot config only.

        Args:
            bot_id: Bot identifier
            config: New config

        Returns:
            Updated state dict or None if not found
        """
        query = select(BotStateModel).where(BotStateModel.bot_id == bot_id)
        existing = await self._fetch_one(query)

        if not existing:
            return None

        existing.config = config
        existing.updated_at = datetime.now(timezone.utc)

        updated = await self._update(existing)
        logger.debug(f"Updated bot config: {bot_id}")
        return self._to_domain(updated)

    # =========================================================================
    # Query Operations
    # =========================================================================

    async def get_all_bots(
        self,
        status: Optional[str] = None,
        bot_type: Optional[str] = None,
    ) -> list[dict]:
        """
        Get all bot states.

        Args:
            status: Status filter (optional)
            bot_type: Bot type filter (optional)

        Returns:
            List of bot state dicts
        """
        from sqlalchemy import and_

        conditions = []

        if status:
            conditions.append(BotStateModel.status == status)
        if bot_type:
            conditions.append(BotStateModel.bot_type == bot_type)

        if conditions:
            query = (
                select(BotStateModel)
                .where(and_(*conditions))
                .order_by(BotStateModel.bot_id)
            )
        else:
            query = select(BotStateModel).order_by(BotStateModel.bot_id)

        orm_models = await self._fetch_all(query)
        return [self._to_domain(m) for m in orm_models]

    async def get_running_bots(self) -> list[dict]:
        """
        Get all running bots.

        Returns:
            List of running bot state dicts
        """
        return await self.get_all_bots(status="running")

    async def delete_state(self, bot_id: str) -> bool:
        """
        Delete bot state.

        Args:
            bot_id: Bot identifier

        Returns:
            True if deleted
        """
        from sqlalchemy import delete

        query = delete(BotStateModel).where(BotStateModel.bot_id == bot_id)

        async with self._db.get_session() as session:
            result = await session.execute(query)
            deleted = result.rowcount > 0

        if deleted:
            logger.debug(f"Deleted bot state: {bot_id}")

        return deleted

    async def bot_exists(self, bot_id: str) -> bool:
        """
        Check if bot state exists.

        Args:
            bot_id: Bot identifier

        Returns:
            True if exists
        """
        from sqlalchemy import func

        query = (
            select(func.count())
            .select_from(BotStateModel)
            .where(BotStateModel.bot_id == bot_id)
        )

        async with self._db.get_session() as session:
            result = await session.execute(query)
            return (result.scalar() or 0) > 0
