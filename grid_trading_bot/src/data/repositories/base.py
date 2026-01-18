"""
Base Repository.

Provides generic base class for all repositories with common
database operations and type conversion patterns.
"""

from abc import ABC, abstractmethod
from typing import Generic, Optional, TypeVar

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import get_logger
from src.data.database import DatabaseManager

logger = get_logger(__name__)

# Type variables for generic repository
T = TypeVar("T")  # Domain model type
M = TypeVar("M")  # ORM model type


class BaseRepository(ABC, Generic[T, M]):
    """
    Abstract base repository with common operations.

    Provides a generic interface for data access with automatic
    conversion between domain models and ORM models.

    Type Parameters:
        T: Domain model type (e.g., Order from core.models)
        M: ORM model type (e.g., OrderModel from data.database)

    Example:
        >>> class OrderRepository(BaseRepository[Order, OrderModel]):
        ...     def _to_domain(self, orm_model: OrderModel) -> Order:
        ...         return orm_model.to_domain()
        ...
        ...     def _to_orm(self, domain_model: Order) -> OrderModel:
        ...         return OrderModel.from_domain(domain_model)
    """

    def __init__(self, db: DatabaseManager):
        """
        Initialize repository.

        Args:
            db: DatabaseManager instance
        """
        self._db = db

    @abstractmethod
    def _to_domain(self, orm_model: M) -> T:
        """
        Convert ORM model to domain model.

        Args:
            orm_model: ORM model instance

        Returns:
            Domain model instance
        """
        pass

    @abstractmethod
    def _to_orm(self, domain_model: T, **kwargs) -> M:
        """
        Convert domain model to ORM model.

        Args:
            domain_model: Domain model instance
            **kwargs: Additional fields (e.g., bot_id)

        Returns:
            ORM model instance
        """
        pass

    async def _get_session(self) -> AsyncSession:
        """Get database session context manager."""
        return self._db.get_session()

    async def _execute_query(self, query):
        """
        Execute a query and return results.

        Args:
            query: SQLAlchemy query

        Returns:
            Query result
        """
        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result

    async def _fetch_one(self, query) -> Optional[M]:
        """
        Fetch single ORM model from query.

        Args:
            query: SQLAlchemy select query

        Returns:
            ORM model or None
        """
        async with self._db.get_session() as session:
            result = await session.execute(query)
            return result.scalar_one_or_none()

    async def _fetch_all(self, query) -> list[M]:
        """
        Fetch all ORM models from query.

        Args:
            query: SQLAlchemy select query

        Returns:
            List of ORM models
        """
        async with self._db.get_session() as session:
            result = await session.execute(query)
            return list(result.scalars().all())

    async def _add(self, orm_model: M) -> M:
        """
        Add ORM model to database.

        Args:
            orm_model: ORM model to add

        Returns:
            Added ORM model with generated fields
        """
        async with self._db.get_session() as session:
            session.add(orm_model)
            await session.flush()
            await session.refresh(orm_model)
            return orm_model

    async def _add_all(self, orm_models: list[M]) -> list[M]:
        """
        Add multiple ORM models to database.

        Args:
            orm_models: List of ORM models to add

        Returns:
            List of added ORM models
        """
        async with self._db.get_session() as session:
            session.add_all(orm_models)
            await session.flush()
            for model in orm_models:
                await session.refresh(model)
            return orm_models

    async def _update(self, orm_model: M) -> M:
        """
        Update ORM model in database.

        Args:
            orm_model: ORM model to update

        Returns:
            Updated ORM model
        """
        async with self._db.get_session() as session:
            merged = await session.merge(orm_model)
            await session.flush()
            return merged

    async def _delete(self, orm_model: M) -> bool:
        """
        Delete ORM model from database.

        Args:
            orm_model: ORM model to delete

        Returns:
            True if deleted
        """
        async with self._db.get_session() as session:
            await session.delete(orm_model)
            return True
