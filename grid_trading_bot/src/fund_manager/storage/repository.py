"""
Allocation Repository.

SQLite-based storage for fund allocation records and balance snapshots.
"""

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from src.core import get_logger

from ..models.records import AllocationRecord, BalanceSnapshot

logger = get_logger(__name__)

DEFAULT_DB_PATH = "data/fund_manager.db"


class AllocationRepository:
    """
    SQLite repository for fund allocation records.

    Provides persistent storage for allocation history and balance snapshots.

    Example:
        >>> repo = AllocationRepository("data/fund_manager.db")
        >>> repo.initialize()
        >>> repo.save_allocation(record)
        >>> history = repo.get_allocation_history(bot_id="grid_btc")
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Initialize repository.

        Args:
            db_path: Path to SQLite database file
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialized = False

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """
        Get database connection context manager.

        Yields:
            SQLite connection
        """
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def initialize(self) -> None:
        """Initialize database schema."""
        if self._initialized:
            return

        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Allocation records table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS allocation_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    bot_id TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    trigger TEXT NOT NULL,
                    previous_allocation TEXT NOT NULL,
                    new_allocation TEXT NOT NULL,
                    success INTEGER NOT NULL,
                    error_message TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Balance snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS balance_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_balance TEXT NOT NULL,
                    available_balance TEXT NOT NULL,
                    allocated_balance TEXT NOT NULL,
                    reserved_balance TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_allocation_bot_id
                ON allocation_records(bot_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_allocation_timestamp
                ON allocation_records(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_snapshot_timestamp
                ON balance_snapshots(timestamp)
            """)

        self._initialized = True
        logger.info(f"AllocationRepository initialized: {self._db_path}")

    # =========================================================================
    # Allocation Records
    # =========================================================================

    def save_allocation(self, record: AllocationRecord) -> None:
        """
        Save an allocation record.

        Args:
            record: AllocationRecord to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR REPLACE INTO allocation_records
                (id, timestamp, bot_id, amount, trigger, previous_allocation,
                 new_allocation, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.timestamp.isoformat(),
                    record.bot_id,
                    str(record.amount),
                    record.trigger,
                    str(record.previous_allocation),
                    str(record.new_allocation),
                    1 if record.success else 0,
                    record.error_message,
                ),
            )
        logger.debug(f"Saved allocation record: {record.id}")

    def save_allocations(self, records: List[AllocationRecord]) -> None:
        """
        Save multiple allocation records.

        Args:
            records: List of AllocationRecord to save
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO allocation_records
                (id, timestamp, bot_id, amount, trigger, previous_allocation,
                 new_allocation, success, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r.id,
                        r.timestamp.isoformat(),
                        r.bot_id,
                        str(r.amount),
                        r.trigger,
                        str(r.previous_allocation),
                        str(r.new_allocation),
                        1 if r.success else 0,
                        r.error_message,
                    )
                    for r in records
                ],
            )
        logger.debug(f"Saved {len(records)} allocation records")

    def get_allocation(self, record_id: str) -> Optional[AllocationRecord]:
        """
        Get allocation record by ID.

        Args:
            record_id: Record identifier

        Returns:
            AllocationRecord if found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM allocation_records WHERE id = ?",
                (record_id,),
            )
            row = cursor.fetchone()
            if row:
                return self._row_to_allocation(row)
        return None

    def get_allocation_history(
        self,
        bot_id: Optional[str] = None,
        trigger: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AllocationRecord]:
        """
        Get allocation history with filters.

        Args:
            bot_id: Filter by bot ID
            trigger: Filter by trigger type
            since: Filter records after this time
            until: Filter records before this time
            limit: Maximum records to return
            offset: Offset for pagination

        Returns:
            List of AllocationRecord
        """
        query = "SELECT * FROM allocation_records WHERE 1=1"
        params: List[Any] = []

        if bot_id:
            query += " AND bot_id = ?"
            params.append(bot_id)
        if trigger:
            query += " AND trigger = ?"
            params.append(trigger)
        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_allocation(row) for row in rows]

    def get_bot_allocations(
        self,
        bot_id: str,
        limit: int = 100,
    ) -> List[AllocationRecord]:
        """
        Get allocation history for a specific bot.

        Args:
            bot_id: Bot identifier
            limit: Maximum records to return

        Returns:
            List of AllocationRecord
        """
        return self.get_allocation_history(bot_id=bot_id, limit=limit)

    def get_latest_allocation(self, bot_id: str) -> Optional[AllocationRecord]:
        """
        Get latest allocation for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Latest AllocationRecord if exists
        """
        records = self.get_allocation_history(bot_id=bot_id, limit=1)
        return records[0] if records else None

    def _row_to_allocation(self, row: sqlite3.Row) -> AllocationRecord:
        """Convert database row to AllocationRecord."""
        return AllocationRecord(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            bot_id=row["bot_id"],
            amount=Decimal(row["amount"]),
            trigger=row["trigger"],
            previous_allocation=Decimal(row["previous_allocation"]),
            new_allocation=Decimal(row["new_allocation"]),
            success=bool(row["success"]),
            error_message=row["error_message"],
        )

    # =========================================================================
    # Balance Snapshots
    # =========================================================================

    def save_snapshot(self, snapshot: BalanceSnapshot) -> int:
        """
        Save a balance snapshot.

        Args:
            snapshot: BalanceSnapshot to save

        Returns:
            ID of saved record
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO balance_snapshots
                (timestamp, total_balance, available_balance,
                 allocated_balance, reserved_balance)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    snapshot.timestamp.isoformat(),
                    str(snapshot.total_balance),
                    str(snapshot.available_balance),
                    str(snapshot.allocated_balance),
                    str(snapshot.reserved_balance),
                ),
            )
            return cursor.lastrowid or 0

    def get_snapshots(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[BalanceSnapshot]:
        """
        Get balance snapshots.

        Args:
            since: Filter snapshots after this time
            until: Filter snapshots before this time
            limit: Maximum records to return

        Returns:
            List of BalanceSnapshot
        """
        query = "SELECT * FROM balance_snapshots WHERE 1=1"
        params: List[Any] = []

        if since:
            query += " AND timestamp >= ?"
            params.append(since.isoformat())
        if until:
            query += " AND timestamp <= ?"
            params.append(until.isoformat())

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_snapshot(row) for row in rows]

    def get_latest_snapshot(self) -> Optional[BalanceSnapshot]:
        """
        Get latest balance snapshot.

        Returns:
            Latest BalanceSnapshot if exists
        """
        snapshots = self.get_snapshots(limit=1)
        return snapshots[0] if snapshots else None

    def _row_to_snapshot(self, row: sqlite3.Row) -> BalanceSnapshot:
        """Convert database row to BalanceSnapshot."""
        return BalanceSnapshot(
            timestamp=datetime.fromisoformat(row["timestamp"]),
            total_balance=Decimal(row["total_balance"]),
            available_balance=Decimal(row["available_balance"]),
            allocated_balance=Decimal(row["allocated_balance"]),
            reserved_balance=Decimal(row["reserved_balance"]),
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(
        self,
        since: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Get allocation statistics.

        Args:
            since: Calculate stats from this time

        Returns:
            Dictionary with statistics
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Base query conditions
            time_condition = ""
            params: List[Any] = []
            if since:
                time_condition = " WHERE timestamp >= ?"
                params.append(since.isoformat())

            # Total allocations
            cursor.execute(
                f"SELECT COUNT(*) as count FROM allocation_records{time_condition}",
                params,
            )
            total_count = cursor.fetchone()["count"]

            # Successful allocations
            cursor.execute(
                f"SELECT COUNT(*) as count FROM allocation_records WHERE success = 1"
                f"{' AND timestamp >= ?' if since else ''}",
                params,
            )
            success_count = cursor.fetchone()["count"]

            # Total amount allocated
            cursor.execute(
                f"SELECT SUM(CAST(amount AS REAL)) as total FROM allocation_records"
                f" WHERE success = 1{' AND timestamp >= ?' if since else ''}",
                params,
            )
            total_amount = cursor.fetchone()["total"] or 0

            # Unique bots
            cursor.execute(
                f"SELECT COUNT(DISTINCT bot_id) as count FROM allocation_records"
                f"{time_condition}",
                params,
            )
            unique_bots = cursor.fetchone()["count"]

            # Allocations by trigger
            cursor.execute(
                f"SELECT trigger, COUNT(*) as count FROM allocation_records"
                f"{time_condition} GROUP BY trigger",
                params,
            )
            by_trigger = {row["trigger"]: row["count"] for row in cursor.fetchall()}

            return {
                "total_allocations": total_count,
                "successful_allocations": success_count,
                "failed_allocations": total_count - success_count,
                "total_amount_allocated": total_amount,
                "unique_bots": unique_bots,
                "by_trigger": by_trigger,
            }

    # =========================================================================
    # Maintenance
    # =========================================================================

    def cleanup_old_records(
        self,
        before: datetime,
        table: str = "allocation_records",
    ) -> int:
        """
        Delete records older than specified date.

        Args:
            before: Delete records before this time
            table: Table to clean up

        Returns:
            Number of deleted records
        """
        if table not in ("allocation_records", "balance_snapshots"):
            raise ValueError(f"Invalid table: {table}")

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {table} WHERE timestamp < ?",
                (before.isoformat(),),
            )
            deleted = cursor.rowcount
            logger.info(f"Cleaned up {deleted} records from {table}")
            return deleted

    def vacuum(self) -> None:
        """Vacuum database to reclaim space."""
        with self._get_connection() as conn:
            conn.execute("VACUUM")
        logger.info("Database vacuumed")
