"""
Snapshot and Backup System.

Provides state snapshots, point-in-time recovery,
and automated backup management.
"""

import asyncio
import gzip
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core import get_logger

logger = get_logger(__name__)


class SnapshotType(Enum):
    """Type of snapshot."""

    FULL = "full"  # Complete state snapshot
    INCREMENTAL = "incremental"  # Only changes since last snapshot
    MANUAL = "manual"  # User-triggered snapshot


class SnapshotStatus(Enum):
    """Snapshot status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CORRUPTED = "corrupted"


@dataclass
class SnapshotMetadata:
    """Metadata for a snapshot."""

    snapshot_id: str
    snapshot_type: SnapshotType
    status: SnapshotStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    checksum: str = ""
    node_id: str = ""
    description: str = ""
    base_snapshot_id: Optional[str] = None  # For incremental
    state_version: int = 0
    entry_count: int = 0
    compressed: bool = True
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "snapshot_type": self.snapshot_type.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "node_id": self.node_id,
            "description": self.description,
            "base_snapshot_id": self.base_snapshot_id,
            "state_version": self.state_version,
            "entry_count": self.entry_count,
            "compressed": self.compressed,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SnapshotMetadata":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            snapshot_type=SnapshotType(data["snapshot_type"]),
            status=SnapshotStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            size_bytes=data.get("size_bytes", 0),
            checksum=data.get("checksum", ""),
            node_id=data.get("node_id", ""),
            description=data.get("description", ""),
            base_snapshot_id=data.get("base_snapshot_id"),
            state_version=data.get("state_version", 0),
            entry_count=data.get("entry_count", 0),
            compressed=data.get("compressed", True),
            tags=data.get("tags", {}),
        )


@dataclass
class BackupConfig:
    """Backup configuration."""

    # Paths
    backup_dir: Path = Path("backups")
    snapshot_prefix: str = "snapshot"

    # Scheduling
    full_backup_interval_hours: int = 24
    incremental_backup_interval_hours: int = 1
    max_snapshots: int = 168  # 7 days of hourly

    # Compression
    compress: bool = True
    compression_level: int = 6

    # Retention
    retention_days: int = 30
    min_snapshots_to_keep: int = 5

    # Verification
    verify_after_create: bool = True
    verify_before_restore: bool = True


class SnapshotManager:
    """
    Manages state snapshots and backups.

    Provides snapshot creation, restoration, and
    automated backup scheduling.
    """

    def __init__(
        self,
        state_provider: Callable[[], Dict[str, Any]],
        state_restorer: Callable[[Dict[str, Any]], None],
        node_id: str = "",
        config: Optional[BackupConfig] = None,
    ):
        """
        Initialize snapshot manager.

        Args:
            state_provider: Function that returns current state
            state_restorer: Function to restore state from dict
            node_id: Node identifier
            config: Backup configuration
        """
        self._get_state = state_provider
        self._restore_state = state_restorer
        self._node_id = node_id
        self._config = config or BackupConfig()

        # Ensure backup directory exists
        self._config.backup_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        self._last_full_snapshot: Optional[datetime] = None
        self._last_incremental_snapshot: Optional[datetime] = None
        self._last_state_hash: Optional[str] = None

        # Callbacks
        self._on_snapshot_complete: List[Callable[[SnapshotMetadata], Any]] = []

    async def start(self) -> None:
        """Start backup scheduler."""
        if self._running:
            return

        self._running = True
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())

        logger.info(
            f"Snapshot manager started, backup dir: {self._config.backup_dir}"
        )

    async def stop(self) -> None:
        """Stop backup scheduler."""
        if not self._running:
            return

        self._running = False

        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass

        logger.info("Snapshot manager stopped")

    async def _scheduler_loop(self) -> None:
        """Automated backup scheduling loop."""
        while self._running:
            try:
                now = datetime.now(timezone.utc)

                # Check for full backup
                if self._should_full_backup(now):
                    await self.create_snapshot(
                        snapshot_type=SnapshotType.FULL,
                        description="Scheduled full backup",
                    )
                    self._last_full_snapshot = now

                # Check for incremental backup
                elif self._should_incremental_backup(now):
                    await self.create_snapshot(
                        snapshot_type=SnapshotType.INCREMENTAL,
                        description="Scheduled incremental backup",
                    )
                    self._last_incremental_snapshot = now

                # Cleanup old snapshots
                await self._cleanup_old_snapshots()

                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(60)

    def _should_full_backup(self, now: datetime) -> bool:
        """Check if full backup is needed."""
        if self._last_full_snapshot is None:
            return True

        elapsed = now - self._last_full_snapshot
        return elapsed >= timedelta(hours=self._config.full_backup_interval_hours)

    def _should_incremental_backup(self, now: datetime) -> bool:
        """Check if incremental backup is needed."""
        if self._last_incremental_snapshot is None:
            return True

        elapsed = now - self._last_incremental_snapshot
        return elapsed >= timedelta(hours=self._config.incremental_backup_interval_hours)

    async def create_snapshot(
        self,
        snapshot_type: SnapshotType = SnapshotType.FULL,
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> SnapshotMetadata:
        """
        Create a new snapshot.

        Args:
            snapshot_type: Type of snapshot
            description: Optional description
            tags: Optional tags

        Returns:
            Snapshot metadata
        """
        snapshot_id = self._generate_snapshot_id()
        now = datetime.now(timezone.utc)

        metadata = SnapshotMetadata(
            snapshot_id=snapshot_id,
            snapshot_type=snapshot_type,
            status=SnapshotStatus.IN_PROGRESS,
            created_at=now,
            node_id=self._node_id,
            description=description,
            compressed=self._config.compress,
            tags=tags or {},
        )

        try:
            # Get current state
            state = self._get_state()
            metadata.entry_count = len(state)

            # For incremental, get only changes
            if snapshot_type == SnapshotType.INCREMENTAL:
                last_full = await self._get_latest_full_snapshot()
                if last_full:
                    metadata.base_snapshot_id = last_full.snapshot_id
                    # In real implementation, would diff against base

            # Serialize state
            state_json = json.dumps(state, sort_keys=True, default=str)
            state_bytes = state_json.encode("utf-8")

            # Calculate checksum before compression
            metadata.checksum = hashlib.sha256(state_bytes).hexdigest()

            # Compress if enabled
            if self._config.compress:
                state_bytes = gzip.compress(
                    state_bytes,
                    compresslevel=self._config.compression_level,
                )

            metadata.size_bytes = len(state_bytes)

            # Write snapshot file
            snapshot_path = self._get_snapshot_path(snapshot_id)
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)

            with open(snapshot_path, "wb") as f:
                f.write(state_bytes)

            # Write metadata
            metadata_path = self._get_metadata_path(snapshot_id)
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Verify if configured
            if self._config.verify_after_create:
                if not await self._verify_snapshot(snapshot_id):
                    metadata.status = SnapshotStatus.CORRUPTED
                    raise ValueError("Snapshot verification failed")

            metadata.status = SnapshotStatus.COMPLETED
            metadata.completed_at = datetime.now(timezone.utc)

            # Update metadata file
            with open(metadata_path, "w") as f:
                json.dump(metadata.to_dict(), f, indent=2)

            logger.info(
                f"Snapshot created: {snapshot_id} "
                f"({metadata.size_bytes} bytes, {metadata.entry_count} entries)"
            )

            # Notify callbacks
            for callback in self._on_snapshot_complete:
                try:
                    result = callback(metadata)
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Snapshot callback error: {e}")

            return metadata

        except Exception as e:
            metadata.status = SnapshotStatus.FAILED
            logger.error(f"Failed to create snapshot: {e}")
            raise

    async def restore_snapshot(
        self,
        snapshot_id: str,
        verify: Optional[bool] = None,
    ) -> bool:
        """
        Restore state from a snapshot.

        Args:
            snapshot_id: Snapshot ID to restore
            verify: Verify snapshot before restore (None = use config)

        Returns:
            True if restore successful
        """
        verify = verify if verify is not None else self._config.verify_before_restore

        metadata = await self.get_snapshot_metadata(snapshot_id)
        if not metadata:
            raise ValueError(f"Snapshot not found: {snapshot_id}")

        if metadata.status != SnapshotStatus.COMPLETED:
            raise ValueError(f"Snapshot not completed: {metadata.status.value}")

        # Verify if configured
        if verify:
            if not await self._verify_snapshot(snapshot_id):
                raise ValueError("Snapshot verification failed")

        try:
            # Read snapshot file
            snapshot_path = self._get_snapshot_path(snapshot_id)

            with open(snapshot_path, "rb") as f:
                state_bytes = f.read()

            # Decompress if needed
            if metadata.compressed:
                state_bytes = gzip.decompress(state_bytes)

            # Verify checksum
            checksum = hashlib.sha256(state_bytes).hexdigest()
            if checksum != metadata.checksum:
                raise ValueError("Checksum mismatch")

            # Parse state
            state = json.loads(state_bytes.decode("utf-8"))

            # Restore state
            self._restore_state(state)

            logger.info(
                f"Snapshot restored: {snapshot_id} ({metadata.entry_count} entries)"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to restore snapshot {snapshot_id}: {e}")
            raise

    async def _verify_snapshot(self, snapshot_id: str) -> bool:
        """
        Verify snapshot integrity.

        Returns:
            True if snapshot is valid
        """
        try:
            metadata = await self.get_snapshot_metadata(snapshot_id)
            if not metadata:
                return False

            snapshot_path = self._get_snapshot_path(snapshot_id)
            if not snapshot_path.exists():
                return False

            with open(snapshot_path, "rb") as f:
                state_bytes = f.read()

            # Check size
            if len(state_bytes) != metadata.size_bytes:
                return False

            # Decompress and check checksum
            if metadata.compressed:
                state_bytes = gzip.decompress(state_bytes)

            checksum = hashlib.sha256(state_bytes).hexdigest()
            return checksum == metadata.checksum

        except Exception as e:
            logger.error(f"Snapshot verification failed: {e}")
            return False

    async def get_snapshot_metadata(
        self,
        snapshot_id: str,
    ) -> Optional[SnapshotMetadata]:
        """Get metadata for a snapshot."""
        metadata_path = self._get_metadata_path(snapshot_id)

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            data = json.load(f)

        return SnapshotMetadata.from_dict(data)

    async def list_snapshots(
        self,
        snapshot_type: Optional[SnapshotType] = None,
        status: Optional[SnapshotStatus] = None,
        limit: int = 100,
    ) -> List[SnapshotMetadata]:
        """
        List available snapshots.

        Args:
            snapshot_type: Filter by type
            status: Filter by status
            limit: Maximum results

        Returns:
            List of snapshot metadata
        """
        snapshots = []

        for metadata_file in self._config.backup_dir.glob("*/metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                metadata = SnapshotMetadata.from_dict(data)

                # Apply filters
                if snapshot_type and metadata.snapshot_type != snapshot_type:
                    continue
                if status and metadata.status != status:
                    continue

                snapshots.append(metadata)

            except Exception as e:
                logger.error(f"Failed to read metadata {metadata_file}: {e}")

        # Sort by creation time, newest first
        snapshots.sort(key=lambda s: s.created_at, reverse=True)

        return snapshots[:limit]

    async def delete_snapshot(self, snapshot_id: str) -> bool:
        """
        Delete a snapshot.

        Args:
            snapshot_id: Snapshot to delete

        Returns:
            True if deleted
        """
        snapshot_dir = self._config.backup_dir / snapshot_id

        if not snapshot_dir.exists():
            return False

        try:
            shutil.rmtree(snapshot_dir)
            logger.info(f"Snapshot deleted: {snapshot_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete snapshot {snapshot_id}: {e}")
            return False

    async def _get_latest_full_snapshot(self) -> Optional[SnapshotMetadata]:
        """Get the latest full snapshot."""
        snapshots = await self.list_snapshots(
            snapshot_type=SnapshotType.FULL,
            status=SnapshotStatus.COMPLETED,
            limit=1,
        )
        return snapshots[0] if snapshots else None

    async def _cleanup_old_snapshots(self) -> int:
        """
        Remove old snapshots beyond retention.

        Returns:
            Number of snapshots deleted
        """
        cutoff = datetime.now(timezone.utc) - timedelta(
            days=self._config.retention_days
        )

        snapshots = await self.list_snapshots(status=SnapshotStatus.COMPLETED)

        # Keep minimum number of snapshots
        if len(snapshots) <= self._config.min_snapshots_to_keep:
            return 0

        # Find snapshots to delete
        to_delete = []
        keep_count = 0

        for snapshot in snapshots:
            if keep_count < self._config.min_snapshots_to_keep:
                keep_count += 1
                continue

            if snapshot.created_at < cutoff:
                to_delete.append(snapshot.snapshot_id)

            if len(snapshots) - len(to_delete) > self._config.max_snapshots:
                to_delete.append(snapshot.snapshot_id)

        # Delete old snapshots
        deleted = 0
        for snapshot_id in to_delete:
            if await self.delete_snapshot(snapshot_id):
                deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old snapshots")

        return deleted

    def _generate_snapshot_id(self) -> str:
        """Generate unique snapshot ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{self._config.snapshot_prefix}_{timestamp}_{self._node_id[:8]}"

    def _get_snapshot_path(self, snapshot_id: str) -> Path:
        """Get path for snapshot data file."""
        suffix = ".gz" if self._config.compress else ".json"
        return self._config.backup_dir / snapshot_id / f"state{suffix}"

    def _get_metadata_path(self, snapshot_id: str) -> Path:
        """Get path for snapshot metadata file."""
        return self._config.backup_dir / snapshot_id / "metadata.json"

    def on_snapshot_complete(
        self,
        callback: Callable[[SnapshotMetadata], Any],
    ) -> Callable[[], None]:
        """
        Register callback for snapshot completion.

        Args:
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_snapshot_complete.append(callback)

        def unsubscribe():
            if callback in self._on_snapshot_complete:
                self._on_snapshot_complete.remove(callback)

        return unsubscribe

    async def export_snapshot(
        self,
        snapshot_id: str,
        destination: Path,
    ) -> bool:
        """
        Export snapshot to external location.

        Args:
            snapshot_id: Snapshot to export
            destination: Destination path

        Returns:
            True if export successful
        """
        snapshot_dir = self._config.backup_dir / snapshot_id

        if not snapshot_dir.exists():
            return False

        try:
            destination.mkdir(parents=True, exist_ok=True)
            shutil.copytree(snapshot_dir, destination / snapshot_id)
            logger.info(f"Snapshot exported: {snapshot_id} -> {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to export snapshot: {e}")
            return False

    async def import_snapshot(
        self,
        source: Path,
    ) -> Optional[SnapshotMetadata]:
        """
        Import snapshot from external location.

        Args:
            source: Source snapshot directory

        Returns:
            Imported snapshot metadata
        """
        if not source.exists():
            return None

        metadata_path = source / "metadata.json"
        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, "r") as f:
                data = json.load(f)
            metadata = SnapshotMetadata.from_dict(data)

            # Copy to backup directory
            dest_dir = self._config.backup_dir / metadata.snapshot_id
            if dest_dir.exists():
                logger.warning(f"Snapshot already exists: {metadata.snapshot_id}")
                return metadata

            shutil.copytree(source, dest_dir)
            logger.info(f"Snapshot imported: {metadata.snapshot_id}")
            return metadata

        except Exception as e:
            logger.error(f"Failed to import snapshot: {e}")
            return None
