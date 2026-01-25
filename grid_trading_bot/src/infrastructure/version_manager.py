"""
Version Manager.

Provides comprehensive version management including:
- State versioning and rollback
- Data migration between versions
- Canary/Grayscale deployment support
- Version compatibility checking
"""

import asyncio
import copy
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.core import get_logger

logger = get_logger(__name__)


# =============================================================================
# Version Types
# =============================================================================


class VersionType(Enum):
    """Type of version."""

    MAJOR = "major"  # Breaking changes, requires migration
    MINOR = "minor"  # New features, backward compatible
    PATCH = "patch"  # Bug fixes, fully compatible


class MigrationDirection(Enum):
    """Direction of data migration."""

    UPGRADE = "upgrade"  # From older to newer version
    DOWNGRADE = "downgrade"  # From newer to older version


class DeploymentPhase(Enum):
    """Grayscale deployment phase."""

    CANARY = "canary"  # Testing with small percentage
    GRADUAL = "gradual"  # Gradually increasing percentage
    FULL = "full"  # Full deployment
    ROLLBACK = "rollback"  # Rolling back to previous version


# =============================================================================
# Version Info
# =============================================================================


@dataclass
class Version:
    """Semantic version representation."""

    major: int
    minor: int
    patch: int
    pre_release: Optional[str] = None
    build: Optional[str] = None

    def __str__(self) -> str:
        """String representation."""
        version = f"{self.major}.{self.minor}.{self.patch}"
        if self.pre_release:
            version += f"-{self.pre_release}"
        if self.build:
            version += f"+{self.build}"
        return version

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Version):
            return False
        return (self.major, self.minor, self.patch) == (other.major, other.minor, other.patch)

    def __lt__(self, other: "Version") -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)

    def __le__(self, other: "Version") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Version") -> bool:
        return not self <= other

    def __ge__(self, other: "Version") -> bool:
        return not self < other

    @classmethod
    def parse(cls, version_str: str) -> "Version":
        """Parse version string."""
        # Handle pre-release and build metadata
        pre_release = None
        build = None

        if "+" in version_str:
            version_str, build = version_str.split("+", 1)
        if "-" in version_str:
            version_str, pre_release = version_str.split("-", 1)

        parts = version_str.split(".")
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")

        return cls(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
            pre_release=pre_release,
            build=build,
        )

    def is_compatible_with(self, other: "Version") -> bool:
        """Check if this version is backward compatible with another."""
        # Same major version = compatible (semver rule)
        return self.major == other.major

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "major": self.major,
            "minor": self.minor,
            "patch": self.patch,
            "pre_release": self.pre_release,
            "build": self.build,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Version":
        """Create from dictionary."""
        return cls(
            major=data["major"],
            minor=data["minor"],
            patch=data["patch"],
            pre_release=data.get("pre_release"),
            build=data.get("build"),
        )


# =============================================================================
# State Snapshot
# =============================================================================


@dataclass
class StateSnapshot:
    """A versioned state snapshot for rollback support."""

    snapshot_id: str
    version: Version
    timestamp: datetime
    state_data: Dict[str, Any]
    checksum: str
    description: str = ""
    bot_id: Optional[str] = None
    is_stable: bool = False  # Marked as known good state
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "snapshot_id": self.snapshot_id,
            "version": self.version.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "state_data": self.state_data,
            "checksum": self.checksum,
            "description": self.description,
            "bot_id": self.bot_id,
            "is_stable": self.is_stable,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateSnapshot":
        """Create from dictionary."""
        return cls(
            snapshot_id=data["snapshot_id"],
            version=Version.from_dict(data["version"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            state_data=data["state_data"],
            checksum=data["checksum"],
            description=data.get("description", ""),
            bot_id=data.get("bot_id"),
            is_stable=data.get("is_stable", False),
            tags=data.get("tags", {}),
        )


# =============================================================================
# Migration
# =============================================================================


@dataclass
class Migration:
    """Defines a data migration between versions."""

    from_version: Version
    to_version: Version
    direction: MigrationDirection
    description: str
    migrate_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    rollback_func: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    is_reversible: bool = True

    def apply(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply migration to data."""
        return self.migrate_func(copy.deepcopy(data))

    def rollback(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Rollback migration if possible."""
        if not self.is_reversible or not self.rollback_func:
            return None
        return self.rollback_func(copy.deepcopy(data))


# =============================================================================
# Migration Registry
# =============================================================================


class MigrationRegistry:
    """
    Registry of data migrations between versions.

    Supports forward (upgrade) and backward (downgrade) migrations.
    """

    def __init__(self):
        """Initialize registry."""
        self._migrations: Dict[str, Migration] = {}

    def register(self, migration: Migration) -> None:
        """Register a migration."""
        key = f"{migration.from_version}->{migration.to_version}"
        self._migrations[key] = migration
        logger.debug(f"Registered migration: {key}")

    def get_migration_path(
        self,
        from_version: Version,
        to_version: Version,
    ) -> List[Migration]:
        """
        Get the migration path between two versions.

        Args:
            from_version: Starting version
            to_version: Target version

        Returns:
            List of migrations to apply in order
        """
        if from_version == to_version:
            return []

        # Determine direction
        direction = (
            MigrationDirection.UPGRADE
            if from_version < to_version
            else MigrationDirection.DOWNGRADE
        )

        # Find path (simple linear search for now)
        path = []
        current = from_version

        # Get all migrations in direction
        relevant_migrations = [
            m for m in self._migrations.values()
            if m.direction == direction
        ]

        # Sort by version
        if direction == MigrationDirection.UPGRADE:
            relevant_migrations.sort(key=lambda m: m.from_version)
        else:
            relevant_migrations.sort(key=lambda m: m.from_version, reverse=True)

        # Build path
        for migration in relevant_migrations:
            if direction == MigrationDirection.UPGRADE:
                if migration.from_version >= current and migration.to_version <= to_version:
                    if migration.from_version == current or not path:
                        path.append(migration)
                        current = migration.to_version
            else:
                if migration.from_version <= current and migration.to_version >= to_version:
                    if migration.from_version == current or not path:
                        path.append(migration)
                        current = migration.to_version

        return path

    def can_migrate(
        self,
        from_version: Version,
        to_version: Version,
    ) -> Tuple[bool, str]:
        """
        Check if migration is possible.

        Returns:
            Tuple of (can_migrate, reason)
        """
        path = self.get_migration_path(from_version, to_version)

        if not path and from_version != to_version:
            return False, f"No migration path from {from_version} to {to_version}"

        # Check if downgrade is reversible
        if from_version > to_version:
            for migration in path:
                if not migration.is_reversible:
                    return False, f"Migration {migration.from_version}->{migration.to_version} is not reversible"

        return True, "Migration is possible"


# =============================================================================
# Version Manager
# =============================================================================


class VersionManager:
    """
    Manages versions, snapshots, and migrations.

    Features:
    - State snapshots for rollback
    - Data migration between versions
    - Compatibility checking
    - Grayscale deployment support
    """

    def __init__(
        self,
        current_version: Version,
        storage_path: Optional[Path] = None,
        max_snapshots: int = 10,
    ):
        """
        Initialize version manager.

        Args:
            current_version: Current application version
            storage_path: Path for storing snapshots
            max_snapshots: Maximum number of snapshots to keep
        """
        self._current_version = current_version
        self._storage_path = storage_path or Path("data/snapshots")
        self._max_snapshots = max_snapshots

        self._snapshots: List[StateSnapshot] = []
        self._migration_registry = MigrationRegistry()
        self._stable_snapshot_id: Optional[str] = None

        # Ensure storage directory exists
        self._storage_path.mkdir(parents=True, exist_ok=True)

        # Load existing snapshots
        self._load_snapshots()

    @property
    def current_version(self) -> Version:
        """Get current version."""
        return self._current_version

    def _load_snapshots(self) -> None:
        """Load snapshots from storage."""
        index_file = self._storage_path / "snapshots_index.json"
        if not index_file.exists():
            return

        try:
            with open(index_file, "r") as f:
                data = json.load(f)

            self._snapshots = [StateSnapshot.from_dict(s) for s in data.get("snapshots", [])]
            self._stable_snapshot_id = data.get("stable_snapshot_id")

            logger.info(f"Loaded {len(self._snapshots)} snapshots")

        except Exception as e:
            logger.error(f"Failed to load snapshots: {e}")

    def _save_snapshots_index(self) -> None:
        """Save snapshots index to storage."""
        index_file = self._storage_path / "snapshots_index.json"

        data = {
            "snapshots": [s.to_dict() for s in self._snapshots],
            "stable_snapshot_id": self._stable_snapshot_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        with open(index_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _generate_snapshot_id(self) -> str:
        """Generate unique snapshot ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"snap_{timestamp}_{len(self._snapshots)}"

    def _compute_checksum(self, data: Dict[str, Any]) -> str:
        """Compute checksum of data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()[:16]

    # =========================================================================
    # Snapshot Management
    # =========================================================================

    def create_snapshot(
        self,
        state_data: Dict[str, Any],
        description: str = "",
        bot_id: Optional[str] = None,
        mark_stable: bool = False,
        tags: Optional[Dict[str, str]] = None,
    ) -> StateSnapshot:
        """
        Create a new state snapshot.

        Args:
            state_data: State data to snapshot
            description: Description of this snapshot
            bot_id: Optional bot ID
            mark_stable: Whether to mark as known good state
            tags: Optional tags

        Returns:
            Created snapshot
        """
        snapshot = StateSnapshot(
            snapshot_id=self._generate_snapshot_id(),
            version=self._current_version,
            timestamp=datetime.now(timezone.utc),
            state_data=copy.deepcopy(state_data),
            checksum=self._compute_checksum(state_data),
            description=description,
            bot_id=bot_id,
            is_stable=mark_stable,
            tags=tags or {},
        )

        # Save snapshot data
        snapshot_file = self._storage_path / f"{snapshot.snapshot_id}.json"
        with open(snapshot_file, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2, default=str)

        self._snapshots.append(snapshot)

        # Update stable marker
        if mark_stable:
            self._stable_snapshot_id = snapshot.snapshot_id

        # Trim old snapshots
        self._trim_snapshots()

        # Save index
        self._save_snapshots_index()

        logger.info(
            f"Created snapshot {snapshot.snapshot_id} "
            f"(version={snapshot.version}, stable={mark_stable})"
        )

        return snapshot

    def _trim_snapshots(self) -> None:
        """Trim old snapshots, keeping stable ones."""
        if len(self._snapshots) <= self._max_snapshots:
            return

        # Sort by timestamp (oldest first)
        sorted_snapshots = sorted(self._snapshots, key=lambda s: s.timestamp)

        # Keep stable snapshots and newest ones
        to_remove = []
        for snapshot in sorted_snapshots:
            if len(self._snapshots) - len(to_remove) <= self._max_snapshots:
                break
            if not snapshot.is_stable and snapshot.snapshot_id != self._stable_snapshot_id:
                to_remove.append(snapshot)

        # Remove snapshots and their files
        for snapshot in to_remove:
            self._snapshots.remove(snapshot)
            snapshot_file = self._storage_path / f"{snapshot.snapshot_id}.json"
            if snapshot_file.exists():
                snapshot_file.unlink()

        logger.info(f"Trimmed {len(to_remove)} old snapshots")

    def get_snapshot(self, snapshot_id: str) -> Optional[StateSnapshot]:
        """Get a snapshot by ID."""
        return next((s for s in self._snapshots if s.snapshot_id == snapshot_id), None)

    def get_stable_snapshot(self) -> Optional[StateSnapshot]:
        """Get the last known stable snapshot."""
        if self._stable_snapshot_id:
            return self.get_snapshot(self._stable_snapshot_id)

        # Fall back to latest stable-marked snapshot
        stable = [s for s in self._snapshots if s.is_stable]
        if stable:
            return max(stable, key=lambda s: s.timestamp)

        return None

    def get_latest_snapshot(
        self,
        bot_id: Optional[str] = None,
    ) -> Optional[StateSnapshot]:
        """Get the latest snapshot, optionally filtered by bot ID."""
        snapshots = self._snapshots
        if bot_id:
            snapshots = [s for s in snapshots if s.bot_id == bot_id]

        if not snapshots:
            return None

        return max(snapshots, key=lambda s: s.timestamp)

    def mark_stable(self, snapshot_id: str) -> bool:
        """Mark a snapshot as stable (known good state)."""
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False

        snapshot.is_stable = True
        self._stable_snapshot_id = snapshot_id
        self._save_snapshots_index()

        logger.info(f"Marked snapshot {snapshot_id} as stable")
        return True

    # =========================================================================
    # Rollback
    # =========================================================================

    def rollback_to_snapshot(
        self,
        snapshot_id: str,
    ) -> Tuple[bool, Dict[str, Any], str]:
        """
        Rollback to a specific snapshot.

        Args:
            snapshot_id: ID of snapshot to rollback to

        Returns:
            Tuple of (success, state_data, message)
        """
        snapshot = self.get_snapshot(snapshot_id)
        if not snapshot:
            return False, {}, f"Snapshot {snapshot_id} not found"

        # Check version compatibility
        if not snapshot.version.is_compatible_with(self._current_version):
            # Try migration
            can_migrate, reason = self._migration_registry.can_migrate(
                snapshot.version, self._current_version
            )
            if not can_migrate:
                return False, {}, f"Version incompatible and cannot migrate: {reason}"

            # Apply migration
            try:
                migrated_data = self._migrate_data(
                    snapshot.state_data,
                    snapshot.version,
                    self._current_version,
                )
                logger.info(
                    f"Migrated data from {snapshot.version} to {self._current_version}"
                )
                return True, migrated_data, "Rollback successful (with migration)"
            except Exception as e:
                return False, {}, f"Migration failed: {e}"

        # Verify checksum
        current_checksum = self._compute_checksum(snapshot.state_data)
        if current_checksum != snapshot.checksum:
            logger.warning(
                f"Checksum mismatch for snapshot {snapshot_id} - data may be corrupted"
            )
            return False, {}, "Snapshot data corrupted (checksum mismatch)"

        logger.info(f"Rolling back to snapshot {snapshot_id}")
        return True, copy.deepcopy(snapshot.state_data), "Rollback successful"

    def rollback_to_stable(self) -> Tuple[bool, Dict[str, Any], str]:
        """
        Rollback to the last known stable state.

        Returns:
            Tuple of (success, state_data, message)
        """
        stable = self.get_stable_snapshot()
        if not stable:
            return False, {}, "No stable snapshot available"

        return self.rollback_to_snapshot(stable.snapshot_id)

    # =========================================================================
    # Migration
    # =========================================================================

    def register_migration(self, migration: Migration) -> None:
        """Register a data migration."""
        self._migration_registry.register(migration)

    def _migrate_data(
        self,
        data: Dict[str, Any],
        from_version: Version,
        to_version: Version,
    ) -> Dict[str, Any]:
        """
        Migrate data between versions.

        Args:
            data: Data to migrate
            from_version: Current data version
            to_version: Target version

        Returns:
            Migrated data
        """
        path = self._migration_registry.get_migration_path(from_version, to_version)

        if not path:
            # No migration needed or not possible
            return data

        result = copy.deepcopy(data)
        for migration in path:
            logger.info(
                f"Applying migration: {migration.from_version} -> {migration.to_version}"
            )
            result = migration.apply(result)

        return result

    def check_compatibility(
        self,
        data_version: Version,
    ) -> Tuple[bool, str, bool]:
        """
        Check if data is compatible with current version.

        Returns:
            Tuple of (is_compatible, message, needs_migration)
        """
        if data_version == self._current_version:
            return True, "Versions match", False

        if data_version.is_compatible_with(self._current_version):
            if data_version < self._current_version:
                can_migrate, reason = self._migration_registry.can_migrate(
                    data_version, self._current_version
                )
                return True, f"Compatible but older - {reason}", can_migrate
            else:
                return True, "Compatible (newer minor/patch)", False

        # Major version mismatch
        can_migrate, reason = self._migration_registry.can_migrate(
            data_version, self._current_version
        )
        if can_migrate:
            return True, f"Major version mismatch but migration available: {reason}", True

        return False, f"Incompatible: data={data_version}, current={self._current_version}", False


# =============================================================================
# Grayscale Deployment Manager
# =============================================================================


@dataclass
class DeploymentStatus:
    """Status of grayscale deployment."""

    phase: DeploymentPhase
    new_version: Version
    old_version: Version
    percentage: int  # 0-100
    started_at: datetime
    updated_at: datetime
    healthy_instances: int
    total_instances: int
    errors: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "phase": self.phase.value,
            "new_version": str(self.new_version),
            "old_version": str(self.old_version),
            "percentage": self.percentage,
            "started_at": self.started_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "healthy_instances": self.healthy_instances,
            "total_instances": self.total_instances,
            "errors": self.errors,
        }


class GrayscaleDeploymentManager:
    """
    Manages grayscale (canary) deployments.

    Features:
    - Gradual rollout with configurable percentages
    - Health monitoring during deployment
    - Automatic rollback on errors
    - Version coexistence support
    """

    def __init__(
        self,
        version_manager: VersionManager,
        health_check: Optional[Callable[[], bool]] = None,
    ):
        """
        Initialize deployment manager.

        Args:
            version_manager: Version manager instance
            health_check: Optional health check function
        """
        self._version_manager = version_manager
        self._health_check = health_check

        self._deployment_status: Optional[DeploymentStatus] = None
        self._deployment_phases = [
            (DeploymentPhase.CANARY, 5),     # 5% canary
            (DeploymentPhase.GRADUAL, 25),   # 25%
            (DeploymentPhase.GRADUAL, 50),   # 50%
            (DeploymentPhase.GRADUAL, 75),   # 75%
            (DeploymentPhase.FULL, 100),     # 100%
        ]
        self._error_threshold = 3  # Max errors before rollback
        self._phase_duration_seconds = 300  # 5 minutes per phase

    def start_deployment(
        self,
        new_version: Version,
        old_version: Version,
        total_instances: int = 1,
    ) -> DeploymentStatus:
        """
        Start a grayscale deployment.

        Args:
            new_version: Version being deployed
            old_version: Version being replaced
            total_instances: Total number of bot instances

        Returns:
            Initial deployment status
        """
        now = datetime.now(timezone.utc)

        self._deployment_status = DeploymentStatus(
            phase=DeploymentPhase.CANARY,
            new_version=new_version,
            old_version=old_version,
            percentage=self._deployment_phases[0][1],
            started_at=now,
            updated_at=now,
            healthy_instances=0,
            total_instances=total_instances,
            errors=[],
        )

        logger.info(
            f"Started grayscale deployment: {old_version} -> {new_version} "
            f"(canary {self._deployment_status.percentage}%)"
        )

        return self._deployment_status

    def should_use_new_version(self, instance_id: str) -> bool:
        """
        Determine if an instance should use the new version.

        Uses consistent hashing to ensure same instance always gets same answer.

        Args:
            instance_id: Unique instance identifier

        Returns:
            True if instance should use new version
        """
        if not self._deployment_status:
            return True  # No deployment in progress, use current

        if self._deployment_status.phase == DeploymentPhase.FULL:
            return True
        if self._deployment_status.phase == DeploymentPhase.ROLLBACK:
            return False

        # Use hash to consistently assign instances
        hash_value = int(hashlib.md5(instance_id.encode()).hexdigest(), 16)
        threshold = self._deployment_status.percentage

        return (hash_value % 100) < threshold

    def record_health(self, instance_id: str, is_healthy: bool, error: Optional[str] = None) -> None:
        """
        Record health status of an instance.

        Args:
            instance_id: Instance identifier
            is_healthy: Whether instance is healthy
            error: Optional error message
        """
        if not self._deployment_status:
            return

        if is_healthy:
            self._deployment_status.healthy_instances = min(
                self._deployment_status.healthy_instances + 1,
                self._deployment_status.total_instances,
            )
        else:
            if error:
                self._deployment_status.errors.append(
                    f"{datetime.now(timezone.utc).isoformat()}: {instance_id}: {error}"
                )

        self._deployment_status.updated_at = datetime.now(timezone.utc)

        # Check for auto-rollback
        if len(self._deployment_status.errors) >= self._error_threshold:
            logger.error(
                f"Error threshold reached ({self._error_threshold}), initiating rollback"
            )
            self.rollback()

    def advance_phase(self) -> Tuple[bool, str]:
        """
        Advance to next deployment phase.

        Returns:
            Tuple of (success, message)
        """
        if not self._deployment_status:
            return False, "No deployment in progress"

        if self._deployment_status.phase == DeploymentPhase.FULL:
            return True, "Deployment already complete"

        if self._deployment_status.phase == DeploymentPhase.ROLLBACK:
            return False, "Deployment is rolling back"

        # Find current phase index
        current_pct = self._deployment_status.percentage
        current_idx = next(
            (i for i, (_, pct) in enumerate(self._deployment_phases) if pct == current_pct),
            0,
        )

        # Check if we can advance
        if current_idx >= len(self._deployment_phases) - 1:
            self._deployment_status.phase = DeploymentPhase.FULL
            self._deployment_status.percentage = 100
            logger.info("Deployment complete (100%)")
            return True, "Deployment complete"

        # Advance to next phase
        next_phase, next_pct = self._deployment_phases[current_idx + 1]
        self._deployment_status.phase = next_phase
        self._deployment_status.percentage = next_pct
        self._deployment_status.updated_at = datetime.now(timezone.utc)

        logger.info(f"Advanced deployment to {next_phase.value} ({next_pct}%)")
        return True, f"Advanced to {next_phase.value} ({next_pct}%)"

    def rollback(self) -> Tuple[bool, str]:
        """
        Rollback the deployment.

        Returns:
            Tuple of (success, message)
        """
        if not self._deployment_status:
            return False, "No deployment in progress"

        self._deployment_status.phase = DeploymentPhase.ROLLBACK
        self._deployment_status.percentage = 0
        self._deployment_status.updated_at = datetime.now(timezone.utc)

        logger.warning(
            f"Rolling back deployment from {self._deployment_status.new_version} "
            f"to {self._deployment_status.old_version}"
        )

        return True, f"Rolling back to {self._deployment_status.old_version}"

    def complete_rollback(self) -> None:
        """Complete the rollback and clear deployment status."""
        if self._deployment_status:
            logger.info(
                f"Rollback complete: reverted to {self._deployment_status.old_version}"
            )
        self._deployment_status = None

    def get_status(self) -> Optional[DeploymentStatus]:
        """Get current deployment status."""
        return self._deployment_status


# =============================================================================
# Default Migrations
# =============================================================================


def create_default_migrations() -> MigrationRegistry:
    """
    Create registry with default migrations.

    Returns:
        Configured migration registry
    """
    registry = MigrationRegistry()

    # Example migration: v1.0.0 -> v2.0.0
    # Add your actual migrations here based on your schema changes

    return registry
