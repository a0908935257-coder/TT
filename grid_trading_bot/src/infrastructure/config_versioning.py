"""
Configuration Version Control.

Provides versioning, history tracking, and rollback
capabilities for configuration changes.
"""

import asyncio
import copy
import difflib
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.core import get_logger

logger = get_logger(__name__)


class ChangeType(Enum):
    """Type of configuration change."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ROLLBACK = "rollback"


class ApprovalStatus(Enum):
    """Approval status for changes."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


@dataclass
class ConfigChange:
    """Record of a configuration change."""

    change_id: str
    version: int
    change_type: ChangeType
    timestamp: datetime
    author: str
    description: str
    path: str  # Config path that changed
    old_value: Any
    new_value: Any
    approval_status: ApprovalStatus = ApprovalStatus.AUTO_APPROVED
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None
    checksum: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "change_id": self.change_id,
            "version": self.version,
            "change_type": self.change_type.value,
            "timestamp": self.timestamp.isoformat(),
            "author": self.author,
            "description": self.description,
            "path": self.path,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "approval_status": self.approval_status.value,
            "approved_by": self.approved_by,
            "approved_at": (
                self.approved_at.isoformat() if self.approved_at else None
            ),
            "checksum": self.checksum,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigChange":
        """Create from dictionary."""
        return cls(
            change_id=data["change_id"],
            version=data["version"],
            change_type=ChangeType(data["change_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            author=data["author"],
            description=data["description"],
            path=data["path"],
            old_value=data["old_value"],
            new_value=data["new_value"],
            approval_status=ApprovalStatus(data.get("approval_status", "auto_approved")),
            approved_by=data.get("approved_by"),
            approved_at=(
                datetime.fromisoformat(data["approved_at"])
                if data.get("approved_at")
                else None
            ),
            checksum=data.get("checksum", ""),
            tags=data.get("tags", {}),
        )


@dataclass
class ConfigSnapshot:
    """Complete configuration snapshot."""

    version: int
    timestamp: datetime
    config: Dict[str, Any]
    checksum: str
    author: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version": self.version,
            "timestamp": self.timestamp.isoformat(),
            "config": self.config,
            "checksum": self.checksum,
            "author": self.author,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConfigSnapshot":
        """Create from dictionary."""
        return cls(
            version=data["version"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            config=data["config"],
            checksum=data["checksum"],
            author=data["author"],
            description=data["description"],
        )


@dataclass
class VersioningConfig:
    """Configuration versioning settings."""

    # Storage
    history_dir: Path = Path("config_history")
    max_versions: int = 100

    # Approval
    require_approval: bool = False
    auto_approve_authors: List[str] = field(default_factory=lambda: ["system"])

    # Validation
    validate_before_apply: bool = True

    # Diff
    show_diff_on_change: bool = True


class ConfigVersionManager:
    """
    Manages configuration versioning and history.

    Provides version tracking, diff, rollback,
    and approval workflow for configuration changes.
    """

    def __init__(
        self,
        config_provider: Callable[[], Dict[str, Any]],
        config_applier: Callable[[Dict[str, Any]], bool],
        config_validator: Optional[Callable[[Dict[str, Any]], Tuple[bool, str]]] = None,
        config: Optional[VersioningConfig] = None,
    ):
        """
        Initialize version manager.

        Args:
            config_provider: Function to get current config
            config_applier: Function to apply config changes
            config_validator: Optional validation function
            config: Versioning configuration
        """
        self._get_config = config_provider
        self._apply_config = config_applier
        self._validate_config = config_validator
        self._config = config or VersioningConfig()

        # Ensure history directory exists
        self._config.history_dir.mkdir(parents=True, exist_ok=True)

        # State
        self._current_version = 0
        self._changes: List[ConfigChange] = []
        self._snapshots: List[ConfigSnapshot] = []

        # Load history
        self._load_history()

        # Callbacks
        self._on_change: List[Callable[[ConfigChange], Any]] = []
        self._on_rollback: List[Callable[[int, int], Any]] = []

    def _load_history(self) -> None:
        """Load change history from disk."""
        # Load changes
        changes_file = self._config.history_dir / "changes.json"
        if changes_file.exists():
            with open(changes_file, "r") as f:
                data = json.load(f)
            self._changes = [ConfigChange.from_dict(c) for c in data]

        # Load snapshots
        snapshots_file = self._config.history_dir / "snapshots.json"
        if snapshots_file.exists():
            with open(snapshots_file, "r") as f:
                data = json.load(f)
            self._snapshots = [ConfigSnapshot.from_dict(s) for s in data]

        # Determine current version
        if self._snapshots:
            self._current_version = max(s.version for s in self._snapshots)
        elif self._changes:
            self._current_version = max(c.version for c in self._changes)

    def _save_history(self) -> None:
        """Save change history to disk."""
        # Save changes
        changes_file = self._config.history_dir / "changes.json"
        with open(changes_file, "w") as f:
            json.dump([c.to_dict() for c in self._changes], f, indent=2)

        # Save snapshots
        snapshots_file = self._config.history_dir / "snapshots.json"
        with open(snapshots_file, "w") as f:
            json.dump([s.to_dict() for s in self._snapshots], f, indent=2)

    def _compute_checksum(self, config: Dict[str, Any]) -> str:
        """Compute checksum of configuration."""
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def _generate_change_id(self) -> str:
        """Generate unique change ID."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"change_{timestamp}_{self._current_version + 1}"

    async def record_change(
        self,
        path: str,
        new_value: Any,
        author: str = "system",
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> ConfigChange:
        """
        Record a configuration change.

        Args:
            path: Config path being changed
            new_value: New value
            author: Who made the change
            description: Change description
            tags: Optional tags

        Returns:
            Change record
        """
        current_config = self._get_config()
        old_value = self._get_value_at_path(current_config, path)

        # Determine change type
        if old_value is None:
            change_type = ChangeType.CREATE
        elif new_value is None:
            change_type = ChangeType.DELETE
        else:
            change_type = ChangeType.UPDATE

        # Create change record
        self._current_version += 1
        change = ConfigChange(
            change_id=self._generate_change_id(),
            version=self._current_version,
            change_type=change_type,
            timestamp=datetime.now(timezone.utc),
            author=author,
            description=description,
            path=path,
            old_value=old_value,
            new_value=new_value,
            tags=tags or {},
        )

        # Determine approval status
        if self._config.require_approval:
            if author in self._config.auto_approve_authors:
                change.approval_status = ApprovalStatus.AUTO_APPROVED
            else:
                change.approval_status = ApprovalStatus.PENDING
        else:
            change.approval_status = ApprovalStatus.AUTO_APPROVED

        # Calculate checksum
        new_config = self._apply_value_at_path(current_config, path, new_value)
        change.checksum = self._compute_checksum(new_config)

        # Add to history
        self._changes.append(change)

        # If approved, create snapshot
        if change.approval_status in [
            ApprovalStatus.APPROVED,
            ApprovalStatus.AUTO_APPROVED,
        ]:
            await self._create_snapshot(
                new_config,
                change.version,
                author,
                description,
            )

        self._save_history()

        # Log diff if enabled
        if self._config.show_diff_on_change:
            diff = self._compute_diff(old_value, new_value)
            if diff:
                logger.info(f"Config change at {path}:\n{diff}")

        # Notify callbacks
        for callback in self._on_change:
            try:
                result = callback(change)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Change callback error: {e}")

        return change

    async def _create_snapshot(
        self,
        config: Dict[str, Any],
        version: int,
        author: str,
        description: str,
    ) -> ConfigSnapshot:
        """Create a configuration snapshot."""
        snapshot = ConfigSnapshot(
            version=version,
            timestamp=datetime.now(timezone.utc),
            config=copy.deepcopy(config),
            checksum=self._compute_checksum(config),
            author=author,
            description=description,
        )

        self._snapshots.append(snapshot)

        # Trim old snapshots
        while len(self._snapshots) > self._config.max_versions:
            self._snapshots.pop(0)

        return snapshot

    async def approve_change(
        self,
        change_id: str,
        approver: str,
    ) -> bool:
        """
        Approve a pending change.

        Args:
            change_id: Change to approve
            approver: Who is approving

        Returns:
            True if approved
        """
        change = self.get_change(change_id)
        if not change:
            return False

        if change.approval_status != ApprovalStatus.PENDING:
            return False

        change.approval_status = ApprovalStatus.APPROVED
        change.approved_by = approver
        change.approved_at = datetime.now(timezone.utc)

        # Apply the change
        current_config = self._get_config()
        new_config = self._apply_value_at_path(
            current_config,
            change.path,
            change.new_value,
        )

        # Validate if enabled
        if self._config.validate_before_apply and self._validate_config:
            valid, error = self._validate_config(new_config)
            if not valid:
                change.approval_status = ApprovalStatus.REJECTED
                logger.error(f"Config validation failed: {error}")
                self._save_history()
                return False

        # Apply config
        success = self._apply_config(new_config)
        if not success:
            change.approval_status = ApprovalStatus.REJECTED
            self._save_history()
            return False

        # Create snapshot
        await self._create_snapshot(
            new_config,
            change.version,
            change.author,
            f"Approved: {change.description}",
        )

        self._save_history()
        logger.info(f"Change {change_id} approved by {approver}")
        return True

    async def reject_change(
        self,
        change_id: str,
        rejector: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a pending change.

        Args:
            change_id: Change to reject
            rejector: Who is rejecting
            reason: Rejection reason

        Returns:
            True if rejected
        """
        change = self.get_change(change_id)
        if not change:
            return False

        if change.approval_status != ApprovalStatus.PENDING:
            return False

        change.approval_status = ApprovalStatus.REJECTED
        change.approved_by = rejector
        change.approved_at = datetime.now(timezone.utc)
        change.tags["rejection_reason"] = reason

        self._save_history()
        logger.info(f"Change {change_id} rejected by {rejector}: {reason}")
        return True

    async def rollback_to_version(
        self,
        target_version: int,
        author: str = "system",
    ) -> bool:
        """
        Rollback configuration to a previous version.

        Args:
            target_version: Version to rollback to
            author: Who is performing rollback

        Returns:
            True if rollback successful
        """
        # Find snapshot
        snapshot = next(
            (s for s in self._snapshots if s.version == target_version),
            None,
        )

        if not snapshot:
            logger.error(f"Version {target_version} not found")
            return False

        current_version = self._current_version

        # Validate if enabled
        if self._config.validate_before_apply and self._validate_config:
            valid, error = self._validate_config(snapshot.config)
            if not valid:
                logger.error(f"Rollback validation failed: {error}")
                return False

        # Apply config
        success = self._apply_config(snapshot.config)
        if not success:
            logger.error("Failed to apply rollback config")
            return False

        # Record rollback as a change
        self._current_version += 1
        rollback_change = ConfigChange(
            change_id=self._generate_change_id(),
            version=self._current_version,
            change_type=ChangeType.ROLLBACK,
            timestamp=datetime.now(timezone.utc),
            author=author,
            description=f"Rollback from v{current_version} to v{target_version}",
            path="*",  # Full config rollback
            old_value=None,
            new_value=None,
            approval_status=ApprovalStatus.AUTO_APPROVED,
            checksum=snapshot.checksum,
        )
        self._changes.append(rollback_change)

        # Create new snapshot
        await self._create_snapshot(
            snapshot.config,
            self._current_version,
            author,
            f"Rollback to v{target_version}",
        )

        self._save_history()

        # Notify callbacks
        for callback in self._on_rollback:
            try:
                result = callback(current_version, target_version)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Rollback callback error: {e}")

        logger.info(f"Rolled back from v{current_version} to v{target_version}")
        return True

    def get_change(self, change_id: str) -> Optional[ConfigChange]:
        """Get a specific change by ID."""
        return next((c for c in self._changes if c.change_id == change_id), None)

    def get_changes(
        self,
        path: Optional[str] = None,
        author: Optional[str] = None,
        change_type: Optional[ChangeType] = None,
        limit: int = 50,
    ) -> List[ConfigChange]:
        """
        Get change history.

        Args:
            path: Filter by config path
            author: Filter by author
            change_type: Filter by change type
            limit: Maximum results

        Returns:
            List of changes
        """
        results = self._changes.copy()

        if path:
            results = [c for c in results if c.path == path or c.path.startswith(path)]
        if author:
            results = [c for c in results if c.author == author]
        if change_type:
            results = [c for c in results if c.change_type == change_type]

        # Sort by version descending
        results.sort(key=lambda c: c.version, reverse=True)

        return results[:limit]

    def get_snapshot(self, version: int) -> Optional[ConfigSnapshot]:
        """Get a specific snapshot."""
        return next((s for s in self._snapshots if s.version == version), None)

    def get_snapshots(self, limit: int = 20) -> List[ConfigSnapshot]:
        """Get recent snapshots."""
        snapshots = sorted(self._snapshots, key=lambda s: s.version, reverse=True)
        return snapshots[:limit]

    @property
    def current_version(self) -> int:
        """Get current version."""
        return self._current_version

    def get_pending_changes(self) -> List[ConfigChange]:
        """Get changes pending approval."""
        return [
            c for c in self._changes
            if c.approval_status == ApprovalStatus.PENDING
        ]

    def diff_versions(
        self,
        version1: int,
        version2: int,
    ) -> Optional[str]:
        """
        Get diff between two versions.

        Args:
            version1: First version
            version2: Second version

        Returns:
            Diff string or None
        """
        snapshot1 = self.get_snapshot(version1)
        snapshot2 = self.get_snapshot(version2)

        if not snapshot1 or not snapshot2:
            return None

        return self._compute_diff(snapshot1.config, snapshot2.config)

    def _compute_diff(self, old: Any, new: Any) -> str:
        """Compute diff between two values."""
        old_str = json.dumps(old, indent=2, sort_keys=True, default=str)
        new_str = json.dumps(new, indent=2, sort_keys=True, default=str)

        diff = difflib.unified_diff(
            old_str.splitlines(keepends=True),
            new_str.splitlines(keepends=True),
            fromfile="old",
            tofile="new",
        )

        return "".join(diff)

    def _get_value_at_path(
        self,
        config: Dict[str, Any],
        path: str,
    ) -> Any:
        """Get value at a config path."""
        keys = path.split(".")
        value = config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None

        return value

    def _apply_value_at_path(
        self,
        config: Dict[str, Any],
        path: str,
        value: Any,
    ) -> Dict[str, Any]:
        """Apply value at a config path."""
        config = copy.deepcopy(config)
        keys = path.split(".")
        current = config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        if value is None:
            current.pop(keys[-1], None)
        else:
            current[keys[-1]] = value

        return config

    def on_change(
        self,
        callback: Callable[[ConfigChange], Any],
    ) -> Callable[[], None]:
        """
        Register callback for config changes.

        Args:
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_change.append(callback)

        def unsubscribe():
            if callback in self._on_change:
                self._on_change.remove(callback)

        return unsubscribe

    def on_rollback(
        self,
        callback: Callable[[int, int], Any],
    ) -> Callable[[], None]:
        """
        Register callback for rollbacks.

        Args:
            callback: Callback function(old_version, new_version)

        Returns:
            Unsubscribe function
        """
        self._on_rollback.append(callback)

        def unsubscribe():
            if callback in self._on_rollback:
                self._on_rollback.remove(callback)

        return unsubscribe
