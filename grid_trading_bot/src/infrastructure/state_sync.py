"""
State Synchronization Module.

Provides state replication and synchronization between
cluster nodes for high availability.
"""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from src.core import get_logger

logger = get_logger(__name__)


class SyncStrategy(Enum):
    """State synchronization strategy."""

    IMMEDIATE = "immediate"  # Sync immediately on change
    BATCHED = "batched"  # Batch changes and sync periodically
    ON_DEMAND = "on_demand"  # Sync only when requested


class ConflictResolution(Enum):
    """Conflict resolution strategy."""

    LAST_WRITE_WINS = "last_write_wins"
    LEADER_WINS = "leader_wins"
    MERGE = "merge"
    REJECT = "reject"


@dataclass
class StateChange:
    """Represents a state change."""

    key: str
    value: Any
    timestamp: datetime
    node_id: str
    version: int
    operation: str  # set, delete, update

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "node_id": self.node_id,
            "version": self.version,
            "operation": self.operation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateChange":
        """Create from dictionary."""
        return cls(
            key=data["key"],
            value=data["value"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            node_id=data["node_id"],
            version=data["version"],
            operation=data["operation"],
        )


@dataclass
class StateVersion:
    """Version information for state."""

    version: int
    checksum: str
    timestamp: datetime
    node_id: str


@dataclass
class StateSyncConfig:
    """State synchronization configuration."""

    # Sync settings
    sync_interval: float = 1.0  # Seconds between batched syncs
    max_batch_size: int = 100  # Max changes per batch
    sync_timeout: float = 5.0

    # Keys
    state_prefix: str = "state:"
    changes_channel: str = "state:changes"
    version_key: str = "state:version"

    # Conflict resolution
    conflict_resolution: ConflictResolution = ConflictResolution.LAST_WRITE_WINS

    # Retention
    change_history_size: int = 1000


class StateManager:
    """
    Manages distributed state with synchronization.

    Provides state storage with automatic replication
    and conflict resolution across cluster nodes.
    """

    def __init__(
        self,
        redis_client: Any,  # RedisManager
        node_id: str,
        is_leader_callback: Callable[[], bool],
        config: Optional[StateSyncConfig] = None,
    ):
        """
        Initialize state manager.

        Args:
            redis_client: Redis client instance
            node_id: This node's ID
            is_leader_callback: Function to check if this node is leader
            config: Sync configuration
        """
        self._redis = redis_client
        self._node_id = node_id
        self._is_leader = is_leader_callback
        self._config = config or StateSyncConfig()

        # Local state cache
        self._local_state: Dict[str, Any] = {}
        self._local_versions: Dict[str, int] = {}
        self._pending_changes: List[StateChange] = []

        # State
        self._running = False
        self._sync_task: Optional[asyncio.Task] = None
        self._subscriber_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_state_change: List[Callable[[str, Any, Any], Any]] = []

    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._is_leader()

    async def start(self) -> None:
        """Start state synchronization."""
        if self._running:
            return

        self._running = True

        # Load initial state
        await self._load_state()

        # Subscribe to changes
        await self._redis.subscribe(
            self._config.changes_channel,
            self._on_remote_change,
        )
        self._subscriber_task = asyncio.create_task(self._dummy_subscriber())
        self._subscriber_task.add_done_callback(self._on_task_done)

        # Start sync loop
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._sync_task.add_done_callback(self._on_task_done)

        logger.info(f"State manager started on node {self._node_id}")

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Log exceptions from background tasks."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.error(f"Background task failed: {exc}", exc_info=(type(exc), exc, exc.__traceback__))

    async def stop(self) -> None:
        """Stop state synchronization."""
        if not self._running:
            return

        self._running = False

        # Flush pending changes
        await self._flush_changes()

        # Cancel tasks
        for task in [self._sync_task, self._subscriber_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Unsubscribe
        await self._redis.unsubscribe(self._config.changes_channel)

        logger.info(f"State manager stopped on node {self._node_id}")

    async def _dummy_subscriber(self) -> None:
        """Dummy task to keep subscriber running."""
        while self._running:
            await asyncio.sleep(1.0)

    async def _load_state(self) -> None:
        """Load state from Redis."""
        keys = await self._redis.keys(f"{self._config.state_prefix}*")

        for key in keys:
            # Remove prefix
            state_key = key[len(self._config.state_prefix) :]
            value = await self._redis.get(f"{self._config.state_prefix}{state_key}")

            if value is not None:
                self._local_state[state_key] = value

                # Get version
                version_data = await self._redis.hget(
                    self._config.version_key,
                    state_key,
                )
                if version_data and isinstance(version_data, dict):
                    self._local_versions[state_key] = version_data.get("version", 0)

        logger.debug(f"Loaded {len(self._local_state)} state entries")

    async def _sync_loop(self) -> None:
        """Periodic sync loop for batched changes."""
        while self._running:
            try:
                await self._flush_changes()
                await asyncio.sleep(self._config.sync_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync error: {e}")
                await asyncio.sleep(1.0)

    async def _flush_changes(self) -> None:
        """Flush pending changes to Redis."""
        if not self._pending_changes:
            return

        # Get changes to process
        changes = self._pending_changes[: self._config.max_batch_size]
        self._pending_changes = self._pending_changes[self._config.max_batch_size :]

        for change in changes:
            try:
                await self._apply_change_to_redis(change)
            except Exception as e:
                logger.error(f"Failed to apply change {change.key}: {e}")

    async def _apply_change_to_redis(self, change: StateChange) -> None:
        """Apply a change to Redis."""
        if change.operation == "delete":
            await self._redis.delete(f"{self._config.state_prefix}{change.key}")
            await self._redis.hdel(self._config.version_key, change.key)
        else:
            await self._redis.set(
                f"{self._config.state_prefix}{change.key}",
                change.value,
            )

            # Update version
            await self._redis.hset(
                self._config.version_key,
                change.key,
                {
                    "version": change.version,
                    "timestamp": change.timestamp.isoformat(),
                    "node_id": change.node_id,
                },
            )

        # Publish change notification
        await self._redis.publish(
            self._config.changes_channel,
            change.to_dict(),
        )

    async def _on_remote_change(self, channel: str, data: Any) -> None:
        """Handle remote state change notification."""
        try:
            if isinstance(data, dict):
                change = StateChange.from_dict(data)

                # Skip our own changes
                if change.node_id == self._node_id:
                    return

                # Apply to local state
                await self._apply_remote_change(change)

        except Exception as e:
            logger.error(f"Error handling remote change: {e}")

    async def _apply_remote_change(self, change: StateChange) -> bool:
        """
        Apply a remote change to local state.

        Returns:
            True if change was applied
        """
        current_version = self._local_versions.get(change.key, 0)

        # Check for conflicts
        if change.version <= current_version:
            # Resolve conflict
            if self._config.conflict_resolution == ConflictResolution.LAST_WRITE_WINS:
                # Compare timestamps
                pass  # Will apply anyway since remote has newer timestamp
            elif self._config.conflict_resolution == ConflictResolution.LEADER_WINS:
                if not self.is_leader:
                    # Accept leader's changes
                    pass
                else:
                    # We're leader, reject remote change
                    return False
            elif self._config.conflict_resolution == ConflictResolution.REJECT:
                return False

        # Apply change
        old_value = self._local_state.get(change.key)

        if change.operation == "delete":
            self._local_state.pop(change.key, None)
            self._local_versions.pop(change.key, None)
        else:
            self._local_state[change.key] = change.value
            self._local_versions[change.key] = change.version

        # Notify callbacks
        for callback in self._on_state_change:
            try:
                result = callback(change.key, old_value, change.value)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"State change callback error: {e}")

        return True

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get state value.

        Args:
            key: State key
            default: Default value if not found

        Returns:
            State value or default
        """
        return self._local_state.get(key, default)

    async def set(self, key: str, value: Any) -> None:
        """
        Set state value.

        Args:
            key: State key
            value: State value
        """
        old_value = self._local_state.get(key)
        new_version = self._local_versions.get(key, 0) + 1

        # Update local state
        self._local_state[key] = value
        self._local_versions[key] = new_version

        # Queue change for sync
        change = StateChange(
            key=key,
            value=value,
            timestamp=datetime.now(timezone.utc),
            node_id=self._node_id,
            version=new_version,
            operation="set",
        )
        self._pending_changes.append(change)

        # Notify local callbacks
        for callback in self._on_state_change:
            try:
                result = callback(key, old_value, value)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def delete(self, key: str) -> None:
        """
        Delete state value.

        Args:
            key: State key
        """
        if key not in self._local_state:
            return

        old_value = self._local_state.pop(key, None)
        version = self._local_versions.pop(key, 0)

        # Queue change for sync
        change = StateChange(
            key=key,
            value=None,
            timestamp=datetime.now(timezone.utc),
            node_id=self._node_id,
            version=version,
            operation="delete",
        )
        self._pending_changes.append(change)

        # Notify local callbacks
        for callback in self._on_state_change:
            try:
                result = callback(key, old_value, None)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"State change callback error: {e}")

    async def get_all(self) -> Dict[str, Any]:
        """
        Get all state values.

        Returns:
            Dict of all state key-value pairs
        """
        return dict(self._local_state)

    async def get_keys(self, pattern: str = "*") -> List[str]:
        """
        Get state keys matching pattern.

        Args:
            pattern: Glob pattern

        Returns:
            List of matching keys
        """
        import fnmatch

        return [k for k in self._local_state.keys() if fnmatch.fnmatch(k, pattern)]

    def on_change(
        self,
        callback: Callable[[str, Any, Any], Any],
    ) -> Callable[[], None]:
        """
        Register callback for state changes.

        Args:
            callback: Callback function(key, old_value, new_value)

        Returns:
            Unsubscribe function
        """
        self._on_state_change.append(callback)

        def unsubscribe():
            if callback in self._on_state_change:
                self._on_state_change.remove(callback)

        return unsubscribe

    async def sync_from_leader(self) -> None:
        """Force sync state from leader."""
        if self.is_leader:
            return

        logger.info("Syncing state from leader")
        await self._load_state()

    def get_checksum(self) -> str:
        """
        Get checksum of current state.

        Returns:
            SHA256 checksum
        """
        state_str = json.dumps(self._local_state, sort_keys=True, default=str)
        return hashlib.sha256(state_str.encode()).hexdigest()[:16]

    async def verify_consistency(self) -> bool:
        """
        Verify state consistency with Redis.

        Returns:
            True if consistent
        """
        local_checksum = self.get_checksum()

        # Get remote state and calculate checksum
        remote_state = {}
        keys = await self._redis.keys(f"{self._config.state_prefix}*")

        for key in keys:
            state_key = key[len(self._config.state_prefix) :]
            value = await self._redis.get(f"{self._config.state_prefix}{state_key}")
            if value is not None:
                remote_state[state_key] = value

        remote_str = json.dumps(remote_state, sort_keys=True, default=str)
        remote_checksum = hashlib.sha256(remote_str.encode()).hexdigest()[:16]

        consistent = local_checksum == remote_checksum

        if not consistent:
            logger.warning(
                f"State inconsistency detected: "
                f"local={local_checksum}, remote={remote_checksum}"
            )

        return consistent
