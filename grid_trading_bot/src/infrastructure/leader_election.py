"""
Leader Election and Active-Standby Failover.

Provides distributed leader election using Redis for
automatic failover in multi-node deployments.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from src.core import get_logger

logger = get_logger(__name__)


class NodeRole(Enum):
    """Node role in the cluster."""

    LEADER = "leader"
    STANDBY = "standby"
    CANDIDATE = "candidate"
    OFFLINE = "offline"


class NodeState(Enum):
    """Node operational state."""

    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"


@dataclass
class NodeInfo:
    """Information about a cluster node."""

    node_id: str
    role: NodeRole
    state: NodeState
    host: str
    port: int
    started_at: datetime
    last_heartbeat: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "node_id": self.node_id,
            "role": self.role.value,
            "state": self.state.value,
            "host": self.host,
            "port": self.port,
            "started_at": self.started_at.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NodeInfo":
        """Create from dictionary."""
        return cls(
            node_id=data["node_id"],
            role=NodeRole(data["role"]),
            state=NodeState(data["state"]),
            host=data["host"],
            port=data["port"],
            started_at=datetime.fromisoformat(data["started_at"]),
            last_heartbeat=datetime.fromisoformat(data["last_heartbeat"]),
            metadata=data.get("metadata", {}),
        )


@dataclass
class LeaderElectionConfig:
    """Leader election configuration."""

    # Lock settings
    lock_key: str = "cluster:leader"
    lock_ttl_seconds: int = 30
    lock_renewal_interval: float = 10.0

    # Heartbeat settings
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 15.0

    # Election settings
    election_timeout: float = 5.0
    max_election_attempts: int = 3

    # Node registration
    nodes_key: str = "cluster:nodes"
    node_ttl_seconds: int = 60


class LeaderElection:
    """
    Distributed leader election using Redis.

    Uses Redis locks for leader election with automatic
    failover when the leader becomes unavailable.
    """

    def __init__(
        self,
        redis_client: Any,  # RedisManager
        node_id: Optional[str] = None,
        host: str = "localhost",
        port: int = 8000,
        config: Optional[LeaderElectionConfig] = None,
    ):
        """
        Initialize leader election.

        Args:
            redis_client: Redis client instance
            node_id: Unique node identifier (auto-generated if not provided)
            host: Node host address
            port: Node port
            config: Election configuration
        """
        self._redis = redis_client
        self._node_id = node_id or str(uuid.uuid4())[:8]
        self._host = host
        self._port = port
        self._config = config or LeaderElectionConfig()

        # State
        self._role = NodeRole.OFFLINE
        self._state = NodeState.STOPPED
        self._running = False
        self._lock_held = False

        # Tasks
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._election_task: Optional[asyncio.Task] = None
        self._lock_renewal_task: Optional[asyncio.Task] = None

        # Callbacks
        self._on_become_leader: List[Callable[[], Any]] = []
        self._on_lose_leadership: List[Callable[[], Any]] = []
        self._on_leader_change: List[Callable[[Optional[str], Optional[str]], Any]] = []

        # Current leader tracking
        self._current_leader: Optional[str] = None

    @property
    def node_id(self) -> str:
        """Get node ID."""
        return self._node_id

    @property
    def role(self) -> NodeRole:
        """Get current role."""
        return self._role

    @property
    def state(self) -> NodeState:
        """Get current state."""
        return self._state

    @property
    def is_leader(self) -> bool:
        """Check if this node is the leader."""
        return self._role == NodeRole.LEADER and self._lock_held

    @property
    def current_leader(self) -> Optional[str]:
        """Get current leader node ID."""
        return self._current_leader

    async def start(self) -> None:
        """Start leader election."""
        if self._running:
            return

        self._running = True
        self._state = NodeState.STARTING

        # Register this node
        await self._register_node()

        # Start election
        self._role = NodeRole.CANDIDATE
        self._state = NodeState.RUNNING

        # Start background tasks
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        self._election_task = asyncio.create_task(self._election_loop())

        logger.info(f"Node {self._node_id} started leader election")

    async def stop(self) -> None:
        """Stop leader election."""
        if not self._running:
            return

        self._running = False
        self._state = NodeState.DRAINING

        # Release leadership if held
        if self._lock_held:
            await self._release_lock()

        # Cancel tasks
        for task in [
            self._heartbeat_task,
            self._election_task,
            self._lock_renewal_task,
        ]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Unregister node
        await self._unregister_node()

        self._role = NodeRole.OFFLINE
        self._state = NodeState.STOPPED

        logger.info(f"Node {self._node_id} stopped")

    async def _register_node(self) -> None:
        """Register this node in the cluster."""
        node_info = NodeInfo(
            node_id=self._node_id,
            role=self._role,
            state=self._state,
            host=self._host,
            port=self._port,
            started_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
        )

        await self._redis.hset(
            self._config.nodes_key,
            self._node_id,
            node_info.to_dict(),
        )

        logger.debug(f"Node {self._node_id} registered")

    async def _unregister_node(self) -> None:
        """Unregister this node from the cluster."""
        await self._redis.hdel(self._config.nodes_key, self._node_id)
        logger.debug(f"Node {self._node_id} unregistered")

    async def _update_heartbeat(self) -> None:
        """Update node heartbeat."""
        node_info = NodeInfo(
            node_id=self._node_id,
            role=self._role,
            state=self._state,
            host=self._host,
            port=self._port,
            started_at=datetime.now(timezone.utc),
            last_heartbeat=datetime.now(timezone.utc),
        )

        await self._redis.hset(
            self._config.nodes_key,
            self._node_id,
            node_info.to_dict(),
        )

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats."""
        while self._running:
            try:
                await self._update_heartbeat()
                await self._check_leader_health()
                await asyncio.sleep(self._config.heartbeat_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(1.0)

    async def _election_loop(self) -> None:
        """Attempt to acquire leadership."""
        while self._running:
            try:
                if not self._lock_held:
                    # Try to acquire leadership
                    acquired = await self._try_acquire_lock()

                    if acquired:
                        await self._become_leader()
                    else:
                        # Update current leader
                        leader = await self._get_current_leader()
                        if leader != self._current_leader:
                            old_leader = self._current_leader
                            self._current_leader = leader
                            await self._notify_leader_change(old_leader, leader)

                        if self._role != NodeRole.STANDBY:
                            self._role = NodeRole.STANDBY
                            logger.info(
                                f"Node {self._node_id} is standby, "
                                f"leader is {leader}"
                            )

                await asyncio.sleep(self._config.election_timeout)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Election error: {e}")
                await asyncio.sleep(1.0)

    async def _try_acquire_lock(self) -> bool:
        """
        Try to acquire the leader lock.

        Returns:
            True if lock acquired
        """
        if not self._redis.client:
            return False

        # Use SET NX EX for atomic lock acquisition
        lock_value = f"{self._node_id}:{datetime.now(timezone.utc).isoformat()}"

        result = await self._redis.client.set(
            self._redis._make_key(self._config.lock_key),
            lock_value,
            nx=True,  # Only set if not exists
            ex=self._config.lock_ttl_seconds,
        )

        return result is not None

    async def _release_lock(self) -> None:
        """Release the leader lock."""
        if not self._redis.client:
            return

        # Only release if we hold the lock
        current = await self._redis.get(self._config.lock_key)
        if current and current.startswith(f"{self._node_id}:"):
            await self._redis.delete(self._config.lock_key)
            self._lock_held = False
            logger.info(f"Node {self._node_id} released leadership")

    async def _renew_lock(self) -> bool:
        """
        Renew the leader lock.

        Returns:
            True if renewal successful
        """
        if not self._redis.client:
            return False

        # Check if we still hold the lock
        current = await self._redis.get(self._config.lock_key)
        if not current or not current.startswith(f"{self._node_id}:"):
            return False

        # Renew TTL
        result = await self._redis.expire(
            self._config.lock_key,
            self._config.lock_ttl_seconds,
        )

        return result

    async def _lock_renewal_loop(self) -> None:
        """Periodically renew the lock."""
        while self._running and self._lock_held:
            try:
                renewed = await self._renew_lock()

                if not renewed:
                    # Lost the lock
                    await self._lose_leadership()
                    break

                await asyncio.sleep(self._config.lock_renewal_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Lock renewal error: {e}")
                await self._lose_leadership()
                break

    async def _become_leader(self) -> None:
        """Handle becoming the leader."""
        self._lock_held = True
        self._role = NodeRole.LEADER
        self._current_leader = self._node_id

        # Start lock renewal
        self._lock_renewal_task = asyncio.create_task(self._lock_renewal_loop())

        logger.info(f"Node {self._node_id} became leader")

        # Notify callbacks
        for callback in self._on_become_leader:
            try:
                result = callback()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Become leader callback error: {e}")

    async def _lose_leadership(self) -> None:
        """Handle losing leadership."""
        was_leader = self._lock_held
        self._lock_held = False
        self._role = NodeRole.STANDBY

        if self._lock_renewal_task:
            self._lock_renewal_task.cancel()
            try:
                await self._lock_renewal_task
            except asyncio.CancelledError:
                pass

        if was_leader:
            logger.warning(f"Node {self._node_id} lost leadership")

            # Notify callbacks
            for callback in self._on_lose_leadership:
                try:
                    result = callback()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"Lose leadership callback error: {e}")

    async def _get_current_leader(self) -> Optional[str]:
        """Get the current leader node ID."""
        lock_value = await self._redis.get(self._config.lock_key)
        if lock_value and ":" in lock_value:
            return lock_value.split(":")[0]
        return None

    async def _check_leader_health(self) -> None:
        """Check if the leader is healthy."""
        leader = await self._get_current_leader()

        if leader and leader != self._node_id:
            # Check if leader has recent heartbeat
            nodes = await self.get_cluster_nodes()
            leader_node = nodes.get(leader)

            if leader_node:
                elapsed = (
                    datetime.now(timezone.utc) - leader_node.last_heartbeat
                ).total_seconds()

                if elapsed > self._config.heartbeat_timeout:
                    logger.warning(
                        f"Leader {leader} heartbeat timeout "
                        f"({elapsed:.1f}s > {self._config.heartbeat_timeout}s)"
                    )

    async def _notify_leader_change(
        self,
        old_leader: Optional[str],
        new_leader: Optional[str],
    ) -> None:
        """Notify callbacks of leader change."""
        for callback in self._on_leader_change:
            try:
                result = callback(old_leader, new_leader)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Leader change callback error: {e}")

    async def get_cluster_nodes(self) -> Dict[str, NodeInfo]:
        """
        Get all registered cluster nodes.

        Returns:
            Dict of node_id to NodeInfo
        """
        data = await self._redis.hgetall(self._config.nodes_key)

        nodes = {}
        for node_id, info in data.items():
            try:
                if isinstance(info, dict):
                    nodes[node_id] = NodeInfo.from_dict(info)
            except Exception as e:
                logger.error(f"Failed to parse node {node_id}: {e}")

        return nodes

    async def get_leader_info(self) -> Optional[NodeInfo]:
        """
        Get information about the current leader.

        Returns:
            Leader NodeInfo or None
        """
        leader_id = await self._get_current_leader()
        if not leader_id:
            return None

        nodes = await self.get_cluster_nodes()
        return nodes.get(leader_id)

    def on_become_leader(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """
        Register callback for when this node becomes leader.

        Args:
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_become_leader.append(callback)

        def unsubscribe():
            if callback in self._on_become_leader:
                self._on_become_leader.remove(callback)

        return unsubscribe

    def on_lose_leadership(self, callback: Callable[[], Any]) -> Callable[[], None]:
        """
        Register callback for when this node loses leadership.

        Args:
            callback: Callback function

        Returns:
            Unsubscribe function
        """
        self._on_lose_leadership.append(callback)

        def unsubscribe():
            if callback in self._on_lose_leadership:
                self._on_lose_leadership.remove(callback)

        return unsubscribe

    def on_leader_change(
        self,
        callback: Callable[[Optional[str], Optional[str]], Any],
    ) -> Callable[[], None]:
        """
        Register callback for any leader change.

        Args:
            callback: Callback function(old_leader, new_leader)

        Returns:
            Unsubscribe function
        """
        self._on_leader_change.append(callback)

        def unsubscribe():
            if callback in self._on_leader_change:
                self._on_leader_change.remove(callback)

        return unsubscribe

    async def step_down(self) -> None:
        """Voluntarily step down from leadership."""
        if self._lock_held:
            logger.info(f"Node {self._node_id} stepping down from leadership")
            await self._release_lock()
            await self._lose_leadership()

    async def force_election(self) -> None:
        """Force a new election by releasing the current lock."""
        if self._lock_held:
            await self.step_down()
