"""
Tests for Infrastructure Module.

Tests for leader election, state sync, snapshot, secrets, and config versioning.
"""

import asyncio
import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.infrastructure import (
    # Leader Election
    LeaderElection,
    LeaderElectionConfig,
    NodeInfo,
    NodeRole,
    NodeState,
    # State Sync
    StateManager,
    StateSyncConfig,
    StateChange,
    StateVersion,
    SyncStrategy,
    ConflictResolution,
    # Snapshot
    SnapshotManager,
    BackupConfig,
    SnapshotMetadata,
    SnapshotType,
    SnapshotStatus,
    # Secrets
    SecretsManager,
    SecretsConfig,
    SecretMetadata,
    SecretType,
    RotationPolicy,
    # Config Versioning
    ConfigVersionManager,
    VersioningConfig,
    ConfigChange,
    ConfigSnapshot,
    ChangeType,
    ApprovalStatus,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_redis():
    """Create mock Redis client."""
    redis = MagicMock()
    redis.client = AsyncMock()
    redis._make_key = lambda x: f"test:{x}"
    redis.get = AsyncMock(return_value=None)
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=True)
    redis.expire = AsyncMock(return_value=True)
    redis.hset = AsyncMock(return_value=True)
    redis.hget = AsyncMock(return_value=None)
    redis.hdel = AsyncMock(return_value=True)
    redis.hgetall = AsyncMock(return_value={})
    redis.keys = AsyncMock(return_value=[])
    redis.publish = AsyncMock(return_value=True)
    redis.subscribe = AsyncMock(return_value=True)
    redis.unsubscribe = AsyncMock(return_value=True)
    return redis


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# =============================================================================
# Leader Election Tests
# =============================================================================


class TestLeaderElection:
    """Tests for LeaderElection class."""

    def test_init(self, mock_redis):
        """Test initialization."""
        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
            host="localhost",
            port=8000,
        )

        assert election.node_id == "test-node"
        assert election.role == NodeRole.OFFLINE
        assert election.state == NodeState.STOPPED
        assert not election.is_leader

    def test_init_with_config(self, mock_redis):
        """Test initialization with custom config."""
        config = LeaderElectionConfig(
            lock_key="custom:leader",
            lock_ttl_seconds=60,
            heartbeat_interval=10.0,
        )

        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
            config=config,
        )

        assert election._config.lock_key == "custom:leader"
        assert election._config.lock_ttl_seconds == 60

    @pytest.mark.asyncio
    async def test_start_stop(self, mock_redis):
        """Test start and stop lifecycle."""
        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        await election.start()
        assert election._running
        assert election.state == NodeState.RUNNING
        assert election.role in [NodeRole.CANDIDATE, NodeRole.STANDBY]

        await election.stop()
        assert not election._running
        assert election.state == NodeState.STOPPED
        assert election.role == NodeRole.OFFLINE

    @pytest.mark.asyncio
    async def test_acquire_lock_success(self, mock_redis):
        """Test successful lock acquisition."""
        mock_redis.client.set = AsyncMock(return_value=True)

        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        result = await election._try_acquire_lock()
        assert result is True
        mock_redis.client.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_acquire_lock_failure(self, mock_redis):
        """Test failed lock acquisition (already held)."""
        mock_redis.client.set = AsyncMock(return_value=None)

        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        result = await election._try_acquire_lock()
        assert result is False

    @pytest.mark.asyncio
    async def test_become_leader_callback(self, mock_redis):
        """Test become leader callback is called."""
        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        callback_called = []

        def on_become_leader():
            callback_called.append(True)

        election.on_become_leader(on_become_leader)
        await election._become_leader()

        assert election.is_leader
        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_lose_leadership_callback(self, mock_redis):
        """Test lose leadership callback is called."""
        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        callback_called = []

        def on_lose_leader():
            callback_called.append(True)

        election.on_lose_leadership(on_lose_leader)

        # First become leader
        await election._become_leader()
        # Then lose leadership
        await election._lose_leadership()

        assert not election.is_leader
        assert len(callback_called) == 1

    @pytest.mark.asyncio
    async def test_step_down(self, mock_redis):
        """Test voluntary step down."""
        mock_redis.get = AsyncMock(return_value="test-node:2024-01-01T00:00:00")

        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        await election._become_leader()
        assert election.is_leader

        await election.step_down()
        assert not election.is_leader

    def test_node_info_serialization(self):
        """Test NodeInfo serialization."""
        now = datetime.now(timezone.utc)
        node = NodeInfo(
            node_id="test-node",
            role=NodeRole.LEADER,
            state=NodeState.RUNNING,
            host="localhost",
            port=8000,
            started_at=now,
            last_heartbeat=now,
            metadata={"version": "1.0"},
        )

        data = node.to_dict()
        restored = NodeInfo.from_dict(data)

        assert restored.node_id == node.node_id
        assert restored.role == node.role
        assert restored.state == node.state
        assert restored.metadata == node.metadata


# =============================================================================
# State Sync Tests
# =============================================================================


class TestStateSync:
    """Tests for StateManager class."""

    def test_init(self, mock_redis):
        """Test initialization."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: False,
        )

        assert manager._node_id == "test-node"
        assert not manager.is_leader

    @pytest.mark.asyncio
    async def test_set_get(self, mock_redis):
        """Test basic set and get operations."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        await manager.set("key1", "value1")
        result = await manager.get("key1")

        assert result == "value1"

    @pytest.mark.asyncio
    async def test_delete(self, mock_redis):
        """Test delete operation."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        await manager.set("key1", "value1")
        await manager.delete("key1")
        result = await manager.get("key1")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_all(self, mock_redis):
        """Test get all operation."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        await manager.set("key1", "value1")
        await manager.set("key2", "value2")

        all_state = await manager.get_all()

        assert len(all_state) == 2
        assert all_state["key1"] == "value1"
        assert all_state["key2"] == "value2"

    @pytest.mark.asyncio
    async def test_get_keys_pattern(self, mock_redis):
        """Test get keys with pattern."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        await manager.set("user:1", "a")
        await manager.set("user:2", "b")
        await manager.set("order:1", "c")

        user_keys = await manager.get_keys("user:*")

        assert len(user_keys) == 2
        assert "user:1" in user_keys
        assert "user:2" in user_keys

    @pytest.mark.asyncio
    async def test_on_change_callback(self, mock_redis):
        """Test state change callback."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        changes = []

        def on_change(key, old_val, new_val):
            changes.append((key, old_val, new_val))

        manager.on_change(on_change)

        await manager.set("key1", "value1")
        await manager.set("key1", "value2")

        assert len(changes) == 2
        assert changes[0] == ("key1", None, "value1")
        assert changes[1] == ("key1", "value1", "value2")

    def test_get_checksum(self, mock_redis):
        """Test state checksum calculation."""
        manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        # Empty state
        checksum1 = manager.get_checksum()

        # Set local state directly
        manager._local_state["key1"] = "value1"
        checksum2 = manager.get_checksum()

        # Checksums should differ
        assert checksum1 != checksum2

    def test_state_change_serialization(self):
        """Test StateChange serialization."""
        now = datetime.now(timezone.utc)
        change = StateChange(
            key="test-key",
            value="test-value",
            timestamp=now,
            node_id="test-node",
            version=1,
            operation="set",
        )

        data = change.to_dict()
        restored = StateChange.from_dict(data)

        assert restored.key == change.key
        assert restored.value == change.value
        assert restored.version == change.version
        assert restored.operation == change.operation


# =============================================================================
# Snapshot Tests
# =============================================================================


class TestSnapshot:
    """Tests for SnapshotManager class."""

    def test_init(self, temp_dir):
        """Test initialization."""
        config = BackupConfig(backup_dir=temp_dir / "backups")

        manager = SnapshotManager(
            state_provider=lambda: {},
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config,
        )

        assert manager._node_id == "test-node"
        assert manager._config.backup_dir.exists()

    @pytest.mark.asyncio
    async def test_create_full_snapshot(self, temp_dir):
        """Test creating full snapshot."""
        config = BackupConfig(backup_dir=temp_dir / "backups")
        test_state = {"key1": "value1", "key2": 123}

        manager = SnapshotManager(
            state_provider=lambda: test_state,
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config,
        )

        metadata = await manager.create_snapshot(
            snapshot_type=SnapshotType.FULL,
            description="Test snapshot",
            tags={"env": "test"},
        )

        assert metadata.status == SnapshotStatus.COMPLETED
        assert metadata.snapshot_type == SnapshotType.FULL
        assert metadata.entry_count == 2
        assert metadata.size_bytes > 0
        assert metadata.checksum != ""
        assert metadata.description == "Test snapshot"

    @pytest.mark.asyncio
    async def test_restore_snapshot(self, temp_dir):
        """Test restoring from snapshot."""
        config = BackupConfig(backup_dir=temp_dir / "backups")
        original_state = {"key1": "value1", "key2": 123}
        restored_state = {}

        def state_restorer(state):
            restored_state.clear()
            restored_state.update(state)

        manager = SnapshotManager(
            state_provider=lambda: original_state,
            state_restorer=state_restorer,
            node_id="test-node",
            config=config,
        )

        # Create snapshot
        metadata = await manager.create_snapshot(snapshot_type=SnapshotType.FULL)

        # Restore snapshot
        success = await manager.restore_snapshot(metadata.snapshot_id)

        assert success
        assert restored_state == original_state

    @pytest.mark.asyncio
    async def test_list_snapshots(self, temp_dir):
        """Test listing snapshots."""
        config = BackupConfig(backup_dir=temp_dir / "backups")

        manager = SnapshotManager(
            state_provider=lambda: {"key": "value"},
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config,
        )

        # Create multiple snapshots with delay to ensure unique IDs
        await manager.create_snapshot(snapshot_type=SnapshotType.FULL)
        await asyncio.sleep(1.1)  # Ensure different timestamps (>1 second)
        await manager.create_snapshot(snapshot_type=SnapshotType.INCREMENTAL)

        # List all
        all_snapshots = await manager.list_snapshots()
        assert len(all_snapshots) == 2

        # Filter by type
        full_only = await manager.list_snapshots(snapshot_type=SnapshotType.FULL)
        assert len(full_only) == 1
        assert full_only[0].snapshot_type == SnapshotType.FULL

    @pytest.mark.asyncio
    async def test_delete_snapshot(self, temp_dir):
        """Test deleting snapshot."""
        config = BackupConfig(backup_dir=temp_dir / "backups")

        manager = SnapshotManager(
            state_provider=lambda: {"key": "value"},
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config,
        )

        metadata = await manager.create_snapshot(snapshot_type=SnapshotType.FULL)

        # Delete
        success = await manager.delete_snapshot(metadata.snapshot_id)
        assert success

        # Verify deleted
        snapshots = await manager.list_snapshots()
        assert len(snapshots) == 0

    @pytest.mark.asyncio
    async def test_snapshot_compression(self, temp_dir):
        """Test snapshot compression."""
        config_compressed = BackupConfig(
            backup_dir=temp_dir / "backups_compressed",
            compress=True,
        )
        config_uncompressed = BackupConfig(
            backup_dir=temp_dir / "backups_uncompressed",
            compress=False,
        )

        large_state = {"key" + str(i): "value" * 100 for i in range(100)}

        manager_compressed = SnapshotManager(
            state_provider=lambda: large_state,
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config_compressed,
        )

        manager_uncompressed = SnapshotManager(
            state_provider=lambda: large_state,
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config_uncompressed,
        )

        meta_compressed = await manager_compressed.create_snapshot()
        meta_uncompressed = await manager_uncompressed.create_snapshot()

        # Compressed should be smaller
        assert meta_compressed.size_bytes < meta_uncompressed.size_bytes

    def test_snapshot_metadata_serialization(self):
        """Test SnapshotMetadata serialization."""
        now = datetime.now(timezone.utc)
        metadata = SnapshotMetadata(
            snapshot_id="test-snapshot",
            snapshot_type=SnapshotType.FULL,
            status=SnapshotStatus.COMPLETED,
            created_at=now,
            completed_at=now,
            size_bytes=1024,
            checksum="abc123",
            entry_count=10,
            tags={"env": "test"},
        )

        data = metadata.to_dict()
        restored = SnapshotMetadata.from_dict(data)

        assert restored.snapshot_id == metadata.snapshot_id
        assert restored.snapshot_type == metadata.snapshot_type
        assert restored.status == metadata.status
        assert restored.size_bytes == metadata.size_bytes


# =============================================================================
# Secrets Tests
# =============================================================================


class TestSecrets:
    """Tests for SecretsManager class."""

    def test_init_with_master_key(self, temp_dir):
        """Test initialization with master key."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        assert manager._fernet is not None

    def test_init_without_master_key_encrypted(self, temp_dir):
        """Test initialization fails without master key when encrypted."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            encrypted=True,
        )

        with pytest.raises(ValueError, match="Master key required"):
            SecretsManager(master_key=None, config=config)

    def test_init_without_master_key_unencrypted(self, temp_dir):
        """Test initialization succeeds without master key when not encrypted."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            encrypted=False,
            enable_audit_log=False,
        )

        manager = SecretsManager(master_key=None, config=config)
        assert manager._fernet is None

    @pytest.mark.asyncio
    async def test_store_and_get_secret(self, temp_dir):
        """Test storing and retrieving secret."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        # Store secret
        metadata = await manager.store_secret(
            secret_id="api-key",
            value="super-secret-value",
            secret_type=SecretType.API_KEY,
            description="Test API key",
        )

        assert metadata.secret_id == "api-key"
        assert metadata.secret_type == SecretType.API_KEY
        assert metadata.version == 1

        # Retrieve secret
        value = await manager.get_secret("api-key")
        assert value == "super-secret-value"

    @pytest.mark.asyncio
    async def test_delete_secret(self, temp_dir):
        """Test deleting secret."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        await manager.store_secret("test-secret", "value")

        # Delete
        success = await manager.delete_secret("test-secret")
        assert success

        # Verify deleted
        value = await manager.get_secret("test-secret")
        assert value is None

    @pytest.mark.asyncio
    async def test_rotate_secret(self, temp_dir):
        """Test rotating secret."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        # Store initial secret
        await manager.store_secret("rotating-secret", "old-value")

        # Rotate
        new_metadata = await manager.rotate_secret(
            "rotating-secret", "new-value"
        )

        assert new_metadata.version == 2

        # Verify new value
        value = await manager.get_secret("rotating-secret")
        assert value == "new-value"

    @pytest.mark.asyncio
    async def test_list_secrets(self, temp_dir):
        """Test listing secrets."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        await manager.store_secret(
            "api-key-1", "value1", secret_type=SecretType.API_KEY
        )
        await manager.store_secret(
            "password-1", "value2", secret_type=SecretType.PASSWORD
        )

        # List all
        all_secrets = await manager.list_secrets()
        assert len(all_secrets) == 2

        # Filter by type
        api_keys = await manager.list_secrets(secret_type=SecretType.API_KEY)
        assert len(api_keys) == 1

    @pytest.mark.asyncio
    async def test_secret_expiration(self, temp_dir):
        """Test expired secrets return None."""
        config = SecretsConfig(
            secrets_dir=temp_dir / "secrets",
            audit_log_file=temp_dir / "logs" / "audit.log",
        )

        manager = SecretsManager(
            master_key="test-master-key-12345",
            config=config,
        )

        # Store expired secret
        past = datetime.now(timezone.utc) - timedelta(days=1)
        await manager.store_secret(
            "expired-secret",
            "value",
            expires_at=past,
        )

        # Should return None
        value = await manager.get_secret("expired-secret")
        assert value is None

    def test_generate_key(self):
        """Test key generation."""
        key1 = SecretsManager.generate_key(32)
        key2 = SecretsManager.generate_key(32)

        assert key1 != key2
        assert len(key1) > 0

    def test_generate_api_key(self):
        """Test API key generation."""
        key1 = SecretsManager.generate_api_key()
        key2 = SecretsManager.generate_api_key()

        assert key1 != key2
        assert len(key1) > 0

    def test_hash_and_verify(self):
        """Test hashing and verification."""
        value = "my-secret-password"

        hash_val, salt = SecretsManager.hash_secret(value)

        assert SecretsManager.verify_hash(value, hash_val, salt)
        assert not SecretsManager.verify_hash("wrong-password", hash_val, salt)

    def test_secret_metadata_serialization(self):
        """Test SecretMetadata serialization."""
        now = datetime.now(timezone.utc)
        metadata = SecretMetadata(
            secret_id="test-secret",
            secret_type=SecretType.API_KEY,
            created_at=now,
            updated_at=now,
            rotation_policy=RotationPolicy.MONTHLY,
            version=2,
        )

        data = metadata.to_dict()
        restored = SecretMetadata.from_dict(data)

        assert restored.secret_id == metadata.secret_id
        assert restored.secret_type == metadata.secret_type
        assert restored.rotation_policy == metadata.rotation_policy
        assert restored.version == metadata.version


# =============================================================================
# Config Versioning Tests
# =============================================================================


class TestConfigVersioning:
    """Tests for ConfigVersionManager class."""

    def test_init(self, temp_dir):
        """Test initialization."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )

        manager = ConfigVersionManager(
            config_provider=lambda: {},
            config_applier=lambda x: True,
            config=config,
        )

        assert manager._config.history_dir.exists()

    @pytest.mark.asyncio
    async def test_record_change(self, temp_dir):
        """Test recording configuration change."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )
        current_config = {"setting1": "old_value"}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        change = await manager.record_change(
            path="setting1",
            new_value="new_value",
            author="test-user",
            description="Updated setting1",
        )

        assert change.version == 1
        assert change.change_type == ChangeType.UPDATE
        assert change.author == "test-user"
        assert change.path == "setting1"

    @pytest.mark.asyncio
    async def test_record_change_create(self, temp_dir):
        """Test recording a create change."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )
        current_config = {}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        change = await manager.record_change(
            path="new_setting",
            new_value="value",
            author="test-user",
            description="Created new setting",
        )

        assert change.change_type == ChangeType.CREATE

    @pytest.mark.asyncio
    async def test_record_change_delete(self, temp_dir):
        """Test recording a delete change."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )
        current_config = {"old_setting": "value"}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        change = await manager.record_change(
            path="old_setting",
            new_value=None,
            author="test-user",
            description="Deleted old setting",
        )

        assert change.change_type == ChangeType.DELETE

    @pytest.mark.asyncio
    async def test_rollback(self, temp_dir):
        """Test config rollback."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )

        current_config = {"setting1": "initial"}

        def config_provider():
            return current_config.copy()

        def config_applier(cfg):
            current_config.clear()
            current_config.update(cfg)
            return True

        manager = ConfigVersionManager(
            config_provider=config_provider,
            config_applier=config_applier,
            config=config,
        )

        # Record first change (creates v1 snapshot)
        await manager.record_change(
            path="setting1",
            new_value="initial",
            author="test",
            description="v1",
        )

        # Record second change (creates v2 snapshot)
        current_config["setting1"] = "changed"
        await manager.record_change(
            path="setting1",
            new_value="changed",
            author="test",
            description="v2",
        )

        # Rollback to v1
        success = await manager.rollback_to_version(1, author="test")

        assert success
        assert current_config["setting1"] == "initial"

    @pytest.mark.asyncio
    async def test_get_changes(self, temp_dir):
        """Test getting change history."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )
        current_config = {}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        # Record multiple changes
        await manager.record_change(
            path="setting1",
            new_value="value1",
            author="user1",
            description="First",
        )
        await manager.record_change(
            path="setting2",
            new_value="value2",
            author="user2",
            description="Second",
        )

        history = manager.get_changes(limit=10)

        assert len(history) == 2
        # Should be in reverse order (newest first)
        assert history[0].description == "Second"
        assert history[1].description == "First"

    @pytest.mark.asyncio
    async def test_diff_versions(self, temp_dir):
        """Test comparing config versions."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
        )

        current_config = {"setting1": "a", "setting2": 1}

        def config_provider():
            return current_config.copy()

        manager = ConfigVersionManager(
            config_provider=config_provider,
            config_applier=lambda x: True,
            config=config,
        )

        # Create first version
        await manager.record_change(
            path="setting1",
            new_value="a",
            author="test",
            description="v1",
        )

        # Update config and create second version
        current_config["setting1"] = "b"
        current_config["setting3"] = "new"
        await manager.record_change(
            path="setting1",
            new_value="b",
            author="test",
            description="v2",
        )

        # Compare
        diff = manager.diff_versions(1, 2)

        assert diff is not None
        assert "setting1" in diff

    @pytest.mark.asyncio
    async def test_approval_workflow(self, temp_dir):
        """Test change approval workflow."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
            require_approval=True,
        )
        current_config = {}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        # Record change with pending approval (non-system author)
        change = await manager.record_change(
            path="setting1",
            new_value="value",
            author="test-user",  # Not in auto_approve_authors
            description="Needs approval",
        )

        assert change.approval_status == ApprovalStatus.PENDING

        # Approve change
        success = await manager.approve_change(
            change_id=change.change_id,
            approver="admin",
        )

        assert success

        # Verify approved
        updated_change = manager.get_change(change.change_id)
        assert updated_change.approval_status == ApprovalStatus.APPROVED
        assert updated_change.approved_by == "admin"

    @pytest.mark.asyncio
    async def test_reject_change(self, temp_dir):
        """Test change rejection workflow."""
        config = VersioningConfig(
            history_dir=temp_dir / "config_history",
            require_approval=True,
        )
        current_config = {}

        manager = ConfigVersionManager(
            config_provider=lambda: current_config,
            config_applier=lambda x: True,
            config=config,
        )

        # Record change with pending approval
        change = await manager.record_change(
            path="setting1",
            new_value="value",
            author="test-user",
            description="Will be rejected",
        )

        assert change.approval_status == ApprovalStatus.PENDING

        # Reject change
        success = await manager.reject_change(
            change_id=change.change_id,
            rejector="admin",
            reason="Not approved",
        )

        assert success

        # Verify rejected
        updated_change = manager.get_change(change.change_id)
        assert updated_change.approval_status == ApprovalStatus.REJECTED

    def test_config_change_serialization(self):
        """Test ConfigChange serialization."""
        now = datetime.now(timezone.utc)
        change = ConfigChange(
            change_id="change_1234_1",
            version=1,
            change_type=ChangeType.UPDATE,
            timestamp=now,
            author="test-user",
            description="Test change",
            path="settings.key",
            old_value="old",
            new_value="new",
        )

        data = change.to_dict()
        restored = ConfigChange.from_dict(data)

        assert restored.change_id == change.change_id
        assert restored.version == change.version
        assert restored.change_type == change.change_type
        assert restored.author == change.author
        assert restored.path == change.path

    def test_config_snapshot_serialization(self):
        """Test ConfigSnapshot serialization."""
        now = datetime.now(timezone.utc)
        snapshot = ConfigSnapshot(
            version=1,
            config={"setting": "value"},
            timestamp=now,
            description="Test snapshot",
            author="test-user",
            checksum="abc123",
        )

        data = snapshot.to_dict()
        restored = ConfigSnapshot.from_dict(data)

        assert restored.version == snapshot.version
        assert restored.config == snapshot.config
        assert restored.checksum == snapshot.checksum


# =============================================================================
# Integration Tests
# =============================================================================


class TestInfrastructureIntegration:
    """Integration tests for infrastructure components."""

    @pytest.mark.asyncio
    async def test_leader_election_with_state_sync(self, mock_redis):
        """Test leader election triggers state sync."""
        # Create leader election
        election = LeaderElection(
            redis_client=mock_redis,
            node_id="test-node",
        )

        # Create state manager
        state_manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: election.is_leader,
        )

        syncs = []

        def on_become_leader():
            syncs.append("leader")

        election.on_become_leader(on_become_leader)

        # Become leader
        await election._become_leader()

        assert election.is_leader
        assert len(syncs) == 1

    @pytest.mark.asyncio
    async def test_snapshot_with_state_manager(self, mock_redis, temp_dir):
        """Test snapshot works with state manager."""
        # Create state manager
        state_manager = StateManager(
            redis_client=mock_redis,
            node_id="test-node",
            is_leader_callback=lambda: True,
        )

        # Add some state
        await state_manager.set("key1", "value1")
        await state_manager.set("key2", "value2")

        # Create snapshot manager using state manager's state
        config = BackupConfig(backup_dir=temp_dir / "backups")
        snapshot_manager = SnapshotManager(
            state_provider=lambda: state_manager._local_state,
            state_restorer=lambda x: None,
            node_id="test-node",
            config=config,
        )

        # Create snapshot
        metadata = await snapshot_manager.create_snapshot()

        assert metadata.entry_count == 2
        assert metadata.status == SnapshotStatus.COMPLETED


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
