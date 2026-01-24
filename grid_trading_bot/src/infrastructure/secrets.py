"""
Secrets and Key Management Service.

Provides secure storage, encryption, and rotation
for sensitive credentials and API keys.
"""

import base64
import hashlib
import hmac
import json
import os
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from src.core import get_logger

logger = get_logger(__name__)


class SecretType(Enum):
    """Type of secret."""

    API_KEY = "api_key"
    API_SECRET = "api_secret"
    PASSWORD = "password"
    TOKEN = "token"
    CERTIFICATE = "certificate"
    ENCRYPTION_KEY = "encryption_key"
    GENERIC = "generic"


class RotationPolicy(Enum):
    """Key rotation policy."""

    NEVER = "never"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


@dataclass
class SecretMetadata:
    """Metadata for a secret."""

    secret_id: str
    secret_type: SecretType
    created_at: datetime
    updated_at: datetime
    expires_at: Optional[datetime] = None
    rotation_policy: RotationPolicy = RotationPolicy.NEVER
    last_rotated: Optional[datetime] = None
    version: int = 1
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    access_count: int = 0
    last_accessed: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "secret_id": self.secret_id,
            "secret_type": self.secret_type.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "rotation_policy": self.rotation_policy.value,
            "last_rotated": (
                self.last_rotated.isoformat() if self.last_rotated else None
            ),
            "version": self.version,
            "description": self.description,
            "tags": self.tags,
            "access_count": self.access_count,
            "last_accessed": (
                self.last_accessed.isoformat() if self.last_accessed else None
            ),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SecretMetadata":
        """Create from dictionary."""
        return cls(
            secret_id=data["secret_id"],
            secret_type=SecretType(data["secret_type"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            rotation_policy=RotationPolicy(
                data.get("rotation_policy", RotationPolicy.NEVER.value)
            ),
            last_rotated=(
                datetime.fromisoformat(data["last_rotated"])
                if data.get("last_rotated")
                else None
            ),
            version=data.get("version", 1),
            description=data.get("description", ""),
            tags=data.get("tags", {}),
            access_count=data.get("access_count", 0),
            last_accessed=(
                datetime.fromisoformat(data["last_accessed"])
                if data.get("last_accessed")
                else None
            ),
        )


@dataclass
class SecretAccessLog:
    """Log entry for secret access."""

    timestamp: datetime
    secret_id: str
    action: str  # read, write, delete, rotate
    actor: str
    success: bool
    ip_address: Optional[str] = None
    details: str = ""


@dataclass
class SecretsConfig:
    """Secrets manager configuration."""

    # Storage
    secrets_dir: Path = Path("secrets")
    encrypted: bool = True

    # Encryption
    key_derivation_iterations: int = 480000
    salt_length: int = 16

    # Audit
    enable_audit_log: bool = True
    audit_log_file: Path = Path("logs/secrets_audit.log")

    # Rotation
    auto_rotate: bool = False
    rotation_check_interval: int = 3600  # seconds

    # Access
    max_access_per_hour: int = 1000


class SecretsManager:
    """
    Secure secrets and key management.

    Provides encrypted storage, access control,
    audit logging, and key rotation.
    """

    def __init__(
        self,
        master_key: Optional[str] = None,
        config: Optional[SecretsConfig] = None,
    ):
        """
        Initialize secrets manager.

        Args:
            master_key: Master encryption key (or env var SECRETS_MASTER_KEY)
            config: Configuration
        """
        self._config = config or SecretsConfig()

        # Get master key
        self._master_key = master_key or os.environ.get("SECRETS_MASTER_KEY")
        if not self._master_key and self._config.encrypted:
            raise ValueError(
                "Master key required for encrypted secrets. "
                "Set SECRETS_MASTER_KEY environment variable."
            )

        # Initialize encryption
        self._fernet: Optional[Fernet] = None
        if self._config.encrypted and self._master_key:
            self._fernet = self._create_fernet(self._master_key)

        # Ensure directories exist
        self._config.secrets_dir.mkdir(parents=True, exist_ok=True)
        if self._config.enable_audit_log:
            self._config.audit_log_file.parent.mkdir(parents=True, exist_ok=True)

        # State
        self._running = False
        self._rotation_task: Optional[Any] = None
        self._access_counts: Dict[str, int] = {}
        self._access_hour: Optional[datetime] = None

        # Callbacks
        self._on_rotation: List[Callable[[str, int], Any]] = []

    def _create_fernet(self, master_key: str) -> Fernet:
        """Create Fernet instance from master key."""
        # Derive key using PBKDF2
        salt = b"trading_secrets_salt"  # Fixed salt for consistency
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._config.key_derivation_iterations,
        )
        key = base64.urlsafe_b64encode(kdf.derive(master_key.encode()))
        return Fernet(key)

    def _encrypt(self, data: str) -> str:
        """Encrypt data."""
        if not self._fernet:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def _decrypt(self, data: str) -> str:
        """Decrypt data."""
        if not self._fernet:
            return data
        return self._fernet.decrypt(data.encode()).decode()

    async def start(self) -> None:
        """Start secrets manager."""
        if self._running:
            return

        self._running = True

        if self._config.auto_rotate:
            import asyncio

            self._rotation_task = asyncio.create_task(self._rotation_loop())

        logger.info("Secrets manager started")

    async def stop(self) -> None:
        """Stop secrets manager."""
        if not self._running:
            return

        self._running = False

        if self._rotation_task:
            self._rotation_task.cancel()

        logger.info("Secrets manager stopped")

    async def _rotation_loop(self) -> None:
        """Check and rotate secrets as needed."""
        import asyncio

        while self._running:
            try:
                await self._check_rotation()
                await asyncio.sleep(self._config.rotation_check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Rotation check error: {e}")
                await asyncio.sleep(60)

    async def _check_rotation(self) -> None:
        """Check all secrets for rotation needs."""
        secrets_list = await self.list_secrets()
        now = datetime.now(timezone.utc)

        for metadata in secrets_list:
            if metadata.rotation_policy == RotationPolicy.NEVER:
                continue

            if await self._needs_rotation(metadata, now):
                logger.info(f"Auto-rotating secret: {metadata.secret_id}")
                # Rotation would require new value - just log warning
                logger.warning(
                    f"Secret {metadata.secret_id} needs rotation "
                    f"(policy: {metadata.rotation_policy.value})"
                )

    async def _needs_rotation(
        self,
        metadata: SecretMetadata,
        now: datetime,
    ) -> bool:
        """Check if secret needs rotation."""
        if metadata.rotation_policy == RotationPolicy.NEVER:
            return False

        last_rotated = metadata.last_rotated or metadata.created_at

        if metadata.rotation_policy == RotationPolicy.DAILY:
            return (now - last_rotated) > timedelta(days=1)
        elif metadata.rotation_policy == RotationPolicy.WEEKLY:
            return (now - last_rotated) > timedelta(weeks=1)
        elif metadata.rotation_policy == RotationPolicy.MONTHLY:
            return (now - last_rotated) > timedelta(days=30)
        elif metadata.rotation_policy == RotationPolicy.QUARTERLY:
            return (now - last_rotated) > timedelta(days=90)

        return False

    async def store_secret(
        self,
        secret_id: str,
        value: str,
        secret_type: SecretType = SecretType.GENERIC,
        description: str = "",
        rotation_policy: RotationPolicy = RotationPolicy.NEVER,
        expires_at: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None,
        actor: str = "system",
    ) -> SecretMetadata:
        """
        Store a secret.

        Args:
            secret_id: Unique identifier
            value: Secret value
            secret_type: Type of secret
            description: Description
            rotation_policy: Rotation policy
            expires_at: Expiration time
            tags: Tags
            actor: Who is storing the secret

        Returns:
            Secret metadata
        """
        now = datetime.now(timezone.utc)

        # Check if exists (for versioning)
        existing = await self.get_metadata(secret_id)
        version = (existing.version + 1) if existing else 1

        # Create metadata
        metadata = SecretMetadata(
            secret_id=secret_id,
            secret_type=secret_type,
            created_at=existing.created_at if existing else now,
            updated_at=now,
            expires_at=expires_at,
            rotation_policy=rotation_policy,
            last_rotated=now if existing else None,
            version=version,
            description=description,
            tags=tags or {},
        )

        # Encrypt and store value
        encrypted_value = self._encrypt(value)

        secret_path = self._get_secret_path(secret_id)
        secret_path.parent.mkdir(parents=True, exist_ok=True)

        with open(secret_path, "w") as f:
            json.dump(
                {
                    "metadata": metadata.to_dict(),
                    "value": encrypted_value,
                },
                f,
                indent=2,
            )

        # Audit log
        await self._audit_log(
            secret_id=secret_id,
            action="write",
            actor=actor,
            success=True,
            details=f"Version {version}",
        )

        logger.debug(f"Secret stored: {secret_id} (v{version})")
        return metadata

    async def get_secret(
        self,
        secret_id: str,
        actor: str = "system",
    ) -> Optional[str]:
        """
        Retrieve a secret value.

        Args:
            secret_id: Secret identifier
            actor: Who is accessing

        Returns:
            Decrypted secret value or None
        """
        # Rate limiting
        if not await self._check_rate_limit():
            logger.warning(f"Rate limit exceeded for secrets access")
            await self._audit_log(
                secret_id=secret_id,
                action="read",
                actor=actor,
                success=False,
                details="Rate limit exceeded",
            )
            return None

        secret_path = self._get_secret_path(secret_id)

        if not secret_path.exists():
            await self._audit_log(
                secret_id=secret_id,
                action="read",
                actor=actor,
                success=False,
                details="Not found",
            )
            return None

        try:
            with open(secret_path, "r") as f:
                data = json.load(f)

            metadata = SecretMetadata.from_dict(data["metadata"])

            # Check expiration
            if metadata.expires_at and metadata.expires_at < datetime.now(timezone.utc):
                await self._audit_log(
                    secret_id=secret_id,
                    action="read",
                    actor=actor,
                    success=False,
                    details="Expired",
                )
                return None

            # Decrypt value
            value = self._decrypt(data["value"])

            # Update access stats
            metadata.access_count += 1
            metadata.last_accessed = datetime.now(timezone.utc)
            metadata.updated_at = datetime.now(timezone.utc)

            with open(secret_path, "w") as f:
                json.dump(
                    {
                        "metadata": metadata.to_dict(),
                        "value": data["value"],
                    },
                    f,
                    indent=2,
                )

            await self._audit_log(
                secret_id=secret_id,
                action="read",
                actor=actor,
                success=True,
            )

            return value

        except Exception as e:
            logger.error(f"Failed to get secret {secret_id}: {e}")
            await self._audit_log(
                secret_id=secret_id,
                action="read",
                actor=actor,
                success=False,
                details=str(e),
            )
            return None

    async def delete_secret(
        self,
        secret_id: str,
        actor: str = "system",
    ) -> bool:
        """
        Delete a secret.

        Args:
            secret_id: Secret identifier
            actor: Who is deleting

        Returns:
            True if deleted
        """
        secret_path = self._get_secret_path(secret_id)

        if not secret_path.exists():
            return False

        try:
            secret_path.unlink()

            await self._audit_log(
                secret_id=secret_id,
                action="delete",
                actor=actor,
                success=True,
            )

            logger.info(f"Secret deleted: {secret_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete secret {secret_id}: {e}")
            await self._audit_log(
                secret_id=secret_id,
                action="delete",
                actor=actor,
                success=False,
                details=str(e),
            )
            return False

    async def rotate_secret(
        self,
        secret_id: str,
        new_value: str,
        actor: str = "system",
    ) -> Optional[SecretMetadata]:
        """
        Rotate a secret with a new value.

        Args:
            secret_id: Secret identifier
            new_value: New secret value
            actor: Who is rotating

        Returns:
            Updated metadata or None
        """
        metadata = await self.get_metadata(secret_id)
        if not metadata:
            return None

        # Store with new value (increments version)
        new_metadata = await self.store_secret(
            secret_id=secret_id,
            value=new_value,
            secret_type=metadata.secret_type,
            description=metadata.description,
            rotation_policy=metadata.rotation_policy,
            expires_at=metadata.expires_at,
            tags=metadata.tags,
            actor=actor,
        )

        await self._audit_log(
            secret_id=secret_id,
            action="rotate",
            actor=actor,
            success=True,
            details=f"v{metadata.version} -> v{new_metadata.version}",
        )

        # Notify callbacks
        for callback in self._on_rotation:
            try:
                result = callback(secret_id, new_metadata.version)
                if hasattr(result, "__await__"):
                    await result
            except Exception as e:
                logger.error(f"Rotation callback error: {e}")

        return new_metadata

    async def get_metadata(self, secret_id: str) -> Optional[SecretMetadata]:
        """Get metadata for a secret."""
        secret_path = self._get_secret_path(secret_id)

        if not secret_path.exists():
            return None

        with open(secret_path, "r") as f:
            data = json.load(f)

        return SecretMetadata.from_dict(data["metadata"])

    async def list_secrets(
        self,
        secret_type: Optional[SecretType] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[SecretMetadata]:
        """
        List all secrets.

        Args:
            secret_type: Filter by type
            tags: Filter by tags

        Returns:
            List of metadata
        """
        results = []

        for secret_file in self._config.secrets_dir.glob("*.json"):
            try:
                with open(secret_file, "r") as f:
                    data = json.load(f)
                metadata = SecretMetadata.from_dict(data["metadata"])

                # Apply filters
                if secret_type and metadata.secret_type != secret_type:
                    continue
                if tags:
                    if not all(
                        metadata.tags.get(k) == v for k, v in tags.items()
                    ):
                        continue

                results.append(metadata)

            except Exception as e:
                logger.error(f"Failed to read secret file {secret_file}: {e}")

        return results

    async def _check_rate_limit(self) -> bool:
        """Check if rate limit allows access."""
        now = datetime.now(timezone.utc)
        current_hour = now.replace(minute=0, second=0, microsecond=0)

        if self._access_hour != current_hour:
            self._access_hour = current_hour
            self._access_counts.clear()

        total_access = sum(self._access_counts.values())
        if total_access >= self._config.max_access_per_hour:
            return False

        self._access_counts["total"] = total_access + 1
        return True

    async def _audit_log(
        self,
        secret_id: str,
        action: str,
        actor: str,
        success: bool,
        ip_address: Optional[str] = None,
        details: str = "",
    ) -> None:
        """Write audit log entry."""
        if not self._config.enable_audit_log:
            return

        log_entry = SecretAccessLog(
            timestamp=datetime.now(timezone.utc),
            secret_id=secret_id,
            action=action,
            actor=actor,
            success=success,
            ip_address=ip_address,
            details=details,
        )

        log_line = json.dumps(
            {
                "timestamp": log_entry.timestamp.isoformat(),
                "secret_id": log_entry.secret_id,
                "action": log_entry.action,
                "actor": log_entry.actor,
                "success": log_entry.success,
                "ip_address": log_entry.ip_address,
                "details": log_entry.details,
            }
        )

        with open(self._config.audit_log_file, "a") as f:
            f.write(log_line + "\n")

    def _get_secret_path(self, secret_id: str) -> Path:
        """Get path for secret file."""
        # Sanitize secret_id for filesystem
        safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in secret_id)
        return self._config.secrets_dir / f"{safe_id}.json"

    def on_rotation(
        self,
        callback: Callable[[str, int], Any],
    ) -> Callable[[], None]:
        """
        Register callback for secret rotation.

        Args:
            callback: Callback function(secret_id, new_version)

        Returns:
            Unsubscribe function
        """
        self._on_rotation.append(callback)

        def unsubscribe():
            if callback in self._on_rotation:
                self._on_rotation.remove(callback)

        return unsubscribe

    @staticmethod
    def generate_key(length: int = 32) -> str:
        """
        Generate a secure random key.

        Args:
            length: Key length in bytes

        Returns:
            Base64-encoded key
        """
        return base64.urlsafe_b64encode(secrets.token_bytes(length)).decode()

    @staticmethod
    def generate_api_key() -> str:
        """Generate an API key."""
        return secrets.token_urlsafe(32)

    @staticmethod
    def hash_secret(value: str, salt: Optional[str] = None) -> tuple[str, str]:
        """
        Hash a secret value.

        Args:
            value: Value to hash
            salt: Optional salt (generated if not provided)

        Returns:
            Tuple of (hash, salt)
        """
        if salt is None:
            salt = secrets.token_hex(16)

        hash_value = hashlib.pbkdf2_hmac(
            "sha256",
            value.encode(),
            salt.encode(),
            iterations=100000,
        )
        return base64.b64encode(hash_value).decode(), salt

    @staticmethod
    def verify_hash(value: str, hash_value: str, salt: str) -> bool:
        """
        Verify a value against its hash.

        Args:
            value: Value to verify
            hash_value: Expected hash
            salt: Salt used for hashing

        Returns:
            True if match
        """
        computed_hash, _ = SecretsManager.hash_secret(value, salt)
        return hmac.compare_digest(computed_hash, hash_value)
