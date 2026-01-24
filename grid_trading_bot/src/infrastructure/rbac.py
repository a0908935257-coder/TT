"""
Role-Based Access Control (RBAC) Module.

Provides fine-grained API-level authorization with role definitions,
permission management, and access control enforcement.
"""

import asyncio
import hashlib
import json
import secrets
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from src.core import get_logger

logger = get_logger(__name__)


class Permission(Enum):
    """System permissions."""

    # Bot Management
    BOT_VIEW = "bot:view"
    BOT_START = "bot:start"
    BOT_STOP = "bot:stop"
    BOT_CREATE = "bot:create"
    BOT_DELETE = "bot:delete"
    BOT_CONFIG = "bot:config"

    # Order Management
    ORDER_VIEW = "order:view"
    ORDER_CREATE = "order:create"
    ORDER_CANCEL = "order:cancel"
    ORDER_MODIFY = "order:modify"

    # Position Management
    POSITION_VIEW = "position:view"
    POSITION_CLOSE = "position:close"
    POSITION_MODIFY = "position:modify"

    # Account Management
    ACCOUNT_VIEW = "account:view"
    ACCOUNT_BALANCE = "account:balance"
    ACCOUNT_WITHDRAW = "account:withdraw"

    # Configuration
    CONFIG_VIEW = "config:view"
    CONFIG_MODIFY = "config:modify"
    CONFIG_SECRETS = "config:secrets"

    # System Administration
    ADMIN_USERS = "admin:users"
    ADMIN_ROLES = "admin:roles"
    ADMIN_SYSTEM = "admin:system"
    ADMIN_LOGS = "admin:logs"
    ADMIN_BACKUP = "admin:backup"

    # API Access
    API_READ = "api:read"
    API_WRITE = "api:write"
    API_ADMIN = "api:admin"

    # Reports
    REPORT_VIEW = "report:view"
    REPORT_EXPORT = "report:export"


class RoleType(Enum):
    """Predefined role types."""

    VIEWER = "viewer"  # Read-only access
    TRADER = "trader"  # Trading operations
    OPERATOR = "operator"  # System operations
    ADMIN = "admin"  # Full administrative access
    SUPER_ADMIN = "super_admin"  # Unrestricted access
    CUSTOM = "custom"  # Custom defined role


@dataclass
class Role:
    """Role definition."""

    role_id: str
    name: str
    role_type: RoleType
    permissions: Set[Permission]
    description: str = ""
    parent_role: Optional[str] = None  # Inherit from parent
    is_system: bool = False  # System role cannot be deleted
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "role_id": self.role_id,
            "name": self.name,
            "role_type": self.role_type.value,
            "permissions": [p.value for p in self.permissions],
            "description": self.description,
            "parent_role": self.parent_role,
            "is_system": self.is_system,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Role":
        """Create from dictionary."""
        return cls(
            role_id=data["role_id"],
            name=data["name"],
            role_type=RoleType(data["role_type"]),
            permissions={Permission(p) for p in data["permissions"]},
            description=data.get("description", ""),
            parent_role=data.get("parent_role"),
            is_system=data.get("is_system", False),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by", "system"),
        )


@dataclass
class User:
    """User definition."""

    user_id: str
    username: str
    roles: Set[str]  # Role IDs
    api_key_hash: Optional[str] = None
    is_active: bool = True
    is_service_account: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    expires_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "roles": list(self.roles),
            "api_key_hash": self.api_key_hash,
            "is_active": self.is_active,
            "is_service_account": self.is_service_account,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            roles=set(data["roles"]),
            api_key_hash=data.get("api_key_hash"),
            is_active=data.get("is_active", True),
            is_service_account=data.get("is_service_account", False),
            metadata=data.get("metadata", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_login=(
                datetime.fromisoformat(data["last_login"])
                if data.get("last_login")
                else None
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
        )

    def is_expired(self) -> bool:
        """Check if user account is expired."""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at


@dataclass
class APIEndpoint:
    """API endpoint permission mapping."""

    path: str  # Path pattern (regex)
    method: str  # HTTP method (GET, POST, etc.) or '*' for all
    required_permissions: Set[Permission]
    description: str = ""
    rate_limit: Optional[int] = None  # Requests per minute

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "path": self.path,
            "method": self.method,
            "required_permissions": [p.value for p in self.required_permissions],
            "description": self.description,
            "rate_limit": self.rate_limit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "APIEndpoint":
        """Create from dictionary."""
        return cls(
            path=data["path"],
            method=data["method"],
            required_permissions={Permission(p) for p in data["required_permissions"]},
            description=data.get("description", ""),
            rate_limit=data.get("rate_limit"),
        )


@dataclass
class AccessCheckResult:
    """Result of access check."""

    allowed: bool
    user: Optional[User] = None
    missing_permissions: Set[Permission] = field(default_factory=set)
    reason: str = ""


@dataclass
class RBACConfig:
    """RBAC configuration."""

    # Storage
    data_file: Path = Path("config/rbac.json")

    # API Key settings
    api_key_length: int = 32
    api_key_prefix: str = "gtb_"  # Grid Trading Bot

    # Session settings
    session_timeout_minutes: int = 60
    max_sessions_per_user: int = 5

    # Audit
    audit_access: bool = True
    audit_file: Path = Path("logs/rbac_audit.log")


class RBACManager:
    """
    Role-Based Access Control manager.

    Provides user management, role assignment, permission
    checking, and API endpoint authorization.
    """

    def __init__(self, config: Optional[RBACConfig] = None):
        """
        Initialize RBAC manager.

        Args:
            config: RBAC configuration
        """
        self._config = config or RBACConfig()
        self._roles: Dict[str, Role] = {}
        self._users: Dict[str, User] = {}
        self._endpoints: Dict[str, APIEndpoint] = {}
        self._api_keys: Dict[str, str] = {}  # hash -> user_id

        # Sessions
        self._sessions: Dict[str, Tuple[str, datetime]] = {}  # token -> (user_id, expiry)

        # Audit callbacks
        self._on_access_denied: List[Callable[[str, str, Set[Permission]], Any]] = []

        # Initialize system roles
        self._init_system_roles()

        # Load data
        self._load_data()

    def _init_system_roles(self) -> None:
        """Initialize predefined system roles."""
        # Viewer role
        self._roles["viewer"] = Role(
            role_id="viewer",
            name="Viewer",
            role_type=RoleType.VIEWER,
            permissions={
                Permission.BOT_VIEW,
                Permission.ORDER_VIEW,
                Permission.POSITION_VIEW,
                Permission.ACCOUNT_VIEW,
                Permission.REPORT_VIEW,
                Permission.API_READ,
            },
            description="Read-only access to trading data",
            is_system=True,
        )

        # Trader role
        self._roles["trader"] = Role(
            role_id="trader",
            name="Trader",
            role_type=RoleType.TRADER,
            permissions={
                # Inherits viewer
                Permission.BOT_VIEW,
                Permission.ORDER_VIEW,
                Permission.POSITION_VIEW,
                Permission.ACCOUNT_VIEW,
                Permission.REPORT_VIEW,
                Permission.API_READ,
                # Trading permissions
                Permission.BOT_START,
                Permission.BOT_STOP,
                Permission.ORDER_CREATE,
                Permission.ORDER_CANCEL,
                Permission.ORDER_MODIFY,
                Permission.POSITION_CLOSE,
                Permission.API_WRITE,
            },
            description="Trading operations access",
            is_system=True,
        )

        # Operator role
        self._roles["operator"] = Role(
            role_id="operator",
            name="Operator",
            role_type=RoleType.OPERATOR,
            permissions={
                # Inherits trader
                Permission.BOT_VIEW,
                Permission.ORDER_VIEW,
                Permission.POSITION_VIEW,
                Permission.ACCOUNT_VIEW,
                Permission.REPORT_VIEW,
                Permission.API_READ,
                Permission.BOT_START,
                Permission.BOT_STOP,
                Permission.ORDER_CREATE,
                Permission.ORDER_CANCEL,
                Permission.ORDER_MODIFY,
                Permission.POSITION_CLOSE,
                Permission.API_WRITE,
                # Operator permissions
                Permission.BOT_CREATE,
                Permission.BOT_CONFIG,
                Permission.POSITION_MODIFY,
                Permission.CONFIG_VIEW,
                Permission.CONFIG_MODIFY,
                Permission.ADMIN_LOGS,
                Permission.REPORT_EXPORT,
            },
            description="System operation access",
            is_system=True,
        )

        # Admin role
        self._roles["admin"] = Role(
            role_id="admin",
            name="Administrator",
            role_type=RoleType.ADMIN,
            permissions={
                # Inherits operator
                Permission.BOT_VIEW,
                Permission.ORDER_VIEW,
                Permission.POSITION_VIEW,
                Permission.ACCOUNT_VIEW,
                Permission.REPORT_VIEW,
                Permission.API_READ,
                Permission.BOT_START,
                Permission.BOT_STOP,
                Permission.ORDER_CREATE,
                Permission.ORDER_CANCEL,
                Permission.ORDER_MODIFY,
                Permission.POSITION_CLOSE,
                Permission.API_WRITE,
                Permission.BOT_CREATE,
                Permission.BOT_CONFIG,
                Permission.POSITION_MODIFY,
                Permission.CONFIG_VIEW,
                Permission.CONFIG_MODIFY,
                Permission.ADMIN_LOGS,
                Permission.REPORT_EXPORT,
                # Admin permissions
                Permission.BOT_DELETE,
                Permission.ACCOUNT_BALANCE,
                Permission.CONFIG_SECRETS,
                Permission.ADMIN_USERS,
                Permission.ADMIN_ROLES,
                Permission.ADMIN_SYSTEM,
                Permission.ADMIN_BACKUP,
                Permission.API_ADMIN,
            },
            description="Administrative access",
            is_system=True,
        )

        # Super Admin role
        self._roles["super_admin"] = Role(
            role_id="super_admin",
            name="Super Administrator",
            role_type=RoleType.SUPER_ADMIN,
            permissions=set(Permission),  # All permissions
            description="Unrestricted access",
            is_system=True,
        )

    def _load_data(self) -> None:
        """Load RBAC data from file."""
        if not self._config.data_file.exists():
            return

        try:
            with open(self._config.data_file, "r") as f:
                data = json.load(f)

            # Load custom roles
            for role_data in data.get("roles", []):
                role = Role.from_dict(role_data)
                if not role.is_system:  # Don't override system roles
                    self._roles[role.role_id] = role

            # Load users
            for user_data in data.get("users", []):
                user = User.from_dict(user_data)
                self._users[user.user_id] = user
                if user.api_key_hash:
                    self._api_keys[user.api_key_hash] = user.user_id

            # Load endpoints
            for endpoint_data in data.get("endpoints", []):
                endpoint = APIEndpoint.from_dict(endpoint_data)
                key = f"{endpoint.method}:{endpoint.path}"
                self._endpoints[key] = endpoint

            logger.info(
                f"Loaded RBAC: {len(self._roles)} roles, "
                f"{len(self._users)} users, {len(self._endpoints)} endpoints"
            )

        except Exception as e:
            logger.error(f"Failed to load RBAC data: {e}")

    def _save_data(self) -> None:
        """Save RBAC data to file."""
        try:
            self._config.data_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "roles": [
                    r.to_dict() for r in self._roles.values() if not r.is_system
                ],
                "users": [u.to_dict() for u in self._users.values()],
                "endpoints": [e.to_dict() for e in self._endpoints.values()],
            }

            with open(self._config.data_file, "w") as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save RBAC data: {e}")

    # === Role Management ===

    def create_role(
        self,
        name: str,
        permissions: Set[Permission],
        description: str = "",
        parent_role: Optional[str] = None,
        created_by: str = "system",
    ) -> Role:
        """
        Create a custom role.

        Args:
            name: Role name
            permissions: Set of permissions
            description: Role description
            parent_role: Parent role to inherit from
            created_by: Creator

        Returns:
            Created role
        """
        role_id = f"role_{name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}"

        # Inherit parent permissions
        if parent_role and parent_role in self._roles:
            permissions = permissions | self._roles[parent_role].permissions

        role = Role(
            role_id=role_id,
            name=name,
            role_type=RoleType.CUSTOM,
            permissions=permissions,
            description=description,
            parent_role=parent_role,
            is_system=False,
            created_by=created_by,
        )

        self._roles[role_id] = role
        self._save_data()

        logger.info(f"Created role: {role_id}")
        return role

    def delete_role(self, role_id: str) -> bool:
        """Delete a role (non-system only)."""
        if role_id not in self._roles:
            return False

        role = self._roles[role_id]
        if role.is_system:
            logger.warning(f"Cannot delete system role: {role_id}")
            return False

        # Remove role from users
        for user in self._users.values():
            user.roles.discard(role_id)

        del self._roles[role_id]
        self._save_data()

        logger.info(f"Deleted role: {role_id}")
        return True

    def get_role(self, role_id: str) -> Optional[Role]:
        """Get a role by ID."""
        return self._roles.get(role_id)

    def list_roles(self, include_system: bool = True) -> List[Role]:
        """List all roles."""
        roles = list(self._roles.values())
        if not include_system:
            roles = [r for r in roles if not r.is_system]
        return roles

    def add_permission_to_role(
        self,
        role_id: str,
        permission: Permission,
    ) -> bool:
        """Add permission to a role."""
        if role_id not in self._roles:
            return False

        role = self._roles[role_id]
        if role.is_system:
            return False

        role.permissions.add(permission)
        self._save_data()
        return True

    def remove_permission_from_role(
        self,
        role_id: str,
        permission: Permission,
    ) -> bool:
        """Remove permission from a role."""
        if role_id not in self._roles:
            return False

        role = self._roles[role_id]
        if role.is_system:
            return False

        role.permissions.discard(permission)
        self._save_data()
        return True

    # === User Management ===

    def create_user(
        self,
        username: str,
        roles: Optional[Set[str]] = None,
        is_service_account: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
        expires_at: Optional[datetime] = None,
    ) -> Tuple[User, Optional[str]]:
        """
        Create a user.

        Args:
            username: Username
            roles: Role IDs to assign
            is_service_account: Is this a service account
            metadata: Additional metadata
            expires_at: Account expiration

        Returns:
            Tuple of (User, API key if service account)
        """
        user_id = f"user_{username.lower()}_{secrets.token_hex(4)}"
        # Only use default viewer role if roles is None (not if empty set)
        if roles is None:
            roles = {"viewer"}

        # Validate roles exist (allow empty set for custom role assignment)
        roles = {r for r in roles if r in self._roles}

        user = User(
            user_id=user_id,
            username=username,
            roles=roles,
            is_service_account=is_service_account,
            metadata=metadata or {},
            expires_at=expires_at,
        )

        api_key = None
        if is_service_account:
            api_key = self._generate_api_key(user)

        self._users[user_id] = user
        self._save_data()

        logger.info(f"Created user: {username} ({user_id})")
        return user, api_key

    def _generate_api_key(self, user: User) -> str:
        """Generate API key for user."""
        raw_key = secrets.token_urlsafe(self._config.api_key_length)
        api_key = f"{self._config.api_key_prefix}{raw_key}"

        # Store hash
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()
        user.api_key_hash = key_hash
        self._api_keys[key_hash] = user.user_id

        return api_key

    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for user."""
        if user_id not in self._users:
            return None

        user = self._users[user_id]

        # Remove old key
        if user.api_key_hash:
            self._api_keys.pop(user.api_key_hash, None)

        # Generate new key
        api_key = self._generate_api_key(user)
        self._save_data()

        logger.info(f"Regenerated API key for: {user_id}")
        return api_key

    def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        if user_id not in self._users:
            return False

        user = self._users[user_id]

        # Remove API key
        if user.api_key_hash:
            self._api_keys.pop(user.api_key_hash, None)

        # Remove sessions
        self._sessions = {
            k: v for k, v in self._sessions.items() if v[0] != user_id
        }

        del self._users[user_id]
        self._save_data()

        logger.info(f"Deleted user: {user_id}")
        return True

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        for user in self._users.values():
            if user.username == username:
                return user
        return None

    def list_users(self, active_only: bool = False) -> List[User]:
        """List all users."""
        users = list(self._users.values())
        if active_only:
            users = [u for u in users if u.is_active and not u.is_expired()]
        return users

    def assign_role(self, user_id: str, role_id: str) -> bool:
        """Assign role to user."""
        if user_id not in self._users or role_id not in self._roles:
            return False

        self._users[user_id].roles.add(role_id)
        self._save_data()
        return True

    def revoke_role(self, user_id: str, role_id: str) -> bool:
        """Revoke role from user."""
        if user_id not in self._users:
            return False

        self._users[user_id].roles.discard(role_id)
        self._save_data()
        return True

    def get_user_permissions(self, user_id: str) -> Set[Permission]:
        """Get all permissions for a user."""
        if user_id not in self._users:
            return set()

        user = self._users[user_id]
        if not user.is_active or user.is_expired():
            return set()

        permissions = set()
        for role_id in user.roles:
            if role_id in self._roles:
                permissions |= self._roles[role_id].permissions

        return permissions

    # === Authentication ===

    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """
        Authenticate user by API key.

        Args:
            api_key: API key

        Returns:
            User if valid, None otherwise
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        if key_hash not in self._api_keys:
            return None

        user_id = self._api_keys[key_hash]
        user = self._users.get(user_id)

        if not user or not user.is_active or user.is_expired():
            return None

        # Update last login
        user.last_login = datetime.now(timezone.utc)
        return user

    def create_session(self, user_id: str) -> Optional[str]:
        """Create a session token for user."""
        if user_id not in self._users:
            return None

        user = self._users[user_id]
        if not user.is_active or user.is_expired():
            return None

        # Generate token
        token = secrets.token_urlsafe(32)
        expiry = datetime.now(timezone.utc) + timedelta(
            minutes=self._config.session_timeout_minutes
        )

        # Limit sessions per user
        user_sessions = [
            k for k, v in self._sessions.items() if v[0] == user_id
        ]
        while len(user_sessions) >= self._config.max_sessions_per_user:
            oldest = user_sessions.pop(0)
            del self._sessions[oldest]

        self._sessions[token] = (user_id, expiry)
        return token

    def validate_session(self, token: str) -> Optional[User]:
        """Validate session token."""
        if token not in self._sessions:
            return None

        user_id, expiry = self._sessions[token]

        if datetime.now(timezone.utc) > expiry:
            del self._sessions[token]
            return None

        return self._users.get(user_id)

    def invalidate_session(self, token: str) -> bool:
        """Invalidate a session."""
        if token in self._sessions:
            del self._sessions[token]
            return True
        return False

    # === Authorization ===

    def check_permission(
        self,
        user_id: str,
        permission: Permission,
    ) -> bool:
        """
        Check if user has a specific permission.

        Args:
            user_id: User ID
            permission: Permission to check

        Returns:
            True if user has permission
        """
        permissions = self.get_user_permissions(user_id)
        return permission in permissions

    def check_permissions(
        self,
        user_id: str,
        permissions: Set[Permission],
    ) -> AccessCheckResult:
        """
        Check if user has all required permissions.

        Args:
            user_id: User ID
            permissions: Required permissions

        Returns:
            Access check result
        """
        user = self._users.get(user_id)
        if not user:
            return AccessCheckResult(
                allowed=False,
                reason="User not found",
            )

        if not user.is_active:
            return AccessCheckResult(
                allowed=False,
                user=user,
                reason="User is not active",
            )

        if user.is_expired():
            return AccessCheckResult(
                allowed=False,
                user=user,
                reason="User account expired",
            )

        user_permissions = self.get_user_permissions(user_id)
        missing = permissions - user_permissions

        if missing:
            # Audit denied access
            self._audit_access_denied(user_id, missing)

            return AccessCheckResult(
                allowed=False,
                user=user,
                missing_permissions=missing,
                reason=f"Missing permissions: {', '.join(p.value for p in missing)}",
            )

        return AccessCheckResult(
            allowed=True,
            user=user,
        )

    def check_endpoint_access(
        self,
        user_id: str,
        path: str,
        method: str = "GET",
    ) -> AccessCheckResult:
        """
        Check if user can access an API endpoint.

        Args:
            user_id: User ID
            path: API path
            method: HTTP method

        Returns:
            Access check result
        """
        import re

        # Find matching endpoint
        for key, endpoint in self._endpoints.items():
            if endpoint.method != "*" and endpoint.method != method:
                continue

            try:
                if re.match(endpoint.path, path):
                    return self.check_permissions(
                        user_id,
                        endpoint.required_permissions,
                    )
            except re.error:
                continue

        # No matching endpoint - check default API permission
        if method == "GET":
            return self.check_permissions(user_id, {Permission.API_READ})
        else:
            return self.check_permissions(user_id, {Permission.API_WRITE})

    def _audit_access_denied(
        self,
        user_id: str,
        missing: Set[Permission],
    ) -> None:
        """Audit access denial."""
        if not self._config.audit_access:
            return

        try:
            self._config.audit_file.parent.mkdir(parents=True, exist_ok=True)

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "user_id": user_id,
                "missing_permissions": [p.value for p in missing],
            }

            with open(self._config.audit_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        except Exception as e:
            logger.error(f"Audit log error: {e}")

        # Notify callbacks
        for callback in self._on_access_denied:
            try:
                result = callback(user_id, str(missing), missing)
                if asyncio.iscoroutine(result):
                    asyncio.create_task(result)
            except Exception as e:
                logger.error(f"Access denied callback error: {e}")

    # === Endpoint Management ===

    def register_endpoint(
        self,
        path: str,
        method: str,
        permissions: Set[Permission],
        description: str = "",
        rate_limit: Optional[int] = None,
    ) -> APIEndpoint:
        """Register an API endpoint with required permissions."""
        endpoint = APIEndpoint(
            path=path,
            method=method,
            required_permissions=permissions,
            description=description,
            rate_limit=rate_limit,
        )

        key = f"{method}:{path}"
        self._endpoints[key] = endpoint
        self._save_data()

        return endpoint

    def unregister_endpoint(self, path: str, method: str) -> bool:
        """Unregister an API endpoint."""
        key = f"{method}:{path}"
        if key in self._endpoints:
            del self._endpoints[key]
            self._save_data()
            return True
        return False

    def list_endpoints(self) -> List[APIEndpoint]:
        """List all registered endpoints."""
        return list(self._endpoints.values())

    def on_access_denied(
        self,
        callback: Callable[[str, str, Set[Permission]], Any],
    ) -> Callable[[], None]:
        """Register callback for access denied events."""
        self._on_access_denied.append(callback)

        def unsubscribe():
            if callback in self._on_access_denied:
                self._on_access_denied.remove(callback)

        return unsubscribe

    def get_stats(self) -> Dict[str, Any]:
        """Get RBAC statistics."""
        return {
            "total_roles": len(self._roles),
            "system_roles": len([r for r in self._roles.values() if r.is_system]),
            "custom_roles": len([r for r in self._roles.values() if not r.is_system]),
            "total_users": len(self._users),
            "active_users": len([u for u in self._users.values() if u.is_active]),
            "service_accounts": len(
                [u for u in self._users.values() if u.is_service_account]
            ),
            "active_sessions": len(self._sessions),
            "registered_endpoints": len(self._endpoints),
        }
