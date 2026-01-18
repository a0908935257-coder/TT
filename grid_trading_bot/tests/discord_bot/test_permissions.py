"""
Tests for Discord Bot Permissions.
"""

import pytest

from src.discord_bot.permissions import (
    PermissionChecker,
    PermissionConfig,
    PermissionLevel,
    load_permission_config_from_env,
)
from tests.discord_bot import (
    MockGuildPermissions,
    MockMember,
    MockRole,
    MockUser,
    create_admin_member,
    create_user_member,
)


# =============================================================================
# PermissionLevel Tests
# =============================================================================


class TestPermissionLevel:
    """Tests for PermissionLevel enum."""

    def test_level_ordering(self):
        """Test permission level ordering."""
        assert PermissionLevel.NONE < PermissionLevel.USER
        assert PermissionLevel.USER < PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN < PermissionLevel.OWNER

    def test_level_comparison_ge(self):
        """Test >= comparison."""
        assert PermissionLevel.OWNER >= PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN >= PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN >= PermissionLevel.USER
        assert not PermissionLevel.USER >= PermissionLevel.ADMIN

    def test_level_comparison_le(self):
        """Test <= comparison."""
        assert PermissionLevel.USER <= PermissionLevel.ADMIN
        assert PermissionLevel.ADMIN <= PermissionLevel.ADMIN
        assert not PermissionLevel.OWNER <= PermissionLevel.ADMIN


# =============================================================================
# PermissionConfig Tests
# =============================================================================


class TestPermissionConfig:
    """Tests for PermissionConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = PermissionConfig()
        assert len(config.owner_ids) == 0
        assert config.admin_role_id is None
        assert config.user_role_id is None
        assert len(config.allowed_channels) == 0
        assert config.allow_server_admins is True

    def test_add_owner(self):
        """Test adding owner."""
        config = PermissionConfig()
        config.add_owner(12345)
        assert 12345 in config.owner_ids

    def test_remove_owner(self):
        """Test removing owner."""
        config = PermissionConfig(owner_ids={12345, 67890})
        config.remove_owner(12345)
        assert 12345 not in config.owner_ids
        assert 67890 in config.owner_ids

    def test_add_allowed_channel(self):
        """Test adding allowed channel."""
        config = PermissionConfig()
        config.add_allowed_channel(111111)
        assert 111111 in config.allowed_channels


# =============================================================================
# PermissionChecker Tests
# =============================================================================


class TestPermissionChecker:
    """Tests for PermissionChecker."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return PermissionConfig(
            owner_ids={12345},
            admin_role_id=99999,
            user_role_id=88888,
            allowed_channels={111111, 222222},
            control_channel_id=111111,
            allow_server_admins=True,
        )

    @pytest.fixture
    def checker(self, config):
        """Create checker with config."""
        return PermissionChecker(config)

    # -------------------------------------------------------------------------
    # get_level Tests
    # -------------------------------------------------------------------------

    def test_get_level_owner(self, checker):
        """Test owner is recognized."""
        owner = MockMember(id=12345, name="Owner")
        assert checker.get_level(owner) == PermissionLevel.OWNER

    def test_get_level_admin_by_role(self, checker):
        """Test admin by role."""
        admin = MockMember(
            id=54321,
            roles=[MockRole(id=99999, name="Admin")],
        )
        assert checker.get_level(admin) == PermissionLevel.ADMIN

    def test_get_level_admin_by_server_permission(self, checker):
        """Test admin by server administrator permission."""
        server_admin = MockMember(
            id=54321,
            guild_permissions=MockGuildPermissions(administrator=True),
        )
        assert checker.get_level(server_admin) == PermissionLevel.ADMIN

    def test_get_level_user_by_role(self, checker):
        """Test user by role."""
        user = MockMember(
            id=54321,
            roles=[MockRole(id=88888, name="User")],
        )
        assert checker.get_level(user) == PermissionLevel.USER

    def test_get_level_default_user(self, checker):
        """Test default level is USER."""
        member = MockMember(id=99999)
        assert checker.get_level(member) == PermissionLevel.USER

    def test_get_level_discord_user_object(self, checker):
        """Test User object (non-member) returns NONE unless owner."""
        user = MockUser(id=99999)
        assert checker.get_level(user) == PermissionLevel.NONE

        owner_user = MockUser(id=12345)
        assert checker.get_level(owner_user) == PermissionLevel.OWNER

    # -------------------------------------------------------------------------
    # check Tests
    # -------------------------------------------------------------------------

    def test_check_owner_can_do_all(self, checker):
        """Test owner can pass all checks."""
        owner = MockMember(id=12345)
        assert checker.check(owner, PermissionLevel.OWNER)
        assert checker.check(owner, PermissionLevel.ADMIN)
        assert checker.check(owner, PermissionLevel.USER)

    def test_check_admin_cannot_owner(self, checker):
        """Test admin cannot pass owner check."""
        admin = create_admin_member(user_id=54321, admin_role_id=99999)
        assert not checker.check(admin, PermissionLevel.OWNER)
        assert checker.check(admin, PermissionLevel.ADMIN)
        assert checker.check(admin, PermissionLevel.USER)

    def test_check_user_limited(self, checker):
        """Test user can only pass user check."""
        user = MockMember(
            id=77777,
            roles=[MockRole(id=88888, name="User")],
        )
        assert not checker.check(user, PermissionLevel.OWNER)
        assert not checker.check(user, PermissionLevel.ADMIN)
        assert checker.check(user, PermissionLevel.USER)

    # -------------------------------------------------------------------------
    # Convenience Methods
    # -------------------------------------------------------------------------

    def test_is_owner(self, checker):
        """Test is_owner method."""
        owner = MockMember(id=12345)
        not_owner = MockMember(id=99999)

        assert checker.is_owner(owner)
        assert not checker.is_owner(not_owner)

    def test_is_admin(self, checker):
        """Test is_admin method."""
        owner = MockMember(id=12345)
        admin = MockMember(id=54321, roles=[MockRole(id=99999)])
        user = MockMember(id=77777, roles=[MockRole(id=88888)])

        assert checker.is_admin(owner)  # Owner is also admin
        assert checker.is_admin(admin)
        assert not checker.is_admin(user)

    def test_is_user(self, checker):
        """Test is_user method."""
        owner = MockMember(id=12345)
        admin = MockMember(id=54321, roles=[MockRole(id=99999)])
        user = MockMember(id=77777, roles=[MockRole(id=88888)])
        regular = MockMember(id=99999)

        assert checker.is_user(owner)
        assert checker.is_user(admin)
        assert checker.is_user(user)
        assert checker.is_user(regular)  # Default is user level

    # -------------------------------------------------------------------------
    # Channel Checks
    # -------------------------------------------------------------------------

    def test_is_allowed_channel(self, checker):
        """Test channel allowlist."""
        assert checker.is_allowed_channel(111111)
        assert checker.is_allowed_channel(222222)
        assert not checker.is_allowed_channel(333333)

    def test_is_allowed_channel_empty_list(self):
        """Test empty allowlist allows all."""
        config = PermissionConfig(allowed_channels=set())
        checker = PermissionChecker(config)
        assert checker.is_allowed_channel(999999)

    def test_is_control_channel(self, checker):
        """Test control channel check."""
        assert checker.is_control_channel(111111)
        assert not checker.is_control_channel(222222)

    def test_is_control_channel_not_set(self):
        """Test control channel not set allows all."""
        config = PermissionConfig(control_channel_id=None)
        checker = PermissionChecker(config)
        assert checker.is_control_channel(999999)


# =============================================================================
# load_permission_config_from_env Tests
# =============================================================================


class TestLoadPermissionConfigFromEnv:
    """Tests for loading config from environment."""

    def test_load_owner_ids(self, monkeypatch):
        """Test loading owner IDs."""
        monkeypatch.setenv("DISCORD_OWNER_IDS", "12345,67890")
        config = load_permission_config_from_env()
        assert 12345 in config.owner_ids
        assert 67890 in config.owner_ids

    def test_load_admin_role(self, monkeypatch):
        """Test loading admin role ID."""
        monkeypatch.setenv("DISCORD_ADMIN_ROLE_ID", "99999")
        config = load_permission_config_from_env()
        assert config.admin_role_id == 99999

    def test_load_user_role(self, monkeypatch):
        """Test loading user role ID."""
        monkeypatch.setenv("DISCORD_USER_ROLE_ID", "88888")
        config = load_permission_config_from_env()
        assert config.user_role_id == 88888

    def test_load_allowed_channels(self, monkeypatch):
        """Test loading allowed channels."""
        monkeypatch.setenv("DISCORD_ALLOWED_CHANNELS", "111,222,333")
        config = load_permission_config_from_env()
        assert 111 in config.allowed_channels
        assert 222 in config.allowed_channels
        assert 333 in config.allowed_channels

    def test_load_control_channel(self, monkeypatch):
        """Test loading control channel."""
        monkeypatch.setenv("DISCORD_CONTROL_CHANNEL_ID", "111111")
        config = load_permission_config_from_env()
        assert config.control_channel_id == 111111

    def test_load_allow_server_admins_true(self, monkeypatch):
        """Test loading allow_server_admins = true."""
        monkeypatch.setenv("DISCORD_ALLOW_SERVER_ADMINS", "true")
        config = load_permission_config_from_env()
        assert config.allow_server_admins is True

    def test_load_allow_server_admins_false(self, monkeypatch):
        """Test loading allow_server_admins = false."""
        monkeypatch.setenv("DISCORD_ALLOW_SERVER_ADMINS", "false")
        config = load_permission_config_from_env()
        assert config.allow_server_admins is False

    def test_load_invalid_values_ignored(self, monkeypatch):
        """Test invalid values are ignored."""
        monkeypatch.setenv("DISCORD_OWNER_IDS", "12345,invalid,67890")
        monkeypatch.setenv("DISCORD_ADMIN_ROLE_ID", "not_a_number")
        config = load_permission_config_from_env()
        assert 12345 in config.owner_ids
        assert 67890 in config.owner_ids
        assert config.admin_role_id is None


# =============================================================================
# Permission Integration Tests
# =============================================================================


class TestPermissionIntegration:
    """Integration tests for permission system."""

    def test_admin_can_create_bot(self):
        """Test admin permission allows bot creation."""
        config = PermissionConfig(admin_role_id=99999)
        checker = PermissionChecker(config)

        admin = create_admin_member(admin_role_id=99999)
        assert checker.is_admin(admin)

    def test_user_cannot_create_bot(self):
        """Test user permission denies bot creation."""
        config = PermissionConfig(admin_role_id=99999, user_role_id=88888)
        checker = PermissionChecker(config)

        user = create_user_member(user_role_id=88888)
        assert not checker.is_admin(user)

    def test_user_can_view_status(self):
        """Test user permission allows viewing status."""
        config = PermissionConfig(user_role_id=88888)
        checker = PermissionChecker(config)

        user = create_user_member(user_role_id=88888)
        assert checker.is_user(user)

    def test_owner_overrides_all(self):
        """Test owner has all permissions regardless of roles."""
        config = PermissionConfig(
            owner_ids={12345},
            admin_role_id=99999,
            user_role_id=88888,
        )
        checker = PermissionChecker(config)

        # Owner without any roles
        owner = MockMember(id=12345, roles=[])
        assert checker.is_owner(owner)
        assert checker.is_admin(owner)
        assert checker.is_user(owner)
