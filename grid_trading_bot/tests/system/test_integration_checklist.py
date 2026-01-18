"""
Tests for Integration Checklist.

Tests the integration checker functionality.
"""

import pytest

from tests.system.integration_checklist import (
    CheckGroup,
    CheckResult,
    CheckStatus,
    IntegrationChecker,
)


# =============================================================================
# CheckResult Tests
# =============================================================================


class TestCheckResult:
    """Tests for CheckResult dataclass."""

    def test_create_pass(self):
        """Test creating a passing check result."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.PASS,
            message="All good",
        )
        assert result.name == "Test Check"
        assert result.status == CheckStatus.PASS
        assert result.message == "All good"
        assert result.error is None

    def test_create_fail(self):
        """Test creating a failing check result."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.FAIL,
            error="Something went wrong",
        )
        assert result.status == CheckStatus.FAIL
        assert result.error == "Something went wrong"

    def test_create_skip(self):
        """Test creating a skipped check result."""
        result = CheckResult(
            name="Test Check",
            status=CheckStatus.SKIP,
            message="No credentials",
        )
        assert result.status == CheckStatus.SKIP


# =============================================================================
# CheckGroup Tests
# =============================================================================


class TestCheckGroup:
    """Tests for CheckGroup dataclass."""

    def test_empty_group(self):
        """Test empty check group."""
        group = CheckGroup(name="Test Group")
        assert group.total == 0
        assert group.passed == 0
        assert group.failed == 0

    def test_group_counts(self):
        """Test group counts."""
        group = CheckGroup(name="Test Group")
        group.checks.append(CheckResult("Test 1", CheckStatus.PASS))
        group.checks.append(CheckResult("Test 2", CheckStatus.PASS))
        group.checks.append(CheckResult("Test 3", CheckStatus.FAIL))
        group.checks.append(CheckResult("Test 4", CheckStatus.SKIP))

        assert group.total == 4
        assert group.passed == 2
        assert group.failed == 1


# =============================================================================
# IntegrationChecker Tests
# =============================================================================


class TestIntegrationChecker:
    """Tests for IntegrationChecker."""

    @pytest.fixture
    def checker(self):
        """Create checker with mocks."""
        return IntegrationChecker(use_mocks=True)

    @pytest.mark.asyncio
    async def test_check_config_to_all(self, checker):
        """Test config checks."""
        group = await checker.check_config_to_all()

        assert group.name == "Configuration Layer"
        assert group.total >= 4  # At least 4 config checks

    @pytest.mark.asyncio
    async def test_check_exchange_connections(self, checker):
        """Test exchange connection checks."""
        group = await checker.check_exchange_connections()

        assert group.name == "Exchange Connections"
        assert group.total >= 1  # At least REST check

    @pytest.mark.asyncio
    async def test_check_db_connections(self, checker):
        """Test database connection checks."""
        group = await checker.check_db_connections()

        assert group.name == "Database Connection"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_redis_connections(self, checker):
        """Test Redis connection checks."""
        group = await checker.check_redis_connections()

        assert group.name == "Redis Connection"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_notification_flow(self, checker):
        """Test notification flow checks."""
        group = await checker.check_notification_flow()

        assert group.name == "Notification Flow"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_data_flow(self, checker):
        """Test data flow checks."""
        group = await checker.check_data_flow()

        assert group.name == "Data Flow"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_order_flow(self, checker):
        """Test order flow checks."""
        group = await checker.check_order_flow()

        assert group.name == "Order Flow"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_bot_lifecycle(self, checker):
        """Test bot lifecycle checks."""
        group = await checker.check_bot_lifecycle()

        assert group.name == "Bot Lifecycle"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_check_ipc_flow(self, checker):
        """Test IPC flow checks."""
        group = await checker.check_ipc_flow()

        assert group.name == "IPC Communication"
        assert group.total >= 1

    @pytest.mark.asyncio
    async def test_run_all(self, checker):
        """Test running all checks."""
        result = await checker.run_all()

        assert "total" in result
        assert "passed" in result
        assert "failed" in result
        assert "success" in result
        assert result["total"] > 0

    @pytest.mark.asyncio
    async def test_get_report_dict(self, checker):
        """Test getting report as dict."""
        await checker.run_all()
        report = checker.get_report_dict()

        assert "groups" in report
        assert "summary" in report
        assert len(report["groups"]) > 0

    @pytest.mark.asyncio
    async def test_all_checks_pass_with_mocks(self, checker):
        """Test that all mock checks pass."""
        result = await checker.run_all()

        # With mocks, all checks should pass
        assert result["failed"] == 0, f"Failed checks: {result}"


# =============================================================================
# CLI Tests
# =============================================================================


class TestCLI:
    """Tests for CLI interface."""

    @pytest.mark.asyncio
    async def test_main_with_mocks(self, capsys):
        """Test main function with mocks."""
        checker = IntegrationChecker(use_mocks=True)
        result = await checker.run_all()
        checker.print_report()

        captured = capsys.readouterr()
        assert "System Integration Check Report" in captured.out
        assert "Total:" in captured.out
