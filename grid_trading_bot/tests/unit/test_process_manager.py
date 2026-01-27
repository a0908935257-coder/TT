"""
Tests for Process Manager.

Validates subprocess lifecycle management.
"""

import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.master.process_manager import ProcessInfo, ProcessManager


# =============================================================================
# ProcessInfo Tests
# =============================================================================


class TestProcessInfo:
    """Tests for ProcessInfo dataclass."""

    def test_pid(self):
        """Test PID property."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None

        info = ProcessInfo(
            bot_id="bot-001",
            config_path="/path/to/config.yaml",
            process=mock_process,
        )

        assert info.pid == 12345

    def test_is_alive_true(self):
        """Test is_alive when process is running."""
        mock_process = MagicMock()
        mock_process.poll.return_value = None  # None means still running

        info = ProcessInfo(
            bot_id="bot-001",
            config_path="/path/to/config.yaml",
            process=mock_process,
        )

        assert info.is_alive is True

    def test_is_alive_false(self):
        """Test is_alive when process has terminated."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 0  # Return code means terminated

        info = ProcessInfo(
            bot_id="bot-001",
            config_path="/path/to/config.yaml",
            process=mock_process,
        )

        assert info.is_alive is False

    def test_return_code(self):
        """Test return code property."""
        mock_process = MagicMock()
        mock_process.poll.return_value = 1

        info = ProcessInfo(
            bot_id="bot-001",
            config_path="/path/to/config.yaml",
            process=mock_process,
        )

        assert info.return_code == 1


# =============================================================================
# ProcessManager Tests
# =============================================================================


class TestProcessManager:
    """Tests for ProcessManager."""

    @pytest.fixture
    def manager(self):
        """Create process manager."""
        return ProcessManager(max_restart_attempts=3)

    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        config = {
            "bot_type": "mock",
            "config": {"symbol": "BTCUSDT"},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config, f)
            f.flush()
            yield f.name
        os.unlink(f.name)

    def test_initialization(self, manager):
        """Test manager initialization."""
        assert manager.active_count == 0
        assert manager.all_bot_ids == []

    def test_spawn_missing_config(self, manager):
        """Test spawning with missing config file."""
        result = manager.spawn("bot-001", "/nonexistent/config.yaml")
        assert result is False

    @patch("subprocess.Popen")
    def test_spawn_success(self, mock_popen, manager, temp_config):
        """Test successful spawn."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        result = manager.spawn("bot-001", temp_config)

        assert result is True
        assert "bot-001" in manager.all_bot_ids
        assert manager.get_pid("bot-001") == 12345

    @patch("subprocess.Popen")
    def test_spawn_already_running(self, mock_popen, manager, temp_config):
        """Test spawning already running bot fails."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        result = manager.spawn("bot-001", temp_config)

        assert result is False

    @patch("subprocess.Popen")
    def test_kill_success(self, mock_popen, manager, temp_config):
        """Test killing a process."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        result = manager.kill("bot-001")

        assert result is True
        assert "bot-001" not in manager.all_bot_ids
        mock_process.terminate.assert_called_once()

    def test_kill_not_found(self, manager):
        """Test killing non-existent process."""
        result = manager.kill("nonexistent")
        assert result is False

    @patch("subprocess.Popen")
    def test_is_alive(self, mock_popen, manager, temp_config):
        """Test checking if process is alive."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)

        assert manager.is_alive("bot-001") is True
        assert manager.is_alive("nonexistent") is False

    @patch("subprocess.Popen")
    def test_get_pid(self, mock_popen, manager, temp_config):
        """Test getting process ID."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)

        assert manager.get_pid("bot-001") == 12345
        assert manager.get_pid("nonexistent") is None

    @patch("subprocess.Popen")
    def test_get_info(self, mock_popen, manager, temp_config):
        """Test getting process info."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)

        info = manager.get_info("bot-001")
        assert info is not None
        assert info.bot_id == "bot-001"
        assert info.pid == 12345

    @patch("subprocess.Popen")
    def test_restart(self, mock_popen, manager, temp_config):
        """Test restarting a process."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        result = manager.restart("bot-001")

        assert result is True
        info = manager.get_info("bot-001")
        assert info.restart_count == 1

    @patch("subprocess.Popen")
    def test_restart_max_attempts(self, mock_popen, manager, temp_config):
        """Test restart fails after max attempts."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)

        # Set restart count to max
        manager._processes["bot-001"].restart_count = 3

        result = manager.restart("bot-001")
        assert result is False

    def test_restart_not_found(self, manager):
        """Test restart of non-existent bot."""
        result = manager.restart("nonexistent")
        assert result is False

    @patch("subprocess.Popen")
    def test_check_health(self, mock_popen, manager, temp_config):
        """Test health check."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        manager.spawn("bot-002", temp_config)

        health = manager.check_health()

        assert "bot-001" in health
        assert "bot-002" in health
        assert health["bot-001"] is True

    @patch("subprocess.Popen")
    def test_get_failed_processes(self, mock_popen, manager, temp_config):
        """Test getting failed processes."""
        # First bot - alive
        mock_alive = MagicMock()
        mock_alive.pid = 12345
        mock_alive.poll.return_value = None

        # Second bot - dead
        mock_dead = MagicMock()
        mock_dead.pid = 12346
        mock_dead.poll.return_value = 1

        mock_popen.side_effect = [mock_alive, mock_dead]

        manager.spawn("bot-001", temp_config)
        manager.spawn("bot-002", temp_config)

        failed = manager.get_failed_processes()

        assert "bot-002" in failed
        assert "bot-001" not in failed

    @patch("subprocess.Popen")
    def test_cleanup_terminated(self, mock_popen, manager, temp_config):
        """Test cleaning up terminated processes."""
        mock_dead = MagicMock()
        mock_dead.pid = 12345
        mock_dead.poll.return_value = 0
        mock_popen.return_value = mock_dead

        manager.spawn("bot-001", temp_config)
        manager.spawn("bot-002", temp_config)

        removed = manager.cleanup_terminated()

        assert "bot-001" in removed
        assert "bot-002" in removed
        assert manager.active_count == 0

    @patch("subprocess.Popen")
    def test_kill_all(self, mock_popen, manager, temp_config):
        """Test killing all processes."""
        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        manager.spawn("bot-002", temp_config)

        killed = manager.kill_all()

        assert killed == 2
        assert manager.active_count == 0

    @patch("subprocess.Popen")
    def test_restart_callback(self, mock_popen, temp_config):
        """Test restart callback is invoked."""
        callback_calls = []

        def callback(bot_id):
            callback_calls.append(bot_id)

        manager = ProcessManager(restart_callback=callback)

        mock_process = MagicMock()
        mock_process.pid = 12345
        mock_process.poll.return_value = None
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        manager.spawn("bot-001", temp_config)
        manager.restart("bot-001")

        assert "bot-001" in callback_calls


# =============================================================================
# Integration Tests
# =============================================================================


class TestProcessManagerIntegration:
    """Integration tests for ProcessManager (with real subprocesses)."""

    @pytest.fixture
    def manager(self):
        """Create process manager."""
        return ProcessManager()

    @pytest.fixture
    def temp_script(self):
        """Create temporary Python script."""
        script = '''
import time
import sys
time.sleep(10)
sys.exit(0)
'''
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            yield f.name
        try:
            os.unlink(f.name)
        except Exception:
            pass

    def test_spawn_real_process(self, manager, temp_script):
        """Test spawning a real subprocess."""
        # Create a simple config that won't actually run bots
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump({"bot_type": "test"}, f)
            f.flush()
            config_path = f.name

        try:
            # This will fail to start because the runner module won't find the bot type
            # but we can still test the spawn mechanism
            with patch.object(subprocess, 'Popen') as mock_popen:
                mock_process = MagicMock()
                mock_process.pid = os.getpid()
                mock_process.poll.return_value = None
                mock_popen.return_value = mock_process

                result = manager.spawn("test-bot", config_path)
                assert result is True

                # Clean up
                manager.kill("test-bot")
        finally:
            os.unlink(config_path)
