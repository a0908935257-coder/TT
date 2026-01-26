"""
Process Manager.

Manages Bot subprocess lifecycle (spawn, monitor, restart, kill).
"""

import os
import signal
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from src.core import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessInfo:
    """
    Information about a bot process.

    Attributes:
        bot_id: Bot identifier
        config_path: Path to bot configuration file
        process: Subprocess Popen object
        started_at: When the process was started
        restart_count: Number of times this process has been restarted
    """

    bot_id: str
    config_path: str
    process: subprocess.Popen
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    restart_count: int = 0

    @property
    def pid(self) -> int:
        """Get process ID."""
        return self.process.pid

    @property
    def is_alive(self) -> bool:
        """Check if process is still running."""
        return self.process.poll() is None

    @property
    def return_code(self) -> Optional[int]:
        """Get return code if process has terminated."""
        return self.process.poll()


class ProcessManager:
    """
    Manages Bot subprocess lifecycle.

    Responsibilities:
    - Spawn new bot processes
    - Monitor process health
    - Restart failed processes (with configurable policy)
    - Kill processes

    Example:
        manager = ProcessManager()
        manager.spawn("bot-001", "config/bot_001.yaml")
        if not manager.is_alive("bot-001"):
            manager.restart("bot-001")
        manager.kill("bot-001")
    """

    def __init__(
        self,
        max_restart_attempts: int = 3,
        restart_callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize ProcessManager.

        Args:
            max_restart_attempts: Maximum restart attempts before giving up
            restart_callback: Callback when a process is restarted
        """
        self._processes: Dict[str, ProcessInfo] = {}
        self._max_restart_attempts = max_restart_attempts
        self._restart_callback = restart_callback

    @property
    def active_count(self) -> int:
        """Get number of active processes."""
        return sum(1 for p in self._processes.values() if p.is_alive)

    @property
    def all_bot_ids(self) -> List[str]:
        """Get list of all managed bot IDs."""
        return list(self._processes.keys())

    def spawn(self, bot_id: str, config_path: str) -> bool:
        """
        Spawn a new bot subprocess.

        Args:
            bot_id: Unique bot identifier
            config_path: Path to bot configuration file

        Returns:
            True if spawned successfully, False if already exists
        """
        # Check if already running
        if bot_id in self._processes and self._processes[bot_id].is_alive:
            logger.warning(f"Bot {bot_id} is already running (PID: {self._processes[bot_id].pid})")
            return False

        # Validate config path
        if not Path(config_path).exists():
            logger.error(f"Config file not found: {config_path}")
            return False

        try:
            # Build command
            cmd = [
                sys.executable,
                "-m", "src.bots.runner",
                "--bot-id", bot_id,
                "--config", config_path,
            ]

            logger.info(f"Spawning bot process: {bot_id}")
            logger.debug(f"Command: {' '.join(cmd)}")

            # Spawn process
            # Use DEVNULL to prevent buffer overflow hang
            # Bot logs are handled by the bot's own logging system
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                # Don't inherit signal handlers
                start_new_session=True,
            )

            # Store process info
            self._processes[bot_id] = ProcessInfo(
                bot_id=bot_id,
                config_path=config_path,
                process=process,
            )

            logger.info(f"Bot process spawned: {bot_id} (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"Failed to spawn bot {bot_id}: {e}")
            return False

    def kill(self, bot_id: str, timeout: float = 10.0) -> bool:
        """
        Kill a bot subprocess.

        First sends SIGTERM for graceful shutdown, then SIGKILL if needed.

        Args:
            bot_id: Bot identifier
            timeout: Timeout in seconds to wait for graceful shutdown

        Returns:
            True if killed successfully, False if not found
        """
        process_info = self._processes.get(bot_id)
        if not process_info:
            logger.warning(f"Bot {bot_id} not found in process manager")
            return False

        if not process_info.is_alive:
            logger.info(f"Bot {bot_id} already terminated")
            del self._processes[bot_id]
            return True

        try:
            logger.info(f"Killing bot process: {bot_id} (PID: {process_info.pid})")

            # Try graceful termination first
            process_info.process.terminate()

            try:
                process_info.process.wait(timeout=timeout)
                logger.info(f"Bot {bot_id} terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill
                logger.warning(f"Bot {bot_id} did not terminate gracefully, sending SIGKILL")
                process_info.process.kill()
                process_info.process.wait(timeout=5)

            del self._processes[bot_id]
            return True

        except Exception as e:
            logger.error(f"Error killing bot {bot_id}: {e}")
            return False

    def is_alive(self, bot_id: str) -> bool:
        """
        Check if a bot process is alive.

        Args:
            bot_id: Bot identifier

        Returns:
            True if process exists and is alive
        """
        process_info = self._processes.get(bot_id)
        return process_info is not None and process_info.is_alive

    def get_pid(self, bot_id: str) -> Optional[int]:
        """
        Get process ID for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            Process ID or None if not found
        """
        process_info = self._processes.get(bot_id)
        return process_info.pid if process_info else None

    def get_info(self, bot_id: str) -> Optional[ProcessInfo]:
        """
        Get process information for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            ProcessInfo or None if not found
        """
        return self._processes.get(bot_id)

    def restart(self, bot_id: str) -> bool:
        """
        Restart a bot subprocess.

        Args:
            bot_id: Bot identifier

        Returns:
            True if restarted successfully
        """
        process_info = self._processes.get(bot_id)
        if not process_info:
            logger.warning(f"Bot {bot_id} not found for restart")
            return False

        # Check restart limit
        if process_info.restart_count >= self._max_restart_attempts:
            logger.error(f"Bot {bot_id} exceeded max restart attempts ({self._max_restart_attempts})")
            return False

        config_path = process_info.config_path
        restart_count = process_info.restart_count

        # Kill existing process
        self.kill(bot_id)

        # Spawn new process
        if self.spawn(bot_id, config_path):
            self._processes[bot_id].restart_count = restart_count + 1
            logger.info(f"Bot {bot_id} restarted (attempt {restart_count + 1})")

            # Invoke callback
            if self._restart_callback:
                try:
                    self._restart_callback(bot_id)
                except Exception as e:
                    logger.error(f"Restart callback error: {e}")

            return True

        return False

    def check_health(self) -> Dict[str, bool]:
        """
        Check health of all managed processes.

        Returns:
            Dictionary mapping bot_id to alive status
        """
        return {bot_id: info.is_alive for bot_id, info in self._processes.items()}

    def get_failed_processes(self) -> List[str]:
        """
        Get list of bot IDs with failed processes.

        Returns:
            List of bot IDs that have terminated unexpectedly
        """
        failed = []
        for bot_id, info in self._processes.items():
            if not info.is_alive:
                failed.append(bot_id)
        return failed

    def cleanup_terminated(self) -> List[str]:
        """
        Remove terminated processes from tracking.

        Returns:
            List of removed bot IDs
        """
        removed = []
        for bot_id in list(self._processes.keys()):
            if not self._processes[bot_id].is_alive:
                del self._processes[bot_id]
                removed.append(bot_id)
        return removed

    def kill_all(self, timeout: float = 10.0) -> int:
        """
        Kill all managed processes.

        Args:
            timeout: Timeout per process

        Returns:
            Number of processes killed
        """
        killed = 0
        for bot_id in list(self._processes.keys()):
            if self.kill(bot_id, timeout):
                killed += 1
        return killed

    def get_output(self, bot_id: str) -> tuple[Optional[bytes], Optional[bytes]]:
        """
        Get stdout/stderr from a terminated process.

        Args:
            bot_id: Bot identifier

        Returns:
            Tuple of (stdout, stderr) or (None, None) if not found
        """
        process_info = self._processes.get(bot_id)
        if not process_info:
            return None, None

        if process_info.is_alive:
            logger.warning(f"Bot {bot_id} is still running, cannot get output")
            return None, None

        try:
            stdout, stderr = process_info.process.communicate(timeout=1)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            return None, None
