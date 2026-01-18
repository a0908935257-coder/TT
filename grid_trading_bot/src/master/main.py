"""
Master Process Entry Point.

Starts the Master Control Console with IPC and Process Management.

Usage:
    python -m src.master.main --redis redis://localhost:6379
"""

import argparse
import asyncio
import logging
import signal
import sys
from typing import Optional

from src.core import get_logger

logger = get_logger(__name__)


async def create_redis_pool(redis_url: str):
    """
    Create Redis connection pool.

    Args:
        redis_url: Redis connection URL

    Returns:
        Redis async client
    """
    try:
        import redis.asyncio as redis
        return redis.from_url(redis_url)
    except ImportError:
        logger.error("redis package not installed. Install with: pip install redis")
        raise


async def run_master(
    redis_url: str = "redis://localhost:6379",
    restore_bots: bool = True,
) -> None:
    """
    Run the Master Control Console.

    Args:
        redis_url: Redis connection URL
        restore_bots: Whether to restore bots from database on start
    """
    from src.master import Master, MasterConfig
    from src.master.ipc_handler import MasterIPCHandler
    from src.master.process_manager import ProcessManager

    redis = None
    master = None
    ipc_handler = None
    process_manager = None
    shutdown_event = asyncio.Event()

    def signal_handler(sig):
        logger.info(f"Received signal {sig}, initiating shutdown")
        shutdown_event.set()

    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            signal.signal(sig, lambda s, f: signal_handler(s))

    try:
        logger.info("Starting Master Control Console")

        # Create Redis connection
        redis = await create_redis_pool(redis_url)
        logger.info(f"Connected to Redis: {redis_url}")

        # Initialize Master
        config = MasterConfig(
            auto_restart=True,
            restore_on_start=restore_bots,
        )
        master = Master(config=config)

        # Initialize IPC Handler
        ipc_handler = MasterIPCHandler(
            redis=redis,
            registry=master.registry,
            heartbeat_monitor=master.heartbeat_monitor,
        )

        # Initialize Process Manager
        process_manager = ProcessManager(
            max_restart_attempts=3,
        )

        # Inject IPC and Process Manager into Master
        # (These would be used by Commander for actual bot control)
        master._ipc_handler = ipc_handler
        master._process_manager = process_manager

        # Start IPC Handler
        await ipc_handler.start()
        logger.info("IPC handler started")

        # Start Master
        await master.start()
        logger.info("Master started")

        # Wait for shutdown signal
        logger.info("Master Control Console running. Press Ctrl+C to stop.")
        await shutdown_event.wait()

    except Exception as e:
        logger.error(f"Master error: {e}")
        raise

    finally:
        logger.info("Shutting down Master Control Console")

        # Stop Master
        if master:
            try:
                await master.stop()
            except Exception as e:
                logger.error(f"Error stopping master: {e}")

        # Stop IPC Handler
        if ipc_handler:
            try:
                await ipc_handler.stop()
            except Exception as e:
                logger.error(f"Error stopping IPC handler: {e}")

        # Kill all bot processes
        if process_manager:
            try:
                killed = process_manager.kill_all()
                logger.info(f"Killed {killed} bot processes")
            except Exception as e:
                logger.error(f"Error killing processes: {e}")

        # Close Redis connection
        if redis:
            try:
                await redis.close()
            except Exception as e:
                logger.error(f"Error closing Redis: {e}")

        logger.info("Master Control Console shutdown complete")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run the Master Control Console"
    )
    parser.add_argument(
        "--redis",
        default="redis://localhost:6379",
        help="Redis connection URL (default: redis://localhost:6379)",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Don't restore bots from database on start",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging
    if args.debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # Run master
    try:
        asyncio.run(run_master(
            redis_url=args.redis,
            restore_bots=not args.no_restore,
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Master failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
