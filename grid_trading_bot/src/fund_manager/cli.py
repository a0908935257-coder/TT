"""
Fund Manager CLI.

Command-line interface for managing fund allocations.
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.core import get_logger

from .manager import FundManager
from .models.config import FundManagerConfig
from .models.records import DispatchResult
from .storage.repository import AllocationRepository

logger = get_logger(__name__)


class FundManagerCLI:
    """
    Command-line interface for Fund Manager.

    Provides commands for managing fund allocations, viewing status,
    and monitoring history.

    Example:
        >>> cli = FundManagerCLI(fund_manager, repository)
        >>> await cli.run(["status"])
        >>> await cli.run(["allocate"])
    """

    def __init__(
        self,
        fund_manager: Optional[FundManager] = None,
        repository: Optional[AllocationRepository] = None,
    ):
        """
        Initialize CLI.

        Args:
            fund_manager: FundManager instance
            repository: AllocationRepository instance
        """
        self._fund_manager = fund_manager
        self._repository = repository
        self._parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser."""
        parser = argparse.ArgumentParser(
            prog="fund-manager",
            description="Fund Manager CLI for managing capital allocation",
        )
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        # start command
        start_parser = subparsers.add_parser("start", help="Start fund manager service")
        start_parser.add_argument(
            "--config", "-c",
            type=str,
            help="Path to configuration file",
        )

        # stop command
        subparsers.add_parser("stop", help="Stop fund manager service")

        # status command
        subparsers.add_parser("status", help="Show current status")

        # allocate command
        allocate_parser = subparsers.add_parser(
            "allocate",
            help="Manually trigger fund allocation",
        )
        allocate_parser.add_argument(
            "--trigger", "-t",
            type=str,
            default="manual",
            choices=["manual", "rebalance"],
            help="Allocation trigger type",
        )

        # set-ratio command
        ratio_parser = subparsers.add_parser(
            "set-ratio",
            help="Adjust allocation ratio for a bot pattern",
        )
        ratio_parser.add_argument("pattern", type=str, help="Bot pattern (e.g., grid_*)")
        ratio_parser.add_argument("ratio", type=float, help="New ratio (0.0 - 1.0)")

        # history command
        history_parser = subparsers.add_parser(
            "history",
            help="View allocation history",
        )
        history_parser.add_argument(
            "--bot", "-b",
            type=str,
            help="Filter by bot ID",
        )
        history_parser.add_argument(
            "--trigger", "-t",
            type=str,
            help="Filter by trigger type",
        )
        history_parser.add_argument(
            "--days", "-d",
            type=int,
            default=7,
            help="Number of days to look back (default: 7)",
        )
        history_parser.add_argument(
            "--limit", "-l",
            type=int,
            default=20,
            help="Maximum records to show (default: 20)",
        )

        # stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="View allocation statistics",
        )
        stats_parser.add_argument(
            "--days", "-d",
            type=int,
            default=30,
            help="Number of days for statistics (default: 30)",
        )

        # pool command
        subparsers.add_parser("pool", help="Show fund pool status")

        # bots command
        subparsers.add_parser("bots", help="List configured bot allocations")

        return parser

    async def run(self, args: List[str]) -> int:
        """
        Run CLI command.

        Args:
            args: Command-line arguments

        Returns:
            Exit code (0 for success)
        """
        if not args:
            self._parser.print_help()
            return 0

        parsed = self._parser.parse_args(args)

        if not parsed.command:
            self._parser.print_help()
            return 0

        try:
            handler = getattr(self, f"_cmd_{parsed.command.replace('-', '_')}", None)
            if handler:
                return await handler(parsed)
            else:
                print(f"Unknown command: {parsed.command}")
                return 1
        except Exception as e:
            print(f"Error: {e}")
            logger.error(f"CLI error: {e}")
            return 1

    # =========================================================================
    # Command Handlers
    # =========================================================================

    async def _cmd_start(self, args: argparse.Namespace) -> int:
        """Handle start command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        if self._fund_manager.is_running:
            print("FundManager is already running")
            return 0

        print("Starting FundManager...")
        await self._fund_manager.start()
        print("FundManager started successfully")
        return 0

    async def _cmd_stop(self, args: argparse.Namespace) -> int:
        """Handle stop command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        if not self._fund_manager.is_running:
            print("FundManager is not running")
            return 0

        print("Stopping FundManager...")
        await self._fund_manager.stop()
        print("FundManager stopped")
        return 0

    async def _cmd_status(self, args: argparse.Namespace) -> int:
        """Handle status command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        status = self._fund_manager.get_status()
        self._print_status(status)
        return 0

    async def _cmd_allocate(self, args: argparse.Namespace) -> int:
        """Handle allocate command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        if not self._fund_manager.is_running:
            print("Warning: FundManager is not running")

        print(f"Dispatching funds (trigger: {args.trigger})...")
        result = await self._fund_manager.dispatch_funds(trigger=args.trigger)
        self._print_dispatch_result(result)
        return 0 if result.success else 1

    async def _cmd_set_ratio(self, args: argparse.Namespace) -> int:
        """Handle set-ratio command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        if args.ratio < 0 or args.ratio > 1:
            print("Error: Ratio must be between 0.0 and 1.0")
            return 1

        success = self._fund_manager.adjust_ratio(
            args.pattern,
            Decimal(str(args.ratio)),
        )
        if success:
            print(f"Set ratio for {args.pattern} to {args.ratio:.2%}")
            return 0
        else:
            print(f"Error: Pattern {args.pattern} not found in config")
            return 1

    async def _cmd_history(self, args: argparse.Namespace) -> int:
        """Handle history command."""
        if not self._repository:
            print("Error: Repository not initialized")
            return 1

        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        records = self._repository.get_allocation_history(
            bot_id=args.bot,
            trigger=args.trigger,
            since=since,
            limit=args.limit,
        )

        if not records:
            print("No allocation records found")
            return 0

        self._print_allocation_history(records)
        return 0

    async def _cmd_stats(self, args: argparse.Namespace) -> int:
        """Handle stats command."""
        if not self._repository:
            print("Error: Repository not initialized")
            return 1

        since = datetime.now(timezone.utc) - timedelta(days=args.days)
        stats = self._repository.get_statistics(since=since)
        self._print_statistics(stats, args.days)
        return 0

    async def _cmd_pool(self, args: argparse.Namespace) -> int:
        """Handle pool command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        pool_status = self._fund_manager.fund_pool.get_status()
        self._print_pool_status(pool_status)
        return 0

    async def _cmd_bots(self, args: argparse.Namespace) -> int:
        """Handle bots command."""
        if not self._fund_manager:
            print("Error: FundManager not initialized")
            return 1

        self._print_bot_allocations(self._fund_manager.config.allocations)
        return 0

    # =========================================================================
    # Output Formatting
    # =========================================================================

    def _print_status(self, status: Dict[str, Any]) -> None:
        """Print status."""
        print("\n=== Fund Manager Status ===")
        print(f"Running:       {'Yes' if status['running'] else 'No'}")
        print(f"Enabled:       {'Yes' if status['enabled'] else 'No'}")
        print(f"Strategy:      {status['strategy']}")
        print(f"Auto Dispatch: {'Yes' if status['auto_dispatch'] else 'No'}")
        print(f"Poll Interval: {status['poll_interval']}s")
        print(f"Total Records: {status['allocation_count']}")

        if "fund_pool" in status:
            pool = status["fund_pool"]
            print("\n--- Fund Pool ---")
            print(f"Total Balance:      {pool['total_balance']} USDT")
            print(f"Available Balance:  {pool['available_balance']} USDT")
            print(f"Allocated Balance:  {pool['allocated_balance']} USDT")
            print(f"Reserved Balance:   {pool['reserved_balance']} USDT")
            print(f"Unallocated:        {pool['unallocated_balance']} USDT")

    def _print_dispatch_result(self, result: DispatchResult) -> None:
        """Print dispatch result."""
        print("\n=== Dispatch Result ===")
        print(f"Success:    {'Yes' if result.success else 'No'}")
        print(f"Trigger:    {result.trigger}")
        print(f"Total:      {result.total_dispatched} USDT")
        print(f"Successful: {result.successful_count}")
        print(f"Failed:     {result.failed_count}")

        if result.allocations:
            print("\n--- Allocations ---")
            for alloc in result.allocations:
                status = "OK" if alloc.success else "FAILED"
                print(
                    f"  {alloc.bot_id}: {alloc.amount} USDT "
                    f"({alloc.previous_allocation} -> {alloc.new_allocation}) "
                    f"[{status}]"
                )

        if result.errors:
            print("\n--- Errors ---")
            for error in result.errors:
                print(f"  - {error}")

    def _print_allocation_history(self, records: list) -> None:
        """Print allocation history."""
        print("\n=== Allocation History ===")
        print(f"{'Timestamp':<20} {'Bot ID':<15} {'Amount':>12} {'Trigger':<10} {'Status':<8}")
        print("-" * 70)

        for record in records:
            timestamp = record.timestamp.strftime("%Y-%m-%d %H:%M")
            status = "OK" if record.success else "FAILED"
            print(
                f"{timestamp:<20} {record.bot_id:<15} "
                f"{str(record.amount):>12} {record.trigger:<10} {status:<8}"
            )

    def _print_statistics(self, stats: Dict[str, Any], days: int) -> None:
        """Print statistics."""
        print(f"\n=== Allocation Statistics (Last {days} Days) ===")
        print(f"Total Allocations:     {stats['total_allocations']}")
        print(f"Successful:            {stats['successful_allocations']}")
        print(f"Failed:                {stats['failed_allocations']}")
        print(f"Total Amount:          {stats['total_amount_allocated']:.2f} USDT")
        print(f"Unique Bots:           {stats['unique_bots']}")

        if stats.get("by_trigger"):
            print("\n--- By Trigger ---")
            for trigger, count in stats["by_trigger"].items():
                print(f"  {trigger}: {count}")

    def _print_pool_status(self, pool: Dict[str, Any]) -> None:
        """Print pool status."""
        print("\n=== Fund Pool Status ===")
        print(f"Total Balance:      {pool['total_balance']} USDT")
        print(f"Available Balance:  {pool['available_balance']} USDT")
        print(f"Allocated Balance:  {pool['allocated_balance']} USDT")
        print(f"Reserved Balance:   {pool['reserved_balance']} USDT")
        print(f"Unallocated:        {pool['unallocated_balance']} USDT")
        print(f"Allocation Count:   {pool['allocation_count']}")
        print(f"Deposit Detected:   {'Yes' if pool['deposit_detected'] else 'No'}")

        if pool["last_update"]:
            print(f"Last Update:        {pool['last_update']}")

    def _print_bot_allocations(self, allocations: list) -> None:
        """Print bot allocation configurations."""
        print("\n=== Bot Allocation Configurations ===")
        print(f"{'Pattern':<20} {'Ratio':>8} {'Min':>10} {'Max':>12} {'Priority':>8} {'Enabled':<8}")
        print("-" * 75)

        for alloc in allocations:
            enabled = "Yes" if alloc.enabled else "No"
            print(
                f"{alloc.bot_pattern:<20} {float(alloc.ratio):>7.1%} "
                f"{str(alloc.min_capital):>10} {str(alloc.max_capital):>12} "
                f"{alloc.priority:>8} {enabled:<8}"
            )


def create_cli(
    fund_manager: Optional[FundManager] = None,
    repository: Optional[AllocationRepository] = None,
) -> FundManagerCLI:
    """
    Create CLI instance.

    Args:
        fund_manager: FundManager instance
        repository: AllocationRepository instance

    Returns:
        FundManagerCLI instance
    """
    return FundManagerCLI(fund_manager, repository)


async def main(args: Optional[List[str]] = None) -> int:
    """
    Main entry point for CLI.

    Args:
        args: Command-line arguments (defaults to sys.argv)

    Returns:
        Exit code
    """
    if args is None:
        args = sys.argv[1:]

    # Get or create instances
    fund_manager = FundManager.get_instance()
    repository = AllocationRepository()
    repository.initialize()

    cli = FundManagerCLI(fund_manager, repository)
    return await cli.run(args)


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
