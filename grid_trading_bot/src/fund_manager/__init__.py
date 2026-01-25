"""
Fund Manager Module.

Provides centralized fund allocation and distribution across trading bots.
Monitors total capital pool and automatically allocates funds based on
configured strategies.

Includes:
- FundManager: Central fund allocation management
- SharedPositionManager: Cross-bot position tracking
- Thread-safe atomic operations for concurrent access
"""

from .cli import FundManagerCLI, create_cli
from .core.allocator import (
    BaseAllocator,
    DynamicWeightAllocator,
    FixedAmountAllocator,
    FixedRatioAllocator,
    create_allocator,
)
from .core.dispatcher import Dispatcher
from .core.fund_pool import FundPool
from .core.position_manager import (
    PositionChange,
    PositionSide,
    SharedPosition,
    SharedPositionManager,
)
from .manager import FundManager
from .models.config import AllocationStrategy, BotAllocation, FundManagerConfig
from .models.records import AllocationRecord, BalanceSnapshot, DispatchResult
from .notifier.bot_notifier import ApiNotifier, BotNotifier, FileNotifier, NotificationMessage
from .storage.repository import AllocationRepository

__all__ = [
    # Manager
    "FundManager",
    # CLI
    "FundManagerCLI",
    "create_cli",
    # Core
    "FundPool",
    "Dispatcher",
    # Position Manager
    "SharedPositionManager",
    "SharedPosition",
    "PositionChange",
    "PositionSide",
    # Allocators
    "BaseAllocator",
    "FixedRatioAllocator",
    "FixedAmountAllocator",
    "DynamicWeightAllocator",
    "create_allocator",
    # Models
    "FundManagerConfig",
    "BotAllocation",
    "AllocationStrategy",
    "AllocationRecord",
    "BalanceSnapshot",
    "DispatchResult",
    # Notifier
    "BotNotifier",
    "FileNotifier",
    "ApiNotifier",
    "NotificationMessage",
    # Storage
    "AllocationRepository",
]
