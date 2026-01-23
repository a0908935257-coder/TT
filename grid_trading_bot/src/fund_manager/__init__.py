"""
Fund Manager Module.

Provides centralized fund allocation and distribution across trading bots.
Monitors total capital pool and automatically allocates funds based on
configured strategies.
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
from .manager import FundManager
from .models.config import AllocationStrategy, BotAllocation, FundManagerConfig
from .models.records import AllocationRecord, BalanceSnapshot, DispatchResult
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
    # Storage
    "AllocationRepository",
]
