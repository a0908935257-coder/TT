"""
Fund Manager Core Components.

Core functionality for fund monitoring, allocation, and position management.
"""

from .allocator import (
    BaseAllocator,
    DynamicWeightAllocator,
    FixedAmountAllocator,
    FixedRatioAllocator,
    create_allocator,
)
from .dispatcher import Dispatcher
from .fund_pool import FundPool
from .position_manager import (
    PositionChange,
    PositionSide,
    SharedPosition,
    SharedPositionManager,
)
from .signal_coordinator import (
    ConflictRecord,
    ConflictResolution,
    ConflictType,
    CoordinationResult,
    SignalCoordinator,
    SignalDirection,
    SignalRequest,
)

__all__ = [
    # Fund Pool
    "FundPool",
    # Allocators
    "BaseAllocator",
    "FixedRatioAllocator",
    "FixedAmountAllocator",
    "DynamicWeightAllocator",
    "create_allocator",
    # Dispatcher
    "Dispatcher",
    # Position Manager
    "SharedPositionManager",
    "SharedPosition",
    "PositionChange",
    "PositionSide",
    # Signal Coordinator
    "SignalCoordinator",
    "SignalDirection",
    "SignalRequest",
    "CoordinationResult",
    "ConflictRecord",
    "ConflictResolution",
    "ConflictType",
]
