"""
Fund Manager Record Models.

Data models for tracking allocations and balance snapshots.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class BalanceSnapshot:
    """
    Snapshot of account balance at a point in time.

    Attributes:
        timestamp: When the snapshot was taken
        total_balance: Total account balance (USDT)
        available_balance: Available balance for allocation
        allocated_balance: Currently allocated to bots
        reserved_balance: Reserved funds (not for allocation)
    """

    timestamp: datetime
    total_balance: Decimal
    available_balance: Decimal
    allocated_balance: Decimal = Decimal("0")
    reserved_balance: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        """Convert values to Decimal if necessary."""
        if isinstance(self.total_balance, (int, float, str)):
            self.total_balance = Decimal(str(self.total_balance))
        if isinstance(self.available_balance, (int, float, str)):
            self.available_balance = Decimal(str(self.available_balance))
        if isinstance(self.allocated_balance, (int, float, str)):
            self.allocated_balance = Decimal(str(self.allocated_balance))
        if isinstance(self.reserved_balance, (int, float, str)):
            self.reserved_balance = Decimal(str(self.reserved_balance))

    @property
    def unallocated_balance(self) -> Decimal:
        """Get balance available for new allocations."""
        return self.available_balance - self.allocated_balance - self.reserved_balance

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_balance": str(self.total_balance),
            "available_balance": str(self.available_balance),
            "allocated_balance": str(self.allocated_balance),
            "reserved_balance": str(self.reserved_balance),
            "unallocated_balance": str(self.unallocated_balance),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BalanceSnapshot":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            timestamp=timestamp or datetime.now(timezone.utc),
            total_balance=Decimal(str(data["total_balance"])),
            available_balance=Decimal(str(data["available_balance"])),
            allocated_balance=Decimal(str(data.get("allocated_balance", "0"))),
            reserved_balance=Decimal(str(data.get("reserved_balance", "0"))),
        )


@dataclass
class AllocationRecord:
    """
    Record of a fund allocation to a bot.

    Attributes:
        id: Unique record identifier
        timestamp: When the allocation was made
        bot_id: Bot identifier
        amount: Amount allocated
        trigger: What triggered the allocation (manual, deposit, rebalance)
        previous_allocation: Previous allocation amount
        new_allocation: New allocation amount after this record
        success: Whether the allocation was successful
        error_message: Error message if failed
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    bot_id: str = ""
    amount: Decimal = Decimal("0")
    trigger: str = "manual"
    previous_allocation: Decimal = Decimal("0")
    new_allocation: Decimal = Decimal("0")
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Convert values to Decimal if necessary."""
        if isinstance(self.amount, (int, float, str)):
            self.amount = Decimal(str(self.amount))
        if isinstance(self.previous_allocation, (int, float, str)):
            self.previous_allocation = Decimal(str(self.previous_allocation))
        if isinstance(self.new_allocation, (int, float, str)):
            self.new_allocation = Decimal(str(self.new_allocation))

    @property
    def change(self) -> Decimal:
        """Calculate the change in allocation."""
        return self.new_allocation - self.previous_allocation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "bot_id": self.bot_id,
            "amount": str(self.amount),
            "trigger": self.trigger,
            "previous_allocation": str(self.previous_allocation),
            "new_allocation": str(self.new_allocation),
            "success": self.success,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationRecord":
        """Create from dictionary."""
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            timestamp=timestamp or datetime.now(timezone.utc),
            bot_id=data.get("bot_id", ""),
            amount=Decimal(str(data.get("amount", "0"))),
            trigger=data.get("trigger", "manual"),
            previous_allocation=Decimal(str(data.get("previous_allocation", "0"))),
            new_allocation=Decimal(str(data.get("new_allocation", "0"))),
            success=data.get("success", True),
            error_message=data.get("error_message"),
        )


@dataclass
class DispatchResult:
    """
    Result of a fund dispatch operation.

    Attributes:
        success: Whether the dispatch was successful
        timestamp: When the dispatch occurred
        trigger: What triggered the dispatch
        total_dispatched: Total amount dispatched
        allocations: List of individual allocation records
        errors: List of error messages
    """

    success: bool = True
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    trigger: str = "manual"
    total_dispatched: Decimal = Decimal("0")
    allocations: List[AllocationRecord] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert values to Decimal if necessary."""
        if isinstance(self.total_dispatched, (int, float, str)):
            self.total_dispatched = Decimal(str(self.total_dispatched))

    @property
    def successful_count(self) -> int:
        """Get count of successful allocations."""
        return sum(1 for a in self.allocations if a.success)

    @property
    def failed_count(self) -> int:
        """Get count of failed allocations."""
        return sum(1 for a in self.allocations if not a.success)

    def add_allocation(self, record: AllocationRecord) -> None:
        """Add an allocation record."""
        self.allocations.append(record)
        if record.success:
            self.total_dispatched += record.amount
        else:
            if record.error_message:
                self.errors.append(record.error_message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "timestamp": self.timestamp.isoformat(),
            "trigger": self.trigger,
            "total_dispatched": str(self.total_dispatched),
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "allocations": [a.to_dict() for a in self.allocations],
            "errors": self.errors,
        }
