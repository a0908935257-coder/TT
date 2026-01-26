"""
Fund Manager Record Models.

Data models for tracking allocations and balance snapshots.
Includes transaction models for atomic allocation operations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid


class TransactionStatus(Enum):
    """
    Status of an allocation transaction.

    Lifecycle: PENDING -> EXECUTING -> COMMITTED or ROLLED_BACK or FAILED
    """

    PENDING = "pending"          # Transaction created, not started
    EXECUTING = "executing"      # Transaction in progress
    COMMITTED = "committed"      # Transaction completed successfully
    ROLLED_BACK = "rolled_back"  # Transaction rolled back
    FAILED = "failed"            # Transaction failed (cannot rollback)


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


@dataclass
class AllocationTransaction:
    """
    Atomic allocation transaction.

    Tracks a batch allocation operation with pre-state snapshot for rollback.
    Ensures all-or-nothing semantics for fund distribution.

    Attributes:
        transaction_id: Unique transaction identifier
        status: Current transaction status
        trigger: What triggered the allocation
        created_at: When transaction was created
        completed_at: When transaction completed (success or failure)
        planned_allocations: Bot allocations planned for this transaction
        pre_state_snapshot: State snapshot before transaction (for rollback)
        executed_allocations: Actually executed allocations
        error_message: Error message if failed
    """

    transaction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: TransactionStatus = TransactionStatus.PENDING
    trigger: str = "manual"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    planned_allocations: Dict[str, Decimal] = field(default_factory=dict)
    pre_state_snapshot: Dict[str, Decimal] = field(default_factory=dict)
    executed_allocations: List[AllocationRecord] = field(default_factory=list)
    error_message: Optional[str] = None

    def __post_init__(self) -> None:
        """Convert values if necessary."""
        # Convert planned allocations to Decimal
        self.planned_allocations = {
            k: Decimal(str(v)) if not isinstance(v, Decimal) else v
            for k, v in self.planned_allocations.items()
        }
        # Convert snapshot to Decimal
        self.pre_state_snapshot = {
            k: Decimal(str(v)) if not isinstance(v, Decimal) else v
            for k, v in self.pre_state_snapshot.items()
        }

    @property
    def is_pending(self) -> bool:
        """Check if transaction is pending."""
        return self.status == TransactionStatus.PENDING

    @property
    def is_executing(self) -> bool:
        """Check if transaction is executing."""
        return self.status == TransactionStatus.EXECUTING

    @property
    def is_completed(self) -> bool:
        """Check if transaction is completed (success or failure)."""
        return self.status in (
            TransactionStatus.COMMITTED,
            TransactionStatus.ROLLED_BACK,
            TransactionStatus.FAILED,
        )

    @property
    def is_successful(self) -> bool:
        """Check if transaction completed successfully."""
        return self.status == TransactionStatus.COMMITTED

    @property
    def successful_count(self) -> int:
        """Get count of successful allocations."""
        return sum(1 for a in self.executed_allocations if a.success)

    @property
    def failed_count(self) -> int:
        """Get count of failed allocations."""
        return sum(1 for a in self.executed_allocations if not a.success)

    @property
    def total_allocated(self) -> Decimal:
        """Get total amount successfully allocated."""
        return sum(
            a.amount for a in self.executed_allocations if a.success
        )

    def mark_executing(self) -> None:
        """Mark transaction as executing."""
        self.status = TransactionStatus.EXECUTING

    def mark_committed(self) -> None:
        """Mark transaction as committed."""
        self.status = TransactionStatus.COMMITTED
        self.completed_at = datetime.now(timezone.utc)

    def mark_rolled_back(self, error: Optional[str] = None) -> None:
        """Mark transaction as rolled back."""
        self.status = TransactionStatus.ROLLED_BACK
        self.completed_at = datetime.now(timezone.utc)
        if error:
            self.error_message = error

    def mark_failed(self, error: str) -> None:
        """Mark transaction as failed."""
        self.status = TransactionStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error_message = error

    def add_executed(self, record: AllocationRecord) -> None:
        """Add an executed allocation record."""
        self.executed_allocations.append(record)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "transaction_id": self.transaction_id,
            "status": self.status.value,
            "trigger": self.trigger,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "planned_allocations": {k: str(v) for k, v in self.planned_allocations.items()},
            "pre_state_snapshot": {k: str(v) for k, v in self.pre_state_snapshot.items()},
            "executed_allocations": [a.to_dict() for a in self.executed_allocations],
            "successful_count": self.successful_count,
            "failed_count": self.failed_count,
            "total_allocated": str(self.total_allocated),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AllocationTransaction":
        """Create from dictionary."""
        created_at = data.get("created_at")
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        completed_at = data.get("completed_at")
        if isinstance(completed_at, str):
            completed_at = datetime.fromisoformat(completed_at)

        return cls(
            transaction_id=data.get("transaction_id", str(uuid.uuid4())),
            status=TransactionStatus(data.get("status", "pending")),
            trigger=data.get("trigger", "manual"),
            created_at=created_at or datetime.now(timezone.utc),
            completed_at=completed_at,
            planned_allocations={
                k: Decimal(str(v))
                for k, v in data.get("planned_allocations", {}).items()
            },
            pre_state_snapshot={
                k: Decimal(str(v))
                for k, v in data.get("pre_state_snapshot", {}).items()
            },
            executed_allocations=[
                AllocationRecord.from_dict(a)
                for a in data.get("executed_allocations", [])
            ],
            error_message=data.get("error_message"),
        )
