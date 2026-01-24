"""
Fund Manager Configuration Models.

Defines configuration structures for fund allocation and distribution.
"""

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional


class AllocationStrategy(Enum):
    """Fund allocation strategy types."""

    FIXED_RATIO = "fixed_ratio"
    FIXED_AMOUNT = "fixed_amount"
    DYNAMIC_WEIGHT = "dynamic_weight"


@dataclass
class BotAllocation:
    """
    Configuration for a single bot's fund allocation.

    Attributes:
        bot_pattern: Glob pattern to match bot IDs (e.g., "bollinger_*", "grid_*")
        ratio: Allocation ratio (0.0 - 1.0) for fixed_ratio strategy
        fixed_amount: Fixed allocation amount for fixed_amount strategy
        min_capital: Minimum capital to allocate
        max_capital: Maximum capital to allocate
        priority: Priority for allocation (higher = first)
        enabled: Whether this allocation is active
    """

    bot_pattern: str
    ratio: Decimal = Decimal("0.0")
    fixed_amount: Optional[Decimal] = None
    min_capital: Decimal = Decimal("0")
    max_capital: Decimal = Decimal("1000000")
    priority: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        """Convert values to Decimal if necessary."""
        if isinstance(self.ratio, (int, float, str)):
            self.ratio = Decimal(str(self.ratio))
        if self.fixed_amount is not None and isinstance(
            self.fixed_amount, (int, float, str)
        ):
            self.fixed_amount = Decimal(str(self.fixed_amount))
        if isinstance(self.min_capital, (int, float, str)):
            self.min_capital = Decimal(str(self.min_capital))
        if isinstance(self.max_capital, (int, float, str)):
            self.max_capital = Decimal(str(self.max_capital))

    def matches(self, bot_id: str) -> bool:
        """
        Check if bot_id matches the pattern.

        Supports simple glob patterns with * wildcard.

        Args:
            bot_id: Bot identifier to check

        Returns:
            True if bot_id matches the pattern
        """
        import fnmatch

        return fnmatch.fnmatch(bot_id, self.bot_pattern)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_pattern": self.bot_pattern,
            "ratio": str(self.ratio),
            "fixed_amount": str(self.fixed_amount) if self.fixed_amount else None,
            "min_capital": str(self.min_capital),
            "max_capital": str(self.max_capital),
            "priority": self.priority,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BotAllocation":
        """Create from dictionary."""
        return cls(
            bot_pattern=data["bot_pattern"],
            ratio=Decimal(str(data.get("ratio", "0.0"))),
            fixed_amount=(
                Decimal(str(data["fixed_amount"]))
                if data.get("fixed_amount")
                else None
            ),
            min_capital=Decimal(str(data.get("min_capital", "0"))),
            max_capital=Decimal(str(data.get("max_capital", "1000000"))),
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
        )


@dataclass
class FundManagerConfig:
    """
    Fund Manager configuration.

    Attributes:
        enabled: Enable/disable the fund manager
        poll_interval: Interval in seconds to check for balance changes
        deposit_threshold: Minimum deposit amount to trigger auto-dispatch
        reserve_ratio: Ratio of total funds to keep in reserve (0.0 - 1.0)
        auto_dispatch: Automatically dispatch funds on deposit detection
        strategy: Default allocation strategy
        allocations: List of bot allocation configurations
    """

    enabled: bool = True
    poll_interval: int = 60
    deposit_threshold: Decimal = Decimal("10")
    reserve_ratio: Decimal = Decimal("0.1")
    auto_dispatch: bool = True
    strategy: AllocationStrategy = AllocationStrategy.FIXED_RATIO
    allocations: List[BotAllocation] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Convert values to proper types."""
        if isinstance(self.deposit_threshold, (int, float, str)):
            self.deposit_threshold = Decimal(str(self.deposit_threshold))
        if isinstance(self.reserve_ratio, (int, float, str)):
            self.reserve_ratio = Decimal(str(self.reserve_ratio))
        if isinstance(self.strategy, str):
            self.strategy = AllocationStrategy(self.strategy)

    def get_allocation_for_bot(self, bot_id: str) -> Optional[BotAllocation]:
        """
        Get allocation config for a specific bot.

        Args:
            bot_id: Bot identifier

        Returns:
            BotAllocation if found, None otherwise
        """
        # Sort by priority (higher first) and return first match
        sorted_allocations = sorted(
            self.allocations, key=lambda x: x.priority, reverse=True
        )
        for alloc in sorted_allocations:
            if alloc.enabled and alloc.matches(bot_id):
                return alloc
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "enabled": self.enabled,
            "poll_interval": self.poll_interval,
            "deposit_threshold": str(self.deposit_threshold),
            "reserve_ratio": str(self.reserve_ratio),
            "auto_dispatch": self.auto_dispatch,
            "strategy": self.strategy.value,
            "allocations": [a.to_dict() for a in self.allocations],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FundManagerConfig":
        """Create from dictionary."""
        allocations = [
            BotAllocation.from_dict(a) for a in data.get("allocations", [])
        ]
        return cls(
            enabled=data.get("enabled", True),
            poll_interval=data.get("poll_interval", 60),
            deposit_threshold=Decimal(str(data.get("deposit_threshold", "10"))),
            reserve_ratio=Decimal(str(data.get("reserve_ratio", "0.1"))),
            auto_dispatch=data.get("auto_dispatch", True),
            strategy=AllocationStrategy(data.get("strategy", "fixed_ratio")),
            allocations=allocations,
        )

    @classmethod
    def from_yaml(cls, yaml_dict: Dict[str, Any]) -> "FundManagerConfig":
        """
        Create from YAML config dictionary.

        The YAML structure has nested keys like:
        - system.poll_interval
        - strategy.type
        - bots (list)

        Args:
            yaml_dict: Dictionary from YAML config file

        Returns:
            FundManagerConfig instance
        """
        # Extract from nested YAML structure
        system = yaml_dict.get("system", {})
        strategy_section = yaml_dict.get("strategy", {})
        bots = yaml_dict.get("bots", [])

        # Convert bots to allocation format
        allocations = []
        for bot in bots:
            if bot.get("status", "active") == "active":
                allocations.append({
                    "bot_pattern": bot.get("bot_id", ""),
                    "ratio": bot.get("ratio", 0),
                    "min_capital": bot.get("min_capital", 0),
                    "max_capital": bot.get("max_capital", 1000000),
                    "priority": bot.get("priority", 0),
                    "enabled": True,
                })

        # Build flat dict for from_dict
        flat_dict = {
            "enabled": True,
            "poll_interval": system.get("poll_interval", 60),
            "deposit_threshold": system.get("min_allocation_unit", 10),
            "reserve_ratio": system.get("reserve_ratio", 0.1),
            "auto_dispatch": strategy_section.get("auto_dispatch", True),
            "strategy": strategy_section.get("type", "fixed_ratio"),
            "allocations": allocations,
        }

        return cls.from_dict(flat_dict)
