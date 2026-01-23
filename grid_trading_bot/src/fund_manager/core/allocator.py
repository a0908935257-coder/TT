"""
Fund Allocation Strategies.

Implements different strategies for allocating funds to trading bots.
"""

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Any, Dict, List, Protocol, runtime_checkable

from src.core import get_logger

from ..models.config import BotAllocation, FundManagerConfig

logger = get_logger(__name__)


@runtime_checkable
class AllocationStrategy(Protocol):
    """Protocol for allocation strategies."""

    def calculate(
        self,
        available_funds: Decimal,
        bot_allocations: List[BotAllocation],
        current_allocations: Dict[str, Decimal],
        bot_ids: List[str],
    ) -> Dict[str, Decimal]:
        """
        Calculate fund allocations for bots.

        Args:
            available_funds: Total funds available for allocation
            bot_allocations: Configuration for each bot allocation
            current_allocations: Current allocations by bot_id
            bot_ids: List of active bot IDs

        Returns:
            Dictionary mapping bot_id to allocation amount
        """
        ...


class BaseAllocator(ABC):
    """Base class for allocation strategies."""

    def __init__(self, config: FundManagerConfig):
        """
        Initialize allocator.

        Args:
            config: Fund manager configuration
        """
        self._config = config

    @abstractmethod
    def calculate(
        self,
        available_funds: Decimal,
        bot_allocations: List[BotAllocation],
        current_allocations: Dict[str, Decimal],
        bot_ids: List[str],
    ) -> Dict[str, Decimal]:
        """
        Calculate fund allocations for bots.

        Args:
            available_funds: Total funds available for allocation
            bot_allocations: Configuration for each bot allocation
            current_allocations: Current allocations by bot_id
            bot_ids: List of active bot IDs

        Returns:
            Dictionary mapping bot_id to allocation amount
        """
        pass

    def _apply_limits(
        self,
        amount: Decimal,
        alloc_config: BotAllocation,
    ) -> Decimal:
        """
        Apply min/max limits to allocation amount.

        Args:
            amount: Calculated allocation amount
            alloc_config: Bot allocation configuration

        Returns:
            Amount with limits applied
        """
        # Apply minimum
        if amount < alloc_config.min_capital:
            if amount > 0:
                # If we have some allocation but less than min, either return 0 or min
                return Decimal("0")  # Don't allocate if we can't meet minimum
            return Decimal("0")

        # Apply maximum
        return min(amount, alloc_config.max_capital)

    def _get_matching_config(
        self,
        bot_id: str,
        bot_allocations: List[BotAllocation],
    ) -> BotAllocation | None:
        """
        Get matching allocation config for a bot.

        Args:
            bot_id: Bot identifier
            bot_allocations: List of allocation configurations

        Returns:
            Matching BotAllocation or None
        """
        # Sort by priority (higher first)
        sorted_allocs = sorted(
            bot_allocations, key=lambda x: x.priority, reverse=True
        )
        for alloc in sorted_allocs:
            if alloc.enabled and alloc.matches(bot_id):
                return alloc
        return None


class FixedRatioAllocator(BaseAllocator):
    """
    Fixed ratio allocation strategy.

    Allocates funds based on configured ratios for each bot pattern.
    Ratios should sum to <= 1.0 to leave some funds unallocated.

    Example:
        - bollinger_*: 0.30 (30%)
        - grid_*: 0.40 (40%)
        - reserve: 0.10 (10% from config)
        - unallocated: 0.20 (20%)
    """

    def calculate(
        self,
        available_funds: Decimal,
        bot_allocations: List[BotAllocation],
        current_allocations: Dict[str, Decimal],
        bot_ids: List[str],
    ) -> Dict[str, Decimal]:
        """
        Calculate allocations based on fixed ratios.

        Args:
            available_funds: Total funds available for allocation
            bot_allocations: Configuration for each bot allocation
            current_allocations: Current allocations by bot_id
            bot_ids: List of active bot IDs

        Returns:
            Dictionary mapping bot_id to allocation amount
        """
        allocations: Dict[str, Decimal] = {}

        if available_funds <= 0 or not bot_ids:
            return allocations

        # Group bots by their matching allocation config
        bot_groups: Dict[str, List[str]] = {}
        for bot_id in bot_ids:
            alloc_config = self._get_matching_config(bot_id, bot_allocations)
            if alloc_config:
                pattern = alloc_config.bot_pattern
                if pattern not in bot_groups:
                    bot_groups[pattern] = []
                bot_groups[pattern].append(bot_id)

        # Calculate allocation for each group
        for pattern, group_bots in bot_groups.items():
            alloc_config = next(
                (a for a in bot_allocations if a.bot_pattern == pattern and a.enabled),
                None,
            )
            if not alloc_config:
                continue

            # Calculate total allocation for this pattern
            pattern_total = available_funds * alloc_config.ratio

            # Distribute equally among bots in the group
            per_bot = pattern_total / len(group_bots)

            for bot_id in group_bots:
                # Apply limits
                final_amount = self._apply_limits(per_bot, alloc_config)
                if final_amount > 0:
                    allocations[bot_id] = final_amount

        logger.debug(f"Fixed ratio allocation calculated: {allocations}")
        return allocations


class FixedAmountAllocator(BaseAllocator):
    """
    Fixed amount allocation strategy.

    Allocates a fixed amount to each bot based on configuration.
    """

    def calculate(
        self,
        available_funds: Decimal,
        bot_allocations: List[BotAllocation],
        current_allocations: Dict[str, Decimal],
        bot_ids: List[str],
    ) -> Dict[str, Decimal]:
        """
        Calculate allocations based on fixed amounts.

        Args:
            available_funds: Total funds available for allocation
            bot_allocations: Configuration for each bot allocation
            current_allocations: Current allocations by bot_id
            bot_ids: List of active bot IDs

        Returns:
            Dictionary mapping bot_id to allocation amount
        """
        allocations: Dict[str, Decimal] = {}
        remaining = available_funds

        if remaining <= 0 or not bot_ids:
            return allocations

        # Sort by priority (higher first)
        sorted_bots = []
        for bot_id in bot_ids:
            alloc_config = self._get_matching_config(bot_id, bot_allocations)
            if alloc_config:
                sorted_bots.append((bot_id, alloc_config))

        sorted_bots.sort(key=lambda x: x[1].priority, reverse=True)

        for bot_id, alloc_config in sorted_bots:
            if remaining <= 0:
                break

            # Use fixed_amount if set, otherwise use min_capital as amount
            amount = alloc_config.fixed_amount or alloc_config.min_capital

            if amount <= 0:
                continue

            # Don't exceed remaining funds
            actual_amount = min(amount, remaining)

            # Apply limits
            final_amount = self._apply_limits(actual_amount, alloc_config)
            if final_amount > 0:
                allocations[bot_id] = final_amount
                remaining -= final_amount

        logger.debug(f"Fixed amount allocation calculated: {allocations}")
        return allocations


class DynamicWeightAllocator(BaseAllocator):
    """
    Dynamic weight allocation strategy.

    Allocates funds based on bot performance or other dynamic factors.
    Falls back to ratio-based allocation if no performance data available.
    """

    def __init__(
        self,
        config: FundManagerConfig,
        performance_weights: Dict[str, Decimal] | None = None,
    ):
        """
        Initialize allocator with optional performance weights.

        Args:
            config: Fund manager configuration
            performance_weights: Optional dict mapping bot_id to weight
        """
        super().__init__(config)
        self._weights = performance_weights or {}

    def set_weights(self, weights: Dict[str, Decimal]) -> None:
        """
        Update performance weights.

        Args:
            weights: Dictionary mapping bot_id to weight
        """
        self._weights = weights

    def calculate(
        self,
        available_funds: Decimal,
        bot_allocations: List[BotAllocation],
        current_allocations: Dict[str, Decimal],
        bot_ids: List[str],
    ) -> Dict[str, Decimal]:
        """
        Calculate allocations based on dynamic weights.

        Args:
            available_funds: Total funds available for allocation
            bot_allocations: Configuration for each bot allocation
            current_allocations: Current allocations by bot_id
            bot_ids: List of active bot IDs

        Returns:
            Dictionary mapping bot_id to allocation amount
        """
        allocations: Dict[str, Decimal] = {}

        if available_funds <= 0 or not bot_ids:
            return allocations

        # Get applicable bots and their weights
        applicable_bots: List[tuple[str, BotAllocation, Decimal]] = []
        for bot_id in bot_ids:
            alloc_config = self._get_matching_config(bot_id, bot_allocations)
            if alloc_config:
                # Use performance weight if available, otherwise use ratio
                weight = self._weights.get(bot_id, alloc_config.ratio)
                applicable_bots.append((bot_id, alloc_config, weight))

        if not applicable_bots:
            return allocations

        # Calculate total weight
        total_weight = sum(w for _, _, w in applicable_bots)
        if total_weight <= 0:
            return allocations

        # Allocate based on normalized weights
        for bot_id, alloc_config, weight in applicable_bots:
            ratio = weight / total_weight
            amount = available_funds * ratio

            # Apply limits
            final_amount = self._apply_limits(amount, alloc_config)
            if final_amount > 0:
                allocations[bot_id] = final_amount

        logger.debug(f"Dynamic weight allocation calculated: {allocations}")
        return allocations


def create_allocator(
    config: FundManagerConfig,
    **kwargs: Any,
) -> BaseAllocator:
    """
    Factory function to create an allocator based on config strategy.

    Args:
        config: Fund manager configuration
        **kwargs: Additional arguments for specific allocators

    Returns:
        Allocator instance

    Raises:
        ValueError: If strategy is not recognized
    """
    from ..models.config import AllocationStrategy as StrategyEnum

    if config.strategy == StrategyEnum.FIXED_RATIO:
        return FixedRatioAllocator(config)
    elif config.strategy == StrategyEnum.FIXED_AMOUNT:
        return FixedAmountAllocator(config)
    elif config.strategy == StrategyEnum.DYNAMIC_WEIGHT:
        return DynamicWeightAllocator(config, **kwargs)
    else:
        raise ValueError(f"Unknown allocation strategy: {config.strategy}")
