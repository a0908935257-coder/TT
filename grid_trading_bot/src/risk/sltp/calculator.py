"""
Stop Loss / Take Profit Calculator.

Provides calculation logic for stop loss, take profit, and trailing stop prices.
Pure calculation functions without side effects - stateless design.
"""

import logging
from decimal import Decimal
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

from src.risk.sltp.models import (
    StopLossConfig,
    StopLossType,
    TakeProfitConfig,
    TakeProfitLevel,
    TakeProfitType,
    TrailingStopConfig,
    TrailingStopType,
)


class SLTPCalculator:
    """Calculator for stop loss and take profit levels."""

    def __init__(
        self,
        indicator_providers: Optional[Dict[str, Callable[[str], Decimal]]] = None,
    ) -> None:
        """
        Initialize calculator.

        Args:
            indicator_providers: Dict mapping indicator keys to provider functions.
                                 Each function takes symbol and returns the indicator value.
                                 Example: {"supertrend": lambda sym: get_supertrend(sym)}
        """
        self._indicator_providers = indicator_providers or {}

    def calculate_stop_loss(
        self,
        config: StopLossConfig,
        entry_price: Decimal,
        is_long: bool,
        atr: Optional[Decimal] = None,
        symbol: Optional[str] = None,
    ) -> Decimal:
        """
        Calculate stop loss price.

        Args:
            config: Stop loss configuration
            entry_price: Entry price of the position
            is_long: True if long position, False if short
            atr: Current ATR value (required for ATR_BASED type)
            symbol: Symbol (required for INDICATOR type)

        Returns:
            Calculated stop loss price
        """
        if not config.enabled:
            # Return a price that won't trigger (very far away)
            return Decimal("0") if is_long else Decimal("999999999")

        if config.stop_type == StopLossType.FIXED:
            if config.fixed_price is None:
                raise ValueError("fixed_price required for FIXED stop loss")
            return config.fixed_price

        elif config.stop_type == StopLossType.PERCENTAGE:
            distance = entry_price * config.value
            if is_long:
                return entry_price - distance
            else:
                return entry_price + distance

        elif config.stop_type == StopLossType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED stop loss")
            distance = atr * config.atr_multiplier
            if is_long:
                return entry_price - distance
            else:
                return entry_price + distance

        elif config.stop_type == StopLossType.INDICATOR:
            if symbol is None or config.indicator_key is None:
                raise ValueError("symbol and indicator_key required for INDICATOR stop loss")
            provider = self._indicator_providers.get(config.indicator_key)
            if provider is None:
                raise ValueError(f"No indicator provider for key: {config.indicator_key}")
            return provider(symbol)

        elif config.stop_type == StopLossType.TRAILING:
            # For trailing, initial SL is based on percentage/ATR
            # Actual trailing is handled separately
            if atr is not None:
                distance = atr * config.atr_multiplier
            else:
                distance = entry_price * config.value
            if is_long:
                return entry_price - distance
            else:
                return entry_price + distance

        raise ValueError(f"Unknown stop loss type: {config.stop_type}")

    def calculate_take_profit(
        self,
        config: TakeProfitConfig,
        entry_price: Decimal,
        is_long: bool,
        stop_loss: Optional[Decimal] = None,
        atr: Optional[Decimal] = None,
        symbol: Optional[str] = None,
    ) -> List[TakeProfitLevel]:
        """
        Calculate take profit levels.

        Args:
            config: Take profit configuration
            entry_price: Entry price of the position
            is_long: True if long position, False if short
            stop_loss: Stop loss price (required for RISK_REWARD type)
            atr: Current ATR value (required for ATR_BASED type)
            symbol: Symbol (required for INDICATOR type)

        Returns:
            List of TakeProfitLevel objects
        """
        if not config.enabled:
            return []

        if config.tp_type == TakeProfitType.FIXED:
            if config.fixed_price is None:
                raise ValueError("fixed_price required for FIXED take profit")
            return [TakeProfitLevel(price=config.fixed_price, percentage=Decimal("1.0"))]

        elif config.tp_type == TakeProfitType.PERCENTAGE:
            distance = entry_price * config.value
            if is_long:
                price = entry_price + distance
            else:
                price = entry_price - distance
            return [TakeProfitLevel(price=price, percentage=Decimal("1.0"))]

        elif config.tp_type == TakeProfitType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED take profit")
            distance = atr * config.atr_multiplier
            if is_long:
                price = entry_price + distance
            else:
                price = entry_price - distance
            return [TakeProfitLevel(price=price, percentage=Decimal("1.0"))]

        elif config.tp_type == TakeProfitType.RISK_REWARD:
            if stop_loss is None:
                raise ValueError("stop_loss required for RISK_REWARD take profit")
            risk = abs(entry_price - stop_loss)
            reward = risk * config.risk_reward_ratio
            if is_long:
                price = entry_price + reward
            else:
                price = entry_price - reward
            return [TakeProfitLevel(price=price, percentage=Decimal("1.0"))]

        elif config.tp_type == TakeProfitType.INDICATOR:
            if symbol is None or config.indicator_key is None:
                raise ValueError("symbol and indicator_key required for INDICATOR take profit")
            provider = self._indicator_providers.get(config.indicator_key)
            if provider is None:
                raise ValueError(f"No indicator provider for key: {config.indicator_key}")
            price = provider(symbol)
            return [TakeProfitLevel(price=price, percentage=Decimal("1.0"))]

        elif config.tp_type == TakeProfitType.MULTI_LEVEL:
            if not config.level_percentages or not config.level_close_pcts:
                raise ValueError(
                    "level_percentages and level_close_pcts required for MULTI_LEVEL"
                )
            # Validate total close percentages don't exceed 100%
            close_pcts = list(config.level_close_pcts)  # Local copy to avoid mutating shared config
            total_close_pct = sum(close_pcts)
            if total_close_pct > Decimal("1.0"):
                logger.warning(
                    f"MULTI_LEVEL TP close_pcts sum={total_close_pct} > 1.0, normalizing"
                )
                close_pcts = [
                    pct / total_close_pct for pct in close_pcts
                ]
            levels = []
            for pct, close_pct in zip(config.level_percentages, close_pcts):
                distance = entry_price * pct
                if is_long:
                    price = entry_price + distance
                else:
                    price = entry_price - distance
                levels.append(TakeProfitLevel(price=price, percentage=close_pct))
            return levels

        raise ValueError(f"Unknown take profit type: {config.tp_type}")

    def calculate_trailing_stop(
        self,
        config: TrailingStopConfig,
        current_price: Decimal,
        highest_price: Decimal,
        lowest_price: Decimal,
        is_long: bool,
        entry_price: Decimal,
        current_stop: Decimal,
        atr: Optional[Decimal] = None,
    ) -> Optional[Decimal]:
        """
        Calculate new trailing stop price.

        Args:
            config: Trailing stop configuration
            current_price: Current market price
            highest_price: Highest price since entry (for long)
            lowest_price: Lowest price since entry (for short)
            is_long: True if long position, False if short
            entry_price: Entry price of the position
            current_stop: Current stop loss price
            atr: Current ATR value (required for ATR_BASED type)

        Returns:
            New stop loss price if it should be updated, None otherwise
        """
        if not config.enabled:
            return None

        # Guard against division by zero (entry_price should never be 0 in practice)
        if entry_price <= 0:
            return None

        # Check if trailing should be activated
        if is_long:
            profit_pct = (current_price - entry_price) / entry_price
            if profit_pct < config.activation_pct:
                return None
            reference_price = highest_price
        else:
            profit_pct = (entry_price - current_price) / entry_price
            if profit_pct < config.activation_pct:
                return None
            reference_price = lowest_price

        # Calculate new trailing stop based on type
        if config.trailing_type == TrailingStopType.FIXED_DISTANCE:
            if is_long:
                new_stop = reference_price - config.distance
            else:
                new_stop = reference_price + config.distance

        elif config.trailing_type == TrailingStopType.PERCENTAGE:
            distance = reference_price * config.distance
            if is_long:
                new_stop = reference_price - distance
            else:
                new_stop = reference_price + distance

        elif config.trailing_type == TrailingStopType.ATR_BASED:
            if atr is None:
                raise ValueError("ATR value required for ATR_BASED trailing stop")
            distance = atr * config.atr_multiplier
            if is_long:
                new_stop = reference_price - distance
            else:
                new_stop = reference_price + distance

        elif config.trailing_type == TrailingStopType.BREAK_EVEN:
            # Move to break even once threshold is reached
            if profit_pct >= config.break_even_trigger:
                new_stop = entry_price
            else:
                return None

        else:
            raise ValueError(f"Unknown trailing stop type: {config.trailing_type}")

        # Only return if new stop is better than current
        if is_long and new_stop > current_stop:
            return new_stop
        elif not is_long and new_stop < current_stop:
            return new_stop

        return None

    def check_stop_loss_hit(
        self,
        stop_loss: Decimal,
        current_price: Decimal,
        high: Decimal,
        low: Decimal,
        is_long: bool,
    ) -> bool:
        """
        Check if stop loss was hit.

        Args:
            stop_loss: Stop loss price
            current_price: Current price
            high: High price of the period
            low: Low price of the period
            is_long: True if long position

        Returns:
            True if stop loss was hit
        """
        if is_long:
            # For long, stop loss is hit if low <= stop_loss
            return low <= stop_loss
        else:
            # For short, stop loss is hit if high >= stop_loss
            return high >= stop_loss

    def check_take_profit_hit(
        self,
        take_profit_levels: List[TakeProfitLevel],
        current_price: Decimal,
        high: Decimal,
        low: Decimal,
        is_long: bool,
    ) -> List[int]:
        """
        Check which take profit levels were hit.

        Args:
            take_profit_levels: List of take profit levels
            current_price: Current price
            high: High price of the period
            low: Low price of the period
            is_long: True if long position

        Returns:
            List of indices of hit levels (not yet triggered)
        """
        hit_indices = []

        for i, level in enumerate(take_profit_levels):
            if level.triggered:
                continue

            if is_long:
                # For long, TP is hit if high >= target
                if high >= level.price:
                    hit_indices.append(i)
            else:
                # For short, TP is hit if low <= target
                if low <= level.price:
                    hit_indices.append(i)

        return hit_indices
