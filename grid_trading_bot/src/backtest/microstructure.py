"""
Market Microstructure Module.

Simulates market microstructure including bid/ask spreads,
intra-bar price sequences, and gap handling.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..core.models import Kline
from .config import IntraBarSequence


class PriceSequence(str, Enum):
    """Price sequence within a bar."""

    HIGH_FIRST = "high_first"  # Open -> High -> Low -> Close
    LOW_FIRST = "low_first"  # Open -> Low -> High -> Close


@dataclass
class SpreadContext:
    """
    Context for spread calculation.

    Attributes:
        volatility_ratio: Current volatility / average volatility
        liquidity_factor: 0-1 where 1 is normal liquidity
        time_of_day_factor: Adjustment based on trading session
    """

    volatility_ratio: Decimal = Decimal("1.0")
    liquidity_factor: Decimal = Decimal("1.0")
    time_of_day_factor: Decimal = Decimal("1.0")


class MarketMicrostructure:
    """
    Simulates market microstructure for realistic order execution.

    Features:
    - Bid/Ask spread simulation
    - Dynamic spread adjustment based on volatility
    - Intra-bar price sequence determination
    - Gap handling for stop loss and take profit

    Example:
        microstructure = MarketMicrostructure(
            base_spread_pct=Decimal("0.0001"),
            spread_volatility_factor=Decimal("0.5"),
        )
        exec_price = microstructure.get_execution_price(
            target_price=Decimal("100"),
            is_buy=True,
            kline=kline,
            is_aggressive=True,
        )
    """

    def __init__(
        self,
        base_spread_pct: Decimal = Decimal("0.0001"),
        spread_volatility_factor: Decimal = Decimal("0.5"),
        intra_bar_sequence: IntraBarSequence = IntraBarSequence.OHLC,
    ) -> None:
        """
        Initialize market microstructure.

        Args:
            base_spread_pct: Base bid/ask spread (0.0001 = 0.01%)
            spread_volatility_factor: How much spread widens with volatility
            intra_bar_sequence: Assumed price sequence within a bar
        """
        if base_spread_pct < 0:
            raise ValueError("base_spread_pct cannot be negative")
        if spread_volatility_factor < 0:
            raise ValueError("spread_volatility_factor cannot be negative")

        self._base_spread_pct = base_spread_pct
        self._spread_volatility_factor = spread_volatility_factor
        self._intra_bar_sequence = intra_bar_sequence

    @property
    def base_spread_pct(self) -> Decimal:
        """Get base spread percentage."""
        return self._base_spread_pct

    @property
    def spread_volatility_factor(self) -> Decimal:
        """Get spread volatility factor."""
        return self._spread_volatility_factor

    @property
    def intra_bar_sequence(self) -> IntraBarSequence:
        """Get intra-bar sequence setting."""
        return self._intra_bar_sequence

    def calculate_spread(
        self,
        price: Decimal,
        kline: Kline,
        context: Optional[SpreadContext] = None,
    ) -> Decimal:
        """
        Calculate the bid/ask spread.

        Base formula: spread = base_spread_pct * price
        Volatility adjustment: spread *= (1 + volatility_factor * (vol_ratio - 1))

        Args:
            price: Reference price
            kline: Current kline for volatility calculation
            context: Additional context for spread adjustment

        Returns:
            Spread amount (half on each side)
        """
        base_spread = price * self._base_spread_pct

        if context and context.volatility_ratio != Decimal("1.0"):
            # Widen spread during high volatility
            vol_adjustment = Decimal("1.0") + self._spread_volatility_factor * (
                context.volatility_ratio - Decimal("1.0")
            )
            base_spread *= max(vol_adjustment, Decimal("0.5"))  # Floor at 50%

        if context and context.liquidity_factor < Decimal("1.0"):
            # Widen spread during low liquidity
            base_spread /= max(context.liquidity_factor, Decimal("0.1"))

        return base_spread

    def calculate_volatility_spread(
        self,
        price: Decimal,
        kline: Kline,
    ) -> Decimal:
        """
        Calculate spread based on kline volatility.

        Uses kline range as a proxy for volatility.

        Args:
            price: Reference price
            kline: Current kline

        Returns:
            Volatility-adjusted spread amount
        """
        # Calculate kline range as volatility proxy
        kline_range = kline.high - kline.low
        range_pct = kline_range / price if price > 0 else Decimal("0")

        # Base spread
        base_spread = price * self._base_spread_pct

        # Add volatility component (use portion of range)
        vol_spread = kline_range * self._spread_volatility_factor * Decimal("0.1")

        return base_spread + vol_spread

    def get_execution_price(
        self,
        target_price: Decimal,
        is_buy: bool,
        kline: Kline,
        is_aggressive: bool = True,
        context: Optional[SpreadContext] = None,
    ) -> Decimal:
        """
        Get execution price including spread simulation.

        Aggressive orders (crossing the spread) get worse prices.
        Passive orders (providing liquidity) can get better prices.

        Args:
            target_price: Target/signal price
            is_buy: True if buying
            kline: Current kline
            is_aggressive: True if taking liquidity (market order or aggressive limit)
            context: Additional context

        Returns:
            Adjusted execution price
        """
        spread = self.calculate_spread(target_price, kline, context)
        half_spread = spread / Decimal("2")

        if is_aggressive:
            # Pay the spread
            if is_buy:
                # Buy at ask (higher)
                exec_price = target_price + half_spread
                exec_price = min(exec_price, kline.high)
            else:
                # Sell at bid (lower)
                exec_price = target_price - half_spread
                exec_price = max(exec_price, kline.low)
        else:
            # Provide liquidity - can get slightly better price
            if is_buy:
                # Buy at or below target
                exec_price = target_price
            else:
                # Sell at or above target
                exec_price = target_price

        return exec_price

    def determine_price_sequence(self, kline: Kline) -> PriceSequence:
        """
        Determine the most likely price sequence within a bar.

        Logic:
        - If configured as OHLC: high comes before low
        - If configured as OLHC: low comes before high
        - If configured as WORST_CASE: return based on context

        Args:
            kline: Current kline

        Returns:
            PriceSequence indicating which extreme was hit first
        """
        if self._intra_bar_sequence == IntraBarSequence.OHLC:
            return PriceSequence.HIGH_FIRST
        elif self._intra_bar_sequence == IntraBarSequence.OLHC:
            return PriceSequence.LOW_FIRST
        else:  # WORST_CASE - use close vs open to guess
            if kline.close >= kline.open:
                # Bullish bar - likely went low first
                return PriceSequence.LOW_FIRST
            else:
                # Bearish bar - likely went high first
                return PriceSequence.HIGH_FIRST

    def determine_sl_tp_execution(
        self,
        position_side: str,
        kline: Kline,
        stop_loss: Optional[Decimal],
        take_profit: Optional[Decimal],
    ) -> tuple[bool, bool, Optional[Decimal], Optional[Decimal]]:
        """
        Determine if SL and/or TP are triggered and their fill prices.

        When both SL and TP could be hit in the same bar, uses
        intra-bar sequence to determine which executes first.

        Args:
            position_side: "LONG" or "SHORT"
            kline: Current kline
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Tuple of (sl_triggered, tp_triggered, sl_fill, tp_fill)
            Only one of sl_triggered/tp_triggered will be True if both hit
        """
        sl_triggered = False
        tp_triggered = False
        sl_fill: Optional[Decimal] = None
        tp_fill: Optional[Decimal] = None

        is_long = position_side == "LONG"

        # Check if SL would be hit
        sl_hit = False
        if stop_loss is not None:
            if is_long and kline.low <= stop_loss:
                sl_hit = True
            elif not is_long and kline.high >= stop_loss:
                sl_hit = True

        # Check if TP would be hit
        tp_hit = False
        if take_profit is not None:
            if is_long and kline.high >= take_profit:
                tp_hit = True
            elif not is_long and kline.low <= take_profit:
                tp_hit = True

        if sl_hit and tp_hit:
            # Both hit - use sequence to determine which first
            sequence = self.determine_price_sequence(kline)

            if is_long:
                # Long: SL hit on low, TP hit on high
                if sequence == PriceSequence.LOW_FIRST:
                    sl_triggered = True
                    sl_fill = self.calculate_gap_fill_price(
                        stop_loss, kline, is_stop=True, is_long=True
                    )
                else:
                    tp_triggered = True
                    tp_fill = take_profit
            else:
                # Short: SL hit on high, TP hit on low
                if sequence == PriceSequence.HIGH_FIRST:
                    sl_triggered = True
                    sl_fill = self.calculate_gap_fill_price(
                        stop_loss, kline, is_stop=True, is_long=False
                    )
                else:
                    tp_triggered = True
                    tp_fill = take_profit

        elif sl_hit:
            sl_triggered = True
            sl_fill = self.calculate_gap_fill_price(
                stop_loss, kline, is_stop=True, is_long=is_long
            )

        elif tp_hit:
            tp_triggered = True
            tp_fill = take_profit  # TP fills at target

        return sl_triggered, tp_triggered, sl_fill, tp_fill

    def calculate_gap_fill_price(
        self,
        target_price: Decimal,
        kline: Kline,
        is_stop: bool,
        is_long: bool,
    ) -> Decimal:
        """
        Calculate fill price when price gaps through target.

        For stop orders, gaps result in slippage (worse fills).
        For take profits, fills at target or better.

        Args:
            target_price: Target order price
            kline: Current kline
            is_stop: True if this is a stop order
            is_long: True if long position

        Returns:
            Actual fill price accounting for gaps
        """
        if is_stop:
            if is_long:
                # Long stop loss = sell when price drops
                # Gap down: fill at open or worse (kline low)
                if kline.open < target_price:
                    # Gapped through stop
                    return max(kline.open, kline.low)
                else:
                    return target_price
            else:
                # Short stop loss = buy when price rises
                # Gap up: fill at open or worse (kline high)
                if kline.open > target_price:
                    # Gapped through stop
                    return min(kline.open, kline.high)
                else:
                    return target_price
        else:
            # Take profit - fill at target or better
            return target_price

    def get_mid_price(self, kline: Kline) -> Decimal:
        """
        Get the mid price from kline.

        Args:
            kline: Current kline

        Returns:
            Mid price ((high + low) / 2)
        """
        return (kline.high + kline.low) / Decimal("2")

    def get_vwap_estimate(self, kline: Kline) -> Decimal:
        """
        Estimate VWAP from kline data.

        Uses typical price as VWAP approximation.

        Args:
            kline: Current kline

        Returns:
            Estimated VWAP (typical price)
        """
        # Typical price = (high + low + close) / 3
        return (kline.high + kline.low + kline.close) / Decimal("3")


def create_microstructure_from_config(
    enable_spread: bool = False,
    spread_pct: Decimal = Decimal("0.0001"),
    intra_bar_sequence: IntraBarSequence = IntraBarSequence.OHLC,
) -> Optional[MarketMicrostructure]:
    """
    Factory function to create market microstructure from config.

    Args:
        enable_spread: Whether spread simulation is enabled
        spread_pct: Base spread percentage
        intra_bar_sequence: Intra-bar price sequence

    Returns:
        MarketMicrostructure if enabled, None otherwise
    """
    if not enable_spread:
        return None

    return MarketMicrostructure(
        base_spread_pct=spread_pct,
        intra_bar_sequence=intra_bar_sequence,
    )
