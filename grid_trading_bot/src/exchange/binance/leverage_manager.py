"""
Leverage Manager for Binance Futures.

Handles leverage bracket management, margin calculations, and position risk assessment.
"""

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Literal, Optional

from src.core import get_logger

logger = get_logger(__name__)


class RiskLevel(str, Enum):
    """Position risk level based on distance to liquidation."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class LeverageBracket:
    """
    Leverage bracket information for a symbol.

    Binance uses tiered leverage where larger positions have lower max leverage.

    Example:
        >>> bracket = LeverageBracket(
        ...     bracket=1,
        ...     notional_cap=Decimal("50000"),
        ...     notional_floor=Decimal("0"),
        ...     max_leverage=125,
        ...     maint_margin_rate=Decimal("0.004"),
        ... )
    """

    bracket: int
    notional_cap: Decimal
    notional_floor: Decimal
    max_leverage: int
    maint_margin_rate: Decimal

    @classmethod
    def from_binance(cls, data: dict, prev_notional: Decimal = Decimal("0")) -> "LeverageBracket":
        """
        Create LeverageBracket from Binance API response.

        Args:
            data: Bracket data from API
            prev_notional: Previous bracket's notional cap (becomes this bracket's floor)

        Returns:
            LeverageBracket instance
        """
        return cls(
            bracket=data["bracket"],
            notional_cap=Decimal(str(data["notionalCap"])),
            notional_floor=prev_notional,
            max_leverage=data["initialLeverage"],
            maint_margin_rate=Decimal(str(data["maintMarginRatio"])),
        )


@dataclass
class PositionRisk:
    """
    Position risk assessment result.

    Example:
        >>> risk = PositionRisk(
        ...     liquidation_price=Decimal("45000"),
        ...     distance_to_liquidation=Decimal("10.5"),
        ...     risk_level=RiskLevel.MEDIUM,
        ...     initial_margin=Decimal("1000"),
        ...     maint_margin=Decimal("400"),
        ... )
    """

    liquidation_price: Decimal
    distance_to_liquidation: Decimal
    risk_level: RiskLevel
    initial_margin: Decimal
    maint_margin: Decimal


class LeverageManager:
    """
    Manage leverage brackets and calculate margins for Binance Futures.

    Provides caching for leverage brackets and methods for margin/liquidation
    calculations based on Binance's tiered leverage system.

    Example:
        >>> from exchange.binance import BinanceFuturesAPI
        >>> async with BinanceFuturesAPI(api_key="...", api_secret="...") as api:
        ...     manager = LeverageManager(api)
        ...     brackets = await manager.get_brackets("BTCUSDT")
        ...     max_lev = await manager.get_max_leverage("BTCUSDT", Decimal("100000"))
    """

    def __init__(self, futures_api: Optional["BinanceFuturesAPI"] = None):
        """
        Initialize LeverageManager.

        Args:
            futures_api: BinanceFuturesAPI instance for fetching brackets.
                        If None, must use load_brackets_from_data().
        """
        self._api = futures_api
        self._brackets_cache: dict[str, list[LeverageBracket]] = {}

    async def load_brackets(self, symbol: str) -> list[LeverageBracket]:
        """
        Load leverage brackets from Binance API.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Returns:
            List of LeverageBracket sorted by notional cap

        Raises:
            ValueError: If API not configured
        """
        if self._api is None:
            raise ValueError("Futures API not configured. Use load_brackets_from_data() instead.")

        data = await self._api.get_leverage_brackets(symbol)

        # API returns list with one item per symbol
        symbol_data = None
        for item in data:
            if item.get("symbol") == symbol:
                symbol_data = item
                break

        if symbol_data is None:
            raise ValueError(f"No leverage brackets found for {symbol}")

        brackets = self._parse_brackets(symbol_data["brackets"])
        self._brackets_cache[symbol] = brackets

        logger.debug(f"Loaded {len(brackets)} leverage brackets for {symbol}")
        return brackets

    def load_brackets_from_data(self, symbol: str, brackets_data: list[dict]) -> list[LeverageBracket]:
        """
        Load leverage brackets from raw data (for testing or offline use).

        Args:
            symbol: Trading pair
            brackets_data: List of bracket dictionaries

        Returns:
            List of LeverageBracket
        """
        brackets = self._parse_brackets(brackets_data)
        self._brackets_cache[symbol] = brackets
        return brackets

    def _parse_brackets(self, brackets_data: list[dict]) -> list[LeverageBracket]:
        """
        Parse bracket data into LeverageBracket objects.

        Args:
            brackets_data: Raw bracket data from API

        Returns:
            List of LeverageBracket sorted by bracket number
        """
        # Sort by bracket number
        sorted_data = sorted(brackets_data, key=lambda x: x["bracket"])

        brackets = []
        prev_notional = Decimal("0")

        for data in sorted_data:
            bracket = LeverageBracket.from_binance(data, prev_notional)
            brackets.append(bracket)
            prev_notional = bracket.notional_cap

        return brackets

    async def get_brackets(self, symbol: str) -> list[LeverageBracket]:
        """
        Get leverage brackets for a symbol (with caching).

        Args:
            symbol: Trading pair

        Returns:
            List of LeverageBracket
        """
        if symbol not in self._brackets_cache:
            await self.load_brackets(symbol)
        return self._brackets_cache[symbol]

    def get_brackets_sync(self, symbol: str) -> list[LeverageBracket]:
        """
        Get cached leverage brackets synchronously.

        Args:
            symbol: Trading pair

        Returns:
            List of LeverageBracket

        Raises:
            ValueError: If brackets not loaded
        """
        if symbol not in self._brackets_cache:
            raise ValueError(f"Brackets not loaded for {symbol}. Call load_brackets() first.")
        return self._brackets_cache[symbol]

    def get_bracket_for_notional(self, symbol: str, notional: Decimal) -> LeverageBracket:
        """
        Get the applicable bracket for a given notional value.

        Args:
            symbol: Trading pair
            notional: Position notional value

        Returns:
            LeverageBracket that applies to the notional value
        """
        brackets = self.get_brackets_sync(symbol)

        for bracket in brackets:
            if notional <= bracket.notional_cap:
                return bracket

        # Return last bracket if notional exceeds all caps
        return brackets[-1]

    async def get_max_leverage(self, symbol: str, notional: Decimal) -> int:
        """
        Get maximum allowed leverage for a given notional value.

        Args:
            symbol: Trading pair
            notional: Position notional value

        Returns:
            Maximum leverage allowed
        """
        if symbol not in self._brackets_cache:
            await self.load_brackets(symbol)

        bracket = self.get_bracket_for_notional(symbol, notional)
        return bracket.max_leverage

    def get_max_leverage_sync(self, symbol: str, notional: Decimal) -> int:
        """
        Get maximum leverage synchronously (brackets must be pre-loaded).

        Args:
            symbol: Trading pair
            notional: Position notional value

        Returns:
            Maximum leverage allowed
        """
        bracket = self.get_bracket_for_notional(symbol, notional)
        return bracket.max_leverage

    async def validate_leverage(
        self,
        symbol: str,
        leverage: int,
        notional: Decimal,
    ) -> tuple[bool, str]:
        """
        Validate if leverage is allowed for given notional value.

        Args:
            symbol: Trading pair
            leverage: Desired leverage
            notional: Position notional value

        Returns:
            Tuple of (is_valid, message)
        """
        max_leverage = await self.get_max_leverage(symbol, notional)

        if leverage > max_leverage:
            return (
                False,
                f"Leverage {leverage}x exceeds max {max_leverage}x for notional {notional}",
            )

        if leverage < 1:
            return False, "Leverage must be at least 1x"

        return True, "OK"

    def validate_leverage_sync(
        self,
        symbol: str,
        leverage: int,
        notional: Decimal,
    ) -> tuple[bool, str]:
        """
        Validate leverage synchronously (brackets must be pre-loaded).

        Args:
            symbol: Trading pair
            leverage: Desired leverage
            notional: Position notional value

        Returns:
            Tuple of (is_valid, message)
        """
        max_leverage = self.get_max_leverage_sync(symbol, notional)

        if leverage > max_leverage:
            return (
                False,
                f"Leverage {leverage}x exceeds max {max_leverage}x for notional {notional}",
            )

        if leverage < 1:
            return False, "Leverage must be at least 1x"

        return True, "OK"

    def calculate_initial_margin(
        self,
        notional: Decimal,
        leverage: int,
    ) -> Decimal:
        """
        Calculate initial margin required for a position.

        Formula: Initial Margin = Notional Value / Leverage

        Args:
            notional: Position notional value (price * quantity)
            leverage: Leverage multiplier

        Returns:
            Initial margin required
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")

        return notional / Decimal(str(leverage))

    def calculate_maint_margin(
        self,
        symbol: str,
        notional: Decimal,
    ) -> Decimal:
        """
        Calculate maintenance margin for a position.

        Formula: Maintenance Margin = Notional Value * Maintenance Margin Rate

        Args:
            symbol: Trading pair
            notional: Position notional value

        Returns:
            Maintenance margin required
        """
        bracket = self.get_bracket_for_notional(symbol, notional)
        return notional * bracket.maint_margin_rate

    def calculate_liquidation_price(
        self,
        symbol: str,
        entry_price: Decimal,
        quantity: Decimal,
        leverage: int,
        side: Literal["long", "short", "LONG", "SHORT"],
        wallet_balance: Optional[Decimal] = None,
    ) -> Decimal:
        """
        Calculate liquidation price for a position.

        For isolated margin:
        - Long: Liq Price = Entry Price * (1 - 1/Leverage + MMR)
        - Short: Liq Price = Entry Price * (1 + 1/Leverage - MMR)

        Args:
            symbol: Trading pair
            entry_price: Position entry price
            quantity: Position quantity (absolute value)
            leverage: Position leverage
            side: Position side ("long" or "short")
            wallet_balance: Wallet balance (for cross margin, not used in isolated)

        Returns:
            Estimated liquidation price
        """
        notional = entry_price * abs(quantity)
        bracket = self.get_bracket_for_notional(symbol, notional)
        mmr = bracket.maint_margin_rate

        side_lower = side.lower()

        if side_lower == "long":
            # Long liquidation: price drops
            # Liq = Entry * (1 - 1/leverage + MMR)
            liq_price = entry_price * (Decimal("1") - Decimal("1") / Decimal(str(leverage)) + mmr)
        else:
            # Short liquidation: price rises
            # Liq = Entry * (1 + 1/leverage - MMR)
            liq_price = entry_price * (Decimal("1") + Decimal("1") / Decimal(str(leverage)) - mmr)

        # Ensure non-negative
        return max(Decimal("0"), liq_price)

    def assess_position_risk(
        self,
        symbol: str,
        entry_price: Decimal,
        current_price: Decimal,
        quantity: Decimal,
        leverage: int,
        side: Literal["long", "short", "LONG", "SHORT"],
    ) -> PositionRisk:
        """
        Assess the risk level of a position.

        Risk levels based on distance to liquidation:
        - LOW: > 20%
        - MEDIUM: > 10%
        - HIGH: > 5%
        - CRITICAL: <= 5%

        Args:
            symbol: Trading pair
            entry_price: Position entry price
            current_price: Current market price
            quantity: Position quantity (absolute value)
            leverage: Position leverage
            side: Position side ("long" or "short")

        Returns:
            PositionRisk with liquidation price, distance, and risk level
        """
        notional = entry_price * abs(quantity)

        # Calculate margins
        initial_margin = self.calculate_initial_margin(notional, leverage)
        maint_margin = self.calculate_maint_margin(symbol, notional)

        # Calculate liquidation price
        liq_price = self.calculate_liquidation_price(
            symbol=symbol,
            entry_price=entry_price,
            quantity=quantity,
            leverage=leverage,
            side=side,
        )

        # Calculate distance to liquidation
        side_lower = side.lower()
        if side_lower == "long":
            # For long: how far current price is above liquidation price
            if liq_price > 0:
                distance = ((current_price - liq_price) / current_price) * Decimal("100")
            else:
                distance = Decimal("100")  # No liquidation risk
        else:
            # For short: how far current price is below liquidation price
            if liq_price > 0:
                distance = ((liq_price - current_price) / current_price) * Decimal("100")
            else:
                distance = Decimal("100")

        # Ensure non-negative
        distance = max(Decimal("0"), distance)

        # Determine risk level
        if distance > Decimal("20"):
            risk_level = RiskLevel.LOW
        elif distance > Decimal("10"):
            risk_level = RiskLevel.MEDIUM
        elif distance > Decimal("5"):
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.CRITICAL

        return PositionRisk(
            liquidation_price=liq_price,
            distance_to_liquidation=distance,
            risk_level=risk_level,
            initial_margin=initial_margin,
            maint_margin=maint_margin,
        )

    def clear_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear cached brackets.

        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            self._brackets_cache.pop(symbol, None)
        else:
            self._brackets_cache.clear()

    def is_cached(self, symbol: str) -> bool:
        """
        Check if brackets are cached for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            True if cached
        """
        return symbol in self._brackets_cache


# Type hint for avoiding circular import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .futures_api import BinanceFuturesAPI
