"""
Pre-Trade Risk Checker.

Provides pre-trade risk validation for orders before submission.
Implements the first line of defense in the risk management system.

Checks performed:
- Position limit per symbol
- Total position limit
- Order size limit
- Price deviation check
- Order frequency limit
- Available funds check
- Blacklist check
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Deque, Dict, List, Optional, Protocol, Set

from src.core import get_logger

logger = get_logger(__name__)


class CheckResult(Enum):
    """Result of a pre-trade check."""

    PASSED = "passed"  # Check passed
    REJECTED = "rejected"  # Order rejected
    WARNING = "warning"  # Warning issued but order allowed
    SPLIT_REQUIRED = "split_required"  # Order needs to be split


class RejectionReason(Enum):
    """Reason for order rejection."""

    POSITION_LIMIT_SYMBOL = "position_limit_symbol"
    POSITION_LIMIT_TOTAL = "position_limit_total"
    ORDER_SIZE_LIMIT = "order_size_limit"
    PRICE_DEVIATION = "price_deviation"
    FREQUENCY_LIMIT = "frequency_limit"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    BLACKLISTED_SYMBOL = "blacklisted_symbol"
    CIRCUIT_BREAKER_ACTIVE = "circuit_breaker_active"
    MAX_ORDERS_PER_SYMBOL = "max_orders_per_symbol"


@dataclass
class PreTradeConfig:
    """Configuration for pre-trade risk checks."""

    # Position limits
    max_position_pct_per_symbol: Decimal = Decimal("0.30")  # 30% max per symbol
    max_total_position_pct: Decimal = Decimal("0.80")  # 80% max total exposure
    max_positions_count: int = 10  # Max number of open positions

    # Order limits
    max_order_value: Decimal = Decimal("10000")  # Max single order value
    max_order_quantity: Optional[Decimal] = None  # Max single order quantity
    min_order_value: Decimal = Decimal("10")  # Min order value
    max_orders_per_symbol: int = 20  # Max open orders per symbol

    # Price deviation
    max_price_deviation_pct: Decimal = Decimal("0.02")  # 2% max deviation from market
    price_deviation_warning_pct: Decimal = Decimal("0.01")  # 1% warning threshold

    # Frequency limits
    max_orders_per_second: int = 5  # Max orders per second
    max_orders_per_minute: int = 60  # Max orders per minute
    order_cooldown_ms: int = 100  # Min ms between orders

    # Fund requirements
    min_available_balance_pct: Decimal = Decimal("0.05")  # Keep 5% as reserve
    margin_buffer_pct: Decimal = Decimal("0.20")  # 20% margin buffer for futures

    # Blacklist
    blacklisted_symbols: Set[str] = field(default_factory=set)

    # Behavior settings
    reject_on_warning: bool = False  # Whether to reject on warning
    allow_order_splitting: bool = True  # Whether to suggest order splitting
    enabled: bool = True  # Whether pre-trade checks are enabled


@dataclass
class OrderRequest:
    """Order request to be validated."""

    symbol: str
    side: str  # BUY or SELL
    order_type: str  # MARKET, LIMIT, etc.
    quantity: Decimal
    price: Optional[Decimal] = None  # For limit orders
    reduce_only: bool = False  # Whether this is a reduce-only order
    client_order_id: Optional[str] = None


@dataclass
class CheckDetail:
    """Detail of a single check result."""

    check_name: str
    result: CheckResult
    message: str
    current_value: Optional[Decimal] = None
    threshold: Optional[Decimal] = None
    rejection_reason: Optional[RejectionReason] = None


@dataclass
class PreTradeCheckResult:
    """Result of pre-trade risk validation."""

    passed: bool
    order_request: OrderRequest
    checks: List[CheckDetail] = field(default_factory=list)
    rejection_reasons: List[RejectionReason] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_quantity: Optional[Decimal] = None  # If split required
    suggested_price: Optional[Decimal] = None  # If price adjustment needed
    timestamp: datetime = field(default_factory=datetime.now)

    def add_check(self, detail: CheckDetail) -> None:
        """Add a check result."""
        self.checks.append(detail)
        if detail.result == CheckResult.REJECTED:
            self.passed = False
            if detail.rejection_reason:
                self.rejection_reasons.append(detail.rejection_reason)
        elif detail.result == CheckResult.WARNING:
            self.warnings.append(detail.message)

    @property
    def rejection_message(self) -> str:
        """Get combined rejection message."""
        if not self.rejection_reasons:
            return ""
        messages = [c.message for c in self.checks if c.result == CheckResult.REJECTED]
        return "; ".join(messages)


class AccountDataProvider(Protocol):
    """Protocol for account data access."""

    def get_available_balance(self, asset: str = "USDT") -> Decimal:
        """Get available balance for an asset."""
        ...

    def get_total_balance(self, asset: str = "USDT") -> Decimal:
        """Get total balance for an asset."""
        ...

    def get_position_value(self, symbol: str) -> Decimal:
        """Get current position value for a symbol."""
        ...

    def get_total_position_value(self) -> Decimal:
        """Get total value of all positions."""
        ...

    def get_position_count(self) -> int:
        """Get number of open positions."""
        ...

    def get_open_orders_count(self, symbol: str) -> int:
        """Get number of open orders for a symbol."""
        ...

    def get_margin_ratio(self) -> Optional[Decimal]:
        """Get current margin ratio (for futures)."""
        ...


class MarketDataProvider(Protocol):
    """Protocol for market data access."""

    def get_market_price(self, symbol: str) -> Optional[Decimal]:
        """Get current market price for a symbol."""
        ...

    def get_bid_price(self, symbol: str) -> Optional[Decimal]:
        """Get current bid price."""
        ...

    def get_ask_price(self, symbol: str) -> Optional[Decimal]:
        """Get current ask price."""
        ...


class PreTradeRiskChecker:
    """
    Pre-trade risk checker for order validation.

    Validates orders against risk limits before submission to exchange.
    Implements the first line of defense in risk management.

    Example:
        >>> checker = PreTradeRiskChecker(config, account_provider, market_provider)
        >>> order = OrderRequest(symbol="BTCUSDT", side="BUY", ...)
        >>> result = checker.check(order)
        >>> if result.passed:
        ...     # Submit order
        ... else:
        ...     print(f"Rejected: {result.rejection_message}")
    """

    def __init__(
        self,
        config: PreTradeConfig,
        account_provider: Optional[AccountDataProvider] = None,
        market_provider: Optional[MarketDataProvider] = None,
        circuit_breaker_check: Optional[callable] = None,
    ):
        """
        Initialize PreTradeRiskChecker.

        Args:
            config: Pre-trade check configuration
            account_provider: Provider for account data
            market_provider: Provider for market data
            circuit_breaker_check: Callable that returns True if circuit breaker is active
        """
        self._config = config
        self._account = account_provider
        self._market = market_provider
        self._circuit_breaker_check = circuit_breaker_check

        # Order frequency tracking
        self._order_timestamps: Deque[datetime] = deque(maxlen=1000)
        self._last_order_time: Optional[datetime] = None

        # Statistics
        self._total_checks: int = 0
        self._total_rejections: int = 0
        self._rejection_counts: Dict[RejectionReason, int] = {
            reason: 0 for reason in RejectionReason
        }

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def config(self) -> PreTradeConfig:
        """Get configuration."""
        return self._config

    @property
    def is_enabled(self) -> bool:
        """Check if pre-trade checks are enabled."""
        return self._config.enabled

    # =========================================================================
    # Main Check Method
    # =========================================================================

    def check(self, order: OrderRequest) -> PreTradeCheckResult:
        """
        Perform all pre-trade risk checks on an order.

        Args:
            order: Order request to validate

        Returns:
            PreTradeCheckResult with all check details
        """
        result = PreTradeCheckResult(passed=True, order_request=order)

        if not self._config.enabled:
            result.add_check(
                CheckDetail(
                    check_name="enabled",
                    result=CheckResult.PASSED,
                    message="Pre-trade checks disabled",
                )
            )
            return result

        self._total_checks += 1

        # Run all checks
        self._check_circuit_breaker(order, result)
        self._check_blacklist(order, result)
        self._check_frequency_limit(order, result)
        self._check_position_limit_symbol(order, result)
        self._check_position_limit_total(order, result)
        self._check_max_orders_per_symbol(order, result)
        self._check_order_size(order, result)
        self._check_price_deviation(order, result)
        self._check_available_funds(order, result)

        # Record order time if passed
        if result.passed:
            now = datetime.now()
            self._order_timestamps.append(now)
            self._last_order_time = now
        else:
            self._total_rejections += 1
            for reason in result.rejection_reasons:
                self._rejection_counts[reason] += 1

        # Log result
        if result.passed:
            if result.warnings:
                logger.warning(
                    f"Order {order.symbol} passed with warnings: {result.warnings}"
                )
            else:
                logger.debug(f"Order {order.symbol} passed all pre-trade checks")
        else:
            logger.warning(
                f"Order {order.symbol} rejected: {result.rejection_message}"
            )

        return result

    # =========================================================================
    # Individual Checks
    # =========================================================================

    def _check_circuit_breaker(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check if circuit breaker is active."""
        if self._circuit_breaker_check is None:
            return

        try:
            is_active = self._circuit_breaker_check()
        except Exception as e:
            logger.error(f"Error checking circuit breaker: {e}")
            is_active = False

        if is_active and not order.reduce_only:
            result.add_check(
                CheckDetail(
                    check_name="circuit_breaker",
                    result=CheckResult.REJECTED,
                    message="Circuit breaker is active, only reduce-only orders allowed",
                    rejection_reason=RejectionReason.CIRCUIT_BREAKER_ACTIVE,
                )
            )
        else:
            result.add_check(
                CheckDetail(
                    check_name="circuit_breaker",
                    result=CheckResult.PASSED,
                    message="Circuit breaker not active",
                )
            )

    def _check_blacklist(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check if symbol is blacklisted."""
        if order.symbol in self._config.blacklisted_symbols:
            result.add_check(
                CheckDetail(
                    check_name="blacklist",
                    result=CheckResult.REJECTED,
                    message=f"Symbol {order.symbol} is blacklisted",
                    rejection_reason=RejectionReason.BLACKLISTED_SYMBOL,
                )
            )
        else:
            result.add_check(
                CheckDetail(
                    check_name="blacklist",
                    result=CheckResult.PASSED,
                    message="Symbol not blacklisted",
                )
            )

    def _check_frequency_limit(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check order frequency limits."""
        now = datetime.now()

        # Check cooldown since last order
        if self._last_order_time:
            elapsed_ms = (now - self._last_order_time).total_seconds() * 1000
            if elapsed_ms < self._config.order_cooldown_ms:
                result.add_check(
                    CheckDetail(
                        check_name="order_cooldown",
                        result=CheckResult.REJECTED,
                        message=f"Order cooldown not elapsed ({elapsed_ms:.0f}ms < {self._config.order_cooldown_ms}ms)",
                        current_value=Decimal(str(elapsed_ms)),
                        threshold=Decimal(str(self._config.order_cooldown_ms)),
                        rejection_reason=RejectionReason.FREQUENCY_LIMIT,
                    )
                )
                return

        # Count orders in last second
        one_second_ago = now - timedelta(seconds=1)
        orders_last_second = sum(
            1 for ts in self._order_timestamps if ts > one_second_ago
        )

        if orders_last_second >= self._config.max_orders_per_second:
            result.add_check(
                CheckDetail(
                    check_name="orders_per_second",
                    result=CheckResult.REJECTED,
                    message=f"Max orders per second exceeded ({orders_last_second} >= {self._config.max_orders_per_second})",
                    current_value=Decimal(str(orders_last_second)),
                    threshold=Decimal(str(self._config.max_orders_per_second)),
                    rejection_reason=RejectionReason.FREQUENCY_LIMIT,
                )
            )
            return

        # Count orders in last minute
        one_minute_ago = now - timedelta(minutes=1)
        orders_last_minute = sum(
            1 for ts in self._order_timestamps if ts > one_minute_ago
        )

        if orders_last_minute >= self._config.max_orders_per_minute:
            result.add_check(
                CheckDetail(
                    check_name="orders_per_minute",
                    result=CheckResult.REJECTED,
                    message=f"Max orders per minute exceeded ({orders_last_minute} >= {self._config.max_orders_per_minute})",
                    current_value=Decimal(str(orders_last_minute)),
                    threshold=Decimal(str(self._config.max_orders_per_minute)),
                    rejection_reason=RejectionReason.FREQUENCY_LIMIT,
                )
            )
            return

        result.add_check(
            CheckDetail(
                check_name="frequency_limit",
                result=CheckResult.PASSED,
                message=f"Frequency OK ({orders_last_second}/s, {orders_last_minute}/min)",
            )
        )

    def _check_position_limit_symbol(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check position limit for specific symbol."""
        if self._account is None:
            return

        # Skip for reduce-only orders
        if order.reduce_only:
            result.add_check(
                CheckDetail(
                    check_name="position_limit_symbol",
                    result=CheckResult.PASSED,
                    message="Reduce-only order, skipping position limit check",
                )
            )
            return

        try:
            total_balance = self._account.get_total_balance()
            current_position = self._account.get_position_value(order.symbol)

            # Calculate order value
            order_value = self._calculate_order_value(order)
            if order_value is None:
                return

            # Calculate new position value
            if order.side == "BUY":
                new_position = current_position + order_value
            else:
                new_position = abs(current_position - order_value)

            # Check against limit
            max_position = total_balance * self._config.max_position_pct_per_symbol
            position_pct = new_position / total_balance if total_balance > 0 else Decimal("0")

            if new_position > max_position:
                result.add_check(
                    CheckDetail(
                        check_name="position_limit_symbol",
                        result=CheckResult.REJECTED,
                        message=f"Position for {order.symbol} would exceed limit "
                        f"({position_pct:.1%} > {self._config.max_position_pct_per_symbol:.0%})",
                        current_value=position_pct,
                        threshold=self._config.max_position_pct_per_symbol,
                        rejection_reason=RejectionReason.POSITION_LIMIT_SYMBOL,
                    )
                )

                # Suggest reduced quantity if splitting allowed
                if self._config.allow_order_splitting and current_position < max_position:
                    available_value = max_position - current_position
                    result.suggested_quantity = self._value_to_quantity(
                        order, available_value
                    )
            else:
                result.add_check(
                    CheckDetail(
                        check_name="position_limit_symbol",
                        result=CheckResult.PASSED,
                        message=f"Position limit OK ({position_pct:.1%})",
                        current_value=position_pct,
                        threshold=self._config.max_position_pct_per_symbol,
                    )
                )

        except Exception as e:
            logger.error(f"Error checking position limit for symbol: {e}")

    def _check_position_limit_total(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check total position limit across all symbols."""
        if self._account is None:
            return

        # Skip for reduce-only orders
        if order.reduce_only:
            result.add_check(
                CheckDetail(
                    check_name="position_limit_total",
                    result=CheckResult.PASSED,
                    message="Reduce-only order, skipping total position limit check",
                )
            )
            return

        try:
            total_balance = self._account.get_total_balance()
            total_position = self._account.get_total_position_value()
            position_count = self._account.get_position_count()

            # Calculate order value
            order_value = self._calculate_order_value(order)
            if order_value is None:
                return

            # Check position count
            if position_count >= self._config.max_positions_count:
                # Check if this is a new position
                current_symbol_position = self._account.get_position_value(order.symbol)
                if current_symbol_position == Decimal("0"):
                    result.add_check(
                        CheckDetail(
                            check_name="position_count",
                            result=CheckResult.REJECTED,
                            message=f"Max position count reached ({position_count} >= {self._config.max_positions_count})",
                            current_value=Decimal(str(position_count)),
                            threshold=Decimal(str(self._config.max_positions_count)),
                            rejection_reason=RejectionReason.POSITION_LIMIT_TOTAL,
                        )
                    )
                    return

            # Check total exposure
            new_total = total_position + order_value
            max_total = total_balance * self._config.max_total_position_pct
            exposure_pct = new_total / total_balance if total_balance > 0 else Decimal("0")

            if new_total > max_total:
                result.add_check(
                    CheckDetail(
                        check_name="position_limit_total",
                        result=CheckResult.REJECTED,
                        message=f"Total exposure would exceed limit "
                        f"({exposure_pct:.1%} > {self._config.max_total_position_pct:.0%})",
                        current_value=exposure_pct,
                        threshold=self._config.max_total_position_pct,
                        rejection_reason=RejectionReason.POSITION_LIMIT_TOTAL,
                    )
                )
            else:
                result.add_check(
                    CheckDetail(
                        check_name="position_limit_total",
                        result=CheckResult.PASSED,
                        message=f"Total exposure OK ({exposure_pct:.1%})",
                        current_value=exposure_pct,
                        threshold=self._config.max_total_position_pct,
                    )
                )

        except Exception as e:
            logger.error(f"Error checking total position limit: {e}")

    def _check_max_orders_per_symbol(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check max open orders per symbol."""
        if self._account is None:
            return

        try:
            open_orders = self._account.get_open_orders_count(order.symbol)

            if open_orders >= self._config.max_orders_per_symbol:
                result.add_check(
                    CheckDetail(
                        check_name="max_orders_per_symbol",
                        result=CheckResult.REJECTED,
                        message=f"Max open orders for {order.symbol} reached "
                        f"({open_orders} >= {self._config.max_orders_per_symbol})",
                        current_value=Decimal(str(open_orders)),
                        threshold=Decimal(str(self._config.max_orders_per_symbol)),
                        rejection_reason=RejectionReason.MAX_ORDERS_PER_SYMBOL,
                    )
                )
            else:
                result.add_check(
                    CheckDetail(
                        check_name="max_orders_per_symbol",
                        result=CheckResult.PASSED,
                        message=f"Open orders OK ({open_orders}/{self._config.max_orders_per_symbol})",
                    )
                )

        except Exception as e:
            logger.error(f"Error checking max orders per symbol: {e}")

    def _check_order_size(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check order size limits."""
        order_value = self._calculate_order_value(order)
        if order_value is None:
            return

        # Check minimum
        if order_value < self._config.min_order_value:
            result.add_check(
                CheckDetail(
                    check_name="min_order_value",
                    result=CheckResult.REJECTED,
                    message=f"Order value {order_value} below minimum {self._config.min_order_value}",
                    current_value=order_value,
                    threshold=self._config.min_order_value,
                    rejection_reason=RejectionReason.ORDER_SIZE_LIMIT,
                )
            )
            return

        # Check maximum value
        if order_value > self._config.max_order_value:
            result.add_check(
                CheckDetail(
                    check_name="max_order_value",
                    result=CheckResult.REJECTED
                    if not self._config.allow_order_splitting
                    else CheckResult.SPLIT_REQUIRED,
                    message=f"Order value {order_value} exceeds maximum {self._config.max_order_value}",
                    current_value=order_value,
                    threshold=self._config.max_order_value,
                    rejection_reason=RejectionReason.ORDER_SIZE_LIMIT
                    if not self._config.allow_order_splitting
                    else None,
                )
            )

            if self._config.allow_order_splitting:
                result.suggested_quantity = self._value_to_quantity(
                    order, self._config.max_order_value
                )
            return

        # Check maximum quantity if set
        if (
            self._config.max_order_quantity
            and order.quantity > self._config.max_order_quantity
        ):
            result.add_check(
                CheckDetail(
                    check_name="max_order_quantity",
                    result=CheckResult.REJECTED
                    if not self._config.allow_order_splitting
                    else CheckResult.SPLIT_REQUIRED,
                    message=f"Order quantity {order.quantity} exceeds maximum {self._config.max_order_quantity}",
                    current_value=order.quantity,
                    threshold=self._config.max_order_quantity,
                    rejection_reason=RejectionReason.ORDER_SIZE_LIMIT
                    if not self._config.allow_order_splitting
                    else None,
                )
            )

            if self._config.allow_order_splitting:
                result.suggested_quantity = self._config.max_order_quantity
            return

        result.add_check(
            CheckDetail(
                check_name="order_size",
                result=CheckResult.PASSED,
                message=f"Order size OK (value: {order_value})",
                current_value=order_value,
            )
        )

    def _check_price_deviation(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check if order price deviates too much from market."""
        # Skip for market orders
        if order.order_type == "MARKET" or order.price is None:
            result.add_check(
                CheckDetail(
                    check_name="price_deviation",
                    result=CheckResult.PASSED,
                    message="Market order, no price deviation check needed",
                )
            )
            return

        if self._market is None:
            return

        try:
            market_price = self._market.get_market_price(order.symbol)
            if market_price is None or market_price <= 0:
                logger.warning(f"Could not get market price for {order.symbol}")
                return

            # Calculate deviation
            deviation = abs(order.price - market_price) / market_price

            # Check against thresholds
            if deviation > self._config.max_price_deviation_pct:
                result.add_check(
                    CheckDetail(
                        check_name="price_deviation",
                        result=CheckResult.REJECTED,
                        message=f"Price deviation {deviation:.2%} exceeds max {self._config.max_price_deviation_pct:.1%} "
                        f"(order: {order.price}, market: {market_price})",
                        current_value=deviation,
                        threshold=self._config.max_price_deviation_pct,
                        rejection_reason=RejectionReason.PRICE_DEVIATION,
                    )
                )

                # Suggest adjusted price
                if order.side == "BUY":
                    result.suggested_price = market_price * (
                        Decimal("1") + self._config.max_price_deviation_pct
                    )
                else:
                    result.suggested_price = market_price * (
                        Decimal("1") - self._config.max_price_deviation_pct
                    )

            elif deviation > self._config.price_deviation_warning_pct:
                check_result = (
                    CheckResult.REJECTED
                    if self._config.reject_on_warning
                    else CheckResult.WARNING
                )
                result.add_check(
                    CheckDetail(
                        check_name="price_deviation",
                        result=check_result,
                        message=f"Price deviation {deviation:.2%} exceeds warning threshold "
                        f"{self._config.price_deviation_warning_pct:.1%}",
                        current_value=deviation,
                        threshold=self._config.price_deviation_warning_pct,
                        rejection_reason=RejectionReason.PRICE_DEVIATION
                        if self._config.reject_on_warning
                        else None,
                    )
                )
            else:
                result.add_check(
                    CheckDetail(
                        check_name="price_deviation",
                        result=CheckResult.PASSED,
                        message=f"Price deviation OK ({deviation:.2%})",
                        current_value=deviation,
                        threshold=self._config.max_price_deviation_pct,
                    )
                )

        except Exception as e:
            logger.error(f"Error checking price deviation: {e}")

    def _check_available_funds(
        self, order: OrderRequest, result: PreTradeCheckResult
    ) -> None:
        """Check if sufficient funds are available."""
        if self._account is None:
            return

        # Skip for reduce-only orders (they don't require new funds)
        if order.reduce_only:
            result.add_check(
                CheckDetail(
                    check_name="available_funds",
                    result=CheckResult.PASSED,
                    message="Reduce-only order, no fund check needed",
                )
            )
            return

        try:
            available = self._account.get_available_balance()
            total = self._account.get_total_balance()

            # Calculate required funds
            order_value = self._calculate_order_value(order)
            if order_value is None:
                return

            # Check margin ratio for futures (if available)
            margin_ratio = self._account.get_margin_ratio()
            if margin_ratio is not None:
                min_margin = Decimal("1") - self._config.margin_buffer_pct
                if margin_ratio < min_margin:
                    result.add_check(
                        CheckDetail(
                            check_name="margin_ratio",
                            result=CheckResult.REJECTED,
                            message=f"Margin ratio {margin_ratio:.1%} below minimum {min_margin:.1%}",
                            current_value=margin_ratio,
                            threshold=min_margin,
                            rejection_reason=RejectionReason.INSUFFICIENT_FUNDS,
                        )
                    )
                    return

            # Check available balance
            min_reserve = total * self._config.min_available_balance_pct
            available_for_trading = available - min_reserve

            if order_value > available_for_trading:
                result.add_check(
                    CheckDetail(
                        check_name="available_funds",
                        result=CheckResult.REJECTED,
                        message=f"Insufficient funds: need {order_value}, "
                        f"available {available_for_trading} (after {self._config.min_available_balance_pct:.0%} reserve)",
                        current_value=available_for_trading,
                        threshold=order_value,
                        rejection_reason=RejectionReason.INSUFFICIENT_FUNDS,
                    )
                )

                # Suggest max affordable quantity
                if self._config.allow_order_splitting and available_for_trading > 0:
                    result.suggested_quantity = self._value_to_quantity(
                        order, available_for_trading
                    )
            else:
                result.add_check(
                    CheckDetail(
                        check_name="available_funds",
                        result=CheckResult.PASSED,
                        message=f"Funds OK (need: {order_value}, available: {available_for_trading})",
                        current_value=available_for_trading,
                    )
                )

        except Exception as e:
            logger.error(f"Error checking available funds: {e}")

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _calculate_order_value(self, order: OrderRequest) -> Optional[Decimal]:
        """Calculate order value in quote currency."""
        if order.price:
            return order.quantity * order.price

        # For market orders, need market price
        if self._market:
            market_price = self._market.get_market_price(order.symbol)
            if market_price:
                return order.quantity * market_price

        logger.warning(f"Could not calculate order value for {order.symbol}")
        return None

    def _value_to_quantity(
        self, order: OrderRequest, value: Decimal
    ) -> Optional[Decimal]:
        """Convert value to quantity."""
        price = order.price
        if price is None and self._market:
            price = self._market.get_market_price(order.symbol)

        if price and price > 0:
            return value / price
        return None

    # =========================================================================
    # Blacklist Management
    # =========================================================================

    def add_to_blacklist(self, symbol: str) -> None:
        """Add symbol to blacklist."""
        self._config.blacklisted_symbols.add(symbol)
        logger.info(f"Added {symbol} to blacklist")

    def remove_from_blacklist(self, symbol: str) -> None:
        """Remove symbol from blacklist."""
        self._config.blacklisted_symbols.discard(symbol)
        logger.info(f"Removed {symbol} from blacklist")

    def get_blacklist(self) -> Set[str]:
        """Get current blacklist."""
        return self._config.blacklisted_symbols.copy()

    def clear_blacklist(self) -> None:
        """Clear all symbols from blacklist."""
        self._config.blacklisted_symbols.clear()
        logger.info("Cleared blacklist")

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict:
        """Get checker statistics."""
        rejection_rate = (
            self._total_rejections / self._total_checks
            if self._total_checks > 0
            else Decimal("0")
        )

        return {
            "total_checks": self._total_checks,
            "total_rejections": self._total_rejections,
            "rejection_rate": rejection_rate,
            "rejection_counts": dict(self._rejection_counts),
            "blacklist_size": len(self._config.blacklisted_symbols),
            "enabled": self._config.enabled,
        }

    def reset_statistics(self) -> None:
        """Reset statistics counters."""
        self._total_checks = 0
        self._total_rejections = 0
        self._rejection_counts = {reason: 0 for reason in RejectionReason}
        logger.info("Pre-trade checker statistics reset")

    # =========================================================================
    # Configuration Updates
    # =========================================================================

    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.

        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
                logger.info(f"Updated pre-trade config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")

    def enable(self) -> None:
        """Enable pre-trade checks."""
        self._config.enabled = True
        logger.info("Pre-trade checks enabled")

    def disable(self) -> None:
        """Disable pre-trade checks."""
        self._config.enabled = False
        logger.warning("Pre-trade checks disabled")
