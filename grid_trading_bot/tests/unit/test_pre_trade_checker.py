"""
Tests for PreTradeRiskChecker.

Tests pre-trade risk validation including:
- Position limits
- Order size limits
- Price deviation checks
- Frequency limits
- Fund availability checks
- Blacklist checks
"""

import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Optional
from unittest.mock import MagicMock

from src.risk.pre_trade_checker import (
    CheckResult,
    OrderRequest,
    PreTradeConfig,
    PreTradeRiskChecker,
    RejectionReason,
)


class MockAccountProvider:
    """Mock account data provider for testing."""

    def __init__(
        self,
        available_balance: Decimal = Decimal("10000"),
        total_balance: Decimal = Decimal("10000"),
        position_values: dict = None,
        position_count: int = 0,
        open_orders_counts: dict = None,
        margin_ratio: Optional[Decimal] = None,
    ):
        self._available = available_balance
        self._total = total_balance
        self._positions = position_values or {}
        self._position_count = position_count
        self._open_orders = open_orders_counts or {}
        self._margin_ratio = margin_ratio

    def get_available_balance(self, asset: str = "USDT") -> Decimal:
        return self._available

    def get_total_balance(self, asset: str = "USDT") -> Decimal:
        return self._total

    def get_position_value(self, symbol: str) -> Decimal:
        return self._positions.get(symbol, Decimal("0"))

    def get_total_position_value(self) -> Decimal:
        return sum(self._positions.values())

    def get_position_count(self) -> int:
        return self._position_count

    def get_open_orders_count(self, symbol: str) -> int:
        return self._open_orders.get(symbol, 0)

    def get_margin_ratio(self) -> Optional[Decimal]:
        return self._margin_ratio


class MockMarketProvider:
    """Mock market data provider for testing."""

    def __init__(self, prices: dict = None):
        self._prices = prices or {"BTCUSDT": Decimal("50000")}

    def get_market_price(self, symbol: str) -> Optional[Decimal]:
        return self._prices.get(symbol)

    def get_bid_price(self, symbol: str) -> Optional[Decimal]:
        price = self._prices.get(symbol)
        return price * Decimal("0.999") if price else None

    def get_ask_price(self, symbol: str) -> Optional[Decimal]:
        price = self._prices.get(symbol)
        return price * Decimal("1.001") if price else None


@pytest.fixture
def config() -> PreTradeConfig:
    """Default pre-trade config for testing."""
    return PreTradeConfig(
        max_position_pct_per_symbol=Decimal("0.30"),
        max_total_position_pct=Decimal("0.80"),
        max_positions_count=10,
        max_order_value=Decimal("5000"),
        min_order_value=Decimal("10"),
        max_orders_per_symbol=20,
        max_price_deviation_pct=Decimal("0.02"),
        price_deviation_warning_pct=Decimal("0.01"),
        max_orders_per_second=5,
        max_orders_per_minute=60,
        order_cooldown_ms=100,
        min_available_balance_pct=Decimal("0.05"),
        enabled=True,
    )


@pytest.fixture
def account_provider() -> MockAccountProvider:
    """Default account provider for testing."""
    return MockAccountProvider(
        available_balance=Decimal("10000"),
        total_balance=Decimal("10000"),
    )


@pytest.fixture
def market_provider() -> MockMarketProvider:
    """Default market provider for testing."""
    return MockMarketProvider({"BTCUSDT": Decimal("50000")})


@pytest.fixture
def checker(
    config: PreTradeConfig,
    account_provider: MockAccountProvider,
    market_provider: MockMarketProvider,
) -> PreTradeRiskChecker:
    """Pre-trade risk checker with mocked providers."""
    return PreTradeRiskChecker(
        config=config,
        account_provider=account_provider,
        market_provider=market_provider,
    )


class TestBasicChecks:
    """Test basic pre-trade check functionality."""

    def test_simple_order_passes(self, checker: PreTradeRiskChecker):
        """Test that a simple valid order passes all checks."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is True
        assert len(result.rejection_reasons) == 0

    def test_disabled_checker_passes_all(self, config: PreTradeConfig):
        """Test that disabled checker passes all orders."""
        config.enabled = False
        checker = PreTradeRiskChecker(config=config)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1000"),  # Very large
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is True


class TestBlacklist:
    """Test blacklist functionality."""

    def test_blacklisted_symbol_rejected(self, config: PreTradeConfig):
        """Test that blacklisted symbols are rejected."""
        config.blacklisted_symbols = {"BTCUSDT"}
        checker = PreTradeRiskChecker(config=config)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.BLACKLISTED_SYMBOL in result.rejection_reasons

    def test_add_remove_blacklist(self, checker: PreTradeRiskChecker):
        """Test adding and removing symbols from blacklist."""
        checker.add_to_blacklist("ETHUSDT")
        assert "ETHUSDT" in checker.get_blacklist()

        checker.remove_from_blacklist("ETHUSDT")
        assert "ETHUSDT" not in checker.get_blacklist()

    def test_clear_blacklist(self, checker: PreTradeRiskChecker):
        """Test clearing the blacklist."""
        checker.add_to_blacklist("ETHUSDT")
        checker.add_to_blacklist("BTCUSDT")
        checker.clear_blacklist()

        assert len(checker.get_blacklist()) == 0


class TestPositionLimits:
    """Test position limit checks."""

    def test_symbol_position_limit_exceeded(self, config: PreTradeConfig):
        """Test rejection when symbol position limit exceeded."""
        account = MockAccountProvider(
            total_balance=Decimal("10000"),
            position_values={"BTCUSDT": Decimal("2500")},  # 25% already
        )
        checker = PreTradeRiskChecker(config=config, account_provider=account)

        # Try to add 10% more (total would be 35%, exceeds 30% limit)
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.02"),  # ~1000 USDT at 50000
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.POSITION_LIMIT_SYMBOL in result.rejection_reasons

    def test_total_position_limit_exceeded(self, config: PreTradeConfig):
        """Test rejection when total position limit exceeded."""
        account = MockAccountProvider(
            total_balance=Decimal("10000"),
            position_values={
                "BTCUSDT": Decimal("2000"),
                "ETHUSDT": Decimal("3000"),
                "SOLUSDT": Decimal("2500"),
            },  # 75% total
            position_count=3,
        )
        checker = PreTradeRiskChecker(config=config, account_provider=account)

        # Try to add 10% more (total would be 85%, exceeds 80% limit)
        order = OrderRequest(
            symbol="BNBUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("3"),
            price=Decimal("333"),  # ~1000 USDT
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.POSITION_LIMIT_TOTAL in result.rejection_reasons

    def test_position_count_limit_exceeded(self, config: PreTradeConfig):
        """Test rejection when position count limit exceeded."""
        config.max_positions_count = 3
        account = MockAccountProvider(
            total_balance=Decimal("10000"),
            position_values={
                "BTCUSDT": Decimal("1000"),
                "ETHUSDT": Decimal("1000"),
                "SOLUSDT": Decimal("1000"),
            },
            position_count=3,
        )
        checker = PreTradeRiskChecker(config=config, account_provider=account)

        # Try to open new position
        order = OrderRequest(
            symbol="BNBUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("1"),
            price=Decimal("500"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.POSITION_LIMIT_TOTAL in result.rejection_reasons

    def test_reduce_only_bypasses_position_limits(
        self, config: PreTradeConfig
    ):
        """Test that reduce-only orders bypass position limits."""
        account = MockAccountProvider(
            total_balance=Decimal("10000"),
            position_values={"BTCUSDT": Decimal("5000")},  # 50% already
        )
        checker = PreTradeRiskChecker(config=config, account_provider=account)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="SELL",  # Reducing position
            order_type="MARKET",
            quantity=Decimal("0.05"),
            reduce_only=True,
        )

        result = checker.check(order)

        # Should not be rejected for position limits
        assert RejectionReason.POSITION_LIMIT_SYMBOL not in result.rejection_reasons
        assert RejectionReason.POSITION_LIMIT_TOTAL not in result.rejection_reasons


class TestOrderSizeLimits:
    """Test order size limit checks."""

    def test_order_value_too_large(self, config: PreTradeConfig):
        """Test rejection when order value exceeds limit."""
        # Use large balance to avoid position/fund limits
        config.allow_order_splitting = False  # Disable splitting to get rejection
        account = MockAccountProvider(
            available_balance=Decimal("100000"),
            total_balance=Decimal("100000"),
        )
        market = MockMarketProvider({"BTCUSDT": Decimal("50000")})
        checker = PreTradeRiskChecker(
            config=config, account_provider=account, market_provider=market
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.2"),  # 10000 USDT at 50000
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.ORDER_SIZE_LIMIT in result.rejection_reasons

    def test_order_value_too_small(self, config: PreTradeConfig):
        """Test rejection when order value below minimum."""
        account = MockAccountProvider(
            available_balance=Decimal("10000"),
            total_balance=Decimal("10000"),
        )
        market = MockMarketProvider({"BTCUSDT": Decimal("50000")})
        checker = PreTradeRiskChecker(
            config=config, account_provider=account, market_provider=market
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.0001"),  # 5 USDT at 50000
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.ORDER_SIZE_LIMIT in result.rejection_reasons

    def test_suggested_quantity_on_split(self, config: PreTradeConfig):
        """Test that suggested quantity is provided when splitting allowed."""
        config.allow_order_splitting = True
        market = MockMarketProvider({"BTCUSDT": Decimal("50000")})
        checker = PreTradeRiskChecker(config=config, market_provider=market)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.2"),  # 10000 USDT
            price=Decimal("50000"),
        )

        result = checker.check(order)

        # Should suggest max allowed quantity
        assert result.suggested_quantity is not None
        assert result.suggested_quantity <= Decimal("0.1")  # 5000/50000


class TestPriceDeviation:
    """Test price deviation checks."""

    def test_price_deviation_rejected(
        self, config: PreTradeConfig, market_provider: MockMarketProvider
    ):
        """Test rejection when price deviates too much."""
        checker = PreTradeRiskChecker(
            config=config, market_provider=market_provider
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("52000"),  # 4% above market (50000)
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.PRICE_DEVIATION in result.rejection_reasons

    def test_price_deviation_warning(
        self, config: PreTradeConfig, market_provider: MockMarketProvider
    ):
        """Test warning when price deviates slightly."""
        config.reject_on_warning = False
        checker = PreTradeRiskChecker(
            config=config, market_provider=market_provider
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50750"),  # 1.5% above market
        )

        result = checker.check(order)

        assert result.passed is True
        assert len(result.warnings) > 0

    def test_market_order_bypasses_price_check(self, checker: PreTradeRiskChecker):
        """Test that market orders bypass price deviation check."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.01"),
        )

        result = checker.check(order)

        # Should not have price deviation rejection
        assert RejectionReason.PRICE_DEVIATION not in result.rejection_reasons


class TestFrequencyLimits:
    """Test order frequency limit checks."""

    def test_orders_per_second_limit(self, config: PreTradeConfig):
        """Test rejection when orders per second limit exceeded."""
        config.max_orders_per_second = 2
        config.order_cooldown_ms = 0  # Disable cooldown for this test
        checker = PreTradeRiskChecker(config=config)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.01"),
        )

        # First two should pass
        result1 = checker.check(order)
        result2 = checker.check(order)
        assert result1.passed is True
        assert result2.passed is True

        # Third should fail
        result3 = checker.check(order)
        assert result3.passed is False
        assert RejectionReason.FREQUENCY_LIMIT in result3.rejection_reasons


class TestFundAvailability:
    """Test fund availability checks."""

    def test_insufficient_funds_rejected(self, config: PreTradeConfig):
        """Test rejection when insufficient funds."""
        account = MockAccountProvider(
            available_balance=Decimal("100"),
            total_balance=Decimal("1000"),
        )
        market = MockMarketProvider({"BTCUSDT": Decimal("50000")})
        checker = PreTradeRiskChecker(
            config=config, account_provider=account, market_provider=market
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),  # 500 USDT
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.INSUFFICIENT_FUNDS in result.rejection_reasons

    def test_margin_ratio_check(self, config: PreTradeConfig):
        """Test rejection when margin ratio too low."""
        account = MockAccountProvider(
            available_balance=Decimal("5000"),
            total_balance=Decimal("10000"),
            margin_ratio=Decimal("0.75"),  # Below 80% buffer
        )
        checker = PreTradeRiskChecker(config=config, account_provider=account)

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.INSUFFICIENT_FUNDS in result.rejection_reasons


class TestCircuitBreaker:
    """Test circuit breaker integration."""

    def test_circuit_breaker_active_rejects_new_orders(self, config: PreTradeConfig):
        """Test that active circuit breaker rejects non-reduce-only orders."""
        checker = PreTradeRiskChecker(
            config=config, circuit_breaker_check=lambda: True
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="MARKET",
            quantity=Decimal("0.01"),
        )

        result = checker.check(order)

        assert result.passed is False
        assert RejectionReason.CIRCUIT_BREAKER_ACTIVE in result.rejection_reasons

    def test_circuit_breaker_allows_reduce_only(self, config: PreTradeConfig):
        """Test that active circuit breaker allows reduce-only orders."""
        checker = PreTradeRiskChecker(
            config=config, circuit_breaker_check=lambda: True
        )

        order = OrderRequest(
            symbol="BTCUSDT",
            side="SELL",
            order_type="MARKET",
            quantity=Decimal("0.01"),
            reduce_only=True,
        )

        result = checker.check(order)

        # Should not be rejected for circuit breaker
        assert RejectionReason.CIRCUIT_BREAKER_ACTIVE not in result.rejection_reasons


class TestStatistics:
    """Test statistics tracking."""

    def test_statistics_tracking(self, checker: PreTradeRiskChecker):
        """Test that statistics are properly tracked."""
        checker.add_to_blacklist("BADCOIN")

        # Valid order
        order1 = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        checker.check(order1)

        # Rejected order
        order2 = OrderRequest(
            symbol="BADCOIN",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("100"),
        )
        checker.check(order2)

        stats = checker.get_statistics()

        assert stats["total_checks"] == 2
        assert stats["total_rejections"] == 1
        assert stats["rejection_counts"][RejectionReason.BLACKLISTED_SYMBOL] == 1

    def test_reset_statistics(self, checker: PreTradeRiskChecker):
        """Test statistics reset."""
        order = OrderRequest(
            symbol="BTCUSDT",
            side="BUY",
            order_type="LIMIT",
            quantity=Decimal("0.01"),
            price=Decimal("50000"),
        )
        checker.check(order)

        checker.reset_statistics()
        stats = checker.get_statistics()

        assert stats["total_checks"] == 0
        assert stats["total_rejections"] == 0


class TestConfigUpdates:
    """Test configuration updates."""

    def test_update_config(self, checker: PreTradeRiskChecker):
        """Test updating configuration parameters."""
        checker.update_config(max_order_value=Decimal("20000"))

        assert checker.config.max_order_value == Decimal("20000")

    def test_enable_disable(self, checker: PreTradeRiskChecker):
        """Test enabling and disabling checker."""
        checker.disable()
        assert checker.is_enabled is False

        checker.enable()
        assert checker.is_enabled is True
