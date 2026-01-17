"""
Pytest configuration and fixtures for grid trading bot tests.
"""

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Optional

import pytest

from core.models import Kline, MarketType

from src.strategy.grid import (
    ATRData,
    GridConfig,
    GridLevel,
    GridSetup,
    GridType,
    LevelSide,
    LevelState,
    RiskConfig,
    RiskLevel,
)
from tests.mocks import MockDataManager, MockExchangeClient, MockNotifier


# =============================================================================
# Sample Kline Data Fixture
# =============================================================================


@pytest.fixture
def sample_klines() -> list[Kline]:
    """
    Generate 50 simulated klines for testing.

    Properties:
    - Starting price: 50000
    - Volatility: ±5%
    - Includes uptrend, downtrend, and sideways movements
    """
    klines = []
    base_price = Decimal("50000")
    current_price = base_price
    start_time = datetime.now(timezone.utc) - timedelta(hours=200)

    # Set seed for reproducibility
    random.seed(42)

    for i in range(50):
        # Simulate price movement with different phases
        if i < 15:
            # Uptrend phase
            change_pct = Decimal(str(random.uniform(-2, 5))) / Decimal("100")
        elif i < 30:
            # Downtrend phase
            change_pct = Decimal(str(random.uniform(-5, 2))) / Decimal("100")
        else:
            # Sideways/volatile phase
            change_pct = Decimal(str(random.uniform(-3, 3))) / Decimal("100")

        # Calculate OHLC
        open_price = current_price
        close_price = open_price * (Decimal("1") + change_pct)

        # Random high/low within 2% of open-close range
        high_extra = Decimal(str(random.uniform(0, 2))) / Decimal("100")
        low_extra = Decimal(str(random.uniform(0, 2))) / Decimal("100")

        high_price = max(open_price, close_price) * (Decimal("1") + high_extra)
        low_price = min(open_price, close_price) * (Decimal("1") - low_extra)

        # Random volume
        volume = Decimal(str(random.uniform(100, 1000)))

        kline = Kline(
            symbol="BTCUSDT",
            interval="4h",
            open_time=start_time + timedelta(hours=i * 4),
            close_time=start_time + timedelta(hours=(i + 1) * 4 - 1),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume,
            quote_volume=volume * close_price,
            trades=random.randint(100, 1000),
        )
        klines.append(kline)

        current_price = close_price

    return klines


@pytest.fixture
def atr_test_klines() -> list[Kline]:
    """
    Generate specific klines for ATR calculation verification.

    High:  [100, 102, 101, 103, 102]
    Low:   [98,  99,  98,  100, 99]
    Close: [99,  101, 100, 102, 101]
    """
    base_time = datetime.now(timezone.utc) - timedelta(hours=20)

    highs = [Decimal("100"), Decimal("102"), Decimal("101"), Decimal("103"), Decimal("102")]
    lows = [Decimal("98"), Decimal("99"), Decimal("98"), Decimal("100"), Decimal("99")]
    closes = [Decimal("99"), Decimal("101"), Decimal("100"), Decimal("102"), Decimal("101")]
    opens = [Decimal("99"), Decimal("100"), Decimal("101"), Decimal("101"), Decimal("102")]

    klines = []
    for i in range(5):
        kline = Kline(
            symbol="TESTUSDT",
            interval="4h",
            open_time=base_time + timedelta(hours=i * 4),
            close_time=base_time + timedelta(hours=(i + 1) * 4 - 1),
            open=opens[i],
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=Decimal("1000"),
            quote_volume=Decimal("100000"),
            trades=100,
        )
        klines.append(kline)

    return klines


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_exchange() -> MockExchangeClient:
    """Create a mock exchange client for testing."""
    exchange = MockExchangeClient()
    exchange.set_price(Decimal("50000"))  # Default price
    exchange.set_balance("USDT", Decimal("100000"))
    exchange.set_balance("BTC", Decimal("2"))
    return exchange


@pytest.fixture
def mock_data_manager() -> MockDataManager:
    """Create a mock data manager for testing."""
    return MockDataManager()


@pytest.fixture
def mock_notifier() -> MockNotifier:
    """Create a mock notifier for testing."""
    return MockNotifier()


# =============================================================================
# Grid Configuration Fixtures
# =============================================================================


@pytest.fixture
def grid_config() -> GridConfig:
    """Create a default grid configuration for testing."""
    return GridConfig(
        symbol="BTCUSDT",
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MEDIUM,
        grid_type=GridType.GEOMETRIC,
        min_grid_count=5,
        max_grid_count=50,
        min_order_value=Decimal("10"),
        atr_period=14,
        atr_timeframe="4h",
    )


@pytest.fixture
def manual_grid_config() -> GridConfig:
    """Create a grid configuration with manual overrides."""
    return GridConfig(
        symbol="BTCUSDT",
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MEDIUM,
        grid_type=GridType.GEOMETRIC,
        manual_upper_price=Decimal("55000"),
        manual_lower_price=Decimal("45000"),
        manual_grid_count=10,
    )


@pytest.fixture
def arithmetic_grid_config() -> GridConfig:
    """Create an arithmetic grid configuration."""
    return GridConfig(
        symbol="BTCUSDT",
        total_investment=Decimal("10000"),
        risk_level=RiskLevel.MEDIUM,
        grid_type=GridType.ARITHMETIC,
        manual_upper_price=Decimal("55000"),
        manual_lower_price=Decimal("45000"),
        manual_grid_count=10,
    )


# =============================================================================
# Grid Setup Fixtures
# =============================================================================


@pytest.fixture
def grid_setup(grid_config: GridConfig) -> GridSetup:
    """Create a sample grid setup for testing."""
    # Create ATR data
    atr_data = ATRData(
        value=Decimal("1000"),
        period=14,
        timeframe="4h",
        current_price=Decimal("50000"),
    )

    # Create grid levels (10 levels from 45000 to 55000)
    levels = []
    current_price = Decimal("50000")
    lower_price = Decimal("45000")
    upper_price = Decimal("55000")
    grid_count = 10

    # Calculate geometric spacing
    ratio = (upper_price / lower_price) ** (Decimal("1") / Decimal(grid_count))

    for i in range(grid_count + 1):
        price = lower_price * (ratio ** Decimal(i))

        # Determine side based on current price
        side = LevelSide.BUY if price < current_price else LevelSide.SELL

        level = GridLevel(
            index=i,
            price=price,
            side=side,
            state=LevelState.EMPTY,
            allocated_amount=Decimal("1000"),  # 10000 / 10
        )
        levels.append(level)

    return GridSetup(
        config=grid_config,
        atr_data=atr_data,
        upper_price=upper_price,
        lower_price=lower_price,
        current_price=current_price,
        grid_count=grid_count,
        grid_spacing_percent=Decimal("2.03"),  # Approximate geometric spacing
        amount_per_grid=Decimal("1000"),
        levels=levels,
        expected_profit_per_trade=Decimal("1.83"),  # 2.03 - 0.2 fees
    )


# =============================================================================
# Risk Configuration Fixture
# =============================================================================


@pytest.fixture
def risk_config() -> RiskConfig:
    """Create a default risk configuration for testing."""
    return RiskConfig(
        upper_breakout_action="pause",
        lower_breakout_action="pause",
        stop_loss_percent=Decimal("20"),
        breakout_buffer=Decimal("0.5"),
        auto_reset_enabled=False,
        daily_loss_limit=Decimal("5"),
        max_consecutive_losses=5,
        volatility_threshold=Decimal("10"),
        order_failure_threshold=3,
        health_check_interval=30,
        monitoring_interval=5,
    )


# =============================================================================
# Helper Fixtures
# =============================================================================


@pytest.fixture
def current_price() -> Decimal:
    """Default current price for testing."""
    return Decimal("50000")


@pytest.fixture
def market_type() -> MarketType:
    """Default market type for testing."""
    return MarketType.SPOT
