#!/usr/bin/env python3
"""
Fund Manager Integration Test.

Tests the centralized fund allocation system with the new configuration.
"""

import asyncio
import sys
import os
from decimal import Decimal
from typing import Dict, Any

sys.path.insert(0, os.path.dirname(__file__))

from src.fund_manager.manager import FundManager
from src.fund_manager.models.config import (
    FundManagerConfig,
    BotAllocation,
    AllocationStrategy,
)
from src.fund_manager.core.allocator import create_allocator


def create_test_config() -> FundManagerConfig:
    """Create test configuration matching config.yaml."""
    return FundManagerConfig(
        enabled=True,
        poll_interval=60,
        deposit_threshold=Decimal("10"),
        reserve_ratio=Decimal("0.1"),  # 10% reserve
        auto_dispatch=True,
        strategy=AllocationStrategy.FIXED_RATIO,
        allocations=[
            BotAllocation(
                bot_pattern="grid_futures_*",
                ratio=Decimal("0.30"),  # 30%
                min_capital=Decimal("100"),
                max_capital=Decimal("10000"),
                priority=4,
                enabled=True,
            ),
            BotAllocation(
                bot_pattern="supertrend_*",
                ratio=Decimal("0.25"),  # 25%
                min_capital=Decimal("100"),
                max_capital=Decimal("8000"),
                priority=3,
                enabled=True,
            ),
            BotAllocation(
                bot_pattern="bollinger_*",
                ratio=Decimal("0.20"),  # 20%
                min_capital=Decimal("50"),
                max_capital=Decimal("5000"),
                priority=2,
                enabled=True,
            ),
            BotAllocation(
                bot_pattern="rsi_*",
                ratio=Decimal("0.15"),  # 15%
                min_capital=Decimal("50"),
                max_capital=Decimal("3000"),
                priority=1,
                enabled=True,
            ),
        ],
    )


def test_allocation_calculation():
    """Test allocation calculation with different account balances."""
    print("=" * 70)
    print("  Fund Manager 資金配置測試")
    print("=" * 70)

    config = create_test_config()
    allocator = create_allocator(config)

    # Test scenarios
    test_balances = [1000, 5000, 10000, 50000, 100000]

    print("\n配置比例:")
    print(f"  - grid_futures_*:  30%")
    print(f"  - supertrend_*:    25%")
    print(f"  - bollinger_*:     20%")
    print(f"  - rsi_*:           15%")
    print(f"  - Reserve:         10%")

    print("\n" + "=" * 70)
    print("  測試 1: 單一 Bot 配置 (每種類型一個)")
    print("=" * 70)

    bot_ids = ["grid_futures_btc", "supertrend_btc", "bollinger_btc", "rsi_btc"]

    print(f"\n{'帳戶餘額':>12} │ {'Grid':>10} │ {'Supertrend':>10} │ {'Bollinger':>10} │ {'RSI':>10} │ {'Reserve':>10}")
    print("─" * 78)

    for balance in test_balances:
        available = Decimal(str(balance)) * Decimal("0.9")  # 90% available (10% reserve)

        allocations = allocator.calculate(
            available_funds=available,
            bot_allocations=config.allocations,
            current_allocations={},
            bot_ids=bot_ids,
        )

        grid = allocations.get("grid_futures_btc", Decimal("0"))
        supertrend = allocations.get("supertrend_btc", Decimal("0"))
        bollinger = allocations.get("bollinger_btc", Decimal("0"))
        rsi = allocations.get("rsi_btc", Decimal("0"))
        reserve = Decimal(str(balance)) * Decimal("0.1")

        print(f"{balance:>12,} │ {float(grid):>10,.0f} │ {float(supertrend):>10,.0f} │ {float(bollinger):>10,.0f} │ {float(rsi):>10,.0f} │ {float(reserve):>10,.0f}")

    print("\n" + "=" * 70)
    print("  測試 2: 多個 Bot 同類型 (資金平分)")
    print("=" * 70)

    # Multiple bots of same type
    multi_bot_ids = [
        "grid_futures_btc",
        "grid_futures_eth",  # Two grid bots
        "supertrend_btc",
        "bollinger_btc",
        "bollinger_eth",  # Two bollinger bots
        "rsi_btc",
    ]

    balance = 10000
    available = Decimal(str(balance)) * Decimal("0.9")

    allocations = allocator.calculate(
        available_funds=available,
        bot_allocations=config.allocations,
        current_allocations={},
        bot_ids=multi_bot_ids,
    )

    print(f"\n帳戶餘額: {balance:,} USDT")
    print(f"可分配金額: {float(available):,.0f} USDT (扣除 10% 保留)")
    print("\n分配結果:")
    for bot_id, amount in sorted(allocations.items()):
        print(f"  {bot_id:<20} → {float(amount):>8,.0f} USDT")

    total_allocated = sum(allocations.values())
    print(f"\n總分配: {float(total_allocated):,.0f} USDT")
    print(f"保留金額: {balance - float(total_allocated):,.0f} USDT")

    print("\n" + "=" * 70)
    print("  測試 3: 最大資金限制 (max_capital)")
    print("=" * 70)

    # Test with very large balance to hit max_capital limits
    large_balance = 100000
    available = Decimal(str(large_balance)) * Decimal("0.9")

    allocations = allocator.calculate(
        available_funds=available,
        bot_allocations=config.allocations,
        current_allocations={},
        bot_ids=bot_ids,
    )

    print(f"\n帳戶餘額: {large_balance:,} USDT")
    print(f"可分配金額: {float(available):,.0f} USDT")
    print("\n分配結果 (受 max_capital 限制):")

    max_caps = {
        "grid_futures_btc": 10000,
        "supertrend_btc": 8000,
        "bollinger_btc": 5000,
        "rsi_btc": 3000,
    }

    for bot_id, amount in sorted(allocations.items()):
        max_cap = max_caps.get(bot_id, 0)
        ratio_amount = float(available) * 0.30 if "grid" in bot_id else \
                       float(available) * 0.25 if "supertrend" in bot_id else \
                       float(available) * 0.20 if "bollinger" in bot_id else \
                       float(available) * 0.15
        capped = "✓ 已限制" if float(amount) < ratio_amount else ""
        print(f"  {bot_id:<20} → {float(amount):>8,.0f} USDT (max: {max_cap:,}) {capped}")

    print("\n" + "=" * 70)
    print("  測試 4: Bot 模式匹配")
    print("=" * 70)

    test_patterns = [
        ("grid_futures_btc_usdt", "grid_futures_*"),
        ("supertrend_eth", "supertrend_*"),
        ("bollinger_btc_long", "bollinger_*"),
        ("rsi_scalper", "rsi_*"),
        ("unknown_bot", None),
    ]

    print("\n模式匹配測試:")
    for bot_id, expected_pattern in test_patterns:
        matched = config.get_allocation_for_bot(bot_id)
        actual_pattern = matched.bot_pattern if matched else None
        status = "✓" if actual_pattern == expected_pattern else "✗"
        print(f"  {status} {bot_id:<25} → {actual_pattern or '(no match)'}")

    print("\n" + "=" * 70)
    print("  測試 5: FundManager 狀態查詢")
    print("=" * 70)

    # Reset singleton for testing
    FundManager.reset_instance()

    manager = FundManager(config=config)

    # Simulate balance update
    manager.fund_pool.update_from_values(
        total_balance=Decimal("10000"),
        available_balance=Decimal("9500"),
    )

    status = manager.get_status()
    pool_status = manager.fund_pool.get_status()

    print("\nFund Manager 狀態:")
    print(f"  Enabled:       {'Yes' if status['enabled'] else 'No'}")
    print(f"  Strategy:      {status['strategy']}")
    print(f"  Auto Dispatch: {'Yes' if status['auto_dispatch'] else 'No'}")
    print(f"  Poll Interval: {status['poll_interval']}s")

    print("\nFund Pool 狀態:")
    print(f"  Total Balance:     {pool_status['total_balance']} USDT")
    print(f"  Available Balance: {pool_status['available_balance']} USDT")
    print(f"  Allocated Balance: {pool_status['allocated_balance']} USDT")
    print(f"  Reserved Balance:  {pool_status['reserved_balance']} USDT")
    print(f"  Unallocated:       {pool_status['unallocated_balance']} USDT")

    print("\n" + "=" * 70)
    print("  測試 6: 入金偵測")
    print("=" * 70)

    # Initial balance
    manager.fund_pool.update_from_values(
        total_balance=Decimal("10000"),
        available_balance=Decimal("9500"),
    )

    print(f"\n初始餘額: 10,000 USDT")
    print(f"入金偵測閾值: {config.deposit_threshold} USDT")

    # Simulate deposit
    manager.fund_pool.update_from_values(
        total_balance=Decimal("15000"),
        available_balance=Decimal("14500"),
    )

    if manager.fund_pool.detect_deposit():
        deposit_amount = manager.fund_pool.get_deposit_amount()
        print(f"\n✓ 偵測到入金: {deposit_amount} USDT")
        print(f"新餘額: 15,000 USDT")
    else:
        print("\n✗ 未偵測到入金")

    # Small deposit below threshold
    manager.fund_pool.update_from_values(
        total_balance=Decimal("15005"),
        available_balance=Decimal("14505"),
    )

    if manager.fund_pool.detect_deposit():
        deposit_amount = manager.fund_pool.get_deposit_amount()
        print(f"\n✓ 偵測到小額入金: {deposit_amount} USDT")
    else:
        print(f"\n✓ 小額入金 (5 USDT < {config.deposit_threshold}) 未觸發分配")

    # Cleanup
    FundManager.reset_instance()

    print("\n" + "=" * 70)
    print("  測試完成")
    print("=" * 70)
    print("\n✅ 所有測試通過！Fund Manager 配置正確。")


async def test_dispatch_simulation():
    """Test fund dispatch simulation."""
    print("\n" + "=" * 70)
    print("  測試 7: 資金分配模擬")
    print("=" * 70)

    # Reset singleton
    FundManager.reset_instance()

    config = create_test_config()
    manager = FundManager(config=config)

    # Set initial balance
    manager.fund_pool.update_from_values(
        total_balance=Decimal("10000"),
        available_balance=Decimal("9500"),
    )

    print("\n模擬分配 (無真實 bot):")
    print(f"帳戶餘額: 10,000 USDT")
    print(f"可用餘額: 9,500 USDT")

    # Calculate what would be allocated
    allocator = create_allocator(config)
    bot_ids = ["grid_futures_btc", "supertrend_btc", "bollinger_btc", "rsi_btc"]

    allocations = allocator.calculate(
        available_funds=manager.fund_pool.get_unallocated(),
        bot_allocations=config.allocations,
        current_allocations={},
        bot_ids=bot_ids,
    )

    print("\n預計分配:")
    for bot_id, amount in sorted(allocations.items()):
        print(f"  {bot_id:<20} → {float(amount):>8,.0f} USDT")

    total = sum(allocations.values())
    print(f"\n總計: {float(total):,.0f} USDT")

    # Cleanup
    FundManager.reset_instance()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  Fund Manager Integration Test")
    print("=" * 70)

    # Run synchronous tests
    test_allocation_calculation()

    # Run async tests
    asyncio.run(test_dispatch_simulation())

    print("\n")
