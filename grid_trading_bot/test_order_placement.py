#!/usr/bin/env python3
"""
Test Order Placement Script.

Tests order placement and stop loss functionality for each bot type.
Uses minimal amounts and immediately cancels orders.

WARNING: This script places REAL orders on LIVE exchange.
"""

import asyncio
import os
import sys
from decimal import Decimal

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.core import get_logger
from src.core.models import OrderSide, OrderType
from src.exchange import ExchangeClient
from src.exchange.binance.futures_api import PositionSide

logger = get_logger(__name__)

# Test configuration
SYMBOL = "BTCUSDT"
LEVERAGE = 50  # Higher leverage = less margin required
MIN_NOTIONAL = Decimal("100")  # Minimum notional value for BTCUSDT futures


async def create_exchange_client() -> ExchangeClient:
    """Create and connect exchange client."""
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('BINANCE_TESTNET', 'false').lower() == 'true'

    print(f"Testnet mode: {testnet}")

    client = ExchangeClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    await client.connect()
    return client


async def get_symbol_info(client: ExchangeClient) -> dict:
    """Get symbol trading info."""
    info = await client.futures.get_exchange_info()
    for symbol_info in info.get("symbols", []):
        if symbol_info["symbol"] == SYMBOL:
            return symbol_info
    return {}


async def test_futures_order(client: ExchangeClient):
    """Test futures order placement and stop loss."""
    print("\n" + "=" * 60)
    print("Testing Futures Order Placement")
    print("=" * 60)

    # Get current price
    ticker = await client.futures.get_ticker(SYMBOL)
    current_price = ticker.price
    print(f"\nCurrent {SYMBOL} price: {current_price}")

    # Get account balance
    try:
        balance = await client.futures.get_balance()
        for b in balance:
            if hasattr(b, 'asset') and b.asset == "USDT":
                print(f"USDT Balance: {b.available_balance} (available)")
                break
            elif isinstance(b, dict) and b.get('asset') == 'USDT':
                print(f"USDT Balance: {b.get('availableBalance', 'N/A')} (available)")
                break
    except Exception as e:
        print(f"Could not get balance: {e}")

    # Calculate minimum quantity
    # Min notional is $100 for BTCUSDT futures, min qty is 0.001
    # Use ROUND_UP to ensure we meet the minimum notional
    from decimal import ROUND_UP
    min_qty = (MIN_NOTIONAL / current_price).quantize(Decimal("0.001"), rounding=ROUND_UP)
    min_qty = max(min_qty, Decimal("0.001"))
    notional_value = min_qty * current_price
    print(f"Minimum quantity for test: {min_qty} BTC (~${notional_value:.2f})")

    # Set leverage
    print(f"\n1. Setting leverage to {LEVERAGE}x...")
    try:
        await client.futures.set_leverage(SYMBOL, LEVERAGE)
        print(f"   ✓ Leverage set to {LEVERAGE}x")
    except Exception as e:
        print(f"   ✗ Failed to set leverage: {e}")
        return False

    # Set margin type
    print(f"\n2. Setting margin type to ISOLATED...")
    try:
        await client.futures.set_margin_type(SYMBOL, "ISOLATED")
        print(f"   ✓ Margin type set to ISOLATED")
    except Exception as e:
        if "No need to change margin type" in str(e):
            print(f"   ✓ Margin type already ISOLATED")
        else:
            print(f"   ✗ Failed to set margin type: {e}")

    # Test 1: Place LIMIT order (won't fill immediately)
    print(f"\n3. Placing LIMIT BUY order (far below market)...")
    limit_price = (current_price * Decimal("0.95")).quantize(Decimal("0.1"))  # 5% below

    try:
        order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=min_qty,
            price=limit_price,
            time_in_force="GTC",
        )
        print(f"   ✓ Limit order placed: ID={order.order_id}, Price={limit_price}")

        # Cancel the order immediately
        print(f"\n4. Canceling limit order...")
        cancelled = await client.futures.cancel_order(SYMBOL, order_id=str(order.order_id))
        print(f"   ✓ Order cancelled: {cancelled.status}")
    except Exception as e:
        print(f"   ✗ Limit order test failed: {e}")
        return False

    # Test 2: Place STOP_MARKET order (stop loss)
    print(f"\n5. Placing STOP_MARKET order (stop loss simulation)...")
    stop_price = (current_price * Decimal("0.97")).quantize(Decimal("0.1"))  # 3% below

    try:
        stop_order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.SELL,
            order_type="STOP_MARKET",
            quantity=min_qty,
            stop_price=stop_price,
            reduce_only=False,  # For testing, not reduce only
        )
        print(f"   ✓ Stop order placed: ID={stop_order.order_id}, Stop Price={stop_price}")

        # Cancel the stop order immediately
        print(f"\n6. Canceling stop order...")
        cancelled = await client.futures.cancel_order(SYMBOL, order_id=str(stop_order.order_id))
        print(f"   ✓ Stop order cancelled: {cancelled.status}")
    except Exception as e:
        print(f"   ✗ Stop order test failed: {e}")
        return False

    # Test 3: Place actual MARKET order and stop loss (REAL TRADE)
    print(f"\n" + "=" * 60)
    print("REAL TRADE TEST")
    print("=" * 60)
    confirm = input(f"\n⚠️  Place REAL market order for {min_qty} BTC (~${MIN_NOTIONAL})? [y/N]: ")

    if confirm.lower() != 'y':
        print("Skipped real trade test.")
        return True

    try:
        # Place market order
        print(f"\n7. Placing MARKET BUY order...")
        market_order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=min_qty,
        )
        entry_price = Decimal(str(market_order.avg_price or current_price))
        print(f"   ✓ Market order filled: ID={market_order.order_id}, Price={entry_price}")

        # Place stop loss
        print(f"\n8. Placing stop loss order...")
        sl_price = (entry_price * Decimal("0.99")).quantize(Decimal("0.1"))  # 1% below
        stop_loss = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.SELL,
            order_type="STOP_MARKET",
            quantity=min_qty,
            stop_price=sl_price,
            reduce_only=True,
        )
        print(f"   ✓ Stop loss placed: ID={stop_loss.order_id}, Stop={sl_price}")

        # Cancel stop loss
        print(f"\n9. Canceling stop loss...")
        await client.futures.cancel_order(SYMBOL, order_id=str(stop_loss.order_id))
        print(f"   ✓ Stop loss cancelled")

        # Close position
        print(f"\n10. Closing position (MARKET SELL)...")
        close_order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=min_qty,
            reduce_only=True,
        )
        print(f"   ✓ Position closed: ID={close_order.order_id}")

        # Calculate PnL
        exit_price = Decimal(str(close_order.avg_price or current_price))
        pnl = (exit_price - entry_price) * min_qty
        print(f"\n   Test trade PnL: {pnl:.4f} USDT")

    except Exception as e:
        print(f"   ✗ Real trade test failed: {e}")
        # Try to close any open position
        try:
            print("\n   Attempting to close any open position...")
            positions = await client.futures.get_positions(SYMBOL)
            for pos in positions:
                if abs(pos.position_amt) > 0:
                    side = OrderSide.SELL if pos.position_amt > 0 else OrderSide.BUY
                    await client.futures.create_order(
                        symbol=SYMBOL,
                        side=side,
                        order_type=OrderType.MARKET,
                        quantity=abs(pos.position_amt),
                        reduce_only=True,
                    )
                    print(f"   ✓ Closed position: {pos.position_amt}")
        except Exception as close_err:
            print(f"   ✗ Failed to close position: {close_err}")
        return False

    return True


async def check_open_positions(client: ExchangeClient):
    """Check and display open positions."""
    print("\n" + "=" * 60)
    print("Current Open Positions")
    print("=" * 60)

    try:
        positions = await client.futures.get_positions()
        active_positions = [p for p in positions if abs(p.position_amt) > 0]

        if not active_positions:
            print("No open positions.")
        else:
            for pos in active_positions:
                print(f"  {pos.symbol}: {pos.position_amt} @ {pos.entry_price} (PnL: {pos.unrealized_pnl})")
    except Exception as e:
        print(f"Failed to get positions: {e}")


async def check_open_orders(client: ExchangeClient):
    """Check and display open orders."""
    print("\n" + "=" * 60)
    print("Current Open Orders")
    print("=" * 60)

    try:
        orders = await client.futures.get_open_orders(SYMBOL)

        if not orders:
            print("No open orders.")
        else:
            for order in orders:
                print(f"  {order.order_id}: {order.side} {order.quantity} @ {order.price} ({order.order_type})")
    except Exception as e:
        print(f"Failed to get orders: {e}")


async def main():
    """Main test function."""
    print("\n" + "=" * 60)
    print("       Order Placement Test Script")
    print("=" * 60)
    print("\n⚠️  WARNING: This script tests REAL order placement!")
    print("    - Limit/Stop orders will be cancelled immediately")
    print("    - Market orders require confirmation")
    print()

    client = None
    try:
        # Connect to exchange
        print("Connecting to exchange...")
        client = await create_exchange_client()
        print("✓ Connected to exchange")

        # Check current state
        await check_open_positions(client)
        await check_open_orders(client)

        # Run tests
        success = await test_futures_order(client)

        if success:
            print("\n" + "=" * 60)
            print("✓ All tests completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ Some tests failed!")
            print("=" * 60)

        # Final state check
        await check_open_positions(client)
        await check_open_orders(client)

    except Exception as e:
        print(f"\n❌ Test error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            await client.disconnect()
            print("\nDisconnected from exchange.")


if __name__ == "__main__":
    asyncio.run(main())
