#!/usr/bin/env python3
"""
Test Real Trade - Simple market order and stop loss test.

WARNING: This script places REAL orders on LIVE exchange.
"""

import asyncio
import os
import sys
from decimal import Decimal, ROUND_UP

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(__file__))

from src.core import get_logger
from src.core.models import OrderSide, OrderType
from src.exchange import ExchangeClient

logger = get_logger(__name__)

SYMBOL = "BTCUSDT"
LEVERAGE = 50


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


async def test_real_trade(client: ExchangeClient):
    """Test real market order with stop loss."""
    print("\n" + "=" * 60)
    print("Real Trade Test: Market Order + Stop Loss + Close")
    print("=" * 60)

    # Get current price
    ticker = await client.futures.get_ticker(SYMBOL)
    current_price = ticker.price
    print(f"\nCurrent {SYMBOL} price: {current_price}")

    # Calculate minimum quantity ($100 min notional)
    min_qty = (Decimal("100") / current_price).quantize(Decimal("0.001"), rounding=ROUND_UP)
    min_qty = max(min_qty, Decimal("0.001"))
    notional = min_qty * current_price
    margin_needed = notional / LEVERAGE
    print(f"Quantity: {min_qty} BTC (~${notional:.2f}, margin ~${margin_needed:.2f})")

    # Set leverage
    print(f"\n1. Setting leverage to {LEVERAGE}x...")
    await client.futures.set_leverage(SYMBOL, LEVERAGE)
    print(f"   ✓ Leverage set")

    # Set margin type
    print(f"\n2. Setting margin type to ISOLATED...")
    try:
        await client.futures.set_margin_type(SYMBOL, "ISOLATED")
        print(f"   ✓ Margin type set")
    except Exception as e:
        if "No need to change" in str(e):
            print(f"   ✓ Already ISOLATED")
        else:
            raise

    entry_price = None
    stop_order_id = None

    try:
        # Place market order (LONG)
        print(f"\n3. Placing MARKET BUY order...")
        market_order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=min_qty,
        )
        entry_price = market_order.avg_price or current_price
        print(f"   ✓ Market order filled: ID={market_order.order_id}, Price={entry_price}")

        # Place stop loss using STOP_LOSS_LIMIT (alternative to STOP_MARKET)
        print(f"\n4. Placing stop loss order...")
        sl_price = (entry_price * Decimal("0.99")).quantize(Decimal("0.1"))  # 1% below
        sl_limit = (sl_price * Decimal("0.995")).quantize(Decimal("0.1"))  # Slightly below stop price

        try:
            # Try STOP_MARKET first
            stop_order = await client.futures.create_order(
                symbol=SYMBOL,
                side=OrderSide.SELL,
                order_type="STOP_MARKET",
                quantity=min_qty,
                stop_price=sl_price,
                reduce_only=True,
            )
            stop_order_id = stop_order.order_id
            print(f"   ✓ Stop loss (STOP_MARKET) placed: ID={stop_order.order_id}, Stop={sl_price}")
        except Exception as e:
            if "-4120" in str(e):
                # Try STOP_LOSS (limit stop) instead
                print(f"   ⚠️ STOP_MARKET not supported, trying STOP...")
                stop_order = await client.futures.create_order(
                    symbol=SYMBOL,
                    side=OrderSide.SELL,
                    order_type="STOP",
                    quantity=min_qty,
                    price=sl_limit,
                    stop_price=sl_price,
                    reduce_only=True,
                )
                stop_order_id = stop_order.order_id
                print(f"   ✓ Stop loss (STOP) placed: ID={stop_order.order_id}, Stop={sl_price}, Limit={sl_limit}")
            else:
                raise

        # Cancel stop loss before closing position (use Algo cancel for conditional orders)
        print(f"\n5. Canceling stop loss...")
        await client.futures.cancel_algo_order(SYMBOL, algo_id=str(stop_order_id))
        print(f"   ✓ Stop loss cancelled")
        stop_order_id = None

        # Close position
        print(f"\n6. Closing position (MARKET SELL)...")
        close_order = await client.futures.create_order(
            symbol=SYMBOL,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=min_qty,
            reduce_only=True,
        )
        exit_price = close_order.avg_price or current_price
        print(f"   ✓ Position closed: ID={close_order.order_id}, Price={exit_price}")

        # Calculate PnL
        pnl = (exit_price - entry_price) * min_qty
        fee = notional * Decimal("0.0004") * 2  # Taker fee for open + close
        net_pnl = pnl - fee
        print(f"\n   Trade PnL: {pnl:.4f} USDT (raw)")
        print(f"   Fees: ~{fee:.4f} USDT")
        print(f"   Net PnL: ~{net_pnl:.4f} USDT")

        return True

    except Exception as e:
        print(f"\n   ✗ Error: {e}")

        # Cancel any pending stop order (use Algo cancel)
        if stop_order_id:
            try:
                print(f"\n   Canceling stop order...")
                await client.futures.cancel_algo_order(SYMBOL, algo_id=str(stop_order_id))
                print(f"   ✓ Stop order cancelled")
            except:
                pass

        # Try to close position
        try:
            print(f"\n   Attempting to close position...")
            positions = await client.futures.get_positions(SYMBOL)
            for pos in positions:
                qty = pos.quantity
                if qty > 0:
                    close_order = await client.futures.create_order(
                        symbol=SYMBOL,
                        side=OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=qty,
                        reduce_only=True,
                    )
                    print(f"   ✓ Position closed: {qty}")
                elif qty < 0:
                    close_order = await client.futures.create_order(
                        symbol=SYMBOL,
                        side=OrderSide.BUY,
                        order_type=OrderType.MARKET,
                        quantity=abs(qty),
                        reduce_only=True,
                    )
                    print(f"   ✓ Position closed: {qty}")
        except Exception as close_err:
            print(f"   ✗ Failed to close position: {close_err}")

        return False


async def main():
    print("\n" + "=" * 60)
    print("       Real Trade Test")
    print("=" * 60)
    print("\n⚠️  WARNING: This will place REAL orders!")
    print("    - Open LONG position (~$100 notional)")
    print("    - Set stop loss (1% below entry)")
    print("    - Close position immediately")
    print()

    client = None
    try:
        print("Connecting to exchange...")
        client = await create_exchange_client()
        print("✓ Connected")

        success = await test_real_trade(client)

        if success:
            print("\n" + "=" * 60)
            print("✓ Real trade test completed successfully!")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("✗ Real trade test failed!")
            print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if client:
            await client.disconnect()
            print("\nDisconnected.")


if __name__ == "__main__":
    asyncio.run(main())
