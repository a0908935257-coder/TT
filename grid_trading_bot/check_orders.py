#!/usr/bin/env python3
"""檢查交易所掛單"""
import asyncio
import os
import sys

sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

from src.exchange import ExchangeClient

async def check():
    client = ExchangeClient(
        os.getenv('BINANCE_API_KEY'),
        os.getenv('BINANCE_API_SECRET'),
        testnet=True
    )
    await client.connect()

    orders = await client.get_open_orders('BTCUSDT')
    print(f'\n交易所 BTCUSDT 掛單數: {len(orders)}')

    if orders:
        print('-' * 50)
        for o in orders:
            side = o.side.value if hasattr(o.side, 'value') else o.side
            print(f'  {side:4} {o.quantity} @ {o.price}')
    else:
        print('  (無掛單)')

    await client.disconnect()

if __name__ == '__main__':
    asyncio.run(check())
