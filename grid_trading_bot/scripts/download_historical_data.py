#!/usr/bin/env python3
"""
Download Historical Kline Data.

Downloads historical kline data from Binance and saves to local file.
Handles API rate limits and pagination for large datasets.

Usage:
    python scripts/download_historical_data.py --symbol BTCUSDT --interval 15m --days 730
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.models import Kline
from src.exchange import ExchangeClient


# Output directory
DATA_DIR = Path("data/historical")


async def download_klines(
    symbol: str,
    interval: str,
    days: int,
    output_file: Path,
) -> list[Kline]:
    """
    Download historical klines in batches.

    Args:
        symbol: Trading pair (e.g., BTCUSDT)
        interval: Kline interval (e.g., 15m, 1h)
        days: Number of days to download
        output_file: Output file path

    Returns:
        List of Kline objects
    """
    print(f"\n{'='*60}")
    print(f"  下載歷史數據")
    print(f"{'='*60}")
    print(f"\n配置:")
    print(f"  交易對: {symbol}")
    print(f"  週期: {interval}")
    print(f"  天數: {days}")
    print(f"  輸出: {output_file}")

    # Calculate bars needed
    interval_hours = {
        "1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
        "1h": 1, "2h": 2, "4h": 4, "1d": 24
    }
    hours_per_bar = interval_hours.get(interval, 1)
    total_bars = int(days * 24 / hours_per_bar)

    print(f"  預計 K 線數: {total_bars:,}")

    # Connect to exchange
    client = ExchangeClient(
        api_key=os.getenv("BINANCE_API_KEY", ""),
        api_secret=os.getenv("BINANCE_API_SECRET", ""),
        testnet=False,
    )

    all_klines = []

    try:
        await client.connect()

        # Calculate time range
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=days)

        print(f"\n時間範圍:")
        print(f"  開始: {start_time.strftime('%Y-%m-%d %H:%M')}")
        print(f"  結束: {end_time.strftime('%Y-%m-%d %H:%M')}")

        # Download in batches of 1000
        batch_size = 1000
        current_end = end_time
        batch_num = 0

        print(f"\n開始下載...")

        while current_end > start_time:
            batch_num += 1

            # Fetch batch
            klines = await client.futures.get_klines(
                symbol=symbol,
                interval=interval,
                limit=batch_size,
                end_time=int(current_end.timestamp() * 1000),
            )

            if not klines:
                break

            # Filter klines within time range
            filtered = [k for k in klines if k.open_time >= start_time]
            all_klines.extend(filtered)

            # Update progress
            oldest = klines[0].open_time if klines else current_end
            print(f"  批次 {batch_num}: 獲取 {len(klines)} 根 (至 {oldest.strftime('%Y-%m-%d')}), 總計: {len(all_klines):,}")

            # Move to earlier time
            if klines:
                current_end = klines[0].open_time - timedelta(seconds=1)
            else:
                break

            # Rate limiting
            await asyncio.sleep(0.2)

            # Check if we've gone past start time
            if oldest < start_time:
                break

        # Sort by time
        all_klines.sort(key=lambda k: k.open_time)

        # Remove duplicates
        seen = set()
        unique_klines = []
        for k in all_klines:
            key = k.open_time.timestamp()
            if key not in seen:
                seen.add(key)
                unique_klines.append(k)

        all_klines = unique_klines

        print(f"\n下載完成!")
        print(f"  總 K 線數: {len(all_klines):,}")
        if all_klines:
            print(f"  實際範圍: {all_klines[0].open_time.strftime('%Y-%m-%d')} ~ {all_klines[-1].open_time.strftime('%Y-%m-%d')}")

        # Save to file
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "symbol": symbol,
                "interval": interval,
                "start_date": all_klines[0].open_time.isoformat() if all_klines else None,
                "end_date": all_klines[-1].open_time.isoformat() if all_klines else None,
                "total_bars": len(all_klines),
                "downloaded_at": datetime.now(timezone.utc).isoformat(),
            },
            "klines": [
                {
                    "open_time": k.open_time.isoformat(),
                    "close_time": k.close_time.isoformat(),
                    "open": str(k.open),
                    "high": str(k.high),
                    "low": str(k.low),
                    "close": str(k.close),
                    "volume": str(k.volume),
                }
                for k in all_klines
            ]
        }

        with open(output_file, "w") as f:
            json.dump(data, f)

        print(f"\n數據已儲存至: {output_file}")
        print(f"  檔案大小: {output_file.stat().st_size / 1024 / 1024:.2f} MB")

        return all_klines

    finally:
        await client.close()


def load_klines_from_file(filepath: Path) -> list[Kline]:
    """Load klines from saved JSON file."""
    from decimal import Decimal

    with open(filepath, "r") as f:
        data = json.load(f)

    klines = []
    for k in data["klines"]:
        klines.append(Kline(
            symbol=data["metadata"]["symbol"],
            interval=data["metadata"]["interval"],
            open_time=datetime.fromisoformat(k["open_time"]),
            close_time=datetime.fromisoformat(k["close_time"]),
            open=Decimal(k["open"]),
            high=Decimal(k["high"]),
            low=Decimal(k["low"]),
            close=Decimal(k["close"]),
            volume=Decimal(k["volume"]),
        ))

    return klines


async def main():
    parser = argparse.ArgumentParser(description="下載歷史 K 線數據")
    parser.add_argument(
        "--symbol", "-s",
        default="BTCUSDT",
        help="交易對 (default: BTCUSDT)"
    )
    parser.add_argument(
        "--interval", "-i",
        default="15m",
        help="K 線週期 (default: 15m)"
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=730,
        help="天數 (default: 730 = 2 years)"
    )
    parser.add_argument(
        "--output", "-o",
        help="輸出檔案路徑 (default: data/historical/{symbol}_{interval}_{days}d.json)"
    )

    args = parser.parse_args()

    # Default output path
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = DATA_DIR / f"{args.symbol}_{args.interval}_{args.days}d.json"

    await download_klines(
        symbol=args.symbol,
        interval=args.interval,
        days=args.days,
        output_file=output_file,
    )


if __name__ == "__main__":
    asyncio.run(main())
