"""
Discord Bot Entry Point.

Run with: python -m src.discord_bot
"""

import asyncio
import os
import sys

from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.core import get_logger
from src.discord_bot import TradingBot, load_discord_config

logger = get_logger(__name__)


async def main():
    """Main entry point."""
    # Load environment variables
    load_dotenv()

    # Load Discord config
    config = load_discord_config()

    # Validate config
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Config error: {error}")
        print("Discord bot configuration is incomplete.")
        print("Please set the following environment variables:")
        print("  - DISCORD_BOT_TOKEN")
        print("  - DISCORD_GUILD_ID")
        return

    # Optional: Initialize Master and RiskEngine
    # For standalone Discord bot, these can be None
    master = None
    risk_engine = None

    # Try to initialize Master if exchange is available
    try:
        from src.exchange import ExchangeClient
        from src.master import Master, MasterConfig

        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")

        if api_key and api_secret:
            logger.info("Initializing exchange client...")
            exchange = ExchangeClient(
                api_key=api_key,
                api_secret=api_secret,
                testnet=os.getenv("BINANCE_TESTNET", "true").lower() == "true",
            )
            await exchange.connect()

            logger.info("Initializing Master...")
            master_config = MasterConfig(
                auto_restart=False,
                max_bots=10,
            )
            master = Master(
                exchange=exchange,
                config=master_config,
            )
            await master.start()
            logger.info("Master initialized")
    except Exception as e:
        logger.warning(f"Could not initialize Master: {e}")
        logger.info("Running Discord bot without Master integration")

    # Create and start Discord bot
    bot = TradingBot(config, master=master, risk_engine=risk_engine)

    print("=" * 60)
    print("       Discord Trading Bot")
    print("=" * 60)
    print(f"  Guild ID: {config.guild_id}")
    print(f"  Master: {'Connected' if master else 'Not connected'}")
    print("=" * 60)

    try:
        await bot.start_bot()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        await bot.stop_bot()
        if master:
            await master.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
