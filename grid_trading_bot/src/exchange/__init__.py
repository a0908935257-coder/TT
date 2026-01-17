# Exchange module - API clients for various exchanges
from .client import ExchangeClient
from .binance.spot_api import BinanceSpotAPI
from .binance.futures_api import BinanceFuturesAPI
from .binance.websocket import BinanceWebSocket
from .binance.auth import BinanceAuth

__all__ = [
    # Unified client
    "ExchangeClient",
    # Individual clients
    "BinanceSpotAPI",
    "BinanceFuturesAPI",
    "BinanceWebSocket",
    "BinanceAuth",
]
