"""
Binance API constants and endpoint definitions.
"""

from dataclasses import dataclass
from enum import Enum


# =============================================================================
# Base URLs
# =============================================================================

SPOT_REST_URL = "https://api.binance.com"
SPOT_TESTNET_URL = "https://testnet.binance.vision"


# =============================================================================
# Endpoint Definition
# =============================================================================


@dataclass(frozen=True)
class Endpoint:
    """API endpoint definition."""

    path: str
    method: str = "GET"


# =============================================================================
# Public Endpoints (No signature required)
# =============================================================================


class PublicEndpoints(Enum):
    """Public API endpoints."""

    PING = Endpoint("/api/v3/ping", "GET")
    SERVER_TIME = Endpoint("/api/v3/time", "GET")
    EXCHANGE_INFO = Endpoint("/api/v3/exchangeInfo", "GET")
    KLINES = Endpoint("/api/v3/klines", "GET")
    TICKER_24H = Endpoint("/api/v3/ticker/24hr", "GET")
    TICKER_PRICE = Endpoint("/api/v3/ticker/price", "GET")
    DEPTH = Endpoint("/api/v3/depth", "GET")
    TRADES = Endpoint("/api/v3/trades", "GET")


# =============================================================================
# Private Endpoints (Signature required)
# =============================================================================


class PrivateEndpoints(Enum):
    """Private API endpoints (require signature)."""

    ACCOUNT = Endpoint("/api/v3/account", "GET")
    ORDER = Endpoint("/api/v3/order")  # GET/POST/DELETE
    OPEN_ORDERS = Endpoint("/api/v3/openOrders", "GET")
    ALL_ORDERS = Endpoint("/api/v3/allOrders", "GET")
    MY_TRADES = Endpoint("/api/v3/myTrades", "GET")


# =============================================================================
# Dictionary format for convenience
# =============================================================================

PUBLIC_ENDPOINTS = {
    "PING": {"path": "/api/v3/ping", "method": "GET"},
    "SERVER_TIME": {"path": "/api/v3/time", "method": "GET"},
    "EXCHANGE_INFO": {"path": "/api/v3/exchangeInfo", "method": "GET"},
    "KLINES": {"path": "/api/v3/klines", "method": "GET"},
    "TICKER_24H": {"path": "/api/v3/ticker/24hr", "method": "GET"},
    "TICKER_PRICE": {"path": "/api/v3/ticker/price", "method": "GET"},
    "DEPTH": {"path": "/api/v3/depth", "method": "GET"},
    "TRADES": {"path": "/api/v3/trades", "method": "GET"},
}

PRIVATE_ENDPOINTS = {
    "ACCOUNT": {"path": "/api/v3/account", "method": "GET"},
    "ORDER": {"path": "/api/v3/order", "method": "POST"},  # Default to POST
    "ORDER_GET": {"path": "/api/v3/order", "method": "GET"},
    "ORDER_DELETE": {"path": "/api/v3/order", "method": "DELETE"},
    "OPEN_ORDERS": {"path": "/api/v3/openOrders", "method": "GET"},
    "ALL_ORDERS": {"path": "/api/v3/allOrders", "method": "GET"},
    "MY_TRADES": {"path": "/api/v3/myTrades", "method": "GET"},
}


# =============================================================================
# Binance Error Codes
# =============================================================================

BINANCE_ERROR_CODES = {
    -1000: "UNKNOWN",           # Unknown error
    -1002: "UNAUTHORIZED",      # Not authorized
    -1003: "TOO_MANY_REQUESTS", # Rate limit
    -1021: "INVALID_TIMESTAMP", # Timestamp outside of recvWindow
    -1022: "INVALID_SIGNATURE", # Signature verification failed
    -2010: "INSUFFICIENT_BALANCE",  # Insufficient balance
    -2011: "ORDER_NOT_FOUND",   # Order does not exist
    -2013: "NO_SUCH_ORDER",     # Order does not exist
    -2014: "BAD_API_KEY_FMT",   # API key format invalid
    -2015: "REJECTED_MBX_KEY",  # Invalid API key or IP
}
