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

# Futures (USDT-M)
FUTURES_REST_URL = "https://fapi.binance.com"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"


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
# Futures Public Endpoints
# =============================================================================


class FuturesPublicEndpoints(Enum):
    """Futures public API endpoints."""

    PING = Endpoint("/fapi/v1/ping", "GET")
    SERVER_TIME = Endpoint("/fapi/v1/time", "GET")
    EXCHANGE_INFO = Endpoint("/fapi/v1/exchangeInfo", "GET")
    KLINES = Endpoint("/fapi/v1/klines", "GET")
    TICKER_24H = Endpoint("/fapi/v1/ticker/24hr", "GET")
    MARK_PRICE = Endpoint("/fapi/v1/premiumIndex", "GET")
    FUNDING_RATE = Endpoint("/fapi/v1/fundingRate", "GET")
    DEPTH = Endpoint("/fapi/v1/depth", "GET")


# =============================================================================
# Futures Private Endpoints
# =============================================================================


class FuturesPrivateEndpoints(Enum):
    """Futures private API endpoints (require signature)."""

    ACCOUNT = Endpoint("/fapi/v2/account", "GET")
    BALANCE = Endpoint("/fapi/v2/balance", "GET")
    POSITION = Endpoint("/fapi/v2/positionRisk", "GET")
    LEVERAGE = Endpoint("/fapi/v1/leverage", "POST")
    MARGIN_TYPE = Endpoint("/fapi/v1/marginType", "POST")
    POSITION_MODE_GET = Endpoint("/fapi/v1/positionSide/dual", "GET")
    POSITION_MODE_SET = Endpoint("/fapi/v1/positionSide/dual", "POST")
    ORDER = Endpoint("/fapi/v1/order")  # GET/POST/DELETE
    OPEN_ORDERS = Endpoint("/fapi/v1/openOrders", "GET")
    ALL_ORDERS = Endpoint("/fapi/v1/allOrders", "GET")
    USER_TRADES = Endpoint("/fapi/v1/userTrades", "GET")
    LEVERAGE_BRACKET = Endpoint("/fapi/v1/leverageBracket", "GET")
    CANCEL_ALL_ORDERS = Endpoint("/fapi/v1/allOpenOrders", "DELETE")


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

# Futures endpoints dictionary format
FUTURES_PUBLIC_ENDPOINTS = {
    "PING": {"path": "/fapi/v1/ping", "method": "GET"},
    "SERVER_TIME": {"path": "/fapi/v1/time", "method": "GET"},
    "EXCHANGE_INFO": {"path": "/fapi/v1/exchangeInfo", "method": "GET"},
    "KLINES": {"path": "/fapi/v1/klines", "method": "GET"},
    "TICKER_24H": {"path": "/fapi/v1/ticker/24hr", "method": "GET"},
    "MARK_PRICE": {"path": "/fapi/v1/premiumIndex", "method": "GET"},
    "FUNDING_RATE": {"path": "/fapi/v1/fundingRate", "method": "GET"},
    "DEPTH": {"path": "/fapi/v1/depth", "method": "GET"},
}

FUTURES_PRIVATE_ENDPOINTS = {
    "ACCOUNT": {"path": "/fapi/v2/account", "method": "GET"},
    "BALANCE": {"path": "/fapi/v2/balance", "method": "GET"},
    "POSITION": {"path": "/fapi/v2/positionRisk", "method": "GET"},
    "LEVERAGE": {"path": "/fapi/v1/leverage", "method": "POST"},
    "MARGIN_TYPE": {"path": "/fapi/v1/marginType", "method": "POST"},
    "POSITION_MODE_GET": {"path": "/fapi/v1/positionSide/dual", "method": "GET"},
    "POSITION_MODE_SET": {"path": "/fapi/v1/positionSide/dual", "method": "POST"},
    "ORDER": {"path": "/fapi/v1/order", "method": "POST"},
    "ORDER_GET": {"path": "/fapi/v1/order", "method": "GET"},
    "ORDER_DELETE": {"path": "/fapi/v1/order", "method": "DELETE"},
    "OPEN_ORDERS": {"path": "/fapi/v1/openOrders", "method": "GET"},
    "ALL_ORDERS": {"path": "/fapi/v1/allOrders", "method": "GET"},
    "USER_TRADES": {"path": "/fapi/v1/userTrades", "method": "GET"},
    "LEVERAGE_BRACKET": {"path": "/fapi/v1/leverageBracket", "method": "GET"},
    "CANCEL_ALL_ORDERS": {"path": "/fapi/v1/allOpenOrders", "method": "DELETE"},
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
