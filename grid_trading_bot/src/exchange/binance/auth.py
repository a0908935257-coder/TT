"""
Binance API authentication and signature handling.
"""

import hashlib
import hmac
from urllib.parse import urlencode

from core.utils import now_timestamp


class BinanceAuth:
    """
    Handles Binance API authentication and request signing.

    Uses HMAC-SHA256 algorithm to sign requests as required by Binance API.

    Example:
        >>> auth = BinanceAuth("api_key", "api_secret")
        >>> params = {"symbol": "BTCUSDT", "side": "BUY"}
        >>> signed_params = auth.sign_params(params)
        >>> headers = auth.get_headers()
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        Initialize BinanceAuth.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
        """
        self.api_key = api_key
        self.api_secret = api_secret

    def sign_params(self, params: dict | None = None) -> dict:
        """
        Sign parameters with HMAC-SHA256.

        Steps:
        1. Add timestamp to params
        2. Sort params by key and create query string
        3. Calculate HMAC-SHA256 signature
        4. Add signature to params

        Args:
            params: Request parameters to sign

        Returns:
            Parameters with timestamp and signature added

        Example:
            >>> auth = BinanceAuth("key", "secret")
            >>> params = auth.sign_params({"symbol": "BTCUSDT"})
            >>> "timestamp" in params and "signature" in params
            True
        """
        if params is None:
            params = {}
        else:
            # Create a copy to avoid modifying original
            params = params.copy()

        # Add timestamp
        params["timestamp"] = now_timestamp(unit="ms")

        # Sort by key and create query string
        sorted_params = sorted(params.items())
        query_string = urlencode(sorted_params)

        # Calculate HMAC-SHA256 signature
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # Add signature to params
        params["signature"] = signature

        return params

    def get_headers(self) -> dict:
        """
        Get request headers with API key.

        Returns:
            Headers dict with X-MBX-APIKEY
        """
        return {"X-MBX-APIKEY": self.api_key}

    def sign_query_string(self, query_string: str) -> str:
        """
        Sign an existing query string.

        Args:
            query_string: URL encoded query string

        Returns:
            HMAC-SHA256 signature as hex string
        """
        return hmac.new(
            self.api_secret.encode("utf-8"),
            query_string.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
