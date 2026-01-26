"""
Binance Spot REST API client.

Provides async interface to Binance Spot trading API with support for
both public and private (authenticated) endpoints.
"""

import asyncio
from datetime import datetime
from decimal import Decimal
from typing import Optional

import aiohttp

from src.core import get_logger
from src.core.exceptions import (
    AuthenticationError,
    ConnectionError,
    ExchangeError,
    InsufficientBalanceError,
    OrderError,
    RateLimitError,
)
from src.core.models import (
    AccountInfo,
    Balance,
    Kline,
    KlineInterval,
    Order,
    OrderSide,
    OrderType,
    SymbolInfo,
    Ticker,
    Trade,
)
from src.core.utils import timestamp_to_datetime

from .auth import BinanceAuth
from .constants import (
    BINANCE_ERROR_CODES,
    PRIVATE_ENDPOINTS,
    PUBLIC_ENDPOINTS,
    SPOT_REST_URL,
    SPOT_TESTNET_URL,
)

logger = get_logger(__name__)


class BinanceSpotAPI:
    """
    Binance Spot REST API client.

    Supports both public endpoints (no auth) and private endpoints (requires API key).

    Example:
        >>> async with BinanceSpotAPI() as api:
        ...     await api.ping()
        ...     klines = await api.get_klines("BTCUSDT", KlineInterval.h1)

        >>> # With authentication
        >>> async with BinanceSpotAPI(api_key="...", api_secret="...") as api:
        ...     account = await api.get_account()
        ...     order = await api.market_buy("BTCUSDT", Decimal("0.001"))
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
        timeout: int = 30,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """
        Initialize BinanceSpotAPI.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Use testnet URL if True
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self._base_url = SPOT_TESTNET_URL if testnet else SPOT_REST_URL
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._auth: Optional[BinanceAuth] = None

        if api_key and api_secret:
            self._auth = BinanceAuth(api_key, api_secret)

        self._testnet = testnet

        # Retry configuration
        self._max_retries = max_retries
        self._retry_delay = retry_delay

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def connect(self) -> None:
        """Create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            logger.debug(f"Connected to {self._base_url}")

    async def sync_time(self) -> None:
        """Synchronize local time with Binance server time."""
        try:
            data = await self._request("GET", "/api/v3/time")
            server_time_ms = data["serverTime"]
            if self._auth:
                self._auth.set_time_offset(server_time_ms)
                logger.info(f"Time synced, offset: {self._auth.time_offset}ms")
        except Exception as e:
            logger.warning(f"Failed to sync time: {e}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Session closed")

    async def __aenter__(self) -> "BinanceSpotAPI":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Internal Request Methods
    # =========================================================================

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: dict | None = None,
        signed: bool = False,
        api_key_required: bool = False,
    ) -> dict:
        """
        Send HTTP request to Binance API with retry logic.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether to sign the request
            api_key_required: Whether to include API key header (without signature)

        Returns:
            JSON response as dict

        Raises:
            ConnectionError: Connection failed after retries
            AuthenticationError: Authentication failed
            RateLimitError: Rate limit exceeded after retries
            InsufficientBalanceError: Insufficient balance
            OrderError: Order-related error
            ExchangeError: Other exchange errors
        """
        if self._session is None or self._session.closed:
            await self.connect()

        url = f"{self._base_url}{endpoint}"
        original_params = params.copy() if params else {}

        last_exception = None
        for attempt in range(self._max_retries + 1):
            headers = {}
            params = original_params.copy()

            # Sign request if needed (must re-sign on each attempt due to timestamp)
            if signed:
                if self._auth is None:
                    raise AuthenticationError("API key and secret required for signed requests")
                params = self._auth.sign_params(params)
                headers = self._auth.get_headers()
            elif api_key_required:
                # Some endpoints require API key but not signature
                if self._auth is None:
                    raise AuthenticationError("API key required for this request")
                headers = self._auth.get_headers()

            logger.debug(f"Request: {method} {endpoint} (attempt {attempt + 1}/{self._max_retries + 1})")

            try:
                if method == "GET":
                    async with self._session.get(url, params=params, headers=headers) as resp:
                        return await self._handle_response(resp)
                elif method == "POST":
                    async with self._session.post(url, params=params, headers=headers) as resp:
                        return await self._handle_response(resp)
                elif method == "PUT":
                    async with self._session.put(url, params=params, headers=headers) as resp:
                        return await self._handle_response(resp)
                elif method == "DELETE":
                    async with self._session.delete(url, params=params, headers=headers) as resp:
                        return await self._handle_response(resp)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

            except (RateLimitError, ConnectionError) as e:
                # Retry on rate limit and connection errors
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    # For rate limits, use retry-after header if available
                    if isinstance(e, RateLimitError):
                        delay = max(delay, 1.0)  # Minimum 1 second for rate limits
                    logger.warning(
                        f"Retryable error on {endpoint}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            except aiohttp.ClientError as e:
                # Retry on transient network errors
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Network error on {endpoint}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Connection error after {self._max_retries + 1} attempts: {e}")
                raise ConnectionError(f"Failed to connect to Binance: {e}")

        # Should not reach here, but handle it just in case
        if last_exception:
            raise last_exception
        raise ExchangeError(f"Request failed after {self._max_retries + 1} attempts")

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict:
        """
        Handle API response and raise appropriate exceptions.

        Args:
            response: aiohttp response object

        Returns:
            Parsed JSON response

        Raises:
            Various exceptions based on response status and error codes
        """
        status = response.status
        logger.debug(f"Response status: {status}")

        # Handle HTTP 429 Rate Limit (before parsing JSON)
        if status == 429:
            retry_after = response.headers.get("Retry-After", "1")
            raise RateLimitError(
                f"Rate limited (HTTP 429). Retry after: {retry_after}s",
                code="429"
            )

        # Try to parse JSON
        try:
            data = await response.json()
        except Exception:
            text = await response.text()
            if status >= 400:
                raise ExchangeError(f"HTTP {status}: {text}")
            return {}

        # Check for Binance error
        if "code" in data and "msg" in data:
            code = data["code"]
            msg = data["msg"]
            self._raise_exception(code, msg)

        # Check HTTP status
        if status >= 400:
            raise ExchangeError(f"HTTP {status}: {data}")

        return data

    def _raise_exception(self, code: int, msg: str) -> None:
        """
        Raise appropriate exception based on Binance error code.

        Args:
            code: Binance error code
            msg: Error message
        """
        error_info = f"[{code}] {msg}"

        if code == -1002 or code in (-2014, -2015):
            raise AuthenticationError(error_info, code=str(code))
        elif code == -1003:
            raise RateLimitError(error_info, code=str(code))
        elif code == -2010:
            raise InsufficientBalanceError(error_info, code=str(code))
        elif code in (-2011, -2013):
            raise OrderError(error_info, code=str(code))
        elif code == -1021 or code == -1022:
            raise ExchangeError(f"Signature/timestamp error: {error_info}", code=str(code))
        else:
            raise ExchangeError(error_info, code=str(code))

    # =========================================================================
    # Public API - System
    # =========================================================================

    async def ping(self) -> bool:
        """
        Test connectivity to Binance API.

        Returns:
            True if successful
        """
        await self._request("GET", PUBLIC_ENDPOINTS["PING"]["path"])
        return True

    async def get_server_time(self) -> datetime:
        """
        Get Binance server time.

        Returns:
            Server time as datetime (UTC)
        """
        data = await self._request("GET", PUBLIC_ENDPOINTS["SERVER_TIME"]["path"])
        return timestamp_to_datetime(data["serverTime"])

    async def get_exchange_info(self, symbol: str | None = None) -> dict | SymbolInfo:
        """
        Get exchange trading rules and symbol information.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            SymbolInfo if symbol specified, else full exchange info dict
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", PUBLIC_ENDPOINTS["EXCHANGE_INFO"]["path"], params)

        if symbol and data.get("symbols"):
            return SymbolInfo.from_binance(data["symbols"][0])

        return data

    # =========================================================================
    # Public API - Market Data
    # =========================================================================

    async def get_klines(
        self,
        symbol: str,
        interval: KlineInterval | str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Kline]:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            interval: Kline interval (e.g., KlineInterval.h1 or "1h")
            limit: Number of klines (max 1000, default 500)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of Kline objects
        """
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval

        params = {
            "symbol": symbol,
            "interval": interval_str,
            "limit": limit,
        }
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", PUBLIC_ENDPOINTS["KLINES"]["path"], params)

        # Convert interval string to enum for Kline constructor
        interval_enum = KlineInterval(interval_str) if isinstance(interval, str) else interval

        return [Kline.from_binance(k, symbol, interval_enum) for k in data]

    async def get_ticker(self, symbol: str | None = None) -> Ticker | list[Ticker]:
        """
        Get 24hr ticker price change statistics.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            Ticker if symbol specified, else list of Tickers
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", PUBLIC_ENDPOINTS["TICKER_24H"]["path"], params)

        if symbol:
            return Ticker.from_binance(data)

        return [Ticker.from_binance(t) for t in data]

    async def get_price(self, symbol: str | None = None) -> Decimal | dict[str, Decimal]:
        """
        Get latest price for symbol(s).

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            Decimal price if symbol specified, else dict of symbol->price
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", PUBLIC_ENDPOINTS["TICKER_PRICE"]["path"], params)

        if symbol:
            return Decimal(str(data["price"]))

        return {item["symbol"]: Decimal(str(item["price"])) for item in data}

    async def get_orderbook(self, symbol: str, limit: int = 100) -> dict:
        """
        Get order book (market depth).

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Dict with 'bids' and 'asks' as list of [price, quantity]
        """
        params = {"symbol": symbol, "limit": limit}
        data = await self._request("GET", PUBLIC_ENDPOINTS["DEPTH"]["path"], params)

        return {
            "bids": [[Decimal(p), Decimal(q)] for p, q in data["bids"]],
            "asks": [[Decimal(p), Decimal(q)] for p, q in data["asks"]],
            "lastUpdateId": data["lastUpdateId"],
        }

    # =========================================================================
    # Private API - Account
    # =========================================================================

    async def get_account(self) -> AccountInfo:
        """
        Get current account information.

        Returns:
            AccountInfo with balances

        Raises:
            AuthenticationError: If not authenticated
        """
        data = await self._request(
            "GET",
            PRIVATE_ENDPOINTS["ACCOUNT"]["path"],
            signed=True,
        )
        return AccountInfo.from_binance_spot(data)

    async def get_balance(self, asset: str) -> Balance | None:
        """
        Get balance for specific asset.

        Args:
            asset: Asset name (e.g., "USDT", "BTC")

        Returns:
            Balance object or None if not found
        """
        account = await self.get_account()
        return account.get_balance(asset)

    # =========================================================================
    # Private API - Orders
    # =========================================================================

    async def create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal | str,
        price: Decimal | str | None = None,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
        stop_price: Decimal | str | None = None,
        self_trade_prevention: str | None = "EXPIRE_TAKER",
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/etc.)
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Custom order ID
            stop_price: Stop price for stop orders
            self_trade_prevention: STP mode (EXPIRE_TAKER, EXPIRE_MAKER, EXPIRE_BOTH, NONE)
                - EXPIRE_TAKER: Cancel the new order (default, protects existing orders)
                - EXPIRE_MAKER: Cancel the existing order
                - EXPIRE_BOTH: Cancel both orders
                - NONE: Allow self-trade

        Returns:
            Created Order object
        """
        side_str = side.value if isinstance(side, OrderSide) else side
        type_str = order_type.value if isinstance(order_type, OrderType) else order_type

        # Format decimal values without scientific notation
        def format_decimal(value: Decimal | str) -> str:
            d = Decimal(str(value))
            # Use fixed-point notation, remove trailing zeros
            return f"{d:f}".rstrip('0').rstrip('.')

        params = {
            "symbol": symbol,
            "side": side_str,
            "type": type_str,
            "quantity": format_decimal(quantity),
        }

        # Add price for limit orders
        if price is not None:
            params["price"] = format_decimal(price)

        # Add timeInForce for limit orders
        if type_str in ("LIMIT", "STOP_LIMIT"):
            params["timeInForce"] = time_in_force

        # Add stop price
        if stop_price is not None:
            params["stopPrice"] = format_decimal(stop_price)

        # Add client order ID
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Add self-trade prevention mode
        if self_trade_prevention:
            params["selfTradePreventionMode"] = self_trade_prevention

        # Request full order response
        params["newOrderRespType"] = "FULL"

        data = await self._request(
            "POST",
            PRIVATE_ENDPOINTS["ORDER"]["path"],
            params,
            signed=True,
        )
        return Order.from_binance(data)

    async def cancel_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Cancel an active order.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            client_order_id: Client order ID (alternative to order_id)

        Returns:
            Cancelled Order object
        """
        params = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        data = await self._request(
            "DELETE",
            PRIVATE_ENDPOINTS["ORDER_DELETE"]["path"],
            params,
            signed=True,
        )
        return Order.from_binance(data)

    async def get_order(
        self,
        symbol: str,
        order_id: str | None = None,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Get order status.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            client_order_id: Client order ID

        Returns:
            Order object
        """
        params = {"symbol": symbol}

        if order_id:
            params["orderId"] = order_id
        elif client_order_id:
            params["origClientOrderId"] = client_order_id
        else:
            raise ValueError("Either order_id or client_order_id required")

        data = await self._request(
            "GET",
            PRIVATE_ENDPOINTS["ORDER_GET"]["path"],
            params,
            signed=True,
        )
        return Order.from_binance(data)

    async def get_open_orders(self, symbol: str | None = None) -> list[Order]:
        """
        Get all open orders.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            List of open Order objects
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request(
            "GET",
            PRIVATE_ENDPOINTS["OPEN_ORDERS"]["path"],
            params,
            signed=True,
        )
        return [Order.from_binance(o) for o in data]

    async def get_all_orders(
        self,
        symbol: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Order]:
        """
        Get all orders (active, cancelled, filled).

        Args:
            symbol: Trading pair
            limit: Number of orders (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of Order objects
        """
        params = {"symbol": symbol, "limit": limit}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request(
            "GET",
            PRIVATE_ENDPOINTS["ALL_ORDERS"]["path"],
            params,
            signed=True,
        )
        return [Order.from_binance(o) for o in data]

    async def get_my_trades(
        self,
        symbol: str,
        limit: int = 500,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Trade]:
        """
        Get account trade history.

        Args:
            symbol: Trading pair
            limit: Number of trades (max 1000)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of Trade objects
        """
        params = {"symbol": symbol, "limit": limit}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request(
            "GET",
            PRIVATE_ENDPOINTS["MY_TRADES"]["path"],
            params,
            signed=True,
        )
        return [Trade.from_binance(t) for t in data]

    # =========================================================================
    # Convenience Order Methods
    # =========================================================================

    async def market_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a market buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id,
        )

    async def market_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a market sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            client_order_id=client_order_id,
        )

    async def limit_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a limit buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            price: Limit price
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    async def limit_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a limit sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            price: Limit price
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            price=price,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    # =========================================================================
    # User Data Stream
    # =========================================================================

    async def create_listen_key(self) -> str:
        """
        Create a new listen key for User Data Stream.

        Returns:
            Listen key string
        """
        data = await self._request(
            "POST",
            PRIVATE_ENDPOINTS["USER_DATA_STREAM"]["path"],
            api_key_required=True,
        )
        return data["listenKey"]

    async def keep_alive_listen_key(self, listen_key: str) -> bool:
        """
        Keep alive a listen key (should be called every 30 minutes).

        Args:
            listen_key: The listen key to keep alive

        Returns:
            True if successful
        """
        await self._request(
            "PUT",
            PRIVATE_ENDPOINTS["USER_DATA_STREAM_KEEPALIVE"]["path"],
            params={"listenKey": listen_key},
            api_key_required=True,
        )
        return True

    async def delete_listen_key(self, listen_key: str) -> bool:
        """
        Delete a listen key.

        Args:
            listen_key: The listen key to delete

        Returns:
            True if successful
        """
        await self._request(
            "DELETE",
            PRIVATE_ENDPOINTS["USER_DATA_STREAM_DELETE"]["path"],
            params={"listenKey": listen_key},
            api_key_required=True,
        )
        return True
