"""
Binance USDT-M Futures REST API client.

Provides async interface to Binance Futures trading API with support for
both public and private (authenticated) endpoints.
"""

import asyncio
from datetime import datetime, timezone
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
    Position,
    PositionSide,
    SymbolInfo,
    Ticker,
    Trade,
)
from src.core.utils import timestamp_to_datetime

from .auth import BinanceAuth
from .constants import (
    FUTURES_PRIVATE_ENDPOINTS,
    FUTURES_PUBLIC_ENDPOINTS,
    FUTURES_REST_URL,
    FUTURES_TESTNET_URL,
)
from src.exchange.rate_limiter import RateLimiter, RateLimitConfig

logger = get_logger(__name__)


class BinanceFuturesAPI:
    """
    Binance USDT-M Futures REST API client.

    Supports both public endpoints (no auth) and private endpoints (requires API key).
    Handles futures-specific features like leverage, margin type, and position modes.

    Example:
        >>> async with BinanceFuturesAPI() as api:
        ...     await api.ping()
        ...     klines = await api.get_klines("BTCUSDT", KlineInterval.h1)

        >>> # With authentication
        >>> async with BinanceFuturesAPI(api_key="...", api_secret="...") as api:
        ...     await api.set_leverage("BTCUSDT", 10)
        ...     positions = await api.get_positions()
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
        Initialize BinanceFuturesAPI.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Use testnet URL if True
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for transient errors
            retry_delay: Initial delay between retries (exponential backoff)
        """
        self._base_url = FUTURES_TESTNET_URL if testnet else FUTURES_REST_URL
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._auth: Optional[BinanceAuth] = None

        if api_key and api_secret:
            self._auth = BinanceAuth(api_key, api_secret)

        self._testnet = testnet

        # Retry configuration
        self._max_retries = max_retries
        self._retry_delay = retry_delay

        # FIX S-3: Rate limiter integration
        self._rate_limiter = RateLimiter(
            config=RateLimitConfig(max_weight_per_minute=2400)
        )

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def connect(self) -> None:
        """Create HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
            logger.debug(f"Connected to {self._base_url}")

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.debug("Session closed")

    async def sync_time(self) -> None:
        """Sync local time with Binance Futures server time."""
        if self._auth:
            import time as time_module
            start_time = time_module.time()
            data = await self._request("GET", "/fapi/v1/time")
            end_time = time_module.time()
            latency_ms = int((end_time - start_time) * 1000)

            server_time_ms = data.get("serverTime", 0)
            if server_time_ms:
                self._auth.set_time_offset(server_time_ms, latency_ms)
                logger.info(
                    f"Futures time synced, offset: {self._auth.time_offset}ms "
                    f"(latency: {latency_ms}ms)"
                )

    async def __aenter__(self) -> "BinanceFuturesAPI":
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
    ) -> dict | list:
        """
        Send HTTP request to Binance Futures API with retry logic.

        Args:
            method: HTTP method (GET, POST, DELETE)
            endpoint: API endpoint path
            params: Request parameters
            signed: Whether to sign the request
            api_key_required: Whether to include API key header (without signature)

        Returns:
            JSON response as dict or list

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
                # Some endpoints require API key but not signature (e.g., listen key)
                if self._auth is None:
                    raise AuthenticationError("API key required for this request")
                headers = self._auth.get_headers()

            logger.debug(f"Request: {method} {endpoint} (attempt {attempt + 1}/{self._max_retries + 1})")

            # FIX S-3: Acquire rate limit capacity before request
            weight = self._rate_limiter.get_endpoint_weight(endpoint)
            is_order = endpoint in ("/fapi/v1/order", "/fapi/v1/batchOrders", "/sapi/v1/algo/futures/newOrderTwap")
            acquired = await self._rate_limiter.acquire(
                weight=weight, is_order=is_order, endpoint=endpoint, timeout=10.0
            )
            if not acquired:
                raise RateLimitError("Rate limit capacity not available within timeout")

            try:
                if method == "GET":
                    async with self._session.get(url, params=params, headers=headers) as resp:
                        self._rate_limiter.update_from_headers(dict(resp.headers))
                        return await self._handle_response(resp)
                elif method == "POST":
                    async with self._session.post(url, params=params, headers=headers) as resp:
                        self._rate_limiter.update_from_headers(dict(resp.headers))
                        return await self._handle_response(resp)
                elif method == "PUT":
                    async with self._session.put(url, params=params, headers=headers) as resp:
                        self._rate_limiter.update_from_headers(dict(resp.headers))
                        return await self._handle_response(resp)
                elif method == "DELETE":
                    async with self._session.delete(url, params=params, headers=headers) as resp:
                        self._rate_limiter.update_from_headers(dict(resp.headers))
                        return await self._handle_response(resp)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

            except (RateLimitError, ConnectionError) as e:
                # Retry on rate limit and connection errors
                last_exception = e
                if attempt < self._max_retries:
                    delay = self._retry_delay * (2 ** attempt)
                    # For rate limits, use longer delay
                    if isinstance(e, RateLimitError):
                        delay = max(delay, 1.0)  # Minimum 1 second for rate limits
                    logger.warning(
                        f"Retryable error on {endpoint}: {e}. "
                        f"Retrying in {delay:.1f}s (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

            except ExchangeError as e:
                # Handle timestamp/signature errors (-1021, -1022) with re-sync and retry
                error_str = str(e)
                if ("-1021" in error_str or "-1022" in error_str) and attempt < self._max_retries:
                    logger.warning(
                        f"Timestamp error on {endpoint}: {e}. "
                        f"Re-syncing time and retrying (attempt {attempt + 1}/{self._max_retries})"
                    )
                    await self.sync_time()
                    await asyncio.sleep(0.1)  # Brief pause after sync
                    last_exception = e
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
                raise ConnectionError(f"Failed to connect to Binance Futures: {e}") from e

        # Should not reach here, but handle it just in case
        if last_exception:
            raise last_exception
        raise ExchangeError(f"Request failed after {self._max_retries + 1} attempts")

    async def _handle_response(self, response: aiohttp.ClientResponse) -> dict | list:
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

        # FIX S-4: Handle HTTP 418 IP ban - requires longer backoff
        if status == 418:
            logger.critical(
                "IP banned by Binance (HTTP 418). All requests paused for 5 minutes."
            )
            raise RateLimitError(
                "IP banned by Binance (HTTP 418). Must wait before retrying.",
                code="418"
            )

        # Try to parse JSON
        try:
            data = await response.json()
        except Exception as e:
            text = await response.text()
            if status >= 400:
                raise ExchangeError(f"HTTP {status}: {text}")
            logger.error(
                f"JSON parse failed for response (status={status}): {e}. "
                f"Response text: {text[:200] if text else '(empty)'}"
            )
            raise ExchangeError(
                f"Unexpected non-JSON response (HTTP {status}): {text[:200] if text else '(empty)'}"
            )

        # Check for Binance error (but not success codes)
        if isinstance(data, dict) and "code" in data and "msg" in data:
            code = data["code"]
            msg = data["msg"]
            # Success codes: 200, 0 (check both int and str formats)
            if code not in (0, 200, "0", "200"):
                self._raise_exception(int(code) if isinstance(code, str) else code, msg)

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
        elif code == -2010 or code == -2019:
            raise InsufficientBalanceError(error_info, code=str(code))
        elif code in (-2011, -2013, -2021, -2022):
            raise OrderError(error_info, code=str(code))
        elif code == -1021 or code == -1022:
            raise ExchangeError(f"Signature/timestamp error: {error_info}", code=str(code))
        elif code == -4046:
            # No need to change margin type - it's already set
            raise ExchangeError(error_info, code=str(code))
        else:
            raise ExchangeError(error_info, code=str(code))

    # =========================================================================
    # Public API - System
    # =========================================================================

    async def ping(self) -> bool:
        """
        Test connectivity to Binance Futures API.

        Returns:
            True if successful
        """
        await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["PING"]["path"])
        return True

    async def get_server_time(self) -> datetime:
        """
        Get Binance server time.

        Returns:
            Server time as datetime (UTC)
        """
        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["SERVER_TIME"]["path"])
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

        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["EXCHANGE_INFO"]["path"], params)

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
            limit: Number of klines (max 1500, default 500)
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

        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["KLINES"]["path"], params)

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

        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["TICKER_24H"]["path"], params)

        def parse_futures_ticker(item: dict) -> Ticker:
            """Parse futures ticker (different format from spot)."""
            return Ticker(
                symbol=item["symbol"],
                price=Decimal(str(item["lastPrice"])),
                bid=Decimal(str(item.get("bidPrice", item["lastPrice"]))),  # Fallback to lastPrice
                ask=Decimal(str(item.get("askPrice", item["lastPrice"]))),  # Fallback to lastPrice
                high_24h=Decimal(str(item.get("highPrice", item["lastPrice"]))),
                low_24h=Decimal(str(item.get("lowPrice", item["lastPrice"]))),
                volume_24h=Decimal(str(item.get("volume", "0"))),
                change_24h=Decimal(str(item.get("priceChangePercent", "0"))),
                timestamp=timestamp_to_datetime(item.get("closeTime", 0)),
            )

        if symbol:
            return parse_futures_ticker(data)

        return [parse_futures_ticker(t) for t in data]

    async def get_mark_price(self, symbol: str | None = None) -> dict | list[dict]:
        """
        Get mark price and funding rate.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            Dict with mark price info if symbol specified, else list
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["MARK_PRICE"]["path"], params)

        def parse_mark_price(item: dict) -> dict:
            return {
                "symbol": item["symbol"],
                "mark_price": Decimal(str(item["markPrice"])),
                "index_price": Decimal(str(item["indexPrice"])),
                "estimated_settle_price": Decimal(str(item.get("estimatedSettlePrice", "0"))),
                "last_funding_rate": Decimal(str(item.get("lastFundingRate", "0"))),
                "next_funding_time": timestamp_to_datetime(item.get("nextFundingTime", 0)),
                "interest_rate": Decimal(str(item.get("interestRate", "0"))),
                "timestamp": timestamp_to_datetime(item.get("time", 0)),
            }

        if symbol:
            return parse_mark_price(data)

        return [parse_mark_price(item) for item in data]

    async def get_funding_rate(
        self,
        symbol: str,
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[dict]:
        """
        Get funding rate history.

        Args:
            symbol: Trading pair
            limit: Number of records (max 1000, default 100)
            start_time: Start time in milliseconds
            end_time: End time in milliseconds

        Returns:
            List of funding rate records
        """
        params = {"symbol": symbol, "limit": limit}

        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time

        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["FUNDING_RATE"]["path"], params)

        return [
            {
                "symbol": item["symbol"],
                "funding_rate": Decimal(str(item.get("fundingRate", "0"))),
                "funding_time": timestamp_to_datetime(item.get("fundingTime", 0)),
                "mark_price": Decimal(str(item.get("markPrice", "0"))),
            }
            for item in data
        ]

    async def get_orderbook(self, symbol: str, limit: int = 100) -> dict:
        """
        Get order book (market depth).

        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)

        Returns:
            Dict with 'bids' and 'asks' as list of [price, quantity]
        """
        params = {"symbol": symbol, "limit": limit}
        data = await self._request("GET", FUTURES_PUBLIC_ENDPOINTS["DEPTH"]["path"], params)

        return {
            "bids": [[Decimal(str(p)), Decimal(str(q))] for p, q in data.get("bids", [])],
            "asks": [[Decimal(str(p)), Decimal(str(q))] for p, q in data.get("asks", [])],
            "lastUpdateId": data.get("lastUpdateId", 0),
        }

    # =========================================================================
    # Private API - Account
    # =========================================================================

    async def get_account(self) -> AccountInfo:
        """
        Get current account information including positions.

        Returns:
            AccountInfo with balances and positions

        Raises:
            AuthenticationError: If not authenticated
        """
        data = await self._request(
            "GET",
            FUTURES_PRIVATE_ENDPOINTS["ACCOUNT"]["path"],
            signed=True,
        )
        return AccountInfo.from_binance_futures(data)

    async def get_balance(self, asset: str = "USDT") -> Balance | None:
        """
        Get balance for specific asset.

        Args:
            asset: Asset name (default "USDT")

        Returns:
            Balance object or None if not found
        """
        data = await self._request(
            "GET",
            FUTURES_PRIVATE_ENDPOINTS["BALANCE"]["path"],
            signed=True,
        )

        for item in data:
            if item.get("asset") == asset:
                return Balance(
                    asset=item["asset"],
                    free=Decimal(str(item.get("availableBalance", "0"))),
                    locked=Decimal(str(item.get("balance", "0"))) - Decimal(str(item.get("availableBalance", "0"))),
                )
        return None

    async def get_positions(self, symbol: str | None = None) -> list[Position]:
        """
        Get all positions or position for specific symbol.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            List of Position objects (only non-zero positions)
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request(
            "GET",
            FUTURES_PRIVATE_ENDPOINTS["POSITION"]["path"],
            params,
            signed=True,
        )

        positions = []
        for item in data:
            qty = Decimal(str(item["positionAmt"]))
            if qty != 0:
                positions.append(Position.from_binance(item))

        return positions

    async def get_position(self, symbol: str) -> Position | None:
        """
        Get position for specific symbol.

        Args:
            symbol: Trading pair

        Returns:
            Position object or None if no position
        """
        positions = await self.get_positions(symbol)
        return positions[0] if positions else None

    # =========================================================================
    # Private API - Settings
    # =========================================================================

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """
        Set leverage for a symbol.

        Args:
            symbol: Trading pair
            leverage: Leverage value (1-125 depending on symbol)

        Returns:
            Dict with leverage and max notional value
        """
        params = {"symbol": symbol, "leverage": leverage}

        data = await self._request(
            "POST",
            FUTURES_PRIVATE_ENDPOINTS["LEVERAGE"]["path"],
            params,
            signed=True,
        )

        return {
            "symbol": data.get("symbol", symbol),
            "leverage": data.get("leverage", leverage),
            "max_notional_value": Decimal(str(data.get("maxNotionalValue", "0"))),
        }

    async def set_margin_type(self, symbol: str, margin_type: str) -> bool:
        """
        Set margin type for a symbol.

        Args:
            symbol: Trading pair
            margin_type: "ISOLATED" or "CROSSED"

        Returns:
            True if successful
        """
        params = {"symbol": symbol, "marginType": margin_type.upper()}

        try:
            await self._request(
                "POST",
                FUTURES_PRIVATE_ENDPOINTS["MARGIN_TYPE"]["path"],
                params,
                signed=True,
            )
            return True
        except ExchangeError as e:
            error_str = str(e)
            # -4046 means margin type already set to target value
            if "-4046" in error_str:
                return True
            # -4047 means cannot change margin type with open orders
            # This is OK if margin type is already correct
            if "-4047" in error_str:
                logger.warning(
                    f"Cannot change margin type for {symbol} (open orders exist). "
                    f"Assuming current margin type is acceptable."
                )
                return True
            raise

    async def get_position_mode(self) -> dict:
        """
        Get current position mode.

        Returns:
            Dict with dual_side_position (True = hedge mode, False = one-way)
        """
        data = await self._request(
            "GET",
            FUTURES_PRIVATE_ENDPOINTS["POSITION_MODE_GET"]["path"],
            signed=True,
        )

        return {"dual_side_position": data["dualSidePosition"]}

    async def set_position_mode(self, dual_side: bool) -> bool:
        """
        Set position mode.

        Args:
            dual_side: True for hedge mode (LONG/SHORT), False for one-way (BOTH)

        Returns:
            True if successful
        """
        params = {"dualSidePosition": str(dual_side).lower()}

        try:
            await self._request(
                "POST",
                FUTURES_PRIVATE_ENDPOINTS["POSITION_MODE_SET"]["path"],
                params,
                signed=True,
            )
            return True
        except ExchangeError as e:
            # -4059 means position mode already set to target value
            if "-4059" in str(e):
                return True
            raise

    async def get_leverage_brackets(self, symbol: str | None = None) -> list[dict]:
        """
        Get leverage brackets (notional value and leverage limits).

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            List of leverage bracket info
        """
        params = {}
        if symbol:
            params["symbol"] = symbol

        data = await self._request(
            "GET",
            FUTURES_PRIVATE_ENDPOINTS["LEVERAGE_BRACKET"]["path"],
            params,
            signed=True,
        )

        return data

    # =========================================================================
    # Private API - Orders
    # =========================================================================

    # Conditional order types that require Algo Order API (since 2025-12-09)
    ALGO_ORDER_TYPES = {
        "STOP_MARKET",
        "TAKE_PROFIT_MARKET",
        "STOP",
        "TAKE_PROFIT",
        "TRAILING_STOP_MARKET",
    }

    async def create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal | str,
        price: Decimal | str | None = None,
        position_side: PositionSide | str = PositionSide.BOTH,
        reduce_only: bool = False,
        stop_price: Decimal | str | None = None,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
        self_trade_prevention: str | None = "EXPIRE_TAKER",
    ) -> Order:
        """
        Create a new futures order.

        Automatically routes conditional orders (STOP_MARKET, TAKE_PROFIT_MARKET, etc.)
        to the Algo Order API as required by Binance since 2025-12-09.

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/STOP_MARKET/etc.)
            quantity: Order quantity
            price: Limit price (required for LIMIT orders)
            position_side: Position side (BOTH for one-way, LONG/SHORT for hedge)
            reduce_only: Only reduce position, no reverse (default False)
            stop_price: Stop price for stop orders (trigger price)
            time_in_force: Time in force (GTC, IOC, FOK)
            client_order_id: Custom order ID
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
        pos_side_str = position_side.value if isinstance(position_side, PositionSide) else position_side

        # Route conditional orders to Algo Order API
        if type_str in self.ALGO_ORDER_TYPES:
            return await self._create_algo_order(
                symbol=symbol,
                side=side_str,
                order_type=type_str,
                quantity=quantity,
                price=price,
                position_side=pos_side_str,
                reduce_only=reduce_only,
                trigger_price=stop_price,
                client_order_id=client_order_id,
                self_trade_prevention=self_trade_prevention,
            )

        # Standard order flow for LIMIT/MARKET orders
        params = {
            "symbol": symbol,
            "side": side_str,
            "type": type_str,
            "quantity": str(quantity),
            "positionSide": pos_side_str,
        }

        # Add price for limit orders
        if price is not None:
            params["price"] = str(price)

        # Add timeInForce for limit orders
        if type_str == "LIMIT":
            params["timeInForce"] = time_in_force

        # Add reduce_only (not allowed when positionSide is LONG or SHORT in Hedge Mode)
        if reduce_only and pos_side_str == "BOTH":
            params["reduceOnly"] = "true"

        # Add client order ID
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Add self-trade prevention mode
        if self_trade_prevention:
            params["selfTradePreventionMode"] = self_trade_prevention

        # Request full order response
        params["newOrderRespType"] = "RESULT"

        data = await self._request(
            "POST",
            FUTURES_PRIVATE_ENDPOINTS["ORDER"]["path"],
            params,
            signed=True,
        )
        return Order.from_binance(data)

    async def _create_algo_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal | str,
        price: Decimal | str | None = None,
        position_side: str = "BOTH",
        reduce_only: bool = False,
        trigger_price: Decimal | str | None = None,
        client_order_id: str | None = None,
        self_trade_prevention: str | None = "EXPIRE_TAKER",
    ) -> Order:
        """
        Create an Algo Order (for conditional orders like STOP_MARKET, TAKE_PROFIT_MARKET).

        Required by Binance since 2025-12-09 for conditional order types.

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (STOP_MARKET/TAKE_PROFIT_MARKET/etc.)
            quantity: Order quantity
            price: Limit price (for STOP/TAKE_PROFIT limit orders)
            position_side: Position side (BOTH/LONG/SHORT)
            reduce_only: Only reduce position
            trigger_price: Price that triggers the order
            client_order_id: Custom order ID
            self_trade_prevention: STP mode (EXPIRE_TAKER, EXPIRE_MAKER, EXPIRE_BOTH, NONE)

        Returns:
            Created Order object
        """
        if trigger_price is None:
            raise OrderError("trigger_price (stop_price) is required for conditional orders")

        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "algoType": "CONDITIONAL",
            "triggerPrice": str(trigger_price),
            "quantity": str(quantity),
            "positionSide": position_side,
            # FIX S-2: Use MARK_PRICE to prevent manipulation-triggered stop losses
            "workingType": "MARK_PRICE",
        }

        # Add price for limit-type conditional orders (STOP, TAKE_PROFIT)
        if price is not None:
            params["price"] = str(price)

        # Add reduce_only (not allowed when positionSide is LONG or SHORT in Hedge Mode)
        if reduce_only and position_side == "BOTH":
            params["reduceOnly"] = "true"

        # Add client order ID
        if client_order_id:
            params["newClientOrderId"] = client_order_id

        # Add self-trade prevention mode
        if self_trade_prevention:
            params["selfTradePreventionMode"] = self_trade_prevention

        data = await self._request(
            "POST",
            FUTURES_PRIVATE_ENDPOINTS["ALGO_ORDER"]["path"],
            params,
            signed=True,
        )

        # Convert Algo Order response to Order object
        return self._algo_response_to_order(data, symbol, side, order_type, quantity, trigger_price)

    def _algo_response_to_order(
        self,
        data: dict,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal | str,
        trigger_price: Decimal | str,
    ) -> Order:
        """Convert Algo Order API response to Order object."""
        from src.core.models import OrderStatus

        # Algo Order API returns different fields
        return Order(
            order_id=str(data.get("algoId", data.get("orderId", ""))),
            client_order_id=data.get("clientOrderId", ""),
            symbol=symbol,
            side=OrderSide(side),
            order_type=OrderType(order_type) if order_type in [e.value for e in OrderType] else OrderType.LIMIT,
            status=OrderStatus.NEW,  # Algo orders start as NEW
            price=Decimal(str(trigger_price)),  # Use trigger price as reference
            quantity=Decimal(str(quantity)),
            filled_qty=Decimal("0"),
            avg_price=None,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

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
            FUTURES_PRIVATE_ENDPOINTS["ORDER_DELETE"]["path"],
            params,
            signed=True,
        )
        return Order.from_binance(data)

    async def cancel_algo_order(
        self,
        symbol: str,
        algo_id: str,
    ) -> dict:
        """
        Cancel an Algo Order (conditional order).

        Args:
            symbol: Trading pair
            algo_id: Algo order ID (returned as algoId from create_algo_order)

        Returns:
            Cancellation response dict
        """
        params = {
            "symbol": symbol,
            "algoId": algo_id,
        }

        data = await self._request(
            "DELETE",
            FUTURES_PRIVATE_ENDPOINTS["ALGO_ORDER_DELETE"]["path"],
            params,
            signed=True,
        )
        return data

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
            FUTURES_PRIVATE_ENDPOINTS["ORDER_GET"]["path"],
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
            FUTURES_PRIVATE_ENDPOINTS["OPEN_ORDERS"]["path"],
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
            FUTURES_PRIVATE_ENDPOINTS["ALL_ORDERS"]["path"],
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
            FUTURES_PRIVATE_ENDPOINTS["USER_TRADES"]["path"],
            params,
            signed=True,
        )
        return [Trade.from_binance(t) for t in data]

    async def cancel_all_orders(self, symbol: str) -> bool:
        """
        Cancel all open orders for a symbol.

        Args:
            symbol: Trading pair

        Returns:
            True if successful
        """
        params = {"symbol": symbol}

        await self._request(
            "DELETE",
            FUTURES_PRIVATE_ENDPOINTS["CANCEL_ALL_ORDERS"]["path"],
            params,
            signed=True,
        )
        return True

    # =========================================================================
    # Convenience Methods - Position Management
    # =========================================================================

    async def close_position(
        self,
        symbol: str,
        position_side: PositionSide | str = PositionSide.BOTH,
    ) -> Order | None:
        """
        Close position for a symbol with market order.

        For one-way mode (BOTH):
            - Long position: SELL order
            - Short position: BUY order

        For hedge mode (LONG/SHORT):
            - Close LONG: SELL with position_side=LONG
            - Close SHORT: BUY with position_side=SHORT

        Args:
            symbol: Trading pair
            position_side: Position side to close (BOTH/LONG/SHORT)

        Returns:
            Order object if position exists, None otherwise
        """
        pos_side_str = position_side.value if isinstance(position_side, PositionSide) else position_side

        # Get current positions
        positions = await self.get_positions(symbol)

        if not positions:
            return None

        for position in positions:
            # Match position side
            if pos_side_str != "BOTH" and position.side != pos_side_str:
                continue

            # Determine close order side
            # For one-way mode: positive qty = long, negative qty = short (but we use abs)
            # We need to check the position side
            if position.side == PositionSide.LONG.value or (position.side == PositionSide.BOTH.value and position.quantity > 0):
                close_side = OrderSide.SELL
            else:
                close_side = OrderSide.BUY

            # Place market close order
            return await self.create_order(
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                position_side=position_side,
                reduce_only=True,
            )

        return None

    async def close_all_positions(self) -> list[Order]:
        """
        Close all open positions with market orders.

        Returns:
            List of close orders
        """
        positions = await self.get_positions()
        orders = []

        for position in positions:
            # Determine close order side
            if position.side == PositionSide.LONG.value:
                close_side = OrderSide.SELL
                pos_side = PositionSide.LONG
            elif position.side == PositionSide.SHORT.value:
                close_side = OrderSide.BUY
                pos_side = PositionSide.SHORT
            else:  # BOTH - one-way mode
                # Check if long or short by entry vs mark price context
                # In one-way mode, we just use BOTH and opposite side
                close_side = OrderSide.SELL if position.quantity > 0 else OrderSide.BUY
                pos_side = PositionSide.BOTH

            order = await self.create_order(
                symbol=position.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                position_side=pos_side,
                reduce_only=True,
            )
            orders.append(order)

        return orders

    # =========================================================================
    # Convenience Methods - Quick Orders
    # =========================================================================

    async def market_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        position_side: PositionSide | str = PositionSide.BOTH,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a market buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            position_side: Position side (BOTH/LONG/SHORT)
            reduce_only: Only reduce position
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            position_side=position_side,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    async def market_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        position_side: PositionSide | str = PositionSide.BOTH,
        reduce_only: bool = False,
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a market sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            position_side: Position side (BOTH/LONG/SHORT)
            reduce_only: Only reduce position
            client_order_id: Custom order ID

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            position_side=position_side,
            reduce_only=reduce_only,
            client_order_id=client_order_id,
        )

    async def limit_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        position_side: PositionSide | str = PositionSide.BOTH,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a limit buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            price: Limit price
            position_side: Position side (BOTH/LONG/SHORT)
            reduce_only: Only reduce position
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
            position_side=position_side,
            reduce_only=reduce_only,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    async def limit_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        position_side: PositionSide | str = PositionSide.BOTH,
        reduce_only: bool = False,
        time_in_force: str = "GTC",
        client_order_id: str | None = None,
    ) -> Order:
        """
        Place a limit sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            price: Limit price
            position_side: Position side (BOTH/LONG/SHORT)
            reduce_only: Only reduce position
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
            position_side=position_side,
            reduce_only=reduce_only,
            time_in_force=time_in_force,
            client_order_id=client_order_id,
        )

    # =========================================================================
    # User Data Stream
    # =========================================================================

    async def create_listen_key(self) -> str:
        """
        Create a new listen key for Futures User Data Stream.

        The listen key is used to subscribe to real-time order/account updates
        via WebSocket. It expires after 60 minutes if not kept alive.

        Returns:
            Listen key string
        """
        data = await self._request(
            "POST",
            FUTURES_PRIVATE_ENDPOINTS["USER_DATA_STREAM"]["path"],
            api_key_required=True,
        )
        return data["listenKey"]

    async def keep_alive_listen_key(self, listen_key: str) -> bool:
        """
        Keep alive a listen key (should be called every 30 minutes).

        Binance listen keys expire after 60 minutes of inactivity.
        Call this method periodically to keep the stream alive.

        Args:
            listen_key: The listen key to keep alive

        Returns:
            True if successful
        """
        await self._request(
            "PUT",
            FUTURES_PRIVATE_ENDPOINTS["USER_DATA_STREAM_KEEPALIVE"]["path"],
            params={"listenKey": listen_key},
            api_key_required=True,
        )
        return True

    async def delete_listen_key(self, listen_key: str) -> bool:
        """
        Delete a listen key.

        Call this when unsubscribing from the User Data Stream
        to clean up resources on Binance's side.

        Args:
            listen_key: The listen key to delete

        Returns:
            True if successful
        """
        await self._request(
            "DELETE",
            FUTURES_PRIVATE_ENDPOINTS["USER_DATA_STREAM_DELETE"]["path"],
            params={"listenKey": listen_key},
            api_key_required=True,
        )
        return True
