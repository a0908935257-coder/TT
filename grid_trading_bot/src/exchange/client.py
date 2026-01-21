"""
Unified Exchange Client.

Provides a single entry point for all exchange operations including
REST API (Spot & Futures) and WebSocket streaming.
"""

import asyncio
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Optional

from src.core import get_logger
from src.core.models import (
    AccountInfo,
    Balance,
    Kline,
    KlineInterval,
    MarketType,
    Order,
    OrderSide,
    OrderType,
    Position,
    SymbolInfo,
    Ticker,
)

from .binance.futures_api import BinanceFuturesAPI
from .binance.spot_api import BinanceSpotAPI
from .binance.websocket import BinanceWebSocket

logger = get_logger(__name__)


class ExchangeClient:
    """
    Unified exchange client integrating REST APIs and WebSocket.

    Provides a single entry point for:
    - Spot REST API operations
    - Futures REST API operations
    - WebSocket subscriptions for real-time data

    Example:
        >>> async with ExchangeClient(api_key, api_secret, testnet=True) as client:
        ...     # Get price (spot by default)
        ...     price = await client.get_price("BTCUSDT")
        ...
        ...     # Place order
        ...     order = await client.limit_buy("BTCUSDT", Decimal("0.001"), price)
        ...
        ...     # Futures operations
        ...     await client.futures.set_leverage("BTCUSDT", 10)
        ...     positions = await client.get_positions("BTCUSDT")
        ...
        ...     # Real-time data
        ...     await client.subscribe_ticker("BTCUSDT", on_ticker)
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = False,
    ):
        """
        Initialize ExchangeClient.

        Args:
            api_key: Binance API key (optional for public endpoints)
            api_secret: Binance API secret (optional for public endpoints)
            testnet: Use testnet URLs if True
        """
        self._api_key = api_key
        self._api_secret = api_secret
        self._testnet = testnet

        # Initialize API clients
        self._spot = BinanceSpotAPI(api_key, api_secret, testnet)
        self._futures = BinanceFuturesAPI(api_key, api_secret, testnet)

        # WebSocket clients (lazy init)
        self._spot_ws: Optional[BinanceWebSocket] = None
        self._futures_ws: Optional[BinanceWebSocket] = None

        # Symbol info cache
        self._symbol_cache: dict[str, dict[MarketType, SymbolInfo]] = {}

        # Connection state
        self._connected = False

        # User Data Stream state (Spot)
        self._user_data_listen_key: Optional[str] = None
        self._user_data_callback: Optional[Callable[[dict], Any]] = None
        self._user_data_keepalive_task: Optional[asyncio.Task] = None

        # User Data Stream state (Futures)
        self._futures_user_data_listen_key: Optional[str] = None
        self._futures_user_data_callback: Optional[Callable[[dict], Any]] = None
        self._futures_user_data_keepalive_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def spot(self) -> BinanceSpotAPI:
        """Get Spot API client."""
        return self._spot

    @property
    def futures(self) -> BinanceFuturesAPI:
        """Get Futures API client."""
        return self._futures

    @property
    def ws(self) -> Optional[BinanceWebSocket]:
        """Get Spot WebSocket client (primary WS)."""
        return self._spot_ws

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish all connections.

        Returns:
            True if all connections successful
        """
        try:
            # Connect REST APIs
            await self._spot.connect()
            await self._futures.connect()

            # Sync time with server (both spot and futures)
            await self._spot.sync_time()
            await self._futures.sync_time()

            # Initialize WebSocket clients
            self._spot_ws = BinanceWebSocket(
                market_type=MarketType.SPOT,
                testnet=self._testnet,
            )
            self._futures_ws = BinanceWebSocket(
                market_type=MarketType.FUTURES,
                testnet=self._testnet,
            )

            # Connect WebSockets
            await self._spot_ws.connect()
            await self._futures_ws.connect()

            self._connected = True
            logger.info("ExchangeClient connected")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.close()
            return False

    async def close(self) -> None:
        """Close all connections gracefully."""
        logger.info("Closing ExchangeClient")

        self._connected = False

        # Cancel Spot user data keepalive task
        if self._user_data_keepalive_task:
            self._user_data_keepalive_task.cancel()
            try:
                await self._user_data_keepalive_task
            except asyncio.CancelledError:
                pass
            self._user_data_keepalive_task = None

        # Cancel Futures user data keepalive task
        if self._futures_user_data_keepalive_task:
            self._futures_user_data_keepalive_task.cancel()
            try:
                await self._futures_user_data_keepalive_task
            except asyncio.CancelledError:
                pass
            self._futures_user_data_keepalive_task = None

        # Delete Futures listen key if exists
        if self._futures_user_data_listen_key:
            try:
                await self._futures.delete_listen_key(self._futures_user_data_listen_key)
            except Exception:
                pass
            self._futures_user_data_listen_key = None

        # Close WebSockets
        if self._spot_ws:
            await self._spot_ws.disconnect()
            self._spot_ws = None
        if self._futures_ws:
            await self._futures_ws.disconnect()
            self._futures_ws = None

        # Close REST APIs
        await self._spot.close()
        await self._futures.close()

        logger.info("ExchangeClient closed")

    async def disconnect(self) -> None:
        """Alias for close() for API consistency."""
        await self.close()

    async def __aenter__(self) -> "ExchangeClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    async def get_price(
        self,
        symbol: str,
        market: MarketType = MarketType.SPOT,
    ) -> Decimal:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            market: SPOT or FUTURES

        Returns:
            Current price as Decimal
        """
        api = self._get_api(market)
        return await api.get_price(symbol)

    async def get_ticker(
        self,
        symbol: str,
        market: MarketType = MarketType.SPOT,
    ) -> Ticker:
        """
        Get 24hr ticker for a symbol.

        Args:
            symbol: Trading pair
            market: SPOT or FUTURES

        Returns:
            Ticker object
        """
        api = self._get_api(market)
        return await api.get_ticker(symbol)

    async def get_klines(
        self,
        symbol: str,
        interval: KlineInterval | str,
        limit: int = 500,
        market: MarketType = MarketType.SPOT,
    ) -> list[Kline]:
        """
        Get kline/candlestick data.

        Args:
            symbol: Trading pair
            interval: Kline interval
            limit: Number of klines
            market: SPOT or FUTURES

        Returns:
            List of Kline objects
        """
        api = self._get_api(market)
        return await api.get_klines(symbol, interval, limit)

    # =========================================================================
    # Account Methods
    # =========================================================================

    async def get_balance(
        self,
        asset: str,
        market: MarketType = MarketType.SPOT,
    ) -> Optional[Balance]:
        """
        Get balance for specific asset.

        Args:
            asset: Asset name (e.g., "USDT", "BTC")
            market: SPOT or FUTURES

        Returns:
            Balance object or None if not found
        """
        api = self._get_api(market)
        return await api.get_balance(asset)

    async def get_account(
        self,
        market: MarketType = MarketType.SPOT,
    ) -> AccountInfo:
        """
        Get account information.

        Args:
            market: SPOT or FUTURES

        Returns:
            AccountInfo object
        """
        api = self._get_api(market)
        return await api.get_account()

    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> list[Position]:
        """
        Get futures positions.

        Args:
            symbol: Trading pair (optional, returns all if None)

        Returns:
            List of Position objects (Futures only)
        """
        return await self._futures.get_positions(symbol)

    # =========================================================================
    # Order Methods
    # =========================================================================

    async def create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
        **kwargs,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/etc.)
            quantity: Order quantity
            market: SPOT or FUTURES
            **kwargs: Additional order parameters (price, time_in_force, etc.)

        Returns:
            Created Order object
        """
        api = self._get_api(market)
        return await api.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            **kwargs,
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Cancel an active order.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            market: SPOT or FUTURES

        Returns:
            Cancelled Order object
        """
        api = self._get_api(market)
        return await api.cancel_order(symbol, order_id=order_id)

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Get order status.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            market: SPOT or FUTURES

        Returns:
            Order object
        """
        api = self._get_api(market)
        return await api.get_order(symbol, order_id=order_id)

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        market: MarketType = MarketType.SPOT,
    ) -> list[Order]:
        """
        Get all open orders.

        Args:
            symbol: Trading pair (optional, returns all if None)
            market: SPOT or FUTURES

        Returns:
            List of open Order objects
        """
        api = self._get_api(market)
        return await api.get_open_orders(symbol)

    # =========================================================================
    # Futures Order Wrapper Methods (for OrderExecutor Protocol)
    # =========================================================================

    async def futures_create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal | str,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: Optional[str] = None,
        reduce_only: bool = False,
    ) -> Order:
        """
        Create a futures order (wrapper for OrderExecutor).

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/STOP_MARKET/etc.)
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            time_in_force: Time in force (GTC, IOC, etc.)
            reduce_only: Whether order is reduce-only

        Returns:
            Created Order object
        """
        return await self._futures.create_order(
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            time_in_force=time_in_force,
            reduce_only=reduce_only,
        )

    async def futures_cancel_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Order:
        """
        Cancel a futures order (wrapper for OrderExecutor).

        Note: For Algo orders (STOP_MARKET, etc.), use futures_cancel_algo_order instead.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID

        Returns:
            Cancelled Order object
        """
        return await self._futures.cancel_order(symbol, order_id=order_id)

    async def futures_cancel_algo_order(
        self,
        symbol: str,
        algo_id: str,
    ) -> dict:
        """
        Cancel a futures Algo order (STOP_MARKET, TAKE_PROFIT_MARKET, etc.).

        Since 2025-12-09, conditional orders use Algo Order API.

        Args:
            symbol: Trading pair
            algo_id: Algo order ID (same as order_id from create_order)

        Returns:
            Cancellation response
        """
        return await self._futures.cancel_algo_order(symbol, algo_id=algo_id)

    async def futures_get_order(
        self,
        symbol: str,
        order_id: str,
    ) -> Order:
        """
        Get futures order status (wrapper for OrderExecutor).

        Args:
            symbol: Trading pair
            order_id: Exchange order ID

        Returns:
            Order object
        """
        return await self._futures.get_order(symbol, order_id=order_id)

    # =========================================================================
    # Convenience Order Methods
    # =========================================================================

    async def market_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a market buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            market: SPOT or FUTURES

        Returns:
            Order object
        """
        api = self._get_api(market)
        return await api.market_buy(symbol, quantity)

    async def market_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a market sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            market: SPOT or FUTURES

        Returns:
            Order object
        """
        api = self._get_api(market)
        return await api.market_sell(symbol, quantity)

    async def limit_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a limit buy order.

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            price: Limit price
            market: SPOT or FUTURES

        Returns:
            Order object
        """
        api = self._get_api(market)
        return await api.limit_buy(symbol, quantity, price)

    async def limit_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Place a limit sell order.

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            price: Limit price
            market: SPOT or FUTURES

        Returns:
            Order object
        """
        api = self._get_api(market)
        return await api.limit_sell(symbol, quantity, price)

    # =========================================================================
    # Subscription Methods
    # =========================================================================

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[Ticker], None],
        market: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Subscribe to ticker stream.

        Args:
            symbol: Trading pair
            callback: Callback receiving Ticker objects
            market: SPOT or FUTURES

        Returns:
            True if successful
        """
        ws = self._get_ws(market)
        if ws is None:
            logger.error("WebSocket not connected")
            return False
        return await ws.subscribe_ticker(symbol, callback)

    async def subscribe_kline(
        self,
        symbol: str,
        interval: KlineInterval | str,
        callback: Callable[[Kline], None],
        market: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Subscribe to kline stream.

        Args:
            symbol: Trading pair
            interval: Kline interval
            callback: Callback receiving Kline objects
            market: SPOT or FUTURES

        Returns:
            True if successful
        """
        ws = self._get_ws(market)
        if ws is None:
            logger.error("WebSocket not connected")
            return False
        return await ws.subscribe_kline(symbol, interval, callback)

    async def unsubscribe_kline(
        self,
        symbol: str,
        interval: KlineInterval | str,
        market: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Unsubscribe from kline stream.

        Args:
            symbol: Trading pair
            interval: Kline interval
            market: SPOT or FUTURES

        Returns:
            True if successful
        """
        ws = self._get_ws(market)
        if ws is None:
            logger.error("WebSocket not connected")
            return False
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval
        stream = f"{symbol.lower()}@kline_{interval_str}"
        return await ws.unsubscribe([stream])

    async def unsubscribe_all(self) -> None:
        """Unsubscribe from all streams."""
        if self._spot_ws and self._spot_ws._subscriptions:
            streams = list(self._spot_ws._subscriptions.keys())
            await self._spot_ws.unsubscribe(streams)

        if self._futures_ws and self._futures_ws._subscriptions:
            streams = list(self._futures_ws._subscriptions.keys())
            await self._futures_ws.unsubscribe(streams)

    # =========================================================================
    # User Data Stream
    # =========================================================================

    async def subscribe_user_data(
        self,
        callback: Callable[[dict], Any],
        market: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Subscribe to User Data Stream for order/account updates.

        Args:
            callback: Callback function receiving user data events
            market: SPOT or FUTURES

        Returns:
            True if subscription successful
        """
        try:
            # Create listen key based on market type
            if market == MarketType.SPOT:
                self._user_data_listen_key = await self._spot.create_listen_key()
                self._user_data_callback = callback
                listen_key = self._user_data_listen_key
            else:
                # Futures User Data Stream
                self._futures_user_data_listen_key = await self._futures.create_listen_key()
                self._futures_user_data_callback = callback
                listen_key = self._futures_user_data_listen_key
                logger.info(f"Created Futures listen key: {listen_key[:20]}...")

            # Subscribe to the user data stream via WebSocket
            ws = self._get_ws(market)
            if ws is None:
                logger.error("WebSocket not connected")
                return False

            stream = listen_key

            async def wrapper(data: dict):
                if market == MarketType.SPOT:
                    if self._user_data_callback:
                        await self._invoke_user_data_callback(data, market)
                else:
                    if self._futures_user_data_callback:
                        await self._invoke_user_data_callback(data, market)

            # Subscribe to user data stream
            success = await ws.subscribe([stream], wrapper)

            if success:
                # Start keep-alive task (ping every 30 minutes)
                if market == MarketType.SPOT:
                    self._user_data_keepalive_task = asyncio.create_task(
                        self._user_data_keepalive_loop(market)
                    )
                else:
                    self._futures_user_data_keepalive_task = asyncio.create_task(
                        self._user_data_keepalive_loop(market)
                    )
                logger.info(f"Subscribed to {market.value} user data stream")

            return success

        except Exception as e:
            logger.error(f"Failed to subscribe to {market.value} user data stream: {e}")
            return False

    async def unsubscribe_user_data(
        self,
        market: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Unsubscribe from User Data Stream.

        Args:
            market: SPOT or FUTURES

        Returns:
            True if unsubscription successful
        """
        try:
            if market == MarketType.SPOT:
                # Stop Spot keep-alive task
                if self._user_data_keepalive_task:
                    self._user_data_keepalive_task.cancel()
                    try:
                        await self._user_data_keepalive_task
                    except asyncio.CancelledError:
                        pass
                    self._user_data_keepalive_task = None

                # Unsubscribe from WebSocket
                if self._user_data_listen_key and self._spot_ws:
                    await self._spot_ws.unsubscribe([self._user_data_listen_key])

                # Delete listen key
                if self._user_data_listen_key:
                    await self._spot.delete_listen_key(self._user_data_listen_key)
                    self._user_data_listen_key = None

                self._user_data_callback = None
            else:
                # Stop Futures keep-alive task
                if self._futures_user_data_keepalive_task:
                    self._futures_user_data_keepalive_task.cancel()
                    try:
                        await self._futures_user_data_keepalive_task
                    except asyncio.CancelledError:
                        pass
                    self._futures_user_data_keepalive_task = None

                # Unsubscribe from WebSocket
                if self._futures_user_data_listen_key and self._futures_ws:
                    await self._futures_ws.unsubscribe([self._futures_user_data_listen_key])

                # Delete listen key
                if self._futures_user_data_listen_key:
                    await self._futures.delete_listen_key(self._futures_user_data_listen_key)
                    self._futures_user_data_listen_key = None

                self._futures_user_data_callback = None

            logger.info(f"Unsubscribed from {market.value} user data stream")
            return True

        except Exception as e:
            logger.error(f"Failed to unsubscribe from {market.value} user data stream: {e}")
            return False

    async def _user_data_keepalive_loop(self, market: MarketType) -> None:
        """
        Keep the user data stream alive by pinging every 30 minutes.

        Args:
            market: SPOT or FUTURES
        """
        try:
            while True:
                await asyncio.sleep(30 * 60)  # 30 minutes

                try:
                    if market == MarketType.SPOT:
                        if not self._user_data_listen_key:
                            break
                        await self._spot.keep_alive_listen_key(
                            self._user_data_listen_key
                        )
                        logger.debug("Spot user data stream keep-alive sent")
                    else:
                        if not self._futures_user_data_listen_key:
                            break
                        await self._futures.keep_alive_listen_key(
                            self._futures_user_data_listen_key
                        )
                        logger.debug("Futures user data stream keep-alive sent")
                except Exception as e:
                    logger.error(f"Failed to keep-alive {market.value} user data stream: {e}")

        except asyncio.CancelledError:
            logger.debug(f"{market.value} user data keep-alive task cancelled")
            raise

    async def _invoke_user_data_callback(
        self,
        data: dict,
        market: MarketType = MarketType.SPOT,
    ) -> None:
        """
        Invoke user data callback, handling both sync and async.

        Args:
            data: User data event dict
            market: SPOT or FUTURES
        """
        callback = (
            self._user_data_callback
            if market == MarketType.SPOT
            else self._futures_user_data_callback
        )

        if callback:
            if asyncio.iscoroutinefunction(callback):
                await callback(data)
            else:
                callback(data)

    # =========================================================================
    # Symbol Info & Precision Methods
    # =========================================================================

    async def get_symbol_info(
        self,
        symbol: str,
        market: MarketType = MarketType.SPOT,
    ) -> SymbolInfo:
        """
        Get symbol trading rules and precision.

        Results are cached to avoid repeated requests.

        Args:
            symbol: Trading pair
            market: SPOT or FUTURES

        Returns:
            SymbolInfo object
        """
        # Check cache
        if symbol in self._symbol_cache:
            if market in self._symbol_cache[symbol]:
                return self._symbol_cache[symbol][market]
        else:
            self._symbol_cache[symbol] = {}

        # Fetch from API
        api = self._get_api(market)
        info = await api.get_exchange_info(symbol)

        if isinstance(info, SymbolInfo):
            self._symbol_cache[symbol][market] = info
            return info

        raise ValueError(f"Could not get symbol info for {symbol}")

    def get_price_precision(
        self,
        symbol: str,
        market: MarketType = MarketType.SPOT,
    ) -> int:
        """
        Get price precision for a symbol from cache.

        Note: Must call get_symbol_info first to populate cache.

        Args:
            symbol: Trading pair
            market: SPOT or FUTURES

        Returns:
            Price precision (decimal places)
        """
        if symbol in self._symbol_cache and market in self._symbol_cache[symbol]:
            return self._symbol_cache[symbol][market].price_precision
        return 8  # Default

    def get_quantity_precision(
        self,
        symbol: str,
        market: MarketType = MarketType.SPOT,
    ) -> int:
        """
        Get quantity precision for a symbol from cache.

        Note: Must call get_symbol_info first to populate cache.

        Args:
            symbol: Trading pair
            market: SPOT or FUTURES

        Returns:
            Quantity precision (decimal places)
        """
        if symbol in self._symbol_cache and market in self._symbol_cache[symbol]:
            return self._symbol_cache[symbol][market].quantity_precision
        return 8  # Default

    def round_price(
        self,
        symbol: str,
        price: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Decimal:
        """
        Round price to symbol's precision.

        Args:
            symbol: Trading pair
            price: Price to round
            market: SPOT or FUTURES

        Returns:
            Rounded price
        """
        price = Decimal(str(price))
        precision = self.get_price_precision(symbol, market)
        return self._round_decimal(price, precision)

    def round_quantity(
        self,
        symbol: str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
    ) -> Decimal:
        """
        Round quantity to symbol's precision.

        Args:
            symbol: Trading pair
            quantity: Quantity to round
            market: SPOT or FUTURES

        Returns:
            Rounded quantity
        """
        quantity = Decimal(str(quantity))
        precision = self.get_quantity_precision(symbol, market)
        return self._round_decimal(quantity, precision)

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_api(self, market: MarketType) -> BinanceSpotAPI | BinanceFuturesAPI:
        """Get API client for market type."""
        if market == MarketType.SPOT:
            return self._spot
        return self._futures

    def _get_ws(self, market: MarketType) -> Optional[BinanceWebSocket]:
        """Get WebSocket client for market type."""
        if market == MarketType.SPOT:
            return self._spot_ws
        return self._futures_ws

    @staticmethod
    def _round_decimal(value: Decimal, precision: int) -> Decimal:
        """
        Round a Decimal value to specified precision.

        Args:
            value: Value to round
            precision: Number of decimal places

        Returns:
            Rounded value
        """
        if precision <= 0:
            return value.quantize(Decimal("1"), rounding=ROUND_DOWN)

        quantize_str = "0." + "0" * precision
        return value.quantize(Decimal(quantize_str), rounding=ROUND_DOWN)
