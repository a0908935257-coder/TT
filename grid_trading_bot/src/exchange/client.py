"""
Unified Exchange Client.

Provides a single entry point for all exchange operations including
REST API (Spot & Futures) and WebSocket streaming.

Thread-safe order placement with:
- asyncio.Lock for order operations
- Request queue for serializing orders
- Cross-bot coordination (single lock shared by all bots)
"""

import asyncio
from collections import deque
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime, timezone
import time
import uuid

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


@dataclass
class OrderRequest:
    """Represents a pending order request in the queue."""

    request_id: str
    bot_id: str
    operation: str  # "create", "cancel", "modify"
    params: dict
    future: asyncio.Future  # Must be set by caller
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0  # Higher priority = processed first


@dataclass
class OrderQueueStats:
    """Statistics for the order queue."""

    total_processed: int = 0
    total_errors: int = 0
    avg_wait_time_ms: float = 0.0
    max_wait_time_ms: float = 0.0
    current_queue_size: int = 0


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

        # Symbol info cache with TTL (1 hour)
        self._symbol_cache: dict[str, dict[MarketType, SymbolInfo]] = {}
        self._symbol_cache_time: dict[str, dict[MarketType, float]] = {}
        self._symbol_cache_ttl: float = 3600.0  # 1 hour in seconds

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

        # =======================================================================
        # Order Synchronization (Cross-bot coordination)
        # =======================================================================
        # Main lock for order operations - ensures only one order at a time
        self._order_lock = asyncio.Lock()

        # Request queue for serializing orders
        self._order_queue: deque[OrderRequest] = deque()
        self._queue_lock = asyncio.Lock()
        self._queue_processor_task: Optional[asyncio.Task] = None
        self._queue_running = False

        # Queue limits to prevent signal stacking
        self._max_queue_size = 100  # Maximum pending orders
        self._max_orders_per_bot_per_minute = 10  # Rate limit per bot
        self._bot_order_timestamps: Dict[str, List[float]] = {}  # Bot order history

        # Queue statistics
        self._queue_stats = OrderQueueStats()

        # Request ID counter
        self._request_counter = 0

        # Reconnect callbacks (called when WebSocket reconnects)
        self._reconnect_callbacks: list[Callable] = []

        # Minimum delay between orders (ms) - helps with rate limiting
        self._min_order_interval_ms = 100

        # Time sync health flag — set to False when sync fails repeatedly
        self._time_sync_healthy = True

        # =======================================================================
        # Time Synchronization (WSL2 drift protection)
        # =======================================================================
        self._time_sync_task: Optional[asyncio.Task] = None
        self._time_sync_interval = 30  # Base sync interval (adaptive: 10-60 seconds)
        self._time_sync_interval_min = 10  # Minimum interval when drift detected
        self._time_sync_interval_max = 60  # Maximum interval when stable
        self._time_offset_warning_ms = 1000  # Warn if offset > 1 second
        self._time_offset_critical_ms = 3000  # Critical if offset > 3 seconds
        self._last_sync_time: float = 0  # Track last successful sync time

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
    def futures_ws(self) -> Optional[BinanceWebSocket]:
        """Get Futures WebSocket client."""
        return self._futures_ws

    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected

    def register_reconnect_callback(self, callback: Callable) -> None:
        """Register a callback to be invoked when WebSocket reconnects."""
        self._reconnect_callbacks.append(callback)

    async def _on_ws_reconnect(self) -> None:
        """Internal handler: fan-out reconnect event to all registered callbacks."""
        logger.warning("WebSocket reconnected — triggering position resync for all registered callbacks")
        for cb in self._reconnect_callbacks:
            try:
                result = cb()
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error(f"Reconnect callback error: {e}")

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

            # Sync time with server (both spot and futures) with validation
            await self._initial_time_sync()

            # Initialize WebSocket clients
            self._spot_ws = BinanceWebSocket(
                market_type=MarketType.SPOT,
                testnet=self._testnet,
            )
            self._futures_ws = BinanceWebSocket(
                market_type=MarketType.FUTURES,
                testnet=self._testnet,
                on_reconnect=self._on_ws_reconnect,
            )

            # Connect WebSockets
            await self._spot_ws.connect()
            await self._futures_ws.connect()

            # Check API key permissions (warn if withdraw enabled)
            await self._check_api_permissions()

            # Start order queue processor
            await self._start_order_queue_processor()

            # Start periodic time sync task
            await self._start_time_sync_task()

            self._connected = True
            logger.info("ExchangeClient connected (with order queue and time sync)")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.close()
            return False

    async def _initial_time_sync(self) -> None:
        """
        Perform initial time synchronization with validation.

        Syncs with both Spot and Futures servers and logs the result.
        Issues warnings if time offset exceeds thresholds.
        """
        # Sync spot
        await self._spot.sync_time()
        spot_offset = self._spot._auth.time_offset if self._spot._auth else 0

        # Sync futures
        await self._futures.sync_time()
        futures_offset = self._futures._auth.time_offset if self._futures._auth else 0

        # Log sync result
        logger.info(
            f"Initial time sync: spot_offset={spot_offset}ms, "
            f"futures_offset={futures_offset}ms"
        )

        # Check for excessive drift
        max_offset = max(abs(spot_offset), abs(futures_offset))

        if max_offset > self._time_offset_critical_ms:
            logger.error(
                f"CRITICAL: System time offset is {max_offset}ms "
                f"(threshold: {self._time_offset_critical_ms}ms). "
                f"API requests may fail due to timestamp issues. "
                f"Please sync your system clock with NTP."
            )
        elif max_offset > self._time_offset_warning_ms:
            logger.warning(
                f"System time offset is {max_offset}ms "
                f"(threshold: {self._time_offset_warning_ms}ms). "
                f"Consider syncing your system clock."
            )

    async def _check_api_permissions(self) -> None:
        """
        Check API key permissions and warn if withdraw is enabled.

        Does not block startup, only logs a warning.
        """
        try:
            account = await self._spot.get_account()
            permissions = getattr(account, "permissions", None)
            if permissions and isinstance(permissions, (list, set)):
                # Check for withdraw-related permissions
                withdraw_perms = {"WITHDRAW", "INTERNAL_TRANSFER"}
                found = withdraw_perms & set(permissions)
                if found:
                    logger.warning(
                        f"⚠️ SECURITY WARNING: API key has {found} permissions. "
                        f"It is strongly recommended to use an API key WITHOUT "
                        f"withdrawal permissions for trading bots."
                    )
                else:
                    logger.info("API key permissions OK (no withdraw permission)")
        except Exception as e:
            logger.debug(f"Could not check API permissions: {e}")

    async def close(self) -> None:
        """Close all connections gracefully."""
        logger.info("Closing ExchangeClient")

        self._connected = False

        # Stop time sync task
        await self._stop_time_sync_task()

        # Stop order queue processor
        await self._stop_order_queue_processor()

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
            except Exception as e:
                logger.warning(f"Failed to delete futures listen key during close: {e}")
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
    # Time Synchronization Management
    # =========================================================================

    async def _start_time_sync_task(self) -> None:
        """Start the background time synchronization task."""
        if self._time_sync_task is not None:
            return

        self._time_sync_task = asyncio.create_task(self._time_sync_loop())
        logger.info("Time sync task started")

    async def _stop_time_sync_task(self) -> None:
        """Stop the time synchronization task."""
        if self._time_sync_task:
            self._time_sync_task.cancel()
            try:
                await self._time_sync_task
            except asyncio.CancelledError:
                pass
            self._time_sync_task = None
            logger.debug("Time sync task stopped")

    async def _time_sync_loop(self) -> None:
        """
        Periodically synchronize time with exchange servers.

        Monitors time offset and logs warnings if drift is detected.
        Includes exponential backoff on consecutive failures.
        """
        consecutive_failures = 0
        max_failures_before_critical = 5
        try:
            while self._connected:
                await asyncio.sleep(self._time_sync_interval)

                if not self._connected:
                    break

                try:
                    # Sync both spot and futures
                    await self._sync_time_with_warning()
                    consecutive_failures = 0  # Reset on success
                    if not self._time_sync_healthy:
                        self._time_sync_healthy = True
                        logger.info("Time sync recovered — order submission re-enabled")
                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures_before_critical:
                        if self._time_sync_healthy:
                            self._time_sync_healthy = False
                            logger.critical(
                                f"Time sync failed {consecutive_failures} consecutive times — "
                                "pausing order submission to prevent -1021 rejections"
                            )
                    else:
                        logger.error(f"Time sync failed ({consecutive_failures}/{max_failures_before_critical}): {e}")

        except asyncio.CancelledError:
            logger.debug("Time sync loop cancelled")
            raise

    async def _sync_time_with_warning(self) -> None:
        """
        Synchronize time and check for excessive drift.

        Issues warnings if time offset exceeds thresholds.
        Implements adaptive interval adjustment based on drift severity.
        """
        # Sync spot (check _auth exists to avoid AttributeError)
        await self._spot.sync_time()
        spot_offset = self._spot._auth.time_offset if self._spot._auth else 0

        # Sync futures (check _auth exists to avoid AttributeError)
        await self._futures.sync_time()
        futures_offset = self._futures._auth.time_offset if self._futures._auth else 0

        # Update last sync time
        self._last_sync_time = time.time()

        # Determine max offset for adaptive interval adjustment
        max_offset = max(abs(spot_offset), abs(futures_offset))

        # Adaptive interval adjustment based on drift severity
        if max_offset > 500:
            # High drift: use minimum interval
            self._time_sync_interval = self._time_sync_interval_min
            logger.warning(
                f"High time drift detected ({max_offset}ms), "
                f"reducing sync interval to {self._time_sync_interval}s"
            )
        elif max_offset < 200:
            # Low drift: gradually increase interval (max 60s)
            self._time_sync_interval = min(
                self._time_sync_interval + 5,
                self._time_sync_interval_max
            )
            logger.debug(
                f"Stable time offset ({max_offset}ms), "
                f"sync interval: {self._time_sync_interval}s"
            )

        # Check spot offset
        if abs(spot_offset) > self._time_offset_critical_ms:
            logger.error(
                f"CRITICAL: Spot time offset is {spot_offset}ms "
                f"(threshold: {self._time_offset_critical_ms}ms). "
                "Orders may be rejected!"
            )
        elif abs(spot_offset) > self._time_offset_warning_ms:
            logger.warning(
                f"Spot time offset is {spot_offset}ms "
                f"(threshold: {self._time_offset_warning_ms}ms)"
            )
        else:
            logger.debug(f"Spot time synced, offset: {spot_offset}ms")

        # Check futures offset
        if abs(futures_offset) > self._time_offset_critical_ms:
            logger.error(
                f"CRITICAL: Futures time offset is {futures_offset}ms "
                f"(threshold: {self._time_offset_critical_ms}ms). "
                "Orders may be rejected!"
            )
        elif abs(futures_offset) > self._time_offset_warning_ms:
            logger.warning(
                f"Futures time offset is {futures_offset}ms "
                f"(threshold: {self._time_offset_warning_ms}ms)"
            )
        else:
            logger.debug(f"Futures time synced, offset: {futures_offset}ms")

    def get_time_offsets(self) -> dict[str, int]:
        """
        Get current time offsets for both spot and futures.

        Returns:
            Dictionary with 'spot' and 'futures' offset values in milliseconds
        """
        return {
            "spot": self._spot._auth.time_offset if self._spot._auth else 0,
            "futures": self._futures._auth.time_offset if self._futures._auth else 0,
        }

    async def force_time_sync(self) -> dict[str, int]:
        """
        Force immediate time synchronization.

        Returns:
            Dictionary with updated offset values
        """
        await self._sync_time_with_warning()
        return self.get_time_offsets()

    def _should_sync_before_order(self) -> bool:
        """
        Check if time sync is needed before placing an order.

        Returns True if last sync was more than 10 seconds ago,
        which helps prevent -1021 timestamp errors in WSL2 environments.

        Returns:
            True if sync is recommended before order submission
        """
        return time.time() - self._last_sync_time > 10

    async def _quick_time_sync(self, futures_only: bool = False) -> None:
        """
        Perform quick time sync before order submission.

        Used to prevent -1021 errors in WSL2 environments with clock drift.

        Args:
            futures_only: If True, only sync Futures API. If False, sync both.
        """
        try:
            if futures_only:
                await self._futures.sync_time()
                futures_offset = self._futures._auth.time_offset if self._futures._auth else 0
                logger.debug(f"Pre-order time sync (futures), offset: {futures_offset}ms")
            else:
                # Sync both APIs in parallel for efficiency
                await asyncio.gather(
                    self._spot.sync_time(),
                    self._futures.sync_time(),
                    return_exceptions=True,  # Don't fail if one sync fails
                )
                spot_offset = self._spot._auth.time_offset if self._spot._auth else 0
                futures_offset = self._futures._auth.time_offset if self._futures._auth else 0
                logger.debug(
                    f"Pre-order time sync completed, "
                    f"spot: {spot_offset}ms, futures: {futures_offset}ms"
                )
            self._last_sync_time = time.time()
        except Exception as e:
            logger.warning(f"Pre-order time sync failed: {e}")

    # =========================================================================
    # Order Queue Management (Cross-bot Coordination)
    # =========================================================================

    async def _start_order_queue_processor(self) -> None:
        """Start the background order queue processor."""
        if self._queue_running:
            return

        self._queue_running = True
        self._queue_processor_task = asyncio.create_task(
            self._process_order_queue()
        )
        logger.info("Order queue processor started")

    async def _stop_order_queue_processor(self) -> None:
        """Stop the order queue processor gracefully."""
        self._queue_running = False

        if self._queue_processor_task:
            self._queue_processor_task.cancel()
            try:
                await self._queue_processor_task
            except asyncio.CancelledError:
                pass
            self._queue_processor_task = None

        # Cancel any pending requests in the queue
        async with self._queue_lock:
            while self._order_queue:
                request = self._order_queue.popleft()
                if not request.future.done():
                    request.future.cancel()
            logger.info("Order queue processor stopped")

    async def _process_order_queue(self) -> None:
        """
        Background task that processes orders from the queue sequentially.
        Ensures only one order is being processed at a time across all bots.
        """
        last_order_time = datetime.now(timezone.utc)

        while self._queue_running:
            try:
                # Check if there's a request to process
                request = None
                async with self._queue_lock:
                    if self._order_queue:
                        request = self._order_queue.popleft()
                        self._queue_stats.current_queue_size = len(self._order_queue)

                if request is None:
                    # No request, wait a bit
                    await asyncio.sleep(0.01)  # 10ms
                    continue

                # Enforce minimum interval between orders
                elapsed = (datetime.now(timezone.utc) - last_order_time).total_seconds() * 1000
                if elapsed < self._min_order_interval_ms:
                    # Use max(0, ...) to handle potential clock drift from NTP sync
                    sleep_time = max(0, self._min_order_interval_ms - elapsed) / 1000
                    await asyncio.sleep(sleep_time)

                # Process the request with the order lock
                async with self._order_lock:
                    try:
                        result = await asyncio.wait_for(
                            self._execute_order_request(request),
                            timeout=30.0,
                        )
                        if not request.future.done():
                            request.future.set_result(result)

                        # Update statistics
                        wait_time = (datetime.now(timezone.utc) - request.created_at).total_seconds() * 1000
                        self._queue_stats.max_wait_time_ms = max(
                            self._queue_stats.max_wait_time_ms, wait_time
                        )
                        # Running average (calculate before incrementing counter)
                        n = self._queue_stats.total_processed
                        if n > 0:
                            self._queue_stats.avg_wait_time_ms = (
                                (self._queue_stats.avg_wait_time_ms * n + wait_time) / (n + 1)
                            )
                        else:
                            self._queue_stats.avg_wait_time_ms = wait_time
                        self._queue_stats.total_processed += 1

                    except Exception as e:
                        self._queue_stats.total_errors += 1
                        if not request.future.done():
                            request.future.set_exception(e)
                        logger.error(f"Order request failed: {e}")

                last_order_time = datetime.now(timezone.utc)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Order queue processor error: {e}")
                await asyncio.sleep(0.1)

    async def _execute_order_request(self, request: OrderRequest) -> Any:
        """Execute a single order request."""
        if not self._time_sync_healthy:
            raise RuntimeError(
                "Order rejected: time sync unhealthy — Binance would reject with -1021"
            )

        params = request.params
        operation = request.operation
        bot_id = request.bot_id

        # Pre-order time sync for WSL2 drift protection
        if self._should_sync_before_order():
            # Determine if we only need Futures sync
            futures_only = operation.startswith("futures_") or (
                operation in ("create", "cancel") and
                params.get("market") == MarketType.FUTURES
            )
            await self._quick_time_sync(futures_only=futures_only)

        # Generate client_order_id with bot prefix for order tracking
        client_order_id = self.generate_client_order_id(bot_id)

        if operation == "create":
            api = self._get_api(params.get("market", MarketType.SPOT))
            kwargs = params.get("kwargs", {})
            # Add client_order_id if not already provided
            if "client_order_id" not in kwargs:
                kwargs["client_order_id"] = client_order_id
            return await api.create_order(
                symbol=params["symbol"],
                side=params["side"],
                order_type=params["order_type"],
                quantity=params["quantity"],
                **kwargs,
            )
        elif operation == "cancel":
            api = self._get_api(params.get("market", MarketType.SPOT))
            return await api.cancel_order(
                params["symbol"],
                order_id=params["order_id"],
            )
        elif operation == "futures_create":
            return await self._futures.create_order(
                symbol=params["symbol"],
                side=params["side"],
                order_type=params["order_type"],
                quantity=params["quantity"],
                price=params.get("price"),
                stop_price=params.get("stop_price"),
                time_in_force=params.get("time_in_force"),
                reduce_only=params.get("reduce_only", False),
                position_side=params.get("position_side", "BOTH"),
                client_order_id=params.get("client_order_id", client_order_id),
            )
        elif operation == "futures_cancel":
            return await self._futures.cancel_order(
                params["symbol"],
                order_id=params["order_id"],
            )
        elif operation == "futures_cancel_algo":
            return await self._futures.cancel_algo_order(
                params["symbol"],
                algo_id=params["algo_id"],
            )
        else:
            raise ValueError(f"Unknown operation: {operation}")

    async def _enqueue_order(
        self,
        operation: str,
        params: dict,
        bot_id: str = "unknown",
        priority: int = 0,
    ) -> Any:
        """
        Enqueue an order request and wait for the result.

        Args:
            operation: Order operation type
            params: Order parameters
            bot_id: ID of the bot making the request
            priority: Request priority (higher = first)

        Returns:
            Result of the order operation

        Raises:
            RuntimeError: If queue is full or bot rate limit exceeded
        """
        async with self._queue_lock:
            # Check queue size limit (prevent signal stacking)
            if len(self._order_queue) >= self._max_queue_size:
                self._queue_stats.total_errors += 1
                raise RuntimeError(
                    f"Order queue full ({self._max_queue_size}). "
                    f"Signal stacking detected - rejecting order from {bot_id}"
                )

            # Check per-bot rate limit
            current_time = time.time()
            one_minute_ago = current_time - 60

            if bot_id not in self._bot_order_timestamps:
                self._bot_order_timestamps[bot_id] = []

            # Clean old timestamps
            self._bot_order_timestamps[bot_id] = [
                ts for ts in self._bot_order_timestamps[bot_id]
                if ts > one_minute_ago
            ]

            # Check rate limit
            if len(self._bot_order_timestamps[bot_id]) >= self._max_orders_per_bot_per_minute:
                self._queue_stats.total_errors += 1
                raise RuntimeError(
                    f"Bot {bot_id} exceeded rate limit "
                    f"({self._max_orders_per_bot_per_minute} orders/minute)"
                )

            # Record this order timestamp
            self._bot_order_timestamps[bot_id].append(current_time)

            # Generate request ID inside lock to avoid race condition
            self._request_counter += 1
            request_id = f"{bot_id}_{self._request_counter}_{uuid.uuid4().hex[:8]}"

        # Create future for this event loop
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        request = OrderRequest(
            request_id=request_id,
            bot_id=bot_id,
            operation=operation,
            params=params,
            future=future,
            priority=priority,
        )

        async with self._queue_lock:
            # Insert based on priority (higher priority first)
            if priority > 0 and self._order_queue:
                # Find insertion point
                inserted = False
                for i, existing in enumerate(self._order_queue):
                    if existing.priority < priority:
                        self._order_queue.insert(i, request)
                        inserted = True
                        break
                if not inserted:
                    self._order_queue.append(request)
            else:
                self._order_queue.append(request)

            self._queue_stats.current_queue_size = len(self._order_queue)

        logger.debug(
            f"Order queued: {request_id} ({operation}) - queue size: {self._queue_stats.current_queue_size}"
        )

        # Wait for the result
        return await future

    def get_order_queue_stats(self) -> OrderQueueStats:
        """Get current order queue statistics."""
        return self._queue_stats

    @staticmethod
    def generate_client_order_id(bot_id: str) -> str:
        """
        Generate a unique client order ID with bot identifier prefix.

        Format: {bot_id}_{timestamp}_{random}
        Example: grid_futures_001_1706123456_a3f2

        This allows filtering orders by bot when syncing with exchange.

        Args:
            bot_id: Bot identifier

        Returns:
            Unique client order ID string (max 36 chars for Binance)
        """
        # Shorten bot_id if needed — leave safety margin for 36-char Binance limit
        # Format: {bot_id}_{timestamp}_{random} = N + 1 + 10 + 1 + 4 = N + 16
        max_bot_len = 36 - 16  # = 20, but use 18 for safety margin
        short_bot_id = bot_id[:18] if len(bot_id) > 18 else bot_id
        timestamp = int(datetime.now(timezone.utc).timestamp())
        random_suffix = uuid.uuid4().hex[:4]
        result = f"{short_bot_id}_{timestamp}_{random_suffix}"
        # Hard limit: truncate to 36 chars if somehow exceeded
        return result[:36]

    @staticmethod
    def get_bot_id_from_client_order_id(client_order_id: str) -> Optional[str]:
        """
        Extract bot ID from a client order ID.

        Args:
            client_order_id: Client order ID in format {bot_id}_{timestamp}_{random}

        Returns:
            Bot ID or None if format doesn't match
        """
        if not client_order_id:
            return None
        parts = client_order_id.rsplit("_", 2)
        if len(parts) >= 3:
            # Reconstruct bot_id from all parts except last 2
            return "_".join(client_order_id.rsplit("_", 2)[:-2]) or parts[0]
        return None

    # =========================================================================
    # Direct Order Methods (bypass queue - use with caution)
    # =========================================================================

    async def _direct_create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
        **kwargs,
    ) -> Order:
        """
        Create order directly without queue (for internal use).

        WARNING: This bypasses the order queue and should only be used
        when the caller already has exclusive access or for emergency orders.
        """
        async with self._order_lock:
            api = self._get_api(market)
            return await api.create_order(
                symbol=symbol,
                side=side,
                order_type=order_type,
                quantity=quantity,
                **kwargs,
            )

    async def _direct_cancel_order(
        self,
        symbol: str,
        order_id: str,
        market: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Cancel order directly without queue (for internal use).

        WARNING: This bypasses the order queue and should only be used
        when the caller already has exclusive access or for emergency orders.
        """
        async with self._order_lock:
            api = self._get_api(market)
            return await api.cancel_order(symbol, order_id=order_id)

    # =========================================================================
    # Order Methods (Thread-safe with queue)
    # =========================================================================

    async def create_order(
        self,
        symbol: str,
        side: OrderSide | str,
        order_type: OrderType | str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
        bot_id: str = "unknown",
        **kwargs,
    ) -> Order:
        """
        Create a new order (thread-safe, queued).

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/etc.)
            quantity: Order quantity
            market: SPOT or FUTURES
            bot_id: ID of the bot placing the order (for tracking)
            **kwargs: Additional order parameters (price, time_in_force, etc.)

        Returns:
            Created Order object
        """
        return await self._enqueue_order(
            operation="create",
            params={
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "market": market,
                "kwargs": kwargs,
            },
            bot_id=bot_id,
        )

    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market: MarketType = MarketType.SPOT,
        bot_id: str = "unknown",
    ) -> Order:
        """
        Cancel an active order (thread-safe, queued).

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            market: SPOT or FUTURES
            bot_id: ID of the bot cancelling the order (for tracking)

        Returns:
            Cancelled Order object
        """
        return await self._enqueue_order(
            operation="cancel",
            params={
                "symbol": symbol,
                "order_id": order_id,
                "market": market,
            },
            bot_id=bot_id,
            priority=1,  # Cancel orders have higher priority
        )

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

    async def get_open_orders_for_bot(
        self,
        bot_id: str,
        symbol: Optional[str] = None,
        market: MarketType = MarketType.SPOT,
    ) -> list[Order]:
        """
        Get open orders belonging to a specific bot.

        Filters orders by checking if their client_order_id starts with the bot_id.
        This ensures each bot only sees its own orders.

        Args:
            bot_id: Bot identifier to filter by
            symbol: Trading pair (optional, returns all if None)
            market: SPOT or FUTURES

        Returns:
            List of open Order objects belonging to the specified bot
        """
        all_orders = await self.get_open_orders(symbol, market)

        # Filter orders by bot_id prefix in client_order_id
        bot_orders = []
        for order in all_orders:
            if order.client_order_id:
                extracted_bot_id = self.get_bot_id_from_client_order_id(order.client_order_id)
                if extracted_bot_id == bot_id:
                    bot_orders.append(order)

        logger.debug(
            f"Filtered orders for {bot_id}: {len(bot_orders)}/{len(all_orders)} "
            f"(symbol={symbol}, market={market.value})"
        )
        return bot_orders

    # =========================================================================
    # Futures Order Wrapper Methods (for OrderExecutor Protocol, thread-safe)
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
        position_side: str = "BOTH",
        bot_id: str = "unknown",
    ) -> Order:
        """
        Create a futures order (thread-safe, queued).

        Args:
            symbol: Trading pair
            side: Order side (BUY/SELL)
            order_type: Order type (LIMIT/MARKET/STOP_MARKET/etc.)
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            stop_price: Stop price (for STOP orders)
            time_in_force: Time in force (GTC, IOC, etc.)
            reduce_only: Whether order is reduce-only
            position_side: Position side (BOTH for one-way, LONG/SHORT for hedge mode)
            bot_id: ID of the bot placing the order (for tracking)

        Returns:
            Created Order object
        """
        return await self._enqueue_order(
            operation="futures_create",
            params={
                "symbol": symbol,
                "side": side,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "stop_price": stop_price,
                "time_in_force": time_in_force,
                "reduce_only": reduce_only,
                "position_side": position_side,
            },
            bot_id=bot_id,
        )

    async def futures_cancel_order(
        self,
        symbol: str,
        order_id: str,
        bot_id: str = "unknown",
    ) -> Order:
        """
        Cancel a futures order (thread-safe, queued).

        Note: For Algo orders (STOP_MARKET, etc.), use futures_cancel_algo_order instead.

        Args:
            symbol: Trading pair
            order_id: Exchange order ID
            bot_id: ID of the bot cancelling the order (for tracking)

        Returns:
            Cancelled Order object
        """
        return await self._enqueue_order(
            operation="futures_cancel",
            params={
                "symbol": symbol,
                "order_id": order_id,
            },
            bot_id=bot_id,
            priority=1,  # Cancel orders have higher priority
        )

    async def futures_cancel_algo_order(
        self,
        symbol: str,
        algo_id: str,
        bot_id: str = "unknown",
    ) -> dict:
        """
        Cancel a futures Algo order (thread-safe, queued).

        Since 2025-12-09, conditional orders use Algo Order API.

        Args:
            symbol: Trading pair
            algo_id: Algo order ID (same as order_id from create_order)
            bot_id: ID of the bot cancelling the order (for tracking)

        Returns:
            Cancellation response
        """
        return await self._enqueue_order(
            operation="futures_cancel_algo",
            params={
                "symbol": symbol,
                "algo_id": algo_id,
            },
            bot_id=bot_id,
            priority=1,  # Cancel orders have higher priority
        )

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
    # Convenience Order Methods (thread-safe, uses queue)
    # =========================================================================

    async def market_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
        position_side: str = "BOTH",
        bot_id: str = "unknown",
    ) -> Order:
        """
        Place a market buy order (thread-safe, queued).

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            market: SPOT or FUTURES
            position_side: Position side (BOTH for one-way, LONG/SHORT for hedge mode)
            bot_id: ID of the bot placing the order

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=quantity,
            market=market,
            bot_id=bot_id,
            position_side=position_side,
        )

    async def market_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        market: MarketType = MarketType.SPOT,
        position_side: str = "BOTH",
        bot_id: str = "unknown",
    ) -> Order:
        """
        Place a market sell order (thread-safe, queued).

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            market: SPOT or FUTURES
            position_side: Position side (BOTH for one-way, LONG/SHORT for hedge mode)
            bot_id: ID of the bot placing the order

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=quantity,
            market=market,
            bot_id=bot_id,
            position_side=position_side,
        )

    async def limit_buy(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        market: MarketType = MarketType.SPOT,
        bot_id: str = "unknown",
    ) -> Order:
        """
        Place a limit buy order (thread-safe, queued).

        Args:
            symbol: Trading pair
            quantity: Quantity to buy
            price: Limit price
            market: SPOT or FUTURES
            bot_id: ID of the bot placing the order

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            market=market,
            bot_id=bot_id,
            price=price,
            time_in_force="GTC",
        )

    async def limit_sell(
        self,
        symbol: str,
        quantity: Decimal | str,
        price: Decimal | str,
        market: MarketType = MarketType.SPOT,
        bot_id: str = "unknown",
    ) -> Order:
        """
        Place a limit sell order (thread-safe, queued).

        Args:
            symbol: Trading pair
            quantity: Quantity to sell
            price: Limit price
            market: SPOT or FUTURES
            bot_id: ID of the bot placing the order

        Returns:
            Order object
        """
        return await self.create_order(
            symbol=symbol,
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            market=market,
            bot_id=bot_id,
            price=price,
            time_in_force="GTC",
        )

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

            # Capture callback at wrapper creation time to avoid closure issues
            # (if callback is changed later, this wrapper still uses the original)
            captured_callback = callback

            async def wrapper(data: dict):
                if captured_callback:
                    if asyncio.iscoroutinefunction(captured_callback):
                        await captured_callback(data)
                    else:
                        captured_callback(data)

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

                # Delete listen key (ensure cleanup even on failure)
                if self._user_data_listen_key:
                    try:
                        await self._spot.delete_listen_key(self._user_data_listen_key)
                    except Exception as e:
                        logger.warning(f"Failed to delete spot listen key: {e}")
                    finally:
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

                # Delete listen key (ensure cleanup even on failure)
                if self._futures_user_data_listen_key:
                    try:
                        await self._futures.delete_listen_key(self._futures_user_data_listen_key)
                    except Exception as e:
                        logger.warning(f"Failed to delete futures listen key: {e}")
                    finally:
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
        # Check cache with TTL
        current_time = time.time()
        if symbol in self._symbol_cache and market in self._symbol_cache[symbol]:
            cache_time = self._symbol_cache_time.get(symbol, {}).get(market, 0)
            if current_time - cache_time < self._symbol_cache_ttl:
                return self._symbol_cache[symbol][market]
            # Cache expired, will refresh below

        # Initialize cache dicts if needed
        if symbol not in self._symbol_cache:
            self._symbol_cache[symbol] = {}
        if symbol not in self._symbol_cache_time:
            self._symbol_cache_time[symbol] = {}

        # Fetch from API
        api = self._get_api(market)
        info = await api.get_exchange_info(symbol)

        if isinstance(info, SymbolInfo):
            self._symbol_cache[symbol][market] = info
            self._symbol_cache_time[symbol][market] = current_time
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
