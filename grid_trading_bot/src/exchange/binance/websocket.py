"""
Binance WebSocket client for real-time market data.

Supports both Spot and Futures market streams with automatic reconnection.
"""

import asyncio
import json
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from src.core import get_logger
from src.core.models import (
    Kline,
    KlineInterval,
    MarketType,
    Ticker,
)
from src.core.utils import timestamp_to_datetime

from .constants import (
    FUTURES_WS_TESTNET_URL,
    FUTURES_WS_URL,
    SPOT_WS_TESTNET_URL,
    SPOT_WS_URL,
)

logger = get_logger(__name__)


class BinanceWebSocket:
    """
    Binance WebSocket client for real-time market data streaming.

    Supports:
    - Spot and Futures markets
    - Multiple stream subscriptions
    - Automatic reconnection with exponential backoff
    - Heartbeat/ping to maintain connection

    Example:
        >>> async def on_ticker(ticker: Ticker):
        ...     print(f"{ticker.symbol}: {ticker.price}")
        ...
        >>> ws = BinanceWebSocket(market_type=MarketType.SPOT)
        >>> await ws.connect()
        >>> await ws.subscribe_ticker("BTCUSDT", on_ticker)
        >>> # Keep running...
        >>> await ws.disconnect()
    """

    def __init__(
        self,
        market_type: MarketType = MarketType.SPOT,
        testnet: bool = False,
        on_message: Optional[Callable[[dict], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_close: Optional[Callable[[], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ):
        """
        Initialize BinanceWebSocket.

        Args:
            market_type: SPOT or FUTURES
            testnet: Use testnet URLs if True
            on_message: Global message callback (raw dict)
            on_error: Error callback
            on_close: Connection close callback
            on_reconnect: Callback invoked after successful reconnection
        """
        self._market_type = market_type
        self._testnet = testnet

        # Select WebSocket URL
        if market_type == MarketType.SPOT:
            self._base_url = SPOT_WS_TESTNET_URL if testnet else SPOT_WS_URL
        else:
            self._base_url = FUTURES_WS_TESTNET_URL if testnet else FUTURES_WS_URL

        # Callbacks
        self._on_message = on_message
        self._on_error = on_error
        self._on_close = on_close
        self._on_reconnect = on_reconnect

        # Connection state
        self._ws: Optional[ClientConnection] = None
        self._connected = False
        self._running = False

        # Subscription tracking
        self._subscriptions: dict[str, Callable] = {}  # stream -> callback
        self._subscriptions_lock: Optional[asyncio.Lock] = None  # Lazy init to avoid event loop issues
        self._request_id = 0

        # Reconnection settings
        self.reconnect_delay = 5  # Initial delay in seconds
        self.max_reconnect_delay = 60  # Maximum delay
        self.max_reconnect_attempts = 10  # Maximum attempts
        self._reconnect_attempts = 0
        self._current_delay = self.reconnect_delay
        self._intentional_close = False  # Flag to prevent reconnection on intentional close

        # Heartbeat settings
        self._heartbeat_interval = 8  # seconds (ping every 8s)
        self._heartbeat_timeout = 10  # seconds (detect dead connection faster)
        # NOTE: asyncio single-thread safe, no lock needed for timing counters
        self._last_pong_time: Optional[datetime] = None

        # Timeout tracking (reduce warning frequency)
        self._consecutive_timeouts = 0
        self._max_silent_timeouts = 3  # Only warn after this many consecutive timeouts

        # Background tasks
        self._message_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None  # Track reconnect to prevent task leak

        # Message deduplication (track recent message timestamps)
        self._recent_msg_ids: dict[str, float] = {}  # msg_id -> timestamp
        self._dedup_window: float = 5.0  # seconds to keep message IDs
        self._dedup_lock: Optional[asyncio.Lock] = None  # Lazy init with event loop

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def is_connected(self) -> bool:
        """Check if WebSocket is connected."""
        if not self._connected or self._ws is None:
            return False
        # Check connection state - websockets 14+ uses state attribute
        try:
            from websockets.protocol import State
            return self._ws.state == State.OPEN
        except (ImportError, AttributeError):
            # Fallback for older versions or different API
            return self._connected

    @property
    def market_type(self) -> MarketType:
        """Get market type."""
        return self._market_type

    # =========================================================================
    # Connection Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful
        """
        if self.is_connected:
            logger.debug("Already connected")
            return True

        try:
            logger.info(f"Connecting to {self._base_url}")
            self._ws = await websockets.connect(
                self._base_url,
                ping_interval=None,  # We handle our own heartbeat
                ping_timeout=None,
            )
            self._connected = True
            self._running = True
            self._reconnect_attempts = 0
            self._current_delay = self.reconnect_delay
            self._last_pong_time = datetime.now(timezone.utc)
            self._consecutive_timeouts = 0  # Reset timeout counter on new connection

            # Initialize locks (in event loop context)
            if self._subscriptions_lock is None:
                self._subscriptions_lock = asyncio.Lock()
            if self._dedup_lock is None:
                self._dedup_lock = asyncio.Lock()

            # Start background tasks
            self._message_task = asyncio.create_task(self._message_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            logger.info("WebSocket connected")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self._connected = False
            if self._on_error:
                self._on_error(e)
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection gracefully."""
        logger.debug("Disconnecting WebSocket")

        # Set flags FIRST to prevent reconnection
        self._intentional_close = True
        self._running = False
        self._connected = False

        # Cancel background tasks
        if self._message_task and not self._message_task.done():
            self._message_task.cancel()
            try:
                await self._message_task
            except asyncio.CancelledError:
                pass
            self._message_task = None

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Cancel any pending reconnect task
        if self._reconnect_task and not self._reconnect_task.done():
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass
            self._reconnect_task = None

        # Close WebSocket
        if self._ws:
            try:
                await self._ws.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self._ws = None

        logger.debug("WebSocket disconnected")

        if self._on_close:
            self._on_close()

    async def reconnect(self) -> bool:
        """
        Attempt to reconnect with exponential backoff.

        Uses iterative retry loop (not recursive) to avoid stack overflow.

        Returns:
            True if reconnection successful
        """
        while True:
            # Don't reconnect if intentionally closed
            if self._intentional_close:
                logger.debug("Skipping reconnection - intentional close")
                return False

            if self._reconnect_attempts >= self.max_reconnect_attempts:
                logger.error(f"Max reconnection attempts ({self.max_reconnect_attempts}) reached")
                if self._on_error:
                    self._on_error(Exception("Max reconnection attempts reached"))
                return False

            self._reconnect_attempts += 1
            logger.info(
                f"Reconnection attempt {self._reconnect_attempts}/{self.max_reconnect_attempts} "
                f"in {self._current_delay}s"
            )

            await asyncio.sleep(self._current_delay)

            # Exponential backoff
            self._current_delay = min(self._current_delay * 2, self.max_reconnect_delay)

            # Attempt connection
            if not await self.connect():
                continue  # Retry with next attempt
            # Resubscribe to all streams with lock protection for entire operation
            # This prevents race conditions where subscriptions are modified during resubscription
            if self._subscriptions_lock:
                async with self._subscriptions_lock:
                    if self._subscriptions:
                        streams_to_resubscribe = list(self._subscriptions.keys())
                        if streams_to_resubscribe:
                            logger.info(f"Resubscribing to {len(streams_to_resubscribe)} streams")
                            # Retry subscription up to 3 times (within lock to ensure consistency)
                            for attempt in range(3):
                                success = await self._send_subscribe(streams_to_resubscribe)
                                if success:
                                    logger.info(f"Successfully resubscribed to {len(streams_to_resubscribe)} streams")
                                    break
                                logger.warning(f"Resubscription attempt {attempt + 1}/3 failed")
                                await asyncio.sleep(1)
                            else:
                                logger.error("Failed to resubscribe after 3 attempts")
            elif self._subscriptions:
                # Ensure lock exists (connect() initializes it, but be safe)
                if self._subscriptions_lock is None:
                    self._subscriptions_lock = asyncio.Lock()
                async with self._subscriptions_lock:
                    streams_to_resubscribe = list(self._subscriptions.keys())
                    if streams_to_resubscribe:
                        logger.info(f"Resubscribing to {len(streams_to_resubscribe)} streams")
                        for attempt in range(3):
                            success = await self._send_subscribe(streams_to_resubscribe)
                            if success:
                                logger.info(f"Successfully resubscribed to {len(streams_to_resubscribe)} streams")
                                break
                            logger.warning(f"Resubscription attempt {attempt + 1}/3 failed")
                            await asyncio.sleep(1)
                        else:
                            logger.error("Failed to resubscribe after 3 attempts")

            # Notify caller about reconnection (e.g., to trigger position resync)
            logger.warning(
                "WebSocket reconnected — messages may have been missed during downtime, "
                "caller should resync state"
            )
            if self._on_reconnect:
                try:
                    result = self._on_reconnect()
                    if asyncio.iscoroutine(result):
                        await result
                except Exception as e:
                    logger.error(f"on_reconnect callback error: {e}")

            return True

    # =========================================================================
    # Subscription Management
    # =========================================================================

    async def subscribe(self, streams: list[str], callback: Callable) -> bool:
        """
        Subscribe to streams.

        Args:
            streams: List of stream names (e.g., ["btcusdt@ticker"])
            callback: Callback function for messages

        Returns:
            True if subscription successful
        """
        if not self.is_connected:
            logger.error("Not connected, cannot subscribe")
            return False

        # Store callbacks with lock protection
        if self._subscriptions_lock:
            async with self._subscriptions_lock:
                for stream in streams:
                    self._subscriptions[stream.lower()] = callback
        else:
            for stream in streams:
                self._subscriptions[stream.lower()] = callback

        return await self._send_subscribe(streams)

    async def unsubscribe(self, streams: list[str]) -> bool:
        """
        Unsubscribe from streams.

        Args:
            streams: List of stream names to unsubscribe

        Returns:
            True if unsubscription successful
        """
        if not self.is_connected:
            logger.error("Not connected, cannot unsubscribe")
            return False

        # Remove callbacks with lock protection
        if self._subscriptions_lock:
            async with self._subscriptions_lock:
                for stream in streams:
                    self._subscriptions.pop(stream.lower(), None)
        else:
            for stream in streams:
                self._subscriptions.pop(stream.lower(), None)

        return await self._send_unsubscribe(streams)

    async def _send_subscribe(self, streams: list[str]) -> bool:
        """Send subscribe message to WebSocket."""
        self._request_id += 1
        message = {
            "method": "SUBSCRIBE",
            "params": [s.lower() for s in streams],
            "id": self._request_id,
        }

        try:
            await self._ws.send(json.dumps(message))
            logger.debug(f"Subscribed to: {streams}")
            return True
        except Exception as e:
            logger.error(f"Subscribe failed: {e}")
            return False

    async def _send_unsubscribe(self, streams: list[str]) -> bool:
        """Send unsubscribe message to WebSocket."""
        self._request_id += 1
        message = {
            "method": "UNSUBSCRIBE",
            "params": [s.lower() for s in streams],
            "id": self._request_id,
        }

        try:
            await self._ws.send(json.dumps(message))
            logger.debug(f"Unsubscribed from: {streams}")
            return True
        except Exception as e:
            logger.error(f"Unsubscribe failed: {e}")
            return False

    # =========================================================================
    # Convenience Subscription Methods
    # =========================================================================

    async def subscribe_ticker(self, symbol: str, callback: Callable[[Ticker], None]) -> bool:
        """
        Subscribe to 24hr ticker stream.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            callback: Callback receiving Ticker objects

        Returns:
            True if successful
        """
        stream = f"{symbol.lower()}@ticker"

        async def wrapper(data: dict):
            ticker = self._parse_ticker(data)
            await self._invoke_callback(callback, ticker)

        return await self.subscribe([stream], wrapper)

    async def subscribe_kline(
        self,
        symbol: str,
        interval: KlineInterval | str,
        callback: Callable[[Kline], None],
    ) -> bool:
        """
        Subscribe to kline/candlestick stream.

        Args:
            symbol: Trading pair
            interval: Kline interval (e.g., KlineInterval.h1 or "1h")
            callback: Callback receiving Kline objects

        Returns:
            True if successful
        """
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval
        stream = f"{symbol.lower()}@kline_{interval_str}"

        async def wrapper(data: dict):
            # Only invoke callback when kline is closed
            k = data.get("k", {})
            if not k.get("x", False):
                return  # Skip unclosed klines
            kline = self._parse_kline(data)
            await self._invoke_callback(callback, kline)

        return await self.subscribe([stream], wrapper)

    async def unsubscribe_kline(
        self,
        symbol: str,
        interval: KlineInterval | str,
    ) -> bool:
        """
        Unsubscribe from kline/candlestick stream.

        Args:
            symbol: Trading pair
            interval: Kline interval (e.g., KlineInterval.h1 or "1h")

        Returns:
            True if successful
        """
        interval_str = interval.value if isinstance(interval, KlineInterval) else interval
        stream = f"{symbol.lower()}@kline_{interval_str}"
        return await self.unsubscribe([stream])

    async def subscribe_depth(
        self,
        symbol: str,
        callback: Callable[[dict], None],
        level: int = 5,
    ) -> bool:
        """
        Subscribe to order book depth stream.

        Args:
            symbol: Trading pair
            callback: Callback receiving depth dict
            level: Depth level (5, 10, or 20)

        Returns:
            True if successful
        """
        stream = f"{symbol.lower()}@depth{level}"

        async def wrapper(data: dict):
            depth = self._parse_depth(data)
            await self._invoke_callback(callback, depth)

        return await self.subscribe([stream], wrapper)

    async def subscribe_trade(self, symbol: str, callback: Callable[[dict], None]) -> bool:
        """
        Subscribe to aggregate trade stream.

        Args:
            symbol: Trading pair
            callback: Callback receiving trade dict

        Returns:
            True if successful
        """
        stream = f"{symbol.lower()}@aggTrade"

        async def wrapper(data: dict):
            trade = self._parse_agg_trade(data)
            await self._invoke_callback(callback, trade)

        return await self.subscribe([stream], wrapper)

    async def subscribe_book_ticker(self, symbol: str, callback: Callable[[dict], None]) -> bool:
        """
        Subscribe to best bid/ask stream.

        Args:
            symbol: Trading pair
            callback: Callback receiving book ticker dict

        Returns:
            True if successful
        """
        stream = f"{symbol.lower()}@bookTicker"

        async def wrapper(data: dict):
            book_ticker = self._parse_book_ticker(data)
            await self._invoke_callback(callback, book_ticker)

        return await self.subscribe([stream], wrapper)

    async def subscribe_mark_price(self, symbol: str, callback: Callable[[dict], None]) -> bool:
        """
        Subscribe to mark price stream (Futures only).

        Args:
            symbol: Trading pair
            callback: Callback receiving mark price dict

        Returns:
            True if successful
        """
        if self._market_type != MarketType.FUTURES:
            logger.warning("Mark price stream is only available for futures")

        stream = f"{symbol.lower()}@markPrice"

        async def wrapper(data: dict):
            mark_price = self._parse_mark_price(data)
            await self._invoke_callback(callback, mark_price)

        return await self.subscribe([stream], wrapper)

    # =========================================================================
    # Message Processing
    # =========================================================================

    async def _message_loop(self) -> None:
        """Main message receiving loop."""
        try:
            while self._running and self._ws:
                try:
                    message = await asyncio.wait_for(
                        self._ws.recv(),
                        timeout=self._heartbeat_timeout,
                    )
                    # Reset timeout counter on successful message
                    self._consecutive_timeouts = 0
                    await self._handle_message(message)

                except asyncio.TimeoutError:
                    self._consecutive_timeouts += 1
                    # Only warn after multiple consecutive timeouts
                    if self._consecutive_timeouts > self._max_silent_timeouts:
                        logger.warning(
                            f"Message timeout ({self._consecutive_timeouts}x), checking connection"
                        )
                    else:
                        logger.debug(
                            f"Message timeout ({self._consecutive_timeouts}/{self._max_silent_timeouts})"
                        )
                    if not await self._check_connection():
                        break

                except websockets.ConnectionClosed as e:
                    logger.warning(f"Connection closed: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug("Message loop cancelled")
            raise

        except Exception as e:
            logger.error(f"Message loop error: {e}")
            if self._on_error:
                self._on_error(e)

        finally:
            self._connected = False
            # Only attempt reconnection if not intentionally closed
            if self._running and not self._intentional_close:
                # Cancel any existing reconnect task to prevent task accumulation
                if self._reconnect_task and not self._reconnect_task.done():
                    self._reconnect_task.cancel()
                    try:
                        await self._reconnect_task
                    except asyncio.CancelledError:
                        pass
                # Track the reconnect task to prevent leak
                self._reconnect_task = asyncio.create_task(self.reconnect())

    async def _handle_message(self, raw_message: str) -> None:
        """
        Handle incoming WebSocket message.

        Args:
            raw_message: Raw JSON string
        """
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return

        # Handle pong response
        if isinstance(data, dict) and data.get("result") is None and "id" in data:
            # This is a response to subscribe/unsubscribe
            logger.debug(f"Received response for request {data.get('id')}")
            return

        # Update pong time for any valid message
        now = datetime.now(timezone.utc)
        self._last_pong_time = now

        # Message deduplication - create unique ID from event time + stream
        if isinstance(data, dict):
            event_time = data.get("E", 0)  # Event time from Binance
            stream = data.get("stream") or self._get_stream_from_data(data) or ""
            msg_id = f"{stream}:{event_time}"

            current_time = now.timestamp()

            # Lock-protected dedup check and update
            if self._dedup_lock:
                async with self._dedup_lock:
                    if msg_id in self._recent_msg_ids:
                        logger.debug(f"Duplicate message filtered: {msg_id}")
                        return
                    self._recent_msg_ids[msg_id] = current_time

                    # Cleanup old message IDs periodically
                    should_cleanup = (
                        len(self._recent_msg_ids) > 100 or
                        (hasattr(self, '_last_dedup_cleanup') and
                         current_time - self._last_dedup_cleanup > 60)
                    )
                    if should_cleanup or not hasattr(self, '_last_dedup_cleanup'):
                        cutoff = current_time - self._dedup_window
                        self._recent_msg_ids = {
                            k: v for k, v in self._recent_msg_ids.items() if v > cutoff
                        }
                        self._last_dedup_cleanup = current_time
            else:
                # Fallback without lock (before connect() initializes it)
                if msg_id in self._recent_msg_ids:
                    logger.debug(f"Duplicate message filtered: {msg_id}")
                    return
                self._recent_msg_ids[msg_id] = current_time

        # Call global message handler if set
        if self._on_message:
            self._on_message(data)

        # Determine stream type and call appropriate callback
        stream = data.get("stream") or self._get_stream_from_data(data)

        # Get callback with lock protection (minimize lock time)
        callback = None
        if stream:
            if self._subscriptions_lock:
                async with self._subscriptions_lock:
                    callback = self._subscriptions.get(stream)
            else:
                callback = self._subscriptions.get(stream)

        if callback:
            try:
                await callback(data)
            except Exception as e:
                logger.error(f"Callback error for {stream}: {e}")

    def _get_stream_from_data(self, data: dict) -> Optional[str]:
        """Extract stream name from message data."""
        event_type = data.get("e")
        symbol = data.get("s", "").lower()

        if not event_type or not symbol:
            return None

        # Map event types to stream suffixes
        stream_map = {
            "24hrTicker": f"{symbol}@ticker",
            "kline": f"{symbol}@kline_{data.get('k', {}).get('i', '')}",
            "depthUpdate": f"{symbol}@depth",
            "aggTrade": f"{symbol}@aggTrade",
            "bookTicker": f"{symbol}@bookTicker",
            "markPriceUpdate": f"{symbol}@markPrice",
        }

        return stream_map.get(event_type)

    async def _invoke_callback(self, callback: Callable, data: Any) -> None:
        """Invoke callback, handling both sync and async callbacks."""
        if asyncio.iscoroutinefunction(callback):
            await callback(data)
        else:
            callback(data)

    # =========================================================================
    # Data Parsing
    # =========================================================================

    def _parse_ticker(self, data: dict) -> Ticker:
        """Parse WebSocket ticker message to Ticker model."""
        return Ticker(
            symbol=data.get("s", ""),
            price=Decimal(str(data.get("c", "0"))),
            bid=Decimal(str(data.get("b", "0"))),
            ask=Decimal(str(data.get("a", "0"))),
            high_24h=Decimal(str(data.get("h", "0"))),
            low_24h=Decimal(str(data.get("l", "0"))),
            volume_24h=Decimal(str(data.get("v", "0"))),
            change_24h=Decimal(str(data.get("P", "0"))),
            timestamp=timestamp_to_datetime(data.get("E", 0)),
        )

    def _parse_kline(self, data: dict) -> Kline:
        """Parse WebSocket kline message to Kline model."""
        k = data.get("k", {})
        return Kline(
            symbol=k.get("s", ""),
            interval=KlineInterval(k.get("i", "1m")),
            open_time=timestamp_to_datetime(k.get("t", 0)),
            open=Decimal(str(k.get("o", "0"))),
            high=Decimal(str(k.get("h", "0"))),
            low=Decimal(str(k.get("l", "0"))),
            close=Decimal(str(k.get("c", "0"))),
            volume=Decimal(str(k.get("v", "0"))),
            close_time=timestamp_to_datetime(k.get("T", 0)),
            quote_volume=Decimal(str(k.get("q", "0"))),
            trades_count=k.get("n", 0),
        )

    def _parse_depth(self, data: dict) -> dict:
        """Parse WebSocket depth message."""
        return {
            "bids": [[Decimal(p), Decimal(q)] for p, q in data.get("bids", data.get("b", []))],
            "asks": [[Decimal(p), Decimal(q)] for p, q in data.get("asks", data.get("a", []))],
            "timestamp": timestamp_to_datetime(data.get("E", 0)),
        }

    def _parse_agg_trade(self, data: dict) -> dict:
        """Parse WebSocket aggregate trade message."""
        return {
            "symbol": data.get("s", ""),
            "trade_id": data.get("a"),
            "price": Decimal(str(data.get("p", "0"))),
            "quantity": Decimal(str(data.get("q", "0"))),
            "first_trade_id": data.get("f"),
            "last_trade_id": data.get("l"),
            "timestamp": timestamp_to_datetime(data.get("T", 0)),
            "is_buyer_maker": data.get("m", False),
        }

    def _parse_book_ticker(self, data: dict) -> dict:
        """Parse WebSocket book ticker message."""
        return {
            "symbol": data.get("s", ""),
            "bid_price": Decimal(str(data.get("b", "0"))),
            "bid_qty": Decimal(str(data.get("B", "0"))),
            "ask_price": Decimal(str(data.get("a", "0"))),
            "ask_qty": Decimal(str(data.get("A", "0"))),
            "timestamp": timestamp_to_datetime(data.get("E", data.get("u", 0))),
        }

    def _parse_mark_price(self, data: dict) -> dict:
        """Parse WebSocket mark price message (Futures)."""
        return {
            "symbol": data.get("s", ""),
            "mark_price": Decimal(str(data.get("p", "0"))),
            "index_price": Decimal(str(data.get("i", "0"))),
            "funding_rate": Decimal(str(data.get("r", "0"))),
            "next_funding_time": timestamp_to_datetime(data.get("T", 0)),
            "timestamp": timestamp_to_datetime(data.get("E", 0)),
        }

    # =========================================================================
    # Heartbeat Management
    # =========================================================================

    async def _heartbeat_loop(self) -> None:
        """Send periodic pings to maintain connection."""
        try:
            while self._running and self._ws:
                await asyncio.sleep(self._heartbeat_interval)

                if not self.is_connected:
                    break

                try:
                    # Send ping frame
                    pong_waiter = await self._ws.ping()
                    await asyncio.wait_for(pong_waiter, timeout=10)
                    self._last_pong_time = datetime.now(timezone.utc)
                    logger.debug("Heartbeat ping/pong successful")

                except asyncio.TimeoutError:
                    logger.warning("Heartbeat pong timeout")
                    # Check if we should reconnect
                    if not await self._check_connection():
                        break

                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    break

        except asyncio.CancelledError:
            logger.debug("Heartbeat loop cancelled")
            raise

    async def _check_connection(self) -> bool:
        """
        Check if connection is still alive.

        Returns:
            True if connection is healthy
        """
        if not self._last_pong_time:
            return False

        elapsed = (datetime.now(timezone.utc) - self._last_pong_time).total_seconds()

        if elapsed > self._heartbeat_timeout:
            logger.warning(f"No response for {elapsed:.0f}s, connection may be dead")
            return False

        return True

    # =========================================================================
    # Context Manager Support
    # =========================================================================

    async def __aenter__(self) -> "BinanceWebSocket":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()
