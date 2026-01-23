"""
Stream Data Processor with Validation.

Integrates data validation and anomaly detection into real-time data streams.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union

from src.core import get_logger
from src.core.models import Kline, Ticker

from .validators import KlineValidator, PriceValidator, OrderBookValidator
from .cleaners import KlineCleaner, PriceCleaner
from .anomaly import AnomalyDetector, AnomalyRecord, AnomalySeverity

logger = get_logger(__name__)

T = TypeVar("T")


class StreamAction(str, Enum):
    """Action to take when anomaly is detected."""
    PASS = "pass"           # Pass data through (log only)
    WARN = "warn"           # Pass with warning
    FILTER = "filter"       # Filter out (don't pass to callback)
    CORRECT = "correct"     # Attempt to correct and pass


@dataclass
class StreamConfig:
    """
    Configuration for stream processor.

    Attributes:
        enable_validation: Enable data validation
        enable_anomaly_detection: Enable anomaly detection
        enable_auto_correction: Enable automatic data correction
        action_on_invalid: Action when validation fails
        action_on_anomaly: Action when anomaly detected
        anomaly_severity_threshold: Minimum severity to trigger action
        buffer_size: Size of data buffer for smoothing
        emit_raw_on_error: Emit raw data if processing fails
    """
    enable_validation: bool = True
    enable_anomaly_detection: bool = True
    enable_auto_correction: bool = True
    action_on_invalid: StreamAction = StreamAction.WARN
    action_on_anomaly: StreamAction = StreamAction.PASS
    anomaly_severity_threshold: AnomalySeverity = AnomalySeverity.HIGH
    buffer_size: int = 10
    emit_raw_on_error: bool = True


@dataclass
class ProcessingStats:
    """Statistics for stream processing."""
    total_received: int = 0
    total_processed: int = 0
    total_filtered: int = 0
    total_corrected: int = 0
    validation_failures: int = 0
    anomalies_detected: int = 0
    processing_errors: int = 0
    last_received: Optional[datetime] = None
    last_anomaly: Optional[AnomalyRecord] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_received": self.total_received,
            "total_processed": self.total_processed,
            "total_filtered": self.total_filtered,
            "total_corrected": self.total_corrected,
            "validation_failures": self.validation_failures,
            "anomalies_detected": self.anomalies_detected,
            "processing_errors": self.processing_errors,
            "last_received": self.last_received.isoformat() if self.last_received else None,
            "last_anomaly": self.last_anomaly.to_dict() if self.last_anomaly else None,
        }


class StreamProcessor:
    """
    Processes real-time market data streams with validation and anomaly detection.

    Integrates with WebSocket streams to provide:
    - Data validation (OHLC relationships, price sanity)
    - Anomaly detection (spikes, gaps, stale data)
    - Automatic correction (outlier smoothing, gap filling)

    Example:
        >>> processor = StreamProcessor()
        >>>
        >>> async def on_validated_kline(kline: Kline):
        ...     print(f"Validated: {kline.symbol} {kline.close}")
        ...
        >>> # Use as WebSocket callback wrapper
        >>> await ws.subscribe_kline(
        ...     "BTCUSDT", "1h",
        ...     processor.wrap_kline_callback(on_validated_kline)
        ... )
    """

    def __init__(
        self,
        config: Optional[StreamConfig] = None,
        anomaly_detector: Optional[AnomalyDetector] = None,
        on_anomaly: Optional[Callable[[AnomalyRecord], None]] = None,
        on_validation_error: Optional[Callable[[str, Any], None]] = None,
    ):
        """
        Initialize stream processor.

        Args:
            config: Stream processing configuration
            anomaly_detector: Custom anomaly detector (creates default if None)
            on_anomaly: Callback when anomaly is detected
            on_validation_error: Callback when validation fails
        """
        self.config = config or StreamConfig()

        # Initialize validators
        self._kline_validator = KlineValidator()
        self._price_validator = PriceValidator()
        self._orderbook_validator = OrderBookValidator()

        # Initialize cleaners
        self._kline_cleaner = KlineCleaner()
        self._price_cleaner = PriceCleaner()

        # Initialize anomaly detector
        self._anomaly_detector = anomaly_detector or AnomalyDetector(
            on_anomaly=self._handle_anomaly,
        )

        # Callbacks
        self._on_anomaly = on_anomaly
        self._on_validation_error = on_validation_error

        # State per symbol
        self._last_kline: Dict[str, Kline] = {}
        self._last_ticker: Dict[str, Ticker] = {}
        self._price_buffer: Dict[str, List[Decimal]] = {}

        # Statistics
        self.stats = ProcessingStats()

    # =========================================================================
    # Public Methods - Callback Wrappers
    # =========================================================================

    def wrap_kline_callback(
        self,
        callback: Callable[[Kline], Any],
    ) -> Callable[[Kline], Any]:
        """
        Wrap a kline callback with validation and anomaly detection.

        Args:
            callback: Original callback function

        Returns:
            Wrapped callback that validates data before passing through
        """
        async def wrapper(kline: Kline) -> None:
            self.stats.total_received += 1
            self.stats.last_received = datetime.now(timezone.utc)

            try:
                # Process kline
                processed = await self.process_kline(kline)

                if processed is not None:
                    self.stats.total_processed += 1
                    # Call original callback
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed)
                    else:
                        callback(processed)
                else:
                    self.stats.total_filtered += 1

            except Exception as e:
                self.stats.processing_errors += 1
                logger.error(f"Stream processing error for kline: {e}")

                # Emit raw data if configured
                if self.config.emit_raw_on_error:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(kline)
                    else:
                        callback(kline)

        return wrapper

    def wrap_ticker_callback(
        self,
        callback: Callable[[Ticker], Any],
    ) -> Callable[[Ticker], Any]:
        """
        Wrap a ticker callback with validation and anomaly detection.

        Args:
            callback: Original callback function

        Returns:
            Wrapped callback that validates data before passing through
        """
        async def wrapper(ticker: Ticker) -> None:
            self.stats.total_received += 1
            self.stats.last_received = datetime.now(timezone.utc)

            try:
                # Process ticker
                processed = await self.process_ticker(ticker)

                if processed is not None:
                    self.stats.total_processed += 1
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed)
                    else:
                        callback(processed)
                else:
                    self.stats.total_filtered += 1

            except Exception as e:
                self.stats.processing_errors += 1
                logger.error(f"Stream processing error for ticker: {e}")

                if self.config.emit_raw_on_error:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(ticker)
                    else:
                        callback(ticker)

        return wrapper

    def wrap_depth_callback(
        self,
        callback: Callable[[dict], Any],
    ) -> Callable[[dict], Any]:
        """
        Wrap a depth/orderbook callback with validation.

        Args:
            callback: Original callback function

        Returns:
            Wrapped callback that validates data before passing through
        """
        async def wrapper(depth: dict) -> None:
            self.stats.total_received += 1
            self.stats.last_received = datetime.now(timezone.utc)

            try:
                # Process depth
                processed = await self.process_depth(depth)

                if processed is not None:
                    self.stats.total_processed += 1
                    if asyncio.iscoroutinefunction(callback):
                        await callback(processed)
                    else:
                        callback(processed)
                else:
                    self.stats.total_filtered += 1

            except Exception as e:
                self.stats.processing_errors += 1
                logger.error(f"Stream processing error for depth: {e}")

                if self.config.emit_raw_on_error:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(depth)
                    else:
                        callback(depth)

        return wrapper

    # =========================================================================
    # Public Methods - Data Processing
    # =========================================================================

    async def process_kline(self, kline: Kline) -> Optional[Kline]:
        """
        Process and validate a kline.

        Args:
            kline: Input kline data

        Returns:
            Processed kline, or None if filtered out
        """
        symbol = kline.symbol
        processed = kline

        # Step 1: Validation
        if self.config.enable_validation:
            # Get previous close for gap detection
            prev_close = None
            if symbol in self._last_kline:
                prev_close = self._last_kline[symbol].close

            result = self._kline_validator.validate(
                open_price=kline.open,
                high=kline.high,
                low=kline.low,
                close=kline.close,
                volume=kline.volume,
                timestamp=kline.open_time or datetime.now(timezone.utc),
                prev_close=prev_close,
            )

            if not result.is_valid:
                self.stats.validation_failures += 1
                logger.warning(f"Kline validation failed for {symbol}: {result.errors}")

                if self._on_validation_error:
                    self._on_validation_error(symbol, result.errors)

                action = self.config.action_on_invalid

                if action == StreamAction.FILTER:
                    return None

                elif action == StreamAction.CORRECT and self.config.enable_auto_correction:
                    # Attempt correction
                    processed = self._correct_kline(kline)
                    self.stats.total_corrected += 1

        # Step 2: Anomaly Detection
        if self.config.enable_anomaly_detection:
            anomalies = self._detect_kline_anomalies(processed)

            if anomalies:
                self.stats.anomalies_detected += len(anomalies)

                # Check severity threshold
                max_severity = max(a.severity for a in anomalies)
                if self._severity_exceeds_threshold(max_severity):
                    action = self.config.action_on_anomaly

                    if action == StreamAction.FILTER:
                        return None

        # Step 3: Check for gaps
        if symbol in self._last_kline:
            last = self._last_kline[symbol]
            if kline.open_time and last.close_time:
                expected_interval = self._get_kline_interval_duration(kline.interval)
                if expected_interval:
                    self._anomaly_detector.check_data_gap(
                        symbol,
                        kline.open_time,
                        last.close_time,
                        expected_interval,
                    )

        # Update state
        self._last_kline[symbol] = processed

        return processed

    async def process_ticker(self, ticker: Ticker) -> Optional[Ticker]:
        """
        Process and validate a ticker.

        Args:
            ticker: Input ticker data

        Returns:
            Processed ticker, or None if filtered out
        """
        symbol = ticker.symbol
        processed = ticker

        # Step 1: Price Validation
        if self.config.enable_validation:
            result = self._price_validator.validate(
                symbol=symbol,
                price=ticker.price,
                timestamp=ticker.timestamp,
            )

            if not result.is_valid:
                self.stats.validation_failures += 1
                logger.warning(f"Ticker validation failed for {symbol}: {result.errors}")

                if self._on_validation_error:
                    self._on_validation_error(symbol, result.errors)

                if self.config.action_on_invalid == StreamAction.FILTER:
                    return None

        # Step 2: Anomaly Detection
        if self.config.enable_anomaly_detection:
            # Check price anomalies
            price_anomalies = self._anomaly_detector.check_price(
                symbol,
                ticker.price,
                ticker.timestamp,
            )

            # Check spread if bid/ask available
            if ticker.bid and ticker.ask:
                spread_anomalies = self._anomaly_detector.check_spread(
                    symbol,
                    ticker.bid,
                    ticker.ask,
                    ticker.timestamp,
                )
                price_anomalies.extend(spread_anomalies)

            # Check volume if available
            if ticker.volume_24h:
                volume_anomalies = self._anomaly_detector.check_volume(
                    symbol,
                    ticker.volume_24h,
                    ticker.timestamp,
                )
                price_anomalies.extend(volume_anomalies)

            if price_anomalies:
                self.stats.anomalies_detected += len(price_anomalies)

                max_severity = max(a.severity for a in price_anomalies)
                if self._severity_exceeds_threshold(max_severity):
                    if self.config.action_on_anomaly == StreamAction.FILTER:
                        return None

        # Step 3: Smoothing (if enabled)
        if self.config.enable_auto_correction:
            processed = self._smooth_ticker_price(ticker)

        # Update state
        self._last_ticker[symbol] = processed

        return processed

    async def process_depth(self, depth: dict) -> Optional[dict]:
        """
        Process and validate order book depth.

        Args:
            depth: Input depth data with 'bids' and 'asks'

        Returns:
            Processed depth, or None if filtered out
        """
        bids = depth.get("bids", [])
        asks = depth.get("asks", [])

        # Convert to proper format for validator if needed
        bids_tuples = []
        asks_tuples = []

        for bid in bids:
            if isinstance(bid, (list, tuple)) and len(bid) >= 2:
                bids_tuples.append((Decimal(str(bid[0])), Decimal(str(bid[1]))))
        for ask in asks:
            if isinstance(ask, (list, tuple)) and len(ask) >= 2:
                asks_tuples.append((Decimal(str(ask[0])), Decimal(str(ask[1]))))

        # Step 1: Validation
        if self.config.enable_validation:
            result = self._orderbook_validator.validate(bids_tuples, asks_tuples)

            if not result.is_valid:
                self.stats.validation_failures += 1
                logger.warning(f"Depth validation failed: {result.errors}")

                if self._on_validation_error:
                    self._on_validation_error("depth", result.errors)

                if self.config.action_on_invalid == StreamAction.FILTER:
                    return None

        # Step 2: Spread anomaly check
        if self.config.enable_anomaly_detection and bids and asks:
            best_bid = bids[0][0] if isinstance(bids[0], (list, tuple)) else bids[0]
            best_ask = asks[0][0] if isinstance(asks[0], (list, tuple)) else asks[0]

            if isinstance(best_bid, (int, float)):
                best_bid = Decimal(str(best_bid))
            if isinstance(best_ask, (int, float)):
                best_ask = Decimal(str(best_ask))

            spread_anomalies = self._anomaly_detector.check_spread(
                depth.get("symbol", "unknown"),
                best_bid,
                best_ask,
            )

            if spread_anomalies:
                self.stats.anomalies_detected += len(spread_anomalies)

                max_severity = max(a.severity for a in spread_anomalies)
                if self._severity_exceeds_threshold(max_severity):
                    if self.config.action_on_anomaly == StreamAction.FILTER:
                        return None

        return depth

    # =========================================================================
    # Private Methods - Correction
    # =========================================================================

    def _correct_kline(self, kline: Kline) -> Kline:
        """Apply corrections to invalid kline."""
        # Convert to dict format for cleaner
        kline_dict = {
            "open": str(kline.open),
            "high": str(kline.high),
            "low": str(kline.low),
            "close": str(kline.close),
        }

        # Use cleaner to fix OHLC relationships
        corrected_list, _ = self._kline_cleaner._correct_ohlc([kline_dict])
        corrected = corrected_list[0]

        return Kline(
            symbol=kline.symbol,
            interval=kline.interval,
            open_time=kline.open_time,
            open=Decimal(corrected["open"]),
            high=Decimal(corrected["high"]),
            low=Decimal(corrected["low"]),
            close=Decimal(corrected["close"]),
            volume=kline.volume if kline.volume >= Decimal("0") else Decimal("0"),
            close_time=kline.close_time,
            quote_volume=kline.quote_volume,
            trades_count=kline.trades_count,
        )

    def _smooth_ticker_price(self, ticker: Ticker) -> Ticker:
        """Apply smoothing to ticker price if it's an outlier."""
        symbol = ticker.symbol
        price = ticker.price

        # Initialize buffer
        if symbol not in self._price_buffer:
            self._price_buffer[symbol] = []

        buffer = self._price_buffer[symbol]

        # Check if price is outlier compared to recent history
        if len(buffer) >= 5:
            avg = sum(buffer) / len(buffer)
            if avg > Decimal("0"):
                deviation = abs(price - avg) / avg
                # If more than 10% deviation, use smoothed value
                if deviation > Decimal("0.10"):
                    smoothed = (price + avg) / 2
                    logger.debug(
                        f"Smoothed {symbol} price: {price} -> {smoothed} "
                        f"(avg: {avg}, dev: {deviation:.2%})"
                    )
                    ticker = Ticker(
                        symbol=ticker.symbol,
                        price=smoothed,
                        bid=ticker.bid,
                        ask=ticker.ask,
                        high_24h=ticker.high_24h,
                        low_24h=ticker.low_24h,
                        volume_24h=ticker.volume_24h,
                        change_24h=ticker.change_24h,
                        timestamp=ticker.timestamp,
                    )
                    self.stats.total_corrected += 1

        # Update buffer
        buffer.append(price)
        if len(buffer) > self.config.buffer_size:
            buffer.pop(0)

        return ticker

    # =========================================================================
    # Private Methods - Anomaly Detection
    # =========================================================================

    def _detect_kline_anomalies(self, kline: Kline) -> List[AnomalyRecord]:
        """Detect anomalies in kline data."""
        anomalies: List[AnomalyRecord] = []
        symbol = kline.symbol

        # Check close price
        price_anomalies = self._anomaly_detector.check_price(
            symbol,
            kline.close,
            kline.close_time,
        )
        anomalies.extend(price_anomalies)

        # Check volume
        if kline.volume > Decimal("0"):
            volume_anomalies = self._anomaly_detector.check_volume(
                symbol,
                kline.volume,
                kline.close_time,
            )
            anomalies.extend(volume_anomalies)

        # Check for abnormal candle (wick too long)
        if kline.high > Decimal("0") and kline.low > Decimal("0"):
            body = abs(kline.close - kline.open)
            total_range = kline.high - kline.low

            if total_range > Decimal("0"):
                body_ratio = body / total_range
                # If body is less than 10% of total range, might be manipulation
                if body_ratio < Decimal("0.10") and total_range / kline.close > Decimal("0.05"):
                    anomalies.append(AnomalyRecord(
                        timestamp=kline.close_time or datetime.now(timezone.utc),
                        anomaly_type="manipulation",
                        severity=AnomalySeverity.MEDIUM,
                        symbol=symbol,
                        description=f"Suspicious candle: body={body_ratio:.1%} of range",
                        value=float(body_ratio),
                        threshold=0.10,
                        metadata={
                            "open": float(kline.open),
                            "high": float(kline.high),
                            "low": float(kline.low),
                            "close": float(kline.close),
                        },
                    ))

        return anomalies

    def _handle_anomaly(self, anomaly: AnomalyRecord) -> None:
        """Handle detected anomaly."""
        self.stats.last_anomaly = anomaly

        if self._on_anomaly:
            self._on_anomaly(anomaly)

        # Log based on severity
        if anomaly.severity == AnomalySeverity.CRITICAL:
            logger.error(f"CRITICAL anomaly: {anomaly.description}")
        elif anomaly.severity == AnomalySeverity.HIGH:
            logger.warning(f"HIGH anomaly: {anomaly.description}")
        else:
            logger.info(f"Anomaly: {anomaly.description}")

    def _severity_exceeds_threshold(self, severity: AnomalySeverity) -> bool:
        """Check if severity exceeds configured threshold."""
        severity_order = [
            AnomalySeverity.LOW,
            AnomalySeverity.MEDIUM,
            AnomalySeverity.HIGH,
            AnomalySeverity.CRITICAL,
        ]
        return severity_order.index(severity) >= severity_order.index(
            self.config.anomaly_severity_threshold
        )

    # =========================================================================
    # Private Methods - Utilities
    # =========================================================================

    def _get_kline_interval_duration(self, interval) -> Optional[timedelta]:
        """Get expected duration for kline interval."""
        interval_str = interval.value if hasattr(interval, "value") else str(interval)

        # Parse interval string (e.g., "1m", "5m", "1h", "1d")
        unit = interval_str[-1]
        try:
            value = int(interval_str[:-1])
        except ValueError:
            return None

        if unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "w":
            return timedelta(weeks=value)

        return None

    # =========================================================================
    # Public Methods - Status
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "processing": self.stats.to_dict(),
            "anomaly_detection": self._anomaly_detector.get_stats(),
        }

    def reset_stats(self) -> None:
        """Reset all statistics."""
        self.stats = ProcessingStats()
        self._anomaly_detector.reset_stats()

    def clear_state(self, symbol: Optional[str] = None) -> None:
        """
        Clear internal state.

        Args:
            symbol: Symbol to clear, or None for all
        """
        if symbol:
            self._last_kline.pop(symbol, None)
            self._last_ticker.pop(symbol, None)
            self._price_buffer.pop(symbol, None)
            self._anomaly_detector.clear_history(symbol)
        else:
            self._last_kline.clear()
            self._last_ticker.clear()
            self._price_buffer.clear()
            self._anomaly_detector.clear_history()


class ValidatedWebSocketWrapper:
    """
    Wrapper that adds validation to any WebSocket client.

    Provides a convenient way to add data validation to existing
    WebSocket implementations without modifying them.

    Example:
        >>> from src.exchange.binance.websocket import BinanceWebSocket
        >>>
        >>> ws = BinanceWebSocket()
        >>> validated_ws = ValidatedWebSocketWrapper(ws)
        >>>
        >>> # Subscribe with automatic validation
        >>> await validated_ws.subscribe_kline("BTCUSDT", "1h", my_callback)
    """

    def __init__(
        self,
        websocket: Any,
        processor: Optional[StreamProcessor] = None,
        config: Optional[StreamConfig] = None,
    ):
        """
        Initialize validated WebSocket wrapper.

        Args:
            websocket: Underlying WebSocket client
            processor: Stream processor to use
            config: Stream configuration
        """
        self._ws = websocket
        self._processor = processor or StreamProcessor(config)

    @property
    def processor(self) -> StreamProcessor:
        """Get stream processor."""
        return self._processor

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._ws.is_connected

    async def connect(self) -> bool:
        """Connect WebSocket."""
        return await self._ws.connect()

    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        await self._ws.disconnect()

    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to validated kline stream."""
        wrapped = self._processor.wrap_kline_callback(callback)
        return await self._ws.subscribe_kline(symbol, interval, wrapped)

    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to validated ticker stream."""
        wrapped = self._processor.wrap_ticker_callback(callback)
        return await self._ws.subscribe_ticker(symbol, wrapped)

    async def subscribe_depth(
        self,
        symbol: str,
        callback: Callable,
        level: int = 5,
    ) -> bool:
        """Subscribe to validated depth stream."""
        wrapped = self._processor.wrap_depth_callback(callback)
        return await self._ws.subscribe_depth(symbol, wrapped, level)

    async def subscribe_trade(
        self,
        symbol: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to trade stream (no validation, pass through)."""
        return await self._ws.subscribe_trade(symbol, callback)

    async def subscribe_book_ticker(
        self,
        symbol: str,
        callback: Callable,
    ) -> bool:
        """Subscribe to book ticker stream."""
        # Wrap with depth validation since it has bid/ask
        wrapped = self._processor.wrap_depth_callback(
            lambda data: callback(data)
        )
        return await self._ws.subscribe_book_ticker(symbol, wrapped)

    def get_stats(self) -> Dict[str, Any]:
        """Get validation statistics."""
        return self._processor.get_stats()

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
