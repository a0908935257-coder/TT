"""
Stream Processor Unit Tests.

Tests for real-time data stream validation and processing.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import Kline, KlineInterval, Ticker
from src.data.validation import (
    StreamProcessor,
    StreamConfig,
    StreamAction,
    ValidatedWebSocketWrapper,
    AnomalyRecord,
    AnomalyType,
)
from src.data.validation.anomaly import AnomalySeverity


class TestStreamConfig:
    """Test StreamConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamConfig()

        assert config.enable_validation is True
        assert config.enable_anomaly_detection is True
        assert config.enable_auto_correction is True
        assert config.action_on_invalid == StreamAction.WARN
        assert config.action_on_anomaly == StreamAction.PASS
        assert config.buffer_size == 10

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamConfig(
            enable_validation=False,
            action_on_invalid=StreamAction.FILTER,
            buffer_size=20,
        )

        assert config.enable_validation is False
        assert config.action_on_invalid == StreamAction.FILTER
        assert config.buffer_size == 20


class TestStreamProcessor:
    """Test StreamProcessor."""

    @pytest.fixture
    def processor(self):
        """Create processor instance."""
        return StreamProcessor()

    @pytest.fixture
    def valid_kline(self):
        """Create valid kline."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
            quote_volume=Decimal("50000000"),
            trades_count=5000,
        )

    @pytest.fixture
    def invalid_kline(self):
        """Create invalid kline (high < low)."""
        return Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),  # Invalid: high < low
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
            quote_volume=Decimal("50000000"),
            trades_count=5000,
        )

    @pytest.fixture
    def valid_ticker(self):
        """Create valid ticker."""
        return Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("2.5"),
            timestamp=datetime.now(timezone.utc),
        )

    @pytest.mark.asyncio
    async def test_process_valid_kline(self, processor, valid_kline):
        """Test processing valid kline."""
        result = await processor.process_kline(valid_kline)

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.close == Decimal("50500")

    @pytest.mark.asyncio
    async def test_process_invalid_kline_warn(self, valid_kline):
        """Test processing invalid kline with WARN action."""
        # Create kline with invalid OHLC
        invalid = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),  # Invalid
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        config = StreamConfig(action_on_invalid=StreamAction.WARN)
        processor = StreamProcessor(config)

        result = await processor.process_kline(invalid)

        # Should pass through with warning
        assert result is not None
        assert processor.stats.validation_failures == 1

    @pytest.mark.asyncio
    async def test_process_invalid_kline_filter(self):
        """Test processing invalid kline with FILTER action."""
        invalid = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),  # Invalid
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        config = StreamConfig(action_on_invalid=StreamAction.FILTER)
        processor = StreamProcessor(config)

        result = await processor.process_kline(invalid)

        # Should be filtered out
        assert result is None
        assert processor.stats.validation_failures == 1

    @pytest.mark.asyncio
    async def test_process_invalid_kline_correct(self):
        """Test processing invalid kline with CORRECT action."""
        invalid = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),  # Invalid
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        config = StreamConfig(action_on_invalid=StreamAction.CORRECT)
        processor = StreamProcessor(config)

        result = await processor.process_kline(invalid)

        # Should be corrected
        assert result is not None
        assert result.high >= result.low  # OHLC corrected
        assert processor.stats.total_corrected >= 1

    @pytest.mark.asyncio
    async def test_process_valid_ticker(self, processor, valid_ticker):
        """Test processing valid ticker."""
        result = await processor.process_ticker(valid_ticker)

        assert result is not None
        assert result.symbol == "BTCUSDT"
        assert result.price == Decimal("50000")

    @pytest.mark.asyncio
    async def test_process_ticker_detects_price_spike(self):
        """Test anomaly detection for price spike."""
        config = StreamConfig(enable_anomaly_detection=True)
        processor = StreamProcessor(config)

        # First, establish baseline
        for i in range(25):
            ticker = Ticker(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49990"),
                ask=Decimal("50010"),
                high_24h=Decimal("51000"),
                low_24h=Decimal("49000"),
                volume_24h=Decimal("10000"),
                change_24h=Decimal("2.5"),
                timestamp=datetime.now(timezone.utc),
            )
            await processor.process_ticker(ticker)

        # Now send a spike
        spike_ticker = Ticker(
            symbol="BTCUSDT",
            price=Decimal("55000"),  # 10% spike
            bid=Decimal("54990"),
            ask=Decimal("55010"),
            high_24h=Decimal("56000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("10.0"),
            timestamp=datetime.now(timezone.utc),
        )
        result = await processor.process_ticker(spike_ticker)

        assert result is not None
        assert processor.stats.anomalies_detected > 0

    @pytest.mark.asyncio
    async def test_process_depth_valid(self, processor):
        """Test processing valid depth data."""
        depth = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("49990"), Decimal("10")], [Decimal("49980"), Decimal("20")]],
            "asks": [[Decimal("50010"), Decimal("10")], [Decimal("50020"), Decimal("20")]],
        }

        result = await processor.process_depth(depth)

        assert result is not None
        assert result["bids"] == depth["bids"]

    @pytest.mark.asyncio
    async def test_process_depth_wide_spread(self):
        """Test anomaly detection for wide spread."""
        config = StreamConfig(enable_anomaly_detection=True)
        processor = StreamProcessor(config)

        depth = {
            "symbol": "BTCUSDT",
            "bids": [[Decimal("48000"), Decimal("10")]],  # Wide spread
            "asks": [[Decimal("52000"), Decimal("10")]],
        }

        result = await processor.process_depth(depth)

        assert result is not None
        assert processor.stats.anomalies_detected > 0


class TestStreamProcessorCallbackWrapper:
    """Test callback wrapper functionality."""

    @pytest.mark.asyncio
    async def test_wrap_kline_callback(self):
        """Test wrapping kline callback."""
        processor = StreamProcessor()
        received_klines = []

        async def callback(kline: Kline):
            received_klines.append(kline)

        wrapped = processor.wrap_kline_callback(callback)

        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        await wrapped(kline)

        assert len(received_klines) == 1
        assert received_klines[0].symbol == "BTCUSDT"
        assert processor.stats.total_received == 1
        assert processor.stats.total_processed == 1

    @pytest.mark.asyncio
    async def test_wrap_ticker_callback(self):
        """Test wrapping ticker callback."""
        processor = StreamProcessor()
        received_tickers = []

        def callback(ticker: Ticker):
            received_tickers.append(ticker)

        wrapped = processor.wrap_ticker_callback(callback)

        ticker = Ticker(
            symbol="BTCUSDT",
            price=Decimal("50000"),
            bid=Decimal("49990"),
            ask=Decimal("50010"),
            high_24h=Decimal("51000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("2.5"),
            timestamp=datetime.now(timezone.utc),
        )

        await wrapped(ticker)

        assert len(received_tickers) == 1
        assert processor.stats.total_processed == 1

    @pytest.mark.asyncio
    async def test_wrap_callback_filters_invalid(self):
        """Test that wrapped callback filters invalid data."""
        config = StreamConfig(action_on_invalid=StreamAction.FILTER)
        processor = StreamProcessor(config)
        received = []

        async def callback(kline: Kline):
            received.append(kline)

        wrapped = processor.wrap_kline_callback(callback)

        # Invalid kline
        invalid = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        await wrapped(invalid)

        assert len(received) == 0  # Filtered out
        assert processor.stats.total_filtered == 1

    @pytest.mark.asyncio
    async def test_callback_error_handling(self):
        """Test error handling in callback wrapper."""
        config = StreamConfig(emit_raw_on_error=True)
        processor = StreamProcessor(config)
        received = []

        async def callback(kline: Kline):
            received.append(kline)

        wrapped = processor.wrap_kline_callback(callback)

        kline = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )

        await wrapped(kline)

        assert len(received) == 1


class TestStreamProcessorDataGapDetection:
    """Test data gap detection."""

    @pytest.mark.asyncio
    async def test_detects_kline_gap(self):
        """Test detection of gaps between klines."""
        config = StreamConfig(enable_anomaly_detection=True)
        processor = StreamProcessor(config)

        now = datetime.now(timezone.utc)

        # First kline
        kline1 = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=now - timedelta(hours=2),
            open=Decimal("50000"),
            high=Decimal("51000"),
            low=Decimal("49000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=now - timedelta(hours=1),
        )
        await processor.process_kline(kline1)

        # Second kline with gap (should be 1 hour after, but is 3 hours)
        kline2 = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=now + timedelta(hours=2),  # 3 hour gap
            open=Decimal("50500"),
            high=Decimal("52000"),
            low=Decimal("50000"),
            close=Decimal("51500"),
            volume=Decimal("1200"),
            close_time=now + timedelta(hours=3),
        )
        await processor.process_kline(kline2)

        # Should detect gap
        assert processor.stats.anomalies_detected > 0


class TestStreamProcessorPriceSmoothing:
    """Test price smoothing functionality."""

    @pytest.mark.asyncio
    async def test_smooths_outlier_price(self):
        """Test that outlier prices are smoothed."""
        config = StreamConfig(
            enable_auto_correction=True,
            enable_anomaly_detection=False,  # Disable to focus on smoothing
            buffer_size=10,
        )
        processor = StreamProcessor(config)

        # Establish baseline
        for i in range(10):
            ticker = Ticker(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49990"),
                ask=Decimal("50010"),
                high_24h=Decimal("51000"),
                low_24h=Decimal("49000"),
                volume_24h=Decimal("10000"),
                change_24h=Decimal("2.5"),
                timestamp=datetime.now(timezone.utc),
            )
            await processor.process_ticker(ticker)

        # Send outlier
        outlier = Ticker(
            symbol="BTCUSDT",
            price=Decimal("60000"),  # 20% deviation
            bid=Decimal("59990"),
            ask=Decimal("60010"),
            high_24h=Decimal("61000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("20.0"),
            timestamp=datetime.now(timezone.utc),
        )
        result = await processor.process_ticker(outlier)

        # Price should be smoothed (not 60000)
        assert result.price < Decimal("60000")
        assert result.price > Decimal("50000")


class TestStreamProcessorStats:
    """Test statistics tracking."""

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test that stats are tracked correctly using wrapper."""
        processor = StreamProcessor()
        received = []

        async def callback(k):
            received.append(k)

        wrapped = processor.wrap_kline_callback(callback)

        for i in range(5):
            kline = Kline(
                symbol="BTCUSDT",
                interval=KlineInterval.h1,
                open_time=datetime.now(timezone.utc),
                open=Decimal("50000"),
                high=Decimal("51000"),
                low=Decimal("49000"),
                close=Decimal("50500"),
                volume=Decimal("1000"),
                close_time=datetime.now(timezone.utc),
            )
            await wrapped(kline)

        stats = processor.get_stats()
        assert len(received) == 5

        assert stats["processing"]["total_received"] == 5
        assert stats["processing"]["total_processed"] == 5

    def test_reset_stats(self):
        """Test stats reset."""
        processor = StreamProcessor()
        processor.stats.total_received = 100

        processor.reset_stats()

        assert processor.stats.total_received == 0

    def test_clear_state(self):
        """Test state clearing."""
        processor = StreamProcessor()
        processor._last_kline["BTCUSDT"] = MagicMock()
        processor._price_buffer["BTCUSDT"] = [Decimal("50000")]

        processor.clear_state("BTCUSDT")

        assert "BTCUSDT" not in processor._last_kline
        assert "BTCUSDT" not in processor._price_buffer


class TestValidatedWebSocketWrapper:
    """Test ValidatedWebSocketWrapper."""

    @pytest.fixture
    def mock_ws(self):
        """Create mock WebSocket."""
        ws = MagicMock()
        ws.is_connected = True
        ws.connect = AsyncMock(return_value=True)
        ws.disconnect = AsyncMock()
        ws.subscribe_kline = AsyncMock(return_value=True)
        ws.subscribe_ticker = AsyncMock(return_value=True)
        ws.subscribe_depth = AsyncMock(return_value=True)
        ws.subscribe_trade = AsyncMock(return_value=True)
        ws.subscribe_book_ticker = AsyncMock(return_value=True)
        return ws

    @pytest.mark.asyncio
    async def test_wrapper_creation(self, mock_ws):
        """Test wrapper creation."""
        wrapper = ValidatedWebSocketWrapper(mock_ws)

        assert wrapper.is_connected is True
        assert wrapper.processor is not None

    @pytest.mark.asyncio
    async def test_subscribe_kline_wraps_callback(self, mock_ws):
        """Test that subscribe_kline wraps the callback."""
        wrapper = ValidatedWebSocketWrapper(mock_ws)

        async def callback(kline):
            pass

        await wrapper.subscribe_kline("BTCUSDT", "1h", callback)

        # Original subscribe should have been called with wrapped callback
        mock_ws.subscribe_kline.assert_called_once()
        call_args = mock_ws.subscribe_kline.call_args
        assert call_args[0][0] == "BTCUSDT"
        assert call_args[0][1] == "1h"
        # Callback should be wrapped (not the original)
        assert call_args[0][2] != callback

    @pytest.mark.asyncio
    async def test_context_manager(self, mock_ws):
        """Test async context manager."""
        wrapper = ValidatedWebSocketWrapper(mock_ws)

        async with wrapper as ws:
            assert ws is wrapper
            mock_ws.connect.assert_called_once()

        mock_ws.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self, mock_ws):
        """Test getting stats from wrapper."""
        wrapper = ValidatedWebSocketWrapper(mock_ws)

        stats = wrapper.get_stats()

        assert "processing" in stats
        assert "anomaly_detection" in stats


class TestAnomalyCallbacks:
    """Test anomaly callback functionality."""

    @pytest.mark.asyncio
    async def test_on_anomaly_callback(self):
        """Test that on_anomaly callback is called."""
        received_anomalies = []

        def on_anomaly(anomaly: AnomalyRecord):
            received_anomalies.append(anomaly)

        processor = StreamProcessor(on_anomaly=on_anomaly)

        # Establish baseline
        for i in range(25):
            ticker = Ticker(
                symbol="BTCUSDT",
                price=Decimal("50000"),
                bid=Decimal("49990"),
                ask=Decimal("50010"),
                high_24h=Decimal("51000"),
                low_24h=Decimal("49000"),
                volume_24h=Decimal("10000"),
                change_24h=Decimal("2.5"),
                timestamp=datetime.now(timezone.utc),
            )
            await processor.process_ticker(ticker)

        # Trigger spike
        spike = Ticker(
            symbol="BTCUSDT",
            price=Decimal("55000"),
            bid=Decimal("54990"),
            ask=Decimal("55010"),
            high_24h=Decimal("56000"),
            low_24h=Decimal("49000"),
            volume_24h=Decimal("10000"),
            change_24h=Decimal("10.0"),
            timestamp=datetime.now(timezone.utc),
        )
        await processor.process_ticker(spike)

        assert len(received_anomalies) > 0

    @pytest.mark.asyncio
    async def test_on_validation_error_callback(self):
        """Test that on_validation_error callback is called."""
        received_errors = []

        def on_error(symbol: str, errors):
            received_errors.append((symbol, errors))

        config = StreamConfig(action_on_invalid=StreamAction.WARN)
        processor = StreamProcessor(config, on_validation_error=on_error)

        invalid = Kline(
            symbol="BTCUSDT",
            interval=KlineInterval.h1,
            open_time=datetime.now(timezone.utc),
            open=Decimal("50000"),
            high=Decimal("49000"),
            low=Decimal("51000"),
            close=Decimal("50500"),
            volume=Decimal("1000"),
            close_time=datetime.now(timezone.utc),
        )
        await processor.process_kline(invalid)

        assert len(received_errors) > 0
        assert received_errors[0][0] == "BTCUSDT"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
