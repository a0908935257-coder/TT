"""
Unit tests for Rate Limiter.

Tests for:
- Token bucket algorithm
- Sliding window rate limiting
- Request queue with priority
- Rate limiter integration
- Response header parsing
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal

from src.exchange.rate_limiter import (
    RateLimiter,
    RateLimiterManager,
    RateLimitConfig,
    RateLimitStatus,
    RequestPriority,
    TokenBucket,
    SlidingWindow,
    RequestQueue,
    get_rate_limiter_manager,
)


# =============================================================================
# Token Bucket Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_bucket_initialization(self):
        """Test bucket initializes with correct capacity."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)

        assert bucket.capacity == 100
        assert bucket.refill_rate == 10.0
        assert bucket.tokens == 100

    def test_bucket_initialization_with_initial_tokens(self):
        """Test bucket initializes with custom initial tokens."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0, initial_tokens=50)

        # Tokens may have refilled slightly
        assert 50 <= bucket.tokens <= 51

    @pytest.mark.asyncio
    async def test_bucket_acquire_success(self):
        """Test successful token acquisition."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0)

        result = await bucket.acquire(10)

        assert result is True
        # Tokens may have refilled slightly
        assert 89 <= bucket.tokens <= 91

    @pytest.mark.asyncio
    async def test_bucket_acquire_insufficient(self):
        """Test acquisition fails with insufficient tokens."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0, initial_tokens=5)

        result = await bucket.acquire(10, timeout=0)

        assert result is False
        # Tokens unchanged (may have refilled slightly)
        assert 5 <= bucket.tokens <= 6

    @pytest.mark.asyncio
    async def test_bucket_try_acquire(self):
        """Test try_acquire doesn't wait."""
        bucket = TokenBucket(capacity=10, refill_rate=1.0, initial_tokens=5)

        result = await bucket.try_acquire(3)
        assert result is True

        result = await bucket.try_acquire(5)
        assert result is False

    @pytest.mark.asyncio
    async def test_bucket_refill(self):
        """Test tokens refill over time."""
        bucket = TokenBucket(capacity=100, refill_rate=100.0, initial_tokens=0)

        # Wait for refill
        await asyncio.sleep(0.1)

        # Should have ~10 tokens (100 tokens/sec * 0.1 sec)
        assert bucket.tokens >= 5
        assert bucket.tokens <= 15

    def test_bucket_peek(self):
        """Test peek returns correct wait time."""
        bucket = TokenBucket(capacity=100, refill_rate=10.0, initial_tokens=50)

        # Need 30 more tokens, 10/sec = 3 seconds
        wait = bucket.peek(80)
        assert wait >= 2.5
        assert wait <= 3.5

    def test_bucket_is_empty(self):
        """Test is_empty property."""
        # A bucket with 0 initial tokens and minimal refill should be nearly empty
        bucket = TokenBucket(capacity=100, refill_rate=0.001, initial_tokens=0)
        # Even with tiny refill, tokens should be very close to 0
        assert bucket.tokens < 1

        bucket = TokenBucket(capacity=100, refill_rate=10.0, initial_tokens=10)
        assert bucket.is_empty is False
        assert bucket.tokens >= 10


# =============================================================================
# Sliding Window Tests
# =============================================================================


class TestSlidingWindow:
    """Tests for SlidingWindow."""

    def test_window_initialization(self):
        """Test window initializes correctly."""
        window = SlidingWindow(window_seconds=60.0, max_requests=100)

        assert window.window_seconds == 60.0
        assert window.max_requests == 100
        assert window.current_count == 0

    @pytest.mark.asyncio
    async def test_window_record(self):
        """Test recording requests."""
        window = SlidingWindow(window_seconds=60.0, max_requests=100)

        await window.record(weight=5)
        await window.record(weight=3)

        assert window.current_count == 2
        assert window.current_weight == 8

    @pytest.mark.asyncio
    async def test_window_can_proceed(self):
        """Test can_proceed check."""
        window = SlidingWindow(window_seconds=60.0, max_requests=10)

        # Should be able to proceed initially
        assert await window.can_proceed(weight=5) is True

        # Record some weight
        for _ in range(8):
            await window.record(weight=1)

        # Should still be able to proceed
        assert await window.can_proceed(weight=2) is True

        # But not with weight 5
        assert await window.can_proceed(weight=5) is False

    @pytest.mark.asyncio
    async def test_window_available_capacity(self):
        """Test available capacity calculation."""
        window = SlidingWindow(window_seconds=60.0, max_requests=100)

        assert window.available_capacity == 100

        # Record 30 requests
        for _ in range(30):
            await window.record(weight=1)

        assert window.available_capacity == 70

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        """Test requests expire after window."""
        window = SlidingWindow(window_seconds=0.1, max_requests=10)

        await window.record(weight=5)
        assert window.current_weight == 5

        # Wait for expiry
        await asyncio.sleep(0.15)

        assert window.current_weight == 0


# =============================================================================
# Request Queue Tests
# =============================================================================


class TestRequestQueue:
    """Tests for RequestQueue."""

    def test_queue_initialization(self):
        """Test queue initializes correctly."""
        queue = RequestQueue(max_size=100)

        assert queue.max_size == 100
        assert queue.size == 0
        assert queue.is_full is False

    @pytest.mark.asyncio
    async def test_queue_enqueue_dequeue(self):
        """Test basic enqueue/dequeue."""
        queue = RequestQueue()

        async def dummy_callback():
            return "result"

        future = await queue.enqueue(
            dummy_callback,
            priority=RequestPriority.MEDIUM,
        )

        assert queue.size == 1

        request = await queue.dequeue()

        assert request is not None
        assert request.priority == RequestPriority.MEDIUM
        assert queue.size == 0

    @pytest.mark.asyncio
    async def test_queue_priority_ordering(self):
        """Test requests are dequeued by priority."""
        queue = RequestQueue()

        async def callback():
            return None

        # Add in reverse priority order
        await queue.enqueue(callback, priority=RequestPriority.LOW)
        await queue.enqueue(callback, priority=RequestPriority.MEDIUM)
        await queue.enqueue(callback, priority=RequestPriority.CRITICAL)
        await queue.enqueue(callback, priority=RequestPriority.HIGH)

        # Dequeue should be in priority order
        r1 = await queue.dequeue()
        assert r1.priority == RequestPriority.CRITICAL

        r2 = await queue.dequeue()
        assert r2.priority == RequestPriority.HIGH

        r3 = await queue.dequeue()
        assert r3.priority == RequestPriority.MEDIUM

        r4 = await queue.dequeue()
        assert r4.priority == RequestPriority.LOW

    @pytest.mark.asyncio
    async def test_queue_stats(self):
        """Test queue statistics."""
        queue = RequestQueue()

        async def callback():
            return None

        await queue.enqueue(callback, priority=RequestPriority.CRITICAL)
        await queue.enqueue(callback, priority=RequestPriority.HIGH)
        await queue.enqueue(callback, priority=RequestPriority.HIGH)
        await queue.enqueue(callback, priority=RequestPriority.LOW)

        stats = queue.get_stats()

        assert stats["total"] == 4
        assert stats["critical"] == 1
        assert stats["high"] == 2
        assert stats["low"] == 1


# =============================================================================
# Rate Limiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def config(self):
        """Create test config."""
        return RateLimitConfig(
            max_weight_per_minute=1200,
            max_orders_per_second=10,
        )

    @pytest.fixture
    def limiter(self, config):
        """Create test rate limiter."""
        return RateLimiter(config)

    def test_limiter_initialization(self, limiter, config):
        """Test limiter initializes correctly."""
        assert limiter.config == config
        assert limiter.is_limited is False
        assert limiter.current_weight == 0

    @pytest.mark.asyncio
    async def test_limiter_acquire_success(self, limiter):
        """Test successful acquisition."""
        result = await limiter.acquire(weight=10)

        assert result is True
        assert limiter.current_weight == 10

    @pytest.mark.asyncio
    async def test_limiter_multiple_acquisitions(self, limiter):
        """Test multiple acquisitions."""
        for i in range(10):
            result = await limiter.acquire(weight=10)
            assert result is True

        assert limiter.current_weight == 100

    def test_limiter_endpoint_weight(self, limiter):
        """Test endpoint weight lookup."""
        # Known endpoint
        weight = limiter.get_endpoint_weight("/api/v3/depth")
        assert weight == 5

        # Unknown endpoint
        weight = limiter.get_endpoint_weight("/api/v3/unknown")
        assert weight == 1

    def test_limiter_update_from_headers(self, limiter):
        """Test updating from response headers."""
        headers = {
            "X-MBX-USED-WEIGHT-1M": "500",
            "X-MBX-ORDER-COUNT-1S": "3",
        }

        limiter.update_from_headers(headers)

        # Server weight should be updated
        assert limiter.current_weight == 500

    def test_limiter_handle_rate_limit_error(self, limiter):
        """Test rate limit error handling."""
        limiter.handle_rate_limit_error(retry_after=30.0)

        assert limiter.is_limited is True
        assert limiter._retry_after == 30.0
        assert limiter._consecutive_limits == 1

    def test_limiter_handle_success_resets_backoff(self, limiter):
        """Test success resets backoff."""
        limiter.handle_rate_limit_error()
        limiter.handle_rate_limit_error()

        assert limiter._consecutive_limits == 2

        limiter.handle_success()

        assert limiter._consecutive_limits == 0

    def test_limiter_get_status(self, limiter):
        """Test status retrieval."""
        status = limiter.get_status()

        assert isinstance(status, RateLimitStatus)
        assert status.current_weight == 0
        assert status.max_weight == 1200
        assert status.is_limited is False

    @pytest.mark.asyncio
    async def test_limiter_statistics(self, limiter):
        """Test statistics collection."""
        await limiter.acquire(weight=10, endpoint="/api/v3/ticker")
        await limiter.acquire(weight=5, is_order=True)

        stats = limiter.get_statistics()

        assert stats["current_weight"] > 0
        assert stats["requests_last_5min"] == 2
        assert stats["orders_last_5min"] == 1


# =============================================================================
# Rate Limiter Manager Tests
# =============================================================================


class TestRateLimiterManager:
    """Tests for RateLimiterManager."""

    def test_manager_get_limiter(self):
        """Test getting limiter for exchange."""
        manager = RateLimiterManager()

        limiter1 = manager.get_limiter("binance", "spot")
        limiter2 = manager.get_limiter("binance", "spot")

        # Should return same instance
        assert limiter1 is limiter2

    def test_manager_different_markets(self):
        """Test different limiters for spot/futures."""
        manager = RateLimiterManager()

        spot_limiter = manager.get_limiter("binance", "spot")
        futures_limiter = manager.get_limiter("binance", "futures")

        # Should be different instances
        assert spot_limiter is not futures_limiter

        # Futures should have higher limit
        assert futures_limiter.config.max_weight_per_minute == 2400
        assert spot_limiter.config.max_weight_per_minute == 1200

    def test_manager_get_all_status(self):
        """Test getting all limiter statuses."""
        manager = RateLimiterManager()

        manager.get_limiter("binance", "spot")
        manager.get_limiter("binance", "futures")

        all_status = manager.get_all_status()

        assert "binance_spot" in all_status
        assert "binance_futures" in all_status


# =============================================================================
# Global Manager Tests
# =============================================================================


class TestGlobalManager:
    """Tests for global rate limiter manager."""

    def test_global_manager_singleton(self):
        """Test global manager is singleton."""
        manager1 = get_rate_limiter_manager()
        manager2 = get_rate_limiter_manager()

        assert manager1 is manager2
