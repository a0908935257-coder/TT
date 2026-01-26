"""
State Synchronizer for Exchange State Management.

Provides synchronization between local state and exchange state:
- Order state cache with TTL
- Position cache with validation
- Balance cache with periodic refresh
- Conflict detection and resolution
- Event-driven updates from WebSocket
- Periodic reconciliation

This ensures the local application always has an accurate view
of the actual exchange state, handling network issues and
missed WebSocket messages gracefully.
"""

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Set,
    TypeVar,
)

from src.core import get_logger
from src.core.models import MarketType

logger = get_logger(__name__)


# =============================================================================
# Enums and Types
# =============================================================================


class SyncState(str, Enum):
    """Synchronization state."""
    SYNCED = "synced"           # In sync with exchange
    SYNCING = "syncing"         # Currently syncing
    STALE = "stale"             # Data may be outdated
    CONFLICT = "conflict"       # Conflict detected
    ERROR = "error"             # Sync error


class ConflictResolution(str, Enum):
    """Conflict resolution strategy."""
    EXCHANGE_WINS = "exchange_wins"  # Trust exchange data
    LOCAL_WINS = "local_wins"        # Trust local data
    NEWEST_WINS = "newest_wins"      # Trust most recent
    MANUAL = "manual"                # Require manual resolution


class CacheEventType(str, Enum):
    """Cache event types."""
    ADDED = "added"
    UPDATED = "updated"
    REMOVED = "removed"
    EXPIRED = "expired"
    SYNCED = "synced"
    CONFLICT = "conflict"


T = TypeVar("T")


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """
    Cache entry with metadata.

    Attributes:
        data: Cached data
        created_at: When entry was created
        updated_at: When entry was last updated
        expires_at: When entry expires (None for no expiry)
        version: Version number for conflict detection
        source: Where data came from (rest, websocket, local)
    """
    data: T
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    expires_at: Optional[datetime] = None
    version: int = 1
    source: str = "unknown"

    @property
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return (datetime.now(timezone.utc) - self.updated_at).total_seconds()


@dataclass
class SyncConfig:
    """
    State synchronization configuration.

    Attributes:
        order_ttl_seconds: Order cache TTL
        position_ttl_seconds: Position cache TTL
        balance_ttl_seconds: Balance cache TTL
        sync_interval_seconds: Periodic sync interval
        stale_threshold_seconds: When data is considered stale
        enable_websocket_updates: Use WebSocket for updates
        conflict_resolution: How to resolve conflicts
        max_retries: Max sync retries on error
        retry_delay_seconds: Delay between retries
    """
    order_ttl_seconds: float = 300.0       # 5 minutes
    position_ttl_seconds: float = 60.0     # 1 minute
    balance_ttl_seconds: float = 60.0      # 1 minute
    sync_interval_seconds: float = 30.0    # 30 seconds
    stale_threshold_seconds: float = 120.0 # 2 minutes
    enable_websocket_updates: bool = True
    conflict_resolution: ConflictResolution = ConflictResolution.EXCHANGE_WINS
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


@dataclass
class OrderState:
    """
    Local order state.

    Attributes:
        order_id: Exchange order ID
        client_order_id: Client-provided order ID
        symbol: Trading symbol
        side: BUY or SELL
        order_type: LIMIT, MARKET, etc.
        quantity: Original quantity
        price: Limit price
        status: Order status
        filled_quantity: Filled amount
        average_price: Average fill price
        created_at: Creation time
        updated_at: Last update time
    """
    order_id: str
    client_order_id: Optional[str]
    symbol: str
    side: str
    order_type: str
    quantity: Decimal
    price: Optional[Decimal]
    status: str
    filled_quantity: Decimal = field(default_factory=lambda: Decimal("0"))
    average_price: Optional[Decimal] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    market_type: MarketType = MarketType.SPOT

    @property
    def is_open(self) -> bool:
        """Check if order is still open."""
        return self.status in ("NEW", "PARTIALLY_FILLED")

    @property
    def is_closed(self) -> bool:
        """Check if order is closed."""
        return self.status in ("FILLED", "CANCELED", "CANCELLED", "REJECTED", "EXPIRED")

    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining unfilled quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class PositionState:
    """
    Local position state.

    Attributes:
        symbol: Trading symbol
        side: LONG, SHORT, or BOTH
        quantity: Position size
        entry_price: Average entry price
        mark_price: Current mark price
        unrealized_pnl: Unrealized P&L
        leverage: Position leverage
        margin_type: ISOLATED or CROSS
        liquidation_price: Liquidation price
    """
    symbol: str
    side: str
    quantity: Decimal
    entry_price: Decimal
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = field(default_factory=lambda: Decimal("0"))
    leverage: int = 1
    margin_type: str = "ISOLATED"
    liquidation_price: Optional[Decimal] = None
    updated_at: Optional[datetime] = None

    @property
    def is_open(self) -> bool:
        """Check if position is open."""
        return self.quantity > 0


@dataclass
class BalanceState:
    """
    Local balance state.

    Attributes:
        asset: Asset symbol
        free: Available balance
        locked: Locked balance
        total: Total balance
    """
    asset: str
    free: Decimal
    locked: Decimal
    updated_at: Optional[datetime] = None

    @property
    def total(self) -> Decimal:
        """Get total balance."""
        return self.free + self.locked


@dataclass
class SyncEvent:
    """Event from state synchronizer."""
    event_type: CacheEventType
    entity_type: str  # order, position, balance
    entity_id: str
    old_data: Optional[Any] = None
    new_data: Optional[Any] = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# =============================================================================
# Protocols
# =============================================================================


class ExchangeDataProvider(Protocol):
    """Protocol for exchange data access."""

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> List[Dict[str, Any]]:
        """Get open orders from exchange."""
        ...

    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Dict[str, Any]]:
        """Get specific order from exchange."""
        ...

    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get positions from exchange."""
        ...

    async def get_balance(
        self,
        asset: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> Dict[str, Dict[str, Any]]:
        """Get balances from exchange."""
        ...


# =============================================================================
# Cache Implementation
# =============================================================================


class StateCache(Generic[T]):
    """
    Generic cache with TTL and event support.

    Provides:
    - Automatic expiration
    - Version tracking
    - Event callbacks
    - Thread-safe operations
    """

    def __init__(
        self,
        default_ttl: Optional[float] = None,
        max_size: int = 10000,
    ):
        """
        Initialize cache.

        Args:
            default_ttl: Default TTL in seconds (None for no expiry)
            max_size: Maximum cache size
        """
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._cache: Dict[str, CacheEntry[T]] = {}
        self._lock = asyncio.Lock()
        self._event_callbacks: List[Callable[[SyncEvent], None]] = []

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def add_event_callback(self, callback: Callable[[SyncEvent], None]) -> None:
        """Add event callback."""
        self._event_callbacks.append(callback)

    def _emit_event(
        self,
        event_type: CacheEventType,
        entity_id: str,
        entity_type: str,
        old_data: Optional[T] = None,
        new_data: Optional[T] = None,
    ) -> None:
        """Emit cache event."""
        event = SyncEvent(
            event_type=event_type,
            entity_type=entity_type,
            entity_id=entity_id,
            old_data=old_data,
            new_data=new_data,
        )
        for callback in self._event_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.warning(f"Event callback error: {e}")

    async def get(self, key: str) -> Optional[T]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                self._emit_event(
                    CacheEventType.EXPIRED,
                    key,
                    "cache",
                    old_data=entry.data,
                )
                return None

            return entry.data

    async def get_entry(self, key: str) -> Optional[CacheEntry[T]]:
        """Get full cache entry with metadata."""
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                return None

            if entry.is_expired:
                del self._cache[key]
                return None

            return entry

    async def set(
        self,
        key: str,
        value: T,
        ttl: Optional[float] = None,
        source: str = "unknown",
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            source: Data source identifier
        """
        async with self._lock:
            # Check size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                # Remove oldest entry
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].updated_at
                )
                del self._cache[oldest_key]

            ttl_seconds = ttl if ttl is not None else self.default_ttl
            expires_at = None
            if ttl_seconds is not None:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=ttl_seconds
                )

            old_entry = self._cache.get(key)
            is_update = old_entry is not None

            entry = CacheEntry(
                data=value,
                expires_at=expires_at,
                version=(old_entry.version + 1) if old_entry else 1,
                source=source,
            )

            self._cache[key] = entry

            self._emit_event(
                CacheEventType.UPDATED if is_update else CacheEventType.ADDED,
                key,
                "cache",
                old_data=old_entry.data if old_entry else None,
                new_data=value,
            )

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            entry = self._cache.pop(key, None)

            if entry:
                self._emit_event(
                    CacheEventType.REMOVED,
                    key,
                    "cache",
                    old_data=entry.data,
                )
                return True

            return False

    async def get_all(self) -> Dict[str, T]:
        """Get all non-expired values."""
        async with self._lock:
            result = {}
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired:
                    expired_keys.append(key)
                else:
                    result[key] = entry.data

            # Clean expired entries
            for key in expired_keys:
                del self._cache[key]

            return result

    async def get_keys(self) -> List[str]:
        """Get all non-expired keys."""
        async with self._lock:
            return [
                key for key, entry in self._cache.items()
                if not entry.is_expired
            ]

    async def clear(self) -> int:
        """
        Clear all entries.

        Returns:
            Number of entries cleared
        """
        async with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    async def cleanup_expired(self) -> int:
        """
        Remove expired entries.

        Returns:
            Number of entries removed
        """
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]

            for key in expired_keys:
                entry = self._cache.pop(key)
                self._emit_event(
                    CacheEventType.EXPIRED,
                    key,
                    "cache",
                    old_data=entry.data,
                )

            return len(expired_keys)

    async def set_if_newer(
        self,
        key: str,
        value: T,
        timestamp: datetime,
        ttl: Optional[float] = None,
        source: str = "unknown",
    ) -> bool:
        """
        Atomically set value only if it's newer than existing data.

        This prevents race conditions where REST sync might overwrite
        more recent WebSocket updates.

        Args:
            key: Cache key
            value: Value to cache
            timestamp: Timestamp of this data (e.g., updated_at)
            ttl: TTL in seconds (uses default if None)
            source: Data source identifier

        Returns:
            True if value was set, False if existing data is newer
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry and not entry.is_expired:
                # Check if existing data has a timestamp
                existing_ts = getattr(entry.data, 'updated_at', None)
                if existing_ts and timestamp and existing_ts > timestamp:
                    # Existing data is newer, don't update
                    return False

            # Proceed with setting (reuse existing set logic)
            # Check size limit
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(
                    self._cache.keys(),
                    key=lambda k: self._cache[k].updated_at
                )
                del self._cache[oldest_key]

            ttl_seconds = ttl if ttl is not None else self.default_ttl
            expires_at = None
            if ttl_seconds is not None:
                expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=ttl_seconds
                )

            old_entry = self._cache.get(key)
            is_update = old_entry is not None

            new_entry = CacheEntry(
                data=value,
                expires_at=expires_at,
                version=(old_entry.version + 1) if old_entry else 1,
                source=source,
            )

            self._cache[key] = new_entry

            self._emit_event(
                CacheEventType.UPDATED if is_update else CacheEventType.ADDED,
                key,
                "cache",
                old_data=old_entry.data if old_entry else None,
                new_data=value,
            )

            return True


# =============================================================================
# State Synchronizer
# =============================================================================


class StateSynchronizer:
    """
    Synchronizes local state with exchange state.

    Manages:
    - Order cache with updates from REST and WebSocket
    - Position cache with real-time updates
    - Balance cache with periodic refresh
    - Conflict detection and resolution
    - Periodic reconciliation

    Example:
        >>> sync = StateSynchronizer(exchange_provider, config)
        >>> await sync.start()
        >>>
        >>> # Get orders (from cache or exchange)
        >>> orders = await sync.get_open_orders("BTCUSDT")
        >>>
        >>> # Update from WebSocket event
        >>> sync.handle_order_update(order_data)
        >>>
        >>> await sync.stop()
    """

    def __init__(
        self,
        exchange_provider: ExchangeDataProvider,
        config: Optional[SyncConfig] = None,
        market_type: MarketType = MarketType.SPOT,
    ):
        """
        Initialize state synchronizer.

        Args:
            exchange_provider: Exchange data provider
            config: Synchronization configuration
            market_type: Market type
        """
        self.exchange = exchange_provider
        self.config = config or SyncConfig()
        self.market_type = market_type

        # Caches
        self._order_cache: StateCache[OrderState] = StateCache(
            default_ttl=self.config.order_ttl_seconds
        )
        self._position_cache: StateCache[PositionState] = StateCache(
            default_ttl=self.config.position_ttl_seconds
        )
        self._balance_cache: StateCache[BalanceState] = StateCache(
            default_ttl=self.config.balance_ttl_seconds
        )

        # State
        self._sync_state = SyncState.STALE
        self._last_sync: Optional[datetime] = None
        self._sync_errors: List[str] = []
        self._conflicts: List[Dict[str, Any]] = []

        # Background tasks
        self._sync_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # Event callbacks
        self._event_callbacks: List[Callable[[SyncEvent], None]] = []

        # Tracked symbols
        self._tracked_symbols: Set[str] = set()

        # Lock for sync operations
        self._sync_lock = asyncio.Lock()

    @property
    def sync_state(self) -> SyncState:
        """Get current sync state."""
        return self._sync_state

    @property
    def last_sync(self) -> Optional[datetime]:
        """Get last sync time."""
        return self._last_sync

    @property
    def is_stale(self) -> bool:
        """Check if data is stale."""
        if self._last_sync is None:
            return True

        age = (datetime.now(timezone.utc) - self._last_sync).total_seconds()
        return age > self.config.stale_threshold_seconds

    def add_event_callback(self, callback: Callable[[SyncEvent], None]) -> None:
        """Add event callback for sync events."""
        self._event_callbacks.append(callback)
        self._order_cache.add_event_callback(callback)
        self._position_cache.add_event_callback(callback)
        self._balance_cache.add_event_callback(callback)

    def track_symbol(self, symbol: str) -> None:
        """Add symbol to tracking list."""
        self._tracked_symbols.add(symbol)

    def untrack_symbol(self, symbol: str) -> None:
        """Remove symbol from tracking list."""
        self._tracked_symbols.discard(symbol)

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def start(self) -> None:
        """Start synchronizer with background tasks."""
        if self._running:
            return

        self._running = True

        # Initial sync
        await self.sync_all()

        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

        logger.info("State synchronizer started")

    async def stop(self) -> None:
        """Stop synchronizer and cleanup."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("State synchronizer stopped")

    async def _sync_loop(self) -> None:
        """Background sync loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.sync_interval_seconds)
                await self.sync_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(5.0)

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self._running:
            try:
                await asyncio.sleep(60.0)  # Every minute
                await self._order_cache.cleanup_expired()
                await self._position_cache.cleanup_expired()
                await self._balance_cache.cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")

    # =========================================================================
    # Sync Operations
    # =========================================================================

    async def sync_all(self) -> bool:
        """
        Sync all state from exchange.

        Returns:
            True if successful
        """
        async with self._sync_lock:
            self._sync_state = SyncState.SYNCING
            success = True

            try:
                # Sync orders
                await self._sync_orders()

                # Sync positions (futures only)
                if self.market_type == MarketType.FUTURES:
                    await self._sync_positions()

                # Sync balances
                await self._sync_balances()

                self._sync_state = SyncState.SYNCED
                self._last_sync = datetime.now(timezone.utc)
                self._sync_errors.clear()

                logger.debug("Full sync completed")

            except Exception as e:
                self._sync_state = SyncState.ERROR
                self._sync_errors.append(str(e))
                logger.error(f"Sync failed: {e}")
                success = False

            return success

    async def _sync_orders(self) -> None:
        """Sync open orders from exchange."""
        try:
            # Get open orders from exchange
            exchange_orders = await self.exchange.get_open_orders(
                market_type=self.market_type
            )

            # Get current cached order IDs
            cached_keys = await self._order_cache.get_keys()
            exchange_order_ids = set()

            # Update cache with exchange data
            for order_data in exchange_orders:
                order_state = self._parse_order(order_data)
                order_id = order_state.order_id
                exchange_order_ids.add(order_id)

                # Check for conflicts and timestamp
                cached_entry = await self._order_cache.get_entry(order_id)
                if cached_entry:
                    cached_order = cached_entry.data
                    self._check_order_conflict(cached_order, order_state)

                    # Don't overwrite with older data (compare updated_at timestamps)
                    if (
                        cached_order.updated_at
                        and order_state.updated_at
                        and cached_order.updated_at > order_state.updated_at
                    ):
                        logger.debug(
                            f"Skipping REST sync for {order_id}: "
                            f"cached={cached_order.updated_at} > incoming={order_state.updated_at}"
                        )
                        continue

                await self._order_cache.set(
                    order_id,
                    order_state,
                    source="rest_sync",
                )

            # Mark closed orders that are no longer open
            for key in cached_keys:
                if key not in exchange_order_ids:
                    order = await self._order_cache.get(key)
                    if order and order.is_open:
                        # Order was closed, fetch final state
                        await self._refresh_order(order.symbol, key)

            logger.debug(f"Synced {len(exchange_orders)} orders")

        except Exception as e:
            logger.error(f"Order sync failed: {e}")
            raise

    async def _sync_positions(self) -> None:
        """Sync positions from exchange."""
        try:
            exchange_positions = await self.exchange.get_positions()

            for pos_data in exchange_positions:
                position = self._parse_position(pos_data)

                if position.quantity > 0:
                    await self._position_cache.set(
                        position.symbol,
                        position,
                        source="rest_sync",
                    )
                else:
                    # No position, remove from cache
                    await self._position_cache.delete(position.symbol)

            logger.debug(f"Synced {len(exchange_positions)} positions")

        except Exception as e:
            logger.error(f"Position sync failed: {e}")
            raise

    async def _sync_balances(self) -> None:
        """Sync balances from exchange."""
        try:
            exchange_balances = await self.exchange.get_balance(
                market_type=self.market_type
            )

            for asset, balance_data in exchange_balances.items():
                balance = self._parse_balance(asset, balance_data)

                if balance.total > 0:
                    await self._balance_cache.set(
                        asset,
                        balance,
                        source="rest_sync",
                    )

            logger.debug(f"Synced {len(exchange_balances)} balances")

        except Exception as e:
            logger.error(f"Balance sync failed: {e}")
            raise

    async def _refresh_order(self, symbol: str, order_id: str) -> None:
        """Refresh single order from exchange."""
        try:
            order_data = await self.exchange.get_order(
                symbol,
                order_id,
                self.market_type,
            )

            if order_data:
                order = self._parse_order(order_data)
                await self._order_cache.set(
                    order_id,
                    order,
                    source="rest_refresh",
                )

        except Exception as e:
            logger.warning(f"Order refresh failed: {e}")

    # =========================================================================
    # WebSocket Updates
    # =========================================================================

    def handle_order_update(self, order_data: Dict[str, Any]) -> None:
        """
        Handle order update from WebSocket.

        Args:
            order_data: Order update data from WebSocket
        """
        asyncio.create_task(self._handle_order_update_async(order_data))

    async def _handle_order_update_async(self, order_data: Dict[str, Any]) -> None:
        """Async handler for order update."""
        try:
            order = self._parse_order(order_data)

            # Check for conflict and timestamp
            cached_entry = await self._order_cache.get_entry(order.order_id)
            if cached_entry:
                cached_order = cached_entry.data
                self._check_order_conflict(cached_order, order)

                # Don't overwrite with older data (compare updated_at timestamps)
                if (
                    cached_order.updated_at
                    and order.updated_at
                    and cached_order.updated_at > order.updated_at
                ):
                    logger.debug(
                        f"Skipping WS update for {order.order_id}: "
                        f"cached={cached_order.updated_at} > incoming={order.updated_at}"
                    )
                    return

            await self._order_cache.set(
                order.order_id,
                order,
                source="websocket",
            )

            logger.debug(f"Order updated from WS: {order.order_id} -> {order.status}")

        except Exception as e:
            logger.error(f"Order update handling failed: {e}")

    def handle_position_update(self, position_data: Dict[str, Any]) -> None:
        """Handle position update from WebSocket."""
        asyncio.create_task(self._handle_position_update_async(position_data))

    async def _handle_position_update_async(
        self,
        position_data: Dict[str, Any]
    ) -> None:
        """Async handler for position update."""
        try:
            position = self._parse_position(position_data)

            await self._position_cache.set(
                position.symbol,
                position,
                source="websocket",
            )

            logger.debug(f"Position updated from WS: {position.symbol}")

        except Exception as e:
            logger.error(f"Position update handling failed: {e}")

    def handle_balance_update(self, balance_data: Dict[str, Any]) -> None:
        """Handle balance update from WebSocket."""
        asyncio.create_task(self._handle_balance_update_async(balance_data))

    async def _handle_balance_update_async(
        self,
        balance_data: Dict[str, Any]
    ) -> None:
        """Async handler for balance update."""
        try:
            asset = balance_data.get("a") or balance_data.get("asset", "")
            balance = self._parse_balance(asset, balance_data)

            await self._balance_cache.set(
                asset,
                balance,
                source="websocket",
            )

            logger.debug(f"Balance updated from WS: {asset}")

        except Exception as e:
            logger.error(f"Balance update handling failed: {e}")

    # =========================================================================
    # Data Access
    # =========================================================================

    async def get_order(
        self,
        order_id: str,
        symbol: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Optional[OrderState]:
        """
        Get order by ID.

        Args:
            order_id: Order ID
            symbol: Symbol (required for refresh)
            force_refresh: Force fetch from exchange

        Returns:
            OrderState or None
        """
        if force_refresh and symbol:
            await self._refresh_order(symbol, order_id)

        return await self._order_cache.get(order_id)

    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
    ) -> List[OrderState]:
        """
        Get open orders.

        Args:
            symbol: Filter by symbol

        Returns:
            List of open orders
        """
        all_orders = await self._order_cache.get_all()

        orders = [
            order for order in all_orders.values()
            if order.is_open
        ]

        if symbol:
            orders = [o for o in orders if o.symbol == symbol]

        return orders

    async def get_position(
        self,
        symbol: str,
        force_refresh: bool = False,
    ) -> Optional[PositionState]:
        """
        Get position for symbol.

        Args:
            symbol: Trading symbol
            force_refresh: Force fetch from exchange

        Returns:
            PositionState or None
        """
        if force_refresh:
            await self._sync_positions()

        return await self._position_cache.get(symbol)

    async def get_all_positions(self) -> List[PositionState]:
        """Get all positions."""
        all_positions = await self._position_cache.get_all()
        return list(all_positions.values())

    async def get_balance(
        self,
        asset: str,
        force_refresh: bool = False,
    ) -> Optional[BalanceState]:
        """
        Get balance for asset.

        Args:
            asset: Asset symbol
            force_refresh: Force fetch from exchange

        Returns:
            BalanceState or None
        """
        if force_refresh:
            await self._sync_balances()

        return await self._balance_cache.get(asset)

    async def get_all_balances(self) -> Dict[str, BalanceState]:
        """Get all balances."""
        return await self._balance_cache.get_all()

    # =========================================================================
    # Local Updates
    # =========================================================================

    async def record_order_sent(
        self,
        order_id: str,
        client_order_id: Optional[str],
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal],
    ) -> None:
        """
        Record locally sent order.

        Call this when placing an order, before exchange confirms.
        """
        order = OrderState(
            order_id=order_id,
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            status="NEW",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            market_type=self.market_type,
        )

        await self._order_cache.set(
            order_id,
            order,
            source="local",
        )

    async def record_order_cancelled(self, order_id: str) -> None:
        """Record locally cancelled order."""
        order = await self._order_cache.get(order_id)
        if order:
            order.status = "CANCELED"
            order.updated_at = datetime.now(timezone.utc)
            await self._order_cache.set(
                order_id,
                order,
                source="local",
            )

    # =========================================================================
    # Conflict Detection
    # =========================================================================

    def _check_order_conflict(
        self,
        cached: OrderState,
        exchange: OrderState,
    ) -> None:
        """Check for conflicts between cached and exchange order."""
        conflicts = []

        # Status conflict
        if cached.status != exchange.status:
            # This is expected for updates
            if cached.is_open and exchange.is_closed:
                logger.debug(f"Order {cached.order_id} closed: {exchange.status}")
            elif cached.is_closed and exchange.is_open:
                conflicts.append(f"Status mismatch: local={cached.status}, exchange={exchange.status}")

        # Filled quantity conflict
        if cached.filled_quantity > exchange.filled_quantity:
            conflicts.append(
                f"Filled qty mismatch: local={cached.filled_quantity}, "
                f"exchange={exchange.filled_quantity}"
            )

        if conflicts:
            self._handle_conflict("order", cached.order_id, conflicts)

    def _handle_conflict(
        self,
        entity_type: str,
        entity_id: str,
        conflicts: List[str],
    ) -> None:
        """Handle detected conflict."""
        conflict_record = {
            "entity_type": entity_type,
            "entity_id": entity_id,
            "conflicts": conflicts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolution": self.config.conflict_resolution.value,
        }

        self._conflicts.append(conflict_record)
        self._sync_state = SyncState.CONFLICT

        logger.warning(f"Conflict detected for {entity_type}/{entity_id}: {conflicts}")

        # Emit conflict event
        for callback in self._event_callbacks:
            try:
                callback(SyncEvent(
                    event_type=CacheEventType.CONFLICT,
                    entity_type=entity_type,
                    entity_id=entity_id,
                    new_data=conflict_record,
                ))
            except Exception as e:
                logger.warning(f"Conflict callback error: {e}")

    def get_conflicts(self) -> List[Dict[str, Any]]:
        """Get all detected conflicts."""
        return self._conflicts.copy()

    def clear_conflicts(self) -> None:
        """Clear conflict records."""
        self._conflicts.clear()
        if self._sync_state == SyncState.CONFLICT:
            self._sync_state = SyncState.SYNCED

    # =========================================================================
    # Parsing Helpers
    # =========================================================================

    def _parse_order(self, data: Dict[str, Any]) -> OrderState:
        """Parse order data from exchange."""
        return OrderState(
            order_id=str(data.get("orderId") or data.get("i", "")),
            client_order_id=data.get("clientOrderId") or data.get("c"),
            symbol=data.get("symbol") or data.get("s", ""),
            side=data.get("side") or data.get("S", ""),
            order_type=data.get("type") or data.get("o", ""),
            quantity=Decimal(str(data.get("origQty") or data.get("q", 0))),
            price=Decimal(str(data.get("price") or data.get("p", 0))) or None,
            status=data.get("status") or data.get("X", "NEW"),
            filled_quantity=Decimal(str(
                data.get("executedQty") or data.get("z", 0)
            )),
            average_price=Decimal(str(
                data.get("avgPrice") or data.get("ap", 0)
            )) or None,
            created_at=datetime.fromtimestamp(
                int(data.get("time", 0)) / 1000,
                tz=timezone.utc,
            ) if data.get("time") else None,
            updated_at=datetime.fromtimestamp(
                int(data.get("updateTime") or data.get("T", 0)) / 1000,
                tz=timezone.utc,
            ) if (data.get("updateTime") or data.get("T")) else datetime.now(timezone.utc),
            market_type=self.market_type,
        )

    def _parse_position(self, data: Dict[str, Any]) -> PositionState:
        """Parse position data from exchange."""
        quantity = Decimal(str(data.get("positionAmt") or data.get("pa", 0)))
        side = "LONG" if quantity > 0 else "SHORT" if quantity < 0 else "BOTH"

        return PositionState(
            symbol=data.get("symbol") or data.get("s", ""),
            side=side,
            quantity=abs(quantity),
            entry_price=Decimal(str(data.get("entryPrice") or data.get("ep", 0))),
            mark_price=Decimal(str(data.get("markPrice") or data.get("mp", 0))) or None,
            unrealized_pnl=Decimal(str(
                data.get("unrealizedProfit") or data.get("up", 0)
            )),
            leverage=int(data.get("leverage", 1)),
            margin_type=data.get("marginType", "ISOLATED"),
            liquidation_price=Decimal(str(
                data.get("liquidationPrice") or data.get("lp", 0)
            )) or None,
            updated_at=datetime.now(timezone.utc),
        )

    def _parse_balance(
        self,
        asset: str,
        data: Dict[str, Any],
    ) -> BalanceState:
        """Parse balance data from exchange."""
        return BalanceState(
            asset=asset,
            free=Decimal(str(
                data.get("free") or data.get("wb", 0)
            )),
            locked=Decimal(str(
                data.get("locked") or data.get("cw", 0)
            )),
            updated_at=datetime.now(timezone.utc),
        )

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get synchronizer statistics."""
        return {
            "sync_state": self._sync_state.value,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "is_stale": self.is_stale,
            "order_cache_size": self._order_cache.size,
            "position_cache_size": self._position_cache.size,
            "balance_cache_size": self._balance_cache.size,
            "tracked_symbols": len(self._tracked_symbols),
            "conflicts": len(self._conflicts),
            "sync_errors": len(self._sync_errors),
        }
