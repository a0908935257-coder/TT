"""
Market Data Manager.

Provides unified data access layer integrating PostgreSQL, Redis,
K-line management, and all repositories.
"""

from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Optional

from core import get_logger
from core.models import (
    Balance,
    Kline,
    KlineInterval,
    MarketType,
    Order,
    Position,
    Ticker,
    Trade,
)
from exchange import ExchangeClient

from .cache import AccountCache, MarketCache, RedisManager
from .database import DatabaseManager
from .kline import KlineManager, TechnicalIndicators
from .repositories import (
    BalanceRepository,
    BotStateRepository,
    OrderRepository,
    PositionRepository,
    TradeRepository,
)

logger = get_logger(__name__)


class MarketDataManager:
    """
    Unified data manager integrating all data sources.

    Provides a single entry point for all data operations including:
    - PostgreSQL database (persistent storage)
    - Redis cache (fast access)
    - K-line management (three-tier data source)
    - All repositories (CRUD operations)

    Example:
        >>> async with MarketDataManager(db_config, redis_config) as data:
        ...     price = await data.get_price("BTCUSDT")
        ...     klines = await data.get_klines("BTCUSDT", "1h", 100)
        ...     await data.save_order(order, bot_id="bot_1")
    """

    def __init__(
        self,
        db_config: dict,
        redis_config: dict,
        exchange: Optional[ExchangeClient] = None,
    ):
        """
        Initialize MarketDataManager.

        Args:
            db_config: Database configuration dict with keys:
                - host, port, database, user, password, pool_size, etc.
            redis_config: Redis configuration dict with keys:
                - host, port, db, password, key_prefix, etc.
            exchange: ExchangeClient instance (optional, for live data)
        """
        self._db_config = db_config
        self._redis_config = redis_config
        self._exchange = exchange
        self._connected = False

        # Data sources (initialized on connect)
        self._db: Optional[DatabaseManager] = None
        self._redis: Optional[RedisManager] = None
        self._kline_manager: Optional[KlineManager] = None

        # Caches
        self._market_cache: Optional[MarketCache] = None
        self._account_cache: Optional[AccountCache] = None

        # Repositories
        self._orders: Optional[OrderRepository] = None
        self._trades: Optional[TradeRepository] = None
        self._positions: Optional[PositionRepository] = None
        self._balances: Optional[BalanceRepository] = None
        self._bot_states: Optional[BotStateRepository] = None

    # =========================================================================
    # Properties - Data Sources
    # =========================================================================

    @property
    def db(self) -> DatabaseManager:
        """Get DatabaseManager instance."""
        if not self._db:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._db

    @property
    def cache(self) -> RedisManager:
        """Get RedisManager instance."""
        if not self._redis:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._redis

    @property
    def klines(self) -> KlineManager:
        """Get KlineManager instance."""
        if not self._kline_manager:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._kline_manager

    @property
    def market_cache(self) -> MarketCache:
        """Get MarketCache instance."""
        if not self._market_cache:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._market_cache

    @property
    def account_cache(self) -> AccountCache:
        """Get AccountCache instance."""
        if not self._account_cache:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._account_cache

    # =========================================================================
    # Properties - Repositories
    # =========================================================================

    @property
    def orders(self) -> OrderRepository:
        """Get OrderRepository instance."""
        if not self._orders:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._orders

    @property
    def trades(self) -> TradeRepository:
        """Get TradeRepository instance."""
        if not self._trades:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._trades

    @property
    def positions(self) -> PositionRepository:
        """Get PositionRepository instance."""
        if not self._positions:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._positions

    @property
    def balances(self) -> BalanceRepository:
        """Get BalanceRepository instance."""
        if not self._balances:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._balances

    @property
    def bot_states(self) -> BotStateRepository:
        """Get BotStateRepository instance."""
        if not self._bot_states:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._bot_states

    @property
    def is_connected(self) -> bool:
        """Check if all data sources are connected."""
        return self._connected

    # =========================================================================
    # Lifecycle Management
    # =========================================================================

    async def connect(self) -> bool:
        """
        Connect to all data sources.

        Returns:
            True if all connections successful
        """
        if self._connected:
            logger.debug("Already connected")
            return True

        try:
            logger.info("Connecting to data sources...")

            # Initialize and connect database
            self._db = DatabaseManager(**self._db_config)
            db_ok = await self._db.connect()
            if not db_ok:
                raise RuntimeError("Database connection failed")

            # Initialize and connect Redis
            self._redis = RedisManager(**self._redis_config)
            redis_ok = await self._redis.connect()
            if not redis_ok:
                raise RuntimeError("Redis connection failed")

            # Initialize caches
            self._market_cache = MarketCache(self._redis)
            self._account_cache = AccountCache(self._redis)

            # Initialize KlineManager
            if self._exchange:
                self._kline_manager = KlineManager(
                    self._redis,
                    self._db,
                    self._exchange,
                )

            # Initialize repositories
            self._orders = OrderRepository(self._db)
            self._trades = TradeRepository(self._db)
            self._positions = PositionRepository(self._db)
            self._balances = BalanceRepository(self._db)
            self._bot_states = BotStateRepository(self._db)

            self._connected = True
            logger.info("All data sources connected")
            return True

        except Exception as e:
            logger.error(f"Connection failed: {e}")
            await self.disconnect()
            return False

    async def disconnect(self) -> None:
        """Disconnect from all data sources."""
        logger.info("Disconnecting from data sources...")

        if self._redis:
            await self._redis.disconnect()
            self._redis = None

        if self._db:
            await self._db.disconnect()
            self._db = None

        self._kline_manager = None
        self._market_cache = None
        self._account_cache = None
        self._orders = None
        self._trades = None
        self._positions = None
        self._balances = None
        self._bot_states = None

        self._connected = False
        logger.info("All data sources disconnected")

    async def __aenter__(self) -> "MarketDataManager":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.disconnect()

    async def health_check(self) -> dict[str, bool]:
        """
        Check health of all data sources.

        Returns:
            Dict with health status of each component
        """
        status = {
            "database": False,
            "redis": False,
            "overall": False,
        }

        try:
            if self._db:
                status["database"] = await self._db.health_check()
        except Exception as e:
            logger.error(f"Database health check failed: {e}")

        try:
            if self._redis:
                status["redis"] = await self._redis.health_check()
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")

        status["overall"] = status["database"] and status["redis"]
        return status

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    async def get_price(
        self,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Decimal]:
        """
        Get latest price for a symbol (cache-first).

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            market_type: Market type (SPOT or FUTURES)

        Returns:
            Latest price or None if unavailable
        """
        symbol = symbol.upper()

        # Try cache first
        if self._market_cache:
            cached_price = await self._market_cache.get_price(symbol)
            if cached_price is not None:
                return cached_price

        # Fallback to exchange
        if self._exchange:
            try:
                if market_type == MarketType.SPOT:
                    price = await self._exchange.spot.get_price(symbol)
                else:
                    price = await self._exchange.futures.get_price(symbol)

                # Cache the price
                if price and self._market_cache:
                    await self._market_cache.set_price(symbol, price)

                return price
            except Exception as e:
                logger.warning(f"Failed to get price from exchange: {e}")

        return None

    async def get_ticker(
        self,
        symbol: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[Ticker]:
        """
        Get ticker data for a symbol (cache-first).

        Args:
            symbol: Trading pair
            market_type: Market type

        Returns:
            Ticker or None if unavailable
        """
        symbol = symbol.upper()

        # Try cache first
        if self._market_cache:
            cached = await self._market_cache.get_ticker(symbol)
            if cached:
                return Ticker(**cached)

        # Fallback to exchange
        if self._exchange:
            try:
                if market_type == MarketType.SPOT:
                    ticker = await self._exchange.spot.get_ticker(symbol)
                else:
                    ticker = await self._exchange.futures.get_ticker(symbol)

                # Cache the ticker
                if ticker and self._market_cache:
                    await self._market_cache.set_ticker(symbol, ticker)

                return ticker
            except Exception as e:
                logger.warning(f"Failed to get ticker from exchange: {e}")

        return None

    async def get_klines(
        self,
        symbol: str,
        interval: str | KlineInterval,
        limit: int = 500,
    ) -> list[Kline]:
        """
        Get K-lines for a symbol.

        Args:
            symbol: Trading pair
            interval: K-line interval (e.g., "1h")
            limit: Number of K-lines

        Returns:
            List of Kline objects
        """
        if self._kline_manager:
            return await self._kline_manager.get_klines(symbol, interval, limit)

        # Fallback to exchange directly
        if self._exchange:
            interval_enum = (
                KlineInterval(interval)
                if isinstance(interval, str)
                else interval
            )
            return await self._exchange.spot.get_klines(
                symbol, interval_enum, limit
            )

        return []

    # =========================================================================
    # Order Methods
    # =========================================================================

    async def save_order(
        self,
        order: Order,
        bot_id: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> Order:
        """
        Save order to database.

        Args:
            order: Order to save
            bot_id: Bot identifier
            market_type: Market type

        Returns:
            Saved Order
        """
        return await self.orders.create(order, bot_id, market_type)

    async def update_order(
        self,
        order: Order,
        bot_id: Optional[str] = None,
    ) -> Order:
        """
        Update existing order.

        Args:
            order: Order with updated fields
            bot_id: Bot identifier

        Returns:
            Updated Order
        """
        return await self.orders.update(order, bot_id)

    async def get_open_orders(
        self,
        bot_id: str,
        symbol: Optional[str] = None,
    ) -> list[Order]:
        """
        Get open orders for a bot.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair filter (optional)

        Returns:
            List of open orders
        """
        return await self.orders.get_open_orders(bot_id, symbol)

    async def get_order(self, order_id: str) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order ID

        Returns:
            Order or None
        """
        return await self.orders.get_by_order_id(order_id)

    # =========================================================================
    # Trade Methods
    # =========================================================================

    async def save_trade(
        self,
        trade: Trade,
        bot_id: Optional[str] = None,
    ) -> Trade:
        """
        Save trade to database.

        Args:
            trade: Trade to save
            bot_id: Bot identifier

        Returns:
            Saved Trade
        """
        return await self.trades.create(trade, bot_id)

    async def save_trades(
        self,
        trades: list[Trade],
        bot_id: Optional[str] = None,
    ) -> list[Trade]:
        """
        Save multiple trades.

        Args:
            trades: Trades to save
            bot_id: Bot identifier

        Returns:
            Saved Trades
        """
        return await self.trades.create_batch(trades, bot_id)

    # =========================================================================
    # Position Methods
    # =========================================================================

    async def update_position(
        self,
        bot_id: str,
        position: Position,
    ) -> Position:
        """
        Update or create position.

        Args:
            bot_id: Bot identifier
            position: Position to save

        Returns:
            Saved Position
        """
        return await self.positions.upsert(position, bot_id)

    async def get_position(
        self,
        bot_id: str,
        symbol: str,
    ) -> Optional[Position]:
        """
        Get position for a symbol.

        Args:
            bot_id: Bot identifier
            symbol: Trading pair

        Returns:
            Position or None
        """
        # Try cache first
        if self._account_cache:
            cached = await self._account_cache.get_position(bot_id, symbol)
            if cached:
                return Position(**cached)

        # Fallback to database
        return await self.positions.get_position(bot_id, symbol)

    async def get_all_positions(self, bot_id: str) -> list[Position]:
        """
        Get all positions for a bot.

        Args:
            bot_id: Bot identifier

        Returns:
            List of positions
        """
        return await self.positions.get_all_positions(bot_id)

    # =========================================================================
    # Balance Methods
    # =========================================================================

    async def save_balances(
        self,
        bot_id: str,
        balances: list[Balance],
        market_type: MarketType = MarketType.SPOT,
    ) -> list[Balance]:
        """
        Save balance snapshot.

        Args:
            bot_id: Bot identifier
            balances: List of balances
            market_type: Market type

        Returns:
            Saved balances
        """
        return await self.balances.save_snapshot(bot_id, balances, market_type)

    async def get_balance(
        self,
        bot_id: str,
        asset: str,
    ) -> Optional[Balance]:
        """
        Get latest balance for an asset.

        Args:
            bot_id: Bot identifier
            asset: Asset name (e.g., "USDT")

        Returns:
            Balance or None
        """
        # Try cache first
        if self._account_cache:
            cached = await self._account_cache.get_balance(bot_id, asset)
            if cached:
                return Balance(**cached)

        # Fallback to database (get from latest snapshot)
        latest = await self.balances.get_latest_snapshot(bot_id)
        for balance in latest:
            if balance.asset == asset.upper():
                return balance
        return None

    # =========================================================================
    # Bot State Methods
    # =========================================================================

    async def save_bot_state(
        self,
        bot_id: str,
        bot_type: str,
        status: str,
        config: dict[str, Any],
        state_data: dict[str, Any],
    ) -> dict:
        """
        Save bot state.

        Args:
            bot_id: Bot identifier
            bot_type: Bot type
            status: Bot status
            config: Bot configuration
            state_data: Runtime state data

        Returns:
            Saved state dict
        """
        return await self.bot_states.save_state(
            bot_id, bot_type, status, config, state_data
        )

    async def get_bot_state(self, bot_id: str) -> Optional[dict]:
        """
        Get bot state.

        Args:
            bot_id: Bot identifier

        Returns:
            State dict or None
        """
        return await self.bot_states.get_state(bot_id)

    async def update_bot_status(
        self,
        bot_id: str,
        status: str,
    ) -> Optional[dict]:
        """
        Update bot status.

        Args:
            bot_id: Bot identifier
            status: New status

        Returns:
            Updated state dict or None
        """
        return await self.bot_states.update_status(bot_id, status)

    # =========================================================================
    # Statistics Methods
    # =========================================================================

    async def get_daily_stats(
        self,
        bot_id: str,
        target_date: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Get daily statistics for a bot.

        Args:
            bot_id: Bot identifier
            target_date: Date to get stats for (default: today)

        Returns:
            Dict with daily statistics
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc).date()

        start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
        end = datetime.combine(target_date, datetime.max.time(), tzinfo=timezone.utc)

        return await self.get_period_stats(bot_id, start, end)

    async def get_period_stats(
        self,
        bot_id: str,
        start: datetime,
        end: datetime,
    ) -> dict[str, Any]:
        """
        Get statistics for a time period.

        Args:
            bot_id: Bot identifier
            start: Start time
            end: End time

        Returns:
            Dict with period statistics
        """
        # Get trades in period
        trades_list = await self.trades.get_trades_in_range(start, end, bot_id)

        # Calculate statistics
        total_trades = len(trades_list)
        total_volume = await self.trades.get_trade_volume(bot_id, start, end)
        total_fees = await self.trades.get_total_fees(bot_id, start, end)
        realized_pnl = await self.trades.get_realized_pnl(bot_id, start, end)

        # Get orders in period
        orders_list = await self.orders.get_orders_in_range(start, end, bot_id)
        filled_orders = [o for o in orders_list if o.is_filled]

        return {
            "bot_id": bot_id,
            "period_start": start,
            "period_end": end,
            "total_trades": total_trades,
            "total_orders": len(orders_list),
            "filled_orders": len(filled_orders),
            "total_volume": total_volume,
            "total_fees": total_fees,
            "realized_pnl": realized_pnl,
            "net_pnl": realized_pnl - total_fees,
        }

    # =========================================================================
    # Technical Analysis Helpers
    # =========================================================================

    @staticmethod
    def calculate_atr(klines: list[Kline], period: int = 14) -> Optional[Decimal]:
        """Calculate ATR for klines."""
        return TechnicalIndicators.atr(klines, period)

    @staticmethod
    def calculate_rsi(klines: list[Kline], period: int = 14) -> Optional[Decimal]:
        """Calculate RSI for klines."""
        return TechnicalIndicators.rsi(klines, period)

    @staticmethod
    def calculate_sma(klines: list[Kline], period: int) -> Optional[Decimal]:
        """Calculate SMA for klines."""
        return TechnicalIndicators.sma(klines, period)

    @staticmethod
    def calculate_ema(klines: list[Kline], period: int) -> Optional[Decimal]:
        """Calculate EMA for klines."""
        return TechnicalIndicators.ema(klines, period)
