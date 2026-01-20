"""
Grid Futures Bot - Futures-based Grid Trading with Leverage.

Features:
- Leverage trading (1x-125x)
- Bidirectional trading (long/short)
- Trend-following mode
- Dynamic ATR-based grid range
- Automatic grid rebuilding

Optimized for:
- 年化 >30%
- 回撤 <50%
- Sharpe >1.0
"""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import Kline
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.master.models import BotState
from src.notification import NotificationManager

from .models import (
    GridFuturesConfig,
    GridDirection,
    GridLevel,
    GridLevelState,
    GridSetup,
    FuturesPosition,
    PositionSide,
    GridTrade,
    GridFuturesStats,
    ExitReason,
)

logger = get_logger(__name__)


class GridFuturesBot(BaseBot):
    """
    Futures-based Grid Trading Bot.

    Implements grid trading strategy on futures market with:
    - Configurable leverage
    - Trend-following direction
    - Dynamic ATR-based grid range
    - Automatic position management
    """

    def __init__(
        self,
        bot_id: str,
        config: GridFuturesConfig,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: Optional[NotificationManager] = None,
    ):
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier

        # State
        self._grid: Optional[GridSetup] = None
        self._position: Optional[FuturesPosition] = None
        self._current_trend: int = 0  # 1=up, -1=down, 0=neutral
        self._capital: Decimal = Decimal("0")
        self._initial_capital: Decimal = Decimal("0")

        # Indicators cache
        self._closes: List[Decimal] = []
        self._klines: List[Kline] = []

        # Statistics
        self._grid_stats = GridFuturesStats()

        # Tasks
        self._monitor_task: Optional[asyncio.Task] = None

    # =========================================================================
    # BaseBot Abstract Properties
    # =========================================================================

    @property
    def bot_type(self) -> str:
        return "grid_futures"

    @property
    def symbol(self) -> str:
        return self._config.symbol

    # =========================================================================
    # BaseBot Abstract Methods
    # =========================================================================

    async def _do_start(self) -> None:
        """Initialize and start the bot."""
        logger.info(f"Starting Grid Futures Bot for {self._config.symbol}")

        # Set leverage and margin type
        await self._setup_futures_account()

        # Get initial balance
        await self._update_capital()
        self._initial_capital = self._capital
        self._grid_stats._peak_equity = self._capital

        logger.info(f"Initial capital: {self._capital} USDT")

        # Check minimum capital requirement (need at least ~$100 for 0.001 BTC)
        min_trade_value = Decimal("100")  # Approximate for 0.001 BTC at ~$90k
        if self._capital < min_trade_value:
            logger.warning(
                f"Capital ({self._capital} USDT) may be too low to trade. "
                f"Minimum recommended: {min_trade_value} USDT"
            )

        # Load historical data for indicators
        await self._load_historical_data()

        # Initialize grid
        current_price = await self._get_current_price()
        self._initialize_grid(current_price)

        # Check existing position
        await self._sync_position()

        # Start monitoring
        self._monitor_task = asyncio.create_task(self._monitor_loop())

        logger.info("Grid Futures Bot started successfully")

    async def _do_stop(self, clear_position: bool = False) -> None:
        """
        Stop the bot.

        Args:
            clear_position: If True, close any open position before stopping
        """
        logger.info("Stopping Grid Futures Bot")

        # Close position if requested
        if clear_position and self._position:
            current_price = await self._get_current_price()
            await self._close_position(current_price, ExitReason.BOT_STOP)

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Grid Futures Bot stopped")

    async def _do_pause(self) -> None:
        """Pause the bot."""
        logger.info("Pausing Grid Futures Bot")
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

    async def _do_resume(self) -> None:
        """Resume the bot."""
        logger.info("Resuming Grid Futures Bot")
        self._monitor_task = asyncio.create_task(self._monitor_loop())

    async def _do_health_check(self) -> bool:
        """Check bot health."""
        try:
            price = await self._get_current_price()
            return price > 0
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = {
            "bot_id": self._bot_id,
            "bot_type": self.bot_type,
            "symbol": self._config.symbol,
            "state": self._state.value,
            "leverage": self._config.leverage,
            "direction": self._config.direction.value,
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "has_position": self._position is not None,
        }

        if self._grid:
            status["grid"] = {
                "center": str(self._grid.center_price),
                "upper": str(self._grid.upper_price),
                "lower": str(self._grid.lower_price),
                "count": self._grid.grid_count,
                "version": self._grid.version,
            }

        if self._position:
            status["position"] = {
                "side": self._position.side.value,
                "entry_price": str(self._position.entry_price),
                "quantity": str(self._position.quantity),
                "unrealized_pnl": str(self._position.unrealized_pnl),
            }

        status["stats"] = self._grid_stats.to_dict()

        return status

    # =========================================================================
    # Futures Account Setup
    # =========================================================================

    async def _setup_futures_account(self) -> None:
        """Setup futures account (leverage, margin type)."""
        try:
            # Set leverage
            await self._exchange.futures.set_leverage(
                symbol=self._config.symbol,
                leverage=self._config.leverage,
            )
            logger.info(f"Set leverage to {self._config.leverage}x")

            # Set margin type
            try:
                await self._exchange.futures.set_margin_type(
                    symbol=self._config.symbol,
                    margin_type=self._config.margin_type,
                )
                logger.info(f"Set margin type to {self._config.margin_type}")
            except Exception as e:
                # Margin type may already be set
                logger.debug(f"Margin type setting: {e}")

        except Exception as e:
            logger.error(f"Failed to setup futures account: {e}")
            raise

    async def _update_capital(self) -> None:
        """Update available capital."""
        try:
            # Use direct futures API get_balance method
            balance = await self._exchange.futures.get_balance("USDT")
            available = balance.free if balance else Decimal("0")

            if self._config.max_capital:
                self._capital = min(available, self._config.max_capital)
            else:
                self._capital = available

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")

    # =========================================================================
    # Data Loading
    # =========================================================================

    async def _load_historical_data(self) -> None:
        """Load historical klines for indicators."""
        try:
            # Need enough data for trend calculation
            limit = max(self._config.trend_period + 50, 200)

            klines = await self._data_manager.get_klines(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                limit=limit,
            )

            self._klines = klines
            self._closes = [k.close for k in klines]

            logger.info(f"Loaded {len(klines)} historical klines")

        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            raise

    async def _get_current_price(self) -> Decimal:
        """Get current market price."""
        try:
            ticker = await self._exchange.futures.get_ticker(self._config.symbol)
            return ticker.price if ticker else Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return Decimal("0")

    # =========================================================================
    # Indicator Calculations
    # =========================================================================

    def _calculate_sma(self, period: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average."""
        if len(self._closes) < period:
            return None
        return sum(self._closes[-period:]) / Decimal(period)

    def _calculate_atr(self) -> Optional[Decimal]:
        """Calculate Average True Range."""
        period = self._config.atr_period
        if len(self._klines) < period + 1:
            return None

        tr_values = []
        for i in range(1, period + 1):
            idx = -i
            if abs(idx) > len(self._klines) or abs(idx - 1) > len(self._klines):
                break

            high = self._klines[idx].high
            low = self._klines[idx].low
            prev_close = self._klines[idx - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close)
            )
            tr_values.append(tr)

        if not tr_values:
            return None

        return sum(tr_values) / Decimal(len(tr_values))

    def _determine_trend(self, current_price: Decimal) -> int:
        """
        Determine market trend.

        Returns:
            1: Uptrend (price > SMA)
            -1: Downtrend (price < SMA)
            0: Neutral
        """
        if not self._config.use_trend_filter:
            return 0

        sma = self._calculate_sma(self._config.trend_period)
        if sma is None:
            return 0

        diff_pct = (current_price - sma) / sma * Decimal("100")

        if diff_pct > Decimal("1"):  # >1% above SMA
            return 1
        elif diff_pct < Decimal("-1"):  # <1% below SMA
            return -1
        return 0

    def _should_trade_direction(self, direction: str) -> bool:
        """Check if trading in this direction is allowed."""
        if self._config.direction == GridDirection.LONG_ONLY:
            return direction == "long"
        elif self._config.direction == GridDirection.SHORT_ONLY:
            return direction == "short"
        elif self._config.direction == GridDirection.NEUTRAL:
            return True
        elif self._config.direction == GridDirection.TREND_FOLLOW:
            if self._current_trend == 1:
                return direction == "long"
            elif self._current_trend == -1:
                return direction == "short"
            return True  # Neutral trend, allow both
        return True

    # =========================================================================
    # Grid Management
    # =========================================================================

    def _initialize_grid(self, center_price: Decimal) -> None:
        """Initialize grid around center price."""
        atr = self._calculate_atr()

        # Calculate range
        if self._config.use_atr_range and atr:
            range_value = atr * self._config.atr_multiplier
            range_pct = range_value / center_price
            # Clamp between 3% and 20%
            range_pct = min(max(range_pct, Decimal("0.03")), Decimal("0.20"))
        else:
            range_pct = self._config.fallback_range_pct

        upper_price = center_price * (Decimal("1") + range_pct)
        lower_price = center_price * (Decimal("1") - range_pct)
        grid_spacing = (upper_price - lower_price) / Decimal(self._config.grid_count)

        # Create levels
        levels = []
        for i in range(self._config.grid_count + 1):
            price = lower_price + Decimal(i) * grid_spacing
            levels.append(GridLevel(
                index=i,
                price=price,
                state=GridLevelState.EMPTY,
            ))

        version = 1
        if self._grid:
            version = self._grid.version + 1
            self._grid_stats.grid_rebuilds += 1

        self._grid = GridSetup(
            center_price=center_price,
            upper_price=upper_price,
            lower_price=lower_price,
            grid_spacing=grid_spacing,
            levels=levels,
            atr_value=atr,
            range_pct=range_pct,
            version=version,
        )

        logger.info(
            f"Grid initialized: center={center_price:.2f}, "
            f"range=±{range_pct*100:.1f}%, levels={len(levels)}, v{version}"
        )

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid:
            return True

        threshold = Decimal("1") + self._config.rebuild_threshold_pct

        if current_price > self._grid.upper_price * threshold:
            return True
        if current_price < self._grid.lower_price / threshold:
            return True

        return False

    # =========================================================================
    # Position Management
    # =========================================================================

    async def _sync_position(self) -> None:
        """Sync position from exchange."""
        try:
            positions = await self._exchange.futures.get_positions(self._config.symbol)

            for pos in positions:
                if pos.quantity > 0:
                    self._position = FuturesPosition(
                        symbol=self._config.symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        quantity=pos.quantity,
                        leverage=self._config.leverage,
                        unrealized_pnl=pos.unrealized_pnl,
                    )
                    logger.info(f"Synced existing position: {pos.side.value} {pos.quantity}")
                    return

            self._position = None

        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

    async def _open_position(self, side: PositionSide, price: Decimal) -> bool:
        """Open a new position or add to existing."""
        try:
            # Calculate position size
            trade_value = self._capital * self._config.position_size_pct
            quantity = trade_value / price

            # Round quantity to exchange precision (0.001 for BTCUSDT)
            quantity = quantity.quantize(Decimal("0.001"))

            # Minimum order for BTCUSDT is 0.001 BTC
            min_quantity = Decimal("0.001")
            if quantity < min_quantity:
                # Don't log repeatedly - capital is simply too low
                return False

            # Check max position limit
            if self._position:
                current_value = self._position.quantity * price
                if current_value >= self._capital * self._config.max_position_pct:
                    logger.debug("Max position reached, skipping")
                    return False

            # Place market order using convenience methods
            if side == PositionSide.LONG:
                order = await self._exchange.futures.market_buy(
                    symbol=self._config.symbol,
                    quantity=quantity,
                )
            else:
                order = await self._exchange.futures.market_sell(
                    symbol=self._config.symbol,
                    quantity=quantity,
                )

            if order:
                fill_price = order.avg_price if order.avg_price else price
                fill_qty = order.filled_qty

                # Update position
                if self._position and self._position.side == side:
                    # Add to position
                    total_value = (
                        self._position.quantity * self._position.entry_price +
                        fill_qty * fill_price
                    )
                    self._position.quantity += fill_qty
                    self._position.entry_price = total_value / self._position.quantity
                else:
                    # New position (or flip)
                    if self._position:
                        await self._close_position(price, ExitReason.TREND_CHANGE)

                    self._position = FuturesPosition(
                        symbol=self._config.symbol,
                        side=side,
                        entry_price=fill_price,
                        quantity=fill_qty,
                        leverage=self._config.leverage,
                        entry_time=datetime.now(timezone.utc),
                    )

                logger.info(f"Opened {side.value} position: {fill_qty} @ {fill_price}")
                return True

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

        return False

    async def _close_position(
        self,
        current_price: Decimal,
        reason: ExitReason,
        quantity: Optional[Decimal] = None,
    ) -> bool:
        """Close position (full or partial)."""
        if not self._position:
            return False

        try:
            close_qty = quantity or self._position.quantity
            # Round to exchange precision
            close_qty = close_qty.quantize(Decimal("0.001"))

            # Place closing order (reduce_only)
            if self._position.side == PositionSide.LONG:
                order = await self._exchange.futures.market_sell(
                    symbol=self._config.symbol,
                    quantity=close_qty,
                    reduce_only=True,
                )
            else:
                order = await self._exchange.futures.market_buy(
                    symbol=self._config.symbol,
                    quantity=close_qty,
                    reduce_only=True,
                )

            if order:
                fill_price = order.avg_price if order.avg_price else current_price

                # Calculate PnL
                if self._position.side == PositionSide.LONG:
                    pnl = close_qty * (fill_price - self._position.entry_price) * Decimal(self._config.leverage)
                else:
                    pnl = close_qty * (self._position.entry_price - fill_price) * Decimal(self._config.leverage)

                fee = close_qty * fill_price * self._config.fee_rate * 2  # Entry + exit fee

                # Record trade
                trade = GridTrade(
                    trade_id=str(uuid.uuid4())[:8],
                    symbol=self._config.symbol,
                    side=self._position.side,
                    entry_price=self._position.entry_price,
                    exit_price=fill_price,
                    quantity=close_qty,
                    pnl=pnl,
                    fee=fee,
                    leverage=self._config.leverage,
                    entry_time=self._position.entry_time or datetime.now(timezone.utc),
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason,
                    grid_level=0,
                )
                self._grid_stats.record_trade(trade)

                # Update position
                if quantity and quantity < self._position.quantity:
                    self._position.quantity -= close_qty
                else:
                    self._position = None

                logger.info(
                    f"Closed {trade.side.value}: {close_qty} @ {fill_price}, "
                    f"PnL: {pnl:+.2f}, Reason: {reason.value}"
                )

                # Send notification
                if self._notifier:
                    await self._notifier.send_trade_notification(
                        symbol=self._config.symbol,
                        side=trade.side.value,
                        action="CLOSE",
                        price=float(fill_price),
                        quantity=float(close_qty),
                        pnl=float(pnl),
                    )

                return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

        return False

    # =========================================================================
    # Grid Trading Logic
    # =========================================================================

    async def _process_grid(self, current_price: Decimal) -> None:
        """Process grid trading logic."""
        if not self._grid:
            return

        for level in self._grid.levels:
            grid_price = level.price

            # Long entry: price at or below grid level
            if current_price <= grid_price and level.state == GridLevelState.EMPTY:
                if self._should_trade_direction("long"):
                    success = await self._open_position(PositionSide.LONG, current_price)
                    if success:
                        level.state = GridLevelState.LONG_FILLED
                        level.filled_at = datetime.now(timezone.utc)

            # Long exit: price at or above grid level
            elif current_price >= grid_price and level.state == GridLevelState.LONG_FILLED:
                if self._position and self._position.side == PositionSide.LONG:
                    partial_qty = self._position.quantity / Decimal(self._config.grid_count)
                    await self._close_position(current_price, ExitReason.GRID_PROFIT, partial_qty)
                level.state = GridLevelState.EMPTY

            # Short entry: price at or above grid level
            if current_price >= grid_price and level.state == GridLevelState.EMPTY:
                if self._should_trade_direction("short"):
                    success = await self._open_position(PositionSide.SHORT, current_price)
                    if success:
                        level.state = GridLevelState.SHORT_FILLED
                        level.filled_at = datetime.now(timezone.utc)

            # Short exit: price at or below grid level
            elif current_price <= grid_price and level.state == GridLevelState.SHORT_FILLED:
                if self._position and self._position.side == PositionSide.SHORT:
                    partial_qty = self._position.quantity / Decimal(self._config.grid_count)
                    await self._close_position(current_price, ExitReason.GRID_PROFIT, partial_qty)
                level.state = GridLevelState.EMPTY

    # =========================================================================
    # Main Loop
    # =========================================================================

    async def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        logger.info("Starting monitor loop")

        while self._state == BotState.RUNNING:
            try:
                # Get current price
                current_price = await self._get_current_price()
                if current_price <= 0:
                    await asyncio.sleep(5)
                    continue

                # Update closes for indicators
                self._closes.append(current_price)
                if len(self._closes) > 500:
                    self._closes = self._closes[-500:]

                # Update trend
                old_trend = self._current_trend
                self._current_trend = self._determine_trend(current_price)

                # Log trend change
                if old_trend != self._current_trend and self._current_trend != 0:
                    trend_str = "BULLISH" if self._current_trend == 1 else "BEARISH"
                    logger.info(f"Trend changed to {trend_str}")

                # Check if grid needs rebuild
                if self._check_rebuild_needed(current_price):
                    logger.info(f"Price {current_price} out of range, rebuilding grid")

                    # Close position before rebuild
                    if self._position:
                        await self._close_position(current_price, ExitReason.GRID_REBUILD)

                    # Rebuild grid
                    self._initialize_grid(current_price)

                # Process grid logic
                await self._process_grid(current_price)

                # Update capital and drawdown
                await self._update_capital()
                self._grid_stats.update_drawdown(self._capital, self._initial_capital)

                # Update position PnL
                if self._position:
                    self._position.unrealized_pnl = self._position.calculate_pnl(current_price)

                # Wait for next tick (heartbeat is handled by BaseBot)
                await asyncio.sleep(1)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)

        logger.info("Monitor loop stopped")

    # =========================================================================
    # BaseBot Abstract Methods - Extra Status and Health Checks
    # =========================================================================

    def _get_extra_status(self) -> Dict[str, Any]:
        """Return extra status fields specific to Grid Futures bot."""
        extra = {
            "grid": None,
            "position": None,
            "stats": self._grid_stats.to_dict(),
        }

        if self._grid:
            extra["grid"] = {
                "center": str(self._grid.center_price),
                "upper": str(self._grid.upper_price),
                "lower": str(self._grid.lower_price),
                "count": self._grid.grid_count,
                "version": self._grid.version,
            }

        if self._position:
            extra["position"] = {
                "side": self._position.side.value,
                "entry_price": str(self._position.entry_price),
                "quantity": str(self._position.quantity),
                "unrealized_pnl": str(self._position.unrealized_pnl),
            }

        return extra

    async def _extra_health_checks(self) -> Dict[str, bool]:
        """Perform extra health checks specific to Grid Futures bot."""
        checks = {}

        # Check if grid is valid
        checks["grid_valid"] = self._grid is not None

        # Check price is reasonable
        try:
            price = await self._get_current_price()
            checks["price_available"] = price > 0
        except Exception:
            checks["price_available"] = False

        # Check position sync
        if self._position:
            checks["position_synced"] = self._position.quantity > 0
        else:
            checks["position_synced"] = True  # No position is valid

        return checks

