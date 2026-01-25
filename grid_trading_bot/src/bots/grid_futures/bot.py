"""
Grid Futures Bot - Futures-based Grid Trading with Leverage.

Features:
- Leverage trading (1x-125x)
- Bidirectional trading (long/short)
- Trend-following mode
- Dynamic ATR-based grid range
- Automatic grid rebuilding
- Kline-based grid detection (與回測一致)

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割):
- Walk-Forward 一致性: 100% (8/8 時段獲利)
- 報酬: +123.9% (2 年), 年化 +62.0%
- Sharpe: 4.50, 最大回撤: 3.5%

驗證參數:
- leverage: 2x
- grid_count: 10
- trend_period: 20
- atr_multiplier: 3.0

進場邏輯 (與回測一致):
- Long entry: K 線低點觸及 grid level (買跌)
- Short entry: K 線高點觸及 grid level (賣漲)
- 使用 grid level 價格進場（非市場價格）
"""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import Kline, MarketType, OrderSide
from typing import Callable
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.fund_manager import SignalCoordinator, SignalDirection, CoordinationResult
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
        heartbeat_callback: Optional[callable] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
    ):
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier

        # Signal Coordinator for multi-bot conflict prevention
        self._signal_coordinator = signal_coordinator or SignalCoordinator.get_instance()

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

        # Signal cooldown to prevent signal stacking
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = 2  # Minimum bars between signals

        # Slippage tracking
        self._slippage_records: List[Dict] = []
        self._max_slippage_records: int = 100

        # Hysteresis: track last triggered level to prevent oscillation
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = Decimal("0.002")  # 0.2% buffer zone

        # Tasks
        self._monitor_task: Optional[asyncio.Task] = None

        # Kline subscription callback
        self._kline_callback: Optional[Callable] = None

        # State persistence
        self._save_task: Optional[asyncio.Task] = None
        self._save_interval_minutes: int = 5

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

        # Subscribe to kline updates (使用 K 線高低點檢測，與回測一致)
        def on_kline_sync(kline: Kline) -> None:
            asyncio.create_task(self._on_kline(kline))

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )

        # Start background monitoring (capital update, drawdown tracking)
        self._monitor_task = asyncio.create_task(self._background_monitor())

        # Start periodic state saving
        self._start_save_task()

        logger.info("Grid Futures Bot started successfully")
        logger.info(f"  Subscribed to {self._config.symbol} {self._config.timeframe} klines")

    async def _do_stop(self, clear_position: bool = False) -> None:
        """
        Stop the bot.

        Args:
            clear_position: If True, close any open position before stopping
        """
        logger.info("Stopping Grid Futures Bot")

        # Unsubscribe from kline updates
        try:
            await self._data_manager.klines.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                callback=self._kline_callback,
                market_type=MarketType.FUTURES,
            )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from klines: {e}")

        # Close position if requested
        if clear_position and self._position:
            current_price = await self._get_current_price()
            await self._close_position(current_price, ExitReason.BOT_STOP)
        elif self._position and self._position.stop_loss_order_id:
            # Cancel stop loss order but keep position
            await self._cancel_stop_loss_order()

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Stop periodic save task and save final state
        self._stop_save_task()
        await self._save_state()

        logger.info("Grid Futures Bot stopped")

    async def _do_pause(self) -> None:
        """Pause the bot."""
        logger.info("Pausing Grid Futures Bot")

        # Unsubscribe from kline updates
        try:
            await self._data_manager.klines.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                callback=self._kline_callback,
                market_type=MarketType.FUTURES,
            )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe: {e}")

        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Grid Futures Bot paused")

    async def _do_resume(self) -> None:
        """Resume the bot."""
        logger.info("Resuming Grid Futures Bot")

        # Re-subscribe to kline updates
        def on_kline_sync(kline: Kline) -> None:
            asyncio.create_task(self._on_kline(kline))

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )

        # Restart background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor())
        logger.info("Grid Futures Bot resumed")

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
                "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                "stop_loss_order_id": self._position.stop_loss_order_id,
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

    async def _on_capital_updated(self, new_max_capital: Decimal) -> None:
        """
        Handle capital update from FundManager.

        When capital allocation changes:
        1. Update internal capital tracking
        2. Log the change for monitoring
        3. Grid will adjust automatically on next rebuild cycle

        Args:
            new_max_capital: New maximum capital allocation
        """
        previous_capital = self._capital

        # Update capital with new limit
        await self._update_capital()

        logger.info(
            f"[FundManager] Capital updated for {self._bot_id}: "
            f"max_capital={new_max_capital}, "
            f"actual_capital: {previous_capital} -> {self._capital}"
        )

        # Note: Grid position sizing will automatically use new capital
        # on next trade. No immediate grid rebuild needed as existing
        # positions should continue with current sizing.

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

    def _is_valid_decimal(self, value: Optional[Decimal]) -> bool:
        """Check if a Decimal value is valid (not None, NaN, or Inf)."""
        if value is None:
            return False
        try:
            return not (value.is_nan() or value.is_infinite())
        except (AttributeError, TypeError):
            return False

    def _calculate_sma(self, period: int) -> Optional[Decimal]:
        """Calculate Simple Moving Average with NaN/Inf protection."""
        if len(self._closes) < period:
            return None

        try:
            # Filter valid values
            valid_closes = [c for c in self._closes[-period:] if self._is_valid_decimal(c)]
            if len(valid_closes) < period:
                return None

            result = sum(valid_closes) / Decimal(period)
            if not self._is_valid_decimal(result):
                logger.warning(f"Invalid SMA result: {result}")
                return None
            return result
        except Exception as e:
            logger.warning(f"SMA calculation error: {e}")
            return None

    def _calculate_atr(self) -> Optional[Decimal]:
        """Calculate Average True Range with NaN/Inf protection."""
        period = self._config.atr_period
        if len(self._klines) < period + 1:
            return None

        try:
            tr_values = []
            for i in range(1, period + 1):
                idx = -i
                if abs(idx) > len(self._klines) or abs(idx - 1) > len(self._klines):
                    break

                high = self._klines[idx].high
                low = self._klines[idx].low
                prev_close = self._klines[idx - 1].close

                # Validate inputs
                if not all(self._is_valid_decimal(v) for v in [high, low, prev_close]):
                    continue

                tr = max(
                    high - low,
                    abs(high - prev_close),
                    abs(low - prev_close)
                )

                if self._is_valid_decimal(tr):
                    tr_values.append(tr)

            if not tr_values:
                return None

            result = sum(tr_values) / Decimal(len(tr_values))
            if not self._is_valid_decimal(result):
                logger.warning(f"Invalid ATR result: {result}")
                return None
            return result
        except Exception as e:
            logger.warning(f"ATR calculation error: {e}")
            return None

    def _determine_trend(self, current_price: Decimal) -> int:
        """
        Determine market trend with NaN/Inf protection.

        Returns:
            1: Uptrend (price > SMA)
            -1: Downtrend (price < SMA)
            0: Neutral
        """
        if not self._config.use_trend_filter:
            return 0

        sma = self._calculate_sma(self._config.trend_period)
        if sma is None or sma == 0:
            return 0

        if not self._is_valid_decimal(current_price):
            return 0

        try:
            diff_pct = (current_price - sma) / sma * Decimal("100")

            if not self._is_valid_decimal(diff_pct):
                return 0

            if diff_pct > Decimal("1"):  # >1% above SMA
                return 1
            elif diff_pct < Decimal("-1"):  # <1% below SMA
                return -1
            return 0
        except Exception:
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
                    # pos.side is a string due to Pydantic's use_enum_values=True
                    # Convert to local PositionSide enum
                    side_str = pos.side if isinstance(pos.side, str) else pos.side.value
                    local_side = PositionSide(side_str)

                    self._position = FuturesPosition(
                        symbol=self._config.symbol,
                        side=local_side,
                        entry_price=pos.entry_price,
                        quantity=pos.quantity,
                        leverage=self._config.leverage,
                        unrealized_pnl=pos.unrealized_pnl,
                    )
                    logger.info(f"Synced existing position: {side_str} {pos.quantity}")
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

            # Check signal coordinator for multi-bot conflicts
            if self._signal_coordinator:
                signal_dir = SignalDirection.LONG if side == PositionSide.LONG else SignalDirection.SHORT
                result = await self._signal_coordinator.request_signal(
                    bot_id=self._bot_id,
                    symbol=self._config.symbol,
                    direction=signal_dir,
                    quantity=quantity,
                    price=price,
                    reason=f"Grid level entry",
                )
                if not result.approved:
                    logger.warning(
                        f"Signal blocked by coordinator: {result.message} "
                        f"(conflict with {result.conflicting_bot})"
                    )
                    return False

            # Place market order (through order queue for cross-bot coordination)
            if side == PositionSide.LONG:
                order = await self._exchange.market_buy(
                    symbol=self._config.symbol,
                    quantity=quantity,
                    market=MarketType.FUTURES,
                    bot_id=self._bot_id,
                )
            else:
                order = await self._exchange.market_sell(
                    symbol=self._config.symbol,
                    quantity=quantity,
                    market=MarketType.FUTURES,
                    bot_id=self._bot_id,
                )

            if order:
                fill_price = order.avg_price if order.avg_price else price
                fill_qty = order.filled_qty

                # Record slippage for backtest vs live comparison
                slippage = fill_price - price
                slippage_pct = (slippage / price * Decimal("100")) if price > 0 else Decimal("0")
                self._record_slippage(
                    expected_price=price,
                    actual_price=fill_price,
                    slippage=slippage,
                    slippage_pct=slippage_pct,
                    side=side.value,
                    quantity=fill_qty,
                )

                # Update position
                if self._position and self._position.side == side:
                    # Add to position (DCA)
                    total_value = (
                        self._position.quantity * self._position.entry_price +
                        fill_qty * fill_price
                    )
                    self._position.quantity += fill_qty
                    self._position.entry_price = total_value / self._position.quantity

                    # Update stop loss order with new quantity and average price
                    if self._config.use_exchange_stop_loss:
                        await self._update_stop_loss_order()
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

                    # Place exchange stop loss order
                    if self._config.use_exchange_stop_loss:
                        await self._place_stop_loss_order()

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

            # Cancel stop loss order if closing full position
            is_full_close = (quantity is None or quantity >= self._position.quantity)
            if is_full_close and self._position.stop_loss_order_id:
                await self._cancel_stop_loss_order()

            # Place closing order (reduce_only, through order queue)
            close_side = "SELL" if self._position.side == PositionSide.LONG else "BUY"
            order = await self._exchange.futures_create_order(
                symbol=self._config.symbol,
                side=close_side,
                order_type="MARKET",
                quantity=close_qty,
                reduce_only=True,
                bot_id=self._bot_id,
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
                    # Update stop loss for reduced position
                    if self._config.use_exchange_stop_loss and self._position.stop_loss_order_id:
                        await self._update_stop_loss_order()
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
    # Kline Event Handler (與回測一致的進場邏輯)
    # =========================================================================

    async def _on_kline(self, kline: Kline) -> None:
        """
        Process kline update - main trading logic.

        使用 K 線高低點檢測網格觸發，與回測策略一致:
        - Long entry: kline.low 觸及 grid level
        - Short entry: kline.high 觸及 grid level
        - 使用 grid level 價格進場（非市場價格）

        Note: Only processes closed klines to match backtest behavior.
        """
        # Validate kline before processing (matches backtest behavior)
        if not self._should_process_kline(kline, require_closed=True, check_symbol=False):
            return

        try:
            current_price = kline.close
            kline_low = kline.low
            kline_high = kline.high

            # Decrement signal cooldown
            if self._signal_cooldown > 0:
                self._signal_cooldown -= 1

            # Update closes for indicators
            self._closes.append(current_price)
            if len(self._closes) > 500:
                self._closes = self._closes[-500:]

            # Update trend
            old_trend = self._current_trend
            self._current_trend = self._determine_trend(current_price)

            if old_trend != self._current_trend and self._current_trend != 0:
                trend_str = "BULLISH" if self._current_trend == 1 else "BEARISH"
                logger.info(f"Trend changed to {trend_str}")

            # Check if grid needs rebuild
            if self._check_rebuild_needed(current_price):
                logger.info(f"Price {current_price} out of range, rebuilding grid")
                if self._position:
                    await self._close_position(current_price, ExitReason.GRID_REBUILD)
                self._initialize_grid(current_price)
                return

            # Process grid logic with kline high/low (與回測一致)
            await self._process_grid_kline(kline_low, kline_high, current_price)

            # Update position PnL and check stop loss
            if self._position:
                self._position.unrealized_pnl = self._position.calculate_pnl(current_price)
                if await self._check_stop_loss(current_price):
                    logger.warning(f"Stop loss triggered at {current_price}")

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    # =========================================================================
    # Grid Trading Logic (與回測一致)
    # =========================================================================

    def _check_hysteresis(self, level_index: int, direction: str, grid_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if price has moved enough from last triggered level (hysteresis).

        Prevents oscillation around grid levels by requiring price to move
        away by a minimum percentage before retriggering the same level.

        Args:
            level_index: Grid level index
            direction: "long" or "short"
            grid_price: Grid level price
            current_price: Current market price

        Returns:
            True if entry is allowed (passed hysteresis check)
        """
        # First trigger is always allowed
        if self._last_triggered_level is None:
            return True

        # Different level is always allowed
        if self._last_triggered_level != level_index:
            return True

        # Same level - check if price moved enough (hysteresis buffer)
        hysteresis_buffer = grid_price * self._hysteresis_pct

        if direction == "long":
            # For long, price should have moved up before coming back down
            if current_price < grid_price - hysteresis_buffer:
                return True
        else:  # short
            # For short, price should have moved down before coming back up
            if current_price > grid_price + hysteresis_buffer:
                return True

        logger.debug(
            f"Hysteresis active: level {level_index} recently triggered, "
            f"price {current_price} too close to {grid_price}"
        )
        return False

    async def _process_grid_kline(
        self,
        kline_low: Decimal,
        kline_high: Decimal,
        current_price: Decimal,
    ) -> None:
        """
        Process grid trading logic using kline high/low.

        與回測策略一致的進場邏輯:
        - Long entry: K 線低點觸及 grid level (買跌)
        - Short entry: K 線高點觸及 grid level (賣漲)
        - 使用 grid level 價格進場

        Includes hysteresis to prevent oscillation around grid levels.

        Args:
            kline_low: K 線最低價
            kline_high: K 線最高價
            current_price: K 線收盤價
        """
        if not self._grid:
            return

        for i, level in enumerate(self._grid.levels):
            grid_price = level.price

            # Long entry: K 線低點觸及 grid level (買跌，與回測一致)
            if level.state == GridLevelState.EMPTY and kline_low <= grid_price:
                # Check signal cooldown to prevent signal stacking
                if self._signal_cooldown > 0:
                    logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping long entry")
                    continue

                # Check hysteresis to prevent oscillation
                if not self._check_hysteresis(i, "long", grid_price, current_price):
                    continue

                if self._should_trade_direction("long"):
                    # 使用 grid level 價格進場（與回測一致）
                    success = await self._open_position(PositionSide.LONG, grid_price)
                    if success:
                        level.state = GridLevelState.LONG_FILLED
                        level.filled_at = datetime.now(timezone.utc)
                        self._signal_cooldown = self._cooldown_bars  # Reset cooldown
                        self._last_triggered_level = i  # Track for hysteresis
                        logger.info(f"Long entry at grid level {i}: {grid_price}")

            # Long exit: K 線高點觸及 grid level
            elif level.state == GridLevelState.LONG_FILLED and kline_high >= grid_price:
                if self._position and self._position.side == PositionSide.LONG:
                    filled_long_count = sum(1 for lv in self._grid.levels if lv.state == GridLevelState.LONG_FILLED)
                    if filled_long_count > 0:
                        partial_qty = self._position.quantity / Decimal(filled_long_count)
                    else:
                        partial_qty = self._position.quantity
                    await self._close_position(grid_price, ExitReason.GRID_PROFIT, partial_qty)
                level.state = GridLevelState.EMPTY
                # Clear hysteresis on exit to allow fresh entry
                if self._last_triggered_level == i:
                    self._last_triggered_level = None

            # Short entry: K 線高點觸及 grid level (賣漲，與回測一致)
            if level.state == GridLevelState.EMPTY and kline_high >= grid_price:
                # Check signal cooldown to prevent signal stacking
                if self._signal_cooldown > 0:
                    logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping short entry")
                    continue

                # Check hysteresis to prevent oscillation
                if not self._check_hysteresis(i, "short", grid_price, current_price):
                    continue

                if self._should_trade_direction("short"):
                    # 使用 grid level 價格進場（與回測一致）
                    success = await self._open_position(PositionSide.SHORT, grid_price)
                    if success:
                        level.state = GridLevelState.SHORT_FILLED
                        level.filled_at = datetime.now(timezone.utc)
                        self._signal_cooldown = self._cooldown_bars  # Reset cooldown
                        self._last_triggered_level = i  # Track for hysteresis
                        logger.info(f"Short entry at grid level {i}: {grid_price}")

            # Short exit: K 線低點觸及 grid level
            elif level.state == GridLevelState.SHORT_FILLED and kline_low <= grid_price:
                if self._position and self._position.side == PositionSide.SHORT:
                    filled_short_count = sum(1 for lv in self._grid.levels if lv.state == GridLevelState.SHORT_FILLED)
                    if filled_short_count > 0:
                        partial_qty = self._position.quantity / Decimal(filled_short_count)
                    else:
                        partial_qty = self._position.quantity
                    await self._close_position(grid_price, ExitReason.GRID_PROFIT, partial_qty)
                level.state = GridLevelState.EMPTY
                # Clear hysteresis on exit to allow fresh entry
                if self._last_triggered_level == i:
                    self._last_triggered_level = None

    # =========================================================================
    # Stop Loss Check
    # =========================================================================

    # =========================================================================
    # Slippage Tracking
    # =========================================================================

    def _record_slippage(
        self,
        expected_price: Decimal,
        actual_price: Decimal,
        slippage: Decimal,
        slippage_pct: Decimal,
        side: str,
        quantity: Decimal,
    ) -> None:
        """
        Record slippage for monitoring backtest vs live discrepancy.

        Args:
            expected_price: Grid level price (same as backtest)
            actual_price: Actual fill price from exchange
            slippage: Absolute slippage (actual - expected)
            slippage_pct: Percentage slippage
            side: Trade side (long/short)
            quantity: Trade quantity
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "expected_price": str(expected_price),
            "actual_price": str(actual_price),
            "slippage": str(slippage),
            "slippage_pct": str(slippage_pct),
            "side": side,
            "quantity": str(quantity),
        }
        self._slippage_records.append(record)

        # Trim old records
        if len(self._slippage_records) > self._max_slippage_records:
            self._slippage_records = self._slippage_records[-self._max_slippage_records:]

        # Log significant slippage
        if abs(slippage_pct) > Decimal("0.1"):  # > 0.1%
            logger.warning(
                f"Significant slippage: {slippage_pct:.4f}% "
                f"(expected {expected_price}, got {actual_price})"
            )
        else:
            logger.debug(f"Slippage: {slippage_pct:.4f}%")

    def get_slippage_stats(self) -> Dict:
        """Get slippage statistics for monitoring."""
        if not self._slippage_records:
            return {"count": 0, "avg_pct": "0", "max_pct": "0"}

        slippages = [Decimal(r["slippage_pct"]) for r in self._slippage_records]
        return {
            "count": len(slippages),
            "avg_pct": str(sum(slippages) / len(slippages)),
            "max_pct": str(max(abs(s) for s in slippages)),
            "min_pct": str(min(slippages)),
            "recent": self._slippage_records[-5:],
        }

    # =========================================================================
    # Stop Loss Check
    # =========================================================================

    async def _check_stop_loss(self, current_price: Decimal) -> bool:
        """
        Check if stop loss should be triggered.

        Returns:
            True if position was closed due to stop loss
        """
        if not self._position:
            return False

        # Calculate PnL percentage
        if self._position.side == PositionSide.LONG:
            pnl_pct = (current_price - self._position.entry_price) / self._position.entry_price
        else:
            pnl_pct = (self._position.entry_price - current_price) / self._position.entry_price

        # Check if loss exceeds stop loss threshold
        if pnl_pct < -self._config.stop_loss_pct:
            logger.warning(
                f"Stop loss triggered: PnL {pnl_pct*100:.2f}% < -{self._config.stop_loss_pct*100:.1f}%"
            )
            await self._close_position(current_price, ExitReason.STOP_LOSS)
            return True

        return False

    # =========================================================================
    # Exchange Stop Loss
    # =========================================================================

    def _calculate_stop_loss_price(self, entry_price: Decimal, side: PositionSide) -> Decimal:
        """Calculate stop loss price based on entry and config."""
        if side == PositionSide.LONG:
            stop_price = entry_price * (Decimal("1") - self._config.stop_loss_pct)
        else:
            stop_price = entry_price * (Decimal("1") + self._config.stop_loss_pct)
        # Round to tick size
        return stop_price.quantize(Decimal("0.1"))

    async def _place_stop_loss_order(self) -> None:
        """Place stop loss order on exchange using Algo Order API."""
        if not self._position:
            return

        try:
            # Calculate stop loss price
            stop_price = self._calculate_stop_loss_price(
                self._position.entry_price,
                self._position.side
            )

            # Determine close side (opposite of position)
            if self._position.side == PositionSide.LONG:
                close_side = OrderSide.SELL
            else:
                close_side = OrderSide.BUY

            # Place STOP_MARKET order (through order queue)
            sl_order = await self._exchange.futures_create_order(
                symbol=self._config.symbol,
                side=close_side.value,  # Convert enum to string
                order_type="STOP_MARKET",
                quantity=self._position.quantity,
                stop_price=stop_price,
                reduce_only=True,
                bot_id=self._bot_id,
            )

            if sl_order:
                self._position.stop_loss_order_id = str(sl_order.order_id)
                self._position.stop_loss_price = stop_price
                logger.info(
                    f"Stop loss order placed: {close_side.value} {self._position.quantity} "
                    f"@ {stop_price}, ID={sl_order.order_id}"
                )

        except Exception as e:
            logger.error(f"Failed to place stop loss order: {e}")

    async def _update_stop_loss_order(self) -> None:
        """Update stop loss order when position changes (e.g., DCA)."""
        if not self._position or not self._config.use_exchange_stop_loss:
            return

        # Cancel existing stop loss
        await self._cancel_stop_loss_order()

        # Place new stop loss with updated entry price
        await self._place_stop_loss_order()

    async def _cancel_stop_loss_order(self) -> None:
        """Cancel stop loss order on exchange using Algo Order API."""
        if not self._position or not self._position.stop_loss_order_id:
            return

        try:
            # Cancel using Algo Order API (required since 2025-12-09)
            await self._exchange.futures.cancel_algo_order(
                symbol=self._config.symbol,
                algo_id=self._position.stop_loss_order_id,
            )
            logger.info(f"Stop loss order cancelled: {self._position.stop_loss_order_id}")
            self._position.stop_loss_order_id = None
            self._position.stop_loss_price = None

        except Exception as e:
            logger.debug(f"Failed to cancel stop loss order: {e}")

    # =========================================================================
    # Background Monitor (資金更新、回撤追蹤)
    # =========================================================================

    async def _background_monitor(self) -> None:
        """
        Background monitoring loop for capital updates and drawdown tracking.

        Note: Trading logic is now handled by _on_kline callback.
        """
        logger.info("Starting background monitor")

        while self._state == BotState.RUNNING:
            try:
                # Update capital and drawdown periodically
                await self._update_capital()
                self._grid_stats.update_drawdown(self._capital, self._initial_capital)

                # Wait 30 seconds between updates
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background monitor: {e}")
                await asyncio.sleep(30)

        logger.info("Background monitor stopped")

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

    # =========================================================================
    # State Persistence
    # =========================================================================

    async def _save_state(self) -> None:
        """Save bot state to database."""
        try:
            config = {
                "symbol": self._config.symbol,
                "grid_count": self._config.grid_count,
                "leverage": self._config.leverage,
                "direction": self._config.direction.value,
                "trend_period": self._config.trend_period,
                "atr_multiplier": str(self._config.atr_multiplier),
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            state_data = {
                "capital": str(self._capital),
                "initial_capital": str(self._initial_capital),
                "current_trend": self._current_trend,
                "stats": {
                    "total_trades": self._grid_stats.total_trades,
                    "winning_trades": self._grid_stats.winning_trades,
                    "losing_trades": self._grid_stats.losing_trades,
                    "total_pnl": str(self._grid_stats.total_pnl),
                    "max_drawdown_pct": str(self._grid_stats.max_drawdown_pct),
                },
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }

            if self._grid:
                state_data["grid"] = {
                    "center_price": str(self._grid.center_price),
                    "upper_price": str(self._grid.upper_price),
                    "lower_price": str(self._grid.lower_price),
                    "grid_count": self._grid.grid_count,
                    "version": self._grid.version,
                }

            if self._position:
                state_data["position"] = {
                    "side": self._position.side.value,
                    "entry_price": str(self._position.entry_price),
                    "quantity": str(self._position.quantity),
                    "entry_time": self._position.entry_time.isoformat() if self._position.entry_time else None,
                    "stop_loss_order_id": self._position.stop_loss_order_id,
                    "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                }

            await self._data_manager.save_bot_state(
                bot_id=self._bot_id,
                bot_type="grid_futures",
                status=self._state.value,
                config=config,
                state_data=state_data,
            )

            logger.debug(f"Bot state saved: {self._bot_id}")

        except Exception as e:
            logger.warning(f"Failed to save bot state: {e}")

    def _start_save_task(self) -> None:
        """Start periodic save task."""
        if self._save_task is not None:
            return

        async def save_loop():
            while self._running:
                await asyncio.sleep(self._save_interval_minutes * 60)
                if self._running:
                    await self._save_state()

        self._save_task = asyncio.create_task(save_loop())

    def _stop_save_task(self) -> None:
        """Stop periodic save task."""
        if self._save_task:
            self._save_task.cancel()
            self._save_task = None

    @classmethod
    async def restore(
        cls,
        bot_id: str,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: Optional[NotificationManager] = None,
    ) -> Optional["GridFuturesBot"]:
        """
        Restore a GridFuturesBot from saved state.

        Args:
            bot_id: Bot ID to restore
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance

        Returns:
            Restored GridFuturesBot or None if not found
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            config_data = state_data.get("config", {})
            config = GridFuturesConfig(
                symbol=config_data.get("symbol", ""),
                grid_count=config_data.get("grid_count", 10),
                leverage=config_data.get("leverage", 2),
                direction=GridDirection(config_data.get("direction", "both")),
                trend_period=config_data.get("trend_period", 20),
                atr_multiplier=Decimal(config_data.get("atr_multiplier", "3.0")),
                max_capital=Decimal(config_data["max_capital"]) if config_data.get("max_capital") else None,
            )

            bot = cls(
                bot_id=bot_id,
                config=config,
                exchange=exchange,
                data_manager=data_manager,
                notifier=notifier,
            )

            # Restore state
            saved_state = state_data.get("state_data", {})
            bot._capital = Decimal(saved_state.get("capital", "0"))
            bot._initial_capital = Decimal(saved_state.get("initial_capital", "0"))
            bot._current_trend = saved_state.get("current_trend", 0)

            # Restore stats
            stats_data = saved_state.get("stats", {})
            bot._grid_stats.total_trades = stats_data.get("total_trades", 0)
            bot._grid_stats.winning_trades = stats_data.get("winning_trades", 0)
            bot._grid_stats.losing_trades = stats_data.get("losing_trades", 0)
            bot._grid_stats.total_pnl = Decimal(stats_data.get("total_pnl", "0"))
            bot._grid_stats.max_drawdown_pct = Decimal(stats_data.get("max_drawdown_pct", "0"))

            # Restore grid
            grid_data = saved_state.get("grid")
            if grid_data:
                bot._grid = GridSetup(
                    symbol=config.symbol,
                    center_price=Decimal(grid_data["center_price"]),
                    upper_price=Decimal(grid_data["upper_price"]),
                    lower_price=Decimal(grid_data["lower_price"]),
                    grid_count=grid_data["grid_count"],
                    levels=[],  # Will be recalculated on start
                    version=grid_data.get("version", 1),
                )

            # Restore position
            position_data = saved_state.get("position")
            if position_data:
                bot._position = FuturesPosition(
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    leverage=config.leverage,
                    entry_time=datetime.fromisoformat(position_data["entry_time"]) if position_data.get("entry_time") else None,
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                )

            logger.info(f"Restored GridFuturesBot: {bot_id}, PnL={bot._grid_stats.total_pnl}")
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot: {e}")
            return None

