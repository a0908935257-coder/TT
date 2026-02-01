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
import time
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

        # Signal cooldown to prevent signal stacking (use config values)
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = self._config.cooldown_bars

        # Slippage tracking (uses BaseBot's enhanced tracking)
        self._init_slippage_tracking()

        # Hysteresis: track last triggered level to prevent oscillation (use config values)
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = self._config.hysteresis_pct

        # Bar counting for timeout exit
        self._current_bar: int = 0
        self._entry_bar: int = 0

        # Tasks
        self._monitor_task: Optional[asyncio.Task] = None

        # Kline subscription callback
        self._kline_callback: Optional[Callable] = None

        # Kline callback tasks tracking (prevent memory leak)
        self._kline_tasks: set[asyncio.Task] = set()

        # State persistence
        self._save_task: Optional[asyncio.Task] = None
        self._save_interval_minutes: int = 5

        # Initialize state lock for concurrent protection
        self._init_state_lock()

        # Data health tracking (for stale/gap detection)
        self._init_data_health_tracking()
        self._prev_kline: Optional[Kline] = None
        self._interval_seconds: int = self._parse_interval_to_seconds(config.timeframe)

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

        # Initialize per-strategy risk tracking (風控相互影響隔離)
        self.set_strategy_initial_capital(self._capital)

        # Register for global risk tracking (多策略風控協調)
        await self.register_bot_for_global_risk(self._bot_id, self._capital)

        # Register for circuit breaker coordination (部分熔斷協調)
        await self.register_strategy_for_cb(
            bot_id=self._bot_id,
            strategy_type="grid_futures",
            dependencies=None,  # No dependencies on other strategies
        )

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

        # Initialize grid (skip if restored from saved state)
        if not self._grid:
            current_price = await self._get_current_price()
            self._initialize_grid(current_price)

        # Check existing position
        await self._sync_position()

        # Subscribe to kline updates (使用 K 線高低點檢測，與回測一致)
        def on_kline_sync(kline: Kline) -> None:
            task = asyncio.create_task(self._on_kline(kline))
            self._kline_tasks.add(task)
            task.add_done_callback(self._on_kline_task_done)

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )

        # Start background monitoring (capital update, drawdown tracking)
        self._monitor_task = asyncio.create_task(self._background_monitor_safe())

        # Start periodic state saving
        self._start_save_task()

        # Start position reconciliation (detect manual operations)
        self._start_position_reconciliation()

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

        # Stop position reconciliation
        self._stop_position_reconciliation()

        # Cancel and cleanup kline callback tasks
        for task in list(self._kline_tasks):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._kline_tasks.clear()

        # Stop periodic save task and save final state
        await self._stop_save_task()
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
            task = asyncio.create_task(self._on_kline(kline))
            self._kline_tasks.add(task)
            task.add_done_callback(self._on_kline_task_done)

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )

        # Restart background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor_safe())
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
            # Validate leverage range before sending to exchange
            if not (1 <= self._config.leverage <= 125):
                raise ValueError(
                    f"Leverage {self._config.leverage}x out of valid range [1, 125]"
                )

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

        # Guard against division by zero
        if self._config.grid_count <= 0:
            grid_spacing = upper_price - lower_price
        else:
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
            # Validate price before calculation (indicator boundary check)
            if not self._validate_price(price, "entry_price"):
                logger.warning(f"Invalid entry price: {price}")
                return False

            # Calculate position size (notional = margin × leverage)
            trade_value = self._capital * self._config.position_size_pct * Decimal(self._config.leverage)
            quantity = self._safe_divide(trade_value, price, context="position_size")

            # Validate and normalize price/quantity to exchange requirements
            order_side = "BUY" if side == PositionSide.LONG else "SELL"
            is_valid, norm_price, norm_quantity, precision_msg = await self._validate_order_precision(
                symbol=self._config.symbol,
                price=price,
                quantity=quantity,
            )
            if not is_valid:
                logger.warning(f"Order precision validation failed: {precision_msg}")
                return False

            # Use normalized values
            quantity = norm_quantity

            # Check for duplicate order (prevent double-entry on retry)
            if self._is_duplicate_order(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
            ):
                logger.warning(f"Duplicate order blocked: {order_side} {quantity}")
                return False

            # Pre-trade validation (time sync + data health + position reconciliation)
            if not await self._validate_pre_trade(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
                check_time_sync=True,
                check_liquidity=False,  # Grid bots use small sizes
            ):
                return False

            # Network health check (網路彈性檢查)
            network_ok, network_reason = await self.check_network_before_trade()
            if not network_ok:
                logger.warning(f"Network check failed: {network_reason}")
                return False

            # SSL certificate check (SSL 證書檢查)
            ssl_ok, ssl_reason = await self.check_ssl_before_trade()
            if not ssl_ok:
                logger.warning(f"SSL check failed: {ssl_reason}")
                return False

            # Check entry allowed (circuit breaker, cooldown, oscillation prevention)
            entry_allowed, entry_reason = self.check_entry_allowed()
            if not entry_allowed:
                logger.warning(f"Entry blocked: {entry_reason}")
                return False

            # Check balance before order (prevent rejection)
            balance_ok, balance_msg = await self._check_balance_for_order(
                symbol=self._config.symbol,
                quantity=quantity,
                price=price,
                leverage=self._config.leverage,
            )
            if not balance_ok:
                logger.warning(f"Order blocked: {balance_msg}")
                return False

            # Apply position size reduction from oscillation prevention
            size_mult = self.get_position_size_reduction()
            if size_mult < Decimal("1.0"):
                quantity = (quantity * size_mult).quantize(Decimal("0.001"))
                logger.info(f"Position size reduced to {size_mult*100:.0f}%: {quantity}")
                if quantity <= 0:
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

            # Pre-trade risk check (risk gate + global risk limits)
            is_allowed, ptc_msg, ptc_details = await self.pre_trade_with_global_check(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
                price=price,
            )
            if not is_allowed:
                logger.warning(f"[{self._bot_id}] Pre-trade check rejected: {ptc_msg}")
                return False

            # Mark order as pending (for deduplication)
            order_key = self._mark_order_pending(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
            )

            # Place market order with timeout protection
            async def place_order():
                if side == PositionSide.LONG:
                    return await self._exchange.market_buy(
                        symbol=self._config.symbol,
                        quantity=quantity,
                        market=MarketType.FUTURES,
                        bot_id=self._bot_id,
                    )
                else:
                    return await self._exchange.market_sell(
                        symbol=self._config.symbol,
                        quantity=quantity,
                        market=MarketType.FUTURES,
                        bot_id=self._bot_id,
                    )

            order = await self._place_order_with_timeout(
                place_order,
                order_side=order_side,
                order_quantity=quantity,
            )

            # Clear pending order marker
            self._clear_pending_order(order_key)

            if order:
                order_id = str(getattr(order, "order_id", ""))

                # Confirm fill with polling fallback (handles lost notifications)
                if order_id:
                    is_confirmed, fill_data = await self._confirm_fill_with_polling(
                        order_id=order_id,
                        symbol=self._config.symbol,
                        expected_quantity=quantity,
                        max_wait_seconds=30,  # 30 second timeout for market orders
                    )

                    if is_confirmed and fill_data:
                        # Use confirmed fill data
                        fill_price = Decimal(fill_data.get("avg_price", str(price)))
                        fill_qty = Decimal(fill_data.get("filled_qty", str(order.filled_qty or quantity)))
                    else:
                        # Fallback to order response data
                        fill_price = order.avg_price if order.avg_price else price
                        fill_qty = order.filled_qty if order.filled_qty else quantity
                        logger.warning(
                            f"Fill confirmation failed for {order_id}, using order response data"
                        )
                else:
                    fill_price = order.avg_price if order.avg_price else price
                    fill_qty = order.filled_qty if order.filled_qty else quantity

                # Check for partial fill
                if fill_qty < quantity:
                    fill_result = await self._handle_partial_fill(
                        order_id=str(getattr(order, "order_id", "")),
                        symbol=self._config.symbol,
                        expected_quantity=quantity,
                        filled_quantity=fill_qty,
                        avg_price=fill_price,
                    )
                    if not fill_result["is_acceptable"]:
                        logger.warning(f"Partial fill not acceptable: {fill_result}")
                        return False

                # Record and check slippage (using BaseBot method)
                slippage_pct = self._record_slippage(
                    expected_price=price,
                    actual_price=fill_price,
                    side=order_side,
                    quantity=fill_qty,
                )

                # Check if slippage is acceptable
                is_acceptable, _ = self._check_slippage_acceptable(
                    expected_price=price,
                    actual_price=fill_price,
                    side=order_side,
                )
                if not is_acceptable:
                    logger.warning(f"Slippage exceeded limit but order already filled")

                # Update position
                if self._position and self._position.side == side:
                    # Add to position (DCA)
                    if fill_qty > 0:
                        total_value = (
                            self._position.quantity * self._position.entry_price +
                            fill_qty * fill_price
                        )
                        self._position.quantity += fill_qty
                        # Avoid division by zero
                        if self._position.quantity > 0:
                            self._position.entry_price = total_value / self._position.quantity

                        # Update stop loss order with new quantity and average price
                        if self._config.use_exchange_stop_loss:
                            await self._update_stop_loss_order()
                    else:
                        logger.warning(f"Skipping DCA update: fill_qty is zero")
                else:
                    # New position (or flip)
                    if self._position:
                        close_ok = await self._close_position(price, ExitReason.TREND_CHANGE)
                        if not close_ok:
                            logger.error("Failed to close position before flip - aborting open")
                            return False

                    self._position = FuturesPosition(
                        symbol=self._config.symbol,
                        side=side,
                        entry_price=fill_price,
                        quantity=fill_qty,
                        leverage=self._config.leverage,
                        entry_time=datetime.now(timezone.utc),
                        highest_price=fill_price,
                        lowest_price=fill_price,
                    )

                    # Record entry bar for timeout tracking
                    self._entry_bar = self._current_bar

                    # Place exchange stop loss order
                    if self._config.use_exchange_stop_loss:
                        await self._place_stop_loss_order()

                # Record virtual fill (淨倉位管理 - track per-bot position)
                order_id_str = str(getattr(order, "order_id", ""))
                fee = fill_qty * fill_price * self._config.fee_rate
                self.record_virtual_fill(
                    symbol=self._config.symbol,
                    side=order_side,
                    quantity=fill_qty,
                    price=fill_price,
                    order_id=order_id_str,
                    fee=fee,
                    is_reduce_only=False,
                    leverage=self._config.leverage,
                )

                # Record cost basis entry (持倉歸屬 - FIFO tracking)
                self.record_cost_basis_entry(
                    symbol=self._config.symbol,
                    quantity=fill_qty,
                    price=fill_price,
                    order_id=order_id_str,
                    fee=fee,
                )

                logger.info(f"Opened {side.value} position: {fill_qty} @ {fill_price}")

                # Verify position sync with exchange after order execution
                is_synced, exchange_pos = await self._verify_position_sync(
                    expected_quantity=self._position.quantity,
                    expected_side=side.value,
                )
                if not is_synced:
                    logger.warning(
                        f"Position sync verification failed after order - "
                        f"forcing resync"
                    )
                    await self._sync_position()

                return True

        except Exception as e:
            # Classify and handle the error
            await self._handle_order_rejection(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
                error=e,
            )
        finally:
            self.release_risk_gate()

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
                    pnl = close_qty * (fill_price - self._position.entry_price)
                else:
                    pnl = close_qty * (self._position.entry_price - fill_price)

                fee = close_qty * fill_price * self._config.fee_rate * 2  # Entry + exit fee

                # Record trade
                # Calculate MFE/MAE
                mfe, mae = self.calculate_mfe_mae(
                    side=self._position.side.value,
                    entry_price=self._position.entry_price,
                    highest_price=self._position.highest_price,
                    lowest_price=self._position.lowest_price,
                    leverage=self._config.leverage,
                )

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
                    mfe=mfe,
                    mae=mae,
                )
                self._grid_stats.record_trade(trade)

                # Close cost basis with FIFO (持倉歸屬 - P&L attribution)
                order_id_str = str(getattr(order, "order_id", ""))
                close_fee = close_qty * fill_price * self._config.fee_rate
                cost_basis_result = self.close_cost_basis_fifo(
                    symbol=self._config.symbol,
                    close_quantity=close_qty,
                    close_price=fill_price,
                    close_order_id=order_id_str,
                    close_fee=close_fee,
                    leverage=self._config.leverage,
                )
                logger.debug(
                    f"Cost basis closed: {len(cost_basis_result.get('matched_lots', []))} lots, "
                    f"Attributed P&L: {cost_basis_result.get('total_realized_pnl')}"
                )

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
            await self._handle_order_rejection(
                symbol=self._config.symbol,
                side="SELL" if self._position and self._position.side.value == "LONG" else "BUY",
                quantity=quantity or (self._position.quantity if self._position else Decimal("0")),
                error=e,
            )
            await self._on_close_position_failure(self._config.symbol, e)

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

        # === Data Protection: Validate data quality ===
        # Check data freshness
        if not self._validate_kline_freshness(kline):
            await self._handle_data_anomaly(
                "stale_data", "high",
                f"Kline data is stale: age > 120s"
            )
            # Continue processing but log the warning

        # Check data integrity (OHLC relationships)
        if not self._validate_kline_integrity(kline, self._prev_kline):
            await self._handle_data_anomaly(
                "invalid_data", "critical",
                f"Invalid OHLC data detected"
            )
            return  # Don't process invalid data

        # Check for data gaps
        if self._prev_kline and not self._check_data_gap(
            kline, self._prev_kline, self._interval_seconds
        ):
            await self._handle_data_anomaly(
                "data_gap", "high",
                f"Data gap detected - indicators may be inaccurate"
            )
            # Continue but indicators may be affected

        # Update data health tracking
        self._update_data_health(kline)
        self._prev_kline = kline

        try:
            current_price = kline.close
            kline_low = kline.low
            kline_high = kline.high

            # Increment bar counter for timeout tracking
            self._current_bar += 1

            # Update closes and klines for indicators (ATR uses _klines)
            self._closes.append(current_price)
            if len(self._closes) > 500:
                self._closes = self._closes[-500:]
            self._klines.append(kline)
            if len(self._klines) > 500:
                self._klines = self._klines[-500:]

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

            # Update position PnL and check stop loss / timeout
            if self._position:
                self._position.unrealized_pnl = self._position.calculate_pnl(current_price)
                self._position.update_extremes(current_price)
                if await self._check_stop_loss(current_price):
                    logger.warning(f"Stop loss triggered at {current_price}")
                elif self._check_timeout_exit(current_price):
                    logger.warning(f"Timeout exit triggered: held {self._current_bar - self._entry_bar} bars")
                    await self._close_position(current_price, ExitReason.TIMEOUT)

            # Decrement signal cooldown (after entry logic to avoid off-by-one)
            if self._signal_cooldown > 0:
                self._signal_cooldown -= 1

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
        Uses state lock to prevent concurrent grid level modifications.

        Args:
            kline_low: K 線最低價
            kline_high: K 線最高價
            current_price: K 線收盤價
        """
        if not self._grid:
            return

        # Acquire state lock to prevent concurrent modifications
        async with self._state_lock:
            for i, level in enumerate(self._grid.levels):
                grid_price = level.price

                # Long entry: K 線低點觸及 grid level (買跌，與回測一致)
                if level.state == GridLevelState.EMPTY and kline_low <= grid_price:
                    # Check signal cooldown to prevent signal stacking (if enabled)
                    if self._config.use_signal_cooldown and self._signal_cooldown > 0:
                        logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping long entry")
                        continue

                    # Check hysteresis to prevent oscillation (if enabled)
                    if self._config.use_hysteresis and not self._check_hysteresis(i, "long", grid_price, current_price):
                        continue

                    if self._should_trade_direction("long"):
                        # 使用 grid level 價格進場（與回測一致）
                        success = await self._open_position(PositionSide.LONG, grid_price)
                        if success:
                            level.state = GridLevelState.LONG_FILLED
                            level.filled_at = datetime.now(timezone.utc)
                            if self._config.use_signal_cooldown:
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
                        close_success = await self._close_position(grid_price, ExitReason.GRID_PROFIT, partial_qty)
                        # Only update state if close was successful
                        if close_success:
                            level.state = GridLevelState.EMPTY
                            # Clear hysteresis on exit to allow fresh entry
                            if self._last_triggered_level == i:
                                self._last_triggered_level = None
                    else:
                        # No position but level marked as filled - reset state
                        level.state = GridLevelState.EMPTY

                # Short entry: K 線高點觸及 grid level (賣漲，與回測一致)
                elif level.state == GridLevelState.EMPTY and kline_high >= grid_price:
                    # Check signal cooldown to prevent signal stacking (if enabled)
                    if self._config.use_signal_cooldown and self._signal_cooldown > 0:
                        logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping short entry")
                        continue

                    # Check hysteresis to prevent oscillation (if enabled)
                    if self._config.use_hysteresis and not self._check_hysteresis(i, "short", grid_price, current_price):
                        continue

                    if self._should_trade_direction("short"):
                        # 使用 grid level 價格進場（與回測一致）
                        success = await self._open_position(PositionSide.SHORT, grid_price)
                        if success:
                            level.state = GridLevelState.SHORT_FILLED
                            level.filled_at = datetime.now(timezone.utc)
                            if self._config.use_signal_cooldown:
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
                        close_success = await self._close_position(grid_price, ExitReason.GRID_PROFIT, partial_qty)
                        # Only update state if close was successful
                        if close_success:
                            level.state = GridLevelState.EMPTY
                            # Clear hysteresis on exit to allow fresh entry
                            if self._last_triggered_level == i:
                                self._last_triggered_level = None
                    else:
                        # No position but level marked as filled - reset state
                        level.state = GridLevelState.EMPTY

    # =========================================================================
    # Stop Loss Check
    # =========================================================================

    # =========================================================================
    # Slippage Tracking
    # =========================================================================
    # Note: Slippage tracking methods (_record_slippage, get_slippage_stats)
    # are inherited from BaseBot with enhanced functionality
    # =========================================================================

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

        # Protect against division by zero
        if self._position.entry_price <= 0:
            logger.warning("Cannot check stop loss: entry_price is zero or negative")
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
            # Record stop loss trigger for oscillation prevention
            self.record_stop_loss_trigger()
            self.clear_stop_loss_sync()
            return True

        return False

    def _check_timeout_exit(self, current_price: Decimal) -> bool:
        """檢查是否超時出場（僅虧損時）。"""
        max_hold = self._config.max_hold_bars
        if max_hold <= 0 or not self._position:
            return False
        bars_held = self._current_bar - self._entry_bar
        if bars_held < max_hold:
            return False
        # Only exit if losing
        if self._position.side == PositionSide.LONG:
            return current_price < self._position.entry_price
        else:
            return current_price > self._position.entry_price

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
        return stop_price.quantize(getattr(self, '_tick_size', Decimal("0.1")))

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

    async def _cancel_stop_loss_order(self) -> bool:
        """
        Cancel stop loss order with verification.

        Returns:
            True if cancelled successfully or was triggered
        """
        if not self._position or not self._position.stop_loss_order_id:
            return True

        # Use BaseBot's robust algo cancel with verification
        result = await self._cancel_algo_order_with_verification(
            algo_id=self._position.stop_loss_order_id,
            symbol=self._config.symbol,
        )

        if result["is_cancelled"]:
            logger.info(f"Stop loss order cancelled: {self._position.stop_loss_order_id}")
            self._position.stop_loss_order_id = None
            self._position.stop_loss_price = None
            return True

        elif result["was_triggered"]:
            # Stop loss was executed - position may have changed
            logger.warning(
                f"Stop loss {self._position.stop_loss_order_id} was triggered - "
                f"forcing position sync"
            )
            self._position.stop_loss_order_id = None
            self._position.stop_loss_price = None
            await self._sync_position()
            return True

        else:
            logger.error(
                f"Failed to cancel stop loss after {result['attempts']} attempts: "
                f"{result['error_message']}"
            )
            return False

    # =========================================================================
    # Background Monitor (資金更新、回撤追蹤)
    # =========================================================================

    def _on_kline_task_done(self, task: asyncio.Task) -> None:
        """Handle kline task completion with exception logging."""
        self._kline_tasks.discard(task)
        if not task.cancelled() and task.exception():
            logger.error(
                f"Kline task failed: {task.exception()}",
                exc_info=task.exception(),
            )

    async def _background_monitor_safe(self) -> None:
        """Auto-restart wrapper for background monitor with max retries."""
        consecutive_failures = 0
        max_consecutive_failures = 5
        while self._state == BotState.RUNNING:
            try:
                await self._background_monitor()
                consecutive_failures = 0  # Reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_failures += 1
                logger.error(
                    f"Background monitor crashed ({consecutive_failures}/{max_consecutive_failures}): {e}"
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.critical(
                        f"Background monitor failed {max_consecutive_failures} times consecutively, stopping bot"
                    )
                    break
                await asyncio.sleep(5)

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
                self.mark_capital_updated()  # Track data freshness for bypass prevention
                self._grid_stats.update_drawdown(self._capital, self._initial_capital)

                # Record capital for CB validation (熔斷誤觸發防護)
                self.record_capital_for_validation(self._capital)

                # Apply consecutive loss decay (prevents permanent lockout)
                self.apply_consecutive_loss_decay()

                # Update virtual position unrealized P&L
                if self._position:
                    current_price = await self._get_current_price()
                    self.update_virtual_unrealized_pnl(self._config.symbol, current_price, leverage=self._config.leverage)
                    # Record price for CB validation
                    self.record_price_for_validation(current_price)

                # Check per-strategy risk (風控隔離 - only affects this bot)
                risk_result = await self.check_strategy_risk()
                if risk_result["risk_level"] in ["DANGER", "CRITICAL"]:
                    logger.warning(
                        f"Strategy risk triggered: {risk_result['risk_level']} - "
                        f"{risk_result.get('action', 'none')}"
                    )
                    # Trigger circuit breaker on CRITICAL risk level (with validation)
                    if risk_result["risk_level"] == "CRITICAL":
                        current_price = await self._get_current_price()
                        cb_result = await self.trigger_circuit_breaker_safe(
                            reason=f"CRITICAL_RISK: {risk_result.get('action', 'unknown')}",
                            current_price=current_price,
                            current_capital=self._capital,
                            partial=True,  # Only affect this strategy
                        )
                        if cb_result["triggered"]:
                            # Verify exchange position is actually closed before clearing local state
                            try:
                                positions = await self._exchange.futures.get_positions(self._config.symbol)
                                has_position = any(
                                    p.quantity != 0 for p in positions
                                    if p.symbol == self._config.symbol
                                )
                                if not has_position:
                                    self._position = None
                                else:
                                    logger.warning(
                                        f"[{self._bot_id}] Circuit breaker triggered but position still on exchange - keeping local state"
                                    )
                            except Exception as e:
                                logger.warning(f"[{self._bot_id}] Failed to verify position after circuit breaker: {e}")

                # Reconcile virtual position with exchange (drift detection)
                if self._position:
                    exchange_qty = self._position.quantity
                    exchange_side = self._position.side.value.upper() if self._position.side else None
                    recon_result = await self.reconcile_virtual_position(
                        symbol=self._config.symbol,
                        exchange_quantity=exchange_qty,
                        exchange_side=exchange_side,
                    )
                    if recon_result.get("drift_detected"):
                        logger.warning(
                            f"Position drift detected: {recon_result.get('action_needed')}, "
                            f"Virtual={recon_result.get('virtual_quantity')}, "
                            f"Exchange={exchange_qty}"
                        )

                # Stop loss sync check (止損同步管理)
                if self._position and self._config.use_exchange_stop_loss:
                    sync_result = await self.sync_stop_loss_order(
                        symbol=self._config.symbol,
                        position_side=self._position.side.value.upper(),
                        current_entry_price=self._position.entry_price,
                        current_quantity=self._position.quantity,
                        current_stop_loss_pct=self._config.stop_loss_pct,
                    )
                    if sync_result["action_taken"] == "SYNCED":
                        self._position.stop_loss_order_id = sync_result["new_order_id"]
                        self._position.stop_loss_price = sync_result["new_stop_price"]

                # Network health monitoring (網路彈性監控)
                try:
                    start_time = time.time()
                    test_price = await self._get_current_price()
                    latency_ms = (time.time() - start_time) * 1000
                    self.record_network_request(True, latency_ms)

                    # Check if network needs reconnection
                    net_healthy, net_reason = self.is_network_healthy()
                    if not net_healthy:
                        logger.warning(f"Network unhealthy: {net_reason}")
                        # Attempt reconnection if disconnected
                        if not self._network_health_state.get("is_connected", True):
                            reconnected = await self.attempt_network_reconnect()
                            if not reconnected:
                                logger.error("Network reconnection failed")
                except Exception as net_err:
                    # Record network failure
                    error_result = await self.handle_network_error(net_err, "background_monitor")
                    if error_result.get("action") == "reconnect":
                        await self.attempt_network_reconnect()

                # SSL certificate monitoring (SSL 證書監控)
                try:
                    ssl_healthy, ssl_reason = self.is_ssl_healthy()
                    if not ssl_healthy:
                        logger.warning(f"SSL unhealthy: {ssl_reason}")
                        # Periodic full SSL check (every 24h or on error)
                        await self.check_ssl_certificate()
                except Exception as ssl_err:
                    await self.handle_ssl_error(ssl_err, "background_monitor")

                # Send heartbeat
                await self._send_heartbeat()

                # Comprehensive stop loss check (三層止損保護)
                if self._position:
                    current_price = await self._get_current_price()
                    sl_check = await self.comprehensive_stop_loss_check(
                        symbol=self._config.symbol,
                        current_price=current_price,
                        entry_price=self._position.entry_price,
                        position_side=self._position.side.value.upper(),
                        quantity=self._position.quantity,
                        stop_loss_pct=self._config.stop_loss_pct,
                        stop_loss_order_id=self._position.stop_loss_order_id,
                        leverage=self._config.leverage,
                    )

                    if sl_check["action_needed"]:
                        logger.warning(
                            f"Stop loss protection triggered: {sl_check['action_type']} "
                            f"(urgency: {sl_check['urgency']})"
                        )

                        if sl_check["action_type"] == "EMERGENCY_CLOSE":
                            # Emergency close with highest priority
                            await self.execute_emergency_close(
                                symbol=self._config.symbol,
                                side=self._position.side.value.upper(),
                                quantity=self._position.quantity,
                                reason=sl_check["details"].get("emergency", {}).get("reason", "UNKNOWN"),
                            )
                            self._position = None
                            self.reset_stop_loss_protection()

                        elif sl_check["action_type"] == "REPLACE_SL":
                            # Replace failed stop loss order
                            replace_result = await self.replace_failed_stop_loss(
                                symbol=self._config.symbol,
                                side=self._position.side.value.upper(),
                                quantity=self._position.quantity,
                                entry_price=self._position.entry_price,
                                stop_loss_pct=self._config.stop_loss_pct,
                            )
                            if replace_result["success"]:
                                self._position.stop_loss_order_id = replace_result["new_order_id"]
                                self._position.stop_loss_price = replace_result.get("stop_price")

                        elif sl_check["action_type"] == "BACKUP_CLOSE":
                            # Backup stop loss triggered - close position
                            await self._close_position(current_price, ExitReason.STOP_LOSS)
                            self.reset_stop_loss_protection()

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

        # Verify stop loss order still exists on exchange
        checks["stop_loss_valid"] = await self._verify_stop_loss()

        # Network health check (網路彈性)
        net_healthy, _ = self.is_network_healthy()
        checks["network_healthy"] = net_healthy

        # DNS resolution check
        try:
            dns_ok, _ = await self._verify_dns_resolution()
            checks["dns_ok"] = dns_ok
        except Exception:
            checks["dns_ok"] = False

        # SSL certificate health check (SSL 證書)
        ssl_healthy, _ = self.is_ssl_healthy()
        checks["ssl_healthy"] = ssl_healthy

        return checks

    async def _verify_stop_loss(self) -> bool:
        """
        Verify that the stop loss order still exists on the exchange.

        Returns:
            True if no SL expected, or SL is confirmed on exchange.
            False if SL should exist but is missing.
        """
        if not self._position or not self._position.stop_loss_order_id:
            return True  # No SL expected

        try:
            order = await self._exchange.futures.get_order(
                symbol=self._config.symbol,
                order_id=self._position.stop_loss_order_id,
            )
            if order:
                status = getattr(order, "status", "").upper()
                if status in ("NEW", "PARTIALLY_FILLED"):
                    return True
                # SL was filled or cancelled — position may be unprotected
                logger.warning(
                    f"[{self._bot_id}] Stop loss order {self._position.stop_loss_order_id} "
                    f"has status {status} — position may be unprotected"
                )
                return False
            else:
                logger.warning(
                    f"[{self._bot_id}] Stop loss order not found on exchange"
                )
                return False
        except Exception as e:
            logger.warning(f"[{self._bot_id}] Failed to verify stop loss: {e}")
            return True  # Don't fail health check on transient errors

    # =========================================================================
    # State Persistence (with validation, checksum, and concurrent protection)
    # =========================================================================

    def _get_state_data(self) -> Dict[str, Any]:
        """
        Override BaseBot method to provide grid futures specific state.

        Returns:
            Dictionary of state data for persistence
        """
        state_data = {
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "signal_cooldown": self._signal_cooldown,
            "last_triggered_level": self._last_triggered_level,
            "current_bar": self._current_bar,
            "entry_bar": self._entry_bar,
            "stats": {
                "total_trades": self._grid_stats.total_trades,
                "winning_trades": self._grid_stats.winning_trades,
                "losing_trades": self._grid_stats.losing_trades,
                "total_pnl": str(self._grid_stats.total_pnl),
                "total_fees": str(self._grid_stats.total_fees),
                "long_trades": self._grid_stats.long_trades,
                "short_trades": self._grid_stats.short_trades,
                "max_drawdown_pct": str(self._grid_stats.max_drawdown_pct),
                "grid_rebuilds": self._grid_stats.grid_rebuilds,
                "_peak_equity": str(self._grid_stats._peak_equity),
            },
        }

        if self._grid:
            # Save grid level states for recovery
            level_states = [
                {"index": lv.index, "state": lv.state.value}
                for lv in self._grid.levels
            ]
            state_data["grid"] = {
                "center_price": str(self._grid.center_price),
                "upper_price": str(self._grid.upper_price),
                "lower_price": str(self._grid.lower_price),
                "grid_count": self._grid.grid_count,
                "version": self._grid.version,
                "level_states": level_states,
            }

        if self._position:
            state_data["position"] = {
                "side": self._position.side.value,
                "entry_price": str(self._position.entry_price),
                "quantity": str(self._position.quantity),
                "entry_time": self._position.entry_time.isoformat() if self._position.entry_time else None,
                "stop_loss_order_id": self._position.stop_loss_order_id,
                "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                "highest_price": str(self._position.highest_price),
                "lowest_price": str(self._position.lowest_price),
            }

        return state_data

    async def _save_state(self) -> None:
        """Save bot state to database with concurrent protection."""

        async def do_save():
            config = {
                "symbol": self._config.symbol,
                "grid_count": self._config.grid_count,
                "leverage": self._config.leverage,
                "direction": self._config.direction.value,
                "trend_period": self._config.trend_period,
                "atr_multiplier": str(self._config.atr_multiplier),
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            # Create validated snapshot with metadata
            snapshot = self._create_state_snapshot()

            async def persist_to_db(snapshot_data: Dict[str, Any]):
                await self._data_manager.save_bot_state(
                    bot_id=self._bot_id,
                    bot_type="grid_futures",
                    status=self._state.value,
                    config=config,
                    state_data=snapshot_data,
                )

            await self._save_state_atomic(snapshot["state"], persist_to_db)

        # Use lock to prevent concurrent state modifications during save
        await self._modify_state_safely(do_save, "save_state")

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

    async def _stop_save_task(self) -> None:
        """Stop periodic save task and wait for cleanup."""
        if self._save_task:
            self._save_task.cancel()
            try:
                await self._save_task
            except asyncio.CancelledError:
                pass
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
        Restore a GridFuturesBot from saved state with validation.

        Includes:
        - Schema version validation
        - Timestamp staleness check
        - Checksum integrity verification
        - Position sync with exchange

        Args:
            bot_id: Bot ID to restore
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance

        Returns:
            Restored GridFuturesBot or None if not found/invalid
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            # Extract and validate state_data section
            saved_state = state_data.get("state_data", {})

            # Validate state freshness (use timestamp from state_data if available)
            timestamp_str = saved_state.get("timestamp")
            if timestamp_str:
                try:
                    saved_time = datetime.fromisoformat(timestamp_str)
                    age_hours = (datetime.now(timezone.utc) - saved_time).total_seconds() / 3600
                    max_age = 24  # MAX_STATE_AGE_HOURS

                    if age_hours > max_age:
                        logger.warning(
                            f"State too old for {bot_id}: {age_hours:.1f}h > {max_age}h - "
                            f"starting fresh to avoid stale state issues"
                        )
                        return None
                except Exception as e:
                    logger.warning(f"Invalid timestamp in saved state: {e}")

            # Validate schema version if present
            schema_version = saved_state.get("schema_version", 1)
            if schema_version > 2:  # STATE_SCHEMA_VERSION
                logger.warning(
                    f"State schema version {schema_version} > current 2 for {bot_id} - "
                    f"cannot restore future schema"
                )
                return None

            config_data = state_data.get("config", {})
            config = GridFuturesConfig(
                symbol=config_data.get("symbol", ""),
                grid_count=config_data.get("grid_count", 10),
                leverage=config_data.get("leverage", 2),
                direction=GridDirection(config_data.get("direction", "neutral")),
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

            # Get state (could be nested in "state" key from new format)
            inner_state = saved_state.get("state", saved_state)

            # Restore core state
            bot._capital = Decimal(inner_state.get("capital", "0"))
            bot._initial_capital = Decimal(inner_state.get("initial_capital", "0"))
            bot._current_trend = inner_state.get("current_trend", 0)
            bot._signal_cooldown = inner_state.get("signal_cooldown", 0)
            bot._last_triggered_level = inner_state.get("last_triggered_level")
            bot._current_bar = inner_state.get("current_bar", 0)
            bot._entry_bar = inner_state.get("entry_bar", 0)

            # Restore stats
            stats_data = inner_state.get("stats", {})
            bot._grid_stats.total_trades = stats_data.get("total_trades", 0)
            bot._grid_stats.winning_trades = stats_data.get("winning_trades", 0)
            bot._grid_stats.losing_trades = stats_data.get("losing_trades", 0)
            bot._grid_stats.total_pnl = Decimal(stats_data.get("total_pnl", "0"))
            bot._grid_stats.total_fees = Decimal(stats_data.get("total_fees", "0"))
            bot._grid_stats.long_trades = stats_data.get("long_trades", 0)
            bot._grid_stats.short_trades = stats_data.get("short_trades", 0)
            bot._grid_stats.max_drawdown_pct = Decimal(stats_data.get("max_drawdown_pct", "0"))
            bot._grid_stats.grid_rebuilds = stats_data.get("grid_rebuilds", 0)
            bot._grid_stats._peak_equity = Decimal(stats_data.get("_peak_equity", str(bot._capital)))

            # Restore grid with level states
            grid_data = inner_state.get("grid")
            if grid_data:
                # Rebuild grid levels
                center = Decimal(grid_data["center_price"])
                upper = Decimal(grid_data["upper_price"])
                lower = Decimal(grid_data["lower_price"])
                grid_count = grid_data["grid_count"]
                # Guard against division by zero
                if grid_count <= 0:
                    grid_spacing = upper - lower
                else:
                    grid_spacing = (upper - lower) / Decimal(grid_count)

                levels = []
                level_states = grid_data.get("level_states", [])
                level_state_map = {ls["index"]: ls["state"] for ls in level_states}

                for i in range(grid_count + 1):
                    price = lower + Decimal(i) * grid_spacing
                    state = GridLevelState(level_state_map.get(i, "empty"))
                    levels.append(GridLevel(index=i, price=price, state=state))

                bot._grid = GridSetup(
                    center_price=center,
                    upper_price=upper,
                    lower_price=lower,
                    grid_spacing=grid_spacing,
                    levels=levels,
                    version=grid_data.get("version", 1),
                )

            # Restore position from saved state
            position_data = inner_state.get("position")
            if position_data:
                bot._position = FuturesPosition(
                    symbol=config.symbol,
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    leverage=config.leverage,
                    entry_time=datetime.fromisoformat(position_data["entry_time"]) if position_data.get("entry_time") else None,
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                    highest_price=Decimal(position_data.get("highest_price", position_data["entry_price"])),
                    lowest_price=Decimal(position_data.get("lowest_price", position_data["entry_price"])),
                )

            # Verify position sync with exchange before returning
            try:
                exchange_positions = await exchange.futures.get_positions(config.symbol)
                exchange_pos = None
                for pos in exchange_positions:
                    if pos.quantity > 0:
                        exchange_pos = pos
                        break

                if bot._position and not exchange_pos:
                    logger.warning(
                        f"Restored position not found on exchange for {bot_id} - "
                        f"position may have been closed (stop loss/liquidation)"
                    )
                    # Clear local position to match exchange
                    bot._position = None
                elif exchange_pos and not bot._position:
                    logger.warning(
                        f"Exchange has position but saved state doesn't for {bot_id} - "
                        f"syncing from exchange"
                    )
                    side_str = exchange_pos.side if isinstance(exchange_pos.side, str) else exchange_pos.side.value
                    bot._position = FuturesPosition(
                        symbol=config.symbol,
                        side=PositionSide(side_str),
                        entry_price=exchange_pos.entry_price,
                        quantity=exchange_pos.quantity,
                        leverage=config.leverage,
                    )
            except Exception as e:
                logger.warning(f"Failed to verify position sync on restore: {e}")

            logger.info(
                f"Restored GridFuturesBot: {bot_id}, "
                f"PnL={bot._grid_stats.total_pnl}, "
                f"position={'yes' if bot._position else 'no'}"
            )
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return None

