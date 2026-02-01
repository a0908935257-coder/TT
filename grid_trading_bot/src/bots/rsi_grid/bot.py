"""
RSI-Grid Hybrid Bot.

A futures trading bot combining RSI zone filtering with Grid entry mechanism.

Strategy Logic:
- RSI Zone Filter: Oversold=LONG only, Overbought=SHORT only, Neutral=follow trend
- Trend Filter: SMA-based trend direction
- Grid Entry: ATR-based dynamic grid levels, kline high/low detection
- Risk Management: ATR-based stop loss, grid-based take profit
- WebSocket subscription for real-time kline updates

Entry Logic (與回測一致):
- Long entry: K 線低點觸及 grid level (買跌)
- Short entry: K 線高點觸及 grid level (賣漲)
- 使用 grid level 價格進場

Design Goals:
- Target Sharpe > 3.0
- Walk-Forward Consistency > 90%
- Win Rate > 70%
- Max Drawdown < 5%
"""

import asyncio
import math
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional

from typing import Callable

from src.bots.base import BaseBot
from src.core import get_logger
from src.core.models import Kline, MarketType, OrderSide
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.fund_manager import SignalCoordinator, SignalDirection, CoordinationResult
from src.master.models import BotState
from src.notification import NotificationManager

from .indicators import RSICalculator, ATRCalculator, SMACalculator
from .models import (
    RSIGridConfig,
    RSIZone,
    PositionSide,
    GridLevel,
    GridLevelState,
    GridSetup,
    RSIGridPosition,
    RSIGridTrade,
    RSIGridStats,
    ExitReason,
)

logger = get_logger(__name__)


class RSIGridBot(BaseBot):
    """
    RSI-Grid Hybrid Trading Bot.

    Combines RSI mean reversion with Grid entry mechanism:
    - RSI Zone determines allowed direction (oversold=long, overbought=short)
    - SMA trend filter provides additional direction bias
    - ATR-based dynamic grid adapts to market volatility
    - Grid levels provide precise entry points

    Entry Logic:
    1. Calculate RSI zone (oversold/neutral/overbought)
    2. Calculate trend direction (above/below SMA)
    3. Determine allowed direction based on RSI zone + trend
    4. Enter when price touches a grid level in allowed direction

    Exit Logic:
    - Take profit: 1 grid spacing (ATR * multiplier / grid_count)
    - Stop loss: ATR * 1.5 (max 3%)
    - RSI reversal: Exit long when RSI > 70, exit short when RSI < 30

    Design Goals:
    - Target Sharpe > 3.0 (Grid Futures baseline: 4.50)
    - Walk-Forward Consistency > 90%
    - Win Rate > 70%
    - Max Drawdown < 5%
    """

    def __init__(
        self,
        bot_id: str,
        config: RSIGridConfig,
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

        # Indicators
        self._rsi_calc: Optional[RSICalculator] = None
        self._atr_calc: Optional[ATRCalculator] = None
        self._sma_calc: Optional[SMACalculator] = None

        # State
        self._grid: Optional[GridSetup] = None
        self._position: Optional[RSIGridPosition] = None
        self._current_trend: int = 0  # 1=up, -1=down, 0=neutral
        self._capital: Decimal = Decimal("0")
        self._initial_capital: Decimal = Decimal("0")
        self._prev_rsi: Optional[Decimal] = None

        # Indicators cache
        self._closes: List[Decimal] = []
        self._klines: List[Kline] = []

        # Statistics
        self._stats = RSIGridStats()

        # Tasks
        self._monitor_task: Optional[asyncio.Task] = None

        # Kline subscription callback
        self._kline_callback: Optional[Callable] = None

        # Kline callback tasks tracking (prevent memory leak)
        self._kline_tasks: set[asyncio.Task] = set()

        # State persistence
        self._save_task: Optional[asyncio.Task] = None
        self._save_interval_minutes: int = 5

        # Risk control
        self._daily_pnl = Decimal("0")
        self._daily_start_time: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._consecutive_losses: int = 0
        self._risk_paused: bool = False

        # Signal cooldown to prevent signal stacking (configurable)
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = config.cooldown_bars  # Use config value

        # Slippage tracking (from BaseBot)
        self._init_slippage_tracking()

        # Hysteresis: track last triggered level to prevent oscillation (configurable)
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = config.hysteresis_pct  # Use config value

        # Initialize state lock for concurrent protection
        self._init_state_lock()

        # v2: Bar counter and volatility baseline
        self._current_bar: int = 0
        self._atr_history: list[Decimal] = []
        self._atr_baseline: Optional[Decimal] = None

        # Data health tracking (for stale/gap detection)
        self._init_data_health_tracking()
        self._prev_kline: Optional[Kline] = None
        self._interval_seconds: int = self._parse_interval_to_seconds(config.timeframe)

    # =========================================================================
    # BaseBot Abstract Properties
    # =========================================================================

    @property
    def bot_type(self) -> str:
        return "rsi_grid"

    @property
    def symbol(self) -> str:
        return self._config.symbol

    # =========================================================================
    # BaseBot Abstract Methods
    # =========================================================================

    async def _do_start(self) -> None:
        """Initialize and start the bot."""
        logger.info(f"Starting RSI-Grid Bot for {self._config.symbol}")

        # Set leverage and margin type
        await self._setup_futures_account()

        # Get initial balance
        await self._update_capital()
        self._initial_capital = self._capital
        self._stats._peak_equity = self._capital

        # Initialize per-strategy risk tracking (風控相互影響隔離)
        self.set_strategy_initial_capital(self._capital)

        # Register for global risk tracking (多策略風控協調)
        await self.register_bot_for_global_risk(self._bot_id, self._capital)

        # Register for circuit breaker coordination (部分熔斷協調)
        await self.register_strategy_for_cb(
            bot_id=self._bot_id,
            strategy_type="rsi_grid",
            dependencies=None,
        )

        logger.info(f"Initial capital: {self._capital} USDT")

        # Check minimum capital requirement
        min_trade_value = Decimal("100")
        if self._capital < min_trade_value:
            logger.warning(
                f"Capital ({self._capital} USDT) may be too low to trade. "
                f"Minimum recommended: {min_trade_value} USDT"
            )

        # Load historical data for indicators
        await self._load_historical_data()

        # Initialize indicators
        self._initialize_indicators()

        # Initialize grid
        current_price = await self._get_current_price()
        self._initialize_grid(current_price)

        # Check existing position
        await self._sync_position()

        # Subscribe to kline updates (WebSocket, 與回測一致)
        def on_kline_sync(kline: Kline) -> None:
            task = asyncio.create_task(self._on_kline(kline))
            self._kline_tasks.add(task)
            task.add_done_callback(lambda t: self._kline_tasks.discard(t))

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

        # Start position reconciliation (detect manual operations)
        self._start_position_reconciliation()

        logger.info("RSI-Grid Bot started successfully")
        logger.info(f"  Symbol: {self._config.symbol}")
        logger.info(f"  Timeframe: {self._config.timeframe}")
        logger.info(f"  RSI: period={self._config.rsi_period}, oversold={self._config.oversold_level}, overbought={self._config.overbought_level}")
        logger.info(f"  Grid: count={self._config.grid_count}, ATR mult={self._config.atr_multiplier}")
        logger.info(f"  Trend SMA: {self._config.trend_sma_period}")
        logger.info(f"  Subscribed to {self._config.symbol} {self._config.timeframe} klines (WebSocket)")

    async def _do_stop(self, clear_position: bool = False) -> None:
        """Stop the bot."""
        logger.info("Stopping RSI-Grid Bot")

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

        logger.info("RSI-Grid Bot stopped")

    async def _do_pause(self) -> None:
        """Pause the bot."""
        logger.info("Pausing RSI-Grid Bot")

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
        logger.info("RSI-Grid Bot paused")

    async def _do_resume(self) -> None:
        """Resume the bot."""
        logger.info("Resuming RSI-Grid Bot")

        # Re-subscribe to kline updates (with task tracking)
        def on_kline_sync(kline: Kline) -> None:
            task = asyncio.create_task(self._on_kline(kline))
            self._kline_tasks.add(task)
            task.add_done_callback(lambda t: self._kline_tasks.discard(t))

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )

        # Restart background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor())
        logger.info("RSI-Grid Bot resumed")

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
        rsi_state = self._rsi_calc.get_state() if self._rsi_calc else {}

        status = {
            "bot_id": self._bot_id,
            "bot_type": self.bot_type,
            "symbol": self._config.symbol,
            "state": self._state.value,
            "leverage": self._config.leverage,
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "rsi": rsi_state.get("rsi"),
            "rsi_zone": rsi_state.get("zone"),
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
                "entry_rsi": str(self._position.entry_rsi),
                "entry_zone": self._position.entry_zone.value,
            }

        status["stats"] = self._stats.to_dict()

        return status

    # =========================================================================
    # Setup
    # =========================================================================

    async def _setup_futures_account(self) -> None:
        """Setup futures account (leverage, margin type)."""
        try:
            await self._exchange.futures.set_leverage(
                symbol=self._config.symbol,
                leverage=self._config.leverage,
            )
            logger.info(f"Set leverage to {self._config.leverage}x")

            try:
                await self._exchange.futures.set_margin_type(
                    symbol=self._config.symbol,
                    margin_type=self._config.margin_type,
                )
                logger.info(f"Set margin type to {self._config.margin_type}")
            except Exception as e:
                logger.debug(f"Margin type setting: {e}")

        except Exception as e:
            logger.error(f"Failed to setup futures account: {e}")
            raise

    async def _update_capital(self) -> None:
        """Update available capital."""
        try:
            balance = await self._exchange.futures.get_balance("USDT")
            available = balance.free if balance else Decimal("0")

            if self._config.max_capital:
                self._capital = min(available, self._config.max_capital)
            else:
                self._capital = available

        except Exception as e:
            logger.error(f"Failed to get balance: {e}")

    async def _on_capital_updated(self, new_max_capital: Decimal) -> None:
        """Handle capital update from FundManager."""
        previous_capital = self._capital
        await self._update_capital()

        logger.info(
            f"[FundManager] Capital updated for {self._bot_id}: "
            f"max_capital={new_max_capital}, "
            f"actual_capital: {previous_capital} -> {self._capital}"
        )

    # =========================================================================
    # Data Loading and Indicators
    # =========================================================================

    async def _load_historical_data(self) -> None:
        """Load historical klines for indicators."""
        try:
            limit = max(
                self._config.rsi_period + 50,
                self._config.atr_period + 50,
                self._config.trend_sma_period + 50,
                200
            )

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

    def _initialize_indicators(self) -> None:
        """Initialize RSI, ATR, and SMA calculators."""
        # RSI Calculator
        self._rsi_calc = RSICalculator(
            period=self._config.rsi_period,
            oversold=self._config.oversold_level,
            overbought=self._config.overbought_level,
        )
        rsi_result = self._rsi_calc.initialize(self._klines)
        if rsi_result:
            logger.info(f"RSI initialized: {rsi_result.rsi:.2f}, zone={rsi_result.zone.value}")

        # ATR Calculator
        self._atr_calc = ATRCalculator(period=self._config.atr_period)
        atr_result = self._atr_calc.initialize(self._klines)
        if atr_result:
            logger.info(f"ATR initialized: {atr_result.atr:.2f}")

        # SMA Calculator
        self._sma_calc = SMACalculator(period=self._config.trend_sma_period)
        for close in self._closes:
            self._sma_calc.update(close)

    async def _get_current_price(self) -> Decimal:
        """Get current market price."""
        try:
            ticker = await self._exchange.futures.get_ticker(self._config.symbol)
            return ticker.price if ticker else Decimal("0")
        except Exception as e:
            logger.error(f"Failed to get price: {e}")
            return Decimal("0")

    # =========================================================================
    # Grid Management
    # =========================================================================

    def _initialize_grid(self, center_price: Decimal) -> None:
        """Initialize grid levels around center price (symmetric ATR-based)."""
        atr_value = self._atr_calc.atr if self._atr_calc else None

        # Symmetric ATR-based range (aligned with backtest strategy)
        if atr_value and atr_value > 0:
            range_size = atr_value * self._config.atr_multiplier
        else:
            range_size = center_price * Decimal("0.05")

        upper_price = center_price + range_size
        lower_price = center_price - range_size
        range_pct = range_size / center_price if center_price > 0 else Decimal("0")

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
            self._stats.grid_rebuilds += 1

        self._grid = GridSetup(
            center_price=center_price,
            upper_price=upper_price,
            lower_price=lower_price,
            grid_spacing=grid_spacing,
            levels=levels,
            atr_value=atr_value,
            range_pct=range_pct,
            version=version,
        )

        logger.info(
            f"Grid initialized: center={center_price:.2f}, "
            f"range=±{range_pct*100:.1f}%, levels={len(levels)}, v{version}"
        )

    def _check_rebuild_needed(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding (price out of range, no buffer)."""
        if not self._grid:
            return True

        if current_price > self._grid.upper_price:
            return True
        if current_price < self._grid.lower_price:
            return True

        return False

    # =========================================================================
    # Direction Logic
    # =========================================================================

    def _get_allowed_direction(self, rsi_zone: RSIZone, trend: int) -> str:
        """
        Determine allowed trading direction.

        Returns:
            "long_only", "short_only", "both", or "none"
        """
        if rsi_zone == RSIZone.OVERSOLD:
            return "long_only"
        elif rsi_zone == RSIZone.OVERBOUGHT:
            return "short_only"
        else:  # NEUTRAL
            if self._config.use_trend_filter:
                if trend > 0:
                    return "long_only"
                elif trend < 0:
                    return "short_only"
            return "both"

    def _rsi_score(self, rsi: Decimal) -> float:
        """Continuous RSI bias score using tanh. Returns [-1, +1]."""
        return math.tanh((float(rsi) - 50) / 50 * 1.5)

    def _find_current_grid_index(self, price: Decimal) -> Optional[int]:
        """Find the grid level index for the current price."""
        if not self._grid or not self._grid.levels:
            return None

        for i, level in enumerate(self._grid.levels):
            if i < len(self._grid.levels) - 1:
                if level.price <= price < self._grid.levels[i + 1].price:
                    return i

        if price >= self._grid.levels[-1].price:
            return len(self._grid.levels) - 1

        return 0

    def _reset_filled_levels(self) -> None:
        """Reset all filled grid levels when no position is held."""
        if self._grid:
            for level in self._grid.levels:
                level.state = GridLevelState.EMPTY

    # =========================================================================
    # Position Management
    # =========================================================================

    async def _sync_position(self) -> None:
        """Sync position from exchange."""
        try:
            positions = await self._exchange.futures.get_positions(self._config.symbol)

            for pos in positions:
                if pos.quantity > 0:
                    self._position = RSIGridPosition(
                        symbol=self._config.symbol,
                        side=pos.side,
                        entry_price=pos.entry_price,
                        quantity=pos.quantity,
                        leverage=self._config.leverage,
                        unrealized_pnl=pos.unrealized_pnl,
                        highest_price=pos.entry_price,
                        lowest_price=pos.entry_price,
                    )
                    logger.info(f"Synced existing position: {pos.side.value} {pos.quantity}")
                    return

            self._position = None

        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

    async def _open_position(
        self,
        side: PositionSide,
        price: Decimal,
        rsi: Decimal,
        rsi_zone: RSIZone,
        grid_level: int,
    ) -> bool:
        """Open a new position."""
        # Check risk limits
        if self._check_risk_limits():
            logger.warning(f"Trading paused due to risk limits")
            return False

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
                    logger.debug("Max position reached")
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
                    reason=f"RSI Grid level {grid_level}, RSI={rsi:.1f}",
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
                        max_wait_seconds=30,
                    )

                    if is_confirmed and fill_data:
                        fill_price = Decimal(fill_data.get("avg_price", str(price)))
                        fill_qty = Decimal(fill_data.get("filled_qty", str(order.filled_qty or quantity)))
                    else:
                        fill_price = order.avg_price if order.avg_price else price
                        fill_qty = order.filled_qty if order.filled_qty else quantity
                        logger.warning(
                            f"Fill confirmation failed for {order_id}, using order response data"
                        )
                else:
                    fill_price = order.avg_price if order.avg_price else price
                    fill_qty = order.filled_qty if order.filled_qty else quantity

                # Validate fill_qty before proceeding
                if fill_qty <= 0:
                    logger.error(f"Invalid fill_qty: {fill_qty}, skipping position update")
                    return False

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
                    logger.warning(
                        f"Slippage exceeded limit ({slippage_pct:.4f}% > {self.DEFAULT_MAX_SLIPPAGE_PCT}%)"
                    )

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

                        if self._config.use_exchange_stop_loss:
                            await self._update_stop_loss_order()
                    else:
                        logger.warning(f"Skipping DCA update: fill_qty is zero")
                else:
                    # New position
                    if self._position:
                        await self._close_position(price, ExitReason.TREND_CHANGE)

                    self._position = RSIGridPosition(
                        symbol=self._config.symbol,
                        side=side,
                        entry_price=fill_price,
                        quantity=fill_qty,
                        leverage=self._config.leverage,
                        entry_time=datetime.now(timezone.utc),
                        entry_rsi=rsi,
                        entry_zone=rsi_zone,
                        grid_level=grid_level,
                        entry_bar=self._current_bar,
                        highest_price=fill_price,
                        lowest_price=fill_price,
                    )

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

                logger.info(f"Opened {side.value} position: {fill_qty} @ {fill_price}, RSI={rsi:.1f}, zone={rsi_zone.value}")

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
            close_qty = close_qty.quantize(Decimal("0.001"))

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

                # Validate fill_price before calculations
                if not fill_price or fill_price <= 0:
                    logger.error(f"Invalid fill_price: {fill_price}, using current_price as fallback")
                    fill_price = current_price
                    if not fill_price or fill_price <= 0:
                        logger.error("Cannot determine valid fill_price, aborting close")
                        return False

                # Calculate PnL
                if self._position.side == PositionSide.LONG:
                    pnl = close_qty * (fill_price - self._position.entry_price)
                else:
                    pnl = close_qty * (self._position.entry_price - fill_price)

                fee = close_qty * fill_price * self._config.fee_rate * 2

                # Calculate MFE/MAE
                mfe, mae = self.calculate_mfe_mae(
                    side=self._position.side.value,
                    entry_price=self._position.entry_price,
                    highest_price=self._position.highest_price,
                    lowest_price=self._position.lowest_price,
                    leverage=self._config.leverage,
                )

                # Record trade
                exit_rsi = self._rsi_calc.rsi if self._rsi_calc else Decimal("50")
                trade = RSIGridTrade(
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
                    entry_rsi=self._position.entry_rsi,
                    exit_rsi=exit_rsi,
                    entry_zone=self._position.entry_zone,
                    grid_level=self._position.grid_level,
                    mfe=mfe,
                    mae=mae,
                )
                self._stats.record_trade(trade)

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

                # Update risk tracking
                self._update_risk_tracking(pnl)

                # Update position
                if close_qty < self._position.quantity:
                    self._position.quantity -= close_qty
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
                side="SELL" if self._position and self._position.side == PositionSide.LONG else "BUY",
                quantity=quantity or (self._position.quantity if self._position else Decimal("0")),
                error=e,
            )
            await self._on_close_position_failure(self._config.symbol, e)

        return False

    # =========================================================================
    # Stop Loss Management
    # =========================================================================

    def _calculate_stop_loss_price(self) -> Decimal:
        """Calculate stop loss price based on ATR."""
        if not self._position:
            return Decimal("0")

        atr = self._atr_calc.atr if self._atr_calc else None
        entry_price = self._position.entry_price

        if atr:
            sl_distance = atr * self._config.stop_loss_atr_mult
        else:
            sl_distance = entry_price * self._config.max_stop_loss_pct

        # Cap at max stop loss
        max_sl_distance = entry_price * self._config.max_stop_loss_pct
        sl_distance = min(sl_distance, max_sl_distance)

        if self._position.side == PositionSide.LONG:
            stop_price = entry_price - sl_distance
        else:
            stop_price = entry_price + sl_distance

        return stop_price.quantize(getattr(self, '_tick_size', Decimal("0.1")))

    async def _place_stop_loss_order(self) -> None:
        """Place stop loss order on exchange."""
        if not self._position:
            return

        try:
            stop_price = self._calculate_stop_loss_price()

            if self._position.side == PositionSide.LONG:
                close_side = OrderSide.SELL
            else:
                close_side = OrderSide.BUY

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
                logger.info(f"Stop loss placed: {close_side.value} @ {stop_price}")

        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")

    async def _update_stop_loss_order(self) -> None:
        """Update stop loss order when position changes."""
        if not self._position or not self._config.use_exchange_stop_loss:
            return

        await self._cancel_stop_loss_order()
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
            logger.info(f"Stop loss cancelled: {self._position.stop_loss_order_id}")
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
    # Risk Control
    # =========================================================================

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats if it's a new day (UTC)."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if today_start > self._daily_start_time:
            logger.info("New trading day - resetting daily stats")
            self._daily_pnl = Decimal("0")
            self._daily_start_time = today_start
            self._risk_paused = False

    def _check_risk_limits(self) -> bool:
        """Check if risk limits have been exceeded."""
        self._reset_daily_stats_if_needed()

        if self._risk_paused:
            return True

        capital = self._config.max_capital or self._capital or Decimal("1000")

        # Check daily loss limit
        daily_loss_pct = abs(self._daily_pnl) / capital if self._daily_pnl < 0 else Decimal("0")
        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            logger.warning(f"Daily loss limit reached: {daily_loss_pct:.1%}")
            self._risk_paused = True
            return True

        # Check consecutive losses
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            logger.warning(f"Max consecutive losses reached: {self._consecutive_losses}")
            self._risk_paused = True
            return True

        return False

    def _update_risk_tracking(self, pnl: Decimal) -> None:
        """Update risk tracking after a trade."""
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    # =========================================================================
    # Kline Event Handler (WebSocket 訂閱模式)
    # =========================================================================

    async def _on_kline(self, kline: Kline) -> None:
        """
        Process kline update from WebSocket subscription.

        This is the main trading logic handler, called on each kline update.
        Only processes closed klines to match backtest behavior.
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
            await self._process_kline(kline)
        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    # =========================================================================
    # Background Monitor (資金更新、回撤追蹤)
    # =========================================================================

    async def _background_monitor(self) -> None:
        """
        Background monitoring loop for capital updates and heartbeat.

        Note: Trading logic is handled by _on_kline callback.
        """
        logger.info("Starting RSI-Grid background monitor")

        while self._state == BotState.RUNNING:
            try:
                # Update capital and drawdown periodically
                await self._update_capital()
                self.mark_capital_updated()  # Track data freshness for bypass prevention
                self._stats.update_drawdown(self._capital, self._initial_capital)

                # Record capital for CB validation (熔斷誤觸發防護)
                self.record_capital_for_validation(self._capital)

                # Apply consecutive loss decay (prevents permanent lockout)
                self.apply_consecutive_loss_decay()

                # Update virtual position unrealized P&L and record price
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
                            partial=True,
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
                            f"Position drift detected: {recon_result.get('action_needed')}"
                        )

                # Comprehensive stop loss check (三層止損保護)
                if self._position:
                    current_price = await self._get_current_price()
                    sl_check = await self.comprehensive_stop_loss_check(
                        symbol=self._config.symbol,
                        current_price=current_price,
                        entry_price=self._position.entry_price,
                        position_side=self._position.side.value.upper(),
                        quantity=self._position.quantity,
                        stop_loss_pct=self._config.max_stop_loss_pct,
                        stop_loss_order_id=self._position.stop_loss_order_id,
                        leverage=self._config.leverage,
                    )

                    if sl_check["action_needed"]:
                        logger.warning(
                            f"Stop loss protection triggered: {sl_check['action_type']} "
                            f"(urgency: {sl_check['urgency']})"
                        )

                        if sl_check["action_type"] == "EMERGENCY_CLOSE":
                            # Use _close_position to properly record trade and update risk tracking
                            emergency_reason = sl_check["details"].get("emergency", {}).get("reason", "UNKNOWN")
                            await self._close_position(
                                price=await self._get_current_price(),
                                reason=ExitReason.STOP_LOSS,
                            )
                            self.reset_stop_loss_protection()

                        elif sl_check["action_type"] == "REPLACE_SL":
                            replace_result = await self.replace_failed_stop_loss(
                                symbol=self._config.symbol,
                                side=self._position.side.value.upper(),
                                quantity=self._position.quantity,
                                entry_price=self._position.entry_price,
                                stop_loss_pct=self._config.max_stop_loss_pct,
                            )
                            if replace_result["success"]:
                                self._position.stop_loss_order_id = replace_result["new_order_id"]
                                self._position.stop_loss_price = replace_result.get("stop_price")

                        elif sl_check["action_type"] == "BACKUP_CLOSE":
                            await self._close_position(current_price, ExitReason.STOP_LOSS)
                            self.reset_stop_loss_protection()

                # Network health monitoring (網路彈性監控)
                try:
                    start_time = time.time()
                    test_price = await self._get_current_price()
                    latency_ms = (time.time() - start_time) * 1000
                    self.record_network_request(True, latency_ms)

                    net_healthy, net_reason = self.is_network_healthy()
                    if not net_healthy:
                        logger.warning(f"Network unhealthy: {net_reason}")
                        if not self._network_health_state.get("is_connected", True):
                            reconnected = await self.attempt_network_reconnect()
                            if not reconnected:
                                logger.error("Network reconnection failed")
                except Exception as net_err:
                    error_result = await self.handle_network_error(net_err, "background_monitor")
                    if error_result.get("action") == "reconnect":
                        await self.attempt_network_reconnect()

                # SSL certificate monitoring (SSL 證書監控)
                try:
                    ssl_healthy, ssl_reason = self.is_ssl_healthy()
                    if not ssl_healthy:
                        logger.warning(f"SSL unhealthy: {ssl_reason}")
                        await self.check_ssl_certificate()
                except Exception as ssl_err:
                    await self.handle_ssl_error(ssl_err, "background_monitor")

                # Send heartbeat
                await self._send_heartbeat()

                # Wait 30 seconds between updates
                await asyncio.sleep(30)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in background monitor: {e}")
                await asyncio.sleep(30)

        logger.info("Background monitor stopped")

    async def _process_kline(self, kline: Kline) -> None:
        """Process new kline data."""
        current_price = kline.close

        # Decrement signal cooldown
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        # Update indicators
        rsi_result = self._rsi_calc.update(kline) if self._rsi_calc else None
        atr_result = self._atr_calc.update(kline) if self._atr_calc else None
        self._sma_calc.update(current_price) if self._sma_calc else None

        if not rsi_result:
            return

        current_rsi = rsi_result.rsi
        rsi_zone = rsi_result.zone

        # v2: Increment bar counter
        self._current_bar += 1

        # Reset filled levels when no position (aligned with backtest)
        if not self._position:
            self._reset_filled_levels()

        # v2: Update volatility baseline
        atr = self._atr_calc.atr if self._atr_calc else None
        if atr:
            self._update_volatility_baseline(atr)

        # Update trend
        self._current_trend = self._sma_calc.get_trend(current_price) if self._sma_calc else 0

        # Check if grid needs rebuild
        if self._check_rebuild_needed(current_price):
            logger.info(f"Price {current_price} out of range, rebuilding grid")
            if self._position:
                await self._close_position(current_price, ExitReason.GRID_REBUILD)
            self._initialize_grid(current_price)
            self._prev_rsi = current_rsi
            return

        # Position management
        if self._position:
            self._position.unrealized_pnl = self._position.calculate_pnl(current_price)
            self._position.update_extremes(current_price)

            # v2: Trailing stop check
            if self._config.use_trailing_stop:
                if self._check_trailing_stop(current_price):
                    logger.warning(f"Trailing stop triggered at {current_price}")
                    await self._close_position(current_price, ExitReason.STOP_LOSS)
                    self.record_stop_loss_trigger()
                    self.clear_stop_loss_sync()
                    self._prev_rsi = current_rsi
                    return

            # v2: Timeout exit check
            if self._check_timeout_exit(current_price):
                bars_held = self._current_bar - self._position.entry_bar
                logger.warning(f"Timeout exit triggered: held {bars_held} bars")
                await self._close_position(current_price, ExitReason.TIMEOUT_EXIT)
                self._prev_rsi = current_rsi
                return

            await self._check_exit(current_price, current_rsi, rsi_zone)
        else:
            if not self._check_risk_limits():
                await self._check_entry(kline, current_price, current_rsi, rsi_zone)

        self._prev_rsi = current_rsi

    def _check_hysteresis(self, level_index: int, direction: str, grid_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if price has moved enough from last triggered level (hysteresis).

        Prevents oscillation around grid levels by requiring price to move
        away by a minimum percentage before retriggering the same level.

        Note: This feature is disabled by default for RSI Grid strategy based on
        backtest results showing it hurts performance for low-frequency strategies.

        Args:
            level_index: Grid level index
            direction: "long" or "short"
            grid_price: Grid level price
            current_price: Current market price

        Returns:
            True if entry is allowed (passed hysteresis check)
        """
        # Check if hysteresis is enabled in config
        if not self._config.use_hysteresis:
            return True

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

    async def _check_entry(
        self,
        kline: Kline,
        current_price: Decimal,
        rsi: Decimal,
        rsi_zone: RSIZone,
    ) -> None:
        """Check for entry signals using tanh RSI score + grid index (aligned with backtest)."""
        if not self._grid:
            return

        # Check signal cooldown to prevent signal stacking (if enabled)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping entry check")
            return

        # v2: Volatility regime filter
        if not self._check_volatility_regime():
            return

        # Use tanh RSI score instead of zone-based direction (aligned with backtest)
        score = self._rsi_score(rsi)
        threshold = self._config.rsi_block_threshold
        can_long = score < threshold    # block long when RSI too high
        can_short = score > -threshold  # block short when RSI too low

        if not can_long and not can_short:
            return

        # Find current grid level index (aligned with backtest)
        level_idx = self._find_current_grid_index(current_price)
        if level_idx is None:
            return

        curr_low = kline.low
        curr_high = kline.high

        # Long entry: grid_levels[level_idx] (current level, price dips to touch)
        if can_long and level_idx > 0:
            entry_level = self._grid.levels[level_idx]
            if entry_level.state == GridLevelState.EMPTY and curr_low <= entry_level.price:
                # Check hysteresis to prevent oscillation
                if not self._check_hysteresis(entry_level.index, "long", entry_level.price, current_price):
                    pass
                else:
                    success = await self._open_position(
                        side=PositionSide.LONG,
                        price=entry_level.price,
                        rsi=rsi,
                        rsi_zone=rsi_zone,
                        grid_level=entry_level.index,
                    )
                    if success:
                        entry_level.state = GridLevelState.LONG_FILLED
                        entry_level.filled_at = datetime.now(timezone.utc)
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._cooldown_bars
                        if self._config.use_hysteresis:
                            self._last_triggered_level = entry_level.index
                        return

        # Short entry: grid_levels[level_idx + 1] (upper level, price rises to touch)
        if can_short and level_idx < len(self._grid.levels) - 1:
            entry_level = self._grid.levels[level_idx + 1]
            if entry_level.state == GridLevelState.EMPTY and curr_high >= entry_level.price:
                # Check hysteresis to prevent oscillation
                if not self._check_hysteresis(entry_level.index, "short", entry_level.price, current_price):
                    pass
                else:
                    success = await self._open_position(
                        side=PositionSide.SHORT,
                        price=entry_level.price,
                        rsi=rsi,
                        rsi_zone=rsi_zone,
                        grid_level=entry_level.index,
                    )
                    if success:
                        entry_level.state = GridLevelState.SHORT_FILLED
                        entry_level.filled_at = datetime.now(timezone.utc)
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._cooldown_bars
                        if self._config.use_hysteresis:
                            self._last_triggered_level = entry_level.index
                        return

    async def _check_exit(
        self,
        current_price: Decimal,
        rsi: Decimal,
        rsi_zone: RSIZone,
    ) -> None:
        """Check for exit conditions."""
        if not self._position:
            return

        # RSI extreme reversal exit (aligned with backtest: hard thresholds)
        if self._position.side == PositionSide.LONG:
            if rsi > Decimal("75"):
                await self._close_position(current_price, ExitReason.RSI_EXIT)
                return
        else:  # SHORT
            if rsi < Decimal("25"):
                await self._close_position(current_price, ExitReason.RSI_EXIT)
                return

        # Take profit check (grid-based)
        if self._grid:
            tp_distance = self._grid.grid_spacing * Decimal(self._config.take_profit_grids)
            if self._position.side == PositionSide.LONG:
                if current_price >= self._position.entry_price + tp_distance:
                    await self._close_position(current_price, ExitReason.GRID_PROFIT)
                    return
            else:  # SHORT
                if current_price <= self._position.entry_price - tp_distance:
                    await self._close_position(current_price, ExitReason.GRID_PROFIT)
                    return

        # Software stop loss check (backup)
        pnl_pct = self._position.calculate_pnl_pct(current_price)
        if pnl_pct < -self._config.max_stop_loss_pct:
            await self._close_position(current_price, ExitReason.STOP_LOSS)
            self.record_stop_loss_trigger()
            self.clear_stop_loss_sync()

    # =========================================================================
    # v2 Helpers: Volatility Filter, Timeout Exit, Trailing Stop
    # =========================================================================

    def _update_volatility_baseline(self, current_atr: Decimal) -> None:
        """更新 ATR 滾動歷史，計算基線。"""
        self._atr_history.append(current_atr)
        bp = self._config.vol_atr_baseline_period
        if len(self._atr_history) > bp:
            self._atr_history = self._atr_history[-bp:]
        if len(self._atr_history) >= bp:
            self._atr_baseline = sum(self._atr_history) / Decimal(len(self._atr_history))

    def _check_volatility_regime(self) -> bool:
        """檢查當前波動率是否在可交易範圍內。"""
        if not self._config.use_volatility_filter:
            return True
        if self._atr_baseline is None or self._atr_baseline == 0:
            return True
        current_atr = self._atr_calc.atr if self._atr_calc else None
        if current_atr is None:
            return True
        ratio = float(current_atr / self._atr_baseline)
        low = self._config.vol_ratio_low
        high = self._config.vol_ratio_high
        if not (low <= ratio <= high):
            logger.info(f"Volatility filter blocked: ATR ratio={ratio:.2f} outside [{low}, {high}]")
            return False
        return True

    def _check_timeout_exit(self, current_price: Decimal) -> bool:
        """檢查是否超時出場（僅虧損時）。"""
        max_hold = self._config.max_hold_bars
        if max_hold <= 0 or not self._position:
            return False
        bars_held = self._current_bar - self._position.entry_bar
        if bars_held < max_hold:
            return False
        # Only exit if losing
        if self._position.side == PositionSide.LONG:
            return current_price < self._position.entry_price
        else:
            return current_price > self._position.entry_price

    def _check_trailing_stop(self, current_price: Decimal) -> bool:
        """檢查追蹤止損是否觸發。"""
        if not self._position:
            return False
        stop_pct = self._config.trailing_stop_pct
        if self._position.side == PositionSide.LONG:
            if self._position.max_price is not None:
                stop_price = self._position.max_price * (Decimal("1") - stop_pct)
                if current_price <= stop_price:
                    logger.info(
                        f"Trailing stop: price {current_price:.2f} <= "
                        f"stop {stop_price:.2f} (max: {self._position.max_price:.2f})"
                    )
                    return True
        else:
            if self._position.min_price is not None:
                stop_price = self._position.min_price * (Decimal("1") + stop_pct)
                if current_price >= stop_price:
                    logger.info(
                        f"Trailing stop: price {current_price:.2f} >= "
                        f"stop {stop_price:.2f} (min: {self._position.min_price:.2f})"
                    )
                    return True
        return False

    def _parse_timeframe_seconds(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return value * multipliers.get(unit, 60)

    # =========================================================================
    # Slippage Tracking - inherited from BaseBot
    # Methods: _record_slippage, _check_slippage_acceptable, get_slippage_stats
    # =========================================================================

    # =========================================================================
    # BaseBot Extra Methods
    # =========================================================================

    def _get_extra_status(self) -> Dict[str, Any]:
        """Return extra status fields specific to RSI-Grid bot."""
        rsi_state = self._rsi_calc.get_state() if self._rsi_calc else {}

        extra = {
            "rsi": rsi_state.get("rsi"),
            "rsi_zone": rsi_state.get("zone"),
            "current_trend": self._current_trend,
            "grid": None,
            "position": None,
            "stats": self._stats.to_dict(),
            "risk_paused": self._risk_paused,
            "daily_pnl": float(self._daily_pnl),
            "consecutive_losses": self._consecutive_losses,
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
                "entry_rsi": str(self._position.entry_rsi),
                "entry_zone": self._position.entry_zone.value,
            }

        return extra

    async def _extra_health_checks(self) -> Dict[str, bool]:
        """Perform extra health checks."""
        checks = {}

        checks["rsi_initialized"] = self._rsi_calc is not None
        checks["atr_initialized"] = self._atr_calc is not None
        checks["grid_valid"] = self._grid is not None

        try:
            price = await self._get_current_price()
            checks["price_available"] = price > 0
        except Exception:
            checks["price_available"] = False

        if self._position:
            checks["position_valid"] = self._position.quantity > 0
        else:
            checks["position_valid"] = True

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

    # =========================================================================
    # State Persistence (with validation, checksum, and concurrent protection)
    # =========================================================================

    def _get_state_data(self) -> Dict[str, Any]:
        """
        Override BaseBot method to provide RSI Grid specific state.

        Returns:
            Dictionary of state data for persistence
        """
        state_data = {
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "daily_pnl": str(self._daily_pnl),
            "consecutive_losses": self._consecutive_losses,
            "risk_paused": self._risk_paused,
            "signal_cooldown": self._signal_cooldown,
            "last_triggered_level": self._last_triggered_level,
            "current_bar": self._current_bar,
            "stats": self._stats.to_dict(),
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
                "version": self._grid.version,
                "level_states": level_states,
            }

        if self._position:
            state_data["position"] = {
                "side": self._position.side.value,
                "entry_price": str(self._position.entry_price),
                "quantity": str(self._position.quantity),
                "entry_time": self._position.entry_time.isoformat() if self._position.entry_time else None,
                "entry_rsi": str(self._position.entry_rsi),
                "entry_zone": self._position.entry_zone.value,
                "entry_bar": self._position.entry_bar,
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
                "timeframe": self._config.timeframe,
                "rsi_period": self._config.rsi_period,
                "oversold_level": self._config.oversold_level,
                "overbought_level": self._config.overbought_level,
                "grid_count": self._config.grid_count,
                "atr_multiplier": str(self._config.atr_multiplier),
                "leverage": self._config.leverage,
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            # Create validated snapshot with metadata
            snapshot = self._create_state_snapshot()

            async def persist_to_db(snapshot_data: Dict[str, Any]):
                await self._data_manager.save_bot_state(
                    bot_id=self._bot_id,
                    bot_type="rsi_grid",
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
    ) -> Optional["RSIGridBot"]:
        """
        Restore an RSIGridBot from saved state with validation.

        Includes:
        - Schema version validation
        - Timestamp staleness check
        - Position sync with exchange

        Args:
            bot_id: Bot ID to restore
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance

        Returns:
            Restored RSIGridBot or None if not found/invalid
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            # Extract and validate state_data section
            saved_state = state_data.get("state_data", {})

            # Validate state freshness
            timestamp_str = saved_state.get("timestamp")
            if timestamp_str:
                try:
                    saved_time = datetime.fromisoformat(timestamp_str)
                    age_hours = (datetime.now(timezone.utc) - saved_time).total_seconds() / 3600
                    max_age = 24

                    if age_hours > max_age:
                        logger.warning(
                            f"State too old for {bot_id}: {age_hours:.1f}h > {max_age}h - "
                            f"starting fresh"
                        )
                        return None
                except Exception as e:
                    logger.warning(f"Invalid timestamp in saved state: {e}")

            config_data = state_data.get("config", {})
            config = RSIGridConfig(
                symbol=config_data.get("symbol", ""),
                timeframe=config_data.get("timeframe", "1h"),
                rsi_period=config_data.get("rsi_period", 7),
                oversold_level=config_data.get("oversold_level", 33),
                overbought_level=config_data.get("overbought_level", 66),
                grid_count=config_data.get("grid_count", 8),
                atr_multiplier=Decimal(config_data.get("atr_multiplier", "3.0")),
                leverage=config_data.get("leverage", 10),
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
            bot._daily_pnl = Decimal(inner_state.get("daily_pnl", "0"))
            bot._consecutive_losses = inner_state.get("consecutive_losses", 0)
            bot._risk_paused = inner_state.get("risk_paused", False)
            bot._signal_cooldown = inner_state.get("signal_cooldown", 0)
            bot._last_triggered_level = inner_state.get("last_triggered_level")
            bot._current_bar = inner_state.get("current_bar", 0)

            # Restore stats
            stats_data = inner_state.get("stats", {})
            bot._stats.total_trades = stats_data.get("total_trades", 0)
            bot._stats.winning_trades = stats_data.get("winning_trades", 0)
            bot._stats.losing_trades = stats_data.get("losing_trades", 0)

            # Restore grid with level states
            grid_data = inner_state.get("grid")
            if grid_data:
                center = Decimal(grid_data["center_price"])
                upper = Decimal(grid_data["upper_price"])
                lower = Decimal(grid_data["lower_price"])
                grid_count = config.grid_count
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
                bot._position = RSIGridPosition(
                    symbol=config.symbol,
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    leverage=config.leverage,
                    entry_time=datetime.fromisoformat(position_data["entry_time"]) if position_data.get("entry_time") else None,
                    entry_rsi=Decimal(position_data.get("entry_rsi", "50")),
                    entry_zone=RSIZone(position_data.get("entry_zone", "neutral")),
                    entry_bar=position_data.get("entry_bar", 0),
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                    highest_price=Decimal(position_data.get("highest_price", position_data["entry_price"])),
                    lowest_price=Decimal(position_data.get("lowest_price", position_data["entry_price"])),
                )

            # Verify position sync with exchange
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
                        f"clearing local state"
                    )
                    bot._position = None
                elif exchange_pos and not bot._position:
                    logger.warning(
                        f"Exchange has position but saved state doesn't for {bot_id} - "
                        f"syncing from exchange"
                    )
                    side_str = exchange_pos.side if isinstance(exchange_pos.side, str) else exchange_pos.side.value
                    bot._position = RSIGridPosition(
                        symbol=config.symbol,
                        side=PositionSide(side_str),
                        entry_price=exchange_pos.entry_price,
                        quantity=exchange_pos.quantity,
                        leverage=config.leverage,
                    )
            except Exception as e:
                logger.warning(f"Failed to verify position sync on restore: {e}")

            logger.info(
                f"Restored RSIGridBot: {bot_id}, "
                f"position={'yes' if bot._position else 'no'}"
            )
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return None
