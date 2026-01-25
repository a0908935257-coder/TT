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

        # Signal cooldown to prevent signal stacking
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = 2  # Minimum bars between signals

        # Slippage tracking
        self._slippage_records: List[Dict] = []
        self._max_slippage_records: int = 100

        # Hysteresis: track last triggered level to prevent oscillation
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = Decimal("0.002")  # 0.2% buffer zone

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

        # Stop periodic save task and save final state
        self._stop_save_task()
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
        """Initialize grid levels around center price."""
        atr_value = self._atr_calc.atr if self._atr_calc else None

        # Calculate range
        if atr_value and atr_value > 0:
            range_value = atr_value * self._config.atr_multiplier
            range_pct = range_value / center_price
            # Clamp between 3% and 20%
            range_pct = min(max(range_pct, Decimal("0.03")), Decimal("0.20"))
        else:
            range_pct = Decimal("0.08")  # Fallback 8%

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
        """Check if grid needs rebuilding (price out of range)."""
        if not self._grid:
            return True

        threshold = Decimal("1.1")  # 10% beyond range

        if current_price > self._grid.upper_price * threshold:
            return True
        if current_price < self._grid.lower_price / threshold:
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

            # Calculate position size
            trade_value = self._capital * self._config.position_size_pct
            quantity = self._safe_divide(trade_value, price, context="position_size")

            # Round quantity
            quantity = quantity.quantize(Decimal("0.001"))

            # Validate quantity (indicator boundary check)
            if not self._validate_quantity(quantity, "order_quantity"):
                return False

            min_quantity = Decimal("0.001")
            if quantity < min_quantity:
                return False

            # Pre-trade validation (time sync + data health)
            order_side = "BUY" if side == PositionSide.LONG else "SELL"
            if not await self._validate_pre_trade(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
                check_time_sync=True,
                check_liquidity=False,  # Grid bots use small sizes
            ):
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

                if self._position and self._position.side == side:
                    # Add to position (DCA)
                    total_value = (
                        self._position.quantity * self._position.entry_price +
                        fill_qty * fill_price
                    )
                    self._position.quantity += fill_qty
                    self._position.entry_price = total_value / self._position.quantity

                    if self._config.use_exchange_stop_loss:
                        await self._update_stop_loss_order()
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
                    )

                    if self._config.use_exchange_stop_loss:
                        await self._place_stop_loss_order()

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

                # Calculate PnL
                if self._position.side == PositionSide.LONG:
                    pnl = close_qty * (fill_price - self._position.entry_price) * Decimal(self._config.leverage)
                else:
                    pnl = close_qty * (self._position.entry_price - fill_price) * Decimal(self._config.leverage)

                fee = close_qty * fill_price * self._config.fee_rate * 2

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
                )
                self._stats.record_trade(trade)

                # Update risk tracking
                self._update_risk_tracking(pnl)

                # Update position
                if quantity and quantity < self._position.quantity:
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

        return stop_price.quantize(Decimal("0.1"))

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

    async def _cancel_stop_loss_order(self) -> None:
        """Cancel stop loss order."""
        if not self._position or not self._position.stop_loss_order_id:
            return

        try:
            await self._exchange.futures.cancel_algo_order(
                symbol=self._config.symbol,
                algo_id=self._position.stop_loss_order_id,
            )
            logger.info(f"Stop loss cancelled: {self._position.stop_loss_order_id}")
            self._position.stop_loss_order_id = None
            self._position.stop_loss_price = None

        except Exception as e:
            logger.debug(f"Failed to cancel stop loss: {e}")

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
                self._stats.update_drawdown(self._capital, self._initial_capital)

                # Send heartbeat
                self._send_heartbeat()

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

    async def _check_entry(
        self,
        kline: Kline,
        current_price: Decimal,
        rsi: Decimal,
        rsi_zone: RSIZone,
    ) -> None:
        """Check for entry signals with hysteresis protection."""
        if not self._grid:
            return

        # Check signal cooldown to prevent signal stacking
        if self._signal_cooldown > 0:
            logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping entry check")
            return

        allowed_dir = self._get_allowed_direction(rsi_zone, self._current_trend)
        if allowed_dir == "none":
            return

        curr_low = kline.low
        curr_high = kline.high

        # Calculate stop loss distance for take profit
        atr = self._atr_calc.atr if self._atr_calc else current_price * self._config.max_stop_loss_pct
        tp_distance = self._grid.grid_spacing * Decimal(self._config.take_profit_grids)

        # Long entry: price dips to touch grid level
        if allowed_dir in ["long_only", "both"]:
            for level in self._grid.levels:
                if level.state == GridLevelState.EMPTY and curr_low <= level.price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level.index, "long", level.price, current_price):
                        continue

                    level.state = GridLevelState.LONG_FILLED
                    level.filled_at = datetime.now(timezone.utc)

                    success = await self._open_position(
                        side=PositionSide.LONG,
                        price=level.price,
                        rsi=rsi,
                        rsi_zone=rsi_zone,
                        grid_level=level.index,
                    )
                    if success:
                        self._signal_cooldown = self._cooldown_bars  # Reset cooldown
                        self._last_triggered_level = level.index  # Track for hysteresis
                        return

        # Short entry: price rises to touch grid level
        if allowed_dir in ["short_only", "both"]:
            for level in reversed(self._grid.levels):
                if level.state == GridLevelState.EMPTY and curr_high >= level.price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level.index, "short", level.price, current_price):
                        continue

                    level.state = GridLevelState.SHORT_FILLED
                    level.filled_at = datetime.now(timezone.utc)

                    success = await self._open_position(
                        side=PositionSide.SHORT,
                        price=level.price,
                        rsi=rsi,
                        rsi_zone=rsi_zone,
                        grid_level=level.index,
                    )
                    if success:
                        self._signal_cooldown = self._cooldown_bars  # Reset cooldown
                        self._last_triggered_level = level.index  # Track for hysteresis
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

        # RSI-based exit
        if self._position.side == PositionSide.LONG:
            if rsi_zone == RSIZone.OVERBOUGHT:
                await self._close_position(current_price, ExitReason.RSI_EXIT)
                return
        else:  # SHORT
            if rsi_zone == RSIZone.OVERSOLD:
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

    def _parse_timeframe_seconds(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return value * multipliers.get(unit, 60)

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
                "stop_loss_order_id": self._position.stop_loss_order_id,
                "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
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
                timeframe=config_data.get("timeframe", "15m"),
                rsi_period=config_data.get("rsi_period", 14),
                oversold_level=config_data.get("oversold_level", 30),
                overbought_level=config_data.get("overbought_level", 70),
                grid_count=config_data.get("grid_count", 10),
                atr_multiplier=Decimal(config_data.get("atr_multiplier", "3.0")),
                leverage=config_data.get("leverage", 2),
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
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
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
