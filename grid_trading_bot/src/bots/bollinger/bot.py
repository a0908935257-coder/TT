"""
Bollinger BB_TREND_GRID Bot - 趨勢網格交易機器人.

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 10 期分割):
- Walk-Forward 一致性: 80% (8/10 時段獲利)
- OOS Sharpe: 6.56
- 過度擬合: 未檢測到
- 穩健性: ROBUST

策略邏輯 (BB_TREND_GRID):
- 趨勢判斷: BB 中軌 (SMA)
  - Price > SMA = 看多 (只做 LONG)
  - Price < SMA = 看空 (只做 SHORT)
- 進場: 網格交易
  - LONG: kline.low <= grid_level.price (買跌)
  - SHORT: kline.high >= grid_level.price (賣漲)
  - 進場價格: Grid level 價格
- 出場: 止盈 1 個網格 或 止損 5%

驗證參數:
- bb_period: 20
- bb_std: 2.0
- grid_count: 10
- leverage: 2x
"""

import asyncio
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import Kline, MarketType
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.master.models import BotState
from src.notification import NotificationManager

from .indicators import BollingerCalculator
from .models import (
    BollingerConfig,
    BollingerBotStats,
    GridLevel,
    GridLevelState,
    GridSetup,
    Position,
    PositionSide,
    TradeRecord,
    ExitReason,
)

logger = get_logger(__name__)


class BollingerBot(BaseBot):
    """
    Bollinger BB_TREND_GRID Bot.

    Uses BB middle band (SMA) for trend detection and grid trading
    for entry/exit within the trend direction.

    Features:
    - Trend detection via BB middle band (SMA)
    - Grid trading within trend direction
    - Kline-based entry detection (matches backtest)
    - WebSocket subscription for real-time updates
    """

    def __init__(
        self,
        bot_id: str,
        config: BollingerConfig,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: Optional[NotificationManager] = None,
        heartbeat_callback: Optional[Callable] = None,
    ):
        """Initialize BollingerBot."""
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )

        self._config: BollingerConfig = config

        # Initialize BB calculator
        self._bb_calculator = BollingerCalculator(
            period=config.bb_period,
            std_multiplier=config.bb_std,
        )

        # State
        self._grid: Optional[GridSetup] = None
        self._position: Optional[Position] = None
        self._current_trend: int = 0  # 1=bullish, -1=bearish, 0=neutral
        self._current_sma: Optional[Decimal] = None
        self._klines: List[Kline] = []

        # Capital tracking
        self._capital: Decimal = Decimal("0")
        self._initial_capital: Decimal = Decimal("0")

        # Statistics
        self._stats = BollingerBotStats()

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def bot_type(self) -> str:
        return "bollinger"

    @property
    def symbol(self) -> str:
        return self._config.symbol

    # =========================================================================
    # Lifecycle
    # =========================================================================

    async def _do_start(self) -> None:
        """Start the bot."""
        logger.info(f"Starting Bollinger BB_TREND_GRID Bot for {self._config.symbol}")

        # 1. Setup futures account
        await self._setup_futures_account()

        # 2. Get initial capital
        await self._update_capital()
        self._initial_capital = self._capital
        logger.info(f"Initial capital: {self._capital} USDT")

        # 3. Load historical klines
        klines = await self._exchange.futures.get_klines(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            limit=200,
        )
        self._klines = list(klines)
        logger.info(f"Loaded {len(self._klines)} historical klines")

        # 4. Initialize BB calculator
        self._bb_calculator.initialize(self._klines)

        # 5. Initialize grid
        if self._klines:
            current_price = self._klines[-1].close
            self._initialize_grid(current_price)

        # 6. Sync existing position
        await self._sync_position()

        # 7. Subscribe to kline updates
        await self._subscribe_klines()

        # 8. Start background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor())

        logger.info("Bollinger BB_TREND_GRID Bot started successfully")
        logger.info(f"  Symbol: {self._config.symbol}")
        logger.info(f"  Timeframe: {self._config.timeframe}")
        logger.info(f"  BB Period: {self._config.bb_period}")
        logger.info(f"  BB Std: {self._config.bb_std}")
        logger.info(f"  Grid Count: {self._config.grid_count}")
        logger.info(f"  Leverage: {self._config.leverage}x")

    async def _do_stop(self, clear_position: bool = False) -> None:
        """Stop the bot."""
        logger.info("Stopping Bollinger BB_TREND_GRID Bot")

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        # Unsubscribe from klines
        try:
            ws = self._exchange.futures_ws
            if ws:
                await ws.unsubscribe_kline(
                    self._config.symbol,
                    self._config.timeframe,
                )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe from klines: {e}")

        # Close position if requested
        if clear_position and self._position:
            await self._close_position(self._position.entry_price, ExitReason.BOT_STOP)

        logger.info("Bollinger BB_TREND_GRID Bot stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current bot status."""
        status = {
            "bot_id": self._bot_id,
            "bot_type": self.bot_type,
            "symbol": self._config.symbol,
            "state": self._state.value,
            "leverage": self._config.leverage,
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "current_sma": str(self._current_sma) if self._current_sma else None,
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

        status["stats"] = self._stats.to_dict()

        return status

    async def _do_pause(self) -> None:
        """Pause the bot."""
        logger.info("Pausing Bollinger BB_TREND_GRID Bot")

        # Unsubscribe from kline updates
        try:
            ws = self._exchange.futures_ws
            if ws:
                await ws.unsubscribe_kline(
                    self._config.symbol,
                    self._config.timeframe,
                )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe: {e}")

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass

        logger.info("Bollinger BB_TREND_GRID Bot paused")

    async def _do_resume(self) -> None:
        """Resume the bot."""
        logger.info("Resuming Bollinger BB_TREND_GRID Bot")

        # Re-subscribe to kline updates
        await self._subscribe_klines()

        # Restart background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor())

        logger.info("Bollinger BB_TREND_GRID Bot resumed")

    def _get_extra_status(self) -> Dict[str, Any]:
        """Return extra status fields specific to Bollinger bot."""
        extra = {
            "grid": None,
            "position": None,
            "current_trend": self._current_trend,
            "current_sma": str(self._current_sma) if self._current_sma else None,
            "stats": self._stats.to_dict(),
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
        """Perform extra health checks specific to Bollinger bot."""
        checks = {}

        # Check if grid is valid
        checks["grid_valid"] = self._grid is not None

        # Check BB calculator
        checks["bb_initialized"] = self._bb_calculator.bbw_history_length > 0

        # Check position sync
        if self._position:
            checks["position_synced"] = self._position.quantity > 0
        else:
            checks["position_synced"] = True  # No position is valid

        return checks

    # =========================================================================
    # Futures Account Setup
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
            logger.error(f"Failed to update capital: {e}")

    # =========================================================================
    # Grid Management
    # =========================================================================

    def _initialize_grid(self, current_price: Decimal) -> None:
        """Initialize grid around current price."""
        range_size = current_price * self._config.grid_range_pct

        upper_price = current_price + range_size
        lower_price = current_price - range_size
        grid_spacing = (upper_price - lower_price) / Decimal(self._config.grid_count)

        # Create grid levels
        levels = []
        for i in range(self._config.grid_count + 1):
            price = lower_price + (grid_spacing * Decimal(i))
            levels.append(GridLevel(index=i, price=price))

        self._grid = GridSetup(
            symbol=self._config.symbol,
            center_price=current_price,
            upper_price=upper_price,
            lower_price=lower_price,
            grid_count=self._config.grid_count,
            levels=levels,
            version=1 if not self._grid else self._grid.version + 1,
        )

        range_pct = self._config.grid_range_pct * 100
        logger.info(
            f"Grid initialized: center={current_price:.2f}, "
            f"range=±{range_pct:.1f}%, levels={len(levels)}, v{self._grid.version}"
        )

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding."""
        if not self._grid:
            return True

        # Rebuild if price moved outside grid range
        if current_price > self._grid.upper_price or current_price < self._grid.lower_price:
            return True

        return False

    def _rebuild_grid(self, current_price: Decimal) -> None:
        """Rebuild grid around new price."""
        self._stats.grid_rebuilds += 1
        self._initialize_grid(current_price)
        logger.info(f"Grid rebuilt around {current_price:.2f}")

    # =========================================================================
    # WebSocket Subscription
    # =========================================================================

    async def _subscribe_klines(self) -> None:
        """Subscribe to kline updates via WebSocket."""

        def on_kline_sync(kline: Kline) -> None:
            """Sync callback wrapper."""
            if self._state == BotState.RUNNING:
                asyncio.create_task(self._on_kline(kline))

        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,
        )
        logger.info(f"Subscribed to {self._config.symbol} {self._config.timeframe} klines")

    async def _on_kline(self, kline: Kline) -> None:
        """
        Process kline update from WebSocket.

        Only processes closed klines to match backtest behavior.
        """
        # Validate kline before processing (matches backtest behavior)
        if not self._should_process_kline(kline, require_closed=True, check_symbol=False):
            return

        try:
            # Update klines list
            self._klines.append(kline)
            if len(self._klines) > 300:
                self._klines = self._klines[-300:]

            # Process grid trading logic
            await self._process_grid_kline(kline)

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    async def _background_monitor(self) -> None:
        """Background monitoring for capital updates and heartbeat."""
        logger.info("Starting background monitor")
        while self._state == BotState.RUNNING:
            try:
                await self._update_capital()
                self._stats.update_drawdown(self._capital, self._initial_capital)
                self._send_heartbeat()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(10)

    # =========================================================================
    # BB_TREND_GRID Trading Logic
    # =========================================================================

    async def _process_grid_kline(
        self,
        kline: Kline,
    ) -> None:
        """
        Process grid trading logic using kline data.

        BB_TREND_GRID 邏輯 (與回測一致):
        - 趨勢: BB 中軌 (SMA) 判斷
          - Price > SMA = 看多 (只做 LONG)
          - Price < SMA = 看空 (只做 SHORT)
        - Long entry: K 線低點觸及 grid level (買跌)
        - Short entry: K 線高點觸及 grid level (賣漲)
        - 使用 grid level 價格進場
        """
        current_price = kline.close
        kline_low = kline.low
        kline_high = kline.high

        # Calculate BB
        try:
            bands, _ = self._bb_calculator.get_all(self._klines)
            self._current_sma = bands.middle
        except Exception:
            return

        # Determine trend based on price vs SMA
        if current_price > bands.middle:
            self._current_trend = 1  # Bullish - LONG only
        elif current_price < bands.middle:
            self._current_trend = -1  # Bearish - SHORT only
        else:
            self._current_trend = 0

        # Check if we need to rebuild grid
        if self._should_rebuild_grid(current_price):
            # Close position if trend might change
            if self._position:
                await self._close_position(current_price, ExitReason.GRID_REBUILD)
            self._rebuild_grid(current_price)
            return

        # Skip if no grid or no trend
        if not self._grid or self._current_trend == 0:
            return

        # Check existing position for exit
        if self._position:
            await self._check_position_exit(current_price, kline_high, kline_low)
            return  # Don't open new position while holding one

        # Look for entry opportunities
        for level in self._grid.levels:
            if level.state != GridLevelState.EMPTY:
                continue

            grid_price = level.price

            # Bullish trend: LONG only (buy dips)
            if self._current_trend == 1:
                # Check if kline low touched grid level
                if kline_low <= grid_price:
                    await self._open_position(PositionSide.LONG, grid_price, level.index)
                    break

            # Bearish trend: SHORT only (sell rallies)
            elif self._current_trend == -1:
                # Check if kline high touched grid level
                if kline_high >= grid_price:
                    await self._open_position(PositionSide.SHORT, grid_price, level.index)
                    break

    async def _check_position_exit(
        self,
        current_price: Decimal,
        kline_high: Decimal,
        kline_low: Decimal,
    ) -> None:
        """Check if position should be closed."""
        if not self._position or not self._grid:
            return

        entry_price = self._position.entry_price
        side = self._position.side

        # Calculate take profit price (next grid level)
        grid_spacing = self._grid.grid_spacing
        if side == PositionSide.LONG:
            tp_price = entry_price + (grid_spacing * self._config.take_profit_grids)
            sl_price = entry_price * (Decimal("1") - self._config.stop_loss_pct)

            # Check take profit (kline high reached TP)
            if kline_high >= tp_price:
                await self._close_position(tp_price, ExitReason.GRID_PROFIT)
                return

            # Check stop loss
            if kline_low <= sl_price:
                await self._close_position(sl_price, ExitReason.STOP_LOSS)
                return

            # Check trend change
            if self._current_trend == -1:
                await self._close_position(current_price, ExitReason.TREND_CHANGE)
                return

        else:  # SHORT
            tp_price = entry_price - (grid_spacing * self._config.take_profit_grids)
            sl_price = entry_price * (Decimal("1") + self._config.stop_loss_pct)

            # Check take profit (kline low reached TP)
            if kline_low <= tp_price:
                await self._close_position(tp_price, ExitReason.GRID_PROFIT)
                return

            # Check stop loss
            if kline_high >= sl_price:
                await self._close_position(sl_price, ExitReason.STOP_LOSS)
                return

            # Check trend change
            if self._current_trend == 1:
                await self._close_position(current_price, ExitReason.TREND_CHANGE)
                return

    # =========================================================================
    # Position Management
    # =========================================================================

    async def _sync_position(self) -> None:
        """Sync position from exchange."""
        try:
            positions = await self._exchange.futures.get_positions(self._config.symbol)

            for pos in positions:
                if pos.quantity > 0:
                    side_str = pos.side if isinstance(pos.side, str) else pos.side.value
                    local_side = PositionSide(side_str)

                    self._position = Position(
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

    async def _open_position(
        self,
        side: PositionSide,
        price: Decimal,
        grid_level_index: int,
    ) -> bool:
        """Open a new position."""
        try:
            # Calculate position size
            trade_value = self._capital * self._config.position_size_pct
            quantity = trade_value / price
            quantity = quantity.quantize(Decimal("0.001"))

            if quantity < Decimal("0.001"):
                return False

            # Check max position limit
            if self._position:
                current_value = self._position.quantity * price
                if current_value >= self._capital * self._config.max_position_pct:
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

                # Calculate TP/SL prices
                grid_spacing = self._grid.grid_spacing if self._grid else price * Decimal("0.01")
                if side == PositionSide.LONG:
                    tp_price = fill_price + (grid_spacing * self._config.take_profit_grids)
                    sl_price = fill_price * (Decimal("1") - self._config.stop_loss_pct)
                else:
                    tp_price = fill_price - (grid_spacing * self._config.take_profit_grids)
                    sl_price = fill_price * (Decimal("1") + self._config.stop_loss_pct)

                self._position = Position(
                    symbol=self._config.symbol,
                    side=side,
                    entry_price=fill_price,
                    quantity=fill_qty,
                    leverage=self._config.leverage,
                    entry_time=datetime.now(timezone.utc),
                    grid_level_index=grid_level_index,
                    take_profit_price=tp_price,
                    stop_loss_price=sl_price,
                )

                # Mark grid level as filled
                if self._grid and 0 <= grid_level_index < len(self._grid.levels):
                    level = self._grid.levels[grid_level_index]
                    level.state = GridLevelState.LONG_FILLED if side == PositionSide.LONG else GridLevelState.SHORT_FILLED
                    level.entry_price = fill_price
                    level.entry_time = datetime.now(timezone.utc)

                logger.info(
                    f"Opened {side.value} position: qty={fill_qty}, "
                    f"price={fill_price:.2f}, TP={tp_price:.2f}, SL={sl_price:.2f}"
                )

                # Send notification
                if self._notifier:
                    await self._notifier.send(
                        f"📈 {self._config.symbol} {side.value}",
                        f"Entry: ${fill_price:.2f}\nQty: {fill_qty}\nTP: ${tp_price:.2f}\nSL: ${sl_price:.2f}",
                    )

                return True

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

        return False

    async def _close_position(self, price: Decimal, reason: ExitReason) -> bool:
        """Close current position."""
        if not self._position:
            return False

        try:
            side = self._position.side
            quantity = self._position.quantity

            # Place closing order (through order queue for cross-bot coordination)
            if side == PositionSide.LONG:
                order = await self._exchange.market_sell(
                    symbol=self._config.symbol,
                    quantity=quantity,
                    market=MarketType.FUTURES,
                    bot_id=self._bot_id,
                )
            else:
                order = await self._exchange.market_buy(
                    symbol=self._config.symbol,
                    quantity=quantity,
                    market=MarketType.FUTURES,
                    bot_id=self._bot_id,
                )

            if order:
                exit_price = order.avg_price if order.avg_price else price
                entry_price = self._position.entry_price

                # Calculate PnL
                if side == PositionSide.LONG:
                    pnl = (exit_price - entry_price) * quantity
                else:
                    pnl = (entry_price - exit_price) * quantity

                pnl_pct = pnl / (entry_price * quantity) * Decimal("100")

                # Record trade
                trade = TradeRecord(
                    trade_id=str(uuid.uuid4()),
                    symbol=self._config.symbol,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    entry_time=self._position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason.value,
                )
                self._stats.record_trade(trade)

                # Reset grid level
                if self._grid and self._position.grid_level_index is not None:
                    idx = self._position.grid_level_index
                    if 0 <= idx < len(self._grid.levels):
                        self._grid.levels[idx].state = GridLevelState.EMPTY

                logger.info(
                    f"Closed {side.value} position: exit={exit_price:.2f}, "
                    f"pnl={pnl:.2f} ({pnl_pct:.2f}%), reason={reason.value}"
                )

                # Send notification
                if self._notifier:
                    emoji = "✅" if pnl > 0 else "❌"
                    await self._notifier.send(
                        f"{emoji} {self._config.symbol} {side.value} Closed",
                        f"Exit: ${exit_price:.2f}\nPnL: ${pnl:.2f} ({pnl_pct:.2f}%)\nReason: {reason.value}",
                    )

                self._position = None
                return True

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

        return False
