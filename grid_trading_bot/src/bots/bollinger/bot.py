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
import time
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import Kline, MarketType, OrderSide
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.fund_manager import SignalCoordinator, SignalDirection, CoordinationResult
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
        signal_coordinator: Optional[SignalCoordinator] = None,
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

        # Signal Coordinator for multi-bot conflict prevention
        self._signal_coordinator = signal_coordinator or SignalCoordinator.get_instance()

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

        # Hysteresis: track last triggered level to prevent oscillation (configurable)
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = config.hysteresis_pct  # Use config value

        # Signal cooldown to prevent signal stacking (configurable)
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = config.cooldown_bars  # Use config value

        # Bar counting for timeout exit
        self._current_bar: int = 0
        self._entry_bar: int = 0

        # Background tasks
        self._monitor_task: Optional[asyncio.Task] = None

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

        # Slippage tracking (from BaseBot)
        self._init_slippage_tracking()

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

        # Initialize per-strategy risk tracking (風控相互影響隔離)
        self.set_strategy_initial_capital(self._capital)

        # Register for global risk tracking (多策略風控協調)
        await self.register_bot_for_global_risk(self._bot_id, self._capital)

        # Register for circuit breaker coordination (部分熔斷協調)
        await self.register_strategy_for_cb(
            bot_id=self._bot_id,
            strategy_type="bollinger",
            dependencies=None,
        )

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
        if self._klines and len(self._klines) > 0:
            current_price = self._klines[-1].close
            self._initialize_grid(current_price)

        # 6. Sync existing position
        await self._sync_position()

        # 7. Subscribe to kline updates
        await self._subscribe_klines()

        # 8. Start background monitor
        self._monitor_task = asyncio.create_task(self._background_monitor())

        # 9. Start periodic state saving
        self._start_save_task()

        # 10. Start position reconciliation (detect manual operations)
        self._start_position_reconciliation()

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

        # Guard against division by zero
        if self._config.grid_count <= 0:
            grid_spacing = range_size
        else:
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
            """Sync callback wrapper with task tracking."""
            if self._state == BotState.RUNNING:
                task = asyncio.create_task(self._on_kline(kline))
                self._kline_tasks.add(task)
                task.add_done_callback(lambda t: self._kline_tasks.discard(t))

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
            # Update klines list
            self._klines.append(kline)
            if len(self._klines) > 300:
                self._klines = self._klines[-300:]

            # Increment bar counter for timeout tracking
            self._current_bar += 1

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
                self.mark_capital_updated()  # Track data freshness for bypass prevention
                self._stats.update_drawdown(self._capital, self._initial_capital)

                # Record capital for CB validation (熔斷誤觸發防護)
                self.record_capital_for_validation(self._capital)

                # Apply consecutive loss decay (prevents permanent lockout)
                self.apply_consecutive_loss_decay()

                # Update virtual position unrealized P&L
                if self._position:
                    current_price = self._klines[-1].close if self._klines and len(self._klines) > 0 else Decimal("0")
                    if current_price > 0:
                        self.update_virtual_unrealized_pnl(self._config.symbol, current_price)
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
                        current_price = self._klines[-1].close if self._klines and len(self._klines) > 0 else await self._get_current_price()
                        cb_result = await self.trigger_circuit_breaker_safe(
                            reason=f"CRITICAL_RISK: {risk_result.get('action', 'unknown')}",
                            current_price=current_price,
                            current_capital=self._capital,
                            partial=True,
                        )
                        if cb_result["triggered"]:
                            self._position = None

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
                    current_price = self._klines[-1].close if self._klines and len(self._klines) > 0 else await self._get_current_price()
                    stop_loss_order_id = getattr(self._position, 'stop_loss_order_id', None)
                    sl_check = await self.comprehensive_stop_loss_check(
                        symbol=self._config.symbol,
                        current_price=current_price,
                        entry_price=self._position.entry_price,
                        position_side=self._position.side.value.upper(),
                        quantity=self._position.quantity,
                        stop_loss_pct=self._config.stop_loss_pct,
                        stop_loss_order_id=stop_loss_order_id,
                        leverage=self._config.leverage,
                    )

                    if sl_check["action_needed"]:
                        logger.warning(
                            f"Stop loss protection triggered: {sl_check['action_type']} "
                            f"(urgency: {sl_check['urgency']})"
                        )

                        if sl_check["action_type"] == "EMERGENCY_CLOSE":
                            await self.execute_emergency_close(
                                symbol=self._config.symbol,
                                side=self._position.side.value.upper(),
                                quantity=self._position.quantity,
                                reason=sl_check["details"].get("emergency", {}).get("reason", "UNKNOWN"),
                            )
                            self._position = None
                            self.reset_stop_loss_protection()

                        elif sl_check["action_type"] == "REPLACE_SL":
                            replace_result = await self.replace_failed_stop_loss(
                                symbol=self._config.symbol,
                                side=self._position.side.value.upper(),
                                quantity=self._position.quantity,
                                entry_price=self._position.entry_price,
                                stop_loss_pct=self._config.stop_loss_pct,
                            )
                            if replace_result["success"] and hasattr(self._position, 'stop_loss_order_id'):
                                self._position.stop_loss_order_id = replace_result["new_order_id"]

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

                await self._send_heartbeat()
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background monitor error: {e}")
                await asyncio.sleep(10)

    # =========================================================================
    # BB_TREND_GRID Trading Logic
    # =========================================================================

    def _check_hysteresis(self, level_index: int, direction: str, grid_price: Decimal, current_price: Decimal) -> bool:
        """
        Check if price has moved enough from last triggered level (hysteresis).

        Prevents oscillation around grid levels by requiring price to move
        away by a minimum percentage before retriggering the same level.

        Note: This feature is disabled by default based on backtest results
        showing it slightly reduces returns while marginally improving drawdown.
        """
        # Check if hysteresis is enabled in config
        if not self._config.use_hysteresis:
            return True

        if self._last_triggered_level is None:
            return True
        if self._last_triggered_level != level_index:
            return True

        hysteresis_buffer = grid_price * self._hysteresis_pct

        if direction == "long":
            if current_price < grid_price - hysteresis_buffer:
                return True
        else:
            if current_price > grid_price + hysteresis_buffer:
                return True

        logger.debug(
            f"Hysteresis active: level {level_index} recently triggered, "
            f"price {current_price} too close to {grid_price}"
        )
        return False

    async def _process_grid_kline(self, kline: Kline) -> None:
        """
        Process grid trading logic using kline data with hysteresis protection.

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

        # Decrement signal cooldown
        if self._signal_cooldown > 0:
            self._signal_cooldown -= 1

        # Check existing position for exit
        if self._position:
            await self._check_position_exit(current_price, kline_high, kline_low)
            return  # Don't open new position while holding one

        # Check signal cooldown (if enabled)
        if self._config.use_signal_cooldown and self._signal_cooldown > 0:
            logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping entry")
            return

        # Look for entry opportunities
        for level in self._grid.levels:
            if level.state != GridLevelState.EMPTY:
                continue

            grid_price = level.price

            # Bullish trend: LONG only (buy dips)
            if self._current_trend == 1:
                # Check if kline low touched grid level
                if kline_low <= grid_price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level.index, "long", grid_price, current_price):
                        continue

                    success = await self._open_position(PositionSide.LONG, grid_price, level.index)
                    if success:
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._cooldown_bars
                        if self._config.use_hysteresis:
                            self._last_triggered_level = level.index
                    break

            # Bearish trend: SHORT only (sell rallies)
            elif self._current_trend == -1:
                # Check if kline high touched grid level
                if kline_high >= grid_price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level.index, "short", grid_price, current_price):
                        continue

                    success = await self._open_position(PositionSide.SHORT, grid_price, level.index)
                    if success:
                        if self._config.use_signal_cooldown:
                            self._signal_cooldown = self._cooldown_bars
                        if self._config.use_hysteresis:
                            self._last_triggered_level = level.index
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
                self.record_stop_loss_trigger()
                self.clear_stop_loss_sync()
                return

            # Check trend change
            if self._current_trend == -1:
                await self._close_position(current_price, ExitReason.TREND_CHANGE)
                return

            # Check timeout exit (close losing position after max_hold_bars)
            if self._check_timeout_exit(current_price):
                logger.warning(f"Timeout exit triggered: held {self._current_bar - self._entry_bar} bars")
                await self._close_position(current_price, ExitReason.TIMEOUT)
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
                self.record_stop_loss_trigger()
                self.clear_stop_loss_sync()
                return

            # Check trend change
            if self._current_trend == 1:
                await self._close_position(current_price, ExitReason.TREND_CHANGE)
                return

            # Check timeout exit (close losing position after max_hold_bars)
            if self._check_timeout_exit(current_price):
                logger.warning(f"Timeout exit triggered: held {self._current_bar - self._entry_bar} bars")
                await self._close_position(current_price, ExitReason.TIMEOUT)
                return

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
            # Stage 6: Close old position if reversing direction
            if self._position and self._position.side != side:
                logger.info(f"Reversing direction: closing {self._position.side.value} before opening {side.value}")
                await self._close_position(price, ExitReason.TREND_CHANGE)

            # Validate price before calculation (indicator boundary check)
            if not self._validate_price(price, "entry_price"):
                logger.warning(f"Invalid entry price: {price}")
                return False

            # Calculate position size
            trade_value = self._capital * self._config.position_size_pct
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
                    reason=f"BB grid level {grid_level_index}",
                )
                if not result.approved:
                    logger.warning(
                        f"Signal blocked by coordinator: {result.message} "
                        f"(conflict with {result.conflicting_bot})"
                    )
                    return False

            # Check global risk limits (多策略風控協調 - 防止總體風險超標)
            exposure = quantity * price
            global_ok, global_msg, global_details = await self.check_global_risk_limits(
                bot_id=self._bot_id,
                symbol=self._config.symbol,
                additional_exposure=exposure,
            )
            if not global_ok:
                logger.warning(f"Global risk check failed: {global_msg}")
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

                # Calculate TP/SL prices
                grid_spacing = self._grid.grid_spacing if self._grid else price * Decimal("0.01")
                if side == PositionSide.LONG:
                    tp_price = fill_price + (grid_spacing * self._config.take_profit_grids)
                    sl_price = fill_price * (Decimal("1") - self._config.stop_loss_pct)
                else:
                    tp_price = fill_price - (grid_spacing * self._config.take_profit_grids)
                    sl_price = fill_price * (Decimal("1") + self._config.stop_loss_pct)

                # Normalize TP/SL prices to exchange tick size
                symbol_info = await self._get_symbol_info(self._config.symbol)
                tp_price = self._normalize_price(tp_price, symbol_info)
                sl_price = self._normalize_price(sl_price, symbol_info)

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

                # Record entry bar for timeout tracking
                self._entry_bar = self._current_bar

                # Place exchange stop loss order if enabled
                if self._config.use_exchange_stop_loss:
                    await self._place_stop_loss_order()

                # Mark grid level as filled
                if self._grid and 0 <= grid_level_index < len(self._grid.levels):
                    level = self._grid.levels[grid_level_index]
                    level.state = GridLevelState.LONG_FILLED if side == PositionSide.LONG else GridLevelState.SHORT_FILLED
                    level.entry_price = fill_price
                    level.entry_time = datetime.now(timezone.utc)

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
                )

                # Record cost basis entry (持倉歸屬 - FIFO tracking)
                self.record_cost_basis_entry(
                    symbol=self._config.symbol,
                    quantity=fill_qty,
                    price=fill_price,
                    order_id=order_id_str,
                    fee=fee,
                )

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

        return False

    async def _place_stop_loss_order(self) -> None:
        """Place stop loss order on exchange using STOP_MARKET."""
        if not self._position or not self._position.stop_loss_price:
            return

        try:
            # Determine close side (opposite of position)
            if self._position.side == PositionSide.LONG:
                close_side = OrderSide.SELL
            else:
                close_side = OrderSide.BUY

            # Place STOP_MARKET order
            sl_order = await self._exchange.futures_create_order(
                symbol=self._config.symbol,
                side=close_side.value,
                order_type="STOP_MARKET",
                quantity=self._position.quantity,
                stop_price=self._position.stop_loss_price,
                reduce_only=True,
                bot_id=self._bot_id,
            )

            if sl_order:
                self._position.stop_loss_order_id = str(sl_order.order_id)
                logger.info(
                    f"Stop loss order placed: {close_side.value} {self._position.quantity} "
                    f"@ {self._position.stop_loss_price}, ID={sl_order.order_id}"
                )

        except Exception as e:
            logger.error(f"Failed to place stop loss order: {e}")

    async def _cancel_stop_loss_order(self) -> bool:
        """Cancel stop loss order with verification."""
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
            return True

        elif result["was_triggered"]:
            logger.warning(
                f"Stop loss {self._position.stop_loss_order_id} was triggered - "
                f"forcing position sync"
            )
            self._position.stop_loss_order_id = None
            await self._sync_position()
            return True

        else:
            logger.error(
                f"Failed to cancel stop loss after {result['attempts']} attempts: "
                f"{result['error_message']}"
            )
            return False

    async def _close_position(self, price: Decimal, reason: ExitReason) -> bool:
        """Close current position."""
        if not self._position:
            return False

        try:
            # Cancel stop loss order first (if any)
            if self._position.stop_loss_order_id:
                await self._cancel_stop_loss_order()

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

                # Apply leverage to PnL
                pnl *= Decimal(self._config.leverage)

                # Calculate fee (entry + exit)
                fee = quantity * (entry_price + exit_price) * self._config.fee_rate

                # Calculate PnL percentage with zero-division protection
                denominator = entry_price * quantity
                pnl_pct = (pnl / denominator * Decimal("100")) if denominator > 0 else Decimal("0")

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
                    fee=fee,
                    entry_time=self._position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason.value,
                )
                self._stats.record_trade(trade)

                # Close cost basis with FIFO (持倉歸屬 - P&L attribution)
                order_id_str = str(getattr(order, "order_id", ""))
                close_fee = quantity * exit_price * self._config.fee_rate
                cost_basis_result = self.close_cost_basis_fifo(
                    symbol=self._config.symbol,
                    close_quantity=quantity,
                    close_price=exit_price,
                    close_order_id=order_id_str,
                    close_fee=close_fee,
                )
                logger.debug(
                    f"Cost basis closed: {len(cost_basis_result.get('matched_lots', []))} lots, "
                    f"Attributed P&L: {cost_basis_result.get('total_realized_pnl')}"
                )

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
            await self._handle_order_rejection(
                symbol=self._config.symbol,
                side="SELL" if self._position and self._position.side == PositionSide.LONG else "BUY",
                quantity=self._position.quantity if self._position else Decimal("0"),
                error=e,
            )

        return False

    # =========================================================================
    # State Persistence (with validation, checksum, and concurrent protection)
    # =========================================================================

    def _get_state_data(self) -> Dict[str, Any]:
        """
        Override BaseBot method to provide Bollinger specific state.

        Returns:
            Dictionary of state data for persistence
        """
        state_data = {
            "capital": str(self._capital),
            "initial_capital": str(self._initial_capital),
            "current_trend": self._current_trend,
            "current_sma": str(self._current_sma) if self._current_sma else None,
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
                "grid_level_index": self._position.grid_level_index,
                "take_profit_price": str(self._position.take_profit_price) if self._position.take_profit_price else None,
                "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                "stop_loss_order_id": self._position.stop_loss_order_id,
            }

        return state_data

    async def _save_state(self) -> None:
        """Save bot state to database with concurrent protection."""

        async def do_save():
            config = {
                "symbol": self._config.symbol,
                "timeframe": self._config.timeframe,
                "bb_period": self._config.bb_period,
                "bb_std": str(self._config.bb_std),
                "grid_count": self._config.grid_count,
                "leverage": self._config.leverage,
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            # Create validated snapshot with metadata
            snapshot = self._create_state_snapshot()

            async def persist_to_db(snapshot_data: Dict[str, Any]):
                await self._data_manager.save_bot_state(
                    bot_id=self._bot_id,
                    bot_type="bollinger",
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
            while self._state == BotState.RUNNING:
                await asyncio.sleep(self._save_interval_minutes * 60)
                if self._state == BotState.RUNNING:
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
    ) -> Optional["BollingerBot"]:
        """
        Restore a BollingerBot from saved state with validation.

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
            Restored BollingerBot or None if not found/invalid
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
                    from datetime import timezone
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
            config = BollingerConfig(
                symbol=config_data.get("symbol", ""),
                timeframe=config_data.get("timeframe", "15m"),
                bb_period=config_data.get("bb_period", 20),
                bb_std=Decimal(config_data.get("bb_std", "2.0")),
                grid_count=config_data.get("grid_count", 10),
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
            if inner_state.get("current_sma"):
                bot._current_sma = Decimal(inner_state["current_sma"])
            bot._signal_cooldown = inner_state.get("signal_cooldown", 0)
            bot._last_triggered_level = inner_state.get("last_triggered_level")

            # Restore stats
            stats_data = inner_state.get("stats", {})
            bot._stats.total_trades = stats_data.get("total_trades", 0)
            bot._stats.winning_trades = stats_data.get("winning_trades", 0)
            bot._stats.losing_trades = stats_data.get("losing_trades", 0)
            bot._stats.total_pnl = Decimal(stats_data.get("total_pnl", "0"))

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
                    symbol=config.symbol,
                    center_price=center,
                    upper_price=upper,
                    lower_price=lower,
                    grid_count=grid_count,
                    levels=levels,
                    version=grid_data.get("version", 1),
                )

            # Restore position from saved state
            position_data = inner_state.get("position")
            if position_data:
                bot._position = Position(
                    symbol=config.symbol,
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    leverage=config.leverage,
                    entry_time=datetime.fromisoformat(position_data["entry_time"]) if position_data.get("entry_time") else None,
                    grid_level_index=position_data.get("grid_level_index"),
                    take_profit_price=Decimal(position_data["take_profit_price"]) if position_data.get("take_profit_price") else None,
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
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
                    bot._position = Position(
                        symbol=config.symbol,
                        side=PositionSide(side_str),
                        entry_price=exchange_pos.entry_price,
                        quantity=exchange_pos.quantity,
                        leverage=config.leverage,
                    )
            except Exception as e:
                logger.warning(f"Failed to verify position sync on restore: {e}")

            logger.info(
                f"Restored BollingerBot: {bot_id}, "
                f"position={'yes' if bot._position else 'no'}"
            )
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return None
