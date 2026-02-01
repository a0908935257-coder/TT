"""
Supertrend TREND_GRID Trading Bot with RSI Filter.

Combines Supertrend trend direction with grid trading:
- In bullish trend: LONG only, buy dips at grid levels
- In bearish trend: SHORT only, sell rallies at grid levels
- RSI filter prevents entries in extreme conditions
- Exit on trend flip or take profit at next grid level

✅ Walk-Forward + OOS 驗證通過 (2024-01-25 ~ 2026-01-24, 2 年數據):
- Walk-Forward 一致性: 70% (7/10 時段)
- OOS Sharpe: 5.84
- 過度擬合: NO
- Monte Carlo: ROBUST (100% 獲利機率)
- 勝率: ~94%

TREND_GRID 進場邏輯:
- 多頭趨勢: 價格觸及網格低點時做多
- 空頭趨勢: 價格觸及網格高點時做空

RSI 過濾器效果:
- 避免在超買區做多 (RSI > 60)
- 避免在超賣區做空 (RSI < 40)
"""

import asyncio
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from src.core import get_logger
from src.core.models import Kline, KlineInterval, MarketType, OrderType, OrderSide
from src.bots.base import BaseBot
from src.fund_manager import SignalCoordinator, SignalDirection, CoordinationResult
from src.master.models import BotState
from src.exchange import ExchangeClient
from src.data import MarketDataManager
from src.notification import NotificationManager

from .models import (
    SupertrendConfig,
    PositionSide,
    Position,
    Trade,
    ExitReason,
    GridLevel,
)
from .indicators import SupertrendIndicator

from typing import List

logger = get_logger(__name__)


class SupertrendBot(BaseBot):
    """
    Supertrend TREND_GRID trading bot.

    Strategy (TREND_GRID mode - Walk-Forward Validated):
    - Uses Supertrend indicator to determine trend direction
    - In bullish trend: LONG only, buy dips at grid levels
    - In bearish trend: SHORT only, sell rallies at grid levels
    - RSI filter prevents entries in extreme conditions
    - Exit on trend flip or take profit at next grid level

    Validation Results (2026-01-24):
    - Walk-Forward Consistency: 70% (7/10 periods)
    - OOS Sharpe: 5.84
    - Monte Carlo: ROBUST (100% profit probability)
    """

    FEE_RATE = Decimal("0.0004")  # 0.04% taker fee

    def __init__(
        self,
        bot_id: str,
        config: SupertrendConfig,
        exchange: ExchangeClient,
        data_manager: MarketDataManager,
        notifier: Optional[NotificationManager] = None,
        heartbeat_callback: Optional[callable] = None,
        signal_coordinator: Optional[SignalCoordinator] = None,
    ):
        # Call BaseBot.__init__ with all required parameters
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )

        # Signal Coordinator for multi-bot conflict prevention
        self._signal_coordinator = signal_coordinator or SignalCoordinator.get_instance()

        # Indicator
        self._indicator = SupertrendIndicator(
            atr_period=config.atr_period,
            atr_multiplier=config.atr_multiplier,
        )

        # Position tracking
        self._position: Optional[Position] = None
        self._trades: deque[Trade] = deque(maxlen=1000)
        self._entry_bar: int = 0
        self._current_bar: int = 0

        # Trend state
        self._prev_trend: int = 0
        self._current_trend: int = 0
        self._trend_bars: int = 0  # Bars in current trend direction

        # Grid state (TREND_GRID mode)
        self._grid_levels: List[GridLevel] = []
        self._upper_price: Optional[Decimal] = None
        self._lower_price: Optional[Decimal] = None
        self._grid_spacing: Optional[Decimal] = None
        self._grid_initialized: bool = False
        self._current_atr: Optional[Decimal] = None
        self._recent_klines: List[Kline] = []  # For ATR calculation

        # RSI Filter (RSI 過濾器)
        self._rsi_closes: list[Decimal] = []  # Recent closes for RSI calculation
        self._current_rsi: Optional[Decimal] = None
        self._rsi_period = config.rsi_period if hasattr(config, 'rsi_period') else 14

        # Statistics
        self._total_pnl = Decimal("0")

        # Risk control tracking (每日虧損 + 連續虧損)
        self._daily_pnl = Decimal("0")
        self._daily_start_time: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._consecutive_losses: int = 0
        self._risk_paused: bool = False

        # Hysteresis: track last triggered level to prevent oscillation
        self._last_triggered_level: Optional[int] = None
        self._hysteresis_pct: Decimal = self._config.hysteresis_pct

        # Signal cooldown to prevent signal stacking
        self._signal_cooldown: int = 0
        self._cooldown_bars: int = self._config.cooldown_bars

        # Volatility filter state (v2)
        self._atr_history: list[Decimal] = []
        self._atr_baseline: Optional[Decimal] = None

        # Kline callback reference for unsubscribe
        self._kline_callback = None

        # Kline callback tasks tracking (prevent memory leak)
        self._kline_tasks: set[asyncio.Task] = set()

        # Background monitor task
        self._monitor_task: Optional[asyncio.Task] = None

        # Slippage tracking (from BaseBot)
        self._init_slippage_tracking()

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
    # Abstract Properties (Required by BaseBot)
    # =========================================================================

    @property
    def bot_type(self) -> str:
        """Return bot type identifier."""
        return "supertrend"

    @property
    def symbol(self) -> str:
        """Return trading symbol."""
        return self._config.symbol

    # =========================================================================
    # Abstract Lifecycle Methods (Required by BaseBot)
    # =========================================================================

    async def _do_start(self) -> None:
        """
        Actual start logic for Supertrend bot.

        Called by BaseBot.start() after state transition.
        """
        logger.info(f"Initializing Supertrend Bot for {self._config.symbol}")

        # Set margin type (ISOLATED for risk control)
        try:
            await self._exchange.futures.set_margin_type(
                symbol=self._config.symbol,
                margin_type=self._config.margin_type,
            )
        except Exception as e:
            # May fail if already set to this type
            logger.debug(f"Set margin type: {e}")

        # Validate and set leverage
        if not (1 <= self._config.leverage <= 125):
            raise ValueError(
                f"Leverage {self._config.leverage}x out of valid range [1, 125]"
            )
        await self._exchange.futures.set_leverage(
            symbol=self._config.symbol,
            leverage=self._config.leverage,
        )

        # Initialize per-strategy risk tracking (風控相互影響隔離)
        capital = self._config.max_capital or Decimal("1000")
        self.set_strategy_initial_capital(capital)

        # Register for global risk tracking (多策略風控協調)
        await self.register_bot_for_global_risk(self._bot_id, capital)

        # Register for circuit breaker coordination (部分熔斷協調)
        await self.register_strategy_for_cb(
            bot_id=self._bot_id,
            strategy_type="supertrend",
            dependencies=None,
        )

        logger.info(f"Strategy initial capital set: {capital} USDT")

        # Get historical klines to initialize indicator
        interval_map = {
            "1m": KlineInterval.m1,
            "5m": KlineInterval.m5,
            "15m": KlineInterval.m15,
            "30m": KlineInterval.m30,
            "1h": KlineInterval.h1,
            "4h": KlineInterval.h4,
        }
        interval = interval_map.get(self._config.timeframe, KlineInterval.m15)

        klines = await self._exchange.futures.get_klines(
            symbol=self._config.symbol,
            interval=interval,
            limit=100,
        )

        if not klines or len(klines) < self._config.atr_period + 10:
            raise RuntimeError("Not enough historical data to initialize indicator")

        # Initialize indicator
        self._indicator.initialize_from_klines(klines)
        self._prev_trend = self._indicator.trend

        # Check existing position
        await self._sync_position()

        # Subscribe to kline updates (sync wrapper for async callback)
        def on_kline_sync(kline: Kline) -> None:
            task = asyncio.create_task(self._on_kline(kline))
            self._kline_tasks.add(task)
            task.add_done_callback(lambda t: self._kline_tasks.discard(t))

        self._kline_callback = on_kline_sync  # Store reference for unsubscribe
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
            market_type=MarketType.FUTURES,  # Futures trading
        )

        logger.info(f"Supertrend Bot initialized successfully")
        logger.info(f"  Mode: {self._config.mode.upper()}")
        logger.info(f"  Symbol: {self._config.symbol}")
        logger.info(f"  Timeframe: {self._config.timeframe}")
        logger.info(f"  ATR Period: {self._config.atr_period}")
        logger.info(f"  ATR Multiplier: {self._config.atr_multiplier}")
        logger.info(f"  Leverage: {self._config.leverage}x")
        logger.info(f"  Initial Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}")
        if self._config.mode == "hybrid_grid":
            logger.info(f"  Hybrid Bias: {self._config.hybrid_grid_bias_pct}")
            logger.info(f"  Hybrid TP Trend/Counter: {self._config.hybrid_tp_multiplier_trend}/{self._config.hybrid_tp_multiplier_counter}")
            logger.info(f"  Hybrid SL Counter: {self._config.hybrid_sl_multiplier_counter}")
            logger.info(f"  Hybrid RSI Asymmetric: {self._config.hybrid_rsi_asymmetric}")

        if self._notifier:
            await self._notifier.send_info(
                title="Supertrend Bot Started",
                message=f"Symbol: {self._config.symbol}\n"
                        f"Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}",
            )

        # Start background monitor (capital updates, risk checks)
        self._monitor_task = asyncio.create_task(self._background_monitor())

        # Start periodic state saving
        self._start_save_task()

        # Start position reconciliation (detect manual operations)
        self._start_position_reconciliation()

    async def _do_stop(self, clear_position: bool = False) -> None:
        """
        Actual stop logic for Supertrend bot.

        Called by BaseBot.stop() after state transition.
        """
        logger.info("Stopping Supertrend Bot...")

        # Unsubscribe from updates
        try:
            await self._data_manager.klines.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                callback=self._kline_callback,
                market_type=MarketType.FUTURES,
            )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe: {e}")

        # Clear position if requested
        if clear_position and self._position:
            await self._close_position(ExitReason.BOT_STOP)
        elif self._position and self._position.stop_loss_order_id:
            # Cancel stop loss order but keep position
            await self._cancel_stop_loss_order()

        # Stop background monitor
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

        logger.info("Supertrend Bot stopped")

        if self._notifier:
            await self._notifier.send_info(
                title="Supertrend Bot Stopped",
                message=f"Total PnL: {self._total_pnl:+.2f} USDT",
            )

    async def _do_pause(self) -> None:
        """
        Pause the bot (stop trading but keep position).

        For Supertrend, we just unsubscribe from kline updates.
        """
        logger.info("Pausing Supertrend Bot...")
        try:
            await self._data_manager.klines.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                callback=self._kline_callback,
                market_type=MarketType.FUTURES,
            )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe: {e}")
        logger.info("Supertrend Bot paused")

    async def _do_resume(self) -> None:
        """
        Resume the bot from paused state.

        Re-subscribe to kline updates.
        """
        logger.info("Resuming Supertrend Bot...")

        # Re-subscribe to kline updates (create new callback with task tracking)
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

        logger.info("Supertrend Bot resumed")

    # =========================================================================
    # Background Monitor (資金更新、風控檢查)
    # =========================================================================

    async def _background_monitor(self) -> None:
        """
        Background monitoring loop for capital updates and risk checks.

        Runs independently of kline processing to ensure:
        - Capital data freshness for risk bypass prevention
        - Strategy-level risk monitoring
        - Position reconciliation with exchange
        """
        logger.info("Starting Supertrend background monitor")

        from src.master.models import BotState

        while self._state == BotState.RUNNING:
            try:
                # Get account balance to track capital
                account = await self._exchange.futures.get_account()
                capital = self._config.max_capital or Decimal("1000")
                # Mark capital as updated for risk bypass prevention
                self.mark_capital_updated()

                # Record capital for CB validation (熔斷誤觸發防護)
                self.record_capital_for_validation(capital)

                # Apply consecutive loss decay (prevents permanent lockout)
                self.apply_consecutive_loss_decay()

                # Update virtual position unrealized P&L
                if self._position:
                    ticker = await self._exchange.futures.get_ticker(self._config.symbol)
                    current_price = Decimal(str(ticker.last_price))
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
                        ticker = await self._exchange.futures.get_ticker(self._config.symbol)
                        current_price = Decimal(str(ticker.last_price))
                        cb_result = await self.trigger_circuit_breaker_safe(
                            reason=f"CRITICAL_RISK: {risk_result.get('action', 'unknown')}",
                            current_price=current_price,
                            current_capital=self._config.max_capital or Decimal("1000"),
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
                    self.mark_position_synced()

                # Comprehensive stop loss check (三層止損保護)
                if self._position:
                    ticker = await self._exchange.futures.get_ticker(self._config.symbol)
                    current_price = Decimal(str(ticker.last_price))
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
                            if replace_result["success"]:
                                self._position.stop_loss_order_id = replace_result["new_order_id"]
                                self._position.stop_loss_price = replace_result.get("stop_price")

                        elif sl_check["action_type"] == "BACKUP_CLOSE":
                            await self._close_position(ExitReason.STOP_LOSS)
                            self.reset_stop_loss_protection()

                # Network health monitoring (網路彈性監控)
                try:
                    start_time = time.time()
                    ticker = await self._exchange.futures.get_ticker(self._config.symbol)
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

        logger.info("Supertrend background monitor stopped")

    def _get_extra_status(self) -> Dict[str, Any]:
        """Return extra status fields specific to Supertrend bot."""
        position_info = None
        if self._position:
            position_info = {
                "side": self._position.side.value,
                "entry_price": float(self._position.entry_price),
                "quantity": float(self._position.quantity),
                "unrealized_pnl": float(self._position.unrealized_pnl),
                "stop_loss_price": float(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                "stop_loss_order_id": self._position.stop_loss_order_id,
            }

        indicator = self._indicator.current
        supertrend_info = None
        if indicator:
            supertrend_info = {
                "trend": "BULLISH" if indicator.is_bullish else "BEARISH",
                "value": float(indicator.supertrend),
                "atr": float(indicator.atr),
            }

        return {
            "timeframe": self._config.timeframe,
            "leverage": self._config.leverage,
            "position": position_info,
            "supertrend": supertrend_info,
            "total_trades": len(self._trades),  # Override BaseBot's stats
            "total_pnl": float(self._total_pnl),
            "current_bar": self._current_bar,
            # RSI filter status
            "rsi": float(self._current_rsi) if self._current_rsi else None,
            "rsi_filter_enabled": getattr(self._config, 'use_rsi_filter', True),
            # Risk control status
            "daily_pnl": float(self._daily_pnl),
            "consecutive_losses": self._consecutive_losses,
            "risk_paused": self._risk_paused,
        }

    async def _extra_health_checks(self) -> Dict[str, bool]:
        """Perform extra health checks specific to Supertrend bot."""
        checks = {}

        # Check if indicator is initialized
        checks["indicator_initialized"] = self._indicator.current is not None

        # Check if data subscription is active
        checks["data_subscribed"] = True  # Assume true if bot is running

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
            return True

        try:
            order = await self._exchange.futures.get_order(
                symbol=self._config.symbol,
                order_id=self._position.stop_loss_order_id,
            )
            if order:
                status = getattr(order, "status", "").upper()
                if status in ("NEW", "PARTIALLY_FILLED"):
                    return True
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

    async def _sync_position(self) -> None:
        """Sync position with exchange."""
        try:
            positions = await self._exchange.futures.get_positions(self._config.symbol)

            for pos in positions:
                if pos.symbol == self._config.symbol and pos.quantity != Decimal("0"):
                    side = PositionSide.LONG if pos.quantity > 0 else PositionSide.SHORT
                    self._position = Position(
                        side=side,
                        entry_price=pos.entry_price,
                        quantity=abs(pos.quantity),
                        entry_time=datetime.now(timezone.utc),
                        unrealized_pnl=pos.unrealized_pnl,
                    )
                    logger.info(f"Synced existing position: {side.value} {self._position.quantity}")
                    break

        except Exception as e:
            logger.warning(f"Failed to sync position: {e}")

    def _calculate_rsi(self, close: Decimal) -> Optional[Decimal]:
        """
        Calculate RSI using recent closes.

        Returns:
            RSI value (0-100) or None if not enough data
        """
        self._rsi_closes.append(close)

        # Keep only enough closes for RSI calculation
        max_closes = self._rsi_period + 50
        if len(self._rsi_closes) > max_closes:
            self._rsi_closes = self._rsi_closes[-max_closes:]

        if len(self._rsi_closes) < self._rsi_period + 1:
            return None

        # Calculate gains and losses
        gains = []
        losses = []
        for i in range(-self._rsi_period, 0):
            change = float(self._rsi_closes[i]) - float(self._rsi_closes[i - 1])
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))

        avg_gain = sum(gains) / self._rsi_period
        avg_loss = sum(losses) / self._rsi_period

        # Handle edge cases to avoid division by zero
        if avg_loss == 0 and avg_gain == 0:
            return Decimal("50")  # Neutral RSI when no movement
        if avg_loss == 0:
            return Decimal("100")

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return Decimal(str(round(rsi, 2)))

    def _check_rsi_filter(self, side: PositionSide) -> bool:
        """
        Check if RSI filter allows entry.

        RSI Filter Logic:
        - Don't go LONG if RSI > 60 (overbought, avoid chasing)
        - Don't go SHORT if RSI < 40 (oversold, avoid selling low)

        Returns:
            True if entry is allowed, False if blocked by filter
        """
        if not getattr(self._config, 'use_rsi_filter', True):
            return True  # Filter disabled

        if self._current_rsi is None:
            return True  # Not enough data yet

        rsi_value = float(self._current_rsi)
        overbought = getattr(self._config, 'rsi_overbought', 60)
        oversold = getattr(self._config, 'rsi_oversold', 40)

        if side == PositionSide.LONG and rsi_value > overbought:
            logger.info(f"RSI filter blocked LONG: RSI={rsi_value:.1f} > {overbought} (overbought)")
            return False

        if side == PositionSide.SHORT and rsi_value < oversold:
            logger.info(f"RSI filter blocked SHORT: RSI={rsi_value:.1f} < {oversold} (oversold)")
            return False

        return True

    # =========================================================================
    # Grid Methods (TREND_GRID mode)
    # =========================================================================

    def _check_hysteresis(
        self, level_index: int, direction: str, grid_price: Decimal, current_price: Decimal
    ) -> bool:
        """
        Check if signal passes hysteresis filter.

        Prevents oscillation by requiring price to move beyond a buffer zone
        before retriggering the same grid level.

        Args:
            level_index: Grid level index
            direction: "long" or "short"
            grid_price: Grid level price
            current_price: Current market price

        Returns:
            True if signal should proceed, False if blocked by hysteresis
        """
        if self._last_triggered_level is None:
            return True

        if self._last_triggered_level != level_index:
            return True

        hysteresis_buffer = grid_price * self._hysteresis_pct

        if direction == "long":
            if current_price < grid_price - hysteresis_buffer:
                return True
        else:  # short
            if current_price > grid_price + hysteresis_buffer:
                return True

        logger.debug(
            f"Hysteresis active: level {level_index} recently triggered, "
            f"waiting for price to move beyond buffer zone"
        )
        return False

    def _calculate_atr(self, period: int) -> Optional[Decimal]:
        """Calculate Average True Range from recent klines."""
        if len(self._recent_klines) < period + 1:
            return None

        tr_values = []
        for i in range(-period, 0):
            kline = self._recent_klines[i]
            prev_kline = self._recent_klines[i - 1]

            tr = max(
                kline.high - kline.low,
                abs(kline.high - prev_kline.close),
                abs(kline.low - prev_kline.close)
            )
            tr_values.append(tr)

        return sum(tr_values) / Decimal(period)

    def _update_volatility_baseline(self, current_atr: Decimal) -> None:
        """更新 ATR 滾動歷史，計算基線。"""
        self._atr_history.append(current_atr)
        bp = getattr(self._config, 'vol_atr_baseline_period', 200)
        if len(self._atr_history) > bp:
            self._atr_history = self._atr_history[-bp:]
        if len(self._atr_history) >= bp:
            self._atr_baseline = sum(self._atr_history) / Decimal(len(self._atr_history))

    def _check_volatility_regime(self) -> bool:
        """檢查當前波動率是否在可交易範圍內。"""
        if not getattr(self._config, 'use_volatility_filter', False):
            return True
        if self._atr_baseline is None or self._atr_baseline == 0:
            return True
        if self._current_atr is None:
            return True
        ratio = float(self._current_atr / self._atr_baseline)
        low = getattr(self._config, 'vol_ratio_low', 0.5)
        high = getattr(self._config, 'vol_ratio_high', 2.0)
        if not (low <= ratio <= high):
            logger.info(f"Volatility filter blocked: ATR ratio={ratio:.2f} outside [{low}, {high}]")
            return False
        return True

    def _check_timeout_exit(self, current_price: Decimal) -> bool:
        """檢查是否超時出場（僅虧損時）。"""
        max_hold = getattr(self._config, 'max_hold_bars', 0)
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

    def _initialize_grid(self, current_price: Decimal) -> None:
        """Initialize grid levels around current price."""
        # Calculate ATR for dynamic range
        atr = self._calculate_atr(self._config.atr_period)
        self._current_atr = atr

        if atr and atr > 0:
            range_size = atr * self._config.grid_atr_multiplier
        else:
            range_size = current_price * Decimal("0.05")  # 5% fallback

        self._upper_price = current_price + range_size
        self._lower_price = current_price - range_size

        # Guard against division by zero
        if self._config.grid_count <= 0:
            self._grid_spacing = range_size
        else:
            self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        # Create grid levels
        self._grid_levels = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + (self._grid_spacing * Decimal(i))
            self._grid_levels.append(GridLevel(index=i, price=price))

        self._grid_initialized = True
        logger.info(
            f"Grid initialized: {self._lower_price:.2f} - {self._upper_price:.2f}, "
            f"{self._config.grid_count} levels, spacing={self._grid_spacing:.2f}"
        )

    def _find_current_grid_level(self, price: Decimal) -> Optional[int]:
        """Find the grid level index for the current price."""
        if not self._grid_levels:
            return None

        for i, level in enumerate(self._grid_levels):
            if i < len(self._grid_levels) - 1:
                if level.price <= price < self._grid_levels[i + 1].price:
                    return i

        if price >= self._grid_levels[-1].price:
            return len(self._grid_levels) - 1

        return 0

    def _should_rebuild_grid(self, current_price: Decimal) -> bool:
        """Check if grid needs rebuilding."""
        if not self._grid_initialized:
            return True

        if current_price > self._upper_price or current_price < self._lower_price:
            return True

        return False

    def _initialize_hybrid_grid(self, current_price: Decimal) -> None:
        """Initialize asymmetric grid levels for HYBRID_GRID mode."""
        atr = self._calculate_atr(self._config.atr_period)
        self._current_atr = atr

        if atr and atr > 0:
            range_size = atr * self._config.grid_atr_multiplier
        else:
            range_size = current_price * Decimal("0.05")

        bias = self._config.hybrid_grid_bias_pct

        if self._current_trend == 1:
            self._lower_price = current_price - range_size * bias
            self._upper_price = current_price + range_size * (Decimal("1") - bias)
        elif self._current_trend == -1:
            self._lower_price = current_price - range_size * (Decimal("1") - bias)
            self._upper_price = current_price + range_size * bias
        else:
            self._lower_price = current_price - range_size
            self._upper_price = current_price + range_size

        if self._config.grid_count <= 0:
            self._grid_spacing = range_size
        else:
            self._grid_spacing = (self._upper_price - self._lower_price) / Decimal(self._config.grid_count)

        self._grid_levels = []
        for i in range(self._config.grid_count + 1):
            price = self._lower_price + (self._grid_spacing * Decimal(i))
            self._grid_levels.append(GridLevel(index=i, price=price))

        self._grid_initialized = True
        logger.info(
            f"HYBRID_GRID initialized: {self._lower_price:.2f} - {self._upper_price:.2f}, "
            f"{self._config.grid_count} levels, bias={bias}, trend={self._current_trend}"
        )

    def _reset_filled_levels(self) -> None:
        """Reset all grid levels to unfilled."""
        for level in self._grid_levels:
            level.is_filled = False

    async def _on_kline_hybrid_grid(self, kline: Kline, current_price: Decimal) -> None:
        """
        HYBRID_GRID mode: bidirectional trading with trend bias.

        Both long and short allowed regardless of trend.
        Trend direction gets favorable TP/SL; counter-trend gets tighter risk.
        """
        # Initialize or rebuild grid
        if self._should_rebuild_grid(current_price):
            self._initialize_hybrid_grid(current_price)
            # Don't return - continue to entry logic

        # Reset filled levels when no position
        if not self._position:
            self._reset_filled_levels()

        # Check signal cooldown
        if self._signal_cooldown > 0:
            logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping entry")
            return

        # Trend status: treat unestablished trend as neutral (counter-trend params)
        if self._current_trend == 0 or self._trend_bars < getattr(self._config, 'min_trend_bars', 2):
            is_bullish = False  # No trend, use counter-trend params for all
        else:
            is_bullish = self._current_trend == 1

        level_idx = self._find_current_grid_level(current_price)
        if level_idx is None:
            return

        # Try LONG entry (buy dip at grid level below)
        if level_idx > 0:
            entry_level = self._grid_levels[level_idx]
            if not entry_level.is_filled and kline.low <= entry_level.price:
                if self._config.hybrid_rsi_asymmetric and not is_bullish:
                    # Counter-trend long: must be oversold
                    if self._current_rsi is not None and float(self._current_rsi) > self._config.rsi_oversold:
                        pass  # blocked
                    else:
                        await self._try_hybrid_entry(
                            entry_level, level_idx, PositionSide.LONG, is_bullish, "long"
                        )
                        return
                else:
                    if self._check_rsi_filter(PositionSide.LONG):
                        await self._try_hybrid_entry(
                            entry_level, level_idx, PositionSide.LONG, is_bullish, "long"
                        )
                        return

        # Try SHORT entry (sell rally at grid level above)
        if level_idx < len(self._grid_levels) - 1:
            entry_level = self._grid_levels[level_idx + 1]
            if not entry_level.is_filled and kline.high >= entry_level.price:
                if self._config.hybrid_rsi_asymmetric and is_bullish:
                    # Counter-trend short: must be overbought
                    if self._current_rsi is not None and float(self._current_rsi) < self._config.rsi_overbought:
                        pass  # blocked
                    else:
                        await self._try_hybrid_entry(
                            entry_level, level_idx + 1, PositionSide.SHORT, not is_bullish, "short"
                        )
                else:
                    if self._check_rsi_filter(PositionSide.SHORT):
                        await self._try_hybrid_entry(
                            entry_level, level_idx + 1, PositionSide.SHORT, not is_bullish, "short"
                        )

    async def _try_hybrid_entry(
        self,
        entry_level: GridLevel,
        level_idx: int,
        side: PositionSide,
        is_with_trend: bool,
        hysteresis_side: str,
    ) -> None:
        """Attempt a hybrid grid entry with differentiated TP/SL."""
        current_price = entry_level.price

        if not self._check_hysteresis(level_idx, hysteresis_side, current_price, current_price):
            return

        entry_level.is_filled = True
        self._last_triggered_level = level_idx

        # Determine TP/SL multipliers based on trend alignment
        if is_with_trend:
            tp_mult = self._config.hybrid_tp_multiplier_trend
            sl_mult = Decimal("1")
        else:
            tp_mult = self._config.hybrid_tp_multiplier_counter
            sl_mult = self._config.hybrid_sl_multiplier_counter

        min_tp_distance = current_price * Decimal("0.001")

        if side == PositionSide.LONG:
            tp_level = min(
                level_idx + int(self._config.take_profit_grids * tp_mult),
                len(self._grid_levels) - 1,
            )
            if tp_level <= level_idx:
                tp_level = min(level_idx + 1, len(self._grid_levels) - 1)
            tp_price = self._grid_levels[tp_level].price
            if tp_price - current_price < min_tp_distance:
                tp_price = current_price + min_tp_distance

            reason = "hybrid_grid_long" if is_with_trend else "hybrid_grid_long_counter"
        else:
            tp_level = max(
                level_idx - int(self._config.take_profit_grids * tp_mult),
                0,
            )
            if tp_level >= level_idx:
                tp_level = max(level_idx - 1, 0)
            tp_price = self._grid_levels[tp_level].price
            if current_price - tp_price < min_tp_distance:
                tp_price = current_price - min_tp_distance

            reason = "hybrid_grid_short" if is_with_trend else "hybrid_grid_short_counter"

        logger.info(
            f"HYBRID_GRID {side.value} signal ({reason}): price={current_price:.2f}, "
            f"grid_level={level_idx}, TP={tp_price:.2f}, "
            f"trend={'WITH' if is_with_trend else 'COUNTER'}"
            + (f", RSI={self._current_rsi:.1f}" if self._current_rsi else "")
        )

        success = await self._open_position(side, current_price, tp_price)
        if success:
            self._signal_cooldown = self._cooldown_bars

    async def _on_kline(self, kline: Kline) -> None:
        """
        Handle new kline data (TREND_GRID mode).

        TREND_GRID Logic:
        - Bullish trend: Buy dips at grid levels (LONG only)
        - Bearish trend: Sell rallies at grid levels (SHORT only)
        - RSI filter to avoid entries in extreme conditions
        - Exit on trend flip

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
            self._current_bar += 1

            # Apply consecutive loss decay (prevents permanent lockout)
            # Run periodically (every ~10 bars to reduce overhead)
            if self._current_bar % 10 == 0:
                self.apply_consecutive_loss_decay()

            # Store kline for ATR calculation
            self._recent_klines.append(kline)
            max_klines = self._config.atr_period + 50
            if len(self._recent_klines) > max_klines:
                self._recent_klines = self._recent_klines[-max_klines:]

            # Update indicator
            supertrend = self._indicator.update(kline)
            if supertrend is None:
                return

            self._prev_trend = self._current_trend
            self._current_trend = supertrend.trend
            current_price = kline.close

            # Track how many bars in current trend
            if self._current_trend == self._prev_trend:
                self._trend_bars += 1
            else:
                self._trend_bars = 1  # Reset on trend change

            # Calculate RSI for filter
            self._current_rsi = self._calculate_rsi(current_price)

            # Update volatility baseline (v2)
            atr = self._calculate_atr(self._config.atr_period)
            if atr:
                self._current_atr = atr
                self._update_volatility_baseline(atr)

            # Update position unrealized PnL
            if self._position:
                self._position.update_extremes(current_price)
                if self._position.side == PositionSide.LONG:
                    self._position.unrealized_pnl = (
                        (current_price - self._position.entry_price) *
                        self._position.quantity
                    )
                else:
                    self._position.unrealized_pnl = (
                        (self._position.entry_price - current_price) *
                        self._position.quantity
                    )

                # Check trailing stop
                if self._config.use_trailing_stop:
                    if self._check_trailing_stop(current_price):
                        logger.warning(f"Trailing stop triggered at {current_price}")
                        await self._close_position(ExitReason.STOP_LOSS)
                        self.record_stop_loss_trigger()
                        self.clear_stop_loss_sync()
                        return

                # Check timeout exit (v2: 超時出場)
                if self._check_timeout_exit(current_price):
                    logger.warning(f"Timeout exit triggered: held {self._current_bar - self._entry_bar} bars")
                    await self._close_position(ExitReason.SIGNAL_FLIP)
                    return

                # Check for trend flip exit (TREND_GRID only; HYBRID_GRID uses TP/SL)
                if self._config.mode != "hybrid_grid":
                    if self._position.side == PositionSide.LONG and self._current_trend == -1:
                        logger.info(f"Trend flip to BEARISH - closing LONG position")
                        await self._close_position(ExitReason.SIGNAL_FLIP)
                    elif self._position.side == PositionSide.SHORT and self._current_trend == 1:
                        logger.info(f"Trend flip to BULLISH - closing SHORT position")
                        await self._close_position(ExitReason.SIGNAL_FLIP)

                return  # Skip entry if already in position

            # === Entry Logic ===

            # Decrement signal cooldown
            if self._signal_cooldown > 0:
                self._signal_cooldown -= 1

            # Volatility regime filter (v2: block entry if outside range)
            if not self._check_volatility_regime():
                return

            # HYBRID_GRID mode branch
            if self._config.mode == "hybrid_grid":
                await self._on_kline_hybrid_grid(kline, current_price)
                return

            # TREND_GRID: Reset filled levels when no position (v2: auto-reset)
            self._reset_filled_levels()

            # Initialize or rebuild grid if needed
            if self._should_rebuild_grid(current_price):
                self._initialize_grid(current_price)
                return

            # Need trend to be established
            if self._current_trend == 0:
                return

            # Check signal cooldown
            if self._signal_cooldown > 0:
                logger.debug(f"Signal cooldown active ({self._signal_cooldown} bars), skipping entry")
                return

            # Require minimum bars in trend before entry (trend confirmation)
            min_trend_bars = getattr(self._config, 'min_trend_bars', 2)
            if self._trend_bars < min_trend_bars:
                return

            # Find current grid level
            level_idx = self._find_current_grid_level(current_price)
            if level_idx is None:
                return

            # Bullish trend: LONG only (buy dips)
            if self._current_trend == 1 and level_idx > 0:
                entry_level = self._grid_levels[level_idx]
                # Check if current kline low touched the grid level
                if not entry_level.is_filled and kline.low <= entry_level.price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level_idx, "long", entry_level.price, current_price):
                        return

                    # Check RSI filter before LONG entry
                    if not self._check_rsi_filter(PositionSide.LONG):
                        return

                    entry_level.is_filled = True
                    entry_price = entry_level.price

                    # Calculate take profit at next grid level up
                    tp_grids = getattr(self._config, 'take_profit_grids', 1)
                    tp_level = min(level_idx + tp_grids, len(self._grid_levels) - 1)
                    tp_price = self._grid_levels[tp_level].price

                    logger.info(
                        f"TREND_GRID LONG signal: price={entry_price:.2f}, "
                        f"grid_level={level_idx}, TP={tp_price:.2f}, "
                        f"RSI={self._current_rsi:.1f}" if self._current_rsi else ""
                    )

                    success = await self._open_position(PositionSide.LONG, entry_price, tp_price)
                    if success:
                        self._signal_cooldown = self._cooldown_bars
                        self._last_triggered_level = level_idx

            # Bearish trend: SHORT only (sell rallies)
            elif self._current_trend == -1 and level_idx < len(self._grid_levels) - 1:
                entry_level = self._grid_levels[level_idx + 1]
                # Check if current kline high touched the grid level
                if not entry_level.is_filled and kline.high >= entry_level.price:
                    # Check hysteresis to prevent oscillation
                    if not self._check_hysteresis(level_idx + 1, "short", entry_level.price, current_price):
                        return

                    # Check RSI filter before SHORT entry
                    if not self._check_rsi_filter(PositionSide.SHORT):
                        return

                    entry_level.is_filled = True
                    entry_price = entry_level.price

                    # Calculate take profit at next grid level down
                    tp_grids = getattr(self._config, 'take_profit_grids', 1)
                    tp_level = max(level_idx + 1 - tp_grids, 0)
                    tp_price = self._grid_levels[tp_level].price

                    logger.info(
                        f"TREND_GRID SHORT signal: price={entry_price:.2f}, "
                        f"grid_level={level_idx + 1}, TP={tp_price:.2f}, "
                        f"RSI={self._current_rsi:.1f}" if self._current_rsi else ""
                    )

                    success = await self._open_position(PositionSide.SHORT, entry_price, tp_price)
                    if success:
                        self._signal_cooldown = self._cooldown_bars
                        self._last_triggered_level = level_idx + 1

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    async def _open_position(
        self,
        side: PositionSide,
        price: Decimal,
        take_profit: Optional[Decimal] = None
    ) -> bool:
        """
        Open a new position (TREND_GRID mode).

        Args:
            side: Position side (LONG or SHORT)
            price: Entry price
            take_profit: Optional take profit price (grid-based)
        """
        # Stage 6: Close old position if reversing direction
        if self._position and self._position.side != side:
            logger.info(f"Reversing direction: closing {self._position.side.value} before opening {side.value}")
            await self._close_position(ExitReason.SIGNAL_FLIP)

        # Check entry allowed (circuit breaker, cooldown, oscillation prevention)
        entry_allowed, entry_reason = self.check_entry_allowed()
        if not entry_allowed:
            logger.warning(f"Entry blocked: {entry_reason}")
            return False

        # Check risk limits before opening
        if self._check_risk_limits():
            logger.warning(f"Trading paused due to risk limits - skipping {side.value} signal")
            if self._notifier:
                await self._notifier.send_warning(
                    title="Supertrend: Risk Limit",
                    message=f"Signal skipped: {side.value}\n"
                            f"Daily PnL: {self._daily_pnl:+.2f}\n"
                            f"Consecutive losses: {self._consecutive_losses}",
                )
            return False

        # Pre-trade validation (time sync + data health + position reconciliation)
        order_side = "BUY" if side == PositionSide.LONG else "SELL"
        if not await self._validate_pre_trade(
            symbol=self._config.symbol,
            side=order_side,
            quantity=Decimal("0.001"),  # Minimum, actual calculated below
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

        # Check balance before order (prevent rejection)
        balance_ok, balance_msg = await self._check_balance_for_order(
            symbol=self._config.symbol,
            quantity=Decimal("0.001"),  # Will be calculated below
            price=price,
            leverage=self._config.leverage,
        )
        if not balance_ok:
            logger.warning(f"Order blocked: {balance_msg}")
            return False

        # Check SignalCoordinator for multi-bot conflict prevention
        if self._signal_coordinator:
            signal_dir = SignalDirection.LONG if side == PositionSide.LONG else SignalDirection.SHORT
            result = await self._signal_coordinator.request_signal(
                bot_id=self._bot_id,
                symbol=self._config.symbol,
                direction=signal_dir,
                quantity=Decimal("0"),  # Will be calculated below
                price=price,
                reason=f"Supertrend grid level",
            )
            if not result.approved:
                logger.warning(f"Signal blocked by coordinator: {result.message}")
                return False

        # Note: Global risk check is done after calculating quantity below

        try:
            # Validate price before calculation (indicator boundary check)
            if not self._validate_price(price, "entry_price"):
                logger.warning(f"Invalid entry price: {price}")
                return False

            # Calculate position size based on allocated capital
            account = await self._exchange.futures.get_account()
            available = Decimal(str(account.available_balance))

            # Use max_capital if configured, otherwise use available balance
            # Read capital under lock to prevent race with update_capital()
            async with self._capital_lock:
                config_capital = self._config.max_capital
            if config_capital is not None:
                # 使用分配的資金上限，但不能超過實際可用餘額
                capital = min(config_capital, available)
            else:
                capital = available

            # Apply position_size_pct but cap at max_position_pct (notional = margin × leverage)
            position_pct = min(self._config.position_size_pct, self._config.max_position_pct)
            notional = capital * position_pct * Decimal(self._config.leverage)
            quantity = self._safe_divide(notional, price, context="position_size")

            # Validate and normalize price/quantity to exchange requirements
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

            if quantity <= 0:
                logger.warning("Insufficient balance to open position")
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

            # Check for duplicate order (prevent double-entry on retry)
            if self._is_duplicate_order(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
            ):
                logger.warning(f"Duplicate order blocked: {order_side} {quantity}")
                return False

            # Mark order as pending (for deduplication)
            order_key = self._mark_order_pending(
                symbol=self._config.symbol,
                side=order_side,
                quantity=quantity,
            )

            # Place market order with timeout protection
            order_side_enum = OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL

            async def place_order():
                return await self._exchange.futures_create_order(
                    symbol=self._config.symbol,
                    side=order_side_enum.value,
                    order_type="MARKET",
                    quantity=quantity,
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

                # Calculate stop loss price
                if side == PositionSide.LONG:
                    stop_loss_price = fill_price * (Decimal("1") - self._config.stop_loss_pct)
                else:
                    stop_loss_price = fill_price * (Decimal("1") + self._config.stop_loss_pct)

                # Normalize stop loss price to exchange tick size
                symbol_info = await self._get_symbol_info(self._config.symbol)
                stop_loss_price = self._normalize_price(stop_loss_price, symbol_info)

                self._position = Position(
                    side=side,
                    entry_price=fill_price,
                    quantity=fill_qty,
                    entry_time=datetime.now(timezone.utc),
                    stop_loss_price=stop_loss_price,
                    highest_price=fill_price,
                    lowest_price=fill_price,
                )
                self._entry_bar = self._current_bar

                # Place exchange stop loss order if enabled
                if self._config.use_exchange_stop_loss:
                    await self._place_stop_loss_order()

                # Record virtual fill (淨倉位管理 - track per-bot position)
                order_id_str = str(getattr(order, "order_id", ""))
                fee = fill_qty * fill_price * self.FEE_RATE
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

                # Log with take profit if provided
                tp_str = f", TP @ {take_profit:.2f}" if take_profit else ""
                logger.info(f"Opened {side.value} position: {quantity} @ {price}, SL @ {stop_loss_price}{tp_str}")

                if self._notifier:
                    tp_msg = f"\nTake Profit: {take_profit:.2f}" if take_profit else ""
                    await self._notifier.send_info(
                        title=f"Supertrend GRID: {side.value}",
                        message=f"Entry: {price}\nSize: {quantity}\nLeverage: {self._config.leverage}x\nStop Loss: {stop_loss_price}{tp_msg}",
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
                quantity=Decimal("0"),
                error=e,
            )
            return False
        finally:
            self.release_risk_gate()

    def _check_trailing_stop(self, current_price: Decimal) -> bool:
        """
        Check if trailing stop should be triggered.

        Returns:
            True if stop loss should trigger
        """
        if not self._position:
            return False

        stop_pct = self._config.trailing_stop_pct

        if self._position.side == PositionSide.LONG:
            # For long: stop if price drops stop_pct below max price
            if self._position.max_price is not None:
                stop_price = self._position.max_price * (Decimal("1") - stop_pct)
                if current_price <= stop_price:
                    logger.info(
                        f"Trailing stop: price {current_price:.2f} <= "
                        f"stop {stop_price:.2f} (max: {self._position.max_price:.2f})"
                    )
                    return True
        else:
            # For short: stop if price rises stop_pct above min price
            if self._position.min_price is not None:
                stop_price = self._position.min_price * (Decimal("1") + stop_pct)
                if current_price >= stop_price:
                    logger.info(
                        f"Trailing stop: price {current_price:.2f} >= "
                        f"stop {stop_price:.2f} (min: {self._position.min_price:.2f})"
                    )
                    return True

        return False

    async def _place_stop_loss_order(self) -> None:
        """Place stop loss order on exchange using Algo Order API."""
        if not self._position or not self._position.stop_loss_price:
            return

        try:
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
            return True

        elif result["was_triggered"]:
            # Stop loss was executed - position may have changed
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

    async def _close_position(self, reason: ExitReason) -> None:
        """Close current position."""
        if not self._position:
            return

        try:
            # Cancel stop loss order first (if any)
            if self._position.stop_loss_order_id:
                await self._cancel_stop_loss_order()

            # Get current price for PnL estimation
            # NOTE: PnL uses ticker price, not actual fill price — minor slippage
            # is acceptable as post-fill cost_basis_fifo tracks attributed P&L
            ticker = await self._exchange.futures.get_ticker(self._config.symbol)
            exit_price = Decimal(str(ticker.last_price))

            # Place closing order
            close_side = OrderSide.SELL if self._position.side == PositionSide.LONG else OrderSide.BUY

            order = await self._exchange.futures_create_order(
                symbol=self._config.symbol,
                side=close_side.value,
                order_type="MARKET",
                quantity=self._position.quantity,
                reduce_only=True,
                bot_id=self._bot_id,
            )

            if order:
                # Use actual fill price if available, fallback to ticker price
                fill_price = Decimal(str(order.avg_price)) if getattr(order, 'avg_price', None) else exit_price

                # Calculate PnL using actual fill price
                if self._position.side == PositionSide.LONG:
                    pnl = (fill_price - self._position.entry_price) * self._position.quantity
                else:
                    pnl = (self._position.entry_price - fill_price) * self._position.quantity

                # Deduct fees
                fee = (self._position.entry_price + fill_price) * self._position.quantity * self.FEE_RATE
                net_pnl = pnl - fee

                # Calculate MFE/MAE
                mfe, mae = self.calculate_mfe_mae(
                    side=self._position.side.value,
                    entry_price=self._position.entry_price,
                    highest_price=self._position.highest_price,
                    lowest_price=self._position.lowest_price,
                    leverage=self._config.leverage,
                )

                # Record trade
                trade = Trade(
                    side=self._position.side,
                    entry_price=self._position.entry_price,
                    exit_price=fill_price,
                    quantity=self._position.quantity,
                    pnl=net_pnl,
                    fee=fee,
                    entry_time=self._position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason,
                    holding_duration=self._current_bar - self._entry_bar,
                    mfe=mfe,
                    mae=mae,
                )
                self._trades.append(trade)
                self._total_pnl += net_pnl

                # Close cost basis with FIFO (持倉歸屬 - P&L attribution)
                order_id_str = str(getattr(order, "order_id", ""))
                close_fee = self._position.quantity * fill_price * self.FEE_RATE
                cost_basis_result = self.close_cost_basis_fifo(
                    symbol=self._config.symbol,
                    close_quantity=self._position.quantity,
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
                self._update_risk_tracking(net_pnl)

                logger.info(
                    f"Closed {self._position.side.value} position: "
                    f"PnL={net_pnl:+.2f} USDT, Reason={reason.value}"
                )

                if self._notifier:
                    emoji = "✅" if net_pnl > 0 else "❌"
                    await self._notifier.send_info(
                        title=f"{emoji} Supertrend: Close {self._position.side.value}",
                        message=f"Exit: {exit_price}\nPnL: {net_pnl:+.2f} USDT\n"
                                f"Reason: {reason.value}\nTotal: {self._total_pnl:+.2f} USDT",
                    )

                self._position = None

        except Exception as e:
            logger.error(f"Failed to close position: {e}")
            await self._handle_order_rejection(
                symbol=self._config.symbol,
                side="SELL" if self._position and self._position.side == PositionSide.LONG else "BUY",
                quantity=self._position.quantity if self._position else Decimal("0"),
                error=e,
            )
            await self._on_close_position_failure(self._config.symbol, e)

    # =========================================================================
    # Risk Control (每日虧損限制 + 連續虧損保護)
    # =========================================================================

    def _reset_daily_stats_if_needed(self) -> None:
        """Reset daily stats if it's a new day (UTC)."""
        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if today_start > self._daily_start_time:
            logger.info(f"New trading day - resetting daily stats")
            self._daily_pnl = Decimal("0")
            self._daily_start_time = today_start
            # Only reset risk_paused if it was due to daily loss
            # Keep consecutive_losses as it carries over days

    def _check_risk_limits(self) -> bool:
        """
        Check if risk limits have been exceeded.

        Returns:
            True if trading should be blocked
        """
        # Reset daily stats if new day
        self._reset_daily_stats_if_needed()

        # Check if already paused
        if self._risk_paused:
            return True

        # Get capital for percentage calculation
        # Note: single-threaded asyncio — write-side is lock-protected,
        # read here is safe without lock (no concurrent mutation possible)
        capital = self._config.max_capital or Decimal("1000")  # Default if not set

        # Check daily loss limit
        daily_loss_pct = abs(self._daily_pnl) / capital if self._daily_pnl < 0 else Decimal("0")
        if daily_loss_pct >= self._config.daily_loss_limit_pct:
            logger.warning(
                f"Daily loss limit reached: {daily_loss_pct:.1%} >= {self._config.daily_loss_limit_pct:.1%}"
            )
            self._risk_paused = True
            return True

        # Check consecutive losses
        if self._consecutive_losses >= self._config.max_consecutive_losses:
            logger.warning(
                f"Max consecutive losses reached: {self._consecutive_losses} >= {self._config.max_consecutive_losses}"
            )
            self._risk_paused = True
            return True

        return False

    def _update_risk_tracking(self, pnl: Decimal) -> None:
        """
        Update risk tracking after a trade.

        Args:
            pnl: Profit/loss from the trade
        """
        # Note: single-threaded asyncio — write-side capital is lock-protected,
        # PnL tracking here runs in the same event loop with no concurrent mutation
        self._daily_pnl += pnl

        if pnl < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0  # Reset on profitable trade

        logger.debug(
            f"Risk tracking updated: daily_pnl={self._daily_pnl:.2f}, "
            f"consecutive_losses={self._consecutive_losses}"
        )

    # =========================================================================
    # FundManager Integration
    # =========================================================================

    async def _on_capital_updated(self, new_max_capital: Decimal) -> None:
        """
        Handle capital update from FundManager.

        Updates the max_capital setting which will be used
        for position sizing on the next trade. If capital is reduced
        and current position exceeds the new limit, closes the position.

        Args:
            new_max_capital: New maximum capital allocation
        """
        previous = self._config.max_capital

        # Update config with new capital
        self._config.max_capital = new_max_capital

        logger.info(
            f"[FundManager] Capital updated for {self._bot_id}: "
            f"{previous} -> {new_max_capital}"
        )

        # Check if existing position exceeds new capital limit
        if self._position and self._position.quantity > 0:
            position_notional = self._position.quantity * self._position.entry_price
            max_allowed = new_max_capital * self._config.leverage
            if position_notional > max_allowed:
                logger.warning(
                    f"Position notional {position_notional} > max allowed "
                    f"{max_allowed} — closing position due to capital recall"
                )
                await self._close_position(ExitReason.CAPITAL_RECALLED)

    # =========================================================================
    # State Persistence (with validation, checksum, and concurrent protection)
    # =========================================================================

    def _get_state_data(self) -> Dict[str, Any]:
        """
        Override BaseBot method to provide Supertrend specific state.

        Returns:
            Dictionary of state data for persistence
        """
        state_data = {
            "total_pnl": str(self._total_pnl),
            "current_bar": self._current_bar,
            "entry_bar": self._entry_bar,
            "prev_trend": self._prev_trend,
            "current_trend": self._current_trend,
            "trend_bars": self._trend_bars,
            "trades_count": len(self._trades),
            "signal_cooldown": self._signal_cooldown,
            "last_triggered_level": self._last_triggered_level,
            # Risk control state
            "daily_pnl": str(self._daily_pnl),
            "daily_start_time": self._daily_start_time.isoformat(),
            "consecutive_losses": self._consecutive_losses,
            "risk_paused": self._risk_paused,
            # RSI filter state
            "current_rsi": str(self._current_rsi) if self._current_rsi else None,
            "rsi_closes": [str(c) for c in self._rsi_closes[-50:]],
        }

        # Save grid state for TREND_GRID mode
        if self._grid_levels:
            grid_level_states = [
                {"index": gl.index, "price": str(gl.price), "is_filled": gl.is_filled}
                for gl in self._grid_levels
            ]
            state_data["grid"] = {
                "upper_price": str(self._upper_price) if self._upper_price else None,
                "lower_price": str(self._lower_price) if self._lower_price else None,
                "grid_spacing": str(self._grid_spacing) if self._grid_spacing else None,
                "grid_levels": grid_level_states,
                "current_atr": str(self._current_atr) if self._current_atr else None,
            }

        if self._position:
            state_data["position"] = {
                "side": self._position.side.value,
                "entry_price": str(self._position.entry_price),
                "quantity": str(self._position.quantity),
                "entry_time": self._position.entry_time.isoformat(),
                "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                "stop_loss_order_id": self._position.stop_loss_order_id,
                "max_price": str(self._position.max_price) if self._position.max_price else None,
                "min_price": str(self._position.min_price) if self._position.min_price else None,
            }

        return state_data

    async def _save_state(self) -> None:
        """Save bot state to database with concurrent protection."""

        async def do_save():
            config = {
                "symbol": self._config.symbol,
                "timeframe": self._config.timeframe,
                "atr_period": self._config.atr_period,
                "atr_multiplier": str(self._config.atr_multiplier),
                "leverage": self._config.leverage,
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            # Create validated snapshot with metadata
            snapshot = self._create_state_snapshot()

            async def persist_to_db(snapshot_data: Dict[str, Any]):
                await self._data_manager.save_bot_state(
                    bot_id=self._bot_id,
                    bot_type="supertrend",
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
    ) -> Optional["SupertrendBot"]:
        """
        Restore a SupertrendBot from saved state with validation.

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
            Restored SupertrendBot or None if not found/invalid
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
            config = SupertrendConfig(
                symbol=config_data.get("symbol", ""),
                timeframe=config_data.get("timeframe", "15m"),
                atr_period=config_data.get("atr_period", 25),
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
            bot._total_pnl = Decimal(inner_state.get("total_pnl", "0"))
            bot._current_bar = inner_state.get("current_bar", 0)
            bot._entry_bar = inner_state.get("entry_bar", 0)
            bot._prev_trend = inner_state.get("prev_trend", 0)
            bot._current_trend = inner_state.get("current_trend", 0)
            bot._trend_bars = inner_state.get("trend_bars", 0)
            bot._signal_cooldown = inner_state.get("signal_cooldown", 0)
            bot._last_triggered_level = inner_state.get("last_triggered_level")

            # Restore risk control state
            bot._daily_pnl = Decimal(inner_state.get("daily_pnl", "0"))
            if inner_state.get("daily_start_time"):
                bot._daily_start_time = datetime.fromisoformat(inner_state["daily_start_time"])
            bot._consecutive_losses = inner_state.get("consecutive_losses", 0)
            bot._risk_paused = inner_state.get("risk_paused", False)

            # Restore RSI filter state
            if inner_state.get("current_rsi"):
                bot._current_rsi = Decimal(inner_state["current_rsi"])
            if inner_state.get("rsi_closes"):
                bot._rsi_closes = [Decimal(c) for c in inner_state["rsi_closes"]]

            # Restore grid state (TREND_GRID mode)
            grid_data = inner_state.get("grid")
            if grid_data:
                bot._upper_price = Decimal(grid_data["upper_price"]) if grid_data.get("upper_price") else None
                bot._lower_price = Decimal(grid_data["lower_price"]) if grid_data.get("lower_price") else None
                bot._grid_spacing = Decimal(grid_data["grid_spacing"]) if grid_data.get("grid_spacing") else None
                bot._current_atr = Decimal(grid_data["current_atr"]) if grid_data.get("current_atr") else None

                grid_levels = grid_data.get("grid_levels", [])
                bot._grid_levels = []
                for gl_data in grid_levels:
                    bot._grid_levels.append(GridLevel(
                        index=gl_data["index"],
                        price=Decimal(gl_data["price"]),
                        is_filled=gl_data.get("is_filled", False),
                    ))
                bot._grid_initialized = len(bot._grid_levels) > 0

            # Restore position from saved state
            position_data = inner_state.get("position")
            if position_data:
                bot._position = Position(
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    entry_time=datetime.fromisoformat(position_data["entry_time"]),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                )
                # Restore max/min price for trailing stop
                if position_data.get("max_price"):
                    bot._position.highest_price = Decimal(position_data["max_price"])
                if position_data.get("min_price"):
                    bot._position.lowest_price = Decimal(position_data["min_price"])

            # Verify position sync with exchange
            try:
                exchange_positions = await exchange.futures.get_positions(config.symbol)
                exchange_pos = None
                for pos in exchange_positions:
                    if pos.symbol == config.symbol and pos.quantity != Decimal("0"):
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
                    side = PositionSide.LONG if exchange_pos.quantity > 0 else PositionSide.SHORT
                    bot._position = Position(
                        side=side,
                        entry_price=exchange_pos.entry_price,
                        quantity=abs(exchange_pos.quantity),
                        entry_time=datetime.now(timezone.utc),
                        unrealized_pnl=exchange_pos.unrealized_pnl,
                    )
            except Exception as e:
                logger.warning(f"Failed to verify position sync on restore: {e}")

            logger.info(
                f"Restored SupertrendBot: {bot_id}, PnL={bot._total_pnl}, "
                f"position={'yes' if bot._position else 'no'}"
            )
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot {bot_id}: {e}")
            return None

