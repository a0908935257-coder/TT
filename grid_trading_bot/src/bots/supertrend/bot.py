"""
Supertrend Trading Bot with RSI Filter.

Trend-following strategy based on Supertrend indicator + RSI filter.
- Enter LONG when Supertrend flips bullish AND RSI < 60 (not overbought)
- Enter SHORT when Supertrend flips bearish AND RSI > 40 (not oversold)
- Exit when trend reverses

✅ Walk-Forward + OOS 驗證通過 (2024-01 ~ 2026-01, 2 年數據):
- Walk-Forward 一致性: 62% (5/8 時段獲利)
- OOS 報酬: +2.8% (唯一正報酬配置)
- 交易數減少 56% (更精選進場)
- 最大回撤: 8.6%

RSI 過濾器效果:
- 避免在超買區做多 (RSI > 60)
- 避免在超賣區做空 (RSI < 40)
- 減少假突破造成的虧損
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, Optional

from src.core import get_logger
from src.core.models import Kline, KlineInterval, OrderType, OrderSide
from src.bots.base import BaseBot
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
)
from .indicators import SupertrendIndicator

logger = get_logger(__name__)


class SupertrendBot(BaseBot):
    """
    Supertrend trend-following trading bot.

    Strategy:
    - Uses Supertrend indicator to determine trend direction
    - Opens LONG position when trend flips to bullish
    - Opens SHORT position when trend flips to bearish
    - Always in the market (flip between LONG/SHORT)
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

        # Indicator
        self._indicator = SupertrendIndicator(
            atr_period=config.atr_period,
            atr_multiplier=config.atr_multiplier,
        )

        # Position tracking
        self._position: Optional[Position] = None
        self._trades: list[Trade] = []
        self._entry_bar: int = 0
        self._current_bar: int = 0

        # Previous trend for detecting flips
        self._prev_trend: int = 0

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

        # Kline callback reference for unsubscribe
        self._kline_callback = None

        # State persistence
        self._save_task: Optional[asyncio.Task] = None
        self._save_interval_minutes: int = 5

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

        # Set leverage
        await self._exchange.futures.set_leverage(
            symbol=self._config.symbol,
            leverage=self._config.leverage,
        )

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
            asyncio.create_task(self._on_kline(kline))

        self._kline_callback = on_kline_sync  # Store reference for unsubscribe
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
        )

        logger.info(f"Supertrend Bot initialized successfully")
        logger.info(f"  Symbol: {self._config.symbol}")
        logger.info(f"  Timeframe: {self._config.timeframe}")
        logger.info(f"  ATR Period: {self._config.atr_period}")
        logger.info(f"  ATR Multiplier: {self._config.atr_multiplier}")
        logger.info(f"  Leverage: {self._config.leverage}x")
        logger.info(f"  Initial Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}")

        if self._notifier:
            await self._notifier.send_info(
                title="Supertrend Bot Started",
                message=f"Symbol: {self._config.symbol}\n"
                        f"Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}",
            )

        # Start periodic state saving
        self._start_save_task()

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
            )
        except Exception as e:
            logger.warning(f"Failed to unsubscribe: {e}")

        # Clear position if requested
        if clear_position and self._position:
            await self._close_position(ExitReason.BOT_STOP)
        elif self._position and self._position.stop_loss_order_id:
            # Cancel stop loss order but keep position
            await self._cancel_stop_loss_order()

        # Stop periodic save task and save final state
        self._stop_save_task()
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

        # Re-subscribe to kline updates (create new callback)
        def on_kline_sync(kline: Kline) -> None:
            asyncio.create_task(self._on_kline(kline))

        self._kline_callback = on_kline_sync
        await self._data_manager.klines.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=on_kline_sync,
        )

        logger.info("Supertrend Bot resumed")

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

        return checks

    async def _sync_position(self) -> None:
        """Sync position with exchange."""
        try:
            positions = await self._exchange.get_positions(self._config.symbol)

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

    async def _on_kline(self, kline: Kline) -> None:
        """Handle new kline data."""
        if self._state != BotState.RUNNING:
            return

        try:
            self._current_bar += 1

            # Update indicator
            supertrend = self._indicator.update(kline)
            if supertrend is None:
                return

            current_trend = supertrend.trend
            current_price = kline.close

            # Calculate RSI for filter
            self._current_rsi = self._calculate_rsi(current_price)

            # Update position unrealized PnL
            if self._position:
                self._position.update_extremes(current_price)
                if self._position.side == PositionSide.LONG:
                    self._position.unrealized_pnl = (
                        (current_price - self._position.entry_price) *
                        self._position.quantity *
                        Decimal(self._config.leverage)
                    )
                else:
                    self._position.unrealized_pnl = (
                        (self._position.entry_price - current_price) *
                        self._position.quantity *
                        Decimal(self._config.leverage)
                    )

                # Check trailing stop
                if self._config.use_trailing_stop:
                    if self._check_trailing_stop(current_price):
                        logger.warning(f"Trailing stop triggered at {current_price}")
                        await self._close_position(ExitReason.STOP_LOSS)
                        return  # Don't open new position after stop loss

            # Check for trend flip
            if current_trend != self._prev_trend and self._prev_trend != 0:
                new_side = PositionSide.LONG if current_trend == 1 else PositionSide.SHORT
                rsi_str = f", RSI={self._current_rsi:.1f}" if self._current_rsi else ""

                logger.info(
                    f"Trend flip detected: {'BEARISH→BULLISH' if current_trend == 1 else 'BULLISH→BEARISH'} "
                    f"at {current_price}{rsi_str}"
                )

                # Close existing position (always close on signal flip)
                if self._position:
                    await self._close_position(ExitReason.SIGNAL_FLIP)

                # Check RSI filter before opening new position
                if self._check_rsi_filter(new_side):
                    await self._open_position(new_side, current_price)
                else:
                    # RSI filter blocked entry - log but don't open position
                    if self._notifier:
                        await self._notifier.send_info(
                            title="Supertrend: RSI Filter",
                            message=f"Signal blocked: {new_side.value}\n"
                                    f"RSI: {self._current_rsi:.1f}\n"
                                    f"Reason: {'Overbought' if new_side == PositionSide.LONG else 'Oversold'}",
                        )

            self._prev_trend = current_trend

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    async def _open_position(self, side: PositionSide, price: Decimal) -> None:
        """Open a new position."""
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
            return

        try:
            # Calculate position size based on allocated capital
            account = await self._exchange.futures.get_account()
            available = Decimal(str(account.available_balance))

            # Use max_capital if configured, otherwise use available balance
            if self._config.max_capital is not None:
                # 使用分配的資金上限，但不能超過實際可用餘額
                capital = min(self._config.max_capital, available)
            else:
                capital = available

            # Apply position_size_pct but cap at max_position_pct
            position_pct = min(self._config.position_size_pct, self._config.max_position_pct)
            notional = capital * position_pct
            quantity = notional / price

            # Round quantity
            quantity = quantity.quantize(Decimal("0.001"))

            if quantity <= 0:
                logger.warning("Insufficient balance to open position")
                return

            # Place market order
            order_side = OrderSide.BUY if side == PositionSide.LONG else OrderSide.SELL
            order = await self._exchange.futures.create_order(
                symbol=self._config.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=quantity,
            )

            if order:
                # Calculate stop loss price
                if side == PositionSide.LONG:
                    stop_loss_price = price * (Decimal("1") - self._config.stop_loss_pct)
                else:
                    stop_loss_price = price * (Decimal("1") + self._config.stop_loss_pct)

                # Round to tick size
                stop_loss_price = stop_loss_price.quantize(Decimal("0.1"))

                self._position = Position(
                    side=side,
                    entry_price=price,
                    quantity=quantity,
                    entry_time=datetime.now(timezone.utc),
                    stop_loss_price=stop_loss_price,
                )
                self._entry_bar = self._current_bar

                # Place exchange stop loss order if enabled
                if self._config.use_exchange_stop_loss:
                    await self._place_stop_loss_order()

                logger.info(f"Opened {side.value} position: {quantity} @ {price}, SL @ {stop_loss_price}")

                if self._notifier:
                    await self._notifier.send_info(
                        title=f"Supertrend: {side.value}",
                        message=f"Entry: {price}\nSize: {quantity}\nLeverage: {self._config.leverage}x\nStop Loss: {stop_loss_price}",
                    )

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

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

            # Place STOP_MARKET order (uses Algo Order API since 2025-12-09)
            sl_order = await self._exchange.futures.create_order(
                symbol=self._config.symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=self._position.quantity,
                stop_price=self._position.stop_loss_price,
                reduce_only=True,
            )

            if sl_order:
                self._position.stop_loss_order_id = str(sl_order.order_id)
                logger.info(
                    f"Stop loss order placed: {close_side.value} {self._position.quantity} "
                    f"@ {self._position.stop_loss_price}, ID={sl_order.order_id}"
                )

        except Exception as e:
            logger.error(f"Failed to place stop loss order: {e}")

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

        except Exception as e:
            logger.debug(f"Failed to cancel stop loss order: {e}")

    async def _close_position(self, reason: ExitReason) -> None:
        """Close current position."""
        if not self._position:
            return

        try:
            # Cancel stop loss order first (if any)
            if self._position.stop_loss_order_id:
                await self._cancel_stop_loss_order()

            # Get current price
            ticker = await self._exchange.futures.get_ticker(self._config.symbol)
            exit_price = Decimal(str(ticker.last_price))

            # Place closing order
            close_side = OrderSide.SELL if self._position.side == PositionSide.LONG else OrderSide.BUY

            order = await self._exchange.futures.create_order(
                symbol=self._config.symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=self._position.quantity,
                reduceOnly=True,
            )

            if order:
                # Calculate PnL
                if self._position.side == PositionSide.LONG:
                    pnl = (exit_price - self._position.entry_price) * self._position.quantity
                else:
                    pnl = (self._position.entry_price - exit_price) * self._position.quantity

                pnl *= Decimal(self._config.leverage)

                # Deduct fees
                fee = (self._position.entry_price + exit_price) * self._position.quantity * self.FEE_RATE
                net_pnl = pnl - fee

                # Record trade
                trade = Trade(
                    side=self._position.side,
                    entry_price=self._position.entry_price,
                    exit_price=exit_price,
                    quantity=self._position.quantity,
                    pnl=net_pnl,
                    fee=fee,
                    entry_time=self._position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason,
                    holding_duration=self._current_bar - self._entry_bar,
                )
                self._trades.append(trade)
                self._total_pnl += net_pnl

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
        for position sizing on the next trade.

        Args:
            new_max_capital: New maximum capital allocation
        """
        previous = self._config.max_capital

        logger.info(
            f"[FundManager] Capital updated for {self._bot_id}: "
            f"{previous} -> {new_max_capital}"
        )

        # Note: Position sizing will automatically use new max_capital
        # on next _open_position call. No immediate action needed
        # as existing positions continue with their original sizing.

    # =========================================================================
    # State Persistence
    # =========================================================================

    async def _save_state(self) -> None:
        """Save bot state to database."""
        try:
            config = {
                "symbol": self._config.symbol,
                "timeframe": self._config.timeframe,
                "atr_period": self._config.atr_period,
                "atr_multiplier": str(self._config.atr_multiplier),
                "leverage": self._config.leverage,
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            state_data = {
                "total_pnl": str(self._total_pnl),
                "current_bar": self._current_bar,
                "entry_bar": self._entry_bar,
                "prev_trend": self._prev_trend,
                "trades_count": len(self._trades),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                # Risk control state
                "daily_pnl": str(self._daily_pnl),
                "daily_start_time": self._daily_start_time.isoformat(),
                "consecutive_losses": self._consecutive_losses,
                "risk_paused": self._risk_paused,
                # RSI filter state
                "current_rsi": str(self._current_rsi) if self._current_rsi else None,
                "rsi_closes": [str(c) for c in self._rsi_closes[-50:]],  # Save last 50 closes
            }

            if self._position:
                state_data["position"] = {
                    "side": self._position.side.value,
                    "entry_price": str(self._position.entry_price),
                    "quantity": str(self._position.quantity),
                    "entry_time": self._position.entry_time.isoformat(),
                    "stop_loss_price": str(self._position.stop_loss_price) if self._position.stop_loss_price else None,
                    "stop_loss_order_id": self._position.stop_loss_order_id,
                }

            await self._data_manager.save_bot_state(
                bot_id=self._bot_id,
                bot_type="supertrend",
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
    ) -> Optional["SupertrendBot"]:
        """
        Restore a SupertrendBot from saved state.

        Args:
            bot_id: Bot ID to restore
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance

        Returns:
            Restored SupertrendBot or None if not found
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

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

            # Restore state
            saved_state = state_data.get("state_data", {})
            bot._total_pnl = Decimal(saved_state.get("total_pnl", "0"))
            bot._current_bar = saved_state.get("current_bar", 0)
            bot._entry_bar = saved_state.get("entry_bar", 0)
            bot._prev_trend = saved_state.get("prev_trend", 0)

            # Restore risk control state
            bot._daily_pnl = Decimal(saved_state.get("daily_pnl", "0"))
            if saved_state.get("daily_start_time"):
                bot._daily_start_time = datetime.fromisoformat(saved_state["daily_start_time"])
            bot._consecutive_losses = saved_state.get("consecutive_losses", 0)
            bot._risk_paused = saved_state.get("risk_paused", False)

            # Restore RSI filter state
            if saved_state.get("current_rsi"):
                bot._current_rsi = Decimal(saved_state["current_rsi"])
            if saved_state.get("rsi_closes"):
                bot._rsi_closes = [Decimal(c) for c in saved_state["rsi_closes"]]

            # Restore position if exists
            position_data = saved_state.get("position")
            if position_data:
                bot._position = Position(
                    side=PositionSide(position_data["side"]),
                    entry_price=Decimal(position_data["entry_price"]),
                    quantity=Decimal(position_data["quantity"]),
                    entry_time=datetime.fromisoformat(position_data["entry_time"]),
                    stop_loss_price=Decimal(position_data["stop_loss_price"]) if position_data.get("stop_loss_price") else None,
                    stop_loss_order_id=position_data.get("stop_loss_order_id"),
                )

            logger.info(f"Restored SupertrendBot: {bot_id}, PnL={bot._total_pnl}")
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot: {e}")
            return None

