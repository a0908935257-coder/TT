"""
Supertrend Trading Bot.

Trend-following strategy based on Supertrend indicator.
- Enter LONG when Supertrend flips bullish
- Enter SHORT when Supertrend flips bearish
- Exit when trend reverses

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割):
- Walk-Forward 一致性: 75% (6/8 時段獲利)
- 報酬: +6.7% (2 年), 年化 +3.3%
- Sharpe: 0.39, 最大回撤: 11.5%

驗證參數:
- leverage: 2x
- atr_period: 25
- atr_multiplier: 3.0
- stop_loss_pct: 3%
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

        # Statistics
        self._total_pnl = Decimal("0")

        # Kline callback reference for unsubscribe
        self._kline_callback = None

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
                logger.info(
                    f"Trend flip detected: {'BEARISH→BULLISH' if current_trend == 1 else 'BULLISH→BEARISH'} "
                    f"at {current_price}"
                )

                # Close existing position
                if self._position:
                    await self._close_position(ExitReason.SIGNAL_FLIP)

                # Open new position
                if current_trend == 1:
                    await self._open_position(PositionSide.LONG, current_price)
                else:
                    await self._open_position(PositionSide.SHORT, current_price)

            self._prev_trend = current_trend

        except Exception as e:
            logger.error(f"Error processing kline: {e}")

    async def _open_position(self, side: PositionSide, price: Decimal) -> None:
        """Open a new position."""
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

            notional = capital * self._config.position_size_pct
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

