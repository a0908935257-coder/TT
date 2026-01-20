"""
RSI Mean Reversion Bot.

A futures trading bot using RSI for mean reversion entries.
- Buy when RSI < oversold (20)
- Sell when RSI > overbought (80)
- Exit when RSI returns to neutral (50) or stop loss/take profit hit
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from src.bots.base import BaseBot
from src.bots.rsi.indicators import RSICalculator
from src.bots.rsi.models import (
    ExitReason,
    Position,
    PositionSide,
    RSIConfig,
    Trade,
)
from src.core import get_logger
from src.core.models import Kline, OrderSide, OrderType

logger = get_logger(__name__)


class RSIBot(BaseBot):
    """
    RSI Mean Reversion Trading Bot.

    Uses RSI indicator to identify oversold/overbought conditions
    and trades mean reversion with proper risk management.

    Optimized configuration (83% walk-forward consistency):
    - RSI Period: 14
    - Oversold: 20, Overbought: 80
    - Leverage: 7x
    - Stop Loss: 2%, Take Profit: 3%
    """

    def __init__(
        self,
        bot_id: str,
        config: RSIConfig,
        exchange: Any,
        data_manager: Any = None,
        notifier: Any = None,
        heartbeat_callback: Optional[callable] = None,
    ):
        """Initialize RSI Bot."""
        super().__init__(
            bot_id=bot_id,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )

        self._config = config
        self._rsi_calc: Optional[RSICalculator] = None
        self._position: Optional[Position] = None
        self._trades: list[Trade] = []
        self._kline_task: Optional[asyncio.Task] = None
        self._fee_rate = Decimal("0.0004")

        # Statistics
        self._total_pnl = Decimal("0")
        self._win_count = 0
        self._loss_count = 0

    async def _do_start(self) -> bool:
        """Start the RSI bot."""
        logger.info(f"Initializing RSI Bot for {self._config.symbol}")

        try:
            # Set leverage and margin type
            await self._exchange.futures.set_leverage(
                symbol=self._config.symbol,
                leverage=self._config.leverage,
            )
            await self._exchange.futures.set_margin_type(
                symbol=self._config.symbol,
                margin_type=self._config.margin_type,
            )

            # Initialize RSI calculator
            self._rsi_calc = RSICalculator(
                period=self._config.rsi_period,
                oversold=self._config.oversold,
                overbought=self._config.overbought,
            )

            # Get historical klines for initialization
            klines = await self._exchange.futures.get_klines(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                limit=200,
            )

            if not klines or len(klines) < self._config.rsi_period + 10:
                logger.error("Insufficient klines for RSI initialization")
                return False

            # Initialize RSI
            result = self._rsi_calc.initialize(klines)
            if not result:
                logger.error("Failed to initialize RSI calculator")
                return False

            # Check for existing position
            await self._sync_position()

            # Start kline monitoring
            self._kline_task = asyncio.create_task(self._kline_loop())

            logger.info(f"RSI Bot initialized successfully")
            logger.info(f"  Symbol: {self._config.symbol}")
            logger.info(f"  Timeframe: {self._config.timeframe}")
            logger.info(f"  RSI Period: {self._config.rsi_period}")
            logger.info(f"  Oversold: {self._config.oversold}")
            logger.info(f"  Overbought: {self._config.overbought}")
            logger.info(f"  Leverage: {self._config.leverage}x")
            logger.info(f"  Initial RSI: {result.rsi:.2f}")

            return True

        except Exception as e:
            logger.error(f"Failed to start RSI Bot: {e}")
            return False

    async def _do_stop(self) -> bool:
        """Stop the RSI bot."""
        logger.info(f"Stopping RSI Bot for {self._config.symbol}")

        # Cancel kline task
        if self._kline_task:
            self._kline_task.cancel()
            try:
                await self._kline_task
            except asyncio.CancelledError:
                pass
            self._kline_task = None

        # Cancel stop loss order if exists
        if self._position and self._position.stop_loss_order_id:
            await self._cancel_stop_loss_order()

        logger.info("RSI Bot stopped")
        return True

    async def _do_pause(self) -> bool:
        """Pause the bot."""
        if self._kline_task:
            self._kline_task.cancel()
            try:
                await self._kline_task
            except asyncio.CancelledError:
                pass
            self._kline_task = None
        return True

    async def _do_resume(self) -> bool:
        """Resume the bot."""
        self._kline_task = asyncio.create_task(self._kline_loop())
        return True

    async def _kline_loop(self):
        """Main loop for monitoring klines."""
        logger.info("Starting RSI monitor loop")

        # Calculate sleep interval based on timeframe
        interval_seconds = self._parse_timeframe_seconds(self._config.timeframe)

        while True:
            try:
                # Get latest kline
                klines = await self._exchange.futures.get_klines(
                    symbol=self._config.symbol,
                    interval=self._config.timeframe,
                    limit=2,
                )

                if klines and len(klines) >= 2:
                    # Use the completed kline (second to last)
                    kline = klines[-2]
                    await self._process_kline(kline)

                # Send heartbeat
                self._send_heartbeat()

                # Sleep until next bar
                await asyncio.sleep(min(interval_seconds / 4, 60))

            except asyncio.CancelledError:
                logger.info("Kline loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in kline loop: {e}")
                await asyncio.sleep(10)

    async def _process_kline(self, kline: Kline):
        """Process new kline data."""
        # Update RSI
        result = self._rsi_calc.update(kline)
        if not result:
            return

        current_price = Decimal(str(kline.close))
        rsi = result.rsi

        # Check position management
        if self._position:
            await self._check_exit(current_price, rsi)
        else:
            await self._check_entry(current_price, rsi)

    async def _check_entry(self, price: Decimal, rsi: Decimal):
        """Check for entry signals."""
        if rsi < self._config.oversold:
            # Oversold - go long
            await self._open_position(PositionSide.LONG, price, rsi)
        elif rsi > self._config.overbought:
            # Overbought - go short
            await self._open_position(PositionSide.SHORT, price, rsi)

    async def _check_exit(self, price: Decimal, rsi: Decimal):
        """Check for exit conditions."""
        if not self._position:
            return

        pnl_pct = self._position.calculate_pnl_pct(price)
        exit_reason = None

        # Check stop loss
        if pnl_pct < -self._config.stop_loss_pct:
            exit_reason = ExitReason.STOP_LOSS

        # Check take profit
        elif pnl_pct > self._config.take_profit_pct:
            exit_reason = ExitReason.TAKE_PROFIT

        # Check RSI exit (return to neutral)
        elif self._position.side == PositionSide.LONG and rsi > self._config.exit_level:
            exit_reason = ExitReason.RSI_EXIT
        elif self._position.side == PositionSide.SHORT and rsi < self._config.exit_level:
            exit_reason = ExitReason.RSI_EXIT

        if exit_reason:
            await self._close_position(price, rsi, exit_reason)

    async def _open_position(self, side: PositionSide, price: Decimal, rsi: Decimal):
        """Open a new position."""
        try:
            # Calculate position size
            balance = await self._get_available_balance()
            if balance <= 0:
                logger.warning("Insufficient balance")
                return

            # Apply max_capital limit
            if self._config.max_capital:
                balance = min(balance, self._config.max_capital)

            position_value = balance * self._config.position_size_pct * self._config.leverage
            quantity = position_value / price

            # Round quantity
            quantity = quantity.quantize(Decimal("0.001"))

            if quantity <= 0:
                logger.warning("Calculated quantity too small")
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
                self._position = Position(
                    side=side,
                    entry_price=price,
                    quantity=quantity,
                    entry_time=datetime.now(timezone.utc),
                    entry_rsi=rsi,
                )

                logger.info(f"Opened {side.value} position: {quantity} @ {price}, RSI={rsi:.1f}")

                # Place exchange stop loss
                if self._config.use_exchange_stop_loss:
                    await self._place_stop_loss_order(price)

                # Send notification
                if self._notifier:
                    await self._notifier.send_info(
                        title=f"RSI Bot: {side.value} Entry",
                        message=f"Symbol: {self._config.symbol}\n"
                               f"Price: {price}\n"
                               f"RSI: {rsi:.1f}\n"
                               f"Quantity: {quantity}",
                    )

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

    async def _close_position(self, price: Decimal, rsi: Decimal, reason: ExitReason):
        """Close current position."""
        if not self._position:
            return

        try:
            # Cancel stop loss order first
            if self._position.stop_loss_order_id:
                await self._cancel_stop_loss_order()

            # Place close order
            order_side = OrderSide.SELL if self._position.side == PositionSide.LONG else OrderSide.BUY

            order = await self._exchange.futures.create_order(
                symbol=self._config.symbol,
                side=order_side,
                order_type=OrderType.MARKET,
                quantity=self._position.quantity,
                reduce_only=True,
            )

            if order:
                # Calculate PnL
                pnl_pct = self._position.calculate_pnl_pct(price)
                gross_pnl = pnl_pct * self._position.quantity * price
                fee = self._position.quantity * price * self._fee_rate * 2
                net_pnl = gross_pnl - fee

                # Record trade
                trade = Trade(
                    side=self._position.side,
                    entry_price=self._position.entry_price,
                    exit_price=price,
                    quantity=self._position.quantity,
                    pnl=net_pnl,
                    fee=fee,
                    entry_time=self._position.entry_time,
                    exit_time=datetime.now(timezone.utc),
                    exit_reason=reason,
                    entry_rsi=self._position.entry_rsi,
                    exit_rsi=rsi,
                )
                self._trades.append(trade)

                # Update statistics
                self._total_pnl += net_pnl
                if net_pnl > 0:
                    self._win_count += 1
                else:
                    self._loss_count += 1

                logger.info(
                    f"Closed {self._position.side.value} position: "
                    f"PnL={net_pnl:.4f} USDT ({pnl_pct*100:.2f}%), "
                    f"Reason={reason.value}, RSI={rsi:.1f}"
                )

                # Send notification
                if self._notifier:
                    pnl_emoji = "+" if net_pnl > 0 else ""
                    await self._notifier.send_info(
                        title=f"RSI Bot: Position Closed ({reason.value})",
                        message=f"Symbol: {self._config.symbol}\n"
                               f"PnL: {pnl_emoji}{net_pnl:.4f} USDT ({pnl_pct*100:.2f}%)\n"
                               f"Exit RSI: {rsi:.1f}\n"
                               f"Total PnL: {self._total_pnl:.4f} USDT",
                    )

                self._position = None

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    async def _place_stop_loss_order(self, entry_price: Decimal):
        """Place stop loss order on exchange."""
        if not self._position:
            return

        try:
            # Calculate stop loss price
            if self._position.side == PositionSide.LONG:
                stop_price = entry_price * (Decimal("1") - self._config.stop_loss_pct)
                close_side = OrderSide.SELL
            else:
                stop_price = entry_price * (Decimal("1") + self._config.stop_loss_pct)
                close_side = OrderSide.BUY

            stop_price = stop_price.quantize(Decimal("0.1"))

            # Place STOP_MARKET order
            sl_order = await self._exchange.futures.create_order(
                symbol=self._config.symbol,
                side=close_side,
                order_type="STOP_MARKET",
                quantity=self._position.quantity,
                stop_price=stop_price,
                reduce_only=True,
            )

            if sl_order:
                self._position.stop_loss_order_id = str(sl_order.order_id)
                self._position.stop_loss_price = stop_price
                logger.info(f"Stop loss placed: {close_side.value} @ {stop_price}")

        except Exception as e:
            logger.error(f"Failed to place stop loss: {e}")

    async def _cancel_stop_loss_order(self):
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
        except Exception as e:
            logger.debug(f"Failed to cancel stop loss: {e}")

    async def _sync_position(self):
        """Sync position from exchange."""
        try:
            positions = await self._exchange.futures.get_positions(
                symbol=self._config.symbol
            )

            for pos in positions:
                if float(pos.get("positionAmt", 0)) != 0:
                    amt = Decimal(str(pos["positionAmt"]))
                    entry = Decimal(str(pos["entryPrice"]))
                    side = PositionSide.LONG if amt > 0 else PositionSide.SHORT

                    self._position = Position(
                        side=side,
                        entry_price=entry,
                        quantity=abs(amt),
                        entry_time=datetime.now(timezone.utc),
                    )
                    logger.info(f"Synced existing position: {side.value} {abs(amt)} @ {entry}")
                    break

        except Exception as e:
            logger.error(f"Failed to sync position: {e}")

    async def _get_available_balance(self) -> Decimal:
        """Get available USDT balance."""
        try:
            balance = await self._exchange.futures.get_balance()
            for asset in balance:
                if asset.get("asset") == "USDT":
                    return Decimal(str(asset.get("availableBalance", 0)))
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")
        return Decimal("0")

    def _parse_timeframe_seconds(self, timeframe: str) -> int:
        """Parse timeframe string to seconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        multipliers = {"m": 60, "h": 3600, "d": 86400}
        return value * multipliers.get(unit, 60)

    def get_status(self) -> dict:
        """Get current bot status."""
        rsi_state = self._rsi_calc.get_state() if self._rsi_calc else {}

        return {
            "symbol": self._config.symbol,
            "timeframe": self._config.timeframe,
            "leverage": self._config.leverage,
            "rsi": rsi_state.get("rsi"),
            "rsi_signal": rsi_state.get("signal"),
            "position": {
                "side": self._position.side.value if self._position else None,
                "entry_price": float(self._position.entry_price) if self._position else None,
                "quantity": float(self._position.quantity) if self._position else None,
            } if self._position else None,
            "total_trades": len(self._trades),
            "total_pnl": float(self._total_pnl),
            "win_rate": self._win_count / (self._win_count + self._loss_count) * 100
                        if (self._win_count + self._loss_count) > 0 else 0,
        }

    def get_statistics(self) -> dict:
        """Get trading statistics."""
        return {
            "total_trades": len(self._trades),
            "winning_trades": self._win_count,
            "losing_trades": self._loss_count,
            "total_profit": float(self._total_pnl),
            "win_rate": self._win_count / (self._win_count + self._loss_count) * 100
                        if (self._win_count + self._loss_count) > 0 else 0,
        }
