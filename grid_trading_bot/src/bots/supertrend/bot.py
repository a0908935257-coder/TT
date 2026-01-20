"""
Supertrend Trading Bot.

Trend-following strategy based on Supertrend indicator.
- Enter LONG when Supertrend flips bullish
- Enter SHORT when Supertrend flips bearish
- Exit when trend reverses
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from src.core import get_logger
from src.core.models import Kline, KlineInterval
from src.bots.base import BaseBot, BotState
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
    ):
        super().__init__(bot_id)
        self._config = config
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier

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

    async def start(self) -> bool:
        """Start the trading bot."""
        try:
            self._state = BotState.STARTING
            logger.info(f"Starting Supertrend Bot for {self._config.symbol}")

            # Set margin type (ISOLATED for risk control)
            await self._exchange.set_margin_type(
                symbol=self._config.symbol,
                margin_type=self._config.margin_type,
            )

            # Set leverage
            await self._exchange.set_leverage(
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

            klines = await self._exchange.get_klines(
                symbol=self._config.symbol,
                interval=interval,
                limit=100,
            )

            if not klines or len(klines) < self._config.atr_period + 10:
                logger.error("Not enough historical data to initialize indicator")
                return False

            # Initialize indicator
            self._indicator.initialize_from_klines(klines)
            self._prev_trend = self._indicator.trend

            # Check existing position
            await self._sync_position()

            # Subscribe to kline updates
            await self._data_manager.subscribe_klines(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                callback=self._on_kline,
            )

            self._state = BotState.RUNNING
            logger.info(f"Supertrend Bot started successfully")
            logger.info(f"  Symbol: {self._config.symbol}")
            logger.info(f"  Timeframe: {self._config.timeframe}")
            logger.info(f"  ATR Period: {self._config.atr_period}")
            logger.info(f"  ATR Multiplier: {self._config.atr_multiplier}")
            logger.info(f"  Leverage: {self._config.leverage}x")
            logger.info(f"  Initial Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}")

            if self._notifier:
                await self._notifier.send_notification(
                    title="Supertrend Bot Started",
                    message=f"Symbol: {self._config.symbol}\n"
                            f"Trend: {'BULLISH' if self._indicator.is_bullish else 'BEARISH'}",
                )

            return True

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            self._state = BotState.ERROR
            return False

    async def stop(self, clear_position: bool = False) -> None:
        """Stop the trading bot."""
        logger.info("Stopping Supertrend Bot...")
        self._state = BotState.STOPPING

        # Unsubscribe from updates
        await self._data_manager.unsubscribe_klines(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
        )

        # Clear position if requested
        if clear_position and self._position:
            await self._close_position(ExitReason.BOT_STOP)

        self._state = BotState.STOPPED
        logger.info("Supertrend Bot stopped")

        if self._notifier:
            await self._notifier.send_notification(
                title="Supertrend Bot Stopped",
                message=f"Total PnL: {self._total_pnl:+.2f} USDT",
            )

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
            balance = await self._exchange.get_balance()
            available = balance.available_balance

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

            # Place order
            order = await self._exchange.place_market_order(
                symbol=self._config.symbol,
                side="BUY" if side == PositionSide.LONG else "SELL",
                quantity=quantity,
            )

            if order:
                self._position = Position(
                    side=side,
                    entry_price=price,
                    quantity=quantity,
                    entry_time=datetime.now(timezone.utc),
                )
                self._entry_bar = self._current_bar

                logger.info(f"Opened {side.value} position: {quantity} @ {price}")

                if self._notifier:
                    await self._notifier.send_notification(
                        title=f"Supertrend: {side.value}",
                        message=f"Entry: {price}\nSize: {quantity}\nLeverage: {self._config.leverage}x",
                    )

        except Exception as e:
            logger.error(f"Failed to open position: {e}")

    async def _close_position(self, reason: ExitReason) -> None:
        """Close current position."""
        if not self._position:
            return

        try:
            # Get current price
            ticker = await self._exchange.get_ticker(self._config.symbol)
            exit_price = ticker.last_price

            # Place closing order
            close_side = "SELL" if self._position.side == PositionSide.LONG else "BUY"

            order = await self._exchange.place_market_order(
                symbol=self._config.symbol,
                side=close_side,
                quantity=self._position.quantity,
                reduce_only=True,
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
                    await self._notifier.send_notification(
                        title=f"{emoji} Supertrend: Close {self._position.side.value}",
                        message=f"Exit: {exit_price}\nPnL: {net_pnl:+.2f} USDT\n"
                                f"Reason: {reason.value}\nTotal: {self._total_pnl:+.2f} USDT",
                    )

                self._position = None

        except Exception as e:
            logger.error(f"Failed to close position: {e}")

    def get_status(self) -> dict:
        """Get bot status."""
        position_info = None
        if self._position:
            position_info = {
                "side": self._position.side.value,
                "entry_price": float(self._position.entry_price),
                "quantity": float(self._position.quantity),
                "unrealized_pnl": float(self._position.unrealized_pnl),
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
            "bot_id": self._bot_id,
            "state": self._state.value,
            "symbol": self._config.symbol,
            "timeframe": self._config.timeframe,
            "leverage": self._config.leverage,
            "position": position_info,
            "supertrend": supertrend_info,
            "total_trades": len(self._trades),
            "total_pnl": float(self._total_pnl),
            "current_bar": self._current_bar,
        }
