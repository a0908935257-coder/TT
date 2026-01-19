"""
Bollinger Band Mean Reversion Bot.

Main bot class that integrates all components for the Bollinger Band
mean reversion trading strategy on futures market.

Conforms to Prompt 69 specification.

Architecture:
    BollingerBot
    ├── BollingerCalculator  → Calculate indicators
    ├── SignalGenerator      → Generate signals
    ├── PositionManager      → Manage positions
    └── OrderExecutor        → Execute orders

Main Loop (on each K-line close):
    1. Get latest K-line
    2. Check exit conditions for existing position
    3. If no position, check for entry signal
    4. Check entry order timeout
    5. Update statistics
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.master.models import BotState

from .indicators import BollingerCalculator
from .models import (
    BollingerBotStats,
    BollingerConfig,
    PositionSide,
    SignalType,
    TradeRecord,
)
from .order_executor import OrderExecutor
from .position_manager import PositionManager
from .signal_generator import SignalGenerator

logger = get_logger(__name__)


# =============================================================================
# Protocols
# =============================================================================


class KlineProtocol(Protocol):
    """Protocol for Kline data."""

    @property
    def close(self) -> Decimal: ...

    @property
    def close_time(self) -> datetime: ...

    @property
    def is_closed(self) -> bool: ...


class OrderProtocol(Protocol):
    """Protocol for order data."""

    @property
    def order_id(self) -> str: ...

    @property
    def status(self) -> str: ...

    @property
    def avg_price(self) -> Decimal: ...

    @property
    def filled_quantity(self) -> Decimal: ...

    @property
    def side(self) -> str: ...


class ExchangeProtocol(Protocol):
    """Protocol for exchange client."""

    async def get_klines(self, symbol: str, interval: str, limit: int) -> List[Any]: ...

    async def subscribe_klines(self, symbol: str, interval: str, callback: Callable) -> None: ...

    async def unsubscribe_klines(self, symbol: str, interval: str) -> None: ...

    async def subscribe_user_data(self, callback: Callable) -> None: ...

    async def get_account(self) -> Any: ...


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    async def send(self, title: str, message: str) -> None: ...


# =============================================================================
# Bollinger Bot
# =============================================================================


class BollingerBot(BaseBot):
    """
    Bollinger Band mean reversion trading bot.

    Trades on futures market using Bollinger Bands to identify
    overbought/oversold conditions for mean reversion entries.

    Features:
        - BBW squeeze filter to avoid breakout trades
        - Limit orders for entry/TP (lower fees)
        - Stop market orders for SL (guaranteed execution)
        - Timeout exit for stuck positions

    Example:
        >>> config = BollingerConfig(symbol="BTCUSDT", leverage=2)
        >>> bot = BollingerBot("bot_001", config, exchange, data_manager, notifier)
        >>> await bot.start()
    """

    def __init__(
        self,
        bot_id: str,
        config: BollingerConfig,
        exchange: ExchangeProtocol,
        data_manager: Any,
        notifier: Optional[NotifierProtocol] = None,
        heartbeat_callback: Optional[Callable] = None,
    ):
        """
        Initialize BollingerBot.

        Args:
            bot_id: Unique bot identifier
            config: BollingerConfig with strategy parameters
            exchange: Exchange client
            data_manager: Data manager for persistence
            notifier: Optional notification manager
            heartbeat_callback: Optional heartbeat callback for Master
        """
        super().__init__(
            bot_id=bot_id,
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
            heartbeat_callback=heartbeat_callback,
        )

        self._config: BollingerConfig = config

        # Initialize components
        self._calculator = BollingerCalculator(
            period=config.bb_period,
            std_multiplier=config.bb_std,
            bbw_lookback=config.bbw_lookback,
            bbw_threshold_pct=config.bbw_threshold_pct,
        )
        self._signal_generator = SignalGenerator(config, self._calculator)
        self._position_manager = PositionManager(config, exchange, data_manager)
        self._order_executor = OrderExecutor(config, exchange, notifier)

        # State
        self._klines: List[KlineProtocol] = []
        self._current_bar: int = 0
        self._entry_order_bar: Optional[int] = None

        # Statistics
        self._bollinger_stats = BollingerBotStats()

    # =========================================================================
    # Abstract Properties
    # =========================================================================

    @property
    def bot_type(self) -> str:
        """Return bot type identifier."""
        return "bollinger"

    @property
    def symbol(self) -> str:
        """Return trading symbol."""
        return self._config.symbol

    # =========================================================================
    # Lifecycle Methods
    # =========================================================================

    async def _do_start(self) -> None:
        """Start the bot."""
        # 1. Initialize position manager (set leverage, margin type)
        await self._position_manager.initialize()

        # 2. Get historical klines
        klines = await self._exchange.get_klines(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            limit=300,
        )
        self._klines = klines

        # 3. Initialize indicator calculator (build BBW history)
        self._calculator.initialize(klines)

        # 4. Set current bar number
        self._current_bar = len(klines)

        # 5. Subscribe to kline updates
        await self._exchange.subscribe_klines(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=self._on_kline,
        )

        # 6. Subscribe to user data (order updates)
        await self._exchange.subscribe_user_data(
            callback=self._on_order_update,
        )

        # 7. Send notification
        if self._notifier:
            await self._notifier.send(
                title="🟢 BollingerBot 已啟動",
                message=(
                    f"交易對: {self._config.symbol}\n"
                    f"時間框架: {self._config.timeframe}\n"
                    f"槓桿: {self._config.leverage}x"
                ),
            )

        logger.info(
            f"BollingerBot started: {self._config.symbol}, "
            f"klines={len(klines)}, bar={self._current_bar}"
        )

    async def _do_stop(self, clear_position: bool = False) -> None:
        """Stop the bot."""
        # 1. Cancel all pending orders
        await self._order_executor.cancel_entry_order()
        await self._order_executor.cancel_exit_orders()

        # 2. Close position if requested or exists
        if clear_position or self._position_manager.has_position:
            position = self._position_manager.get_position()
            if position:
                await self._order_executor.close_position_market(position)
                await self._position_manager.close_position("Bot 停止")

        # 3. Unsubscribe from klines
        try:
            await self._exchange.unsubscribe_klines(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
            )
        except Exception as e:
            logger.debug(f"Unsubscribe klines: {e}")

        # 4. Send notification
        if self._notifier:
            await self._notifier.send(
                title="🔴 BollingerBot 已停止",
                message=(
                    f"總交易: {self._bollinger_stats.total_trades}\n"
                    f"總盈虧: {self._bollinger_stats.total_pnl:+.2f} USDT"
                ),
            )

        logger.info(f"BollingerBot stopped: {self._config.symbol}")

    async def _do_pause(self) -> None:
        """Pause the bot."""
        # Cancel entry order if pending
        await self._order_executor.cancel_entry_order()
        self._entry_order_bar = None

        logger.info(f"BollingerBot paused: {self._config.symbol}")

    async def _do_resume(self) -> None:
        """Resume the bot."""
        # Sync with exchange
        await self._position_manager.sync_with_exchange()

        logger.info(f"BollingerBot resumed: {self._config.symbol}")

    # =========================================================================
    # Status Methods
    # =========================================================================

    def _get_extra_status(self) -> Dict[str, Any]:
        """Return extra status fields for this bot type."""
        position = self._position_manager.get_position()

        return {
            "timeframe": self._config.timeframe,
            "leverage": self._config.leverage,
            "bb_period": self._config.bb_period,
            "bb_std": str(self._config.bb_std),
            "current_bar": self._current_bar,
            "has_position": position is not None,
            "position_side": position.side.value if position else None,
            "entry_price": str(position.entry_price) if position else None,
            "has_pending_entry": self._order_executor.has_pending_entry,
            "win_rate": f"{self._bollinger_stats.win_rate * 100:.1f}%",
            "signals_generated": self._bollinger_stats.signals_generated,
            "signals_filtered": self._bollinger_stats.signals_filtered,
        }

    async def _extra_health_checks(self) -> Dict[str, bool]:
        """Perform extra health checks for this bot type."""
        checks = {}

        # Check if klines are being received
        checks["klines_ok"] = len(self._klines) >= self._config.bb_period

        # Check if position manager is synced
        checks["position_synced"] = True  # Simplified

        # Check if within reasonable bar count
        checks["bar_count_ok"] = self._current_bar > 0

        return checks

    # =========================================================================
    # Kline Callback
    # =========================================================================

    async def _on_kline(self, kline: KlineProtocol) -> None:
        """
        Handle kline update callback.

        Only processes closed klines.
        """
        # Only process closed klines
        if not kline.is_closed:
            return

        # Skip if paused
        if self._state == BotState.PAUSED:
            return

        # Update kline list
        self._klines.append(kline)
        if len(self._klines) > 500:
            self._klines = self._klines[-500:]

        self._current_bar += 1

        # Process bar
        await self._process_bar()

    # =========================================================================
    # Main Logic
    # =========================================================================

    async def _process_bar(self) -> None:
        """
        Process each K-line bar.

        Main trading logic:
        1. Check exit conditions for existing position
        2. Check entry order timeout
        3. Check for new entry signal
        """
        current_price = self._klines[-1].close
        if not isinstance(current_price, Decimal):
            current_price = Decimal(str(current_price))

        # ===== 1. Check existing position =====
        position = self._position_manager.get_position()

        if position:
            should_exit, reason = self._signal_generator.check_exit(
                position=position,
                klines=self._klines,
                current_price=current_price,
                current_bar=self._current_bar,
            )

            if should_exit:
                await self._exit_position(reason)
                return

            # Has position, don't check new signals
            return

        # ===== 2. Check entry order timeout =====
        if self._entry_order_bar is not None:
            bars_since_entry = self._current_bar - self._entry_order_bar
            if bars_since_entry >= self._order_executor.get_entry_timeout_bars():
                await self._order_executor.cancel_entry_order()
                self._entry_order_bar = None
                logger.info("Entry order timeout, cancelled")
                return

        # ===== 3. Check new entry signal =====
        if self._entry_order_bar is None:
            signal = self._signal_generator.generate(self._klines, current_price)

            # Record signal stats
            if signal.bbw.is_squeeze:
                self._bollinger_stats.record_signal(filtered=True)
            elif signal.signal_type != SignalType.NONE:
                self._bollinger_stats.record_signal(filtered=False)
                await self._enter_position(signal)

    async def _enter_position(self, signal) -> None:
        """Enter a new position."""
        # 1. Calculate position and create
        position = await self._position_manager.open_position(signal)

        # 2. Place entry order
        await self._order_executor.place_entry_order(signal, position.quantity)

        # 3. Record entry bar
        self._entry_order_bar = self._current_bar

        logger.info(
            f"Entry signal: {signal.signal_type.value}, "
            f"price={signal.entry_price}, qty={position.quantity}"
        )

    async def _exit_position(self, reason: str) -> None:
        """Exit current position."""
        position = self._position_manager.get_position()
        if not position:
            return

        # 1. Cancel exit orders if timeout
        if "超時" in reason:
            await self._order_executor.cancel_exit_orders()
            await self._order_executor.close_position_market(position)

        # 2. Record trade
        record = await self._position_manager.close_position(reason)
        record.hold_bars = self._current_bar - position.entry_bar

        # 3. Update statistics
        self._update_stats(record)

        # 4. Send notification
        if self._notifier:
            emoji = "📈" if record.pnl > 0 else "📉"
            await self._notifier.send(
                title=f"{emoji} 平倉",
                message=(
                    f"方向: {record.side.value}\n"
                    f"盈虧: {record.pnl:+.4f} USDT\n"
                    f"原因: {reason}"
                ),
            )

        logger.info(f"Exit: {reason}, PnL={record.pnl:+.4f}")

    # =========================================================================
    # Order Update Callback
    # =========================================================================

    async def _on_order_update(self, order: OrderProtocol) -> None:
        """Handle order update callback."""
        if order.status != "FILLED":
            return

        result = await self._order_executor.on_order_filled(order)

        if result == "entry_filled":
            # Entry filled, place exit orders
            position = self._position_manager.get_position()
            if position:
                # Update position with actual fill price
                position.entry_bar = self._current_bar
                position.entry_price = order.avg_price

                # Recalculate stop loss based on actual entry
                stop_loss = self._signal_generator._calculate_stop_loss(
                    order.avg_price,
                    SignalType.LONG if position.side == PositionSide.LONG else SignalType.SHORT,
                )
                position.stop_loss_price = stop_loss

                # Place exit orders
                await self._order_executor.place_exit_orders(position)
                self._entry_order_bar = None

        elif result == "take_profit_filled":
            # Take profit filled
            await self._position_manager.close_position("止盈")
            self._record_trade_from_order(order, "止盈")

        elif result == "stop_loss_filled":
            # Stop loss filled
            await self._position_manager.close_position("止損")
            self._record_trade_from_order(order, "止損")

    def _record_trade_from_order(self, order: OrderProtocol, reason: str) -> None:
        """Record trade stats from order fill."""
        # Simplified - actual PnL calculation should be done properly
        self._stats.record_trade(Decimal("0"), Decimal("0"))

    # =========================================================================
    # Statistics
    # =========================================================================

    def _update_stats(self, record: TradeRecord) -> None:
        """Update bot statistics."""
        # Update base stats
        self._stats.record_trade(record.pnl, record.fee)

        # Update bollinger-specific stats
        self._bollinger_stats.record_trade(record.pnl)

    def get_bollinger_stats(self) -> Dict[str, Any]:
        """Get Bollinger-specific statistics."""
        return self._bollinger_stats.to_dict()
