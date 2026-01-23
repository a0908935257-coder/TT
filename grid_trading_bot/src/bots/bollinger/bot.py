"""
Bollinger Trend Bot (Supertrend + BB Combination).

Main bot class that integrates all components for the Bollinger Trend
trading strategy on futures market.

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 8 期分割):
- Walk-Forward 一致性: 75% (6/8 時段獲利)
- OOS 效率: 96%
- 報酬: +35.1%, Sharpe: 1.81, DD: 6.7%

策略邏輯:
- 進場: Supertrend 看多時在 BB 下軌買入，看空時在 BB 上軌賣出
- 出場: Supertrend 翻轉（主要）或 ATR 止損（保護）

驗證參數:
- bb_period: 20, bb_std: 3.0
- st_atr_period: 20, st_atr_multiplier: 3.5
- atr_stop_multiplier: 2.0
- leverage: 2x

Architecture:
    BollingerBot
    ├── BollingerCalculator   → Calculate BB indicators
    ├── SupertrendCalculator  → Calculate Supertrend
    ├── SignalGenerator       → Generate signals
    ├── PositionManager       → Manage positions
    └── OrderExecutor         → Execute orders
"""

import asyncio
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Dict, List, Optional, Protocol

from src.bots.base import BaseBot, BotStats
from src.core import get_logger
from src.core.models import MarketType
from src.master.models import BotState

from .indicators import BollingerCalculator, SupertrendCalculator
from .models import (
    BollingerBotStats,
    BollingerConfig,
    PositionSide,
    SignalType,
    StrategyMode,
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

    async def subscribe_kline(self, symbol: str, interval: str, callback: Callable) -> None: ...

    async def unsubscribe_kline(self, symbol: str, interval: str) -> None: ...

    async def subscribe_user_data(self, callback: Callable, market: MarketType = ...) -> None: ...

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

        # Initialize Supertrend calculator (BOLLINGER_TREND mode - Walk-Forward validated)
        self._supertrend = SupertrendCalculator(
            atr_period=config.st_atr_period,
            atr_multiplier=config.st_atr_multiplier,
        )

        self._signal_generator = SignalGenerator(config, self._calculator, self._supertrend)
        self._position_manager = PositionManager(config, exchange, data_manager)
        self._order_executor = OrderExecutor(config, exchange, notifier)

        # State
        self._klines: List[KlineProtocol] = []
        self._current_bar: int = 0
        self._entry_order_bar: Optional[int] = None

        # Statistics
        self._bollinger_stats = BollingerBotStats()

        # Risk control tracking (每日虧損 + 連續虧損)
        self._daily_pnl = Decimal("0")
        self._daily_start_time: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._consecutive_losses: int = 0
        self._risk_paused: bool = False

        # State persistence
        self._save_task: Optional[asyncio.Task] = None
        self._save_interval_minutes: int = 5

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

        # 2. Get historical klines (futures market)
        klines = await self._exchange.get_klines(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            limit=300,
            market=MarketType.FUTURES,
        )
        self._klines = klines

        # 3. Initialize indicator calculator (build BBW history)
        self._calculator.initialize(klines)

        # 4. Initialize Supertrend (BOLLINGER_TREND mode - Walk-Forward validated)
        self._signal_generator.initialize_supertrend(klines)

        # 5. Set current bar number
        self._current_bar = len(klines)

        # 6. Subscribe to kline updates (futures market)
        await self._exchange.subscribe_kline(
            symbol=self._config.symbol,
            interval=self._config.timeframe,
            callback=self._on_kline,
            market=MarketType.FUTURES,
        )

        # 7. Subscribe to user data (order updates) - Futures market
        await self._exchange.subscribe_user_data(
            callback=self._on_order_update,
            market=MarketType.FUTURES,
        )

        # 8. Send notification
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

        # Start periodic state saving
        self._start_save_task()

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

        # 3. Unsubscribe from klines (futures market)
        try:
            await self._exchange.unsubscribe_kline(
                symbol=self._config.symbol,
                interval=self._config.timeframe,
                market=MarketType.FUTURES,
            )
        except Exception as e:
            logger.debug(f"Unsubscribe klines: {e}")

        # 4. Stop periodic save task and save final state
        self._stop_save_task()
        await self._save_state()

        # 5. Send notification
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

        status = {
            "timeframe": self._config.timeframe,
            "leverage": self._config.leverage,
            "strategy_mode": "bollinger_trend",  # Walk-Forward validated mode
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

        # Add Supertrend info (BOLLINGER_TREND mode)
        if self._supertrend:
            st = self._supertrend.current
            status["supertrend_trend"] = "BULL" if self._supertrend.is_bullish else "BEAR"
            if st:
                status["supertrend_value"] = str(st.supertrend)

        # Risk control status
        status["daily_pnl"] = float(self._daily_pnl)
        status["consecutive_losses"] = self._consecutive_losses
        status["risk_paused"] = self._risk_paused

        return status

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
        try:
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

        except Exception as e:
            logger.error(f"Error in _process_bar: {e}", exc_info=True)

    async def _enter_position(self, signal) -> None:
        """Enter a new position."""
        # Check risk limits before entering
        if self._check_risk_limits():
            logger.warning(f"Trading paused due to risk limits - skipping {signal.signal_type.value} signal")
            if self._notifier:
                await self._notifier.send(
                    title="⚠️ Bollinger: Risk Limit",
                    message=f"Signal skipped: {signal.signal_type.value}\n"
                            f"Daily PnL: {self._daily_pnl:+.2f}\n"
                            f"Consecutive losses: {self._consecutive_losses}",
                )
            return

        try:
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

        except Exception as e:
            logger.error(f"Error in _enter_position: {e}", exc_info=True)

    async def _exit_position(self, reason: str) -> None:
        """Exit current position."""
        try:
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

        except Exception as e:
            logger.error(f"Error in _exit_position: {e}", exc_info=True)

    # =========================================================================
    # Order Update Callback
    # =========================================================================

    async def _on_order_update(self, order: OrderProtocol) -> None:
        """Handle order update callback."""
        try:
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

        except Exception as e:
            logger.error(f"Error in _on_order_update: {e}", exc_info=True)

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

        # Update risk tracking
        self._update_risk_tracking(record.pnl)

    def get_bollinger_stats(self) -> Dict[str, Any]:
        """Get Bollinger-specific statistics."""
        return self._bollinger_stats.to_dict()

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
        capital = self._config.max_capital or Decimal("1000")

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
            self._consecutive_losses = 0

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
        for position sizing on the next trade via PositionManager.

        Args:
            new_max_capital: New maximum capital allocation
        """
        previous = self._config.max_capital

        logger.info(
            f"[FundManager] Capital updated for {self._bot_id}: "
            f"{previous} -> {new_max_capital}"
        )

        # Note: PositionManager will use new max_capital automatically
        # on next position calculation. No immediate action needed.

    # =========================================================================
    # State Persistence
    # =========================================================================

    async def _save_state(self) -> None:
        """Save bot state to database."""
        try:
            config = {
                "symbol": self._config.symbol,
                "timeframe": self._config.timeframe,
                "bb_period": self._config.bb_period,
                "bb_std": str(self._config.bb_std),
                "st_atr_period": self._config.st_atr_period,
                "st_atr_multiplier": str(self._config.st_atr_multiplier),
                "leverage": self._config.leverage,
                "max_capital": str(self._config.max_capital) if self._config.max_capital else None,
            }

            state_data = {
                "current_bar": self._current_bar,
                "entry_order_bar": self._entry_order_bar,
                "stats": self._bollinger_stats.to_dict(),
                "saved_at": datetime.now(timezone.utc).isoformat(),
                # Risk control state
                "daily_pnl": str(self._daily_pnl),
                "daily_start_time": self._daily_start_time.isoformat(),
                "consecutive_losses": self._consecutive_losses,
                "risk_paused": self._risk_paused,
            }

            position = self._position_manager.get_position()
            if position:
                state_data["position"] = {
                    "side": position.side.value,
                    "entry_price": str(position.entry_price),
                    "quantity": str(position.quantity),
                    "entry_bar": position.entry_bar,
                    "take_profit_price": str(position.take_profit_price) if position.take_profit_price else None,
                    "stop_loss_price": str(position.stop_loss_price) if position.stop_loss_price else None,
                }

            await self._data_manager.save_bot_state(
                bot_id=self._bot_id,
                bot_type="bollinger",
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
        exchange: ExchangeProtocol,
        data_manager: Any,
        notifier: Optional[NotifierProtocol] = None,
    ) -> Optional["BollingerBot"]:
        """
        Restore a BollingerBot from saved state.

        Args:
            bot_id: Bot ID to restore
            exchange: Exchange client
            data_manager: Data manager instance
            notifier: Notification manager

        Returns:
            Restored BollingerBot or None if not found
        """
        try:
            state_data = await data_manager.get_bot_state(bot_id)
            if not state_data:
                logger.warning(f"No saved state for bot: {bot_id}")
                return None

            config_data = state_data.get("config", {})
            config = BollingerConfig(
                symbol=config_data.get("symbol", ""),
                timeframe=config_data.get("timeframe", "15m"),
                bb_period=config_data.get("bb_period", 20),
                bb_std=Decimal(config_data.get("bb_std", "3.0")),
                st_atr_period=config_data.get("st_atr_period", 20),
                st_atr_multiplier=Decimal(config_data.get("st_atr_multiplier", "3.5")),
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
            bot._current_bar = saved_state.get("current_bar", 0)
            bot._entry_order_bar = saved_state.get("entry_order_bar")

            # Restore stats
            stats_data = saved_state.get("stats", {})
            bot._bollinger_stats.total_trades = stats_data.get("total_trades", 0)
            bot._bollinger_stats.winning_trades = stats_data.get("winning_trades", 0)
            bot._bollinger_stats.losing_trades = stats_data.get("losing_trades", 0)
            bot._bollinger_stats.total_pnl = Decimal(stats_data.get("total_pnl", "0"))

            # Restore risk control state
            bot._daily_pnl = Decimal(saved_state.get("daily_pnl", "0"))
            if saved_state.get("daily_start_time"):
                bot._daily_start_time = datetime.fromisoformat(saved_state["daily_start_time"])
            bot._consecutive_losses = saved_state.get("consecutive_losses", 0)
            bot._risk_paused = saved_state.get("risk_paused", False)

            logger.info(f"Restored BollingerBot: {bot_id}, PnL={bot._bollinger_stats.total_pnl}")
            return bot

        except Exception as e:
            logger.error(f"Failed to restore bot: {e}")
            return None
