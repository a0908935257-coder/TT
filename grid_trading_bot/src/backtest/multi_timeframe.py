"""
Multi-Timeframe Backtest Module.

Provides support for strategies that use multiple timeframes.

Features:
- Automatic timeframe resampling (1m -> 5m -> 15m -> 1h -> 4h -> 1d)
- Synchronized access to all timeframes
- Proper alignment of higher timeframe bars
- Warmup handling for each timeframe

Example:
    # Strategy using 1h for entries and 4h for trend
    class MultiTFStrategy(MultiTimeframeStrategy):
        def timeframes(self) -> list[str]:
            return ["1h", "4h"]

        def on_kline(self, kline, context):
            # Access 4h data for trend
            tf_4h = context.get_timeframe("4h")
            trend_sma = self._calculate_sma(tf_4h.get_closes(20), 20)

            # Access 1h data for entry
            tf_1h = context.get_timeframe("1h")
            if trend_sma and kline.close > trend_sma:
                return Signal.long_entry(...)
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Optional

from ..core.models import Kline
from .order import Signal
from .position import Position
from .result import Trade
from .strategy.base import BacktestContext, BacktestStrategy


class Timeframe(str, Enum):
    """Supported timeframes."""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H2 = "2h"
    H4 = "4h"
    H8 = "8h"
    H12 = "12h"
    D1 = "1d"
    W1 = "1w"


# Timeframe in minutes
TIMEFRAME_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "12h": 720,
    "1d": 1440,
    "1w": 10080,
}


def get_timeframe_minutes(tf: str) -> int:
    """Get timeframe duration in minutes."""
    return TIMEFRAME_MINUTES.get(tf, 60)


def can_resample(source_tf: str, target_tf: str) -> bool:
    """Check if source timeframe can be resampled to target."""
    source_mins = get_timeframe_minutes(source_tf)
    target_mins = get_timeframe_minutes(target_tf)
    return target_mins > source_mins and target_mins % source_mins == 0


@dataclass
class TimeframeData:
    """
    Data container for a single timeframe.

    Provides access to klines and helper methods for a specific timeframe.
    """

    timeframe: str
    klines: list[Kline] = field(default_factory=list)
    bar_index: int = 0

    @property
    def current_kline(self) -> Optional[Kline]:
        """Get the current (most recent complete) kline."""
        if not self.klines or self.bar_index < 0:
            return None
        idx = min(self.bar_index, len(self.klines) - 1)
        return self.klines[idx] if idx >= 0 else None

    @property
    def current_price(self) -> Optional[Decimal]:
        """Get the current close price."""
        kline = self.current_kline
        return kline.close if kline else None

    def get_klines_window(self, lookback: int) -> list[Kline]:
        """Get a window of recent klines."""
        if not self.klines:
            return []
        end = min(self.bar_index + 1, len(self.klines))
        start = max(0, end - lookback)
        return self.klines[start:end]

    def get_closes(self, lookback: int) -> list[Decimal]:
        """Get recent close prices."""
        return [k.close for k in self.get_klines_window(lookback)]

    def get_highs(self, lookback: int) -> list[Decimal]:
        """Get recent high prices."""
        return [k.high for k in self.get_klines_window(lookback)]

    def get_lows(self, lookback: int) -> list[Decimal]:
        """Get recent low prices."""
        return [k.low for k in self.get_klines_window(lookback)]

    def get_volumes(self, lookback: int) -> list[Decimal]:
        """Get recent volumes."""
        return [k.volume for k in self.get_klines_window(lookback)]

    @property
    def bars_available(self) -> int:
        """Number of bars available."""
        return min(self.bar_index + 1, len(self.klines))


@dataclass
class MultiTimeframeContext(BacktestContext):
    """
    Extended context with multi-timeframe support.

    Provides access to data from multiple synchronized timeframes.

    Attributes:
        timeframe_data: Dictionary of TimeframeData by timeframe string
        base_timeframe: The primary (lowest) timeframe
    """

    timeframe_data: dict[str, TimeframeData] = field(default_factory=dict)
    base_timeframe: str = "1h"

    def get_timeframe(self, tf: str) -> Optional[TimeframeData]:
        """
        Get data for a specific timeframe.

        Args:
            tf: Timeframe string (e.g., "4h", "1d")

        Returns:
            TimeframeData for the timeframe, or None if not available
        """
        return self.timeframe_data.get(tf)

    @property
    def available_timeframes(self) -> list[str]:
        """List of available timeframes."""
        return list(self.timeframe_data.keys())

    def has_timeframe(self, tf: str) -> bool:
        """Check if a timeframe is available."""
        return tf in self.timeframe_data


class TimeframeResampler:
    """
    Resamples klines from lower to higher timeframes.

    Aggregates OHLCV data properly:
    - Open: First bar's open
    - High: Maximum high
    - Low: Minimum low
    - Close: Last bar's close
    - Volume: Sum of volumes
    """

    def __init__(self, source_timeframe: str):
        """
        Initialize resampler.

        Args:
            source_timeframe: The source (lowest) timeframe
        """
        self._source_tf = source_timeframe
        self._source_minutes = get_timeframe_minutes(source_timeframe)

    def resample(
        self,
        klines: list[Kline],
        target_timeframe: str,
    ) -> list[Kline]:
        """
        Resample klines to a higher timeframe.

        Args:
            klines: Source klines (lower timeframe)
            target_timeframe: Target timeframe to resample to

        Returns:
            List of resampled klines
        """
        if not klines:
            return []

        target_minutes = get_timeframe_minutes(target_timeframe)

        if target_minutes <= self._source_minutes:
            # Can't resample to same or lower timeframe
            return klines.copy()

        if target_minutes % self._source_minutes != 0:
            raise ValueError(
                f"Cannot resample {self._source_tf} to {target_timeframe}: "
                f"not evenly divisible"
            )

        bars_per_candle = target_minutes // self._source_minutes
        resampled = []

        # Group klines by target timeframe periods
        i = 0
        while i < len(klines):
            # Find the start of the period
            period_start = self._get_period_start(
                klines[i].open_time, target_minutes
            )

            # Collect all bars in this period
            period_bars = []
            while i < len(klines):
                bar_period = self._get_period_start(
                    klines[i].open_time, target_minutes
                )
                if bar_period != period_start:
                    break
                period_bars.append(klines[i])
                i += 1

            if period_bars:
                # Aggregate the bars
                resampled_kline = self._aggregate_bars(
                    period_bars, target_timeframe
                )
                resampled.append(resampled_kline)

        return resampled

    def _get_period_start(self, dt: datetime, period_minutes: int) -> datetime:
        """Get the start of the period containing the datetime."""
        # Convert to minutes since midnight
        minutes_since_midnight = dt.hour * 60 + dt.minute

        # For daily and weekly, align to day start
        if period_minutes >= 1440:  # 1 day or more
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)

        # Find period start
        period_start_minutes = (minutes_since_midnight // period_minutes) * period_minutes
        return dt.replace(
            hour=period_start_minutes // 60,
            minute=period_start_minutes % 60,
            second=0,
            microsecond=0,
        )

    def _aggregate_bars(
        self,
        bars: list[Kline],
        target_tf: str
    ) -> Kline:
        """Aggregate multiple bars into one."""
        if not bars:
            raise ValueError("Cannot aggregate empty bar list")

        return Kline(
            symbol=bars[0].symbol,
            interval=target_tf,
            open_time=bars[0].open_time,
            close_time=bars[-1].close_time,
            open=bars[0].open,
            high=max(b.high for b in bars),
            low=min(b.low for b in bars),
            close=bars[-1].close,
            volume=sum(b.volume for b in bars),
            quote_volume=sum(b.quote_volume for b in bars),
            trades_count=sum(b.trades_count for b in bars),
            is_closed=bars[-1].is_closed,
        )


class MultiTimeframeStrategy(BacktestStrategy):
    """
    Abstract base class for multi-timeframe strategies.

    Extends BacktestStrategy with multi-timeframe capabilities.
    Override `timeframes()` to specify which timeframes the strategy needs.

    Example:
        class TrendFollowingMTF(MultiTimeframeStrategy):
            def timeframes(self) -> list[str]:
                return ["1h", "4h", "1d"]

            def warmup_periods(self) -> dict[str, int]:
                return {"1h": 20, "4h": 20, "1d": 20}

            def on_kline(self, kline, context):
                # Get daily trend
                daily = context.get_timeframe("1d")
                daily_sma = self._sma(daily.get_closes(20), 20)

                # Get 4h momentum
                h4 = context.get_timeframe("4h")

                # Use 1h for entry timing
                if daily_sma and kline.close > daily_sma:
                    return Signal.long_entry(...)
    """

    @abstractmethod
    def timeframes(self) -> list[str]:
        """
        Return list of timeframes needed by this strategy.

        The first timeframe is the base (primary) timeframe.
        Higher timeframes will be resampled from the base.

        Returns:
            List of timeframe strings (e.g., ["1h", "4h", "1d"])
        """
        pass

    def warmup_periods(self) -> dict[str, int]:
        """
        Return warmup periods for each timeframe.

        Override to specify different warmup periods for each timeframe.
        Default is the same as `warmup_period()` for all timeframes.

        Returns:
            Dictionary of timeframe to warmup period
        """
        default = self.warmup_period()
        return {tf: default for tf in self.timeframes()}

    def warmup_period(self) -> int:
        """Return the warmup period for the base timeframe."""
        return 50

    @abstractmethod
    def on_kline(
        self,
        kline: Kline,
        context: MultiTimeframeContext
    ) -> Optional[Signal]:
        """
        Process a kline with multi-timeframe context.

        Args:
            kline: Current kline (base timeframe)
            context: Multi-timeframe context with all timeframe data

        Returns:
            Signal if a trade should be made, None otherwise
        """
        pass


class MultiTimeframeEngine:
    """
    Backtest engine with multi-timeframe support.

    Handles:
    - Automatic timeframe resampling
    - Synchronized bar updates across timeframes
    - Proper warmup for each timeframe

    Example:
        engine = MultiTimeframeEngine(config)
        result = engine.run(klines_1h, mtf_strategy)
    """

    def __init__(self, config) -> None:
        """
        Initialize multi-timeframe engine.

        Args:
            config: Backtest configuration
        """
        from .engine import BacktestEngine

        self._config = config
        self._base_engine = BacktestEngine(config)
        self._resampler: Optional[TimeframeResampler] = None
        self._timeframe_klines: dict[str, list[Kline]] = {}
        self._timeframe_bar_indices: dict[str, int] = {}

    def run(
        self,
        klines: list[Kline],
        strategy: MultiTimeframeStrategy,
    ):
        """
        Execute multi-timeframe backtest.

        Args:
            klines: Base timeframe klines
            strategy: Multi-timeframe strategy

        Returns:
            BacktestResult with all metrics
        """
        from .result import BacktestResult
        from .position import PositionManager
        from .order import OrderSimulator
        from .metrics import MetricsCalculator

        if not klines:
            return BacktestResult()

        # Get required timeframes
        timeframes = strategy.timeframes()
        if not timeframes:
            raise ValueError("Strategy must specify at least one timeframe")

        base_tf = timeframes[0]

        # Detect source timeframe from klines
        source_tf = klines[0].interval if klines else base_tf

        # Initialize resampler
        self._resampler = TimeframeResampler(source_tf)

        # Resample klines for all timeframes
        self._prepare_timeframe_data(klines, timeframes, source_tf)

        # Calculate total warmup needed (in base timeframe bars)
        warmup_periods = strategy.warmup_periods()
        max_warmup = self._calculate_max_warmup(
            warmup_periods, timeframes, source_tf
        )

        if len(klines) <= max_warmup:
            return BacktestResult()

        # Initialize engine components
        position_manager = PositionManager(max_positions=self._config.max_positions)
        order_simulator = OrderSimulator(self._config)
        metrics = MetricsCalculator(initial_capital=self._config.initial_capital)

        equity_curve = []
        strategy.reset()

        # Process each bar
        for bar_idx in range(max_warmup, len(klines)):
            kline = klines[bar_idx]

            # Update timeframe bar indices
            self._update_timeframe_indices(kline, timeframes)

            # Create multi-timeframe context
            context = self._create_mtf_context(
                bar_idx, klines, position_manager, base_tf
            )

            # Check exits for existing positions
            position = position_manager.current_position
            if position:
                exit_signal = strategy.check_exit(position, kline, context)
                if exit_signal:
                    self._process_exit(
                        position, kline, bar_idx,
                        position_manager, order_simulator
                    )

            # Update context after potential exit
            context = self._create_mtf_context(
                bar_idx, klines, position_manager, base_tf
            )

            # Get strategy signal
            signal = strategy.on_kline(kline, context)

            # Process entry signal
            if signal and not position_manager.has_position:
                self._process_entry(
                    signal, kline, bar_idx,
                    position_manager, order_simulator
                )

            # Update equity
            current_price = kline.close
            equity = position_manager.total_equity(
                current_price,
                self._config.initial_capital,
                self._config.leverage
            )
            equity_curve.append(equity)

            # Callback
            strategy.on_bar_close(kline, context)

        # Close remaining positions
        if position_manager.has_position:
            final_kline = klines[-1]
            self._close_remaining_positions(
                position_manager, final_kline, len(klines) - 1, order_simulator
            )

        # Calculate metrics
        trades = position_manager.trades

        # Build daily returns from equity curve and klines
        daily_pnl: dict[str, Decimal] = {}
        prev_equity = self._config.initial_capital
        for i, (eq, kline) in enumerate(zip(equity_curve, klines[max_warmup:])):
            date_str = kline.close_time.strftime("%Y-%m-%d")
            daily_pnl[date_str] = daily_pnl.get(date_str, Decimal("0")) + (eq - prev_equity)
            prev_equity = eq

        return metrics.calculate_all(trades, equity_curve, daily_pnl)

    def _prepare_timeframe_data(
        self,
        klines: list[Kline],
        timeframes: list[str],
        source_tf: str,
    ) -> None:
        """Prepare klines for all timeframes."""
        self._timeframe_klines = {}
        self._timeframe_bar_indices = {}

        for tf in timeframes:
            if tf == source_tf:
                self._timeframe_klines[tf] = klines
            else:
                self._timeframe_klines[tf] = self._resampler.resample(klines, tf)
            self._timeframe_bar_indices[tf] = 0

    def _calculate_max_warmup(
        self,
        warmup_periods: dict[str, int],
        timeframes: list[str],
        source_tf: str,
    ) -> int:
        """Calculate maximum warmup in base timeframe bars."""
        source_minutes = get_timeframe_minutes(source_tf)
        max_warmup = 0

        for tf in timeframes:
            tf_warmup = warmup_periods.get(tf, 50)
            tf_minutes = get_timeframe_minutes(tf)

            # Convert to base timeframe bars
            bars_per_tf = tf_minutes // source_minutes
            warmup_in_base = tf_warmup * bars_per_tf

            max_warmup = max(max_warmup, warmup_in_base)

        return max_warmup

    def _update_timeframe_indices(
        self,
        current_kline: Kline,
        timeframes: list[str]
    ) -> None:
        """Update bar indices for all timeframes."""
        current_time = current_kline.close_time

        for tf in timeframes:
            tf_klines = self._timeframe_klines.get(tf, [])
            if not tf_klines:
                continue

            # Find the latest completed bar for this timeframe
            for i, k in enumerate(tf_klines):
                if k.close_time <= current_time:
                    self._timeframe_bar_indices[tf] = i
                else:
                    break

    def _create_mtf_context(
        self,
        bar_idx: int,
        klines: list[Kline],
        position_manager,
        base_tf: str,
    ) -> MultiTimeframeContext:
        """Create multi-timeframe context."""
        timeframe_data = {}

        for tf, tf_klines in self._timeframe_klines.items():
            bar_index = self._timeframe_bar_indices.get(tf, 0)
            timeframe_data[tf] = TimeframeData(
                timeframe=tf,
                klines=tf_klines,
                bar_index=bar_index,
            )

        current_price = klines[bar_idx].close

        return MultiTimeframeContext(
            bar_index=bar_idx,
            klines=klines[:bar_idx + 1],
            current_position=position_manager.current_position,
            realized_pnl=position_manager.realized_pnl,
            equity=position_manager.total_equity(
                current_price,
                self._config.initial_capital,
                self._config.leverage
            ),
            initial_capital=self._config.initial_capital,
            timeframe_data=timeframe_data,
            base_timeframe=base_tf,
        )

    def _process_entry(
        self,
        signal,
        kline: Kline,
        bar_idx: int,
        position_manager,
        order_simulator,
    ) -> None:
        """Process entry signal."""
        from .order import SignalType
        from .position import Position

        if signal.signal_type in (SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY):
            side = "LONG" if signal.signal_type == SignalType.LONG_ENTRY else "SHORT"
            is_buy = side == "LONG"
            entry_price = signal.price if signal.price else kline.close

            # Apply slippage using calculate_fill_price
            entry_price = order_simulator.calculate_fill_price(
                target_price=entry_price,
                is_buy=is_buy,
                kline=kline,
            )

            # Calculate quantity based on position sizing
            position_size_pct = self._config.position_size_pct
            available_capital = self._config.initial_capital * position_size_pct
            notional = available_capital * Decimal(self._config.leverage)
            quantity = notional / entry_price

            # Calculate entry fee
            entry_fee = order_simulator.fee_rate * notional

            # Create position object
            position = Position(
                side=side,
                entry_price=entry_price,
                quantity=quantity,
                entry_time=kline.close_time,
                entry_bar=bar_idx,
                entry_fee=entry_fee,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            position_manager.open_position(position)

    def _process_exit(
        self,
        position,
        kline: Kline,
        bar_idx: int,
        position_manager,
        order_simulator,
    ) -> None:
        """Process exit."""
        from .result import ExitReason

        # Exit is opposite of entry: sell for long, buy for short
        is_buy = position.side == "SHORT"
        exit_price = order_simulator.calculate_fill_price(
            target_price=kline.close,
            is_buy=is_buy,
            kline=kline,
        )

        # Calculate exit fee
        notional = position.quantity * exit_price
        exit_fee = order_simulator.fee_rate * notional

        position_manager.close_position(
            position=position,
            exit_price=exit_price,
            exit_time=kline.close_time,
            exit_bar=bar_idx,
            exit_fee=exit_fee,
            exit_reason=ExitReason.SIGNAL,
            leverage=self._config.leverage,
        )

    def _close_remaining_positions(
        self,
        position_manager,
        kline: Kline,
        bar_idx: int,
        order_simulator,
    ) -> None:
        """Close any remaining positions at end of backtest."""
        from .result import ExitReason

        while position_manager.has_position:
            position = position_manager.current_position
            is_buy = position.side == "SHORT"
            exit_price = order_simulator.calculate_fill_price(
                target_price=kline.close,
                is_buy=is_buy,
                kline=kline,
            )

            # Calculate exit fee
            notional = position.quantity * exit_price
            exit_fee = order_simulator.fee_rate * notional

            position_manager.close_position(
                position=position,
                exit_price=exit_price,
                exit_time=kline.close_time,
                exit_bar=bar_idx,
                exit_fee=exit_fee,
                exit_reason=ExitReason.SIGNAL,  # End of data
                leverage=self._config.leverage,
            )
