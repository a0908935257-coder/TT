"""
Technical Indicators Calculator.

Provides static methods for calculating common technical indicators
from K-line (candlestick) data.
"""

from decimal import Decimal
from typing import Optional

from src.core.models import Kline


class TechnicalIndicators:
    """
    Technical indicators calculator.

    All methods are static and work with lists of Kline objects.
    Returns Decimal values for precision in financial calculations.

    Example:
        >>> klines = await kline_manager.get_klines("BTCUSDT", "1h", 100)
        >>> sma_20 = TechnicalIndicators.sma(klines, 20)
        >>> rsi = TechnicalIndicators.rsi(klines, 14)
    """

    # =========================================================================
    # Moving Averages
    # =========================================================================

    @staticmethod
    def sma(klines: list[Kline], period: int) -> Optional[Decimal]:
        """
        Calculate Simple Moving Average.

        Args:
            klines: List of Kline objects (most recent last)
            period: Number of periods for SMA

        Returns:
            SMA value or None if insufficient data
        """
        if len(klines) < period:
            return None

        closes = [k.close for k in klines[-period:]]
        return sum(closes) / Decimal(period)

    @staticmethod
    def sma_series(klines: list[Kline], period: int) -> list[Optional[Decimal]]:
        """
        Calculate SMA series for all klines.

        Args:
            klines: List of Kline objects
            period: Number of periods for SMA

        Returns:
            List of SMA values (None for insufficient data points)
        """
        result = []
        for i in range(len(klines)):
            if i < period - 1:
                result.append(None)
            else:
                closes = [k.close for k in klines[i - period + 1 : i + 1]]
                result.append(sum(closes) / Decimal(period))
        return result

    @staticmethod
    def ema(klines: list[Kline], period: int) -> Optional[Decimal]:
        """
        Calculate Exponential Moving Average.

        Args:
            klines: List of Kline objects (most recent last)
            period: Number of periods for EMA

        Returns:
            EMA value or None if insufficient data
        """
        if len(klines) < period:
            return None

        multiplier = Decimal(2) / (Decimal(period) + Decimal(1))

        # Initialize EMA with SMA
        closes = [k.close for k in klines[:period]]
        ema_value = sum(closes) / Decimal(period)

        # Calculate EMA for remaining periods
        for k in klines[period:]:
            ema_value = (k.close - ema_value) * multiplier + ema_value

        return ema_value

    @staticmethod
    def ema_series(klines: list[Kline], period: int) -> list[Optional[Decimal]]:
        """
        Calculate EMA series for all klines.

        Args:
            klines: List of Kline objects
            period: Number of periods for EMA

        Returns:
            List of EMA values (None for insufficient data points)
        """
        if len(klines) < period:
            return [None] * len(klines)

        result: list[Optional[Decimal]] = [None] * (period - 1)
        multiplier = Decimal(2) / (Decimal(period) + Decimal(1))

        # Initialize with SMA
        closes = [k.close for k in klines[:period]]
        ema_value = sum(closes) / Decimal(period)
        result.append(ema_value)

        # Calculate EMA for remaining periods
        for k in klines[period:]:
            ema_value = (k.close - ema_value) * multiplier + ema_value
            result.append(ema_value)

        return result

    # =========================================================================
    # Volatility Indicators
    # =========================================================================

    @staticmethod
    def atr(klines: list[Kline], period: int = 14) -> Optional[Decimal]:
        """
        Calculate Average True Range.

        TR = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        ATR = EMA(TR, period)

        Args:
            klines: List of Kline objects (most recent last)
            period: Number of periods for ATR (default 14)

        Returns:
            ATR value or None if insufficient data
        """
        if len(klines) < period + 1:
            return None

        # Calculate True Range series
        tr_values: list[Decimal] = []

        for i in range(1, len(klines)):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            tr_values.append(tr)

        # Calculate ATR using EMA of True Range
        if len(tr_values) < period:
            return None

        multiplier = Decimal(2) / (Decimal(period) + Decimal(1))

        # Initialize with SMA
        atr_value = sum(tr_values[:period]) / Decimal(period)

        # Calculate EMA for remaining periods
        for tr in tr_values[period:]:
            atr_value = (tr - atr_value) * multiplier + atr_value

        return atr_value

    @staticmethod
    def atr_series(klines: list[Kline], period: int = 14) -> list[Optional[Decimal]]:
        """
        Calculate ATR series for all klines.

        Args:
            klines: List of Kline objects
            period: Number of periods for ATR

        Returns:
            List of ATR values (None for insufficient data points)
        """
        if len(klines) < 2:
            return [None] * len(klines)

        # Calculate True Range series
        tr_values: list[Decimal] = []
        for i in range(1, len(klines)):
            high = klines[i].high
            low = klines[i].low
            prev_close = klines[i - 1].close

            tr = max(
                high - low,
                abs(high - prev_close),
                abs(low - prev_close),
            )
            tr_values.append(tr)

        # First kline has no TR
        result: list[Optional[Decimal]] = [None]

        if len(tr_values) < period:
            return result + [None] * len(tr_values)

        # Pad with None until we have enough data
        result.extend([None] * (period - 1))

        # Initialize ATR with SMA
        multiplier = Decimal(2) / (Decimal(period) + Decimal(1))
        atr_value = sum(tr_values[:period]) / Decimal(period)
        result.append(atr_value)

        # Calculate EMA for remaining periods
        for tr in tr_values[period:]:
            atr_value = (tr - atr_value) * multiplier + atr_value
            result.append(atr_value)

        return result

    @staticmethod
    def bollinger_bands(
        klines: list[Kline],
        period: int = 20,
        std: int = 2,
    ) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """
        Calculate Bollinger Bands.

        Args:
            klines: List of Kline objects (most recent last)
            period: Number of periods for SMA (default 20)
            std: Number of standard deviations (default 2)

        Returns:
            Tuple of (upper_band, middle_band, lower_band) or None
        """
        if len(klines) < period:
            return None

        closes = [k.close for k in klines[-period:]]
        middle = sum(closes) / Decimal(period)

        # Calculate standard deviation
        variance = sum((c - middle) ** 2 for c in closes) / Decimal(period)
        std_dev = variance.sqrt()

        upper = middle + (Decimal(std) * std_dev)
        lower = middle - (Decimal(std) * std_dev)

        return (upper, middle, lower)

    @staticmethod
    def bollinger_bands_series(
        klines: list[Kline],
        period: int = 20,
        std: int = 2,
    ) -> list[Optional[tuple[Decimal, Decimal, Decimal]]]:
        """
        Calculate Bollinger Bands series.

        Args:
            klines: List of Kline objects
            period: Number of periods for SMA
            std: Number of standard deviations

        Returns:
            List of (upper, middle, lower) tuples
        """
        result: list[Optional[tuple[Decimal, Decimal, Decimal]]] = []

        for i in range(len(klines)):
            if i < period - 1:
                result.append(None)
            else:
                closes = [k.close for k in klines[i - period + 1 : i + 1]]
                middle = sum(closes) / Decimal(period)

                variance = sum((c - middle) ** 2 for c in closes) / Decimal(period)
                std_dev = variance.sqrt()

                upper = middle + (Decimal(std) * std_dev)
                lower = middle - (Decimal(std) * std_dev)

                result.append((upper, middle, lower))

        return result

    # =========================================================================
    # Momentum Indicators
    # =========================================================================

    @staticmethod
    def rsi(klines: list[Kline], period: int = 14) -> Optional[Decimal]:
        """
        Calculate Relative Strength Index.

        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss

        Args:
            klines: List of Kline objects (most recent last)
            period: Number of periods for RSI (default 14)

        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(klines) < period + 1:
            return None

        # Calculate price changes
        changes: list[Decimal] = []
        for i in range(1, len(klines)):
            changes.append(klines[i].close - klines[i - 1].close)

        # Separate gains and losses
        gains = [c if c > 0 else Decimal(0) for c in changes]
        losses = [abs(c) if c < 0 else Decimal(0) for c in changes]

        # Calculate initial average gain/loss
        avg_gain = sum(gains[:period]) / Decimal(period)
        avg_loss = sum(losses[:period]) / Decimal(period)

        # Smooth averages for remaining periods
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * Decimal(period - 1) + gains[i]) / Decimal(period)
            avg_loss = (avg_loss * Decimal(period - 1) + losses[i]) / Decimal(period)

        # Calculate RSI
        if avg_loss == 0:
            # No losses: RSI is 100 if there are gains, 50 if no movement
            return Decimal(100) if avg_gain > 0 else Decimal(50)

        rs = avg_gain / avg_loss
        rsi_value = Decimal(100) - (Decimal(100) / (Decimal(1) + rs))

        return rsi_value

    @staticmethod
    def rsi_series(klines: list[Kline], period: int = 14) -> list[Optional[Decimal]]:
        """
        Calculate RSI series for all klines.

        Args:
            klines: List of Kline objects
            period: Number of periods for RSI

        Returns:
            List of RSI values (None for insufficient data points)
        """
        if len(klines) < period + 1:
            return [None] * len(klines)

        # Calculate price changes
        changes: list[Decimal] = []
        for i in range(1, len(klines)):
            changes.append(klines[i].close - klines[i - 1].close)

        # Separate gains and losses
        gains = [c if c > 0 else Decimal(0) for c in changes]
        losses = [abs(c) if c < 0 else Decimal(0) for c in changes]

        result: list[Optional[Decimal]] = [None] * period

        # Calculate initial average gain/loss
        avg_gain = sum(gains[:period]) / Decimal(period)
        avg_loss = sum(losses[:period]) / Decimal(period)

        # First RSI value
        if avg_loss == 0:
            # No losses: RSI is 100 if there are gains, 50 if no movement
            result.append(Decimal(100) if avg_gain > 0 else Decimal(50))
        else:
            rs = avg_gain / avg_loss
            result.append(Decimal(100) - (Decimal(100) / (Decimal(1) + rs)))

        # Calculate remaining RSI values
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * Decimal(period - 1) + gains[i]) / Decimal(period)
            avg_loss = (avg_loss * Decimal(period - 1) + losses[i]) / Decimal(period)

            if avg_loss == 0:
                # No losses: RSI is 100 if there are gains, 50 if no movement
                result.append(Decimal(100) if avg_gain > 0 else Decimal(50))
            else:
                rs = avg_gain / avg_loss
                result.append(Decimal(100) - (Decimal(100) / (Decimal(1) + rs)))

        return result

    @staticmethod
    def macd(
        klines: list[Kline],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> Optional[tuple[Decimal, Decimal, Decimal]]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD Line = EMA(fast) - EMA(slow)
        Signal Line = EMA(MACD Line, signal)
        Histogram = MACD Line - Signal Line

        Args:
            klines: List of Kline objects (most recent last)
            fast: Fast EMA period (default 12)
            slow: Slow EMA period (default 26)
            signal: Signal line period (default 9)

        Returns:
            Tuple of (macd_line, signal_line, histogram) or None
        """
        if len(klines) < slow + signal:
            return None

        # Calculate fast and slow EMA series
        fast_ema_values = TechnicalIndicators.ema_series(klines, fast)
        slow_ema_values = TechnicalIndicators.ema_series(klines, slow)

        # Calculate MACD line series
        macd_values: list[Decimal] = []
        for i in range(len(klines)):
            if fast_ema_values[i] is not None and slow_ema_values[i] is not None:
                macd_values.append(fast_ema_values[i] - slow_ema_values[i])

        if len(macd_values) < signal:
            return None

        # Calculate signal line (EMA of MACD)
        multiplier = Decimal(2) / (Decimal(signal) + Decimal(1))
        signal_value = sum(macd_values[:signal]) / Decimal(signal)

        for mv in macd_values[signal:]:
            signal_value = (mv - signal_value) * multiplier + signal_value

        macd_line = macd_values[-1]
        histogram = macd_line - signal_value

        return (macd_line, signal_value, histogram)

    @staticmethod
    def macd_series(
        klines: list[Kline],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> list[Optional[tuple[Decimal, Decimal, Decimal]]]:
        """
        Calculate MACD series for all klines.

        Args:
            klines: List of Kline objects
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period

        Returns:
            List of (macd_line, signal_line, histogram) tuples
        """
        result: list[Optional[tuple[Decimal, Decimal, Decimal]]] = []

        # Calculate fast and slow EMA series
        fast_ema_values = TechnicalIndicators.ema_series(klines, fast)
        slow_ema_values = TechnicalIndicators.ema_series(klines, slow)

        # Calculate MACD line series
        macd_values: list[Optional[Decimal]] = []
        for i in range(len(klines)):
            if fast_ema_values[i] is not None and slow_ema_values[i] is not None:
                macd_values.append(fast_ema_values[i] - slow_ema_values[i])
            else:
                macd_values.append(None)

        # Calculate signal line series (EMA of MACD)
        multiplier = Decimal(2) / (Decimal(signal) + Decimal(1))
        signal_value: Optional[Decimal] = None
        signal_init_idx: Optional[int] = None

        # Find where we have enough MACD values for signal
        non_none_count = 0
        for i, mv in enumerate(macd_values):
            if mv is not None:
                non_none_count += 1
                if non_none_count == signal:
                    signal_init_idx = i
                    break

        for i in range(len(klines)):
            if signal_init_idx is None or i < signal_init_idx:
                result.append(None)
            elif i == signal_init_idx:
                # Initialize signal with SMA of MACD
                valid_macd = [
                    m for m in macd_values[: i + 1] if m is not None
                ][-signal:]
                signal_value = sum(valid_macd) / Decimal(signal)
                macd_line = macd_values[i]
                histogram = macd_line - signal_value
                result.append((macd_line, signal_value, histogram))
            else:
                # Continue EMA calculation
                if macd_values[i] is not None and signal_value is not None:
                    signal_value = (
                        macd_values[i] - signal_value
                    ) * multiplier + signal_value
                    macd_line = macd_values[i]
                    histogram = macd_line - signal_value
                    result.append((macd_line, signal_value, histogram))
                else:
                    result.append(None)

        return result
