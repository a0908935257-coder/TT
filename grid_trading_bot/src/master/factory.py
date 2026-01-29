"""
Bot Factory for creating bot instances.

Creates and configures trading bot instances based on type and configuration.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any, Optional, Protocol

from src.core import get_logger
from src.master.models import BotType

if TYPE_CHECKING:
    from src.bots.base import BaseBot
    from src.bots.grid.bot import GridBot, GridBotConfig
    from src.data import MarketDataManager
    from src.exchange import ExchangeClient
    from src.notification import NotificationManager

logger = get_logger(__name__)


class ExchangeClientProtocol(Protocol):
    """Protocol for exchange client."""

    async def get_ticker(self, symbol: str) -> dict[str, Any]: ...


class DataManagerProtocol(Protocol):
    """Protocol for data manager."""

    pass


class NotifierProtocol(Protocol):
    """Protocol for notification manager."""

    pass


class BotProtocol(Protocol):
    """Protocol for bot instances."""

    @property
    def bot_id(self) -> str: ...

    async def start(self) -> bool: ...
    async def stop(self, reason: str = "", clear_position: bool = False) -> bool: ...
    async def pause(self, reason: str = "") -> bool: ...
    async def resume(self) -> bool: ...


class UnsupportedBotTypeError(Exception):
    """Raised when trying to create an unsupported bot type."""

    def __init__(self, bot_type: BotType):
        self.bot_type = bot_type
        super().__init__(f"Unsupported bot type: {bot_type.value}")


class InvalidBotConfigError(Exception):
    """Raised when bot configuration is invalid."""

    def __init__(self, message: str):
        super().__init__(message)


class BotFactory:
    """
    Factory for creating bot instances.

    Creates and configures trading bot instances based on type.

    Example:
        >>> factory = BotFactory(exchange, data_manager, notifier)
        >>> bot = factory.create(BotType.GRID, "bot_001", config)
        >>> await bot.start()
    """

    def __init__(
        self,
        exchange: Optional[ExchangeClientProtocol] = None,
        data_manager: Optional[DataManagerProtocol] = None,
        notifier: Optional[NotifierProtocol] = None,
        heartbeat_callback: Optional[callable] = None,
    ):
        """
        Initialize BotFactory.

        Args:
            exchange: ExchangeClient instance
            data_manager: MarketDataManager instance
            notifier: NotificationManager instance
            heartbeat_callback: Optional callback for bot heartbeats
        """
        self._exchange = exchange
        self._data_manager = data_manager
        self._notifier = notifier
        self._heartbeat_callback = heartbeat_callback

        # Registry of bot creators
        self._creators: dict[BotType, callable] = {
            BotType.GRID: self._create_grid_bot,
            BotType.BOLLINGER: self._create_bollinger_bot,
            BotType.SUPERTREND: self._create_supertrend_bot,
            BotType.RSI_GRID: self._create_rsi_grid_bot,
            BotType.GRID_FUTURES: self._create_grid_futures_bot,
            BotType.DCA: self._create_dca_bot,
            BotType.TRAILING_STOP: self._create_trailing_stop_bot,
            BotType.SIGNAL: self._create_signal_bot,
        }

    def create(
        self,
        bot_type: BotType,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a bot instance.

        Args:
            bot_type: Type of bot to create
            bot_id: Unique bot identifier
            config: Bot configuration dictionary

        Returns:
            Bot instance (not started)

        Raises:
            UnsupportedBotTypeError: If bot type is not supported
            InvalidBotConfigError: If configuration is invalid
        """
        # Validate basic config
        self._validate_base_config(config)

        # Get creator function
        creator = self._creators.get(bot_type)
        if not creator:
            raise UnsupportedBotTypeError(bot_type)

        try:
            instance = creator(bot_id, config)
            logger.info(f"Created bot instance: {bot_id} ({bot_type.value})")
            return instance
        except Exception as e:
            logger.error(f"Failed to create bot {bot_id}: {e}")
            raise InvalidBotConfigError(f"Failed to create {bot_type.value} bot: {e}")

    def _validate_base_config(self, config: dict[str, Any]) -> None:
        """
        Validate base configuration fields.

        Args:
            config: Configuration dictionary

        Raises:
            InvalidBotConfigError: If required fields are missing
        """
        required_fields = ["symbol"]
        for field in required_fields:
            if field not in config:
                raise InvalidBotConfigError(f"Missing required field: {field}")

    def _create_grid_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a GridBot instance.

        Args:
            bot_id: Bot identifier
            config: Grid bot configuration

        Returns:
            GridBot instance
        """
        # Import here to avoid circular imports
        from src.bots.grid.bot import GridBot, GridBotConfig
        from src.bots.grid.models import RiskLevel
        from src.core.models import MarketType

        # Handle market_type (convert lowercase to uppercase if needed)
        market_type_str = config.get("market_type", "SPOT")
        if market_type_str.lower() == "spot":
            market_type = MarketType.SPOT
        elif market_type_str.lower() == "futures":
            market_type = MarketType.FUTURES
        else:
            market_type = MarketType(market_type_str)

        # Build GridBotConfig from dict
        grid_config = GridBotConfig(
            symbol=config["symbol"],
            market_type=market_type,
            total_investment=Decimal(str(config.get("total_investment", "1000"))),
            risk_level=RiskLevel(config.get("risk_level", "moderate")),
        )

        # Handle optional fields
        if config.get("manual_upper"):
            grid_config.manual_upper = Decimal(str(config["manual_upper"]))
        if config.get("manual_lower"):
            grid_config.manual_lower = Decimal(str(config["manual_lower"]))
        if config.get("manual_grid_count"):
            grid_config.manual_grid_count = int(config["manual_grid_count"])

        # Create bot instance
        bot = GridBot(
            bot_id=bot_id,
            config=grid_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            heartbeat_callback=self._heartbeat_callback,
        )

        return bot

    def _create_bollinger_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a BollingerBot instance.

        默認值來自 BollingerConfig (實戰配置)，確保一致性。

        Args:
            bot_id: Bot identifier
            config: Bollinger bot configuration

        Returns:
            BollingerBot instance
        """
        # Import here to avoid circular imports
        from decimal import Decimal

        from src.bots.bollinger.bot import BollingerBot
        from src.bots.bollinger.models import BollingerConfig

        # 使用 BollingerConfig 默認值作為備用 (單一來源)
        _defaults = BollingerConfig(symbol=config["symbol"])

        # Build BollingerConfig from dict (未指定的參數使用實戰配置默認值)
        bollinger_config = BollingerConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", _defaults.timeframe),
            # Bollinger Bands
            bb_period=int(config.get("bb_period", _defaults.bb_period)),
            bb_std=Decimal(str(config.get("bb_std", _defaults.bb_std))),
            # Grid parameters
            grid_count=int(config.get("grid_count", _defaults.grid_count)),
            grid_range_pct=Decimal(str(config.get("grid_range_pct", _defaults.grid_range_pct))),
            take_profit_grids=int(config.get("take_profit_grids", _defaults.take_profit_grids)),
            # Position settings
            leverage=int(config.get("leverage", _defaults.leverage)),
            margin_type=config.get("margin_type", _defaults.margin_type),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else _defaults.max_capital,
            position_size_pct=Decimal(str(config.get("position_size_pct", _defaults.position_size_pct))),
            max_position_pct=Decimal(str(config.get("max_position_pct", _defaults.max_position_pct))),
            # Risk management
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", _defaults.stop_loss_pct))),
            rebuild_threshold_pct=Decimal(str(config.get("rebuild_threshold_pct", _defaults.rebuild_threshold_pct))),
            # BBW filter
            bbw_lookback=int(config.get("bbw_lookback", _defaults.bbw_lookback)),
            bbw_threshold_pct=int(config.get("bbw_threshold_pct", _defaults.bbw_threshold_pct)),
            # Protective features
            use_hysteresis=config.get("use_hysteresis", _defaults.use_hysteresis),
            hysteresis_pct=Decimal(str(config.get("hysteresis_pct", _defaults.hysteresis_pct))),
            use_signal_cooldown=config.get("use_signal_cooldown", _defaults.use_signal_cooldown),
            cooldown_bars=int(config.get("cooldown_bars", _defaults.cooldown_bars)),
        )

        # Create bot instance
        bot = BollingerBot(
            bot_id=bot_id,
            config=bollinger_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            heartbeat_callback=self._heartbeat_callback,
        )

        return bot

    def _create_supertrend_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a SupertrendBot instance (TREND_GRID mode).

        Walk-Forward 驗證通過 (2024-01-25 ~ 2026-01-24, 2 年數據):
        - 一致性: 70% (7/10 時段), OOS Sharpe: 5.84
        - 過度擬合: NO, 穩健性: ROBUST
        - 獲利機率: 100%, 勝率: ~94%

        TREND_GRID 模式:
        - 在多頭趨勢中，於網格低點做多
        - 在空頭趨勢中，於網格高點做空
        - RSI 過濾器避免極端進場

        Args:
            bot_id: Bot identifier
            config: Supertrend bot configuration

        Returns:
            SupertrendBot instance
        """
        # Import here to avoid circular imports
        from decimal import Decimal

        from src.bots.supertrend.bot import SupertrendBot
        from src.bots.supertrend.models import SupertrendConfig

        # Build SupertrendConfig from dict (TREND_GRID mode)
        supertrend_config = SupertrendConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "1h"),  # Walk-Forward validated: 1h
            # Supertrend Settings
            atr_period=int(config.get("atr_period", 14)),  # Walk-Forward validated
            atr_multiplier=Decimal(str(config.get("atr_multiplier", "3.0"))),
            leverage=int(config.get("leverage", 2)),  # Walk-Forward validated: 2x
            margin_type=config.get("margin_type", "ISOLATED"),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            # Grid Settings (TREND_GRID 模式)
            grid_count=int(config.get("grid_count", 10)),
            grid_atr_multiplier=Decimal(str(config.get("grid_atr_multiplier", "3.0"))),
            take_profit_grids=int(config.get("take_profit_grids", 1)),
            # RSI Filter (減少假訊號)
            use_rsi_filter=config.get("use_rsi_filter", True),
            rsi_period=int(config.get("rsi_period", 14)),
            rsi_overbought=int(config.get("rsi_overbought", 60)),
            rsi_oversold=int(config.get("rsi_oversold", 40)),
            # Trend confirmation
            min_trend_bars=int(config.get("min_trend_bars", 2)),
            # Risk Management
            use_trailing_stop=config.get("use_trailing_stop", False),
            trailing_stop_pct=Decimal(str(config.get("trailing_stop_pct", "0.02"))),
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", True),
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", "0.05"))),  # Walk-Forward: 5%
            # HYBRID_GRID mode (v3)
            mode=config.get("mode", "hybrid_grid"),
            hybrid_grid_bias_pct=Decimal(str(config.get("hybrid_grid_bias_pct", "0.75"))),
            hybrid_tp_multiplier_trend=Decimal(str(config.get("hybrid_tp_multiplier_trend", "1.25"))),
            hybrid_tp_multiplier_counter=Decimal(str(config.get("hybrid_tp_multiplier_counter", "0.75"))),
            hybrid_sl_multiplier_counter=Decimal(str(config.get("hybrid_sl_multiplier_counter", "0.5"))),
            hybrid_rsi_asymmetric=config.get("hybrid_rsi_asymmetric", True),
        )

        # Create bot instance
        bot = SupertrendBot(
            bot_id=bot_id,
            config=supertrend_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            heartbeat_callback=self._heartbeat_callback,
        )

        return bot

    def _create_grid_futures_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a GridFuturesBot instance.

        默認值來自 GridFuturesConfig (實戰配置)，確保一致性。

        Args:
            bot_id: Bot identifier
            config: Grid Futures bot configuration

        Returns:
            GridFuturesBot instance
        """
        # Import here to avoid circular imports
        from decimal import Decimal

        from src.bots.grid_futures.bot import GridFuturesBot
        from src.bots.grid_futures.models import GridFuturesConfig, GridDirection

        # 使用 GridFuturesConfig 默認值作為備用 (單一來源)
        _defaults = GridFuturesConfig(symbol=config["symbol"])

        # Parse direction
        direction_str = config.get("direction", _defaults.direction.value)
        direction = GridDirection(direction_str)

        # Build GridFuturesConfig from dict (未指定的參數使用實戰配置默認值)
        grid_futures_config = GridFuturesConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", _defaults.timeframe),
            leverage=int(config.get("leverage", _defaults.leverage)),
            margin_type=config.get("margin_type", _defaults.margin_type),
            grid_count=int(config.get("grid_count", _defaults.grid_count)),
            direction=direction,
            use_trend_filter=config.get("use_trend_filter", _defaults.use_trend_filter),
            trend_period=int(config.get("trend_period", _defaults.trend_period)),
            use_atr_range=config.get("use_atr_range", _defaults.use_atr_range),
            atr_period=int(config.get("atr_period", _defaults.atr_period)),
            atr_multiplier=Decimal(str(config.get("atr_multiplier", _defaults.atr_multiplier))),
            fallback_range_pct=Decimal(str(config.get("fallback_range_pct", _defaults.fallback_range_pct))),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else _defaults.max_capital,
            position_size_pct=Decimal(str(config.get("position_size_pct", _defaults.position_size_pct))),
            max_position_pct=Decimal(str(config.get("max_position_pct", _defaults.max_position_pct))),
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", _defaults.stop_loss_pct))),
            rebuild_threshold_pct=Decimal(str(config.get("rebuild_threshold_pct", _defaults.rebuild_threshold_pct))),
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", _defaults.use_exchange_stop_loss),
            # Protective Features
            use_hysteresis=config.get("use_hysteresis", _defaults.use_hysteresis),
            hysteresis_pct=Decimal(str(config.get("hysteresis_pct", _defaults.hysteresis_pct))),
            use_signal_cooldown=config.get("use_signal_cooldown", _defaults.use_signal_cooldown),
            cooldown_bars=int(config.get("cooldown_bars", _defaults.cooldown_bars)),
        )

        # Create bot instance
        bot = GridFuturesBot(
            bot_id=bot_id,
            config=grid_futures_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            heartbeat_callback=self._heartbeat_callback,
        )

        return bot

    def _create_rsi_grid_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create an RSI-Grid Hybrid Bot instance.

        Combines RSI zone filtering with Grid entry mechanism:
        - RSI Zone determines allowed direction (oversold=long, overbought=short)
        - SMA trend filter provides additional direction bias
        - ATR-based dynamic grid adapts to market volatility
        - Grid levels provide precise entry points

        Design Goals:
        - Target Sharpe > 3.0
        - Walk-Forward Consistency > 90%
        - Win Rate > 70%
        - Max Drawdown < 5%

        Args:
            bot_id: Bot identifier
            config: RSI-Grid bot configuration

        Returns:
            RSIGridBot instance
        """
        # Import here to avoid circular imports
        from decimal import Decimal

        from src.bots.rsi_grid.bot import RSIGridBot
        from src.bots.rsi_grid.models import RSIGridConfig

        # Build RSIGridConfig from dict
        rsi_grid_config = RSIGridConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "15m"),
            leverage=int(config.get("leverage", 2)),
            margin_type=config.get("margin_type", "ISOLATED"),
            # RSI Parameters
            rsi_period=int(config.get("rsi_period", 14)),
            oversold_level=int(config.get("oversold_level", 30)),
            overbought_level=int(config.get("overbought_level", 70)),
            # Grid Parameters
            grid_count=int(config.get("grid_count", 10)),
            atr_period=int(config.get("atr_period", 14)),
            atr_multiplier=Decimal(str(config.get("atr_multiplier", "3.0"))),
            # Trend Filter
            trend_sma_period=int(config.get("trend_sma_period", 20)),
            use_trend_filter=config.get("use_trend_filter", True),
            # Capital allocation
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            max_position_pct=Decimal(str(config.get("max_position_pct", "0.5"))),
            # Risk Management
            stop_loss_atr_mult=Decimal(str(config.get("stop_loss_atr_mult", "1.5"))),
            max_stop_loss_pct=Decimal(str(config.get("max_stop_loss_pct", "0.03"))),
            take_profit_grids=int(config.get("take_profit_grids", 1)),
            max_positions=int(config.get("max_positions", 5)),
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", True),
        )

        # Create bot instance
        bot = RSIGridBot(
            bot_id=bot_id,
            config=rsi_grid_config,
            exchange=self._exchange,
            data_manager=self._data_manager,
            notifier=self._notifier,
            heartbeat_callback=self._heartbeat_callback,
        )

        return bot

    def _create_dca_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a DCA bot instance.

        Args:
            bot_id: Bot identifier
            config: DCA bot configuration

        Returns:
            DCA bot instance

        Note:
            DCA bot is not yet implemented.
        """
        raise UnsupportedBotTypeError(BotType.DCA)

    def _create_trailing_stop_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a TrailingStop bot instance.

        Args:
            bot_id: Bot identifier
            config: Trailing stop bot configuration

        Returns:
            TrailingStop bot instance

        Note:
            TrailingStop bot is not yet implemented.
        """
        raise UnsupportedBotTypeError(BotType.TRAILING_STOP)

    def _create_signal_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a Signal bot instance.

        Args:
            bot_id: Bot identifier
            config: Signal bot configuration

        Returns:
            Signal bot instance

        Note:
            Signal bot is not yet implemented.
        """
        raise UnsupportedBotTypeError(BotType.SIGNAL)

    def register_creator(
        self,
        bot_type: BotType,
        creator: callable,
    ) -> None:
        """
        Register a custom bot creator.

        Allows extending the factory with custom bot types.

        Args:
            bot_type: Bot type to register
            creator: Factory function (bot_id, config) -> BotProtocol
        """
        self._creators[bot_type] = creator
        logger.info(f"Registered custom creator for {bot_type.value}")

    @property
    def supported_types(self) -> list[BotType]:
        """Get list of supported bot types."""
        return list(self._creators.keys())
