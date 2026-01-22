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
            BotType.RSI: self._create_rsi_bot,
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

        Walk-Forward 驗證通過的參數 (75% 一致性, Sharpe 1.81):
        - bb_period: 20, bb_std: 3.0
        - st_atr_period: 20, st_atr_multiplier: 3.5
        - atr_stop_multiplier: 2.0
        - leverage: 2x
        - 報酬: +35.1%, 最大回撤: 6.7%

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

        # Build BollingerConfig from dict with walk-forward validated defaults
        bollinger_config = BollingerConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "15m"),
            # Bollinger Bands (Walk-Forward validated: BB(20, 3.0))
            bb_period=int(config.get("bb_period", 20)),
            bb_std=Decimal(str(config.get("bb_std", "3.0"))),
            # Supertrend (Walk-Forward validated: ST(20, 3.5))
            st_atr_period=int(config.get("st_atr_period", 20)),
            st_atr_multiplier=Decimal(str(config.get("st_atr_multiplier", "3.5"))),
            # ATR Stop Loss (Walk-Forward validated: 2.0x ATR)
            atr_stop_multiplier=Decimal(str(config.get("atr_stop_multiplier", "2.0"))),
            # Position settings
            leverage=int(config.get("leverage", 2)),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            # BBW filter
            bbw_lookback=int(config.get("bbw_lookback", 200)),
            bbw_threshold_pct=int(config.get("bbw_threshold_pct", 20)),
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
        Create a SupertrendBot instance.

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

        # Build SupertrendConfig from dict
        supertrend_config = SupertrendConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "15m"),
            atr_period=int(config.get("atr_period", 25)),
            atr_multiplier=Decimal(str(config.get("atr_multiplier", "3.0"))),
            leverage=int(config.get("leverage", 10)),
            margin_type=config.get("margin_type", "ISOLATED"),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            use_trailing_stop=config.get("use_trailing_stop", False),
            trailing_stop_pct=Decimal(str(config.get("trailing_stop_pct", "0.02"))),
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", True),
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", "0.02"))),
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

        Walk-Forward 驗證通過的參數 (83% 一致性, Sharpe 1.85):
        - leverage: 2x
        - grid_count: 12
        - trend_period: 50
        - 預期年化: ~16.6%, 最大回撤: 8.2%

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

        # Parse direction (default to trend_follow for validated performance)
        direction_str = config.get("direction", "trend_follow")
        direction = GridDirection(direction_str)

        # Build GridFuturesConfig from dict with walk-forward validated defaults
        grid_futures_config = GridFuturesConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "1h"),
            leverage=int(config.get("leverage", 2)),  # Validated: 2x
            margin_type=config.get("margin_type", "ISOLATED"),
            grid_count=int(config.get("grid_count", 12)),  # Validated: 12 grids
            direction=direction,
            use_trend_filter=config.get("use_trend_filter", True),
            trend_period=int(config.get("trend_period", 50)),  # Validated: 50-period
            use_atr_range=config.get("use_atr_range", True),
            atr_period=int(config.get("atr_period", 14)),
            atr_multiplier=Decimal(str(config.get("atr_multiplier", "2.0"))),
            fallback_range_pct=Decimal(str(config.get("fallback_range_pct", "0.08"))),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            max_position_pct=Decimal(str(config.get("max_position_pct", "0.5"))),
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", "0.05"))),
            rebuild_threshold_pct=Decimal(str(config.get("rebuild_threshold_pct", "0.02"))),
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", True),
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

    def _create_rsi_bot(
        self,
        bot_id: str,
        config: dict[str, Any],
    ) -> BotProtocol:
        """
        Create a RSI Momentum Bot instance.

        Uses momentum strategy (trend following) instead of mean reversion:
        - Long when RSI crosses above entry_level + momentum_threshold
        - Short when RSI crosses below entry_level - momentum_threshold

        Optimized defaults (67% walk-forward consistency, Sharpe 1.03):
        - RSI Period: 21
        - Entry Level: 50, Momentum Threshold: 5
        - Leverage: 5x
        - Stop Loss: 2%, Take Profit: 4%

        Args:
            bot_id: Bot identifier
            config: RSI bot configuration

        Returns:
            RSIBot instance
        """
        # Import here to avoid circular imports
        from decimal import Decimal

        from src.bots.rsi.bot import RSIBot
        from src.bots.rsi.models import RSIConfig

        # Build RSIConfig from dict with momentum strategy defaults
        rsi_config = RSIConfig(
            symbol=config["symbol"],
            timeframe=config.get("timeframe", "15m"),
            rsi_period=int(config.get("rsi_period", 21)),  # Optimized: 21
            entry_level=int(config.get("entry_level", 50)),  # Momentum crossover level
            momentum_threshold=int(config.get("momentum_threshold", 5)),  # Crossover threshold
            leverage=int(config.get("leverage", 5)),  # Optimized: 5x
            margin_type=config.get("margin_type", "ISOLATED"),
            max_capital=Decimal(str(config["max_capital"])) if config.get("max_capital") else None,
            position_size_pct=Decimal(str(config.get("position_size_pct", "0.1"))),
            stop_loss_pct=Decimal(str(config.get("stop_loss_pct", "0.02"))),  # 2%
            take_profit_pct=Decimal(str(config.get("take_profit_pct", "0.04"))),  # 4%
            use_exchange_stop_loss=config.get("use_exchange_stop_loss", True),
        )

        # Create bot instance
        bot = RSIBot(
            bot_id=bot_id,
            config=rsi_config,
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
