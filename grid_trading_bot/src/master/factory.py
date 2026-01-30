"""
Bot Factory for creating bot instances.

Creates and configures trading bot instances based on type and configuration.
"""

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
        from src.bots.bollinger.bot import BollingerBot
        from src.bots.bollinger.models import BollingerConfig

        # 從 settings.yaml 載入參數（單一來源），runtime config 作為 override
        bollinger_config = BollingerConfig.from_yaml(
            symbol=config["symbol"],
            **{k: v for k, v in config.items() if k != "symbol"},
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
        Create a SupertrendBot instance (HYBRID_GRID mode).

        嚴格成本約束優化 + Walk-Forward 驗證 (2026-01-30):
        - 年化: 227%, Sharpe: 9.64, MDD: 1.80%
        - W-F 一致性: 100% (9/9), Monte Carlo: 100%
        - 成本模型: 手續費 0.06% + 滑價 0.05%

        HYBRID_GRID 模式:
        - 雙向交易 + Supertrend 趨勢偏移
        - RSI 過濾器避免極端進場
        - 超時出場 + 網格自動重置

        Args:
            bot_id: Bot identifier
            config: Supertrend bot configuration

        Returns:
            SupertrendBot instance
        """
        # Import here to avoid circular imports
        from src.bots.supertrend.bot import SupertrendBot
        from src.bots.supertrend.models import SupertrendConfig

        # 從 settings.yaml 載入參數（單一來源），runtime config 作為 override
        supertrend_config = SupertrendConfig.from_yaml(
            symbol=config["symbol"],
            **{k: v for k, v in config.items() if k != "symbol"},
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
        from src.bots.grid_futures.bot import GridFuturesBot
        from src.bots.grid_futures.models import GridFuturesConfig

        # 從 settings.yaml 載入參數（單一來源），runtime config 作為 override
        grid_futures_config = GridFuturesConfig.from_yaml(
            symbol=config["symbol"],
            **{k: v for k, v in config.items() if k != "symbol"},
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
        from src.bots.rsi_grid.bot import RSIGridBot
        from src.bots.rsi_grid.models import RSIGridConfig

        # 從 settings.yaml 載入參數（單一來源），runtime config 作為 override
        rsi_grid_config = RSIGridConfig.from_yaml(
            symbol=config["symbol"],
            **{k: v for k, v in config.items() if k != "symbol"},
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
