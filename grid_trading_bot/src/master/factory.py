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
