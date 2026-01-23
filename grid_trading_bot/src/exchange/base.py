"""
Exchange Base Classes.

Provides abstract base classes for exchange implementations,
enabling multi-exchange support and easy switching.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

from src.core import get_logger
from src.core.models import Kline, MarketType

logger = get_logger(__name__)

T = TypeVar("T")


class ExchangeType(str, Enum):
    """Supported exchange types."""
    BINANCE = "binance"
    OKX = "okx"
    BYBIT = "bybit"
    MOCK = "mock"  # For testing


class ExchangeStatus(str, Enum):
    """Exchange connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    ERROR = "error"
    RATE_LIMITED = "rate_limited"
    MAINTENANCE = "maintenance"


@dataclass
class ExchangeConfig:
    """
    Base exchange configuration.

    Attributes:
        exchange_type: Type of exchange
        api_key: API key
        api_secret: API secret
        testnet: Use testnet/sandbox mode
        timeout: Request timeout in seconds
        rate_limit: Max requests per minute
        priority: Priority for failover (lower = higher priority)
        enabled: Whether this exchange is enabled
    """
    exchange_type: ExchangeType
    api_key: str = ""
    api_secret: str = ""
    testnet: bool = False
    timeout: float = 30.0
    rate_limit: int = 1200
    priority: int = 0
    enabled: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrderResult:
    """Result of an order operation."""
    success: bool
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    order_type: str = ""
    quantity: Decimal = Decimal("0")
    price: Optional[Decimal] = None
    status: str = ""
    filled_quantity: Decimal = Decimal("0")
    average_price: Optional[Decimal] = None
    fee: Decimal = Decimal("0")
    fee_asset: str = ""
    timestamp: Optional[datetime] = None
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error_code: Optional[str] = None
    error_message: Optional[str] = None


@dataclass
class TickerData:
    """Ticker/price data."""
    symbol: str
    price: Decimal
    bid: Optional[Decimal] = None
    ask: Optional[Decimal] = None
    high_24h: Optional[Decimal] = None
    low_24h: Optional[Decimal] = None
    volume_24h: Optional[Decimal] = None
    change_24h: Optional[Decimal] = None
    timestamp: Optional[datetime] = None


@dataclass
class BalanceData:
    """Account balance data."""
    asset: str
    free: Decimal
    locked: Decimal
    total: Decimal = field(init=False)

    def __post_init__(self):
        self.total = self.free + self.locked


@dataclass
class PositionData:
    """Futures position data."""
    symbol: str
    side: str  # LONG, SHORT, BOTH
    quantity: Decimal
    entry_price: Decimal
    mark_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    leverage: int = 1
    margin_type: str = "ISOLATED"
    liquidation_price: Optional[Decimal] = None


class BaseExchangeAPI(ABC):
    """
    Abstract base class for exchange REST API implementations.

    All exchange API classes should inherit from this and implement
    the abstract methods.
    """

    def __init__(self, config: ExchangeConfig):
        """
        Initialize exchange API.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self._status = ExchangeStatus.DISCONNECTED
        self._last_request_time: Optional[datetime] = None
        self._request_count = 0

    @property
    def exchange_type(self) -> ExchangeType:
        """Get exchange type."""
        return self.config.exchange_type

    @property
    def status(self) -> ExchangeStatus:
        """Get current status."""
        return self._status

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._status == ExchangeStatus.CONNECTED

    # =========================================================================
    # Market Data Methods
    # =========================================================================

    @abstractmethod
    async def get_ticker(self, symbol: str) -> TickerData:
        """
        Get current ticker for a symbol.

        Args:
            symbol: Trading symbol

        Returns:
            TickerData with current prices
        """
        pass

    @abstractmethod
    async def get_orderbook(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Dict[str, List[Tuple[Decimal, Decimal]]]:
        """
        Get order book for a symbol.

        Args:
            symbol: Trading symbol
            limit: Number of levels

        Returns:
            Dict with 'bids' and 'asks' lists of (price, quantity)
        """
        pass

    @abstractmethod
    async def get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> List[Kline]:
        """
        Get K-line/candlestick data.

        Args:
            symbol: Trading symbol
            interval: K-line interval (1m, 5m, 15m, etc.)
            limit: Maximum number of klines
            start_time: Start time filter
            end_time: End time filter

        Returns:
            List of Kline objects
        """
        pass

    # =========================================================================
    # Account Methods
    # =========================================================================

    @abstractmethod
    async def get_balance(
        self,
        asset: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> Dict[str, BalanceData]:
        """
        Get account balances.

        Args:
            asset: Specific asset to query (None for all)
            market_type: Market type (SPOT or FUTURES)

        Returns:
            Dict mapping asset to BalanceData
        """
        pass

    @abstractmethod
    async def get_positions(
        self,
        symbol: Optional[str] = None,
    ) -> List[PositionData]:
        """
        Get futures positions.

        Args:
            symbol: Specific symbol (None for all)

        Returns:
            List of PositionData
        """
        pass

    # =========================================================================
    # Order Methods
    # =========================================================================

    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        market_type: MarketType = MarketType.SPOT,
        **kwargs: Any,
    ) -> OrderResult:
        """
        Place an order.

        Args:
            symbol: Trading symbol
            side: BUY or SELL
            order_type: LIMIT, MARKET, etc.
            quantity: Order quantity
            price: Limit price (for LIMIT orders)
            market_type: Market type
            **kwargs: Additional order parameters

        Returns:
            OrderResult
        """
        pass

    @abstractmethod
    async def cancel_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> bool:
        """
        Cancel an order.

        Args:
            symbol: Trading symbol
            order_id: Order ID to cancel
            market_type: Market type

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def get_order(
        self,
        symbol: str,
        order_id: str,
        market_type: MarketType = MarketType.SPOT,
    ) -> Optional[OrderResult]:
        """
        Get order status.

        Args:
            symbol: Trading symbol
            order_id: Order ID
            market_type: Market type

        Returns:
            OrderResult or None if not found
        """
        pass

    @abstractmethod
    async def get_open_orders(
        self,
        symbol: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> List[OrderResult]:
        """
        Get open orders.

        Args:
            symbol: Symbol filter (None for all)
            market_type: Market type

        Returns:
            List of OrderResult
        """
        pass

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @abstractmethod
    async def ping(self) -> bool:
        """
        Check connectivity to exchange.

        Returns:
            True if connected
        """
        pass

    @abstractmethod
    async def get_server_time(self) -> datetime:
        """
        Get exchange server time.

        Returns:
            Server time
        """
        pass

    @abstractmethod
    async def get_exchange_info(
        self,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol info.

        Args:
            symbol: Specific symbol (None for all)

        Returns:
            Exchange info dictionary
        """
        pass


class BaseExchangeWebSocket(ABC):
    """
    Abstract base class for exchange WebSocket implementations.
    """

    def __init__(self, config: ExchangeConfig):
        """
        Initialize WebSocket client.

        Args:
            config: Exchange configuration
        """
        self.config = config
        self._connected = False
        self._subscriptions: Dict[str, Callable] = {}

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected

    @abstractmethod
    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connected
        """
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect WebSocket."""
        pass

    @abstractmethod
    async def subscribe_ticker(
        self,
        symbol: str,
        callback: Callable[[TickerData], None],
    ) -> bool:
        """
        Subscribe to ticker updates.

        Args:
            symbol: Trading symbol
            callback: Callback for ticker updates

        Returns:
            True if subscribed
        """
        pass

    @abstractmethod
    async def subscribe_kline(
        self,
        symbol: str,
        interval: str,
        callback: Callable[[Kline], None],
    ) -> bool:
        """
        Subscribe to K-line updates.

        Args:
            symbol: Trading symbol
            interval: K-line interval
            callback: Callback for kline updates

        Returns:
            True if subscribed
        """
        pass

    @abstractmethod
    async def subscribe_orderbook(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        level: int = 20,
    ) -> bool:
        """
        Subscribe to order book updates.

        Args:
            symbol: Trading symbol
            callback: Callback for orderbook updates
            level: Depth level

        Returns:
            True if subscribed
        """
        pass

    @abstractmethod
    async def unsubscribe(self, symbol: str, stream_type: str) -> bool:
        """
        Unsubscribe from a stream.

        Args:
            symbol: Trading symbol
            stream_type: Type of stream (ticker, kline, orderbook)

        Returns:
            True if unsubscribed
        """
        pass
