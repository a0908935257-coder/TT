"""
Base data models for Grid Trading Bot.

Pydantic v2 models for trading system core data structures including
K-lines, orders, positions, accounts, and more.
"""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

from .utils import timestamp_to_datetime


# =============================================================================
# Enums
# =============================================================================


class OrderSide(str, Enum):
    """Order side - buy or sell."""

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """Order type."""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"


class OrderStatus(str, Enum):
    """Order status."""

    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionSide(str, Enum):
    """Position side for futures trading."""

    LONG = "LONG"
    SHORT = "SHORT"
    BOTH = "BOTH"


class MarketType(str, Enum):
    """Market type."""

    SPOT = "SPOT"
    FUTURES = "FUTURES"


class KlineInterval(str, Enum):
    """Kline/candlestick interval."""

    m1 = "1m"
    m5 = "5m"
    m15 = "15m"
    m30 = "30m"
    h1 = "1h"
    h4 = "4h"
    d1 = "1d"
    w1 = "1w"


# =============================================================================
# Base Model Configuration
# =============================================================================


class TradingBaseModel(BaseModel):
    """Base model with common configuration for all trading models."""

    model_config = ConfigDict(
        use_enum_values=True,
        validate_assignment=True,
        from_attributes=True,
    )


# =============================================================================
# Kline Model
# =============================================================================


class Kline(TradingBaseModel):
    """K-line (candlestick) data model."""

    symbol: str
    interval: KlineInterval
    open_time: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    close_time: datetime
    quote_volume: Decimal = Decimal("0")
    trades_count: int = 0

    @computed_field
    @property
    def is_bullish(self) -> bool:
        """True if bullish candle (close > open)."""
        return self.close > self.open

    @computed_field
    @property
    def is_bearish(self) -> bool:
        """True if bearish candle (close < open)."""
        return self.close < self.open

    @computed_field
    @property
    def body(self) -> Decimal:
        """Absolute body size (|close - open|)."""
        return abs(self.close - self.open)

    @computed_field
    @property
    def range(self) -> Decimal:
        """Price range (high - low)."""
        return self.high - self.low

    @computed_field
    @property
    def body_ratio(self) -> Decimal:
        """Body to range ratio (body / range)."""
        if self.range == 0:
            return Decimal("0")
        return self.body / self.range

    @classmethod
    def from_binance(cls, data: list, symbol: str, interval: KlineInterval) -> "Kline":
        """
        Create Kline from Binance API kline data.

        Args:
            data: Binance kline array [open_time, open, high, low, close, volume, ...]
            symbol: Trading pair symbol
            interval: Kline interval

        Returns:
            Kline instance
        """
        return cls(
            symbol=symbol,
            interval=interval,
            open_time=timestamp_to_datetime(data[0]),
            open=Decimal(str(data[1])),
            high=Decimal(str(data[2])),
            low=Decimal(str(data[3])),
            close=Decimal(str(data[4])),
            volume=Decimal(str(data[5])),
            close_time=timestamp_to_datetime(data[6]),
            quote_volume=Decimal(str(data[7])) if len(data) > 7 else Decimal("0"),
            trades_count=int(data[8]) if len(data) > 8 else 0,
        )


# =============================================================================
# Ticker Model
# =============================================================================


class Ticker(TradingBaseModel):
    """Real-time ticker data model."""

    symbol: str
    price: Decimal
    bid: Decimal
    ask: Decimal
    high_24h: Decimal
    low_24h: Decimal
    volume_24h: Decimal
    change_24h: Decimal
    timestamp: datetime

    @computed_field
    @property
    def spread(self) -> Decimal:
        """Bid-ask spread (ask - bid)."""
        return self.ask - self.bid

    @computed_field
    @property
    def spread_percent(self) -> Decimal:
        """Spread as percentage of price."""
        if self.price == 0:
            return Decimal("0")
        return (self.spread / self.price) * Decimal("100")

    @classmethod
    def from_binance(cls, data: dict) -> "Ticker":
        """
        Create Ticker from Binance API ticker data.

        Args:
            data: Binance 24hr ticker response

        Returns:
            Ticker instance
        """
        return cls(
            symbol=data["symbol"],
            price=Decimal(str(data["lastPrice"])),
            bid=Decimal(str(data["bidPrice"])),
            ask=Decimal(str(data["askPrice"])),
            high_24h=Decimal(str(data["highPrice"])),
            low_24h=Decimal(str(data["lowPrice"])),
            volume_24h=Decimal(str(data["volume"])),
            change_24h=Decimal(str(data["priceChangePercent"])),
            timestamp=timestamp_to_datetime(data.get("closeTime", 0)),
        )


# =============================================================================
# Order Model
# =============================================================================


class Order(TradingBaseModel):
    """Order data model."""

    order_id: str
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    status: OrderStatus
    price: Optional[Decimal] = None
    quantity: Decimal
    filled_qty: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    fee: Decimal = Decimal("0")
    fee_asset: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    @computed_field
    @property
    def is_active(self) -> bool:
        """True if order is active (NEW or PARTIALLY_FILLED)."""
        return self.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)

    @computed_field
    @property
    def is_filled(self) -> bool:
        """True if order is completely filled."""
        return self.status == OrderStatus.FILLED

    @computed_field
    @property
    def filled_percent(self) -> Decimal:
        """Filled percentage (filled_qty / quantity * 100)."""
        if self.quantity == 0:
            return Decimal("0")
        return (self.filled_qty / self.quantity) * Decimal("100")

    @computed_field
    @property
    def remaining_qty(self) -> Decimal:
        """Remaining quantity to be filled."""
        return self.quantity - self.filled_qty

    @computed_field
    @property
    def filled_value(self) -> Decimal:
        """Filled value (filled_qty * avg_price)."""
        if self.avg_price is None:
            return Decimal("0")
        return self.filled_qty * self.avg_price

    @classmethod
    def from_binance(cls, data: dict) -> "Order":
        """
        Create Order from Binance API order data.

        Args:
            data: Binance order response

        Returns:
            Order instance
        """
        return cls(
            order_id=str(data["orderId"]),
            client_order_id=data.get("clientOrderId"),
            symbol=data["symbol"],
            side=OrderSide(data["side"]),
            order_type=OrderType(data["type"]),
            status=OrderStatus(data["status"]),
            price=Decimal(str(data["price"])) if data.get("price") else None,
            quantity=Decimal(str(data["origQty"])),
            filled_qty=Decimal(str(data.get("executedQty", "0"))),
            avg_price=Decimal(str(data["avgPrice"])) if data.get("avgPrice") else None,
            created_at=timestamp_to_datetime(data.get("time", data.get("transactTime", 0))),
            updated_at=timestamp_to_datetime(data["updateTime"]) if data.get("updateTime") else None,
        )


# =============================================================================
# Position Model (Futures)
# =============================================================================


class Position(TradingBaseModel):
    """Futures position data model."""

    symbol: str
    side: PositionSide
    quantity: Decimal
    entry_price: Decimal
    mark_price: Decimal
    liquidation_price: Optional[Decimal] = None
    leverage: int
    margin: Decimal
    unrealized_pnl: Decimal
    margin_type: str = "isolated"
    updated_at: datetime

    @computed_field
    @property
    def notional_value(self) -> Decimal:
        """Notional value (quantity * mark_price)."""
        return self.quantity * self.mark_price

    @computed_field
    @property
    def roe(self) -> Decimal:
        """Return on equity percentage (unrealized_pnl / margin * 100)."""
        if self.margin == 0:
            return Decimal("0")
        return (self.unrealized_pnl / self.margin) * Decimal("100")

    @computed_field
    @property
    def is_long(self) -> bool:
        """True if long position."""
        return self.side == PositionSide.LONG

    @computed_field
    @property
    def is_short(self) -> bool:
        """True if short position."""
        return self.side == PositionSide.SHORT

    @classmethod
    def from_binance(cls, data: dict) -> "Position":
        """
        Create Position from Binance Futures API position data.

        Args:
            data: Binance position response

        Returns:
            Position instance
        """
        return cls(
            symbol=data["symbol"],
            side=PositionSide(data["positionSide"]),
            quantity=abs(Decimal(str(data["positionAmt"]))),
            entry_price=Decimal(str(data["entryPrice"])),
            mark_price=Decimal(str(data["markPrice"])),
            liquidation_price=Decimal(str(data["liquidationPrice"])) if data.get("liquidationPrice") else None,
            leverage=int(data["leverage"]),
            margin=Decimal(str(data.get("isolatedMargin", data.get("maintMargin", "0")))),
            unrealized_pnl=Decimal(str(data["unRealizedProfit"])),
            margin_type=data.get("marginType", "isolated").lower(),
            updated_at=timestamp_to_datetime(data.get("updateTime", 0)),
        )


# =============================================================================
# Balance Model
# =============================================================================


class Balance(TradingBaseModel):
    """Account balance for a single asset."""

    asset: str
    free: Decimal
    locked: Decimal

    @computed_field
    @property
    def total(self) -> Decimal:
        """Total balance (free + locked)."""
        return self.free + self.locked

    @classmethod
    def from_binance(cls, data: dict) -> "Balance":
        """
        Create Balance from Binance API balance data.

        Args:
            data: Binance balance response

        Returns:
            Balance instance
        """
        return cls(
            asset=data["asset"],
            free=Decimal(str(data["free"])),
            locked=Decimal(str(data["locked"])),
        )


# =============================================================================
# AccountInfo Model
# =============================================================================


class AccountInfo(TradingBaseModel):
    """Account information including balances and positions."""

    market_type: MarketType
    balances: list[Balance]
    positions: list[Position] = Field(default_factory=list)
    updated_at: datetime

    def get_balance(self, asset: str) -> Optional[Balance]:
        """
        Get balance for a specific asset.

        Args:
            asset: Asset name (e.g., "USDT")

        Returns:
            Balance instance or None if not found
        """
        for balance in self.balances:
            if balance.asset == asset:
                return balance
        return None

    @classmethod
    def from_binance_spot(cls, data: dict) -> "AccountInfo":
        """
        Create AccountInfo from Binance Spot API account data.

        Args:
            data: Binance spot account response

        Returns:
            AccountInfo instance
        """
        balances = [
            Balance.from_binance(b)
            for b in data.get("balances", [])
            if Decimal(str(b["free"])) > 0 or Decimal(str(b["locked"])) > 0
        ]
        return cls(
            market_type=MarketType.SPOT,
            balances=balances,
            positions=[],
            updated_at=timestamp_to_datetime(data.get("updateTime", 0)),
        )

    @classmethod
    def from_binance_futures(cls, data: dict) -> "AccountInfo":
        """
        Create AccountInfo from Binance Futures API account data.

        Args:
            data: Binance futures account response

        Returns:
            AccountInfo instance
        """
        balances = [
            Balance(
                asset=b["asset"],
                free=Decimal(str(b["availableBalance"])),
                locked=Decimal(str(b["balance"])) - Decimal(str(b["availableBalance"])),
            )
            for b in data.get("assets", [])
            if Decimal(str(b["walletBalance"])) > 0
        ]
        positions = [
            Position.from_binance(p)
            for p in data.get("positions", [])
            if Decimal(str(p.get("positionAmt", "0"))) != 0
        ]
        return cls(
            market_type=MarketType.FUTURES,
            balances=balances,
            positions=positions,
            updated_at=datetime.now(timezone.utc),
        )


# =============================================================================
# Trade Model
# =============================================================================


class Trade(TradingBaseModel):
    """Trade (fill) record data model."""

    trade_id: str
    order_id: str
    symbol: str
    side: OrderSide
    price: Decimal
    quantity: Decimal
    fee: Decimal
    fee_asset: str
    realized_pnl: Optional[Decimal] = None
    is_maker: bool
    timestamp: datetime

    @computed_field
    @property
    def value(self) -> Decimal:
        """Trade value (price * quantity)."""
        return self.price * self.quantity

    @classmethod
    def from_binance(cls, data: dict) -> "Trade":
        """
        Create Trade from Binance API trade data.

        Args:
            data: Binance trade response

        Returns:
            Trade instance
        """
        return cls(
            trade_id=str(data["id"]),
            order_id=str(data["orderId"]),
            symbol=data["symbol"],
            side=OrderSide.BUY if data.get("isBuyer", False) else OrderSide.SELL,
            price=Decimal(str(data["price"])),
            quantity=Decimal(str(data["qty"])),
            fee=Decimal(str(data["commission"])),
            fee_asset=data["commissionAsset"],
            realized_pnl=Decimal(str(data["realizedPnl"])) if data.get("realizedPnl") else None,
            is_maker=data.get("isMaker", False),
            timestamp=timestamp_to_datetime(data["time"]),
        )


# =============================================================================
# SymbolInfo Model
# =============================================================================


class SymbolInfo(TradingBaseModel):
    """Trading pair information and constraints."""

    symbol: str
    base_asset: str
    quote_asset: str
    price_precision: int
    quantity_precision: int
    min_quantity: Decimal
    min_notional: Decimal
    tick_size: Decimal
    step_size: Decimal

    @classmethod
    def from_binance(cls, data: dict) -> "SymbolInfo":
        """
        Create SymbolInfo from Binance API exchange info.

        Args:
            data: Binance symbol info from exchange info

        Returns:
            SymbolInfo instance
        """
        # Extract filters
        filters = {f["filterType"]: f for f in data.get("filters", [])}

        lot_size = filters.get("LOT_SIZE", {})
        price_filter = filters.get("PRICE_FILTER", {})
        notional = filters.get("NOTIONAL", filters.get("MIN_NOTIONAL", {}))

        return cls(
            symbol=data["symbol"],
            base_asset=data["baseAsset"],
            quote_asset=data["quoteAsset"],
            price_precision=data.get("pricePrecision", data.get("quotePrecision", 8)),
            quantity_precision=data.get("quantityPrecision", data.get("baseAssetPrecision", 8)),
            min_quantity=Decimal(str(lot_size.get("minQty", "0.00000001"))),
            min_notional=Decimal(str(notional.get("minNotional", "0"))),
            tick_size=Decimal(str(price_filter.get("tickSize", "0.00000001"))),
            step_size=Decimal(str(lot_size.get("stepSize", "0.00000001"))),
        )
