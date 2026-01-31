"""
SQLAlchemy ORM Models for Trading Bot Database.

Defines database tables and provides conversion methods between
ORM models and domain models.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Optional

from sqlalchemy import (
    DECIMAL,
    TIMESTAMP,
    Boolean,
    Index,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.core.models import (
    Balance as DomainBalance,
    Kline as DomainKline,
    KlineInterval,
    MarketType,
    Order as DomainOrder,
    OrderSide,
    OrderStatus,
    OrderType,
    Position as DomainPosition,
    PositionSide,
    Trade as DomainTrade,
)


# =============================================================================
# Base Model
# =============================================================================


class Base(DeclarativeBase):
    """SQLAlchemy declarative base for all ORM models."""

    pass


# =============================================================================
# Order Model
# =============================================================================


class OrderModel(Base):
    """Order database model."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    bot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(10), nullable=False)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    order_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8), nullable=True)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    filled_quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    average_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8), nullable=True)
    fee: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    fee_asset: Mapped[Optional[str]] = mapped_column(String(10), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=True,
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_domain(self) -> DomainOrder:
        """Convert to domain Order model."""
        return DomainOrder(
            order_id=self.order_id,
            client_order_id=self.client_order_id,
            symbol=self.symbol,
            side=OrderSide(self.side),
            order_type=OrderType(self.order_type),
            status=OrderStatus(self.status),
            price=self.price,
            quantity=self.quantity,
            filled_qty=self.filled_quantity,
            avg_price=self.average_price,
            fee=self.fee,
            fee_asset=self.fee_asset,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(
        cls,
        order: DomainOrder,
        bot_id: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> "OrderModel":
        """Create from domain Order model."""
        return cls(
            order_id=order.order_id,
            client_order_id=order.client_order_id,
            bot_id=bot_id,
            symbol=order.symbol,
            market_type=market_type.value if isinstance(market_type, MarketType) else market_type,
            side=order.side.value if isinstance(order.side, OrderSide) else order.side,
            order_type=order.order_type.value if isinstance(order.order_type, OrderType) else order.order_type,
            status=order.status.value if isinstance(order.status, OrderStatus) else order.status,
            price=order.price,
            quantity=order.quantity,
            filled_quantity=order.filled_qty,
            average_price=order.avg_price,
            fee=order.fee,
            fee_asset=order.fee_asset,
            created_at=order.created_at,
            updated_at=order.updated_at,
        )


# =============================================================================
# Trade Model
# =============================================================================


class TradeModel(Base):
    """Trade (fill) database model."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trade_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    order_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    bot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    fee: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    fee_asset: Mapped[str] = mapped_column(String(10), nullable=False)
    realized_pnl: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8), nullable=True)
    is_maker: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_domain(self) -> DomainTrade:
        """Convert to domain Trade model."""
        return DomainTrade(
            trade_id=self.trade_id,
            order_id=self.order_id,
            symbol=self.symbol,
            side=OrderSide(self.side),
            price=self.price,
            quantity=self.quantity,
            fee=self.fee,
            fee_asset=self.fee_asset,
            realized_pnl=self.realized_pnl,
            is_maker=self.is_maker,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_domain(
        cls,
        trade: DomainTrade,
        bot_id: Optional[str] = None,
    ) -> "TradeModel":
        """Create from domain Trade model."""
        return cls(
            trade_id=trade.trade_id,
            order_id=trade.order_id,
            bot_id=bot_id,
            symbol=trade.symbol,
            side=trade.side.value if isinstance(trade.side, OrderSide) else trade.side,
            price=trade.price,
            quantity=trade.quantity,
            fee=trade.fee,
            fee_asset=trade.fee_asset,
            realized_pnl=trade.realized_pnl,
            is_maker=trade.is_maker,
            timestamp=trade.timestamp,
        )


# =============================================================================
# Position Model
# =============================================================================


class PositionModel(Base):
    """Futures position database model."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    side: Mapped[str] = mapped_column(String(10), nullable=False)
    quantity: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    entry_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    mark_price: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    liquidation_price: Mapped[Optional[Decimal]] = mapped_column(DECIMAL(20, 8), nullable=True)
    leverage: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    margin: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    unrealized_pnl: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    margin_type: Mapped[str] = mapped_column(String(10), nullable=False, default="isolated")
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    # Unique constraint on bot_id + symbol + side
    __table_args__ = (
        UniqueConstraint("bot_id", "symbol", "side", name="uq_position_bot_symbol_side"),
    )

    def to_domain(self) -> DomainPosition:
        """Convert to domain Position model."""
        return DomainPosition(
            symbol=self.symbol,
            side=PositionSide(self.side),
            quantity=self.quantity,
            entry_price=self.entry_price,
            mark_price=self.mark_price,
            liquidation_price=self.liquidation_price,
            leverage=self.leverage,
            margin=self.margin,
            unrealized_pnl=self.unrealized_pnl,
            margin_type=self.margin_type,
            updated_at=self.updated_at,
        )

    @classmethod
    def from_domain(
        cls,
        position: DomainPosition,
        bot_id: Optional[str] = None,
    ) -> "PositionModel":
        """Create from domain Position model."""
        return cls(
            bot_id=bot_id,
            symbol=position.symbol,
            side=position.side.value if isinstance(position.side, PositionSide) else position.side,
            quantity=position.quantity,
            entry_price=position.entry_price,
            mark_price=position.mark_price,
            liquidation_price=position.liquidation_price,
            leverage=position.leverage,
            margin=position.margin,
            unrealized_pnl=position.unrealized_pnl,
            margin_type=position.margin_type,
            updated_at=position.updated_at,
        )


# =============================================================================
# Balance Model
# =============================================================================


class BalanceModel(Base):
    """Balance snapshot database model."""

    __tablename__ = "balances"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    asset: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    market_type: Mapped[str] = mapped_column(String(10), nullable=False)
    free: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    locked: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    total: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )

    def to_domain(self) -> DomainBalance:
        """Convert to domain Balance model."""
        return DomainBalance(
            asset=self.asset,
            free=self.free,
            locked=self.locked,
        )

    @classmethod
    def from_domain(
        cls,
        balance: DomainBalance,
        bot_id: Optional[str] = None,
        market_type: MarketType = MarketType.SPOT,
    ) -> "BalanceModel":
        """Create from domain Balance model."""
        return cls(
            bot_id=bot_id,
            asset=balance.asset,
            market_type=market_type.value if isinstance(market_type, MarketType) else market_type,
            free=balance.free,
            locked=balance.locked,
            total=balance.total,
        )


# =============================================================================
# Kline Model
# =============================================================================


class KlineModel(Base):
    """Kline (candlestick) database model."""

    __tablename__ = "klines"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    symbol: Mapped[str] = mapped_column(String(20), nullable=False)
    interval: Mapped[str] = mapped_column(String(5), nullable=False)
    open_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    open: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    high: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    low: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    close: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False)
    volume: Mapped[Decimal] = mapped_column(DECIMAL(20, 8), nullable=False, default=Decimal("0"))
    close_time: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)

    # Unique index on symbol + interval + open_time
    __table_args__ = (
        UniqueConstraint("symbol", "interval", "open_time", name="uq_kline_symbol_interval_time"),
        Index("ix_kline_symbol_interval", "symbol", "interval"),
    )

    def to_domain(self) -> DomainKline:
        """Convert to domain Kline model."""
        return DomainKline(
            symbol=self.symbol,
            interval=KlineInterval(self.interval),
            open_time=self.open_time,
            open=self.open,
            high=self.high,
            low=self.low,
            close=self.close,
            volume=self.volume,
            close_time=self.close_time,
        )

    @classmethod
    def from_domain(cls, kline: DomainKline) -> "KlineModel":
        """Create from domain Kline model."""
        return cls(
            symbol=kline.symbol,
            interval=kline.interval.value if isinstance(kline.interval, KlineInterval) else kline.interval,
            open_time=kline.open_time,
            open=kline.open,
            high=kline.high,
            low=kline.low,
            close=kline.close,
            volume=kline.volume,
            close_time=kline.close_time,
        )


# =============================================================================
# Bot State Model
# =============================================================================


class BotStateModel(Base):
    """Bot state database model."""

    __tablename__ = "bot_states"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    bot_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="stopped")
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    state_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "bot_type": self.bot_type,
            "status": self.status,
            "config": self.config,
            "state_data": self.state_data,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BotStateModel":
        """Create from dictionary."""
        return cls(
            bot_id=data["bot_id"],
            bot_type=data["bot_type"],
            status=data.get("status", "stopped"),
            config=data.get("config", {}),
            state_data=data.get("state_data", {}),
        )


class BotStateBackupModel(Base):
    """Bot state backup database model."""

    __tablename__ = "bot_state_backups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    bot_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    bot_type: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    state_data: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    backed_up_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "bot_id": self.bot_id,
            "bot_type": self.bot_type,
            "status": self.status,
            "config": self.config,
            "state_data": self.state_data,
            "backed_up_at": self.backed_up_at,
        }
