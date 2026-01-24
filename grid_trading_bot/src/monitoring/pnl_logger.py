"""
P&L logging module.

Provides daily settlement logging and P&L audit trails
for regulatory compliance and performance tracking.
"""

import asyncio
import json
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.core import get_logger

logger = get_logger(__name__)


class SettlementStatus(Enum):
    """Settlement status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TradeRecord:
    """Individual trade record."""

    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # buy/sell
    quantity: Decimal
    price: Decimal
    commission: Decimal
    realized_pnl: Decimal
    order_id: str
    strategy_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trade_id": self.trade_id,
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "side": self.side,
            "quantity": str(self.quantity),
            "price": str(self.price),
            "commission": str(self.commission),
            "realized_pnl": str(self.realized_pnl),
            "order_id": self.order_id,
            "strategy_id": self.strategy_id,
        }


@dataclass
class PositionSnapshot:
    """Position snapshot at settlement time."""

    symbol: str
    quantity: Decimal
    avg_cost: Decimal
    current_price: Decimal
    unrealized_pnl: Decimal
    market_value: Decimal

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "quantity": str(self.quantity),
            "avg_cost": str(self.avg_cost),
            "current_price": str(self.current_price),
            "unrealized_pnl": str(self.unrealized_pnl),
            "market_value": str(self.market_value),
        }


@dataclass
class DailySettlement:
    """Daily settlement record."""

    settlement_date: date
    status: SettlementStatus = SettlementStatus.PENDING
    settlement_time: Optional[datetime] = None

    # P&L Summary
    realized_pnl: Decimal = Decimal("0")
    unrealized_pnl: Decimal = Decimal("0")
    total_commission: Decimal = Decimal("0")
    net_pnl: Decimal = Decimal("0")

    # Account Summary
    starting_equity: Decimal = Decimal("0")
    ending_equity: Decimal = Decimal("0")
    equity_change: Decimal = Decimal("0")
    equity_change_percent: float = 0.0

    # Trading Activity
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    total_volume: Decimal = Decimal("0")
    total_turnover: Decimal = Decimal("0")

    # Risk Metrics
    max_drawdown: Decimal = Decimal("0")
    max_drawdown_percent: float = 0.0
    sharpe_ratio: Optional[float] = None
    win_rate: float = 0.0

    # Positions
    positions: List[PositionSnapshot] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)

    # Notes and errors
    notes: str = ""
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "settlement_date": self.settlement_date.isoformat(),
            "status": self.status.value,
            "settlement_time": (
                self.settlement_time.isoformat() if self.settlement_time else None
            ),
            "pnl": {
                "realized": str(self.realized_pnl),
                "unrealized": str(self.unrealized_pnl),
                "commission": str(self.total_commission),
                "net": str(self.net_pnl),
            },
            "account": {
                "starting_equity": str(self.starting_equity),
                "ending_equity": str(self.ending_equity),
                "equity_change": str(self.equity_change),
                "equity_change_percent": self.equity_change_percent,
            },
            "activity": {
                "total_trades": self.total_trades,
                "buy_trades": self.buy_trades,
                "sell_trades": self.sell_trades,
                "total_volume": str(self.total_volume),
                "total_turnover": str(self.total_turnover),
            },
            "risk": {
                "max_drawdown": str(self.max_drawdown),
                "max_drawdown_percent": self.max_drawdown_percent,
                "sharpe_ratio": self.sharpe_ratio,
                "win_rate": self.win_rate,
            },
            "positions": [p.to_dict() for p in self.positions],
            "trade_count": len(self.trades),
            "notes": self.notes,
            "errors": self.errors,
        }


class PnLLogger:
    """
    P&L logging and daily settlement manager.

    Records all P&L changes and generates daily settlement reports.
    """

    def __init__(
        self,
        db_path: Optional[Path] = None,
        log_dir: Optional[Path] = None,
        settlement_hour: int = 16,  # 4 PM local time
    ):
        """
        Initialize the P&L logger.

        Args:
            db_path: SQLite database path
            log_dir: Directory for settlement log files
            settlement_hour: Hour to run daily settlement
        """
        self._db_path = db_path or Path("data/pnl.db")
        self._log_dir = log_dir or Path("logs/pnl")
        self._settlement_hour = settlement_hour

        # Ensure directories exist
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

        # Current day tracking
        self._current_date: Optional[date] = None
        self._current_settlement: Optional[DailySettlement] = None

        # Callbacks
        self._settlement_callbacks: List[Callable[[DailySettlement], None]] = []

    def _init_db(self) -> None:
        """Initialize the SQLite database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS daily_settlements (
                    settlement_date TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    settlement_time TEXT,
                    realized_pnl TEXT NOT NULL,
                    unrealized_pnl TEXT NOT NULL,
                    total_commission TEXT NOT NULL,
                    net_pnl TEXT NOT NULL,
                    starting_equity TEXT NOT NULL,
                    ending_equity TEXT NOT NULL,
                    equity_change TEXT NOT NULL,
                    equity_change_percent REAL,
                    total_trades INTEGER,
                    buy_trades INTEGER,
                    sell_trades INTEGER,
                    total_volume TEXT,
                    total_turnover TEXT,
                    max_drawdown TEXT,
                    max_drawdown_percent REAL,
                    sharpe_ratio REAL,
                    win_rate REAL,
                    positions_json TEXT,
                    notes TEXT,
                    errors_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id TEXT PRIMARY KEY,
                    settlement_date TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    quantity TEXT NOT NULL,
                    price TEXT NOT NULL,
                    commission TEXT NOT NULL,
                    realized_pnl TEXT NOT NULL,
                    order_id TEXT NOT NULL,
                    strategy_id TEXT,
                    FOREIGN KEY (settlement_date) REFERENCES daily_settlements(settlement_date)
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_date
                ON trades(settlement_date)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol
                ON trades(symbol)
            """)

            conn.commit()

    def record_trade(self, trade: TradeRecord) -> None:
        """
        Record a trade.

        Args:
            trade: Trade record to log
        """
        self._ensure_current_settlement(trade.timestamp.date())

        # Add to current settlement
        if self._current_settlement:
            self._current_settlement.trades.append(trade)
            self._current_settlement.total_trades += 1
            self._current_settlement.realized_pnl += trade.realized_pnl
            self._current_settlement.total_commission += trade.commission
            self._current_settlement.total_volume += trade.quantity
            self._current_settlement.total_turnover += trade.quantity * trade.price

            if trade.side.lower() == "buy":
                self._current_settlement.buy_trades += 1
            else:
                self._current_settlement.sell_trades += 1

        # Persist to database
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO trades
                (trade_id, settlement_date, timestamp, symbol, side,
                 quantity, price, commission, realized_pnl, order_id, strategy_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    trade.trade_id,
                    trade.timestamp.date().isoformat(),
                    trade.timestamp.isoformat(),
                    trade.symbol,
                    trade.side,
                    str(trade.quantity),
                    str(trade.price),
                    str(trade.commission),
                    str(trade.realized_pnl),
                    trade.order_id,
                    trade.strategy_id,
                ),
            )
            conn.commit()

        logger.debug(
            f"Trade recorded: {trade.trade_id} {trade.side} "
            f"{trade.quantity} {trade.symbol} @ {trade.price}"
        )

    def update_equity(
        self,
        current_equity: Decimal,
        unrealized_pnl: Decimal,
        positions: List[PositionSnapshot],
    ) -> None:
        """
        Update current equity snapshot.

        Args:
            current_equity: Current account equity
            unrealized_pnl: Total unrealized P&L
            positions: Current positions
        """
        today = date.today()
        self._ensure_current_settlement(today)

        if self._current_settlement:
            self._current_settlement.ending_equity = current_equity
            self._current_settlement.unrealized_pnl = unrealized_pnl
            self._current_settlement.positions = positions

            # Update equity change
            if self._current_settlement.starting_equity > 0:
                change = current_equity - self._current_settlement.starting_equity
                self._current_settlement.equity_change = change
                self._current_settlement.equity_change_percent = float(
                    change / self._current_settlement.starting_equity * 100
                )

            # Update net P&L
            self._current_settlement.net_pnl = (
                self._current_settlement.realized_pnl
                + unrealized_pnl
                - self._current_settlement.total_commission
            )

    def update_risk_metrics(
        self,
        max_drawdown: Decimal,
        max_drawdown_percent: float,
        sharpe_ratio: Optional[float] = None,
        win_rate: float = 0.0,
    ) -> None:
        """
        Update risk metrics.

        Args:
            max_drawdown: Maximum drawdown amount
            max_drawdown_percent: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio if available
            win_rate: Win rate percentage
        """
        if self._current_settlement:
            self._current_settlement.max_drawdown = max_drawdown
            self._current_settlement.max_drawdown_percent = max_drawdown_percent
            self._current_settlement.sharpe_ratio = sharpe_ratio
            self._current_settlement.win_rate = win_rate

    def _ensure_current_settlement(self, target_date: date) -> None:
        """Ensure we have a settlement for the target date."""
        if self._current_date != target_date:
            # Finalize previous day if exists
            if self._current_settlement and self._current_date:
                self._save_settlement(self._current_settlement)

            # Load or create settlement for target date
            self._current_date = target_date
            self._current_settlement = self._load_settlement(target_date)

            if not self._current_settlement:
                self._current_settlement = DailySettlement(
                    settlement_date=target_date,
                    status=SettlementStatus.IN_PROGRESS,
                )
                # Set starting equity from previous day's ending
                prev_settlement = self._get_previous_settlement(target_date)
                if prev_settlement:
                    self._current_settlement.starting_equity = (
                        prev_settlement.ending_equity
                    )

    def _load_settlement(self, settlement_date: date) -> Optional[DailySettlement]:
        """Load settlement from database."""
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT * FROM daily_settlements WHERE settlement_date = ?
            """,
                (settlement_date.isoformat(),),
            )
            row = cursor.fetchone()

            if not row:
                return None

            # Load trades
            trades_cursor = conn.execute(
                """
                SELECT * FROM trades WHERE settlement_date = ?
            """,
                (settlement_date.isoformat(),),
            )

            trades = []
            for trade_row in trades_cursor:
                trades.append(
                    TradeRecord(
                        trade_id=trade_row["trade_id"],
                        timestamp=datetime.fromisoformat(trade_row["timestamp"]),
                        symbol=trade_row["symbol"],
                        side=trade_row["side"],
                        quantity=Decimal(trade_row["quantity"]),
                        price=Decimal(trade_row["price"]),
                        commission=Decimal(trade_row["commission"]),
                        realized_pnl=Decimal(trade_row["realized_pnl"]),
                        order_id=trade_row["order_id"],
                        strategy_id=trade_row["strategy_id"],
                    )
                )

            positions_json = row["positions_json"]
            positions = []
            if positions_json:
                for p in json.loads(positions_json):
                    positions.append(
                        PositionSnapshot(
                            symbol=p["symbol"],
                            quantity=Decimal(p["quantity"]),
                            avg_cost=Decimal(p["avg_cost"]),
                            current_price=Decimal(p["current_price"]),
                            unrealized_pnl=Decimal(p["unrealized_pnl"]),
                            market_value=Decimal(p["market_value"]),
                        )
                    )

            errors_json = row["errors_json"]
            errors = json.loads(errors_json) if errors_json else []

            return DailySettlement(
                settlement_date=date.fromisoformat(row["settlement_date"]),
                status=SettlementStatus(row["status"]),
                settlement_time=(
                    datetime.fromisoformat(row["settlement_time"])
                    if row["settlement_time"]
                    else None
                ),
                realized_pnl=Decimal(row["realized_pnl"]),
                unrealized_pnl=Decimal(row["unrealized_pnl"]),
                total_commission=Decimal(row["total_commission"]),
                net_pnl=Decimal(row["net_pnl"]),
                starting_equity=Decimal(row["starting_equity"]),
                ending_equity=Decimal(row["ending_equity"]),
                equity_change=Decimal(row["equity_change"]),
                equity_change_percent=row["equity_change_percent"] or 0.0,
                total_trades=row["total_trades"] or 0,
                buy_trades=row["buy_trades"] or 0,
                sell_trades=row["sell_trades"] or 0,
                total_volume=Decimal(row["total_volume"] or "0"),
                total_turnover=Decimal(row["total_turnover"] or "0"),
                max_drawdown=Decimal(row["max_drawdown"] or "0"),
                max_drawdown_percent=row["max_drawdown_percent"] or 0.0,
                sharpe_ratio=row["sharpe_ratio"],
                win_rate=row["win_rate"] or 0.0,
                positions=positions,
                trades=trades,
                notes=row["notes"] or "",
                errors=errors,
            )

    def _save_settlement(self, settlement: DailySettlement) -> None:
        """Save settlement to database."""
        now = datetime.now(timezone.utc).isoformat()

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_settlements
                (settlement_date, status, settlement_time, realized_pnl,
                 unrealized_pnl, total_commission, net_pnl, starting_equity,
                 ending_equity, equity_change, equity_change_percent,
                 total_trades, buy_trades, sell_trades, total_volume,
                 total_turnover, max_drawdown, max_drawdown_percent,
                 sharpe_ratio, win_rate, positions_json, notes, errors_json,
                 created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    settlement.settlement_date.isoformat(),
                    settlement.status.value,
                    (
                        settlement.settlement_time.isoformat()
                        if settlement.settlement_time
                        else None
                    ),
                    str(settlement.realized_pnl),
                    str(settlement.unrealized_pnl),
                    str(settlement.total_commission),
                    str(settlement.net_pnl),
                    str(settlement.starting_equity),
                    str(settlement.ending_equity),
                    str(settlement.equity_change),
                    settlement.equity_change_percent,
                    settlement.total_trades,
                    settlement.buy_trades,
                    settlement.sell_trades,
                    str(settlement.total_volume),
                    str(settlement.total_turnover),
                    str(settlement.max_drawdown),
                    settlement.max_drawdown_percent,
                    settlement.sharpe_ratio,
                    settlement.win_rate,
                    json.dumps([p.to_dict() for p in settlement.positions]),
                    settlement.notes,
                    json.dumps(settlement.errors),
                    now,
                    now,
                ),
            )
            conn.commit()

    def _get_previous_settlement(
        self, current_date: date
    ) -> Optional[DailySettlement]:
        """Get the previous day's settlement."""
        prev_date = current_date - timedelta(days=1)
        return self._load_settlement(prev_date)

    async def run_daily_settlement(
        self,
        settlement_date: Optional[date] = None,
    ) -> DailySettlement:
        """
        Run daily settlement process.

        Args:
            settlement_date: Date to settle (default today)

        Returns:
            Completed settlement record
        """
        target_date = settlement_date or date.today()
        self._ensure_current_settlement(target_date)

        if not self._current_settlement:
            self._current_settlement = DailySettlement(settlement_date=target_date)

        settlement = self._current_settlement

        try:
            settlement.status = SettlementStatus.IN_PROGRESS
            settlement.settlement_time = datetime.now(timezone.utc)

            # Calculate win rate
            if settlement.trades:
                winning = sum(1 for t in settlement.trades if t.realized_pnl > 0)
                settlement.win_rate = winning / len(settlement.trades) * 100

            # Mark as completed
            settlement.status = SettlementStatus.COMPLETED

            # Save to database
            self._save_settlement(settlement)

            # Write log file
            self._write_settlement_log(settlement)

            # Notify callbacks
            for callback in self._settlement_callbacks:
                try:
                    callback(settlement)
                except Exception as e:
                    logger.error(f"Settlement callback error: {e}")
                    settlement.errors.append(str(e))

            logger.info(
                f"Daily settlement completed for {target_date}: "
                f"Net P&L={settlement.net_pnl}, Trades={settlement.total_trades}"
            )

        except Exception as e:
            settlement.status = SettlementStatus.FAILED
            settlement.errors.append(str(e))
            self._save_settlement(settlement)
            logger.error(f"Daily settlement failed: {e}")
            raise

        return settlement

    def _write_settlement_log(self, settlement: DailySettlement) -> None:
        """Write settlement to log file."""
        log_file = self._log_dir / f"settlement_{settlement.settlement_date}.json"

        with open(log_file, "w") as f:
            json.dump(settlement.to_dict(), f, indent=2)

        logger.debug(f"Settlement log written to {log_file}")

    def get_settlement(self, settlement_date: date) -> Optional[DailySettlement]:
        """Get settlement for a specific date."""
        return self._load_settlement(settlement_date)

    def get_settlements_range(
        self,
        start_date: date,
        end_date: date,
    ) -> List[DailySettlement]:
        """Get settlements for a date range."""
        settlements = []
        current = start_date

        while current <= end_date:
            settlement = self._load_settlement(current)
            if settlement:
                settlements.append(settlement)
            current += timedelta(days=1)

        return settlements

    def get_cumulative_pnl(
        self,
        start_date: date,
        end_date: date,
    ) -> Decimal:
        """Calculate cumulative P&L for a period."""
        settlements = self.get_settlements_range(start_date, end_date)
        return sum((s.net_pnl for s in settlements), Decimal("0"))

    def add_settlement_callback(
        self,
        callback: Callable[[DailySettlement], None],
    ) -> None:
        """Add a callback for settlement completion."""
        self._settlement_callbacks.append(callback)


class PnLSettlementScheduler:
    """
    Scheduler for automatic daily settlements.

    Runs daily settlement at configured time.
    """

    def __init__(
        self,
        pnl_logger: PnLLogger,
        settlement_hour: int = 16,
        settlement_minute: int = 0,
        timezone_offset: int = 0,
    ):
        """
        Initialize the scheduler.

        Args:
            pnl_logger: P&L logger instance
            settlement_hour: Hour to run settlement (24-hour format)
            settlement_minute: Minute to run settlement
            timezone_offset: Timezone offset from UTC in hours
        """
        self._logger = pnl_logger
        self._settlement_hour = settlement_hour
        self._settlement_minute = settlement_minute
        self._tz_offset = timedelta(hours=timezone_offset)
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._schedule_loop())
        logger.info(
            f"P&L settlement scheduler started "
            f"(daily at {self._settlement_hour:02d}:{self._settlement_minute:02d})"
        )

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("P&L settlement scheduler stopped")

    async def _schedule_loop(self) -> None:
        """Main scheduling loop."""
        while self._running:
            try:
                # Calculate next settlement time
                now = datetime.now(timezone.utc) + self._tz_offset
                settlement_time = datetime.combine(
                    now.date(),
                    time(self._settlement_hour, self._settlement_minute),
                ).replace(tzinfo=timezone.utc)

                # If we've passed today's time, schedule for tomorrow
                if now.time() >= time(self._settlement_hour, self._settlement_minute):
                    settlement_time += timedelta(days=1)

                # Wait until settlement time
                wait_seconds = (settlement_time - now).total_seconds()
                logger.debug(f"Next settlement in {wait_seconds:.0f} seconds")

                await asyncio.sleep(wait_seconds)

                # Run settlement
                if self._running:
                    await self._logger.run_daily_settlement()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Settlement scheduler error: {e}")
                await asyncio.sleep(60)  # Wait a minute before retry
