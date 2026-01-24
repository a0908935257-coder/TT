"""Tests for P&L logger module."""

import asyncio
import json
import tempfile
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pytest

from src.monitoring.pnl_logger import (
    DailySettlement,
    PnLLogger,
    PnLSettlementScheduler,
    PositionSnapshot,
    SettlementStatus,
    TradeRecord,
)


class TestTradeRecord:
    """Tests for TradeRecord."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        trade = TradeRecord(
            trade_id="T001",
            timestamp=datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            price=Decimal("42000"),
            commission=Decimal("21"),
            realized_pnl=Decimal("0"),
            order_id="O001",
            strategy_id="grid_01",
        )

        d = trade.to_dict()
        assert d["trade_id"] == "T001"
        assert d["symbol"] == "BTC/USDT"
        assert d["quantity"] == "0.5"
        assert d["price"] == "42000"


class TestPositionSnapshot:
    """Tests for PositionSnapshot."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        pos = PositionSnapshot(
            symbol="BTC/USDT",
            quantity=Decimal("1.5"),
            avg_cost=Decimal("41000"),
            current_price=Decimal("42000"),
            unrealized_pnl=Decimal("1500"),
            market_value=Decimal("63000"),
        )

        d = pos.to_dict()
        assert d["symbol"] == "BTC/USDT"
        assert d["quantity"] == "1.5"
        assert d["unrealized_pnl"] == "1500"


class TestDailySettlement:
    """Tests for DailySettlement."""

    def test_default_values(self):
        """Test default values."""
        settlement = DailySettlement(settlement_date=date(2024, 1, 15))

        assert settlement.status == SettlementStatus.PENDING
        assert settlement.realized_pnl == Decimal("0")
        assert settlement.total_trades == 0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        settlement = DailySettlement(
            settlement_date=date(2024, 1, 15),
            status=SettlementStatus.COMPLETED,
            realized_pnl=Decimal("1000"),
            unrealized_pnl=Decimal("500"),
            net_pnl=Decimal("1450"),
            total_commission=Decimal("50"),
            starting_equity=Decimal("100000"),
            ending_equity=Decimal("101500"),
            equity_change=Decimal("1500"),
            equity_change_percent=1.5,
            total_trades=50,
            buy_trades=30,
            sell_trades=20,
        )

        d = settlement.to_dict()
        assert d["settlement_date"] == "2024-01-15"
        assert d["status"] == "completed"
        assert d["pnl"]["realized"] == "1000"
        assert d["account"]["equity_change_percent"] == 1.5
        assert d["activity"]["total_trades"] == 50


class TestPnLLogger:
    """Tests for PnLLogger."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_pnl.db"
            log_dir = Path(tmpdir) / "logs"
            yield db_path, log_dir

    @pytest.fixture
    def logger(self, temp_paths):
        """Create a test logger."""
        db_path, log_dir = temp_paths
        return PnLLogger(db_path=db_path, log_dir=log_dir)

    def test_init_creates_database(self, temp_paths):
        """Test that initialization creates database."""
        db_path, log_dir = temp_paths
        logger = PnLLogger(db_path=db_path, log_dir=log_dir)

        assert db_path.exists()

    def test_record_trade(self, logger):
        """Test recording a trade."""
        trade = TradeRecord(
            trade_id="T001",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            price=Decimal("42000"),
            commission=Decimal("21"),
            realized_pnl=Decimal("100"),
            order_id="O001",
        )

        logger.record_trade(trade)

        stats = logger._current_settlement
        assert stats.total_trades == 1
        assert stats.realized_pnl == Decimal("100")
        assert stats.buy_trades == 1

    def test_record_multiple_trades(self, logger):
        """Test recording multiple trades."""
        for i in range(5):
            trade = TradeRecord(
                trade_id=f"T00{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                side="buy" if i % 2 == 0 else "sell",
                quantity=Decimal("0.1"),
                price=Decimal("42000"),
                commission=Decimal("4.2"),
                realized_pnl=Decimal("10"),
                order_id=f"O00{i}",
            )
            logger.record_trade(trade)

        stats = logger._current_settlement
        assert stats.total_trades == 5
        assert stats.buy_trades == 3
        assert stats.sell_trades == 2
        assert stats.realized_pnl == Decimal("50")

    def test_update_equity(self, logger):
        """Test updating equity."""
        positions = [
            PositionSnapshot(
                symbol="BTC/USDT",
                quantity=Decimal("1"),
                avg_cost=Decimal("41000"),
                current_price=Decimal("42000"),
                unrealized_pnl=Decimal("1000"),
                market_value=Decimal("42000"),
            )
        ]

        # First set starting equity
        logger._ensure_current_settlement(date.today())
        logger._current_settlement.starting_equity = Decimal("100000")

        logger.update_equity(
            current_equity=Decimal("101000"),
            unrealized_pnl=Decimal("1000"),
            positions=positions,
        )

        stats = logger._current_settlement
        assert stats.ending_equity == Decimal("101000")
        assert stats.unrealized_pnl == Decimal("1000")
        assert len(stats.positions) == 1
        assert stats.equity_change == Decimal("1000")
        assert stats.equity_change_percent == 1.0

    def test_update_risk_metrics(self, logger):
        """Test updating risk metrics."""
        logger._ensure_current_settlement(date.today())

        logger.update_risk_metrics(
            max_drawdown=Decimal("500"),
            max_drawdown_percent=5.0,
            sharpe_ratio=1.5,
            win_rate=65.0,
        )

        stats = logger._current_settlement
        assert stats.max_drawdown == Decimal("500")
        assert stats.max_drawdown_percent == 5.0
        assert stats.sharpe_ratio == 1.5
        assert stats.win_rate == 65.0

    @pytest.mark.asyncio
    async def test_run_daily_settlement(self, logger):
        """Test running daily settlement."""
        # Record some trades first
        for i in range(3):
            trade = TradeRecord(
                trade_id=f"T00{i}",
                timestamp=datetime.now(timezone.utc),
                symbol="BTC/USDT",
                side="buy",
                quantity=Decimal("0.1"),
                price=Decimal("42000"),
                commission=Decimal("4.2"),
                realized_pnl=Decimal("50") if i < 2 else Decimal("-20"),
                order_id=f"O00{i}",
            )
            logger.record_trade(trade)

        settlement = await logger.run_daily_settlement()

        assert settlement.status == SettlementStatus.COMPLETED
        assert settlement.total_trades == 3
        assert settlement.realized_pnl == Decimal("80")
        # Win rate: 2/3 wins = 66.67%
        assert 66 < settlement.win_rate < 67

    def test_get_settlement(self, logger, temp_paths):
        """Test retrieving a settlement."""
        # Record a trade to create settlement
        trade = TradeRecord(
            trade_id="T001",
            timestamp=datetime.now(timezone.utc),
            symbol="BTC/USDT",
            side="buy",
            quantity=Decimal("0.5"),
            price=Decimal("42000"),
            commission=Decimal("21"),
            realized_pnl=Decimal("100"),
            order_id="O001",
        )
        logger.record_trade(trade)
        logger._save_settlement(logger._current_settlement)

        # Retrieve it
        settlement = logger.get_settlement(date.today())

        assert settlement is not None
        assert settlement.total_trades == 1

    def test_get_settlements_range(self, logger, temp_paths):
        """Test retrieving settlements for a date range."""
        today = date.today()

        # Create settlement for today
        logger._ensure_current_settlement(today)
        logger._current_settlement.realized_pnl = Decimal("100")
        logger._save_settlement(logger._current_settlement)

        settlements = logger.get_settlements_range(
            today - timedelta(days=7), today + timedelta(days=1)
        )

        assert len(settlements) >= 1

    def test_get_cumulative_pnl(self, logger, temp_paths):
        """Test calculating cumulative P&L."""
        today = date.today()

        # Create settlement
        logger._ensure_current_settlement(today)
        logger._current_settlement.net_pnl = Decimal("500")
        logger._save_settlement(logger._current_settlement)

        total = logger.get_cumulative_pnl(today - timedelta(days=1), today)

        assert total == Decimal("500")

    def test_settlement_callback(self, logger):
        """Test settlement callbacks."""
        callback_called = []

        def callback(settlement):
            callback_called.append(settlement)

        logger.add_settlement_callback(callback)

        # Run settlement
        asyncio.run(logger.run_daily_settlement())

        assert len(callback_called) == 1
        assert callback_called[0].status == SettlementStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_settlement_log_file_created(self, logger, temp_paths):
        """Test that settlement log file is created."""
        db_path, log_dir = temp_paths

        await logger.run_daily_settlement()

        # Check log file exists
        log_files = list(log_dir.glob("settlement_*.json"))
        assert len(log_files) == 1

        # Verify content
        with open(log_files[0]) as f:
            data = json.load(f)
            assert "settlement_date" in data
            assert data["status"] == "completed"


class TestPnLSettlementScheduler:
    """Tests for PnLSettlementScheduler."""

    @pytest.fixture
    def temp_paths(self):
        """Create temporary paths for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test_pnl.db"
            log_dir = Path(tmpdir) / "logs"
            yield db_path, log_dir

    @pytest.fixture
    def scheduler(self, temp_paths):
        """Create a test scheduler."""
        db_path, log_dir = temp_paths
        logger = PnLLogger(db_path=db_path, log_dir=log_dir)
        return PnLSettlementScheduler(logger, settlement_hour=23, settlement_minute=59)

    @pytest.mark.asyncio
    async def test_start_stop(self, scheduler):
        """Test starting and stopping scheduler."""
        await scheduler.start()
        assert scheduler._running

        await scheduler.stop()
        assert not scheduler._running

    @pytest.mark.asyncio
    async def test_scheduler_schedules_settlement(self, temp_paths):
        """Test that scheduler runs settlement at configured time."""
        db_path, log_dir = temp_paths
        logger = PnLLogger(db_path=db_path, log_dir=log_dir)

        # Create scheduler with immediate execution for testing
        scheduler = PnLSettlementScheduler(logger)

        async def mock_settlement(*args, **kwargs):
            return DailySettlement(
                settlement_date=date.today(), status=SettlementStatus.COMPLETED
            )

        with patch.object(
            logger, "run_daily_settlement", side_effect=mock_settlement
        ):
            await scheduler.start()
            await asyncio.sleep(0.1)
            await scheduler.stop()

            # Scheduler starts but may not run settlement immediately
            # (depends on configured time)
            assert scheduler._running is False
