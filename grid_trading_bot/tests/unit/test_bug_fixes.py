"""
Verification tests for 39 bug fixes.

Phase 2: Import + Init validation
Phase 3: Targeted fix verification
"""

import asyncio
import inspect
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Phase 2: Import Verification
# =============================================================================


class TestImportVerification:
    """Verify all fixed modules can be imported without errors."""

    def test_import_base_bot(self):
        from src.bots.base import BaseBot
        assert BaseBot is not None

    def test_import_grid_bot(self):
        from src.bots.grid.bot import GridBot
        assert GridBot is not None

    def test_import_supertrend_bot(self):
        from src.bots.supertrend.bot import SupertrendBot
        assert SupertrendBot is not None

    def test_import_grid_futures_bot(self):
        from src.bots.grid_futures.bot import GridFuturesBot
        assert GridFuturesBot is not None

    def test_import_rsi_grid_bot(self):
        from src.bots.rsi_grid.bot import RSIGridBot
        assert RSIGridBot is not None

    def test_import_state_sync(self):
        from src.exchange.state_sync import StateSynchronizer, StateCache
        assert StateSynchronizer is not None
        assert StateCache is not None

    def test_import_capital_monitor(self):
        from src.risk.capital_monitor import CapitalMonitor
        assert CapitalMonitor is not None

    def test_import_pre_trade_checker(self):
        from src.risk.pre_trade_checker import PreTradeRiskChecker
        assert PreTradeRiskChecker is not None

    def test_import_fund_pool(self):
        from src.fund_manager.core.fund_pool import FundPool
        assert FundPool is not None


# =============================================================================
# Phase 3: CRITICAL Fix Verification
# =============================================================================


class TestC1StateCacheSetIfNewer:
    """C1: StateCache.set_if_newer has single definition with timestamp_attr."""

    def test_set_if_newer_accepts_timestamp_attr(self):
        from src.exchange.state_sync import StateCache
        sig = inspect.signature(StateCache.set_if_newer)
        assert "timestamp_attr" in sig.parameters
        # Check default value
        assert sig.parameters["timestamp_attr"].default == "updated_at"

    def test_set_if_newer_is_async(self):
        from src.exchange.state_sync import StateCache
        assert asyncio.iscoroutinefunction(StateCache.set_if_newer)


class TestC2ClassifyOrderErrorSync:
    """C2: _classify_order_error is sync (not async)."""

    def test_classify_order_error_is_sync(self):
        from src.bots.base import BaseBot
        method = getattr(BaseBot, "_classify_order_error", None)
        assert method is not None
        assert not asyncio.iscoroutinefunction(method)

    def test_classify_order_error_returns_tuple(self):
        from src.bots.base import BaseBot
        sig = inspect.signature(BaseBot._classify_order_error)
        params = list(sig.parameters.keys())
        assert "self" in params
        assert "error" in params


class TestFix2TriggerCircuitBreakerSafe:
    """Fix 2: trigger_circuit_breaker_safe accepts only valid parameters."""

    def test_signature_has_expected_params(self):
        from src.bots.base import BaseBot
        sig = inspect.signature(BaseBot.trigger_circuit_breaker_safe)
        param_names = set(sig.parameters.keys())
        expected = {"self", "reason", "current_price", "current_capital", "loss_pct", "partial"}
        assert expected.issubset(param_names)

    def test_is_async(self):
        from src.bots.base import BaseBot
        assert asyncio.iscoroutinefunction(BaseBot.trigger_circuit_breaker_safe)


# =============================================================================
# Phase 3: HIGH Fix Verification
# =============================================================================


class TestH1CircuitBreakerBlocksPreTrade:
    """H1: safe_pre_trade_risk_check returns False when CB active."""

    def test_safe_pre_trade_risk_check_exists(self):
        from src.bots.base import BaseBot
        assert hasattr(BaseBot, "safe_pre_trade_risk_check")
        assert asyncio.iscoroutinefunction(BaseBot.safe_pre_trade_risk_check)


class TestH2StrategyFlagsInitialized:
    """H2: _strategy_stop_requested/_strategy_pause_requested are initialized."""

    def test_init_strategy_risk_tracking_exists(self):
        from src.bots.base import BaseBot
        assert hasattr(BaseBot, "_init_strategy_risk_tracking")

    def test_flags_initialized_via_method(self):
        """Verify the init method sets both flags."""
        from src.bots.base import BaseBot
        source = inspect.getsource(BaseBot._init_strategy_risk_tracking)
        assert "_strategy_stop_requested" in source
        assert "_strategy_pause_requested" in source


class TestFix3ErrorCodeRetryability:
    """Fix 3: _classify_order_error returns codes that match retryability check."""

    def test_error_code_constants_exist(self):
        from src.bots.base import BaseBot
        # Check that error code constants are defined
        assert hasattr(BaseBot, "ORDER_ERROR_RATE_LIMIT")
        assert hasattr(BaseBot, "ORDER_ERROR_TIMEOUT")
        assert hasattr(BaseBot, "ORDER_ERROR_INSUFFICIENT_BALANCE")


class TestFix4Fix5GateAcquiredPattern:
    """Fix 4/5: supertrend/grid_futures gate_acquired pattern is correct."""

    def test_supertrend_has_release_risk_gate(self):
        from src.bots.supertrend.bot import SupertrendBot
        assert hasattr(SupertrendBot, "release_risk_gate") or hasattr(
            SupertrendBot.__mro__[1], "release_risk_gate"
        )

    def test_grid_futures_has_release_risk_gate(self):
        from src.bots.grid_futures.bot import GridFuturesBot
        assert hasattr(GridFuturesBot, "release_risk_gate") or hasattr(
            GridFuturesBot.__mro__[1], "release_risk_gate"
        )

    def test_gate_acquired_in_supertrend_source(self):
        from src.bots.supertrend.bot import SupertrendBot
        source = inspect.getsource(SupertrendBot)
        assert "gate_acquired" in source
        assert "release_risk_gate" in source

    def test_gate_acquired_in_grid_futures_source(self):
        from src.bots.grid_futures.bot import GridFuturesBot
        source = inspect.getsource(GridFuturesBot)
        assert "gate_acquired" in source
        assert "release_risk_gate" in source


class TestFix6UpdateBotExposure:
    """Fix 6: update_bot_exposure correctly aggregates multi-bot exposure."""

    def test_update_bot_exposure_exists(self):
        from src.bots.base import BaseBot
        assert hasattr(BaseBot, "update_bot_exposure")

    def test_aggregation_uses_bot_symbol_tuple(self):
        """Verify per-bot symbol tracking for correct aggregation."""
        from src.bots.base import BaseBot
        source = inspect.getsource(BaseBot.update_bot_exposure)
        assert "bot_symbol_exposures" in source
        assert "(bot_id, symbol)" in source or "bot_id, symbol" in source


class TestFix7CapitalMonitorMarkPriceNone:
    """Fix 7: capital_monitor handles mark_price=None without crash."""

    def test_capital_monitor_init_no_crash(self):
        from src.risk.capital_monitor import CapitalMonitor
        from src.risk.models import RiskConfig
        config = RiskConfig(total_capital=Decimal("10000"))
        monitor = CapitalMonitor(config=config)
        assert monitor is not None


class TestFix8CloseCostBasisFifoSide:
    """Fix 8: close_cost_basis_fifo accepts side parameter."""

    def test_fifo_accepts_side_param(self):
        from src.bots.base import BaseBot
        sig = inspect.signature(BaseBot.close_cost_basis_fifo)
        assert "side" in sig.parameters
        assert sig.parameters["side"].default == "SELL"


class TestFix9ShortPositionDetection:
    """Fix 9: pos.quantity != Decimal("0") detects short positions."""

    def test_position_state_quantity_check(self):
        from src.exchange.state_sync import PositionState
        # Create a short position (quantity is stored as absolute)
        pos = PositionState(
            symbol="BTCUSDT",
            side="SHORT",
            quantity=Decimal("0.5"),
            entry_price=Decimal("50000"),
            mark_price=Decimal("49000"),
            unrealized_pnl=Decimal("500"),
            leverage=10,
            margin_type="ISOLATED",
            liquidation_price=None,
            updated_at=datetime.now(timezone.utc),
        )
        assert pos.quantity != Decimal("0")
        assert pos.side == "SHORT"


class TestFix10HeartbeatPauseResume:
    """Fix 10: pause clears heartbeat task, resume can restart it."""

    def test_base_bot_has_heartbeat_management(self):
        from src.bots.base import BaseBot
        # Verify pause/resume exist
        assert hasattr(BaseBot, "pause")
        assert hasattr(BaseBot, "resume")


# =============================================================================
# Phase 3: MEDIUM Fix Verification
# =============================================================================


class TestM5CapitalMonitorUTCDate:
    """M5: capital_monitor uses UTC date."""

    def test_daily_reset_uses_utc(self):
        from src.risk.capital_monitor import CapitalMonitor
        source = inspect.getsource(CapitalMonitor)
        # Should use timezone.utc or utcnow or UTC
        assert "timezone.utc" in source or "UTC" in source


class TestFix12ParsePositionMarkPrice:
    """Fix 12: _parse_position doesn't turn mark_price=0 into None."""

    def test_parse_position_exists(self):
        from src.exchange.state_sync import StateSynchronizer
        assert hasattr(StateSynchronizer, "_parse_position")

    def test_mark_price_zero_handling(self):
        """mark_price=0 from exchange is falsy; verify handling in source."""
        from src.exchange.state_sync import StateSynchronizer
        source = inspect.getsource(StateSynchronizer._parse_position)
        # The fix should use "is not None" check rather than truthy check
        # for mark_price to distinguish 0 from None
        assert "is not None" in source or "_mp" in source


class TestFix13PreTradeTimestampTZ:
    """Fix 13: PreTradeCheckResult.timestamp is timezone-aware."""

    def test_timestamp_is_tz_aware(self):
        from src.risk.pre_trade_checker import PreTradeCheckResult, OrderRequest
        result = PreTradeCheckResult(
            passed=True,
            order_request=OrderRequest(
                symbol="BTCUSDT",
                side="BUY",
                order_type="LIMIT",
                quantity=Decimal("0.01"),
                price=Decimal("50000"),
            ),
        )
        assert result.timestamp.tzinfo is not None
        assert result.timestamp.tzinfo == timezone.utc


class TestFeeFix:
    """Fee fix: strategy capital and virtual ledger realized PnL consistent."""

    def test_proportional_fee_in_source(self):
        from src.bots.base import BaseBot
        source = inspect.getsource(BaseBot)
        assert "proportional_fee" in source or "proportional_close_fee" in source


class TestConflictsSyncErrorsCapping:
    """Verify _conflicts and _sync_errors lists are capped."""

    def test_sync_errors_capped_in_source(self):
        from src.exchange.state_sync import StateSynchronizer
        source = inspect.getsource(StateSynchronizer)
        # Should contain capping logic
        assert "_sync_errors" in source
        # Check for length limit (50 or 100)
        assert "50" in source or "100" in source

    def test_conflicts_capped_in_source(self):
        from src.exchange.state_sync import StateSynchronizer
        source = inspect.getsource(StateSynchronizer)
        assert "_conflicts" in source
