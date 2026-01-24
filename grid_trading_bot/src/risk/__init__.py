"""
Risk Management Module.

Provides global risk management for the grid trading bot system:
- Capital monitoring
- Drawdown calculation
- Circuit breaker protection
- Emergency stop functionality
- Unified SLTP (Stop Loss / Take Profit) management
- Pre-trade risk validation
- Concentration risk monitoring
"""

from src.risk.capital_monitor import CapitalMonitor
from src.risk.circuit_breaker import CircuitBreaker, CircuitState, CooldownNotFinishedError
from src.risk.concentration_monitor import (
    AssetCategory,
    ConcentrationAlert,
    ConcentrationAlertType,
    ConcentrationConfig,
    ConcentrationLevel,
    ConcentrationMonitor,
    ConcentrationSnapshot,
    PositionInfo,
)
from src.risk.drawdown_calculator import DrawdownCalculator
from src.risk.emergency_stop import EmergencyConfig, EmergencyStop, EmergencyStopStatus
from src.risk.models import (
    CapitalSnapshot,
    CircuitBreakerState,
    DailyPnL,
    DrawdownInfo,
    GlobalRiskStatus,
    RiskAction,
    RiskAlert,
    RiskConfig,
    RiskLevel,
)
from src.risk.pre_trade_checker import (
    CheckDetail,
    CheckResult,
    OrderRequest,
    PreTradeCheckResult,
    PreTradeConfig,
    PreTradeRiskChecker,
    RejectionReason,
)
from src.risk.risk_engine import RiskEngine
from src.risk.sltp import (
    ExchangeAdapter,
    MockExchangeAdapter,
    SLTPCalculator,
    SLTPConfig,
    SLTPExchangeAdapter,
    SLTPManager,
    SLTPState,
    StopLossConfig,
    StopLossType,
    TakeProfitConfig,
    TakeProfitLevel,
    TakeProfitType,
    TrailingStopConfig,
    TrailingStopType,
)

__all__ = [
    # Risk models
    "RiskLevel",
    "RiskAction",
    "RiskConfig",
    "CapitalSnapshot",
    "DrawdownInfo",
    "DailyPnL",
    "RiskAlert",
    "CircuitBreakerState",
    "GlobalRiskStatus",
    # Risk components
    "CapitalMonitor",
    "DrawdownCalculator",
    "CircuitBreaker",
    "CircuitState",
    "CooldownNotFinishedError",
    "EmergencyConfig",
    "EmergencyStop",
    "EmergencyStopStatus",
    "RiskEngine",
    # Pre-trade risk checker
    "PreTradeRiskChecker",
    "PreTradeConfig",
    "PreTradeCheckResult",
    "OrderRequest",
    "CheckResult",
    "CheckDetail",
    "RejectionReason",
    # Concentration monitor
    "ConcentrationMonitor",
    "ConcentrationConfig",
    "ConcentrationSnapshot",
    "ConcentrationAlert",
    "ConcentrationAlertType",
    "ConcentrationLevel",
    "PositionInfo",
    "AssetCategory",
    # SLTP types
    "StopLossType",
    "TakeProfitType",
    "TrailingStopType",
    # SLTP configs
    "StopLossConfig",
    "TakeProfitConfig",
    "TakeProfitLevel",
    "TrailingStopConfig",
    "SLTPConfig",
    # SLTP state
    "SLTPState",
    # SLTP components
    "SLTPCalculator",
    "SLTPManager",
    "ExchangeAdapter",
    "SLTPExchangeAdapter",
    "MockExchangeAdapter",
]
