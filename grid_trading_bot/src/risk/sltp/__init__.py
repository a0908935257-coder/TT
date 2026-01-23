"""
Stop Loss / Take Profit Management Module.

Provides unified SLTP management for all trading bots:
- Centralized SLTP calculations
- Multiple SL/TP types support
- Trailing stop management
- Exchange order integration
"""

from src.risk.sltp.calculator import SLTPCalculator
from src.risk.sltp.exchange_adapter import (
    ExchangeClient,
    MockExchangeAdapter,
    SLTPExchangeAdapter,
)
from src.risk.sltp.manager import ExchangeAdapter, SLTPManager
from src.risk.sltp.models import (
    SLTPConfig,
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
    # Types/Enums
    "StopLossType",
    "TakeProfitType",
    "TrailingStopType",
    # Configs
    "StopLossConfig",
    "TakeProfitConfig",
    "TakeProfitLevel",
    "TrailingStopConfig",
    "SLTPConfig",
    # State
    "SLTPState",
    # Calculator
    "SLTPCalculator",
    # Manager
    "SLTPManager",
    "ExchangeAdapter",
    # Exchange Adapters
    "ExchangeClient",
    "SLTPExchangeAdapter",
    "MockExchangeAdapter",
]
