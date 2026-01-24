"""
Bollinger BB_TREND_GRID Trading Bot.

✅ Walk-Forward 驗證通過 (2024-01 ~ 2026-01, 2 年數據, 10 期分割):
- Walk-Forward 一致性: 80% (8/10 時段獲利)
- OOS Sharpe: 6.56
- 過度擬合: 未檢測到
- 穩健性: ROBUST

策略邏輯 (BB_TREND_GRID):
- 趨勢判斷: BB 中軌 (SMA)
- 進場: 網格交易，K線觸及網格線時進場
- 出場: 止盈 1 個網格 或 止損 5%
"""

from .models import (
    BBWData,
    BollingerBands,
    BollingerBotStats,
    BollingerConfig,
    ExitReason,
    GridLevel,
    GridLevelState,
    GridSetup,
    Position,
    PositionSide,
    SignalType,
    StrategyMode,
    TradeRecord,
)
from .indicators import (
    BandPosition,
    BollingerCalculator,
    InsufficientDataError,
)
from .bot import BollingerBot

__all__ = [
    # Models
    "BollingerConfig",
    "BollingerBands",
    "BBWData",
    "SignalType",
    "StrategyMode",
    "PositionSide",
    "Position",
    "TradeRecord",
    "BollingerBotStats",
    "ExitReason",
    "GridLevel",
    "GridLevelState",
    "GridSetup",
    # Indicators
    "BollingerCalculator",
    "BandPosition",
    "InsufficientDataError",
    # Bot
    "BollingerBot",
]
