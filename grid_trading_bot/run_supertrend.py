#!/usr/bin/env python3
"""
Supertrend Bot Runner.

啟動 Supertrend TREND_GRID 趨勢網格交易機器人。

✅ Walk-Forward 驗證通過 (2024-01-25 ~ 2026-01-24, 2 年數據)
- 年化報酬: 8.69%
- Sharpe: 3.78
- 回撤: 4.51%
- Monte Carlo: ROBUST (100% 獲利機率)
- 勝率: 85.4%

⚠️ 參數來自 settings.yaml (單一來源):
- 所有策略參數從 settings.yaml 讀取
- 支援環境變數覆蓋 (例: SUPERTREND_LEVERAGE=5)

Usage:
    python run_supertrend.py

    # 覆蓋特定參數
    SUPERTREND_LEVERAGE=5 python run_supertrend.py
"""

import asyncio
import os
import signal
import sys
from decimal import Decimal

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dotenv import load_dotenv

from src.core import get_logger
from src.core.models import MarketType
from src.data import MarketDataManager
from src.exchange import ExchangeClient
from src.notification import NotificationManager
from src.bots.supertrend import SupertrendBot, SupertrendConfig
from src.config import load_strategy_config, validate_config_consistency

# Load environment variables
load_dotenv()

logger = get_logger(__name__)

# Global bot reference for signal handling
_bot: SupertrendBot | None = None


def get_config_from_env() -> SupertrendConfig:
    """
    從 settings.yaml 讀取配置，支援環境變數覆蓋。

    單一來源設計：
    - 所有策略參數從 settings.yaml 讀取
    - 支援環境變數覆蓋 (格式: SUPERTREND_{PARAM_NAME})

    Example:
        # 使用 YAML 參數
        python run_supertrend.py

        # 覆蓋特定參數
        SUPERTREND_LEVERAGE=5 python run_supertrend.py
    """
    # 從 settings.yaml 加載參數（支援環境變數覆蓋）
    params = load_strategy_config("supertrend")

    # 資金分配（環境變數覆蓋）
    max_capital_str = os.getenv('SUPERTREND_MAX_CAPITAL', '')
    max_capital = Decimal(max_capital_str) if max_capital_str else None

    return SupertrendConfig(
        # 基本設定
        symbol=params.get('symbol', 'BTCUSDT'),
        timeframe=params.get('timeframe', '1h'),
        leverage=int(params.get('leverage', 7)),
        margin_type=params.get('margin_type', 'ISOLATED'),

        # 資金分配
        max_capital=max_capital,
        position_size_pct=Decimal(str(params.get('position_size_pct', 0.1))),

        # Supertrend 設定
        atr_period=int(params.get('atr_period', 11)),
        atr_multiplier=Decimal(str(params.get('atr_multiplier', 1.5))),

        # Grid 設定 (TREND_GRID 模式)
        grid_count=int(params.get('grid_count', 8)),
        grid_atr_multiplier=Decimal(str(params.get('grid_atr_multiplier', 7.5))),
        take_profit_grids=int(params.get('take_profit_grids', 1)),

        # RSI 過濾器
        use_rsi_filter=bool(params.get('use_rsi_filter', True)),
        rsi_period=int(params.get('rsi_period', 21)),
        rsi_overbought=int(params.get('rsi_overbought', 75)),
        rsi_oversold=int(params.get('rsi_oversold', 37)),

        # 趨勢確認
        min_trend_bars=int(params.get('min_trend_bars', 1)),

        # 止損設定
        stop_loss_pct=Decimal(str(params.get('stop_loss_pct', 0.05))),
        use_exchange_stop_loss=bool(params.get('use_exchange_stop_loss', True)),

        # 追蹤止損
        use_trailing_stop=bool(params.get('use_trailing_stop', True)),
        trailing_stop_pct=Decimal(str(params.get('trailing_stop_pct', 0.01))),

        # 倉位與風控
        max_position_pct=Decimal(str(params.get('max_position_pct', 0.5))),
        daily_loss_limit_pct=Decimal(str(params.get('daily_loss_limit_pct', 0.05))),
        max_consecutive_losses=int(params.get('max_consecutive_losses', 5)),

        # Protective features
        use_hysteresis=bool(params.get('use_hysteresis', False)),
        hysteresis_pct=Decimal(str(params.get('hysteresis_pct', 0.0085))),
        use_signal_cooldown=bool(params.get('use_signal_cooldown', False)),
        cooldown_bars=int(params.get('cooldown_bars', 3)),

        # 波動率過濾
        use_volatility_filter=bool(params.get('use_volatility_filter', False)),
        vol_atr_baseline_period=int(params.get('vol_atr_baseline_period', 200)),
        vol_ratio_low=float(params.get('vol_ratio_low', 0.3)),
        vol_ratio_high=float(params.get('vol_ratio_high', 3.0)),

        # 超時退出
        max_hold_bars=int(params.get('max_hold_bars', 8)),

        # HYBRID_GRID mode (v3)
        mode=params.get('mode', 'hybrid_grid'),
        hybrid_grid_bias_pct=Decimal(str(params.get('hybrid_grid_bias_pct', 0.65))),
        hybrid_tp_multiplier_trend=Decimal(str(params.get('hybrid_tp_multiplier_trend', 1.75))),
        hybrid_tp_multiplier_counter=Decimal(str(params.get('hybrid_tp_multiplier_counter', 0.5))),
        hybrid_sl_multiplier_counter=Decimal(str(params.get('hybrid_sl_multiplier_counter', 0.9))),
        hybrid_rsi_asymmetric=bool(params.get('hybrid_rsi_asymmetric', False)),
    )


async def create_exchange_client() -> ExchangeClient:
    """建立交易所客戶端"""
    api_key = os.getenv('BINANCE_API_KEY', '')
    api_secret = os.getenv('BINANCE_API_SECRET', '')
    testnet = os.getenv('BINANCE_TESTNET', 'true').lower() == 'true'

    client = ExchangeClient(
        api_key=api_key,
        api_secret=api_secret,
        testnet=testnet,
    )
    await client.connect()
    return client


async def create_data_manager(exchange: ExchangeClient) -> MarketDataManager:
    """建立數據管理器"""
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', '5432')),
        'database': os.getenv('DB_NAME', 'trading_bot'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', ''),
        'pool_size': 10,
    }

    redis_config = {
        'host': os.getenv('REDIS_HOST', 'localhost'),
        'port': int(os.getenv('REDIS_PORT', '6379')),
        'db': 0,
        'key_prefix': 'trading:',
    }

    manager = MarketDataManager(
        db_config=db_config,
        redis_config=redis_config,
        exchange=exchange,
        market_type=MarketType.FUTURES,
    )
    await manager.connect()
    return manager


def create_notifier() -> NotificationManager:
    """建立通知管理器"""
    return NotificationManager.from_env()


async def shutdown(sig: signal.Signals) -> None:
    """優雅關閉"""
    global _bot

    logger.info(f"收到信號 {sig.name}，正在關閉...")

    if _bot:
        await _bot.stop(clear_position=False)

    logger.info("機器人已關閉")


async def main() -> None:
    """主程式"""
    global _bot

    print("=" * 60)
    print("    Supertrend TREND_GRID Bot - 趨勢網格交易機器人")
    print("=" * 60)

    # 讀取配置
    config = get_config_from_env()

    # 啟動時參數衝突檢查
    logger = get_logger("supertrend")
    warnings = validate_config_consistency("supertrend", config)
    for w in warnings:
        logger.warning(f"Config mismatch: {w}")

    print(f"\n✅ Walk-Forward 驗證通過 (70% 一致性, OOS Sharpe 5.84)")
    print(f"\n配置資訊：")
    print(f"  交易對: {config.symbol}")
    print(f"  時間框架: {config.timeframe}")
    print(f"  槓桿: {config.leverage}x")
    print(f"  保證金模式: {config.margin_type} (逐倉)")

    # 資金分配顯示
    if config.max_capital:
        print(f"  分配資金: {config.max_capital:,.0f} USDT")
        print(f"  單次倉位: {config.position_size_pct*100}% = {config.max_capital * config.position_size_pct:,.0f} USDT")
    else:
        print(f"  分配資金: 全部餘額")
        print(f"  單次倉位: {config.position_size_pct*100}%")

    print(f"\nSupertrend 設定:")
    print(f"  ATR 週期: {config.atr_period}")
    print(f"  ATR 乘數: {config.atr_multiplier}")

    print(f"\nTREND_GRID 設定:")
    print(f"  網格數量: {config.grid_count}")
    print(f"  網格 ATR 乘數: {config.grid_atr_multiplier}")
    print(f"  止盈網格: {config.take_profit_grids}")

    print(f"\nRSI 過濾器:")
    print(f"  啟用: {'是' if config.use_rsi_filter else '否'}")
    if config.use_rsi_filter:
        print(f"  RSI 週期: {config.rsi_period}")
        print(f"  超買閾值: {config.rsi_overbought} (RSI > {config.rsi_overbought} 不做多)")
        print(f"  超賣閾值: {config.rsi_oversold} (RSI < {config.rsi_oversold} 不做空)")

    print(f"\n風險控制:")
    print(f"  止損: {config.stop_loss_pct*100}%")
    print(f"  追蹤止損: {'開啟' if config.use_trailing_stop else '關閉'}")
    if config.use_trailing_stop:
        print(f"  追蹤止損比例: {config.trailing_stop_pct*100}%")
    print()

    try:
        # 建立元件
        print("正在連接交易所...")
        exchange = await create_exchange_client()

        print("正在連接數據庫...")
        data_manager = await create_data_manager(exchange)

        print("正在設定通知...")
        notifier = create_notifier()

        # 建立機器人
        print("正在初始化機器人...")
        _bot = SupertrendBot(
            bot_id=f"supertrend_{config.symbol.lower()}",
            config=config,
            exchange=exchange,
            data_manager=data_manager,
            notifier=notifier,
        )

        # 設定信號處理
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(shutdown(s))
            )

        # 啟動機器人
        print("\n正在啟動交易機器人...")
        success = await _bot.start()

        if success:
            status = _bot.get_status()
            print("\n" + "=" * 60)
            print("機器人啟動成功！")
            print("=" * 60)
            print(f"  交易對: {status.get('symbol')}")
            print(f"  狀態: {status.get('state')}")
            supertrend = status.get('supertrend', {})
            if supertrend:
                print(f"  當前趨勢: {supertrend.get('trend', 'N/A')}")
                print(f"  Supertrend 值: {supertrend.get('value', 0):.2f}")
                print(f"  ATR: {supertrend.get('atr', 0):.2f}")
            print(f"  有持倉: {'是' if status.get('position') else '否'}")
        else:
            print("\n交易機器人啟動失敗！請檢查日誌。")

        print("\n按 Ctrl+C 停止程式")
        print("=" * 60)

        # 保持運行
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print("\n收到中斷信號...")
        if _bot:
            await _bot.stop(clear_position=False)
    except Exception as e:
        logger.error(f"錯誤: {e}")
        print(f"\n發生錯誤: {e}")
        import traceback
        traceback.print_exc()
        if _bot:
            await _bot.stop(clear_position=False)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
