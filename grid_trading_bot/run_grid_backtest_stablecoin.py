"""
Grid Bot Backtest - 穩定幣模擬（區間震盪市場）

由於 Binance 期貨沒有純穩定幣對，這裡模擬一個區間震盪的市場環境，
展示 Grid Bot 在適合的市場條件下的表現。

模擬設定：
- 價格在 0.98 - 1.02 之間震盪（±2% 波動）
- 模擬 180 天的 1h K 線數據
- 對比 Grid Bot 和持有策略
"""

import random
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from typing import List, Tuple


@dataclass
class SimulatedKline:
    """模擬 K 線數據"""
    open_time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def generate_range_bound_prices(
    center_price: float = 1.0,
    volatility_pct: float = 0.02,  # ±2%
    days: int = 180,
    hours_per_day: int = 24,
    seed: int = 42,
) -> List[SimulatedKline]:
    """
    生成區間震盪的價格數據（模擬穩定幣對）

    使用均值回歸的隨機過程，價格會在中心價格附近震盪
    """
    random.seed(seed)

    klines = []
    current_price = center_price
    start_time = datetime(2023, 7, 1, 0, 0, 0)

    upper_bound = center_price * (1 + volatility_pct)
    lower_bound = center_price * (1 - volatility_pct)

    for day in range(days):
        for hour in range(hours_per_day):
            # 均值回歸隨機遊走
            # 越接近邊界，越傾向於回歸中心
            distance_from_center = (current_price - center_price) / (center_price * volatility_pct)
            mean_reversion_force = -distance_from_center * 0.1

            # 隨機變動 + 均值回歸
            random_change = random.gauss(0, volatility_pct / 10)
            price_change = mean_reversion_force + random_change

            open_price = current_price
            close_price = current_price * (1 + price_change)

            # 確保價格在範圍內
            close_price = max(lower_bound * 0.99, min(upper_bound * 1.01, close_price))

            # 生成 high/low
            intra_volatility = abs(close_price - open_price) + current_price * 0.001
            high_price = max(open_price, close_price) + random.uniform(0, intra_volatility)
            low_price = min(open_price, close_price) - random.uniform(0, intra_volatility)

            kline = SimulatedKline(
                open_time=start_time + timedelta(days=day, hours=hour),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=random.uniform(1000, 10000),
            )
            klines.append(kline)

            current_price = close_price

    return klines


def run_grid_bot_backtest(
    klines: List[SimulatedKline],
    initial_capital: float,
    grid_count: int,
    upper_price: float,
    lower_price: float,
    leverage: int = 1,
    position_pct: float = 0.1,
) -> dict:
    """
    運行 Grid Bot 回測
    """
    capital = initial_capital
    position = 0.0  # 持有的數量
    avg_entry_price = 0.0

    # 計算網格
    grid_spacing = (upper_price - lower_price) / grid_count
    grids = []
    for i in range(grid_count + 1):
        price = lower_price + i * grid_spacing
        grids.append({
            "price": price,
            "buy_filled": False,
            "sell_filled": False,
        })

    total_trades = 0
    grid_profit = 0.0
    winning_trades = 0
    losing_trades = 0

    # 追蹤資金曲線
    equity_curve = []
    max_equity = initial_capital
    max_drawdown = 0.0

    for kline in klines:
        current_price = kline.close

        # 計算當前權益
        unrealized_pnl = position * (current_price - avg_entry_price) if position != 0 else 0
        current_equity = capital + unrealized_pnl
        equity_curve.append(current_equity)

        # 更新最大回撤
        if current_equity > max_equity:
            max_equity = current_equity
        drawdown = (max_equity - current_equity) / max_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

        # 檢查每個網格
        for i, grid in enumerate(grids):
            grid_price = grid["price"]

            # 買入條件：價格低於網格價格且未買入
            if current_price <= grid_price and not grid["buy_filled"]:
                # 計算買入數量
                trade_value = capital * position_pct
                quantity = trade_value / current_price

                if trade_value > 10:  # 最小交易金額
                    # 更新平均成本
                    total_value = position * avg_entry_price + quantity * current_price
                    position += quantity
                    avg_entry_price = total_value / position if position > 0 else 0
                    capital -= trade_value

                    grid["buy_filled"] = True
                    grid["sell_filled"] = False
                    total_trades += 1

            # 賣出條件：價格高於網格價格且已買入
            elif current_price >= grid_price and grid["buy_filled"] and not grid["sell_filled"]:
                # 計算賣出數量（賣出對應網格的買入量）
                sell_quantity = min(position, capital * position_pct / grid_price)

                if sell_quantity > 0 and position > 0:
                    # 計算這筆交易的盈虧
                    trade_pnl = sell_quantity * (current_price - avg_entry_price)
                    grid_profit += sell_quantity * grid_spacing  # 網格利潤

                    if trade_pnl > 0:
                        winning_trades += 1
                    else:
                        losing_trades += 1

                    capital += sell_quantity * current_price
                    position -= sell_quantity

                    grid["sell_filled"] = True
                    grid["buy_filled"] = False
                    total_trades += 1

    # 最終平倉
    final_price = klines[-1].close
    if position > 0:
        capital += position * final_price
        position = 0

    final_equity = capital
    total_return = (final_equity - initial_capital) / initial_capital

    # 計算 Sharpe Ratio
    if len(equity_curve) > 1:
        returns = []
        for i in range(1, len(equity_curve)):
            ret = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            returns.append(ret)

        if returns:
            avg_return = sum(returns) / len(returns)
            variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
            std_return = math.sqrt(variance) if variance > 0 else 0.0001

            # 年化 (假設 1h K 線)
            annual_factor = math.sqrt(365 * 24)
            sharpe = (avg_return / std_return) * annual_factor if std_return > 0 else 0
        else:
            sharpe = 0
    else:
        sharpe = 0

    win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0

    return {
        "initial_capital": initial_capital,
        "final_equity": final_equity,
        "total_return": total_return,
        "total_trades": total_trades,
        "grid_profit": grid_profit,
        "winning_trades": winning_trades,
        "losing_trades": losing_trades,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "sharpe_ratio": sharpe,
        "start_price": klines[0].close,
        "end_price": klines[-1].close,
        "price_change": (klines[-1].close - klines[0].close) / klines[0].close,
    }


def run_hold_strategy(
    klines: List[SimulatedKline],
    initial_capital: float,
) -> dict:
    """持有策略回測"""
    start_price = klines[0].close
    end_price = klines[-1].close

    quantity = initial_capital / start_price
    final_value = quantity * end_price
    total_return = (final_value - initial_capital) / initial_capital

    return {
        "initial_capital": initial_capital,
        "final_equity": final_value,
        "total_return": total_return,
        "price_change": (end_price - start_price) / start_price,
    }


def main():
    print("=" * 70)
    print("Grid Bot 回測 - 穩定幣模擬（區間震盪市場）")
    print("=" * 70)

    # 測試參數
    initial_capital = 10000.0
    days = 180

    # 測試不同的波動率設定
    test_cases = [
        {"name": "穩定幣模擬 (±1%)", "volatility": 0.01, "center": 1.0},
        {"name": "穩定幣模擬 (±2%)", "volatility": 0.02, "center": 1.0},
        {"name": "低波動資產 (±5%)", "volatility": 0.05, "center": 100.0},
    ]

    for case in test_cases:
        print(f"\n{'='*70}")
        print(f"測試: {case['name']}")
        print(f"{'='*70}")

        # 生成價格數據
        klines = generate_range_bound_prices(
            center_price=case["center"],
            volatility_pct=case["volatility"],
            days=days,
            seed=42,
        )

        # 設定網格參數
        center = case["center"]
        vol = case["volatility"]
        upper_price = center * (1 + vol)
        lower_price = center * (1 - vol)
        grid_count = 10

        print(f"\n市場設定:")
        print(f"  中心價格: {center}")
        print(f"  價格範圍: {lower_price:.4f} - {upper_price:.4f}")
        print(f"  波動率: ±{vol*100}%")
        print(f"  測試天數: {days} 天")
        print(f"  K 線數量: {len(klines)}")

        print(f"\nGrid Bot 設定:")
        print(f"  網格數量: {grid_count}")
        print(f"  網格間距: {(upper_price - lower_price) / grid_count:.6f}")
        print(f"  單次交易: 10% 資金")

        # 運行 Grid Bot 回測
        grid_result = run_grid_bot_backtest(
            klines=klines,
            initial_capital=initial_capital,
            grid_count=grid_count,
            upper_price=upper_price,
            lower_price=lower_price,
            leverage=1,
            position_pct=0.1,
        )

        # 運行持有策略
        hold_result = run_hold_strategy(klines, initial_capital)

        # 輸出結果
        print(f"\n{'='*40}")
        print("Grid Bot 結果:")
        print(f"{'='*40}")
        print(f"  初始資金: ${initial_capital:,.2f}")
        print(f"  最終資金: ${grid_result['final_equity']:,.2f}")
        print(f"  總收益: ${grid_result['final_equity'] - initial_capital:,.2f}")
        print(f"  ROI: {grid_result['total_return']*100:.2f}%")
        print(f"  網格利潤: ${grid_result['grid_profit']:,.2f}")
        print(f"  總交易次數: {grid_result['total_trades']}")
        print(f"  勝率: {grid_result['win_rate']*100:.1f}%")
        print(f"  最大回撤: {grid_result['max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {grid_result['sharpe_ratio']:.2f}")

        print(f"\n{'='*40}")
        print("持有策略結果:")
        print(f"{'='*40}")
        print(f"  初始資金: ${initial_capital:,.2f}")
        print(f"  最終資金: ${hold_result['final_equity']:,.2f}")
        print(f"  總收益: ${hold_result['final_equity'] - initial_capital:,.2f}")
        print(f"  ROI: {hold_result['total_return']*100:.2f}%")
        print(f"  價格變化: {hold_result['price_change']*100:.2f}%")

        print(f"\n{'='*40}")
        print("策略比較:")
        print(f"{'='*40}")
        outperform = grid_result['total_return'] - hold_result['total_return']
        print(f"  Grid Bot 超額收益: {outperform*100:.2f}%")
        if outperform > 0:
            print(f"  ✓ Grid Bot 表現優於持有策略")
        else:
            print(f"  ✗ 持有策略表現優於 Grid Bot")

    # 對比不同市場條件
    print("\n" + "=" * 70)
    print("總結：Grid Bot 適合的市場條件")
    print("=" * 70)
    print("""
    Grid Bot 策略分析：

    ✓ 適合市場：
      - 區間震盪市場（穩定幣對、ETF 等）
      - 低波動、價格圍繞中心震盪的資產
      - 沒有明顯趨勢的市場

    ✗ 不適合市場：
      - 強趨勢市場（如 BTC 牛市/熊市）
      - 高波動突破型市場
      - 價格單邊上漲或下跌的市場

    結論：
    - 在區間震盪市場，Grid Bot 可以穩定獲利
    - 在趨勢市場，Grid Bot 會因為持倉方向錯誤而虧損
    - 選擇合適的交易標的是 Grid Bot 成功的關鍵
    """)


if __name__ == "__main__":
    main()
