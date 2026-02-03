# 交易系統完整審計與修復指令

> **文件版本**: 1.0
> **創建日期**: 2026-02-03
> **目標系統**: `/mnt/c/trading/grid_trading_bot/`

## 重要警告

這是金融交易系統，錯誤會導致直接財務損失。請徹底且保守地進行分析，發現問題後立即修復。

## 系統路徑

- 主代碼: `/mnt/c/trading/grid_trading_bot/src/`
- 測試: `/mnt/c/trading/grid_trading_bot/tests/`
- 配置: `/mnt/c/trading/grid_trading_bot/config/`

---

## 系統結構摘要

```
/mnt/c/trading/grid_trading_bot/src/
├── exchange/binance/          # Binance API (futures_api.py, spot_api.py, auth.py, websocket.py)
├── bots/                      # 策略機器人 (grid, grid_futures, supertrend, bollinger, rsi_grid)
├── risk/                      # 風險管理 (risk_engine.py, circuit_breaker.py)
├── core/                      # 核心模型 (models.py, exceptions.py)
├── fund_manager/              # 資金管理
├── data/                      # 數據層 (PostgreSQL + Redis)
├── master/                    # 主控制器
├── execution/                 # 訂單執行
├── monitoring/                # 監控系統
└── config/                    # 配置管理
```

---

## 審計清單 (73 項檢查)

### A. API 端點驗證 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| A-1 | Futures REST URL | `Grep(pattern="fapi.binance.com", path="grid_trading_bot/src/exchange/")` | 僅在 futures_api.py 和 constants.py | 移除錯誤位置的 URL |
| A-2 | Spot REST URL | `Grep(pattern="api.binance.com", path="grid_trading_bot/src/exchange/")` | 僅在 spot_api.py 和 constants.py | 移除錯誤位置的 URL |
| A-3 | Futures WS URL | `Grep(pattern="fstream.binance.com", path="grid_trading_bot/src/exchange/")` | 在 websocket.py 和 constants.py | 確保正確端點 |
| A-4 | Spot WS URL | `Grep(pattern="stream.binance.com", path="grid_trading_bot/src/exchange/")` | 在 websocket.py 和 constants.py | 確保正確端點 |
| A-5 | 無端點污染 | `Grep(pattern="/api/v3/", path="grid_trading_bot/src/exchange/binance/futures_api.py")` | 應無結果 | 移除現貨端點 |
| A-6 | Testnet 隔離 | `Grep(pattern="testnet", path="grid_trading_bot/src/exchange/")` | 僅在 constants.py | 統一到常量檔案 |

### B. 認證與簽名 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| B-1 | recvWindow 值 | `Read("grid_trading_bot/src/exchange/binance/auth.py")` | recvWindow <= 10000ms | 降低到安全值 |
| B-2 | 時間偏移校正 | `Grep(pattern="time_offset", path="grid_trading_bot/src/exchange/")` | client.py 應用偏移 | 確保簽名使用偏移 |
| B-3 | HMAC-SHA256 | `Grep(pattern="hmac.new.*sha256", path="grid_trading_bot/src/exchange/")` | auth.py 中存在 | 使用正確算法 |
| B-4 | API Key Header | `Grep(pattern="X-MBX-APIKEY", path="grid_trading_bot/src/exchange/")` | auth.py 中設置 | 確保正確頭部 |
| B-5 | 時間同步驗證 | `Grep(pattern="_time_sync\|sync_time", path="grid_trading_bot/src/exchange/client.py")` | 有同步機制 | 添加時間同步 |

### C. 買賣方向正確性 (8 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| C-1 | 平多倉方向 | `Grep(pattern="close.*long\|LONG.*SELL", path="grid_trading_bot/src/", -i=true)` | Long 平倉 → SELL | 修正方向邏輯 |
| C-2 | 平空倉方向 | `Grep(pattern="close.*short\|SHORT.*BUY", path="grid_trading_bot/src/", -i=true)` | Short 平倉 → BUY | 修正方向邏輯 |
| C-3 | reduceOnly 使用 | `Grep(pattern="reduce_only.*True\|reduceOnly.*true", path="grid_trading_bot/src/")` | 平倉訂單必須設置 | 添加 reduceOnly |
| C-4 | PositionSide 對沖 | `Grep(pattern="PositionSide\\.LONG\|PositionSide\\.SHORT", path="grid_trading_bot/src/")` | 對沖模式正確 | 修正 PositionSide |
| C-5 | 雙向持倉 | `Grep(pattern="dualSidePosition\|BOTH", path="grid_trading_bot/src/exchange/")` | 支持兩種模式 | 添加模式檢測 |
| C-6 | 反向訂單邏輯 | `Read("grid_trading_bot/src/bots/grid/order_manager.py")` | BUY→SELL, SELL→BUY | 驗證反向邏輯 |
| C-7 | 期貨開倉方向 | `Read("grid_trading_bot/src/bots/grid_futures/bot.py")` | 多頭用 BUY, 空頭用 SELL | 修正開倉方向 |
| C-8 | Supertrend 方向 | `Read("grid_trading_bot/src/bots/supertrend/bot.py")` | 趨勢方向匹配訂單 | 修正策略邏輯 |

### D. 精度處理 (8 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| D-1 | Decimal 使用 | `Grep(pattern="from decimal import", path="grid_trading_bot/src/")` | 金融計算用 Decimal | 替換 float |
| D-2 | float→Decimal 危險 | `Grep(pattern="Decimal\\([^\"'s]", path="grid_trading_bot/src/", type="py")` | 應無直接轉換 | 用 Decimal(str()) |
| D-3 | ROUND_DOWN 數量 | `Grep(pattern="ROUND_DOWN", path="grid_trading_bot/src/bots/")` | 數量用向下取整 | 添加 ROUND_DOWN |
| D-4 | stepSize 應用 | `Grep(pattern="step_size\|stepSize", path="grid_trading_bot/src/")` | 數量符合精度 | 應用 stepSize |
| D-5 | tickSize 應用 | `Grep(pattern="tick_size\|tickSize", path="grid_trading_bot/src/")` | 價格符合精度 | 應用 tickSize |
| D-6 | 科學記號防止 | `Grep(pattern=":f}\|:\..*f}\|normalize", path="grid_trading_bot/src/exchange/")` | 避免 1e-8 格式 | 用 format 或 normalize |
| D-7 | 最小名義價值 | `Grep(pattern="min_notional\|minNotional", path="grid_trading_bot/src/")` | 訂單 > minNotional | 添加驗證 |
| D-8 | 數量精度函數 | `Grep(pattern="def.*quantity.*precision\|def.*format.*quantity", path="grid_trading_bot/src/")` | 有統一函數 | 創建工具函數 |

### E. API 回應解析 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| E-1 | from_binance 解析 | `Grep(pattern="def from_binance\|from_dict", path="grid_trading_bot/src/core/")` | Order, Position 有解析器 | 添加解析方法 |
| E-2 | 空值處理 | `Grep(pattern='\\.get\\(.*None\\)\|or ""', path="grid_trading_bot/src/core/models.py")` | 正確處理缺失字段 | 添加默認值 |
| E-3 | liquidationPrice | `Grep(pattern="liquidation.*price\|liquidationPrice", path="grid_trading_bot/src/")` | 處理 "0" 字串 | 添加特殊處理 |
| E-4 | 時間戳解析 | `Grep(pattern="timestamp.*datetime\|datetime.*timestamp", path="grid_trading_bot/src/")` | 毫秒正確轉換 | 修正轉換邏輯 |
| E-5 | 錯誤碼映射 | `Read("grid_trading_bot/src/exchange/binance/constants.py")` | 完整錯誤碼處理 | 添加缺失錯誤碼 |
| E-6 | HTTP 狀態處理 | `Grep(pattern="status_code\|response\\.status", path="grid_trading_bot/src/exchange/")` | 處理 4xx, 5xx | 添加錯誤處理 |

### F. 去重機制 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| F-1 | Fill 去重 | `Grep(pattern="_processed.*id\|_fill.*id", path="grid_trading_bot/src/bots/")` | 防止重複處理 | 添加去重集合 |
| F-2 | 去重大小限制 | 檢查 F-1 結果 | 有 maxlen 或清理 | 添加大小限制 |
| F-3 | WS 消息去重 | `Grep(pattern="_recent.*id\|_dedup\|_seen", path="grid_trading_bot/src/exchange/")` | WS 消息去重 | 添加去重邏輯 |
| F-4 | 信號冷卻 | `Grep(pattern="cooldown\|_signal_cooldown\|_last_signal", path="grid_trading_bot/src/bots/")` | 防止信號堆疊 | 添加冷卻機制 |
| F-5 | K線級別去重 | `Grep(pattern="_last.*bar\|_bar_processed", path="grid_trading_bot/src/bots/")` | 每K線最多一信號 | 添加K線鎖定 |
| F-6 | 訂單重複檢查 | `Grep(pattern="duplicate.*order\|order.*exist", path="grid_trading_bot/src/")` | 防止重複下單 | 添加訂單檢查 |

### G. 錯誤處理 (7 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| G-1 | 靜默 except | `Grep(pattern="except.*:$", path="grid_trading_bot/src/", output_mode="content", -A=2)` | 無靜默 pass | 添加日誌 |
| G-2 | exc_info 日誌 | `Grep(pattern="logger\\.error.*exc_info\|exception=", path="grid_trading_bot/src/")` | 錯誤包含堆疊 | 添加 exc_info=True |
| G-3 | 重試邏輯 | `Grep(pattern="@.*retry\|RetryConfig\|max_retries", path="grid_trading_bot/src/")` | 關鍵操作有重試 | 添加重試裝飾器 |
| G-4 | 特定異常 | `Grep(pattern="except \\w+Error", path="grid_trading_bot/src/exchange/")` | 捕獲具體異常 | 細化異常類型 |
| G-5 | 異常鏈保留 | `Grep(pattern="raise.*from", path="grid_trading_bot/src/")` | 保留原始異常 | 使用 raise from |
| G-6 | finally 清理 | `Grep(pattern="finally:", path="grid_trading_bot/src/exchange/")` | 資源正確清理 | 添加 finally |
| G-7 | 超時異常 | `Grep(pattern="TimeoutError\|asyncio\\.TimeoutError", path="grid_trading_bot/src/")` | 處理超時 | 添加超時處理 |

### H. 異步與競態條件 (7 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| H-1 | gather return_exceptions | `Grep(pattern="asyncio\\.gather", path="grid_trading_bot/src/", output_mode="content", -A=1)` | return_exceptions=True | 添加參數 |
| H-2 | Lock 保護 | `Grep(pattern="asyncio\\.Lock", path="grid_trading_bot/src/")` | 關鍵操作有鎖 | 添加 Lock |
| H-3 | Task done_callback | `Grep(pattern="add_done_callback", path="grid_trading_bot/src/")` | 背景任務有回調 | 添加回調 |
| H-4 | 訂單隊列鎖 | `Grep(pattern="_order_lock\|order.*lock", path="grid_trading_bot/src/exchange/")` | 訂單執行串行 | 添加訂單鎖 |
| H-5 | 超時保護 | `Grep(pattern="asyncio\\.wait_for\|timeout=", path="grid_trading_bot/src/")` | 異步有超時 | 添加 wait_for |
| H-6 | 任務取消處理 | `Grep(pattern="CancelledError\|task\\.cancel", path="grid_trading_bot/src/")` | 正確處理取消 | 添加取消處理 |
| H-7 | 共享狀態保護 | `Grep(pattern="self\\._\|cls\\._", path="grid_trading_bot/src/bots/base.py", output_mode="content")` | 狀態修改有鎖 | 添加狀態鎖 |

### I. 風險控制 (8 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| I-1 | 止損實現 | `Grep(pattern="stop_loss\|STOP_MARKET", path="grid_trading_bot/src/")` | 有止損機制 | 添加止損 |
| I-2 | MARK_PRICE 止損 | `Grep(pattern="workingType.*MARK\|MARK_PRICE", path="grid_trading_bot/src/")` | SL 用標記價格 | 設置 workingType |
| I-3 | 止損 vs 強平 | `Grep(pattern="liquidation.*stop\|stop.*liquidation", path="grid_trading_bot/src/")` | SL 在強平前 | 添加驗證邏輯 |
| I-4 | 熔斷器 | `Read("grid_trading_bot/src/risk/circuit_breaker.py")` | 熔斷保護存在 | 驗證實現 |
| I-5 | 倉位限制 | `Grep(pattern="max_position\|position_limit", path="grid_trading_bot/src/")` | 有倉位上限 | 添加限制 |
| I-6 | 每日虧損限制 | `Grep(pattern="daily_loss\|daily_limit", path="grid_trading_bot/src/")` | 每日限制 | 添加限制 |
| I-7 | 槓桿驗證 | `Grep(pattern="leverage.*validate\|max_leverage", path="grid_trading_bot/src/")` | 槓桿不超限 | 添加驗證 |
| I-8 | 資金費率避開 | `Grep(pattern="funding.*time\|avoid.*funding", path="grid_trading_bot/src/")` | 避開結算 | 添加時間檢查 |

### J. 狀態管理 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| J-1 | 狀態機定義 | `Read("grid_trading_bot/src/core/models.py")` | BotState 枚舉完整 | 補充狀態 |
| J-2 | 狀態轉換驗證 | `Grep(pattern="VALID_STATE_TRANSITIONS", path="grid_trading_bot/src/")` | 強制有效轉換 | 添加轉換表 |
| J-3 | 狀態持久化 | `Grep(pattern="_save_state\|save.*state", path="grid_trading_bot/src/")` | 狀態可恢復 | 添加持久化 |
| J-4 | 孤兒訂單清理 | `Grep(pattern="orphan\|stale.*order\|cleanup.*order", path="grid_trading_bot/src/")` | 重啟後清理 | 添加清理邏輯 |
| J-5 | 交易所同步 | `Grep(pattern="sync_orders\|sync_position\|reconcile", path="grid_trading_bot/src/")` | 本地與交易所同步 | 添加同步 |
| J-6 | 崩潰恢復 | `Grep(pattern="recover\|resume\|restore", path="grid_trading_bot/src/bots/")` | 支持恢復 | 添加恢復邏輯 |

### K. 時間處理 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 | 修復方式 |
|------|----------|----------|----------|----------|
| K-1 | UTC 時區 | `Grep(pattern="datetime\\.now\\(\\)", path="grid_trading_bot/src/", output_mode="content")` | 全用 timezone.utc | 添加 tz=timezone.utc |
| K-2 | 時間戳單位 | `Grep(pattern="\\* 1000\|/ 1000", path="grid_trading_bot/src/")` | 毫秒/秒正確 | 驗證轉換 |
| K-3 | 時間同步任務 | `Grep(pattern="sync.*time\|time.*sync", path="grid_trading_bot/src/exchange/")` | 與交易所同步 | 添加同步 |
| K-4 | 資金費率時間 | `Grep(pattern="funding.*time\|next_funding", path="grid_trading_bot/src/")` | 計算正確 | 驗證邏輯 |
| K-5 | 定時器精度 | `Grep(pattern="asyncio\\.sleep", path="grid_trading_bot/src/")` | 適當精度 | 調整間隔 |
| K-6 | 訂單過期 | `Grep(pattern="time_in_force\|goodTillDate", path="grid_trading_bot/src/")` | 正確設置 | 驗證設置 |

---

## 執行指令

### 完整審計命令

```
請對 /mnt/c/trading/grid_trading_bot/src/ 執行完整審計：
1. 按順序檢查 A-K 所有類別 (73 項)
2. 對每個失敗項目提供修復代碼
3. 自動執行修復
4. 生成總結報告
```

### 快速審計命令 (僅關鍵項)

```
請對交易系統執行快速審計，僅檢查：
- A: API 端點
- C: 買賣方向
- D: 精度處理
- I: 風險控制
發現問題立即修復。
```

### 單類別審計命令

```
請對交易系統執行 [類別字母] 類別審計：
例如：請對交易系統執行 C 類別審計（買賣方向正確性）
```

---

## 關鍵文件優先級

| 優先級 | 文件路徑 | 原因 |
|--------|----------|------|
| 1 | `src/exchange/binance/futures_api.py` | 核心合約 API |
| 2 | `src/exchange/binance/spot_api.py` | 現貨 API |
| 3 | `src/exchange/binance/auth.py` | 認證簽名 |
| 4 | `src/exchange/client.py` | 統一客戶端 |
| 5 | `src/bots/base.py` | 策略基類 |
| 6 | `src/bots/grid/order_manager.py` | 訂單管理 |
| 7 | `src/bots/grid_futures/bot.py` | 期貨網格 |
| 8 | `src/bots/supertrend/bot.py` | Supertrend |
| 9 | `src/bots/rsi_grid/bot.py` | RSI-Grid |
| 10 | `src/risk/circuit_breaker.py` | 熔斷器 |
| 11 | `src/risk/risk_engine.py` | 風險引擎 |
| 12 | `src/core/models.py` | 核心模型 |

---

## 修復優先級

| 優先級 | 等級 | 時限 | 範例 |
|--------|------|------|------|
| 1 | **嚴重** | 立即修復 | 買賣方向錯誤、認證失敗、風險控制缺失 |
| 2 | **重要** | 當日修復 | 精度問題、去重缺失、狀態不同步 |
| 3 | **中等** | 本週修復 | 錯誤處理、異步問題、時間處理 |
| 4 | **輕微** | 計劃修復 | 代碼優化、日誌改進 |

---

## 報告格式

### [類別 X]: [名稱]

**狀態**: [通過 / 警告 / 失敗]

**問題清單**:
1. **[嚴重程度]** `[文件:行號]`
   - 問題: [描述]
   - 修復: [已修復 / 待修復]
   - 代碼變更: [變更摘要]

**修復統計**: X 個問題修復，Y 個待處理

---

## 最終總結模板

| 類別 | 檢查項 | 通過 | 警告 | 失敗 | 已修復 |
|------|--------|------|------|------|--------|
| A. API | 6 | ? | ? | ? | ? |
| B. 認證 | 5 | ? | ? | ? | ? |
| C. 方向 | 8 | ? | ? | ? | ? |
| D. 精度 | 8 | ? | ? | ? | ? |
| E. 解析 | 6 | ? | ? | ? | ? |
| F. 去重 | 6 | ? | ? | ? | ? |
| G. 錯誤 | 7 | ? | ? | ? | ? |
| H. 異步 | 7 | ? | ? | ? | ? |
| I. 風險 | 8 | ? | ? | ? | ? |
| J. 狀態 | 6 | ? | ? | ? | ? |
| K. 時間 | 6 | ? | ? | ? | ? |
| **總計** | **73** | ? | ? | ? | ? |

**整體評估**: [生產就緒 / 需要修復 / 嚴重問題]

---

## 審計執行注意事項

1. **每次修復後提交**
   ```bash
   git add .
   git commit -m "audit: fix [類別]-[編號] [問題描述]"
   git push origin main
   ```

2. **修復驗證**
   - 修復後重新執行該項檢查
   - 確認問題已解決
   - 運行相關測試

3. **風險評估**
   - 修復前評估影響範圍
   - 高風險修復需備份
   - 優先修復嚴重問題

4. **文檔更新**
   - 記錄所有修復
   - 更新相關文檔
   - 標記已知限制

---

## 常見問題修復模式

### 模式 1: 添加去重機制
```python
from collections import deque

class Bot:
    def __init__(self):
        self._processed_fills = deque(maxlen=1000)  # 限制大小

    def _on_fill(self, fill_id: str):
        if fill_id in self._processed_fills:
            return  # 跳過重複
        self._processed_fills.append(fill_id)
        # 處理 fill...
```

### 模式 2: 添加訂單鎖
```python
import asyncio

class OrderManager:
    def __init__(self):
        self._order_lock = asyncio.Lock()

    async def place_order(self, order):
        async with self._order_lock:
            # 串行執行訂單
            return await self._execute_order(order)
```

### 模式 3: 精度處理
```python
from decimal import Decimal, ROUND_DOWN

def format_quantity(quantity: float, step_size: str) -> str:
    qty = Decimal(str(quantity))
    step = Decimal(step_size)
    precision = abs(step.as_tuple().exponent)
    formatted = qty.quantize(step, rounding=ROUND_DOWN)
    return f"{formatted:.{precision}f}"
```

### 模式 4: 時間同步
```python
from datetime import datetime, timezone

# 錯誤
now = datetime.now()

# 正確
now = datetime.now(timezone.utc)
```

### 模式 5: 異步 gather 安全
```python
# 錯誤
results = await asyncio.gather(*tasks)

# 正確
results = await asyncio.gather(*tasks, return_exceptions=True)
for result in results:
    if isinstance(result, Exception):
        logger.error(f"Task failed: {result}", exc_info=True)
```

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0 | 2026-02-03 | 初始版本，73 項檢查 |
