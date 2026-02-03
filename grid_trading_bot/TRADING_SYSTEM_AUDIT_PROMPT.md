# 加密貨幣交易系統完整審計指令

## 系統概述

你正在審計一個 Python 加密貨幣交易系統，架構如下：
- **交易所**: Binance Spot & Futures API (REST + WebSocket)
- **策略**: Grid, Bollinger, Supertrend, RSI-Grid, GridFutures
- **數據層**: PostgreSQL + Redis
- **協調**: 多機器人 Master 系統
- **風險管理**: 熔斷器、回撤保護、倉位限制

**重要警告**: 這是金融交易系統，錯誤會導致直接財務損失。請徹底且保守地進行分析。

---

## 審計清單 (48 項檢查)

### A. API 與端點驗證 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| A-1 | Futures REST URL 正確 | `Grep(pattern="fapi.binance.com", path="src/exchange/")` | 僅在 futures_api.py 中出現 |
| A-2 | Spot REST URL 正確 | `Grep(pattern="api.binance.com", path="src/exchange/")` | 僅在 spot_api.py 中出現 |
| A-3 | Futures WebSocket URL | `Grep(pattern="fstream.binance.com", path="src/exchange/")` | 合約 WS 使用正確端點 |
| A-4 | Spot WebSocket URL | `Grep(pattern="stream.binance.com", path="src/exchange/")` | 現貨 WS 使用正確端點 |
| A-5 | 無端點交叉污染 | `Grep(pattern="/api/v3/", path="src/exchange/binance/futures_api.py")` | 應無結果 |
| A-6 | Testnet 隔離 | `Grep(pattern="testnet", path="src/exchange/")` | 僅在配置/測試中出現 |

### B. 認證與簽名 (4 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| B-1 | recvWindow 安全值 | `Read("src/exchange/binance/auth.py")` | recvWindow <= 10000ms |
| B-2 | 時間偏移校正 | `Grep(pattern="time_offset", path="src/exchange/")` | 簽名時應用時間偏移 |
| B-3 | HMAC-SHA256 簽名 | `Grep(pattern="hmac.new.*sha256", path="src/exchange/")` | 使用正確簽名算法 |
| B-4 | API Key Header | `Grep(pattern="X-MBX-APIKEY", path="src/exchange/")` | 正確設置 API 密鑰頭 |

### C. 買賣方向正確性 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| C-1 | 平多倉方向 | `Read("src/exchange/binance/futures_api.py", offset=1210, limit=100)` | Long 平倉 → SELL |
| C-2 | 平空倉方向 | 同上 | Short 平倉 → BUY |
| C-3 | reduceOnly 使用 | `Grep(pattern="reduce_only.*True\|reduceOnly.*true", path="src/")` | 平倉訂單必須設置 |
| C-4 | PositionSide 對沖模式 | `Grep(pattern="PositionSide\\.LONG\|PositionSide\\.SHORT", path="src/")` | 對沖模式正確使用 |
| C-5 | 雙向持倉支持 | `Grep(pattern="dualSidePosition\|BOTH", path="src/exchange/")` | 支持單向/對沖模式 |
| C-6 | 反向訂單邏輯 | `Read("src/bots/grid/order_manager.py")` 搜索 `place_reverse_order` | BUY成交→SELL，SELL成交→BUY |

### D. 精度處理 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| D-1 | Decimal 使用 | `Grep(pattern="from decimal import", path="src/")` | 金融計算使用 Decimal |
| D-2 | 危險 float 轉換 | `Grep(pattern="Decimal\\((?!str\\|\"\\|')", path="src/", type="py")` | 應無直接 float→Decimal |
| D-3 | ROUND_DOWN 數量 | `Grep(pattern="ROUND_DOWN", path="src/bots/")` | 數量計算使用向下取整 |
| D-4 | stepSize 應用 | `Grep(pattern="step_size\|stepSize", path="src/")` | 數量符合交易所精度 |
| D-5 | tickSize 應用 | `Grep(pattern="tick_size\|tickSize", path="src/")` | 價格符合交易所精度 |
| D-6 | 科學記號防止 | `Grep(pattern="format.*:f\|normalize", path="src/exchange/")` | 避免 1e-8 格式 |

### E. API 回應解析 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| E-1 | from_binance 解析器 | `Grep(pattern="def from_binance", path="src/core/models.py")` | Order, Position 有解析器 |
| E-2 | 空值處理 | `Grep(pattern='\\.get\\(.*""\\|None\\)', path="src/core/models.py")` | 正確處理缺失字段 |
| E-3 | liquidationPrice 解析 | 讀取 Position.from_binance | 正確處理 "0" 字串 |
| E-4 | 時間戳解析 | `Grep(pattern="timestamp_to_datetime", path="src/")` | 毫秒正確轉換 |
| E-5 | 錯誤碼映射 | `Read("src/exchange/binance/futures_api.py")` 搜索 `error_code_map` | 完整的錯誤碼處理 |

### F. 去重機制 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| F-1 | Fill 去重 | `Grep(pattern="_processed_fill_ids", path="src/bots/")` | 防止重複處理成交 |
| F-2 | 去重集合大小限制 | 檢查 F-1 結果 | 有 maxlen 或定期清理 |
| F-3 | WebSocket 消息去重 | `Grep(pattern="_recent_msg_ids\|_dedup", path="src/exchange/")` | WS 消息去重 |
| F-4 | 信號冷卻機制 | `Grep(pattern="cooldown\|_signal_cooldown", path="src/bots/")` | 防止信號堆疊 |
| F-5 | K線級別去重 | `Grep(pattern="_last_signal_bar\|kline.*lock", path="src/bots/")` | 每根K線最多一個信號 |

### G. 錯誤處理 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| G-1 | 靜默 except 塊 | `Grep(pattern="except.*:\\s*$", path="src/", output_mode="content", -A=2)` | 無靜默 pass |
| G-2 | exc_info 日誌 | `Grep(pattern="logger\\.error\\(", path="src/", output_mode="content")` | 錯誤日誌包含堆疊 |
| G-3 | 重試邏輯 | `Grep(pattern="@.*retry\|RetryConfig", path="src/")` | 關鍵操作有重試 |
| G-4 | 特定異常類型 | `Grep(pattern="except (\\w+Error)", path="src/exchange/")` | 捕獲具體異常 |
| G-5 | 異常鏈保留 | `Grep(pattern="raise.*from", path="src/")` | 保留原始異常 |

### H. 異步與競態條件 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| H-1 | gather return_exceptions | `Grep(pattern="asyncio\\.gather", path="src/", output_mode="content", -A=1)` | 有 return_exceptions=True |
| H-2 | Lock 保護共享狀態 | `Grep(pattern="asyncio\\.Lock", path="src/")` | 關鍵操作有鎖 |
| H-3 | Task done_callback | `Grep(pattern="add_done_callback", path="src/")` | 背景任務有回調 |
| H-4 | 訂單隊列序列化 | `Read("src/exchange/client.py")` 搜索 `_order_lock` | 訂單執行串行化 |
| H-5 | 超時保護 | `Grep(pattern="asyncio\\.wait_for\|timeout=", path="src/")` | 異步操作有超時 |

### I. 風險控制 (6 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| I-1 | 止損實現 | `Grep(pattern="stop_loss\|STOP_MARKET", path="src/")` | 有止損機制 |
| I-2 | MARK_PRICE 工作類型 | `Grep(pattern="workingType.*MARK\|MARK_PRICE", path="src/")` | SL 使用標記價格 |
| I-3 | 止損 vs 強平驗證 | `Grep(pattern="liquidation.*stop\|stop.*liquidation", path="src/")` | SL 在強平價之前 |
| I-4 | 熔斷器 | `Grep(pattern="circuit_breaker\|CircuitBreaker", path="src/risk/")` | 熔斷保護存在 |
| I-5 | 倉位限制 | `Grep(pattern="max_position\|position_limit", path="src/")` | 有倉位上限 |
| I-6 | 每日虧損限制 | `Grep(pattern="daily_loss\|daily_limit", path="src/")` | 每日虧損限制 |

### J. 狀態管理 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| J-1 | 狀態機定義 | `Read("src/core/models.py")` 搜索 `BotState` | 完整狀態枚舉 |
| J-2 | 狀態轉換驗證 | `Grep(pattern="VALID_STATE_TRANSITIONS", path="src/")` | 強制有效轉換 |
| J-3 | 狀態持久化 | `Grep(pattern="_save_state\|save_bot_state", path="src/")` | 狀態可恢復 |
| J-4 | 孤兒訂單清理 | `Grep(pattern="orphan\|stale.*order\|cleanup", path="src/")` | 重啟後清理 |
| J-5 | 交易所同步 | `Grep(pattern="sync_orders\|sync_position", path="src/")` | 本地與交易所同步 |

### K. 時間處理 (5 項)

| 編號 | 檢查項目 | 搜索指令 | 預期結果 |
|------|----------|----------|----------|
| K-1 | UTC 時區一致 | `Grep(pattern="datetime\\.now\\(\\)", path="src/", output_mode="content")` | 全部使用 timezone.utc |
| K-2 | 時間戳單位 | `Grep(pattern="\\* 1000\|/ 1000", path="src/")` | 毫秒/秒轉換正確 |
| K-3 | 時間同步任務 | `Grep(pattern="sync.*time\|time.*sync", path="src/exchange/")` | 與交易所時間同步 |
| K-4 | 資金費率時間 | `Grep(pattern="funding.*time\|next_funding", path="src/")` | 避開結算時段 |
| K-5 | 定時器精度 | `Grep(pattern="asyncio\\.sleep\|Timer", path="src/")` | 適當的定時精度 |

---

## 執行順序

按以下順序執行審計（有依賴關係）：

1. **A: API 端點** - 所有 API 調用的基礎
2. **B: 認證** - 私有 API 必需
3. **D: 精度** - 影響所有訂單
4. **C: 方向** - 核心交易邏輯
5. **E: 解析** - 數據完整性
6. **F: 去重** - 防止重複執行
7. **G: 錯誤處理** - 系統可靠性
8. **H: 異步/競態** - 並發安全
9. **I: 風險控制** - 損失預防
10. **J: 狀態管理** - 崩潰恢復
11. **K: 時間處理** - 時序正確性

---

## 報告格式

對每個類別，使用以下格式報告：

### [類別 X]: [名稱]

**狀態**: [通過 / 警告 / 失敗]

**發現的問題**:
1. **[嚴重程度: 嚴重/重要/中等/輕微]** `[文件:行號]`
   - 描述: [問題描述]
   - 代碼: `[相關代碼片段]`
   - 風險: [可能的後果]
   - 修復建議: [建議的修復方式]

**已檢查的文件**:
- [文件列表]

---

## 關鍵文件清單

審計時優先檢查以下文件：

| 優先級 | 文件路徑 | 原因 |
|--------|----------|------|
| 1 | `src/exchange/binance/futures_api.py` | 核心合約 API，訂單執行 |
| 2 | `src/exchange/binance/spot_api.py` | 現貨 API |
| 3 | `src/bots/base.py` | 所有策略的基類 |
| 4 | `src/bots/grid/order_manager.py` | 訂單管理核心 |
| 5 | `src/core/models.py` | 數據模型，API 解析 |
| 6 | `src/exchange/client.py` | 統一交易所客戶端 |
| 7 | `src/risk/circuit_breaker.py` | 熔斷器 |
| 8 | `src/exchange/binance/websocket.py` | WebSocket 處理 |
| 9 | `src/bots/grid_futures/bot.py` | 合約網格策略 |
| 10 | `src/execution/router.py` | 訂單路由 |

---

## 最終總結模板

審計完成後，生成以下總結：

| 類別 | 檢查項 | 通過 | 警告 | 失敗 |
|------|--------|------|------|------|
| A. API 端點 | 6 | ? | ? | ? |
| B. 認證 | 4 | ? | ? | ? |
| C. 方向 | 6 | ? | ? | ? |
| D. 精度 | 6 | ? | ? | ? |
| E. 解析 | 5 | ? | ? | ? |
| F. 去重 | 5 | ? | ? | ? |
| G. 錯誤處理 | 5 | ? | ? | ? |
| H. 異步 | 5 | ? | ? | ? |
| I. 風險 | 6 | ? | ? | ? |
| J. 狀態 | 5 | ? | ? | ? |
| K. 時間 | 5 | ? | ? | ? |
| **總計** | **48** | ? | ? | ? |

**整體評估**: [生產就緒 / 需要修復 / 嚴重問題]

**優先修復項目**:
1. [最高優先級問題]
2. [次高優先級問題]
3. ...

---

## 使用說明

### 如何使用此審計提示詞

1. **複製此文件內容**到 Claude Code 對話中
2. **執行命令**: 告訴 Claude "請根據審計清單對交易系統進行完整審計"
3. **審閱結果**: Claude 會按順序執行所有 48 項檢查
4. **修復問題**: 根據報告中的建議修復發現的問題
5. **重新審計**: 修復後重新運行審計以確認問題已解決

### 審計頻率建議

- **每次重大更改後**: 完整審計
- **每周**: 關鍵類別審計 (A, C, D, I)
- **部署前**: 完整審計 + 手動驗證

### 自定義審計

可以根據需要修改此提示詞：
- 添加特定於您系統的檢查項目
- 調整搜索路徑以匹配您的項目結構
- 增加或修改預期結果

---

## 版本歷史

| 版本 | 日期 | 變更 |
|------|------|------|
| 1.0 | 2026-02-03 | 初始版本，48 項檢查 |
