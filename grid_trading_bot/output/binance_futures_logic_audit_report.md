# Binance 合約交易系統（Python）邏輯掃描報告

## 掃描資訊
- 掃描時間：2026-02-02
- Python 版本：3.12
- 使用的 SDK：自建 REST/WebSocket 客戶端（aiohttp + websockets + hmac）
- 交易對：BTCUSDT 等多幣種
- 帳戶模式：單向（One-way, positionSide=BOTH）
- 保證金模式：逐倉（ISOLATED）

---

## 致命問題（直接導致虧損）

### F-1: 止損價未與強平價交叉驗證
- **位置**：全系統（無此邏輯）
- **問題**：系統沒有驗證止損價是否在強平價之前觸發。若止損設在強平價之後，止損永遠不會觸發，直接被強制平倉（更高手續費）
- **建議**：開倉後比較 `stop_loss_price` 與 `liquidation_price`，若 SL 在爆倉之後則拒絕或調整

### F-2: `asyncio.gather` 未設 `return_exceptions=True`（訂單執行路由）
- **位置**：`src/execution/router.py:972`
- **問題**：`await asyncio.gather(*tasks)` 未設 `return_exceptions=True`。若任一子訂單失敗，所有兄弟訂單被取消，可能留下部分成交無法恢復的狀態
- **建議**：加上 `return_exceptions=True`，逐一檢查結果

---

## 嚴重問題（可能導致系統異常）

### S-1: 止損價格截斷邏輯錯誤
- **位置**：`src/bots/grid_futures/bot.py:1501-1509`
- **問題**：`_calculate_stop_loss_price()` 使用 `quantize(tick_size)` 而非 `(price / tick_size).quantize(1, ROUND_DOWN) * tick_size`。對非 10 的冪次的 tick_size（如 0.05）會產生錯誤結果。另外 `_tick_size` 使用 `getattr` 回退到硬編碼 `Decimal("0.1")`
- **建議**：統一使用 `base.py:_normalize_price()` 的截斷邏輯

### S-2: 止損單未設 `workingType="MARK_PRICE"`
- **位置**：`src/exchange/binance/futures_api.py:831-860`, `:906-914`
- **問題**：STOP_MARKET 訂單未送 `workingType` 參數，Binance 預設用 `CONTRACT_PRICE`（最新成交價），容易被插針觸發止損
- **建議**：所有止損/止盈單加上 `workingType="MARK_PRICE"`

### S-3: Rate Limiter 未整合至請求流程
- **位置**：`src/exchange/rate_limiter.py` vs `src/exchange/binance/futures_api.py._request()`
- **問題**：完整的 `RateLimiter` 類別存在但 `_request()` 從未呼叫 `acquire()` 或 `update_from_headers()`。權重追蹤形同虛設
- **建議**：在 `_request()` 中整合 rate limiter

### S-4: HTTP 418（IP 封鎖）無專門處理
- **位置**：`src/exchange/binance/futures_api.py`
- **問題**：Binance 嚴重超限時回傳 418 封鎖 IP，系統僅有通用錯誤處理，無特殊退避邏輯
- **建議**：偵測 418 後暫停所有請求 5 分鐘並通知管理員

---

## 中等問題（邏輯不嚴謹）

### M-1: 資金費率結算時段無迴避邏輯
- **位置**：全系統（無此邏輯）
- **問題**：系統獲取了 `next_funding_time` 但未用於控制交易時機。在結算前一秒開倉會立即被收取資金費率
- **建議**：結算前 5 分鐘暫停新開倉

### M-2: `recvWindow=60000`（過度寬鬆）
- **位置**：`src/exchange/binance/auth.py:83`
- **問題**：60 秒是 Binance 允許的最大值，增加重放攻擊風險
- **建議**：降至 5000-10000ms

### M-3: 錯誤碼 -2010 分類不精確
- **位置**：`src/exchange/binance/futures_api.py:314`
- **問題**：-2010（New order rejected）映射為 `InsufficientBalanceError`，但此錯誤碼涵蓋多種拒絕原因
- **建議**：解析 `msg` 欄位區分具體原因

### M-4: `liquidationPrice="0"` 邊界處理
- **位置**：`src/core/models.py:466`
- **問題**：`if data.get("liquidationPrice")` — 字串 `"0"` 在 Python 中為 truthy，會被解析為 `Decimal("0")` 而非 `None`
- **建議**：改為 `if data.get("liquidationPrice") not in (None, "", "0")`

### M-5: `create_task` 缺少異常回調
- **位置**：`src/optimization/pool.py:230,547`、`src/infrastructure/state_sync.py:166,169`
- **問題**：背景任務若拋出異常會被靜默吞掉
- **建議**：加上 `task.add_done_callback(handle_exception)`

### M-6: 大量 `except ... pass` 吞掉錯誤
- **位置**：多處（`src/backtest/versioning.py:167`、`src/backtest/git_integration.py:229,304` 等數十處）
- **問題**：靜默吞掉異常可能掩蓋生產環境的真實問題
- **建議**：至少加上 `logger.debug()` 記錄

### M-7: `raise` 未保留異常鏈
- **位置**：`src/backtest/git_integration.py:184,186`、`src/config/models/risk.py:120,137`
- **問題**：`raise XxxError(...)` 在 `except` 區塊內未使用 `from e`，丟失原始異常資訊
- **建議**：加上 `from e`

---

## 建議改善

### G-1: Runner 腳本使用 `print()` 而非 `logger`
- **位置**：`run_bollinger.py:187-228` 等
- **建議**：改用 logger 統一管理

### G-2: 活 PnL 未扣除資金費率
- **位置**：回測引擎有扣（`src/backtest/engine.py:499`），實盤未扣
- **建議**：實盤 PnL 追蹤加入資金費率扣除

### G-3: Runner 全域變數
- **位置**：`run_bollinger.py:173` 等 `global _bot`
- **說明**：用於信號處理器的優雅關機，可接受但非最佳實踐

---

## Binance API 特有驗證

| 項目 | 狀態 |
|------|------|
| 端點使用 fapi（合約），非 api（現貨） | ✅ |
| WebSocket 使用 fstream，非 stream | ✅ |
| 所有 API 返回值都有 Decimal(str()) 轉換 | ✅ |
| side 方向全部驗證正確 | ✅ |
| positionSide 與帳戶模式匹配（BOTH） | ✅ |
| reduceOnly 用於所有平倉訂單 | ✅ |
| 價格精度符合 tickSize | ⚠️ 主流程正確，止損計算有問題（S-1） |
| 數量精度符合 stepSize（ROUND_DOWN） | ✅ |
| MIN_NOTIONAL 檢查已實現 | ✅ |
| Listen Key 有定期續期（30 分鐘） | ✅ |
| workingType 設定正確 | ❌ 未設定（S-2） |
| 時間戳使用毫秒 | ✅ |
| HMAC-SHA256 簽名正確 | ✅ |
| 429/Rate Limit 處理 | ⚠️ 有處理但未整合（S-3） |

## Python 特有驗證

| 項目 | 狀態 |
|------|------|
| 無浮點數 == 比較（金額） | ✅ |
| Decimal 初始化都用字串（非 float） | ✅ |
| 所有 async 調用都有 await | ✅ |
| 所有 create_task 都有異常回調 | ⚠️ 核心有，部分輔助模組缺少（M-5） |
| 無可變默認參數 | ✅ |
| 無空 except/pass | ⚠️ 有數十處（M-6） |
| 無 time.sleep（改用 asyncio.sleep） | ✅ |
| 所有 datetime 都使用 UTC | ✅ |
| asyncio.gather 有 return_exceptions | ❌ router.py 缺少（F-2） |

---

## 修復優先順序

| 優先級 | 編號 | 檔案 | 問題 | 風險 |
|--------|------|------|------|------|
| P0 | F-1 | 全系統 | 止損 vs 強平未交叉驗證 | 爆倉 |
| P0 | F-2 | execution/router.py:972 | gather 無 return_exceptions | 部分成交 |
| P1 | S-1 | grid_futures/bot.py:1501 | 止損價截斷邏輯錯誤 | 下單失敗 |
| P1 | S-2 | futures_api.py | 無 workingType=MARK_PRICE | 插針止損 |
| P1 | S-3 | futures_api.py + rate_limiter.py | Rate limiter 未整合 | IP 封鎖 |
| P1 | S-4 | futures_api.py | 無 418 處理 | IP 封鎖 |
| P2 | M-1 | 全系統 | 資金費率結算迴避 | 額外成本 |
| P2 | M-2 | auth.py:83 | recvWindow 過大 | 安全性 |

---

## 結論

**系統整體架構品質高**，核心交易邏輯（方向、精度、positionSide、reduceOnly）全部正確。Decimal 精度處理、WebSocket 管理、Listen Key 續期等關鍵基礎設施完善。

**需修復後方可上線的問題**：
1. **F-2**（gather 無 return_exceptions）— 可能導致部分成交後狀態不一致
2. **S-2**（無 MARK_PRICE）— 插針觸發止損是實際交易中的常見虧損來源

**殘留風險**：
- F-1（止損 vs 強平）在低槓桿時風險較低，但高槓桿（>10x）必須修復
- S-3（Rate Limiter 未整合）在低頻交易時影響不大，但多 bot 並行時可能觸發 IP 封鎖
