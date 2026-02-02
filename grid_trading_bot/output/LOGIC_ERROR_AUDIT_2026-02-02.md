# 程式邏輯錯誤深度掃描報告

> **掃描日期**: 2026-02-02
> **掃描範圍**: grid_trading_bot/src/ — 224 個 Python 檔案，127,642 行
> **審計依據**: `trading_logic_error_check.md` 8 章檢查清單

---

## 掃描統計

- 檢查的程式碼行數：127,642
- 發現的邏輯錯誤：47
- 🔴 致命（直接導致虧損/崩潰）：5
- 🟠 嚴重（可能導致虧損）：16
- 🟡 中等（邏輯不嚴謹）：17
- 🟢 輕微（程式碼品質）：9

---

## 第一章：數學運算邏輯錯誤

### 1.1 浮點數精度陷阱

**結論: ✅ 大部分正確**

- [x] **浮點數比較**: 全系統未發現 `==` 比較 float 價格/數量。Decimal 使用一致。
- [x] **浮點數加減**: 財務計算全部使用 Decimal。
- [⚠️] **浮點數累積**: Supertrend bot RSI 全程用 float 計算再轉 Decimal（`bots/supertrend/bot.py:776-795`）。RSI-Grid 的初始化也用 float 種子值（`bots/rsi_grid/indicators.py:156-166`）。

**具體問題**:

| # | 檔案:行 | 嚴重度 | 問題 | 修復方向 |
|---|---------|--------|------|----------|
| 1.1-A | `bots/base.py:5353` | 🟠 | `Decimal(str(0.1 * (attempt+1)))` float 中間值 | 改為 `Decimal("0.1") * (attempt+1)` |
| 1.1-B | `bots/supertrend/bot.py:776` | 🟡 | RSI 全程 float 再轉 Decimal | 改用 Decimal 計算 |
| 1.1-C | `bots/rsi_grid/indicators.py:156` | 🟡 | 初始 gain/loss 用 float | 改用 Decimal |

### 1.2 除法陷阱

**結論: ⚠️ 發現 2 個問題**

- [x] **_safe_divide**: 正確實作，零值回傳預設值
- [x] **勝率/統計**: 有 `if total > 0` 防護
- [❌] **槓桿除法**: 無零值防護

| # | 檔案:行 | 嚴重度 | 問題 | 修復方向 |
|---|---------|--------|------|----------|
| 1.2-A | `bots/base.py:2263` | 🔴 | `notional_value / Decimal(leverage)` leverage=0 崩潰 | 加入 `if leverage <= 0: raise` |
| 1.2-B | `bots/base.py:2058` | 🟡 | `qty_diff / local_qty` 未取 abs，做空時分母為負 | 改為 `abs(local_qty)` |

### 1.3 數值溢出與異常值

**結論: ✅ 良好**

- [x] **NaN 防護**: `_validate_indicator_value` 檢查 NaN/Infinity
- [x] **負數防護**: 餘額、數量有正值驗證
- [⚠️] **WS 餘額可能為負**: `exchange/state_sync.py:1345`

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 1.3-A | `exchange/state_sync.py:1345` | 🟠 | `free = wb - cw` 可為負 |
| 1.3-B | `core/models.py:575` | 🟡 | `locked = walletBalance - availableBalance` 可為負 |

### 1.4 精度截斷邏輯

**結論: ⚠️ 主要問題在止損價**

- [x] **round_decimal**: 使用 `ROUND_DOWN` ✅ 正確（截斷而非四捨五入）
- [x] **round_to_tick**: 正確向下截斷到 tick size
- [❌] **止損價**: 硬編碼 tick size

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 1.4-A | `bots/base.py:8867,9336` | 🔴 | `stop_price.quantize(Decimal("0.1"))` 硬編碼精度 |

---

## 第二章：條件判斷邏輯錯誤

### 2.1 比較運算符錯誤

**結論: ✅ 全部正確**

止損/止盈方向全面驗證通過：
- ✅ LONG SL: `entry * (1 - sl_pct)` → 低於進場 → `kline_low <= sl_price` 觸發
- ✅ SHORT SL: `entry * (1 + sl_pct)` → 高於進場 → `kline_high >= sl_price` 觸發
- ✅ LONG TP: `kline_high >= tp_price` 觸發
- ✅ SHORT TP: `kline_low <= tp_price` 觸發
- ✅ 平倉方向: LONG → SELL, SHORT → BUY

### 2.2 條件遺漏

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 2.2-A | `bots/grid_futures/bot.py:1331` | 🟠 | 單根 K 線遍歷所有 level 無 break，多重觸發 |
| 2.2-B | `master/heartbeat.py:775` | 🟡 | PAUSED 狀態無法合法轉換到 ERROR |
| 2.2-C | `core/models.py:448` | 🟡 | `position_amt==0` 預設分類為 SHORT |

### 2.3 條件順序錯誤

**結論: ✅ 正確**

- [x] 風控檢查在交易前執行（`PreTradeRiskChecker.check()` 先於下單）
- [x] `RiskEngine.check()` 順序: capital → drawdown → alert → circuit breaker → emergency

### 2.4 布林邏輯錯誤

**結論: ✅ 未發現**

- [x] 無德摩根定律混淆
- [x] 狀態旗標無矛盾組合
- [x] 三值邏輯使用 Enum (LONG/SHORT/BOTH)

---

## 第三章：迴圈與遞迴邏輯錯誤

### 3.1 迴圈邊界

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 3.1-A | `bots/grid/order_manager.py:149` | 🟠 | `deque(maxlen=10000)` 驅逐未配對買單 |
| 3.1-B | `bots/grid/order_manager.py:221` | 🟠 | Grid rebuild `_pending_buy_fills.clear()` 丟失記錄 |
| 3.1-C | `bots/grid_futures/bot.py:1359` | 🟠 | 同根 K 線多次平倉 `filled_long_count` 未更新 |

### 3.2 技術指標計算中的迴圈錯誤

**結論: ⚠️ Supertrend 指標有問題**

#### SMA 驗算
```
closes = [1, 2, 3, 4, 5]
SMA(3) = (3+4+5)/3 = 4.0
```
- ✅ `data/kline/indicators.py` SMA 計算正確（取最後 period 根）

#### EMA 驗算
```
multiplier = 2 / (period + 1)
ema = close * multiplier + prevEma * (1 - multiplier)
```
- ✅ `data/kline/indicators.py` EMA 正確
- ⚠️ `data/kline/indicators.py:172` ATR 用 EMA(2/(n+1)) 而非 Wilder's(1/n)

#### RSI 驗算
```
avgGain = Wilder's smoothing: (prevAvgGain * (period-1) + currentGain) / period
RS = avgGain / avgLoss
RSI = 100 - 100/(1+RS)
```
- ✅ `bots/rsi_grid/indicators.py` RSI 正確使用 Wilder's Smoothing
- ✅ `avgLoss == 0` 防護: 回傳 RSI=100
- ❌ `bots/supertrend/bot.py:773-795` RSI 用簡單平均，非 Wilder's

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 3.2-A | `bots/supertrend/bot.py:773` | 🟠 | RSI 用 SMA 而非 Wilder's Smoothing |
| 3.2-B | `bots/supertrend/indicators.py:122` | 🟠 | ATR 用 SMA 而非 Wilder's Smoothing |
| 3.2-C | `data/kline/indicators.py:172` | 🟡 | ATR 用 EMA 而非 Wilder's，與主流平台不同 |

#### Bollinger Bands 驗算
```
Middle = SMA(period)
StdDev = sqrt(sum((c - middle)^2) / N)  ← 母體標準差
Upper/Lower = Middle ± multiplier × StdDev
```
- ✅ `bots/bollinger/indicators.py:182` 使用母體標準差（N）符合業界標準
- ✅ multiplier 可配置

#### Supertrend 驗算
- ❌ **趨勢判定使用前一根 band 而非當前調整值**（`indicators.py:91-99`）
- ❌ **upper_band None 檢查遺漏**（`indicators.py:82-86`）

---

## 第四章：時序與狀態轉換邏輯錯誤

### 4.1 狀態機邏輯

**結論: ✅ 大致正確**

Bot 狀態轉換圖：
```
REGISTERED → RUNNING（start）
RUNNING → PAUSED（pause）
RUNNING → STOPPING → STOPPED
RUNNING → ERROR → STOPPED
PAUSED → RUNNING（resume）
PAUSED → STOPPING → STOPPED
STOPPED → RUNNING（restart）
ERROR → STOPPED
```

- [x] 每個轉換有驗證（`VALID_STATE_TRANSITIONS`）
- [x] 不可跳躍（例如 REGISTERED 不可直接到 PAUSED）
- [⚠️] PAUSED → ERROR 未定義（heartbeat 超時時問題）

| # | 嚴重度 | 問題 |
|---|--------|------|
| 4.1-A | 🟡 | PAUSED 狀態的 bot heartbeat 超時無法轉 ERROR |

### 4.2 操作順序錯誤

**結論: ✅ 正確**

開倉順序驗證：
1. ✅ 計算訊號
2. ✅ 風控檢查（PreTradeRiskChecker）
3. ✅ 計算倉位大小
4. ✅ 提交訂單
5. ✅ 等待成交
6. ✅ 設定止損止盈
7. ✅ 記錄日誌

平倉順序驗證：
1. ✅ 取消關聯掛單
2. ✅ 提交平倉訂單
3. ✅ 計算盈虧
4. ✅ 記錄日誌

關閉順序：`fund_manager → commander.stop_all → heartbeat → dashboard → save_state` ✅

### 4.3 異步操作時序錯誤

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 4.3-A | `bots/bollinger/bot.py`, `grid_futures/bot.py` | 🟠 | 無 kline 重入保護，create_task 可並行 |
| 4.3-B | `risk/risk_engine.py:392` | 🟠 | callback 同步呼叫，async callback 被丟棄 |
| 4.3-C | `risk/circuit_breaker.py:226,274` | 🟠 | 同上 |
| 4.3-D | `risk/emergency_stop.py:239` | 🟠 | 同上 |
| 4.3-E | `execution/algorithms.py:965` | 🔴 | float × Decimal → TypeError 崩潰 |

---

## 第五章：字串與類型邏輯錯誤

### 5.1 字串比較陷阱

**結論: ⚠️ 有重大枚舉衝突**

- [x] 大小寫: 使用 `str, Enum` 確保一致性
- [x] 交易對格式: 統一使用 ccxt 標準化格式

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 5.1-A | `fund_manager/core/position_manager.py:23` vs `core/models.py:51` | 🟠 | `PositionSide` 枚舉值 "long" vs "LONG" |

### 5.2 類型轉換陷阱

**結論: ⚠️ 有問題**

- [x] API 回傳數字皆有 `Decimal(str(x))` 轉換
- [⚠️] `getattr(order, "filled_qty", Decimal("0"))` 可能拿到字串

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 5.2-A | `bots/base.py` 多處 | 🟡 | getattr 假設回傳 Decimal，API 可能回傳 str |
| 5.2-B | `bots/base.py:5357` | 🟢 | `limit_price * (1 + adjustment_pct)` int + Decimal |

---

## 第六章：集合與數據結構邏輯錯誤

### 6.1 陣列操作

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 6.1-A | `bots/grid/order_manager.py:149` | 🟠 | 有界 deque 驅逐記錄（同 3.1-A） |

### 6.2 Map/Dict 操作

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 6.2-A | `fund_manager/core/position_manager.py:297` | 🟡 | 同 symbol 多空部位直接累加無衝突檢查 |
| 6.2-B | `fund_manager/core/position_manager.py:505` | 🟡 | exchange sync 覆蓋 quantity 但 bot_contributions 不同步 |

---

## 第七章：交易特有邏輯錯誤

### 7.1 移動止損邏輯

**結論: ✅ 正確**

驗算（做多）:
```
進場 $100，追蹤距離 $3
價格: $100 → $105 → $103 → $108 → $104
止損: $97  → $102 → $102  → $105 → $105（觸發平倉）
```

- [x] SLTP calculator 正確實作「只能往有利方向移動」
- [x] 做空方向正確（只能往下移）

### 7.2 加倉/減倉邏輯

**結論: N/A**

Grid bot 不支援傳統加減倉，使用 grid level 管理。Grid Futures 有 DCA 模式但透過 grid level 實現。

### 7.3 部分成交處理邏輯

**結論: ⚠️ 有問題**

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 7.3-A | `bots/grid/order_manager.py:1136` | 🔴 | 反向單用目標 level 配額而非成交量（同 F-02） |
| 7.3-B | `bots/grid/calculator.py:479` | 🔴 | Sell level 配額為零（同 F-01） |
| 7.3-C | `bots/grid_futures/bot.py:1359` | 🟠 | 同根 K 線多次平倉數量漂移（同 3.1-C） |

### 7.4 手續費扣除邏輯

**結論: ⚠️ 輕微問題**

- [x] Maker/Taker 區分: 透過 `fee_rate` 配置
- [x] FIFO 成本追蹤: 正確實作

| # | 檔案:行 | 嚴重度 | 問題 |
|---|---------|--------|------|
| 7.4-A | `bots/base.py:6600` | 🟡 | `close_fee = fee * close_qty / quantity` 若 `close_qty > quantity` 則過度歸屬 |

---

## 第八章：總結

### 驗算結果

| 項目 | 輸入 | 預期 | 系統 | 結果 |
|------|------|------|------|------|
| SMA(3) | closes=[1,2,3,4,5] | 4.0 | 4.0 | ✅ |
| EMA multiplier | period=12 | 2/13=0.1538 | 0.1538 | ✅ |
| RSI (rsi_grid) | Wilder's smoothing | 標準 | 標準 | ✅ |
| RSI (supertrend) | 應為 Wilder's | 簡單平均 | 不一致 | ❌ |
| Bollinger StdDev | 母體(÷N) | 母體 | 母體 | ✅ |
| Supertrend 趨勢切換 | 用當前 band | 用前一根 | 延遲一根 | ❌ |
| 止損方向(做多) | price ≤ SL | ≤ | ≤ | ✅ |
| 止損方向(做空) | price ≥ SL | ≥ | ≥ | ✅ |
| 移動止損(只升) | 只能往有利方向 | 是 | 是 | ✅ |
| round_decimal | 截斷不四捨五入 | ROUND_DOWN | ROUND_DOWN | ✅ |
| Grid 反向單數量 | 應=成交量 | 0(sell level) | 零量 | ❌ |

### 狀態轉換圖驗證

```
REGISTERED ─── start() ──→ RUNNING ─── pause() ──→ PAUSED
                              │                        │
                              │ stop()             resume() / stop()
                              ▼                        │
                           STOPPING ←─────────────────┘
                              │
                              ▼
                           STOPPED ←── ERROR
                              │           ↑
                              └── start() ─┘ (未實作 PAUSED→ERROR)
```

- ✅ REGISTERED → RUNNING：已驗證
- ✅ RUNNING → PAUSED → RUNNING：已驗證
- ✅ RUNNING → STOPPING → STOPPED：已驗證
- ✅ ERROR → STOPPED：已驗證
- ⚠️ PAUSED → ERROR：未定義（heartbeat timeout 時有問題）

### 整體健康度評估

**系統整體評分: 7.5 / 10**

**優點**:
- Decimal 使用一致，財務精度有保障
- 止損/止盈方向全部正確
- 風控檢查在交易前執行
- 狀態機定義完整
- FIFO 成本追蹤正確
- 原子性資金鎖定防 TOCTOU

**主要風險**:
1. 🔴 Grid bot 反向單零數量 — **核心功能受損**
2. 🔴 VWAP float×Decimal 崩潰 — **執行引擎故障**
3. 🔴 止損價硬編碼精度 — **跨幣種風險**
4. 🟠 Supertrend 指標計算不標準 — **訊號品質下降**
5. 🟠 Risk callback 丟棄 — **風控通知失效**

### 上線建議

**❌ 不建議立即上線**

必須先修復以下項目：
1. Grid 反向單數量（F-01 + F-02）— 核心交易邏輯
2. VWAP float×Decimal（F-03）— 執行引擎崩潰
3. 止損價精度（F-05）— 風控安全
4. 槓桿零值防護（F-04）— 崩潰風險

修復後可先以最小資金上線測試：
- Grid bot 需重新驗證完整買賣往返
- Supertrend bot 建議與 TradingView 對比指標值
- VWAP 執行需端對端測試
