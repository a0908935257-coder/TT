# 交易系統全面掃描報告

> **掃描日期**: 2026-02-02
> **掃描範圍**: grid_trading_bot/src/ 全部 224 個 Python 檔案，127,642 行
> **掃描方式**: 5 個並行審計 agents 分區深度掃描

---

## 掃描統計

| 項目 | 數量 |
|------|------|
| 掃描檔案數 | 224 |
| 掃描程式碼行數 | 127,642 |
| 發現的問題總數 | 47 |
| 🔴 致命（直接導致虧損/崩潰） | 5 |
| 🟠 嚴重（可能導致虧損） | 16 |
| 🟡 中等（邏輯不嚴謹） | 17 |
| 🟢 輕微（程式碼品質） | 9 |

---

## 🔴 致命問題（5 個）

### F-01: Grid 反向賣單數量為零
- **檔案**: `bots/grid/calculator.py:479-488`
- **問題**: Sell levels 的 `allocated_amount=Decimal("0")`，當 BUY 成交後反向 SELL 的數量 = `0 / price = 0`，訂單會被交易所拒絕或下出零數量單
- **影響**: Grid bot 無法完成買賣往返交易，完全失去獲利能力
- **修復方向**: 將成交的 BUY 數量傳遞給反向 SELL，或在 level 上記錄成交量

### F-02: Grid 反向訂單使用目標層級配額而非實際成交量
- **檔案**: `bots/grid/order_manager.py:1136-1190, 454`
- **問題**: `place_reverse_order` 使用 `target_level.allocated_amount / price` 計算數量，而非實際成交量
- **影響**: 結合 F-01，反向單數量錯誤；即使 sell levels 有配額，partial fill 也無法正確對應
- **修復方向**: 傳入 `filled_quantity` 參數直接使用

### F-03: VWAP 演算法 float × Decimal 會崩潰
- **檔案**: `execution/algorithms.py:965`
- **問題**: `elapsed_pct`(float) × `Decimal("100")` 在 Python 會 raise `TypeError`
- **影響**: 啟用 VWAP adaptive participation 時直接崩潰
- **修復方向**: `Decimal(str(elapsed_pct)) * Decimal("100")`

### F-04: 槓桿為零時除以零崩潰
- **檔案**: `bots/base.py:2263`
- **問題**: `required_margin = notional_value / Decimal(leverage)` 無零值防護
- **影響**: 配置錯誤或預設值為 0 時直接崩潰
- **修復方向**: 加入 `if leverage <= 0: raise ValueError()`

### F-05: 止損價硬編碼 tick size
- **檔案**: `bots/base.py:8867, 9336`
- **問題**: `stop_price.quantize(Decimal("0.1"))` 對所有交易對使用固定精度
- **影響**: BTC (tick=0.01) 或低價幣 (tick=0.0001) 止損價精度錯誤，可能被交易所拒絕或在錯誤價位觸發
- **修復方向**: 從 symbol info 取得 `tick_size`

---

## 🟠 嚴重問題（16 個）

### S-01: Supertrend RSI 用簡單平均而非 Wilder's Smoothing
- **檔案**: `bots/supertrend/bot.py:773-795`
- **問題**: RSI 每次只看最近 period 根 K 線，不帶前值平滑
- **影響**: RSI 值波動較大，過濾訊號不穩定

### S-02: Supertrend ATR 用簡單平均而非 Wilder's Smoothing
- **檔案**: `bots/supertrend/indicators.py:122-140`
- **問題**: ATR 每次重新計算平均，失去平滑效果
- **影響**: Supertrend bands 過度波動

### S-03: Supertrend 上軌 None 檢查遺漏
- **檔案**: `bots/supertrend/indicators.py:82-86`
- **問題**: 外層 if 檢查 `_prev_lower_band is not None` 但內層直接使用 `_prev_upper_band` 未檢查
- **影響**: 首次計算可能 NoneType 比較崩潰

### S-04: Supertrend 趨勢判定使用前一根 band 而非當前調整值
- **檔案**: `bots/supertrend/indicators.py:91-99`
- **問題**: 標準 Supertrend 應使用當前已調整的 band，此處用前一根，造成一根延遲
- **影響**: 趨勢轉換訊號延遲一根 K 線

### S-05: Grid Futures 單根 K 線觸發多個進出場
- **檔案**: `bots/grid_futures/bot.py:1331-1414`
- **問題**: 迴圈遍歷所有 grid levels 無 break/return，一根波動大的 K 線可觸發多筆交易
- **影響**: 與回測行為不一致，可能過度交易

### S-06: Bollinger 和 Grid Futures bot 缺乏重入保護
- **檔案**: `bots/bollinger/bot.py`, `bots/grid_futures/bot.py`
- **問題**: 用 `asyncio.create_task` 處理 kline，無 Lock 防護，慢網路下可並行執行
- **影響**: 可能重複開倉或狀態錯亂（Supertrend bot 有正確的 `_kline_lock`）

### S-07: Grid 有界 deque 驅逐未配對買單記錄
- **檔案**: `bots/grid/order_manager.py:149`
- **問題**: `_filled_history` maxlen=10000，舊買單記錄被驅逐後永遠無法配對賣單
- **影響**: 利潤追蹤遺失，止損計算部位可能不正確

### S-08: Risk engine callback 同步呼叫可能丟棄 async coroutine
- **檔案**: `risk/risk_engine.py:392`, `risk/circuit_breaker.py:226,274`, `risk/emergency_stop.py:239`
- **問題**: `self._on_level_change(old, new)` 如果 callback 是 async function，coroutine 被靜默丟棄
- **影響**: 風險等級變更、熔斷器觸發/重置、緊急停止的回調不執行

### S-09: WS 餘額解析可能產生負 free balance
- **檔案**: `exchange/state_sync.py:1345`
- **問題**: `free = wallet_balance - cross_wallet` 當未實現虧損大於錢包餘額時為負
- **影響**: 下游可用餘額判斷異常

### S-10: Grid 擴展動作不實際修改 grid
- **檔案**: `bots/grid/risk_manager.py:647-683`
- **問題**: `_action_expand_grid` 計算新邊界但只發通知，不重建 grid，狀態卻設為 NORMAL
- **影響**: 突破偵測後 grid 保持原範圍，系統認為已處理

### S-11: Grid rebuild 清除未配對買單
- **檔案**: `bots/grid/order_manager.py:221`
- **問題**: `initialize()` 呼叫 `_pending_buy_fills.clear()`，重建時丟失未平倉的買單記錄
- **影響**: 利潤追蹤永久遺失

### S-12: PositionSide 枚舉重複且值不同
- **檔案**: `fund_manager/core/position_manager.py:23` vs `core/models.py:51`
- **問題**: 前者用 `"long"/"short"`，後者用 `"LONG"/"SHORT"`
- **影響**: 跨模組傳遞 PositionSide 時 ValueError

### S-13: Fund Pool 可用餘額判斷不一致
- **檔案**: `fund_manager/core/fund_pool.py:302` vs `fund_manager/manager.py:382`
- **問題**: `get_unallocated()` 用 `available - allocated - reserved`，`dispatch_funds()` 用 `total - allocated`
- **影響**: 期貨場景下 dispatch 可能在無可用資金時仍嘗試分配

### S-14: Dynamic adjust threshold 單位不一致
- **檔案**: `bots/grid/risk_manager.py:2204-2209`
- **問題**: `DynamicAdjustConfig.breakout_threshold` 是小數(0.04)，`RiskConfig.breakout_threshold` 是百分比(4.0)
- **影響**: 混用配置時 threshold 偏差 100 倍

### S-15: Grid Futures 部分平倉數量計算漂移
- **檔案**: `bots/grid_futures/bot.py:1359-1363`
- **問題**: 同一根 K 線觸發多個平倉時，`filled_long_count` 未更新導致後續 `partial_qty` 錯誤
- **影響**: 最後一筆平倉可能殘留或超額

### S-16: base.py 浮點中間值傳入 Decimal
- **檔案**: `bots/base.py:5353`
- **問題**: `Decimal(str(0.1 * (attempt + 1)))` — float 中間運算引入精度誤差
- **影響**: 訂單價格調整微小偏差

---

## 🟡 中等問題（17 個）

| # | 檔案 | 行 | 問題 |
|---|------|----|------|
| M-01 | `bots/grid/bot.py:1074` + `risk_manager.py:1047` | | 重複的 position 計算且無 lock |
| M-02 | `bots/grid/calculator.py:422-425` | | 等距 grid 間距百分比以最低價為基準，上層偏低 |
| M-03 | `bots/grid/order_manager.py:777` | | fill 可能被 sync 和 WebSocket 重複處理（需冪等性） |
| M-04 | `bots/grid/risk_manager.py:727-761` | | 止損百分比基於配置金額而非實際部署資金 |
| M-05 | `bots/grid/calculator.py:420` | | Decimal 非整數冪次內部轉 float |
| M-06 | `bots/grid/risk_manager.py:1614-1627` | | 連續虧損計數午夜重置，跨日虧損串不觸發 |
| M-07 | `bots/supertrend/bot.py:776-795` | | RSI 計算全程用 float 再轉 Decimal |
| M-08 | `bots/rsi_grid/indicators.py:156-166` | | RSI 初始化用 float 種子值 |
| M-09 | `bots/bollinger/bot.py:836-874` | | 開倉失敗 break 不嘗試其他 level |
| M-10 | `execution/algorithms.py:1830` | | POV volume_rate 可為負值 |
| M-11 | `exchange/state_sync.py:1222` | | 部分成交量差異誤報為衝突 |
| M-12 | `execution/router.py:928` | | 定時執行用不可取消的 sleep |
| M-13 | `core/models.py:448-450` | | `position_amt==0` 時預設分類為 SHORT |
| M-14 | `core/models.py:575` | | 期貨鎖定餘額可為負 |
| M-15 | `data/kline/indicators.py:172` | | ATR 用 EMA 而非 Wilder's，與主流平台不同 |
| M-16 | `master/heartbeat.py:775` | | PAUSED 狀態無法轉換到 ERROR |
| M-17 | `bots/base.py:2058` | | 部位差異計算未取絕對值 `local_qty` |

---

## 🟢 輕微問題（9 個）

| # | 檔案 | 問題 |
|---|------|------|
| L-01 | `bots/base.py:5357` | int + Decimal 混合運算 |
| L-02 | `bots/base.py:10280` | 冗餘的零值檢查 |
| L-03 | `bots/base.py:3000` | 訂單去重用 `time.time()` 而非 `time.monotonic()` |
| L-04 | `execution/router.py:742` | 返回 float 但標註 int |
| L-05 | `risk/pre_trade_checker.py:1000` | 拒絕率回傳 float/Decimal 不一致 |
| L-06 | `fund_manager/manager.py:384` | 無資金時回傳模糊的 result |
| L-07 | `fund_manager/core/atomic_allocator.py:152` | 錯誤記錄 new_allocation 語義不一致 |
| L-08 | `fund_manager/core/fund_pool.py:358` | 負金額靜默移除分配 |
| L-09 | `fund_manager/core/position_manager.py:297` | 多空部位直接累加無衝突檢查 |

---

## 模組健康度

| 模組 | 行數 | 🔴 | 🟠 | 🟡 | 🟢 | 評級 |
|------|------|-----|-----|-----|-----|------|
| bots/base.py | 12,813 | 2 | 1 | 2 | 3 | ⚠️ |
| bots/grid/ | 7,412 | 2 | 4 | 4 | 0 | ❌ 最需修復 |
| bots/supertrend/ | 2,590 | 0 | 4 | 2 | 0 | ⚠️ |
| bots/bollinger/ | 2,354 | 0 | 1 | 1 | 0 | ✅ |
| bots/rsi_grid/ | 2,891 | 0 | 0 | 1 | 0 | ✅ |
| bots/grid_futures/ | 2,729 | 0 | 2 | 0 | 0 | ⚠️ |
| execution/ | 3,080 | 1 | 0 | 2 | 1 | ⚠️ |
| exchange/ | 3,200+ | 0 | 1 | 1 | 0 | ✅ |
| risk/ | 4,700+ | 0 | 3 | 0 | 0 | ⚠️ |
| core/ | 1,075 | 0 | 0 | 2 | 0 | ✅ |
| fund_manager/ | 2,639+ | 0 | 2 | 0 | 3 | ⚠️ |
| data/ | 1,917+ | 0 | 0 | 1 | 0 | ✅ |
| master/ | 1,638+ | 0 | 0 | 1 | 0 | ✅ |

---

## 正面發現

1. **Decimal 使用一致** — 所有財務計算（價格、數量、餘額）全面使用 Decimal，未發現 float `==` 比較
2. **Division by zero 防護完善** — 絕大部分除法有零值防護（`_safe_divide`、`if x > 0` guard）
3. **止損方向全部正確** — 所有 bot 的多空止損/止盈比較方向驗證通過
4. **Bollinger Bands 用母體標準差** — 符合業界標準
5. **RSI-Grid 的 RSI 計算正確** — 使用 Wilder's Smoothing
6. **狀態機轉換定義完整** — `VALID_STATE_TRANSITIONS` 覆蓋所有狀態
7. **原子性資金鎖定** — TOCTOU 防護正確實作
8. **Heartbeat 指數退避** — 自動重啟有正確的 backoff + reset
