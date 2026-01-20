#!/bin/bash
# 停止所有交易機器人
# 使用方式: ./stop_all.sh

echo "============================================================"
echo "       停止交易系統"
echo "============================================================"
echo ""

# 顯示當前運行的機器人
echo "當前運行的機器人："
ps aux | grep -E "python.*run_(bollinger|supertrend|grid_futures|discord)" | grep -v grep | awk '{print "  PID: "$2" - "$11" "$12}'
echo ""

# 停止機器人
echo "停止 Bollinger Bot..."
pkill -f "python.*run_bollinger" 2>/dev/null && echo "  ✓ 已停止" || echo "  - 未運行"

echo "停止 Supertrend Bot..."
pkill -f "python.*run_supertrend" 2>/dev/null && echo "  ✓ 已停止" || echo "  - 未運行"

echo "停止 Grid Futures Bot..."
pkill -f "python.*run_grid_futures" 2>/dev/null && echo "  ✓ 已停止" || echo "  - 未運行"

echo "停止 Discord Bot..."
pkill -f "python.*discord_bot" 2>/dev/null && echo "  ✓ 已停止" || echo "  - 未運行"

sleep 1

echo ""
echo "============================================================"
echo "       所有機器人已停止"
echo "============================================================"
