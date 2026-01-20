#!/bin/bash
# 一鍵啟動所有交易機器人
# 使用方式: ./start_all.sh

cd /mnt/c/trading/grid_trading_bot
source .venv/bin/activate

echo "============================================================"
echo "       啟動交易系統"
echo "============================================================"

# 檢查是否已有機器人在運行
RUNNING=$(ps aux | grep -E "python.*run_(bollinger|supertrend|grid_futures)" | grep -v grep | wc -l)
if [ "$RUNNING" -gt 0 ]; then
    echo "⚠️  發現 $RUNNING 個機器人已在運行"
    echo ""
    ps aux | grep -E "python.*run_(bollinger|supertrend|grid_futures)" | grep -v grep
    echo ""
    read -p "是否要停止它們並重新啟動？(y/n): " choice
    if [ "$choice" = "y" ]; then
        echo "停止現有機器人..."
        pkill -f "python.*run_bollinger"
        pkill -f "python.*run_supertrend"
        pkill -f "python.*run_grid_futures"
        sleep 2
    else
        echo "保持現有機器人運行"
        exit 0
    fi
fi

echo ""
echo "啟動 Bollinger Bot..."
nohup python run_bollinger.py > /tmp/bollinger_bot.log 2>&1 &
sleep 2

echo "啟動 Supertrend Bot..."
nohup python run_supertrend.py > /tmp/supertrend_bot.log 2>&1 &
sleep 2

echo "啟動 Grid Futures Bot..."
nohup python run_grid_futures.py > /tmp/grid_futures_bot.log 2>&1 &
sleep 2

echo ""
echo "============================================================"
echo "       啟動完成！"
echo "============================================================"
echo ""
echo "運行中的機器人："
ps aux | grep -E "python.*run_(bollinger|supertrend|grid_futures)" | grep -v grep | awk '{print "  PID: "$2" - "$11" "$12}'
echo ""
echo "查看日誌："
echo "  tail -f /tmp/bollinger_bot.log"
echo "  tail -f /tmp/supertrend_bot.log"
echo "  tail -f /tmp/grid_futures_bot.log"
echo ""
echo "停止所有機器人："
echo "  ./stop_all.sh"
echo "============================================================"
