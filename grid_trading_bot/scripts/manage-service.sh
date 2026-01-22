#!/bin/bash
# Trading Bot Service Manager
# 使用方式: ./manage-service.sh [start|stop|restart|status|logs|enable|disable]

SERVICE_NAME="trading-bot"

case "$1" in
    start)
        echo "Starting trading bot service..."
        systemctl --user start $SERVICE_NAME
        systemctl --user status $SERVICE_NAME
        ;;
    stop)
        echo "Stopping trading bot service..."
        systemctl --user stop $SERVICE_NAME
        echo "Service stopped."
        ;;
    restart)
        echo "Restarting trading bot service..."
        systemctl --user restart $SERVICE_NAME
        systemctl --user status $SERVICE_NAME
        ;;
    status)
        systemctl --user status $SERVICE_NAME
        ;;
    logs)
        echo "=== Recent logs ==="
        tail -50 /mnt/c/trading/grid_trading_bot/logs/systemd.log 2>/dev/null
        echo ""
        echo "=== Recent errors ==="
        tail -20 /mnt/c/trading/grid_trading_bot/logs/systemd-error.log 2>/dev/null
        ;;
    enable)
        echo "Enabling trading bot service (auto-start on login)..."
        systemctl --user daemon-reload
        systemctl --user enable $SERVICE_NAME
        # Enable lingering to keep services running after logout
        loginctl enable-linger $USER 2>/dev/null || echo "Note: enable-linger may require sudo"
        echo "Service enabled."
        ;;
    disable)
        echo "Disabling trading bot service..."
        systemctl --user disable $SERVICE_NAME
        echo "Service disabled."
        ;;
    install)
        echo "Installing systemd service..."
        mkdir -p ~/.config/systemd/user
        cp /mnt/c/trading/grid_trading_bot/scripts/trading-bot.service ~/.config/systemd/user/
        systemctl --user daemon-reload
        echo "Service installed. Run './manage-service.sh enable' to enable auto-start."
        ;;
    *)
        echo "Trading Bot Service Manager"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable|install}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the trading bot"
        echo "  stop     - Stop the trading bot"
        echo "  restart  - Restart the trading bot"
        echo "  status   - Show service status"
        echo "  logs     - Show recent logs"
        echo "  enable   - Enable auto-start on login"
        echo "  disable  - Disable auto-start"
        echo "  install  - Install systemd service file"
        exit 1
        ;;
esac
