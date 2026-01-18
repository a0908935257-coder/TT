#!/bin/bash
# Grid Trading Bot 啟動腳本
# 自動使用虛擬環境

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "錯誤: 找不到虛擬環境，請先執行: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

echo "使用虛擬環境: $VENV_PYTHON"
exec "$VENV_PYTHON" "$SCRIPT_DIR/run.py" "$@"
