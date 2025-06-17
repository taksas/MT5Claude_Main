#!/bin/bash
echo "🚀 CHECKING MT5 API CONNECTION..."
python3 mt5_client.py

if [ $? -eq 0 ]; then
    echo "✅ MT5 API CONNECTED - STARTING LIVE TRADING!"
    python3 main.py --mode live --force
else
    echo "❌ MT5 API NOT AVAILABLE"
    echo "Please start MT5 Bridge API server on Windows host first"
    echo ""
    echo "Required steps:"
    echo "1. Open MT5 terminal and login"
    echo "2. Enable algorithmic trading"
    echo "3. Start: uvicorn main:app --host 0.0.0.0 --port 8000"
    echo "4. Run this script again"
fi