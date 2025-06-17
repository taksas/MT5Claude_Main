#!/bin/bash
echo "🔍 CHECKING MT5 API AND DISCOVERING SYMBOLS..."
echo "=============================================="

# Test connection
echo "Testing connection..."
curl -s --connect-timeout 3 http://10.255.255.254:8000/status/ping

if [ $? -eq 0 ]; then
    echo "✅ API Connected - Discovering symbols..."
    python3 discover_symbols.py
else
    echo "❌ API Not Available"
    echo ""
    echo "TO START LIVE TRADING:"
    echo "1. On Windows: cd to MT5 Bridge directory"
    echo "2. Run: uvicorn main:app --host 0.0.0.0 --port 8000"
    echo "3. Run this script again"
    echo ""
    echo "Then I'll immediately discover your available symbols"
    echo "and start live trading with the correct symbol names!"
fi