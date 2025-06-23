#!/usr/bin/env python3
"""
Debug signal generation issues
"""

import pytz
from datetime import datetime
from components.engine_core import UltraTradingEngine
from components.trading_config import CONFIG

# Check current time in JST
jst = pytz.timezone('Asia/Tokyo')
now_jst = datetime.now(jst)
print(f"Current time (JST): {now_jst.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Current hour: {now_jst.hour}")

# Check if in trading hours
if 1 <= now_jst.hour < 19:
    print("❌ NOT in trading hours (1:00-19:00 JST blocked)")
    print("⚠️  This is why no signals are generated!")
    print("✅ Trading hours: 19:00-00:59 JST")
else:
    print("✅ In trading hours")

# Initialize engine to check settings
print("\n" + "="*50)
print("Checking engine configuration...")
engine = UltraTradingEngine()

# Check trading hours function
is_trading = engine._is_trading_hours()
print(f"Engine says trading allowed: {is_trading}")

# Check other settings
print(f"\nSignal generation settings:")
print(f"- MIN_CONFIDENCE: {CONFIG.get('MIN_CONFIDENCE', 0.7)}")
print(f"- AGGRESSIVE_MODE: {CONFIG.get('AGGRESSIVE_MODE', False)}")
print(f"- IGNORE_SPREAD: {CONFIG.get('IGNORE_SPREAD', False)}")
print(f"- MAX_SPREAD: {CONFIG.get('MAX_SPREAD', 2.5)}")

# Check discovered symbols
symbols = engine._discover_symbols()
print(f"\nDiscovered {len(symbols)} symbols")
print(f"Top 5 symbols: {symbols[:5]}")