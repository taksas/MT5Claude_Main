#!/usr/bin/env python3
"""Test the live system with correct order format"""

import time
import logging

# Set up logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

print("=" * 70)
print("   TESTING LIVE SYSTEM WITH FIXED ORDER FORMAT")
print("=" * 70)
print("\nThis test will run the engine for 60 seconds to see if it places orders.\n")

from components.engine_core import UltraTradingEngine

# Create engine
engine = UltraTradingEngine()

# Override some settings for faster testing
from components import trading_config
trading_config.CONFIG["FORCE_TRADE_INTERVAL"] = 10  # Force trade every 10 seconds
trading_config.CONFIG["MIN_CONFIDENCE"] = 0.1  # Lower confidence for testing

print("Starting engine with aggressive settings...")
print("- Force trade every 10 seconds")
print("- Minimum confidence: 10%")
print("- Focus on USDJPY# and other available symbols\n")

# Initialize engine
if not engine.api_client.check_connection():
    print("✗ Cannot connect to API")
    exit(1)

engine.balance = engine.api_client.get_balance()
if not engine.balance:
    print("✗ Cannot get balance")
    exit(1)

# Discover symbols
engine.tradable_symbols = engine._discover_symbols()
if not engine.tradable_symbols:
    print("✗ No tradable symbols found")
    exit(1)

# Make sure USDJPY# is first if available
if "USDJPY#" in engine.tradable_symbols:
    engine.tradable_symbols.remove("USDJPY#")
    engine.tradable_symbols.insert(0, "USDJPY#")

print(f"✓ Engine initialized")
print(f"  Balance: ¥{engine.balance:,.0f}")
print(f"  Symbols: {len(engine.tradable_symbols)}")
print(f"  Priority symbols: {engine.tradable_symbols[:5]}\n")

engine.running = True

# Run for 60 seconds
start_time = time.time()
iteration = 0

while time.time() - start_time < 60:
    iteration += 1
    print(f"\n--- Iteration {iteration} (elapsed: {int(time.time() - start_time)}s) ---")
    
    engine.run_once()
    
    print(f"Active trades: {len(engine.active_trades)}")
    print(f"Daily trades: {engine.daily_trades}")
    print(f"Force attempts: {engine.force_trade_attempts}")
    
    # Show any active trades
    if engine.active_trades:
        for ticket, trade in engine.active_trades.items():
            print(f"  Trade {ticket}: {trade.symbol} {trade.type} @ {trade.entry_price}")
    
    time.sleep(2)  # Wait 2 seconds between iterations

print("\n" + "=" * 70)
print("TEST COMPLETE")
print(f"Total trades opened: {engine.daily_trades}")
print(f"Force trade attempts: {engine.force_trade_attempts}")
print("=" * 70)

if engine.daily_trades > 0:
    print("\n✓ SUCCESS: System is placing orders correctly!")
else:
    print("\n✗ No trades placed. Check the logs above for errors.")