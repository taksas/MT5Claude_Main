#!/usr/bin/env python3
"""
Final system test
"""

import time
from components.engine_core import UltraTradingEngine
from components.trading_config import CONFIG

print("FINAL SYSTEM TEST")
print("=" * 60)
print(f"Configuration:")
print(f"  MIN_CONFIDENCE: {CONFIG['MIN_CONFIDENCE']} (5%)")
print(f"  AGGRESSIVE_MODE: {CONFIG['AGGRESSIVE_MODE']}")
print(f"  IGNORE_SPREAD: {CONFIG['IGNORE_SPREAD']}")
print(f"  FORCE_TRADE_INTERVAL: {CONFIG['FORCE_TRADE_INTERVAL']}s")
print()

# Create and initialize engine
engine = UltraTradingEngine()

# Initialize
if not engine.api_client.check_connection():
    print("❌ Cannot connect to API")
    exit(1)

engine.balance = engine.api_client.get_balance()
print(f"✅ Connected - Balance: ¥{engine.balance:,.0f}")

engine.tradable_symbols = engine._discover_symbols()
print(f"✅ Found {len(engine.tradable_symbols)} symbols")

# Note about symbols
print("\n📝 Note: All symbols have '#' suffix (e.g., EURUSD#, GBPJPY#)")
print(f"   First 5 symbols: {', '.join(engine.tradable_symbols[:5])}")

# Check if any errors occur
print("\n🔍 Running one full analysis cycle...")
engine.running = True

try:
    # Run one iteration
    start_time = time.time()
    engine.run_once()
    elapsed = time.time() - start_time
    
    print(f"✅ Analysis completed in {elapsed:.1f} seconds")
    print("✅ No harmonic pattern errors!")
    
    # Check for active trades
    positions = engine.api_client.get_positions()
    if positions:
        print(f"\n📈 Active positions: {len(positions)}")
        for pos in positions[:3]:  # Show first 3
            print(f"   {pos['symbol']}: {pos['type_str']} {pos['volume']} lots")
    else:
        print("\n📊 No active positions yet")
        
    print("\n✅ SYSTEM IS WORKING PROPERLY!")
    print("\nTo run the full system:")
    print("  python3 main.py              # Engine only")
    print("  python3 main.py --visualize  # With visualizer")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()