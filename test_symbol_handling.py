#!/usr/bin/env python3
"""
Test symbol handling with # suffix
"""

from components.engine_core import UltraTradingEngine
from components.trading_config import get_symbol_config, HIGH_PROFIT_SYMBOLS

print("Testing Symbol Handling with # Suffix")
print("=" * 60)

# Test get_symbol_config
print("\n1. Testing get_symbol_config:")
test_symbols = ["EURUSD#", "GBPJPY#", "USDTRY#", "UNKNOWN#"]
for symbol in test_symbols:
    config = get_symbol_config(symbol)
    print(f"  {symbol}: risk_factor={config['risk_factor']}, "
          f"typical_spread={config.get('typical_spread', 'default')}")

# Test engine symbol discovery
print("\n2. Testing engine symbol discovery:")
engine = UltraTradingEngine()
engine.balance = 1000  # Mock balance
symbols = engine._discover_symbols()

print(f"\nTotal symbols discovered: {len(symbols)}")
print("\nHigh-profit symbols with # suffix:")
for symbol in symbols[:15]:
    symbol_base = symbol.rstrip('#')
    if symbol_base in HIGH_PROFIT_SYMBOLS:
        print(f"  ✓ {symbol} (priority symbol)")
    else:
        print(f"    {symbol}")

# Check if all symbols have # suffix
symbols_without_hash = [s for s in symbols if not s.endswith('#')]
if symbols_without_hash:
    print(f"\n⚠️  WARNING: Found {len(symbols_without_hash)} symbols without # suffix:")
    print(f"   {symbols_without_hash[:5]}")
else:
    print("\n✅ All symbols have # suffix correctly!")

print("\n3. Testing with actual API:")
# Test actual symbol info retrieval
from components.mt5_api_client import MT5APIClient
api = MT5APIClient("http://172.28.144.1:8000")

test_symbol = "EURUSD#"
symbol_info = api.get_symbol_info(test_symbol)
if symbol_info:
    print(f"  ✓ Successfully retrieved info for {test_symbol}")
else:
    print(f"  ✗ Failed to get info for {test_symbol}")