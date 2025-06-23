#!/usr/bin/env python3
"""
Test fixed API with proper URL encoding
"""

from components.mt5_api_client import MT5APIClient
from components.trading_config import CONFIG

api = MT5APIClient(CONFIG["API_BASE"])

print("Testing Fixed API Client")
print("=" * 60)

# Test symbols with #
test_symbols = ["EURUSD#", "GBPJPY#", "USDTRY#"]

print("\n1. Testing get_symbol_info with # symbols:")
for symbol in test_symbols:
    info = api.get_symbol_info(symbol)
    if info:
        print(f"  ✓ {symbol}: Bid={info.get('bid', 'N/A')}, Ask={info.get('ask', 'N/A')}, "
              f"Spread={info.get('spread', 'N/A')}")
    else:
        print(f"  ✗ {symbol}: Failed to get info")

print("\n2. Testing get_current_price with # symbols:")
for symbol in test_symbols:
    price = api.get_current_price(symbol)
    if price:
        print(f"  ✓ {symbol}: {price}")
    else:
        print(f"  ✗ {symbol}: Failed to get price")

print("\n3. Testing market history with # symbols:")
for symbol in test_symbols[:1]:  # Just test one
    history = api.get_market_history(symbol, "M5", 5)
    if history:
        print(f"  ✓ {symbol}: Got {len(history)} candles")
    else:
        print(f"  ✗ {symbol}: Failed to get history")

# Compare EURUSD vs EURUSD#
print("\n4. Comparing EURUSD vs EURUSD#:")
for symbol in ["EURUSD", "EURUSD#"]:
    info = api.get_symbol_info(symbol)
    if info:
        print(f"  {symbol}: name={info.get('name')}, bid={info.get('bid')}, ask={info.get('ask')}")

print("\nConclusion: The system should use symbols WITH # suffix as they are the tradable ones!")