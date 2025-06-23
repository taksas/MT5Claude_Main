#!/usr/bin/env python3
"""
Test symbol info to understand data structure
"""

import requests
import json

API_BASE = "http://172.28.144.1:8000"

# Get tradable symbols first
response = requests.get(f"{API_BASE}/market/symbols/tradable")
symbols = response.json()

print("Testing symbol info for first 3 symbols...")
print("=" * 60)

for symbol in symbols[:3]:
    print(f"\nSymbol: {symbol}")
    clean_symbol = symbol.rstrip('#')
    
    # Try to get symbol info
    response = requests.get(f"{API_BASE}/market/symbols/{clean_symbol}")
    if response.status_code == 200:
        data = response.json()
        print(f"Keys: {list(data.keys())}")
        
        # Look for price-related fields
        for key in ['bid', 'ask', 'last', 'price', 'spread']:
            if key in data:
                print(f"  {key}: {data[key]}")
                
        # Print full data for first symbol
        if symbols.index(symbol) == 0:
            print(f"\nFull data for {symbol}:")
            print(json.dumps(data, indent=2))
    else:
        print(f"  Error {response.status_code}: {response.text}")

# Test getting latest candle as alternative
print("\n\nTesting latest candle as price source...")
test_symbol = symbols[0]
payload = {"symbol": test_symbol, "timeframe": "M1", "count": 1}
response = requests.post(f"{API_BASE}/market/history", json=payload)
if response.status_code == 200:
    candles = response.json()
    if candles:
        latest = candles[0]
        print(f"Latest candle for {test_symbol}:")
        print(f"  Time: {latest['time']}")
        print(f"  Close: {latest['close']}")
        print(f"  Spread: {latest['spread']}")